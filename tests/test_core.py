from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from cleartrans.common import guard_output, read_jsonl, validate_instance, write_jsonl
from cleartrans.data import DuplicateIndex
from cleartrans.detectors import DetectorError, SonarDetector
from cleartrans.evaluation import evaluate, export_human, import_human, infer, summarize, summarize_seeds
from cleartrans.prompts import parse_response, response_segments, student_prompt, unwrap_code
from cleartrans.rewards import Observation, batched_group_rewards, group_rewards, observe
from cleartrans.scoring import select_candidates, tokenize_record


def instance(identifier="one"):
    return {"id": identifier, "problem_id": f"problem:{identifier}", "source_lang": "python", "target_lang": "cpp",
            "source_code": "print(int(input()) + 1)", "tests": [{"input": "2\n", "output": "3\n"}]}


def response():
    return ("<think>\nRefactor.\n[Begin Clean Source Code]\nprint(int(input()) + 1)\n"
            "[End Clean Source Code]\nTranslate.\n</think>\n```cpp\nint main() { return 0; }\n```")


def good_observation(**kwargs):
    base = Observation(instance_id="one", format_ok=True, source_count=4, clean_count=0, target_count=0,
                       clean_correct=True, target_correct=True, clean_analyzable=True, target_analyzable=True)
    return replace(base, **kwargs)


class FormatTests(unittest.TestCase):
    def test_extract_both_programs(self):
        result = parse_response(response())
        self.assertEqual(result["clean_source"], "print(int(input()) + 1)")
        self.assertEqual(result["target_code"], "int main() { return 0; }")

    def test_missing_anchor_is_invalid(self):
        with self.assertRaises(ValueError):
            parse_response(response().replace("[End Clean Source Code]", ""))

    def test_extra_target_block_is_invalid(self):
        with self.assertRaises(ValueError):
            parse_response(response() + "\n```cpp\nextra\n```")

    def test_repeated_think_is_invalid(self):
        with self.assertRaises(ValueError):
            parse_response(response() + "<think>x</think>")

    def test_unwrap_teacher_markers(self):
        self.assertEqual(unwrap_code("[BEGIN]\n```python\nprint(1)\n```\n[END]"), "print(1)")

    def test_empty_tests_rejected(self):
        with self.assertRaises(ValueError):
            validate_instance({**instance(), "tests": []})

    def test_language_aliases(self):
        self.assertEqual(validate_instance({**instance(), "target_lang": "C++"})["target_lang"], "cpp")

    def test_segments_use_same_tokens_for_scoring_and_sft(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3]
            def encode(self, value, **kwargs):
                return list(value.encode())
        row = {**instance(), "cot": "reason", "target_code": "target"}
        prefix, thought, target = tokenize_record(row, Tokenizer())
        left, right = response_segments("reason", "target", "cpp")
        self.assertEqual(prefix, [1, 2, 3])
        self.assertEqual(bytes(thought).decode(), left)
        self.assertEqual(bytes(target).decode(), right)


class RewardTests(unittest.TestCase):
    def test_clean_correct_response_gets_two(self):
        self.assertAlmostEqual(group_rewards([good_observation()])[0], 2)

    def test_exact_paper_equation(self):
        a = good_observation(clean_count=2, target_count=3)
        b = good_observation(target_count=6)
        expected = 0.2 * (2 - 2 / (4 + 1e-6)) + 0.8 * (2 - 3 / (6 + 1e-6))
        self.assertAlmostEqual(group_rewards([a, b])[0], expected)

    def test_group_denominator_is_shared(self):
        a, b = good_observation(target_count=2), good_observation(target_count=8)
        together = group_rewards([a, b])[0]
        isolated = group_rewards([a])[0]
        self.assertGreater(together, isolated)

    def test_failed_test_gates_each_component(self):
        scores = group_rewards([good_observation(clean_correct=False), good_observation(target_correct=False)])
        self.assertAlmostEqual(scores[0], 1.6)
        self.assertAlmostEqual(scores[1], 0.4)

    def test_invalid_format_is_zero(self):
        self.assertEqual(group_rewards([good_observation(format_ok=False)])[0], 0)

    def test_zero_source_smells_and_introduced_smell(self):
        score = group_rewards([good_observation(source_count=0, clean_count=1)], alpha=1)[0]
        self.assertEqual(score, 1)

    def test_unanalyzed_code_cannot_get_clean_bonus(self):
        score = group_rewards([good_observation(clean_analyzable=False, target_analyzable=False)])[0]
        self.assertEqual(score, 0)

    def test_cannot_mix_sources(self):
        with self.assertRaises(ValueError):
            group_rewards([good_observation(), good_observation(instance_id="two")])

    def test_partial_group_rejected(self):
        with self.assertRaises(ValueError):
            batched_group_rewards([good_observation()] * 11)

    def test_two_complete_groups(self):
        rows = [good_observation()] * 12 + [good_observation(instance_id="two")] * 12
        self.assertEqual(batched_group_rewards(rows), [2] * 24)

    def test_detector_failure_is_not_zero_smells(self):
        detector = Mock()
        detector.analyze.side_effect = DetectorError("offline")
        with self.assertRaises(DetectorError):
            observe(instance(), response(), Mock(), detector)


class DataAndSelectionTests(unittest.TestCase):
    def test_identifier_and_comment_normalization(self):
        index = DuplicateIndex()
        index.add("def add(x):\n return x + 1\n", "python")
        self.assertTrue(index.duplicate("# comment\ndef inc(y):\n    return y + 1\n", "python"))

    def test_semantically_different_literal_not_exact_duplicate(self):
        index = DuplicateIndex(threshold=1)
        index.add("print(1)", "python")
        self.assertFalse(index.duplicate("print(2)", "python"))

    def test_language_separation(self):
        index = DuplicateIndex()
        index.add("print(1)", "python")
        self.assertFalse(index.duplicate("print(1)", "java"))

    def test_top_k_and_ablation(self):
        with tempfile.TemporaryDirectory() as directory:
            source, out = Path(directory) / "in.jsonl", Path(directory) / "out.jsonl"
            rows = [{"id": "one", "candidate_id": str(i), "status": "ok", "score": i, "score_eff": -i} for i in range(4)]
            write_jsonl(source, rows)
            select_candidates(source, out, k=2)
            self.assertEqual([r["candidate_id"] for r in read_jsonl(out)], ["3", "2"])
            select_candidates(source, out, k=2, mode="effectiveness")
            self.assertEqual([r["candidate_id"] for r in read_jsonl(out)], ["0", "1"])

    def test_missing_candidate_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "in.jsonl"
            write_jsonl(source, [{"id": "one", "candidate_id": "0", "status": "rejected"}])
            with self.assertRaises(ValueError):
                select_candidates(source, Path(directory) / "out.jsonl")

    def test_resume_rejects_different_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            source, out = Path(directory) / "in", Path(directory) / "out"
            source.write_text("one")
            guard_output(out, {"seed": 42}, [source], "infer")
            guard_output(out, {"seed": 42}, [source], "infer")
            with self.assertRaises(ValueError):
                guard_output(out, {"seed": 43}, [source], "infer")


class EvaluationTests(unittest.TestCase):
    def test_ca_sfa_joint_event_and_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            source, out = Path(directory) / "pred", Path(directory) / "eval"
            write_jsonl(source, [{**instance("a"), "target_code": "a"}, {**instance("b"), "target_code": "b"}])
            executor, detector = Mock(), Mock()
            executor.run.side_effect = [{"correct": True, "failure": None}, {"correct": False, "failure": "wrong_answer"}]
            detector.analyze.return_value = {"count": 0, "smell_free": True, "issues": []}
            evaluate(source, out, executor, detector, "sonarqube")
            stats = summarize([out])[0]
            self.assertEqual(stats["ca_percent"], 50)
            self.assertEqual(stats["sfa_percent"], 50)
            self.assertEqual(detector.analyze.call_count, 1)

    def test_correct_smelly_program_counts_only_ca(self):
        with tempfile.TemporaryDirectory() as directory:
            source, out = Path(directory) / "pred", Path(directory) / "eval"
            write_jsonl(source, [{**instance(), "target_code": "a"}])
            executor, detector = Mock(), Mock()
            executor.run.return_value = {"correct": True}
            detector.analyze.return_value = {"count": 2, "smell_free": False, "issues": []}
            evaluate(source, out, executor, detector, "sonarqube")
            stats = summarize([out])[0]
            self.assertEqual((stats["ca_percent"], stats["sfa_percent"]), (100, 0))

    def test_analysis_failure_aborts_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pred"
            write_jsonl(source, [{**instance(), "target_code": "a"}])
            executor, detector = Mock(), Mock()
            executor.run.return_value = {"correct": True}
            detector.analyze.side_effect = DetectorError("offline")
            with self.assertRaises(DetectorError):
                evaluate(source, Path(directory) / "eval", executor, detector, "sonarqube")

    def test_unknown_token_usage_stays_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            source, out = Path(directory) / "input", Path(directory) / "pred"
            write_jsonl(source, [instance()])
            client = Mock()
            client.generate.return_value = {"text": response(), "usage": None}
            infer(source, out, client, {"seed": 42})
            self.assertIsNone(next(read_jsonl(out))["token_usage"]["total_tokens"])

    def test_actual_usage_adds_all_repair_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            source, out = Path(directory) / "input", Path(directory) / "pred"
            write_jsonl(source, [instance()])
            client, executor, detector = Mock(), Mock(), Mock()
            client.generate.return_value = {"text": "```cpp\ncode\n```", "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
            executor.run.return_value = {"correct": False}
            detector.analyze.return_value = {"smell_free": False}
            infer(source, out, client, {"repair_rounds": 2}, "repair", executor, detector)
            self.assertEqual(next(read_jsonl(out))["token_usage"]["total_tokens"], 90)

    def test_seed_summary_detects_different_instances(self):
        rows = [{"method": "m", "evaluator": "e", "source_lang": "c", "target_lang": "cpp", "seed": i,
                 "n": 1, "instance_ids_sha256": str(i), "ca_percent": 100, "sfa_percent": 100} for i in (42, 43)]
        with self.assertRaises(ValueError):
            summarize_seeds(rows)


class HumanTests(unittest.TestCase):
    def test_export_blinds_models_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pred"
            write_jsonl(source, [{**instance(), "method": "secret_model", "target_code": "target", "ca": True,
                                  "analysis": {"count": 3}}])
            export_human([source], directory, sample_size=1)
            public = next(read_jsonl(Path(directory) / "assessor_materials.jsonl"))
            self.assertEqual(len(public["assigned_assessors"]), 3)
            self.assertNotIn("method", public)
            self.assertNotIn("analysis", public)

    def test_consensus_is_not_substituted_for_independent_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "map", [{"blind_id": "s", "method": "m", "correct": True,
                                        "assigned_assessors": ["A", "B", "C"]}])
            write_jsonl(root / "labels", [{"blind_id": "s", "assessor": a, "smell_free": a != "C"} for a in "ABC"])
            write_jsonl(root / "consensus", [{"blind_id": "s", "smell_free": True}])
            result = import_human(root / "labels", root / "consensus", root / "map")
            self.assertEqual(result["unanimous_rate"], 0)
            self.assertEqual(result["methods"]["m"]["sfa_percent"], 100)


class SonarContractTests(unittest.TestCase):
    def detector(self, directory):
        detector = SonarDetector.__new__(SonarDetector)
        detector.config = {"work_dir": directory, "profiles": {"python": "test"}}
        detector.url, detector.manifest = "http://localhost", {"version": "test"}
        return detector

    def test_completed_analysis_and_paginated_issues(self):
        with tempfile.TemporaryDirectory() as directory:
            detector = self.detector(directory)
            def request(endpoint, params=None, **kwargs):
                if endpoint == "ce/task":
                    return {"task": {"status": "SUCCESS", "analysisId": "analysis"}}
                if endpoint == "measures/component":
                    return {"component": {"measures": [{"metric": "ncloc", "value": "1"},
                        {"metric": "ncloc_language_distribution", "value": "py=1"}]}}
                if endpoint == "issues/search":
                    return {"issues": [{"rule": "py:test"}] * (500 if params["p"] == 1 else 1), "paging": {"total": 501}}
                return {}
            detector.request = Mock(side_effect=request)
            def scanner(command, cwd, **kwargs):
                target = Path(cwd) / ".scannerwork"
                target.mkdir()
                (target / "report-task.txt").write_text("ceTaskId=task\n")
                return Mock(returncode=0)
            with patch.dict("os.environ", {"SONAR_TOKEN": "test-placeholder"}), patch("cleartrans.detectors.subprocess.run", scanner):
                result = detector._scan("print(1)", "python", {"source_name": "main.py"}, "key")
            self.assertEqual(result["count"], 501)
            self.assertFalse(result["smell_free"])
            deleted = [call for call in detector.request.call_args_list if call.args[0] == "projects/delete"]
            self.assertEqual(len(deleted), 1)

    def test_missing_language_coverage_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            detector = self.detector(directory)
            def request(endpoint, params=None, **kwargs):
                if endpoint == "ce/task":
                    return {"task": {"status": "SUCCESS"}}
                if endpoint == "measures/component":
                    return {"component": {"measures": [{"metric": "ncloc", "value": "1"}]}}
                return {}
            detector.request = request
            def scanner(command, cwd, **kwargs):
                root = Path(cwd) / ".scannerwork"
                root.mkdir()
                (root / "report-task.txt").write_text("ceTaskId=task\n")
                return Mock(returncode=0)
            with patch.dict("os.environ", {"SONAR_TOKEN": "test-placeholder"}), patch("cleartrans.detectors.subprocess.run", scanner):
                with self.assertRaises(DetectorError):
                    detector._scan("print(1)", "python", {"source_name": "main.py"}, "key")


if __name__ == "__main__":
    unittest.main()
