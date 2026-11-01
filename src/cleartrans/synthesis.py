import json

from .common import append_jsonl, digest, read_jsonl, validate_instance
from .detectors import InvalidProgram
from .prompts import (CLEAN_BEGIN, CLEAN_END, parse_response, refactor_prompt,
                      response_segments, synthesis_prompt, translation_prompt, unwrap_code)


class AnchorSynthesizer:
    def __init__(self, client, executor, detector, config):
        self.client, self.executor, self.detector, self.config = client, executor, detector, config

    def generate_candidate(self, row, index):
        cfg, calls, history = self.config, [], []
        seed = cfg.get("seed", 42) + int(digest(row["id"])[:8], 16) % 1000000 + index * 100

        def generate(prompt, offset):
            result = self.client.generate(prompt, seed=seed + offset)
            calls.append(result)
            return result["text"]

        source_execution = self.executor.run(row["source_code"], row["source_lang"], row["tests"])
        if not source_execution["correct"]:
            raise ValueError(f"Original source fails reference tests: {row['id']}")
        source_report = self.detector.analyze(row["source_code"], row["source_lang"])
        clean, report, execution = row["source_code"], source_report, source_execution
        for attempt in range(cfg.get("max_refinement_rounds", 5) + 1):
            if execution["correct"] and report is not None and report["smell_free"]:
                break
            if attempt == cfg.get("max_refinement_rounds", 5):
                raise ValueError("Could not construct a correct smell-free source anchor")
            clean = unwrap_code(generate(refactor_prompt(row, clean, json.dumps(report), json.dumps(execution)), attempt))
            execution = self.executor.run(clean, row["source_lang"], row["tests"])
            try:
                report = self.detector.analyze(clean, row["source_lang"])
            except InvalidProgram:
                report = None
            history.append({"stage": "source", "attempt": attempt, "code": clean,
                            "execution": execution, "analysis": report})
        target = unwrap_code(generate(translation_prompt(row, clean), 20))
        for attempt in range(cfg.get("max_refinement_rounds", 5) + 1):
            target_execution = self.executor.run(target, row["target_lang"], row["tests"])
            try:
                target_report = self.detector.analyze(target, row["target_lang"])
            except InvalidProgram:
                target_report = None
            history.append({"stage": "target", "attempt": attempt, "code": target,
                            "execution": target_execution, "analysis": target_report})
            if target_execution["correct"] and target_report is not None and target_report["smell_free"]:
                break
            if attempt == cfg.get("max_refinement_rounds", 5):
                raise ValueError("Could not construct a correct smell-free target anchor")
            feedback = json.dumps({"execution": target_execution, "smells": target_report})
            target = unwrap_code(generate(translation_prompt(row, clean, target, feedback), 21 + attempt))
        cot = generate(synthesis_prompt(row, clean, target), 40).strip()
        if cot.startswith("<think>") and cot.endswith("</think>"):
            cot = cot[len("<think>"):-len("</think>")].strip()
        thought, answer = response_segments(cot, target, row["target_lang"])
        parsed = parse_response(thought + answer)
        if parsed["clean_source"].strip() != clean.strip():
            raise ValueError("Synthesized reasoning changed the validated clean-source anchor")
        if target.strip() in cot:
            raise ValueError("Synthesized reasoning leaked the complete final target code")
        return {**row, "candidate_id": f"{row['id']}:{index}", "candidate_index": index,
                "status": "ok", "cot": cot, "clean_source": clean, "target_code": target,
                "source_analysis": source_report, "clean_source_analysis": report,
                "target_analysis": target_report, "history": history, "calls": calls, "seed": seed}


def synthesize(input_path, output_path, synthesizer):
    completed = {row["candidate_id"] for row in read_jsonl(output_path)} if __import__("pathlib").Path(output_path).exists() else set()
    count = 0
    for original in read_jsonl(input_path):
        row = validate_instance(original)
        for index in range(synthesizer.config.get("n_candidates", 15)):
            candidate_id = f"{row['id']}:{index}"
            if candidate_id in completed:
                continue
            try:
                result = synthesizer.generate_candidate(row, index)
            except ValueError as exc:
                # Invalid anchors are rejected; detector/transport failures abort instead.
                result = {"id": row["id"], "candidate_id": candidate_id, "candidate_index": index,
                          "status": "rejected", "reason": str(exc)}
            append_jsonl(output_path, result)
            completed.add(candidate_id)
            count += 1
    return count
