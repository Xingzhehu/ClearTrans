from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from .common import append_jsonl, digest, read_jsonl, validate_instance, write_jsonl
from .detectors import InvalidProgram
from .prompts import direct_prompt, parse_response, student_prompt, translation_prompt, unwrap_code


def infer(input_path, output_path, client, config, method="cleartrans", executor=None, detector=None):
    completed = {r["id"] for r in read_jsonl(output_path)} if Path(output_path).exists() else set()
    for original in read_jsonl(input_path):
        row = validate_instance(original)
        if row["id"] in completed:
            continue
        calls = []
        seed = config.get("seed", 42) + int(digest(row["id"])[:8], 16) % 1000000
        result = client.generate(student_prompt(row) if method == "cleartrans" else direct_prompt(row), seed=seed)
        calls.append(result)
        response = result["text"]
        if method == "cleartrans":
            try:
                target = parse_response(response)["target_code"]
                format_ok = True
            except ValueError:
                target, format_ok = "", False
        else:
            target, format_ok = unwrap_code(response), True
        if method == "repair":
            if executor is None or detector is None:
                raise ValueError("Translate-and-repair requires execution and Sonar feedback")
            for attempt in range(config.get("repair_rounds", 3)):
                execution = executor.run(target, row["target_lang"], row["tests"])
                try:
                    analysis = detector.analyze(target, row["target_lang"])
                except InvalidProgram:
                    analysis = None
                if execution["correct"] and analysis is not None and analysis["smell_free"]:
                    break
                prompt = translation_prompt(row, row["source_code"], target,
                                            json.dumps({"execution": execution, "analysis": analysis}))
                result = client.generate(prompt, seed=seed + attempt + 1)
                calls.append(result)
                response, target = result["text"], unwrap_code(result["text"])
        usages = [call["usage"] for call in calls]
        totals = {key: sum(u[key] for u in usages) if all(u is not None and key in u for u in usages) else None
                  for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        append_jsonl(output_path, {**row, "method": method, "response": response, "target_code": target,
                                  "format_ok": format_ok, "calls": calls, "token_usage": totals,
                                  "seed": config.get("seed", 42), "generation_seed": seed})
        completed.add(row["id"])


def evaluate(input_path, output_path, executor, detector, detector_name):
    completed = {r["id"] for r in read_jsonl(output_path)} if Path(output_path).exists() else set()
    evaluated, skipped = 0, 0
    for row in read_jsonl(input_path):
        if (detector_name == "pylint" and row["target_lang"] != "python") or (detector_name == "pmd" and row["target_lang"] != "java"):
            skipped += 1
            continue
        if row["id"] in completed:
            continue
        code = row.get("target_code", "")
        execution = executor.run(code, row["target_lang"], row["tests"]) if code.strip() else {"correct": False, "failure": "empty_output"}
        analysis = None
        if execution["correct"]:
            # All correct outputs must be evaluated; analyzer failure aborts rather than inflating SFA.
            analysis = detector.analyze(code, row["target_lang"])
        append_jsonl(output_path, {**row, "execution": execution, "analysis": analysis,
            "evaluator": detector_name, "ca": bool(execution["correct"]),
            "sfa": bool(execution["correct"] and analysis is not None and analysis["smell_free"])})
        completed.add(row["id"])
        evaluated += 1
    return {"newly_evaluated": evaluated, "outside_analyzer_language_scope": skipped, "evaluator": detector_name}


def summarize(paths):
    groups = defaultdict(list)
    seen = set()
    for path in paths:
        for row in read_jsonl(path):
            key = (row.get("method", "unknown"), row["evaluator"], row["source_lang"], row["target_lang"], row.get("seed", 42))
            identity = (key, row.get("seed"), row["id"])
            if identity in seen:
                raise ValueError(f"Duplicate evaluation row: {identity}")
            seen.add(identity)
            groups[key].append(row)
    results = []
    for key, rows in sorted(groups.items()):
        correct = [r for r in rows if r["ca"]]
        numeric = [r["analysis"]["count"] for r in correct if r["analysis"]["count"] is not None]
        issues = Counter()
        for row in correct:
            for issue in row["analysis"].get("issues", []):
                issues[issue.get("rule", issue.get("message-id", "unknown"))] += 1
        results.append({"method": key[0], "evaluator": key[1], "source_lang": key[2], "target_lang": key[3], "seed": key[4],
            "n": len(rows), "ca_percent": 100 * len(correct) / len(rows),
            "instance_ids_sha256": digest(sorted(r["id"] for r in rows)),
            "sfa_percent": 100 * sum(r["sfa"] for r in rows) / len(rows),
            "mean_smells_on_correct": sum(numeric) / len(numeric) if numeric else None,
            "mean_prompt_tokens": _mean_usage(rows, "prompt_tokens"),
            "mean_completion_tokens": _mean_usage(rows, "completion_tokens"),
            "mean_total_tokens": _mean_usage(rows, "total_tokens"),
            "rule_counts_on_correct": dict(issues), "failure_counts": dict(Counter(r["execution"].get("failure") for r in rows if not r["ca"]))})
    return results


def _mean_usage(rows, key):
    values = [row.get("token_usage", {}).get(key) for row in rows]
    return sum(values) / len(values) if all(value is not None for value in values) else None


def summarize_seeds(results):
    import statistics
    groups = defaultdict(list)
    for row in results:
        groups[(row["method"], row["evaluator"], row["source_lang"], row["target_lang"])].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        if len({row["n"] for row in rows}) != 1:
            raise ValueError("Seed runs contain different instance counts")
        if len({row["instance_ids_sha256"] for row in rows}) != 1:
            raise ValueError("Seed runs contain different instance IDs")
        item = dict(zip(("method", "evaluator", "source_lang", "target_lang"), key))
        item["seeds"] = [row["seed"] for row in rows]
        for metric in ("ca_percent", "sfa_percent"):
            values = [row[metric] for row in rows]
            item[metric + "_mean"] = statistics.mean(values)
            item[metric + "_sample_std"] = statistics.stdev(values) if len(values) > 1 else None
        output.append(item)
    return output


def export_human(input_paths, output, sample_size=300, seed=42):
    """Shared instance sampling; model identities/reports are kept in a private key file."""
    methods = {}
    for path in input_paths:
        rows = list(read_jsonl(path))
        if not rows:
            raise ValueError(f"Empty prediction file: {path}")
        if len({r.get("method") for r in rows}) != 1:
            raise ValueError("Each human-evaluation input must contain one method")
        name = rows[0].get("method", "unknown")
        if name in methods or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Use unique method names and instance IDs for human export")
        methods[name] = {r["id"]: r for r in rows}
    common = sorted(set.intersection(*(set(rows) for rows in methods.values())))
    if len(common) < sample_size:
        raise ValueError(f"Need {sample_size} shared instances; only {len(common)} available")
    rng = random.Random(seed)
    chosen = rng.sample(common, sample_size)
    pairs = [(name, instance_id) for name in sorted(methods) for instance_id in chosen]
    rng.shuffle(pairs)
    public, private = [], []
    for index, (name, instance_id) in enumerate(pairs):
        row = methods[name][instance_id]
        blind_id = f"sample_{index:05d}"
        assessors = sorted(rng.sample(["A", "B", "C", "D", "E"], 3))
        public.append({"blind_id": blind_id, "language": row["target_lang"], "code": row["target_code"],
                       "assigned_assessors": assessors})
        private.append({"blind_id": blind_id, "method": name, "id": instance_id,
                        "source_lang": row["source_lang"], "target_lang": row["target_lang"],
                        "correct": row.get("ca"), "assigned_assessors": assessors})
    root = Path(output)
    write_jsonl(root / "assessor_materials.jsonl", public)
    write_jsonl(root / "PRIVATE_mapping.jsonl", private)
    return {"shared_instances": sample_size, "programs": len(public), "assessors_per_program": 3}


def import_human(labels_path, consensus_path, mapping_path):
    mapping = {r["blind_id"]: r for r in read_jsonl(mapping_path)}
    if not mapping:
        raise ValueError("Empty human evaluation mapping")
    labels = defaultdict(dict)
    for row in read_jsonl(labels_path):
        key, assessor = row["blind_id"], row["assessor"]
        if key not in mapping or assessor not in mapping[key]["assigned_assessors"]:
            raise ValueError("Unknown sample or unassigned assessor")
        if assessor in labels[key] or type(row.get("smell_free")) is not bool:
            raise ValueError("Duplicate or non-Boolean human label")
        labels[key][assessor] = row["smell_free"]
    consensus = {}
    for row in read_jsonl(consensus_path):
        if row["blind_id"] not in mapping or row["blind_id"] in consensus:
            raise ValueError("Unknown or duplicate consensus sample")
        consensus[row["blind_id"]] = row
    results, unanimous = [], 0
    for key, metadata in mapping.items():
        votes = labels[key]
        if len(votes) != 3 or type(consensus.get(key, {}).get("smell_free")) is not bool:
            raise ValueError(f"Need three independent labels and a final consensus for {key}")
        if type(metadata.get("correct")) is not bool:
            raise ValueError("Export human materials from evaluated outputs to compute joint SFA")
        unanimous += int(len(set(votes.values())) == 1)
        results.append({**metadata, "independent_labels": votes, "consensus": consensus[key]["smell_free"],
                        "sfa": metadata["correct"] and consensus[key]["smell_free"]})
    by_method = defaultdict(list)
    for row in results:
        by_method[row["method"]].append(row)
    return {"task": "human_smell_evaluation", "n": len(results), "unanimous_rate": unanimous / len(results),
            "agreement_measure": "pre-discussion observed unanimity, not chance-corrected kappa",
            "methods": {name: {"n": len(rows), "sfa_percent": 100 * sum(r["sfa"] for r in rows) / len(rows)}
                        for name, rows in by_method.items()}}
