import argparse
import json
from pathlib import Path
import shutil

from .common import guard_output, load_config, read_jsonl, record_run, validate_instance
from .execution import DockerExecutor


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="ClearTrans paper-method reconstruction")
    parser.add_argument("--config", default="configs/paper.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    commands.add_parser("sonar-snapshot")
    p = commands.add_parser("check-data")
    p.add_argument("--input", required=True)
    p.add_argument("--execute", action="store_true")
    p = commands.add_parser("prepare")
    p.add_argument("--input", required=True)
    p.add_argument("--benchmarks", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    for name in ("synthesize", "score", "select", "infer", "evaluate"):
        p = commands.add_parser(name)
        p.add_argument("--input", required=True)
        p.add_argument("--output", required=True)
        if name == "select":
            p.add_argument("--k", type=int, default=3)
            p.add_argument("--mode", choices=["dual", "plausibility", "effectiveness", "random"], default="dual")
        if name == "infer":
            p.add_argument("--method", choices=["cleartrans", "direct", "repair"], default="cleartrans")
        if name == "evaluate":
            p.add_argument("--detector", choices=["sonarqube", "pylint", "pmd", "llm"], default="sonarqube")
    for name in ("train-sft", "train-grpo"):
        p = commands.add_parser(name)
        p.add_argument("--input", required=True)
        p.add_argument("--resume")
    p = commands.add_parser("summarize")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("human-export")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample-size", type=int, default=300)
    p = commands.add_parser("human-import")
    p.add_argument("--labels", required=True)
    p.add_argument("--consensus", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    executor = DockerExecutor(cfg["execution"])
    cmd = args.command
    if cmd == "doctor":
        import os
        emit({"commands": {name: shutil.which(name) for name in ("docker", "sonar-scanner", "gcc", "g++", "pmd", "pylint")},
              "sonar_environment": {name: bool(os.environ.get(name)) for name in ("SONAR_HOST_URL", "SONAR_TOKEN")},
              "note": "Presence check only; no models, candidate programs, or analysis jobs were started."})
    elif cmd == "sonar-snapshot":
        from .detectors import SonarDetector
        detector = SonarDetector(cfg["sonarqube"], executor, require_manifest=False)
        path = Path(cfg["sonarqube"]["manifest_path"])
        if path.exists():
            raise ValueError("Manifest already exists; preserve it and explicitly choose another path for a new setup")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(detector.manifest, indent=2), encoding="utf-8")
        emit({"manifest": str(path), "server_version": detector.manifest["server_version"]})
    elif cmd == "check-data":
        seen = set()
        for original in read_jsonl(args.input):
            row = validate_instance(original)
            if row["id"] in seen:
                raise ValueError(f"Duplicate ID: {row['id']}")
            seen.add(row["id"])
            if args.execute and not executor.run(row["source_code"], row["source_lang"], row["tests"])["correct"]:
                raise ValueError(f"Source fails reference tests: {row['id']}")
        if not seen:
            raise ValueError("Empty dataset")
        emit({"instances": len(seen), "executed": args.execute})
    elif cmd == "prepare":
        from .data import prepare
        emit(prepare(args.input, args.benchmarks, args.output_dir, cfg["data"]))
        record_run(args.output_dir, cfg)
    elif cmd in ("synthesize", "score", "select", "infer", "evaluate"):
        provenance = {"configuration": cfg, "options": vars(args)}
        guard_output(args.output, provenance, [args.input], cmd)
        if cmd == "synthesize":
            from .detectors import SonarDetector
            from .llm import ChatClient
            from .synthesis import AnchorSynthesizer, synthesize
            worker = AnchorSynthesizer(ChatClient(cfg["teacher"]), executor,
                                       SonarDetector(cfg["sonarqube"], executor), cfg["synthesis"])
            emit({"new_candidates": synthesize(args.input, args.output, worker)})
        elif cmd == "score":
            from .scoring import score_candidates
            score_candidates(args.input, args.output, cfg)
        elif cmd == "select":
            from .scoring import select_candidates
            emit(select_candidates(args.input, args.output, args.k, args.mode, cfg["data"].get("seed", 42)))
        elif cmd == "infer":
            from .detectors import SonarDetector
            from .evaluation import infer
            from .llm import ChatClient
            detector = SonarDetector(cfg["sonarqube"], executor) if args.method == "repair" else None
            infer(args.input, args.output, ChatClient(cfg["inference"]), cfg["inference"], args.method, executor, detector)
        else:
            from .detectors import make_detector
            from .evaluation import evaluate
            emit(evaluate(args.input, args.output, executor, make_detector(args.detector, cfg, executor), args.detector))
    elif cmd == "train-sft":
        from .training import train_sft
        train_sft(args.input, cfg, args.resume)
    elif cmd == "train-grpo":
        from .training import train_grpo
        train_grpo(args.input, cfg, args.resume)
    elif cmd == "summarize":
        from .evaluation import summarize, summarize_seeds
        results = summarize(args.inputs)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({"per_run": results, "across_seeds": summarize_seeds(results)}, indent=2), encoding="utf-8")
        emit({"output": args.output, "direction_run_groups": len(results)})
    elif cmd == "human-export":
        from .evaluation import export_human
        emit(export_human(args.inputs, args.output_dir, args.sample_size, cfg["data"].get("seed", 42)))
    elif cmd == "human-import":
        from .evaluation import import_human
        results = import_human(args.labels, args.consensus, args.mapping)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
        emit(results)
