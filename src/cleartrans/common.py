import hashlib
import json
import os
import random
from pathlib import Path

import yaml

LANGUAGES = {"c": "c", "cpp": "cpp", "go": "go", "java": "java", "python": "py"}
ALIASES = {"c++": "cpp", "py": "python", "python3": "python", "golang": "go"}


def language(value):
    value = ALIASES.get(value.lower(), value.lower())
    if value not in LANGUAGES:
        raise ValueError(f"Unsupported language: {value}")
    return value


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def guard_output(path, config, inputs, stage):
    """Reject accidental resume with a different input, model, prompt, or configuration."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    signature = {"stage": stage, "config": config,
                 "inputs": {str(Path(p).resolve()): file_digest(p) for p in inputs},
                 "implementation_version": "0.1.0",
                 "implementation_sha256": digest({p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))})}
    # Freeze referenced rule/config files as well as their filenames.
    def references(value):
        found = {}
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("manifest_path", "rcfile", "ruleset", "deepspeed") and isinstance(item, str) and Path(item).is_file():
                    found[item] = file_digest(item)
                found.update(references(item))
        return found
    signature["referenced_files"] = references(config)
    if sidecar.exists():
        if json.loads(sidecar.read_text(encoding="utf-8")) != signature:
            raise ValueError(f"Provenance mismatch for {path}; choose a fresh output path")
    elif path.exists():
        raise ValueError(f"Existing output has no provenance: {path}; choose a fresh output path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_instance(row):
    for key in ("id", "problem_id", "source_lang", "target_lang", "source_code", "tests"):
        if key not in row:
            raise ValueError(f"Missing {key}: {row.get('id')}")
    row = dict(row)
    row["id"], row["problem_id"] = str(row["id"]), str(row["problem_id"])
    row["source_lang"] = language(row["source_lang"])
    row["target_lang"] = language(row["target_lang"])
    if row["source_lang"] == row["target_lang"]:
        raise ValueError("Source and target languages must differ")
    if not row["source_code"].strip() or not row["tests"]:
        raise ValueError("Nonempty source code and test cases are required")
    for test in row["tests"]:
        if not isinstance(test.get("input"), str) or not isinstance(test.get("output"), str):
            raise ValueError("Each test requires string input and output")
    return row


def record_run(output, config, argv=None):
    import platform
    import subprocess
    from importlib.metadata import PackageNotFoundError, version
    versions = {}
    for name in ("cleartrans-replication", "torch", "transformers", "trl", "accelerate", "datasets", "deepspeed"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    # Configuration contains environment-variable NAMES, never credentials.
    info = {"config": config, "config_sha256": digest(config), "argv": argv,
            "python": platform.python_version(), "platform": platform.platform(), "packages": versions}
    try:
        info["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True, timeout=10).strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        info["gpu"] = []
    Path(output).mkdir(parents=True, exist_ok=True)
    Path(output, "run_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
