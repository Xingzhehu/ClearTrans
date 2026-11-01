"""Runs ONLY inside the disposable Docker container. No host-code execution."""
import base64
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import tempfile


def invoke(command, stdin, timeout, cwd):
    # Bound both output streams on disk; subprocess PIPE alone would be unbounded.
    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                   cwd=cwd, start_new_session=True, preexec_fn=limits,
                                   env={**os.environ, "HOME": "/work", "GOCACHE": "/work/go-cache",
                                        "GOTOOLCHAIN": "local", "GOPROXY": "off", "GOMAXPROCS": "2",
                                        "GOFLAGS": "-p=1", "CGO_ENABLED": "0"})
        timed_out = False
        try:
            process.communicate(stdin.encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Kill children even if the parent exits before them.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        stdout.seek(0)
        stderr.seek(0)
        return {"returncode": process.returncode, "timeout": timed_out,
                "stdout": stdout.read(1024 * 1024).decode(errors="replace"),
                "stderr": stderr.read(1024 * 1024).decode(errors="replace")}


def run(payload):
    lang, code = payload["language"], payload["code"]
    with tempfile.TemporaryDirectory(dir="/work") as directory:
        root = Path(directory)
        name = {"c": "main.c", "cpp": "main.cpp", "go": "main.go", "python": "main.py", "java": "Main.java"}[lang]
        main = "Main"
        if lang == "java":
            if re.search(r"^\s*package\s+", code, re.M):
                return {"correct": False, "failure": "unsupported_java_package", "tests": []}
            match = re.search(r"\bpublic\s+(?:final\s+)?class\s+(\w+)", code)
            if not match:
                match = re.search(r"\bclass\s+(\w+)", code)
            if match:
                main, name = match.group(1), match.group(1) + ".java"
        path = root / name
        path.write_text(code)
        compilers = {"c": ["gcc", "-std=c11", "-O2", name, "-lm", "-o", "program"],
                     "cpp": ["g++", "-std=c++17", "-O2", name, "-o", "program"],
                     "go": ["go", "build", "-o", "program", name],
                     "java": ["javac", "-encoding", "UTF-8", "-d", "classes", name],
                     "python": ["python3", "-m", "py_compile", name]}
        compiled = invoke(compilers[lang], "", payload.get("compile_timeout", 90), root)
        if compiled["returncode"] != 0 or compiled["timeout"]:
            return {"correct": False, "failure": "compile_error", "compile": compiled, "tests": []}
        if payload.get("compile_only"):
            files = {str(p.relative_to(root)): base64.b64encode(p.read_bytes()).decode()
                     for p in root.glob("classes/**/*.class")}
            return {"correct": True, "failure": None, "files": files, "source_name": name}
        commands = {"c": ["./program"], "cpp": ["./program"], "go": ["./program"],
                    "java": ["java", "-Xmx512m", "-cp", "classes", main], "python": ["python3", name]}
        results = []
        for test in payload["tests"]:
            result = invoke(commands[lang], test["input"], payload["timeout"], root)
            actual, expected = result["stdout"], test["output"]
            equal = actual.split() == expected.split() if payload["comparison"] == "tokens" else actual == expected
            result["passed"] = result["returncode"] == 0 and not result["timeout"] and equal
            results.append(result)
        failure = None
        if not all(r["passed"] for r in results):
            failure = ("timeout" if any(r["timeout"] for r in results) else
                       "runtime_error" if any(r["returncode"] for r in results) else "wrong_answer")
        return {"correct": bool(results) and failure is None, "failure": failure, "tests": results}


if __name__ == "__main__":
    try:
        print(json.dumps(run(json.load(sys.stdin))))
    except Exception as exc:
        print(json.dumps({"infrastructure_error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
