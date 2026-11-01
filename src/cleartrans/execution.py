import json
import subprocess


class ExecutionError(RuntimeError):
    pass


class DockerExecutor:
    def __init__(self, config):
        self.config = config

    def run(self, code, lang, tests, compile_only=False):
        cfg = self.config
        payload = {"code": code, "language": lang, "tests": tests, "compile_only": compile_only,
                   "timeout": cfg.get("timeout", 10), "compile_timeout": cfg.get("compile_timeout", 90),
                   "comparison": cfg.get("comparison", "tokens")}
        command = ["docker", "run", "--rm", "--network=none", "--read-only", "--cap-drop=ALL",
                   "--security-opt=no-new-privileges", "--pids-limit=128", "--cpus=1",
                   f"--memory={cfg.get('memory', '1g')}", "--tmpfs=/work:rw,exec,size=512m,mode=1777",
                   "--tmpfs=/tmp:rw,noexec,nosuid,size=128m,mode=1777",
                   "--user=65534:65534", "-i", cfg.get("image", "cleartrans-runner:0.1"),
                   "python3", "/opt/runner.py"]
        try:
            result = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True,
                                    timeout=cfg.get("compile_timeout", 90) + len(tests) * payload["timeout"] + 30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExecutionError("Docker executor unavailable or exceeded outer deadline") from exc
        if result.returncode:
            raise ExecutionError(f"Docker runner failed: {result.stderr[-2000:]}")
        try:
            return json.loads(result.stdout)
        except ValueError as exc:
            raise ExecutionError("Docker runner produced invalid JSON") from exc
