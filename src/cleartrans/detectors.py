import base64
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import uuid

import requests

from .common import LANGUAGES, digest
from .llm import ChatClient
from .prompts import judge_prompt


class DetectorError(RuntimeError):
    """Unavailable or incomplete analysis; never interpret as smell-free."""


class InvalidProgram(ValueError):
    pass


@contextmanager
def file_lock(path, nonblocking=False):
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class SonarDetector:
    """One isolated project per scan, completed CE task, frozen profiles, paginated issues."""

    def __init__(self, config, executor, require_manifest=True):
        self.config, self.executor = config, executor
        self.url = os.environ.get(config.get("url_env", "SONAR_HOST_URL"), "").rstrip("/")
        token = os.environ.get(config.get("token_env", "SONAR_TOKEN"), "")
        if not self.url or not token:
            raise DetectorError("Set SONAR_HOST_URL and SONAR_TOKEN (or configured environment names)")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"
        self.cache = Path(config.get("cache_dir", "artifacts/sonar_cache"))
        self.cache.mkdir(parents=True, exist_ok=True)
        self.manifest = self.snapshot()
        if require_manifest:
            manifest_path = Path(config["manifest_path"])
            if not manifest_path.exists():
                raise DetectorError("Missing Sonar manifest. Run cleartrans sonar-snapshot before analysis")
            expected = json.loads(manifest_path.read_text())
            if expected != self.manifest:
                raise DetectorError("Sonar server/plugins/quality profiles differ from the frozen manifest")

    def request(self, endpoint, params=None, post=False, raw=False):
        method = self.session.post if post else self.session.get
        try:
            response = method(self.url + "/api/" + endpoint, **({"data": params} if post else {"params": params}), timeout=60)
        except requests.RequestException as exc:
            raise DetectorError(f"Sonar API unavailable: {endpoint}") from exc
        if not response.ok:
            raise DetectorError(f"Sonar API {endpoint}: HTTP {response.status_code}")
        return response.text if raw else response.json()

    def snapshot(self):
        version = self.request("server/version", raw=True).strip()
        if not version.startswith(self.config.get("server_version_prefix", "2025.1")):
            raise DetectorError(f"Unexpected Sonar server version: {version}")
        plugins = self.request("plugins/installed")["plugins"]
        plugins = sorted((p["key"], p.get("version", "")) for p in plugins)
        profiles = {}
        configured = self.config.get("profiles", {})
        if set(configured) != set(LANGUAGES):
            raise DetectorError("Explicit quality profile names are required for all five languages")
        for lang, name in configured.items():
            xml = self.request("qualityprofiles/export", {"language": LANGUAGES[lang], "qualityProfile": name}, raw=True)
            if not xml.strip():
                raise DetectorError(f"Empty profile export for {lang}")
            profiles[lang] = {"name": name, "export_sha256": digest(xml), "export_xml": xml}
        return {"server_version": version, "plugins": [list(p) for p in plugins], "profiles": profiles}

    def verify_profiles(self):
        # A mutable server profile must not silently invalidate cached results mid-run.
        if self.snapshot() != self.manifest:
            raise DetectorError("Sonar configuration changed during this run")

    @contextmanager
    def scanner_slot(self):
        import fcntl
        deadline = time.monotonic() + self.config.get("queue_timeout", 3600)
        while time.monotonic() < deadline:
            for slot in range(self.config.get("max_concurrent_scans", 2)):
                stream = open(self.cache / f"scanner_{slot}.lock", "a")
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    continue
                try:
                    yield
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)
                    stream.close()
                return
            time.sleep(0.5)
        raise DetectorError("Timed out waiting for a Sonar scanner slot")

    def analyze(self, code, lang):
        self.verify_profiles()
        key = digest({"code": code, "language": lang, "manifest": self.manifest, "schema": 1})
        destination = self.cache / f"{key}.json"
        with file_lock(self.cache / f"{key}.lock"):
            if destination.exists():
                return json.loads(destination.read_text())
            compiled = self.executor.run(code, lang, [], compile_only=True)
            if not compiled["correct"]:
                raise InvalidProgram(json.dumps(compiled, ensure_ascii=False))
            with self.scanner_slot():
                result = self._scan(code, lang, compiled, key)
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            temporary.replace(destination)
            return result

    def _scan(self, code, lang, compiled, key):
        project = self.config.get("project_prefix", "cleartrans") + "_" + uuid.uuid4().hex
        work_root = Path(self.config.get("work_dir", "/tmp/cleartrans-sonar"))
        work_root.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            self.request("projects/create", {"project": project, "name": project}, post=True)
            created = True
            self.request("qualityprofiles/add_project", {"project": project, "language": LANGUAGES[lang],
                         "qualityProfile": self.config["profiles"][lang]}, post=True, raw=True)
            with tempfile.TemporaryDirectory(dir=work_root) as directory:
                root = Path(directory).resolve()
                filename = compiled.get("source_name", f"main.{LANGUAGES[lang]}")
                source = root / filename
                source.write_text(code, encoding="utf-8")
                properties = {"sonar.projectKey": project, "sonar.projectName": project, "sonar.sources": ".",
                              "sonar.sourceEncoding": "UTF-8", "sonar.scm.disabled": "true",
                              "sonar.inclusions": filename, "sonar.exclusions": "classes/**"}
                if lang == "java":
                    for relative, encoded in compiled.get("files", {}).items():
                        path = root / relative
                        if not path.resolve().is_relative_to(root):
                            raise DetectorError("Unsafe compiled artifact path")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(base64.b64decode(encoded))
                    properties["sonar.java.binaries"] = "classes"
                if lang in ("c", "cpp"):
                    compiler = shutil.which("gcc" if lang == "c" else "g++")
                    if not compiler:
                        raise DetectorError("CFamily scanner requires matching GCC/G++ on the scanner host")
                    standard = "-std=c11" if lang == "c" else "-std=c++17"
                    database = [{"directory": str(root), "file": str(source),
                                 "arguments": [compiler, standard, "-O2", "-c", str(source), "-o", str(root / "main.o")]}]
                    (root / "compile_commands.json").write_text(json.dumps(database))
                    properties["sonar.cfamily.compile-commands"] = str(root / "compile_commands.json")
                (root / "sonar-project.properties").write_text("\n".join(f"{k}={v}" for k, v in properties.items()))
                env = {**os.environ, "SONAR_HOST_URL": self.url,
                       "SONAR_TOKEN": os.environ[self.config.get("token_env", "SONAR_TOKEN")]}
                try:
                    process = subprocess.run(self.config.get("scanner_command", ["sonar-scanner"]), cwd=root,
                                             env=env, capture_output=True, text=True, timeout=self.config.get("scan_timeout", 600))
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise DetectorError("SonarScanner unavailable or timed out") from exc
                if process.returncode:
                    # Scanner logs are not echoed: some versions print connection configuration.
                    raise DetectorError(f"SonarScanner exited {process.returncode}; check server configuration")
                task_file = root / ".scannerwork" / "report-task.txt"
                if not task_file.exists():
                    raise DetectorError("Scanner did not submit an analysis task")
                task = dict(line.split("=", 1) for line in task_file.read_text().splitlines() if "=" in line)
                deadline = time.monotonic() + self.config.get("scan_timeout", 600)
                while time.monotonic() < deadline:
                    ce = self.request("ce/task", {"id": task["ceTaskId"]})["task"]
                    if ce["status"] == "SUCCESS":
                        break
                    if ce["status"] in ("FAILED", "CANCELED"):
                        raise DetectorError(f"Sonar compute-engine task {ce['status']}")
                    time.sleep(1)
                else:
                    raise DetectorError("Sonar compute-engine task timed out")
                measures = self.request("measures/component", {"component": project,
                    "metricKeys": "ncloc,ncloc_language_distribution"})["component"]["measures"]
                measures = {m["metric"]: m["value"] for m in measures}
                distribution = measures.get("ncloc_language_distribution", "")
                observed_languages = {entry.split("=")[0] for entry in distribution.split(";") if "=" in entry}
                if float(measures.get("ncloc", 0)) <= 0 or LANGUAGES[lang] not in observed_languages:
                    raise DetectorError(f"No confirmed {lang} analysis coverage; refusing to return zero smells")
                issues, page = [], 1
                while True:
                    data = self.request("issues/search", {"componentKeys": project, "types": "CODE_SMELL",
                                                         "resolved": "false", "ps": 500, "p": page})
                    issues.extend(data["issues"])
                    total = data.get("paging", {}).get("total", data.get("total"))
                    if total is None:
                        raise DetectorError("Missing issue pagination metadata")
                    if len(issues) >= total:
                        break
                    if not data["issues"]:
                        raise DetectorError("Incomplete Sonar issue pagination")
                    page += 1
                return {"detector": "sonarqube", "count": len(issues), "smell_free": not issues,
                        "issues": issues, "measures": measures, "analysis_id": ce.get("analysisId"),
                        "manifest_sha256": digest(self.manifest), "cache_key": key}
        finally:
            if created and self.config.get("delete_temporary_projects", True):
                # Delete only the UUID project created by this invocation.
                self.request("projects/delete", {"project": project}, post=True, raw=True)


class PylintDetector:
    def __init__(self, config):
        self.config = config

    def analyze(self, code, lang):
        if lang != "python":
            raise ValueError("Pylint only supports Python")
        command = self.config.get("command", ["pylint"])
        try:
            version = subprocess.check_output(command + ["--version"], text=True, timeout=30)
            if not re.search(r"pylint\s+3\.3\.8\b", version):
                raise DetectorError("Pylint 3.3.8 is required")
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "main.py"
                path.write_text(code)
                process = subprocess.run(command + ["--rcfile", str(Path(self.config["rcfile"]).resolve()),
                                         "--output-format=json", "--score=no", str(path)],
                                         capture_output=True, text=True, timeout=60)
                if process.returncode < 0 or process.returncode & (1 | 32):
                    raise DetectorError("Pylint fatal/configuration error")
                issues = json.loads(process.stdout)
                return {"detector": "pylint", "count": len(issues), "smell_free": not issues, "issues": issues}
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise DetectorError("Pylint could not complete analysis") from exc


class PMDDetector:
    def __init__(self, config):
        self.config = config

    def analyze(self, code, lang):
        if lang != "java":
            raise ValueError("PMD evaluation is configured only for Java")
        command = self.config.get("command", ["pmd"])
        try:
            version = subprocess.check_output(command + ["--version"], text=True, timeout=30)
            if "7.17.0" not in version:
                raise DetectorError("PMD 7.17.0 is required")
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "Main.java"
                path.write_text(code)
                process = subprocess.run(command + ["check", "-d", str(path), "-R", self.config["ruleset"],
                                         "-f", "json", "--no-cache"], capture_output=True, text=True, timeout=60)
                if process.returncode not in (0, 4):
                    raise DetectorError("PMD analysis/configuration error")
                data = json.loads(process.stdout)
                if data.get("processingErrors") or data.get("configurationErrors"):
                    raise DetectorError("Incomplete PMD analysis")
                issues = [v for file in data.get("files", []) for v in file.get("violations", [])]
                return {"detector": "pmd", "count": len(issues), "smell_free": not issues, "issues": issues}
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise DetectorError("PMD could not complete analysis") from exc


class LLMDetector:
    def __init__(self, config):
        self.client = ChatClient(config)

    def analyze(self, code, lang):
        result = self.client.generate(judge_prompt(code), temperature=0)
        verdict = re.match(r"^\s*(YES|NO)\b", result["text"])
        if not verdict:
            raise DetectorError("LLM evaluator did not return a valid YES/NO verdict")
        # A qualitative YES/NO judgment has no comparable numeric smell count.
        return {"detector": "llm", "count": None, "smell_free": verdict.group(1) == "NO", **result}


def make_detector(name, config, executor):
    if name == "sonarqube":
        return SonarDetector(config["sonarqube"], executor)
    if name == "pylint":
        return PylintDetector(config["pylint"])
    if name == "pmd":
        return PMDDetector(config["pmd"])
    if name == "llm":
        return LLMDetector(config["judge"])
    raise ValueError(f"Unknown detector: {name}")
