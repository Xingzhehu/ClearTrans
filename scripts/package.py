"""Build a source-only archive with SHA256 inventory. Run in the coder environment."""
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", "__pycache__", ".pytest_cache", "artifacts", "data"}


def main():
    files = sorted(path for path in ROOT.rglob("*") if path.is_file()
                   and not any(part in EXCLUDED or part.endswith(".egg-info") for part in path.relative_to(ROOT).parts)
                   and path.suffix not in (".pyc", ".zip", ".gz")
                   and path.name not in (".env", "SOURCE_MANIFEST.json", "SHA256SUMS"))
    inventory = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    manifest = ROOT / "SOURCE_MANIFEST.json"
    manifest.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8", newline="\n")
    files.append(manifest)
    checksums = ROOT / "SHA256SUMS"
    checksums.write_text("".join(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.relative_to(ROOT).as_posix() + "\n"
                                 for path in files), encoding="utf-8", newline="\n")
    files.append(checksums)
    tar_path = ROOT.parent / (ROOT.name + ".tar.gz")
    with tarfile.open(tar_path, "w:gz") as archive:
        for path in files:
            info = archive.gettarinfo(str(path), arcname=path.relative_to(ROOT).as_posix())
            info.uid, info.gid, info.uname, info.gname = 0, 0, "", ""
            info.mode = 0o755 if path.suffix == ".sh" else 0o644
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    zip_path = ROOT.parent / (ROOT.name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=ROOT.name + "/" + path.relative_to(ROOT).as_posix())
    print(json.dumps({"source_files": len(files), "tar": str(tar_path), "zip": str(zip_path)}, indent=2))


if __name__ == "__main__":
    main()
