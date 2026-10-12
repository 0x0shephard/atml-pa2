"""Package reproducible code and evidence, excluding credentials/data/weights."""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import base64
from scripts.kaggle_runner import ROOT, read_manifest, source_snapshot
from scripts.validate_results import validate


def main():
    destination = ROOT / "submission"
    destination.mkdir(exist_ok=True)
    runs = read_manifest()["runs"]
    issues = validate()
    evidence = {"experiment_jobs": len([r for r in runs if not r["id"].startswith("benchmark_")]),
                "benchmark_jobs": len([r for r in runs if r["id"].startswith("benchmark_")]),
                "job_receipts": {}, "validation_issues": issues,
                "submission_ready": False,
                "remaining": ["Student blind safety audit", "Student-written eight-page report",
                              "Accessible student GitHub repository and final commit", "LMS upload"]}
    for run in runs:
        path = ROOT / "results/jobs" / (run["id"] + ".json")
        receipt = json.loads(path.read_text())
        if receipt["status"] != "complete":
            raise ValueError(f"Incomplete job: {run['id']}")
        evidence["job_receipts"][run["id"]] = receipt
    audit = ROOT / "results/task4_safety/manual_audit.csv"
    with audit.open(newline="") as file:
        rows = list(csv.DictReader(file))
    evidence["manual_audit"] = {"rows": len(rows), "labeled": sum(bool(r["manual_label"].strip()) for r in rows)}
    (destination / "status.json").write_text(json.dumps(evidence, indent=2))
    payload, _ = source_snapshot()
    files = {}
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as source:
        files.update({name:source.read(name) for name in source.namelist()})
    for path in sorted((ROOT / "results").rglob("*")):
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
            files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    files[".gitignore"] = (ROOT / ".gitignore").read_bytes()
    for name in ("submission/README.md", "submission/manual_audit.html", "manual/README.md"):
        files[name] = (ROOT / name).read_bytes()
    files["submission/status.json"] = (destination / "status.json").read_bytes()
    hashes = {name: hashlib.sha256(data).hexdigest() for name,data in files.items()}
    (destination / "file_manifest.json").write_text(json.dumps(hashes, indent=2))
    files["submission/file_manifest.json"] = json.dumps(hashes, indent=2).encode()
    with zipfile.ZipFile(destination / "pa2-code-and-evidence.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name,data in files.items():
            if any(part in {"credentials.json", "kaggle.json"} for part in Path(name).parts):
                raise ValueError("Credential path encountered")
            archive.writestr(name,data)
    print(f"Packaged {len(files)} files; manual audit labels {evidence['manual_audit']['labeled']}/{len(rows)}")


if __name__ == "__main__":
    main()
