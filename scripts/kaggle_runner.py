"""Build private, self-contained Kaggle notebooks and operate separate CLI profiles.

No credential values are ever embedded in source snapshots or notebook metadata.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "runs.json"


def read_manifest():
    manifest = json.loads(MANIFEST.read_text())
    ids = [run["id"] for run in manifest["runs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate run IDs")
    for run in manifest["runs"]:
        if any(dependency not in ids for dependency in run["depends_on"]):
            raise ValueError(f"Unknown dependency for {run['id']}")
    visiting, visited = set(), set()
    lookup = {run["id"]: run for run in manifest["runs"]}
    def visit(key):
        if key in visiting:
            raise ValueError("Run dependency cycle")
        if key in visited:
            return
        visiting.add(key)
        for dependency in lookup[key]["depends_on"]:
            visit(dependency)
        visiting.remove(key)
        visited.add(key)
    for key in ids:
        visit(key)
    return manifest


def source_snapshot():
    stream = io.BytesIO()
    # Allowlist avoids including checkpoints, assets, credentials, environments or result records.
    allowed_dirs = {"common", "configs", "scripts", "task1_dpo", "task2_ppo", "task3_grpo",
                    "task4_safety", "task5_feedback", "tests", "experiments"}
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(ROOT.rglob("*")):
            relative = file.relative_to(ROOT)
            if not file.is_file() or file.is_symlink() or any(x.startswith(".") or x == "__pycache__" for x in relative.parts):
                continue
            include = ((relative.parts[0] in allowed_dirs and file.suffix in {".py", ".yaml", ".json"})
                       or str(relative) in {"requirements.txt", "README.md", "data/word_limit_prompts.jsonl"})
            if include:
                info = zipfile.ZipInfo(str(relative), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, file.read_bytes())
    payload = stream.getvalue()
    return base64.b64encode(payload).decode(), hashlib.sha256(payload).hexdigest()


def notebook_source(run, payload, digest):
    return f'''# Generated from a fixed source snapshot; experiment logic lives in Python modules.
import base64, hashlib, json, os, pathlib, shutil, subprocess, sys, time, zipfile, io
root = pathlib.Path("/kaggle/working/pa2")
root.mkdir(parents=True, exist_ok=True)
payload = base64.b64decode({payload!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(root)
os.chdir(root)
os.environ["HF_HOME"] = "/kaggle/temp/pa2-hf-cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MPLCONFIGDIR"] = "/kaggle/temp/pa2-matplotlib"
# Resume only a matching run/source snapshot; inherited dependency artifacts are copied too.
if {run.get('resume', False)!r}:
    previous = []
    for status_file in pathlib.Path("/kaggle/input").rglob("job_status.json"):
        status = json.loads(status_file.read_text())
        if status.get("run_id") == {run['id']!r}:
            if status.get("source_sha256") != {digest!r}: raise RuntimeError("Resume source snapshot differs")
            previous.append(status_file.parent)
    if len(previous) != 1: raise RuntimeError("Need exactly one previous output for this run")
    for prefix in ("outputs", "results"):
        if (previous[0] / prefix).exists():
            shutil.copytree(previous[0] / prefix, root / prefix, dirs_exist_ok=True)
# Copy only declared dependency outputs into the same relative paths.
dependencies = {run['dependency_artifacts']!r}
for relative in dependencies:
    if (root / relative).exists(): continue
    matches = [p for p in pathlib.Path("/kaggle/input").rglob(relative) if p.is_dir() or p.is_file()]
    bundles = []
    for bundle in pathlib.Path("/kaggle/input").rglob("dependency_bundle.zip"):
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            if relative in names or any(name.startswith(relative + "/") for name in names):
                bundles.append(bundle)
    if len(matches) + len(bundles) != 1:
        raise RuntimeError(f"Need exactly one mounted dependency {{relative}}, found {{matches + bundles}}")
    if bundles:
        with zipfile.ZipFile(bundles[0]) as archive:
            for name in archive.namelist():
                if name == relative or name.startswith(relative + "/") or name == str(pathlib.Path(relative).with_suffix(".meta.json")):
                    target = root / name
                    if not target.resolve().is_relative_to(root.resolve()): raise RuntimeError("Unsafe dependency archive path")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(name))
        continue
    if len(matches) != 1:
        raise RuntimeError(f"Need exactly one mounted dependency {{relative}}, found {{matches}}")
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if matches[0].is_dir():
        shutil.copytree(matches[0], destination, dirs_exist_ok=True)
    else:
        shutil.copy2(matches[0], destination)
    # Evaluation sidecars are required for stale-output validation.
    if matches[0].suffix == ".jsonl":
        sidecar = matches[0].with_suffix(".meta.json")
        if sidecar.exists(): shutil.copy2(sidecar, destination.with_suffix(".meta.json"))
def execute(command):
    print("RUN", command, flush=True)
    subprocess.run(command, check=True, cwd=root)
started = time.time()
(root / "job_status.json").write_text(json.dumps({{"run_id": {run['id']!r}, "status": "running", "source_sha256": {digest!r}}}))
try:
    execute([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    execute([sys.executable, "-m", "scripts.check_environment"])
    execute([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    execute([sys.executable, "-m", "scripts.download_assets"])
    execute([sys.executable, "-m", "scripts.validate_assets"])
    for command in {run['commands']!r}:
        execute([sys.executable] + command)
    missing = [p for p in {run['artifacts']!r} if not (root / p).exists()]
    if missing: raise RuntimeError(f"Job finished without required artifacts: {{missing}}")
    (root / "job_status.json").write_text(json.dumps({{"run_id": {run['id']!r}, "status": "complete", "source_sha256": {digest!r}, "wall_clock_seconds": time.time() - started}}))
except BaseException as exc:
    (root / "job_status.json").write_text(json.dumps({{"run_id": {run['id']!r}, "status": "failed", "error": str(exc), "source_sha256": {digest!r}}}))
    raise
finally:
    # Retain small experiment evidence and adapters; downloaded assets/models are reproducible inputs.
    for name in ("checkpoints", "cached", "data", "manifests", ".cache"):
        shutil.rmtree(root / name, ignore_errors=True)
'''


def build(run_id, username, datasets=(), kernels=(), resume_kernel=None, attempt=1):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise ValueError("Invalid Kaggle username")
    run = next((r for r in read_manifest()["runs"] if r["id"] == run_id), None)
    if run is None:
        raise ValueError(f"Unknown run ID: {run_id}")
    if run["depends_on"] and not (datasets or kernels or resume_kernel):
        raise ValueError("This run requires mounted dependency outputs; pass --kernel or --dataset")
    if resume_kernel and attempt < 2:
        raise ValueError("Use --attempt 2 (or higher) with --resume-kernel to avoid self-referencing the prior notebook")
    run = {**run, "resume": resume_kernel is not None}
    payload, digest = source_snapshot()
    title = f"PA2 {run_id.replace('_', ' ')}"
    if attempt > 1:
        title += f" attempt {attempt}"
    slug = title.lower().replace(" ", "-")
    directory = ROOT / "kaggle_jobs" / username / run_id
    if attempt > 1:
        directory = directory / f"attempt_{attempt}"
    directory.mkdir(parents=True, exist_ok=True)
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [{"cell_type": "code", "id": "launch", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": notebook_source(run, payload, digest).splitlines(True)}]}
    (directory / "run.ipynb").write_text(json.dumps(notebook, indent=2))
    metadata = {"id": f"{username}/{slug}", "title": title, "code_file": "run.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": run.get("gpu", True),
                "enable_internet": True, "dataset_sources": list(datasets), "kernel_sources": list(kernels) + ([resume_kernel] if resume_kernel else []),
                "competition_sources": [], "model_sources": []}
    if run.get("gpu", True):
        metadata["machine_shape"] = "NvidiaTeslaT4"
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    (directory / "snapshot.json").write_text(json.dumps({"sha256": digest, "run_id": run_id, "dependencies": run["depends_on"]}, indent=2))
    return directory


def cli(command, profile):
    env = os.environ.copy()
    if profile:
        env["KAGGLE_CONFIG_DIR"] = str(Path(profile).expanduser().resolve())
        env["PA2_KAGGLE_PROFILE"] = env["KAGGLE_CONFIG_DIR"]
        binary = shutil.which("kaggle")
        if not binary:
            raise FileNotFoundError("Install Kaggle CLI and add its environment's bin directory to PATH")
        launcher = [str(Path(binary).parent / "python"), str(ROOT / "scripts/kaggle_profile_cli.py")]
    else:
        launcher = ["kaggle"]
    subprocess.run(launcher + command, check=True, env=env)


def import_output(directory, run_id):
    run = next(r for r in read_manifest()["runs"] if r["id"] == run_id)
    candidates = list(Path(directory).rglob("job_status.json"))
    statuses = [(file, json.loads(file.read_text())) for file in candidates]
    matches = [(file, status) for file, status in statuses if status.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError("Need exactly one matching job_status.json")
    status_path, status = matches[0]
    if status["status"] != "complete":
        raise ValueError(f"Job was not successful: {status}")
    source_root = status_path.parent
    if any(not (source_root / artifact).exists() for artifact in run["artifacts"]):
        raise ValueError("Downloaded job is missing required artifacts")
    files = [file for prefix in ("outputs", "results") for file in (source_root / prefix).rglob("*") if file.is_file()]
    # Each independent judge job has its own memoization cache. Keep those in
    # the original download; only its experiment records/metrics are merged.
    files = [file for file in files if not (
        file.parent.relative_to(source_root).as_posix() == "results/task5_feedback"
        and file.name.startswith("pairwise_cache."))]
    for file in files:
        if file.is_symlink():
            raise ValueError("Downloaded artifacts must not contain symlinks")
        target = ROOT / file.relative_to(source_root)
        if target.exists() and target.read_bytes() != file.read_bytes():
            raise FileExistsError(f"Conflicting artifact: {target}; preserve separate run versions")
    for file in files:
        target = ROOT / file.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    receipt = ROOT / "results" / "jobs" / f"{run_id}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(status, indent=2))
    return len(files)


def dependency_dataset(run_id, username):
    run = next(r for r in read_manifest()["runs"] if r["id"] == run_id)
    if not run["dependency_artifacts"]:
        raise ValueError("Run has no dependency artifacts")
    directory = ROOT / "kaggle_jobs" / username / (run_id + "_inputs")
    directory.mkdir(parents=True, exist_ok=True)
    bundle = directory / "dependency_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in run["dependency_artifacts"]:
            source = ROOT / relative
            if not source.exists():
                raise FileNotFoundError(f"Import completed upstream outputs first: {source}")
            files = sorted(source.rglob("*")) if source.is_dir() else [source]
            if source.suffix == ".jsonl":
                files.append(source.with_suffix(".meta.json"))
            for file in files:
                if file.is_file():
                    if file.is_symlink():
                        raise ValueError("Dependency exports must not contain symlinks")
                    info = zipfile.ZipInfo(str(file.relative_to(ROOT)), date_time=(2026, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, file.read_bytes())
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()[:10]
    slug = f"pa2-inputs-{run_id.replace('_', '-')[:26]}-{digest}"
    metadata = {"id": f"{username}/{slug}", "title": slug.replace("-", " "),
                "licenses": [{"name": "other"}],
                "description": "Private experiment dependencies. Original authors retain rights; upstream model and dataset licenses apply."}
    (directory / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    return directory


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    build_parser = sub.add_parser("build")
    build_parser.add_argument("run_id")
    build_parser.add_argument("--username", required=True)
    build_parser.add_argument("--dataset", action="append", default=[])
    build_parser.add_argument("--kernel", action="append", default=[])
    build_parser.add_argument("--resume-kernel")
    build_parser.add_argument("--attempt", type=int, default=1)
    push = sub.add_parser("push")
    push.add_argument("directory", type=Path)
    push.add_argument("--profile", type=Path)
    status = sub.add_parser("status")
    status.add_argument("kernel")
    status.add_argument("--profile", type=Path)
    output = sub.add_parser("output")
    output.add_argument("kernel")
    output.add_argument("--directory", type=Path, required=True)
    output.add_argument("--profile", type=Path)
    imported = sub.add_parser("import")
    imported.add_argument("run_id")
    imported.add_argument("directory", type=Path)
    dataset = sub.add_parser("dependencies")
    dataset.add_argument("run_id")
    dataset.add_argument("--username", required=True)
    upload = sub.add_parser("upload-dependencies")
    upload.add_argument("directory", type=Path)
    upload.add_argument("--profile", type=Path)
    args = ap.parse_args()
    if args.action == "list":
        for run in read_manifest()["runs"]:
            print(run["id"], "account=" + run["account"], "depends=" + ",".join(run["depends_on"]))
    elif args.action == "build":
        print(build(args.run_id, args.username, args.dataset, args.kernel, args.resume_kernel, args.attempt))
    elif args.action == "push":
        metadata = json.loads((args.directory / "kernel-metadata.json").read_text())
        if metadata.get("is_private") is not True:
            raise ValueError("Runner only submits private notebooks")
        command = ["kernels", "push", "-p", str(args.directory), "-t", "28800"]
        if metadata.get("enable_gpu"):
            command += ["--accelerator", "NvidiaTeslaT4"]
        cli(command, args.profile)
    elif args.action == "status":
        cli(["kernels", "status", args.kernel], args.profile)
    elif args.action == "output":
        args.directory.mkdir(parents=True, exist_ok=True)
        cli(["kernels", "output", args.kernel, "-p", str(args.directory)], args.profile)
    elif args.action == "import":
        print("Imported files:", import_output(args.directory, args.run_id))
    elif args.action == "dependencies":
        print(dependency_dataset(args.run_id, args.username))
    elif args.action == "upload-dependencies":
        # Kaggle datasets create defaults to private; never pass --public.
        cli(["datasets", "create", "-p", str(args.directory)], args.profile)


if __name__ == "__main__":
    main()
