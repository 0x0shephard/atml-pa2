"""Hourly, restartable controller for the private PA2 experiment queue."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from scripts.kaggle_runner import ROOT, build, import_output, read_manifest, dependency_dataset

STATE_DIR = ROOT / "kaggle_jobs" / "autopilot"


def log(message):
    print(datetime.datetime.now().astimezone().isoformat(), message, flush=True)


def command(cli, args, profile=None):
    env = os.environ.copy()
    if profile:
        env["PA2_KAGGLE_PROFILE"] = profile
    if cli.endswith(".py"):
        python = ROOT.parent / ".kaggle-cli/bin/python"
        launcher = [str(python) if python.exists() else sys.executable, cli]
    elif profile:
        launcher = [str(Path(cli).parent / "python"), str(ROOT / "scripts/kaggle_profile_cli.py")]
    else:
        launcher = [cli]
    result = subprocess.run(launcher + args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    output = result.stdout + result.stderr
    if result.returncode:
        raise RuntimeError(output[-2000:])
    return output


def tick(state, username, cli):
    runs = read_manifest()["runs"]
    jobs = state["jobs"]
    accounts = state.get("accounts", {username: None})
    def save():
        temp = STATE_DIR / "state.tmp"
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(STATE_DIR / "state.json")
    for run in runs:
        key = run["id"]
        if key not in jobs:
            complete = all((ROOT / p).exists() for p in run["artifacts"])
            jobs[key] = {"status": "complete" if complete else "pending"}
    save()
    for run in runs:
        key, job = run["id"], jobs[run["id"]]
        if job["status"] not in ("submitted", "submitting", "download"):
            continue
        try:
            owner = job["kernel"].split("/")[0]
            profile = accounts[owner]
            output = command(cli, ["kernels", "status", job["kernel"]], profile)
            match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
            if not match:
                raise RuntimeError("Unrecognized status: " + output)
            status = match[1]
            log(f"{key}: {status}")
            job["remote_status"] = status
            if status == "COMPLETE":
                job["status"] = "download"
                save()
                directory = ROOT / "kaggle_downloads" / (key + "_auto")
                command(cli, ["kernels", "output", job["kernel"], "-p", str(directory),
                              "--file-pattern", ".*(outputs/|results/|job_status).*", "-q"], profile)
                import_output(directory, key)
                job["status"] = "complete"
                job.pop("error", None)
                log(f"{key}: imported successfully")
            elif status in ("ERROR", "CANCELLED", "CANCELED"):
                job["status"] = "failed"
                save()
                command(cli, ["kernels", "output", job["kernel"], "-p",
                              str(ROOT / "kaggle_downloads" / (key + "_failed_auto")),
                              "--file-pattern", ".*job_status.*", "-q"], profile)
        except Exception as exc:
            job["error"] = str(exc)
            log(f"{key}: {exc}")
        save()
    active = [r for r in runs if jobs[r["id"]]["status"] in ("submitted", "submitting", "download")]
    counts = {owner: {True: 0, False: 0} for owner in accounts}
    for r in active:
        counts[jobs[r["id"]]["kernel"].split("/")[0]][r.get("gpu", True)] += 1
    for run in runs:
        key, job = run["id"], jobs[run["id"]]
        if job["status"] != "pending" or any(jobs[d]["status"] != "complete" for d in run["depends_on"]):
            continue
        gpu = run.get("gpu", True)
        available = [owner for owner in accounts if counts[owner][gpu] < (2 if gpu else 1)]
        if not available:
            continue
        owner = min(available, key=lambda a: counts[a][gpu])
        profile = accounts[owner]
        kernels = [jobs[d].get("kernel", f"{username}/pa2-{d.replace('_', '-')}") for d in run["depends_on"]]
        try:
            datasets = []
            if any(k.split("/")[0] != owner for k in kernels):
                inputs = dependency_dataset(key, owner)
                dataset = json.loads((inputs / "dataset-metadata.json").read_text())["id"]
                output = command(cli, ["datasets", "create", "-p", str(inputs)], profile)
                if "Dataset creation error" in output or "https://www.kaggle.com/datasets/" not in output:
                    raise RuntimeError("Dependency dataset upload failed: " + output)
                datasets = [dataset]
                kernels = []
            directory = build(key, owner, datasets=datasets, kernels=kernels)
            metadata = json.loads((directory / "kernel-metadata.json").read_text())
            job.update(status="submitting", kernel=metadata["id"])
            save()  # Crash-safe: query this slug before any further submission.
            args = ["kernels", "push", "-p", str(directory), "-t", "28800"]
            if gpu:
                args += ["--accelerator", "NvidiaTeslaT4"]
            output = command(cli, args, profile)
            if "successfully pushed" not in output:
                job.update(status="pending", error=output)
                log(f"{key}: submission rejected: {output}")
                save()
                break
            job["status"] = "submitted"
            job.pop("error", None)
            log(f"{key}: submitted {job['kernel']}")
            counts[owner][gpu] += 1
        except Exception as exc:
            # Ambiguous submission stays 'submitting' to prevent duplicate pushes.
            job["error"] = str(exc)
            log(f"{key}: {exc}")
            save()
            break
        save()
    state["last_poll"] = datetime.datetime.now().astimezone().isoformat()
    save()
    return all(j["status"] in ("complete", "failed") for j in jobs.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--cli", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock = (STATE_DIR / "lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (STATE_DIR / "pid").write_text(str(os.getpid()))
    path = STATE_DIR / "state.json"
    state = json.loads(path.read_text()) if path.exists() else {"jobs": {}}
    while True:
        try:
            finished = tick(state, args.username, str(Path(args.cli).resolve()))
            if finished:
                log("Queue finished; manual audit and report remain user tasks.")
                break
        except Exception as exc:
            log(f"Controller error: {exc}")
        if args.once:
            break
        log("Next poll in one hour")
        time.sleep(3600)


if __name__ == "__main__":
    main()
