"""Report evidence coverage and fail when required results are incomplete."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def validate():
    issues = []
    manifest = json.loads((ROOT / "experiments/runs.json").read_text())
    for run in manifest["runs"]:
        if run["id"].startswith("benchmark_"):
            continue
        for relative in run["artifacts"]:
            if not (ROOT / relative).exists():
                issues.append(f"{run['id']}: missing {relative}")
    for task, budget in (("task1_dpo", "epochs"), ("task2_ppo", "updates"), ("task3_grpo", "updates")):
        path = ROOT / "results" / task / "standard_run.json"
        if not path.exists():
            issues.append(f"{task}: missing standard run provenance")
            continue
        data = json.loads(path.read_text())
        expected = 1 if task == "task1_dpo" else 20
        if data.get("status") != "complete" or data.get(budget) != expected:
            issues.append(f"{task}: standard budget/status mismatch")
        if task != "task1_dpo" and (not data.get("peak_vram_bytes") or data.get("wall_clock_seconds", 0) <= 0):
            issues.append(f"{task}: missing GPU memory/time evidence")
    safety = ROOT / "results/task4_safety/safety_summary.json"
    if safety.exists() and not json.loads(safety.read_text()).get("manual_audit", {}).get("complete"):
        issues.append("Task 4: manual audit is pending or incomplete")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    issues = validate()
    print(json.dumps({"complete": not issues, "issues": issues}, indent=2))
    if issues and not args.allow_incomplete:
        sys.exit(1)


if __name__ == "__main__":
    main()
