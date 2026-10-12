"""Validate saved per-example coverage and finite logged measurements."""
import json
import math
from scripts.kaggle_runner import ROOT


def main():
    issues = []
    checked = 0
    def records(path):
        nonlocal checked
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        checked += len(rows)
        return rows
    def ids(rows):
        return [str(r.get("source_index", r.get("record_id"))) for r in rows]
    for dataset, expected in (("gsm", 300), ("transfer", 100)):
        groups = [records(ROOT / f"results/task5_feedback/{dataset}_{policy}_generated.jsonl")
                  for policy in ("sft", "rlvr", "rlaif")]
        if any(len(rows) != expected for rows in groups) or any(ids(rows) != ids(groups[0]) for rows in groups[1:]):
            issues.append(f"{dataset}: policy IDs/count mismatch")
    safety = [records(ROOT / f"results/task4_safety/judged_{policy}.jsonl") for policy in ("sft", "dpo", "ppo", "grpo")]
    if any([r["xstest_id"] for r in rows] != [r["xstest_id"] for r in safety[0]] for rows in safety[1:]):
        issues.append("Safety policy IDs differ")
    for task in ("task2_ppo", "task3_grpo"):
        rows = records(ROOT / f"results/{task}/standard_train.jsonl")
        if [r["update"] for r in rows] != list(range(1,21)):
            issues.append(task + ": update sequence mismatch")
        for row in rows:
            for key in ("reward", "kl", "policy_loss", "gradient_norm", "entropy"):
                if not math.isfinite(row[key]):
                    issues.append(f"{task}: nonfinite {key}")
    report = {"checked_records": checked, "issues": issues, "passed": not issues,
              "scope": "Stored IDs, required evaluation counts, update sequences, and finite standard training measurements"}
    (ROOT / "results/validation/evidence_integrity.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
