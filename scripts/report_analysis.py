"""Reproduce the report's exploratory paired exact-accuracy checks."""
import json
import math
from scripts.kaggle_runner import ROOT
from task5_feedback.rlvr import exact_reward


def main():
    root = ROOT / "results/task5_feedback"
    output = {}
    for dataset in ("gsm", "transfer"):
        rows = {policy: [json.loads(line) for line in (root / f"{dataset}_{policy}_generated.jsonl").read_text().splitlines()]
                for policy in ("sft", "rlvr", "rlaif")}
        ids = {p:[r["source_index"] for r in values] for p,values in rows.items()}
        if any(ids[p] != ids["sft"] for p in ids):
            raise ValueError("Paired IDs differ")
        scores = {p:[exact_reward(r["response"],r["gold_final"]) for r in values] for p,values in rows.items()}
        output[dataset] = {}
        for policy in ("rlvr", "rlaif"):
            better = sum(a == 1 and b == 0 for a,b in zip(scores[policy],scores["sft"]))
            worse = sum(a == 0 and b == 1 for a,b in zip(scores[policy],scores["sft"]))
            n = better + worse
            probability = min(1., 2 * sum(math.comb(n,k) for k in range(min(better,worse)+1)) / 2**n) if n else 1.
            output[dataset][policy] = {"better":better, "worse":worse, "paired_exact_p":probability,
                                       "truncated":sum(r["truncated"] for r in rows[policy])}
    path = ROOT / "results/validation/report_paired_checks.json"
    path.write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()
