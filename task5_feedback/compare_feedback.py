from __future__ import annotations
import argparse
import json
from common.data import load_yaml, repo_path
from common.runtime import atomic_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    root = repo_path(cfg["results_dir"]) / "task5_feedback"
    source = {name: json.loads((root / f"{name}_summary.json").read_text()) for name in ("gsm", "transfer", "diagnostics")}
    drops = {}
    for name in ("sft", "rlvr", "rlaif"):
        drops[name] = {}
        for metric in ("exact_accuracy", "ai_pairwise_win_rate_against_sft"):
            if metric not in source["gsm"][name] or metric not in source["transfer"][name]:
                raise ValueError(f"Incomplete math evaluation: {name}/{metric}")
            drops[name][metric] = source["gsm"][name][metric] - source["transfer"][name][metric]
    atomic_json(root / "feedback_comparison.json", {**source, "in_domain_minus_transfer": drops})

if __name__ == "__main__":
    main()
