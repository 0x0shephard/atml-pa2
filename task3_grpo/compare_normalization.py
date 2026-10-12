from __future__ import annotations
import argparse
import gc
import json
from common.data import load_yaml, read_jsonl, repo_path
from common.runtime import atomic_json, completed

def main():
    from task3_grpo.continue_train import run_grpo
    from common.evaluation import evaluate_policy
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = {}
    for loss_type in ([args.loss_type] if args.loss_type else ["grpo", "dr_grpo"]):
        name = f"normalization_{loss_type}"
        output = f"outputs/task3_grpo/{name}"
        if not completed(output):
            run_grpo(args.config, output, int(cfg["fork_updates"]), loss_type, name, args.resume)
        gc.collect()
        metrics = evaluate_policy(args.config, output, name, "task3_grpo")
        logs = read_jsonl(repo_path(output) / "train.jsonl")
        results[loss_type] = {"heldout": metrics, "generated_tokens": logs[-1]["generated_tokens"],
                              "length_conditioned_gradient": [{"update": x["update"], **x["length_conditioned_gradient"]} for x in logs]}
    # Same updates/prompts/caps are matched; realized token counts can differ after policies diverge.
    results["budget_convention"] = "Equal updates, prompts, K and maximum generated-token allocation; report realized token counts separately"
    atomic_json(f"{cfg['results_dir']}/normalization_summary{'_' + args.loss_type if args.loss_type else ''}.json", results)

if __name__ == "__main__":
    main()
