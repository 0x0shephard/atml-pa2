from __future__ import annotations
import argparse
import gc
from common.data import load_yaml
from common.runtime import atomic_json, completed

def main():
    from task1_dpo.train import run_training
    from common.evaluation import evaluate_policy
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = {}
    for beta in ([args.beta] if args.beta is not None else cfg["betas"]):
        name = f"beta_{beta:.2f}"
        output = f"outputs/task1_dpo/{name}"
        if not completed(output):
            run_training(args.config, name, output_path=output, beta=beta,
                         max_examples=int(cfg["short_ablation_examples"]), resume=args.resume)
        gc.collect()
        results[name] = evaluate_policy(args.config, output, name, "task1_dpo", beta)
    atomic_json(f"{cfg['results_dir']}/beta_summary{'_' + format(args.beta, '.2f') if args.beta is not None else ''}.json", results)

if __name__ == "__main__":
    main()
