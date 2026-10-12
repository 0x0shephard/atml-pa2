from __future__ import annotations
import argparse
import gc
from common.data import load_yaml
from common.runtime import atomic_json, completed

def main():
    from task2_ppo.continue_train import run_ppo
    from common.evaluation import evaluate_policy
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = {}
    for beta in ([args.beta] if args.beta is not None else cfg["kl_values"]):
        name = f"kl_{beta:.2f}"
        output = f"outputs/task2_ppo/{name}"
        if not completed(output):
            run_ppo(args.config, output, int(cfg["fork_updates"]), kl_beta=beta, run_name=name, resume=args.resume)
        gc.collect()
        results[name] = evaluate_policy(args.config, output, name, "task2_ppo")
    atomic_json(f"{cfg['results_dir']}/kl_summary{'_' + format(args.beta, '.2f') if args.beta is not None else ''}.json", results)

if __name__ == "__main__":
    main()
