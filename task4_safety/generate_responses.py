from __future__ import annotations

import argparse
import pandas as pd

from common.data import load_yaml, repo_path
from common.generation import batch_generate
from common.models import load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 1):
    from common.evaluation import generated_records
    specs = policy_specs(cfg)
    df = load_xstest(cfg)
    rows = [{**row, "source_index": str(row["xstest_id"])} for row in df.to_dict("records")]
    cfg = {**cfg, "evaluation_policy": policy_name}
    return generated_records(cfg, rows, specs[policy_name], repo_path(cfg["results_dir"]) / "task4_safety" / f"generated_{policy_name}.jsonl",
                             max_new_tokens=int(cfg["safety_max_new_tokens"]), do_sample=False, score=False)


def main():
    from common.logging_utils import set_seed
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policy", choices=["sft", "dpo", "ppo", "grpo"])
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    for policy in ([args.policy] if args.policy else policy_specs(cfg)):
        generate_for_policy(cfg, policy)

if __name__ == "__main__":
    main()
