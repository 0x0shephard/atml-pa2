from __future__ import annotations
import argparse
import gc
from collections import defaultdict
from common.data import load_yaml, read_jsonl, repo_path, prompt_messages
from common.metrics import word_limit_compliance
from common.runtime import atomic_json, completed, numeric_stats

def main():
    from task1_dpo.train import run_training
    from common.evaluation import preference_records, generated_records, generation_summary
    from common.models import load_policy, load_tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if not completed(cfg["length_output"]):
        run_training(args.config, "length_balanced", dataset_path=cfg["paths"]["dpo_length_train"],
                     output_path=cfg["length_output"], resume=args.resume)
    rows = read_jsonl(cfg["paths"]["dpo_length_eval"])
    summary = {}
    for name, adapter in (("standard", cfg["standard_output"]), ("length_balanced", cfg["length_output"])):
        model = load_policy(cfg, adapter_path=adapter, trainable=False)
        tokenizer = load_tokenizer(cfg["base_model"])
        pairs = preference_records(cfg, model, tokenizer, rows, adapter, name + "_length", "dpo_length_eval")
        strata = defaultdict(list)
        for row in pairs:
            if row["stratum"] is None:
                raise ValueError("Length evaluation rows require the course stratum field; inspect downloaded schema")
            strata[str(row["stratum"])].append(row["correct"])
        del model
        gc.collect()
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        prompts = read_jsonl(cfg["paths"]["word_limit_prompts"])
        generated = generated_records(cfg, prompts, adapter, repo_path(cfg["results_dir"]) / f"{name}_word_limit.jsonl",
                                      max_new_tokens=int(cfg["max_generation_tokens"]), do_sample=bool(cfg["generation"]["do_sample"]))
        compliance = [word_limit_compliance(" ".join(m["content"] for m in r["prompt"]), r["response"]) for r in generated]
        if any(x is None for x in compliance):
            raise ValueError("A fixed word-limit prompt was not recognized by the course helper")
        summary[name] = {"strata_preference_accuracy": {k: numeric_stats(v) for k, v in strata.items()},
                         "word_limit_compliance": numeric_stats(compliance), **generation_summary(generated)}
    atomic_json(repo_path(cfg["results_dir"]) / "length_summary.json", summary)

if __name__ == "__main__":
    main()
