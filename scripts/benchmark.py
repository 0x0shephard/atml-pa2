"""Small, separately named feasibility runs; never used as assignment results."""
from __future__ import annotations

import argparse
import gc
import time

from common.data import load_yaml, read_jsonl, repo_path
from common.runtime import atomic_json, completed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["dpo", "ppo", "grpo"], required=True)
    args = ap.parse_args()
    task = {"dpo": "task1_dpo", "ppo": "task2_ppo", "grpo": "task3_grpo"}[args.task]
    config = f"configs/{args.task}.yaml"
    cfg = load_yaml(config)
    output = f"outputs/benchmarks/{args.task}"
    started = time.perf_counter()
    if not completed(output):
        if args.task == "dpo":
            from task1_dpo.train import run_training
            run_training(config, "benchmark", output_path=output, max_examples=16, resume=True)
        elif args.task == "ppo":
            from task2_ppo.continue_train import run_ppo
            run_ppo(config, output, updates=1, run_name="benchmark", resume=True)
        else:
            from task3_grpo.continue_train import run_grpo
            run_grpo(config, output, updates=1, run_name="benchmark", resume=True)
    training_seconds = time.perf_counter() - started
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from common.evaluation import generated_records, generation_summary
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])[:4]
    started = time.perf_counter()
    generated = generated_records(cfg, rows, output, f"results/benchmarks/{args.task}_generated.jsonl",
                                  max_new_tokens=int(cfg.get("max_generation_tokens", cfg.get("eval_max_response_length", cfg.get("max_completion_length", 512)))),
                                  do_sample=True)
    generation_seconds = time.perf_counter() - started
    result = {"benchmark_only": True, "task": args.task, "training_process_seconds": training_seconds,
              "generation_process_seconds": generation_seconds, "generation": generation_summary(generated),
              "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    if args.task == "dpo":
        from task4_safety.judge_responses import load_judge, judge_one
        feedback = load_yaml("configs/feedback.yaml")
        tok, model = load_judge(feedback)
        started = time.perf_counter()
        result["judge_samples"] = [judge_one(tok, model, "\n".join(m["content"] for m in row["prompt"]), row["response"], int(feedback["judge_max_new_tokens"])) for row in generated]
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        result["judge_seconds"] = time.perf_counter() - started
    atomic_json(f"results/benchmarks/{args.task}.json", result)


if __name__ == "__main__":
    main()
