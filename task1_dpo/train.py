from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    filter_preference_prompts,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    tokenizer = load_tokenizer(cfg["base_model"])
    rows, coverage = filter_preference_prompts(tokenizer, rows, int(cfg["max_sequence_length"]))
    if max_examples is not None:
        rows = rows[: int(max_examples)]
    cfg = {**cfg, "prompt_length_coverage": coverage}
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None, resume: bool = False):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    from common.runtime import Run, scaler_for, optimizer_step, identifier
    from common.generation import response_sequence_logprobs
    from common.models import reference_mode
    import random

    cfg = {**bundle["cfg"], "beta": bundle["beta"], "training_data": dataset_path or bundle["cfg"]["paths"]["dpo_standard_train"],
           "max_examples": max_examples}
    output = repo_path(output_path or f"outputs/task1_dpo/{run_name}")
    rows, model, tokenizer, optimizer = (bundle[k] for k in ("rows", "model", "tokenizer", "optimizer"))
    if not rows:
        raise ValueError("Training data is empty")
    run = Run(output, cfg, rows, run_name, resume)
    scaler = scaler_for(model)
    progress = run.restore({"policy": model}, {"policy": optimizer}, scaler)
    device = next(model.parameters()).device
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    batch_size, accum = int(cfg["batch_size"]), int(cfg["grad_accum_steps"])
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(progress["epoch"], int(cfg["epochs"])):
        order = list(range(len(rows)))
        random.Random(int(cfg["seed"]) + epoch).shuffle(order)
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        cursor = progress["cursor"] if epoch == progress["epoch"] else 0
        while cursor < len(batches):
            chunk = batches[cursor:cursor + accum]
            examples = sum(len(indices) for indices in chunk)
            losses, accuracies, used_ids = [], [], []
            for indices in chunk:
                chosen, rejected = collate([rows[i] for i in indices])
                chosen = {k: v.to(device) for k, v in chosen.items()}
                rejected = {k: v.to(device) for k, v in rejected.items()}
                with torch.no_grad(), reference_mode(model):
                    ref_chosen = response_sequence_logprobs(model, chosen)[0]
                    ref_rejected = response_sequence_logprobs(model, rejected)[0]
                policy_chosen = response_sequence_logprobs(model, chosen)[0]
                policy_rejected = response_sequence_logprobs(model, rejected)[0]
                loss, diagnostics = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, cfg["beta"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite DPO loss")
                scaler.scale(loss * len(indices) / examples).backward()
                losses.extend([float(loss.detach())] * len(indices))
                accuracies.extend([float(diagnostics["preference_accuracy"])] * len(indices))
                used_ids.extend(identifier(rows[i], i) for i in indices)
            norm = optimizer_step(model, optimizer, scaler, float(cfg["max_grad_norm"]))
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            cursor += len(chunk)
            progress = {"update": progress["update"] + 1, "epoch": epoch, "cursor": cursor}
            run.save({"policy": model}, {"policy": optimizer}, scaler, progress,
                     {"loss": sum(losses) / examples, "preference_accuracy": sum(accuracies) / examples,
                      "gradient_norm": norm, "examples": examples, "example_ids": used_ids})
        progress = {**progress, "epoch": epoch + 1, "cursor": 0}
    run.finish(model, tokenizer, updates=progress["update"], training_examples=len(rows), epochs=int(cfg["epochs"]))
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples, args.resume)


if __name__ == "__main__":
    main()
