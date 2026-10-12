"""Matched frozen-policy evaluation and incremental generation records."""
from __future__ import annotations

import gc
import torch

from common.data import (encode_prompt_response, pad_batch, preference_responses,
                         prompt_messages, prompt_messages_from_preference, read_jsonl, repo_path)
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs, score_reward_pairs
from common.logging_utils import set_seed
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from common.online import entropy_from_logits
from common.runtime import RecordStore, atomic_json, identifier, numeric_stats, policy_provenance
from task1_dpo.dpo import dpo_loss


@torch.no_grad()
def preference_records(cfg, model, tokenizer, rows, adapter, name, dataset_key="dpo_standard_eval"):
    from common.data import filter_preference_prompts
    rows, coverage = filter_preference_prompts(tokenizer, rows, int(cfg["max_sequence_length"]))
    cfg = {**cfg, "prompt_length_coverage": coverage}
    path = repo_path(cfg["results_dir"]) / f"{name}_preferences.jsonl"
    store = RecordStore(path, {"config": cfg, "rows": rows, "policy": policy_provenance(adapter), "dataset": dataset_key})
    device = next(model.parameters()).device
    for index, row in enumerate(rows):
        key = identifier(row, index)
        if key in store.keys:
            continue
        prompts = prompt_messages_from_preference(row)
        chosen, rejected = preference_responses(row)
        batches = [pad_batch(tokenizer, [encode_prompt_response(tokenizer, prompts, text, int(cfg["max_sequence_length"]))])
                   for text in (chosen, rejected)]
        batches = [{k: v.to(device) for k, v in batch.items()} for batch in batches]
        pc, pr = [response_sequence_logprobs(model, batch)[0] for batch in batches]
        with reference_mode(model):
            rc, rr = [response_sequence_logprobs(model, batch)[0] for batch in batches]
        loss, _ = dpo_loss(pc, pr, rc, rr, float(cfg["beta"]))
        margin = float((pc - rc - pr + rr)[0])
        store.append({"record_id": key, "source_index": key, "margin": margin,
                      "correct": float(margin > 0), "loss": float(loss),
                      "stratum": row.get("length_stratum", row.get("stratum", row.get("length_bucket"))),
                      "chosen_response_tokens": int(batches[0]["response_mask"].sum()),
                      "rejected_response_tokens": int(batches[1]["response_mask"].sum())})
    return store.rows


@torch.no_grad()
def generated_records(cfg, rows, adapter, path, prompt_builder=prompt_messages,
                      max_new_tokens=256, do_sample=False, score=True):
    provenance = {"config": cfg, "rows": rows, "policy": policy_provenance(adapter),
                  "max_new_tokens": max_new_tokens, "do_sample": do_sample, "score": score}
    store = RecordStore(path, provenance)
    if len(store.rows) == len(rows):
        return store.rows
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    reward = load_reward_model(cfg) if score else None
    try:
        for index, row in enumerate(rows):
            key = identifier(row, index)
            if key in store.keys:
                continue
            # Per-example seeds make restart/sharding independent of preceding generations.
            set_seed(int(cfg["seed"]) + index)
            prompts = prompt_builder(row)
            gen = batch_generate(model, tokenizer, [prompts], int(cfg.get("max_prompt_length", 256)),
                                 max_new_tokens, temperature=cfg["generation"]["temperature"],
                                 top_p=cfg["generation"]["top_p"], do_sample=do_sample)
            record = {"record_id": key, "source_index": key, "prompt": prompts,
                      "response": gen["responses"][0], "response_tokens": gen["response_lengths"][0],
                      "terminated_with_eos": gen["terminated_with_eos"][0], "truncated": gen["truncated"][0]}
            record.update({k: row[k] for k in ("xstest_id", "benchmark_class", "type", "gold_final", "question") if k in row})
            if "evaluation_policy" in cfg:
                record["policy"] = cfg["evaluation_policy"]
            if score:
                lp, logits = response_token_logprobs(model, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
                record["entropy_sum"] = float(entropy_from_logits(logits, gen["response_mask"])) * record["response_tokens"]
                del logits
                with reference_mode(model):
                    ref, logits = response_token_logprobs(model, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
                    del logits
                record["kl_sum"] = float(((lp - ref) * gen["response_mask"]).sum())
                record["sample_entropy_sum"] = float((-lp * gen["response_mask"]).sum())
                record["reward"] = float(score_reward_pairs(*reward, [prompts], gen["responses"], int(cfg.get("reward_max_length", 1024)))[0])
            store.append(record)
    finally:
        del model, reward
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return store.rows


def generation_summary(records):
    tokens = sum(row["response_tokens"] for row in records)
    summary = {"n": len(records), "response_length": numeric_stats([r["response_tokens"] for r in records]),
               "truncated_fraction": sum(r["truncated"] for r in records) / max(len(records), 1)}
    if records and "reward" in records[0]:
        summary.update({"reward": numeric_stats([r["reward"] for r in records]),
                        "kl": sum(r["kl_sum"] for r in records) / max(tokens, 1),
                        "entropy": sum(r["entropy_sum"] for r in records) / max(tokens, 1),
                        "sample_entropy": sum(r["sample_entropy_sum"] for r in records) / max(tokens, 1),
                        "kl_averaging": "mean over all valid generated response tokens"})
    return summary


def evaluate_policy(config_path, adapter, name, task, beta=None):
    from common.data import load_yaml
    cfg = load_yaml(config_path)
    if beta is not None:
        cfg["beta"] = float(beta)
    if task == "task1_dpo":
        rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])
        model = load_policy(cfg, adapter_path=adapter, trainable=False)
        tok = load_tokenizer(cfg["base_model"])
        pairs = preference_records(cfg, model, tok, rows, adapter, name)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pref = {"heldout_dpo_loss": numeric_stats([r["loss"] for r in pairs])["mean"],
                "preference_accuracy": numeric_stats([r["correct"] for r in pairs])["mean"]}
        cap = int(cfg["max_generation_tokens"])
        builder = prompt_messages_from_preference
    else:
        rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
        pref = {}
        cap = int(cfg.get("eval_max_response_length", cfg.get("max_completion_length", 512)))
        builder = prompt_messages
    generated = generated_records(cfg, rows, adapter, repo_path(cfg["results_dir"]) / f"{name}_generated.jsonl",
                                  builder, cap, do_sample=bool(cfg["generation"]["do_sample"]))
    summary = {**pref, **generation_summary(generated), "config": cfg, "policy": policy_provenance(adapter)}
    atomic_json(repo_path(cfg["results_dir"]) / f"{name}_metrics.json", summary)
    return summary
