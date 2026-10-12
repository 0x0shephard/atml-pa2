from __future__ import annotations

import argparse
import torch

from common.data import load_yaml, repo_path


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def cached_geometry(cfg, rows):
    from common.models import load_policy, load_tokenizer, load_value_model, token_values
    from common.data import prompt_messages, read_jsonl
    from common.generation import response_token_logprobs
    from common.runtime import atomic_json, identifier
    from task2_ppo.ppo import shaped_rewards, compute_gae, normalize_advantages, ppo_policy_loss
    import numpy as np
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, cfg["paths"]["ppo_midpoint_policy"], trainable=False)
    value = None
    # The released cache uses held-out eval prompts; prompt_id disambiguates
    # source_index values that can overlap across source splits.
    pool_rows = read_jsonl(cfg["paths"]["rl_prompt_eval"]) + read_jsonl(cfg["paths"]["rl_prompt_train"])
    pool = {str(row["prompt_id"]): row for row in pool_rows if "prompt_id" in row}
    results = []
    device = next(policy.parameters()).device
    with torch.no_grad():
        for row in rows:
            old = torch.as_tensor(row["old_logprobs"], dtype=torch.float32, device=device).reshape(1, -1)
            ref = torch.as_tensor(row["ref_logprobs"], dtype=torch.float32, device=device).reshape(1, -1)
            if old.shape != ref.shape:
                raise ValueError("Cached old/reference logprobs disagree in shape")
            mask = torch.as_tensor(row.get("response_mask", [1] * old.shape[1]), dtype=torch.float32, device=device).reshape(1, -1)
            if mask.shape != old.shape:
                raise ValueError("Cached response mask shape mismatch")
            if "new_logprobs" in row:
                new = torch.as_tensor(row["new_logprobs"], dtype=torch.float32, device=device).reshape(1, -1)
                generation = None
            else:
                source = row if "prompt" in row or "messages" in row else pool.get(str(row.get("prompt_id")))
                if source is None:
                    matches = [p for p in pool_rows if str(p.get("source_index")) == str(row["source_index"])]
                    if len(matches) == 1:
                        source = matches[0]
                if source is None:
                    raise ValueError("Cache prompt ID absent from fixed pool")
                prompt = tokenizer.apply_chat_template(prompt_messages(source), tokenize=False, add_generation_prompt=True)
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(cfg["max_prompt_length"]))
                ids = row.get("response_ids", row.get("response_token_ids"))
                if ids is None:
                    ids = tokenizer(str(row["response"]), add_special_tokens=False)["input_ids"]
                    valid = int(mask.sum())
                    if len(ids) + 1 == valid and tokenizer.eos_token_id is not None:
                        ids = ids + [tokenizer.eos_token_id]
                    elif len(ids) != valid:
                        raise ValueError("Cannot reconstruct exact cached tokens from response text; cache must provide response_ids")
                response = torch.as_tensor(ids, dtype=torch.long, device=device).reshape(1, -1)
                if response.shape[1] == int(mask.sum()) and response.shape[1] < old.shape[1]:
                    response = torch.nn.functional.pad(response, (0, old.shape[1] - response.shape[1]), value=tokenizer.pad_token_id)
                if response.shape != old.shape:
                    raise ValueError("Cached token/logprob length mismatch")
                prompt_ids = encoded["input_ids"].to(device)
                sequences = torch.cat([prompt_ids, response], dim=1)
                attention = torch.cat([encoded["attention_mask"].to(device), mask.long()], dim=1)
                generation = (sequences, attention, prompt_ids.shape[1], response)
                new, logits = response_token_logprobs(policy, *generation)
                del logits
            if new.shape != old.shape:
                raise ValueError("Cached new logprob shape mismatch")
            if "advantages" in row:
                advantage = torch.as_tensor(row["advantages"], dtype=torch.float32, device=device).reshape(1, -1)
            else:
                values = row.get("values", row.get("value_estimates"))
                if values is None:
                    if generation is None:
                        raise ValueError("Cached new logprobs without values/advantages cannot reconstruct GAE")
                    if value is None:
                        value = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], "frozen")
                    sequences, attention, width, response = generation
                    values = token_values(value, sequences, attention)[:, width - 1: width - 1 + old.shape[1]].to(device)
                else:
                    values = torch.as_tensor(values, dtype=torch.float32, device=device).reshape(1, -1)
                scalar_reward = row.get("effective_terminal_reward", row.get("reward", row.get("rm_reward", row.get("task_reward", row.get("raw_terminal_reward")))))
                if scalar_reward is None:
                    raise ValueError("PPO cache requires reward or precomputed advantages")
                scalar_reward = float(scalar_reward)
                if "effective_terminal_reward" not in row and row.get("terminated_with_eos") is False:
                    scalar_reward -= float(cfg.get("missing_eos_penalty", 0))
                rewards = shaped_rewards(torch.tensor([scalar_reward], device=device), old, ref, mask, cfg["kl_beta"])
                advantage, returns = compute_gae(rewards, values, mask, cfg["gamma"], cfg["gae_lambda"])
                advantage = normalize_advantages(advantage, mask)
            if advantage.shape != old.shape:
                raise ValueError("PPO cache advantage shape mismatch")
            for eps in cfg["clip_values"]:
                loss, ratio, clip = ppo_policy_loss(new, old, advantage, mask, eps)
                changed = ((ratio * advantage) != (ratio.clamp(1-eps, 1+eps) * advantage)) & mask.bool()
                results.append({"source_index": str(row["source_index"]), "epsilon": float(eps), "loss": float(loss),
                                "valid_tokens": int(mask.sum()), "clip_fraction": float(clip),
                                "affected_token_fraction": float(clip),
                                "changed_surrogate_fraction": float(changed.sum() / mask.sum().clamp_min(1))})
    summary = {}
    for eps in cfg["clip_values"]:
        selected = [r for r in results if r["epsilon"] == eps]
        total = sum(r["valid_tokens"] for r in selected)
        summary[str(eps)] = {key: sum(r[key] * r["valid_tokens"] for r in selected) / max(total, 1)
                            for key in ("loss", "clip_fraction", "affected_token_fraction", "changed_surrogate_fraction")}
    atomic_json(repo_path(cfg["results_dir"]) / "cached_clipping.json", {"records": results, "summary": summary,
                "averaging": "valid-response-token weighted"})
    return summary


def main():
    from common.runtime import atomic_json, completed, numeric_stats
    from common.data import read_jsonl
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--epsilon", type=float)
    ap.add_argument("--cached-only", action="store_true")
    ap.add_argument("--forks-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if not args.forks_only:
        cached_geometry(cfg, load_cached_rollouts(cfg["cached_rollouts"]))
    if args.cached_only:
        return
    from task2_ppo.continue_train import run_ppo
    from common.evaluation import evaluate_policy
    results = {}
    for eps in ([args.epsilon] if args.epsilon is not None else cfg["clip_values"]):
        name = f"clip_{eps:.2f}"
        output = f"outputs/task2_ppo/{name}"
        if not completed(output):
            run_ppo(args.config, output, int(cfg["fork_updates"]), clip_epsilon=eps, run_name=name, resume=args.resume)
        metrics = evaluate_policy(args.config, output, name, "task2_ppo")
        logs = read_jsonl(repo_path(output) / "train.jsonl")
        metrics["stability_gradient_norm"] = numeric_stats([x["gradient_norm"] for x in logs])
        metrics["max_gradient_norm"] = max(x["gradient_norm"] for x in logs)
        results[name] = metrics
    atomic_json(f"{cfg['results_dir']}/clipping_summary{'_' + format(args.epsilon, '.2f') if args.epsilon is not None else ''}.json", results)

if __name__ == "__main__":
    main()
