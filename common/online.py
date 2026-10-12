"""Shared rollout plumbing; PPO and GRPO keep distinct update objectives."""
from __future__ import annotations

import random
import torch

from common.data import prompt_messages
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.metrics import masked_mean, sampled_kl, sample_entropy
from common.models import reference_mode, token_values
from common.runtime import Run, identifier, numeric_stats, optimizer_step, scaler_for
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def rollout_logprobs(model, generation):
    if torch.is_grad_enabled() and generation["sequences"].shape[0] > 1:
        from torch.utils.checkpoint import checkpoint
        # Recompute one response at a time during backward. The loss still uses
        # the complete group, with its original advantages and normalization.
        def score(sequence, attention, response):
            return response_token_logprobs(model, sequence, attention,
                                          generation["prompt_width"], response)[0]
        chosen = [checkpoint(score, generation["sequences"][i:i + 1],
                             generation["attention_mask"][i:i + 1],
                             generation["response_ids"][i:i + 1], use_reentrant=False)
                  for i in range(generation["sequences"].shape[0])]
        return torch.cat(chosen, dim=0), None
    chosen, logits = response_token_logprobs(model, generation["sequences"], generation["attention_mask"],
                                            generation["prompt_width"], generation["response_ids"])
    return chosen, logits


def entropy_from_logits(logits, mask):
    # Chunk the vocabulary-sized calculation along the response dimension to limit temporary memory.
    total = torch.zeros((), device=logits.device)
    for start in range(0, logits.shape[1], 32):
        logp = torch.log_softmax(logits[:, start:start + 32].float(), -1)
        total += (-(logp.exp() * logp).sum(-1) * mask[:, start:start + 32]).sum()
    return total / mask.sum().clamp_min(1)


def response_values(model, generation):
    values = token_values(model, generation["sequences"], generation["attention_mask"])
    width = generation["prompt_width"]
    return values[:, width - 1:width - 1 + generation["response_ids"].shape[1]]


def run_online(bundle, output, run_name, algorithm, resume=False, loss_type="grpo"):
    cfg, policy, tokenizer = (bundle[k] for k in ("cfg", "policy", "tokenizer"))
    cfg = {**cfg, "algorithm": algorithm, "loss_type": loss_type if algorithm == "grpo" else None}
    rows = bundle["prompt_rows"]
    if not rows:
        raise ValueError("Prompt pool is empty")
    models = {"policy": policy}
    optimizers = {"policy": bundle["policy_optimizer" if algorithm == "ppo" else "optimizer"]}
    if algorithm == "ppo":
        models["value"] = bundle["value_model"]
        optimizers["value"] = bundle["value_optimizer"]
    run = Run(output, cfg, rows, run_name, resume)
    scaler = scaler_for(policy)
    progress = run.restore(models, optimizers, scaler)
    # Dropout must not change importance ratios between old/new scoring of identical actions.
    policy.eval()
    if algorithm == "ppo":
        models["value"].eval()
    device = next(policy.parameters()).device
    order = list(range(len(rows)))
    random.Random(int(cfg["seed"])).shuffle(order)
    total_tokens = 0
    if run.state:
        total_tokens = progress.get("generated_tokens", 0)
    for update in range(progress["update"], int(cfg["updates"])):
        n = int(cfg["prompts_per_update"])
        indices = [order[(update * n + j) % len(order)] for j in range(n)]
        k = int(cfg["num_generations"]) if algorithm == "grpo" else 1
        prompts = [prompt_messages(rows[i]) for i in indices for _ in range(k)]
        cap = int(cfg["max_completion_length"] if algorithm == "grpo" else cfg["max_response_length"])
        generation = batch_generate(policy, tokenizer, prompts, int(cfg["max_prompt_length"]), cap,
                                    **cfg["generation"])
        # generate uses inference_mode: clone its tensors before gradient-enabled scoring.
        generation = {key: value.clone() if torch.is_tensor(value) else value
                      for key, value in generation.items()}
        mask = generation["response_mask"]
        with torch.no_grad():
            old, old_logits = rollout_logprobs(policy, generation)
            entropy = float(entropy_from_logits(old_logits, mask))
            del old_logits
            with reference_mode(policy):
                ref, logits = rollout_logprobs(policy, generation)
                del logits
            reward = score_reward_pairs(bundle["reward_model"], bundle["reward_tokenizer"], prompts,
                                        generation["responses"], int(cfg.get("reward_max_length", 1024))).to(device)
        records = []
        lengths = generation["response_lengths"]
        total_tokens += sum(lengths)
        diagnostics = {"reward": float(reward.mean()), "kl": float(sampled_kl(old, ref, mask)),
                       "entropy": entropy, "sample_entropy": float(sample_entropy(old, mask)),
                       "response_length": numeric_stats(lengths), "generated_tokens": total_tokens,
                       "prompt_ids": [identifier(rows[i], i) for i in indices],
                       "truncated_fraction": sum(generation["truncated"]) / len(lengths)}
        if algorithm == "ppo":
            task_reward = reward - float(cfg.get("missing_eos_penalty", 0)) * torch.tensor(
                [not x for x in generation["terminated_with_eos"]], device=device)
            with torch.no_grad():
                old_values = response_values(models["value"], generation).to(device)
                shaped = shaped_rewards(task_reward, old, ref, mask, cfg["kl_beta"])
                advantages, returns = compute_gae(shaped, old_values, mask, cfg["gamma"], cfg["gae_lambda"])
                advantages = normalize_advantages(advantages, mask)
            for _ in range(int(cfg["ppo_epochs"])):
                optimizers["policy"].zero_grad(set_to_none=True)
                new, logits = rollout_logprobs(policy, generation)
                del logits
                loss, ratio, clip = ppo_policy_loss(new, old, advantages, mask, cfg["clip_epsilon"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO loss")
                scaler.scale(loss).backward()
                policy_norm = optimizer_step(policy, optimizers["policy"], scaler, cfg["max_grad_norm"])
                optimizers["value"].zero_grad(set_to_none=True)
                predicted = response_values(models["value"], generation).to(device)
                value_loss = value_mse_loss(predicted, returns, mask)
                scaler.scale(float(cfg["value_coef"]) * value_loss).backward()
                value_norm = optimizer_step(models["value"], optimizers["value"], scaler, cfg["max_grad_norm"])
                scaler.update()
                records.append({"policy_loss": float(loss.detach()), "value_loss": float(value_loss.detach()),
                                "gradient_norm": policy_norm, "value_gradient_norm": value_norm,
                                "clip_fraction": float(clip)})
            diagnostics["critic_return_mse"] = float(value_mse_loss(old_values, returns, mask))
        else:
            group_ids = torch.arange(n, device=device).repeat_interleave(k)
            advantages = group_relative_advantages(reward, group_ids)
            stds = [float(reward[group_ids == gid].std(unbiased=False)) for gid in range(n)]
            training_mask = mask_truncated_sequences(mask, generation["truncated"]) if cfg.get("mask_truncated_completions") else mask
            diagnostics.update({"group_reward_std": sum(stds) / len(stds),
                                "uninformative_group_fraction": sum(x <= 1e-6 for x in stds) / len(stds),
                                "informative_group_rate": sum(x > 1e-6 for x in stds) / len(stds)})
            for _ in range(int(cfg["policy_epochs"])):
                optimizers["policy"].zero_grad(set_to_none=True)
                new, logits = rollout_logprobs(policy, generation)
                del logits
                loss, details = grpo_policy_loss(new, old, advantages, training_mask, ref,
                                                cfg["clip_epsilon"], cfg["kl_beta"], loss_type, cap)
                # Directly measure d(policy loss)/d(token logp), grouped by response length.
                score_gradient = torch.autograd.grad(loss, new, retain_graph=True)[0].detach()
                lengths_tensor = mask.sum(-1)
                median_length = lengths_tensor.median()
                conditioned = {}
                for label, selected in (("short", lengths_tensor <= median_length), ("long", lengths_tensor > median_length)):
                    conditioned[label] = {"count": int(selected.sum()),
                        "mean_length": float(lengths_tensor[selected].mean()) if selected.any() else None,
                        "mean_abs_logp_gradient_sum": float(score_gradient[selected].abs().sum(-1).mean()) if selected.any() else None}
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite GRPO loss")
                scaler.scale(loss).backward()
                norm = optimizer_step(policy, optimizers["policy"], scaler, cfg["max_grad_norm"])
                scaler.update()
                records.append({"policy_loss": float(loss.detach()), "gradient_norm": norm,
                                "clip_fraction": float(details["clip_fraction"]),
                                "training_kl_penalty": float(details["sampled_kl"])})
                diagnostics["length_conditioned_gradient"] = conditioned
        for key in records[0]:
            diagnostics[key] = sum(record[key] for record in records) / len(records)
        diagnostics["rollouts"] = [{"prompt_id": identifier(rows[i], i), "response": generation["responses"][j],
            "reward": float(reward[j]), "response_tokens": lengths[j], "truncated": generation["truncated"][j]}
            for j, i in enumerate([i for i in indices for _ in range(k)])]
        progress = {"update": update + 1, "epoch": 0, "cursor": (update + 1) * n, "generated_tokens": total_tokens}
        run.save(models, optimizers, scaler, progress, diagnostics)
    if algorithm == "ppo":
        models["value"].save_pretrained(run.output / "value_adapter")
    run.finish(policy, tokenizer, updates=int(cfg["updates"]), generated_tokens=total_tokens)
    return run.output
