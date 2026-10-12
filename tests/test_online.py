"""Exercise complete updates on tiny randomly initialized models without downloads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from peft import get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2ForSequenceClassification
from common.models import make_lora_config, make_value_lora_config, trainable_parameters
from common.online import run_online


class Tokenizer:
    def save_pretrained(self, output):
        (Path(output) / "tokenizer_config.json").write_text("{}")


class OnlineUpdates(unittest.TestCase):
    def bundle(self, directory, algorithm):
        cfg = {"seed": 17, "updates": 1, "prompts_per_update": 1, "num_generations": 4,
               "max_prompt_length": 8, "max_response_length": 4, "max_completion_length": 4,
               "generation": {"temperature": .7, "top_p": .9, "do_sample": True},
               "ppo_epochs": 2, "policy_epochs": 1, "clip_epsilon": .2, "kl_beta": .1,
               "max_grad_norm": 1., "gamma": 1., "gae_lambda": .95, "value_coef": .5,
               "missing_eos_penalty": 1., "mask_truncated_completions": True,
               "results_dir": str(Path(directory) / "results"),
               "lora": {"r": 2, "alpha": 4, "dropout": 0., "target_modules": ["q_proj", "v_proj"]}}
        mc = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, pad_token_id=0, eos_token_id=1, num_labels=1)
        mc.base_model_tp_plan = {}
        policy = get_peft_model(Qwen2ForCausalLM(mc), make_lora_config(cfg))
        optimizer = torch.optim.AdamW(trainable_parameters(policy), lr=.01)
        bundle = {"cfg": cfg, "policy": policy, "tokenizer": Tokenizer(), "prompt_rows": [{"source_index": 7, "prompt": "p"}],
                  "reward_model": None, "reward_tokenizer": None, "optimizer": optimizer, "policy_optimizer": optimizer}
        if algorithm == "ppo":
            value = get_peft_model(Qwen2ForSequenceClassification(mc), make_value_lora_config(cfg))
            bundle.update(value_model=value, value_optimizer=torch.optim.AdamW(trainable_parameters(value), lr=.01))
        n = 4 if algorithm == "grpo" else 1
        ids = torch.tensor([[2, 3, 4 + (i % 2), 1] for i in range(n)])
        generation = {"sequences": ids, "attention_mask": torch.ones_like(ids), "prompt_width": 2,
                      "response_ids": ids[:, 2:], "response_mask": torch.ones(n, 2),
                      "responses": [str(i) for i in range(n)], "response_lengths": [2] * n,
                      "terminated_with_eos": [True] * n, "truncated": [False] * n}
        return bundle, generation

    def check_update(self, algorithm, loss_type="grpo"):
        with tempfile.TemporaryDirectory() as directory:
            bundle, gen = self.bundle(directory, algorithm)
            frozen = {name: p.detach().clone() for name, p in bundle["policy"].named_parameters() if not p.requires_grad}
            before = {name: p.detach().clone() for name, p in bundle["policy"].named_parameters() if p.requires_grad}
            with patch("common.online.batch_generate", return_value=gen), patch("common.online.score_reward_pairs", return_value=torch.arange(len(gen["responses"]), dtype=torch.float32) + 1):
                output = run_online(bundle, Path(directory) / "adapter", "fixture", algorithm, loss_type=loss_type)
            metadata = json.loads((output / "run.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            log = json.loads((output / "train.jsonl").read_text())
            self.assertEqual(log["update"], 1)
            self.assertTrue(torch.isfinite(torch.tensor(log["policy_loss"])))
            self.assertTrue(any(not torch.equal(before[name], p.detach()) for name, p in bundle["policy"].named_parameters() if name in before))
            for name, param in bundle["policy"].named_parameters():
                if name in frozen:
                    torch.testing.assert_close(param.detach(), frozen[name], rtol=0, atol=0)
            if algorithm == "ppo":
                self.assertTrue((output / "value_adapter" / "adapter_config.json").exists())
            else:
                self.assertEqual(log["informative_group_rate"], 1)
                self.assertIn("length_conditioned_gradient", log)

    def test_ppo_rollout_actor_and_critic_update(self):
        self.check_update("ppo")

    def test_grpo_rollout_update(self):
        self.check_update("grpo")

    def test_dr_grpo_rollout_update(self):
        self.check_update("grpo", "dr_grpo")


if __name__ == "__main__":
    unittest.main()
