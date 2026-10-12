import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from peft import get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from common.models import make_lora_config, trainable_parameters
from common.runtime import Run
from task1_dpo.train import run_training


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        return [2, 3]

    def __call__(self, text, **kwargs):
        return {"input_ids": [4 if text == "chosen" else 5]}

    def save_pretrained(self, output):
        (Path(output) / "tokenizer_config.json").write_text("{}")


class DPOTraining(unittest.TestCase):
    def bundle(self, directory):
        torch.manual_seed(17)
        cfg = {"seed": 17, "epochs": 1, "beta": .1, "batch_size": 2, "grad_accum_steps": 2,
               "max_sequence_length": 8, "max_grad_norm": 1., "paths": {"dpo_standard_train": "fixture"},
               "results_dir": str(Path(directory) / "results"),
               "lora": {"r": 2, "alpha": 4, "dropout": 0., "target_modules": ["q_proj", "v_proj"]}}
        mc = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, pad_token_id=0, eos_token_id=1)
        mc.base_model_tp_plan = {}
        model = get_peft_model(Qwen2ForCausalLM(mc), make_lora_config(cfg))
        optimizer = torch.optim.AdamW(trainable_parameters(model), lr=.01)
        rows = [{"source_index": i, "prompt": "p", "chosen": "chosen", "rejected": "rejected"} for i in range(5)]
        return {"cfg": cfg, "model": model, "tokenizer": Tokenizer(), "optimizer": optimizer,
                "rows": rows, "beta": .1}

    def test_partial_accumulation_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.bundle(directory)
            with patch("task1_dpo.train.prepare_dpo_run", return_value=bundle):
                full = run_training("unused", "fixture", output_path=str(Path(directory) / "full"))
            full_state = {n: p.detach().clone() for n, p in bundle["model"].named_parameters() if p.requires_grad}
            original_save = Run.save
            def interrupted_save(run, *args, **kwargs):
                original_save(run, *args, **kwargs)
                raise RuntimeError("simulated interruption")
            partial = Path(directory) / "resumed"
            with patch("task1_dpo.train.prepare_dpo_run", return_value=self.bundle(directory)), patch.object(Run, "save", interrupted_save):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    run_training("unused", "fixture", output_path=str(partial))
            resumed_bundle = self.bundle(directory)
            with patch("task1_dpo.train.prepare_dpo_run", return_value=resumed_bundle):
                run_training("unused", "fixture", output_path=str(partial), resume=True)
            for name, param in resumed_bundle["model"].named_parameters():
                if name in full_state:
                    torch.testing.assert_close(param, full_state[name], rtol=0, atol=0)
            logs = [json.loads(line) for line in (partial / "train.jsonl").read_text().splitlines()]
            self.assertEqual([row["examples"] for row in logs], [4, 1])
            self.assertEqual(json.loads((full / "run.json").read_text())["training_examples"], 5)


if __name__ == "__main__":
    unittest.main()
