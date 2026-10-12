import unittest
import torch
from peft import get_peft_model
from transformers import Qwen2Config, Qwen2ForSequenceClassification
from common.models import make_value_lora_config, token_values


class ValueModel(unittest.TestCase):
    def test_peft_critic_uses_backbone_and_active_head(self):
        cfg = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=2, num_labels=1, pad_token_id=0)
        cfg.base_model_tp_plan = {}  # Single-device fixture; no tensor-parallel backend required.
        model = Qwen2ForSequenceClassification(cfg)
        model = get_peft_model(model, make_value_lora_config({"lora": {"r": 2, "alpha": 4, "dropout": 0., "target_modules": ["q_proj", "v_proj"]}}))
        # Original head can differ in dtype from the active trainable head.
        model.get_base_model().score.original_module.half()
        ids = torch.tensor([[0, 2, 3, 4]])
        values = token_values(model, ids, torch.tensor([[0, 1, 1, 1]]))
        self.assertEqual(tuple(values.shape), (1, 4))
        values[:, 1:].sum().backward()
        head = model.get_base_model().score.modules_to_save["default"]
        self.assertIsNotNone(head.weight.grad)


if __name__ == "__main__":
    unittest.main()
