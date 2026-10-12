import json
import tempfile
from pathlib import Path
import unittest
import torch
from common.runtime import Run, optimizer_step


class Checkpointing(unittest.TestCase):
    def train(self, output, stop, resume=False):
        torch.manual_seed(17)
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        run = Run(output, {"seed": 17, "updates": 3}, [{"source_index": 7}], "test", resume)
        progress = run.restore({"policy": model}, {"policy": optimizer}, scaler)
        for index in range(progress["update"], stop):
            optimizer.zero_grad()
            loss = model(torch.randn(4, 3)).square().mean()
            scaler.scale(loss).backward()
            norm = optimizer_step(model, optimizer, scaler, 1.)
            scaler.update()
            run.save({"policy": model}, {"policy": optimizer}, scaler,
                     {"update": index + 1, "cursor": index + 1, "epoch": 0}, {"loss": float(loss.detach()), "gradient_norm": norm})
        return model.state_dict()

    def test_resume_matches_uninterrupted_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            full = self.train(Path(directory) / "full", 3)
            partial_path = Path(directory) / "resumed"
            self.train(partial_path, 1)
            # Simulate a trailing log written before a crash and checkpoint replacement.
            with (partial_path / "train.jsonl").open("a") as file:
                file.write(json.dumps({"update": 2, "loss": 123}) + "\n")
            resumed = self.train(partial_path, 3, True)
            for key in full:
                torch.testing.assert_close(full[key], resumed[key], rtol=0, atol=0)
            updates = [json.loads(line)["update"] for line in (partial_path / "train.jsonl").read_text().splitlines()]
            self.assertEqual(updates, [1, 2, 3])

    def test_resume_rejects_changed_config(self):
        with tempfile.TemporaryDirectory() as directory:
            self.train(Path(directory), 1)
            with self.assertRaises(ValueError):
                Run(directory, {"seed": 18, "updates": 3}, [{"source_index": 7}], "test", True)


if __name__ == "__main__":
    unittest.main()
