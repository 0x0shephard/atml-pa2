import base64
import io
import json
from pathlib import Path
import unittest
import zipfile
from scripts.kaggle_runner import read_manifest, source_snapshot, notebook_source


class KaggleRunner(unittest.TestCase):
    def test_manifest_and_generated_notebooks(self):
        manifest = read_manifest()
        self.assertGreater(len(manifest["runs"]), 25)
        payload, digest = source_snapshot()
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
            names = archive.namelist()
            self.assertIn("data/word_limit_prompts.jsonl", names)
            self.assertIn("common/runtime.py", names)
            self.assertFalse(any("credentials" in name or ".venv" in name or "outputs/" in name for name in names))
        for run in manifest["runs"]:
            compile(notebook_source(run, payload, digest), run["id"], "exec")

    def test_declared_modules_exist(self):
        root = Path(__file__).resolve().parents[1]
        for run in read_manifest()["runs"]:
            for command in run["commands"]:
                self.assertEqual(command[0], "-m")
                self.assertTrue((root / (command[1].replace(".", "/") + ".py")).exists(), command)


if __name__ == "__main__":
    unittest.main()
