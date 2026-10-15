import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTest(unittest.TestCase):
    def test_frozen_training_recipe(self):
        protocol = json.loads((ROOT / "configs" / "protocol.json").read_text())
        train = protocol["train"]
        self.assertEqual(train["epochs"], 30)
        self.assertEqual(train["batch"], 8)
        self.assertEqual(train["imgsz"], 640)
        self.assertEqual(train["optimizer"], "auto")
        self.assertEqual(train["seed"], 0)
        self.assertFalse(train["resume"])
        self.assertFalse(train["exist_ok"])

    def test_source_is_cloudv2_best(self):
        protocol = json.loads((ROOT / "configs" / "protocol.json").read_text())
        self.assertTrue(protocol["source_checkpoint"].endswith("finetune20_batch8_from_e18_restart2/weights/best.pt"))
        self.assertEqual(len(protocol["source_sha256"]), 64)

    def test_runner_imports_without_ultralytics(self):
        path = ROOT / "scripts" / "train_cloudv3.py"
        spec = importlib.util.spec_from_file_location("train_cloudv3", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.load_and_validate_protocol))


if __name__ == "__main__":
    unittest.main()
