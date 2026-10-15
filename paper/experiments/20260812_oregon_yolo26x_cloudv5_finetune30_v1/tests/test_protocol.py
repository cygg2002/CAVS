import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V4_ROOT = ROOT.parent / "20260810_oregon_yolo26x_cloudv4_finetune30_v1"


class ProtocolTest(unittest.TestCase):
    def test_training_recipe_exactly_matches_cloudv4(self):
        protocol = json.loads((ROOT / "configs" / "protocol.json").read_text())
        cloudv4 = json.loads((V4_ROOT / "configs" / "protocol.json").read_text())
        self.assertEqual(protocol["train"], cloudv4["train"])

    def test_source_is_cloudv4_best_and_destination_is_isolated(self):
        protocol = json.loads((ROOT / "configs" / "protocol.json").read_text())
        self.assertTrue(
            protocol["source_checkpoint"].endswith(
                "finetune30_batch8_from_cloudv3_best_v1/weights/best.pt"
            )
        )
        self.assertEqual(
            protocol["source_sha256"],
            "8beb7c7e49c5a0ec60222723816f9c943f59a24d902c451be3e5c6910bffbacb",
        )
        self.assertEqual(protocol["source_best_ap50_95"], 0.55068)
        self.assertEqual(
            protocol["name"], "finetune30_batch8_from_cloudv4_best_v1"
        )
        self.assertNotEqual(
            protocol["name"], "finetune30_batch8_from_cloudv3_best_v1"
        )

    def test_runner_imports_without_ultralytics(self):
        path = ROOT / "scripts" / "train_cloudv5.py"
        spec = importlib.util.spec_from_file_location("train_cloudv5", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.load_and_validate_protocol))


if __name__ == "__main__":
    unittest.main()
