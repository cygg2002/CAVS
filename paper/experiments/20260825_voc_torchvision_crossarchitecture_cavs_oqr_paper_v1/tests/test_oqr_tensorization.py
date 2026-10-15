from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DATA = load("voc_oqr_tensor_data", ROOT / "scripts/data_core.py")
ADAPTER = load("voc_oqr_tensor_adapter", ROOT / "scripts/training_adapter.py")
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


class OqrTensorizationTest(unittest.TestCase):
    def test_four_actions_reach_the_frozen_learner_with_distinct_types(self) -> None:
        samples = [
            {
                "image_id": 1,
                "identity": "2007:000001",
                "width": 100,
                "height": 80,
                "payload_bytes": {
                    "original": 1000,
                    "q90": 600,
                    "current@official": 250,
                },
                "available": {
                    "original": True,
                    "q90": True,
                    "current@official": True,
                },
                "roi": {"xyxy": [5, 10, 70, 75]},
            }
        ]
        edge = {
            1: [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 20, 30, 40],
                    "score": 0.9,
                }
            ]
        }
        records = DATA.build_action_records(
            samples, edge, category_ids=list(range(1, 21)), action_space="OQR"
        )
        tensors = ADAPTER._SOURCE.tensorize(records, None, PROTOCOL)
        self.assertEqual(tuple(tensors["actions"].shape), (1, 4, 18))
        self.assertEqual(tensors["action_mask"][0].tolist(), [True, True, True, True])
        self.assertEqual(tensors["action_type"][0].tolist(), [0, 1, 3, 2])


if __name__ == "__main__":
    unittest.main()
