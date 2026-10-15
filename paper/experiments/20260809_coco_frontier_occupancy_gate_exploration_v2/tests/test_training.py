from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "frontier_occupancy_training", ROOT / "scripts/training.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(ROOT / "scripts/training.py")
TRAINING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAINING
SPEC.loader.exec_module(TRAINING)
PROTOCOL = json.loads(
    (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
)


class OccupancyTrainingTests(unittest.TestCase):
    def test_roi_occupancy_loss_is_finite_and_backpropagates(self) -> None:
        batch, actions, boxes = 4, 4, 3
        data = {
            "boxes": torch.rand(batch, boxes, 6),
            "classes": torch.randint(0, 5, (batch, boxes)),
            "box_mask": torch.ones((batch, boxes), dtype=torch.bool),
            "scene": torch.rand(batch, 9),
            "actions": torch.rand(batch, actions, 18),
            "action_type": torch.tensor([[0, 1, 3, 2]]).expand(batch, -1),
            "action_mask": torch.ones((batch, actions), dtype=torch.bool),
            "occupancy_target": torch.tensor(
                [[False, True, True, True], [False, True, True, False]]
            ).repeat(2, 1),
        }
        model = TRAINING.build_model(
            category_count=5,
            scene_dim=9,
            action_dim=18,
            config=PROTOCOL["model"],
        )
        result = TRAINING.occupancy_objective(model, data)
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
