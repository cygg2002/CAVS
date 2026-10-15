from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "frontier_training", ROOT / "scripts/training.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(ROOT / "scripts/training.py")
TRAINING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAINING
SPEC.loader.exec_module(TRAINING)
PROTOCOL = json.loads(
    (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
)


def synthetic_data() -> dict[str, torch.Tensor]:
    batch, actions, boxes = 4, 4, 3
    action_mask = torch.ones((batch, actions), dtype=torch.bool)
    return {
        "boxes": torch.rand(batch, boxes, 6),
        "classes": torch.randint(0, 5, (batch, boxes)),
        "box_mask": torch.ones((batch, boxes), dtype=torch.bool),
        "scene": torch.rand(batch, 9),
        "actions": torch.rand(batch, actions, 18),
        "action_type": torch.tensor([[0, 1, 2, 3]]).expand(batch, -1),
        "action_mask": action_mask,
        "target": torch.tensor(
            [
                [-1.0, 1.0, 0.5, 0.0],
                [-1.0, -0.2, 0.8, 0.3],
                [-1.0, 0.4, 0.5, 1.0],
                [-1.0, 0.7, 0.6, -0.1],
            ]
        ),
        "raw_target": torch.tensor(
            [
                [0.0, 0.8, 0.5, 0.2],
                [0.0, 0.1, 0.6, 0.3],
                [0.0, 0.4, 0.5, 0.8],
                [0.0, 0.7, 0.6, -0.1],
            ]
        ),
        "cost_ratio": torch.tensor(
            [[0.0, 1.0, 0.25, 0.1]]
        ).expand(batch, -1),
    }


class TrainingObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = synthetic_data()
        self.model = TRAINING.build_model(
            category_count=5,
            scene_dim=9,
            action_dim=18,
            config=PROTOCOL["model"],
        )

    def test_all_screen_methods_are_finite_and_backpropagate(self) -> None:
        for method in PROTOCOL["screen_methods"]:
            self.model.zero_grad(set_to_none=True)
            result = TRAINING.training_objective(
                self.model,
                method,
                self.data,
                reward_mean=0.2,
                reward_std=0.2,
                config=PROTOCOL["model"],
            )
            self.assertTrue(torch.isfinite(result["loss"]))
            result["loss"].backward()
            self.assertTrue(
                any(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    for parameter in self.model.parameters()
                )
            )

    def test_current_actionvalue_has_no_frontier_penalty(self) -> None:
        result = TRAINING.training_objective(
            self.model,
            "CurrentActionValue",
            self.data,
            reward_mean=0.2,
            reward_std=0.2,
            config=PROTOCOL["model"],
        )
        self.assertEqual(float(result["frontier_loss"]), 0.0)

    def test_frontier_methods_add_positive_path_loss(self) -> None:
        result = TRAINING.training_objective(
            self.model,
            "BoundaryPathDistill",
            self.data,
            reward_mean=0.2,
            reward_std=0.2,
            config=PROTOCOL["model"],
        )
        self.assertGreater(float(result["frontier_loss"]), 0.0)


class TensorAssemblyTests(unittest.TestCase):
    def test_cost_stripping_accepts_roi_and_preserves_action_identity(self) -> None:
        data = synthetic_data()
        original_types = data["action_type"].clone()
        stripped = TRAINING.strip_cost_inputs(data)
        self.assertTrue(torch.equal(stripped["action_type"], original_types))
        self.assertTrue(
            torch.equal(
                stripped["actions"][:, :, list(TRAINING.VALUE_CORE.ACTION_COST_INDICES)],
                torch.zeros(
                    (4, 4, len(TRAINING.VALUE_CORE.ACTION_COST_INDICES))
                ),
            )
        )

    def test_reward_and_cost_tensors_follow_record_action_order(self) -> None:
        records = [
            {
                "image_id": 7,
                "full_payload_bytes": 1000,
                "actions": [
                    {"action_id": "edge", "payload_bytes": 0},
                    {"action_id": "q90", "payload_bytes": 200},
                    {"action_id": "roi", "payload_bytes": 100},
                ],
            }
        ]
        rewards, costs = TRAINING.reward_and_cost_tensors(
            records,
            {7: {"q90": 0.4, "roi": 0.3}},
            max_actions=4,
            cost_scale_bytes=1000.0,
        )
        self.assertEqual(rewards.tolist(), [[0.0, 0.4, 0.3, 0.0]])
        self.assertEqual(costs.tolist(), [[0.0, 0.2, 0.1, 0.0]])

    def test_missing_cloud_reward_is_rejected(self) -> None:
        records = [
            {
                "image_id": 7,
                "full_payload_bytes": 1000,
                "actions": [
                    {"action_id": "edge", "payload_bytes": 0},
                    {"action_id": "q90", "payload_bytes": 200},
                ],
            }
        ]
        with self.assertRaises(ValueError):
            TRAINING.reward_and_cost_tensors(
                records, {7: {}}, max_actions=2, cost_scale_bytes=1000.0
            )

    def test_cost_scale_is_shared_across_images(self) -> None:
        records = [
            {
                "image_id": image_id,
                "full_payload_bytes": full_bytes,
                "actions": [
                    {"action_id": "edge", "payload_bytes": 0},
                    {"action_id": "q90", "payload_bytes": 200},
                ],
            }
            for image_id, full_bytes in ((1, 1000), (2, 2000))
        ]
        _, costs = TRAINING.reward_and_cost_tensors(
            records,
            {1: {"q90": 0.1}, 2: {"q90": 0.1}},
            max_actions=2,
            cost_scale_bytes=1000.0,
        )
        self.assertEqual(costs[:, 1].tolist(), [0.2, 0.2])


if __name__ == "__main__":
    unittest.main()
