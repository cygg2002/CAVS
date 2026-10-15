from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from action_core import (  # noqa: E402
    build_action_records,
    deterministic_folds,
    evaluate_gate,
    numeric_score_arrays,
    schedule_fixed_byte,
    schedule_fixed_call,
    validate_protocol,
)
from value_risk import (  # noqa: E402
    ActionValueRiskRouter,
    value_risk_objective,
)


class ProtocolTests(unittest.TestCase):
    def test_frozen_protocol_is_valid(self) -> None:
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        validate_protocol(protocol)
        self.assertEqual(protocol["model"]["pairs_per_budget"], 4000)
        self.assertEqual(protocol["model"]["boundary_weight"], 1.0)

    def test_canonical_dualhead_boundary_contract_is_required(self) -> None:
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        del protocol["model"]["pairs_per_budget"]
        with self.assertRaisesRegex(ValueError, "boundary-pair count"):
            validate_protocol(protocol)

    def test_forbidden_role_is_rejected(self) -> None:
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        protocol["roles"]["final"] = 5000
        with self.assertRaisesRegex(ValueError, "fit and calibration"):
            validate_protocol(protocol)


class FoldTests(unittest.TestCase):
    def test_folds_are_deterministic_balanced_and_exhaustive(self) -> None:
        first = deterministic_folds(range(23), fold_count=5, seed=7)
        second = deterministic_folds(range(23), fold_count=5, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(sorted(value for fold in first for value in fold), list(range(23)))
        self.assertLessEqual(max(map(len, first)) - min(map(len, first)), 1)


class RecordTests(unittest.TestCase):
    def test_record_has_dedicated_q90_token_and_excludes_unavailable_roi(self) -> None:
        samples = [
            {
                "image_id": 1,
                "width": 100,
                "height": 80,
                "payload_bytes": {"original": 1000, "q90": 400, "roi": None},
                "available": {"original": True, "q90": True, "roi": False},
                "roi_area_ratio": 0.0,
            },
            {
                "image_id": 2,
                "width": 200,
                "height": 100,
                "payload_bytes": {"original": 2000, "q90": 800, "roi": 300},
                "available": {"original": True, "q90": True, "roi": True},
                "roi_area_ratio": 0.25,
            },
        ]
        predictions = {
            1: [{"category_id": 3, "bbox": [10, 20, 30, 40], "score": 0.8}],
            2: [],
        }
        records = build_action_records(samples, predictions, category_ids=[1, 3])
        self.assertEqual([a["action_id"] for a in records[0]["actions"]], ["edge", "original", "q90"])
        self.assertEqual(records[0]["actions"][2]["action_type"], "pair")
        self.assertEqual(records[0]["edge_boxes"][0]["cls"], 1)
        roi = next(a for a in records[1]["actions"] if a["action_id"] == "roi")
        self.assertEqual(roi["action_type"], "roi")
        self.assertAlmostEqual(records[1]["regions"][0]["area_ratio"], 0.25)


class ScoreArtifactTests(unittest.TestCase):
    def test_checkpoint_metadata_is_not_coerced_to_numeric_scores(self) -> None:
        artifact = {
            "scores": {
                "baseline": [1.0, 2.0],
                "value": [[0.0, 1.0], [0.0, 2.0]],
                "risk_mean": [[0.0, 1.0], [0.0, 2.0]],
                "risk_lower": [[0.0, 0.5], [0.0, 1.5]],
                "selected_epochs": {"baseline": 12, "value": 20},
            }
        }
        scores = numeric_score_arrays(artifact)
        self.assertEqual(
            set(scores), {"baseline", "value", "risk_mean", "risk_lower"}
        )
        self.assertEqual(scores["baseline"].dtype, np.float64)


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [
            {"image_id": 1, "payload_bytes": {"original": 10, "q90": 4, "roi": None}, "available": {"original": True, "q90": True, "roi": False}},
            {"image_id": 2, "payload_bytes": {"original": 10, "q90": 5, "roi": 2}, "available": {"original": True, "q90": True, "roi": True}},
            {"image_id": 3, "payload_bytes": {"original": 10, "q90": 6, "roi": 3}, "available": {"original": True, "q90": True, "roi": True}},
        ]
        self.scores = {
            1: {"original": 1.0, "q90": 2.0},
            2: {"original": 1.0, "q90": 1.5, "roi": 3.0},
            3: {"original": 1.0, "q90": 0.5, "roi": 1.5},
        }

    def test_fixed_call_uses_exact_count_and_one_action_per_image(self) -> None:
        result = schedule_fixed_call(self.samples, self.scores, budget=2 / 3)
        self.assertEqual(result["selected_images"], 2)
        self.assertEqual(result["selection"], {1: "q90", 2: "roi"})

    def test_fixed_byte_respects_capacity_and_availability(self) -> None:
        result = schedule_fixed_byte(self.samples, self.scores, capacity_bytes=6)
        self.assertLessEqual(result["total_cost_bytes"], 6)
        self.assertNotEqual(result["selection"].get(1), "roi")
        self.assertEqual(result["selection"], {1: "q90", 2: "roi"})


class ModelTests(unittest.TestCase):
    def test_value_risk_router_outputs_mean_and_lower_quantile_per_action(self) -> None:
        model = ActionValueRiskRouter(
            category_count=3,
            scene_dim=9,
            action_dim=18,
            class_embedding_dim=4,
            box_hidden_dim=8,
            action_hidden_dim=8,
            joint_hidden_dim=12,
            dropout=0.0,
        )
        output = model(
            boxes=torch.zeros(2, 3, 6),
            classes=torch.zeros(2, 3, dtype=torch.long),
            box_mask=torch.tensor([[True, False, False], [True, True, False]]),
            scene=torch.zeros(2, 9),
            actions=torch.zeros(2, 4, 18),
            action_type=torch.tensor([[0, 1, 3, 2], [0, 1, 3, 2]]),
            action_mask=torch.tensor([[True, True, True, False], [True, True, True, True]]),
        )
        self.assertEqual(tuple(output["mean"].shape), (2, 4))
        self.assertEqual(tuple(output["lower"].shape), (2, 4))
        self.assertLess(float(output["lower"][0, 3].detach()), -1e8)

    def test_quantile_objective_penalizes_crossing_and_backpropagates(self) -> None:
        mean = torch.tensor([[0.0, 0.5, -0.1]], requires_grad=True)
        lower = torch.tensor([[0.0, 0.8, -0.2]], requires_grad=True)
        target = torch.tensor([[0.0, 0.2, -0.4]])
        mask = torch.tensor([[True, True, True]])
        result = value_risk_objective(
            mean,
            lower,
            target,
            mask,
            action_type=torch.tensor([[0, 1, 2]]),
            quantile=0.1,
            reward_weight=0.5,
            within_weight=0.5,
            quantile_weight=0.5,
            crossing_weight=0.25,
        )
        self.assertGreater(float(result["crossing_loss"].detach()), 0.0)
        result["loss"].backward()
        self.assertIsNotNone(mean.grad)
        self.assertIsNotNone(lower.grad)


class GateTests(unittest.TestCase):
    def test_gate_requires_fixed_byte_gain_and_fixed_call_noninferiority(self) -> None:
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        rows = [
            {"mode": "fixed_byte", "budget": 0.1, "delta_ap": 0.0012, "capacity_ok": True},
            {"mode": "fixed_byte", "budget": 0.25, "delta_ap": 0.0011, "capacity_ok": True},
            {"mode": "fixed_byte", "budget": 0.4, "delta_ap": -0.0009, "capacity_ok": True},
            {"mode": "fixed_call", "budget": 0.1, "delta_ap": 0.0, "capacity_ok": True},
            {"mode": "fixed_call", "budget": 0.25, "delta_ap": -0.0009, "capacity_ok": True},
            {"mode": "fixed_call", "budget": 0.4, "delta_ap": 0.0, "capacity_ok": True},
        ]
        self.assertTrue(evaluate_gate(rows, protocol)["passed"])
        rows[-1]["delta_ap"] = -0.0011
        self.assertFalse(evaluate_gate(rows, protocol)["passed"])


if __name__ == "__main__":
    unittest.main()
