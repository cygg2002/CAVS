from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from joint_core import (  # noqa: E402
    action_feature,
    balanced_binary_weights,
    candidate_matrix,
    choose_ap_safe_configuration,
    choose_reversible_configuration,
    reversible_fusion,
    select_joint_actions,
    validate_roles,
)


class JointCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = {
            "image_id": 7,
            "width": 1920,
            "height": 1080,
            "action_features": [float(value) for value in range(54)],
            "class_ids": [99],
            "qualities": {"edge": 0.1, "original": 0.9, "q90": 0.8},
            "payload_bytes": {"original": 1000, "q90": 400},
        }

    def test_roles_reject_final_and_require_frozen_order(self) -> None:
        self.assertEqual(
            validate_roles({"fit": 9, "calibration": 4, "audit": 8}),
            ("fit", "calibration", "audit"),
        )
        with self.assertRaisesRegex(ValueError, "final"):
            validate_roles({"fit": 9, "calibration": 4, "audit": 8, "final": 4})
        with self.assertRaisesRegex(ValueError, "roles"):
            validate_roles({"calibration": 4, "fit": 9, "audit": 8})

    def test_candidate_matrix_is_exactly_twelve_roi_actions(self) -> None:
        values = candidate_matrix(
            ("current", "moderate", "historical_cost", "fixed_1_8"),
            (160, 320, 640),
        )
        self.assertEqual(len(values), 12)
        self.assertEqual(values[0]["id"], "current@160")
        self.assertEqual(values[-1]["id"], "fixed_1_8@640")

    def test_action_feature_is_70d_and_does_not_read_gt_fields(self) -> None:
        record = {
            "area_ratio": 0.1,
            "crop_width": 640,
            "crop_height": 320,
            "image_width": 1920,
            "image_height": 1080,
        }
        first = action_feature(
            self.sample,
            action_id="moderate@320",
            payload_bytes=250,
            base_payload_bytes=400,
            record=record,
        )
        changed = copy.deepcopy(self.sample)
        changed["class_ids"] = [1, 2, 3]
        changed["qualities"] = {"edge": -99, "original": -99, "q90": -99}
        second = action_feature(
            changed,
            action_id="moderate@320",
            payload_bytes=250,
            base_payload_bytes=400,
            record=record,
        )
        self.assertEqual(len(first), 70)
        self.assertEqual(first, second)

    def test_reversible_fusion_keeps_edge_when_roi_evidence_is_weaker(self) -> None:
        edge = [
            {"image_id": 1, "category_id": 3, "bbox": [10, 10, 20, 20], "score": 0.8}
        ]
        cloud = [
            {"image_id": 1, "category_id": 3, "bbox": [11, 11, 20, 20], "score": 0.7}
        ]
        result = reversible_fusion(
            edge,
            cloud,
            match_iou=0.3,
            score_margin=0.0,
            add_threshold=0.1,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["_source"], "edge")
        self.assertEqual(result[0]["score"], 0.8)

    def test_reversible_fusion_conditionally_replaces_and_adds(self) -> None:
        edge = [
            {"image_id": 1, "category_id": 3, "bbox": [10, 10, 20, 20], "score": 0.6}
        ]
        cloud = [
            {"image_id": 1, "category_id": 3, "bbox": [11, 11, 20, 20], "score": 0.8},
            {"image_id": 1, "category_id": 4, "bbox": [50, 50, 10, 10], "score": 0.2},
        ]
        result = reversible_fusion(
            edge,
            cloud,
            match_iou=0.3,
            score_margin=0.0,
            add_threshold=0.1,
        )
        self.assertEqual({value["category_id"] for value in result}, {3, 4})
        self.assertTrue(all(value["_source"] == "roi" for value in result))

    def test_scheduler_respects_roi_budget_and_uses_base_fallback(self) -> None:
        options = {
            1: [
                {"action_id": "q90", "payload_bytes": 400, "predicted_gain": 0.0, "safe_probability": 1.0},
                {"action_id": "original", "payload_bytes": 1000, "predicted_gain": 0.0, "safe_probability": 1.0},
                {"action_id": "current@160", "payload_bytes": 200, "predicted_gain": 0.4, "safe_probability": 0.9},
            ],
            2: [
                {"action_id": "q90", "payload_bytes": 400, "predicted_gain": 0.0, "safe_probability": 1.0},
                {"action_id": "original", "payload_bytes": 1000, "predicted_gain": 0.0, "safe_probability": 1.0},
                {"action_id": "historical_cost@320", "payload_bytes": 220, "predicted_gain": 0.3, "safe_probability": 0.8},
            ],
            3: [
                {"action_id": "q90", "payload_bytes": 400, "predicted_gain": 0.0, "safe_probability": 1.0},
                {"action_id": "original", "payload_bytes": 1000, "predicted_gain": 0.0, "safe_probability": 1.0},
                {"action_id": "moderate@320", "payload_bytes": 210, "predicted_gain": 0.5, "safe_probability": 0.4},
            ],
        }
        selected = select_joint_actions(
            options,
            frozen_base={1: "q90", 2: "q90", 3: "q90"},
            roi_budget=1 / 3,
            safe_threshold=0.6,
            cost_weight=0.0,
            base_mode="frozen_q90_original",
        )
        self.assertEqual(sum("@" in action for action in selected.values()), 1)
        self.assertEqual(selected[1], "current@160")
        self.assertEqual(selected[2], "q90")
        self.assertEqual(selected[3], "q90")

    def test_calibration_chooses_lowest_kb_ap_safe_configuration(self) -> None:
        chosen = choose_ap_safe_configuration(
            [
                {"name": "base", "official_map50_95": 0.50, "request_kb_per_image": 100.0},
                {"name": "cheap-safe", "official_map50_95": 0.4991, "request_kb_per_image": 70.0},
                {"name": "too-low", "official_map50_95": 0.4989, "request_kb_per_image": 60.0},
                {"name": "safe-expensive", "official_map50_95": 0.501, "request_kb_per_image": 80.0},
            ],
            baseline_name="base",
            tolerance=0.001,
        )
        self.assertEqual(chosen["name"], "cheap-safe")

    def test_binary_weights_balance_labels_and_normalize_each_image(self) -> None:
        weights = balanced_binary_weights(
            labels=[True, False, False, True, False],
            image_ids=[1, 1, 2, 3, 3],
        )
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertAlmostEqual(
            sum(weight for weight, label in zip(weights, [True, False, False, True, False]) if label),
            0.5,
        )
        self.assertAlmostEqual(
            sum(weight for weight, label in zip(weights, [True, False, False, True, False]) if not label),
            0.5,
        )

    def test_reversible_configuration_maximizes_worst_budget_delta(self) -> None:
        chosen = choose_reversible_configuration(
            [
                {"config_id": "peaky", "budget": 0.1, "map_delta_vs_base": 0.04},
                {"config_id": "peaky", "budget": 0.4, "map_delta_vs_base": -0.02},
                {"config_id": "stable", "budget": 0.1, "map_delta_vs_base": 0.01},
                {"config_id": "stable", "budget": 0.4, "map_delta_vs_base": 0.005},
            ]
        )
        self.assertEqual(chosen["config_id"], "stable")
        self.assertEqual(chosen["worst_map_delta"], 0.005)


if __name__ == "__main__":
    unittest.main()
