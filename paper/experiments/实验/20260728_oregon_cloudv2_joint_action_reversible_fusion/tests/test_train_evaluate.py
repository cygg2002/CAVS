from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_evaluate  # noqa: E402
from train_evaluate import (  # noqa: E402
    action_rows,
    actions_from_choice,
    compare_class_ap,
    json_sha256,
    primary_candidate_rows,
    validate_hash_inventory,
)


class TrainEvaluateTest(unittest.TestCase):
    def test_action_rows_build_base_and_roi_targets_against_frozen_base(self) -> None:
        sample = {
            "image_id": 1,
            "width": 100,
            "height": 100,
            "action_features": [0.0] * 54,
            "payload_bytes": {"original": 1000, "q90": 400},
            "qualities": {"original": 0.8, "q90": 0.79},
        }
        data = {"samples": [sample]}
        details = {
            "current@160": {
                1: {
                    "quality": 0.77,
                    "base_quality": 0.79,
                    "payload_bytes": 200,
                    "record": {
                        "area_ratio": 0.25,
                        "crop_width": 50,
                        "crop_height": 50,
                        "image_width": 100,
                        "image_height": 100,
                    },
                }
            }
        }
        rows = action_rows(data, {1: "q90"}, details, epsilon=0.02)
        self.assertEqual(rows["features"].shape, (3, 70))
        self.assertEqual(
            [value["action_id"] for value in rows["metadata"]],
            ["original", "q90", "current@160"],
        )
        self.assertEqual(rows["labels"].tolist(), [True, True, True])
        self.assertAlmostEqual(float(rows["gains"][0]), 0.01, places=6)
        self.assertAlmostEqual(float(rows["gains"][1]), 0.0, places=6)
        self.assertAlmostEqual(float(rows["gains"][2]), -0.02, places=6)

    def test_frozen_base_choice_does_not_reenter_scheduler(self) -> None:
        actions = actions_from_choice(
            {"name": "base"},
            options={},
            base={1: "q90", 2: "original"},
            budget=0.4,
        )
        self.assertEqual(actions, {1: "q90", 2: "original"})

    def test_class_gate_flags_only_supported_large_negative_groups(self) -> None:
        baseline = {
            1: {"category_name": "common", "instances": 100, "map50_95": 0.5},
            2: {"category_name": "rare", "instances": 10, "map50_95": 0.5},
        }
        candidate = {
            1: {"category_name": "common", "instances": 100, "map50_95": 0.48},
            2: {"category_name": "rare", "instances": 10, "map50_95": 0.0},
        }
        rows = compare_class_ap(
            baseline,
            candidate,
            budget=0.4,
            branch="reversible",
            minimum_instances=50,
            negative_delta=-0.01,
        )
        self.assertTrue(rows[0]["significant_negative"])
        self.assertFalse(rows[1]["significant_negative"])

    def test_primary_candidates_keep_exactly_one_base_after_branch_fallbacks(self) -> None:
        rows = primary_candidate_rows(
            {"official_map50_95": 0.5, "request_kb_per_image": 100.0},
            budget=0.4,
            branch_choices={
                "maur": {"name": "base", "branch": "maur"},
                "reversible": {"name": "base", "branch": "reversible"},
            },
        )
        self.assertEqual([row["name"] for row in rows], ["base"])

    def test_completion_marker_is_not_written_when_summary_write_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "COMPLETED"
            with patch.object(
                train_evaluate,
                "write_json",
                side_effect=RuntimeError("simulated interrupted summary write"),
            ):
                with self.assertRaises(RuntimeError):
                    train_evaluate.finalize_stage(
                        root / "summary.json", {"status": "completed"}, marker
                    )
            self.assertFalse(marker.exists())

    def test_protocol_hash_changes_when_nested_configuration_changes(self) -> None:
        original = {"scheduler": {"roi_budgets": [0.1, 0.25, 0.4]}}
        changed = {"scheduler": {"roi_budgets": [0.1, 0.25, 0.5]}}
        self.assertNotEqual(json_sha256(original), json_sha256(changed))

    def test_hash_inventory_rejects_frozen_cache_drift(self) -> None:
        expected = {"audit:current@160": "abc"}
        validate_hash_inventory(expected, expected, label="audit cache")
        with self.assertRaisesRegex(RuntimeError, "audit cache hash drift"):
            validate_hash_inventory(
                {"audit:current@160": "changed"},
                expected,
                label="audit cache",
            )


if __name__ == "__main__":
    unittest.main()
