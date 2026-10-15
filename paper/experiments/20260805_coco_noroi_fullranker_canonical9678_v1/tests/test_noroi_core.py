from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "noroi_core", ROOT / "scripts/noroi_core.py"
)
assert SPEC and SPEC.loader
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def protocol() -> dict:
    return json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


class NoRoiCoreTests(unittest.TestCase):
    def test_protocol_and_matrix_are_frozen(self) -> None:
        value = protocol()
        core.validate_protocol(value)
        self.assertEqual(len(core.METHODS), 16)
        self.assertEqual(len(core.expected_result_keys(value)), 192)

    def test_protocol_rejects_final_selection(self) -> None:
        value = protocol()
        value["registered_final_used_for_selection"] = True
        with self.assertRaisesRegex(ValueError, "val2017"):
            core.validate_protocol(value)

    def test_build_no_roi_policies(self) -> None:
        samples = [
            {"image_id": 1, "available": {"q90": True}},
            {"image_id": 2, "available": {"q90": True}},
            {"image_id": 3, "available": {"q90": False}},
        ]
        maps = core.build_no_roi_action_maps(
            samples, np.asarray([0.9, 0.4, 1.0]), q90_threshold=0.5
        )
        self.assertEqual(
            maps["Original-only"], {1: "original", 2: "original", 3: "original"}
        )
        self.assertEqual(
            maps["SharedAction-NoROI"], {1: "q90", 2: "original", 3: "original"}
        )
        core.assert_no_roi_actions(maps["SharedAction-NoROI"])

    def test_roi_action_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ROI"):
            core.assert_no_roi_actions({1: "original", 2: "roi"})

    def test_result_matrix_and_roi_ratio(self) -> None:
        value = protocol()
        rows = [
            {
                "method": method,
                "payload_policy": policy,
                "mode": mode,
                "nominal_budget": budget,
                "roi_ratio": 0.0,
            }
            for method, policy, mode, budget in sorted(core.expected_result_keys(value))
        ]
        core.assert_result_matrix_complete(rows, value)
        core.assert_result_rows_no_roi(rows)
        rows[0]["roi_ratio"] = 0.01
        with self.assertRaisesRegex(RuntimeError, "ROI"):
            core.assert_result_rows_no_roi(rows)

    def test_fixed_byte_capacity(self) -> None:
        samples = [
            {"image_id": 1, "payload_bytes": {"original": 100, "q90": 25}},
            {"image_id": 2, "payload_bytes": {"original": 100, "q90": 25}},
        ]
        core.assert_fixed_byte_capacity(samples, {1: "q90", 2: "q90"}, capacity=50)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            core.assert_fixed_byte_capacity(samples, {1: "original"}, capacity=50)

    def test_score_alignment_is_order_sensitive(self) -> None:
        core.assert_score_alignment([1, 2, 3], [1, 2, 3], np.ones((16, 3, 3)))
        with self.assertRaisesRegex(RuntimeError, "ordering"):
            core.assert_score_alignment([1, 2, 3], [1, 3, 2], np.ones((16, 3, 3)))

    def test_freeze_strongest_uses_calibration_only(self) -> None:
        rows = [
            {
                "method": "A",
                "payload_policy": "Original-only",
                "mode": "fixed_byte",
                "nominal_budget": 0.1,
                "official_map50_95": 0.5,
                "request_kb_per_image": 10.0,
                "role": "calibration",
            },
            {
                "method": "B",
                "payload_policy": "Original-only",
                "mode": "fixed_byte",
                "nominal_budget": 0.1,
                "official_map50_95": 0.6,
                "request_kb_per_image": 11.0,
                "role": "calibration",
            },
        ]
        self.assertEqual(
            core.freeze_strongest(
                rows,
                candidates=("A", "B"),
                policies=("Original-only",),
                budgets=(0.1,),
            ),
            {"Original-only|0.1": "B"},
        )
        rows[0]["role"] = "final"
        with self.assertRaisesRegex(RuntimeError, "calibration"):
            core.freeze_strongest(
                rows,
                candidates=("A", "B"),
                policies=("Original-only",),
                budgets=(0.1,),
            )


if __name__ == "__main__":
    unittest.main()
