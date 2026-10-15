from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/scan_core.py"


def load_core():
    spec = importlib.util.spec_from_file_location("scan_core", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ScanCoreTest(unittest.TestCase):
    def test_candidate_matrix_is_exactly_four_profiles_by_three_scales(self):
        core = load_core()
        candidates = core.candidate_matrix(
            ["current", "moderate", "historical_cost", "fixed_1_8"],
            [160, 320, 640],
        )
        self.assertEqual(len(candidates), 12)
        self.assertEqual(candidates[0]["id"], "current@160")
        self.assertEqual(candidates[-1]["id"], "fixed_1_8@640")
        self.assertEqual(len({value["id"] for value in candidates}), 12)

    def test_fixed_1_8_uses_one_expansion_for_all_area_buckets(self):
        core = load_core()
        for area in (0.001, 0.03, 0.2):
            self.assertEqual(core.roi_expansion(area, "fixed_1_8"), 1.8)
        self.assertEqual(core.roi_expansion(0.001, "historical_cost"), 4.0)

    def test_merged_roi_geometry_is_clipped_and_rejects_large_crop(self):
        core = load_core()
        boxes = [{"bbox": [90, 90, 20, 20], "score": 0.9}]
        self.assertEqual(
            core.merged_roi_xyxy(100, 100, boxes, profile="fixed_1_8"),
            (82, 82, 100, 100),
        )
        self.assertIsNone(
            core.merged_roi_xyxy(
                100,
                100,
                [{"bbox": [10, 10, 80, 80], "score": 0.9}],
                profile="fixed_1_8",
            )
        )

    def test_best_candidate_tie_breaks_by_quality_payload_then_id(self):
        core = load_core()
        chosen = core.best_candidate(
            [
                {"candidate_id": "b@640", "quality": 0.8, "payload_bytes": 90},
                {"candidate_id": "a@320", "quality": 0.8, "payload_bytes": 90},
                {"candidate_id": "c@160", "quality": 0.8, "payload_bytes": 100},
            ]
        )
        self.assertEqual(chosen["candidate_id"], "a@320")

    def test_ap_safe_point_minimizes_kb_then_maximizes_ap(self):
        core = load_core()
        rows = [
            {"roi_ratio": 0.0, "official_map50_95": 0.50, "request_kb_per_image": 100.0},
            {"roi_ratio": 0.1, "official_map50_95": 0.4992, "request_kb_per_image": 80.0},
            {"roi_ratio": 0.2, "official_map50_95": 0.4991, "request_kb_per_image": 80.0},
            {"roi_ratio": 0.3, "official_map50_95": 0.4980, "request_kb_per_image": 60.0},
        ]
        chosen = core.select_ap_safe_point(rows, tolerance=0.001)
        self.assertEqual(chosen["roi_ratio"], 0.1)

    def test_baseline_deltas_do_not_treat_single_frozen_point_as_base(self):
        core = load_core()
        point = {
            "official_map50_95": 0.55,
            "request_kb_per_image": 70.0,
        }
        baseline = {
            "official_map50_95": 0.50,
            "request_kb_per_image": 100.0,
        }
        updated = core.with_baseline_deltas(point, baseline)
        self.assertAlmostEqual(updated["map_delta_vs_base"], 0.05)
        self.assertAlmostEqual(updated["kb_delta_vs_base"], -30.0)
        self.assertNotIn("map_delta_vs_base", point)

    def test_role_contract_rejects_final(self):
        core = load_core()
        self.assertEqual(
            core.validate_roles({"calibration": 4961, "audit": 9923}),
            ("calibration", "audit"),
        )
        with self.assertRaisesRegex(ValueError, "final"):
            core.validate_roles(
                {"calibration": 4961, "audit": 9923, "final": 4962}
            )

    def test_cache_signature_separates_geometry_from_prediction_scale(self):
        core = load_core()
        geometry = core.canonical_signature(
            {"role": "calibration", "profile": "current", "quality": 90}
        )
        prediction_160 = core.canonical_signature(
            {"geometry": geometry, "imgsz": 160, "model": "abc"}
        )
        prediction_640 = core.canonical_signature(
            {"geometry": geometry, "imgsz": 640, "model": "abc"}
        )
        self.assertEqual(len(geometry), 64)
        self.assertNotEqual(prediction_160, prediction_640)
        self.assertEqual(
            geometry,
            core.canonical_signature(
                json.loads(
                    '{"profile":"current","quality":90,"role":"calibration"}'
                )
            ),
        )

    def test_ranked_candidate_actions_uses_gain_then_image_id(self):
        core = load_core()
        options = {
            9: [
                {
                    "candidate_id": "current@320",
                    "quality": 0.7,
                    "base_quality": 0.5,
                    "payload_bytes": 50,
                }
            ],
            3: [
                {
                    "candidate_id": "historical_cost@160",
                    "quality": 0.9,
                    "base_quality": 0.4,
                    "payload_bytes": 60,
                },
                {
                    "candidate_id": "fixed_1_8@160",
                    "quality": 0.9,
                    "base_quality": 0.4,
                    "payload_bytes": 40,
                },
            ],
        }
        selected = core.ranked_candidate_actions(options, count=1)
        self.assertEqual(list(selected), [3])
        self.assertEqual(selected[3]["candidate_id"], "fixed_1_8@160")
        self.assertEqual(selected[3]["gain"], 0.5)


if __name__ == "__main__":
    unittest.main()
