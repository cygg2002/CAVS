from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from online_methods import build_method_runtime, score_current_frame


class OnlineMethodsTest(unittest.TestCase):
    def test_random_is_deterministic_per_seed_and_image(self) -> None:
        runtime = build_method_runtime("random", random_seed=20260720)
        first = score_current_frame(runtime, 139, [], width=640, height=426, roi_xyxy=None)
        repeated = score_current_frame(runtime, 139, [], width=640, height=426, roi_xyxy=None)
        another = score_current_frame(runtime, 285, [], width=640, height=480, roi_xyxy=None)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, another)
        self.assertEqual(set(first), {"edge", "original"})
        self.assertGreaterEqual(first["original"], 0.0)
        self.assertLess(first["original"], 1.0)

    def test_confidence_uses_frozen_low_surviving_confidence_rule(self) -> None:
        runtime = build_method_runtime("confidence")
        scores = score_current_frame(
            runtime,
            139,
            [
                {"xyxy": [0, 0, 10, 10], "conf": 0.9, "cls": 0},
                {"xyxy": [0, 0, 20, 20], "conf": 0.3, "cls": 1},
                {"xyxy": [0, 0, 30, 30], "conf": 0.1, "cls": 2},
            ],
            width=640,
            height=480,
            roi_xyxy=None,
        )
        self.assertAlmostEqual(scores["original"], 0.7)
        empty = score_current_frame(runtime, 140, [], width=640, height=480, roi_xyxy=None)
        self.assertEqual(empty["original"], 2.0)

    def test_endpoint_references_force_edge_or_original_without_future_data(self) -> None:
        edge = build_method_runtime("edge_only")
        cloud = build_method_runtime("always_original")

        edge_scores = score_current_frame(
            edge, 1, [], width=640, height=480, roi_xyxy=None
        )
        cloud_scores = score_current_frame(
            cloud, 1, [], width=640, height=480, roi_xyxy=None
        )

        self.assertEqual(edge.action_ids, ("edge", "original"))
        self.assertEqual(cloud.action_ids, ("edge", "original"))
        self.assertLess(edge_scores["original"], edge_scores["edge"])
        self.assertGreater(cloud_scores["original"], cloud_scores["edge"])

    def test_unknown_method_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_method_runtime("posthoc-best")


if __name__ == "__main__":
    unittest.main()
