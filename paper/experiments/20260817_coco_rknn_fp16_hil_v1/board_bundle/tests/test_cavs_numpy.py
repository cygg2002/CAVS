from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cavs_numpy import current_roi_xyxy, load_policy, predict_scores


class CavsNumpyTest(unittest.TestCase):
    def test_numpy_policy_matches_frozen_pytorch_golden_case(self) -> None:
        boxes = [
            {"xyxy": [100.25, 120.5, 180.75, 260.125], "conf": 0.91, "cls": 0},
            {"xyxy": [190.0, 130.0, 280.5, 300.0], "conf": 0.73, "cls": 16},
            {"xyxy": [150.0, 80.0, 195.0, 135.0], "conf": 0.31, "cls": 2},
            {"xyxy": [5.0, 5.0, 15.0, 15.0], "conf": 0.0005, "cls": 7},
        ]
        roi = current_roi_xyxy(640, 480, boxes, confidence=0.25, max_area_ratio=0.65)
        self.assertEqual(roi, [76, 58, 308, 351])

        policy = load_policy(ROOT / "policy" / "manifest.json")
        actual = predict_scores(policy, boxes, width=640, height=480, roi_xyxy=roi)
        expected = {
            "edge": 0.0,
            "original": 0.3581651177401041,
            "q90": 0.3242438197537096,
            "current@640": 0.24509042514852064,
        }
        self.assertEqual(set(actual), set(expected))
        for action, value in expected.items():
            self.assertAlmostEqual(actual[action], value, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
