from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fusion import maur_fuse_boxes


class FusionTest(unittest.TestCase):
    def test_maur_replaces_inside_roi_and_keeps_outside_edge_box(self) -> None:
        edge = [
            {"xyxy": [10, 10, 30, 30], "conf": 0.8, "cls": 0},
            {"xyxy": [100, 100, 150, 150], "conf": 0.7, "cls": 1},
        ]
        cloud = [{"xyxy": [98, 98, 152, 152], "conf": 0.9, "cls": 1}]
        fused = maur_fuse_boxes(edge, cloud, [80, 80, 180, 180])
        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0]["cls"], 1)
        self.assertTrue(any(box["cls"] == 0 for box in fused))


if __name__ == "__main__":
    unittest.main()

