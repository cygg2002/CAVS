from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("transition_core", ROOT / "scripts/core.py")
CORE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(CORE)


class TransitionAccountingTest(unittest.TestCase):
    def test_roi_column_is_resolved_per_image(self) -> None:
        available = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
        names = [["edge", "original", "roi"], ["edge", "original", "q90", "roi"]]
        observed = CORE.action_column_mask(names, available, "roi")
        expected = torch.tensor(
            [[0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.bool
        )
        self.assertTrue(torch.equal(observed, expected))

    def test_only_valid_roi_crossings_are_marked(self) -> None:
        rewards = torch.tensor([[0.0, 1.0, 0.5]])
        costs = torch.tensor([[0.0, 2.0, 2.0]])
        mask = torch.ones_like(rewards, dtype=torch.bool)
        roi = torch.tensor([[0, 0, 1]], dtype=torch.bool)
        valid, roi_valid = CORE.crossing_pair_masks(
            rewards, costs, mask, roi, maximum_price=5.0
        )
        # edge/original and edge/ROI cross at positive prices; the two cloud
        # actions have equal costs and therefore do not cross.
        self.assertEqual(int(valid.sum()), 2)
        self.assertEqual(int(roi_valid.sum()), 1)

    def test_duplicate_action_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CORE.action_column_mask(
                [["edge", "roi", "roi"]], torch.ones((1, 3), dtype=torch.bool), "roi"
            )


if __name__ == "__main__":
    unittest.main()
