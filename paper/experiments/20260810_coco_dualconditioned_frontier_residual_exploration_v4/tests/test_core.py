from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("residual_core", ROOT / "scripts/core.py")
if SPEC is None or SPEC.loader is None:
    raise ImportError(ROOT / "scripts/core.py")
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


class CoreTests(unittest.TestCase):
    def test_protocol_boundary(self) -> None:
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        CORE.validate_protocol(protocol)
        self.assertFalse(protocol["val2017_accessed"])

    def test_frontier_state_uses_best_competitor(self) -> None:
        values = torch.tensor([[0.0, 0.8, 0.5, 0.6]])
        costs = torch.tensor([[0.0, 1.0, 0.2, 0.1]])
        mask = torch.ones_like(values, dtype=torch.bool)
        state = CORE.frontier_state(
            values, costs, mask, torch.tensor([0.5]), roi_index=torch.tensor([3])
        )
        self.assertAlmostEqual(float(state["margin"][0]), 0.15, places=6)
        self.assertEqual(int(state["competitor_index"][0]), 2)
        self.assertAlmostEqual(float(state["cost_gap"][0]), -0.1, places=6)

    def test_rectification_changes_only_roi(self) -> None:
        scores = {1: {"original": 0.8, "q90": 0.6, "roi": 0.7}}
        corrected = CORE.rectify_roi_scores(scores, {1: -0.2})
        self.assertEqual(corrected[1]["original"], 0.8)
        self.assertEqual(corrected[1]["q90"], 0.6)
        self.assertAlmostEqual(corrected[1]["roi"], 0.5)


if __name__ == "__main__":
    unittest.main()

