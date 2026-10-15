from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("price_gate_core", ROOT / "scripts/core.py")
if SPEC is None or SPEC.loader is None:
    raise ImportError(ROOT / "scripts/core.py")
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


class CoreTests(unittest.TestCase):
    def test_protocol_keeps_development_boundary(self) -> None:
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        CORE.validate_protocol(protocol)
        self.assertFalse(protocol["threshold_selected_on_oof"])
        self.assertFalse(protocol["final_accessed"])

    def test_roi_price_targets_change_with_shadow_price(self) -> None:
        rewards = torch.tensor([[0.0, 0.8, 0.5, 0.3]])
        costs = torch.tensor([[0.0, 1.0, 0.25, 0.05]])
        mask = torch.ones_like(rewards, dtype=torch.bool)
        prices = torch.tensor([0.0, 0.5, 5.0])
        target = CORE.roi_optimal_by_price(
            rewards, costs, mask, prices, roi_index=2
        )
        self.assertEqual(target.tolist(), [[False, True, False]])

    def test_gate_removes_only_roi(self) -> None:
        scores = {1: {"original": 0.8, "q90": 0.6, "roi": 0.7}}
        gated = CORE.gate_roi_scores(scores, {1: 0.49}, threshold=0.5)
        self.assertEqual(gated, {1: {"original": 0.8, "q90": 0.6}})

    def test_per_image_roi_positions(self) -> None:
        rewards = torch.tensor([[0.0, 0.6, 0.7], [0.0, 0.7, 0.6]])
        costs = torch.zeros_like(rewards)
        mask = torch.ones_like(rewards, dtype=torch.bool)
        target = CORE.roi_optimal_by_price(
            rewards,
            costs,
            mask,
            torch.tensor([0.0]),
            roi_index=torch.tensor([2, 1]),
        )
        self.assertEqual(target.tolist(), [[True], [True]])


if __name__ == "__main__":
    unittest.main()
