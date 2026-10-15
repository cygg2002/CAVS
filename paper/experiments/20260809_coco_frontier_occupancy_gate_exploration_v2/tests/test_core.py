from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "frontier_occupancy_core", ROOT / "scripts/core.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(ROOT / "scripts/core.py")
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_fit_only_and_keeps_final_sealed(self) -> None:
        protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )
        CORE.validate_protocol(protocol)
        self.assertEqual(protocol["roles"], {"fit": 9678})
        self.assertFalse(protocol["threshold_selected_on_oof"])
        self.assertFalse(protocol["final_accessed"])


class OccupancyTargetTests(unittest.TestCase):
    def test_frontier_labels_mark_every_action_seen_on_valid_path(self) -> None:
        targets = torch.tensor([[1, 1, 2, 0], [2, 3, 2, 0]])
        price_mask = torch.tensor(
            [[True, True, True, False], [True, False, True, False]]
        )
        action_mask = torch.tensor(
            [[True, True, True, False], [True, True, True, True]]
        )
        labels = CORE.frontier_occupancy_labels(
            targets, price_mask, action_mask
        )
        self.assertEqual(
            labels.tolist(),
            [[False, True, True, False], [False, False, True, False]],
        )


class RoiGateTests(unittest.TestCase):
    def test_gate_removes_only_low_probability_roi(self) -> None:
        scores = {
            1: {"original": 0.8, "q90": 0.7, "roi": 0.6},
            2: {"original": 0.5, "q90": 0.4, "roi": 0.3},
        }
        gated = CORE.gate_roi_scores(
            scores, {1: 0.49, 2: 0.51}, threshold=0.5
        )
        self.assertEqual(gated[1], {"original": 0.8, "q90": 0.7})
        self.assertEqual(gated[2], scores[2])
        self.assertIn("roi", scores[1])

    def test_gate_rejects_missing_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            CORE.gate_roi_scores(
                {1: {"original": 1.0, "roi": 0.5}}, {}, threshold=0.5
            )


if __name__ == "__main__":
    unittest.main()
