from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "frontier_aligned_core", ROOT / "scripts/core.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(ROOT / "scripts/core.py")
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


class ProtocolTests(unittest.TestCase):
    def test_protocol_preserves_canonical_development_boundaries(self) -> None:
        protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )
        CORE.validate_protocol(protocol)
        self.assertEqual(protocol["roles"], {"fit": 9678, "calibration": 571})
        self.assertFalse(protocol["final_accessed"])
        self.assertFalse(protocol["val2017_accessed"])


class FrontierTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = torch.tensor([[True, True, True, True]])
        self.costs = torch.tensor([[0.0, 1.0, 0.25, 0.1]])

    def test_target_path_changes_from_expensive_to_cheap_to_edge(self) -> None:
        rewards = torch.tensor([[0.0, 0.80, 0.50, 0.20]])
        prices = torch.tensor([0.0, 0.6, 3.0])
        targets = CORE.frontier_target_actions(
            rewards, self.costs, self.mask, prices
        )
        self.assertEqual(targets.tolist(), [[1, 2, 0]])

    def test_adding_image_constant_does_not_change_cloud_frontier(self) -> None:
        rewards = torch.tensor([[0.0, 0.80, 0.50, 0.20]])
        shifted = rewards + 4.0
        prices = torch.tensor([0.0, 0.6])
        first = CORE.frontier_target_actions(
            rewards, self.costs, self.mask, prices
        )
        second = CORE.frontier_target_actions(
            shifted, self.costs, self.mask, prices
        )
        self.assertEqual(first.tolist(), second.tolist())

    def test_dominated_roi_never_enters_target_path(self) -> None:
        rewards = torch.tensor([[0.0, 0.80, 0.50, 0.10]])
        prices = torch.linspace(0.0, 5.0, 101)
        targets = CORE.frontier_target_actions(
            rewards, self.costs, self.mask, prices
        )
        self.assertNotIn(3, targets.flatten().tolist())

    def test_unavailable_action_is_never_selected(self) -> None:
        rewards = torch.tensor([[0.0, 100.0, 0.5, 0.2]])
        mask = torch.tensor([[True, False, True, True]])
        targets = CORE.frontier_target_actions(
            rewards, self.costs, mask, torch.tensor([0.0])
        )
        self.assertEqual(targets.item(), 2)

    def test_probe_prices_cover_both_sides_of_pairwise_breakpoints(self) -> None:
        rewards = torch.tensor([[0.0, 0.80, 0.50]])
        costs = torch.tensor([[0.0, 1.0, 0.25]])
        mask = torch.ones_like(rewards, dtype=torch.bool)
        prices, price_mask = CORE.frontier_probe_prices(
            rewards,
            costs,
            mask,
            relative_offset=0.01,
            maximum_price=3.0,
        )
        observed = prices[0][price_mask[0]].tolist()
        self.assertTrue(any(abs(value - 0.396) < 1e-5 for value in observed))
        self.assertTrue(any(abs(value - 0.404) < 1e-5 for value in observed))
        self.assertTrue(any(abs(value - 1.98) < 1e-5 for value in observed))
        self.assertTrue(any(abs(value - 2.02) < 1e-5 for value in observed))

    def test_padded_equal_cost_actions_cannot_create_nan_probes(self) -> None:
        rewards = torch.tensor([[0.0, 0.5, 0.0, 0.0]])
        costs = torch.tensor([[0.0, 0.2, 0.0, 0.0]])
        mask = torch.tensor([[True, True, False, False]])
        prices, price_mask = CORE.frontier_probe_prices(
            rewards,
            costs,
            mask,
            relative_offset=0.01,
            maximum_price=5.0,
        )
        self.assertTrue(torch.isfinite(prices).all())
        loss = CORE.frontier_path_loss(
            rewards,
            rewards,
            costs,
            mask,
            prices,
            price_mask=price_mask,
            temperature=0.03,
            boundary_scale=0.03,
            boundary_boost=0.0,
        )
        self.assertTrue(torch.isfinite(loss))


class FrontierLossTests(unittest.TestCase):
    def test_matching_frontier_scores_have_lower_loss(self) -> None:
        rewards = torch.tensor(
            [[0.0, 0.8, 0.5], [0.0, 0.1, 0.4]], dtype=torch.float32
        )
        costs = torch.tensor(
            [[0.0, 1.0, 0.25], [0.0, 1.0, 0.25]], dtype=torch.float32
        )
        mask = torch.ones_like(rewards, dtype=torch.bool)
        prices = torch.tensor([0.0, 0.5, 2.0])
        matching = CORE.frontier_path_loss(
            rewards * 8.0,
            rewards,
            costs,
            mask,
            prices,
            temperature=0.03,
            boundary_scale=0.03,
            boundary_boost=0.0,
        )
        reversed_scores = rewards.flip(1) * 8.0
        wrong = CORE.frontier_path_loss(
            reversed_scores,
            rewards,
            costs,
            mask,
            prices,
            temperature=0.03,
            boundary_scale=0.03,
            boundary_boost=0.0,
        )
        self.assertLess(float(matching), float(wrong))

    def test_boundary_weight_is_larger_for_near_tie(self) -> None:
        near = torch.tensor([0.001, 0.01])
        far = torch.tensor([0.2, 0.5])
        self.assertGreater(
            float(CORE.boundary_weights(near, scale=0.03, boost=2.0).mean()),
            float(CORE.boundary_weights(far, scale=0.03, boost=2.0).mean()),
        )

    def test_loss_accepts_per_image_parametric_probe_paths(self) -> None:
        rewards = torch.tensor(
            [[0.0, 0.8, 0.5], [0.0, 0.1, 0.4]], dtype=torch.float32
        )
        costs = torch.tensor(
            [[0.0, 1.0, 0.25], [0.0, 1.0, 0.25]], dtype=torch.float32
        )
        mask = torch.ones_like(rewards, dtype=torch.bool)
        prices, price_mask = CORE.frontier_probe_prices(
            rewards,
            costs,
            mask,
            relative_offset=0.01,
            maximum_price=3.0,
        )
        loss = CORE.frontier_path_loss(
            rewards,
            rewards,
            costs,
            mask,
            prices,
            price_mask=price_mask,
            temperature=0.03,
            boundary_scale=0.03,
            boundary_boost=2.0,
        )
        self.assertTrue(torch.isfinite(loss))


class GateTests(unittest.TestCase):
    def test_screen_gate_requires_two_byte_gains_and_call_safety(self) -> None:
        rows = [
            {"mode": "fixed_byte", "budget": 0.1, "delta_ap": 0.002},
            {"mode": "fixed_byte", "budget": 0.25, "delta_ap": 0.001},
            {"mode": "fixed_byte", "budget": 0.4, "delta_ap": -0.0005},
            {"mode": "fixed_call", "budget": 0.1, "delta_ap": 0.0},
            {"mode": "fixed_call", "budget": 0.25, "delta_ap": -0.0005},
            {"mode": "fixed_call", "budget": 0.4, "delta_ap": 0.0},
        ]
        gate = {
            "minimum_fixed_byte_gain": 0.001,
            "required_fixed_byte_gain_points": 2,
            "maximum_other_loss": 0.001,
            "maximum_fixed_call_loss": 0.001,
        }
        self.assertTrue(CORE.evaluate_screen_gate(rows, gate)["passed"])
        rows[-1]["delta_ap"] = -0.0011
        self.assertFalse(CORE.evaluate_screen_gate(rows, gate)["passed"])

    def test_screen_gate_rejects_underused_byte_capacity(self) -> None:
        rows = [
            {
                "mode": mode,
                "budget": budget,
                "delta_ap": 0.002,
                "capacity_utilization": 0.97 if mode == "fixed_byte" else 1.0,
            }
            for mode in ("fixed_byte", "fixed_call")
            for budget in (0.1, 0.25, 0.4)
        ]
        gate = {
            "minimum_fixed_byte_gain": 0.001,
            "required_fixed_byte_gain_points": 2,
            "maximum_other_loss": 0.001,
            "maximum_fixed_call_loss": 0.001,
            "minimum_capacity_utilization": 0.98,
        }
        result = CORE.evaluate_screen_gate(rows, gate)
        self.assertFalse(result["passed"])
        self.assertFalse(result["capacity_sufficient"])

    def test_confirm_gate_requires_two_seed_support_at_two_budgets(self) -> None:
        ensemble = [
            {
                "mode": mode,
                "budget": budget,
                "delta_ap": 0.002 if mode == "fixed_byte" else 0.0,
                "capacity_utilization": 1.0,
            }
            for mode in ("fixed_byte", "fixed_call")
            for budget in (0.1, 0.25, 0.4)
        ]
        seeds = [
            {
                "seed": seed,
                "mode": "fixed_byte",
                "budget": budget,
                "delta_ap": 0.0012 if budget in (0.1, 0.25) else 0.0002,
            }
            for seed in (1, 2, 3)
            for budget in (0.1, 0.25, 0.4)
        ]
        gate = {
            "minimum_fixed_byte_gain": 0.001,
            "required_fixed_byte_gain_points": 2,
            "maximum_other_loss": 0.001,
            "maximum_fixed_call_loss": 0.001,
            "minimum_capacity_utilization": 0.98,
            "minimum_supporting_seeds": 2,
        }
        result = CORE.evaluate_confirm_gate(seeds, ensemble, gate)
        self.assertTrue(result["passed"])
        for row in seeds:
            if row["seed"] in (2, 3) and row["budget"] == 0.25:
                row["delta_ap"] = 0.0
        self.assertFalse(CORE.evaluate_confirm_gate(seeds, ensemble, gate)["passed"])


class ActionSpaceTests(unittest.TestCase):
    def test_filter_action_space_preserves_order_and_masks_roi(self) -> None:
        records = [
            {
                "image_id": 1,
                "actions": [
                    {"action_id": "edge"},
                    {"action_id": "original"},
                    {"action_id": "q90"},
                    {"action_id": "roi"},
                ],
            }
        ]
        filtered = CORE.filter_action_space_records(
            records, ("edge", "original", "q90")
        )
        self.assertEqual(
            [action["action_id"] for action in filtered[0]["actions"]],
            ["edge", "original", "q90"],
        )
        self.assertEqual(len(records[0]["actions"]), 4)

    def test_filter_action_space_requires_edge(self) -> None:
        with self.assertRaises(ValueError):
            CORE.filter_action_space_records([], ("original", "q90"))


if __name__ == "__main__":
    unittest.main()
