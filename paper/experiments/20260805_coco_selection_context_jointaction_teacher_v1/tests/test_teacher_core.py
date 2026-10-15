from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from teacher_core import (  # noqa: E402
    build_context_ids,
    evaluate_gate,
    schedule_joint_actions,
    selection_overlap,
    validate_protocol,
)


class ContextTests(unittest.TestCase):
    def test_context_is_deterministic_unique_and_fit_only(self) -> None:
        fit_ids = [101, 103, 107, 109, 113, 127]
        first = build_context_ids(fit_ids, 999, context_size=4, seed=17)
        second = build_context_ids(fit_ids, 999, context_size=4, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len(set(first)), 4)
        self.assertTrue(set(first).issubset(set(fit_ids)))
        self.assertNotIn(999, first)

    def test_fit_target_is_excluded_from_its_context(self) -> None:
        fit_ids = [1, 2, 3, 4]
        context = build_context_ids(fit_ids, 3, context_size=10, seed=9)
        self.assertEqual(len(context), 3)
        self.assertNotIn(3, context)


class SchedulingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [
            {
                "image_id": 10,
                "payload_bytes": {"original": 10, "q90": 4, "roi": 3},
                "available": {"original": True, "q90": True, "roi": False},
            },
            {
                "image_id": 20,
                "payload_bytes": {"original": 10, "q90": 5, "roi": 2},
                "available": {"original": True, "q90": True, "roi": True},
            },
        ]
        self.rewards = {
            10: {"original": 3.0, "q90": 2.0, "roi": 100.0},
            20: {"original": 2.0, "q90": 2.5, "roi": 2.0},
        }

    def test_joint_schedule_respects_capacity_and_availability(self) -> None:
        result = schedule_joint_actions(
            self.samples,
            self.rewards,
            candidate_actions=("original", "q90", "roi"),
            capacity_bytes=6,
        )
        self.assertLessEqual(result["total_cost_bytes"], 6)
        self.assertNotEqual(result["selection"].get(10), "roi")
        self.assertEqual(set(result["selection"]), {10, 20})
        self.assertEqual(result["selection"][10], "q90")
        self.assertEqual(result["selection"][20], "roi")

    def test_zero_or_negative_reward_can_stay_on_edge(self) -> None:
        rewards = {
            10: {"original": -1.0, "q90": 0.0, "roi": -2.0},
            20: {"original": -1.0, "q90": -2.0, "roi": -3.0},
        }
        result = schedule_joint_actions(
            self.samples,
            rewards,
            candidate_actions=("original", "q90", "roi"),
            capacity_bytes=20,
        )
        self.assertEqual(result["selection"], {})
        self.assertEqual(result["total_cost_bytes"], 0.0)


class ReportingTests(unittest.TestCase):
    def test_selection_overlap_reports_image_and_exact_action_overlap(self) -> None:
        overlap = selection_overlap(
            {1: "q90", 2: "roi", 3: "original"},
            {2: "q90", 3: "original", 4: "roi"},
        )
        self.assertAlmostEqual(overlap["selected_image_jaccard"], 0.5)
        self.assertAlmostEqual(overlap["exact_action_jaccard"], 0.2)
        self.assertEqual(overlap["common_selected_images"], 2)
        self.assertEqual(overlap["common_exact_actions"], 1)

    def test_gate_requires_two_gains_and_noninferiority_elsewhere(self) -> None:
        protocol = {
            "budgets": [0.1, 0.25, 0.4],
            "gate": {
                "minimum_gain": 0.002,
                "required_gain_points": 2,
                "remaining_noninferiority_margin": 0.001,
                "capacity_tolerance_bytes": 0.001,
            },
        }
        rows = [
            {"budget": 0.1, "delta_ap": 0.0021, "capacity_ok": True},
            {"budget": 0.25, "delta_ap": 0.003, "capacity_ok": True},
            {"budget": 0.4, "delta_ap": -0.0009, "capacity_ok": True},
        ]
        self.assertTrue(evaluate_gate(rows, protocol)["passed"])
        rows[2]["delta_ap"] = -0.0011
        self.assertFalse(evaluate_gate(rows, protocol)["passed"])


class ProtocolTests(unittest.TestCase):
    def test_frozen_protocol_is_valid_and_forbids_evaluation_roles(self) -> None:
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        validate_protocol(protocol)
        self.assertEqual(set(protocol["roles"]), {"fit", "calibration"})
        self.assertTrue(protocol["fit_only_context"])
        self.assertFalse(protocol["audit_accessed"])
        self.assertFalse(protocol["final_accessed"])
        self.assertFalse(protocol["registered_final_accessed"])

    def test_protocol_rejects_role_leakage(self) -> None:
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        protocol["roles"]["final"] = 5000
        with self.assertRaisesRegex(ValueError, "fit and calibration"):
            validate_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
