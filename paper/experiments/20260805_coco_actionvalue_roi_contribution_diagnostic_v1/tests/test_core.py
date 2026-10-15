from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from roi_core import (  # noqa: E402
    apply_action_policy,
    build_action_score_map,
    evaluate_gain_gate,
    roi_risk_rates,
    validate_protocol,
)


class ProtocolTests(unittest.TestCase):
    def test_frozen_protocol_is_valid(self) -> None:
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        validate_protocol(protocol)

    def test_forbidden_final_access_is_rejected(self) -> None:
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        protocol["final_accessed"] = True
        with self.assertRaisesRegex(ValueError, "final_accessed"):
            validate_protocol(protocol)


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scores = {
            1: {"original": 0.4, "q90": 0.5, "roi": 0.8},
            2: {"original": 0.5, "q90": 0.4, "roi": 0.7},
            3: {"original": 0.2, "q90": 0.3},
        }
        self.rewards = {
            1: {"original": 0.1, "q90": 0.2, "roi": 0.3},
            2: {"original": 0.3, "q90": 0.2, "roi": 0.1},
            3: {"original": 0.2, "q90": 0.1},
        }

    def test_oq_policy_disables_roi_without_changing_other_scores(self) -> None:
        result = apply_action_policy(self.scores, self.rewards, "ActionValue-OQ")
        self.assertEqual(result[1], {"original": 0.4, "q90": 0.5})
        self.assertEqual(result[3], self.scores[3])

    def test_oracle_relative_gate_keeps_only_roi_better_than_alternatives(self) -> None:
        result = apply_action_policy(
            self.scores,
            self.rewards,
            "ActionValue-OQR-OracleRelativeGate",
        )
        self.assertIn("roi", result[1])
        self.assertNotIn("roi", result[2])
        self.assertNotIn("roi", result[3])

    def test_oracle_relative_gate_rejects_roi_when_all_cloud_rewards_are_negative(self) -> None:
        scores = {1: {"original": 0.2, "q90": 0.3, "roi": 0.8}}
        rewards = {1: {"original": -0.3, "q90": -0.2, "roi": -0.1}}
        result = apply_action_policy(
            scores,
            rewards,
            "ActionValue-OQR-OracleRelativeGate",
        )
        self.assertNotIn("roi", result[1])

    def test_score_map_follows_each_records_action_order_and_ignores_padding(self) -> None:
        records = [
            {
                "image_id": 9,
                "actions": [
                    {"action_id": "edge"},
                    {"action_id": "original"},
                    {"action_id": "q90"},
                ],
            }
        ]
        scores = np.asarray([[0.0, 0.4, 0.5, -1e9]])
        self.assertEqual(
            build_action_score_map(records, scores),
            {9: {"original": 0.4, "q90": 0.5}},
        )


class GateTests(unittest.TestCase):
    def test_gate_requires_two_gains_and_noninferiority(self) -> None:
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        self.assertTrue(evaluate_gain_gate([0.0012, 0.0011, -0.0009], protocol)["passed"])
        self.assertFalse(evaluate_gain_gate([0.0012, 0.0009, -0.0009], protocol)["passed"])
        self.assertFalse(evaluate_gain_gate([0.0012, 0.0011, -0.0011], protocol)["passed"])


class RiskMetricTests(unittest.TestCase):
    def test_roi_rates_use_selected_roi_as_the_denominator(self) -> None:
        rewards = {
            1: {"original": 0.1, "q90": 0.2, "roi": 0.3},
            2: {"original": 0.2, "q90": 0.1, "roi": -0.1},
            3: {"original": 0.4, "q90": 0.3, "roi": 0.2},
        }
        rates = roi_risk_rates(
            {1: "roi", 2: "roi", 3: "original"}, rewards
        )
        self.assertEqual(rates["roi_count"], 2)
        self.assertEqual(rates["harmful_roi_rate"], 0.5)
        self.assertEqual(rates["relative_suboptimal_roi_rate"], 0.5)

    def test_empty_roi_selection_has_zero_rates(self) -> None:
        rates = roi_risk_rates(
            {1: "q90"}, {1: {"original": 0.1, "q90": 0.2, "roi": 0.3}}
        )
        self.assertEqual(
            rates,
            {"roi_count": 0, "harmful_roi_rate": 0.0, "relative_suboptimal_roi_rate": 0.0},
        )

    def test_roi_rates_compare_only_available_alternatives(self) -> None:
        rates = roi_risk_rates(
            {1: "roi"}, {1: {"original": 0.2, "roi": 0.3}}
        )
        self.assertEqual(rates["relative_suboptimal_roi_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
