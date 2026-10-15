from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dualheadv2_core import (  # noqa: E402
    assert_no_roi_selection,
    blend_mean_lower,
    choose_alphas,
    evaluate_dualheadv2_gate,
    evaluate_selection_context_gate,
    no_roi_score_map,
    validate_protocol,
)


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_canonical_fit_calibration_only(self) -> None:
        protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )
        validate_protocol(protocol)
        self.assertEqual(protocol["roles"], {"fit": 9678, "calibration": 571})
        self.assertEqual(protocol["candidate_actions"], ["original", "q90"])
        self.assertFalse(protocol["final_accessed"])
        self.assertFalse(protocol["registered_final_accessed"])

    def test_protocol_rejects_roi(self) -> None:
        protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )
        protocol["candidate_actions"].append("roi")
        with self.assertRaisesRegex(ValueError, "ROI|action"):
            validate_protocol(protocol)


class ScoreTests(unittest.TestCase):
    def test_alpha_zero_is_mean_and_penalty_is_one_sided(self) -> None:
        mean = np.asarray([[0.5, 0.2], [0.1, -0.1]])
        lower = np.asarray([[0.1, 0.3], [-0.2, -0.4]])
        np.testing.assert_allclose(blend_mean_lower(mean, lower, 0.0), mean)
        np.testing.assert_allclose(
            blend_mean_lower(mean, lower, 0.5),
            [[0.3, 0.2], [-0.05, -0.25]],
        )

    def test_no_roi_score_map_hard_masks_roi_even_when_present(self) -> None:
        records = [
            {
                "image_id": 7,
                "actions": [
                    {"action_id": "edge"},
                    {"action_id": "original"},
                    {"action_id": "q90"},
                    {"action_id": "roi"},
                ],
            }
        ]
        mapped = no_roi_score_map(records, np.asarray([[0.0, 1.0, 2.0, 99.0]]))
        self.assertEqual(mapped, {7: {"original": 1.0, "q90": 2.0}})

    def test_no_roi_selection_rejects_roi(self) -> None:
        assert_no_roi_selection({1: "original", 2: "q90"})
        with self.assertRaisesRegex(RuntimeError, "ROI"):
            assert_no_roi_selection({1: "roi"})


def row(
    method: str,
    mode: str,
    budget: float,
    ap: float,
    *,
    utilization: float = 1.0,
) -> dict:
    return {
        "method": method,
        "mode": mode,
        "budget": budget,
        "official_ap50_95": ap,
        "capacity_ok": True,
        "capacity_utilization": utilization,
        "roi_count": 0,
    }


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_choose_alpha_uses_oof_ap_with_utilization_and_call_guards(self) -> None:
        rows = []
        for alpha, byte_ap, call_ap, utilization in (
            (0.0, 0.50, 0.49, 1.0),
            (0.1, 0.51, 0.4895, 0.99),
            (0.2, 0.52, 0.48, 1.0),
            (0.5, 0.60, 0.50, 0.50),
        ):
            name = f"DualHeadV2-alpha={alpha:g}"
            rows.append(row(name, "fixed_byte", 0.1, byte_ap, utilization=utilization))
            rows.append(row(name, "fixed_call", 0.1, call_ap))
        rows.append(row("ActionValue-OQ", "fixed_call", 0.1, 0.49))
        chosen = choose_alphas(rows, self.protocol, budgets=[0.1])
        self.assertEqual(chosen[0.1]["alpha"], 0.1)
        self.assertEqual(chosen[0.1]["selection_role"], "fit_oof")

    def test_choose_alpha_records_alpha_zero_when_every_candidate_is_ineligible(self) -> None:
        rows = [
            row("ActionValue-OQ", "fixed_call", 0.1, 0.50),
            row("DualHeadV2-alpha=0", "fixed_byte", 0.1, 0.60, utilization=0.5),
            row("DualHeadV2-alpha=0", "fixed_call", 0.1, 0.40),
        ]
        chosen = choose_alphas(rows, self.protocol, budgets=[0.1])
        self.assertEqual(chosen[0.1]["alpha"], 0.0)
        self.assertFalse(chosen[0.1]["selection_constraints_passed"])

    def test_dualhead_gate_requires_two_byte_gains_and_call_noninferiority(self) -> None:
        rows = []
        for budget, gain in ((0.1, 0.002), (0.25, 0.0015), (0.4, -0.0005)):
            rows.extend(
                [
                    row("ActionValue-OQ", "fixed_byte", budget, 0.5),
                    row("DualHeadV2-OQ", "fixed_byte", budget, 0.5 + gain),
                    row("ActionValue-OQ", "fixed_call", budget, 0.49),
                    row("DualHeadV2-OQ", "fixed_call", budget, 0.4895),
                ]
            )
        result = evaluate_dualheadv2_gate(rows, self.protocol)
        self.assertTrue(result["passed"])
        rows[-1]["official_ap50_95"] = 0.47
        self.assertFalse(evaluate_dualheadv2_gate(rows, self.protocol)["passed"])

    def test_selection_context_gate_uses_all_three_budgets(self) -> None:
        values = [
            {"budget": 0.1, "delta_ap": 0.003, "capacity_ok": True},
            {"budget": 0.25, "delta_ap": 0.002, "capacity_ok": True},
            {"budget": 0.4, "delta_ap": -0.0005, "capacity_ok": True},
        ]
        self.assertTrue(evaluate_selection_context_gate(values, self.protocol)["passed"])
        values[2]["delta_ap"] = -0.002
        self.assertFalse(evaluate_selection_context_gate(values, self.protocol)["passed"])


if __name__ == "__main__":
    unittest.main()
