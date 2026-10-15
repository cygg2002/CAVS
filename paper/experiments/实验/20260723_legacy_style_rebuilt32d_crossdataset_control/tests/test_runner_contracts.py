from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation_core import (  # noqa: E402
    assert_bootstrap_rows_complete,
    assert_fixed_byte_gates_complete,
    eligible_roi_ids,
    overlay_roi_actions,
    verify_reproduction_rows,
)


class RunnerContractTest(unittest.TestCase):
    def test_roi_extremes_reproduce_shared_and_all_eligible(self) -> None:
        base = {1: "original", 2: "q90", 3: "original"}
        eligible = {1, 2}
        scores = {1: 0.9, 2: 0.1, 3: 1.0}
        shared = overlay_roi_actions(base, scores, eligible, math.inf)
        all_roi = overlay_roi_actions(base, scores, eligible, -math.inf)
        self.assertEqual(shared, base)
        self.assertEqual(all_roi, {1: "roi", 2: "roi", 3: "original"})

    def test_fixed_byte_requires_one_recalibrated_gate_per_point(self) -> None:
        gates = [
            {"ranker": ranker, "budget": budget, "lambda_value": 0.1}
            for ranker in ("A", "B")
            for budget in (0.1, 0.2)
        ]
        assert_fixed_byte_gates_complete(gates, ("A", "B"), (0.1, 0.2))
        with self.assertRaisesRegex(RuntimeError, "gate"):
            assert_fixed_byte_gates_complete(gates[:-1], ("A", "B"), (0.1, 0.2))
        with self.assertRaisesRegex(RuntimeError, "gate"):
            assert_fixed_byte_gates_complete(gates + [dict(gates[0])], ("A", "B"), (0.1, 0.2))

    def test_roi_cost_eligibility_uses_actual_shared_base(self) -> None:
        samples = [
            {
                "image_id": 1,
                "available": {"roi": True},
                "payload_bytes": {"original": 1000, "q90": 300, "roi": 500},
            },
            {
                "image_id": 2,
                "available": {"roi": True},
                "payload_bytes": {"original": 1000, "q90": 300, "roi": 500},
            },
        ]
        self.assertEqual(
            eligible_roi_ids({"samples": samples}, {1: "original", 2: "q90"}),
            {1},
        )

    def test_bootstrap_requires_exact_unique_comparison_grid(self) -> None:
        rows = [
            {
                "comparison": "GBDT-Hierarchical - SharedAction",
                "method": ranker,
                "nominal_budget": budget,
                "replicates": 200,
            }
            for ranker in ("A", "B")
            for budget in (0.1, 0.25, 0.4)
        ]
        assert_bootstrap_rows_complete(
            rows,
            rankers=("A", "B"),
            budgets=(0.1, 0.25, 0.4),
            comparison="GBDT-Hierarchical - SharedAction",
            replicates=200,
        )
        with self.assertRaisesRegex(RuntimeError, "Bootstrap"):
            assert_bootstrap_rows_complete(
                rows + [dict(rows[0])],
                rankers=("A", "B"),
                budgets=(0.1, 0.25, 0.4),
                comparison="GBDT-Hierarchical - SharedAction",
                replicates=200,
            )

    def test_reproduction_ignores_new_controlled_ranker_only(self) -> None:
        expected = [
            {
                "role": "final",
                "method": "Confidence",
                "mode": "fixed_call",
                "nominal_budget": "0.1",
                "official_map50_95": "0.7",
                "request_kb_per_image": "10.0",
            }
        ]
        actual = [
            {
                "action_policy": "SharedAction",
                "method": "Confidence",
                "mode": "fixed_call",
                "nominal_budget": 0.1,
                "official_map50_95": 0.7,
                "request_kb_per_image": 10.0,
            },
            {
                "action_policy": "SharedAction",
                "method": "Current-Controlled-32D",
                "mode": "fixed_call",
                "nominal_budget": 0.1,
                "official_map50_95": 0.8,
                "request_kb_per_image": 9.0,
            },
        ]
        report = verify_reproduction_rows(
            actual,
            expected,
            canonical_rankers=("Confidence",),
            metrics=("official_map50_95", "request_kb_per_image"),
            tolerance=1e-12,
        )
        self.assertEqual(report["rows"], 1)
        self.assertEqual(report["worst_difference"], 0.0)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            verify_reproduction_rows(
                actual + [dict(actual[0])],
                expected,
                canonical_rankers=("Confidence",),
                metrics=("official_map50_95",),
                tolerance=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
