from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from finalize_source_dataset import (  # noqa: E402
    action_selection_key,
    choose_final_oracle,
    find_frozen_final_row,
)


class SourceFinalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "method": "Confidence",
                "action_policy": "SharedAction",
                "mode": "fixed_call",
                "nominal_budget": "0.1",
                "official_map50_95": "0.70",
                "request_kb_per_image": "20.0",
            },
            {
                "method": "M2E",
                "action_policy": "GBDT-Hierarchical",
                "mode": "fixed_call",
                "nominal_budget": "0.1",
                "official_map50_95": "0.71",
                "request_kb_per_image": "18.0",
            },
        ]

    def test_finds_exact_frozen_source_row(self) -> None:
        row = find_frozen_final_row(
            self.rows,
            {
                "mode": "fixed_call",
                "budget": 0.1,
                "frozen_ranker": "M2E",
                "frozen_policy": "GBDT-Hierarchical",
            },
        )
        self.assertEqual(row["official_map50_95"], "0.71")

    def test_rejects_missing_or_duplicate_frozen_source_row(self) -> None:
        choice = {
            "mode": "fixed_call",
            "budget": 0.1,
            "frozen_ranker": "M2E",
            "frozen_policy": "GBDT-Hierarchical",
        }
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            find_frozen_final_row(self.rows + [dict(self.rows[1])], choice)

    def test_final_oracle_is_diagnostic_max_ap_with_lower_kb_tie_break(self) -> None:
        rows = self.rows + [
            {
                "method": "SemanticRank",
                "action_policy": "SharedAction",
                "mode": "fixed_call",
                "nominal_budget": "0.1",
                "official_map50_95": "0.71",
                "request_kb_per_image": "17.0",
            }
        ]
        oracle = choose_final_oracle(
            rows,
            mode="fixed_call",
            budget=0.1,
            selectable_rankers=("Confidence", "M2E", "SemanticRank"),
            ranker_order=("Confidence", "M2E", "SemanticRank"),
            policy_order=("SharedAction", "GBDT-Hierarchical"),
        )
        self.assertEqual(oracle["method"], "SemanticRank")
        self.assertTrue(oracle["diagnostic_only"])

    def test_action_selection_key_matches_frozen_source_format(self) -> None:
        self.assertEqual(
            action_selection_key(
                mode="fixed_byte",
                policy="SharedAction",
                ranker="Confidence",
                budget=0.05,
            ),
            "fixed_byte|SharedAction|Confidence|0.05",
        )


if __name__ == "__main__":
    unittest.main()
