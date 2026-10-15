from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extension_core import (  # noqa: E402
    assert_result_matrix_complete,
    assert_same_membership,
    expected_result_rows,
    frozen_policy_thresholds,
    validate_protocol,
)


class ExtensionCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_frozen_protocol_is_complete(self) -> None:
        validate_protocol(self.protocol)
        self.assertEqual(expected_result_rows(self.protocol), 40)

    def test_result_matrix_rejects_missing_and_duplicate_rows(self) -> None:
        rows = [
            {
                "method": method,
                "payload_policy": policy,
                "mode": mode,
                "nominal_budget": budget,
            }
            for method in self.protocol["rankers"]
            for policy in self.protocol["payload_policies"]
            for mode, budgets in self.protocol["work_points"].items()
            for budget in budgets
        ]
        assert_result_matrix_complete(rows, self.protocol)
        with self.assertRaises(RuntimeError):
            assert_result_matrix_complete(rows[:-1], self.protocol)
        with self.assertRaises(RuntimeError):
            assert_result_matrix_complete([*rows, dict(rows[0])], self.protocol)

    def test_fixed_call_and_online_membership_must_not_change(self) -> None:
        assert_same_membership(
            {
                "CostEligible-AllROI": {1: "roi", 2: "q90"},
                "GBDT-Hierarchical": {1: "original", 2: "roi"},
            }
        )
        with self.assertRaises(RuntimeError):
            assert_same_membership(
                {
                    "CostEligible-AllROI": {1: "roi", 2: "q90"},
                    "GBDT-Hierarchical": {1: "original"},
                }
            )

    def test_thresholds_are_taken_from_shared_frozen_choices(self) -> None:
        choices = {
            "shared": {
                "fixed_call": {"threshold": 0.9},
                "online": {"threshold": 0.8},
                "fixed_byte": {"threshold": 0.7},
            }
        }
        observed = frozen_policy_thresholds(choices)
        self.assertEqual(
            observed,
            {
                "fixed_call": {
                    "CostEligible-AllROI": float("-inf"),
                    "GBDT-Hierarchical": 0.9,
                },
                "online": {
                    "CostEligible-AllROI": float("-inf"),
                    "GBDT-Hierarchical": 0.8,
                },
                "fixed_byte": {
                    "CostEligible-AllROI": float("-inf"),
                    "GBDT-Hierarchical": 0.7,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
