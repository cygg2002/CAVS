from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from baseline_core import (  # noqa: E402
    assert_result_matrix_complete,
    expected_result_rows,
    validate_protocol,
)


class ProtocolContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_protocol_has_ten_work_points_and_65_rows(self) -> None:
        validate_protocol(self.protocol)
        self.assertEqual(
            sum(len(values) for values in self.protocol["work_points"].values()),
            10,
        )
        self.assertEqual(expected_result_rows(self.protocol), 65)

    def test_final_is_evaluation_only(self) -> None:
        self.assertTrue(self.protocol["fit_only_training"])
        self.assertFalse(self.protocol["calibration_used_for_model_training"])
        self.assertTrue(
            self.protocol["calibration_teacher_gain_used_for_fixed_byte_lambda"]
        )
        self.assertFalse(self.protocol["final_used_for_selection"])

    def test_external_and_adapted_names_are_not_conflated(self) -> None:
        self.assertEqual(self.protocol["native_methods"], ["DCSB-Native"])
        self.assertIn("DCSB-Score-Adapted", self.protocol["continuous_methods"])
        self.assertNotIn("MORIC-style", self.protocol["continuous_methods"])

    def test_result_matrix_rejects_duplicates_and_missing_rows(self) -> None:
        rows = []
        for method in self.protocol["continuous_methods"]:
            for payload in self.protocol["payload_policies"]:
                for mode, budgets in self.protocol["work_points"].items():
                    for budget in budgets:
                        rows.append(
                            {
                                "method": method,
                                "payload_policy": payload,
                                "mode": mode,
                                "nominal_budget": budget,
                            }
                        )
        for method in self.protocol["native_methods"]:
            for payload in self.protocol["payload_policies"]:
                rows.append(
                    {
                        "method": method,
                        "payload_policy": payload,
                        "mode": "native_rule",
                        "nominal_budget": -1.0,
                    }
                )
        rows.extend(
            {
                "method": method,
                "payload_policy": "system",
                "mode": "system",
                "nominal_budget": 0.0,
            }
            for method in self.protocol["system_references"]
        )
        assert_result_matrix_complete(rows, self.protocol)
        with self.assertRaises(RuntimeError):
            assert_result_matrix_complete([*rows, dict(rows[0])], self.protocol)


if __name__ == "__main__":
    unittest.main()
