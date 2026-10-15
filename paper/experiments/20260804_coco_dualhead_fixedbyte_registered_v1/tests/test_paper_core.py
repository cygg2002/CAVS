from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from paper_core import (  # noqa: E402
    assert_final_allowed,
    deterministic_evaluation_roles,
    fixed_byte_audit_gate,
    fixed_byte_final_gate,
    freeze_primary_comparators,
    validate_protocol,
)


class PaperCoreTest(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / "configs/protocol.json").read_text())

    def test_frozen_protocol_is_valid(self):
        validate_protocol(self.protocol)

    def test_roles_are_exact_and_exclude_both_prior_populations(self):
        population = range(1, 100)
        canonical = {2, 4, 6, 8}
        previous = {1, 3, 5, 7, 9}
        first = deterministic_evaluation_roles(
            population,
            canonical | previous,
            {"audit": 10, "registered_final": 20},
            seed=17,
        )
        second = deterministic_evaluation_roles(
            reversed(list(population)),
            previous | canonical,
            {"audit": 10, "registered_final": 20},
            seed=17,
        )
        self.assertEqual(first, second)
        self.assertEqual({key: len(value) for key, value in first.items()}, {"audit": 10, "registered_final": 20})
        observed = {value for values in first.values() for value in values}
        self.assertFalse(observed & canonical)
        self.assertFalse(observed & previous)

    def test_primary_comparators_are_frozen_from_fixed_byte_only(self):
        rows = []
        for mode in ("fixed_call", "fixed_byte"):
            for budget in (0.1, 0.25, 0.4):
                rows.extend(
                    [
                        {"method": "ORIC-AP5095", "mode": mode, "budget": budget, "map50_95": 0.52},
                        {"method": "MORIC-AP5095", "mode": mode, "budget": budget, "map50_95": 0.53 if mode == "fixed_byte" else 0.51},
                    ]
                )
        frozen = freeze_primary_comparators(rows, mode="fixed_byte")
        self.assertEqual(frozen, {"fixed_byte:0.1": "MORIC-AP5095", "fixed_byte:0.25": "MORIC-AP5095", "fixed_byte:0.4": "MORIC-AP5095"})

    def test_audit_gate_requires_two_fixed_byte_wins_and_capacity(self):
        comparisons = [{"ap_gain": 0.002}, {"ap_gain": 0.0012}, {"ap_gain": -0.0008}]
        capacities = [{"used": 10.0, "capacity": 10.0}] * 3
        self.assertTrue(
            fixed_byte_audit_gate(
                comparisons,
                capacities,
                minimum_gain=0.001,
                required_gain_points=2,
                maximum_other_loss=0.001,
            )["passed"]
        )
        capacities[-1] = {"used": 10.1, "capacity": 10.0}
        self.assertFalse(
            fixed_byte_audit_gate(
                comparisons,
                capacities,
                minimum_gain=0.001,
                required_gain_points=2,
                maximum_other_loss=0.001,
            )["passed"]
        )

    def test_final_gate_uses_fixed_byte_intervals_and_capacity(self):
        intervals = [{"ci_low": 0.0002}, {"ci_low": 0.0001}, {"ci_low": -0.0005}]
        capacities = [{"used": 9.9, "capacity": 10.0}] * 6
        gate = fixed_byte_final_gate(
            intervals,
            capacities,
            required_superior_points=2,
            noninferiority_margin=0.001,
        )
        self.assertTrue(gate["passed"])

    def test_final_requires_passed_audit_and_is_one_time(self):
        assert_final_allowed({"complete": True, "passed": True}, final_exists=False)
        with self.assertRaises(RuntimeError):
            assert_final_allowed({"complete": True, "passed": False}, final_exists=False)
        with self.assertRaises(RuntimeError):
            assert_final_allowed({"complete": True, "passed": True}, final_exists=True)


if __name__ == "__main__":
    unittest.main()
