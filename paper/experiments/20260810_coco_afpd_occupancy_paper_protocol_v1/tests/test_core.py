from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_core():
    path = ROOT / "scripts/core.py"
    spec = importlib.util.spec_from_file_location("afpd_paper_core_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_frozen_and_marks_final_reuse(self) -> None:
        core = load_core()
        protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )
        core.validate_protocol(protocol)
        self.assertEqual(
            protocol["roles"], {"fit": 9678, "calibration": 571, "final": 5000}
        )
        self.assertFalse(protocol["final_is_hidden"])
        self.assertFalse(protocol["occupancy_can_replace_primary"])

    def test_expected_row_counts_are_exact(self) -> None:
        core = load_core()
        protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )
        self.assertEqual(core.expected_oof_action_rows(protocol), 96)
        self.assertEqual(core.expected_oof_occupancy_rows(protocol), 72)
        self.assertEqual(core.expected_role_rows(protocol), 36)

    def test_preflight_keeps_final_unread(self) -> None:
        core = load_core()
        protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )
        self.assertEqual(core.preflight_roles(protocol), ("fit", "calibration"))


class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.core = load_core()
        self.gate = {
            "minimum_fixed_byte_gain": 0.001,
            "required_fixed_byte_gain_points": 2,
            "maximum_other_loss": 0.001,
            "maximum_fixed_call_loss": 0.001,
            "minimum_capacity_utilization": 0.98,
            "minimum_supporting_seeds": 2,
        }

    def test_delta_gate_passes_two_primary_gains(self) -> None:
        points = [
            {"mode": "fixed_byte", "budget": 0.1, "delta_ap": 0.0012, "capacity_utilization": 0.999},
            {"mode": "fixed_byte", "budget": 0.25, "delta_ap": 0.0014, "capacity_utilization": 0.999},
            {"mode": "fixed_byte", "budget": 0.4, "delta_ap": -0.0008, "capacity_utilization": 0.999},
            {"mode": "fixed_call", "budget": 0.1, "delta_ap": -0.0009, "capacity_utilization": 1.0},
            {"mode": "fixed_call", "budget": 0.25, "delta_ap": 0.0, "capacity_utilization": 1.0},
            {"mode": "fixed_call", "budget": 0.4, "delta_ap": 0.0001, "capacity_utilization": 1.0},
        ]
        self.assertTrue(self.core.evaluate_delta_gate(points, self.gate)["passed"])

    def test_delta_gate_rejects_fixed_call_regression(self) -> None:
        points = [
            {"mode": "fixed_byte", "budget": b, "delta_ap": 0.002, "capacity_utilization": 1.0}
            for b in (0.1, 0.25, 0.4)
        ] + [
            {"mode": "fixed_call", "budget": b, "delta_ap": -0.0011 if b == 0.25 else 0.0, "capacity_utilization": 1.0}
            for b in (0.1, 0.25, 0.4)
        ]
        self.assertFalse(self.core.evaluate_delta_gate(points, self.gate)["passed"])

    def test_final_gate_depends_only_on_primary_calibration_gate(self) -> None:
        self.assertTrue(
            self.core.should_run_final(
                primary_calibration_gate={"passed": True},
                occupancy_results={"OccupancyFree": {"passed": False}},
            )
        )
        self.assertFalse(
            self.core.should_run_final(
                primary_calibration_gate={"passed": False},
                occupancy_results={"OccupancyFree": {"passed": True}},
            )
        )


class AttributionTests(unittest.TestCase):
    def test_two_by_two_attribution(self) -> None:
        core = load_core()
        values = {
            "Current-OQ": 0.50,
            "Current-OQR": 0.503,
            "AFPD-OQ": 0.501,
            "AFPD-OQR": 0.507,
        }
        result = core.two_by_two_attribution(values)
        self.assertAlmostEqual(result["roi_effect_current"], 0.003)
        self.assertAlmostEqual(result["objective_effect_oq"], 0.001)
        self.assertAlmostEqual(result["objective_effect_oqr"], 0.004)
        self.assertAlmostEqual(result["interaction"], 0.003)
        self.assertAlmostEqual(result["total_effect"], 0.007)

    def test_capacity_assertion_rejects_overflow(self) -> None:
        core = load_core()
        with self.assertRaises(ValueError):
            core.assert_fixed_byte_capacity(101.0, 100.0)


if __name__ == "__main__":
    unittest.main()
