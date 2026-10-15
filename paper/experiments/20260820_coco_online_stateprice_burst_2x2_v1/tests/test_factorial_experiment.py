from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from factorial_control import build_empirical_demand_curve
from factorial_experiment import calibrate_budget, replay_factorial


ACTIONS = ("edge", "original", "q90")


def record(image_id: int, original_score: float, q90_score: float):
    return {
        "image_id": image_id,
        "payload_bytes": {"edge": 0, "original": 100, "q90": 30},
        "scores": {"edge": 0.0, "original": original_score, "q90": q90_score},
        "objective_values": {"edge": 0.0, "original": original_score, "q90": q90_score},
    }


class FactorialExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            record(1, 1.0, 0.8),
            record(2, 0.2, 0.1),
            record(3, 1.1, 0.9),
            record(4, 0.3, 0.2),
        ]
        self.curve = build_empirical_demand_curve(
            self.records,
            action_ids=ACTIONS,
            median_original_bytes=100,
        )
        self.base = {
            "schema": "cavs_online_control_v1",
            "formal": True,
            "source_split": "calibration",
            "final_used_for_tuning": False,
            "budget_numerator": 1,
            "budget_denominator": 4,
            "lambda": 0.3,
            "median_original_bytes": 100,
        }

    def test_calibration_freezes_only_registered_burst_values(self) -> None:
        control, report = calibrate_budget(
            self.records,
            action_ids=ACTIONS,
            base_control=self.base,
            demand_curve=self.curve,
            burst_fraction_grid=(0.0, 0.1),
        )
        self.assertFalse(control["final_used_for_tuning"])
        self.assertEqual(control["source_split"], "calibration")
        self.assertIn(control["burst_fraction_by_price"]["fixed"], (0.0, 0.1))
        self.assertIn(control["burst_fraction_by_price"]["state_aware"], (0.0, 0.1))
        self.assertEqual(set(report["cells"]), {
            "fixed__strict_prefix",
            "fixed__paced_burst",
            "state_aware__strict_prefix",
            "state_aware__paced_burst",
        })

    def test_final_replay_has_four_complete_zero_buffer_cells(self) -> None:
        control, _ = calibrate_budget(
            self.records,
            action_ids=ACTIONS,
            base_control=self.base,
            demand_curve=self.curve,
            burst_fraction_grid=(0.0, 0.1),
        )
        report, selections = replay_factorial(
            self.records,
            action_ids=ACTIONS,
            frozen_control=control,
            demand_curve=self.curve,
        )
        self.assertFalse(report["final_used_for_tuning"])
        self.assertEqual(report["buffer_frames"], 0)
        self.assertTrue(report["zero_lookahead"])
        self.assertEqual(set(selections), set(report["cells"]))
        for cell in report["cells"].values():
            self.assertEqual(cell["frames"], 4)
            self.assertEqual(cell["maximum_envelope_violation_bytes"], 0)
            self.assertLessEqual(cell["spent_bytes"], cell["total_capacity_bytes"])

    def test_reference_selection_mismatch_is_rejected(self) -> None:
        control, _ = calibrate_budget(
            self.records,
            action_ids=ACTIONS,
            base_control=self.base,
            demand_curve=self.curve,
            burst_fraction_grid=(0.0,),
        )
        wrong = {str(row["image_id"]): "original" for row in self.records}
        with self.assertRaises(RuntimeError):
            replay_factorial(
                self.records,
                action_ids=ACTIONS,
                frozen_control=control,
                demand_curve=self.curve,
                expected_fixed_prefix=wrong,
            )


if __name__ == "__main__":
    unittest.main()
