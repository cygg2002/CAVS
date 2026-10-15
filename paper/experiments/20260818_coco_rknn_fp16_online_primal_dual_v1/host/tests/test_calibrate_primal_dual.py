from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


HOST = Path(__file__).resolve().parents[1]
BOARD_SCRIPTS = HOST.parent / "board_bundle" / "scripts"
sys.path.insert(0, str(HOST))
sys.path.insert(0, str(BOARD_SCRIPTS))

from calibrate_primal_dual import calibrate_primal_dual_control


class CalibratePrimalDualTest(unittest.TestCase):
    def _bundle(self) -> dict:
        records = []
        for image_id in range(1, 9):
            records.append({
                "image_id": image_id,
                "payload_bytes": {"edge": 0, "original": 100, "q90": 30},
                "scores": {"edge": 0.0, "original": 1.1, "q90": 1.0},
                "objective_values": {"edge": 0.0, "original": 1.1, "q90": 1.0},
            })
        return {
            "schema": "cavs_online_calibration_records_v1",
            "source_split": "calibration",
            "final_used_for_tuning": False,
            "protocol_id": "coco_rknn_fp16_causal_online_hil_v1",
            "method": "cavs_oqr",
            "action_ids": ["edge", "original", "q90"],
            "calibration_objective": {
                "name": "predicted_action_value",
                "direction": "maximize",
            },
            "source_hashes": {"policy": "a" * 64},
            "records": records,
        }

    def _base(self) -> dict:
        return {
            "schema": "cavs_online_control_v1",
            "formal": True,
            "protocol_id": "coco_rknn_fp16_causal_online_hil_v1",
            "method": "cavs_oqr",
            "budget": 0.25,
            "budget_numerator": 1,
            "budget_denominator": 4,
            "lambda": 0.3,
            "median_original_bytes": 100.0,
            "source_split": "calibration",
            "source_hashes": {
                "policy": "a" * 64,
                "calibration_records": "b" * 64,
                "protocol": "c" * 64,
            },
            "final_used_for_tuning": False,
        }

    def test_freezes_one_calibration_only_adaptive_control(self) -> None:
        control = calibrate_primal_dual_control(
            self._bundle(),
            base_control=self._base(),
            step_size_grid=(0.0, 0.001, 0.01),
            minimum_price=0.0,
            maximum_price=5.0,
            base_control_sha256="d" * 64,
        )
        self.assertEqual(control["schema"], "cavs_online_primal_dual_control_v1")
        self.assertEqual(control["controller"], "projected_primal_dual")
        self.assertIn(control["step_size"], control["step_size_grid"])
        self.assertEqual(control["lambda_initial"], 0.3)
        self.assertFalse(control["final_used_for_tuning"])
        self.assertEqual(control["source_hashes"]["base_control"], "d" * 64)
        self.assertTrue(all(row["maximum_prefix_violation_bytes"] == 0 for row in control["calibration_trials"]))

    def test_rejects_non_calibration_or_final_tainted_inputs(self) -> None:
        for field, value in (("source_split", "final"), ("final_used_for_tuning", True)):
            bundle = copy.deepcopy(self._bundle())
            bundle[field] = value
            with self.assertRaises(ValueError):
                calibrate_primal_dual_control(
                    bundle,
                    base_control=self._base(),
                    step_size_grid=(0.0,),
                    minimum_price=0.0,
                    maximum_price=5.0,
                    base_control_sha256="d" * 64,
                )

    def test_zero_step_trial_matches_the_frozen_fixed_control(self) -> None:
        control = calibrate_primal_dual_control(
            self._bundle(),
            base_control=self._base(),
            step_size_grid=(0.0,),
            minimum_price=0.0,
            maximum_price=5.0,
            base_control_sha256="d" * 64,
        )
        trial = control["calibration_trials"][0]
        self.assertEqual(trial["step_size"], 0.0)
        self.assertEqual(trial["final_price"], 0.3)
        self.assertEqual(trial["maximum_prefix_violation_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
