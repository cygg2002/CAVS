from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


HOST = Path(__file__).resolve().parents[1]
BOARD_SCRIPTS = HOST.parent / "board_bundle" / "scripts"
sys.path.insert(0, str(HOST))
sys.path.insert(0, str(BOARD_SCRIPTS))

from run_primal_dual_replay import compare_replays


class PrimalDualReplayTest(unittest.TestCase):
    def _records(self) -> list[dict]:
        return [
            {
                "image_id": image_id,
                "payload_bytes": {"edge": 0, "original": 100, "q90": 30},
                "scores": {"edge": 0.0, "original": 1.1, "q90": 1.0},
            }
            for image_id in range(1, 21)
        ]

    def _control(self, step_size: float) -> dict:
        return {
            "schema": "cavs_online_primal_dual_control_v1",
            "controller": "projected_primal_dual",
            "method": "cavs_oqr",
            "budget_numerator": 1,
            "budget_denominator": 4,
            "lambda": 0.3,
            "lambda_initial": 0.3,
            "lambda_minimum": 0.0,
            "lambda_maximum": 5.0,
            "step_size": step_size,
            "median_original_bytes": 100.0,
            "source_split": "calibration",
            "final_used_for_tuning": False,
            "buffer_frames": 0,
            "zero_lookahead": True,
            "global_plan": False,
        }

    def test_reports_fixed_parity_and_exact_prefix_feasibility(self) -> None:
        report, selections = compare_replays(
            self._records(),
            action_ids=("edge", "original", "q90"),
            adaptive_control=self._control(0.0),
        )
        self.assertEqual(report["changed_frames"], 0)
        self.assertEqual(report["fixed"]["maximum_prefix_violation_bytes"], 0)
        self.assertEqual(report["adaptive"]["maximum_prefix_violation_bytes"], 0)
        self.assertEqual(len(selections["adaptive"]), 20)

    def test_rejects_a_control_that_buffers_or_uses_final_for_tuning(self) -> None:
        for field, value in (
            ("buffer_frames", 1),
            ("zero_lookahead", False),
            ("global_plan", True),
            ("final_used_for_tuning", True),
        ):
            control = self._control(0.001)
            control[field] = value
            with self.assertRaises(ValueError):
                compare_replays(
                    self._records(),
                    action_ids=("edge", "original", "q90"),
                    adaptive_control=control,
                )

    def test_command_line_entrypoint_resolves_the_board_controller(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(HOST / "run_primal_dual_replay.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
