from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_diagnostic import assert_final_allowed, roles_for_stage  # noqa: E402


class RunnerContractTest(unittest.TestCase):
    def test_calibration_stage_cannot_request_final(self) -> None:
        self.assertEqual(roles_for_stage("calibration"), ("calibration",))
        self.assertEqual(roles_for_stage("final"), ("final",))
        with self.assertRaises(ValueError):
            roles_for_stage("all")

    def test_final_requires_passed_frozen_gate(self) -> None:
        assert_final_allowed({"passed": True, "frozen": True})
        with self.assertRaises(RuntimeError):
            assert_final_allowed({"passed": False, "frozen": True})
        with self.assertRaises(RuntimeError):
            assert_final_allowed({"passed": True, "frozen": False})


if __name__ == "__main__":
    unittest.main()
