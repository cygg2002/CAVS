from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/run_scan.py"
CONFIG = ROOT / "configs/protocol.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_scan", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RunScanTest(unittest.TestCase):
    def test_each_role_has_four_geometry_and_twelve_inference_jobs(self):
        runner = load_runner()
        config = json.loads(CONFIG.read_text())
        jobs = runner.planned_jobs(config, "calibration")
        self.assertEqual(
            sum(job["kind"] == "geometry" for job in jobs), 4
        )
        self.assertEqual(
            sum(job["kind"] == "inference" for job in jobs), 12
        )
        self.assertEqual(sum(job["kind"] == "evaluate" for job in jobs), 1)
        self.assertNotIn("final", json.dumps(jobs).lower())

    def test_stage_contract_only_allows_preflight_calibration_audit(self):
        runner = load_runner()
        self.assertEqual(
            runner.validate_stage("calibration"), "calibration"
        )
        with self.assertRaisesRegex(ValueError, "final"):
            runner.validate_stage("final")


if __name__ == "__main__":
    unittest.main()
