from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_experiment import (  # noqa: E402
    install_read_only_action_links,
    planned_fit_jobs,
    validate_stage,
)


class RunExperimentTest(unittest.TestCase):
    def test_stage_rejects_final(self) -> None:
        self.assertEqual(validate_stage("preflight"), "preflight")
        self.assertEqual(validate_stage("fit-cache"), "fit-cache")
        with self.assertRaisesRegex(ValueError, "final"):
            validate_stage("final")

    def test_fit_plan_contains_four_geometry_and_twelve_inference_jobs(self) -> None:
        config = {
            "profiles": ["current", "moderate", "historical_cost", "fixed_1_8"],
            "imgsz": [160, 320, 640],
        }
        jobs = planned_fit_jobs(config)
        self.assertEqual(sum(value["kind"] == "geometry" for value in jobs), 4)
        self.assertEqual(sum(value["kind"] == "inference" for value in jobs), 12)
        self.assertTrue(all(value["role"] == "fit" for value in jobs))

    def test_action_links_point_only_to_calibration_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            for role in ("calibration", "audit"):
                role_root = source / "runs" / "actions" / role
                role_root.mkdir(parents=True)
                (role_root / "marker.json").write_text(json.dumps({"role": role}))
            config = {
                "source_scan_root": str(source),
                "output_root": str(output),
            }
            installed = install_read_only_action_links(config)
            self.assertEqual(set(installed), {"calibration", "audit"})
            for role, target in installed.items():
                link = output / "runs" / "actions" / role
                self.assertTrue(link.is_symlink())
                self.assertEqual(link.resolve(), Path(target).resolve())
            self.assertFalse((output / "runs" / "actions" / "final").exists())


if __name__ == "__main__":
    unittest.main()
