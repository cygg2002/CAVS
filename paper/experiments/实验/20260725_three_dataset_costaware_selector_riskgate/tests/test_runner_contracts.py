from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_selector import (  # noqa: E402
    build_provisional_choices,
    calibration_input_paths,
    source_root,
)


class RunnerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_calibration_stage_input_allowlist_excludes_final_artifacts(self) -> None:
        paths = calibration_input_paths("coco")
        names = {path.name for path in paths}
        self.assertIn("calibration_curve.csv", names)
        self.assertIn("threshold_choices.json", names)
        self.assertNotIn("results.csv", names)
        self.assertNotIn("selections.json", names)
        self.assertNotIn("paired_bootstrap.csv", names)

    def test_completed_source_builds_ten_choices_and_240_candidates(self) -> None:
        source = source_root()
        choices, candidates = build_provisional_choices(
            dataset="ena24",
            protocol=self.protocol,
            calibration_curve=source / "runs/ena24/calibration_curve.csv",
            threshold_choices=source / "runs/ena24/threshold_choices.json",
        )
        self.assertEqual(len(choices), 10)
        self.assertEqual(len(candidates), 240)
        self.assertEqual(
            {(row["mode"], float(row["budget"])) for row in choices},
            {
                ("fixed_call", 0.1),
                ("fixed_call", 0.25),
                ("fixed_call", 0.4),
                ("online", 0.1),
                ("online", 0.25),
                ("online", 0.4),
                ("fixed_byte", 0.05),
                ("fixed_byte", 0.1),
                ("fixed_byte", 0.2),
                ("fixed_byte", 0.4),
            },
        )
        self.assertTrue(all(row["selection_source_role"] == "calibration" for row in choices))
        self.assertTrue(all(row["dataset"] == "ena24" for row in choices))


if __name__ == "__main__":
    unittest.main()
