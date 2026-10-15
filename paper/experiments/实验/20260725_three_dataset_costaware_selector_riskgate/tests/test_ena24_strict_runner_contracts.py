from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_ena24_strict import (  # noqa: E402
    STRICT_MARKER,
    finalize_identical_bootstrap,
    strict_gate_csv_row,
    strict_input_paths,
    supersede_legacy_provisional,
)


class Ena24StrictRunnerContractTests(unittest.TestCase):
    def test_strict_inputs_do_not_include_final_artifacts(self) -> None:
        names = {path.name for path in strict_input_paths()}
        self.assertNotIn("results.csv", names)
        self.assertNotIn("selections.json", names)
        self.assertNotIn("final", {path.parent.name for path in strict_input_paths()})

    def test_superseding_legacy_marker_records_hashes_and_removes_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "PROVISIONAL_FROZEN").write_text("ok\n", encoding="ascii")
            (output / "provisional_choices.json").write_text(
                json.dumps([{"legacy": True}]), encoding="utf-8"
            )

            record = supersede_legacy_provisional(output)

            self.assertFalse((output / "PROVISIONAL_FROZEN").exists())
            self.assertFalse((output / STRICT_MARKER).exists())
            self.assertEqual(
                record["status"], "superseded_pre_audit_full_calibration_prototype"
            )
            self.assertIn("provisional_choices.json", record["artifact_sha256"])

    def test_identical_actions_shortcut_still_records_200_replicates(self) -> None:
        result = finalize_identical_bootstrap(replicates=200)
        self.assertEqual(result["replicates"], 200)
        self.assertEqual(result["official_map50_95_delta_ci_low"], 0.0)
        self.assertEqual(result["official_map50_95_delta_ci_high"], 0.0)
        self.assertEqual(result["request_kb_per_image_delta_ci_high"], 0.0)
        self.assertTrue(result["identical_action_shortcut"])

    def test_gate_csv_identity_is_not_overwritten_by_bootstrap_metadata(self) -> None:
        row = {
            "dataset": "ena24",
            "mode": "online",
            "budget": 0.1,
            "reference": {"ranker": "Confidence", "policy": "SharedAction"},
            "provisional": {
                "ranker": "Confidence",
                "policy": "GBDT-Hierarchical",
            },
            "frozen": {"ranker": "Confidence", "policy": "SharedAction"},
            "gate_passed": False,
            "used_reference_fallback": True,
            "bootstrap": {
                "dataset": "wrong-dataset",
                "mode": "fixed_call",
                "nominal_budget": 0.25,
                "official_map50_95_delta_ci_low": -0.01,
            },
        }

        flattened = strict_gate_csv_row(row)

        self.assertEqual(flattened["dataset"], "ena24")
        self.assertEqual(flattened["mode"], "online")
        self.assertEqual(flattened["budget"], 0.1)
        self.assertNotIn("nominal_budget", flattened)


if __name__ == "__main__":
    unittest.main()
