from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_analysis import build_analysis, load_analysis_inputs  # noqa: E402


class FrozenInputContractTests(unittest.TestCase):
    def test_registered_candidate_grids_are_complete(self):
        loaded = load_analysis_inputs(ROOT)

        for dataset in ("coco", "ena24", "oregon"):
            self.assertEqual(len(loaded["selector"][dataset]), 240)
            self.assertEqual(len(loaded["final_matrix"][dataset]), 240)
            self.assertEqual(len(loaded["frozen"][dataset]), 10)

    def test_inputs_are_marked_posthoc_and_final_is_not_a_selection_source(self):
        loaded = load_analysis_inputs(ROOT)

        self.assertEqual(
            loaded["protocol"]["output_role"],
            "posthoc_diagnostic_not_confirmatory",
        )
        self.assertFalse(loaded["protocol"]["final_used_for_selection"])

    def test_analysis_outputs_are_complete_and_preserve_strict_infeasibility(self):
        analysis = build_analysis(load_analysis_inputs(ROOT), ROOT)

        self.assertEqual(len(analysis["shared_choices"]), 10)
        self.assertEqual(len(analysis["shared_final"]), 30)
        self.assertEqual(len(analysis["lesion_rows"]), 180)
        self.assertEqual(len(analysis["comparison_rows"]), 120)
        self.assertEqual(
            sum(row["strict_feasible"] for row in analysis["shared_choices"]),
            0,
        )


if __name__ == "__main__":
    unittest.main()
