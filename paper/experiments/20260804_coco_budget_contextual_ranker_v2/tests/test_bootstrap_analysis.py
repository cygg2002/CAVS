from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_fixed_results import selection_overlap, summarize_bootstrap  # noqa: E402


class BootstrapAnalysisTest(unittest.TestCase):
    def test_bootstrap_summary_reports_interval_and_probability(self) -> None:
        summary = summarize_bootstrap(np.asarray([-1.0, 1.0, 2.0, 3.0]))
        self.assertAlmostEqual(summary["mean"], 1.25)
        self.assertEqual(summary["probability_gt_zero"], 0.75)
        self.assertLess(summary["ci_low"], summary["ci_high"])

    def test_selection_overlap_uses_union_and_smaller_set_recall(self) -> None:
        observed = selection_overlap({1, 2, 3}, {2, 3, 4, 5})
        self.assertAlmostEqual(observed["jaccard"], 2 / 5)
        self.assertAlmostEqual(observed["overlap_over_smaller"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
