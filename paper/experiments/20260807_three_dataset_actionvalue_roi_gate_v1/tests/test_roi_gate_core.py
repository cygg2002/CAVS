import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from roi_gate_core import (  # noqa: E402
    apply_roi_advantage_threshold,
    expected_result_keys,
    merge_fold_scores,
    oqr_score_filename,
    select_advantage_threshold,
    validate_protocol,
)


class ProtocolCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_frozen_protocol_is_valid(self):
        validate_protocol(self.protocol)

    def test_oregon_is_canonical_event_v1(self):
        changed = json.loads(json.dumps(self.protocol))
        changed["datasets"]["oregon"]["canonical_run"] = "cloudv2"
        with self.assertRaisesRegex(ValueError, "Oregon canonical run"):
            validate_protocol(changed)

    def test_expected_matrix_has_36_rows_per_dataset(self):
        keys = expected_result_keys(self.protocol, "ena24")
        self.assertEqual(len(keys), 36)

    def test_roi_threshold_preserves_oq_scores(self):
        scores = np.array(
            [[0.0, 0.4, 0.3, 0.6], [0.0, 0.5, 0.2, 0.55]], dtype=np.float64
        )
        gated = apply_roi_advantage_threshold(scores, threshold=0.15)
        np.testing.assert_array_equal(gated[:, :3], scores[:, :3])
        self.assertEqual(gated[0, 3], 0.6)
        self.assertLess(gated[1, 3], -1e8)

    def test_infinite_threshold_disables_roi(self):
        scores = np.array([[0.0, 0.4, 0.3, 100.0]], dtype=np.float64)
        gated = apply_roi_advantage_threshold(scores, threshold=float("inf"))
        self.assertLess(gated[0, 3], -1e8)

    def test_threshold_selection_obeys_harmful_rate_and_tie_break(self):
        candidates = [
            {"threshold": -1.0, "reward": 10.0, "harmful_rate": 0.5, "roi_count": 4},
            {"threshold": 0.1, "reward": 8.0, "harmful_rate": 0.0, "roi_count": 2},
            {"threshold": 0.2, "reward": 8.0, "harmful_rate": 0.0, "roi_count": 1},
        ]
        chosen = select_advantage_threshold(candidates, maximum_harmful_rate=0.1)
        self.assertEqual(chosen["threshold"], 0.2)
        self.assertEqual(chosen["roi_count"], 1)

    def test_fold_scores_preserve_budget_heldout_action_axes(self):
        target = np.full((2, 3, 5, 4), np.nan, dtype=np.float64)
        fold = np.arange(3 * 2 * 4, dtype=np.float64).reshape(3, 2, 4)
        merge_fold_scores(target, seed_index=1, heldout_indices=[1, 4], fold_scores=fold)
        np.testing.assert_array_equal(target[1][:, [1, 4], :], fold)
        self.assertTrue(np.isnan(target[0]).all())

    def test_oqr_score_filename_preserves_internal_role_name(self):
        self.assertEqual(oqr_score_filename("fit_oof"), "fit_oof_oqr.pkl")
        self.assertEqual(oqr_score_filename("calibration"), "calibration_oqr.pkl")


if __name__ == "__main__":
    unittest.main()
