from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dcsb_af_numpy import (  # noqa: E402
    adaptive_feeding_feature,
    dcsb_margin,
    empirical_percentile,
    load_policy,
)


class DcsbAdaptiveFeedingRuntimeTests(unittest.TestCase):
    def test_adaptive_feeding_feature_matches_source_order_and_padding(self):
        boxes = [
            {"xyxy": [10.0, 20.0, 30.0, 60.0], "conf": 0.4, "cls": 1},
            {"xyxy": [0.0, 5.0, 40.0, 25.0], "conf": 0.9, "cls": 0},
        ]
        feature = adaptive_feeding_feature(
            boxes, width=100, height=200, category_ids=(1, 2), top_k=2
        )
        np.testing.assert_allclose(
            feature,
            np.asarray(
                [
                    1.0,
                    1.0,
                    0.9,
                    0.0,
                    0.025,
                    0.4,
                    0.1,
                    0.4,
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                dtype=np.float64,
            ),
        )

    def test_dcsb_margin_preserves_native_easy_gate(self):
        model = {
            "confidence_threshold": 0.2,
            "detector_confidence": 0.5,
            "count_threshold": 2,
            "area_threshold": 0.2,
        }
        easy = [{"xyxy": [0, 0, 50, 50], "conf": 0.8, "cls": 0}]
        hard = easy + [{"xyxy": [0, 0, 10, 10], "conf": 0.3, "cls": 0}]
        self.assertEqual(dcsb_margin(easy, width=100, height=100, model=model), -1.0)
        self.assertGreater(dcsb_margin(hard, width=100, height=100, model=model), 0.0)

    def test_empirical_percentile_is_monotone_and_bounded(self):
        reference = np.asarray([-2.0, -1.0, -1.0, 3.0], dtype=np.float64)
        values = [empirical_percentile(value, reference) for value in (-3, -1, 0, 4)]
        self.assertEqual(values, sorted(values))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertEqual(values[1], 0.75)

    def test_policy_loader_rejects_a_tampered_parameter_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "adaptive_feeding.npz"
            np.savez(model, coefficient=np.zeros(2), intercept=np.asarray([0.0]))
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "policy_type": "dcsb_adaptive_feeding_original_only_rknn",
                        "rknn_model_sha256": "0" * 64,
                        "category_ids": [1, 2],
                        "top_k": 0,
                        "adaptive_feeding": {
                            "file": model.name,
                            "sha256": "f" * 64,
                        },
                        "dcsb": {},
                        "calibration_references": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_policy(manifest)


if __name__ == "__main__":
    unittest.main()
