from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/build_controls.py"
    spec = importlib.util.spec_from_file_location("test_dcsb_af_build_controls", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILD = load_module()


class BuildControlsTests(unittest.TestCase):
    def test_budget_records_use_calibration_percentile_not_raw_margin(self):
        bundle = BUILD.calibration_records_bundle(
            method="dcsb_budget",
            image_ids=np.asarray([1, 2]),
            original_bytes=np.asarray([100, 200]),
            raw_margin=np.asarray([-1.0, 2.0]),
            percentile=np.asarray([0.5, 1.0]),
            protocol_id="protocol",
            rknn_sha256="1" * 64,
            source_hashes={"policy": "2" * 64},
        )
        self.assertEqual(bundle["source_split"], "calibration")
        self.assertEqual(bundle["records"][0]["scores"]["original"], 0.5)
        self.assertFalse(bundle["final_used_for_tuning"])

    def test_native_records_preserve_zero_margin_decision_boundary(self):
        bundle = BUILD.calibration_records_bundle(
            method="adaptive_feeding_native",
            image_ids=np.asarray([1]),
            original_bytes=np.asarray([100]),
            raw_margin=np.asarray([-0.25]),
            percentile=np.asarray([1.0]),
            protocol_id="protocol",
            rknn_sha256="1" * 64,
            source_hashes={"policy": "2" * 64},
        )
        self.assertEqual(bundle["records"][0]["scores"]["original"], -0.25)

    def test_formal_matrix_has_six_budget_rows_and_two_native_rows(self):
        rows = BUILD.formal_rows("policies/model/manifest.json", "a" * 64)
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(row["work_point"] == "budget_matched" for row in rows), 6)
        self.assertEqual(sum(row["work_point"] == "native" for row in rows), 2)
        self.assertTrue(all(row["zero_lookahead"] for row in rows))
        self.assertTrue(all(row["global_plan"] is False for row in rows))
        self.assertTrue(all(row["buffer_frames"] == 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
