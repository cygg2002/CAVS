from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from prepare_int8_bundle import CLOUD_SHA256, PROTOCOL_ID, RKNN_SHA256, prepare_bundle


EXPERIMENTS = ROOT.parent
SOURCE = EXPERIMENTS / "20260828_coco_rknn_fp16_online_hil_final_v2"
POLICY = EXPERIMENTS / "20260816_coco_rknn_cavs_oqr_deployment_retrain_v1/runs/backends/int8-heads/export/policy"


class PrepareInt8BundleTest(unittest.TestCase):
    def test_creates_a_cavs_only_int8_bundle_with_distinct_protocol(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "isolated"
            result = prepare_bundle(
                source_bundle=SOURCE / "board_bundle",
                policy_source=POLICY,
                source_host=SOURCE / "host",
                output=output,
            )
            protocol = json.loads((output / "board_bundle/configs/protocol.json").read_text())
            self.assertEqual(protocol["protocol_id"], PROTOCOL_ID)
            self.assertEqual(protocol["rknn_sha256"], RKNN_SHA256)
            self.assertEqual(protocol["models"]["int8_heads"]["sha256"], RKNN_SHA256)
            self.assertEqual(protocol["models"]["cloud_sha256"], CLOUD_SHA256)
            self.assertEqual(set(protocol["methods"]), {"cavs_oqr"})
            self.assertEqual(protocol["methods"]["cavs_oqr"]["actions"], ["edge", "original", "q90", "current@640"])
            self.assertFalse((output / "board_bundle/BUNDLE_MANIFEST.json").exists())
            self.assertTrue((output / "board_bundle/controls/cavs_oqr").is_dir())
            self.assertIn("int8_heads", (output / "board_bundle/scripts/run_contract.py").read_text())
            self.assertTrue((output / "host/build_calibration_records.py").is_file())
            self.assertTrue((output / "board_formal/prepare_controls.sh").is_file())
            self.assertTrue((output / "board_formal/run_cavs_oqr_three_budgets.sh").is_file())
            self.assertIn(PROTOCOL_ID, (output / "board_bundle/tests/test_online_runner.py").read_text())
            self.assertTrue(Path(result["setup"]).is_file())

    def test_rejects_a_nonempty_output_directory(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "isolated"
            output.mkdir()
            (output / "occupied").write_text("x")
            with self.assertRaises(FileExistsError):
                prepare_bundle(
                    source_bundle=SOURCE / "board_bundle",
                    policy_source=POLICY,
                    source_host=SOURCE / "host",
                    output=output,
                )


if __name__ == "__main__":
    unittest.main()
