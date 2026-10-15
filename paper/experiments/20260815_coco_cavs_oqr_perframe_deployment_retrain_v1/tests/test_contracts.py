from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parents[2] / "examples/YOLO26-Edge-Cloud"))

import contracts
from inference_contract import current_roi_xyxy
try:
    from current_maur_uploader import current_roi_xyxy as deployed_current_roi_xyxy
except ImportError:
    deployed_current_roi_xyxy = None


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads((ROOT / "configs/protocol.json").read_text())

    def test_protocol_and_job_matrix(self) -> None:
        contracts.validate_protocol(self.protocol)
        jobs = contracts.training_jobs(self.protocol)
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len({job["id"] for job in jobs}), 18)

    def test_batch_sources_are_rejected(self) -> None:
        marker = object()
        self.assertIs(contracts.require_single_image(marker), marker)
        with self.assertRaises(TypeError):
            contracts.require_single_image([marker])
        with self.assertRaises(TypeError):
            contracts.require_single_image((marker,))

    def test_full_and_roi_preprocessing_are_explicit(self) -> None:
        full = contracts.action_predict_kwargs(self.protocol, "q90")
        roi = contracts.action_predict_kwargs(self.protocol, "current@640")
        self.assertFalse(full["scaleup"])
        self.assertFalse(full["half"])
        self.assertTrue(roi["scaleup"])
        self.assertTrue(roi["half"])
        self.assertFalse(full["end2end"])
        self.assertTrue(full["rect"])

    def test_canonical_population_is_frozen(self) -> None:
        contracts.validate_population(range(9678), role="fit")
        with self.assertRaises(RuntimeError):
            contracts.validate_population(range(9677), role="fit")

    def test_roi_geometry_matches_deployed_helper(self) -> None:
        if deployed_current_roi_xyxy is None:
            self.skipTest("deployment helper is not installed in this workspace")
        boxes = [
            {"xyxy": [10.2, 20.4, 50.8, 80.1], "conf": 0.9},
            {"xyxy": [100.0, 120.0, 180.0, 220.0], "conf": 0.6},
        ]
        expected = deployed_current_roi_xyxy(640, 480, boxes)
        self.assertEqual(current_roi_xyxy(640, 480, boxes), expected)


if __name__ == "__main__":
    unittest.main()
