from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_core():
    path = ROOT / "scripts/calibration_core.py"
    spec = importlib.util.spec_from_file_location("oq_advantage_calibration_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AdvantageCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_core()

    def test_q_anchor_preserves_q90_and_realizes_calibrated_advantage(self):
        scores = np.asarray([[0.0, 0.8, 0.5, 0.2], [0.0, 0.3, 0.4, 0.1]])
        advantage = np.asarray([0.1, -0.2])
        adjusted = self.core.replace_oq_advantage(scores, advantage, anchor="q90")
        np.testing.assert_allclose(adjusted[:, 2], scores[:, 2])
        np.testing.assert_allclose(adjusted[:, 1] - adjusted[:, 2], advantage)
        np.testing.assert_allclose(adjusted[:, [0, 3]], scores[:, [0, 3]])

    def test_midpoint_preserves_midpoint_and_realizes_calibrated_advantage(self):
        scores = np.asarray([[0.0, 0.8, 0.5, 0.2], [0.0, 0.3, 0.4, 0.1]])
        advantage = np.asarray([0.1, -0.2])
        adjusted = self.core.replace_oq_advantage(scores, advantage, anchor="midpoint")
        np.testing.assert_allclose(
            (adjusted[:, 1] + adjusted[:, 2]) / 2.0,
            (scores[:, 1] + scores[:, 2]) / 2.0,
        )
        np.testing.assert_allclose(adjusted[:, 1] - adjusted[:, 2], advantage)
        np.testing.assert_allclose(adjusted[:, [0, 3]], scores[:, [0, 3]])

    def test_q90_preferred_only_suppresses_original_when_q90_is_available(self):
        scores = np.asarray([[0.0, 0.8, 0.5, 0.2], [0.0, 0.3, -1e9, 0.1]])
        adjusted = self.core.prefer_q90(scores)
        self.assertLess(adjusted[0, 1], -1e8)
        self.assertEqual(adjusted[1, 1], scores[1, 1])
        np.testing.assert_allclose(adjusted[:, [0, 2, 3]], scores[:, [0, 2, 3]])

    def test_original_margin_changes_only_original_scores(self):
        scores = np.asarray(
            [[0.0, 0.8, 0.5, 0.2], [0.0, 0.3, 0.4, 0.1], [0.0, 0.7, -1e9, 0.3]]
        )
        adjusted = self.core.apply_original_margin(scores, margin=0.15)
        np.testing.assert_allclose(adjusted[:2, 1], scores[:2, 1] - 0.15)
        self.assertEqual(adjusted[2, 1], scores[2, 1])
        np.testing.assert_allclose(adjusted[:, [0, 2, 3]], scores[:, [0, 2, 3]])

    def test_protocol_forbids_final_fit_and_source_modification(self):
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        self.core.validate_protocol(protocol)
        self.assertEqual(
            protocol["protocol_id"],
            "oregon_cloudv3_oq_advantage_calibration_diagnostic_v1",
        )
        self.assertEqual(protocol["source_method"], "CAVS-OQR-MAUR")
        protocol["fit_source"] = "calibration"
        with self.assertRaises(ValueError):
            self.core.validate_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
