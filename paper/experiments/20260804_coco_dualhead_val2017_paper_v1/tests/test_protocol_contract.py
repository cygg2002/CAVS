from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProtocolContractTest(unittest.TestCase):
    def test_full_val2017_is_the_only_evaluation_population(self):
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        self.assertEqual(protocol["dataset"], "coco_val2017")
        self.assertEqual(protocol["test_images"], 5000)
        self.assertIn("instances_val2017.json", protocol["paths"]["annotations"])
        self.assertNotIn("audit", protocol)
        self.assertFalse(protocol["val_used_for_training"])
        self.assertFalse(protocol["val_used_for_selection"])

    def test_fixed_byte_comparators_are_frozen_from_development_oof(self):
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        self.assertEqual(
            protocol["frozen_comparators"],
            {
                "fixed_byte:0.1": "ORIC-AP5095",
                "fixed_byte:0.25": "ORIC-AP5095",
                "fixed_byte:0.4": "ORIC-AP5095",
            },
        )

    def test_network_contract_matches_the_frozen_paper_e2e_model(self):
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        self.assertEqual(
            protocol["network"],
            {
                "uplink_mbps": 10.0,
                "downlink_mbps": 20.0,
                "rtt_ms": 40.0,
                "edge_ms": 18.23,
                "router_ms": 1.0,
                "codec_ms": 2.0,
                "cloud_ms": 18.0,
                "fusion_ms": 1.0,
                "response_bytes_per_detection": 24,
            },
        )

    def test_runner_rejects_train2017_paths_and_requires_exact_population(self):
        runner = ROOT / "scripts/run_paper.py"
        if not runner.is_file():
            self.fail("run_paper.py is not implemented")
        source = runner.read_text()
        self.assertIn('"/val2017/"', source)
        self.assertIn('"/train2017/"', source)
        self.assertIn("expected_images=5000", source)

    def test_runner_uses_frozen_checkpoint_without_training(self):
        runner = ROOT / "scripts/run_paper.py"
        if not runner.is_file():
            self.fail("run_paper.py is not implemented")
        source = runner.read_text()
        self.assertNotIn("fit_dual_router(", source)
        self.assertNotIn("fit_value_router(", source)
        self.assertIn("checkpoint_sha256", source)


if __name__ == "__main__":
    unittest.main()
