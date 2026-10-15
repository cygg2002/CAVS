from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("oregon_oqr_core", ROOT / "scripts/core.py")
CORE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CORE)
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())


class CoreContractTest(unittest.TestCase):
    def test_protocol_freezes_coco_matched_single_roi(self) -> None:
        CORE.validate_protocol(PROTOCOL)
        self.assertEqual(
            PROTOCOL["action_space"],
            ["edge", "original", "q90", "current@640"],
        )

    def test_training_matrix_has_eighteen_jobs(self) -> None:
        jobs = CORE.training_jobs(PROTOCOL)
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len({job["id"] for job in jobs}), 18)

    def test_expected_point_and_bootstrap_counts(self) -> None:
        self.assertEqual(CORE.expected_point_rows(PROTOCOL), 12)
        self.assertEqual(CORE.bootstrap_jobs(PROTOCOL), 12)

    def test_validate_selection_rejects_non_contract_action(self) -> None:
        with self.assertRaises(ValueError):
            CORE.validate_action_selection({1: "moderate@320"}, PROTOCOL)


if __name__ == "__main__":
    unittest.main()
