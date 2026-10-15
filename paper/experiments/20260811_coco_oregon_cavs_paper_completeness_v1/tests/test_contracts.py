import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from completeness_core import (  # noqa: E402
    action_mask,
    bootstrap_job_count,
    exact_oq_objective,
    event_groups,
    validate_isolated_sources,
)


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ROOT / "configs/protocol.json").read_text())

    def test_job_counts_and_units(self):
        self.assertEqual(bootstrap_job_count("coco", self.protocol), 12)
        self.assertEqual(bootstrap_job_count("oregon", self.protocol), 24)

    def test_masks_are_closed(self):
        self.assertEqual(action_mask("Edge-Original", self.protocol), {"edge", "original"})
        self.assertEqual(action_mask("Edge-Q90", self.protocol), {"edge", "q90"})
        with self.assertRaises(KeyError):
            action_mask("ROI", self.protocol)

    def test_event_groups_partition(self):
        samples = [
            {"image_id": 1, "event_id": "a"},
            {"image_id": 2, "event_id": "a"},
            {"image_id": 3, "event_id": "b"},
        ]
        groups = event_groups(samples)
        self.assertEqual(sorted(sum(groups, [])), [1, 2, 3])
        self.assertEqual(len(groups), 2)

    def test_exact_solver_not_below_heuristic(self):
        values = {1: {"original": 0.5, "q90": 0.4}, 2: {"original": 0.2, "q90": 0.8}}
        costs = {1: {"original": 5, "q90": 3}, 2: {"original": 5, "q90": 3}}
        result = exact_oq_objective(values, costs, capacity=6)
        self.assertGreaterEqual(result["objective"], 0.8)
        self.assertLessEqual(result["used_bytes"], 6)

    def test_current_run_is_forbidden(self):
        with self.assertRaises(ValueError):
            validate_isolated_sources(["/tmp/20260811_coco_currentoqr12_profile_scale_extension_v1/x"], self.protocol)


if __name__ == "__main__":
    unittest.main()
