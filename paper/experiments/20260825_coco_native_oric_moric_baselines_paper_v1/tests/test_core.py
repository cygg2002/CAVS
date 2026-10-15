import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())
SPEC = importlib.util.spec_from_file_location("native_baseline_core", ROOT / "scripts/core.py")
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


class NativeBaselineContractTest(unittest.TestCase):
    def test_original_action_map_is_original_only(self):
        samples = [{"image_id": 7}, {"image_id": 11}]
        self.assertEqual(CORE.original_action_map(samples), {7: "original", 11: "original"})

    def test_rejects_non_original_native_selection(self):
        with self.assertRaises(ValueError):
            CORE.validate_native_selection({7: "original", 11: "q90"})

    def test_point_matrix_count(self):
        self.assertEqual(CORE.expected_point_rows(PROTOCOL), 120)

    def test_bootstrap_jobs_cover_four_methods_at_three_budgets(self):
        jobs = CORE.bootstrap_jobs(PROTOCOL)
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({job["id"] for job in jobs}), 12)
        self.assertEqual({job["right"] for job in jobs}, set(PROTOCOL["native_methods"]))


if __name__ == "__main__":
    unittest.main()
