import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "voc_action_cache_under_test", ROOT / "scripts/cache_actions.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ActionCacheTest(unittest.TestCase):
    def test_parse_free_gpu_memory_uses_smallest_visible_value(self):
        self.assertEqual(MODULE.parse_free_gpu_mib("12345\n15321\n"), 12345)

    def test_record_population_rejects_duplicates_missing_and_unknowns(self):
        MODULE.validate_record_population([{"image_id": 1}, {"image_id": 2}], [1, 2])
        with self.assertRaises(RuntimeError):
            MODULE.validate_record_population(
                [{"image_id": 1}, {"image_id": 1}, {"image_id": 2}], [1, 2]
            )
        with self.assertRaises(RuntimeError):
            MODULE.validate_record_population([{"image_id": 1}], [1, 2])
        with self.assertRaises(RuntimeError):
            MODULE.validate_record_population([{"image_id": 3}], [1, 2])


if __name__ == "__main__":
    unittest.main()
