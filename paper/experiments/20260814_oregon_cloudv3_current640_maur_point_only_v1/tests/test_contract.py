import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("point_only", ROOT / "scripts/run_point_only.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PointOnlyContractTest(unittest.TestCase):
    def test_protocol_is_narrow_and_frozen(self):
        protocol = MODULE.PROTOCOL
        self.assertEqual(protocol["dataset"], "oregon")
        self.assertEqual(protocol["role"], "final")
        self.assertEqual(protocol["method"], "CAVS-OQR-MAUR")
        self.assertEqual(protocol["policy"], "Joint-OQR")
        self.assertEqual(protocol["fusion"], "MAUR")
        self.assertEqual(protocol["mode"], "fixed_byte")
        self.assertEqual(protocol["budgets"], [0.1, 0.25, 0.4])
        self.assertEqual(protocol["expected_rows"], 3)
        self.assertEqual(protocol["bootstrap_replicates"], 0)
        self.assertFalse(protocol["training_performed"])
        self.assertFalse(protocol["final_used_for_training_or_selection"])

    def test_output_is_isolated_from_source(self):
        self.assertNotEqual(MODULE.OUTPUT.resolve(), MODULE.SOURCE_ROOT.resolve())
        self.assertNotIn(MODULE.SOURCE_ROOT.resolve().as_posix(), MODULE.OUTPUT.resolve().as_posix())

    def test_expected_score_paths_cover_three_seeds(self):
        paths = MODULE.expected_score_paths()
        self.assertEqual(len(paths), 3)
        self.assertEqual(len(set(paths)), 3)
        self.assertTrue(all("CAVS-OQR-MAUR" in str(path) for path in paths))


if __name__ == "__main__":
    unittest.main()

