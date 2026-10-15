import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_core():
    path = ROOT / "scripts/core.py"
    spec = importlib.util.spec_from_file_location("afpd_forced_fullval_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_protocol_discloses_gate_override_and_posthoc_status(self):
        core = load_core()
        core.validate_protocol(self.protocol)
        self.assertFalse(self.protocol["confirmatory"])
        self.assertTrue(self.protocol["calibration_gate_failed"])
        self.assertFalse(self.protocol["final_is_hidden"])
        self.assertTrue(self.protocol["training_allowed"])
        self.assertFalse(self.protocol["tuning_allowed"])
        self.assertTrue(self.protocol["independent_retrain"])
        self.assertFalse(self.protocol["source_checkpoint_reuse"])
        self.assertTrue(self.protocol["full_val_evaluation_unconditional"])

    def test_expected_training_and_result_counts(self):
        core = load_core()
        self.assertEqual(core.expected_oof_action_jobs(self.protocol), 60)
        self.assertEqual(core.expected_oof_occupancy_jobs(self.protocol), 30)
        self.assertEqual(core.expected_fullfit_jobs(self.protocol), 18)
        self.assertEqual(core.expected_result_rows(self.protocol), 36)
        self.assertEqual(core.expected_bootstrap_rows(self.protocol), 15)

    def test_new_checkpoint_paths_are_exact_and_run_scoped(self):
        core = load_core()
        paths = core.checkpoint_relative_paths(self.protocol)
        self.assertEqual(len(paths), 18)
        self.assertEqual(len(set(paths)), 18)
        self.assertTrue(all(path.startswith("runs/calibration/") for path in paths))

    def test_capacity_guard_rejects_overflow(self):
        core = load_core()
        with self.assertRaises(ValueError):
            core.assert_capacity(100.01, 100.0)


if __name__ == "__main__":
    unittest.main()
