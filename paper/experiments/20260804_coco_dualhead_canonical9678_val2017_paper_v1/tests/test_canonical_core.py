import importlib.util
import json
import pickle
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts/canonical_core.py"


def load_core():
    spec = importlib.util.spec_from_file_location("canonical9678_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CanonicalProtocolTest(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_protocol_locks_canonical_populations(self):
        core = load_core()
        core.validate_protocol(self.protocol)
        self.assertEqual(
            self.protocol["roles"],
            {"fit": 9678, "calibration": 571, "registered_final": 5000},
        )

    def test_protocol_rejects_any_fit_count_change(self):
        core = load_core()
        changed = json.loads(json.dumps(self.protocol))
        changed["roles"]["fit"] = 5000
        with self.assertRaisesRegex(ValueError, "9,678"):
            core.validate_protocol(changed)

    def test_protocol_locks_actions_budgets_and_methods(self):
        core = load_core()
        core.validate_protocol(self.protocol)
        self.assertEqual(self.protocol["actions"], ["edge", "original"])
        self.assertEqual(self.protocol["budgets"], [0.1, 0.25, 0.4])
        self.assertEqual(len(self.protocol["methods"]), 7)

    def test_model_hyperparameters_match_the_frozen_predecessor(self):
        predecessor = json.loads(
            (
                ROOT.parent
                / "20260804_coco_dualhead_fixedbyte_registered_v1/configs/protocol.json"
            ).read_text(encoding="utf-8")
        )
        for key in ("set_model", "baseline_mlp", "detector", "network"):
            self.assertEqual(self.protocol[key], predecessor[key])

    def test_protocol_does_not_reference_temporary_training_checkpoint(self):
        serialized = json.dumps(self.protocol, sort_keys=True)
        self.assertNotIn("coco_budgetset_registered_paper_v1", serialized)
        self.assertNotIn("coco_dualhead_fixedbyte_registered_v1/runs", serialized)

    def test_role_manifest_rejects_wrong_split_or_population(self):
        core = load_core()
        valid = {
            "fit": [{"image_id": value, "identity": f"/train2017/{value}.jpg"} for value in range(9678)],
            "calibration": [{"image_id": 20_000 + value, "identity": f"/train2017/{value}.jpg"} for value in range(571)],
            "registered_final": [{"image_id": 30_000 + value, "identity": f"/val2017/{value}.jpg"} for value in range(5000)],
        }
        core.validate_role_populations(valid)
        valid["fit"][0]["identity"] = "/val2017/0.jpg"
        with self.assertRaisesRegex(ValueError, "train2017"):
            core.validate_role_populations(valid)

    def test_reward_ids_must_equal_fit_and_exclude_other_roles(self):
        core = load_core()
        core.validate_reward_population({1, 2}, {1, 2}, {3}, {4})
        with self.assertRaises(ValueError):
            core.validate_reward_population({1, 3}, {1, 2}, {3}, {4})

    def test_runner_exposes_only_the_frozen_stage_sequence(self):
        runner_path = ROOT / "scripts/run_paper.py"
        text = runner_path.read_text(encoding="utf-8")
        for stage in ("prepare", "rewards", "train", "calibration", "final", "all"):
            self.assertIn(f'"{stage}"', text)
        self.assertNotIn("infer_original", text)

    def test_bootstrap_worker_is_importable_and_picklable(self):
        scripts = str(ROOT / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import canonical_core

        restored = pickle.loads(pickle.dumps(canonical_core.bootstrap_worker))
        self.assertIs(restored, canonical_core.bootstrap_worker)


if __name__ == "__main__":
    unittest.main()
