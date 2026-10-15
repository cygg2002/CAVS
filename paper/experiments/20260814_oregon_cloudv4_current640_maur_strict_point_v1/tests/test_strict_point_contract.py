import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROTOCOL_CORE = load("cloudv4_protocol", ROOT / "scripts/protocol_core.py")
CACHE_CORE = load("cloudv4_cache", ROOT / "scripts/cache_core.py")
TRAIN_CORE = load("cloudv4_train", ROOT / "scripts/train_core.py")


class StrictPointContractTest(unittest.TestCase):
    def test_protocol_is_cloudv4_current640_maur_only(self):
        report = PROTOCOL_CORE.validate_protocol(PROTOCOL)
        self.assertEqual(report, {"roi_actions": 1, "learned_methods": 1})
        self.assertEqual(PROTOCOL["cloud_model"]["epoch"], 29)
        self.assertEqual(PROTOCOL["roi_actions"], ["current@640"])
        self.assertEqual(PROTOCOL["fusion_modes"], ["MAUR"])

    def test_cache_plan_is_three_roles_by_four_actions(self):
        jobs = CACHE_CORE.planned_cache_jobs(PROTOCOL)
        self.assertEqual(len(jobs), 12)
        for role in PROTOCOL["roles"]:
            role_jobs = [job for job in jobs if job["role"] == role]
            self.assertEqual(len(role_jobs), 4)
            inference = {job["action_id"] for job in role_jobs if job["kind"] == "inference"}
            self.assertEqual(inference, {"original", "q90", "current@640"})

    def test_training_plan_is_three_seeds_five_oof_plus_full(self):
        jobs = TRAIN_CORE.training_jobs(PROTOCOL)
        self.assertEqual(len(jobs), 18)
        self.assertEqual({job["method"] for job in jobs}, {"CurrentActionValue-OQR12"})
        self.assertEqual({job["fold"] for job in jobs}, {0, 1, 2, 3, 4, "full"})

    def test_point_runner_exists_and_pipeline_skips_bootstrap(self):
        self.assertTrue((ROOT / "scripts/run_point_only.py").is_file())
        pipeline = (ROOT / "scripts/run_pipeline.py").read_text()
        self.assertIn('("point", "run_point_only.py"', pipeline)
        self.assertNotIn('("bootstrap",', pipeline)


if __name__ == "__main__":
    unittest.main()

