import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("unified_cavs_runner", ROOT / "scripts/run_experiment.py")
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class RunnerContractTest(unittest.TestCase):
    def test_dataset_adapter_rejects_unknown_dataset(self):
        with self.assertRaises(ValueError):
            RUNNER.DatasetAdapter("ena24")

    def test_source_paths_are_isolated_under_repo(self):
        for dataset in ("coco", "oregon"):
            adapter = RUNNER.DatasetAdapter(dataset, load_modules=False)
            self.assertTrue(adapter.cavs_root.is_relative_to(RUNNER.REPO))
            self.assertNotEqual(adapter.cavs_root, RUNNER.ROOT)

    def test_completion_is_not_true_for_missing_file(self):
        self.assertFalse(RUNNER.is_complete(ROOT / "does-not-exist.json"))

    def test_runner_does_not_treat_final_as_training_role(self):
        self.assertEqual(RUNNER.TRAINING_ROLES, ("fit",))
        self.assertNotIn("final", RUNNER.TRAINING_ROLES)

    def test_external_source_directory_is_added_before_import(self):
        adapter = RUNNER.DatasetAdapter("coco", load_modules=False)
        directory = adapter.external_root / "scripts"
        RUNNER.ensure_import_path(directory)
        self.assertEqual(Path(sys.path[0]), directory)

    def test_lightweight_adapter_does_not_import_evaluator(self):
        adapter = RUNNER.DatasetAdapter("coco", load_modules=False)
        self.assertIsNone(adapter.cavs)
        self.assertEqual(
            adapter.cavs_protocol["methods"]["single"],
            "CurrentActionValue-OQR-Current640-Reversible",
        )
        self.assertEqual(
            adapter.common_root("fit").name,
            "fit",
        )

    def test_random_aggregation_keeps_calibration_and_final_separate(self):
        rows = []
        for role, ap in (("calibration", 0.4), ("final", 0.5)):
            rows.append(
                {
                    "role": role,
                    "method": "Random",
                    "policy": "SharedAction-OQR",
                    "mode": "fixed_byte",
                    "budget": 0.1,
                    "seed": 1,
                    "official_ap50_95": ap,
                    "official_ap50": ap,
                    "official_ap75": ap,
                    "kb_per_image": 1.0,
                    "request_bytes": 1024,
                    "capacity_utilization": 1.0,
                    "selected_images": 1,
                    "original_count": 1,
                    "q90_count": 0,
                    "roi_count": 0,
                }
            )
        aggregated = RUNNER.aggregate_random(rows)
        self.assertEqual(len(aggregated), 2)
        self.assertEqual({row["role"] for row in aggregated}, {"calibration", "final"})

    def test_artifact_audit_rejects_incomplete_matrix(self):
        rows = [
            {
                "dataset": "coco",
                "role": "final",
                "method": "CAVS-O",
                "policy": "Joint-O",
                "mode": "fixed_byte",
                "budget": 0.1,
            }
        ]
        with self.assertRaises(RuntimeError):
            RUNNER.audit_point_rows("coco", rows, [])


if __name__ == "__main__":
    unittest.main()
