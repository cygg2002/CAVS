import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "voc_point_under_test", ROOT / "scripts/evaluate_point.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PointLogicTest(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {
                "image_id": 1,
                "payload_bytes": {"original": 10, "q90": 5},
                "available": {"original": True, "q90": True},
            },
            {
                "image_id": 2,
                "payload_bytes": {"original": 10, "q90": 5},
                "available": {"original": True, "q90": True},
            },
        ]

    def test_fixed_byte_joint_selection_obeys_floor_capacity(self):
        scores = {1: {"original": 2.0, "q90": 1.5}, 2: {"original": 1.0, "q90": 0.9}}
        selection = MODULE.joint_selection(
            self.samples, scores, mode="fixed_byte", budget=0.25
        )
        spent = sum(
            self.samples[int(image_id) - 1]["payload_bytes"][action]
            for image_id, action in selection.items()
        )
        self.assertLessEqual(spent, 5)

    def test_fixed_call_counts_cloud_images_not_actions(self):
        scores = {1: {"original": 2.0, "q90": 3.0}, 2: {"original": 1.0, "q90": 0.5}}
        selection = MODULE.joint_selection(
            self.samples, scores, mode="fixed_call", budget=0.5
        )
        self.assertEqual(len(selection), 1)
        self.assertEqual(selection, {1: "q90"})

    def test_completed_auxiliary_policies_skip_fit_role_during_point_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory)
            (runs / "policies/shared_action_oq").mkdir(parents=True)
            (runs / "policies/shared_action_oq/completion.json").write_text("{}")
            (runs / "policies/shared_action_oq/artifact.pkl").write_bytes(b"ready")
            (runs / "policies/dcsb").mkdir(parents=True)
            (runs / "policies/dcsb/model.json").write_text("{}")
            with mock.patch.object(MODULE, "RUNS", runs):
                self.assertEqual(
                    MODULE._point_roles_to_load(), ("calibration", "final")
                )

    def test_point_worker_limit_can_be_overridden_without_changing_protocol(self):
        with mock.patch.dict("os.environ", {"VOC_POINT_WORKERS": "3"}, clear=False):
            self.assertEqual(MODULE._point_worker_count(24, 100), 3)

    def test_completed_point_job_is_loaded_without_evaluation_state(self):
        task = {
            "role": "final",
            "method": "CAVS-O-Core",
            "policy": "Joint-O",
            "seed": "ensemble",
            "mode": "fixed_byte",
            "budget": 0.25,
            "selection_index": 7,
        }
        row = {
            "role": "final",
            "method": "CAVS-O-Core",
            "policy": "Joint-O",
            "seed": "ensemble",
            "mode": "fixed_byte",
            "budget": 0.25,
            "official_ap50_95": 0.5,
        }
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(MODULE, "RUNS", Path(directory)):
                path = MODULE._point_job_path(task)
                MODULE.DATA.atomic_json(path, row)
                MODULE._EVAL_STATE.clear()
                self.assertEqual(MODULE._evaluate_task(task), row)

    def test_main_final_ensemble_fixed_byte_point_runs_first(self):
        main = {
            "role": "final",
            "method": "CAVS-OQR-Core",
            "policy": "Joint-OQR",
            "seed": "ensemble",
            "mode": "fixed_byte",
            "budget": 0.1,
        }
        calibration = dict(main, role="calibration")
        single_seed = dict(main, seed="20260825")
        self.assertLess(MODULE._point_task_priority(main), MODULE._point_task_priority(calibration))
        self.assertLess(MODULE._point_task_priority(main), MODULE._point_task_priority(single_seed))


if __name__ == "__main__":
    unittest.main()
