import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("voc_torchvision_runtime_core", ROOT / "scripts/runtime_core.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SourceRoleTest(unittest.TestCase):
    def test_role_loader_verifies_hashes_population_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory)
            source = experiment / "old/runs/data/roles/fit"
            source.mkdir(parents=True)
            gt = {
                "categories": [{"id": 1, "name": "person"}],
                "images": [
                    {"id": 1, "identity": "2007:1", "absolute_path": "/tmp/1.jpg"},
                    {"id": 2, "identity": "2007:2", "absolute_path": "/tmp/2.jpg"},
                ],
                "annotations": [{"image_id": 1, "category_id": 1}],
            }
            gt_path = source / "gt.json"
            gt_path.write_text(json.dumps(gt), encoding="utf-8")
            images_path = source / "images.txt"
            images_path.write_text("/tmp/1.jpg\n/tmp/2.jpg\n", encoding="utf-8")
            identity_hash = MODULE.text_sha256("2007:1\n2007:2")
            protocol = {
                "allowed_roles": ["fit", "calibration"],
                "source_experiment": "old",
                "roles": {
                    "fit": {
                        "images": 2,
                        "gt_sha256": MODULE.file_sha256(gt_path),
                        "images_sha256": MODULE.file_sha256(images_path),
                        "identity_sha256": identity_hash,
                    }
                },
            }
            loaded = MODULE.load_source_role(experiment, protocol, "fit", require_images=False)
            self.assertEqual([row["category_ids"] for row in loaded["images"]], [[1], []])

    def test_role_loader_rejects_final_before_touching_files(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = {"allowed_roles": ["fit", "calibration"], "source_experiment": "missing", "roles": {}}
            with self.assertRaisesRegex(ValueError, "forbidden"):
                MODULE.load_source_role(Path(directory), protocol, "final")


class CheckpointTest(unittest.TestCase):
    def test_gpu_gate_requires_both_memory_and_utilization_limits(self):
        contract = {"minimum_free_gpu_mib": 18000, "maximum_utilization_percent": 5}
        self.assertTrue(MODULE.gpu_gate_open(19000, 4, contract))
        self.assertFalse(MODULE.gpu_gate_open(17000, 4, contract))
        self.assertFalse(MODULE.gpu_gate_open(19000, 6, contract))

    def test_jsonl_checkpoint_rejects_duplicate_images(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            row = {"image_id": 1, "action": "edge", "predictions": []}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                MODULE.read_checkpoint(path, action="edge", expected_ids={1, 2})

    def test_jsonl_checkpoint_rejects_cross_protocol_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            row = {
                "image_id": 1,
                "action": "edge",
                "protocol_sha256": "a" * 64,
                "predictions": [],
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "protocol"):
                MODULE.read_checkpoint(
                    path,
                    action="edge",
                    expected_ids={1},
                    protocol_sha256="b" * 64,
                )

    def test_bound_json_rejects_cross_protocol_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "completion.json"
            path.write_text(
                json.dumps({"complete": True, "protocol_sha256": "a" * 64}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "protocol"):
                MODULE.read_protocol_bound_json(path, "b" * 64)

    def test_atomic_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            MODULE.atomic_json(path, {"complete": True, "value": 3})
            self.assertEqual(json.loads(path.read_text()), {"complete": True, "value": 3})


if __name__ == "__main__":
    unittest.main()
