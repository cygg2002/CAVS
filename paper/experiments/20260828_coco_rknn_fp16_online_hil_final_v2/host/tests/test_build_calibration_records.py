from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[2]
HOST = ROOT / "host"
BOARD_SCRIPTS = ROOT / "board_bundle" / "scripts"
sys.path.insert(0, str(HOST))
sys.path.insert(0, str(BOARD_SCRIPTS))

from build_calibration_records import build_records_bundle
from online_methods import MethodRuntime


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BuildCalibrationRecordsTest(unittest.TestCase):
    def test_builtin_method_uses_only_current_frame_and_predicted_objective(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "000000000007.jpg"
            image.write_bytes(b"native-jpeg")
            candidates = root / "edge_candidates.json"
            report = root / "report.json"
            protocol = root / "protocol.json"
            candidates.write_text(json.dumps([{
                "image_id": 7,
                "file_name": image.name,
                "width": 20,
                "height": 10,
                "edge_boxes": [],
                "roi_xyxy": None,
                "native_jpeg_bytes": len(image.read_bytes()),
            }]), encoding="utf-8")
            report.write_text(json.dumps({
                "complete": True,
                "backend": "fp16",
                "role": "calibration",
                "images": 1,
                "model_sha256": "a" * 64,
            }), encoding="utf-8")
            protocol.write_text(json.dumps({
                "protocol_id": "online-test-v1",
                "rknn_sha256": "a" * 64,
            }), encoding="utf-8")

            bundle = build_records_bundle(
                candidates_path=candidates,
                candidate_report_path=report,
                protocol_path=protocol,
                images_dir=root,
                method_runtime=MethodRuntime("confidence", ("edge", "original")),
                policy_source=BOARD_SCRIPTS / "online_methods.py",
            )

        self.assertEqual(bundle["schema"], "cavs_online_calibration_records_v1")
        self.assertEqual(bundle["source_split"], "calibration")
        self.assertEqual(bundle["method"], "confidence")
        self.assertEqual(bundle["action_ids"], ["edge", "original"])
        self.assertEqual(bundle["records"][0]["payload_bytes"], {
            "edge": 0,
            "original": len(b"native-jpeg"),
        })
        self.assertEqual(
            bundle["records"][0]["objective_values"],
            bundle["records"][0]["scores"],
        )
        self.assertEqual(bundle["calibration_objective"]["name"], "predicted_action_value")
        self.assertNotIn("annotations", json.dumps(bundle).lower())

    def test_rejects_non_calibration_report_or_native_byte_drift(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "one.jpg"
            image.write_bytes(b"jpeg")
            candidates = root / "candidates.json"
            report = root / "report.json"
            protocol = root / "protocol.json"
            candidates.write_text(json.dumps([{
                "image_id": 1,
                "file_name": image.name,
                "width": 1,
                "height": 1,
                "edge_boxes": [],
                "roi_xyxy": None,
                "native_jpeg_bytes": 999,
            }]), encoding="utf-8")
            protocol.write_text(json.dumps({
                "protocol_id": "online-test-v1", "rknn_sha256": "a" * 64,
            }), encoding="utf-8")
            report.write_text(json.dumps({
                "complete": True, "backend": "fp16", "role": "final",
                "images": 1, "model_sha256": "a" * 64,
            }), encoding="utf-8")
            kwargs = dict(
                candidates_path=candidates,
                candidate_report_path=report,
                protocol_path=protocol,
                images_dir=root,
                method_runtime=MethodRuntime("random", ("edge", "original"), random_seed=3),
                policy_source=BOARD_SCRIPTS / "online_methods.py",
            )
            with self.assertRaisesRegex(ValueError, "calibration"):
                build_records_bundle(**kwargs)

            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["role"] = "calibration"
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "native JPEG byte drift"):
                build_records_bundle(**kwargs)

    def test_binds_all_source_files_by_sha256(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "one.jpg"
            image.write_bytes(b"jpeg")
            candidates = root / "candidates.json"
            report = root / "report.json"
            protocol = root / "protocol.json"
            policy = root / "policy.json"
            candidates.write_text(json.dumps([{
                "image_id": 1, "file_name": image.name, "width": 1, "height": 1,
                "edge_boxes": [], "roi_xyxy": None, "native_jpeg_bytes": 4,
            }]), encoding="utf-8")
            report.write_text(json.dumps({
                "complete": True, "backend": "fp16", "role": "calibration",
                "images": 1, "model_sha256": "a" * 64,
            }), encoding="utf-8")
            protocol.write_text(json.dumps({
                "protocol_id": "online-test-v1", "rknn_sha256": "a" * 64,
            }), encoding="utf-8")
            policy.write_text("{}", encoding="utf-8")

            bundle = build_records_bundle(
                candidates_path=candidates,
                candidate_report_path=report,
                protocol_path=protocol,
                images_dir=root,
                method_runtime=MethodRuntime("confidence", ("edge", "original")),
                policy_source=policy,
            )

            self.assertEqual(bundle["source_hashes"], {
                "candidate_report": sha256(report),
                "candidates": sha256(candidates),
                "policy": sha256(policy),
            })


if __name__ == "__main__":
    unittest.main()
