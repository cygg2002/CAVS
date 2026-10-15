from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rknn_candidate_io", ROOT / "scripts" / "candidate_io.py")
assert SPEC and SPEC.loader
CANDIDATE_IO = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CANDIDATE_IO
SPEC.loader.exec_module(CANDIDATE_IO)


def input_manifest(role: str = "fit") -> dict:
    return {
        "schema": "rknn_candidate_input_v1",
        "role": role,
        "images": 2,
        "source_manifest_sha256": "a" * 64,
        "records": [
            {"image_id": 11, "file_name": "000000000011.jpg", "width": 640, "height": 480},
            {"image_id": 12, "file_name": "000000000012.jpg", "width": 800, "height": 600},
        ],
    }


def candidate_manifest(role: str = "fit") -> dict:
    return {
        "schema": "rknn_candidate_bundle_v1",
        "backend": "fp16",
        "model_sha256": "b" * 64,
        "role": role,
        "images": 2,
        "input_manifest_sha256": "c" * 64,
    }


def candidate_record(image_id: int, *, width: int = 640, height: int = 480) -> dict:
    return {
        "schema": "rknn_candidate_record_v1",
        "backend": "fp16",
        "model_sha256": "b" * 64,
        "role": "fit",
        "image_id": image_id,
        "file_name": f"{image_id:012d}.jpg",
        "shape": [height, width],
        "edge_boxes": [{"xyxy": [1.0, 2.0, 3.0, 4.0], "conf": 0.9, "cls": 1}],
    }


class CandidateIOTests(unittest.TestCase):
    def test_input_manifest_rejects_duplicate_image_ids(self) -> None:
        payload = input_manifest()
        payload["records"][1]["image_id"] = 11

        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            CANDIDATE_IO.validate_input_manifest(payload, expected_role="fit", expected_images=2)

    def test_bundle_rejects_cross_backend_or_model_hash(self) -> None:
        payload = candidate_manifest()

        with self.assertRaisesRegex(RuntimeError, "backend"):
            CANDIDATE_IO.validate_candidate_bundle(
                payload,
                backend="int8-heads",
                model_sha256="b" * 64,
                role="fit",
                expected_images=2,
            )
        with self.assertRaisesRegex(RuntimeError, "model hash"):
            CANDIDATE_IO.validate_candidate_bundle(
                payload,
                backend="fp16",
                model_sha256="d" * 64,
                role="fit",
                expected_images=2,
            )

    def test_import_rejects_duplicate_records_and_ignores_old_scores(self) -> None:
        expected = input_manifest()
        first = candidate_record(11)
        second = candidate_record(12, width=800, height=600)
        second["candidate"] = {"scores": {"original": 999.0}}

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidates.jsonl"
            path.write_text("\n".join(json.dumps(value) for value in (first, second)) + "\n", encoding="utf-8")
            records = CANDIDATE_IO.load_candidate_records(
                path,
                input_manifest=expected,
                backend="fp16",
                model_sha256="b" * 64,
                role="fit",
            )
            self.assertEqual(set(records), {11, 12})
            self.assertNotIn("candidate", records[12])

            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(first) + "\n")
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                CANDIDATE_IO.load_candidate_records(
                    path,
                    input_manifest=expected,
                    backend="fp16",
                    model_sha256="b" * 64,
                    role="fit",
                )

    def test_import_rejects_roi_not_derived_from_matching_edge_boxes(self) -> None:
        expected = input_manifest()
        record = candidate_record(11)
        record["roi_xyxy"] = [0, 0, 10, 10]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidates.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "ROI geometry"):
                CANDIDATE_IO.load_candidate_records(
                    path,
                    input_manifest={**expected, "images": 1, "records": expected["records"][:1]},
                    backend="fp16",
                    model_sha256="b" * 64,
                    role="fit",
                )

    def test_import_preserves_clipped_zero_area_board_boxes(self) -> None:
        expected = input_manifest()
        record = candidate_record(11)
        record["edge_boxes"] = [
            {"xyxy": [1.0, 0.0, 639.0, 0.0], "conf": 0.01, "cls": 47}
        ]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidates.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            records = CANDIDATE_IO.load_candidate_records(
                path,
                input_manifest={**expected, "images": 1, "records": expected["records"][:1]},
                backend="fp16",
                model_sha256="b" * 64,
                role="fit",
            )

        self.assertEqual(records[11]["edge_boxes"], record["edge_boxes"])


if __name__ == "__main__":
    unittest.main()
