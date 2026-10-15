from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


HOST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST))

import sequence_manifest


class SequenceManifestTest(unittest.TestCase):
    def test_builds_sorted_images_only_manifest_with_exact_record_fields(self) -> None:
        annotation = {
            "images": [
                {"id": 9, "file_name": "000000000009.jpg", "width": 9, "height": 8, "license": 1},
                {"id": 2, "file_name": "000000000002.jpg", "width": 7, "height": 6, "license": 2},
            ],
            "annotations": [{"id": 99, "image_id": 9, "bbox": [1, 2, 3, 4]}],
            "categories": [{"id": 1, "name": "person"}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            annotation_path = Path(temporary) / "instances.json"
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")

            manifest = sequence_manifest.build_manifest(annotation_path, split="val2017")
            annotation_sha256 = hashlib.sha256(annotation_path.read_bytes()).hexdigest()

        self.assertEqual(manifest["schema"], "coco_images_only_sequence_v1")
        self.assertEqual(manifest["dataset"], "coco")
        self.assertEqual(manifest["split"], "val2017")
        self.assertEqual(manifest["images"], 2)
        self.assertEqual([record["image_id"] for record in manifest["records"]], [2, 9])
        self.assertEqual(
            set(manifest["records"][0]),
            {"image_id", "file_name", "width", "height"},
        )
        self.assertFalse({"annotations", "categories"} & set(manifest))
        self.assertEqual(
            manifest["source_annotations_sha256"],
            annotation_sha256,
        )
        self.assertEqual(
            manifest["sequence_sha256"],
            sequence_manifest.records_sha256(manifest["records"]),
        )

    def test_writes_deterministic_manifest_and_loads_it(self) -> None:
        annotation = {
            "images": [
                {"id": 1, "file_name": "one.jpg", "width": 10, "height": 20},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation_path = root / "instances.json"
            output_path = root / "sequence.json"
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")

            first = sequence_manifest.write_manifest(annotation_path, output_path, split="val2017")
            first_bytes = output_path.read_bytes()
            second = sequence_manifest.write_manifest(annotation_path, output_path, split="val2017")

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, output_path.read_bytes())
            self.assertEqual(sequence_manifest.load_manifest(output_path), first)

    def test_rejects_duplicate_image_ids(self) -> None:
        annotation = {
            "images": [
                {"id": 1, "file_name": "one.jpg", "width": 10, "height": 20},
                {"id": 1, "file_name": "other.jpg", "width": 10, "height": 20},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            annotation_path = Path(temporary) / "instances.json"
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate image_id"):
                sequence_manifest.build_manifest(annotation_path, split="val2017")

    def test_load_rejects_modified_records(self) -> None:
        manifest = {
            "schema": "coco_images_only_sequence_v1",
            "dataset": "coco",
            "split": "val2017",
            "images": 1,
            "source_annotations_sha256": "a" * 64,
            "sequence_sha256": "b" * 64,
            "records": [
                {"image_id": 1, "file_name": "one.jpg", "width": 10, "height": 20},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sequence.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sequence_sha256"):
                sequence_manifest.load_manifest(path)


if __name__ == "__main__":
    unittest.main()
