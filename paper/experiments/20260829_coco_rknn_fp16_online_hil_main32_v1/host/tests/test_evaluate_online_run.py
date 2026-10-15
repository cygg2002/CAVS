from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


HOST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST))

import evaluate_online_run
import sequence_manifest


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class EvaluateOnlineRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {"image_id": 2, "file_name": "two.jpg", "width": 20, "height": 10},
            {"image_id": 9, "file_name": "nine.jpg", "width": 90, "height": 80},
        ]
        self.manifest = {
            "schema": "coco_images_only_sequence_v1",
            "dataset": "coco",
            "split": "val2017",
            "images": len(self.records),
            "source_annotations_sha256": "a" * 64,
            "sequence_sha256": sequence_manifest.records_sha256(self.records),
            "records": self.records,
        }
        self.frames = [
            {
                "status": "committed",
                "position": 0,
                "image_id": 2,
                "file_name": "two.jpg",
                "resolved_boxes": [
                    {"xyxy": [1.0, 2.0, 11.0, 8.0], "conf": 0.75, "cls": 11},
                ],
            },
            {
                "status": "committed",
                "position": 1,
                "image_id": 9,
                "file_name": "nine.jpg",
                "resolved_boxes": [],
            },
        ]

    def test_loads_complete_committed_frames_and_converts_coco_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frames_path = Path(temporary) / "frames.jsonl"
            write_jsonl(frames_path, self.frames)

            frames = evaluate_online_run.load_committed_frames(frames_path, self.manifest)
            predictions = evaluate_online_run.frames_to_coco_predictions(frames)

        self.assertEqual([frame["image_id"] for frame in frames], [2, 9])
        self.assertEqual(
            predictions,
            [
                {
                    "image_id": 2,
                    "category_id": 13,
                    "bbox": [1.0, 2.0, 10.0, 6.0],
                    "score": 0.75,
                }
            ],
        )

    def test_rejects_partial_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frames_path = Path(temporary) / "frames.jsonl"
            write_jsonl(frames_path, self.frames[:1])

            with self.assertRaisesRegex(ValueError, "incomplete frames.jsonl"):
                evaluate_online_run.load_committed_frames(frames_path, self.manifest)

    def test_rejects_duplicate_or_out_of_order_frame(self) -> None:
        duplicate = [self.frames[0], {**self.frames[1], "image_id": 2}]
        with tempfile.TemporaryDirectory() as temporary:
            frames_path = Path(temporary) / "frames.jsonl"
            write_jsonl(frames_path, duplicate)

            with self.assertRaisesRegex(ValueError, "duplicate image_id"):
                evaluate_online_run.load_committed_frames(frames_path, self.manifest)

    def test_rejects_unique_frames_in_the_wrong_order(self) -> None:
        reversed_frames = [
            {**self.frames[1], "position": 0},
            {**self.frames[0], "position": 1},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            frames_path = Path(temporary) / "frames.jsonl"
            write_jsonl(frames_path, reversed_frames)

            with self.assertRaisesRegex(ValueError, "frame order mismatch"):
                evaluate_online_run.load_committed_frames(frames_path, self.manifest)

    def test_rejects_uncommitted_frame(self) -> None:
        frames = [{**self.frames[0], "status": "running"}, self.frames[1]]
        with tempfile.TemporaryDirectory() as temporary:
            frames_path = Path(temporary) / "frames.jsonl"
            write_jsonl(frames_path, frames)

            with self.assertRaisesRegex(ValueError, "not committed"):
                evaluate_online_run.load_committed_frames(frames_path, self.manifest)

    def test_pycocotools_dependency_error_is_actionable(self) -> None:
        with mock.patch("importlib.import_module", side_effect=ModuleNotFoundError("missing")):
            with self.assertRaisesRegex(RuntimeError, "pycocotools is required"):
                evaluate_online_run.load_pycocotools()

    def test_extracts_requested_coco_metrics_with_injected_api(self) -> None:
        predictions = [{"image_id": 2, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5}]

        class FakeCOCO:
            def __init__(self, path: str) -> None:
                self.path = path

            def loadRes(self, path: str):
                self.predictions = json.loads(Path(path).read_text(encoding="utf-8"))
                return object()

        class FakeCOCOeval:
            def __init__(self, truth, detected, kind: str) -> None:
                self.params = types.SimpleNamespace(imgIds=[])
                self.stats = [0.31, 0.52, 0.33]

            def evaluate(self) -> None:
                pass

            def accumulate(self) -> None:
                pass

            def summarize(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "instances.json"
            predictions_path = root / "predictions.json"
            log_path = root / "coco_eval.log"
            annotations.write_text("{}", encoding="utf-8")
            predictions_path.write_text(json.dumps(predictions), encoding="utf-8")

            metrics = evaluate_online_run.evaluate_predictions(
                annotations,
                predictions_path,
                [2, 9],
                log_path,
                coco_class=FakeCOCO,
                coco_eval_class=FakeCOCOeval,
            )

        self.assertEqual(metrics, {"AP50_95": 0.31, "AP50": 0.52, "AP75": 0.33})

    def test_complete_evaluation_writes_predictions_and_auditable_metrics(self) -> None:
        class FakeCOCO:
            def __init__(self, path: str) -> None:
                pass

            def loadRes(self, path: str):
                return object()

        class FakeCOCOeval:
            def __init__(self, truth, detected, kind: str) -> None:
                self.params = types.SimpleNamespace(imgIds=[])
                self.stats = [0.41, 0.62, 0.43]

            def evaluate(self) -> None:
                pass

            def accumulate(self) -> None:
                pass

            def summarize(self) -> None:
                pass

        annotation = {
            "images": [
                {"id": 9, "file_name": "nine.jpg", "width": 90, "height": 80},
                {"id": 2, "file_name": "two.jpg", "width": 20, "height": 10},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations_path = root / "instances.json"
            manifest_path = root / "sequence.json"
            frames_path = root / "frames.jsonl"
            predictions_path = root / "predictions.json"
            metrics_path = root / "metrics.json"
            log_path = root / "eval.log"
            annotations_path.write_text(json.dumps(annotation), encoding="utf-8")
            sequence_manifest.write_manifest(
                annotations_path,
                manifest_path,
                split="val2017",
            )
            write_jsonl(frames_path, self.frames)

            with mock.patch.object(
                evaluate_online_run,
                "load_pycocotools",
                return_value=(FakeCOCO, FakeCOCOeval),
            ):
                report = evaluate_online_run.evaluate_run(
                    annotation_path=annotations_path,
                    manifest_path=manifest_path,
                    frames_path=frames_path,
                    predictions_path=predictions_path,
                    metrics_path=metrics_path,
                    log_path=log_path,
                )

            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
            stored_report = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertEqual(len(predictions), 1)
        self.assertEqual(report["metrics"], {"AP50_95": 0.41, "AP50": 0.62, "AP75": 0.43})
        self.assertEqual(stored_report, report)
        self.assertTrue(report["complete"])
        self.assertEqual(report["images"], 2)

    def test_rejects_annotation_file_not_bound_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations_path = root / "instances.json"
            manifest_path = root / "sequence.json"
            frames_path = root / "frames.jsonl"
            annotations_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 2, "file_name": "two.jpg", "width": 20, "height": 10},
                            {"id": 9, "file_name": "nine.jpg", "width": 90, "height": 80},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            sequence_manifest.write_manifest(annotations_path, manifest_path, split="val2017")
            annotations_path.write_text('{"images": []}', encoding="utf-8")
            write_jsonl(frames_path, self.frames)

            with self.assertRaisesRegex(ValueError, "annotation hash does not match"):
                evaluate_online_run.evaluate_run(
                    annotation_path=annotations_path,
                    manifest_path=manifest_path,
                    frames_path=frames_path,
                    predictions_path=root / "predictions.json",
                    metrics_path=root / "metrics.json",
                    log_path=root / "eval.log",
                )


if __name__ == "__main__":
    unittest.main()
