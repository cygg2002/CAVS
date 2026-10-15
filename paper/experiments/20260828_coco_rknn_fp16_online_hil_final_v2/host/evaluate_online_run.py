#!/usr/bin/env python3
"""Validate a completed online run and evaluate its final boxes on COCO."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from sequence_manifest import file_sha256, load_manifest


COCO80_TO_COCO91 = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
    84, 85, 86, 87, 88, 89, 90,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"blank line in {path} at line {line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path} at line {line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"frame at line {line_number} must be a JSON object")
                rows.append(row)
    except OSError as exc:
        raise ValueError(f"cannot read frames log {path}: {exc}") from exc
    return rows


def load_committed_frames(
    frames_path: str | Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("validated sequence manifest must contain records")
    rows = _load_jsonl(Path(frames_path))

    seen_ids: set[int] = set()
    seen_positions: set[int] = set()
    for line_number, row in enumerate(rows, start=1):
        image_id = row.get("image_id")
        position = row.get("position")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError(f"invalid image_id at frames line {line_number}")
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError(f"invalid position at frames line {line_number}")
        if image_id in seen_ids:
            raise ValueError(f"duplicate image_id {image_id} in frames.jsonl")
        if position in seen_positions:
            raise ValueError(f"duplicate position {position} in frames.jsonl")
        seen_ids.add(image_id)
        seen_positions.add(position)

    if len(rows) < len(records):
        raise ValueError(
            f"incomplete frames.jsonl: expected {len(records)} committed frames, found {len(rows)}"
        )
    if len(rows) > len(records):
        raise ValueError(
            f"extra frames in frames.jsonl: expected {len(records)}, found {len(rows)}"
        )

    for position, (row, expected) in enumerate(zip(rows, records, strict=True)):
        if row.get("status") != "committed":
            raise ValueError(f"frame at position {position} is not committed")
        if row["position"] != position:
            raise ValueError(
                f"frame order mismatch: expected position {position}, found {row['position']}"
            )
        if row["image_id"] != expected["image_id"]:
            raise ValueError(
                f"frame order mismatch at position {position}: expected image_id "
                f"{expected['image_id']}, found {row['image_id']}"
            )
        if row.get("file_name") != expected["file_name"]:
            raise ValueError(
                f"file_name mismatch at position {position}: expected "
                f"{expected['file_name']!r}, found {row.get('file_name')!r}"
            )
        if not isinstance(row.get("resolved_boxes"), list):
            raise ValueError(f"frame at position {position} has no resolved_boxes list")
    return rows


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def frames_to_coco_predictions(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for frame in frames:
        image_id = int(frame["image_id"])
        for box_index, box in enumerate(frame["resolved_boxes"]):
            if not isinstance(box, Mapping):
                raise ValueError(f"image_id {image_id} box {box_index} must be an object")
            xyxy = box.get("xyxy")
            if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
                raise ValueError(f"image_id {image_id} box {box_index} must have four xyxy values")
            x1, y1, x2, y2 = (
                _finite_number(value, f"image_id {image_id} box {box_index} xyxy")
                for value in xyxy
            )
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"image_id {image_id} box {box_index} has non-positive area")
            class_id = box.get("cls")
            if isinstance(class_id, bool) or not isinstance(class_id, int) or not 0 <= class_id < 80:
                raise ValueError(f"image_id {image_id} box {box_index} has invalid cls")
            confidence = _finite_number(
                box.get("conf", box.get("score")),
                f"image_id {image_id} box {box_index} confidence",
            )
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": COCO80_TO_COCO91[class_id],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": confidence,
                }
            )
    return predictions


def load_pycocotools() -> tuple[type, type]:
    try:
        coco_module = importlib.import_module("pycocotools.coco")
        eval_module = importlib.import_module("pycocotools.cocoeval")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "pycocotools is required for post-run COCO evaluation; install it on the "
            "host (for example: python -m pip install pycocotools)"
        ) from exc
    return coco_module.COCO, eval_module.COCOeval


def evaluate_predictions(
    annotation_path: str | Path,
    predictions_path: str | Path,
    image_ids: Sequence[int],
    log_path: str | Path,
    *,
    coco_class: type | None = None,
    coco_eval_class: type | None = None,
) -> dict[str, float]:
    if (coco_class is None) != (coco_eval_class is None):
        raise ValueError("coco_class and coco_eval_class must be provided together")
    if coco_class is None:
        coco_class, coco_eval_class = load_pycocotools()
    assert coco_eval_class is not None

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        truth = coco_class(str(annotation_path))
        detected = truth.loadRes(str(predictions_path))
        evaluator = coco_eval_class(truth, detected, "bbox")
        evaluator.params.imgIds = [int(image_id) for image_id in image_ids]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    if len(evaluator.stats) < 3:
        raise RuntimeError("pycocotools returned fewer than three COCO metrics")
    return {
        "AP50_95": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
    }


def _atomic_json(path: Path, value: Any, *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if pretty:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    else:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def evaluate_run(
    *,
    annotation_path: str | Path,
    manifest_path: str | Path,
    frames_path: str | Path,
    predictions_path: str | Path,
    metrics_path: str | Path,
    log_path: str | Path,
) -> dict[str, Any]:
    annotation_path = Path(annotation_path)
    manifest_path = Path(manifest_path)
    frames_path = Path(frames_path)
    predictions_path = Path(predictions_path)
    metrics_path = Path(metrics_path)

    manifest = load_manifest(manifest_path)
    annotation_sha256 = file_sha256(annotation_path)
    if annotation_sha256 != manifest["source_annotations_sha256"]:
        raise ValueError(
            "COCO annotation hash does not match the sequence manifest; use the exact "
            "trusted annotation file that generated the manifest"
        )
    frames = load_committed_frames(frames_path, manifest)
    predictions = frames_to_coco_predictions(frames)
    coco_class, coco_eval_class = load_pycocotools()
    _atomic_json(predictions_path, predictions, pretty=False)
    metrics = evaluate_predictions(
        annotation_path,
        predictions_path,
        [record["image_id"] for record in manifest["records"]],
        log_path,
        coco_class=coco_class,
        coco_eval_class=coco_eval_class,
    )
    report = {
        "schema": "coco_online_run_evaluation_v1",
        "complete": True,
        "images": len(frames),
        "detections": len(predictions),
        "sequence_sha256": manifest["sequence_sha256"],
        "source_annotations_sha256": annotation_sha256,
        "manifest_sha256": file_sha256(manifest_path),
        "frames_sha256": file_sha256(frames_path),
        "predictions_sha256": file_sha256(predictions_path),
        "predictions_path": str(predictions_path.resolve()),
        "metrics": metrics,
    }
    _atomic_json(metrics_path, report, pretty=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--log-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stem = args.frames.with_suffix("")
    predictions_path = args.predictions_output or stem.with_name(f"{stem.name}.predictions.json")
    metrics_path = args.metrics_output or stem.with_name(f"{stem.name}.metrics.json")
    log_path = args.log_output or stem.with_name(f"{stem.name}.coco_eval.log")
    try:
        report = evaluate_run(
            annotation_path=args.annotations,
            manifest_path=args.manifest,
            frames_path=args.frames,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
            log_path=log_path,
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    metrics = report["metrics"]
    print(
        f"AP50-95={metrics['AP50_95']:.6f} AP50={metrics['AP50']:.6f} "
        f"AP75={metrics['AP75']:.6f} images={report['images']} "
        f"detections={report['detections']}"
    )


if __name__ == "__main__":
    main()
