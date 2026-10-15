#!/usr/bin/env python3
"""Collect raw single-image RKNN Edge candidates for one frozen COCO role.

Copy this file into the board bundle's ``scripts`` directory.  It intentionally
does not load a CAVS policy or export action scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePath
import sys
import time
from typing import Any, Mapping

import cv2
import numpy as np

from board_client import atomic_json, append_jsonl, imports_from_sibling, load_config, sha256
from cavs_numpy import current_roi_xyxy
from protocol import compact_boxes


ROOT = Path(__file__).resolve().parents[1]
ROLE_COUNTS = {"fit": 9678, "calibration": 571, "final": 5000}
INPUT_SCHEMA = "rknn_candidate_input_v1"
RECORD_SCHEMA = "rknn_candidate_record_v1"
BUNDLE_SCHEMA = "rknn_candidate_bundle_v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_input(path: Path, role: str) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != INPUT_SCHEMA or payload.get("role") != role:
        raise RuntimeError("candidate input schema or role mismatch")
    if int(payload.get("images", -1)) != ROLE_COUNTS[role]:
        raise RuntimeError(f"candidate input count is not frozen for {role}")
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != ROLE_COUNTS[role]:
        raise RuntimeError("candidate input records are incomplete")
    records: dict[int, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise RuntimeError("candidate input record is invalid")
        image_id = int(raw["image_id"])
        name = str(raw["file_name"])
        if image_id in records:
            raise RuntimeError("candidate input contains duplicate image IDs")
        if not name or PurePath(name).name != name:
            raise RuntimeError("candidate input filename must be a plain filename")
        width, height = int(raw["width"]), int(raw["height"])
        if width <= 0 or height <= 0:
            raise RuntimeError("candidate input geometry is invalid")
        records[image_id] = {"image_id": image_id, "file_name": name, "width": width, "height": height}
    return payload, records


def read_checkpoint(path: Path, *, model: str, model_sha256: str, role: str, wanted: set[int]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid candidate JSON at line {number}") from error
            image_id = int(row.get("image_id", -1))
            if image_id in rows:
                raise RuntimeError(f"duplicate candidate image ID: {image_id}")
            if image_id not in wanted:
                raise RuntimeError("candidate checkpoint belongs to another role")
            if row.get("schema") != RECORD_SCHEMA or row.get("backend") != model or row.get("role") != role:
                raise RuntimeError("candidate checkpoint provenance mismatch")
            if str(row.get("model_sha256")) != model_sha256:
                raise RuntimeError("candidate checkpoint model hash mismatch")
            rows[image_id] = row
    return rows


def status(output: Path, **values: Any) -> None:
    atomic_json(output / "status.json", {"updated_at": time.time(), **values})


def collect(args: argparse.Namespace) -> None:
    config = load_config()
    role, model = str(args.role), str(args.model)
    input_path = Path(args.manifest).resolve()
    input_payload, expected = read_input(input_path, role)
    images_root = Path(args.images_root).resolve()
    missing = [row["file_name"] for row in expected.values() if not (images_root / row["file_name"]).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} role images; first: {missing[0]}")
    model_path = Path(config["models"][model]["path"])
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()
    model_digest = str(config["models"][model]["sha256"])
    if sha256(model_path) != model_digest:
        raise RuntimeError(f"{model} RKNN checksum changed")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "candidates.jsonl"
    rows = read_checkpoint(checkpoint, model=model, model_sha256=model_digest, role=role, wanted=set(expected))
    remaining = [item for image_id, item in sorted(expected.items()) if image_id not in rows]
    status(output, stage="candidates", backend=model, role=role, done=len(rows), total=len(expected))
    if remaining:
        helpers = imports_from_sibling()
        try:
            from rknnlite.api import RKNNLite
        except ImportError as error:
            raise RuntimeError("RKNNLite is required on the Orange Pi") from error
        capture = helpers["NativeOutputCapture"](output / "rknn_runtime.log")
        capture.open()
        rknn = capture.call(RKNNLite, verbose=False)
        try:
            if capture.call(rknn.load_rknn, str(model_path)) != 0:
                raise RuntimeError("load_rknn failed")
            if capture.call(rknn.init_runtime, core_mask=helpers["core_mask"](RKNNLite, args.core)) != 0:
                raise RuntimeError("init_runtime failed")
            warmup = np.full((1, int(config["runtime"]["imgsz"]), int(config["runtime"]["imgsz"]), 3), 114, dtype=np.uint8)
            if capture.call(rknn.inference, inputs=[warmup]) is None:
                raise RuntimeError("RKNN warmup failed")
            for offset, item in enumerate(remaining, start=1):
                image_id, image_path = int(item["image_id"]), images_root / item["file_name"]
                started = time.perf_counter()
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                decode_ms = (time.perf_counter() - started) * 1000.0
                if image is None:
                    raise RuntimeError(f"could not decode {image_path}")
                if (image.shape[1], image.shape[0]) != (int(item["width"]), int(item["height"])):
                    raise RuntimeError(f"image geometry changed for {image_id}")
                started = time.perf_counter()
                padded, ratio, pad = helpers["letterbox_val"](image, int(config["runtime"]["imgsz"]))
                rknn_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)[None, ...])
                preprocess_ms = (time.perf_counter() - started) * 1000.0
                outputs, inference_ms = capture.timed_call(rknn.inference, inputs=[rknn_input])
                started = time.perf_counter()
                decoded = helpers["decode_rknn_outputs"](outputs, imgsz=int(config["runtime"]["imgsz"]))
                boxes, scores, classes, raw_candidates = helpers["ultralytics_multilabel_nms"](
                    decoded,
                    float(config["runtime"]["conf"]),
                    float(config["runtime"]["iou"]),
                    int(config["runtime"]["max_det"]),
                    int(config["runtime"]["max_nms"]),
                )
                boxes = helpers["scale_boxes"](boxes, image.shape[:2], ratio, pad)
                edge_boxes = [
                    {"xyxy": [float(value) for value in box], "conf": float(score), "cls": int(cls)}
                    for box, score, cls in zip(boxes, scores, classes)
                ]
                nms_ms = (time.perf_counter() - started) * 1000.0
                record = {
                    "schema": RECORD_SCHEMA,
                    "backend": model,
                    "model_sha256": model_digest,
                    "role": role,
                    "image_id": image_id,
                    "file_name": item["file_name"],
                    "shape": [int(image.shape[0]), int(image.shape[1])],
                    "edge_boxes": compact_boxes(edge_boxes),
                    "roi_xyxy": current_roi_xyxy(int(item["width"]), int(item["height"]), edge_boxes),
                    "raw_multilabel_candidates": int(raw_candidates),
                    "decode_ms": decode_ms,
                    "edge_preprocess_ms": preprocess_ms,
                    "edge_inference_ms": inference_ms,
                    "edge_nms_ms": nms_ms,
                }
                append_jsonl(checkpoint, record)
                rows[image_id] = record
                if offset == 1 or len(rows) % int(args.progress_every) == 0 or offset == len(remaining):
                    status(output, stage="candidates", backend=model, role=role, done=len(rows), total=len(expected))
                    print(f"candidates backend={model} role={role} {len(rows)}/{len(expected)} image={image_id}", flush=True)
        finally:
            try:
                capture.call(rknn.release)
            finally:
                capture.close()
    if set(rows) != set(expected):
        raise RuntimeError("candidate collection is incomplete")
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "backend": model,
        "model_sha256": model_digest,
        "role": role,
        "images": len(rows),
        "input_manifest_sha256": _file_sha256(input_path),
        "candidates_sha256": _file_sha256(checkpoint),
        "input_schema": input_payload["schema"],
        "scores_exported": False,
        "roi_source": "raw_rknn_edge_boxes",
    }
    atomic_json(output / "candidate_bundle.json", bundle)
    status(output, stage="complete", backend=model, role=role, done=len(rows), total=len(expected), complete=True)
    print(json.dumps(bundle, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fp16", "int8-heads"), required=True)
    parser.add_argument("--role", choices=tuple(ROLE_COUNTS), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--core", choices=("auto", "0", "1", "2", "all"), default="0")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
