#!/usr/bin/env python3
"""Build annotation-free calibration records for causal byte-price freezing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BOARD_SCRIPTS = ROOT / "board_bundle" / "scripts"
if str(BOARD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BOARD_SCRIPTS))

from cavs_numpy import current_roi_xyxy
from online_methods import MethodRuntime, build_method_runtime, score_current_frame
from protocol import compact_boxes, native_jpeg, q90_jpeg


SCHEMA = "cavs_online_calibration_records_v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read JSON: {path}") from error


def expected_rknn_sha256(protocol: Mapping[str, Any]) -> str:
    direct = protocol.get("rknn_sha256")
    nested = protocol.get("models", {}).get("fp16", {}).get("sha256") if isinstance(protocol.get("models"), Mapping) else None
    value = str(direct or nested or "").lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("protocol must declare the FP16 RKNN SHA-256")
    return value


def _safe_image_path(images_dir: Path, file_name: str) -> Path:
    relative = Path(file_name)
    if relative.is_absolute() or ".." in relative.parts or not file_name:
        raise ValueError(f"unsafe calibration image path: {file_name}")
    root = images_dir.resolve()
    path = (root / relative).resolve()
    if path.parent != root and root not in path.parents:
        raise ValueError(f"calibration image escapes images root: {file_name}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _roi_for_record(
    runtime: MethodRuntime,
    edge_boxes: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> list[int] | None:
    if "current@640" not in runtime.action_ids:
        return None
    if runtime.policy is None or not isinstance(runtime.policy.get("roi"), Mapping):
        raise ValueError("ROI action requires a policy ROI contract")
    roi = runtime.policy["roi"]
    return current_roi_xyxy(
        int(width),
        int(height),
        edge_boxes,
        confidence=float(roi["confidence"]),
        max_area_ratio=float(roi["max_area_ratio"]),
    )


def _frame_record(
    raw: Mapping[str, Any],
    *,
    images_dir: Path,
    runtime: MethodRuntime,
) -> dict[str, Any]:
    image_id = int(raw.get("image_id", 0))
    file_name = str(raw.get("file_name", ""))
    width, height = int(raw.get("width", 0)), int(raw.get("height", 0))
    if image_id <= 0 or width <= 0 or height <= 0:
        raise ValueError("calibration candidate has invalid identity or dimensions")
    path = _safe_image_path(images_dir, file_name)
    original = native_jpeg(path)
    if len(original) != int(raw.get("native_jpeg_bytes", -1)):
        raise RuntimeError(f"native JPEG byte drift for image {image_id}")
    edge_boxes = compact_boxes(raw.get("edge_boxes", ()))
    roi_xyxy = _roi_for_record(runtime, edge_boxes, width=width, height=height)
    payloads: dict[str, int] = {"edge": 0, "original": len(original)}
    if "q90" in runtime.action_ids:
        q90 = q90_jpeg(path)
        if 0 < len(q90) < len(original):
            payloads["q90"] = len(q90)
    if "current@640" in runtime.action_ids and roi_xyxy is not None:
        payloads["current@640"] = len(q90_jpeg(path, crop_xyxy=roi_xyxy))
    scores = score_current_frame(
        runtime,
        image_id,
        edge_boxes,
        width=width,
        height=height,
        roi_xyxy=roi_xyxy,
    )
    scores = {action: float(scores[action]) for action in payloads}
    return {
        "image_id": image_id,
        "payload_bytes": payloads,
        "scores": scores,
        "objective_values": dict(scores),
    }


def build_records_bundle(
    *,
    candidates_path: str | Path,
    candidate_report_path: str | Path,
    protocol_path: str | Path,
    images_dir: str | Path,
    method_runtime: MethodRuntime,
    policy_source: str | Path,
) -> dict[str, Any]:
    candidates_path = Path(candidates_path)
    report_path = Path(candidate_report_path)
    protocol_path = Path(protocol_path)
    policy_path = Path(policy_source)
    candidates = read_json(candidates_path)
    report = read_json(report_path)
    protocol = read_json(protocol_path)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("calibration candidates must be a non-empty list")
    if not isinstance(report, Mapping) or report.get("complete") is not True:
        raise ValueError("calibration candidate report is incomplete")
    if str(report.get("role")) != "calibration":
        raise ValueError("formal records require the calibration candidate split")
    if int(report.get("images", -1)) != len(candidates):
        raise ValueError("calibration candidate population differs from its report")
    model_sha256 = expected_rknn_sha256(protocol)
    if str(report.get("model_sha256", "")).lower() != model_sha256:
        raise ValueError("calibration candidates use a different RKNN model")
    protocol_id = str(protocol.get("protocol_id", ""))
    if not protocol_id:
        raise ValueError("protocol_id is required")

    records = [
        _frame_record(
            raw,
            images_dir=Path(images_dir),
            runtime=method_runtime,
        )
        for raw in candidates
    ]
    image_ids = [int(record["image_id"]) for record in records]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("calibration candidate image IDs are not unique")
    return {
        "schema": SCHEMA,
        "protocol_id": protocol_id,
        "source_split": "calibration",
        "method": method_runtime.method,
        "random_seed": (
            int(method_runtime.random_seed)
            if method_runtime.random_seed is not None else None
        ),
        "calibration_objective": {
            "name": "predicted_action_value",
            "direction": "maximize",
        },
        "action_ids": list(method_runtime.action_ids),
        "rknn_model_sha256": model_sha256,
        "source_hashes": {
            "candidate_report": file_sha256(report_path),
            "candidates": file_sha256(candidates_path),
            "policy": file_sha256(policy_path),
        },
        "records": records,
        "ground_truth_read": False,
        "final_used_for_tuning": False,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--random-seed", type=int, default=20260720)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = read_json(args.protocol)
    rknn_sha256 = expected_rknn_sha256(protocol)
    policy_source = args.policy or (BOARD_SCRIPTS / "online_methods.py")
    runtime = build_method_runtime(
        args.method,
        policy_path=args.policy,
        random_seed=args.random_seed,
        expected_rknn_sha256=rknn_sha256,
    )
    bundle = build_records_bundle(
        candidates_path=args.candidates,
        candidate_report_path=args.candidate_report,
        protocol_path=args.protocol,
        images_dir=args.images,
        method_runtime=runtime,
        policy_source=policy_source,
    )
    _atomic_json(args.output, bundle)
    print(json.dumps({
        "output": str(args.output),
        "method": runtime.method,
        "records": len(bundle["records"]),
        "sha256": file_sha256(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
