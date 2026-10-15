#!/usr/bin/env python3
"""Build protocol-bound Edge, Original, Q90, and Current@official+MAUR caches."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from PIL import Image


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
RUNS = ROOT / "runs"
PROTOCOL_PATH = ROOT / "configs/protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("voc_torchvision_cache_core", ROOT / "scripts/core.py")
DATA = load_module("voc_torchvision_cache_data", ROOT / "scripts/data_core.py")
ROI = load_module("voc_torchvision_cache_roi", ROOT / "scripts/roi_core.py")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL line {number}: {path}") from error
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def flatten_action_rows(
    rows: Sequence[Mapping[str, Any]], action: str, expected_ids: set[int]
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    seen: set[int] = set()
    predictions: list[dict[str, Any]] = []
    payload: dict[int, int] = {}
    for row in rows:
        image_id = int(row["image_id"])
        if image_id in seen:
            raise RuntimeError(f"duplicate action row: {action}/{image_id}")
        if image_id not in expected_ids:
            raise RuntimeError(f"unexpected action row: {action}/{image_id}")
        if str(row["action"]) != str(action):
            raise RuntimeError(f"action row changed: {row['action']}/{action}")
        seen.add(image_id)
        payload[image_id] = int(row.get("payload_bytes", 0))
        for source in row.get("predictions", ()):
            prediction = dict(source)
            if int(prediction["image_id"]) != image_id:
                raise RuntimeError(f"prediction image changed: {action}/{image_id}")
            predictions.append(prediction)
    if seen != expected_ids:
        raise RuntimeError(f"incomplete action population: {action}/{len(seen)}/{len(expected_ids)}")
    return predictions, payload


def q90_manifest_records(
    images: Sequence[Mapping[str, Any]], payload: Mapping[int, int]
) -> list[dict[str, Any]]:
    records = []
    for image in images:
        image_id = int(image["id"])
        original_bytes = int(image["original_bytes"])
        action_bytes = int(payload[image_id])
        records.append(
            {
                "image_id": image_id,
                "identity": str(image["identity"]),
                "path": str(image["absolute_path"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "original_bytes": original_bytes,
                "action_bytes": action_bytes,
                "available": 0 < action_bytes < original_bytes,
            }
        )
    return records


def roi_payload_available(action_bytes: int, original_bytes: int, *, required: bool) -> bool:
    return int(action_bytes) > 0 and (not required or int(action_bytes) < int(original_bytes))


def _protocol_sha256() -> str:
    return DATA.file_sha256(PROTOCOL_PATH)


def _source_data_role(role: str) -> dict[str, Any]:
    path = RUNS / f"data/roles/{role}/gt.json"
    if not path.is_file():
        raise RuntimeError("data import must complete before action caching")
    gt = DATA.read_json(path)
    if len(gt["images"]) != int(PROTOCOL["roles"][role]):
        raise RuntimeError(f"role population changed: {role}")
    contract = PROTOCOL["role_contracts"][role]
    if DATA.file_sha256(path) != str(contract["gt_sha256"]):
        raise RuntimeError(f"role ground truth changed: {role}")
    return gt


def _source_gate_root() -> Path:
    return (ROOT / str(PROTOCOL["source_gate_experiment"])).resolve()


def _models() -> dict[str, tuple[Any, Any, dict[str, Any]]]:
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        SSDLite320_MobileNet_V3_Large_Weights,
        fasterrcnn_resnet50_fpn_v2,
        ssdlite320_mobilenet_v3_large,
    )

    threshold = float(PROTOCOL["runtime"]["score_threshold"])
    maximum = int(PROTOCOL["runtime"]["detections_per_image"])
    return {
        "edge": (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
            {"score_thresh": threshold, "detections_per_img": maximum},
        ),
        "cloud": (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
            {"box_score_thresh": threshold, "box_detections_per_img": maximum},
        ),
    }


def _load_model(kind: str) -> tuple[Any, Any]:
    import torch
    import torchvision

    builder, weights, kwargs = _models()[kind]
    if str(torch.__version__) != str(PROTOCOL["runtime"]["torch"]):
        raise RuntimeError(f"torch version changed: {torch.__version__}")
    if str(torchvision.__version__) != str(PROTOCOL["runtime"]["torchvision"]):
        raise RuntimeError(f"torchvision version changed: {torchvision.__version__}")
    model = builder(weights=weights, progress=True, **kwargs)
    weight_path = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / Path(urlparse(str(weights.url)).path).name
    )
    if DATA.file_sha256(weight_path) != str(PROTOCOL["detectors"][kind]["weights_sha256"]):
        raise RuntimeError(f"official Torchvision weight changed: {kind}")
    model.eval().to(torch.device(PROTOCOL["runtime"]["device"]))
    return model, weights


def _voc_category_ids(weights: Any, gt: Mapping[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    label_names = {index: str(name) for index, name in enumerate(weights.meta["categories"])}
    voc_ids = {str(row["name"]): int(row["id"]) for row in gt["categories"]}
    mapping = {
        str(coco): int(voc_ids[str(voc)])
        for coco, voc in PROTOCOL["category_mapping"].items()
    }
    return label_names, mapping


def _sanitize(
    *,
    image_id: int,
    width: int,
    height: int,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    scores: Sequence[float],
    label_names: Mapping[int, str],
    category_ids: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    audit = Counter()
    output = []
    threshold = float(PROTOCOL["runtime"]["score_threshold"])
    for box, label, score in zip(boxes, labels, scores):
        value = float(score)
        if not math.isfinite(value):
            audit["nonfinite_scores"] += 1
            continue
        if value < threshold:
            audit["below_threshold"] += 1
            continue
        name = str(label_names.get(int(label), ""))
        if name not in category_ids:
            audit["filtered_non_voc"] += 1
            continue
        coordinates = list(map(float, box))
        if len(coordinates) != 4 or not all(map(math.isfinite, coordinates)):
            audit["nonfinite_boxes"] += 1
            continue
        x1, y1, x2, y2 = coordinates
        x1, x2 = min(max(x1, 0.0), float(width)), min(max(x2, 0.0), float(width))
        y1, y2 = min(max(y1, 0.0), float(height)), min(max(y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            audit["degenerate_boxes"] += 1
            continue
        output.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_ids[name]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": value,
            }
        )
        audit["kept"] += 1
    return output, dict(audit)


def _encode_q90(image: Image.Image) -> bytes:
    buffer = BytesIO()
    settings = PROTOCOL["jpeg_q90"]
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=int(settings["quality"]),
        optimize=bool(settings["optimize"]),
        subsampling=int(settings["subsampling"]),
    )
    return buffer.getvalue()


def _infer_pil(
    model: Any,
    weights: Any,
    pil_image: Image.Image,
    *,
    image_id: int,
    label_names: Mapping[int, str],
    category_ids: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    import torch

    device = torch.device(PROTOCOL["runtime"]["device"])
    tensor = weights.transforms()(pil_image).to(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model([tensor])[0]
    torch.cuda.synchronize(device)
    predictions, audit = _sanitize(
        image_id=image_id,
        width=pil_image.width,
        height=pil_image.height,
        boxes=prediction["boxes"].detach().cpu().tolist(),
        labels=prediction["labels"].detach().cpu().tolist(),
        scores=prediction["scores"].detach().cpu().tolist(),
        label_names=label_names,
        category_ids=category_ids,
    )
    return predictions, audit, (time.perf_counter() - started) * 1000.0


def _read_checkpoint(path: Path, action: str, expected_ids: set[int]) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = _read_jsonl(path)
    output = {}
    for row in rows:
        image_id = int(row["image_id"])
        if image_id in output or image_id not in expected_ids:
            raise RuntimeError(f"invalid checkpoint image: {action}/{image_id}")
        if str(row.get("action")) != action or str(row.get("protocol_sha256")) != _protocol_sha256():
            raise RuntimeError(f"checkpoint contract changed: {action}/{image_id}")
        output[image_id] = row
    return output


def _materialize_rows(
    *,
    role: str,
    action: str,
    rows_path: Path,
    expected_ids: set[int],
    output: Path,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    rows = _read_jsonl(rows_path)
    predictions, payload = flatten_action_rows(rows, action, expected_ids)
    DATA.atomic_json(output / "predictions.json", predictions)
    report = {
        "complete": True,
        "role": role,
        "action": action,
        "images": len(rows),
        "predictions": len(predictions),
        "payload_bytes": sum(payload.values()),
        "rows_sha256": DATA.file_sha256(rows_path),
        "predictions_sha256": DATA.file_sha256(output / "predictions.json"),
        "protocol_sha256": _protocol_sha256(),
    }
    if action == "q90":
        gt = _source_data_role(role)
        manifest = {"role": role, "records": q90_manifest_records(gt["images"], payload)}
        DATA.atomic_json(output / "q90_manifest.json", manifest)
        report["manifest_sha256"] = DATA.file_sha256(output / "q90_manifest.json")
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(output / "completion.json", report)
    return predictions, payload


def import_gate_action(role: str, action: str) -> dict[str, Any]:
    if role not in {"fit", "calibration"} or action not in {"edge", "original", "q90"}:
        raise ValueError(f"gate import is forbidden: {role}/{action}")
    gt = _source_data_role(role)
    expected_ids = {int(row["id"]) for row in gt["images"]}
    source_root = _source_gate_root()
    source_protocol = source_root / "configs/protocol.json"
    if DATA.file_sha256(source_protocol) != str(PROTOCOL["source_gate_protocol_sha256"]):
        raise RuntimeError("source gate protocol changed")
    source_rows = source_root / f"runs/actions/full/{role}/{action}/rows.jsonl"
    expected_hash = str(PROTOCOL["source_gate_rows"][role][action])
    if DATA.file_sha256(source_rows) != expected_hash:
        raise RuntimeError(f"source gate rows changed: {role}/{action}")
    rows = _read_jsonl(source_rows)
    old_hash = str(PROTOCOL["source_gate_protocol_sha256"])
    if any(str(row.get("protocol_sha256")) != old_hash for row in rows):
        raise RuntimeError(f"source gate row protocol changed: {role}/{action}")
    output = RUNS / f"actions/{role}/{action}"
    rows_path = output / "rows.jsonl"
    if not rows_path.is_file():
        output.mkdir(parents=True, exist_ok=True)
        for row in rows:
            imported = dict(row)
            imported["source_protocol_sha256"] = imported.pop("protocol_sha256")
            imported["protocol_sha256"] = _protocol_sha256()
            _append_jsonl(rows_path, imported)
    _materialize_rows(
        role=role,
        action=action,
        rows_path=rows_path,
        expected_ids=expected_ids,
        output=output,
    )
    report = DATA.read_json(output / "completion.json")
    report["source_gate_rows_sha256"] = expected_hash
    report["source_gate_protocol_sha256"] = old_hash
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(output / "completion.json", report)
    return report


def infer_final_action(action: str, *, limit: int | None = None) -> dict[str, Any]:
    if action not in {"edge", "original", "q90"}:
        raise ValueError(action)
    role = "final"
    gt = _source_data_role(role)
    images = list(gt["images"])
    expected_ids = {int(row["id"]) for row in images}
    output = RUNS / f"actions/{role}/{action}"
    rows_path = output / "rows.jsonl"
    completion = output / "completion.json"
    if completion.is_file() and limit is None:
        report = DATA.read_json(completion)
        if report.get("rows_sha256") != DATA.file_sha256(rows_path):
            raise RuntimeError(f"completed final action changed: {action}")
        return report
    existing = _read_checkpoint(rows_path, action, expected_ids)
    pending = [row for row in images if int(row["id"]) not in existing]
    if limit is not None:
        pending = pending[: max(0, int(limit))]
    kind = "edge" if action == "edge" else "cloud"
    model, weights = _load_model(kind)
    label_names, category_ids = _voc_category_ids(weights, gt)
    started = time.time()
    for index, image in enumerate(pending, 1):
        source = Path(str(image["absolute_path"]))
        with Image.open(source) as opened:
            if action == "q90":
                encoded = _encode_q90(opened)
                pil_image = Image.open(BytesIO(encoded)).convert("RGB")
                payload_bytes = len(encoded)
            else:
                pil_image = opened.convert("RGB")
                payload_bytes = 0 if action == "edge" else source.stat().st_size
            predictions, audit, latency_ms = _infer_pil(
                model,
                weights,
                pil_image,
                image_id=int(image["id"]),
                label_names=label_names,
                category_ids=category_ids,
            )
        _append_jsonl(
            rows_path,
            {
                "action": action,
                "image_id": int(image["id"]),
                "identity": str(image["identity"]),
                "payload_bytes": int(payload_bytes),
                "latency_ms": float(latency_ms),
                "predictions": predictions,
                "sanitization": audit,
                "protocol_sha256": _protocol_sha256(),
            },
        )
        if index % 50 == 0 or index == len(pending):
            print(f"[final/{action}] {len(existing) + index}/{len(images)}", flush=True)
    current = _read_checkpoint(rows_path, action, expected_ids)
    if set(current) != expected_ids:
        return {
            "complete": False,
            "role": role,
            "action": action,
            "images": len(current),
            "expected_images": len(images),
            "elapsed_seconds_this_invocation": time.time() - started,
        }
    _materialize_rows(
        role=role,
        action=action,
        rows_path=rows_path,
        expected_ids=expected_ids,
        output=output,
    )
    return DATA.read_json(completion)


def prepare_roi(role: str) -> dict[str, Any]:
    gt = _source_data_role(role)
    output = RUNS / f"actions/{role}/current@official"
    manifest_path = output / "roi_manifest.json"
    completion = output / "geometry_completion.json"
    if completion.is_file():
        report = DATA.read_json(completion)
        if report.get("manifest_sha256") != DATA.file_sha256(manifest_path):
            raise RuntimeError(f"completed ROI geometry changed: {role}")
        return report
    edge_path = RUNS / f"actions/{role}/edge/predictions.json"
    if not edge_path.is_file():
        raise RuntimeError(f"edge action must complete before ROI: {role}")
    edge = DATA.group_predictions(DATA.read_json(edge_path))
    settings = PROTOCOL["roi"]
    crop_dir = output / "images"
    crop_dir.mkdir(parents=True, exist_ok=True)

    def encode(image: Mapping[str, Any]) -> dict[str, Any]:
        image_id = int(image["id"])
        xyxy = ROI.merged_roi_xyxy(
            int(image["width"]),
            int(image["height"]),
            edge.get(image_id, ()),
            profile=str(settings["profile"]),
            confidence=float(settings["edge_confidence"]),
            max_area_ratio=float(settings["maximum_area_ratio"]),
        )
        base = {
            "image_id": image_id,
            "identity": str(image["identity"]),
            "original_bytes": int(image["original_bytes"]),
            "profile": str(settings["profile"]),
            "cloud_transform": str(PROTOCOL["detectors"]["cloud"]["transform"]),
        }
        if xyxy is None:
            return {**base, "available": False, "reason": "geometry_unavailable", "action_bytes": 0}
        x1, y1, x2, y2 = map(int, xyxy)
        target = crop_dir / str(image["file_name"])
        if not target.is_file():
            with Image.open(str(image["absolute_path"])) as opened:
                opened.crop((x1, y1, x2, y2)).convert("RGB").save(
                    target,
                    format="JPEG",
                    quality=int(settings["jpeg_quality"]),
                    optimize=bool(settings["jpeg_optimize"]),
                    subsampling=int(settings["jpeg_subsampling"]),
                )
        action_bytes = target.stat().st_size
        available = roi_payload_available(
            action_bytes,
            int(image["original_bytes"]),
            required=bool(settings["require_bytes_below_original"]),
        )
        return {
            **base,
            "path": str(target.resolve()),
            "xyxy": [x1, y1, x2, y2],
            "width": x2 - x1,
            "height": y2 - y1,
            "area_ratio": (x2 - x1) * (y2 - y1) / max(1, int(image["width"]) * int(image["height"])),
            "action_bytes": int(action_bytes),
            "available": bool(available),
            "reason": "available" if available else "not_smaller_than_original",
        }

    with ThreadPoolExecutor(max_workers=int(settings["encode_workers"])) as executor:
        records = list(executor.map(encode, gt["images"]))
    DATA.atomic_json(manifest_path, {"role": role, "records": records})
    report = {
        "complete": True,
        "role": role,
        "images": len(records),
        "geometry_candidates": sum("xyxy" in row for row in records),
        "roi_available": sum(bool(row["available"]) for row in records),
        "manifest_sha256": DATA.file_sha256(manifest_path),
        "protocol_sha256": _protocol_sha256(),
    }
    DATA.atomic_json(output / "geometry_report.json", report)
    DATA.atomic_json(completion, report)
    return report


def infer_roi(role: str, *, limit: int | None = None) -> dict[str, Any]:
    prepare_roi(role)
    gt = _source_data_role(role)
    output = RUNS / f"actions/{role}/current@official"
    manifest = DATA.read_json(output / "roi_manifest.json")["records"]
    records = [row for row in manifest if bool(row["available"])]
    expected_ids = {int(row["image_id"]) for row in records}
    rows_path = output / "rows.jsonl"
    completion = output / "completion.json"
    if completion.is_file() and limit is None:
        report = DATA.read_json(completion)
        if report.get("rows_sha256") != DATA.file_sha256(rows_path):
            raise RuntimeError(f"completed ROI action changed: {role}")
        return report
    existing = _read_checkpoint(rows_path, "current@official", expected_ids)
    pending = [row for row in records if int(row["image_id"]) not in existing]
    if limit is not None:
        pending = pending[: max(0, int(limit))]
    model, weights = _load_model("cloud")
    label_names, category_ids = _voc_category_ids(weights, gt)
    started = time.time()
    for index, row in enumerate(pending, 1):
        with Image.open(str(row["path"])) as opened:
            predictions, audit, latency_ms = _infer_pil(
                model,
                weights,
                opened.convert("RGB"),
                image_id=int(row["image_id"]),
                label_names=label_names,
                category_ids=category_ids,
            )
        _append_jsonl(
            rows_path,
            {
                "action": "current@official",
                "image_id": int(row["image_id"]),
                "identity": str(row["identity"]),
                "payload_bytes": int(row["action_bytes"]),
                "latency_ms": float(latency_ms),
                "predictions": predictions,
                "sanitization": audit,
                "protocol_sha256": _protocol_sha256(),
            },
        )
        if index % 50 == 0 or index == len(pending):
            print(f"[{role}/current@official] {len(existing) + index}/{len(records)}", flush=True)
    current = _read_checkpoint(rows_path, "current@official", expected_ids)
    if set(current) != expected_ids:
        return {
            "complete": False,
            "role": role,
            "action": "current@official",
            "images": len(current),
            "expected_images": len(records),
            "elapsed_seconds_this_invocation": time.time() - started,
        }
    raw_predictions, _ = flatten_action_rows(list(current.values()), "current@official", expected_ids)
    geometry = {int(row["image_id"]): row for row in records}
    mapped = ROI.map_crop_predictions(raw_predictions, geometry)
    DATA.atomic_json(output / "crop_predictions.json", raw_predictions)
    DATA.atomic_json(output / "mapped_predictions.json", mapped)
    image_ids = [int(row["id"]) for row in gt["images"]]
    edge = DATA.group_predictions(
        DATA.read_json(RUNS / f"actions/{role}/edge/predictions.json"), image_ids=image_ids
    )
    cloud = DATA.group_predictions(mapped, image_ids=image_ids)
    fused: dict[int, list[dict[str, Any]]] = {}
    for image_id in sorted(expected_ids):
        candidates = ROI.maur_candidates(
            edge[image_id],
            cloud[image_id],
            geometry[image_id]["xyxy"],
            duplicate_iou=float(PROTOCOL["fusion"]["duplicate_iou"]),
        )
        fused[image_id] = ROI.class_aware_nms(
            candidates, iou_threshold=float(PROTOCOL["fusion"]["nms_iou"])
        )
    DATA.atomic_json(output / "maur_predictions.json", DATA.flat_predictions(fused))
    report = {
        "complete": True,
        "role": role,
        "action": "current@official",
        "images": len(records),
        "rows_sha256": DATA.file_sha256(rows_path),
        "crop_predictions_sha256": DATA.file_sha256(output / "crop_predictions.json"),
        "mapped_predictions_sha256": DATA.file_sha256(output / "mapped_predictions.json"),
        "maur_predictions_sha256": DATA.file_sha256(output / "maur_predictions.json"),
        "manifest_sha256": DATA.file_sha256(output / "roi_manifest.json"),
        "protocol_sha256": _protocol_sha256(),
        "elapsed_seconds_this_invocation": time.time() - started,
    }
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(completion, report)
    return report


def run_role(role: str, *, limit: int | None = None) -> dict[str, Any]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(role)
    CORE.validate_protocol(PROTOCOL)
    reports = {}
    if role in {"fit", "calibration"}:
        for action in ("edge", "original", "q90"):
            reports[action] = import_gate_action(role, action)
    else:
        for action in ("edge", "original", "q90"):
            reports[action] = infer_final_action(action, limit=limit)
            if not reports[action]["complete"]:
                return {"complete": False, "role": role, "actions": reports}
    reports["current@official"] = infer_roi(role, limit=limit)
    complete = all(bool(row["complete"]) for row in reports.values())
    report = {
        "complete": complete,
        "role": role,
        "actions": reports,
        "protocol_sha256": _protocol_sha256(),
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(RUNS / f"actions/{role}/report.json", report)
    if complete:
        DATA.atomic_json(RUNS / f"actions/{role}/completion.json", report)
    return report


def run_all() -> dict[str, Any]:
    reports = {role: run_role(role) for role in PROTOCOL["roles"]}
    report = {
        "complete": all(bool(row["complete"]) for row in reports.values()),
        "roles": {role: bool(row["complete"]) for role, row in reports.items()},
        "actions": list(PROTOCOL["action_spaces"]["OQR"]),
        "protocol_sha256": _protocol_sha256(),
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(RUNS / "actions/report.json", report)
    if report["complete"]:
        DATA.atomic_json(RUNS / "actions/completion.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=(*PROTOCOL["roles"], "all"), default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = run_all() if args.role == "all" else run_role(args.role, limit=args.limit)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
