#!/usr/bin/env python3
"""Build resumable VOC Edge, Original, Q90, and Current@640+MAUR caches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from PIL import Image
import yaml


os.environ.setdefault("YOLO_AUTOINSTALL", "False")
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DATA = load_module("voc_point_cache_data", ROOT / "scripts/data_core.py")
CORE = load_module("voc_point_cache_protocol", ROOT / "scripts/protocol_core.py")
SCAN = load_module(
    "voc_point_current_geometry",
    REPO / "paper/experiments/20260728_oregon_cloudv2_roi_profile_scale_scan/scripts/scan_core.py",
)
PROTOCOL_SOURCE = load_module(
    "voc_point_maur_source",
    REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/protocol_core.py",
)


def checkpoint(kind: str) -> Path:
    return (ROOT / PROTOCOL["checkpoints"][kind]["file"]).resolve()


def role_root(role: str) -> Path:
    if role not in PROTOCOL["roles"]:
        raise ValueError(role)
    return RUNS / f"data/roles/{role}"


def gt_path(role: str) -> Path:
    return role_root(role) / "gt.json"


def _complete(path: Path) -> bool:
    return path.is_file() and DATA.read_json(path).get("complete") is True


def preflight() -> dict[str, Any]:
    output = RUNS / "preflight"
    completion = output / "completion.json"
    if _complete(completion):
        return DATA.read_json(output / "report.json")
    if not _complete(RUNS / "data/completion.json"):
        raise RuntimeError("VOC data preparation must complete first")
    from ultralytics import YOLO

    expected_names = list(map(str, PROTOCOL["data"]["category_names"]))
    models = {}
    for kind in ("edge", "cloud"):
        path = checkpoint(kind)
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_hash = DATA.file_sha256(path)
        expected_hash = str(PROTOCOL["checkpoints"][kind]["sha256"])
        if observed_hash != expected_hash:
            raise RuntimeError(f"{kind} checkpoint hash changed")
        model = YOLO(str(path))
        CORE.validate_category_names(model.names, expected_names)
        train_args = dict((model.ckpt or {}).get("train_args") or {})
        models[kind] = {
            "path": str(path),
            "sha256": observed_hash,
            "task": str(model.task),
            "classes": len(model.names),
            "train_args": {
                key: train_args.get(key)
                for key in ("data", "epochs", "batch", "imgsz", "seed", "split", "name")
            },
        }
        del model
        gc.collect()
    roles = {}
    identities = {}
    for role, expected in PROTOCOL["roles"].items():
        gt = DATA.read_json(gt_path(role))
        if len(gt["images"]) != int(expected):
            raise RuntimeError(f"role population changed: {role}")
        values = {str(image["identity"]) for image in gt["images"]}
        if len(values) != int(expected):
            raise RuntimeError(f"duplicate role identity: {role}")
        identities[role] = values
        roles[role] = len(values)
    if any(identities[left] & identities[right] for left in identities for right in identities if left < right):
        raise RuntimeError("VOC roles overlap")
    report = {
        "complete": True,
        "models": models,
        "roles": roles,
        "community_detector_test_was_used_for_checkpoint_selection": True,
        "final_used_for_router_training_or_selection": False,
    }
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(completion, {"complete": True, "report_sha256": DATA.file_sha256(output / "report.json")})
    return report


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True)


def infer_full(role: str, kind: str) -> dict[str, Any]:
    action = "edge" if kind == "edge" else "original"
    output = RUNS / f"actions/{role}/{action}"
    predictions = output / "predictions.json"
    completion = output / "completion.json"
    if _complete(completion):
        return DATA.read_json(completion)
    output.mkdir(parents=True, exist_ok=True)
    inference = PROTOCOL["inference"]
    validator = ROOT / "scripts/predict_cache.py"
    _run(
        [
            sys.executable,
            str(validator),
            "--gt-json",
            str(gt_path(role)),
            "--model",
            str(checkpoint(kind)),
            "--out-json",
            str(predictions),
            "--imgsz",
            str(inference["imgsz"]),
            "--conf",
            str(inference["conf"]),
            "--iou",
            str(inference["iou"]),
            "--max-det",
            str(inference["max_det"]),
            "--device",
            str(inference["device"]),
            "--batch",
            str(inference["edge_batch"] if kind == "edge" else inference["cloud_batch"]),
            "--half",
        ]
    )
    report = {
        "complete": True,
        "role": role,
        "action": action,
        "checkpoint_sha256": DATA.file_sha256(checkpoint(kind)),
        "predictions_sha256": DATA.file_sha256(predictions),
    }
    DATA.atomic_json(completion, report)
    return report


def extract_fpn(role: str) -> dict[str, Any]:
    output = RUNS / f"actions/{role}/edge_fpn"
    features = output / "global_fpn_features.json"
    completion = output / "completion.json"
    if _complete(completion):
        return DATA.read_json(completion)
    script_dir = REPO / "paper/experiments/20260712_coco_frozen_semantic_ranker/scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    source = load_module("voc_point_fpn_source", script_dir / "extract_global_fpn_features.py")
    output.mkdir(parents=True, exist_ok=True)
    gt = DATA.read_json(gt_path(role))
    image_paths = [Path(image["absolute_path"]) for image in gt["images"]]
    file_to_id = DATA.source_basename_to_id(gt["images"])
    dataset_yaml = source.stage_coco_subset(
        output,
        RUNS / "data/voc_template.yaml",
        gt_path(role),
        image_paths,
    )
    source.GlobalFPNFeatureValidator.output_path = features
    source.GlobalFPNFeatureValidator.file_to_id = file_to_id
    source.GlobalFPNFeatureValidator.run_metadata = {
        "protocol": "voc_edge_native_fpn_mean_v1",
        "role": role,
        "model_sha256": DATA.file_sha256(checkpoint("edge")),
    }
    from ultralytics import YOLO

    inference = PROTOCOL["inference"]
    YOLO(str(checkpoint("edge"))).val(
        validator=source.GlobalFPNFeatureValidator,
        data=str(dataset_yaml),
        split="val",
        imgsz=int(inference["imgsz"]),
        batch=int(inference["edge_batch"]),
        workers=8,
        device=str(inference["device"]),
        conf=float(inference["conf"]),
        iou=float(inference["iou"]),
        max_det=int(inference["max_det"]),
        end2end=False,
        rect=True,
        plots=False,
        save_json=False,
        project=str(output),
        name="validator",
        exist_ok=True,
        verbose=False,
    )
    payload = DATA.read_json(features)
    if len(payload["features"]) != int(PROTOCOL["roles"][role]):
        raise RuntimeError(f"FPN feature population changed: {role}")
    report = {
        "complete": True,
        "role": role,
        "features": len(payload["features"]),
        "features_sha256": DATA.file_sha256(features),
    }
    DATA.atomic_json(completion, report)
    return report


def encode_q90(role: str) -> dict[str, Any]:
    output = RUNS / f"actions/{role}/q90"
    manifest_path = output / "q90_manifest.json"
    completion = output / "encode_completion.json"
    if _complete(completion):
        return DATA.read_json(completion)
    gt = DATA.read_json(gt_path(role))
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    quality = int(PROTOCOL["geometry"]["jpeg_quality"])

    def encode(image: Mapping[str, Any]) -> dict[str, Any]:
        source = Path(str(image["absolute_path"]))
        target = image_dir / str(image["file_name"])
        if not target.is_file():
            with Image.open(source) as opened:
                opened.convert("RGB").save(
                    target,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                    subsampling=2,
                )
        return {
            "image_id": int(image["id"]),
            "identity": str(image["identity"]),
            "path": str(target.resolve()),
            "width": int(image["width"]),
            "height": int(image["height"]),
            "original_bytes": source.stat().st_size,
            "action_bytes": target.stat().st_size,
        }

    with ThreadPoolExecutor(max_workers=int(PROTOCOL["geometry"]["encode_workers"])) as executor:
        records = list(executor.map(encode, gt["images"]))
    DATA.atomic_json(manifest_path, {"role": role, "records": records})
    report = {
        "complete": True,
        "role": role,
        "images": len(records),
        "manifest_sha256": DATA.file_sha256(manifest_path),
    }
    DATA.atomic_json(completion, report)
    return report


def _pseudo_gt(role: str, action: str, records: list[dict[str, Any]]) -> Path:
    source = DATA.read_json(gt_path(role))
    images = {int(image["id"]): dict(image) for image in source["images"]}
    record_ids = {int(record["image_id"]) for record in records}
    for record in records:
        image = images[int(record["image_id"])]
        image["absolute_path"] = str(record["path"])
        image["file_name"] = Path(str(record["path"])).name
        if action == "roi":
            x1, y1, x2, y2 = map(int, record["xyxy"])
            image["width"] = x2 - x1
            image["height"] = y2 - y1
    payload = {
        "images": [images[image_id] for image_id in sorted(record_ids)],
        "annotations": [],
        "categories": source["categories"],
    }
    path = RUNS / f"actions/{role}/{action}/pseudo_gt.json"
    DATA.atomic_json(path, payload)
    return path


def infer_q90(role: str) -> dict[str, Any]:
    encode_q90(role)
    output = RUNS / f"actions/{role}/q90"
    predictions = output / "predictions.json"
    completion = output / "completion.json"
    if _complete(completion):
        return DATA.read_json(completion)
    manifest = DATA.read_json(output / "q90_manifest.json")["records"]
    pseudo = _pseudo_gt(role, "q90", manifest)
    inference = PROTOCOL["inference"]
    predictor = ROOT / "scripts/predict_cache.py"
    _run(
        [
            sys.executable,
            str(predictor),
            "--gt-json",
            str(pseudo),
            "--model",
            str(checkpoint("cloud")),
            "--out-json",
            str(predictions),
            "--imgsz",
            str(inference["imgsz"]),
            "--conf",
            str(inference["conf"]),
            "--iou",
            str(inference["iou"]),
            "--max-det",
            str(inference["max_det"]),
            "--device",
            str(inference["device"]),
            "--batch",
            str(inference["cloud_batch"]),
            "--half",
        ]
    )
    report = {
        "complete": True,
        "role": role,
        "action": "q90",
        "predictions_sha256": DATA.file_sha256(predictions),
        "manifest_sha256": DATA.file_sha256(output / "q90_manifest.json"),
    }
    DATA.atomic_json(completion, report)
    return report


def prepare_roi(role: str) -> dict[str, Any]:
    output = RUNS / f"actions/{role}/roi"
    manifest_path = output / "roi_manifest.json"
    completion = output / "geometry_completion.json"
    if _complete(completion):
        return DATA.read_json(completion)
    gt = DATA.read_json(gt_path(role))
    edge_rows = DATA.read_json(RUNS / f"actions/{role}/edge/predictions.json")
    edge = DATA.group_predictions(edge_rows)
    crop_dir = output / "images"
    crop_dir.mkdir(parents=True, exist_ok=True)
    settings = PROTOCOL["geometry"]

    def encode(image: Mapping[str, Any]) -> dict[str, Any] | None:
        image_id = int(image["id"])
        xyxy = SCAN.merged_roi_xyxy(
            int(image["width"]),
            int(image["height"]),
            edge.get(image_id, []),
            profile=str(settings["profile"]),
            confidence=float(settings["edge_conf"]),
            max_area_ratio=float(settings["max_area_ratio"]),
        )
        if xyxy is None:
            return None
        source = Path(str(image["absolute_path"]))
        target = crop_dir / str(image["file_name"])
        x1, y1, x2, y2 = map(int, xyxy)
        if not target.is_file():
            with Image.open(source) as opened:
                opened.crop((x1, y1, x2, y2)).convert("RGB").save(
                    target,
                    format="JPEG",
                    quality=int(settings["jpeg_quality"]),
                    optimize=True,
                    subsampling=2,
                )
        return {
            "image_id": image_id,
            "identity": str(image["identity"]),
            "path": str(target.resolve()),
            "xyxy": [x1, y1, x2, y2],
            "area_ratio": (x2 - x1) * (y2 - y1) / max(1, int(image["width"]) * int(image["height"])),
            "bytes": target.stat().st_size,
            "profile": "current",
            "imgsz": 640,
        }

    with ThreadPoolExecutor(max_workers=int(settings["encode_workers"])) as executor:
        values = list(executor.map(encode, gt["images"]))
    records = [value for value in values if value is not None]
    DATA.atomic_json(manifest_path, {"role": role, "records": records})
    report = {
        "complete": True,
        "role": role,
        "images": len(gt["images"]),
        "roi_candidates": len(records),
        "manifest_sha256": DATA.file_sha256(manifest_path),
    }
    DATA.atomic_json(completion, report)
    return report


def infer_roi(role: str) -> dict[str, Any]:
    prepare_roi(role)
    output = RUNS / f"actions/{role}/roi"
    completion = output / "completion.json"
    if _complete(completion):
        return DATA.read_json(completion)
    records = DATA.read_json(output / "roi_manifest.json")["records"]
    raw_path = output / "crop_predictions.json"
    mapped_path = output / "mapped_predictions.json"
    fused_path = output / "maur_predictions.json"
    if records:
        pseudo = _pseudo_gt(role, "roi", records)
        inference = PROTOCOL["inference"]
        predictor = ROOT / "scripts/predict_cache.py"
        _run(
            [
                sys.executable,
                str(predictor),
                "--gt-json",
                str(pseudo),
                "--model",
                str(checkpoint("cloud")),
                "--out-json",
                str(raw_path),
                "--imgsz",
                str(PROTOCOL["geometry"]["imgsz"]),
                "--conf",
                str(inference["conf"]),
                "--iou",
                str(inference["iou"]),
                "--max-det",
                str(inference["max_det"]),
                "--device",
                str(inference["device"]),
                "--batch",
                str(inference["cloud_batch"]),
                "--half",
            ]
        )
    else:
        DATA.atomic_json(raw_path, [])
    geometry = {int(record["image_id"]): record for record in records}
    mapped = DATA.map_crop_predictions(DATA.read_json(raw_path), geometry)
    DATA.atomic_json(mapped_path, mapped)
    gt = DATA.read_json(gt_path(role))
    image_ids = [int(image["id"]) for image in gt["images"]]
    edge = DATA.group_predictions(
        DATA.read_json(RUNS / f"actions/{role}/edge/predictions.json"), image_ids=image_ids
    )
    cloud = DATA.group_predictions(mapped, image_ids=image_ids)
    fused = {}
    for image_id in sorted(geometry):
        candidates = PROTOCOL_SOURCE.maur_candidates(
            edge[image_id], cloud[image_id], tuple(map(int, geometry[image_id]["xyxy"]))
        )
        fused[image_id] = PROTOCOL_SOURCE.class_aware_nms(
            candidates, iou_threshold=float(PROTOCOL["fusion"]["nms_iou"])
        )
    DATA.atomic_json(fused_path, DATA.flat_predictions(fused))
    report = {
        "complete": True,
        "role": role,
        "action": "current@640",
        "roi_candidates": len(records),
        "mapped_predictions_sha256": DATA.file_sha256(mapped_path),
        "maur_predictions_sha256": DATA.file_sha256(fused_path),
    }
    DATA.atomic_json(completion, report)
    return report


def run_role(role: str) -> dict[str, Any]:
    infer_full(role, "edge")
    extract_fpn(role)
    infer_full(role, "cloud")
    infer_q90(role)
    infer_roi(role)
    report = {
        "complete": True,
        "role": role,
        "actions": list(PROTOCOL["actions"]),
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(RUNS / f"actions/{role}/completion.json", report)
    return report


def run_all() -> dict[str, Any]:
    preflight()
    reports = {role: run_role(role) for role in PROTOCOL["roles"]}
    report = {"complete": True, "roles": list(reports), "actions": list(PROTOCOL["actions"])}
    DATA.atomic_json(RUNS / "actions/report.json", report)
    DATA.atomic_json(RUNS / "actions/completion.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "all", "role"), default="all")
    parser.add_argument("--role", choices=tuple(PROTOCOL["roles"]))
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
    elif args.stage == "role":
        if args.role is None:
            parser.error("--role is required for --stage role")
        result = run_role(args.role)
    else:
        result = run_all()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
