#!/usr/bin/env python3
"""Assemble VOC action records and observable learner inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


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


DATA = load_module("voc_point_common_data", ROOT / "scripts/data_core.py")
load_module(
    "protocol_core",
    REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/protocol_core.py",
)
BENCHMARK = load_module(
    "voc_point_common_quality",
    REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/benchmark_core.py",
)


def _group(path: Path, image_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    return DATA.group_predictions(DATA.read_json(path), image_ids=image_ids)


def _records(path: Path) -> dict[int, dict[str, Any]]:
    values = DATA.read_json(path).get("records", [])
    output = {int(row["image_id"]): dict(row) for row in values}
    if len(output) != len(values):
        raise RuntimeError(f"duplicate action manifest image: {path}")
    return output


def _quality_map(gt: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in gt["annotations"]:
        output[int(row["image_id"])].append(
            {
                "category_id": int(row["category_id"]),
                "bbox": list(map(float, row["bbox"])),
            }
        )
    return dict(output)


def build_role(role: str) -> dict[str, Any]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(role)
    output = RUNS / f"common/{role}"
    completion = output / "completion.json"
    samples_path = output / "samples.json"
    records_path = output / "records.pkl"
    manifest_path = output / "source_manifest.json"
    if completion.is_file():
        report = DATA.read_json(completion)
        for key, path in (("samples_sha256", samples_path), ("records_sha256", records_path)):
            if report.get(key) != DATA.file_sha256(path):
                raise RuntimeError(f"common artifact changed: {role}/{path.name}")
        return report

    gt_path = RUNS / f"data/roles/{role}/gt.json"
    action_root = RUNS / f"actions/{role}"
    paths = {
        "gt": gt_path,
        "edge": action_root / "edge/predictions.json",
        "original": action_root / "original/predictions.json",
        "q90": action_root / "q90/predictions.json",
        "q90_manifest": action_root / "q90/q90_manifest.json",
        "roi": action_root / "roi/maur_predictions.json",
        "roi_manifest": action_root / "roi/roi_manifest.json",
        "fpn": action_root / "edge_fpn/global_fpn_features.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    gt = DATA.read_json(gt_path)
    images = {int(image["id"]): dict(image) for image in gt["images"]}
    image_ids = sorted(images)
    if len(image_ids) != int(PROTOCOL["roles"][role]):
        raise RuntimeError(f"common role population changed: {role}")
    actions = {
        name: _group(path, image_ids)
        for name, path in paths.items()
        if name in {"edge", "original", "q90", "roi"}
    }
    q90 = _records(paths["q90_manifest"])
    roi = _records(paths["roi_manifest"])
    features = DATA.read_json(paths["fpn"]).get("features", {})
    if set(map(int, features)) != set(image_ids):
        raise RuntimeError(f"FPN feature population changed: {role}")
    truth = _quality_map(gt)
    samples = []
    for image_id in image_ids:
        image = images[image_id]
        q90_row = q90[image_id]
        original_bytes = int(q90_row["original_bytes"])
        q90_bytes = int(q90_row["action_bytes"])
        roi_row = roi.get(image_id)
        payloads = {"original": original_bytes, "q90": q90_bytes}
        available = {
            "original": True,
            "q90": 0 < q90_bytes < original_bytes,
            "current@640": roi_row is not None,
        }
        action_rows = [
            {"action_id": "edge", "payload_bytes": 0, "available": True},
            {"action_id": "original", "payload_bytes": original_bytes, "available": True},
            {"action_id": "q90", "payload_bytes": q90_bytes, "available": available["q90"]},
        ]
        if roi_row is not None:
            payloads["current@640"] = int(roi_row["bytes"])
            action_rows.append(
                {
                    "action_id": "current@640",
                    "payload_bytes": int(roi_row["bytes"]),
                    "available": True,
                    "profile": "current",
                    "imgsz": 640,
                    "xyxy": list(map(int, roi_row["xyxy"])),
                    "area_ratio": float(roi_row["area_ratio"]),
                }
            )
        qualities = {
            name: float(BENCHMARK.image_quality(truth.get(image_id, []), actions[name][image_id]))
            for name in ("edge", "original", "q90")
        }
        if roi_row is not None:
            qualities["current@640"] = float(
                BENCHMARK.image_quality(truth.get(image_id, []), actions["roi"][image_id])
            )
        base_features = list(map(float, features[str(image_id)]))
        roi_bytes = int(roi_row["bytes"]) if roi_row is not None else original_bytes
        roi_area = float(roi_row["area_ratio"]) if roi_row is not None else 0.0
        action_features = [
            *base_features,
            q90_bytes / max(1, original_bytes),
            roi_bytes / max(1, original_bytes),
            roi_area,
            float(available["q90"]),
            float(available["current@640"]),
        ]
        samples.append(
            {
                "image_id": image_id,
                "identity": str(image["identity"]),
                "event_id": str(image["identity"]),
                "camera": str(image["identity"]).split(":", 1)[0],
                "file_name": str(image["file_name"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "payload_bytes": payloads,
                "available": available,
                "actions": action_rows,
                "qualities": qualities,
                "action_features": action_features,
                "roi_area_ratio": roi_area,
            }
        )
    records = DATA.build_action_records(
        samples,
        actions["edge"],
        category_ids=[int(row["id"]) for row in gt["categories"]],
    )
    DATA.atomic_json(samples_path, samples)
    DATA.atomic_pickle(records_path, records)
    manifest = {
        "complete": True,
        "role": role,
        "gt": str(gt_path.resolve()),
        "action_sources": {
            "edge": str(paths["edge"].resolve()),
            "original": str(paths["original"].resolve()),
            "q90": str(paths["q90"].resolve()),
            "current@640": str(paths["roi"].resolve()),
        },
        "samples": str(samples_path.resolve()),
        "records": str(records_path.resolve()),
        "edge_model_sha256": PROTOCOL["checkpoints"]["edge"]["sha256"],
        "cloud_model_sha256": PROTOCOL["checkpoints"]["cloud"]["sha256"],
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(manifest_path, manifest)
    report = {
        "complete": True,
        "role": role,
        "images": len(samples),
        "roi_candidates": len(roi),
        "samples_sha256": DATA.file_sha256(samples_path),
        "records_sha256": DATA.file_sha256(records_path),
        "manifest_sha256": DATA.file_sha256(manifest_path),
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(completion, report)
    return report


def run_all() -> dict[str, Any]:
    reports = {role: build_role(role) for role in PROTOCOL["roles"]}
    report = {"complete": True, "roles": {key: value["images"] for key, value in reports.items()}}
    DATA.atomic_json(RUNS / "common/report.json", report)
    DATA.atomic_json(RUNS / "common/completion.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=(*PROTOCOL["roles"], "all"), default="all")
    args = parser.parse_args()
    result = run_all() if args.role == "all" else build_role(args.role)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
