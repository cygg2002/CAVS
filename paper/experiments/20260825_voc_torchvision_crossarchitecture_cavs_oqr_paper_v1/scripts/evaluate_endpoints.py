#!/usr/bin/env python3
"""Audit all four frozen VOC action endpoints before router evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DATA = load_module("voc_gluoncv_endpoint_data", ROOT / "scripts/data_core.py")
METRIC = load_module("voc_gluoncv_endpoint_metric", ROOT / "scripts/voc_metric.py")


def run() -> dict[str, Any]:
    output = RUNS / "endpoint_gate"
    completion = output / "completion.json"
    report_path = output / "report.json"
    if completion.is_file():
        report = DATA.read_json(completion)
        if report.get("report_sha256") != DATA.file_sha256(report_path):
            raise RuntimeError("completed endpoint gate report changed")
        return DATA.read_json(report_path)
    gt_path = RUNS / "data/roles/final/gt.json"
    gt = DATA.read_json(gt_path)
    image_ids = [int(row["id"]) for row in gt["images"]]
    evaluator = METRIC.VocMetricBackend(gt)
    rows = {}
    paths = {
        "edge": RUNS / "actions/final/edge/predictions.json",
        "original": RUNS / "actions/final/original/predictions.json",
        "q90": RUNS / "actions/final/q90/predictions.json",
        "current@official": RUNS / "actions/final/current@official/maur_predictions.json",
    }
    edge_predictions = DATA.read_json(paths["edge"])
    edge_groups = DATA.group_predictions(edge_predictions, image_ids=image_ids)
    for action, path in paths.items():
        action_completion = RUNS / f"actions/final/{action}/completion.json"
        if not path.is_file() or not action_completion.is_file():
            raise RuntimeError(f"final endpoint cache is incomplete: {action}")
        predictions = DATA.read_json(path)
        available_count = len(image_ids)
        fallback = None
        if action == "current@official":
            manifest = DATA.read_json(
                RUNS / "actions/final/current@official/roi_manifest.json"
            )["records"]
            available = {
                int(row["image_id"])
                for row in manifest
                if bool(row.get("available", False))
            }
            roi_groups = DATA.group_predictions(predictions, image_ids=image_ids)
            predictions = DATA.flat_predictions(
                {
                    image_id: (
                        roi_groups[image_id] if image_id in available else edge_groups[image_id]
                    )
                    for image_id in image_ids
                }
            )
            available_count = len(available)
            fallback = "edge"
        metrics = evaluator.evaluate(predictions, image_ids, return_details=True)
        rows[action] = {
            **metrics,
            "predictions_sha256": DATA.file_sha256(path),
            "available_images": available_count,
            "unavailable_fallback": fallback,
        }
    if float(rows["original"]["map50_95"]) <= float(rows["edge"]["map50_95"]):
        raise RuntimeError("Cloud endpoint is not stronger than Edge on final")
    report = {
        "complete": True,
        "images": len(image_ids),
        "metric": "VOC difficult-aware; AP50-95 101-point plus VOC07 11-point AP50",
        "official_weight_claim": "COCO-trained Torchvision weights; all VOC metrics recomputed here",
        "actions": rows,
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(report_path, report)
    DATA.atomic_json(
        completion,
        {"complete": True, "report_sha256": DATA.file_sha256(report_path)},
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
