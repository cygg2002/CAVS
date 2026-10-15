#!/usr/bin/env python3
"""Evaluate frozen VOC detector endpoints before any router result is produced."""

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


def published_ap_drift(published: float, observed: float) -> float:
    return float(observed) - float(published)


def require_published_ap_consistency(
    published: float, observed: float, *, tolerance: float
) -> None:
    drift = published_ap_drift(published, observed)
    if abs(drift) > float(tolerance):
        raise RuntimeError(
            f"official VOC07 AP50 gate failed: observed={observed:.6f}, "
            f"published={published:.6f}, drift={drift:+.6f}"
        )


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
    for action in ("edge", "original", "q90"):
        path = RUNS / f"actions/final/{action}/predictions.json"
        action_completion = RUNS / f"actions/final/{action}/completion.json"
        if not path.is_file() or not action_completion.is_file():
            raise RuntimeError(f"final endpoint cache is incomplete: {action}")
        metrics = evaluator.evaluate(DATA.read_json(path), image_ids, return_details=True)
        rows[action] = {
            **metrics,
            "predictions_sha256": DATA.file_sha256(path),
        }
    tolerance = 0.02
    for action, role in (("edge", "edge"), ("original", "cloud")):
        published = float(PROTOCOL["detectors"][role]["published_voc07_ap50_no_flip"] if role == "edge" else PROTOCOL["detectors"][role]["published_voc07_ap50"])
        observed = float(rows[action]["voc07_map50"])
        rows[action]["published_voc07_ap50"] = published
        rows[action]["published_ap_drift"] = published_ap_drift(published, observed)
        require_published_ap_consistency(published, observed, tolerance=tolerance)
    report = {
        "complete": True,
        "images": len(image_ids),
        "metric": "VOC difficult-aware; AP50-95 101-point plus VOC07 11-point AP50",
        "published_ap_tolerance": tolerance,
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
