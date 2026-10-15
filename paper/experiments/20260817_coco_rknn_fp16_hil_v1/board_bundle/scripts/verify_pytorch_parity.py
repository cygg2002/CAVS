#!/usr/bin/env python3
"""Mac/Cloud-only numerical parity check between frozen PyTorch and NumPy CAVS."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

from cavs_numpy import current_roi_xyxy, load_policy, predict_scores


ROOT = Path(__file__).resolve().parents[1]


def reference_module(path: Path):
    spec = importlib.util.spec_from_file_location("cavs_pytorch_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_reference_identity(source, manifest) -> None:
    edge_model = manifest.get("edge_model", {})
    model_sha256 = str(manifest.get("rknn_model_sha256", ""))
    if (
        manifest.get("backend") != "fp16"
        or not isinstance(edge_model, dict)
        or str(edge_model.get("sha256", "")) != model_sha256
    ):
        raise ValueError("deployment policy identity mismatch")
    source.EXPECTED_EDGE_MODEL = str(edge_model.get("name", ""))
    source.EXPECTED_EDGE_MODEL_SHA256 = model_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="current_maur_uploader.py from this checkout")
    parser.add_argument("--cases", type=int, default=12)
    args = parser.parse_args()
    source = reference_module(args.reference.resolve())
    manifest_path = ROOT / "policy" / "manifest.json"
    numpy_policy = load_policy(manifest_path)
    configure_reference_identity(source, numpy_policy)
    torch_policy = source.load_current_maur_policy(manifest_path)
    rng, maximum_error = np.random.default_rng(20260816), 0.0
    for case in range(args.cases):
        width, height = int(rng.integers(400, 1600)), int(rng.integers(300, 1200))
        boxes = []
        for _ in range(int(rng.integers(0, 90))):
            x1, y1 = rng.uniform(0, width * 0.7), rng.uniform(0, height * 0.7)
            x2, y2 = rng.uniform(x1 + 1, width), rng.uniform(y1 + 1, height)
            boxes.append({"xyxy": [float(x1), float(y1), float(x2), float(y2)], "conf": float(rng.uniform(0, 1)), "cls": int(rng.integers(0, 80))})
        roi = current_roi_xyxy(width, height, boxes, confidence=0.25, max_area_ratio=0.65)
        reference = source.predict_current_maur_scores(torch_policy, boxes, width=width, height=height, roi_xyxy=roi)
        actual = predict_scores(numpy_policy, boxes, width=width, height=height, roi_xyxy=roi)
        if set(reference) != set(actual):
            raise AssertionError(f"action availability mismatch in case {case}: {reference.keys()} != {actual.keys()}")
        maximum_error = max(maximum_error, *(abs(float(reference[action]) - actual[action]) for action in reference))
    if maximum_error > 1e-5:
        raise AssertionError(f"NumPy/PyTorch score mismatch: {maximum_error:.8g} > 1e-5")
    print(f"PASS cases={args.cases} max_abs_error={maximum_error:.8g}")


if __name__ == "__main__":
    main()
