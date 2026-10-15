#!/usr/bin/env python3
"""Freeze RKNN-aligned DCSB/Adaptive Feeding controls and the 8-run matrix."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
BOARD = ROOT / "board_bundle"
POLICY = BOARD / "policies/dcsb_adaptive_feeding/manifest.json"
PROTOCOL = BOARD / "configs/protocol.json"
TRAINING = ROOT / "artifacts/training"
CALIBRATION = ROOT / "artifacts/calibration"
MATRIX = ROOT / "artifacts/FORMAL_MATRIX.json"
RKNN_IMPORTED = (
    REPO
    / "paper/experiments/20260816_coco_rknn_cavs_oqr_deployment_retrain_v1"
    / "runs/imported/fp16/calibration"
)
PROTOCOL_ID = "coco_rknn_fp16_causal_online_hil_final_v2"
EXPECTED_RKNN_SHA256 = (
    "0915a99235c3be5723fcfeda45b06775ecccc2908cc5b490f093f23bbdede12d"
)
EXPECTED_CLOUD_SHA256 = (
    "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"
)
BUDGETS = (("10", 0.10), ("25", 0.25), ("40", 0.40))
BUDGET_METHODS = ("dcsb_budget", "adaptive_feeding_budget")
NATIVE_METHODS = ("dcsb_native", "adaptive_feeding_native")
LAMBDA_GRID = tuple([index / 100.0 for index in range(201)] + [2.5, 3.0, 4.0, 5.0])

for import_root in (ROOT / "host", BOARD / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CALIBRATE = load_module("dcsb_af_calibrate", ROOT / "host/calibrate_controls.py")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _one_dimensional(name: str, values: Sequence[Any], *, length: int | None = None) -> np.ndarray:
    output = np.asarray(values).reshape(-1)
    if length is not None and len(output) != length:
        raise ValueError(f"{name} length changed")
    if not len(output):
        raise ValueError(f"{name} must not be empty")
    return output


def calibration_records_bundle(
    *,
    method: str,
    image_ids: Sequence[Any],
    original_bytes: Sequence[Any],
    raw_margin: Sequence[Any],
    percentile: Sequence[Any],
    protocol_id: str,
    rknn_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build the calibration-only records consumed by the frozen controller."""
    name = str(method)
    if name not in set(BUDGET_METHODS + NATIVE_METHODS):
        raise ValueError(f"unsupported method: {name}")
    ids = _one_dimensional("image_ids", image_ids)
    count = len(ids)
    costs = _one_dimensional("original_bytes", original_bytes, length=count)
    raw = _one_dimensional("raw_margin", raw_margin, length=count).astype(np.float64)
    ranks = _one_dimensional("percentile", percentile, length=count).astype(np.float64)
    if len(set(map(int, ids))) != count or np.any(ids.astype(np.int64) <= 0):
        raise ValueError("calibration image IDs must be positive and unique")
    if np.any(costs.astype(np.int64) <= 0):
        raise ValueError("Original payload bytes must be positive")
    if not np.isfinite(raw).all() or not np.isfinite(ranks).all():
        raise ValueError("calibration scores must be finite")
    if np.any((ranks < 0.0) | (ranks > 1.0)):
        raise ValueError("empirical percentiles must lie in [0, 1]")
    score = ranks if name in BUDGET_METHODS else raw
    records = []
    for image_id, cost, value in zip(ids, costs, score, strict=True):
        records.append(
            {
                "image_id": int(image_id),
                "payload_bytes": {"edge": 0, "original": int(cost)},
                "scores": {"edge": 0.0, "original": float(value)},
                "objective_values": {"edge": 0.0, "original": float(value)},
            }
        )
    return {
        "schema": CALIBRATE.CALIBRATION_RECORDS_SCHEMA,
        "protocol_id": str(protocol_id),
        "method": name,
        "rknn_model_sha256": str(rknn_sha256),
        "source_split": "calibration",
        "action_ids": ["edge", "original"],
        "calibration_objective": {
            "name": "empirical_percentile_sum" if name in BUDGET_METHODS else "native_margin_sum",
            "direction": "maximize",
        },
        "random_seed": None,
        "source_hashes": {str(key): str(value) for key, value in source_hashes.items()},
        "records": records,
        "score_semantics": "calibration_empirical_percentile" if name in BUDGET_METHODS else "paper_native_raw_margin",
        "native_threshold": 0.0 if name in NATIVE_METHODS else None,
        "final_used_for_tuning": False,
    }


def formal_rows(
    policy_path: str,
    policy_sha256: str,
    *,
    control_hashes: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return the predeclared causal extension matrix in fixed order."""
    hashes = dict(control_hashes or {})
    specifications = [
        ("dcsb_budget", "10", 0.10, "budget_matched"),
        ("adaptive_feeding_budget", "25", 0.25, "budget_matched"),
        ("dcsb_budget", "40", 0.40, "budget_matched"),
        ("adaptive_feeding_native", "100", 1.0, "native"),
        ("adaptive_feeding_budget", "10", 0.10, "budget_matched"),
        ("dcsb_budget", "25", 0.25, "budget_matched"),
        ("adaptive_feeding_budget", "40", 0.40, "budget_matched"),
        ("dcsb_native", "100", 1.0, "native"),
    ]
    rows: list[dict[str, Any]] = []
    for run_index, (method, label, budget, work_point) in enumerate(specifications, start=1):
        relative_control = f"controls/{method}/{label}.json"
        row = {
            "run_index": run_index,
            "method": method,
            "budget_label": label,
            "budget": budget,
            "work_point": work_point,
            "policy": str(policy_path),
            "policy_sha256": str(policy_sha256),
            "control": relative_control,
            "expected_frames": 5000,
            "zero_lookahead": True,
            "global_plan": False,
            "buffer_frames": 0,
            "upload_timing": "before_next_frame_is_opened",
            "transport": "binary_v2",
        }
        if relative_control in hashes:
            row["control_sha256"] = hashes[relative_control]
        rows.append(row)
    return rows


def _protocol(policy_sha256: str) -> dict[str, Any]:
    method = {
        "policy": "policies/dcsb_adaptive_feeding/manifest.json",
        "policy_sha256": policy_sha256,
        "actions": ["edge", "original"],
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "experiment_variant": "dcsb_adaptive_feeding_rknn_aligned_causal_online_extension_v1",
        "source_experiment": ROOT.name,
        "dataset": {
            "sequence": "sequences/coco_val2017_sequence.json",
            "sequence_file_sha256": "930cff19321c3cafb9d205f6d12d2cde7c17f74901077d099299a98a8f57dd67",
            "sequence_records_sha256": "aa36e57fb97291562937947c9cebe221c330a0f8da89d5fc865cd8eee66d058b",
            "source_annotations_sha256": "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f",
            "expected_images": 5000,
            "images_dir": "../yolo26n_coco_val/coco_val2017/images",
        },
        "models": {
            "fp16": {
                "path": "../yolo26n_coco_val/models/yolo26n-rk3588-fp16.rknn",
                "sha256": EXPECTED_RKNN_SHA256,
            },
            "cloud_sha256": EXPECTED_CLOUD_SHA256,
        },
        "methods": {
            "dcsb_budget": {**method, "score": "calibration_empirical_percentile"},
            "adaptive_feeding_budget": {**method, "score": "calibration_empirical_percentile"},
            "dcsb_native": {**method, "score": "paper_native_raw_margin", "threshold": 0.0},
            "adaptive_feeding_native": {**method, "score": "paper_native_raw_margin", "threshold": 0.0},
        },
        "budgets": [0.1, 0.25, 0.4],
        "native_work_points": True,
        "runtime": {
            "rknn_helpers_dir": "../yolo26n_coco_val",
            "core": "0",
            "cloud_timeout_seconds": 60.0,
            "request_retries": 3,
            "progress_every": 100,
            "buffer_frames": 0,
            "zero_lookahead": True,
            "global_plan": False,
        },
        "controller": {
            "id": "calibration_frozen_fixed_lambda",
            "upload": "selected payload is sent before the next image is opened",
            "controls": "controls/{method}/{budget_label}.json",
        },
        "transport": {
            "schema": "cavs_hil_binary_request_v2",
            "media_type": "application/vnd.cavs-hil.v2+octet-stream",
            "magic_ascii": "CAVSHIL2",
            "base64": False,
            "budget_accounting": "jpeg_payload_bytes",
            "secondary_accounting": "http_request_body_bytes",
        },
        "final_used_for_training_or_control": False,
        "global_plan": False,
        "zero_lookahead": True,
    }


def _read_training_inputs() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report_path = TRAINING / "report.json"
    scores_path = TRAINING / "calibration_scores.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("complete") is not True
        or report.get("final_loaded") is not False
        or report.get("final_used_for_training_normalization_or_control") is not False
        or report.get("policy_manifest_sha256") != file_sha256(POLICY)
    ):
        raise RuntimeError("training/export audit is incomplete or changed")
    with np.load(scores_path, allow_pickle=False) as archive:
        scores = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "image_ids",
        "original_bytes",
        "adaptive_feeding_margin",
        "adaptive_feeding_percentile",
        "dcsb_margin",
        "dcsb_percentile",
    }
    if set(scores) != required or len(scores["image_ids"]) != 571:
        raise RuntimeError("calibration score archive changed")
    return report, scores


def build_all() -> dict[str, Any]:
    report, scores = _read_training_inputs()
    policy_hash = file_sha256(POLICY)
    protocol = _protocol(policy_hash)
    _atomic_json(PROTOCOL, protocol)
    protocol_hash = file_sha256(PROTOCOL)
    sources = {
        "policy": policy_hash,
        "training_report": file_sha256(TRAINING / "report.json"),
        "calibration_scores": file_sha256(TRAINING / "calibration_scores.npz"),
        "rknn_calibration_candidates": file_sha256(RKNN_IMPORTED / "edge_candidates.json"),
        "rknn_calibration_report": file_sha256(RKNN_IMPORTED / "report.json"),
        "build_controls": file_sha256(SCRIPT),
    }
    control_hashes: dict[str, str] = {}
    summaries = []
    for method in BUDGET_METHODS + NATIVE_METHODS:
        prefix = "dcsb" if method.startswith("dcsb_") else "adaptive_feeding"
        bundle = calibration_records_bundle(
            method=method,
            image_ids=scores["image_ids"],
            original_bytes=scores["original_bytes"],
            raw_margin=scores[f"{prefix}_margin"],
            percentile=scores[f"{prefix}_percentile"],
            protocol_id=PROTOCOL_ID,
            rknn_sha256=EXPECTED_RKNN_SHA256,
            source_hashes=sources,
        )
        records_path = CALIBRATION / f"{method}.json"
        _atomic_json(records_path, bundle)
        work_points = BUDGETS if method in BUDGET_METHODS else (("100", 1.0),)
        grid = LAMBDA_GRID if method in BUDGET_METHODS else (0.0,)
        for label, budget in work_points:
            control = CALIBRATE.calibrate_control(
                bundle,
                budget=budget,
                lambda_grid=grid,
                calibration_records_sha256=file_sha256(records_path),
                protocol_sha256=protocol_hash,
            )
            destination = BOARD / f"controls/{method}/{label}.json"
            _atomic_json(destination, control)
            relative = destination.relative_to(BOARD).as_posix()
            control_hashes[relative] = file_sha256(destination)
            summaries.append(
                {
                    "method": method,
                    "budget_label": label,
                    "budget": budget,
                    "lambda": control["lambda"],
                    "calibration_action_counts": control["diagnostics"]["action_counts"],
                    "calibration_capacity_utilization": control["diagnostics"]["capacity_utilization"],
                    "control": relative,
                    "control_sha256": control_hashes[relative],
                }
            )
    rows = formal_rows(
        "policies/dcsb_adaptive_feeding/manifest.json",
        policy_hash,
        control_hashes=control_hashes,
    )
    if not all("control_sha256" in row for row in rows):
        raise RuntimeError("formal matrix is missing a frozen control")
    matrix = {
        "schema": "cavs_online_extension_matrix_v1",
        "protocol_id": PROTOCOL_ID,
        "experiment": ROOT.name,
        "rows": rows,
        "formal_runs": len(rows),
        "budget_matched_runs": 6,
        "native_work_point_runs": 2,
        "source_split_for_controls": "calibration",
        "final_used_for_training_normalization_or_control": False,
        "policy_sha256": policy_hash,
        "protocol_sha256": protocol_hash,
    }
    _atomic_json(MATRIX, matrix)
    _atomic_json(BOARD / "configs/formal_extension_matrix.json", matrix)
    completion = {
        "complete": True,
        "policy_sha256": policy_hash,
        "protocol_sha256": protocol_hash,
        "matrix_sha256": file_sha256(MATRIX),
        "training_source_digest": report["source_digest"],
        "controls": summaries,
        "final_loaded": False,
        "final_used_for_training_normalization_or_control": False,
    }
    _atomic_json(ROOT / "artifacts/CONTROL_COMPLETION.json", completion)
    return completion


def main() -> None:
    print(json.dumps(build_all(), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
