#!/usr/bin/env python3
"""Refit DCSB/Adaptive Feeding on RKNN-FP16 fit records and export NumPy."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
SOURCE_ROOT = (
    REPO / "paper/experiments/20260829_dcsb_adaptive_feeding_native_baselines_v1"
)
RKNN_ROOT = (
    REPO / "paper/experiments/20260816_coco_rknn_cavs_oqr_deployment_retrain_v1"
)
RKNN_IMPORTED = RKNN_ROOT / "runs/imported/fp16"
RKNN_PROTOCOL = RKNN_ROOT / "configs/protocol.json"
BOARD_SCRIPTS = ROOT / "board_bundle/scripts"
POLICY_ROOT = ROOT / "board_bundle/policies/dcsb_adaptive_feeding"
ARTIFACTS = ROOT / "artifacts/training"
EXPECTED_RKNN_SHA256 = (
    "0915a99235c3be5723fcfeda45b06775ecccc2908cc5b490f093f23bbdede12d"
)
EXPECTED_CLOUD_SHA256 = (
    "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"
)
ROLE_COUNTS = {"fit": 9678, "calibration": 571}
TOP_K = 50


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module("dcsb_af_rknn_source", SOURCE_ROOT / "scripts/run_experiment.py")
RUNTIME = load_module("dcsb_af_rknn_numpy", BOARD_SCRIPTS / "dcsb_af_numpy.py")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def native_edge_predictions(
    candidate: Mapping[str, Any], *, category_ids: Sequence[int]
) -> list[dict[str, Any]]:
    """Map native RKNN class indices to the frozen COCO category schema."""
    categories = tuple(map(int, category_ids))
    image_id = int(candidate["image_id"])
    output = []
    for box in candidate.get("edge_boxes", ()):
        class_index = int(box["cls"])
        if not 0 <= class_index < len(categories):
            raise RuntimeError(f"RKNN class index is outside COCO: {class_index}")
        x1, y1, x2, y2 = map(float, box["xyxy"])
        if x2 <= x1 or y2 <= y1:
            continue
        output.append(
            {
                "image_id": image_id,
                "category_id": categories[class_index],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(box["conf"]),
            }
        )
    return output


def validate_candidate_alignment(
    samples: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> dict[int, Mapping[str, Any]]:
    if len(samples) != len(candidates) or len(samples) != ROLE_COUNTS.get(role, -1):
        raise RuntimeError(f"RKNN candidate population changed: {role}")
    by_id: dict[int, Mapping[str, Any]] = {}
    for candidate in candidates:
        image_id = int(candidate["image_id"])
        if image_id <= 0 or image_id in by_id:
            raise RuntimeError(f"duplicate or invalid RKNN image ID: {role}/{image_id}")
        if not isinstance(candidate.get("edge_boxes"), list):
            raise RuntimeError(f"RKNN edge boxes are missing: {role}/{image_id}")
        by_id[image_id] = candidate
    if set(by_id) != {int(sample["image_id"]) for sample in samples}:
        raise RuntimeError(f"RKNN/base image identities differ: {role}")
    for sample in samples:
        image_id = int(sample["image_id"])
        candidate = by_id[image_id]
        expected = (
            str(sample["file_name"]),
            int(sample["width"]),
            int(sample["height"]),
            int(sample["payload_bytes"]["original"]),
        )
        observed = (
            str(candidate["file_name"]),
            int(candidate["width"]),
            int(candidate["height"]),
            int(candidate["native_jpeg_bytes"]),
        )
        if observed != expected:
            raise RuntimeError(
                f"RKNN candidate identity/geometry/bytes drift: {role}/{image_id}"
            )
    return by_id


def runtime_feature_matrix(
    role: Mapping[str, Any], candidates: Mapping[int, Mapping[str, Any]]
) -> np.ndarray:
    """Build features from the exact raw boxes seen by the online scorer.

    The committed Edge result is sanitized before evaluation, but current-frame
    scoring intentionally occurs before that sanitization.  Rare zero-area RKNN
    boxes must therefore remain in the policy feature stream while staying out
    of the Edge AP labels.
    """
    rows = []
    for sample in role["samples"]:
        image_id = int(sample["image_id"])
        candidate = candidates[image_id]
        rows.append(
            RUNTIME.adaptive_feeding_feature(
                candidate["edge_boxes"],
                width=int(sample["width"]),
                height=int(sample["height"]),
                category_ids=role["category_ids"],
                top_k=TOP_K,
            )
        )
    matrix = np.asarray(rows, dtype=np.float64)
    expected = (len(role["samples"]), len(role["category_ids"]) + 5 * TOP_K)
    if matrix.shape != expected or not np.isfinite(matrix).all():
        raise RuntimeError("RKNN online feature matrix is invalid")
    return matrix


def _candidate_paths(role: str) -> tuple[Path, Path]:
    return (
        RKNN_IMPORTED / role / "edge_candidates.json",
        RKNN_IMPORTED / role / "report.json",
    )


def _rknn_role(dataset: Any, role_name: str) -> tuple[dict[str, Any], dict[int, Mapping[str, Any]]]:
    candidates_path, report_path = _candidate_paths(role_name)
    report = read_json(report_path)
    if (
        report.get("complete") is not True
        or str(report.get("backend")) != "fp16"
        or str(report.get("role")) != role_name
        or int(report.get("images", -1)) != ROLE_COUNTS[role_name]
        or str(report.get("model_sha256", "")).lower() != EXPECTED_RKNN_SHA256
    ):
        raise RuntimeError(f"RKNN candidate report changed: {role_name}")
    candidates = read_json(candidates_path)
    if not isinstance(candidates, list):
        raise RuntimeError(f"RKNN candidates are not a list: {role_name}")
    base = dataset.role(role_name)
    by_id = validate_candidate_alignment(base["samples"], candidates, role=role_name)
    edge = {
        int(sample["image_id"]): native_edge_predictions(
            by_id[int(sample["image_id"])], category_ids=base["category_ids"]
        )
        for sample in base["samples"]
    }
    return {**base, "actions": {**base["actions"], "edge": edge}}, by_id


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return BASE._classification_metrics(
        np.asarray(labels, dtype=np.int64), np.asarray(predictions, dtype=np.int64)
    )


def _source_hashes(fit: Mapping[str, Any], calibration: Mapping[str, Any]) -> dict[str, str]:
    fit_candidates, fit_report = _candidate_paths("fit")
    cal_candidates, cal_report = _candidate_paths("calibration")
    paths = {
        "train_export": SCRIPT,
        "source_protocol": SOURCE_ROOT / "configs/protocol.json",
        "source_baseline_core": SOURCE_ROOT / "scripts/baseline_core.py",
        "source_data_adapter": SOURCE_ROOT / "scripts/data_adapter.py",
        "source_runner": SOURCE_ROOT / "scripts/run_experiment.py",
        "rknn_protocol": RKNN_PROTOCOL,
        "rknn_fit_candidates": fit_candidates,
        "rknn_fit_report": fit_report,
        "rknn_calibration_candidates": cal_candidates,
        "rknn_calibration_report": cal_report,
        "fit_manifest": Path(fit["manifest_path"]),
        "fit_ground_truth": Path(fit["gt_path"]),
        "calibration_manifest": Path(calibration["manifest_path"]),
        "calibration_ground_truth": Path(calibration["gt_path"]),
        "board_numpy_runtime": BOARD_SCRIPTS / "dcsb_af_numpy.py",
    }
    return {name: file_sha256(path) for name, path in sorted(paths.items())}


def _oof_report(
    fit: Mapping[str, Any],
    features: np.ndarray,
    af_labels: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    folds = BASE._folds(len(fit["samples"]))
    all_indices = np.arange(len(fit["samples"]), dtype=np.int64)
    assignment = np.full(len(all_indices), -1, dtype=np.int64)
    af_prediction = np.zeros(len(all_indices), dtype=np.int64)
    dcsb_prediction = np.zeros(len(all_indices), dtype=np.int64)
    dcsb_target = BASE._dcsb_labels(
        fit, float(BASE.PROTOCOL["dcsb"]["detector_confidence"])
    )
    rows = []
    for fold_index, validation_indices in enumerate(folds):
        train_mask = np.ones(len(all_indices), dtype=bool)
        train_mask[validation_indices] = False
        training_indices = all_indices[train_mask]
        assignment[validation_indices] = fold_index
        af_model = BASE.CORE.fit_adaptive_feeding(
            features[training_indices],
            af_labels[training_indices],
            positive_weight=float(BASE.PROTOCOL["adaptive_feeding"]["positive_weight"]),
        )
        af_prediction[validation_indices] = af_model.predict(features[validation_indices])
        training_ids = [fit["image_ids"][int(index)] for index in training_indices]
        dcsb_model = BASE._fit_dcsb(fit, training_ids)
        validation_role = {
            **fit,
            "samples": [fit["samples"][int(index)] for index in validation_indices],
        }
        dcsb_prediction[validation_indices] = BASE._dcsb_scores(
            validation_role, dcsb_model
        )[0]
        rows.append(
            {
                "fold": fold_index,
                "train_images": len(training_indices),
                "validation_images": len(validation_indices),
                "adaptive_feeding": _classification_metrics(
                    af_labels[validation_indices], af_prediction[validation_indices]
                ),
                "dcsb": _classification_metrics(
                    dcsb_target[validation_indices], dcsb_prediction[validation_indices]
                ),
            }
        )
    if np.any(assignment < 0):
        raise RuntimeError("OOF fold assignment is incomplete")
    return (
        {
            "folds": rows,
            "adaptive_feeding": _classification_metrics(af_labels, af_prediction),
            "dcsb": _classification_metrics(dcsb_target, dcsb_prediction),
        },
        {
            "image_ids": np.asarray(fit["image_ids"], dtype=np.int64),
            "fold": assignment,
            "adaptive_feeding_label": np.asarray(af_labels, dtype=np.int64),
            "adaptive_feeding_prediction": af_prediction,
            "dcsb_label": dcsb_target,
            "dcsb_prediction": dcsb_prediction,
        },
    )


def _score_parity(
    policy: Mapping[str, Any],
    candidates: Mapping[int, Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    af_margin: np.ndarray,
    dcsb_margin: np.ndarray,
) -> dict[str, float]:
    board_af, board_dcsb = [], []
    for sample in samples:
        image_id = int(sample["image_id"])
        candidate = candidates[image_id]
        board_af.append(
            RUNTIME.method_score(
                policy,
                "adaptive_feeding_native",
                candidate["edge_boxes"],
                width=int(sample["width"]),
                height=int(sample["height"]),
            )
        )
        board_dcsb.append(
            RUNTIME.method_score(
                policy,
                "dcsb_native",
                candidate["edge_boxes"],
                width=int(sample["width"]),
                height=int(sample["height"]),
            )
        )
    differences = {
        "adaptive_feeding_max_abs": float(
            np.max(np.abs(np.asarray(board_af) - np.asarray(af_margin)))
        ),
        "dcsb_max_abs": float(
            np.max(np.abs(np.asarray(board_dcsb) - np.asarray(dcsb_margin)))
        ),
    }
    if differences["adaptive_feeding_max_abs"] > 1e-9 or differences["dcsb_max_abs"] > 1e-12:
        raise RuntimeError(f"host/board score parity failed: {differences}")
    return differences


def train_and_export() -> dict[str, Any]:
    started = time.time()
    dataset = BASE.DATA.FrozenDataset("coco")
    fit, fit_candidates = _rknn_role(dataset, "fit")
    calibration, calibration_candidates = _rknn_role(dataset, "calibration")
    if str(BASE.PROTOCOL["datasets"]["coco"]["cloud_model_sha256"]) != EXPECTED_CLOUD_SHA256:
        raise RuntimeError("cloud model hash changed")

    fit_features = runtime_feature_matrix(fit, fit_candidates)
    fit_gt, _, _ = BASE._training_targets(fit)
    af_labels, edge_mapi, cloud_mapi = BASE.CORE.adaptive_feeding_labels(
        fit["image_ids"], fit["actions"]["edge"], fit["actions"]["original"], fit_gt
    )
    af_model = BASE.CORE.fit_adaptive_feeding(
        fit_features,
        af_labels,
        positive_weight=float(BASE.PROTOCOL["adaptive_feeding"]["positive_weight"]),
    )
    dcsb_model = BASE._fit_dcsb(fit, fit["image_ids"])
    oof_report, oof_values = _oof_report(fit, fit_features, af_labels)

    calibration_features = runtime_feature_matrix(calibration, calibration_candidates)
    af_margin = np.asarray(
        af_model.decision_function(calibration_features), dtype=np.float64
    ).reshape(-1)
    _, dcsb_margin = BASE._dcsb_scores(calibration, dcsb_model)
    dcsb_margin = np.asarray(dcsb_margin, dtype=np.float64).reshape(-1)

    POLICY_ROOT.mkdir(parents=True, exist_ok=True)
    model_path = POLICY_ROOT / "adaptive_feeding.npz"
    references_path = POLICY_ROOT / "calibration_references.npz"
    atomic_npz(
        model_path,
        coefficient=np.asarray(af_model.coef_, dtype=np.float64).reshape(-1),
        intercept=np.asarray(af_model.intercept_, dtype=np.float64).reshape(-1),
    )
    atomic_npz(
        references_path,
        adaptive_feeding_margin=np.sort(af_margin),
        dcsb_margin=np.sort(dcsb_margin),
    )
    sources = _source_hashes(fit, calibration)
    source_digest = hashlib.sha256(
        json.dumps(sources, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "policy_type": RUNTIME.POLICY_TYPE,
        "policy_id": "coco_rknn_fp16_dcsb_adaptive_feeding_fit9678_cal571_v1",
        "rknn_model_sha256": EXPECTED_RKNN_SHA256,
        "cloud_model_sha256": EXPECTED_CLOUD_SHA256,
        "category_ids": list(map(int, fit["category_ids"])),
        "top_k": TOP_K,
        "feature_order": "class_histogram_then_topk_confidence_xmin_ymin_width_height",
        "policy_score_input": "raw_post_nms_rknn_boxes_before_commit_sanitization",
        "edge_label_input": "sanitized_non_degenerate_rknn_boxes",
        "adaptive_feeding": {
            "file": model_path.name,
            "sha256": file_sha256(model_path),
            "classifier": "LinearSVC",
            "C": 1.0,
            "positive_weight": 3.0,
            "native_threshold": 0.0,
        },
        "dcsb": dict(dcsb_model),
        "calibration_references": {
            "file": references_path.name,
            "sha256": file_sha256(references_path),
            "records": len(calibration["samples"]),
            "transform": "right_empirical_percentile",
            "used_by": ["dcsb_budget", "adaptive_feeding_budget"],
        },
        "source_hashes": sources,
        "source_digest": source_digest,
        "fit_images": len(fit["samples"]),
        "calibration_images": len(calibration["samples"]),
        "fit_only_model_training": True,
        "calibration_labels_used_for_model_training": False,
        "calibration_only_score_normalization": True,
        "final_loaded": False,
        "final_used_for_training_normalization_or_control": False,
        "uses_cavs": False,
        "uses_q90": False,
        "uses_roi": False,
        "uses_maur": False,
    }
    manifest_path = POLICY_ROOT / "manifest.json"
    atomic_json(manifest_path, manifest)
    policy = RUNTIME.load_policy(manifest_path)
    parity = _score_parity(
        policy,
        calibration_candidates,
        calibration["samples"],
        af_margin,
        dcsb_margin,
    )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    oof_path = ARTIFACTS / "oof_predictions.npz"
    scores_path = ARTIFACTS / "calibration_scores.npz"
    models_path = ARTIFACTS / "full_fit_models.pkl"
    atomic_npz(oof_path, **oof_values)
    atomic_npz(
        scores_path,
        image_ids=np.asarray(calibration["image_ids"], dtype=np.int64),
        original_bytes=np.asarray(
            [int(sample["payload_bytes"]["original"]) for sample in calibration["samples"]],
            dtype=np.int64,
        ),
        adaptive_feeding_margin=af_margin,
        adaptive_feeding_percentile=np.asarray(
            [RUNTIME.empirical_percentile(value, policy["adaptive_feeding_reference"]) for value in af_margin],
            dtype=np.float64,
        ),
        dcsb_margin=dcsb_margin,
        dcsb_percentile=np.asarray(
            [RUNTIME.empirical_percentile(value, policy["dcsb_reference"]) for value in dcsb_margin],
            dtype=np.float64,
        ),
    )
    with models_path.open("wb") as handle:
        pickle.dump(
            {
                "adaptive_feeding": af_model,
                "dcsb": dcsb_model,
                "fit_only_training": True,
                "final_used": False,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    report = {
        "complete": True,
        "runtime_seconds": time.time() - started,
        "fit_images": len(fit["samples"]),
        "calibration_images": len(calibration["samples"]),
        "feature_dimension": int(fit_features.shape[1]),
        "adaptive_feeding": {
            "fit_label_rate": float(np.mean(af_labels)),
            "fit_edge_mapi_mean": float(np.mean(edge_mapi)),
            "fit_cloud_mapi_mean": float(np.mean(cloud_mapi)),
            "full_fit": _classification_metrics(af_labels, af_model.predict(fit_features)),
            "n_iter": int(np.max(np.atleast_1d(af_model.n_iter_))),
        },
        "dcsb": {"model": dcsb_model},
        "oof": oof_report,
        "score_parity": parity,
        "source_digest": source_digest,
        "policy_manifest_sha256": file_sha256(manifest_path),
        "oof_sha256": file_sha256(oof_path),
        "calibration_scores_sha256": file_sha256(scores_path),
        "full_fit_models_sha256": file_sha256(models_path),
        "fit_only_model_training": True,
        "final_loaded": False,
        "final_used_for_training_normalization_or_control": False,
    }
    report_path = ARTIFACTS / "report.json"
    atomic_json(report_path, report)
    atomic_json(
        ARTIFACTS / "completion.json",
        {
            "complete": True,
            "policy_manifest_sha256": report["policy_manifest_sha256"],
            "source_digest": source_digest,
        },
    )
    return report


def main() -> None:
    print(json.dumps(train_and_export(), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
