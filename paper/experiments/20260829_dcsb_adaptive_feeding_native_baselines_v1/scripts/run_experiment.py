#!/usr/bin/env python3
"""Train and evaluate DCSB and Adaptive Feeding on frozen paper caches."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
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


CORE = load_module("dcsb_af_core", ROOT / "scripts/baseline_core.py")
DATA = load_module("dcsb_af_data", ROOT / "scripts/data_adapter.py")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def atomic_pickle(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, target)


def atomic_npz(path: str | Path, **values: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, target)


def atomic_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def validate_protocol() -> None:
    if tuple(PROTOCOL["payload_actions"]) != ("edge", "original"):
        raise ValueError("external payload action space changed")
    forbidden = (
        "cavs_components_allowed",
        "shared_action_allowed",
        "q90_allowed",
        "roi_allowed",
        "maur_allowed",
    )
    if any(bool(PROTOCOL[key]) for key in forbidden):
        raise ValueError("a forbidden CAVS component was enabled")
    if not bool(PROTOCOL["fit_only_training"]):
        raise ValueError("fit-only training is required")
    if bool(PROTOCOL["calibration_used_for_model_training"]):
        raise ValueError("calibration labels cannot fit external baselines")
    if bool(PROTOCOL["final_used_for_training_or_selection"]):
        raise ValueError("final labels cannot fit or tune external baselines")
    expected = (
        2
        + 2
        + len(PROTOCOL["methods"]["budget_matched"])
        * len(PROTOCOL["modes"])
        * len(PROTOCOL["budgets"])
    ) * len(PROTOCOL["evaluation_roles"])
    if expected != int(PROTOCOL["expected_rows_per_dataset"]):
        raise ValueError(f"expected result count drifted: {expected}")


def _valid_annotations(gt: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in gt["annotations"]:
        row = dict(raw)
        output[int(row["image_id"])].append(row)
    for image in gt["images"]:
        output.setdefault(int(image["id"]), [])
    return dict(output)


def _training_targets(
    role: Mapping[str, Any],
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, int], dict[int, float]]:
    grouped = _valid_annotations(role["gt"])
    sample_by_id = {int(row["image_id"]): row for row in role["samples"]}
    counts: dict[int, int] = {}
    minimum_areas: dict[int, float] = {}
    for image_id, sample in sample_by_id.items():
        targets = [
            target
            for target in grouped.get(image_id, ())
            if not bool(target.get("difficult", target.get("ignore", False)))
            and not bool(target.get("iscrowd", False))
        ]
        counts[image_id] = len(targets)
        minimum_areas[image_id] = (
            min(
                max(0.0, float(target["bbox"][2]))
                * max(0.0, float(target["bbox"][3]))
                / max(1.0, float(sample["width"]) * float(sample["height"]))
                for target in targets
            )
            if targets
            else 0.0
        )
    return grouped, counts, minimum_areas


def _feature_matrix(role: Mapping[str, Any], top_k: int) -> np.ndarray:
    rows = []
    for sample in role["samples"]:
        image_id = int(sample["image_id"])
        rows.append(
            CORE.adaptive_feeding_feature(
                role["actions"]["edge"].get(image_id, ()),
                width=int(sample["width"]),
                height=int(sample["height"]),
                category_ids=role["category_ids"],
                top_k=int(top_k),
            )
        )
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise RuntimeError("Adaptive Feeding feature matrix is invalid")
    return matrix


def _dcsb_labels(role: Mapping[str, Any], detector_confidence: float) -> np.ndarray:
    values = []
    for sample in role["samples"]:
        image_id = int(sample["image_id"])
        edge_count = sum(
            CORE._score(row) > detector_confidence
            for row in role["actions"]["edge"].get(image_id, ())
        )
        cloud_count = sum(
            CORE._score(row) > detector_confidence
            for row in role["actions"]["original"].get(image_id, ())
        )
        values.append(int(cloud_count > edge_count))
    return np.asarray(values, dtype=np.int64)


def _fit_dcsb(role: Mapping[str, Any], image_ids: Sequence[int]) -> dict[str, Any]:
    _, counts, minimum_areas = _training_targets(role)
    samples = {int(row["image_id"]): row for row in role["samples"]}
    config = PROTOCOL["dcsb"]
    return CORE.fit_dcsb(
        image_ids=image_ids,
        edge_predictions=role["actions"]["edge"],
        cloud_predictions=role["actions"]["original"],
        image_sizes={
            image_id: (int(samples[image_id]["width"]), int(samples[image_id]["height"]))
            for image_id in map(int, image_ids)
        },
        ground_truth_counts=counts,
        ground_truth_min_areas=minimum_areas,
        detector_confidence=float(config["detector_confidence"]),
        count_thresholds=list(map(int, config["count_thresholds"])),
        area_thresholds=np.arange(
            float(config["area_threshold_start"]),
            float(config["area_threshold_stop_exclusive"]),
            float(config["area_threshold_step"]),
        ).tolist(),
    )


def _dcsb_scores(role: Mapping[str, Any], model: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    decisions, margins = [], []
    for sample in role["samples"]:
        image_id = int(sample["image_id"])
        decision, margin = CORE.dcsb_decision_and_margin(
            role["actions"]["edge"].get(image_id, ()),
            width=int(sample["width"]),
            height=int(sample["height"]),
            model=model,
        )
        decisions.append(decision)
        margins.append(margin)
    return np.asarray(decisions, dtype=np.int64), np.asarray(margins, dtype=np.float64)


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    target = np.asarray(labels, dtype=np.int64)
    estimate = np.asarray(predictions, dtype=np.int64)
    true_positive = int(np.sum((target == 1) & (estimate == 1)))
    false_positive = int(np.sum((target == 0) & (estimate == 1)))
    false_negative = int(np.sum((target == 1) & (estimate == 0)))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "accuracy": float(np.mean(target == estimate)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(1e-12, precision + recall)),
        "positive_label_rate": float(np.mean(target)),
        "positive_prediction_rate": float(np.mean(estimate)),
    }


def _folds(total: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(PROTOCOL["fold_seed"]))
    return [np.asarray(values, dtype=np.int64) for values in np.array_split(rng.permutation(total), int(PROTOCOL["fold_count"]))]


def _source_digest(dataset: DATA.FrozenDataset) -> tuple[str, dict[str, str]]:
    paths = [PROTOCOL_PATH, ROOT / "scripts/baseline_core.py", ROOT / "scripts/data_adapter.py", SCRIPT]
    paths.extend(dataset.source_paths())
    hashes = {str(path.resolve()): file_sha256(path) for path in paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()
    return digest, hashes


def run_preflight(dataset_name: str) -> dict[str, Any]:
    validate_protocol()
    output = RUNS / dataset_name / "preflight"
    dataset = DATA.FrozenDataset(dataset_name)
    expected = PROTOCOL["datasets"][dataset_name]
    populations, identities, manifests = {}, {}, {}
    expected_cloud_hash = str(expected["cloud_model_sha256"])
    for role_name in PROTOCOL["roles"]:
        role = dataset.role(role_name)
        samples = role["samples"]
        if len(samples) != int(expected["roles"][role_name]):
            raise RuntimeError(f"role population changed: {dataset_name}/{role_name}")
        ids = [int(row["image_id"]) for row in samples]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate image ID: {dataset_name}/{role_name}")
        if any(int(row["payload_bytes"]["original"]) <= 0 for row in samples):
            raise RuntimeError(f"invalid Original payload: {dataset_name}/{role_name}")
        if not set(role["actions"]) == {"edge", "original"}:
            raise RuntimeError(f"external action projection changed: {dataset_name}/{role_name}")
        if expected_cloud_hash not in json.dumps(role["manifest"], sort_keys=True):
            raise RuntimeError(f"cloud model hash not bound by manifest: {dataset_name}/{role_name}")
        populations[role_name] = len(samples)
        identities[role_name] = {
            str(row.get("identity", row["image_id"])) for row in samples
        }
        manifests[role_name] = file_sha256(role["manifest_path"])
    overlaps = {
        f"{left}_{right}": len(identities[left] & identities[right])
        for index, left in enumerate(PROTOCOL["roles"])
        for right in PROTOCOL["roles"][index + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError(f"role identity leakage: {dataset_name}/{overlaps}")
    digest, hashes = _source_digest(dataset)
    import sklearn

    report = {
        "complete": True,
        "dataset": dataset_name,
        "role_counts": populations,
        "identity_overlap": overlaps,
        "manifest_sha256": manifests,
        "source_digest": digest,
        "source_hashes": hashes,
        "cloud_model_sha256": expected_cloud_hash,
        "payload_actions": ["edge", "original"],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "source_experiments_modified": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(output / "completion.json", {"complete": True, "source_digest": digest})
    return report


def verify_preflight(dataset_name: str, dataset: DATA.FrozenDataset) -> dict[str, Any]:
    path = RUNS / dataset_name / "preflight/report.json"
    if not path.is_file():
        raise RuntimeError("preflight must complete first")
    report = read_json(path)
    digest, _ = _source_digest(dataset)
    if digest != report["source_digest"]:
        raise RuntimeError("source inputs changed after preflight")
    return report


def run_train(dataset_name: str) -> dict[str, Any]:
    output = RUNS / dataset_name / "training"
    report_path = output / "report.json"
    completion = output / "completion.json"
    if completion.is_file():
        return read_json(report_path)
    dataset = DATA.FrozenDataset(dataset_name)
    preflight = verify_preflight(dataset_name, dataset)
    fit = dataset.role("fit")
    top_k = int(PROTOCOL["datasets"][dataset_name]["top_k"])
    started = time.time()
    fit_features = _feature_matrix(fit, top_k)
    fit_gt, _, _ = _training_targets(fit)
    af_labels, edge_mapi, cloud_mapi = CORE.adaptive_feeding_labels(
        fit["image_ids"], fit["actions"]["edge"], fit["actions"]["original"], fit_gt
    )
    af_model = CORE.fit_adaptive_feeding(
        fit_features,
        af_labels,
        positive_weight=float(PROTOCOL["adaptive_feeding"]["positive_weight"]),
    )
    dcsb_model = _fit_dcsb(fit, fit["image_ids"])

    fold_values = _folds(len(fit["samples"]))
    af_oof = np.zeros(len(fit["samples"]), dtype=np.int64)
    dcsb_oof = np.zeros(len(fit["samples"]), dtype=np.int64)
    fold_assignments = np.full(len(fit["samples"]), -1, dtype=np.int64)
    fold_reports = []
    all_indices = np.arange(len(fit["samples"]), dtype=np.int64)
    dcsb_target = _dcsb_labels(fit, float(PROTOCOL["dcsb"]["detector_confidence"]))
    for fold_index, validation_indices in enumerate(fold_values):
        train_mask = np.ones(len(all_indices), dtype=bool)
        train_mask[validation_indices] = False
        training_indices = all_indices[train_mask]
        fold_assignments[validation_indices] = fold_index
        fold_af = CORE.fit_adaptive_feeding(
            fit_features[training_indices],
            af_labels[training_indices],
            positive_weight=float(PROTOCOL["adaptive_feeding"]["positive_weight"]),
        )
        af_oof[validation_indices] = fold_af.predict(fit_features[validation_indices])
        training_ids = [fit["image_ids"][int(index)] for index in training_indices]
        fold_dcsb = _fit_dcsb(fit, training_ids)
        validation_role = {
            **fit,
            "samples": [fit["samples"][int(index)] for index in validation_indices],
        }
        fold_decisions, _ = _dcsb_scores(validation_role, fold_dcsb)
        dcsb_oof[validation_indices] = fold_decisions
        fold_reports.append(
            {
                "fold": fold_index,
                "train_images": len(training_indices),
                "validation_images": len(validation_indices),
                "adaptive_feeding": _classification_metrics(
                    af_labels[validation_indices], af_oof[validation_indices]
                ),
                "dcsb": _classification_metrics(
                    dcsb_target[validation_indices], dcsb_oof[validation_indices]
                ),
            }
        )
    if np.any(fold_assignments < 0):
        raise RuntimeError("OOF fold coverage is incomplete")

    score_values: dict[str, np.ndarray] = {}
    for role_name in PROTOCOL["roles"]:
        role = dataset.role(role_name)
        features = fit_features if role_name == "fit" else _feature_matrix(role, top_k)
        af_margin = np.asarray(af_model.decision_function(features), dtype=np.float64)
        af_decision = np.asarray(af_model.predict(features), dtype=np.int64)
        dcsb_decision, dcsb_margin = _dcsb_scores(role, dcsb_model)
        score_values[f"image_ids_{role_name}"] = np.asarray(role["image_ids"], dtype=np.int64)
        score_values[f"af_margin_{role_name}"] = af_margin
        score_values[f"af_decision_{role_name}"] = af_decision
        score_values[f"dcsb_margin_{role_name}"] = dcsb_margin
        score_values[f"dcsb_decision_{role_name}"] = dcsb_decision

    artifact_path = output / "full_fit_models.pkl"
    scores_path = output / "scores.npz"
    oof_path = output / "oof_predictions.npz"
    atomic_pickle(
        artifact_path,
        {
            "adaptive_feeding": af_model,
            "dcsb": dcsb_model,
            "top_k": top_k,
            "fit_only_training": True,
            "calibration_used_for_model_training": False,
            "final_used_for_training_or_selection": False,
        },
    )
    atomic_npz(scores_path, **score_values)
    atomic_npz(
        oof_path,
        image_ids=np.asarray(fit["image_ids"], dtype=np.int64),
        fold=fold_assignments,
        af_label=af_labels,
        af_prediction=af_oof,
        dcsb_label=dcsb_target,
        dcsb_prediction=dcsb_oof,
    )
    report = {
        "complete": True,
        "dataset": dataset_name,
        "fit_images": len(fit["samples"]),
        "top_k": top_k,
        "feature_dimension": int(fit_features.shape[1]),
        "adaptive_feeding": {
            "positive_weight": float(PROTOCOL["adaptive_feeding"]["positive_weight"]),
            "fit_label_rate": float(np.mean(af_labels)),
            "fit_edge_mapi_mean": float(np.mean(edge_mapi)),
            "fit_cloud_mapi_mean": float(np.mean(cloud_mapi)),
            "full_fit": _classification_metrics(af_labels, af_model.predict(fit_features)),
            "oof": _classification_metrics(af_labels, af_oof),
            "n_iter": int(np.max(np.atleast_1d(af_model.n_iter_))),
        },
        "dcsb": {
            "model": dcsb_model,
            "oof": _classification_metrics(dcsb_target, dcsb_oof),
        },
        "folds": fold_reports,
        "runtime_seconds": time.time() - started,
        "source_digest": preflight["source_digest"],
        "artifact_sha256": file_sha256(artifact_path),
        "scores_sha256": file_sha256(scores_path),
        "oof_sha256": file_sha256(oof_path),
        "fit_only_training": True,
        "calibration_used_for_model_training": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {
            "complete": True,
            "artifact_sha256": report["artifact_sha256"],
            "scores_sha256": report["scores_sha256"],
        },
    )
    return report


def _metric_row(
    dataset: DATA.FrozenDataset,
    role_name: str,
    *,
    method: str,
    mode: str,
    budget: float,
    selection: Mapping[int, str],
    source_native: bool,
    budget_matched: bool,
) -> dict[str, Any]:
    role = dataset.role(role_name)
    metric = dataset.evaluate(role_name, selection)
    validation = CORE.validate_original_only_selection(
        role["samples"], selection, mode=mode, budget=float(budget)
    )
    return {
        "dataset": dataset.dataset,
        "role": role_name,
        "method": method,
        "mode": mode,
        "nominal_budget": float(budget),
        "official_ap50_95": metric["map50_95"],
        "official_ap50": metric["map50"],
        "official_ap75": metric["map75"],
        **({"voc07_ap50": metric["voc07_map50"], "voc12_ap50": metric["voc12_map50"]} if "voc07_map50" in metric else {}),
        "selected_images": validation["selected_images"],
        "request_bytes": validation["request_bytes"],
        "all_original_bytes": validation["all_original_bytes"],
        "kb_per_image": validation["request_bytes"] / 1024.0 / len(role["samples"]),
        "call_ratio": validation["call_ratio"],
        "byte_ratio": validation["byte_ratio"],
        "capacity": validation["capacity"],
        "capacity_utilization": validation["capacity_utilization"],
        "payload_policy": "Original-only",
        "adaptive_feeding_proposal_source": PROTOCOL["adaptive_feeding"]["proposal_source"],
        "adaptive_feeding_original_paper_proposal_source": PROTOCOL["adaptive_feeding"]["original_paper_proposal_source"],
        "source_native": bool(source_native),
        "budget_matched_adaptation": bool(budget_matched),
        "uses_cavs": False,
        "uses_shared_action": False,
        "uses_q90": False,
        "uses_roi": False,
        "uses_maur": False,
        "fit_only_training": True,
        "final_used_for_training_or_selection": False,
    }


def _audit_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    expected_count = int(PROTOCOL["expected_rows_per_dataset"])
    if len(rows) != expected_count:
        raise RuntimeError(f"point matrix incomplete: {len(rows)} != {expected_count}")
    keys = [
        (str(row["role"]), str(row["method"]), str(row["mode"]), float(row["nominal_budget"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("point matrix contains duplicate rows")
    if any(str(row["payload_policy"]) != "Original-only" for row in rows):
        raise RuntimeError("a non-Original external payload was evaluated")
    if any(
        bool(row[key])
        for row in rows
        for key in ("uses_cavs", "uses_shared_action", "uses_q90", "uses_roi", "uses_maur")
    ):
        raise RuntimeError("a CAVS component leaked into the external baseline matrix")
    if any(float(row["capacity_utilization"]) > 1.0 + 1e-12 for row in rows):
        raise RuntimeError("a result exceeded its capacity")


def run_point(dataset_name: str) -> dict[str, Any]:
    output = RUNS / dataset_name / "point"
    report_path = output / "report.json"
    completion = output / "completion.json"
    if completion.is_file():
        return read_json(report_path)
    dataset = DATA.FrozenDataset(dataset_name)
    preflight = verify_preflight(dataset_name, dataset)
    training = read_json(RUNS / dataset_name / "training/report.json")
    scores_path = RUNS / dataset_name / "training/scores.npz"
    if not scores_path.is_file() or file_sha256(scores_path) != training["scores_sha256"]:
        raise RuntimeError("frozen external scores are missing or changed")
    started = time.time()
    rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, str]] = {}
    with np.load(scores_path, allow_pickle=False) as scores:
        for role_name in PROTOCOL["evaluation_roles"]:
            role = dataset.role(role_name)
            expected_ids = list(map(int, role["image_ids"]))
            if list(map(int, scores[f"image_ids_{role_name}"])) != expected_ids:
                raise RuntimeError(f"score alignment changed: {dataset_name}/{role_name}")

            references = {
                "Edge-only": {},
                "Cloud-only": {image_id: "original" for image_id in expected_ids},
            }
            for method, selection in references.items():
                key = f"{role_name}|{method}|reference|-1"
                selections[key] = {str(image_id): action for image_id, action in selection.items()}
                rows.append(
                    _metric_row(
                        dataset,
                        role_name,
                        method=method,
                        mode="reference",
                        budget=-1.0,
                        selection=selection,
                        source_native=True,
                        budget_matched=False,
                    )
                )

            native_specs = (
                ("DCSB-Native", np.asarray(scores[f"dcsb_decision_{role_name}"], dtype=np.int64)),
                ("AdaptiveFeeding-Native-w3", np.asarray(scores[f"af_decision_{role_name}"], dtype=np.int64)),
            )
            for method, decisions in native_specs:
                selection = {
                    image_id: "original"
                    for image_id, decision in zip(expected_ids, decisions)
                    if int(decision) == 1
                }
                key = f"{role_name}|{method}|native|-1"
                selections[key] = {str(image_id): action for image_id, action in selection.items()}
                rows.append(
                    _metric_row(
                        dataset,
                        role_name,
                        method=method,
                        mode="native",
                        budget=-1.0,
                        selection=selection,
                        source_native=True,
                        budget_matched=False,
                    )
                )

            adapted_specs = (
                ("DCSB-Margin-BudgetMatched", np.asarray(scores[f"dcsb_margin_{role_name}"], dtype=np.float64)),
                ("AdaptiveFeeding-Margin-BudgetMatched", np.asarray(scores[f"af_margin_{role_name}"], dtype=np.float64)),
            )
            for method, margins in adapted_specs:
                for mode in PROTOCOL["modes"]:
                    for budget in map(float, PROTOCOL["budgets"]):
                        selection = CORE.select_original_by_score(
                            role["samples"], margins, mode=mode, budget=budget
                        )
                        key = f"{role_name}|{method}|{mode}|{budget:g}"
                        selections[key] = {
                            str(image_id): action for image_id, action in selection.items()
                        }
                        rows.append(
                            _metric_row(
                                dataset,
                                role_name,
                                method=method,
                                mode=mode,
                                budget=budget,
                                selection=selection,
                                source_native=False,
                                budget_matched=True,
                            )
                        )
    _audit_rows(rows)
    rows.sort(
        key=lambda row: (
            str(row["role"]),
            str(row["method"]),
            str(row["mode"]),
            float(row["nominal_budget"]),
        )
    )
    results_path = output / "results.csv"
    selections_path = output / "selections.json"
    atomic_csv(results_path, rows)
    atomic_json(selections_path, selections)
    report = {
        "complete": True,
        "dataset": dataset_name,
        "rows": len(rows),
        "runtime_seconds": time.time() - started,
        "source_digest": preflight["source_digest"],
        "training_scores_sha256": training["scores_sha256"],
        "results_sha256": file_sha256(results_path),
        "selections_sha256": file_sha256(selections_path),
        "native_methods": PROTOCOL["methods"]["native"],
        "budget_matched_methods": PROTOCOL["methods"]["budget_matched"],
        "payload_policy": "Original-only",
        "source_methods_and_budget_adaptations_separated": True,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {"complete": True, "results_sha256": report["results_sha256"]},
    )
    return report


def run(dataset_name: str, stage: str) -> dict[str, Any]:
    if dataset_name not in PROTOCOL["datasets"]:
        raise ValueError(dataset_name)
    if stage == "preflight":
        return run_preflight(dataset_name)
    if stage == "train":
        if not (RUNS / dataset_name / "preflight/completion.json").is_file():
            run_preflight(dataset_name)
        return run_train(dataset_name)
    if stage == "point":
        if not (RUNS / dataset_name / "training/completion.json").is_file():
            raise RuntimeError("training must complete before point evaluation")
        return run_point(dataset_name)
    if stage != "all":
        raise ValueError(stage)
    run_preflight(dataset_name)
    run_train(dataset_name)
    return run_point(dataset_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=tuple(PROTOCOL["datasets"]))
    parser.add_argument("--stage", required=True, choices=("preflight", "train", "point", "all"))
    args = parser.parse_args()
    result = run(str(args.dataset), str(args.stage))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
