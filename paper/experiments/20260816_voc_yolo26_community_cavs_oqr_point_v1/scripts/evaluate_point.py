#!/usr/bin/env python3
"""Run the frozen VOC point matrix for CAVS and all declared baselines."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(REPO))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DATA = load_module("voc_point_eval_data", ROOT / "scripts/data_core.py")
CORE = load_module("voc_point_eval_protocol", ROOT / "scripts/protocol_core.py")
METRIC = load_module("voc_point_eval_metric", REPO / "paper/experiments/CA_AORIC/ca_aoric.py")
load_module(
    "protocol_core",
    REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/protocol_core.py",
)
BENCHMARK = load_module(
    "voc_point_eval_benchmark",
    REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/benchmark_core.py",
)
EXTERNAL = load_module(
    "voc_point_eval_external",
    REPO / "paper/experiments/20260728_three_dataset_dcsb_oric_moric_external_baselines/scripts/baseline_core.py",
)


def _role(role: str) -> dict[str, Any]:
    root = RUNS / f"common/{role}"
    manifest = DATA.read_json(root / "source_manifest.json")
    samples = DATA.read_json(root / "samples.json")
    image_ids = [int(sample["image_id"]) for sample in samples]
    actions = {
        action: DATA.group_predictions(DATA.read_json(path), image_ids=image_ids)
        for action, path in manifest["action_sources"].items()
    }
    return {
        "role": role,
        "gt": Path(manifest["gt"]),
        "samples": samples,
        "sample_by_id": {int(sample["image_id"]): sample for sample in samples},
        "records": DATA.load_pickle(root / "records.pkl"),
        "image_ids": image_ids,
        "actions": actions,
    }


def _mixed(data: Mapping[str, Any], selection: Mapping[int, str]) -> list[dict[str, Any]]:
    threshold = float(PROTOCOL["model"]["deploy_confidence"])
    output = []
    for image_id in data["image_ids"]:
        action = str(selection.get(int(image_id), "edge"))
        if action not in data["actions"]:
            raise ValueError(f"unknown selected action: {image_id}/{action}")
        output.extend(
            row
            for row in data["actions"][action].get(int(image_id), ())
            if float(row.get("score", 1.0)) >= threshold
        )
    return output


def _evaluate(data: Mapping[str, Any], selection: Mapping[int, str]) -> dict[str, float]:
    key = str(data["role"])
    cache = getattr(_evaluate, "metric_cache", {})
    if key not in cache:
        cache[key] = METRIC.CocoMetricBackend(data["gt"])
        setattr(_evaluate, "metric_cache", cache)
    metric = cache[key]
    return metric.evaluate(_mixed(data, selection), list(map(int, data["image_ids"])))


def _shared_features(samples: Sequence[Mapping[str, Any]], *, roi: bool) -> np.ndarray:
    rows = []
    for sample in samples:
        values = list(map(float, sample["action_features"]))
        if roi:
            original = max(1.0, float(sample["payload_bytes"]["original"]))
            values.extend(
                (
                    float(sample.get("roi_area_ratio", 0.0)),
                    float(sample["payload_bytes"].get("current@640", original)) / original,
                )
            )
        rows.append(values)
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise RuntimeError("SharedAction feature matrix changed")
    return matrix


def _shared_map(
    samples: Sequence[Mapping[str, Any]],
    q90: Sequence[float],
    roi: Mapping[int, float],
    q90_threshold: float,
    roi_threshold: float,
) -> dict[int, str]:
    output = {}
    for sample, q90_probability in zip(samples, q90):
        image_id = int(sample["image_id"])
        available = sample["available"]
        base = (
            "q90"
            if bool(available.get("q90", False))
            and float(q90_probability) >= float(q90_threshold)
            else "original"
        )
        output[image_id] = (
            "current@640"
            if bool(available.get("current@640", False))
            and float(roi.get(image_id, -math.inf)) >= float(roi_threshold)
            else base
        )
    return output


def train_shared_action(roles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output = RUNS / "policies/shared_action"
    completion = output / "completion.json"
    artifact_path = output / "artifact.pkl"
    if completion.is_file():
        report = DATA.read_json(completion)
        if report["artifact_sha256"] != DATA.file_sha256(artifact_path):
            raise RuntimeError("SharedAction artifact changed")
        return DATA.load_pickle(artifact_path)
    fit = roles["fit"]
    calibration = roles["calibration"]
    epsilon = float(PROTOCOL["shared_action"]["epsilon"])
    q90_labels = np.asarray(
        [
            bool(sample["available"].get("q90", False))
            and float(sample["qualities"]["q90"])
            >= float(sample["qualities"]["original"]) - epsilon
            for sample in fit["samples"]
        ],
        dtype=np.float64,
    )
    roi_indices = [
        index
        for index, sample in enumerate(fit["samples"])
        if bool(sample["available"].get("current@640", False))
    ]
    if not roi_indices:
        raise RuntimeError("VOC has no deployable ROI actions")
    roi_labels = np.asarray(
        [
            float(fit["samples"][index]["qualities"]["current@640"])
            >= max(
                float(fit["samples"][index]["qualities"]["original"]),
                float(fit["samples"][index]["qualities"]["q90"]),
            )
            - epsilon
            for index in roi_indices
        ],
        dtype=np.float64,
    )
    q90_model = BENCHMARK.fit_weighted_logistic(
        _shared_features(fit["samples"], roi=False), q90_labels
    )
    roi_model = BENCHMARK.fit_weighted_logistic(
        _shared_features(fit["samples"], roi=True)[roi_indices], roi_labels
    )
    cal_samples = calibration["samples"]
    q90_probabilities = BENCHMARK.predict_weighted_logistic(
        q90_model, _shared_features(cal_samples, roi=False)
    )
    roi_all = BENCHMARK.predict_weighted_logistic(
        roi_model, _shared_features(cal_samples, roi=True)
    )
    roi_probabilities = {
        int(sample["image_id"]): float(value)
        for sample, value in zip(cal_samples, roi_all)
        if bool(sample["available"].get("current@640", False))
    }
    fractions = list(map(float, PROTOCOL["shared_action"]["action_fractions"]))
    q90_eligible = np.asarray(
        [bool(sample["available"].get("q90", False)) for sample in cal_samples]
    )
    q90_thresholds = sorted(
        {
            BENCHMARK.threshold_for_fraction(q90_probabilities, q90_eligible, fraction)
            for fraction in fractions
        },
        reverse=True,
    )
    original = {int(sample["image_id"]): "original" for sample in cal_samples}
    original_ap = float(_evaluate(calibration, original)["map50_95"])
    q90_candidates = []
    for threshold in q90_thresholds:
        action_map = _shared_map(
            cal_samples, q90_probabilities, {}, float(threshold), float("inf")
        )
        metric = _evaluate(calibration, action_map)
        requested = sum(
            calibration["sample_by_id"][image_id]["payload_bytes"][action]
            for image_id, action in action_map.items()
        )
        q90_candidates.append(
            {"threshold": float(threshold), "ap": float(metric["map50_95"]), "request_bytes": int(requested)}
        )
    maximum_loss = float(PROTOCOL["shared_action"]["maximum_calibration_ap_loss"])
    q90_safe = [row for row in q90_candidates if row["ap"] >= original_ap - maximum_loss]
    if not q90_safe:
        raise RuntimeError("no calibration-safe Q90 threshold")
    q90_choice = min(q90_safe, key=lambda row: (row["request_bytes"], -row["ap"], -row["threshold"]))
    base_map = _shared_map(
        cal_samples,
        q90_probabilities,
        {},
        float(q90_choice["threshold"]),
        float("inf"),
    )
    base_ap = float(_evaluate(calibration, base_map)["map50_95"])
    roi_values = np.asarray(list(roi_probabilities.values()), dtype=np.float64)
    roi_thresholds = sorted(
        {
            BENCHMARK.threshold_for_fraction(
                roi_values, np.ones(len(roi_values), dtype=bool), fraction
            )
            for fraction in fractions
        },
        reverse=True,
    )
    roi_candidates = []
    for threshold in roi_thresholds:
        action_map = _shared_map(
            cal_samples,
            q90_probabilities,
            roi_probabilities,
            float(q90_choice["threshold"]),
            float(threshold),
        )
        metric = _evaluate(calibration, action_map)
        requested = sum(
            calibration["sample_by_id"][image_id]["payload_bytes"][action]
            for image_id, action in action_map.items()
        )
        roi_candidates.append(
            {
                "threshold": float(threshold),
                "ap": float(metric["map50_95"]),
                "request_bytes": int(requested),
                "roi_count": sum(action == "current@640" for action in action_map.values()),
            }
        )
    roi_safe = [row for row in roi_candidates if row["ap"] >= base_ap - maximum_loss]
    if not roi_safe:
        raise RuntimeError("no calibration-safe ROI threshold")
    roi_choice = min(roi_safe, key=lambda row: (row["request_bytes"], -row["ap"], -row["threshold"]))
    artifact = {
        "q90_model": q90_model,
        "roi_model": roi_model,
        "q90_threshold": float(q90_choice["threshold"]),
        "roi_threshold": float(roi_choice["threshold"]),
        "q90_choice": q90_choice,
        "roi_choice": roi_choice,
        "fit_only_training": True,
        "calibration_only_thresholds": True,
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_pickle(artifact_path, artifact)
    report_q90 = dict(q90_choice)
    report_roi = dict(roi_choice)
    report_q90["threshold"] = (
        float(report_q90["threshold"])
        if math.isfinite(float(report_q90["threshold"]))
        else None
    )
    report_roi["threshold"] = (
        float(report_roi["threshold"])
        if math.isfinite(float(report_roi["threshold"]))
        else None
    )
    report = {
        "complete": True,
        "artifact_sha256": DATA.file_sha256(artifact_path),
        "q90_choice": report_q90,
        "roi_choice": report_roi,
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(completion, report)
    return artifact


def shared_action_maps(
    roles: Mapping[str, Mapping[str, Any]], artifact: Mapping[str, Any]
) -> dict[str, dict[int, str]]:
    output = {}
    for role in ("calibration", "final"):
        samples = roles[role]["samples"]
        q90 = BENCHMARK.predict_weighted_logistic(
            artifact["q90_model"], _shared_features(samples, roi=False)
        )
        roi_values = BENCHMARK.predict_weighted_logistic(
            artifact["roi_model"], _shared_features(samples, roi=True)
        )
        roi = {
            int(sample["image_id"]): float(value)
            for sample, value in zip(samples, roi_values)
            if bool(sample["available"].get("current@640", False))
        }
        output[role] = _shared_map(
            samples,
            q90,
            roi,
            float(artifact["q90_threshold"]),
            float(artifact["roi_threshold"]),
        )
    return output


def fit_dcsb(roles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output = RUNS / "policies/dcsb"
    path = output / "model.json"
    if path.is_file():
        return DATA.read_json(path)
    fit = roles["fit"]
    gt = DATA.read_json(fit["gt"])
    counts = Counter(int(row["image_id"]) for row in gt["annotations"])
    model = EXTERNAL.fit_dcsb(
        image_ids=fit["image_ids"],
        edge_predictions=fit["actions"]["edge"],
        cloud_predictions=fit["actions"]["original"],
        image_sizes={
            int(sample["image_id"]): (int(sample["width"]), int(sample["height"]))
            for sample in fit["samples"]
        },
        ground_truth_counts={image_id: int(counts[image_id]) for image_id in fit["image_ids"]},
        detector_confidence=float(PROTOCOL["dcsb"]["detector_confidence"]),
        count_thresholds=list(map(int, PROTOCOL["dcsb"]["count_thresholds"])),
        area_thresholds=np.arange(
            float(PROTOCOL["dcsb"]["area_threshold_start"]),
            float(PROTOCOL["dcsb"]["area_threshold_stop"])
            + float(PROTOCOL["dcsb"]["area_threshold_step"]) / 2.0,
            float(PROTOCOL["dcsb"]["area_threshold_step"]),
        ).tolist(),
    )
    model.update(fit_only_training=True, final_used_for_training_or_selection=False)
    DATA.atomic_json(path, model)
    return model


def _score_file(method: str, seed: int) -> Mapping[str, Any]:
    return DATA.load_pickle(RUNS / f"learners/jobs/{method}/seed_{seed}/scores.pkl")


def _scores(method: str, role: str) -> dict[str, np.ndarray]:
    output = {}
    expected = [int(sample["image_id"]) for sample in _EVAL_STATE["roles"][role]["samples"]]
    for seed in map(int, PROTOCOL["model_seeds"]):
        payload = _score_file(method, seed)[role]
        if list(map(int, payload["image_ids"])) != expected:
            raise RuntimeError(f"learner score alignment changed: {method}/{seed}/{role}")
        output[str(seed)] = np.asarray(payload["values"], dtype=np.float64)
    output["ensemble"] = np.mean(list(output.values()), axis=0)
    return output


def _action_scores(
    records: Sequence[Mapping[str, Any]], matrix: np.ndarray, allowed: set[str]
) -> dict[int, dict[str, float]]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(records):
        raise ValueError(f"action score shape changed: {values.shape}")
    output = {}
    for row, record in enumerate(records):
        output[int(record["image_id"])] = {
            str(action["action_id"]): float(values[row, column])
            for column, action in enumerate(record["actions"])
            if str(action["action_id"]) in allowed
        }
        if not output[int(record["image_id"])]:
            raise RuntimeError(f"no available scored action: {record['image_id']}")
    return output


def _joint_selection(
    samples: Sequence[Mapping[str, Any]],
    scores: Mapping[int, Mapping[str, float]],
    *,
    mode: str,
    budget: float,
) -> dict[int, str]:
    if mode == "fixed_call":
        best = []
        for sample in samples:
            image_id = int(sample["image_id"])
            candidates = [
                (float(value), -float(sample["payload_bytes"][action]), str(action))
                for action, value in scores[image_id].items()
                if action in sample["payload_bytes"]
            ]
            value, negative_cost, action = max(candidates)
            best.append((value, negative_cost, -image_id, image_id, action))
        count = DATA.fixed_call_count(len(samples), budget)
        return {
            image_id: action
            for _, _, _, image_id, action in sorted(best, reverse=True)[:count]
        }
    groups = []
    for sample in samples:
        image_id = int(sample["image_id"])
        actions = [{"image_id": image_id, "action_id": "edge", "value": 0.0, "cost": 0.0}]
        for action, value in scores[image_id].items():
            if action in sample["payload_bytes"] and bool(sample["available"].get(action, True)):
                actions.append(
                    {
                        "image_id": image_id,
                        "action_id": action,
                        "value": float(value),
                        "cost": float(sample["payload_bytes"][action]),
                    }
                )
        groups.append(actions)
    capacity = float(budget) * sum(float(sample["payload_bytes"]["original"]) for sample in samples)
    return {
        int(row["image_id"]): str(row["action_id"])
        for row in CORE.select_actions_for_capacity(groups, capacity)
        if str(row["action_id"]) != "edge"
    }


def _percentile(reference: Sequence[float], values: Sequence[float]) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(ordered, np.asarray(values, dtype=np.float64), side="right") / len(ordered)


def _scalar_selection(
    samples: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    reference: Sequence[float],
    action_map: Mapping[int, str],
    *,
    mode: str,
    budget: float,
) -> dict[int, str]:
    if mode == "fixed_call":
        count = DATA.fixed_call_count(len(samples), budget)
        order = sorted(
            range(len(samples)),
            key=lambda index: (-float(scores[index]), int(samples[index]["image_id"])),
        )[:count]
        return {
            int(samples[index]["image_id"]): str(action_map[int(samples[index]["image_id"])])
            for index in order
        }
    percentile = _percentile(reference, scores)
    score_map = {
        int(sample["image_id"]): {
            str(action_map[int(sample["image_id"])]): float(percentile[index])
        }
        for index, sample in enumerate(samples)
    }
    return _joint_selection(samples, score_map, mode=mode, budget=budget)


_EVAL_STATE: dict[str, Any] = {}


def _validate_selection(
    data: Mapping[str, Any], selection: Mapping[int, str], mode: str, budget: float
) -> dict[str, Any]:
    requested = sum(
        float(data["sample_by_id"][int(image_id)]["payload_bytes"][str(action)])
        for image_id, action in selection.items()
    )
    original_total = sum(
        float(sample["payload_bytes"]["original"]) for sample in data["samples"]
    )
    if mode == "fixed_byte":
        capacity = float(budget) * original_total
        if requested > capacity + max(1e-6, capacity * 1e-12):
            raise RuntimeError("fixed-byte capacity exceeded")
        utilization = requested / capacity if capacity else 0.0
    else:
        expected = DATA.fixed_call_count(len(data["samples"]), budget)
        if len(selection) != expected:
            raise RuntimeError("fixed-call population changed")
        capacity = None
        utilization = 1.0
    return {"request_bytes": requested, "capacity_bytes": capacity, "utilization": utilization}


def _evaluate_task(task: Mapping[str, Any]) -> dict[str, Any]:
    role = str(task["role"])
    data = _EVAL_STATE["roles"][role]
    selection = {int(key): str(value) for key, value in task["selection"].items()}
    metric = _evaluate(data, selection)
    validation = _validate_selection(data, selection, str(task["mode"]), float(task["budget"]))
    counts = Counter(selection.values())
    return {
        "role": role,
        "method": str(task["method"]),
        "policy": str(task["policy"]),
        "seed": task["seed"],
        "mode": str(task["mode"]),
        "budget": float(task["budget"]),
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "kb_per_image": validation["request_bytes"] / 1024.0 / len(data["samples"]),
        "request_bytes": validation["request_bytes"],
        "capacity_bytes": validation["capacity_bytes"],
        "capacity_utilization": validation["utilization"],
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["current@640"]),
        "bootstrap_performed": False,
        "final_used_for_training_or_selection": False,
    }


def _aggregate_random(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["role"]), str(row["mode"]), float(row["budget"]))].append(row)
    output = []
    numeric = (
        "official_ap50_95", "official_ap50", "official_ap75", "kb_per_image",
        "request_bytes", "capacity_utilization", "selected_images", "original_count",
        "q90_count", "roi_count",
    )
    for (role, mode, budget), values in sorted(grouped.items()):
        if len(values) != int(PROTOCOL["random"]["seeds"]):
            raise RuntimeError("Random-50 result matrix incomplete")
        row = {
            "role": role,
            "method": "Random",
            "policy": "SharedAction-OQR",
            "seed": "fixed",
            "mode": mode,
            "budget": budget,
            "capacity_bytes": values[0]["capacity_bytes"],
            "bootstrap_performed": False,
            "final_used_for_training_or_selection": False,
        }
        row.update({key: float(np.mean([float(value[key]) for value in values])) for key in numeric})
        row["random_seed_count"] = len(values)
        output.append(row)
    return output


def run() -> dict[str, Any]:
    output = RUNS / "point"
    completion = output / "completion.json"
    if completion.is_file():
        return DATA.read_json(output / "report.json")
    if not (RUNS / "learners/completion.json").is_file():
        raise RuntimeError("learner retraining must complete before point evaluation")
    roles = {role: _role(role) for role in PROTOCOL["roles"]}
    _EVAL_STATE.clear()
    _EVAL_STATE["roles"] = roles
    shared = train_shared_action(roles)
    action_maps = shared_action_maps(roles, shared)
    dcsb = fit_dcsb(roles)
    scalar: dict[str, dict[str, np.ndarray]] = {role: {} for role in ("calibration", "final")}
    for role in scalar:
        data = roles[role]
        scalar[role]["Confidence"] = np.asarray(
            [BENCHMARK.confidence_score(data["actions"]["edge"][image_id]) for image_id in data["image_ids"]],
            dtype=np.float64,
        )
        scalar[role]["DCSB-Score-Adapted"] = np.asarray(
            [
                EXTERNAL.dcsb_decision_and_score(
                    data["actions"]["edge"][int(sample["image_id"])],
                    int(sample["width"]),
                    int(sample["height"]),
                    dcsb,
                )[1]
                for sample in data["samples"]
            ],
            dtype=np.float64,
        )
    tasks = []
    random_tasks = []
    selections = {}
    for role in ("calibration", "final"):
        data = roles[role]
        cavs = _scores("CAVS-OQR-MAUR", role)
        for seed, values in cavs.items():
            matrices = np.asarray(values, dtype=np.float64)
            if matrices.ndim == 2:
                matrices = np.repeat(matrices[None, :, :], len(PROTOCOL["budgets"]), axis=0)
            for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
                for method, allowed, policy in (
                    ("CAVS-OQR-MAUR", {"original", "q90", "current@640"}, "Joint-OQR"),
                    ("CAVS-OQ-from-OQR", {"original", "q90"}, "Joint-OQ"),
                ):
                    score_map = _action_scores(data["records"], matrices[budget_index], allowed)
                    for mode in PROTOCOL["modes"]:
                        selection = _joint_selection(data["samples"], score_map, mode=mode, budget=budget)
                        task = {"role": role, "method": method, "policy": policy, "seed": seed, "mode": mode, "budget": budget, "selection": selection}
                        tasks.append(task)
                        selections[f"{role}|{method}|{seed}|{mode}|{budget:g}"] = selection
        for method in ("ORIC-AP5095", "MORIC-AP5095", "ORIC-AP50", "MORIC-AP50"):
            role_scores = _scores(method, role)
            calibration_scores = _scores(method, "calibration")
            for seed, values in role_scores.items():
                reference = calibration_scores[seed]
                for budget in map(float, PROTOCOL["budgets"]):
                    for mode in PROTOCOL["modes"]:
                        selection = _scalar_selection(data["samples"], values, reference, action_maps[role], mode=mode, budget=budget)
                        task = {"role": role, "method": method, "policy": "SharedAction-OQR", "seed": seed, "mode": mode, "budget": budget, "selection": selection}
                        tasks.append(task)
                        selections[f"{role}|{method}|{seed}|{mode}|{budget:g}"] = selection
        for method in ("DCSB-Score-Adapted", "Confidence"):
            for budget in map(float, PROTOCOL["budgets"]):
                for mode in PROTOCOL["modes"]:
                    selection = _scalar_selection(
                        data["samples"], scalar[role][method], scalar["calibration"][method],
                        action_maps[role], mode=mode, budget=budget,
                    )
                    task = {"role": role, "method": method, "policy": "SharedAction-OQR", "seed": "fixed", "mode": mode, "budget": budget, "selection": selection}
                    tasks.append(task)
                    selections[f"{role}|{method}|fixed|{mode}|{budget:g}"] = selection
        for offset in range(int(PROTOCOL["random"]["seeds"])):
            seed = int(PROTOCOL["random"]["base_seed"]) + offset
            values = np.random.default_rng(seed).random(len(data["samples"]))
            reference = np.random.default_rng(seed).random(len(roles["calibration"]["samples"]))
            for budget in map(float, PROTOCOL["budgets"]):
                for mode in PROTOCOL["modes"]:
                    selection = _scalar_selection(
                        data["samples"], values, reference, action_maps[role], mode=mode, budget=budget
                    )
                    random_tasks.append({"role": role, "method": "Random", "policy": "SharedAction-OQR", "seed": seed, "mode": mode, "budget": budget, "selection": selection})
    DATA.atomic_pickle(output / "selections.pkl", selections)
    workers = min(int(PROTOCOL["reward"]["workers"]), max(1, os.cpu_count() or 1))
    with mp.get_context("fork").Pool(workers) as pool:
        fixed_rows = list(pool.imap_unordered(_evaluate_task, tasks, chunksize=1))
        random_rows = list(pool.imap_unordered(_evaluate_task, random_tasks, chunksize=1))
    rows = fixed_rows + _aggregate_random(random_rows)
    CORE.validate_point_rows(rows, PROTOCOL)
    rows.sort(key=lambda row: (str(row["role"]), str(row["method"]), str(row["seed"]), str(row["mode"]), float(row["budget"])))
    DATA.atomic_csv(output / "point_results.csv", rows)
    DATA.atomic_csv(output / "random_50_raw.csv", random_rows)
    report = {
        "complete": True,
        "rows": len(rows),
        "raw_random_rows": len(random_rows),
        "workers": workers,
        "bootstrap_performed": False,
        "final_used_for_training_or_selection": False,
        "point_results_sha256": DATA.file_sha256(output / "point_results.csv"),
        "random_results_sha256": DATA.file_sha256(output / "random_50_raw.csv"),
    }
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(completion, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
