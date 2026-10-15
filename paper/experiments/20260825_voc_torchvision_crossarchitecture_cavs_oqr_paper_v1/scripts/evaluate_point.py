#!/usr/bin/env python3
"""Evaluate the complete frozen VOC CAVS/baseline point-result matrix."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
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
DEFAULT_POINT_WORKERS = 4
POINT_TASKS_PER_CHILD = 8


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("voc_gluoncv_point_core", ROOT / "scripts/core.py")
DATA = load_module("voc_gluoncv_point_data", ROOT / "scripts/data_core.py")
METRIC = load_module("voc_gluoncv_point_metric", ROOT / "scripts/voc_metric.py")


def _sources() -> dict[str, Any]:
    if hasattr(_sources, "value"):
        return getattr(_sources, "value")
    load_module(
        "protocol_core",
        REPO
        / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/protocol_core.py",
    )
    result = {
        "benchmark": load_module(
            "voc_gluoncv_point_benchmark",
            REPO
            / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/benchmark_core.py",
        ),
        "external": load_module(
            "voc_gluoncv_point_external",
            REPO
            / "paper/experiments/20260728_three_dataset_dcsb_oric_moric_external_baselines/scripts/baseline_core.py",
        ),
        "adapter": load_module(
            "voc_gluoncv_point_adapter", ROOT / "scripts/training_adapter.py"
        ),
    }
    setattr(_sources, "value", result)
    return result


def _role(role: str) -> dict[str, Any]:
    common = RUNS / f"common/{role}"
    manifest = DATA.read_json(common / "source_manifest.json")
    samples = DATA.read_json(common / "samples.json")
    image_ids = [int(row["image_id"]) for row in samples]
    actions = {
        action: DATA.group_predictions(DATA.read_json(path), image_ids=image_ids)
        for action, path in manifest["actions"].items()
    }
    return {
        "role": role,
        "gt": Path(manifest["gt"]),
        "samples": samples,
        "sample_by_id": {int(row["image_id"]): row for row in samples},
        "records": {
            action_space: DATA.load_pickle(common / f"records_{action_space}.pkl")
            for action_space in ("O", "OQ", "OQR")
        },
        "image_ids": image_ids,
        "actions": actions,
    }


def mixed_predictions(
    data: Mapping[str, Any], selection: Mapping[int, str]
) -> list[dict[str, Any]]:
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
        cache[key] = METRIC.VocMetricBackend(data["gt"])
        setattr(_evaluate, "metric_cache", cache)
    return cache[key].evaluate(mixed_predictions(data, selection), data["image_ids"])


def _shared_features(data: Mapping[str, Any]) -> np.ndarray:
    gt = DATA.read_json(data["gt"])
    return _sources()["adapter"].external_features(
        data["records"]["O"],
        [int(row["id"]) for row in gt["categories"]],
        PROTOCOL,
    )


def _shared_map(
    samples: Sequence[Mapping[str, Any]], probabilities: Sequence[float], threshold: float
) -> dict[int, str]:
    return {
        int(sample["image_id"]): (
            "q90"
            if bool(sample["available"].get("q90", False))
            and float(probability) >= float(threshold)
            else "original"
        )
        for sample, probability in zip(samples, probabilities)
    }


def train_shared_action(roles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output = RUNS / "policies/shared_action_oq"
    completion = output / "completion.json"
    artifact_path = output / "artifact.pkl"
    if completion.is_file():
        report = DATA.read_json(completion)
        if report.get("artifact_sha256") != DATA.file_sha256(artifact_path):
            raise RuntimeError("completed SharedAction-OQ artifact changed")
        return DATA.load_pickle(artifact_path)
    benchmark = _sources()["benchmark"]
    fit, calibration = roles["fit"], roles["calibration"]
    fit_metric = DATA.load_pickle(RUNS / "rewards/fit/metric_map.pkl")
    epsilon = float(PROTOCOL["shared_action"]["epsilon"])
    labels = np.asarray(
        [
            bool(sample["available"].get("q90", False))
            and float(fit_metric[int(sample["image_id"])]["q90"]["ap50_95"])
            >= float(fit_metric[int(sample["image_id"])]["original"]["ap50_95"])
            - epsilon
            for sample in fit["samples"]
        ],
        dtype=np.float64,
    )
    model = benchmark.fit_weighted_logistic(_shared_features(fit), labels)
    probabilities = benchmark.predict_weighted_logistic(
        model, _shared_features(calibration)
    )
    eligible = np.asarray(
        [bool(sample["available"].get("q90", False)) for sample in calibration["samples"]]
    )
    thresholds = sorted(
        {
            benchmark.threshold_for_fraction(probabilities, eligible, float(fraction))
            for fraction in PROTOCOL["shared_action"]["action_fractions"]
        },
        reverse=True,
    )
    original = {int(row["image_id"]): "original" for row in calibration["samples"]}
    original_ap = float(_evaluate(calibration, original)["map50_95"])
    candidates = []
    for threshold in thresholds:
        action_map = _shared_map(calibration["samples"], probabilities, float(threshold))
        metric = _evaluate(calibration, action_map)
        requested = sum(
            int(calibration["sample_by_id"][image_id]["payload_bytes"][action])
            for image_id, action in action_map.items()
        )
        candidates.append(
            {
                "threshold": float(threshold),
                "ap50_95": float(metric["map50_95"]),
                "request_bytes": int(requested),
                "q90_count": sum(action == "q90" for action in action_map.values()),
            }
        )
    maximum_loss = float(PROTOCOL["shared_action"]["maximum_calibration_ap_loss"])
    safe = [row for row in candidates if row["ap50_95"] >= original_ap - maximum_loss]
    if not safe:
        raise RuntimeError("no calibration-safe SharedAction-Q90 threshold")
    choice = min(
        safe,
        key=lambda row: (row["request_bytes"], -row["ap50_95"], -row["threshold"]),
    )
    artifact = {
        "model": model,
        "threshold": float(choice["threshold"]),
        "choice": choice,
        "fit_only_training": True,
        "calibration_only_threshold": True,
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_pickle(artifact_path, artifact)
    report_choice = dict(choice)
    if not math.isfinite(float(report_choice["threshold"])):
        report_choice["threshold"] = None
    report = {
        "complete": True,
        "artifact_sha256": DATA.file_sha256(artifact_path),
        "choice": report_choice,
        "original_calibration_ap50_95": original_ap,
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(completion, report)
    return artifact


def shared_action_maps(
    roles: Mapping[str, Mapping[str, Any]], artifact: Mapping[str, Any]
) -> dict[str, dict[int, str]]:
    benchmark = _sources()["benchmark"]
    output = {}
    for role in ("calibration", "final"):
        data = roles[role]
        probabilities = benchmark.predict_weighted_logistic(
            artifact["model"], _shared_features(data)
        )
        output[role] = _shared_map(
            data["samples"], probabilities, float(artifact["threshold"])
        )
    return output


def fit_dcsb(roles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output = RUNS / "policies/dcsb"
    path = output / "model.json"
    if path.is_file():
        return DATA.read_json(path)
    fit = roles["fit"]
    gt = DATA.read_json(fit["gt"])
    counts = Counter(
        int(row["image_id"])
        for row in gt["annotations"]
        if not bool(row.get("difficult", row.get("ignore", False)))
    )
    external = _sources()["external"]
    model = external.fit_dcsb(
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


def _score_path(method: str, seed: int) -> Path:
    return (
        RUNS
        / f"learners/jobs/{method}/seed_{int(seed)}/fold_full/scores.pkl"
    )


def _scores(method: str, role: str, expected_ids: Sequence[int]) -> dict[str, np.ndarray]:
    output = {}
    for seed in map(int, PROTOCOL["model_seeds"]):
        payload = DATA.load_pickle(_score_path(method, seed))[role]
        if list(map(int, payload["image_ids"])) != list(map(int, expected_ids)):
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
    return output


def joint_selection(
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
                and bool(sample["available"].get(action, True))
            ]
            if not candidates:
                continue
            value, negative_cost, action = max(candidates)
            best.append((value, negative_cost, -image_id, image_id, action))
        count = CORE.fixed_call_count(len(samples), budget)
        if len(best) < count:
            raise RuntimeError("not enough cloud actions for fixed-call selection")
        return {
            image_id: action
            for _, _, _, image_id, action in sorted(best, reverse=True)[:count]
        }
    if mode != "fixed_byte":
        raise ValueError(mode)
    groups = []
    for sample in samples:
        image_id = int(sample["image_id"])
        actions = [
            {"image_id": image_id, "action_id": "edge", "value": 0.0, "cost": 0.0}
        ]
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
    capacity = math.floor(
        float(budget)
        * sum(float(sample["payload_bytes"]["original"]) for sample in samples)
    )
    return {
        int(row["image_id"]): str(row["action_id"])
        for row in CORE.select_actions_for_capacity(groups, capacity)
        if str(row["action_id"]) != "edge"
    }


def _percentile(reference: Sequence[float], values: Sequence[float]) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(ordered, np.asarray(values, dtype=np.float64), side="right") / len(ordered)


def scalar_selection(
    samples: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    reference: Sequence[float],
    action_map: Mapping[int, str],
    *,
    mode: str,
    budget: float,
) -> dict[int, str]:
    if mode == "fixed_call":
        count = CORE.fixed_call_count(len(samples), budget)
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
    return joint_selection(samples, score_map, mode=mode, budget=budget)


def _validate_selection(
    data: Mapping[str, Any], selection: Mapping[int, str], mode: str, budget: float
) -> dict[str, Any]:
    requested = sum(
        int(data["sample_by_id"][int(image_id)]["payload_bytes"][str(action)])
        for image_id, action in selection.items()
    )
    if mode == "fixed_byte":
        capacity = math.floor(
            float(budget)
            * sum(int(row["payload_bytes"]["original"]) for row in data["samples"])
        )
        if requested > capacity:
            raise RuntimeError(f"fixed-byte capacity exceeded: {requested}>{capacity}")
        utilization = requested / capacity if capacity else 0.0
    elif mode == "fixed_call":
        expected = CORE.fixed_call_count(len(data["samples"]), budget)
        if len(selection) != expected:
            raise RuntimeError(f"fixed-call count changed: {len(selection)} != {expected}")
        capacity = expected
        utilization = 1.0
    else:
        raise ValueError(mode)
    return {
        "request_bytes": requested,
        "capacity_bytes": capacity,
        "capacity_utilization": utilization,
    }


_EVAL_STATE: dict[str, Any] = {}


def _point_roles_to_load() -> tuple[str, ...]:
    shared_ready = all(
        path.is_file()
        for path in (
            RUNS / "policies/shared_action_oq/completion.json",
            RUNS / "policies/shared_action_oq/artifact.pkl",
        )
    )
    dcsb_ready = (RUNS / "policies/dcsb/model.json").is_file()
    required = {"calibration", "final"}
    if not (shared_ready and dcsb_ready):
        required.add("fit")
    return tuple(role for role in PROTOCOL["roles"] if role in required)


def _point_worker_count(configured: int, task_count: int) -> int:
    raw = os.environ.get("VOC_POINT_WORKERS", str(DEFAULT_POINT_WORKERS))
    try:
        requested = int(raw)
    except ValueError as error:
        raise ValueError(f"VOC_POINT_WORKERS must be an integer, got {raw!r}") from error
    if requested < 1:
        raise ValueError("VOC_POINT_WORKERS must be positive")
    return min(
        int(configured), requested, int(task_count), max(1, os.cpu_count() or 1)
    )


def _point_job_path(task: Mapping[str, Any]) -> Path:
    identity = {
        key: task[key]
        for key in ("role", "method", "policy", "seed", "mode", "budget")
    }
    token = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RUNS / f"point/jobs/{token}.json"


def _row_matches_task(row: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    return all(
        row.get(key) == task.get(key)
        for key in ("role", "method", "policy", "seed", "mode", "budget")
    )


def _point_task_priority(task: Mapping[str, Any]) -> tuple[Any, ...]:
    main_order = {
        "CAVS-OQR-Core": 0,
        "CAVS-OQ-Core": 1,
        "CAVS-O-Core": 2,
    }
    is_main_point = (
        str(task["role"]) == "final"
        and str(task["method"]) in main_order
        and str(task["seed"]) == "ensemble"
        and str(task["mode"]) == "fixed_byte"
    )
    return (
        0 if is_main_point else 1,
        main_order.get(str(task["method"]), len(main_order)),
        float(task["budget"]),
        str(task["role"]),
        str(task["method"]),
        str(task["policy"]),
        str(task["seed"]),
        str(task["mode"]),
    )


def _evaluate_task(task: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = _point_job_path(task)
    if checkpoint.is_file():
        row = DATA.read_json(checkpoint)
        if not _row_matches_task(row, task):
            raise RuntimeError(f"point checkpoint identity changed: {checkpoint}")
        return row
    data = _EVAL_STATE["roles"][str(task["role"])]
    selection = _EVAL_STATE["task_selections"][int(task["selection_index"])]
    metric = _evaluate(data, selection)
    validation = _validate_selection(
        data, selection, str(task["mode"]), float(task["budget"])
    )
    counts = Counter(selection.values())
    row = {
        "role": str(task["role"]),
        "method": str(task["method"]),
        "policy": str(task["policy"]),
        "seed": task["seed"],
        "mode": str(task["mode"]),
        "budget": float(task["budget"]),
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "voc07_ap50": float(metric["voc07_map50"]),
        "voc12_ap50": float(metric["voc12_map50"]),
        "request_bytes": int(validation["request_bytes"]),
        "capacity_bytes": int(validation["capacity_bytes"]),
        "capacity_utilization": float(validation["capacity_utilization"]),
        "kb_per_image": validation["request_bytes"] / 1024.0 / len(data["samples"]),
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "current_official_count": int(counts["current@official"]),
        "bootstrap_performed": False,
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(checkpoint, row)
    return row


def _aggregate_random(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["role"]), str(row["policy"]), str(row["mode"]), float(row["budget"]))].append(row)
    output = []
    numeric = (
        "official_ap50_95",
        "official_ap50",
        "official_ap75",
        "voc07_ap50",
        "voc12_ap50",
        "request_bytes",
        "capacity_utilization",
        "kb_per_image",
        "selected_images",
        "original_count",
        "q90_count",
        "current_official_count",
    )
    for (role, policy, mode, budget), values in sorted(grouped.items()):
        if len(values) != int(PROTOCOL["random"]["seeds"]):
            raise RuntimeError("Random-50 point matrix incomplete")
        row = {
            "role": role,
            "method": "Random",
            "policy": policy,
            "seed": "mean50",
            "mode": mode,
            "budget": budget,
            "capacity_bytes": int(values[0]["capacity_bytes"]),
            "bootstrap_performed": False,
            "final_used_for_training_or_selection": False,
            "random_seed_count": len(values),
        }
        row.update(
            {
                key: float(np.mean([float(value[key]) for value in values]))
                for key in numeric
            }
        )
        output.append(row)
    return output


def _selection_key(
    role: str, method: str, policy: str, seed: Any, mode: str, budget: float
) -> str:
    return f"{role}|{method}|{policy}|seed={seed}|{mode}|{float(budget):g}"


def run() -> dict[str, Any]:
    output = RUNS / "point"
    completion = output / "completion.json"
    if completion.is_file():
        return DATA.read_json(output / "report.json")
    if not (RUNS / "learners/completion.json").is_file():
        raise RuntimeError("learner retraining must complete before point evaluation")
    roles = {role: _role(role) for role in _point_roles_to_load()}
    shared_artifact = train_shared_action(roles)
    shared_maps = shared_action_maps(roles, shared_artifact)
    original_maps = {
        role: {int(row["image_id"]): "original" for row in roles[role]["samples"]}
        for role in ("calibration", "final")
    }
    policy_maps = {
        "Original-only": original_maps,
        "SharedAction-OQ": shared_maps,
    }
    dcsb = fit_dcsb(roles)
    benchmark, external = _sources()["benchmark"], _sources()["external"]
    scalar: dict[str, dict[str, np.ndarray]] = {role: {} for role in ("calibration", "final")}
    for role in scalar:
        data = roles[role]
        scalar[role]["Confidence"] = np.asarray(
            [benchmark.confidence_score(data["actions"]["edge"][image_id]) for image_id in data["image_ids"]],
            dtype=np.float64,
        )
        scalar[role]["DCSB-Score-Adapted"] = np.asarray(
            [
                external.dcsb_decision_and_score(
                    data["actions"]["edge"][int(sample["image_id"])],
                    int(sample["width"]),
                    int(sample["height"]),
                    dcsb,
                )[1]
                for sample in data["samples"]
            ],
            dtype=np.float64,
        )
    tasks: list[dict[str, Any]] = []
    random_tasks: list[dict[str, Any]] = []
    task_selections: list[dict[int, str]] = []
    stored: dict[str, dict[int, str]] = {}
    random_stored: dict[str, dict[int, str]] = {}

    def add(
        target: list[dict[str, Any]],
        *,
        role: str,
        method: str,
        policy: str,
        seed: Any,
        mode: str,
        budget: float,
        selection: Mapping[int, str],
        random: bool = False,
    ) -> None:
        frozen = {int(key): str(value) for key, value in selection.items()}
        selection_index = len(task_selections)
        task_selections.append(frozen)
        target.append(
            {
                "role": role,
                "method": method,
                "policy": policy,
                "seed": seed,
                "mode": mode,
                "budget": float(budget),
                "selection_index": selection_index,
            }
        )
        key = _selection_key(role, method, policy, seed, mode, budget)
        (random_stored if random else stored)[key] = frozen

    for role in ("calibration", "final"):
        data = roles[role]
        for method, action_space, policy, allowed in (
            ("CAVS-O-Core", "O", "Joint-O", {"original"}),
            ("CAVS-OQ-Core", "OQ", "Joint-OQ", {"original", "q90"}),
            (
                "CAVS-OQR-Core",
                "OQR",
                "Joint-OQR",
                {"original", "q90", "current@official"},
            ),
        ):
            method_scores = _scores(method, role, data["image_ids"])
            for seed, matrices in method_scores.items():
                values = np.asarray(matrices, dtype=np.float64)
                if values.ndim != 3 or values.shape[0] != len(PROTOCOL["budgets"]):
                    raise RuntimeError(f"CAVS score shape changed: {method}/{seed}/{values.shape}")
                for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
                    score_map = _action_scores(
                        data["records"][action_space], values[budget_index], allowed
                    )
                    for mode in PROTOCOL["modes"]:
                        add(
                            tasks,
                            role=role,
                            method=method,
                            policy=policy,
                            seed=seed,
                            mode=str(mode),
                            budget=budget,
                            selection=joint_selection(
                                data["samples"], score_map, mode=str(mode), budget=budget
                            ),
                        )
        for method in CORE.EXTERNAL_METHODS:
            role_scores = _scores(method, role, data["image_ids"])
            calibration_scores = _scores(
                method, "calibration", roles["calibration"]["image_ids"]
            )
            for seed, values in role_scores.items():
                for policy in ("Original-only", "SharedAction-OQ"):
                    for budget in map(float, PROTOCOL["budgets"]):
                        for mode in PROTOCOL["modes"]:
                            add(
                                tasks,
                                role=role,
                                method=method,
                                policy=policy,
                                seed=seed,
                                mode=str(mode),
                                budget=budget,
                                selection=scalar_selection(
                                    data["samples"],
                                    values,
                                    calibration_scores[seed],
                                    policy_maps[policy][role],
                                    mode=str(mode),
                                    budget=budget,
                                ),
                            )
        for method in ("DCSB-Score-Adapted", "Confidence"):
            for policy in ("Original-only", "SharedAction-OQ"):
                for budget in map(float, PROTOCOL["budgets"]):
                    for mode in PROTOCOL["modes"]:
                        add(
                            tasks,
                            role=role,
                            method=method,
                            policy=policy,
                            seed="fixed",
                            mode=str(mode),
                            budget=budget,
                            selection=scalar_selection(
                                data["samples"],
                                scalar[role][method],
                                scalar["calibration"][method],
                                policy_maps[policy][role],
                                mode=str(mode),
                                budget=budget,
                            ),
                        )
        for offset in range(int(PROTOCOL["random"]["seeds"])):
            seed = int(PROTOCOL["random"]["base_seed"]) + offset
            values = np.random.default_rng(seed).random(len(data["samples"]))
            reference = np.random.default_rng(seed).random(len(roles["calibration"]["samples"]))
            for policy in ("Original-only", "SharedAction-OQ"):
                for budget in map(float, PROTOCOL["budgets"]):
                    for mode in PROTOCOL["modes"]:
                        add(
                            random_tasks,
                            role=role,
                            method="Random",
                            policy=policy,
                            seed=seed,
                            mode=str(mode),
                            budget=budget,
                            selection=scalar_selection(
                                data["samples"],
                                values,
                                reference,
                                policy_maps[policy][role],
                                mode=str(mode),
                                budget=budget,
                            ),
                            random=True,
                        )
    tasks.sort(key=_point_task_priority)
    _EVAL_STATE.clear()
    _EVAL_STATE.update(roles=roles, task_selections=task_selections)
    workers = _point_worker_count(
        int(PROTOCOL["reward"]["workers"]), len(tasks) + len(random_tasks)
    )
    with mp.get_context("fork").Pool(
        workers, maxtasksperchild=POINT_TASKS_PER_CHILD
    ) as pool:
        fixed_rows = list(pool.imap_unordered(_evaluate_task, tasks, chunksize=1))
        random_rows = list(pool.imap_unordered(_evaluate_task, random_tasks, chunksize=1))
    rows = fixed_rows + _aggregate_random(random_rows)
    CORE.validate_point_rows(rows, PROTOCOL)
    rows.sort(
        key=lambda row: (
            str(row["role"]),
            str(row["method"]),
            str(row["policy"]),
            str(row["seed"]),
            str(row["mode"]),
            float(row["budget"]),
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    DATA.atomic_csv(output / "point_results.csv", rows)
    DATA.atomic_csv(output / "random_50_raw.csv", random_rows)
    DATA.atomic_pickle(output / "selections.pkl", stored)
    DATA.atomic_pickle(output / "random_selections.pkl", random_stored)
    endpoint_rows = []
    for role in ("calibration", "final"):
        data = roles[role]
        for method, selection in (
            ("Edge-only", {}),
            ("Cloud-only", original_maps[role]),
        ):
            metric = _evaluate(data, selection)
            endpoint_rows.append(
                {
                    "role": role,
                    "method": method,
                    "official_ap50_95": metric["map50_95"],
                    "official_ap50": metric["map50"],
                    "official_ap75": metric["map75"],
                    "voc07_ap50": metric["voc07_map50"],
                    "voc12_ap50": metric["voc12_map50"],
                }
            )
    DATA.atomic_csv(output / "endpoint_results.csv", endpoint_rows)
    report = {
        "complete": True,
        "rows": len(rows),
        "raw_random_rows": len(random_rows),
        "workers": workers,
        "bootstrap_performed": False,
        "final_used_for_training_or_selection": False,
        "point_results_sha256": DATA.file_sha256(output / "point_results.csv"),
        "random_results_sha256": DATA.file_sha256(output / "random_50_raw.csv"),
        "selections_sha256": DATA.file_sha256(output / "selections.pkl"),
        "random_selections_sha256": DATA.file_sha256(output / "random_selections.pkl"),
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
