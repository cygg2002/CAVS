#!/usr/bin/env python3
"""Evaluate the complete Cloud-V3 learner/action matrix on Oregon."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
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


TRAIN_CORE = load_module("oregon_cloudv3_eval_train_core", ROOT / "scripts/train_core.py")
LEARNER_CORE = load_module("oregon_cloudv3_eval_learner_core", ROOT / "scripts/learner_core.py")
PROTOCOL_CORE = load_module("oregon_cloudv3_eval_protocol_core", ROOT / "scripts/protocol_core.py")
CANONICAL = load_module(
    "oregon_cloudv3_eval_canonical",
    REPO
    / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/run_multiranker_benchmark.py",
)
CALIBRATION_CORE = load_module(
    "oregon_cloudv3_cross_action_calibration",
    REPO
    / "paper/experiments/20260809_three_dataset_cross_action_calibration_v1/scripts/calibration_core.py",
)
sys.path.insert(0, str(REPO / "paper/experiments/GAIR_v5"))
from gair.selector import select_actions_for_budget  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def group_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in read_json(path):
        output[int(raw["image_id"])].append(
            {
                "image_id": int(raw["image_id"]),
                "category_id": int(raw["category_id"]),
                "bbox": list(map(float, raw["bbox"])),
                "score": float(raw.get("score", 1.0)),
            }
        )
    return dict(output)


_ROLE_CACHE: dict[str, dict[str, Any]] = {}


def load_role(role: str, *, fusion: str = "Reversible") -> dict[str, Any]:
    key = f"{role}:{fusion}"
    if key not in _ROLE_CACHE:
        CANONICAL.COMMON = RUNS / "common"
        data = CANONICAL.load_role("oregon", role)
        manifest = read_json(RUNS / f"common/oregon/{role}/source_manifest.json")
        if fusion == "MAUR":
            for action_id, path in manifest["maur_action_sources"].items():
                data["actions"][str(action_id)] = group_predictions(Path(path))
        elif fusion != "Reversible":
            raise ValueError(fusion)
        if len(data["samples"]) != int(PROTOCOL["roles"][role]):
            raise RuntimeError(f"role population drift: {role}")
        if manifest["cloud_model_sha256"] != PROTOCOL["cloud_model"]["sha256"]:
            raise RuntimeError(f"wrong Cloud-V3 role: {role}")
        _ROLE_CACHE[key] = data
    return _ROLE_CACHE[key]


def role_records(role: str, action_space: str) -> list[dict[str, Any]]:
    data = load_role(role)
    category_ids = [
        int(row["id"])
        for row in read_json(RUNS / f"common/oregon/{role}/gt.json")["categories"]
    ]
    records = TRAIN_CORE.build_action_records(
        data["samples"], data["actions"]["edge"], category_ids=category_ids
    )
    all_actions = ["edge", "original", "q90", *PROTOCOL["roi_actions"]]
    return TRAIN_CORE.filter_records(
        records, TRAIN_CORE.action_space(action_space, all_actions)
    )


def full_job_score(method: str, role: str) -> np.ndarray:
    spec = (
        TRAIN_CORE.NEURAL_METHOD_SPECS.get(method)
        if method in TRAIN_CORE.NEURAL_METHOD_SPECS
        else ("O", "external")
    )
    values = []
    expected_ids = [int(sample["image_id"]) for sample in load_role(role)["samples"]]
    safe = method.replace("/", "_")
    for seed in PROTOCOL["model_seeds"]:
        path = RUNS / f"learners/jobs/{safe}/seed_{int(seed)}/full/scores.npz"
        with np.load(path, allow_pickle=False) as artifact:
            ids = list(map(int, artifact[f"image_ids_{role}"]))
            score = np.asarray(artifact[f"scores_{role}"], dtype=np.float64)
        if ids != expected_ids:
            raise RuntimeError(f"learner score alignment drift: {method}/{role}/{seed}")
        values.append(score)
    if len({array.shape for array in values}) != 1:
        raise RuntimeError(f"seed score shapes differ: {method}/{role}")
    output = np.mean(values, axis=0)
    if not np.isfinite(output).all():
        raise RuntimeError(f"non-finite learner score: {method}/{role}")
    return output


def _budget_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return np.repeat(array[None, :], len(PROTOCOL["budgets"]), axis=0)
    if array.ndim == 2:
        return np.repeat(array[None, :, :], len(PROTOCOL["budgets"]), axis=0)
    if array.ndim == 3 and array.shape[0] == 1:
        return np.repeat(array, len(PROTOCOL["budgets"]), axis=0)
    if array.ndim == 3 and array.shape[0] == len(PROTOCOL["budgets"]):
        return array
    raise ValueError(f"unexpected score shape: {array.shape}")


def scalar_scores(role: str) -> dict[str, np.ndarray]:
    historical = load_pickle(RUNS / "historical/artifact.pkl")["scores"][role]
    output = {
        method: _budget_matrix(np.asarray(values, dtype=np.float64))
        for method, values in historical.items()
    }
    for method in PROTOCOL["external_rankers"]:
        output[method] = _budget_matrix(full_job_score(method, role))
    for method in ("Current-BudgetSet", "SetValue", "DualHead"):
        values = _budget_matrix(full_job_score(method, role))
        # O-space is strictly [Edge, Original]; scalar score is Original value.
        output[method] = values[:, :, 1]
    missing = set(LEARNER_CORE.SCALAR_METHODS) - {"Random"} - set(output)
    if missing:
        raise RuntimeError(f"missing scalar scores: {sorted(missing)}")
    return output


def _score_map(records: Sequence[Mapping[str, Any]], matrix: np.ndarray) -> dict[int, dict[str, float]]:
    return LEARNER_CORE.action_score_map(records, matrix)


def joint_scores() -> dict[str, dict[str, np.ndarray]]:
    methods = [
        method
        for method in LEARNER_CORE.JOINT_METHODS
        if method
        not in {
            "PerAction-Isotonic-OQ",
            "OccupancyFree-AFPD-OQR12",
            "OccupancyAware-AFPD-OQR12",
        }
    ]
    output = {role: {} for role in ("calibration", "final")}
    for role in output:
        for method in methods:
            output[role][method] = _budget_matrix(full_job_score(method, role))

    # Per-action monotonic calibration is fit only on the calibration role.
    calibration_rewards = load_pickle(RUNS / "rewards/calibration/reward_map.pkl")
    raw_cal = output["calibration"]["Raw-ActionValue-OQ"].copy()
    raw_final = output["final"]["Raw-ActionValue-OQ"].copy()
    cal_records = role_records("calibration", "OQ")
    final_records = role_records("final", "OQ")
    isotonic_models = {}
    calibrated = {"calibration": raw_cal.copy(), "final": raw_final.copy()}
    for budget_index, budget in enumerate(PROTOCOL["budgets"]):
        isotonic_models[str(budget)] = {}
        for column, action_id in ((1, "original"), (2, "q90")):
            targets = np.asarray(
                [calibration_rewards[int(record["image_id"])][action_id] for record in cal_records],
                dtype=np.float64,
            )
            model = CALIBRATION_CORE.fit_isotonic(
                raw_cal[budget_index, :, column], targets, bins=32
            )
            isotonic_models[str(budget)][action_id] = model
            calibrated["calibration"][budget_index, :, column] = CALIBRATION_CORE.apply_isotonic(
                model, raw_cal[budget_index, :, column]
            )
            calibrated["final"][budget_index, :, column] = CALIBRATION_CORE.apply_isotonic(
                model, raw_final[budget_index, :, column]
            )
    output["calibration"]["PerAction-Isotonic-OQ"] = calibrated["calibration"]
    output["final"]["PerAction-Isotonic-OQ"] = calibrated["final"]

    for label in ("OccupancyFree-AFPD-OQR12", "OccupancyAware-AFPD-OQR12"):
        probabilities_method = label
        for role in output:
            base = output[role]["AFPD-OQR12"]
            probabilities = _budget_matrix(full_job_score(probabilities_method, role))[0]
            records = role_records(role, "OQR12")
            gated = np.full_like(base, -1e9)
            for budget_index in range(len(PROTOCOL["budgets"])):
                score_map = _score_map(records, base[budget_index])
                probability_map = {
                    int(record["image_id"]): {
                        str(action["action_id"]): float(probabilities[row, column])
                        for column, action in enumerate(record["actions"])
                        if "@" in str(action["action_id"])
                    }
                    for row, record in enumerate(records)
                }
                values = LEARNER_CORE.gate_roi_scores(
                    score_map, probability_map, threshold=0.5
                )
                for row, record in enumerate(records):
                    gated[budget_index, row, 0] = 0.0
                    for column, action in enumerate(record["actions"][1:], 1):
                        gated[budget_index, row, column] = values[int(record["image_id"])][str(action["action_id"])]
            output[role][label] = gated
    atomic_pickle(
        RUNS / "evaluation/controls/isotonic_models.pkl",
        {"models": isotonic_models, "fit_role": "calibration", "final_used": False},
    )
    return output


def schedule_joint(
    samples: Sequence[Mapping[str, Any]],
    score_map: Mapping[int, Mapping[str, float]],
    *,
    mode: str,
    budget: float,
) -> dict[int, str]:
    groups = []
    best_by_image = []
    for sample in samples:
        image_id = int(sample["image_id"])
        actions = [
            {
                "action_id": "edge",
                "image_id": image_id,
                "value": 0.0,
                "cost": 0.0,
            }
        ]
        for action_id, value in score_map[image_id].items():
            if action_id not in sample["payload_bytes"] or not sample["available"].get(action_id, False):
                continue
            actions.append(
                {
                    "action_id": str(action_id),
                    "image_id": image_id,
                    "value": float(value),
                    "cost": float(sample["payload_bytes"][action_id]),
                }
            )
        if len(actions) == 1:
            raise RuntimeError(f"no cloud action: {image_id}")
        groups.append(actions)
        best_by_image.append(max(actions[1:], key=lambda row: (row["value"], -row["cost"], row["action_id"])))
    if mode == "fixed_byte":
        capacity = float(budget) * sum(float(sample["payload_bytes"]["original"]) for sample in samples)
        result = select_actions_for_budget(groups, capacity)
        selection = {
            int(row["image_id"]): str(row["action_id"])
            for row in result["selected"]
            if str(row["action_id"]) != "edge"
        }
    elif mode == "fixed_call":
        count = int(math.floor(len(samples) * float(budget) + 0.5))
        selected = sorted(
            best_by_image,
            key=lambda row: (-float(row["value"]), float(row["cost"]), int(row["image_id"])),
        )[:count]
        selection = {int(row["image_id"]): str(row["action_id"]) for row in selected}
    else:
        raise ValueError(mode)
    LEARNER_CORE.validate_selection(samples, selection, mode=mode, budget=budget)
    return selection


def shared_action_probabilities(
    role: str, artifact: Mapping[str, Any]
) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
    data = load_role(role)
    features = np.asarray(
        [sample["action_features"] for sample in data["samples"]], dtype=np.float64
    )
    q90 = CANONICAL.predict_weighted_logistic(artifact["q90_model"], features)
    roi_features, keys = _gbdt_features(role)
    roi_values = CANONICAL.predict_weighted_logistic(
        artifact["roi_model"], roi_features
    )
    roi: dict[int, dict[str, float]] = defaultdict(dict)
    for (image_id, action_id), value in zip(keys, roi_values):
        roi[int(image_id)][str(action_id)] = float(value)
    return q90, dict(roi)


def _action_maps_for_role(
    role: str,
    shared: Mapping[str, Any],
    gbdt: Mapping[str, Any] | None = None,
) -> dict[str, dict[int, str]]:
    data = load_role(role)
    q90_probability, roi_probability = shared_action_probabilities(role, shared)
    original = {int(sample["image_id"]): "original" for sample in data["samples"]}
    no_roi = {}
    shared_r12 = {}
    for index, sample in enumerate(data["samples"]):
        image_id = int(sample["image_id"])
        no_roi[image_id] = (
            "q90"
            if sample["available"].get("q90", False)
            and float(q90_probability[index]) >= float(shared["q90_threshold"])
            else "original"
        )
        candidates = [
            action
            for action, probability in roi_probability.get(image_id, {}).items()
            if sample["available"].get(action, False)
            and float(probability) >= float(shared["roi_threshold"])
        ]
        shared_r12[image_id] = max(
            candidates,
            key=lambda action: (
                roi_probability[image_id][action],
                -sample["payload_bytes"][action],
                action,
            ),
            default=no_roi[image_id],
        )
    all_roi = {}
    for sample in data["samples"]:
        image_id = int(sample["image_id"])
        base = shared_r12[image_id]
        candidates = [
            action_id
            for action_id in PROTOCOL["roi_actions"]
            if sample["available"].get(action_id, False)
            and sample["payload_bytes"][action_id] < sample["payload_bytes"][base]
        ]
        all_roi[image_id] = min(
            candidates,
            key=lambda action: (
                int(sample["payload_bytes"][action]),
                -float(roi_probability.get(image_id, {}).get(action, 0.0)),
                action,
            ),
            default=base,
        )
    hierarchical = dict(shared_r12)
    if gbdt is not None:
        probabilities = predict_hierarchical_probabilities(role, gbdt)
        threshold = float(gbdt["threshold"])
        for sample in data["samples"]:
            image_id = int(sample["image_id"])
            base = no_roi[image_id]
            candidates = [
                action
                for action in PROTOCOL["roi_actions"]
                if sample["available"].get(action, False)
                and probabilities[image_id].get(action, 0.0) >= threshold
                and sample["payload_bytes"][action] < sample["payload_bytes"][base]
            ]
            hierarchical[image_id] = max(
                candidates,
                key=lambda action: (
                    probabilities[image_id][action],
                    roi_probability.get(image_id, {}).get(action, 0.0),
                    -sample["payload_bytes"][action],
                ),
                default=base,
            )
    return {
        "Original-only": original,
        "SharedAction-NoROI": no_roi,
        "SharedAction-R12": shared_r12,
        "CostEligible-AllROI-R12": all_roi,
        "GBDT-Hierarchical-R12": hierarchical,
    }


def train_shared_action() -> dict[str, Any]:
    path = RUNS / "evaluation/controls/shared_action.pkl"
    if path.is_file():
        return load_pickle(path)
    fit = load_role("fit")
    fit_features = np.asarray(
        [sample["action_features"] for sample in fit["samples"]], dtype=np.float64
    )
    epsilon = float(PROTOCOL["shared_action"]["epsilon"])
    q90_labels = np.asarray(
        [
            sample["available"].get("q90", False)
            and float(sample["qualities"]["q90"])
            >= float(sample["qualities"]["original"]) - epsilon
            for sample in fit["samples"]
        ],
        dtype=np.float64,
    )
    roi_features, roi_keys = _gbdt_features("fit")
    fit_by_id = {int(sample["image_id"]): sample for sample in fit["samples"]}
    roi_labels = np.asarray(
        [
            float(fit_by_id[image_id]["qualities"][action_id])
            >= float(fit_by_id[image_id]["qualities"]["original"]) - epsilon
            for image_id, action_id in roi_keys
        ],
        dtype=np.float64,
    )
    q90_model = CANONICAL.fit_weighted_logistic(fit_features, q90_labels)
    roi_model = CANONICAL.fit_weighted_logistic(roi_features, roi_labels)
    shell = {
        "q90_model": q90_model,
        "roi_model": roi_model,
        "q90_threshold": float("inf"),
        "roi_threshold": float("inf"),
    }
    calibration = load_role("calibration")
    q90_probability, roi_probability = shared_action_probabilities("calibration", shell)
    q90_eligible = np.asarray(
        [sample["available"].get("q90", False) for sample in calibration["samples"]]
    )
    q90_thresholds = sorted(
        {
            CANONICAL.threshold_for_fraction(q90_probability, q90_eligible, fraction)
            for fraction in PROTOCOL["shared_action"]["action_fractions"]
        },
        reverse=True,
    )
    base = {int(sample["image_id"]): "original" for sample in calibration["samples"]}
    base_predictions, _ = CANONICAL.mixed_predictions(calibration, base, 0.001)
    base_ap = float(calibration["evaluator"].evaluate(base_predictions)["map50_95"])
    q90_candidates = []
    for threshold in q90_thresholds:
        shell["q90_threshold"] = float(threshold)
        action_map = _action_maps_for_role("calibration", shell, None)["SharedAction-NoROI"]
        predictions, _ = CANONICAL.mixed_predictions(calibration, action_map, 0.001)
        ap = float(calibration["evaluator"].evaluate(predictions)["map50_95"])
        requested = sum(
            calibration["sample_by_id"][image_id]["payload_bytes"][action]
            for image_id, action in action_map.items()
        )
        q90_candidates.append({"threshold": float(threshold), "ap": ap, "request_bytes": requested})
    eligible_q90 = [
        row for row in q90_candidates
        if row["ap"] >= base_ap - float(PROTOCOL["shared_action"]["maximum_calibration_ap_loss"])
    ]
    q90_choice = min(
        eligible_q90,
        key=lambda row: (row["request_bytes"], -row["ap"], -row["threshold"]),
        default={"threshold": float("inf"), "ap": base_ap, "request_bytes": 0},
    )
    shell["q90_threshold"] = float(q90_choice["threshold"])
    all_roi_probabilities = np.asarray(
        [value for values in roi_probability.values() for value in values.values()],
        dtype=np.float64,
    )
    roi_thresholds = sorted(
        {
            CANONICAL.threshold_for_fraction(
                all_roi_probabilities,
                np.ones(len(all_roi_probabilities), dtype=bool),
                fraction,
            )
            for fraction in PROTOCOL["shared_action"]["action_fractions"]
        },
        reverse=True,
    )
    no_roi_map = _action_maps_for_role("calibration", shell, None)["SharedAction-NoROI"]
    no_roi_predictions, _ = CANONICAL.mixed_predictions(calibration, no_roi_map, 0.001)
    no_roi_ap = float(calibration["evaluator"].evaluate(no_roi_predictions)["map50_95"])
    roi_candidates = []
    for threshold in roi_thresholds:
        shell["roi_threshold"] = float(threshold)
        action_map = _action_maps_for_role("calibration", shell, None)["SharedAction-R12"]
        predictions, _ = CANONICAL.mixed_predictions(calibration, action_map, 0.001)
        ap = float(calibration["evaluator"].evaluate(predictions)["map50_95"])
        requested = sum(
            calibration["sample_by_id"][image_id]["payload_bytes"][action]
            for image_id, action in action_map.items()
        )
        roi_candidates.append({"threshold": float(threshold), "ap": ap, "request_bytes": requested, "roi_count": sum("@" in action for action in action_map.values())})
    eligible_roi = [
        row for row in roi_candidates
        if row["ap"] >= no_roi_ap - float(PROTOCOL["shared_action"]["maximum_calibration_ap_loss"])
    ]
    roi_choice = min(
        eligible_roi,
        key=lambda row: (row["request_bytes"], -row["ap"], -row["threshold"]),
        default={"threshold": float("inf"), "ap": no_roi_ap, "request_bytes": 0, "roi_count": 0},
    )
    artifact = {
        "q90_model": q90_model,
        "roi_model": roi_model,
        "q90_threshold": float(q90_choice["threshold"]),
        "roi_threshold": float(roi_choice["threshold"]),
        "fit_q90_positive_rate": float(q90_labels.mean()),
        "fit_roi_positive_rate": float(roi_labels.mean()),
        "q90_candidates": q90_candidates,
        "roi_candidates": roi_candidates,
        "q90_choice": q90_choice,
        "roi_choice": roi_choice,
        "fit_only_training": True,
        "calibration_only_thresholds": True,
        "final_used_for_training_or_selection": False,
    }
    atomic_pickle(path, artifact)
    return artifact


def _gbdt_features(role: str) -> tuple[np.ndarray, list[tuple[int, str]]]:
    rows, keys = [], []
    data = load_role(role)
    for sample in data["samples"]:
        original = max(1, int(sample["payload_bytes"]["original"]))
        for action_id in PROTOCOL["roi_actions"]:
            if not sample["available"].get(action_id, False):
                continue
            profile, scale_raw = action_id.split("@")
            profile_onehot = [float(profile == value) for value in PROTOCOL["profiles"]]
            scale = int(scale_raw)
            scale_onehot = [float(scale == value) for value in PROTOCOL["imgsz"]]
            source = next(row for row in sample["actions"] if row["action_id"] == action_id)
            rows.append(
                [
                    *map(float, sample["action_features"]),
                    *profile_onehot,
                    *scale_onehot,
                    float(source["area_ratio"]),
                    float(sample["payload_bytes"][action_id]) / original,
                ]
            )
            keys.append((int(sample["image_id"]), action_id))
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"non-finite hierarchical features: {role}")
    return matrix, keys


def predict_hierarchical_probabilities(
    role: str, artifact: Mapping[str, Any]
) -> dict[int, dict[str, float]]:
    features, keys = _gbdt_features(role)
    raw = artifact["model"].predict_proba(features)
    classes = list(map(int, artifact["model"].classes_))
    values = raw[:, classes.index(1)] if 1 in classes else np.zeros(len(features))
    output: dict[int, dict[str, float]] = defaultdict(dict)
    for (image_id, action_id), value in zip(keys, values):
        output[image_id][action_id] = float(value)
    return dict(output)


def train_hierarchical(shared: Mapping[str, Any]) -> dict[str, Any]:
    path = RUNS / "evaluation/controls/hierarchical.pkl"
    if path.is_file():
        return load_pickle(path)
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier

    fit_features, fit_keys = _gbdt_features("fit")
    fit_reward = load_pickle(RUNS / "rewards/fit/reward_map.pkl")
    labels = np.asarray(
        [
            fit_reward[image_id][action_id]
            >= max(0.0, fit_reward[image_id]["original"], fit_reward[image_id]["q90"])
            for image_id, action_id in fit_keys
        ],
        dtype=np.int64,
    )
    model = (
        DummyClassifier(strategy="constant", constant=int(labels[0]))
        if len(np.unique(labels)) == 1
        else HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=int(PROTOCOL["hierarchical_gate"]["seed"]),
        )
    ).fit(fit_features, labels)
    shell = {"model": model, "threshold": float("inf")}
    base = _action_maps_for_role("calibration", shared, None)["SharedAction-NoROI"]
    data = load_role("calibration")
    base_predictions, _ = CANONICAL.mixed_predictions(data, base, 0.001)
    base_ap = float(data["evaluator"].evaluate(base_predictions)["map50_95"])
    candidates = []
    for threshold in PROTOCOL["hierarchical_gate"]["minimum_safe_probability_grid"]:
        shell["threshold"] = float(threshold)
        actions = _action_maps_for_role("calibration", shared, shell)["GBDT-Hierarchical-R12"]
        predictions, _ = CANONICAL.mixed_predictions(data, actions, 0.001)
        metric = data["evaluator"].evaluate(predictions)
        requested = sum(data["sample_by_id"][image_id]["payload_bytes"][action] for image_id, action in actions.items())
        candidates.append(
            {
                "threshold": float(threshold),
                "ap": float(metric["map50_95"]),
                "request_bytes": int(requested),
                "roi_count": sum("@" in action for action in actions.values()),
            }
        )
    eligible = [
        row
        for row in candidates
        if row["ap"] >= base_ap - float(PROTOCOL["hierarchical_gate"]["maximum_calibration_ap_loss"])
    ]
    chosen = min(
        eligible,
        key=lambda row: (row["request_bytes"], -row["ap"], -row["threshold"]),
        default={"threshold": float("inf"), "ap": base_ap, "request_bytes": 0, "roi_count": 0},
    )
    artifact = {
        "model": model,
        "threshold": float(chosen["threshold"]),
        "calibration_base_ap": base_ap,
        "candidates": candidates,
        "chosen": chosen,
        "fit_rows": len(fit_features),
        "fit_positive_rate": float(labels.mean()),
        "fit_only_training": True,
        "calibration_only_threshold": True,
        "final_used_for_training_or_selection": False,
    }
    atomic_pickle(path, artifact)
    return artifact


def _rank_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(ordered, np.asarray(values), side="right") / max(1, len(ordered))


def _scalar_byte_selection(
    samples: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    action_map: Mapping[int, str],
    *,
    budget: float,
    lambda_value: float,
    reference_scores: np.ndarray,
    median_original_bytes: float,
) -> dict[int, str]:
    percentile = _rank_percentile(reference_scores, scores)
    priority = np.asarray(
        [
            percentile[index]
            - float(lambda_value)
            * float(sample["payload_bytes"][action_map[int(sample["image_id"])]])
            / max(1.0, float(median_original_bytes))
            for index, sample in enumerate(samples)
        ]
    )
    capacity = float(budget) * sum(float(sample["payload_bytes"]["original"]) for sample in samples)
    used = 0.0
    selected = {}
    for index in np.argsort(-priority, kind="mergesort"):
        sample = samples[int(index)]
        image_id = int(sample["image_id"])
        action = action_map[image_id]
        cost = float(sample["payload_bytes"][action])
        if used + cost <= capacity + 1e-6:
            selected[image_id] = action
            used += cost
    LEARNER_CORE.validate_selection(samples, selected, mode="fixed_byte", budget=budget)
    return selected


def calibrate_scalar_lambda(
    samples: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    action_map: Mapping[int, str],
    reward_map: Mapping[int, Mapping[str, float]],
    *,
    budget: float,
) -> dict[str, float]:
    median = float(np.median([sample["payload_bytes"]["original"] for sample in samples]))
    sample_by_id = {int(sample["image_id"]): sample for sample in samples}
    candidates = []
    for value in PROTOCOL["fixed_byte_scheduler"]["lambda_grid"]:
        selection = _scalar_byte_selection(
            samples,
            scores,
            action_map,
            budget=budget,
            lambda_value=float(value),
            reference_scores=scores,
            median_original_bytes=median,
        )
        reward = sum(float(reward_map[image_id][action]) for image_id, action in selection.items())
        requested = sum(
            float(sample_by_id[image_id]["payload_bytes"][action])
            for image_id, action in selection.items()
        )
        candidates.append({"lambda": float(value), "teacher_reward": reward, "request_bytes": requested})
    chosen = max(candidates, key=lambda row: (row["teacher_reward"], -row["request_bytes"], -row["lambda"]))
    return {**chosen, "median_original_bytes": median}


_EVAL_STATE: dict[str, Any] = {}


def _evaluate(task: Mapping[str, Any]) -> dict[str, Any]:
    data = _EVAL_STATE[str(task["role"])]
    selection = {int(key): str(value) for key, value in task["selection"].items()}
    predictions, _ = CANONICAL.mixed_predictions(data, selection, 0.001)
    metric = data["evaluator"].evaluate(predictions)
    validation = LEARNER_CORE.validate_selection(
        data["samples"], selection, mode=str(task["mode"]), budget=float(task["budget"])
    )
    counts = Counter(selection.values())
    return {
        "role": str(task["role"]),
        "method": str(task["method"]),
        "policy": str(task["policy"]),
        "mode": str(task["mode"]),
        "budget": float(task["budget"]),
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "kb_per_image": validation["request_bytes"] / 1024.0 / len(data["samples"]),
        "request_bytes": int(validation["request_bytes"]),
        "capacity_utilization": float(validation["utilization"]),
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": sum(count for action, count in counts.items() if "@" in action),
        "cloud_model": "Cloud-V3-epoch28-best",
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "final_is_hidden": False,
        "final_used_for_selection": False,
    }


def _scalar_call(samples: Sequence[Mapping[str, Any]], scores: np.ndarray, action_map: Mapping[int, str], budget: float) -> dict[int, str]:
    count = int(math.floor(len(samples) * float(budget) + 0.5))
    order = np.argsort(-np.asarray(scores), kind="mergesort")[:count]
    selection = {int(samples[index]["image_id"]): action_map[int(samples[index]["image_id"])] for index in order}
    LEARNER_CORE.validate_selection(samples, selection, mode="fixed_call", budget=budget)
    return selection


def run_controls() -> dict[str, Any]:
    completion = RUNS / "evaluation/controls/completion.json"
    if completion.is_file():
        return read_json(RUNS / "evaluation/controls/report.json")
    if not (RUNS / "learners/completion.json").is_file() or not (RUNS / "historical/completion.json").is_file():
        raise RuntimeError("all learners must finish before evaluation controls")
    joint = joint_scores()
    shared = train_shared_action()
    gbdt = train_hierarchical(shared)
    action_maps = {}
    for role in ("calibration", "final"):
        action_maps[role] = _action_maps_for_role(role, shared, gbdt)
    scalar = {role: scalar_scores(role) for role in ("calibration", "final")}
    calibration_reward = load_pickle(RUNS / "rewards/calibration/reward_map.pkl")
    lambda_controls = {}
    for method in LEARNER_CORE.SCALAR_METHODS:
        if method == "Random":
            continue
        for policy in PROTOCOL["scalar_action_policies"]:
            for budget_index, budget in enumerate(PROTOCOL["budgets"]):
                values = scalar["calibration"][method][budget_index]
                key = f"{method}|{policy}|{float(budget):g}"
                lambda_controls[key] = calibrate_scalar_lambda(
                    load_role("calibration")["samples"],
                    values,
                    action_maps["calibration"][policy],
                    calibration_reward,
                    budget=float(budget),
                )
    artifact = {
        "action_maps": action_maps,
        "lambda_controls": lambda_controls,
        "scalar_scores": scalar,
        "joint_scores": joint,
        "gbdt": gbdt,
        "shared_action": shared,
        "calibration_only_controls": True,
        "final_used_for_training_or_selection": False,
    }
    artifact_path = RUNS / "evaluation/controls/artifact.pkl"
    atomic_pickle(artifact_path, artifact)
    report = {
        "complete": True,
        "action_policies": list(PROTOCOL["scalar_action_policies"]),
        "lambda_controls": len(lambda_controls),
        "gbdt_threshold": float(gbdt["threshold"])
        if math.isfinite(float(gbdt["threshold"]))
        else None,
        "gbdt_fit_positive_rate": gbdt["fit_positive_rate"],
        "shared_q90_threshold": float(shared["q90_threshold"])
        if math.isfinite(float(shared["q90_threshold"]))
        else None,
        "shared_roi_threshold": float(shared["roi_threshold"])
        if math.isfinite(float(shared["roi_threshold"]))
        else None,
        "artifact_sha256": file_sha256(artifact_path),
        "final_used_for_training_or_selection": False,
    }
    atomic_json(RUNS / "evaluation/controls/report.json", report)
    atomic_json(completion, {"complete": True, "artifact_sha256": report["artifact_sha256"]})
    return report


def run_point() -> dict[str, Any]:
    completion = RUNS / "evaluation/point/completion.json"
    if completion.is_file():
        return read_json(RUNS / "evaluation/point/report.json")
    controls_report = run_controls()
    artifact_path = RUNS / "evaluation/controls/artifact.pkl"
    if controls_report["artifact_sha256"] != file_sha256(artifact_path):
        raise RuntimeError("evaluation controls changed")
    controls = load_pickle(artifact_path)
    tasks = []
    selections: dict[str, dict[str, str]] = {}
    for role in ("calibration", "final"):
        data = load_role(role)
        samples = data["samples"]
        for method in LEARNER_CORE.SCALAR_METHODS:
            if method == "Random":
                continue
            for policy in PROTOCOL["scalar_action_policies"]:
                action_map = controls["action_maps"][role][policy]
                for budget_index, budget in enumerate(PROTOCOL["budgets"]):
                    score = controls["scalar_scores"][role][method][budget_index]
                    for mode in PROTOCOL["modes"]:
                        if mode == "fixed_call":
                            selection = _scalar_call(samples, score, action_map, float(budget))
                        else:
                            control = controls["lambda_controls"][f"{method}|{policy}|{float(budget):g}"]
                            selection = _scalar_byte_selection(
                                samples,
                                score,
                                action_map,
                                budget=float(budget),
                                lambda_value=float(control["lambda"]),
                                reference_scores=controls["scalar_scores"]["calibration"][method][budget_index],
                                median_original_bytes=float(control["median_original_bytes"]),
                            )
                        key = f"{role}|{method}|{policy}|{mode}|{float(budget):g}"
                        selections[key] = {str(image_id): action for image_id, action in selection.items()}
                        tasks.append({"role": role, "method": method, "policy": policy, "mode": mode, "budget": float(budget), "selection": selection})

        # Random uses 50 frozen seeds. Mean rows are reported separately after evaluation.
        for random_seed_index in range(int(PROTOCOL["random_seeds"])):
            seed = int(PROTOCOL["fold_seed"]) + random_seed_index
            score = CANONICAL.random_scores(samples, seed)
            cal_score = CANONICAL.random_scores(load_role("calibration")["samples"], seed)
            for policy in PROTOCOL["scalar_action_policies"]:
                action_map = controls["action_maps"][role][policy]
                for budget in PROTOCOL["budgets"]:
                    for mode in PROTOCOL["modes"]:
                        if mode == "fixed_call":
                            selection = _scalar_call(samples, score, action_map, float(budget))
                        else:
                            control = calibrate_scalar_lambda(
                                load_role("calibration")["samples"],
                                cal_score,
                                controls["action_maps"]["calibration"][policy],
                                load_pickle(RUNS / "rewards/calibration/reward_map.pkl"),
                                budget=float(budget),
                            )
                            selection = _scalar_byte_selection(
                                samples,
                                score,
                                action_map,
                                budget=float(budget),
                                lambda_value=float(control["lambda"]),
                                reference_scores=cal_score,
                                median_original_bytes=float(control["median_original_bytes"]),
                            )
                        method = f"Random[seed={seed}]"
                        key = f"{role}|{method}|{policy}|{mode}|{float(budget):g}"
                        selections[key] = {str(image_id): action for image_id, action in selection.items()}
                        tasks.append({"role": role, "method": method, "policy": policy, "mode": mode, "budget": float(budget), "selection": selection})

        for method in LEARNER_CORE.JOINT_METHODS:
            space = "OQR12" if method in {
                "CurrentActionValue-OQR12", "AFPD-OQR12",
                "OccupancyFree-AFPD-OQR12", "OccupancyAware-AFPD-OQR12",
            } else "OQ"
            records = role_records(role, space)
            for budget_index, budget in enumerate(PROTOCOL["budgets"]):
                score_map = _score_map(records, controls["joint_scores"][role][method][budget_index])
                for mode in PROTOCOL["modes"]:
                    selection = schedule_joint(samples, score_map, mode=mode, budget=float(budget))
                    key = f"{role}|{method}|Joint|{mode}|{float(budget):g}"
                    selections[key] = {str(image_id): action for image_id, action in selection.items()}
                    tasks.append({"role": role, "method": method, "policy": "Joint", "mode": mode, "budget": float(budget), "selection": selection})

    _EVAL_STATE.clear()
    _EVAL_STATE.update({role: load_role(role) for role in ("calibration", "final")})
    workers = min(int(PROTOCOL["cpu"]["maximum_workers"]), len(tasks), max(1, (os.cpu_count() or 1) - 4))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        raw_rows = list(pool.imap_unordered(_evaluate, tasks, chunksize=1))
    deterministic = [row for row in raw_rows if not row["method"].startswith("Random[")]
    random_rows = [row for row in raw_rows if row["method"].startswith("Random[")]
    rows = list(deterministic)
    for role in ("calibration", "final"):
        for policy in PROTOCOL["scalar_action_policies"]:
            for mode in PROTOCOL["modes"]:
                for budget in PROTOCOL["budgets"]:
                    values = [row for row in random_rows if row["role"] == role and row["policy"] == policy and row["mode"] == mode and math.isclose(row["budget"], float(budget))]
                    if len(values) != int(PROTOCOL["random_seeds"]):
                        raise RuntimeError("Random seed matrix incomplete")
                    average_fields = ("official_ap50_95", "official_ap50", "official_ap75", "kb_per_image", "request_bytes", "capacity_utilization", "selected_images", "original_count", "q90_count", "roi_count")
                    row = {**values[0], "method": "Random", "random_seed_count": len(values)}
                    for field in average_fields:
                        row[field] = float(np.mean([float(value[field]) for value in values]))
                    rows.append(row)
    expected_per_role = len(LEARNER_CORE.evaluation_matrix(PROTOCOL))
    if any(sum(row["role"] == role for row in rows) != expected_per_role for role in ("calibration", "final")):
        raise RuntimeError("complete evaluation matrix missing")
    result_path = RUNS / "evaluation/point/results.csv"
    random_path = RUNS / "evaluation/point/random_seed_results.csv"
    selection_path = RUNS / "evaluation/point/selections.json"
    atomic_csv(result_path, rows)
    atomic_csv(random_path, random_rows)
    atomic_json(selection_path, selections)
    report = {
        "complete": True,
        "rows": len(rows),
        "rows_per_role": expected_per_role,
        "random_seed_rows": len(random_rows),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "results_sha256": file_sha256(result_path),
        "random_results_sha256": file_sha256(random_path),
        "selections_sha256": file_sha256(selection_path),
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "final_is_hidden": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(RUNS / "evaluation/point/report.json", report)
    atomic_json(completion, {"complete": True, "results_sha256": report["results_sha256"], "selections_sha256": report["selections_sha256"]})
    _EVAL_STATE.clear()
    return report


def _teacher_score_map(role: str) -> dict[int, dict[str, float]]:
    output = {}
    for sample in load_role(role)["samples"]:
        image_id = int(sample["image_id"])
        output[image_id] = {
            action: float(sample["gains"].get(action, -1e9))
            for action in ("original", "q90", *PROTOCOL["roi_actions"])
            if sample["available"].get(action, False)
        }
    return output


def run_references() -> dict[str, Any]:
    completion = RUNS / "evaluation/references/completion.json"
    if completion.is_file():
        return read_json(RUNS / "evaluation/references/report.json")
    tasks = []
    selections = {}
    system_rows = []
    for role in ("calibration", "final"):
        for fusion in ("Reversible", "MAUR"):
            data = load_role(role, fusion=fusion)
            for budget in PROTOCOL["budgets"]:
                for mode in PROTOCOL["modes"]:
                    selection = schedule_joint(data["samples"], _teacher_score_map(role), mode=mode, budget=float(budget))
                    method = f"GT-Proxy-Teacher-OQR12-{fusion}"
                    key = f"{role}|{method}|Joint|{mode}|{float(budget):g}"
                    selections[key] = {str(image_id): action for image_id, action in selection.items()}
                    tasks.append({"role": f"{role}:{fusion}", "method": method, "policy": "Joint", "mode": mode, "budget": float(budget), "selection": selection})
        data = load_role(role)
        for method, action in (("Edge-only", None), ("Cloud-only Original", "original"), ("Cloud-only Q90", "q90")):
            selection = {} if action is None else {int(sample["image_id"]): action for sample in data["samples"]}
            predictions, _ = CANONICAL.mixed_predictions(data, selection, 0.001)
            metric = data["evaluator"].evaluate(predictions)
            requested = 0 if action is None else sum(sample["payload_bytes"][action] for sample in data["samples"])
            system_rows.append({"role": role, "method": method, "policy": "System", "mode": "system", "budget": 1.0 if action else 0.0, "official_ap50_95": metric["map50_95"], "official_ap50": metric["map50"], "official_ap75": metric["map75"], "kb_per_image": requested / 1024 / len(data["samples"]), "request_bytes": requested, "selected_images": len(selection), "cloud_model": "Cloud-V3-epoch28-best"})
    _EVAL_STATE.clear()
    _EVAL_STATE.update({f"{role}:{fusion}": load_role(role, fusion=fusion) for role in ("calibration", "final") for fusion in ("Reversible", "MAUR")})
    workers = min(int(PROTOCOL["cpu"]["maximum_workers"]), len(tasks))
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        teacher_rows = list(pool.imap_unordered(_evaluate, tasks, chunksize=1))
    for row in teacher_rows:
        row["role"], row["fusion"] = row["role"].split(":", 1)
    rows = system_rows + teacher_rows
    path = RUNS / "evaluation/references/results.csv"
    atomic_csv(path, rows)
    atomic_json(RUNS / "evaluation/references/selections.json", selections)
    report = {"complete": True, "rows": len(rows), "results_sha256": file_sha256(path), "teacher_kind": "GT-aware per-image gain proxy, not global mAP Oracle", "final_is_hidden": False}
    atomic_json(RUNS / "evaluation/references/report.json", report)
    atomic_json(completion, {"complete": True, "results_sha256": report["results_sha256"]})
    _EVAL_STATE.clear()
    return report


def run_all() -> dict[str, Any]:
    return {"point": run_point(), "references": run_references()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("controls", "point", "references", "all"), default="all")
    args = parser.parse_args()
    result = {"controls": run_controls, "point": run_point, "references": run_references, "all": run_all}[args.stage]()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
