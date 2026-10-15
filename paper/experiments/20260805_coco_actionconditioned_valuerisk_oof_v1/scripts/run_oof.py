#!/usr/bin/env python3
"""Run the cache-only COCO Action-Conditioned Value–Risk OOF diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL_PATH = ROOT / "configs/protocol.json"


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


PROTOCOL = read_json(PROTOCOL_PATH)
SOURCES = {
    name: REPO / path for name, path in PROTOCOL["source_experiments"].items()
}
CANONICAL_ROOT = SOURCES["canonical"]
COMMON = CANONICAL_ROOT / PROTOCOL["canonical_common"]
BENCHMARK = CANONICAL_ROOT / PROTOCOL["canonical_benchmark"]
REWARD_ROOT = SOURCES["rewards"]
SET_SCRIPTS = SOURCES["set_model"] / "scripts"
VALUE_SCRIPTS = SOURCES["value_model"] / "scripts"
CANONICAL_SCRIPTS = CANONICAL_ROOT / "scripts"
TRIACTION_SCRIPTS = (
    REPO
    / "paper/experiments/20260722_ena24_hierarchical_triaction_paper_exploration/scripts"
)
sys.path[:0] = [
    str(ROOT / "scripts"),
    str(SET_SCRIPTS),
    str(VALUE_SCRIPTS),
    str(CANONICAL_SCRIPTS),
    str(TRIACTION_SCRIPTS),
]

from action_core import (  # noqa: E402
    CLOUD_ACTIONS,
    METHODS,
    build_action_records,
    deterministic_folds,
    evaluate_gate,
    numeric_score_arrays,
    schedule_fixed_byte,
    schedule_fixed_call,
    validate_protocol,
)
from value_risk import (  # noqa: E402
    fit_value_risk_router,
    predict_value_risk_router,
)
import run_multiranker_benchmark as canonical  # noqa: E402
import run_exploration as tri  # type: ignore  # noqa: E402
from oof_train import (  # noqa: E402
    fit_dual_router,
    fit_value_router,
    predict_dual_value,
    predict_value_router,
)
from run_experiment import _record_tensors  # noqa: E402


validate_protocol(PROTOCOL)


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
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def set_protocol() -> dict[str, Any]:
    return {
        "budgets": list(PROTOCOL["budgets"]),
        "max_boxes": int(PROTOCOL["model"]["max_boxes"]),
        "max_actions": int(PROTOCOL["model"]["max_actions"]),
        "deploy_confidence": float(PROTOCOL["model"]["deploy_confidence"]),
        "resource_profile": "upload",
        "model": dict(PROTOCOL["model"]),
    }


def _configure_canonical() -> None:
    canonical.COMMON = COMMON.parent
    tri.current.COMMON = COMMON.parent
    if Path(canonical.COMMON) / "coco" != COMMON:
        raise RuntimeError("Canonical COCO cache binding failed")


_ROLE_CACHE: dict[str, dict[str, Any]] = {}


def load_role(role: str) -> dict[str, Any]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(f"Forbidden role: {role}")
    if role not in _ROLE_CACHE:
        _configure_canonical()
        data = canonical.load_role("coco", role)
        if len(data["samples"]) != int(PROTOCOL["roles"][role]):
            raise RuntimeError(f"Canonical {role} population changed")
        _ROLE_CACHE[role] = data
    return _ROLE_CACHE[role]


def category_ids(role: str = "fit") -> list[int]:
    gt = read_json(COMMON / role / "gt.json")
    values = [int(item["id"]) for item in gt["categories"]]
    if len(values) != 80 or len(values) != len(set(values)):
        raise RuntimeError("Canonical COCO category schema changed")
    return values


def reward_path(role: str) -> Path:
    return REWARD_ROOT / PROTOCOL["reward_files"][role]


def reward_map(role: str) -> dict[int, dict[str, float]]:
    values = load_pickle(reward_path(role))
    expected_ids = {int(sample["image_id"]) for sample in load_role(role)["samples"]}
    if set(map(int, values)) != expected_ids:
        raise RuntimeError(f"{role} reward population changed")
    return {
        int(image_id): {str(action): float(value) for action, value in actions.items()}
        for image_id, actions in values.items()
    }


def records_for_role(role: str) -> list[dict[str, Any]]:
    data = load_role(role)
    return build_action_records(
        data["samples"], data["actions"]["edge"], category_ids=category_ids()
    )


def rewards_for_records(
    records: Sequence[Mapping[str, Any]], values: Mapping[int, Mapping[str, float]]
) -> np.ndarray:
    output = np.zeros(
        (len(records), int(PROTOCOL["model"]["max_actions"])), dtype=np.float32
    )
    for image_index, record in enumerate(records):
        image_id = int(record["image_id"])
        for action_index, action in enumerate(record["actions"]):
            action_id = str(action["action_id"])
            if action_id == "edge":
                continue
            if action_id not in values[image_id]:
                raise RuntimeError(f"Missing reward for {image_id}/{action_id}")
            output[image_index, action_index] = float(values[image_id][action_id])
    return output


def baseline_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        value = dict(record)
        value["actions"] = [
            dict(action)
            for action in record["actions"]
            if str(action["action_id"]) in {"edge", "original"}
        ]
        output.append(value)
    return output


def subset_data(
    data: Mapping[str, torch.Tensor], indices: Sequence[int]
) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[index] for key, value in data.items()}


def source_files() -> list[Path]:
    paths = {
        PROTOCOL_PATH,
        SCRIPT,
        ROOT / "scripts/action_core.py",
        ROOT / "scripts/value_risk.py",
        reward_path("fit"),
        reward_path("calibration"),
        BENCHMARK / "models.json",
        BENCHMARK / "action_gate.json",
        CANONICAL_SCRIPTS / "run_multiranker_benchmark.py",
        SET_SCRIPTS / "experiment_core.py",
        SET_SCRIPTS / "run_experiment.py",
        VALUE_SCRIPTS / "oof_core.py",
        VALUE_SCRIPTS / "oof_train.py",
        TRIACTION_SCRIPTS / "run_exploration.py",
    }
    for role in PROTOCOL["roles"]:
        root = COMMON / role
        manifest = read_json(root / "source_manifest.json")
        paths.update({root / "samples.json", root / "gt.json", root / "source_manifest.json"})
        paths.update(Path(value) for value in manifest["action_sources"].values())
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen inputs:\n" + "\n".join(sorted(missing)))
    return sorted(path.resolve() for path in paths)


def capture_hashes() -> dict[str, str]:
    return {str(path): file_sha256(path) for path in source_files()}


def digest(hashes: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(hashes), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_preflight() -> dict[str, Any]:
    completion = RUNS / "preflight/completion.json"
    if completion.is_file():
        return read_json(RUNS / "preflight/report.json")
    samples = {
        role: read_json(COMMON / role / "samples.json") for role in PROTOCOL["roles"]
    }
    ids = {
        role: {int(item["image_id"]) for item in values}
        for role, values in samples.items()
    }
    if len(ids["fit"] & ids["calibration"]):
        raise RuntimeError("Fit and calibration overlap")
    for role, count in PROTOCOL["roles"].items():
        if len(ids[role]) != int(count):
            raise RuntimeError(f"Canonical {role} count changed")
        values = load_pickle(reward_path(role))
        if set(map(int, values)) != ids[role]:
            raise RuntimeError(f"{role} reward population changed")
    hashes = capture_hashes()
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "passed": True,
        "role_counts": {role: len(value) for role, value in ids.items()},
        "fit_calibration_overlap": 0,
        "source_sha256": hashes,
        "source_digest": digest(hashes),
        "actions": list(PROTOCOL["actions"]),
        "audit_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    atomic_json(RUNS / "preflight/report.json", report)
    atomic_json(completion, {"complete": True, "source_digest": report["source_digest"]})
    return report


def verify_hashes() -> dict[str, str]:
    expected = read_json(RUNS / "preflight/report.json")["source_sha256"]
    observed = capture_hashes()
    if observed != expected:
        raise RuntimeError("Frozen source hashes changed after preflight")
    return observed


def shared_action_map(role: str) -> dict[int, str]:
    data = load_role(role)
    models = tri.load_frozen_models(BENCHMARK / "models.json")
    gate = read_json(BENCHMARK / "action_gate.json")
    output = tri.current.chosen_actions(models["action_heads"], gate, data)
    if set(map(int, output)) != {int(sample["image_id"]) for sample in data["samples"]}:
        raise RuntimeError(f"Incomplete frozen SharedAction map for {role}")
    return {int(image_id): str(action) for image_id, action in output.items()}


def _train_models(
    train_records: list[dict[str, Any]],
    train_rewards: np.ndarray,
    train_data: Mapping[str, torch.Tensor],
    target_records: list[dict[str, Any]],
    target_data: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    protocol = set_protocol()
    seed = int(PROTOCOL["model"]["seeds"][0])
    categories = len(category_ids())
    image_ids = [int(record["image_id"]) for record in train_records]

    base_train_records = baseline_records(train_records)
    base_target_records = baseline_records(target_records)
    base_train_rewards = train_rewards[:, :2]
    base_train_data = _record_tensors(base_train_records, base_train_rewards, protocol)
    base_target_data = _record_tensors(base_target_records, None, protocol)
    baseline = fit_dual_router(
        base_train_records,
        base_train_data,
        base_train_rewards,
        protocol,
        category_count=categories,
        seed=seed,
    )
    baseline_scores = predict_dual_value(
        baseline,
        base_target_records,
        base_target_data,
        protocol,
        category_count=categories,
    )[0, :, 1]
    baseline_epoch = int(baseline["selected_epoch"])
    del baseline, base_train_data, base_target_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    value = fit_value_router(
        image_ids,
        train_data,
        protocol,
        category_count=categories,
        seed=seed,
    )
    value_scores = predict_value_router(
        value, target_data, protocol, category_count=categories
    )[0]
    value_epoch = int(value["selected_epoch"])
    del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    value_risk = fit_value_risk_router(
        image_ids,
        train_data,
        PROTOCOL,
        category_count=categories,
        seed=seed,
    )
    risk_scores = predict_value_risk_router(
        value_risk, target_data, PROTOCOL, category_count=categories
    )
    epochs = {
        "CurrentDualHead-SharedAction": baseline_epoch,
        "ActionValue-Joint": value_epoch,
        "ActionValueRiskLCB-Joint": int(value_risk["selected_epoch"]),
    }
    del value_risk
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "baseline": np.asarray(baseline_scores, dtype=np.float64),
        "value": np.asarray(value_scores, dtype=np.float64),
        "risk_mean": np.asarray(risk_scores["mean"], dtype=np.float64),
        "risk_lower": np.asarray(risk_scores["lower"], dtype=np.float64),
        "selected_epochs": epochs,
    }


def run_oof() -> dict[str, Any]:
    completion = RUNS / "oof/completion.json"
    if completion.is_file():
        return read_json(RUNS / "oof/report.json")
    if not (RUNS / "preflight/completion.json").is_file():
        raise RuntimeError("Preflight must complete first")
    role = load_role("fit")
    records = records_for_role("fit")
    rewards = rewards_for_records(records, reward_map("fit"))
    data = _record_tensors(records, rewards, set_protocol())
    image_ids = [int(record["image_id"]) for record in records]
    index_by_id = {image_id: index for index, image_id in enumerate(image_ids)}
    folds = deterministic_folds(
        image_ids,
        fold_count=int(PROTOCOL["fold_count"]),
        seed=int(PROTOCOL["fold_seed"]),
    )
    source_digest = read_json(RUNS / "preflight/report.json")["source_digest"]
    atomic_json(
        RUNS / "oof/fold_manifest.json",
        {"folds": folds, "source_digest": source_digest},
    )
    artifacts = []
    for fold_index, target_ids in enumerate(folds):
        path = RUNS / f"oof/folds/fold_{fold_index:02d}.pkl"
        if path.is_file():
            artifact = load_pickle(path)
            if artifact["source_digest"] != source_digest or artifact["target_ids"] != target_ids:
                raise RuntimeError(f"Fold {fold_index} cache binding changed")
            artifacts.append(artifact)
            print(f"[fold {fold_index}] reused", flush=True)
            continue
        target_indices = [index_by_id[image_id] for image_id in target_ids]
        target_set = set(target_indices)
        train_indices = [index for index in range(len(records)) if index not in target_set]
        started = time.time()
        predictions = _train_models(
            [records[index] for index in train_indices],
            rewards[train_indices],
            subset_data(data, train_indices),
            [records[index] for index in target_indices],
            subset_data(data, target_indices),
        )
        artifact = {
            "fold_index": fold_index,
            "target_ids": target_ids,
            "predictions": predictions,
            "seconds": time.time() - started,
            "source_digest": source_digest,
        }
        atomic_pickle(path, artifact)
        artifacts.append(artifact)
        print(f"[fold {fold_index}] complete in {artifact['seconds']/60:.1f} min", flush=True)
    seen: set[int] = set()
    assembled = {
        "baseline": np.full(len(records), np.nan),
        "value": np.full((len(records), 4), np.nan),
        "risk_mean": np.full((len(records), 4), np.nan),
        "risk_lower": np.full((len(records), 4), np.nan),
    }
    for artifact in artifacts:
        indices = [index_by_id[int(image_id)] for image_id in artifact["target_ids"]]
        if seen & set(artifact["target_ids"]):
            raise RuntimeError("An image received multiple OOF predictions")
        seen.update(map(int, artifact["target_ids"]))
        for name in assembled:
            assembled[name][indices] = artifact["predictions"][name]
    if seen != set(image_ids) or any(np.isnan(value).any() for value in assembled.values()):
        raise RuntimeError("OOF prediction population is incomplete")
    score_path = RUNS / "oof/scores.pkl"
    atomic_pickle(score_path, {"image_ids": image_ids, "scores": assembled})
    report = {
        "complete": True,
        "images": len(image_ids),
        "folds": len(folds),
        "fold_seconds": [float(item["seconds"]) for item in artifacts],
        "scores_sha256": file_sha256(score_path),
        "reward_context_may_cross_oof_folds": True,
        "calibration_accessed": False,
        "audit_accessed": False,
        "final_accessed": False,
    }
    atomic_json(RUNS / "oof/report.json", report)
    atomic_json(completion, report)
    del role
    return report


def run_calibration_scores() -> dict[str, Any]:
    completion = RUNS / "calibration_scores/completion.json"
    if completion.is_file():
        return read_json(RUNS / "calibration_scores/report.json")
    if not (RUNS / "oof/completion.json").is_file():
        raise RuntimeError("OOF must complete before the frozen calibration check")
    fit_records = records_for_role("fit")
    fit_rewards = rewards_for_records(fit_records, reward_map("fit"))
    fit_data = _record_tensors(fit_records, fit_rewards, set_protocol())
    calibration_records = records_for_role("calibration")
    calibration_data = _record_tensors(calibration_records, None, set_protocol())
    started = time.time()
    predictions = _train_models(
        fit_records,
        fit_rewards,
        fit_data,
        calibration_records,
        calibration_data,
    )
    score_path = RUNS / "calibration_scores/scores.pkl"
    atomic_pickle(
        score_path,
        {
            "image_ids": [int(record["image_id"]) for record in calibration_records],
            "scores": predictions,
        },
    )
    report = {
        "complete": True,
        "images": len(calibration_records),
        "seconds": time.time() - started,
        "scores_sha256": file_sha256(score_path),
        "calibration_used_for_training": False,
        "audit_accessed": False,
        "final_accessed": False,
    }
    atomic_json(RUNS / "calibration_scores/report.json", report)
    atomic_json(completion, report)
    return report


def _score_maps(
    records: Sequence[Mapping[str, Any]],
    scores: Mapping[str, np.ndarray],
    shared: Mapping[int, str],
) -> dict[str, dict[int, dict[str, float]]]:
    output = {method: {} for method in METHODS}
    for index, record in enumerate(records):
        image_id = int(record["image_id"])
        shared_action = str(shared[image_id])
        output["CurrentDualHead-SharedAction"][image_id] = {
            shared_action: float(scores["baseline"][index])
        }
        action_indices = {
            str(action["action_id"]): action_index
            for action_index, action in enumerate(record["actions"])
            if str(action["action_id"]) != "edge"
        }
        output["ActionValue-Joint"][image_id] = {
            action: float(scores["value"][index, action_index])
            for action, action_index in action_indices.items()
        }
        output["ActionValueRiskLCB-Joint"][image_id] = {
            action: float(scores["risk_lower"][index, action_index])
            for action, action_index in action_indices.items()
        }
    return output


def _evaluate_selection(
    *,
    role: str,
    method: str,
    mode: str,
    budget: float,
    data: dict[str, Any],
    selection: Mapping[int, str],
    schedule: Mapping[str, Any],
    rewards: Mapping[int, Mapping[str, float]],
    teacher_value: float,
    capacity_bytes: float,
) -> dict[str, Any]:
    predictions, _ = canonical.mixed_predictions(data, dict(selection), 0.001)
    official = data["evaluator"].evaluate(predictions)
    request_bytes = sum(
        int(data["sample_by_id"][int(image_id)]["payload_bytes"][str(action)])
        for image_id, action in selection.items()
    )
    selected_rewards = [
        float(rewards[int(image_id)][str(action)])
        for image_id, action in selection.items()
    ]
    roi_rewards = [
        float(rewards[int(image_id)]["roi"])
        for image_id, action in selection.items()
        if str(action) == "roi"
    ]
    counts = Counter(map(str, selection.values()))
    expected_calls = int(math.floor(len(data["samples"]) * float(budget) + 0.5))
    tolerance = float(PROTOCOL["gate"]["capacity_tolerance_bytes"])
    capacity_ok = (
        len(selection) == expected_calls
        if mode == "fixed_call"
        else request_bytes <= float(capacity_bytes) + tolerance
    )
    return {
        "role": role,
        "method": method,
        "mode": mode,
        "budget": float(budget),
        "official_ap50_95": float(official["map50_95"]),
        "official_ap50": float(official["map50"]),
        "official_ap75": float(official["map75"]),
        "kb_per_image": request_bytes / 1024.0 / len(data["samples"]),
        "request_bytes": int(request_bytes),
        "capacity_bytes": float(capacity_bytes),
        "capacity_ok": bool(capacity_ok),
        "selected_images": len(selection),
        "selected_fraction": len(selection) / len(data["samples"]),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["roi"]),
        "harmful_action_rate": sum(value <= 0.0 for value in selected_rewards)
        / max(1, len(selected_rewards)),
        "harmful_roi_rate": sum(value <= 0.0 for value in roi_rewards)
        / max(1, len(roi_rewards)),
        "selected_teacher_reward": float(sum(selected_rewards)),
        "teacher_reward_ceiling": float(teacher_value),
        "teacher_regret": float(teacher_value - sum(selected_rewards)),
        "predicted_objective": float(schedule["total_value"]),
        "scheduler_lambda": float(schedule.get("lambda", 0.0)),
    }


def evaluate_role(role: str, score_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = load_role(role)
    records = records_for_role(role)
    rewards = reward_map(role)
    score_artifact = load_pickle(score_path)
    image_ids = [int(record["image_id"]) for record in records]
    if list(map(int, score_artifact["image_ids"])) != image_ids:
        raise RuntimeError(f"{role} score/record ordering changed")
    scores = numeric_score_arrays(score_artifact)
    maps = _score_maps(records, scores, shared_action_map(role))
    samples = data["samples"]
    original_total = sum(float(sample["payload_bytes"]["original"]) for sample in samples)
    rows, selections = [], {}
    for budget_value in PROTOCOL["budgets"]:
        budget = float(budget_value)
        capacity = budget * original_total
        true_scores = {int(image_id): dict(values) for image_id, values in rewards.items()}
        teachers = {
            "fixed_call": schedule_fixed_call(samples, true_scores, budget=budget),
            "fixed_byte": schedule_fixed_byte(
                samples, true_scores, capacity_bytes=capacity
            ),
        }
        for method in METHODS:
            for mode in PROTOCOL["modes"]:
                schedule = (
                    schedule_fixed_call(samples, maps[method], budget=budget)
                    if mode == "fixed_call"
                    else schedule_fixed_byte(
                        samples, maps[method], capacity_bytes=capacity
                    )
                )
                key = f"{method}|{mode}|{budget:g}"
                selections[key] = {
                    str(image_id): action
                    for image_id, action in schedule["selection"].items()
                }
                rows.append(
                    _evaluate_selection(
                        role=role,
                        method=method,
                        mode=str(mode),
                        budget=budget,
                        data=data,
                        selection=schedule["selection"],
                        schedule=schedule,
                        rewards=rewards,
                        teacher_value=float(teachers[str(mode)]["total_value"]),
                        capacity_bytes=capacity,
                    )
                )
    if len(rows) != len(METHODS) * len(PROTOCOL["modes"]) * len(PROTOCOL["budgets"]):
        raise RuntimeError(f"Incomplete result matrix for {role}")
    return rows, selections


def _candidate_gate(
    rows: Sequence[Mapping[str, Any]], candidate: str
) -> dict[str, Any]:
    comparisons = []
    for mode in PROTOCOL["modes"]:
        for budget in map(float, PROTOCOL["budgets"]):
            baseline = next(
                row
                for row in rows
                if row["method"] == "CurrentDualHead-SharedAction"
                and row["mode"] == mode
                and float(row["budget"]) == budget
            )
            value = next(
                row
                for row in rows
                if row["method"] == candidate
                and row["mode"] == mode
                and float(row["budget"]) == budget
            )
            comparisons.append(
                {
                    "mode": str(mode),
                    "budget": budget,
                    "delta_ap": float(value["official_ap50_95"])
                    - float(baseline["official_ap50_95"]),
                    "capacity_ok": bool(value["capacity_ok"])
                    and bool(baseline["capacity_ok"]),
                    "candidate_harmful_roi_rate": float(value["harmful_roi_rate"]),
                    "baseline_harmful_roi_rate": float(baseline["harmful_roi_rate"]),
                }
            )
    gate = evaluate_gate(comparisons, PROTOCOL)
    gate["candidate"] = candidate
    gate["comparisons"] = comparisons
    return gate


def _write_notes(report: Mapping[str, Any]) -> None:
    rows = report["rows"]
    lines = [
        "# COCO Action-Conditioned Value–Risk OOF 诊断结果",
        "",
        "> 只使用 canonical train2017 fit=9,678 做5-fold OOF，并在 calibration=571 做一次冻结验证。未访问 audit、val2017/final 或 registered final。",
        "",
    ]
    for role, title in (("fit_oof", "5-fold OOF"), ("calibration", "Frozen calibration")):
        lines.extend(
            [
                f"## {title}",
                "",
                "| 模式 | 预算 | 方法 | Official AP | AP75 | KB/图 | O/Q90/ROI | 有害ROI率 |",
                "|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for mode in PROTOCOL["modes"]:
            for budget in map(float, PROTOCOL["budgets"]):
                for method in METHODS:
                    row = next(
                        item
                        for item in rows
                        if item["role"] == role
                        and item["method"] == method
                        and item["mode"] == mode
                        and float(item["budget"]) == budget
                    )
                    lines.append(
                        f"| {mode} | {budget:.0%} | {method} | "
                        f"{float(row['official_ap50_95']):.6f} | "
                        f"{float(row['official_ap75']):.6f} | "
                        f"{float(row['kb_per_image']):.2f} | "
                        f"{row['original_count']}/{row['q90_count']}/{row['roi_count']} | "
                        f"{float(row['harmful_roi_rate']):.2%} |"
                    )
        lines.extend(["", "### Gate", ""])
        for candidate, gate in report["gates"][role].items():
            lines.append(f"- {candidate}: `passed={str(gate['passed']).lower()}`")
        lines.append("")
    lines.extend(
        [
            "## Final decision",
            "",
            f"`open_three_seed_paper_protocol={str(report['open_three_seed_paper_protocol']).lower()}`",
            "",
            "预计算 contextual reward 的32图 context可能跨OOF fold，因此本实验是机制诊断，不是论文final。",
            "",
        ]
    )
    path = ROOT / "notes/results_zh.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def run_evaluation() -> dict[str, Any]:
    completion = RUNS / "evaluation/completion.json"
    if completion.is_file():
        return read_json(RUNS / "evaluation/report.json")
    if not (RUNS / "oof/completion.json").is_file() or not (
        RUNS / "calibration_scores/completion.json"
    ).is_file():
        raise RuntimeError("OOF and calibration scores must complete first")
    oof_rows, oof_selections = evaluate_role("fit", RUNS / "oof/scores.pkl")
    for row in oof_rows:
        row["role"] = "fit_oof"
    calibration_rows, calibration_selections = evaluate_role(
        "calibration", RUNS / "calibration_scores/scores.pkl"
    )
    rows = [*oof_rows, *calibration_rows]
    gates = {
        role: {
            candidate: _candidate_gate(
                [row for row in rows if row["role"] == role], candidate
            )
            for candidate in METHODS[1:]
        }
        for role in ("fit_oof", "calibration")
    }
    open_paper = bool(
        gates["fit_oof"]["ActionValueRiskLCB-Joint"]["passed"]
        and gates["calibration"]["ActionValueRiskLCB-Joint"]["passed"]
    )
    hashes = verify_hashes()
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "result_type": "single_seed_five_fold_oof_mechanism_diagnostic",
        "rows": rows,
        "gates": gates,
        "open_three_seed_paper_protocol": open_paper,
        "source_digest": digest(hashes),
        "reward_context_may_cross_oof_folds": True,
        "audit_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    atomic_csv(RUNS / "evaluation/results.csv", rows)
    atomic_json(
        RUNS / "evaluation/selections.json",
        {"fit_oof": oof_selections, "calibration": calibration_selections},
    )
    atomic_json(RUNS / "evaluation/report.json", report)
    _write_notes(report)
    atomic_json(
        completion,
        {"complete": True, "open_three_seed_paper_protocol": open_paper},
    )
    atomic_json(
        ROOT / "COMPLETED.json",
        {
            "complete": True,
            "open_three_seed_paper_protocol": open_paper,
            "audit_accessed": False,
            "final_accessed": False,
            "registered_final_accessed": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "oof", "calibration_scores", "evaluation", "all"),
        default="all",
    )
    arguments = parser.parse_args()
    reports = {}
    for stage in ("preflight", "oof", "calibration_scores", "evaluation"):
        if arguments.stage not in (stage, "all"):
            continue
        print(f"[stage] {stage}", flush=True)
        reports[stage] = globals()[f"run_{stage}"]()
    if "evaluation" in reports:
        print(
            json.dumps(
                {
                    "gates": reports["evaluation"]["gates"],
                    "open_three_seed_paper_protocol": reports["evaluation"][
                        "open_three_seed_paper_protocol"
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
