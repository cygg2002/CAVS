#!/usr/bin/env python3
"""Run the frozen COCO/Oregon full scheduler benchmark using cached detections."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from full_benchmark_core import (
    assert_fixed_byte_capacity,
    assert_primary_actions,
    source_tier,
    validate_protocol,
    validate_unique_rows,
)


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
validate_protocol(PROTOCOL)

SOURCES = {name: REPO / path for name, path in PROTOCOL["source_experiments"].items()}
EXTERNAL = SOURCES["external"]
CANONICAL = SOURCES["canonical"]
NINERANKER = SOURCES["nineranker"]
CROSSDATASET = SOURCES["crossdataset"]
CROSSCAL = SOURCES["cross_action_calibration"]
COCO_ACTIONVALUE = REPO / "paper/experiments/20260805_coco_actionconditioned_valuerisk_oof_v1"

sys.dont_write_bytecode = True


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def complete(path: Path) -> bool:
    return path.is_file() and read_json(path).get("complete") is True


_MODULES: dict[str, Any] = {}


def modules() -> dict[str, Any]:
    if _MODULES:
        return _MODULES
    sys.path.insert(0, str(EXTERNAL / "scripts"))
    external = _load_module("full_benchmark_external", EXTERNAL / "scripts/run_dataset.py")
    sys.path.insert(0, str(CROSSDATASET / "scripts"))
    cross = _load_module("full_benchmark_crossdataset", CROSSDATASET / "scripts/run_crossdataset.py")
    sys.path.insert(0, str(CROSSCAL / "scripts"))
    calibration = _load_module("full_benchmark_calibration", CROSSCAL / "scripts/calibration_core.py")
    _MODULES.update(external=external, cross=cross, calibration=calibration)
    return _MODULES


def canonical_layout(dataset: str) -> dict[str, Path]:
    run = PROTOCOL["datasets"][dataset]["canonical_run"]
    base = CANONICAL / "runs" / run
    return {"common": base / "common", "benchmark": base / "benchmark" / dataset}


def load_context(dataset: str) -> dict[str, Any]:
    values = modules()
    layout = canonical_layout(dataset)
    loader = values["external"].load_frozen_benchmark(layout["common"])
    loader.COMMON = layout["common"]
    aliases = {"calibration": "calibration", "final": PROTOCOL["datasets"][dataset]["final_alias"]}
    roles = {role: loader.load_role(dataset, alias) for role, alias in aliases.items()}
    expected = PROTOCOL["datasets"][dataset]["roles"]
    for role, data in roles.items():
        if len(data["samples"]) != int(expected[role]):
            raise RuntimeError(f"{dataset}/{role} count changed")
    models = read_json(layout["benchmark"] / "models.json")
    gate = read_json(layout["benchmark"] / "action_gate.json")
    return {"loader": loader, "layout": layout, "roles": roles, "models": models, "gate": gate}


def _identity_set(data: Mapping[str, Any]) -> set[str]:
    return {str(sample["identity"]) for sample in data["samples"]}


def required_sources(dataset: str) -> list[Path]:
    shared = [
        ROOT / "configs/protocol.json",
        SCRIPT,
        ROOT / "scripts/full_benchmark_core.py",
        EXTERNAL / "configs/protocol.json",
        NINERANKER / "configs/protocol.json",
        CROSSCAL / f"runs/{dataset}/COMPLETED.json",
        CROSSCAL / f"runs/{dataset}/calibration/scores.pkl",
        canonical_layout(dataset)["benchmark"] / "models.json",
        canonical_layout(dataset)["benchmark"] / "action_gate.json",
    ]
    if dataset == "coco":
        shared.extend(
            [
                SOURCES["coco_noroi"] / "runs/point/completion.json",
                COCO_ACTIONVALUE / "runs/calibration_scores/scores.pkl",
                COCO_ACTIONVALUE / "configs/protocol.json",
            ]
        )
    else:
        shared.extend(
            [
                CROSSDATASET / "runs/oregon/COMPLETED.json",
                CROSSDATASET / "runs/oregon/scores/calibration.pkl",
                CROSSDATASET / "runs/oregon/scores/final.pkl",
            ]
        )
    shared.extend(
        [
            EXTERNAL / f"runs/{dataset}/completion.json",
            EXTERNAL / f"runs/{dataset}/evaluation/oric_mlp.pkl",
            EXTERNAL / f"runs/{dataset}/evaluation/moric_mlp.pkl",
            EXTERNAL / f"runs/{dataset}/rewards/dcsb_model.json",
            NINERANKER / f"runs/{dataset}/COMPLETED",
            NINERANKER / f"runs/{dataset}/models/current_controlled_32d.json",
        ]
    )
    return shared


def run_preflight(dataset: str) -> dict[str, Any]:
    output = RUNS / dataset / "preflight"
    if complete(output / "completion.json"):
        return read_json(output / "report.json")
    context = load_context(dataset)
    missing = [str(path) for path in required_sources(dataset) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen sources:\n" + "\n".join(missing))
    fit_alias = "fit"
    fit = context["loader"].load_role(dataset, fit_alias)
    expected = PROTOCOL["datasets"][dataset]["roles"]
    if len(fit["samples"]) != int(expected["fit"]):
        raise RuntimeError(f"{dataset}/fit count changed")
    overlap = {
        "fit_calibration": len(_identity_set(fit) & _identity_set(context["roles"]["calibration"])),
        "fit_final": len(_identity_set(fit) & _identity_set(context["roles"]["final"])),
        "calibration_final": len(_identity_set(context["roles"]["calibration"]) & _identity_set(context["roles"]["final"])),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Identity leakage for {dataset}: {overlap}")
    hashes = {str(path.resolve()): file_sha256(path) for path in required_sources(dataset)}
    report = {
        "dataset": dataset,
        "protocol_id": PROTOCOL["protocol_id"],
        "role_counts": {"fit": len(fit["samples"]), **{role: len(data["samples"]) for role, data in context["roles"].items()}},
        "identity_overlap": overlap,
        "source_sha256": hashes,
        "source_digest": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "detector_inference": False,
        "gpu_required": False,
        "final_used_for_training_or_selection": False,
        "passed": True,
    }
    atomic_json(output / "report.json", report)
    atomic_json(output / "completion.json", {"complete": True, "source_digest": report["source_digest"]})
    return report


def verify_preflight(dataset: str) -> None:
    report = read_json(RUNS / dataset / "preflight/report.json")
    observed = {str(path.resolve()): file_sha256(path) for path in required_sources(dataset)}
    if observed != report["source_sha256"]:
        raise RuntimeError(f"Frozen source changed after preflight for {dataset}")


def _filter_oq_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        item = dict(record)
        item["actions"] = [dict(action) for action in record["actions"] if str(action["action_id"]) in {"edge", "original", "q90"}]
        if [str(action["action_id"]) for action in item["actions"]] != ["edge", "original", "q90"]:
            raise RuntimeError("O/Q action ordering changed")
        output.append(item)
    return output


def _coco_actionvalue_scores() -> dict[str, Any]:
    output = RUNS / "coco/actionvalue"
    score_path = output / "scores.pkl"
    if complete(output / "completion.json"):
        return load_pickle(score_path)
    sys.path.insert(0, str(COCO_ACTIONVALUE / "scripts"))
    source = _load_module("full_benchmark_coco_actionvalue", COCO_ACTIONVALUE / "scripts/run_oof.py")
    import torch

    torch.set_num_threads(int(PROTOCOL["workers"]["torch_threads_per_process"]))
    source._configure_canonical()
    fit_records = source.records_for_role("fit")
    fit_rewards = source.rewards_for_records(fit_records, source.reward_map("fit"))
    tensor_protocol = source.set_protocol()
    train_data = source._record_tensors(fit_records, fit_rewards, tensor_protocol)
    checkpoint = source.fit_value_router(
        [int(record["image_id"]) for record in fit_records],
        train_data,
        tensor_protocol,
        category_count=len(source.category_ids()),
        seed=int(source.PROTOCOL["model"]["seeds"][0]),
        device=torch.device("cpu"),
    )
    role_scores = {}
    role_ids = {}
    for role, alias in (("calibration", "calibration"), ("final", "final")):
        data = source.canonical.load_role("coco", alias)
        records = source.build_action_records(data["samples"], data["actions"]["edge"], category_ids=source.category_ids())
        tensors = source._record_tensors(records, None, tensor_protocol)
        values = np.asarray(source.predict_value_router(checkpoint, tensors, tensor_protocol, category_count=len(source.category_ids())), dtype=np.float64)
        role_scores[role] = values[:, :, :3]
        role_ids[role] = [int(sample["image_id"]) for sample in data["samples"]]
    frozen = load_pickle(COCO_ACTIONVALUE / "runs/calibration_scores/scores.pkl")
    expected = np.repeat(np.asarray(frozen["scores"]["value"], dtype=np.float64)[None, :, :3], len(PROTOCOL["budgets"]), axis=0)
    maximum_difference = float(np.max(np.abs(role_scores["calibration"] - expected)))
    correlation = float(np.corrcoef(role_scores["calibration"].ravel(), expected.ravel())[0, 1])
    if not np.isfinite(correlation) or correlation < 0.99 or maximum_difference > 0.05:
        raise RuntimeError(
            "Fresh COCO CPU full-fit model is incompatible with the frozen "
            f"calibrator: correlation={correlation}, max_diff={maximum_difference}"
        )
    payload = {"raw": role_scores, "image_ids": role_ids, "selected_epoch": int(checkpoint["selected_epoch"]), "fit_only": True, "final_used_for_selection": False}
    atomic_pickle(output / "checkpoint.pkl", checkpoint)
    atomic_pickle(score_path, payload)
    atomic_json(output / "completion.json", {"complete": True, "scores_sha256": file_sha256(score_path), "calibration_correlation": correlation, "calibration_max_abs_difference": maximum_difference})
    return payload


def raw_actionvalue_scores(dataset: str, context: Mapping[str, Any]) -> dict[str, Any]:
    if dataset == "coco":
        return _coco_actionvalue_scores()
    payload = {"raw": {}, "image_ids": {}}
    for role in ("calibration", "final"):
        artifact = load_pickle(CROSSDATASET / f"runs/oregon/scores/{role}.pkl")
        payload["raw"][role] = np.asarray(artifact["ensemble"]["ActionValue-OQ-Joint"], dtype=np.float64)
        payload["image_ids"][role] = list(map(int, artifact["image_ids"]))
    payload.update(fit_only=True, final_used_for_selection=False)
    return payload


def calibrated_actionvalue_scores(dataset: str, raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    artifact = load_pickle(CROSSCAL / f"runs/{dataset}/calibration/scores.pkl")
    calibration = modules()["calibration"]
    output = {}
    for role in ("calibration", "final"):
        values = np.asarray(raw["raw"][role], dtype=np.float64).copy()
        values[:, :, 0] = 0.0
        for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
            for action_index, action in enumerate(("original", "q90"), start=1):
                model = artifact["models"][f"{budget:g}|{action}"]["full_fit"]
                values[budget_index, :, action_index] = calibration.apply_isotonic(model, values[budget_index, :, action_index])
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite calibrated scores for {dataset}/{role}")
        output[role] = values
    return output


def no_roi_action_maps(context: Mapping[str, Any]) -> dict[str, dict[str, dict[int, str]]]:
    loader, models, gate = context["loader"], context["models"], context["gate"]
    output = {}
    for role, data in context["roles"].items():
        q90_probability, _ = loader.action_probabilities(models["action_heads"], data)
        threshold = float(gate["chosen"]["q90_threshold"])
        original, shared = {}, {}
        for sample, probability in zip(data["samples"], q90_probability):
            image_id = int(sample["image_id"])
            original[image_id] = "original"
            shared[image_id] = "q90" if bool(sample.get("available", {}).get("q90", False)) and float(probability) >= threshold else "original"
        assert_primary_actions(original)
        assert_primary_actions(shared)
        output[role] = {"Original-only": original, "SharedAction-NoROI": shared}
    return output


def _align(ids: Sequence[int], data: Mapping[str, Any], values: np.ndarray) -> np.ndarray:
    expected = [int(sample["image_id"]) for sample in data["samples"]]
    if list(map(int, ids)) != expected or np.asarray(values).shape[-1] == 0:
        raise RuntimeError("Score/image order changed")
    return np.asarray(values, dtype=np.float64)


def scalar_scores(dataset: str, context: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    budgets = len(PROTOCOL["budgets"])
    output = {role: {} for role in ("calibration", "final")}
    loader, models = context["loader"], context["models"]
    for role, data in context["roles"].items():
        historical = loader.ranker_scores(models, data)
        for method in PROTOCOL["historical_rankers"]:
            if method in {"Random", "Current-Controlled-32D"}:
                continue
            vector = np.asarray(historical[method], dtype=np.float64)
            output[role][method] = np.repeat(vector[None, :], budgets, axis=0)
        controlled = read_json(NINERANKER / f"runs/{dataset}/models/current_controlled_32d.json")
        pca = np.asarray([sample["pca32"] for sample in data["samples"]], dtype=np.float64)
        vector = np.asarray(loader.predict_pairwise_ranker(controlled, pca), dtype=np.float64)
        output[role]["Current-Controlled-32D"] = np.repeat(vector[None, :], budgets, axis=0)
    if dataset == "coco":
        path = SOURCES["coco_noroi"] / "runs/scores/frozen_scores.npz"
        with np.load(path, allow_pickle=False) as artifact:
            methods = list(map(str, artifact["methods"]))
            for role in output:
                ids = artifact[f"{role}_ids"]
                for method in PROTOCOL["internal_rankers"]:
                    values = artifact[f"{role}_scores"][methods.index(method)]
                    output[role][method] = _align(ids, context["roles"][role], values)
    else:
        mapping = {"Current-BudgetSet": "Current-BudgetSet-SharedAction", "SetValue": "SetValue-SharedAction", "DualHead": "DualHead-SharedAction"}
        for role in output:
            artifact = load_pickle(CROSSDATASET / f"runs/oregon/scores/{role}.pkl")
            for method, source_name in mapping.items():
                output[role][method] = _align(artifact["image_ids"], context["roles"][role], artifact["ensemble"][source_name])
    external = modules()["external"]
    checkpoints = {"ORIC-MLP": load_pickle(EXTERNAL / f"runs/{dataset}/evaluation/oric_mlp.pkl"), "MORIC-MLP": load_pickle(EXTERNAL / f"runs/{dataset}/evaluation/moric_mlp.pkl")}
    fit_artifact = np.load(EXTERNAL / f"runs/{dataset}/rewards/fit_artifacts.npz")
    categories = list(map(int, fit_artifact["category_ids"]))
    dcsb = read_json(EXTERNAL / f"runs/{dataset}/rewards/dcsb_model.json")
    for role, data in context["roles"].items():
        features = external._role_features(data, categories, 25)
        for method, checkpoint in checkpoints.items():
            vector = np.asarray(external.predict_reward_mlp(checkpoint, features), dtype=np.float64)
            output[role][method] = np.repeat(vector[None, :], budgets, axis=0)
        adapted = []
        for sample in data["samples"]:
            image_id = int(sample["image_id"])
            _, score = external.dcsb_decision_and_score(data["actions"]["edge"].get(image_id, []), int(sample["width"]), int(sample["height"]), dcsb)
            adapted.append(score)
        output[role]["DCSB-Score-Adapted"] = np.repeat(np.asarray(adapted, dtype=np.float64)[None, :], budgets, axis=0)
    expected = (set(PROTOCOL["historical_rankers"]) - {"Random"}) | set(PROTOCOL["internal_rankers"]) | {"DCSB-Score-Adapted", "ORIC-MLP", "MORIC-MLP"}
    for role in output:
        if set(output[role]) != expected:
            raise RuntimeError(f"Scalar method matrix changed for {dataset}/{role}: {sorted(output[role])}")
    return output


def actionvalue_map(data: Mapping[str, Any], values: np.ndarray) -> dict[int, dict[str, float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(data["samples"]), 3):
        raise RuntimeError(f"ActionValue shape changed: {array.shape}")
    return {
        int(sample["image_id"]): {"original": float(array[index, 1]), "q90": float(array[index, 2])}
        for index, sample in enumerate(data["samples"])
    }


def _selected_bytes(data: Mapping[str, Any], actions: Mapping[int, str]) -> float:
    return sum(float(data["sample_by_id"][int(image_id)]["payload_bytes"][str(action)]) for image_id, action in actions.items())


_EVAL_STATE: dict[str, Any] = {}


def _evaluate_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    loader = _EVAL_STATE["loader"]
    data = _EVAL_STATE["data"]
    models = _EVAL_STATE["models"]
    actions = {int(image_id): str(action) for image_id, action in task["actions"].items()}
    if task["policy"] != "system":
        assert_primary_actions(actions)
    row = loader.evaluate(str(task["method"]), str(task["mode"]), float(task["budget"]), data, actions, models)
    capacity = task.get("capacity")
    result = {
        **row,
        "dataset": _EVAL_STATE["dataset"],
        "method": str(task["method"]),
        "policy": str(task["policy"]),
        "mode": str(task["mode"]),
        "budget": float(task["budget"]),
        "seed": task.get("seed", "aggregate"),
        "source_tier": "canonical_primary",
        "detector_protocol": "cloud_v1",
        "final_used_for_selection": False,
    }
    if capacity is not None:
        utilization = assert_fixed_byte_capacity(data["samples"], actions, capacity=float(capacity))
        result.update(capacity_bytes=float(capacity), used_capacity_bytes=_selected_bytes(data, actions), capacity_utilization=utilization)
    return result


def _evaluate_tasks(dataset: str, context: Mapping[str, Any], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _EVAL_STATE.clear()
    _EVAL_STATE.update(dataset=dataset, loader=context["loader"], data=context["roles"]["final"], models=context["models"])
    workers = min(int(PROTOCOL["workers"]["point_per_dataset"]), max(1, (os.cpu_count() or 1) - 2), len(tasks))
    ctx = mp.get_context("fork")
    try:
        with ctx.Pool(workers) as pool:
            rows = []
            for index, row in enumerate(pool.imap_unordered(_evaluate_worker, tasks, chunksize=1), start=1):
                rows.append(row)
                if index % 25 == 0 or index == len(tasks):
                    print(f"[{dataset}:point] {index}/{len(tasks)}", flush=True)
            return rows
    finally:
        _EVAL_STATE.clear()


def _schedule_scalar(
    context: Mapping[str, Any], scores: Mapping[str, Mapping[str, np.ndarray]], maps: Mapping[str, Mapping[str, Mapping[int, str]]]
) -> tuple[list[dict[str, Any]], dict[str, dict[int, str]], dict[str, Any]]:
    loader = context["loader"]
    external_protocol = read_json(EXTERNAL / "configs/protocol.json")
    tasks, selections, controls = [], {}, {}
    methods = sorted(scores["final"])
    for method in methods:
        for policy in PROTOCOL["scalar_payload_policies"]:
            for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
                cal_score = scores["calibration"][method][budget_index]
                final_score = scores["final"][method][budget_index]
                selected = loader.top_k_actions(context["roles"]["final"], final_score, maps["final"][policy], budget)
                key = f"{method}|{policy}|fixed_call|{budget:g}"
                selections[key] = selected
                tasks.append({"method": method, "policy": policy, "mode": "fixed_call", "budget": budget, "actions": selected})
                frozen = loader.calibrate_lambda(context["roles"]["calibration"], cal_score, maps["calibration"][policy], budget, external_protocol)
                priorities = loader.audit_priority(cal_score, final_score, context["roles"]["final"], maps["final"][policy], float(frozen["lambda"]), float(frozen["median_original_bytes"]))
                selected = loader.byte_selection(context["roles"]["final"], priorities, maps["final"][policy], budget)
                capacity = budget * sum(float(sample["payload_bytes"]["original"]) for sample in context["roles"]["final"]["samples"])
                key = f"{method}|{policy}|fixed_byte|{budget:g}"
                selections[key] = selected
                tasks.append({"method": method, "policy": policy, "mode": "fixed_byte", "budget": budget, "actions": selected, "capacity": capacity})
                controls[key] = {"lambda": float(frozen["lambda"]), "median_original_bytes": float(frozen["median_original_bytes"]), "calibration_teacher_gain": float(frozen["teacher_gain"])}
    return tasks, selections, controls


def _schedule_actionvalue(
    context: Mapping[str, Any], raw: Mapping[str, Any], calibrated: Mapping[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[str, dict[int, str]]]:
    cross = modules()["cross"]
    tasks, selections = [], {}
    candidates = {"Raw-ActionValue-OQ": raw["raw"], "PerAction-Isotonic-OQ": calibrated}
    for method, role_scores in candidates.items():
        for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
            values = actionvalue_map(context["roles"]["final"], role_scores["final"][budget_index])
            for mode in PROTOCOL["modes"]:
                if mode == "fixed_call":
                    scheduled = cross.schedule_fixed_call(context["roles"]["final"]["samples"], values, budget=budget)
                    capacity = None
                else:
                    capacity = budget * sum(float(sample["payload_bytes"]["original"]) for sample in context["roles"]["final"]["samples"])
                    scheduled = cross.schedule_fixed_byte(context["roles"]["final"]["samples"], values, capacity_bytes=capacity)
                selected = {int(image_id): str(action) for image_id, action in scheduled["selection"].items()}
                key = f"{method}|Joint-OQ|{mode}|{budget:g}"
                selections[key] = selected
                task = {"method": method, "policy": "Joint-OQ", "mode": mode, "budget": budget, "actions": selected}
                if capacity is not None:
                    task["capacity"] = capacity
                tasks.append(task)
    return tasks, selections


def _random_tasks(context: Mapping[str, Any], maps: Mapping[str, Mapping[str, Mapping[int, str]]]) -> tuple[list[dict[str, Any]], dict[str, dict[int, str]]]:
    loader = context["loader"]
    external_protocol = read_json(EXTERNAL / "configs/protocol.json")
    tasks, saved = [], {}
    for offset in range(int(PROTOCOL["random"]["seeds"])):
        seed = int(PROTOCOL["random"]["base_seed"]) + offset
        rng = np.random.default_rng(seed)
        cal_score = rng.random(len(context["roles"]["calibration"]["samples"]))
        final_score = rng.random(len(context["roles"]["final"]["samples"]))
        for policy in PROTOCOL["scalar_payload_policies"]:
            for budget in map(float, PROTOCOL["budgets"]):
                selected = loader.top_k_actions(context["roles"]["final"], final_score, maps["final"][policy], budget)
                tasks.append({"method": "Random", "policy": policy, "mode": "fixed_call", "budget": budget, "actions": selected, "seed": seed})
                frozen = loader.calibrate_lambda(context["roles"]["calibration"], cal_score, maps["calibration"][policy], budget, external_protocol)
                priorities = loader.audit_priority(cal_score, final_score, context["roles"]["final"], maps["final"][policy], float(frozen["lambda"]), float(frozen["median_original_bytes"]))
                selected = loader.byte_selection(context["roles"]["final"], priorities, maps["final"][policy], budget)
                capacity = budget * sum(float(sample["payload_bytes"]["original"]) for sample in context["roles"]["final"]["samples"])
                tasks.append({"method": "Random", "policy": policy, "mode": "fixed_byte", "budget": budget, "actions": selected, "capacity": capacity, "seed": seed})
                if offset == 0:
                    saved[f"Random|{policy}|fixed_call|{budget:g}"] = loader.top_k_actions(context["roles"]["final"], final_score, maps["final"][policy], budget)
                    saved[f"Random|{policy}|fixed_byte|{budget:g}"] = selected
    return tasks, saved


def aggregate_random(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["policy"]), str(row["mode"]), float(row["budget"]))].append(row)
    output = []
    for (policy, mode, budget), values in sorted(grouped.items()):
        item = dict(values[0])
        item["seed"] = "aggregate"
        for key in item:
            numeric = [value.get(key) for value in values]
            if numeric and all(isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) for value in numeric):
                item[key] = float(np.mean(numeric))
        item.update(method="Random", policy=policy, mode=mode, budget=budget, random_seeds=len(values))
        output.append(item)
    return output


def _native_and_system_tasks(dataset: str, context: Mapping[str, Any], maps: Mapping[str, Mapping[str, Mapping[int, str]]]) -> list[dict[str, Any]]:
    external = modules()["external"]
    dcsb = read_json(EXTERNAL / f"runs/{dataset}/rewards/dcsb_model.json")
    final = context["roles"]["final"]
    decisions = {}
    for sample in final["samples"]:
        image_id = int(sample["image_id"])
        decision, _ = external.dcsb_decision_and_score(final["actions"]["edge"].get(image_id, []), int(sample["width"]), int(sample["height"]), dcsb)
        decisions[image_id] = int(decision)
    tasks = []
    for policy in PROTOCOL["scalar_payload_policies"]:
        selected = {image_id: action for image_id, action in maps["final"][policy].items() if decisions[image_id] == 1}
        tasks.append({"method": "DCSB-Native", "policy": policy, "mode": "native_rule", "budget": -1.0, "actions": selected})
    original = {int(sample["image_id"]): "original" for sample in final["samples"]}
    q90 = {int(sample["image_id"]): ("q90" if bool(sample.get("available", {}).get("q90", False)) else "original") for sample in final["samples"]}
    tasks.extend(
        [
            {"method": "Edge-only", "policy": "system", "mode": "system", "budget": 0.0, "actions": {}},
            {"method": "Cloud-only Original", "policy": "system", "mode": "system", "budget": 0.0, "actions": original},
            {"method": "Cloud-only Q90", "policy": "system", "mode": "system", "budget": 0.0, "actions": q90},
        ]
    )
    return tasks


def run_point(dataset: str) -> dict[str, Any]:
    output = RUNS / dataset / "point"
    if complete(output / "completion.json"):
        return read_json(output / "report.json")
    verify_preflight(dataset)
    context = load_context(dataset)
    maps = no_roi_action_maps(context)
    scores = scalar_scores(dataset, context)
    raw = raw_actionvalue_scores(dataset, context)
    for role in ("calibration", "final"):
        expected_ids = [int(sample["image_id"]) for sample in context["roles"][role]["samples"]]
        if list(map(int, raw["image_ids"][role])) != expected_ids:
            raise RuntimeError(f"ActionValue identity alignment changed for {dataset}/{role}")
    calibrated = calibrated_actionvalue_scores(dataset, raw)
    scalar_tasks, scalar_selections, controls = _schedule_scalar(context, scores, maps)
    action_tasks, action_selections = _schedule_actionvalue(context, raw, calibrated)
    random_tasks, random_selections = _random_tasks(context, maps)
    other_tasks = _native_and_system_tasks(dataset, context, maps)
    deterministic_rows = _evaluate_tasks(dataset, context, scalar_tasks + action_tasks + other_tasks)
    random_seed_rows = _evaluate_tasks(dataset, context, random_tasks)
    rows = deterministic_rows + aggregate_random(random_seed_rows)
    validate_unique_rows(rows)
    selections = {**scalar_selections, **action_selections, **random_selections}
    # Freeze the strongest external system using calibration-only teacher gain.
    frozen_external = {}
    for budget in map(float, PROTOCOL["budgets"]):
        candidates = [
            (float(value["calibration_teacher_gain"]), key)
            for key, value in controls.items()
            if key.split("|")[0] in {"DCSB-Score-Adapted", "ORIC-MLP", "MORIC-MLP"}
            and key.endswith(f"fixed_byte|{budget:g}")
        ]
        frozen_external[f"{budget:g}"] = max(candidates, key=lambda item: (item[0], item[1]))[1]
    verify_preflight(dataset)
    atomic_csv(output / "final_results.csv", rows)
    atomic_csv(output / "random_seed_results.csv", random_seed_rows)
    atomic_json(output / "selections.json", {key: {str(image_id): action for image_id, action in value.items()} for key, value in selections.items()})
    atomic_json(output / "controls.json", {"fixed_byte": controls, "strongest_external": frozen_external})
    report = {
        "dataset": dataset,
        "rows": len(rows),
        "random_seed_rows": len(random_seed_rows),
        "frozen_external": frozen_external,
        "actionvalue_fit_only": True,
        "final_used_for_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(output / "completion.json", {"complete": True, "results_sha256": file_sha256(output / "final_results.csv"), "selections_sha256": file_sha256(output / "selections.json")})
    return report


def _bootstrap_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "probability_superior": float(np.mean(values > 0.0)),
    }


def _selection(values: Mapping[str, Mapping[str, str]], key: str) -> dict[int, str]:
    if key not in values:
        raise RuntimeError(f"Missing frozen selection: {key}")
    actions = {int(image_id): str(action) for image_id, action in values[key].items()}
    assert_primary_actions(actions)
    return actions


def run_bootstrap(dataset: str) -> dict[str, Any]:
    output = RUNS / dataset / "bootstrap"
    if complete(output / "completion.json"):
        return read_json(output / "report.json")
    verify_preflight(dataset)
    point = RUNS / dataset / "point"
    if not complete(point / "completion.json"):
        raise RuntimeError("Point evaluation must complete before bootstrap")
    context = load_context(dataset)
    final = context["roles"]["final"]
    stored = read_json(point / "selections.json")
    controls = read_json(point / "controls.json")
    sys.path.insert(0, str(REPO / "paper/experiments/20260804_coco_dualhead_canonical9678_val2017_paper_v1/scripts"))
    core = _load_module(f"full_benchmark_bootstrap_{dataset}", REPO / "paper/experiments/20260804_coco_dualhead_canonical9678_val2017_paper_v1/scripts/canonical_core.py")
    image_ids = list(map(int, final["image_ids"]))
    ground_truth = read_json(Path(final["gt_path"]))
    comparisons = []
    for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
        candidate_key = f"PerAction-Isotonic-OQ|Joint-OQ|fixed_byte|{budget:g}"
        external_key = str(controls["strongest_external"][f"{budget:g}"])
        confidence_candidates = [
            (float(value["calibration_teacher_gain"]), key)
            for key, value in controls["fixed_byte"].items()
            if key.startswith("Confidence|") and key.endswith(f"fixed_byte|{budget:g}")
        ]
        confidence_key = max(confidence_candidates, key=lambda item: (item[0], item[1]))[1]
        random_key = f"Random|SharedAction-NoROI|fixed_byte|{budget:g}"
        for comparison_type, comparator_key in (
            ("strongest_external", external_key),
            ("confidence", confidence_key),
            ("random_seed_20260720", random_key),
        ):
            result_path = output / f"{budget:g}_{comparison_type}.json"
            if result_path.is_file():
                item = read_json(result_path)
                values = np.asarray(item["values"], dtype=np.float64)
            else:
                candidate = _selection(stored, candidate_key)
                comparator = _selection(stored, comparator_key)
                candidate_predictions, _ = context["loader"].mixed_predictions(final, candidate, 0.001)
                comparator_predictions, _ = context["loader"].mixed_predictions(final, comparator, 0.001)
                seed = int(PROTOCOL["bootstrap"]["seed"]) + budget_index * 101
                generator = np.random.default_rng(seed)
                indices = generator.integers(0, len(image_ids), size=(int(PROTOCOL["bootstrap"]["replicates"]), len(image_ids)), dtype=np.int32)
                core.configure_bootstrap(ground_truth, image_ids, candidate_predictions, comparator_predictions)
                values = np.empty(int(PROTOCOL["bootstrap"]["replicates"]), dtype=np.float64)
                workers = min(int(PROTOCOL["bootstrap"]["workers_per_dataset"]), max(1, (os.cpu_count() or 1) - 2))
                ctx = mp.get_context("fork")
                try:
                    with ctx.Pool(workers) as pool:
                        tasks = ((index, indices[index]) for index in range(len(indices)))
                        for count, (replicate, delta) in enumerate(pool.imap_unordered(core.bootstrap_worker, tasks, chunksize=1), start=1):
                            values[int(replicate)] = float(delta)
                            if count % 100 == 0 or count == len(indices):
                                print(f"[{dataset}:bootstrap:{comparison_type}:{budget:g}] {count}/{len(indices)}", flush=True)
                finally:
                    core.clear_bootstrap()
                item = {
                    "dataset": dataset,
                    "budget": budget,
                    "candidate": "PerAction-Isotonic-OQ",
                    "candidate_key": candidate_key,
                    "comparator": comparator_key.split("|")[0],
                    "comparator_key": comparator_key,
                    "comparison_type": comparison_type,
                    "replicates": len(values),
                    "seed": seed,
                    "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                    "resampling_unit": "image" if dataset == "coco" else "image_sensitivity_group_metadata_unavailable",
                    "values": values.tolist(),
                }
                atomic_json(result_path, item)
            comparisons.append({key: value for key, value in item.items() if key != "values"} | _bootstrap_summary(values))
    report = {
        "dataset": dataset,
        "comparisons": comparisons,
        "replicates_per_comparison": int(PROTOCOL["bootstrap"]["replicates"]),
        "oregon_limitation": "Event/camera group IDs are absent from canonical samples; image bootstrap is sensitivity-only." if dataset == "oregon" else None,
        "final_used_for_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(output / "completion.json", {"complete": True, "comparisons": len(comparisons), "report_sha256": file_sha256(output / "report.json")})
    return report


def _copy_csv_rows(source: Path, dataset: str, tier: str) -> list[dict[str, Any]]:
    if not source.is_file():
        return []
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    return [{**row, "dataset": dataset, "source_experiment": source.parent.parent.name, "comparison_tier": tier, "source_file": str(source)} for row in rows]


def run_integrate(dataset: str) -> dict[str, Any]:
    output = RUNS / dataset / "extensions"
    if complete(output / "completion.json"):
        return read_json(output / "report.json")
    canonical_rows = []
    canonical_rows.extend(_copy_csv_rows(NINERANKER / f"runs/{dataset}/results.csv", dataset, "canonical_action_extension"))
    if dataset == "coco":
        canonical_rows.extend(_copy_csv_rows(SOURCES["coco_triaction"] / "runs/point/final_results.csv", dataset, "canonical_action_extension"))
    else:
        canonical_rows.extend(_copy_csv_rows(CROSSDATASET / "runs/oregon/track_b/results.csv", dataset, "canonical_action_extension"))
    roi_rows = _copy_csv_rows(SOURCES["roi_gate"] / f"runs/{dataset}/evaluation/results.csv", dataset, "development_roi_diagnostic")
    if canonical_rows:
        atomic_csv(output / "canonical_action_results.csv", canonical_rows)
    if roi_rows:
        atomic_csv(output / "roi_diagnostics.csv", roi_rows)
    references = []
    if dataset == "coco":
        reversible = REPO / "paper/experiments/20260729_coco_ena24_reversible_ranker_transfer_audit/results/coco/summary.json"
        references.append({"name": "coco_reversible", "path": str(reversible), "sha256": file_sha256(reversible), "tier": "development_roi_diagnostic"})
    else:
        for name, path in (
            ("cloudv2_profile_scale", SOURCES["cloudv2_scan"] / "runs/final_report.json"),
            ("cloudv2_maur_reversible", SOURCES["cloudv2_fusion"] / "runs/final_report.json"),
        ):
            if not path.is_file():
                candidates = sorted((path.parents[1] / "runs").rglob("*report*.json"))
                path = candidates[-1] if candidates else path
            references.append({"name": name, "path": str(path), "sha256": file_sha256(path), "tier": source_tier(str(path))})
    architecture = []
    if dataset == "coco":
        for name in ("dualhead_v2", "relational", "factorized"):
            root = SOURCES[name]
            for path in sorted(root.rglob("results.csv")):
                architecture.extend(_copy_csv_rows(path, dataset, "development_architecture_diagnostic"))
    if architecture:
        atomic_csv(output / "architecture_diagnostics.csv", architecture)
    coverage = {
        "SharedAction": "completed_imported",
        "CostEligible-AllROI": "completed_imported",
        "GBDT-Hierarchical": "completed_imported",
        "ROI": "completed_imported",
        "MAUR_Reversible": "completed_imported",
        "CloudV2_ProfileScale": "oregon_completed_separate" if dataset == "oregon" else "not_applicable_no_coco_cloudv2_cache",
        "DualHead-v3": "not_run_no_frozen_definition",
        "DualHead-v2_RelActionRank_Factorized": "coco_completed_imported" if dataset == "coco" else "not_available_for_oregon",
    }
    atomic_json(output / "references.json", references)
    atomic_json(output / "coverage.json", coverage)
    report = {"dataset": dataset, "canonical_action_rows": len(canonical_rows), "roi_diagnostic_rows": len(roi_rows), "architecture_rows": len(architecture), "references": references, "coverage": coverage}
    atomic_json(output / "report.json", report)
    atomic_json(output / "completion.json", {"complete": True, "report_sha256": file_sha256(output / "report.json")})
    return report


def run_summary(dataset: str) -> dict[str, Any]:
    output = RUNS / dataset
    point = read_json(output / "point/report.json")
    bootstrap = read_json(output / "bootstrap/report.json")
    extensions = read_json(output / "extensions/report.json")
    rows = list(csv.DictReader((output / "point/final_results.csv").open()))
    lines = [f"# {dataset.upper()} full scheduler benchmark", "", f"- 主表行数：{len(rows)}", f"- Random seed 行数：{point['random_seed_rows']}", "- 结果性质：post-hoc final reuse；final 未用于训练、阈值或比较器选择。", "", "## Fixed-byte 主结果", "", "| 方法 | 策略 | 10% AP | 25% AP | 40% AP |", "|---|---|---:|---:|---:|"]
    preferred = ["PerAction-Isotonic-OQ", "Raw-ActionValue-OQ", "DualHead", "Confidence", "Random", "ORIC-MLP", "MORIC-MLP"]
    for method in preferred:
        matches = [row for row in rows if row["method"] == method and row["mode"] == "fixed_byte"]
        policies = sorted({row["policy"] for row in matches})
        for policy in policies:
            by_budget = {float(row["budget"]): float(row["official_map50_95"]) for row in matches if row["policy"] == policy}
            if set(by_budget) == set(map(float, PROTOCOL["budgets"])):
                lines.append(f"| {method} | {policy} | {by_budget[0.1]:.6f} | {by_budget[0.25]:.6f} | {by_budget[0.4]:.6f} |")
    lines.extend(["", "## 扩展覆盖", ""] + [f"- {name}: {status}" for name, status in extensions["coverage"].items()])
    (output / "results_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {"complete": True, "dataset": dataset, "point": point, "bootstrap": bootstrap, "extensions": extensions, "results_sha256": file_sha256(output / "results_zh.md")}
    atomic_json(output / "COMPLETED.json", report)
    return report


def run(dataset: str, stage: str) -> None:
    stages = ["preflight", "point", "bootstrap", "integrate", "summary"]
    target = stages.index(stage) if stage != "all" else len(stages) - 1
    functions = [run_preflight, run_point, run_bootstrap, run_integrate, run_summary]
    for index, function in enumerate(functions):
        if index <= target:
            print(f"[{dataset}] {stages[index]} starting", flush=True)
            function(dataset)
            print(f"[{dataset}] {stages[index]} complete", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("coco", "oregon"))
    parser.add_argument("--stage", default="all", choices=("preflight", "point", "bootstrap", "integrate", "summary", "all"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.dataset, args.stage)
