#!/usr/bin/env python3
"""Run the cache-only three-dataset ActionValue ROI gate diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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
import torch

from roi_gate_core import (
    ACTIONS,
    BUDGETS,
    DATASETS,
    METHODS,
    apply_roi_advantage_threshold,
    expected_result_keys,
    merge_fold_scores,
    oqr_score_filename,
    select_advantage_threshold,
    validate_protocol,
    validate_result_rows,
)


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
validate_protocol(PROTOCOL)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The completed cross-dataset runner supplies the frozen canonical loader,
# evaluator, action scheduler, and the existing ActionValue training helpers.
LEGACY_ROOT = REPO / "paper/experiments/20260807_ena24_oregon_dualhead_crossdataset_v1"
sys.path.insert(0, str(LEGACY_ROOT / "scripts"))
LEGACY = _load_module("roi_gate_crossdataset_legacy", LEGACY_ROOT / "scripts/run_crossdataset.py")
LEGACY.budgetset._device = lambda: torch.device("cpu")

COCO_ROOT = REPO / "paper/experiments/20260805_coco_actionconditioned_valuerisk_oof_v1"
COCO_SOURCE: Any | None = None


def coco_source() -> Any:
    global COCO_SOURCE
    if COCO_SOURCE is None:
        sys.path.insert(0, str(COCO_ROOT / "scripts"))
        COCO_SOURCE = _load_module(
            "roi_gate_coco_actionvalue_source", COCO_ROOT / "scripts/run_oof.py"
        )
    return COCO_SOURCE


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


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def complete(path: Path) -> bool:
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("complete") is True


def dataset_run(dataset: str) -> Path:
    return RUNS / dataset


def backend(dataset: str) -> Any:
    return coco_source() if dataset == "coco" else LEGACY


def load_role(dataset: str, role: str) -> dict[str, Any]:
    if dataset == "coco":
        return backend(dataset).load_role("fit" if role == "fit_oof" else role)
    return backend(dataset).load_role(dataset, "fit" if role == "fit_oof" else role)


def role_name(role: str) -> str:
    return "fit" if role == "fit_oof" else "calibration"


def category_ids(dataset: str, data: Mapping[str, Any]) -> list[int]:
    if dataset == "coco":
        return backend(dataset).category_ids()
    return LEGACY.category_ids(data)


def records_for_role(dataset: str, data: Mapping[str, Any], categories: Sequence[int]) -> list[dict[str, Any]]:
    return LEGACY.build_action_records(data["samples"], data["actions"]["edge"], category_ids=categories)


def identity_map(data: Mapping[str, Any]) -> dict[int, str]:
    return {int(sample["image_id"]): str(sample["identity"]) for sample in data["samples"]}


def canonical_for(dataset: str) -> Any:
    return backend(dataset).canonical


def source_score_paths(dataset: str, role: str) -> tuple[Path, Path]:
    source_role = role_name(role)
    if dataset == "coco":
        root = COCO_ROOT / "runs" / ("oof" if source_role == "fit" else "calibration_scores")
        return root / "scores.pkl", root / "report.json"
    root = LEGACY_ROOT / "runs" / dataset / "scores"
    return root / ("oof.pkl" if source_role == "fit" else "calibration.pkl"), root / ("oof.completion.json" if source_role == "fit" else "calibration.completion.json")


def role_source_files(dataset: str, role: str) -> list[Path]:
    data = load_role(dataset, role)
    root = Path(data["gt_path"]).parent
    manifest_path = root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [manifest_path, root / "samples.json", root / "gt.json"]
    for action in ACTIONS:
        source = Path(manifest["action_sources"][action])
        if not source.is_file():
            raise FileNotFoundError(source)
        paths.append(source)
    score, marker = source_score_paths(dataset, role)
    paths.extend((score, marker))
    return paths


def run_preflight(dataset: str) -> dict[str, Any]:
    output = dataset_run(dataset) / "preflight"
    if complete(output / "completion.json"):
        return json.loads((output / "report.json").read_text(encoding="utf-8"))
    roles = {role: load_role(dataset, role) for role in ("fit_oof", "calibration")}
    ids = {role: {str(sample["identity"]) for sample in data["samples"]} for role, data in roles.items()}
    overlap = len(ids["fit_oof"] & ids["calibration"])
    if overlap:
        raise RuntimeError(f"Role identity overlap for {dataset}: {overlap}")
    paths = [path for role in roles for path in role_source_files(dataset, role)]
    lowered = "\n".join(str(path).lower() for path in paths)
    for forbidden in PROTOCOL["forbidden_source_substrings"]:
        if str(forbidden).lower() in lowered:
            raise RuntimeError(f"Forbidden source path: {forbidden}")
    hashes = {str(path.resolve()): file_sha256(path) for path in sorted(set(paths))}
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "dataset": dataset,
        "role_counts": {role: len(data["samples"]) for role, data in roles.items()},
        "identity_overlap": overlap,
        "source_sha256": hashes,
        "source_digest": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "final_accessed": False,
        "detector_inference": False,
        "gpu_inference": False,
        "passed": True,
    }
    atomic_json(output / "report.json", report)
    atomic_json(output / "completion.json", {"complete": True, "source_digest": report["source_digest"]})
    return report


def verify_preflight(dataset: str) -> None:
    report = json.loads((dataset_run(dataset) / "preflight/report.json").read_text(encoding="utf-8"))
    current = run_preflight(dataset)
    if current["source_sha256"] != report["source_sha256"]:
        raise RuntimeError(f"Frozen source changed for {dataset}")


_REWARD_CACHE: dict[tuple[str, str], dict[int, dict[str, float]]] = {}


def get_rewards(dataset: str, role: str) -> dict[int, dict[str, float]]:
    key = (str(dataset), str(role))
    if key in _REWARD_CACHE:
        return _REWARD_CACHE[key]
    if dataset == "coco":
        values = backend(dataset).reward_map(role_name(role))
        result = {int(image_id): dict(values[int(image_id)]) for image_id in values}
    elif role == "calibration":
        result = load_pickle(dataset_run(dataset) / "rewards/calibration.pkl")
    elif role == "fit_oof":
        fit = load_role(dataset, "fit_oof")
        result = {}
        for fold_index, indices in enumerate(deterministic_folds(fit)):
            fold_rewards = load_pickle(dataset_run(dataset) / f"rewards/fit_fold_{fold_index}.pkl")
            for index in indices:
                image_id = int(fit["samples"][int(index)]["image_id"])
                result[image_id] = dict(fold_rewards[image_id])
        if len(result) != len(fit["samples"]):
            raise RuntimeError(f"Incomplete OOF reward population for {dataset}")
    else:
        raise ValueError(role)
    _REWARD_CACHE[key] = result
    return result


_STATS_STATE: dict[str, Any] = {}


def _stats_worker(item: tuple[int, str]) -> tuple[int, dict[str, Any]]:
    dataset = str(_STATS_STATE["dataset"])
    data = _STATS_STATE["data"]
    sample = data["sample_by_id"][int(item[0])]
    target = data["gt_map"].get(int(item[0]), [])
    canonical = canonical_for(dataset)
    output: dict[str, Any] = {}
    for action in ACTIONS:
        if action != "edge" and not bool(sample["available"].get(action, False)):
            continue
        predictions = canonical.filter_conf(
            data["actions"][action].get(int(item[0]), []),
            float(PROTOCOL["reward"]["confidence"]),
        )
        output[action] = LEGACY.detection_stats(
            predictions,
            target,
            tuple(map(float, PROTOCOL["reward"]["iou_thresholds"])),
        )
    return int(item[0]), output


def _fork_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("fork")
    except ValueError as error:
        raise RuntimeError("The CPU cache pipeline requires fork") from error


def build_stats(dataset: str, role: str, workers: int) -> dict[str, Any]:
    output = dataset_run(dataset) / "stats"
    path = output / f"{role_name(role)}.pkl"
    marker = output / f"{role_name(role)}_completion.json"
    if complete(marker):
        return load_pickle(path)
    data = load_role(dataset, role)
    _STATS_STATE.clear()
    _STATS_STATE.update({"dataset": dataset, "data": data})
    stats = {action: {} for action in ACTIONS}
    started = time.time()
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=_fork_context()) as pool:
        items = [(int(image_id), identity_map(data)[int(image_id)]) for image_id in data["image_ids"]]
        for count, (image_id, values) in enumerate(pool.map(_stats_worker, items, chunksize=16), start=1):
            for action, value in values.items():
                stats[action][int(image_id)] = value
            if count % 512 == 0 or count == len(items):
                print(f"[{dataset}/stats/{role_name(role)}] {count}/{len(items)}", flush=True)
    payload = {
        "dataset": dataset,
        "role": role_name(role),
        "image_ids": list(map(int, data["image_ids"])),
        "identities": identity_map(data),
        "stats": stats,
        "elapsed_seconds": time.time() - started,
    }
    atomic_pickle(path, payload)
    atomic_json(marker, {"complete": True, "images": len(items), "sha256": file_sha256(path)})
    _STATS_STATE.clear()
    return payload


def _context_ids(
    candidate_ids: Sequence[int], identities: Mapping[int, str], target_identity: str
) -> list[int]:
    candidates = [int(image_id) for image_id in candidate_ids if str(identities[int(image_id)]) != str(target_identity)]
    if not candidates:
        return []
    digest = hashlib.sha256(
        f"{int(PROTOCOL['reward']['context_seed'])}:{target_identity}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    take = min(int(PROTOCOL["reward"]["context_size"]), len(candidates))
    return [candidates[int(index)] for index in rng.choice(len(candidates), size=take, replace=False)]


_REWARD_STATE: dict[str, Any] = {}


def _reward_worker(item: tuple[int, str]) -> tuple[int, dict[str, float]]:
    image_id, identity = item
    context_ids = _context_ids(
        _REWARD_STATE["context_ids"],
        _REWARD_STATE["context_identities"],
        str(identity),
    )
    context = [_REWARD_STATE["context_stats"]["edge"][value] for value in context_ids]
    target = _REWARD_STATE["target_stats"]
    edge = target["edge"][int(image_id)]
    values: dict[str, float] = {}
    for action in PROTOCOL["reward"]["candidate_actions"]:
        cloud = target[str(action)].get(int(image_id))
        if cloud is not None:
            values[str(action)] = float(LEGACY.oric_reward(context, edge, cloud)["reward"])
    return int(image_id), values


def compute_rewards(
    dataset: str,
    label: str,
    context_stats: Mapping[str, Any],
    target_stats: Mapping[str, Any],
    context_ids: Sequence[int],
    workers: int,
) -> dict[int, dict[str, float]]:
    output = dataset_run(dataset) / "rewards"
    path = output / f"{label}.pkl"
    marker = output / f"{label}_completion.json"
    if complete(marker):
        return load_pickle(path)
    partial = output / f"{label}_partial.pkl"
    rewards: dict[int, dict[str, float]] = load_pickle(partial) if partial.is_file() else {}
    items = [(int(image_id), str(target_stats["identities"][int(image_id)])) for image_id in target_stats["image_ids"]]
    remaining = [item for item in items if item[0] not in rewards]
    _REWARD_STATE.clear()
    _REWARD_STATE.update(
        {
            "context_ids": list(map(int, context_ids)),
            "context_identities": context_stats["identities"],
            "context_stats": context_stats["stats"],
            "target_stats": target_stats["stats"],
        }
    )
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=_fork_context()) as pool:
        for count, (image_id, values) in enumerate(pool.map(_reward_worker, remaining, chunksize=16), start=1):
            rewards[int(image_id)] = values
            if count % int(PROTOCOL["reward"]["checkpoint_interval"]) == 0 or count == len(remaining):
                atomic_pickle(partial, rewards)
                print(f"[{dataset}/rewards/{label}] {len(rewards)}/{len(items)}", flush=True)
    if set(rewards) != {item[0] for item in items}:
        raise RuntimeError(f"Incomplete reward cache: {dataset}/{label}")
    atomic_pickle(path, rewards)
    atomic_json(marker, {"complete": True, "images": len(rewards), "sha256": file_sha256(path)})
    partial.unlink(missing_ok=True)
    _REWARD_STATE.clear()
    return rewards


def deterministic_folds(data: Mapping[str, Any]) -> list[list[int]]:
    ordered = sorted(
        range(len(data["samples"])),
        key=lambda index: hashlib.sha256(
            f"{int(PROTOCOL['fold_seed'])}:{data['samples'][index]['identity']}".encode("utf-8")
        ).digest(),
    )
    folds = [[] for _ in range(5)]
    for position, index in enumerate(ordered):
        folds[position % 5].append(int(index))
    return folds


def run_rewards(dataset: str, workers: int) -> None:
    output = dataset_run(dataset) / "rewards"
    marker = output / "completion.json"
    if complete(marker):
        return
    verify_preflight(dataset)
    fit = load_role(dataset, "fit_oof")
    fit_stats = build_stats(dataset, "fit_oof", workers)
    calibration_stats = build_stats(dataset, "calibration", workers)
    fit_ids = list(map(int, fit["image_ids"]))
    compute_rewards(dataset, "fit_full", fit_stats, fit_stats, fit_ids, workers)
    folds = deterministic_folds(fit)
    manifest: dict[str, Any] = {}
    for fold_index, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train_ids = [int(fit["samples"][index]["image_id"]) for index in range(len(fit["samples"])) if index not in heldout_set]
        manifest[str(fold_index)] = {"heldout_indices": heldout, "train_ids": train_ids}
        compute_rewards(dataset, f"fit_fold_{fold_index}", fit_stats, fit_stats, train_ids, workers)
    compute_rewards(dataset, "calibration", fit_stats, calibration_stats, fit_ids, workers)
    atomic_json(output / "fold_manifest.json", manifest)
    atomic_json(output / "report.json", {"dataset": dataset, "folds": 5, "final_accessed": False})
    atomic_json(marker, {"complete": True})


def model_protocol() -> dict[str, Any]:
    return {
        "budgets": list(BUDGETS),
        "max_boxes": int(PROTOCOL["model"]["max_boxes"]),
        "max_actions": int(PROTOCOL["model"]["max_actions"]),
        "deploy_confidence": float(PROTOCOL["model"]["deploy_confidence"]),
        "resource_profile": "upload",
        "model": dict(PROTOCOL["model"]),
    }


def reward_matrix(
    records: Sequence[Mapping[str, Any]], rewards: Mapping[int, Mapping[str, float]]
) -> np.ndarray:
    values = np.zeros((len(records), 4), dtype=np.float32)
    for index, record in enumerate(records):
        image_id = int(record["image_id"])
        for action_index, action in enumerate(record["actions"]):
            action_id = str(action["action_id"])
            if action_id == "edge":
                continue
            if action_id not in rewards[image_id]:
                raise RuntimeError(f"Missing reward for {image_id}/{action_id}")
            values[index, action_index] = float(rewards[image_id][action_id])
    return values


def subset_tensor_map(data: Mapping[str, torch.Tensor], indices: Sequence[int]) -> dict[str, torch.Tensor]:
    tensor_indices = torch.as_tensor(list(map(int, indices)), dtype=torch.long)
    return {key: value[tensor_indices] for key, value in data.items()}


_TRAIN_STATE: dict[str, Any] = {}


def _train_oof_job(item: tuple[int, int]) -> dict[str, Any]:
    fold_index, seed = map(int, item)
    torch.set_num_threads(int(PROTOCOL["workers"]["torch_threads_per_worker"]))
    dataset = str(_TRAIN_STATE["dataset"])
    output = dataset_run(dataset) / "models/oof" / f"fold_{fold_index}_seed_{seed}.pkl"
    marker = output.with_suffix(".completion.json")
    if complete(marker):
        return {"fold": fold_index, "seed": seed, "cached": True}
    records = _TRAIN_STATE["records"]
    heldout = list(map(int, _TRAIN_STATE["folds"][fold_index]))
    heldout_set = set(heldout)
    train = [index for index in range(len(records)) if index not in heldout_set]
    rewards = load_pickle(dataset_run(dataset) / f"rewards/fit_fold_{fold_index}.pkl")
    raw = reward_matrix(records, rewards)
    protocol = model_protocol()
    train_records = [records[index] for index in train]
    target_records = [records[index] for index in heldout]
    train_data = LEGACY.budgetset._record_tensors(
        train_records, raw[train], protocol
    )
    target_data = LEGACY.budgetset._record_tensors(
        target_records, None, protocol
    )
    checkpoint = LEGACY.fit_value_router(
        [int(records[index]["image_id"]) for index in train],
        train_data,
        protocol,
        category_count=int(_TRAIN_STATE["category_count"]),
        seed=int(seed),
        device=torch.device("cpu"),
    )
    scores = LEGACY.predict_value_router(
        checkpoint,
        target_data,
        protocol,
        category_count=int(_TRAIN_STATE["category_count"]),
    )
    payload = {
        "fold": fold_index,
        "seed": seed,
        "indices": heldout,
        "scores": np.asarray(scores, dtype=np.float64),
        "selected_epoch": int(checkpoint["selected_epoch"]),
    }
    atomic_pickle(output, payload)
    atomic_json(marker, {"complete": True, "sha256": file_sha256(output), "fit_only": True})
    return {"fold": fold_index, "seed": seed, "cached": False}


def _train_full_job(seed: int) -> dict[str, Any]:
    seed = int(seed)
    torch.set_num_threads(int(PROTOCOL["workers"]["torch_threads_per_worker"]))
    dataset = str(_TRAIN_STATE["dataset"])
    output = dataset_run(dataset) / "models/full" / f"seed_{seed}.pkl"
    marker = output.with_suffix(".completion.json")
    if complete(marker):
        return {"seed": seed, "cached": True}
    records = _TRAIN_STATE["records"]
    calibration = _TRAIN_STATE["calibration_records"]
    rewards = load_pickle(dataset_run(dataset) / "rewards/fit_full.pkl")
    raw = reward_matrix(records, rewards)
    protocol = model_protocol()
    train_data = LEGACY.budgetset._record_tensors(records, raw, protocol)
    target_data = LEGACY.budgetset._record_tensors(calibration, None, protocol)
    checkpoint = LEGACY.fit_value_router(
        [int(record["image_id"]) for record in records],
        train_data,
        protocol,
        category_count=int(_TRAIN_STATE["category_count"]),
        seed=seed,
        device=torch.device("cpu"),
    )
    scores = LEGACY.predict_value_router(
        checkpoint,
        target_data,
        protocol,
        category_count=int(_TRAIN_STATE["category_count"]),
    )
    payload = {
        "seed": seed,
        "scores": np.asarray(scores, dtype=np.float64),
        "selected_epoch": int(checkpoint["selected_epoch"]),
        "fit_only": True,
    }
    atomic_pickle(output, payload)
    atomic_json(marker, {"complete": True, "sha256": file_sha256(output), "fit_only": True})
    return {"seed": seed, "cached": False}


def _run_jobs(function: Any, jobs: Sequence[Any], workers: int) -> list[Any]:
    if not jobs:
        return []
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=_fork_context()) as pool:
        return list(pool.map(function, jobs, chunksize=1))


def run_train(dataset: str, workers: int) -> None:
    output = dataset_run(dataset) / "models"
    marker = output / "completion.json"
    if complete(marker):
        return
    if dataset == "coco":
        # COCO OQ/OQR was already fit in the completed source diagnostic.
        atomic_json(output / "report.json", {"dataset": dataset, "reused": True, "detector_inference": False})
        atomic_json(marker, {"complete": True, "reused": True})
        return
    verify_preflight(dataset)
    fit = load_role(dataset, "fit_oof")
    calibration = load_role(dataset, "calibration")
    categories = category_ids(dataset, fit)
    records = records_for_role(dataset, fit, categories)
    calibration_records = records_for_role(dataset, calibration, categories)
    folds = deterministic_folds(fit)
    _TRAIN_STATE.clear()
    _TRAIN_STATE.update(
        {
            "dataset": dataset,
            "records": records,
            "calibration_records": calibration_records,
            "folds": folds,
            "category_count": len(categories),
        }
    )
    seeds = list(map(int, PROTOCOL["model"]["seeds"]))
    oof_jobs = [(fold_index, seed) for fold_index in range(len(folds)) for seed in seeds]
    oof_reports = _run_jobs(_train_oof_job, oof_jobs, int(workers))
    full_reports = _run_jobs(_train_full_job, seeds, min(int(workers), len(seeds)))
    atomic_json(output / "report.json", {"dataset": dataset, "oof_jobs": oof_reports, "full_jobs": full_reports, "fit_only": True})
    atomic_json(marker, {"complete": True})
    _TRAIN_STATE.clear()


def source_score_artifact(dataset: str, role: str) -> Mapping[str, Any]:
    score_path, _ = source_score_paths(dataset, role)
    return load_pickle(score_path)


def load_oq_scores(dataset: str, role: str) -> np.ndarray:
    artifact = source_score_artifact(dataset, role)
    if dataset == "coco":
        values = np.asarray(artifact["scores"]["value"], dtype=np.float64)
        if values.shape[1] != 4:
            raise RuntimeError(f"Unexpected COCO OQR source shape: {values.shape}")
        return np.repeat(values[:, :3][None, ...], len(BUDGETS), axis=0)
    values = np.asarray(artifact["ensemble"]["ActionValue-OQ-Joint"], dtype=np.float64)
    if values.shape != (len(BUDGETS), DATASETS[dataset]["fit" if role == "fit_oof" else "calibration"], 3):
        raise RuntimeError(f"Unexpected OQ score shape for {dataset}/{role}: {values.shape}")
    return values


def load_oqr_scores(dataset: str, role: str) -> np.ndarray:
    if dataset == "coco":
        artifact = source_score_artifact(dataset, role)
        values = np.asarray(artifact["scores"]["value"], dtype=np.float64)
        return np.repeat(values[None, ...], len(BUDGETS), axis=0)
    path = dataset_run(dataset) / "scores" / oqr_score_filename(role)
    marker = path.with_suffix(".completion.json")
    if not complete(marker):
        raise RuntimeError(f"OQR score artifact missing: {path}")
    return np.asarray(load_pickle(path)["ensemble"], dtype=np.float64)


def ensemble_model_scores(dataset: str) -> None:
    if dataset == "coco":
        return
    output = dataset_run(dataset) / "scores"
    for role in ("fit_oof", "calibration"):
        path = output / oqr_score_filename(role)
        marker = path.with_suffix(".completion.json")
        if complete(marker):
            continue
        data = load_role(dataset, role)
        count = len(data["samples"])
        arrays = np.full((len(PROTOCOL["model"]["seeds"]), len(BUDGETS), count, 4), np.nan, dtype=np.float64)
        if role == "fit_oof":
            for fold_index in range(5):
                for seed_index, seed in enumerate(map(int, PROTOCOL["model"]["seeds"])):
                    artifact = load_pickle(dataset_run(dataset) / f"models/oof/fold_{fold_index}_seed_{seed}.pkl")
                    indices = list(map(int, artifact["indices"]))
                    merge_fold_scores(
                        arrays,
                        seed_index=seed_index,
                        heldout_indices=indices,
                        fold_scores=artifact["scores"],
                    )
        else:
            for seed_index, seed in enumerate(map(int, PROTOCOL["model"]["seeds"])):
                artifact = load_pickle(dataset_run(dataset) / f"models/full/seed_{seed}.pkl")
                arrays[seed_index] = artifact["scores"]
        if not np.isfinite(arrays).all():
            raise RuntimeError(f"Incomplete OQR scores for {dataset}/{role}")
        payload = {"ensemble": arrays.mean(axis=0), "per_seed": arrays}
        atomic_pickle(path, payload)
        atomic_json(marker, {"complete": True, "sha256": file_sha256(path)})


def score_map(data: Mapping[str, Any], values: np.ndarray, *, include_roi: bool) -> dict[int, dict[str, float]]:
    values = np.asarray(values, dtype=np.float64)
    expected_width = 4 if include_roi else 3
    if values.shape != (len(data["samples"]), expected_width):
        raise RuntimeError(f"Score shape changed: {values.shape} != {(len(data['samples']), expected_width)}")
    output = {}
    for index, sample in enumerate(data["samples"]):
        actions = {"original": float(values[index, 1]), "q90": float(values[index, 2])}
        if include_roi:
            actions["roi"] = float(values[index, 3])
        output[int(sample["image_id"])] = actions
    return output


def request_bytes(data: Mapping[str, Any], selection: Mapping[int, str]) -> int:
    return sum(int(data["sample_by_id"][int(image_id)]["payload_bytes"][str(action)]) for image_id, action in selection.items())


def reward_sum(selection: Mapping[int, str], rewards: Mapping[int, Mapping[str, float]]) -> float:
    return sum(float(rewards[int(image_id)].get(str(action), 0.0)) for image_id, action in selection.items())


def roi_rates(selection: Mapping[int, str], rewards: Mapping[int, Mapping[str, float]]) -> tuple[float, float]:
    selected = [int(image_id) for image_id, action in selection.items() if str(action) == "roi"]
    if not selected:
        return 0.0, 0.0
    harmful = 0
    relative = 0
    for image_id in selected:
        values = rewards[image_id]
        roi = float(values.get("roi", 0.0))
        if roi <= 0.0:
            harmful += 1
        if roi < max(float(values.get("original", 0.0)), float(values.get("q90", 0.0))):
            relative += 1
    return harmful / len(selected), relative / len(selected)


def schedule(data: Mapping[str, Any], scores: Mapping[int, Mapping[str, float]], mode: str, budget: float) -> dict[str, Any]:
    if mode == "fixed_call":
        return LEGACY.schedule_fixed_call(data["samples"], scores, budget=float(budget))
    capacity = float(budget) * sum(float(sample["payload_bytes"]["original"]) for sample in data["samples"])
    return LEGACY.schedule_fixed_byte(data["samples"], scores, capacity_bytes=capacity)


def threshold_candidates(values: np.ndarray) -> list[float]:
    advantage = np.asarray(values[:, 3] - np.maximum(values[:, 1], values[:, 2]), dtype=np.float64)
    if not len(advantage):
        return [float("inf")]
    quantiles = np.quantile(advantage, np.linspace(0.0, 1.0, int(PROTOCOL["advantage_gate"]["candidate_quantiles"])))
    maximum = float(np.max(advantage))
    disabled = maximum + max(1e-6, abs(maximum) * 1e-6)
    candidates = [*map(float, quantiles), disabled]
    output = []
    for value in candidates:
        if not output or value != output[-1]:
            output.append(value)
    return output


def fit_gate_thresholds(
    data: Mapping[str, Any], scores: np.ndarray, rewards: Mapping[int, Mapping[str, float]]
) -> dict[str, dict[str, Any]]:
    thresholds: dict[str, dict[str, Any]] = {}
    maximum_harmful = float(PROTOCOL["advantage_gate"]["maximum_harmful_roi_rate"])
    for budget_index, budget in enumerate(BUDGETS):
        for mode in ("fixed_byte", "fixed_call"):
            candidates = []
            for threshold in threshold_candidates(scores[budget_index]):
                gated = apply_roi_advantage_threshold(scores[budget_index], threshold=threshold)
                selection = schedule(data, score_map(data, gated, include_roi=True), mode, float(budget))["selection"]
                harmful, _ = roi_rates(selection, rewards)
                candidates.append(
                    {
                        "threshold": float(threshold),
                        "reward": reward_sum(selection, rewards),
                        "harmful_rate": harmful,
                        "roi_count": sum(str(action) == "roi" for action in selection.values()),
                    }
                )
            chosen = select_advantage_threshold(candidates, maximum_harmful_rate=maximum_harmful)
            key = f"{mode}|{float(budget):g}"
            thresholds[key] = {**chosen, "mode": mode, "budget": float(budget), "source_role": "fit_oof"}
    return thresholds


def evaluate_selection(
    dataset: str,
    role: str,
    method: str,
    mode: str,
    budget: float,
    data: Mapping[str, Any],
    rewards: Mapping[int, Mapping[str, float]],
    selection: Mapping[int, str],
    predicted_value: float,
    gate_threshold: float | None,
) -> dict[str, Any]:
    predictions, _ = canonical_for(dataset).mixed_predictions(data, dict(selection), 0.001)
    official = data["evaluator"].evaluate(predictions)
    used = request_bytes(data, selection)
    capacity = float(budget) * sum(float(sample["payload_bytes"]["original"]) for sample in data["samples"])
    expected_count = int(math.floor(len(data["samples"]) * float(budget) + 0.5))
    harmful, relative = roi_rates(selection, rewards)
    teacher = schedule(data, rewards, mode, float(budget))
    counts = Counter(map(str, selection.values()))
    return {
        "dataset": dataset,
        "role": role,
        "method": method,
        "mode": mode,
        "budget": float(budget),
        "official_ap50_95": float(official["map50_95"]),
        "official_ap50": float(official["map50"]),
        "official_ap75": float(official["map75"]),
        "kb_per_image": used / 1024.0 / len(data["samples"]),
        "request_bytes": int(used),
        "capacity_bytes": float(capacity),
        "capacity_ok": bool(used <= capacity + 1e-6),
        "selected_images": len(selection),
        "expected_call_count": expected_count,
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["roi"]),
        "roi_ratio": float(counts["roi"] / max(1, len(selection))),
        "harmful_roi_rate": float(harmful),
        "relative_suboptimal_roi_rate": float(relative),
        "selected_contextual_reward": float(reward_sum(selection, rewards)),
        "teacher_contextual_reward": float(teacher["total_value"]),
        "teacher_regret": float(teacher["total_value"] - reward_sum(selection, rewards)),
        "predicted_objective": float(predicted_value),
        "gate_threshold": None if gate_threshold is None else float(gate_threshold),
        "threshold_source": "fit_oof_only" if gate_threshold is not None else None,
        "final_accessed": False,
    }


def evaluate_role(
    dataset: str,
    role: str,
    data: Mapping[str, Any],
    oq_scores: np.ndarray,
    oqr_scores: np.ndarray,
    gate_thresholds: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rewards = get_rewards(dataset, role)
    rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, str]] = {}
    for budget_index, budget in enumerate(BUDGETS):
        for mode in ("fixed_byte", "fixed_call"):
            oq_map = score_map(data, oq_scores[budget_index], include_roi=False)
            oqr_map = score_map(data, oqr_scores[budget_index], include_roi=True)
            gate = None if gate_thresholds is None else gate_thresholds[f"{mode}|{float(budget):g}"]
            gated_scores = apply_roi_advantage_threshold(oqr_scores[budget_index], threshold=float(gate["threshold"])) if gate is not None else None
            gate_map = None if gated_scores is None else score_map(data, gated_scores, include_roi=True)
            method_maps = {
                "ActionValue-OQ-Joint": oq_map,
                "ActionValue-OQR-Naive": oqr_map,
                "ActionValue-OQR-AdvantageGate": gate_map,
            }
            for method, values in method_maps.items():
                if values is None:
                    raise RuntimeError(f"Missing gate map for {dataset}/{role}")
                result = schedule(data, values, mode, float(budget))
                selection = result["selection"]
                if mode == "fixed_call" and len(selection) != int(math.floor(len(data["samples"]) * float(budget) + 0.5)):
                    raise RuntimeError(f"Fixed-call count changed: {dataset}/{role}/{method}/{budget}")
                key = f"{method}|{mode}|{float(budget):g}"
                selections[key] = {str(image_id): str(action) for image_id, action in selection.items()}
                threshold = None if method != "ActionValue-OQR-AdvantageGate" else float(gate["threshold"])
                rows.append(
                    evaluate_selection(
                        dataset,
                        role,
                        method,
                        mode,
                        float(budget),
                        data,
                        rewards,
                        selection,
                        float(result["total_value"]),
                        threshold,
                    )
                )
    return rows, selections


def run_evaluation(dataset: str) -> None:
    output = dataset_run(dataset) / "evaluation"
    marker = output / "completion.json"
    if complete(marker):
        return
    verify_preflight(dataset)
    ensemble_model_scores(dataset)
    fit = load_role(dataset, "fit_oof")
    calibration = load_role(dataset, "calibration")
    fit_categories = category_ids(dataset, fit)
    fit_records = records_for_role(dataset, fit, fit_categories)
    calibration_records = records_for_role(dataset, calibration, fit_categories)
    oq_fit = load_oq_scores(dataset, "fit_oof")
    oq_calibration = load_oq_scores(dataset, "calibration")
    oqr_fit = load_oqr_scores(dataset, "fit_oof")
    oqr_calibration = load_oqr_scores(dataset, "calibration")
    fit_gate = fit_gate_thresholds(fit, oqr_fit, get_rewards(dataset, "fit_oof"))
    rows, selections = evaluate_role(dataset, "fit_oof", fit, oq_fit, oqr_fit, fit_gate)
    calibration_rows, calibration_selections = evaluate_role(dataset, "calibration", calibration, oq_calibration, oqr_calibration, fit_gate)
    rows.extend(calibration_rows)
    selections.update({f"calibration|{key}": value for key, value in calibration_selections.items()})
    row_index = {
        (str(row["role"]), str(row["mode"]), float(row["budget"]), str(row["method"])): row
        for row in rows
    }
    for row in rows:
        baseline = row_index[(str(row["role"]), str(row["mode"]), float(row["budget"]), "ActionValue-OQ-Joint")]
        row["delta_ap_vs_oq"] = float(row["official_ap50_95"]) - float(baseline["official_ap50_95"])
    validate_result_rows(rows, PROTOCOL, dataset)
    atomic_csv(output / "results.csv", rows)
    atomic_json(output / "selections.json", selections)
    atomic_json(output / "gate_thresholds.json", fit_gate)
    report = {
        "dataset": dataset,
        "rows": len(rows),
        "methods": list(METHODS),
        "fit_gate_source": "fit_oof_only",
        "calibration_gate_frozen": True,
        "final_accessed": False,
        "detector_inference": False,
        "reused_oq_scores": True,
        "reused_coco_oqr_scores": dataset == "coco",
    }
    atomic_json(output / "report.json", report)
    atomic_json(marker, {"complete": True})


def write_notes(dataset: str) -> None:
    rows = []
    with (dataset_run(dataset) / "evaluation/results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        f"# {dataset} ActionValue ROI gate diagnostic",
        "",
        "> Fit OOF selects gate thresholds; frozen calibration evaluates them. No final role was accessed.",
        "",
        "| Role | Mode | Budget | OQ AP | OQR AP | Gate AP | OQ KB | OQR KB | Gate KB | Gate ROI | Gate harmful ROI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ("fit_oof", "calibration"):
        for mode in ("fixed_byte", "fixed_call"):
            for budget in BUDGETS:
                subset = [row for row in rows if row["role"] == role and row["mode"] == mode and math.isclose(float(row["budget"]), budget)]
                by_method = {row["method"]: row for row in subset}
                lines.append(
                    f"| {role} | {mode} | {budget:.0%} | "
                    f"{float(by_method['ActionValue-OQ-Joint']['official_ap50_95']):.6f} | "
                    f"{float(by_method['ActionValue-OQR-Naive']['official_ap50_95']):.6f} | "
                    f"{float(by_method['ActionValue-OQR-AdvantageGate']['official_ap50_95']):.6f} | "
                    f"{float(by_method['ActionValue-OQ-Joint']['kb_per_image']):.2f} | "
                    f"{float(by_method['ActionValue-OQR-Naive']['kb_per_image']):.2f} | "
                    f"{float(by_method['ActionValue-OQR-AdvantageGate']['kb_per_image']):.2f} | "
                    f"{float(by_method['ActionValue-OQR-AdvantageGate']['roi_ratio']):.2%} | "
                    f"{float(by_method['ActionValue-OQR-AdvantageGate']['harmful_roi_rate']):.2%} |"
                )
    (ROOT / "notes").mkdir(parents=True, exist_ok=True)
    (ROOT / "notes" / f"{dataset}_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset(dataset: str, stage: str, workers: int, train_workers: int) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(dataset)
    if stage in ("preflight", "all"):
        run_preflight(dataset)
    if stage in ("rewards", "all") and dataset != "coco":
        run_rewards(dataset, workers)
    if stage in ("train", "all"):
        run_train(dataset, train_workers)
    if stage in ("evaluate", "all"):
        run_evaluation(dataset)
        write_notes(dataset)
    state = {
        "dataset": dataset,
        "requested_stage": stage,
        "preflight": complete(dataset_run(dataset) / "preflight/completion.json"),
        "rewards": dataset == "coco" or complete(dataset_run(dataset) / "rewards/completion.json"),
        "train": complete(dataset_run(dataset) / "models/completion.json"),
        "evaluation": complete(dataset_run(dataset) / "evaluation/completion.json"),
        "final_accessed": False,
    }
    atomic_json(dataset_run(dataset) / "pipeline_status.json", state)
    if all(state[key] for key in ("preflight", "rewards", "train", "evaluation")):
        atomic_json(dataset_run(dataset) / "COMPLETED.json", {"complete": True, **state})
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--stage", choices=("preflight", "rewards", "train", "evaluate", "all"), default="all")
    parser.add_argument("--workers", type=int, default=int(PROTOCOL["workers"]["cache_per_dataset"]))
    parser.add_argument("--train-workers", type=int, default=int(PROTOCOL["workers"]["training_per_dataset"]))
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    state = run_dataset(arguments.dataset, arguments.stage, arguments.workers, arguments.train_workers)
    print(json.dumps(state, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
