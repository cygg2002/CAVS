#!/usr/bin/env python3
"""Run the final COCO/Oregon CAVS paper-completion experiment."""

from __future__ import annotations

import argparse
from collections import Counter
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


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
sys.dont_write_bytecode = True


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("final_completion_contract", ROOT / "scripts/core.py")
TRAINING = load_module("final_completion_training", ROOT / "scripts/training.py")
CORE.validate_protocol(PROTOCOL)
UNIFIED_ROOT = REPO / "paper/experiments/20260812_coco_oregon_cavs_oqr_unified_paper_v1"
UNIFIED = load_module("final_completion_unified", UNIFIED_ROOT / "scripts/run_experiment.py")
FACTORIAL_ROOT = REPO / PROTOCOL["datasets"]["coco"]["maur_source"]
FACTORIAL = load_module("final_completion_factorial", FACTORIAL_ROOT / "scripts/run_experiment.py")
OREGON_ROOT = REPO / PROTOCOL["datasets"]["oregon"]["canonical_source"]
OREGON_TRAIN = load_module("final_completion_oregon_train", OREGON_ROOT / "scripts/train_learners.py")
OREGON_REWARD = load_module("final_completion_oregon_reward", OREGON_ROOT / "scripts/build_rewards.py")
SAME_MAUR_ROOT = REPO / PROTOCOL["datasets"]["oregon"]["maur_action_source"]
SAME_MAUR = load_module("final_completion_same_maur", SAME_MAUR_ROOT / "scripts/run_experiment.py")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def is_complete(path: Path) -> bool:
    return path.is_file() and read_json(path).get("complete") is True


def filter_records(
    records: Sequence[Mapping[str, Any]], actions: Sequence[str]
) -> list[dict[str, Any]]:
    allowed = set(map(str, actions))
    output = []
    for record in records:
        row = dict(record)
        row["actions"] = [
            dict(action) for action in record["actions"] if str(action["action_id"]) in allowed
        ]
        if not row["actions"] or str(row["actions"][0]["action_id"]) != "edge":
            raise RuntimeError(f"edge action changed: {record.get('image_id')}")
        output.append(row)
    return output


_ADAPTER_CACHE: dict[str, Any] = {}


def adapter(dataset: str) -> Any:
    """Load each heavy detector/evaluator adapter once per process."""
    if dataset not in _ADAPTER_CACHE:
        _ADAPTER_CACHE[dataset] = UNIFIED.DatasetAdapter(dataset)
    return _ADAPTER_CACHE[dataset]


def role_records(dataset: str, role: str, space: str) -> list[dict[str, Any]]:
    values = adapter(dataset).records(role)
    return filter_records(values, PROTOCOL["action_spaces"][space])


def folds(dataset: str) -> list[list[int]]:
    values = FACTORIAL.fit_folds() if dataset == "coco" else OREGON_TRAIN.fit_folds()
    return [list(map(int, fold)) for fold in values]


def reward_map(dataset: str, role: str, method: str) -> dict[int, dict[str, float]]:
    if role not in {"fit", "calibration"}:
        raise ValueError("final rewards are forbidden")
    if dataset == "coco":
        return FACTORIAL.reward_map_for_method(role, "CAVS-Current-MAUR")
    if method == "CAVS-OQR-MAUR":
        return load_pickle(RUNS / f"oregon/rewards/{role}/reward_map.pkl")
    values, _ = OREGON_TRAIN.load_rewards(role)
    return values


def reward_matrix(
    dataset: str, records: Sequence[Mapping[str, Any]], rewards: Mapping[int, Mapping[str, float]]
) -> np.ndarray:
    source = FACTORIAL.BASE.SOURCE_TRAIN.TRAIN_CORE if dataset == "coco" else OREGON_TRAIN.TRAIN_CORE
    return source.reward_matrix(records, rewards)


def training_adapter(dataset: str) -> Any:
    return FACTORIAL.BASE.SOURCE_TRAIN.ADAPTER if dataset == "coco" else OREGON_TRAIN.ADAPTER


def common_manifest(dataset: str, role: str) -> Path:
    root = REPO / PROTOCOL["datasets"][dataset]["canonical_source"]
    return root / f"runs/common/{dataset}/{role}/source_manifest.json"


def common_samples(dataset: str, role: str) -> Path:
    return common_manifest(dataset, role).with_name("samples.json")


def _source_required(dataset: str) -> list[Path]:
    config = PROTOCOL["datasets"][dataset]
    required = [REPO / config["unified_source"] / f"runs/{dataset}/point/completion.json"]
    for role in config["roles"]:
        required.append(common_manifest(dataset, role))
    if dataset == "coco":
        source = REPO / config["dedicated_oq_source"]
        for seed in config["model_seeds"]:
            for fold in range(int(PROTOCOL["fold_count"])):
                required.append(source / f"runs/oof/action/jobs/Current-OQ/seed_{seed}/fold_{fold}/scores.npz")
            required.extend(
                (
                    source / f"runs/calibration/action/jobs/Current-OQ/seed_{seed}/calibration_scores.npz",
                    source / f"runs/final/action/jobs/Current-OQ/seed_{seed}/scores.npz",
                )
            )
        for seed in config["model_seeds"]:
            required.append(FACTORIAL_ROOT / f"runs/training/jobs/CAVS-Current-MAUR/seed_{seed}/full/scores.npz")
    else:
        source = REPO / config["dedicated_oq_source"]
        for method in ("CurrentActionValue-OQ", "ORIC-AP5095", "MORIC-AP5095"):
            for seed in config["model_seeds"]:
                required.append(source / f"runs/learners/jobs/{method}/seed_{seed}/full/scores.npz")
        for role in ("fit", "calibration"):
            required.extend(
                (
                    source / f"runs/rewards/{role}/reward_map.pkl",
                    source / f"runs/rewards/{role}/metric_map.pkl",
                    source / f"runs/rewards/{role}/folds.json",
                    source / f"runs/rewards/{role}/contexts.json",
                )
            )
            manifest = read_json(common_manifest(dataset, role))
            required.append(Path(manifest["maur_action_sources"]["current@640"]))
    return required


def run_preflight(dataset: str) -> dict[str, Any]:
    output = RUNS / f"{dataset}/preflight"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    config = PROTOCOL["datasets"][dataset]
    samples = {}
    for role in config["roles"]:
        raw = read_json(common_samples(dataset, role))
        samples[role] = list(raw["samples"] if isinstance(raw, dict) else raw)
    counts = {role: len(values) for role, values in samples.items()}
    if counts != config["roles"]:
        raise RuntimeError(f"canonical role counts changed: {dataset}/{counts}")
    identities = {
        role: {
            str(sample["identity"])
            if "identity" in sample
            else str(sample["image_id"])
            for sample in values
        }
        for role, values in samples.items()
    }
    overlap = {
        f"{left}_{right}": len(identities[left] & identities[right])
        for index, left in enumerate(config["roles"])
        for right in list(config["roles"])[index + 1 :]
    }
    if any(overlap.values()):
        raise RuntimeError(f"identity leakage: {dataset}/{overlap}")
    manifests = {role: read_json(common_manifest(dataset, role)) for role in config["roles"]}
    if any(value.get("cloud_model_sha256") != config["cloud_model_sha256"] for value in manifests.values()):
        raise RuntimeError(f"cloud cache changed: {dataset}")
    required = _source_required(dataset)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[:20])
    fold_path = (
        FACTORIAL_ROOT / "runs/rewards/fit/folds.json"
        if dataset == "coco"
        else OREGON_ROOT / "runs/rewards/fit/folds.json"
    )
    observed_folds = [list(map(int, fold)) for fold in read_json(fold_path)["folds"]]
    fit_ids = {int(sample["image_id"]) for sample in samples["fit"]}
    fold_ids = [image_id for fold in observed_folds for image_id in fold]
    if len(observed_folds) != 5 or set(fold_ids) != fit_ids or len(fold_ids) != len(set(fold_ids)):
        raise RuntimeError(f"OOF folds changed: {dataset}")
    hashes = {str(path.resolve()): file_sha256(path) for path in required}
    report = {
        "complete": True,
        "dataset": dataset,
        "role_counts": counts,
        "identity_overlap": overlap,
        "fold_sizes": list(map(len, observed_folds)),
        "cloud_model_sha256": config["cloud_model_sha256"],
        "required_sources": len(required),
        "source_digest": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "source_hashes": hashes,
        "final_is_hidden": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "source_digest": report["source_digest"]})
    return report


def run_preflight_all() -> dict[str, Any]:
    reports = {dataset: run_preflight(dataset) for dataset in PROTOCOL["datasets"]}
    result = {"complete": True, "datasets": reports}
    atomic_json(RUNS / "preflight_report.json", result)
    return result


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_oregon_maur_rewards() -> dict[str, Any]:
    output = RUNS / "oregon/rewards"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    run_preflight("oregon")
    source_runs = OREGON_ROOT / "runs"
    reports = {}
    original_runs, original_protocol = OREGON_REWARD.RUNS, OREGON_REWARD.PROTOCOL
    try:
        for role in ("fit", "calibration"):
            OREGON_REWARD.RUNS = source_runs
            data = OREGON_REWARD.load_role(role)
            manifest = read_json(common_manifest("oregon", role))
            maur_path = Path(manifest["maur_action_sources"]["current@640"])
            data["actions"]["current@640"] = OREGON_REWARD.group_predictions(maur_path)
            source_role = source_runs / f"rewards/{role}"
            folds_value = [list(map(int, fold)) for fold in read_json(source_role / "folds.json")["folds"]]
            contexts = {int(key): list(map(int, value)) for key, value in read_json(source_role / "contexts.json").items()}
            membership = {
                int(image_id): fold_index
                for fold_index, fold in enumerate(folds_value)
                for image_id in fold
            }
            OREGON_REWARD.RUNS = RUNS / "oregon"
            reward_protocol = dict(original_protocol)
            reward_protocol["reward"] = dict(original_protocol["reward"])
            reward_protocol["reward"]["workers"] = int(PROTOCOL["resource"]["initial_workers"])
            OREGON_REWARD.PROTOCOL = reward_protocol
            shard = OREGON_REWARD.run_action(role, "current@640", data, contexts, membership)
            rows = _load_jsonl(output / f"{role}/current__640/rows.jsonl")
            source_reward = load_pickle(source_role / "reward_map.pkl")
            source_metric = load_pickle(source_role / "metric_map.pkl")
            rewards = {
                int(image_id): {
                    "original": float(values["original"]),
                    "q90": float(values["q90"]),
                }
                for image_id, values in source_reward.items()
            }
            metrics = {
                int(image_id): {
                    "original": dict(source_metric[int(image_id)]["original"]),
                    "q90": dict(source_metric[int(image_id)]["q90"]),
                }
                for image_id in source_reward
            }
            for row in rows:
                image_id = int(row["target_image_id"])
                rewards[image_id]["current@640"] = float(row["reward_ap50_95"])
                metrics[image_id]["current@640"] = {
                    "ap50": float(row["reward_ap50"]),
                    "ap50_95": float(row["reward_ap50_95"]),
                }
            expected = {"original", "q90", "current@640"}
            if any(set(values) != expected for values in rewards.values()):
                raise RuntimeError(f"Oregon MAUR reward matrix incomplete: {role}")
            role_output = output / role
            atomic_pickle(role_output / "reward_map.pkl", rewards)
            atomic_pickle(role_output / "metric_map.pkl", metrics)
            atomic_json(role_output / "folds.json", {"folds": folds_value})
            atomic_json(role_output / "contexts.json", {str(key): value for key, value in contexts.items()})
            reports[role] = {
                "images": len(rewards),
                "actions": 3,
                "maur_shard": shard,
                "reward_map_sha256": file_sha256(role_output / "reward_map.pkl"),
                "metric_map_sha256": file_sha256(role_output / "metric_map.pkl"),
            }
    finally:
        OREGON_REWARD.RUNS = original_runs
        OREGON_REWARD.PROTOCOL = original_protocol
    report = {
        "complete": True,
        "roles": reports,
        "fusion": "MAUR",
        "detector_inference_performed": False,
        "final_accessed": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(output / "report.json")})
    return report


_TRAIN_STATE: dict[str, Any] = {}


def job_root(job: Mapping[str, Any]) -> Path:
    token = "full" if job["fold"] == "full" else f"fold_{int(job['fold'])}"
    return RUNS / f"{job['dataset']}/training/jobs/{job['method']}/seed_{job['seed']}/{token}"


def _subset(values: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [values[index] for index in indices]


def _initialize_train_state() -> None:
    _TRAIN_STATE.clear()
    for dataset in PROTOCOL["datasets"]:
        records = {role: role_records(dataset, role, "OQR") for role in PROTOCOL["datasets"][dataset]["roles"]}
        category_ids = CORE.category_ids_from_gt(
            read_json(adapter(dataset).gt_path("fit")),
            expected_count=int(PROTOCOL["datasets"][dataset]["category_count"]),
        )
        _TRAIN_STATE[dataset] = {
            "records": records,
            "folds": folds(dataset),
            "category_ids": category_ids,
        }
        if dataset == "coco":
            metric_path = FACTORIAL_ROOT / "runs/rewards/fit/metric_map.pkl"
            _TRAIN_STATE[dataset]["metric_map"] = load_pickle(metric_path)
        else:
            _, metrics = OREGON_TRAIN.load_rewards("fit")
            _TRAIN_STATE[dataset]["metric_map"] = metrics


def _external_features(dataset: str, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return training_adapter(dataset).external_features(
        records, _TRAIN_STATE[dataset]["category_ids"], PROTOCOL
    )


def train_job(job: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(int(PROTOCOL["resource"]["torch_threads_per_worker"]))
    output = job_root(job)
    completion = output / "completion.json"
    if is_complete(completion):
        report = read_json(output / "report.json")
        if report["artifact_sha256"] != file_sha256(output / "artifact.pkl"):
            raise RuntimeError(f"training artifact changed: {job['id']}")
        return report
    dataset, method = str(job["dataset"]), str(job["method"])
    state = _TRAIN_STATE[dataset]
    space = "OQR" if method == "CAVS-OQR-MAUR" else "OQ"
    fit_records = filter_records(state["records"]["fit"], PROTOCOL["action_spaces"][space])
    image_ids = [int(record["image_id"]) for record in fit_records]
    if job["fold"] == "full":
        train_indices = list(range(len(fit_records)))
        predictions = {
            role: filter_records(state["records"][role], PROTOCOL["action_spaces"][space])
            for role in ("calibration", "final")
        }
    else:
        heldout_ids = set(state["folds"][int(job["fold"])])
        heldout = [index for index, image_id in enumerate(image_ids) if image_id in heldout_ids]
        train_indices = [index for index, image_id in enumerate(image_ids) if image_id not in heldout_ids]
        if len(heldout) != len(heldout_ids):
            raise RuntimeError(f"OOF population changed: {job['id']}")
        predictions = {"fit_oof": _subset(fit_records, heldout)}
    started = time.time()
    if method in {"ORIC-AP5095", "MORIC-AP5095"}:
        features = _external_features(dataset, fit_records)
        targets = np.asarray(
            [float(state["metric_map"][image_id]["original"]["ap50_95"]) for image_id in image_ids],
            dtype=np.float64,
        )
        artifact = training_adapter(dataset).fit_external(
            method, features[train_indices], targets[train_indices], PROTOCOL, seed=int(job["seed"])
        )
    else:
        rewards = reward_map(dataset, "fit", method)
        matrix = reward_matrix(dataset, fit_records, rewards)
        train_records = _subset(fit_records, train_indices)
        train_rewards = matrix[train_indices]
        model_seed = int(job["seed"]) + (0 if job["fold"] == "full" else int(job["fold"]) * 10_007)
        if method == "CostInValue-OQ":
            artifact = TRAINING.fit_cost_in_value(
                training_adapter(dataset), train_records, train_rewards, PROTOCOL,
                category_count=len(state["category_ids"]), seed=model_seed,
            )
        elif method == "CAVS-OQR-MAUR":
            artifact = training_adapter(dataset).fit_model(
                "current_action", train_records, train_rewards, PROTOCOL,
                category_count=len(state["category_ids"]), seed=model_seed,
            )
        else:
            raise ValueError(method)
    arrays: dict[str, np.ndarray] = {}
    for role, records in predictions.items():
        if method in {"ORIC-AP5095", "MORIC-AP5095"}:
            values = training_adapter(dataset).predict_external(
                artifact, _external_features(dataset, records)
            )
        elif method == "CostInValue-OQ":
            values = TRAINING.predict_cost_in_value(
                training_adapter(dataset), artifact, records, PROTOCOL
            )
        else:
            values = training_adapter(dataset).predict_model(
                artifact, records, PROTOCOL, category_count=len(state["category_ids"])
            )
        arrays[f"scores_{role}"] = np.asarray(values, dtype=np.float64)
        arrays[f"image_ids_{role}"] = np.asarray([int(record["image_id"]) for record in records], dtype=np.int64)
    output.mkdir(parents=True, exist_ok=True)
    atomic_pickle(
        output / "artifact.pkl",
        {
            "job": dict(job),
            "action_space": space,
            "category_ids": state["category_ids"],
            "cloud_model_sha256": PROTOCOL["datasets"][dataset]["cloud_model_sha256"],
            "artifact": artifact,
        },
    )
    atomic_npz(output / "scores.npz", **arrays)
    metadata = artifact.get("metadata", artifact.get("checkpoint", {}))
    report = {
        "complete": True,
        "id": job["id"],
        "dataset": dataset,
        "method": method,
        "seed": int(job["seed"]),
        "fold": job["fold"],
        "fit_images": len(train_indices),
        "selected_epoch": int(metadata.get("selected_epoch", -1)),
        "elapsed_seconds": time.time() - started,
        "artifact_sha256": file_sha256(output / "artifact.pkl"),
        "scores_sha256": file_sha256(output / "scores.npz"),
        "old_learner_checkpoint_reused": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "artifact_sha256": report["artifact_sha256"]})
    print(f"[train] {job['id']} {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def protected_jobs_complete() -> bool:
    return all((REPO / path).is_file() for path in PROTOCOL["resource"]["protected_completion_markers"])


def allowed_workers() -> int:
    initial = int(PROTOCOL["resource"]["initial_workers"])
    cpu = max(1, os.cpu_count() or 1)
    override = CORE.runtime_worker_override(
        os.environ.get("CAVS_RUNTIME_WORKERS"), cpu_count=cpu
    )
    if override is not None:
        return override
    if not protected_jobs_complete():
        return initial
    try:
        load = float(Path("/proc/loadavg").read_text().split()[0])
    except (FileNotFoundError, ValueError):
        load = 0.0
    idle = max(1.0, cpu - load)
    resource_limit = max(initial, int(idle / float(PROTOCOL["resource"]["minimum_idle_cpus_per_new_worker"])))
    return min(int(PROTOCOL["resource"]["maximum_workers"]), resource_limit)


def _job_process(job: Mapping[str, Any]) -> None:
    train_job(job)


def run_train() -> dict[str, Any]:
    output = RUNS / "training"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    run_preflight_all()
    run_oregon_maur_rewards()
    _initialize_train_state()
    pending = [job for job in CORE.training_jobs(PROTOCOL) if not is_complete(job_root(job) / "completion.json")]
    running: dict[int, tuple[mp.Process, dict[str, Any]]] = {}
    started = time.time()
    while pending or running:
        limit = allowed_workers()
        while pending and len(running) < limit:
            job = pending.pop(0)
            process = mp.get_context("fork").Process(target=_job_process, args=(job,))
            process.start()
            running[process.pid] = (process, job)
        time.sleep(1.0)
        for pid, (process, job) in list(running.items()):
            if process.is_alive():
                continue
            process.join()
            del running[pid]
            if process.exitcode != 0 or not is_complete(job_root(job) / "completion.json"):
                raise RuntimeError(f"training worker failed: {job['id']}/{process.exitcode}")
        if pending and not running:
            time.sleep(2.0)
    _TRAIN_STATE.clear()
    jobs = CORE.training_jobs(PROTOCOL)
    reports = [read_json(job_root(job) / "report.json") for job in jobs]
    report = {
        "complete": True,
        "jobs": len(jobs),
        "elapsed_seconds": time.time() - started,
        "nice": int(PROTOCOL["resource"]["nice"]),
        "protected_jobs_complete_at_end": protected_jobs_complete(),
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "jobs": len(reports)})
    return report


def _normalize_action_scores(
    values: np.ndarray, *, images: int, actions: int
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        if array.shape[0] != images:
            raise RuntimeError(f"score population changed: {array.shape}/{images}")
        array = np.repeat(array[None, :, :], len(PROTOCOL["budgets"]), axis=0)
    if array.ndim != 3 or array.shape[0] != len(PROTOCOL["budgets"]) or array.shape[1] != images:
        raise RuntimeError(f"action score shape changed: {array.shape}")
    if array.shape[2] < actions:
        raise RuntimeError(f"action score width changed: {array.shape[2]}<{actions}")
    return array[:, :, :actions]


def _npz_scores(path: Path, role: str, *, fallback: bool = False) -> tuple[list[int], np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        id_keys = [f"image_ids_{role}", "image_ids"]
        score_keys = [f"scores_{role}", "scores"]
        id_key = next((key for key in id_keys if key in source), None)
        score_key = next((key for key in score_keys if key in source), None)
        if id_key is None or score_key is None:
            if not fallback:
                raise RuntimeError(f"score artifact schema changed: {path}/{source.files}")
            id_key, score_key = source.files[0], source.files[1]
        return list(map(int, source[id_key])), np.asarray(source[score_key], dtype=np.float64)


def source_score_path(dataset: str, method: str, seed: int, role: str) -> Path:
    config = PROTOCOL["datasets"][dataset]
    if method == "CAVS-OQ-Core":
        root = REPO / config["dedicated_oq_source"]
        if dataset == "coco":
            return (
                root / f"runs/calibration/action/jobs/Current-OQ/seed_{seed}/calibration_scores.npz"
                if role == "calibration"
                else root / f"runs/final/action/jobs/Current-OQ/seed_{seed}/scores.npz"
            )
        return root / f"runs/learners/jobs/CurrentActionValue-OQ/seed_{seed}/full/scores.npz"
    if method == "CAVS-OQR-Reversible":
        return adapter(dataset).source_score_path(seed, "full")
    if method == "CAVS-OQR-MAUR":
        if dataset == "coco":
            return FACTORIAL_ROOT / f"runs/training/jobs/CAVS-Current-MAUR/seed_{seed}/full/scores.npz"
        return job_root(
            {"dataset": dataset, "method": method, "seed": seed, "fold": "full"}
        ) / "scores.npz"
    if method == "CostInValue-OQ":
        return job_root(
            {"dataset": dataset, "method": method, "seed": seed, "fold": "full"}
        ) / "scores.npz"
    if method in {"ORIC-AP5095", "MORIC-AP5095"}:
        if dataset == "coco":
            return job_root(
                {"dataset": dataset, "method": method, "seed": seed, "fold": "full"}
            ) / "scores.npz"
        root = REPO / config["dedicated_oq_source"]
        return root / f"runs/learners/jobs/{method}/seed_{seed}/full/scores.npz"
    raise ValueError(method)


def action_scores(dataset: str, method: str, seed: int, role: str) -> np.ndarray:
    records = role_records(dataset, role, "OQR" if "OQR" in method else "OQ")
    expected = [int(record["image_id"]) for record in records]
    ids, values = _npz_scores(source_score_path(dataset, method, seed, role), role)
    if ids != expected:
        raise RuntimeError(f"score/image alignment changed: {dataset}/{method}/{seed}/{role}")
    return _normalize_action_scores(
        values,
        images=len(records),
        actions=CORE.maximum_action_width(records),
    )


def scalar_external_scores(dataset: str, method: str, seed: int, role: str) -> np.ndarray:
    expected = [int(record["image_id"]) for record in role_records(dataset, role, "OQ")]
    ids, values = _npz_scores(source_score_path(dataset, method, seed, role), role)
    if ids != expected:
        raise RuntimeError(f"external score/image alignment changed: {dataset}/{method}/{seed}/{role}")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != len(expected):
        raise RuntimeError(f"external score shape changed: {array.shape}")
    return array


def ensemble_action_scores(dataset: str, method: str, role: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    single = {
        str(seed): action_scores(dataset, method, int(seed), role)
        for seed in PROTOCOL["datasets"][dataset]["model_seeds"]
    }
    return single, np.mean(list(single.values()), axis=0)


def ensemble_external_scores(dataset: str, method: str, role: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    single = {
        str(seed): scalar_external_scores(dataset, method, int(seed), role)
        for seed in PROTOCOL["datasets"][dataset]["model_seeds"]
    }
    return single, np.mean(list(single.values()), axis=0)


def action_score_map(
    records: Sequence[Mapping[str, Any]], values: np.ndarray
) -> dict[int, dict[str, float]]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(records):
        raise ValueError("action score population changed")
    output: dict[int, dict[str, float]] = {}
    for row, record in enumerate(records):
        if len(record["actions"]) > matrix.shape[1]:
            raise ValueError(
                f"action score width changed: {record.get('image_id')}/"
                f"{len(record['actions'])}>{matrix.shape[1]}"
            )
        output[int(record["image_id"])] = {
            str(action["action_id"]): float(matrix[row, column])
            for column, action in enumerate(record["actions"])
        }
    return output


def shared_action_maps_oq(dataset: str) -> dict[str, dict[int, str]]:
    source_adapter = adapter(dataset)
    artifact = UNIFIED.train_shared_action(source_adapter)
    benchmark = load_module(
        f"final_completion_benchmark_{dataset}",
        REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/benchmark_core.py",
    )
    output: dict[str, dict[int, str]] = {}
    for role in ("calibration", "final"):
        samples = source_adapter.load_role(role)["samples"]
        probabilities = benchmark.predict_weighted_logistic(
            artifact["q90_model"], UNIFIED._shared_features(samples, roi=False)
        )
        output[role] = UNIFIED.CORE.shared_action_map(
            samples,
            q90_probabilities=probabilities,
            roi_probabilities={},
            q90_threshold=float(artifact["q90_threshold"]),
            roi_threshold=float("inf"),
        )
        if any(action not in {"original", "q90"} for action in output[role].values()):
            raise RuntimeError(f"No-ROI shared action changed: {dataset}/{role}")
    return output


def data_with_fusion(dataset: str, role: str, fusion: str) -> dict[str, Any]:
    source_adapter = adapter(dataset)
    return (
        source_adapter.load_role(role)
        if fusion == "Reversible"
        else SAME_MAUR.data_with_fusion(source_adapter, role, "MAUR")
    )


def mixed_predictions(
    dataset: str, role: str, selection: Mapping[int, str], fusion: str
) -> dict[int, list[dict[str, Any]]]:
    source_adapter = adapter(dataset)
    data = data_with_fusion(dataset, role, fusion)
    predictions, _ = source_adapter.cloud.CANONICAL.mixed_predictions(
        data, dict(selection), float(PROTOCOL["model"]["deploy_confidence"])
    )
    return predictions


def evaluate_selection(
    dataset: str,
    role: str,
    selection: Mapping[int, str],
    *,
    method: str,
    policy: str,
    mode: str,
    budget: float,
    seed: int | str,
    fusion: str,
) -> dict[str, Any]:
    source_adapter = adapter(dataset)
    data = data_with_fusion(dataset, role, fusion)
    predictions = mixed_predictions(dataset, role, selection, fusion)
    metric = data["evaluator"].evaluate(predictions)
    validation = CORE.validate_selection(data["samples"], selection, mode=mode, budget=budget)
    counts = Counter(selection.values())
    return {
        "dataset": dataset,
        "role": role,
        "method": method,
        "policy": policy,
        "mode": mode,
        "budget": float(budget),
        "seed": seed,
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "kb_per_image": validation["request_bytes"] / 1024.0 / len(data["samples"]),
        "request_bytes": validation["request_bytes"],
        "capacity_utilization": validation["utilization"],
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["current@640"]),
        "primary_roi_fusion": fusion,
        "cloud_model_sha256": PROTOCOL["datasets"][dataset]["cloud_model_sha256"],
        "final_is_hidden": False,
        "final_used_for_training_or_selection": False,
    }


def _selection_key(
    role: str, method: str, policy: str, mode: str, budget: float, seed: int | str
) -> str:
    return f"{role}|{method}|{policy}|{mode}|{budget:g}|seed={seed}"


_POINT_STATE: dict[str, Any] = {}


def _point_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_selection(
        str(task["dataset"]),
        str(task["role"]),
        {int(key): str(value) for key, value in task["selection"].items()},
        method=str(task["method"]),
        policy=str(task["policy"]),
        mode=str(task["mode"]),
        budget=float(task["budget"]),
        seed=task["seed"],
        fusion=str(task["fusion"]),
    )


def _joint_selection(
    dataset: str,
    samples: Sequence[Mapping[str, Any]],
    scores: Mapping[int, Mapping[str, float]],
    *,
    mode: str,
    budget: float,
) -> dict[int, str]:
    return UNIFIED.joint_selection(adapter(dataset), samples, scores, mode=mode, budget=budget)


def _scalar_selection(
    dataset: str,
    samples: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    reference: np.ndarray,
    action_map: Mapping[int, str],
    *,
    mode: str,
    budget: float,
) -> dict[int, str]:
    if mode == "fixed_call":
        return UNIFIED.scalar_fixed_call(samples, scores, action_map, budget=budget)
    return UNIFIED.scalar_fixed_byte(
        adapter(dataset), samples, scores, action_map, budget=budget, reference_scores=reference
    )


def run_point(dataset: str) -> dict[str, Any]:
    output = RUNS / f"{dataset}/point"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    run_train()
    source_adapter = adapter(dataset)
    oq_actions = shared_action_maps_oq(dataset)
    oqr_actions = UNIFIED.shared_action_maps(
        source_adapter, UNIFIED.train_shared_action(source_adapter)
    )
    unified_scalar = UNIFIED.scalar_scores(source_adapter)
    records_by_role = {role: role_records(dataset, role, "OQ") for role in ("calibration", "final")}
    oqr_records = {role: role_records(dataset, role, "OQR") for role in ("calibration", "final")}
    action_methods = ("CAVS-OQ-Core", "CostInValue-OQ")
    external_methods = ("ORIC-AP5095", "MORIC-AP5095")
    action_values: dict[str, dict[str, dict[str, np.ndarray] | np.ndarray]] = {}
    external_values: dict[str, dict[str, dict[str, np.ndarray] | np.ndarray]] = {}
    for method in action_methods:
        action_values[method] = {}
        for role in ("calibration", "final"):
            single, ensemble = ensemble_action_scores(dataset, method, role)
            action_values[method][role] = {**single, "ensemble": ensemble}
    for method in external_methods:
        external_values[method] = {}
        for role in ("calibration", "final"):
            single, ensemble = ensemble_external_scores(dataset, method, role)
            external_values[method][role] = {**single, "ensemble": ensemble}
    for method in ("CAVS-OQR-Reversible", "CAVS-OQR-MAUR"):
        action_values[method] = {}
        for role in ("calibration", "final"):
            single, ensemble = ensemble_action_scores(dataset, method, role)
            action_values[method][role] = {**single, "ensemble": ensemble}

    tasks: list[dict[str, Any]] = []
    selections: dict[str, dict[str, str]] = {}
    random_selections: dict[str, dict[str, str]] = {}

    def add(
        role: str,
        method: str,
        policy: str,
        mode: str,
        budget: float,
        seed: int | str,
        selection: Mapping[int, str],
        fusion: str,
        *,
        random: bool = False,
    ) -> None:
        key = _selection_key(role, method, policy, mode, budget, seed)
        target = random_selections if random else selections
        target[key] = {str(image_id): str(action) for image_id, action in selection.items()}
        tasks.append(
            {
                "dataset": dataset,
                "role": role,
                "method": method,
                "policy": policy,
                "mode": mode,
                "budget": budget,
                "seed": seed,
                "selection": dict(selection),
                "fusion": fusion,
                "random": random,
            }
        )

    for role in ("calibration", "final"):
        samples = source_adapter.load_role(role)["samples"]
        for method in action_methods:
            for seed_token, matrices in action_values[method][role].items():
                for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
                    score_map = action_score_map(records_by_role[role], matrices[budget_index])
                    for mode in PROTOCOL["modes"]:
                        selection = _joint_selection(dataset, samples, score_map, mode=mode, budget=budget)
                        add(role, method, "Joint-OQ", mode, budget, seed_token, selection, "Reversible")
                        image_score, action_map = CORE.two_stage_projection(score_map)
                        two_values = np.asarray([image_score[int(sample["image_id"])] for sample in samples])
                        cal_matrices = action_values[method]["calibration"][seed_token]
                        cal_map = action_score_map(records_by_role["calibration"], cal_matrices[budget_index])
                        cal_image_score, _ = CORE.two_stage_projection(cal_map)
                        cal_samples = source_adapter.load_role("calibration")["samples"]
                        reference = np.asarray([cal_image_score[int(sample["image_id"])] for sample in cal_samples])
                        two_selection = _scalar_selection(
                            dataset, samples, two_values, reference, action_map, mode=mode, budget=budget
                        )
                        add(role, f"{method}-TwoStage", "TwoStage-OQ", mode, budget, seed_token, two_selection, "Reversible")
        for method in external_methods:
            for seed_token, values in external_values[method][role].items():
                reference = external_values[method]["calibration"][seed_token]
                for budget in map(float, PROTOCOL["budgets"]):
                    for mode in PROTOCOL["modes"]:
                        selection = _scalar_selection(
                            dataset, samples, values, reference, oq_actions[role], mode=mode, budget=budget
                        )
                        add(role, method, "SharedAction-OQ", mode, budget, seed_token, selection, "Reversible")
                        oqr_selection = _scalar_selection(
                            dataset, samples, values, reference, oqr_actions[role], mode=mode, budget=budget
                        )
                        add(role, method, "SharedAction-OQR-MAUR", mode, budget, seed_token, oqr_selection, "MAUR")
        confidence = np.asarray(unified_scalar[role]["Confidence"], dtype=np.float64)
        cal_confidence = np.asarray(unified_scalar["calibration"]["Confidence"], dtype=np.float64)
        for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
            for mode in PROTOCOL["modes"]:
                selection = _scalar_selection(
                    dataset, samples, confidence[budget_index], cal_confidence[budget_index],
                    oq_actions[role], mode=mode, budget=budget,
                )
                add(role, "Confidence", "SharedAction-OQ", mode, budget, "fixed", selection, "Reversible")
                oqr_selection = _scalar_selection(
                    dataset, samples, confidence[budget_index], cal_confidence[budget_index],
                    oqr_actions[role], mode=mode, budget=budget,
                )
                add(role, "Confidence", "SharedAction-OQR-MAUR", mode, budget, "fixed", oqr_selection, "MAUR")
            for offset in range(int(PROTOCOL["bootstrap"]["random_seeds"])):
                random_seed = int(PROTOCOL["bootstrap"]["random_base_seed"][dataset]) + offset
                values = np.random.default_rng(random_seed).random(len(samples))
                reference = np.random.default_rng(random_seed).random(
                    len(source_adapter.load_role("calibration")["samples"])
                )
                for mode in PROTOCOL["modes"]:
                    selection = _scalar_selection(
                        dataset, samples, values, reference, oq_actions[role], mode=mode, budget=budget
                    )
                    add(role, "Random", "SharedAction-OQ", mode, budget, random_seed, selection, "Reversible", random=True)
                    oqr_selection = _scalar_selection(
                        dataset, samples, values, reference, oqr_actions[role], mode=mode, budget=budget
                    )
                    add(role, "Random", "SharedAction-OQR-MAUR", mode, budget, random_seed, oqr_selection, "MAUR", random=True)

        for method, fusion in (("CAVS-OQR-Reversible", "Reversible"), ("CAVS-OQR-MAUR", "MAUR")):
            for seed_token, matrices in action_values[method][role].items():
                for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
                    base_map = action_score_map(oqr_records[role], matrices[budget_index])
                    for mode in PROTOCOL["modes"]:
                        selection = _joint_selection(dataset, samples, base_map, mode=mode, budget=budget)
                        add(role, method, "Joint-OQR", mode, budget, seed_token, selection, fusion)
                        oq_map = UNIFIED.CORE.mask_scores(base_map, PROTOCOL["action_spaces"]["OQ"])
                        oq_selection = _joint_selection(dataset, samples, oq_map, mode=mode, budget=budget)
                        add(role, f"{method}-OQMask", "Joint-OQ-from-OQR", mode, budget, seed_token, oq_selection, fusion)

    workers = min(allowed_workers(), len(tasks))
    with mp.get_context("fork").Pool(processes=workers) as pool:
        rows = list(pool.imap_unordered(_point_worker, tasks, chunksize=1))
    fixed_rows = [row for row in rows if row["method"] != "Random"]
    random_rows = [row for row in rows if row["method"] == "Random"]
    random_aggregate: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in random_rows:
        key = tuple(row[name] for name in ("dataset", "role", "policy", "mode", "budget"))
        grouped.setdefault(key, []).append(row)
    for values in grouped.values():
        if len(values) != 50:
            raise RuntimeError("Random-50 point matrix incomplete")
        item = dict(values[0])
        for name in (
            "official_ap50_95", "official_ap50", "official_ap75", "kb_per_image",
            "request_bytes", "capacity_utilization", "selected_images", "original_count",
            "q90_count", "roi_count",
        ):
            item[name] = float(np.mean([float(value[name]) for value in values]))
        item["seed"] = "mean50"
        item["random_seed_count"] = 50
        random_aggregate.append(item)
    aggregate = fixed_rows + random_aggregate
    CORE.validate_unique_rows(aggregate)
    low_capacity = [
        row
        for row in aggregate
        if row["mode"] == "fixed_byte" and float(row["capacity_utilization"]) < 0.98
    ]
    if low_capacity:
        first = low_capacity[0]
        raise RuntimeError(
            "fixed-byte capacity utilization below 98%: "
            f"{first['method']}/{first['policy']}/{first['budget']}/{first['capacity_utilization']}"
        )
    atomic_csv(output / "results.csv", aggregate)
    atomic_csv(output / "random_seed_results.csv", random_rows)
    atomic_json(output / "selections.json", selections)
    atomic_json(output / "random_selections.json", random_selections)
    report = {
        "complete": True,
        "dataset": dataset,
        "aggregate_rows": len(aggregate),
        "random_seed_rows": len(random_rows),
        "workers": workers,
        "results_sha256": file_sha256(output / "results.csv"),
        "selections_sha256": file_sha256(output / "selections.json"),
        "random_selections_sha256": file_sha256(output / "random_selections.json"),
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "results_sha256": report["results_sha256"]})
    return report


_BOOT_STATE: dict[str, Any] = {}


def _bootstrap_sample(rng: np.random.Generator, unit: str) -> list[int]:
    if unit == "image":
        values = _BOOT_STATE["image_ids"]
        return [int(values[index]) for index in rng.integers(0, len(values), size=len(values))]
    if unit == "event":
        groups = _BOOT_STATE["event_groups"]
        return [
            int(image_id)
            for index in rng.integers(0, len(groups), size=len(groups))
            for image_id in groups[int(index)]
        ]
    raise ValueError(unit)


def _bootstrap_rng(job_id: str) -> np.random.Generator:
    token = hashlib.sha256(f"{PROTOCOL['bootstrap']['seed']}:{job_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(token[:8], "big"))


def _bootstrap_output(job: Mapping[str, Any]) -> tuple[Path, Path]:
    safe = hashlib.sha256(str(job["id"]).encode()).hexdigest()[:20]
    path = RUNS / f"{job['dataset']}/bootstrap/jobs/{safe}.json"
    return path, path.with_suffix(".partial.json")


def _bootstrap_summary(job: Mapping[str, Any], values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        **{
            key: value
            for key, value in job.items()
            if key not in {"pair_index", "candidate_index", "random_indices"}
        },
        "replicates": len(array),
        "mean_delta": float(array.mean()),
        "median_delta": float(np.median(array)),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "probability_gt_zero": float(np.mean(array > 0)),
        "final_used_for_training_or_selection": False,
    }


def _standard_bootstrap(job: Mapping[str, Any]) -> dict[str, Any]:
    path, partial = _bootstrap_output(job)
    if path.is_file():
        return read_json(path)
    values = list(map(float, read_json(partial)["values"])) if partial.is_file() else []
    rng = _bootstrap_rng(str(job["id"]))
    for _ in values:
        _bootstrap_sample(rng, str(job["unit"]))
    candidate, comparator = _BOOT_STATE["pairs"][int(job["pair_index"])]
    total = int(PROTOCOL["bootstrap"]["replicates"])
    for replicate in range(len(values), total):
        sampled = _bootstrap_sample(rng, str(job["unit"]))
        gt, left, right = _BOOT_STATE["core"]._bootstrap_remap(
            _BOOT_STATE["ground_truth"], candidate, comparator, sampled
        )
        values.append(
            float(_BOOT_STATE["core"]._in_memory_ap(gt, left) - _BOOT_STATE["core"]._in_memory_ap(gt, right))
        )
        if (replicate + 1) % 100 == 0:
            atomic_json(partial, {"values": values})
    result = _bootstrap_summary(job, values)
    atomic_json(path, result)
    partial.unlink(missing_ok=True)
    return result


def _nested_random_bootstrap(job: Mapping[str, Any]) -> dict[str, Any]:
    path, partial = _bootstrap_output(job)
    if path.is_file():
        return read_json(path)
    values = list(map(float, read_json(partial)["values"])) if partial.is_file() else []
    rng = _bootstrap_rng(str(job["id"]))
    for _ in values:
        _bootstrap_sample(rng, str(job["unit"]))
    candidate = _BOOT_STATE["candidate_predictions"][int(job["candidate_index"])]
    random_predictions = [
        _BOOT_STATE["random_predictions"][index] for index in job["random_indices"]
    ]
    if len(random_predictions) != int(PROTOCOL["bootstrap"]["random_seeds"]):
        raise RuntimeError("nested Random-50 prediction population changed")
    total = int(PROTOCOL["bootstrap"]["replicates"])
    for replicate in range(len(values), total):
        sampled = _bootstrap_sample(rng, str(job["unit"]))
        random_ap = []
        candidate_ap = None
        for comparator in random_predictions:
            gt, left, right = _BOOT_STATE["core"]._bootstrap_remap(
                _BOOT_STATE["ground_truth"], candidate, comparator, sampled
            )
            if candidate_ap is None:
                candidate_ap = float(_BOOT_STATE["core"]._in_memory_ap(gt, left))
            random_ap.append(float(_BOOT_STATE["core"]._in_memory_ap(gt, right)))
        values.append(float(candidate_ap - np.mean(random_ap)))
        if (replicate + 1) % 25 == 0:
            atomic_json(partial, {"values": values})
    result = _bootstrap_summary(job, values)
    result["random_seed_count"] = len(random_predictions)
    result["random_nested_per_resample"] = True
    atomic_json(path, result)
    partial.unlink(missing_ok=True)
    return result


def _bootstrap_worker(job: Mapping[str, Any]) -> dict[str, Any]:
    return _nested_random_bootstrap(job) if job["kind"] == "nested_random" else _standard_bootstrap(job)


def run_bootstrap(dataset: str) -> dict[str, Any]:
    output = RUNS / f"{dataset}/bootstrap"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    run_point(dataset)
    source_adapter = adapter(dataset)
    stored = read_json(RUNS / f"{dataset}/point/selections.json")
    random_stored = read_json(RUNS / f"{dataset}/point/random_selections.json")

    def selection(key: str, *, random: bool = False) -> dict[int, str]:
        values = random_stored if random else stored
        if key not in values:
            raise RuntimeError(f"missing frozen selection: {dataset}/{key}")
        return {int(image_id): str(action) for image_id, action in values[key].items()}

    data = source_adapter.load_role("final")
    image_ids = [int(sample["image_id"]) for sample in data["samples"]]
    event_map: dict[str, list[int]] = {}
    for sample in data["samples"]:
        event_map.setdefault(str(sample.get("event_id", sample["image_id"])), []).append(int(sample["image_id"]))
    jobs: list[dict[str, Any]] = []
    pairs: list[tuple[Any, Any]] = []
    candidate_predictions: list[Any] = []
    random_predictions: list[Any] = []
    prediction_cache: dict[tuple[str, str], Any] = {}

    def predictions(key: str, fusion: str, *, random: bool = False) -> Any:
        cache_key = (key, fusion)
        if cache_key not in prediction_cache:
            prediction_cache[cache_key] = mixed_predictions(
                dataset, "final", selection(key, random=random), fusion
            )
        return prediction_cache[cache_key]

    units = list(map(str, PROTOCOL["datasets"][dataset]["bootstrap_units"]))
    seeds = list(map(int, PROTOCOL["datasets"][dataset]["model_seeds"]))
    for budget in map(float, PROTOCOL["budgets"]):
        core_key = _selection_key("final", "CAVS-OQ-Core", "Joint-OQ", "fixed_byte", budget, "ensemble")
        core_comparators = {
            "ORIC-AP5095-3Seed": _selection_key("final", "ORIC-AP5095", "SharedAction-OQ", "fixed_byte", budget, "ensemble"),
            "MORIC-AP5095-3Seed": _selection_key("final", "MORIC-AP5095", "SharedAction-OQ", "fixed_byte", budget, "ensemble"),
            "Confidence": _selection_key("final", "Confidence", "SharedAction-OQ", "fixed_byte", budget, "fixed"),
            "CostInValue-OQ": _selection_key("final", "CostInValue-OQ", "Joint-OQ", "fixed_byte", budget, "ensemble"),
            "TwoStage-OQ": _selection_key("final", "CAVS-OQ-Core-TwoStage", "TwoStage-OQ", "fixed_byte", budget, "ensemble"),
        }
        maur_key = _selection_key("final", "CAVS-OQR-MAUR", "Joint-OQR", "fixed_byte", budget, "ensemble")
        maur_comparators = {
            "CAVS-OQ-from-OQR": _selection_key("final", "CAVS-OQR-MAUR-OQMask", "Joint-OQ-from-OQR", "fixed_byte", budget, "ensemble"),
            "CAVS-OQR-Reversible": _selection_key("final", "CAVS-OQR-Reversible", "Joint-OQR", "fixed_byte", budget, "ensemble"),
            "ORIC-AP5095-3Seed": _selection_key("final", "ORIC-AP5095", "SharedAction-OQR-MAUR", "fixed_byte", budget, "ensemble"),
            "MORIC-AP5095-3Seed": _selection_key("final", "MORIC-AP5095", "SharedAction-OQR-MAUR", "fixed_byte", budget, "ensemble"),
            "Confidence": _selection_key("final", "Confidence", "SharedAction-OQR-MAUR", "fixed_byte", budget, "fixed"),
        }
        for candidate_name, candidate_key, fusion, comparators in (
            ("CAVS-OQ-Core", core_key, "Reversible", core_comparators),
            ("CAVS-OQR-MAUR", maur_key, "MAUR", maur_comparators),
        ):
            candidate = predictions(candidate_key, fusion)
            for comparator_name, comparator_key in comparators.items():
                comparator_fusion = "Reversible" if comparator_name == "CAVS-OQR-Reversible" else fusion
                comparator = predictions(comparator_key, comparator_fusion)
                pair_index = len(pairs)
                pairs.append((candidate, comparator))
                for unit in units:
                    jobs.append(
                        {
                            "id": f"{candidate_name}|vs|{comparator_name}|budget={budget:g}|{unit}",
                            "kind": "paired",
                            "dataset": dataset,
                            "candidate": candidate_name,
                            "comparator": comparator_name,
                            "budget": budget,
                            "unit": unit,
                            "pair_index": pair_index,
                        }
                    )
            policy = "SharedAction-OQ" if candidate_name == "CAVS-OQ-Core" else "SharedAction-OQR-MAUR"
            random_indices = []
            for random_seed in range(
                int(PROTOCOL["bootstrap"]["random_base_seed"][dataset]),
                int(PROTOCOL["bootstrap"]["random_base_seed"][dataset]) + int(PROTOCOL["bootstrap"]["random_seeds"]),
            ):
                key = _selection_key("final", "Random", policy, "fixed_byte", budget, random_seed)
                random_indices.append(len(random_predictions))
                random_predictions.append(predictions(key, fusion, random=True))
            candidate_index = len(candidate_predictions)
            candidate_predictions.append(candidate)
            for unit in units:
                jobs.append(
                    {
                        "id": f"{candidate_name}|vs|Random-Mean50|budget={budget:g}|{unit}",
                        "kind": "nested_random",
                        "dataset": dataset,
                        "candidate": candidate_name,
                        "comparator": "Random-Mean50",
                        "budget": budget,
                        "unit": unit,
                        "candidate_index": candidate_index,
                        "random_indices": random_indices,
                    }
                )
    if len({job["id"] for job in jobs}) != len(jobs):
        raise RuntimeError("duplicate bootstrap jobs")
    if len(jobs) != CORE.expected_bootstrap_jobs(PROTOCOL, dataset):
        raise RuntimeError(f"bootstrap job matrix changed: {dataset}/{len(jobs)}")
    _BOOT_STATE.clear()
    _BOOT_STATE.update(
        ground_truth=read_json(source_adapter.gt_path("final")),
        image_ids=image_ids,
        event_groups=[event_map[key] for key in sorted(event_map)],
        core=source_adapter.bootstrap_core,
        pairs=pairs,
        candidate_predictions=candidate_predictions,
        random_predictions=random_predictions,
    )
    workers = min(allowed_workers(), len(jobs))
    with mp.get_context("fork").Pool(processes=workers) as pool:
        results = list(pool.imap_unordered(_bootstrap_worker, jobs, chunksize=1))
    _BOOT_STATE.clear()
    results.sort(key=lambda row: (row["unit"], row["budget"], row["candidate"], row["comparator"]))
    if any(int(row["replicates"]) != int(PROTOCOL["bootstrap"]["replicates"]) for row in results):
        raise RuntimeError("bootstrap matrix incomplete")
    atomic_csv(output / "results.csv", results)
    report = {
        "complete": True,
        "dataset": dataset,
        "jobs": len(results),
        "paired_jobs": sum(row["kind"] == "paired" for row in results),
        "nested_random_jobs": sum(row["kind"] == "nested_random" for row in results),
        "replicates_per_job": int(PROTOCOL["bootstrap"]["replicates"]),
        "random_seed_count": int(PROTOCOL["bootstrap"]["random_seeds"]),
        "workers": workers,
        "results_sha256": file_sha256(output / "results.csv"),
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "results_sha256": report["results_sha256"]})
    return report


def run_report() -> dict[str, Any]:
    for dataset in PROTOCOL["datasets"]:
        run_bootstrap(dataset)
    rows = [
        row
        for dataset in PROTOCOL["datasets"]
        for row in read_csv(RUNS / f"{dataset}/point/results.csv")
    ]
    boot = [
        row
        for dataset in PROTOCOL["datasets"]
        for row in read_csv(RUNS / f"{dataset}/bootstrap/results.csv")
    ]
    final = [row for row in rows if row["role"] == "final" and row["mode"] == "fixed_byte"]
    seed_summary = []
    grouped_seed: dict[tuple[str, str, str, float], list[dict[str, str]]] = {}
    for row in final:
        try:
            int(row["seed"])
        except ValueError:
            continue
        key = (row["dataset"], row["method"], row["policy"], float(row["budget"]))
        grouped_seed.setdefault(key, []).append(row)
    for (dataset, method, policy, budget), values in sorted(grouped_seed.items()):
        if len(values) != 3:
            continue
        ap = np.asarray([float(value["official_ap50_95"]) for value in values])
        seed_summary.append(
            {
                "dataset": dataset,
                "method": method,
                "policy": policy,
                "budget": budget,
                "single_model_mean_ap50_95": float(ap.mean()),
                "single_model_sample_sd_ap50_95": float(ap.std(ddof=1)),
                "model_seeds": ",".join(sorted(value["seed"] for value in values)),
            }
        )
    atomic_csv(RUNS / "seed_summary.csv", seed_summary)
    lines = [
        "# CAVS 最终补充实验结果",
        "",
        "> COCO val2017 与 Oregon 已打开 final 均为 post-hoc paper-grade 复用，不是新 hidden final。",
        "",
        "## Fixed-byte 主结果",
        "",
        "| 数据集 | 方法 | Seed | 10% | 25% | 40% |",
        "|---|---|---:|---:|---:|---:|",
    ]
    methods = (
        "CAVS-OQ-Core", "CAVS-OQ-Core-TwoStage", "CostInValue-OQ",
        "ORIC-AP5095", "MORIC-AP5095", "Confidence", "Random",
        "CAVS-OQR-MAUR", "CAVS-OQR-Reversible",
    )
    for dataset in PROTOCOL["datasets"]:
        for method in methods:
            selected = [
                row for row in final
                if row["dataset"] == dataset and row["method"] == method
                and row["seed"] in {"ensemble", "fixed", "mean50"}
                and (row["policy"] in {"Joint-OQ", "TwoStage-OQ", "SharedAction-OQ", "Joint-OQR"})
            ]
            if not selected:
                continue
            by_budget = {float(row["budget"]): row for row in selected}
            if set(by_budget) != set(map(float, PROTOCOL["budgets"])):
                continue
            values = [
                f"{float(by_budget[budget]['official_ap50_95']):.6f} / {float(by_budget[budget]['kb_per_image']):.2f}"
                for budget in map(float, PROTOCOL["budgets"])
            ]
            lines.append(f"| {dataset} | {method} | {selected[0]['seed']} | " + " | ".join(values) + " |")
    lines.extend(
        (
            "",
            "## 统计与边界",
            "",
            f"- Paired/nested bootstrap rows: {len(boot)}.",
            f"- Three-seed mean/SD rows: {len(seed_summary)}.",
            "- Random 比较在每个重采样中先计算 50 个随机策略 AP，再取均值。",
            "- CAVS-OQ-Core 是独立 OQ 训练；OQ-from-OQR 不进入主方法行。",
            "- final_used_for_training_or_selection=false。",
        )
    )
    notes = ROOT / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "RESULTS_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "complete": True,
        "point_rows": len(rows),
        "bootstrap_rows": len(boot),
        "seed_summary_rows": len(seed_summary),
        "seed_summary_sha256": file_sha256(RUNS / "seed_summary.csv"),
        "results_zh_sha256": file_sha256(notes / "RESULTS_ZH.md"),
        "final_is_hidden": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(RUNS / "report.json", report)
    atomic_json(RUNS / "completion.json", {"complete": True, "report_sha256": file_sha256(RUNS / "report.json")})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "rewards", "train", "point", "bootstrap", "report", "all"),
        default="all",
    )
    parser.add_argument("--dataset", choices=("coco", "oregon", "all"), default="all")
    args = parser.parse_args()
    datasets = list(PROTOCOL["datasets"]) if args.dataset == "all" else [args.dataset]
    if args.stage == "preflight":
        result = {dataset: run_preflight(dataset) for dataset in datasets}
    elif args.stage == "rewards":
        result = run_oregon_maur_rewards()
    elif args.stage == "train":
        result = run_train()
    elif args.stage == "point":
        result = {dataset: run_point(dataset) for dataset in datasets}
    elif args.stage == "bootstrap":
        result = {dataset: run_bootstrap(dataset) for dataset in datasets}
    elif args.stage in {"report", "all"}:
        result = run_report()
    else:
        raise ValueError(args.stage)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
