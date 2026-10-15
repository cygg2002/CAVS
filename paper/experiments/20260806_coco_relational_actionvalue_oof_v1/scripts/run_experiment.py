#!/usr/bin/env python3
"""Run the cache-only COCO Relational Action-Value OOF diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import gc
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
ACTIONVALUE_SCRIPTS = SOURCES["actionvalue"] / "scripts"
SET_SCRIPTS = SOURCES["set_model"] / "scripts"
VALUE_SCRIPTS = SOURCES["value_model"] / "scripts"
REWARD_SCRIPTS = SOURCES["reward_core"] / "scripts"
CANONICAL_SCRIPTS = CANONICAL_ROOT / "scripts"
CA_AORIC = REPO / "paper/experiments/CA_AORIC"
sys.path[:0] = [
    str(REPO),
    str(ROOT / "scripts"),
    str(ACTIONVALUE_SCRIPTS),
    str(SET_SCRIPTS),
    str(VALUE_SCRIPTS),
    str(REWARD_SCRIPTS),
    str(CANONICAL_SCRIPTS),
    str(CA_AORIC),
]

from core import (  # noqa: E402
    EXPECTED_ACTIONS,
    EXPECTED_BUDGETS,
    EXPECTED_MODES,
    GLOBAL_METHODS,
    METHODS,
    action_value_objective,
    assert_no_roi_records,
    assert_oof_reward_isolation,
    build_model,
    deterministic_folds,
    deterministic_validation_split,
    evaluate_candidate_gate,
    filter_no_roi_records,
    fold_contained_contexts,
    jaccard,
    should_run_calibration,
    spearman,
    strip_cost_inputs,
    validate_protocol,
    validate_result_matrix,
)
from action_core import schedule_fixed_byte, schedule_fixed_call  # noqa: E402
from experiment_core import ACTION_DIM, SCENE_DIM  # noqa: E402
from paper_core import dual_metric_context_reward  # noqa: E402
import run_multiranker_benchmark as canonical  # noqa: E402
from ca_aoric import CocoMetricBackend  # noqa: E402


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SET_RUNNER = _load_source_module(
    "relational_actionvalue_set_source", SET_SCRIPTS / "run_experiment.py"
)
_record_tensors = _SET_RUNNER._record_tensors


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
    fields = sorted({str(key) for row in rows for key in row})
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


def _configure_canonical() -> None:
    canonical.COMMON = COMMON.parent
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
    values = [
        int(item["id"])
        for item in read_json(COMMON / role / "gt.json")["categories"]
    ]
    if len(values) != 80 or len(values) != len(set(values)):
        raise RuntimeError("Canonical COCO category schema changed")
    return values


def records_for_role(role: str) -> list[dict[str, Any]]:
    source = __import__("action_core").build_action_records(
        load_role(role)["samples"],
        load_role(role)["actions"]["edge"],
        category_ids=category_ids(role),
    )
    return filter_no_roi_records(source)


def _role_ids_from_samples(role: str) -> list[int]:
    samples = read_json(COMMON / role / "samples.json")
    ids = [int(sample["image_id"]) for sample in samples]
    if len(ids) != int(PROTOCOL["roles"][role]) or len(ids) != len(set(ids)):
        raise RuntimeError(f"Canonical {role} sample population changed")
    return ids


def _source_files() -> list[Path]:
    paths = [
        PROTOCOL_PATH,
        ROOT / "scripts/core.py",
        SCRIPT,
        ACTIONVALUE_SCRIPTS / "action_core.py",
        SET_SCRIPTS / "experiment_core.py",
        REWARD_SCRIPTS / "paper_core.py",
        CANONICAL_SCRIPTS / "run_multiranker_benchmark.py",
    ]
    for role in PROTOCOL["roles"]:
        root = COMMON / role
        manifest = read_json(root / "source_manifest.json")
        paths.extend((root / "samples.json", root / "gt.json", root / "source_manifest.json"))
        paths.extend(
            Path(manifest["action_sources"][action])
            for action in EXPECTED_ACTIONS
        )
    return list(dict.fromkeys(map(Path, paths)))


def run_preflight() -> dict[str, Any]:
    completion_path = RUNS / "preflight/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "preflight/report.json")
    role_ids = {role: _role_ids_from_samples(role) for role in PROTOCOL["roles"]}
    if set(role_ids["fit"]) & set(role_ids["calibration"]):
        raise RuntimeError("Fit and calibration overlap")
    folds = deterministic_folds(
        role_ids["fit"],
        fold_count=int(PROTOCOL["fold_count"]),
        seed=int(PROTOCOL["fold_seed"]),
    )
    contexts = fold_contained_contexts(
        role_ids["fit"],
        folds=folds,
        context_size=int(PROTOCOL["reward"]["context_size"]),
        seed=int(PROTOCOL["reward"]["seed"]),
    )
    assert_oof_reward_isolation(contexts, folds)
    sources = _source_files()
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen sources: {missing[:3]}")
    hashes = {str(path): file_sha256(path) for path in sources}
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role_counts": {role: len(ids) for role, ids in role_ids.items()},
        "fit_calibration_overlap": 0,
        "fold_sizes": list(map(len, folds)),
        "context_size": int(PROTOCOL["reward"]["context_size"]),
        "context_cross_fold_violations": 0,
        "source_hashes": hashes,
        "source_digest": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "detector_inference_required": False,
        "audit_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    atomic_json(RUNS / "preflight/folds.json", {"folds": folds})
    atomic_json(RUNS / "preflight/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "report_sha256": file_sha256(RUNS / "preflight/report.json"),
            "source_digest": report["source_digest"],
        },
    )
    return report


def _verify_preflight() -> dict[str, Any]:
    completion = read_json(RUNS / "preflight/completion.json")
    report_path = RUNS / "preflight/report.json"
    if completion.get("complete") is not True:
        raise RuntimeError("Preflight is incomplete")
    if completion.get("report_sha256") != file_sha256(report_path):
        raise RuntimeError("Preflight report changed")
    return read_json(report_path)


_REWARD_STATE: dict[str, Any] = {}


def _init_reward_worker(annotation_path: str) -> None:
    _REWARD_STATE["metric"] = CocoMetricBackend(Path(annotation_path))


def _reward_worker(task: tuple[int, str]) -> dict[str, Any]:
    target_id, action_id = int(task[0]), str(task[1])
    data = _REWARD_STATE["data"]
    contexts = _REWARD_STATE["contexts"]
    context_ids = contexts[target_id]
    context_records = [
        {
            "image_id": image_id,
            "edge_predictions": data["actions"]["edge"].get(image_id, []),
        }
        for image_id in context_ids
    ]
    target = {
        "image_id": target_id,
        "edge_predictions": data["actions"]["edge"].get(target_id, []),
        "cloud_predictions": data["actions"][action_id].get(target_id, []),
    }
    row = dual_metric_context_reward(
        _REWARD_STATE["metric"], context_records, target
    )
    row["action_id"] = action_id
    row["reward"] = float(row["reward_ap50_95"])
    row["fold_index"] = int(_REWARD_STATE["membership"].get(target_id, 0))
    return row


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid JSONL line {line_number}: {path}") from error
    return rows


def _reward_contexts(role: str, image_ids: Sequence[int]) -> tuple[list[list[int]], dict[int, list[int]]]:
    if role == "fit":
        folds = [
            list(map(int, fold))
            for fold in read_json(RUNS / "preflight/folds.json")["folds"]
        ]
    elif role == "calibration":
        folds = [sorted(map(int, image_ids))]
    else:
        raise ValueError(f"Forbidden reward role: {role}")
    contexts = fold_contained_contexts(
        image_ids,
        folds=folds,
        context_size=int(PROTOCOL["reward"]["context_size"]),
        seed=int(PROTOCOL["reward"]["seed"]),
    )
    if role == "fit":
        assert_oof_reward_isolation(contexts, folds)
    return folds, contexts


def run_rewards(role: str) -> dict[str, Any]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(f"Forbidden reward role: {role}")
    completion_path = RUNS / f"rewards/{role}/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / f"rewards/{role}/report.json")
    _verify_preflight()
    data = load_role(role)
    image_ids = [int(sample["image_id"]) for sample in data["samples"]]
    folds, contexts = _reward_contexts(role, image_ids)
    membership = {
        int(image_id): fold_index
        for fold_index, fold in enumerate(folds)
        for image_id in fold
    }
    reward_path = RUNS / f"rewards/{role}/rewards.jsonl"
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_jsonl(reward_path)
    completed = [
        (int(row["target_image_id"]), str(row["action_id"])) for row in existing
    ]
    if len(completed) != len(set(completed)):
        raise RuntimeError("Duplicate reward target/action checkpoint")
    expected = {
        (int(image_id), action)
        for image_id in image_ids
        for action in PROTOCOL["candidate_actions"]
    }
    if not set(completed) <= expected:
        raise RuntimeError("Reward checkpoint contains an invalid target/action")
    pending = sorted(expected - set(completed))
    _REWARD_STATE.clear()
    _REWARD_STATE.update(
        {
            "data": data,
            "contexts": contexts,
            "membership": membership,
        }
    )
    workers = min(
        int(PROTOCOL["reward"]["workers"]), max(1, (os.cpu_count() or 1) - 2)
    )
    context = mp.get_context("fork")
    started = time.time()
    with reward_path.open("a", encoding="utf-8") as handle:
        with context.Pool(
            processes=workers,
            initializer=_init_reward_worker,
            initargs=(str(COMMON / role / "gt.json"),),
        ) as pool:
            for count, row in enumerate(
                pool.imap_unordered(_reward_worker, pending, chunksize=1),
                start=len(completed) + 1,
            ):
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                handle.flush()
                if count % int(PROTOCOL["reward"]["progress_every"]) == 0:
                    os.fsync(handle.fileno())
                    print(f"[reward:{role}] {count}/{len(expected)}", flush=True)
        os.fsync(handle.fileno())
    rows = _read_jsonl(reward_path)
    observed = {
        (int(row["target_image_id"]), str(row["action_id"])) for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError(f"Incomplete {role} reward matrix")
    for row in rows:
        target = int(row["target_image_id"])
        context_ids = list(map(int, row["context_image_ids"]))
        if target in context_ids or any(
            membership[value] != membership[target] for value in context_ids
        ):
            raise RuntimeError("Generated reward context violates isolation")
    reward_map: dict[int, dict[str, float]] = {image_id: {} for image_id in image_ids}
    for row in rows:
        reward_map[int(row["target_image_id"])][str(row["action_id"])] = float(
            row["reward"]
        )
    atomic_pickle(RUNS / f"rewards/{role}/reward_map.pkl", reward_map)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": role,
        "images": len(image_ids),
        "rows": len(rows),
        "actions": list(PROTOCOL["candidate_actions"]),
        "context_size": int(PROTOCOL["reward"]["context_size"]),
        "fold_contained": True,
        "cross_fold_violations": 0,
        "reward_means": {
            action: float(
                np.mean(
                    [float(row["reward"]) for row in rows if row["action_id"] == action]
                )
            )
            for action in PROTOCOL["candidate_actions"]
        },
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "fit_used_for_training": role == "fit",
        "calibration_used_for_training_or_selection": False,
        "audit_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    atomic_json(RUNS / f"rewards/{role}/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "rewards_sha256": file_sha256(reward_path),
            "map_sha256": file_sha256(RUNS / f"rewards/{role}/reward_map.pkl"),
            "report_sha256": file_sha256(RUNS / f"rewards/{role}/report.json"),
        },
    )
    _ROLE_CACHE.pop(role, None)
    _REWARD_STATE.clear()
    gc.collect()
    return report


def _load_reward_map(role: str) -> dict[int, dict[str, float]]:
    completion = read_json(RUNS / f"rewards/{role}/completion.json")
    path = RUNS / f"rewards/{role}/reward_map.pkl"
    if completion.get("complete") is not True:
        raise RuntimeError(f"{role} rewards are incomplete")
    if completion.get("map_sha256") != file_sha256(path):
        raise RuntimeError(f"{role} reward map changed")
    values = load_pickle(path)
    return {
        int(image_id): {str(action): float(reward) for action, reward in actions.items()}
        for image_id, actions in values.items()
    }


def _reward_matrix(
    records: Sequence[Mapping[str, Any]],
    reward_map: Mapping[int, Mapping[str, float]],
) -> np.ndarray:
    values = np.zeros((len(records), int(PROTOCOL["model"]["max_actions"])), dtype=np.float32)
    for image_index, record in enumerate(records):
        image_id = int(record["image_id"])
        if image_id not in reward_map:
            raise RuntimeError(f"Missing rewards for image {image_id}")
        for action_index, action in enumerate(record["actions"]):
            action_id = str(action["action_id"])
            if action_id == "edge":
                continue
            if action_id not in reward_map[image_id]:
                raise RuntimeError(f"Missing reward for {image_id}/{action_id}")
            values[image_index, action_index] = float(reward_map[image_id][action_id])
    return values


def _tensor_protocol() -> dict[str, Any]:
    return {
        "max_boxes": int(PROTOCOL["model"]["max_boxes"]),
        "max_actions": int(PROTOCOL["model"]["max_actions"]),
        "deploy_confidence": float(PROTOCOL["model"]["deploy_confidence"]),
    }


def _role_tensors(role: str) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    records = records_for_role(role)
    assert_no_roi_records(records)
    rewards = _reward_matrix(records, _load_reward_map(role))
    data = strip_cost_inputs(_record_tensors(records, rewards, _tensor_protocol()))
    if data["actions"].shape[1] != 3 or data["target"].shape[1] != 3:
        raise RuntimeError("No-ROI action tensor shape changed")
    return records, data


def _subset_tensors(
    data: Mapping[str, torch.Tensor], indices: Sequence[int]
) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[index] for key, value in data.items()}


def _normalized_targets(
    data: Mapping[str, torch.Tensor], train_indices: Sequence[int]
) -> tuple[dict[str, torch.Tensor], float, float]:
    index = torch.as_tensor(train_indices, dtype=torch.long)
    cloud = data["action_mask"][index] & (data["action_type"][index] != 0)
    values = data["target"][index][cloud]
    mean = float(values.mean())
    std = max(1e-6, float(values.std(unbiased=False)))
    output = dict(data)
    output["target"] = (data["target"] - mean) / std
    return output, mean, std


def _model_inputs(
    data: Mapping[str, torch.Tensor], indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {
        key: data[key][indices]
        for key in (
            "boxes",
            "classes",
            "box_mask",
            "scene",
            "actions",
            "action_type",
            "action_mask",
        )
    }


def _objective(
    model: torch.nn.Module,
    method: str,
    data: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    scores = model(**_model_inputs(data, indices))
    config = PROTOCOL["model"]
    return action_value_objective(
        scores,
        data["target"][indices],
        data["action_mask"][indices],
        data["action_type"][indices],
        cross_image_weight=(
            float(config["cross_image_weight"]) if method in GLOBAL_METHODS else 0.0
        ),
        reward_weight=float(config["reward_loss_weight"]),
        within_weight=float(config["within_action_weight"]),
        minimum_target_gap=float(config["minimum_pair_target_gap"]),
    )


def _run_epoch(
    model: torch.nn.Module,
    method: str,
    data: Mapping[str, torch.Tensor],
    indices: Sequence[int],
    *,
    optimizer: torch.optim.Optimizer | None,
    seed: int,
) -> dict[str, float]:
    model.train(optimizer is not None)
    batch_size = int(PROTOCOL["model"]["batch_size"])
    ordered = torch.as_tensor(indices, dtype=torch.long)
    if optimizer is not None:
        generator = torch.Generator().manual_seed(int(seed))
        ordered = ordered[torch.randperm(len(ordered), generator=generator)]
    totals: Counter[str] = Counter()
    samples = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            result = _objective(model, method, data, batch)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(PROTOCOL["model"]["gradient_clip"])
                )
                optimizer.step()
            count = len(batch)
            samples += count
            for key, value in result.items():
                totals[key] += float(value.detach()) * count
    return {key: value / max(1, samples) for key, value in totals.items()}


def _fit_model(
    method: str,
    data: Mapping[str, torch.Tensor],
    image_ids: Sequence[int],
    train_indices: Sequence[int],
    *,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    train_ids = [int(image_ids[index]) for index in train_indices]
    inner_train_ids, validation_ids = deterministic_validation_split(
        train_ids,
        seed=int(seed),
        fraction=float(PROTOCOL["model"]["validation_fraction"]),
    )
    by_id = {int(image_id): index for index, image_id in enumerate(image_ids)}
    inner_train = [by_id[image_id] for image_id in inner_train_ids]
    validation = [by_id[image_id] for image_id in validation_ids]
    normalized, reward_mean, reward_std = _normalized_targets(data, inner_train)
    torch.manual_seed(int(seed))
    model = build_model(
        method,
        category_count=80,
        scene_dim=SCENE_DIM,
        action_dim=ACTION_DIM,
        config=PROTOCOL["model"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(PROTOCOL["model"]["learning_rate"]),
        weight_decay=float(PROTOCOL["model"]["weight_decay"]),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, best_loss, stale = 0, math.inf, 0
    history = []
    for epoch in range(1, int(PROTOCOL["model"]["epochs"]) + 1):
        train = _run_epoch(
            model,
            method,
            normalized,
            inner_train,
            optimizer=optimizer,
            seed=int(seed) + epoch * 1_000_003,
        )
        validation_metrics = _run_epoch(
            model,
            method,
            normalized,
            validation,
            optimizer=None,
            seed=int(seed),
        )
        value = float(validation_metrics["loss"])
        history.append(
            {"epoch": epoch, "train_loss": train["loss"], "validation_loss": value}
        )
        if value < best_loss - 1e-6:
            best_epoch, best_loss, stale = epoch, value, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= int(PROTOCOL["model"]["patience"]):
            break
    model.load_state_dict(best_state)
    return model, {
        "selected_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "reward_mean": float(reward_mean),
        "reward_std": float(reward_std),
        "inner_train_images": len(inner_train),
        "validation_images": len(validation),
        "history": history,
    }


def _predict(
    model: torch.nn.Module,
    data: Mapping[str, torch.Tensor],
    indices: Sequence[int],
    *,
    reward_mean: float,
    reward_std: float,
) -> np.ndarray:
    model.eval()
    output = []
    batch_size = int(PROTOCOL["model"]["batch_size"])
    ordered = torch.as_tensor(indices, dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            scores = model(**_model_inputs(data, batch))
            output.append(scores.cpu().numpy())
    values = np.concatenate(output, axis=0).astype(np.float64)
    values = values * float(reward_std) + float(reward_mean)
    values[:, 0] = 0.0
    if not np.isfinite(values).all():
        raise RuntimeError("Model prediction contains non-finite values")
    return values


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


_TRAIN_STATE: dict[str, Any] = {}


def _train_oof_job(task: tuple[str, int, int]) -> dict[str, Any]:
    method, base_seed, fold_index = str(task[0]), int(task[1]), int(task[2])
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    job_root = RUNS / f"oof/jobs/{method}/seed_{base_seed}/fold_{fold_index}"
    completion_path = job_root / "completion.json"
    if completion_path.is_file():
        return read_json(job_root / "report.json")
    image_ids = _TRAIN_STATE["image_ids"]
    data = _TRAIN_STATE["data"]
    heldout_ids = set(map(int, _TRAIN_STATE["folds"][fold_index]))
    heldout = [
        index for index, image_id in enumerate(image_ids) if int(image_id) in heldout_ids
    ]
    heldout_indices = set(heldout)
    train = [index for index in range(len(image_ids)) if index not in heldout_indices]
    job_seed = base_seed + fold_index * 10_007
    started = time.time()
    model, fit = _fit_model(method, data, image_ids, train, seed=job_seed)
    scores = _predict(
        model,
        data,
        heldout,
        reward_mean=float(fit["reward_mean"]),
        reward_std=float(fit["reward_std"]),
    )
    score_path = job_root / "scores.npz"
    checkpoint_path = job_root / "checkpoint.pt"
    _atomic_npz(
        score_path,
        image_ids=np.asarray([image_ids[index] for index in heldout], dtype=np.int64),
        scores=scores,
    )
    _atomic_torch_save(
        checkpoint_path,
        {
            "method": method,
            "base_seed": base_seed,
            "fold_index": fold_index,
            "model_state": model.state_dict(),
            "fit": fit,
        },
    )
    report = {
        "method": method,
        "seed": base_seed,
        "fold_index": fold_index,
        "train_images": len(train),
        "heldout_images": len(heldout),
        "selected_epoch": int(fit["selected_epoch"]),
        "best_validation_loss": float(fit["best_validation_loss"]),
        "elapsed_seconds": time.time() - started,
        "score_sha256": file_sha256(score_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    atomic_json(job_root / "report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "score_sha256": report["score_sha256"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "report_sha256": file_sha256(job_root / "report.json"),
        },
    )
    return report


def run_oof() -> dict[str, Any]:
    completion_path = RUNS / "oof/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "oof/report.json")
    _verify_preflight()
    records, data = _role_tensors("fit")
    image_ids = [int(record["image_id"]) for record in records]
    folds = [
        list(map(int, fold)) for fold in read_json(RUNS / "preflight/folds.json")["folds"]
    ]
    _ROLE_CACHE.pop("fit", None)
    gc.collect()
    _TRAIN_STATE.clear()
    _TRAIN_STATE.update({"image_ids": image_ids, "data": data, "folds": folds})
    tasks = [
        (method, int(seed), fold_index)
        for method in METHODS
        for seed in PROTOCOL["seeds"]
        for fold_index in range(int(PROTOCOL["fold_count"]))
    ]
    workers = min(int(PROTOCOL["cpu"]["training_workers"]), len(tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        reports = list(pool.imap_unordered(_train_oof_job, tasks, chunksize=1))
    scores: dict[str, dict[int, np.ndarray]] = {
        method: {
            int(seed): np.full((len(image_ids), 3), np.nan, dtype=np.float64)
            for seed in PROTOCOL["seeds"]
        }
        for method in METHODS
    }
    id_to_index = {image_id: index for index, image_id in enumerate(image_ids)}
    for method, seed, fold_index in tasks:
        path = RUNS / f"oof/jobs/{method}/seed_{seed}/fold_{fold_index}/scores.npz"
        with np.load(path, allow_pickle=False) as artifact:
            heldout_ids = list(map(int, artifact["image_ids"]))
            values = np.asarray(artifact["scores"], dtype=np.float64)
        if values.shape != (len(heldout_ids), 3):
            raise RuntimeError("OOF job score shape changed")
        indices = [id_to_index[image_id] for image_id in heldout_ids]
        target = scores[method][int(seed)]
        if np.isfinite(target[indices]).any():
            raise RuntimeError("An OOF image was predicted more than once")
        target[indices] = values
    for method in METHODS:
        for seed in PROTOCOL["seeds"]:
            if not np.isfinite(scores[method][int(seed)]).all():
                raise RuntimeError(f"Incomplete OOF scores: {method}/{seed}")
    score_path = RUNS / "oof/scores.pkl"
    atomic_pickle(score_path, {"image_ids": image_ids, "scores": scores})
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "jobs": len(tasks),
        "methods": list(METHODS),
        "seeds": list(PROTOCOL["seeds"]),
        "folds": int(PROTOCOL["fold_count"]),
        "images": len(image_ids),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "selected_epoch_summary": {
            method: {
                str(seed): [
                    int(row["selected_epoch"])
                    for row in reports
                    if row["method"] == method and int(row["seed"]) == int(seed)
                ]
                for seed in PROTOCOL["seeds"]
            }
            for method in METHODS
        },
        "score_sha256": file_sha256(score_path),
        "reward_context_may_cross_oof_folds": False,
        "calibration_used": False,
    }
    atomic_json(RUNS / "oof/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "score_sha256": report["score_sha256"],
            "report_sha256": file_sha256(RUNS / "oof/report.json"),
            "calibration_used": False,
        },
    )
    _TRAIN_STATE.clear()
    gc.collect()
    return report


def _score_map(
    records: Sequence[Mapping[str, Any]], values: np.ndarray
) -> dict[int, dict[str, float]]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (len(records), 3):
        raise ValueError("Score matrix and No-ROI records differ")
    output: dict[int, dict[str, float]] = {}
    for row_index, record in enumerate(records):
        actions = {
            str(action["action_id"]): action_index
            for action_index, action in enumerate(record["actions"])
            if str(action["action_id"]) != "edge"
        }
        output[int(record["image_id"])] = {
            action: float(matrix[row_index, action_index])
            for action, action_index in actions.items()
        }
    return output


_EVAL_STATE: dict[str, Any] = {}


def _evaluate_task(task: Mapping[str, Any]) -> dict[str, Any]:
    data = _EVAL_STATE["data"]
    rewards = _EVAL_STATE["rewards"]
    selection = {int(key): str(value) for key, value in task["selection"].items()}
    predictions, _ = canonical.mixed_predictions(data, selection, 0.001)
    official = data["evaluator"].evaluate(predictions)
    request_bytes = sum(
        int(data["sample_by_id"][image_id]["payload_bytes"][action])
        for image_id, action in selection.items()
    )
    selected_rewards = [
        float(rewards[image_id][action]) for image_id, action in selection.items()
    ]
    counts = Counter(selection.values())
    budget = float(task["budget"])
    mode = str(task["mode"])
    capacity = float(task["capacity_bytes"])
    expected_calls = int(math.floor(len(data["samples"]) * budget + 0.5))
    capacity_ok = (
        len(selection) == expected_calls
        if mode == "fixed_call"
        else request_bytes <= capacity + float(PROTOCOL["gate"]["capacity_tolerance_bytes"])
    )
    utilization = (
        len(selection) / max(1, expected_calls)
        if mode == "fixed_call"
        else request_bytes / max(1.0, capacity)
    )
    row = {
        "role": str(task["role"]),
        "method": str(task["method"]),
        "seed": task["seed"],
        "mode": mode,
        "budget": budget,
        "official_ap50_95": float(official["map50_95"]),
        "official_ap50": float(official["map50"]),
        "official_ap75": float(official["map75"]),
        "kb_per_image": request_bytes / 1024.0 / len(data["samples"]),
        "request_bytes": int(request_bytes),
        "capacity_bytes": capacity,
        "capacity_ok": bool(capacity_ok),
        "capacity_utilization": float(utilization),
        "selected_images": len(selection),
        "selected_fraction": len(selection) / len(data["samples"]),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": 0,
        "harmful_action_rate": sum(value <= 0.0 for value in selected_rewards)
        / max(1, len(selected_rewards)),
        "selected_teacher_reward": float(sum(selected_rewards)),
        "teacher_reward_ceiling": float(task["teacher_value"]),
        "teacher_regret": float(task["teacher_value"] - sum(selected_rewards)),
        "predicted_objective": float(task["predicted_objective"]),
        "scheduler_lambda": float(task["scheduler_lambda"]),
        "teacher_selection_jaccard": float(task["teacher_selection_jaccard"]),
        "score_reward_spearman": float(task["score_reward_spearman"]),
    }
    return row


def _evaluation_inputs(
    role: str,
    score_artifact: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, dict[int, str]]]]:
    data = load_role(role)
    records = records_for_role(role)
    image_ids = [int(record["image_id"]) for record in records]
    if list(map(int, score_artifact["image_ids"])) != image_ids:
        raise RuntimeError(f"{role} score and record ordering changed")
    rewards = _load_reward_map(role)
    samples = data["samples"]
    original_total = sum(int(sample["payload_bytes"]["original"]) for sample in samples)
    tasks: list[dict[str, Any]] = []
    selections: dict[str, dict[str, dict[int, str]]] = {}
    for method in METHODS:
        method_scores = score_artifact["scores"][method]
        seed_scores = {
            str(seed): np.asarray(method_scores[int(seed)], dtype=np.float64)
            for seed in PROTOCOL["seeds"]
        }
        seed_scores["ensemble"] = np.mean(list(seed_scores.values()), axis=0)
        selections[method] = {}
        for seed_label, values in seed_scores.items():
            maps = _score_map(records, values)
            flat_score, flat_reward = [], []
            for image_id, actions in maps.items():
                for action, score in actions.items():
                    flat_score.append(float(score))
                    flat_reward.append(float(rewards[image_id][action]))
            rank_correlation = spearman(flat_score, flat_reward)
            for budget in EXPECTED_BUDGETS:
                capacity = float(budget) * original_total
                teachers = {
                    "fixed_call": schedule_fixed_call(samples, rewards, budget=budget),
                    "fixed_byte": schedule_fixed_byte(
                        samples, rewards, capacity_bytes=capacity
                    ),
                }
                for mode in EXPECTED_MODES:
                    schedule = (
                        schedule_fixed_call(samples, maps, budget=budget)
                        if mode == "fixed_call"
                        else schedule_fixed_byte(
                            samples, maps, capacity_bytes=capacity
                        )
                    )
                    key = f"{seed_label}|{mode}|{budget:g}"
                    selection = {
                        int(image_id): str(action)
                        for image_id, action in schedule["selection"].items()
                    }
                    selections[method][key] = selection
                    teacher_selection = {
                        (int(image_id), str(action))
                        for image_id, action in teachers[mode]["selection"].items()
                    }
                    tasks.append(
                        {
                            "role": role,
                            "method": method,
                            "seed": seed_label,
                            "mode": mode,
                            "budget": float(budget),
                            "selection": selection,
                            "capacity_bytes": capacity,
                            "teacher_value": float(teachers[mode]["total_value"]),
                            "predicted_objective": float(schedule["total_value"]),
                            "scheduler_lambda": float(schedule.get("lambda", 0.0)),
                            "teacher_selection_jaccard": jaccard(
                                set(selection.items()), teacher_selection
                            ),
                            "score_reward_spearman": rank_correlation,
                        }
                    )
    return records, tasks, selections


def evaluate_scores(
    role: str,
    score_artifact: Mapping[str, Any],
    *,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    completion_path = output_root / "completion.json"
    if completion_path.is_file():
        with (output_root / "results.csv").open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle)), read_json(output_root / "report.json")
    records, tasks, selections = _evaluation_inputs(role, score_artifact)
    data = load_role(role)
    _EVAL_STATE.clear()
    _EVAL_STATE.update({"data": data, "rewards": _load_reward_map(role)})
    workers = min(int(PROTOCOL["cpu"]["evaluation_workers"]), len(tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        rows = list(pool.imap_unordered(_evaluate_task, tasks, chunksize=1))
    validate_result_matrix(rows, PROTOCOL)
    rows.sort(
        key=lambda row: (
            METHODS.index(str(row["method"])),
            str(row["seed"]),
            EXPECTED_MODES.index(str(row["mode"])),
            float(row["budget"]),
        )
    )
    gate = evaluate_candidate_gate(rows, PROTOCOL)
    atomic_csv(output_root / "results.csv", rows)
    atomic_json(output_root / "selections.json", selections)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": role,
        "rows": len(rows),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "gate": gate,
        "images": len(records),
        "roi_count": 0,
        "calibration_used_for_training_or_selection": False,
    }
    atomic_json(output_root / "report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "results_sha256": file_sha256(output_root / "results.csv"),
            "selections_sha256": file_sha256(output_root / "selections.json"),
            "report_sha256": file_sha256(output_root / "report.json"),
        },
    )
    _ROLE_CACHE.pop(role, None)
    _EVAL_STATE.clear()
    gc.collect()
    return rows, report


def run_oof_evaluation() -> dict[str, Any]:
    completion = read_json(RUNS / "oof/completion.json")
    score_path = RUNS / "oof/scores.pkl"
    if completion.get("score_sha256") != file_sha256(score_path):
        raise RuntimeError("OOF score cache changed")
    _, report = evaluate_scores(
        "fit", load_pickle(score_path), output_root=RUNS / "oof_evaluation"
    )
    return report


def _train_full_job(task: tuple[str, int]) -> dict[str, Any]:
    method, seed = str(task[0]), int(task[1])
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    job_root = RUNS / f"calibration/jobs/{method}/seed_{seed}"
    completion_path = job_root / "completion.json"
    if completion_path.is_file():
        return read_json(job_root / "report.json")
    fit_data = _TRAIN_STATE["fit_data"]
    fit_ids = _TRAIN_STATE["fit_ids"]
    calibration_data = _TRAIN_STATE["calibration_data"]
    calibration_ids = _TRAIN_STATE["calibration_ids"]
    started = time.time()
    model, fit = _fit_model(
        method, fit_data, fit_ids, list(range(len(fit_ids))), seed=seed
    )
    scores = _predict(
        model,
        calibration_data,
        list(range(len(calibration_ids))),
        reward_mean=float(fit["reward_mean"]),
        reward_std=float(fit["reward_std"]),
    )
    score_path = job_root / "scores.npz"
    checkpoint_path = job_root / "checkpoint.pt"
    _atomic_npz(
        score_path,
        image_ids=np.asarray(calibration_ids, dtype=np.int64),
        scores=scores,
    )
    _atomic_torch_save(
        checkpoint_path,
        {
            "method": method,
            "seed": seed,
            "model_state": model.state_dict(),
            "fit": fit,
        },
    )
    report = {
        "method": method,
        "seed": seed,
        "fit_images": len(fit_ids),
        "calibration_images": len(calibration_ids),
        "selected_epoch": int(fit["selected_epoch"]),
        "elapsed_seconds": time.time() - started,
        "score_sha256": file_sha256(score_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    atomic_json(job_root / "report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "score_sha256": report["score_sha256"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "report_sha256": file_sha256(job_root / "report.json"),
        },
    )
    return report


def _frozen_calibration_references() -> list[dict[str, Any]]:
    path = SOURCES["noroi_reference"] / "runs/point/calibration_results.csv"
    if not path.is_file():
        return []
    allowed = {"DualHead", "SetValue", "SemanticRank", "MORIC-AP5095"}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "method": row["method"],
            "mode": row["mode"],
            "budget": float(row["nominal_budget"]),
            "official_ap50_95": float(row["official_map50_95"]),
            "kb_per_image": float(row["request_kb_per_image"]),
            "source": "frozen_noroi_reference",
        }
        for row in rows
        if row["method"] in allowed
        and row["payload_policy"] == "SharedAction-NoROI"
        and row["mode"] in EXPECTED_MODES
        and float(row["nominal_budget"]) in EXPECTED_BUDGETS
    ]


def run_calibration() -> dict[str, Any]:
    completion_path = RUNS / "calibration/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "calibration/report.json")
    oof_report = read_json(RUNS / "oof_evaluation/report.json")
    if not should_run_calibration(oof_report["gate"], PROTOCOL):
        report = {
            "protocol_id": PROTOCOL["protocol_id"],
            "role": "calibration",
            "status": "skipped_oof_gate_failed",
            "evaluated": False,
            "oof_gate": oof_report["gate"],
            "calibration_used_for_training_or_selection": False,
        }
        atomic_json(RUNS / "calibration/report.json", report)
        atomic_json(completion_path, {"complete": True, "evaluated": False})
        return report
    run_rewards("calibration")
    fit_records, fit_data = _role_tensors("fit")
    calibration_records, calibration_data = _role_tensors("calibration")
    fit_ids = [int(record["image_id"]) for record in fit_records]
    calibration_ids = [int(record["image_id"]) for record in calibration_records]
    _ROLE_CACHE.clear()
    gc.collect()
    _TRAIN_STATE.clear()
    _TRAIN_STATE.update(
        {
            "fit_data": fit_data,
            "fit_ids": fit_ids,
            "calibration_data": calibration_data,
            "calibration_ids": calibration_ids,
        }
    )
    tasks = [(method, int(seed)) for method in METHODS for seed in PROTOCOL["seeds"]]
    workers = min(int(PROTOCOL["cpu"]["training_workers"]), len(tasks))
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        list(pool.imap_unordered(_train_full_job, tasks, chunksize=1))
    scores: dict[str, dict[int, np.ndarray]] = {method: {} for method in METHODS}
    for method, seed in tasks:
        path = RUNS / f"calibration/jobs/{method}/seed_{seed}/scores.npz"
        with np.load(path, allow_pickle=False) as artifact:
            if list(map(int, artifact["image_ids"])) != calibration_ids:
                raise RuntimeError("Calibration score ordering changed")
            scores[method][seed] = np.asarray(artifact["scores"], dtype=np.float64)
    score_path = RUNS / "calibration/scores.pkl"
    atomic_pickle(score_path, {"image_ids": calibration_ids, "scores": scores})
    _, evaluation = evaluate_scores(
        "calibration",
        {"image_ids": calibration_ids, "scores": scores},
        output_root=RUNS / "calibration/evaluation",
    )
    references = _frozen_calibration_references()
    atomic_csv(RUNS / "calibration/frozen_references.csv", references)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": "calibration",
        "status": "evaluated_once",
        "evaluated": True,
        "gate": evaluation["gate"],
        "score_sha256": file_sha256(score_path),
        "frozen_reference_rows": len(references),
        "calibration_used_for_training_or_selection": False,
    }
    atomic_json(RUNS / "calibration/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "evaluated": True,
            "score_sha256": report["score_sha256"],
            "report_sha256": file_sha256(RUNS / "calibration/report.json"),
        },
    )
    _TRAIN_STATE.clear()
    gc.collect()
    return report


def _read_result_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    numeric = {
        "budget",
        "official_ap50_95",
        "official_ap50",
        "official_ap75",
        "kb_per_image",
        "capacity_utilization",
        "selected_fraction",
        "score_reward_spearman",
        "teacher_selection_jaccard",
        "teacher_regret",
    }
    integer = {"selected_images", "original_count", "q90_count", "roi_count"}
    output = []
    for row in source:
        converted = dict(row)
        for key in numeric:
            if key in converted:
                converted[key] = float(converted[key])
        for key in integer:
            if key in converted:
                converted[key] = int(converted[key])
        output.append(converted)
    return output


def _ensemble_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row["seed"]) == "ensemble"]


def _method_comparison(
    rows: Sequence[Mapping[str, Any]], method: str
) -> dict[str, Any]:
    values = []
    for mode in EXPECTED_MODES:
        for budget in EXPECTED_BUDGETS:
            baseline = next(
                row
                for row in rows
                if row["method"] == "CurrentActionValue"
                and row["mode"] == mode
                and math.isclose(float(row["budget"]), budget)
            )
            candidate = next(
                row
                for row in rows
                if row["method"] == method
                and row["mode"] == mode
                and math.isclose(float(row["budget"]), budget)
            )
            values.append(
                {
                    "mode": mode,
                    "budget": budget,
                    "delta_ap50_95": float(candidate["official_ap50_95"])
                    - float(baseline["official_ap50_95"]),
                    "delta_spearman": float(candidate["score_reward_spearman"])
                    - float(baseline["score_reward_spearman"]),
                    "delta_teacher_jaccard": float(
                        candidate["teacher_selection_jaccard"]
                    )
                    - float(baseline["teacher_selection_jaccard"]),
                }
            )
    return {"method": method, "versus": "CurrentActionValue", "work_points": values}


def _hypothesis_summary(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def primary_mean(method: str) -> float:
        values = [
            float(row["delta_ap50_95"])
            for row in comparisons[method]["work_points"]
            if row["mode"] == "fixed_byte"
        ]
        return float(np.mean(values))

    objective = primary_mean("GlobalRank")
    representation = primary_mean("RelSetValue")
    complete = primary_mean("RelActionRank")
    strongest = max(
        (
            ("training_objective", objective),
            ("relational_representation", representation),
            ("combined", complete),
        ),
        key=lambda item: item[1],
    )
    return {
        "fixed_byte_mean_delta_ap": {
            "GlobalRank": objective,
            "RelSetValue": representation,
            "RelActionRank": complete,
        },
        "strongest_observed_factor": strongest[0],
        "strongest_mean_delta_ap": float(strongest[1]),
        "interpretation_is_diagnostic_not_final": True,
    }


def _markdown_table(rows: Sequence[Mapping[str, Any]], mode: str) -> list[str]:
    lines = [
        f"### {mode}",
        "",
        "| 方法 | 10% AP / KB | 25% AP / KB | 40% AP / KB |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        cells = []
        for budget in EXPECTED_BUDGETS:
            row = next(
                value
                for value in rows
                if value["method"] == method
                and value["mode"] == mode
                and math.isclose(float(value["budget"]), budget)
            )
            cells.append(
                f'{float(row["official_ap50_95"]):.6f} / '
                f'{float(row["kb_per_image"]):.2f}'
            )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return lines


def run_report() -> dict[str, Any]:
    completion_path = RUNS / "COMPLETED.json"
    if completion_path.is_file():
        return read_json(RUNS / "report.json")
    oof_report = read_json(RUNS / "oof_evaluation/report.json")
    calibration_report = read_json(RUNS / "calibration/report.json")
    rows = _ensemble_rows(_read_result_rows(RUNS / "oof_evaluation/results.csv"))
    comparisons = {
        method: _method_comparison(rows, method) for method in METHODS[1:]
    }
    hypotheses = _hypothesis_summary(comparisons)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "result_type": PROTOCOL["result_type"],
        "dataset": "coco",
        "actions": list(EXPECTED_ACTIONS),
        "fit_images": int(PROTOCOL["roles"]["fit"]),
        "calibration_images": int(PROTOCOL["roles"]["calibration"]),
        "oof_gate": oof_report["gate"],
        "calibration": calibration_report,
        "comparisons": comparisons,
        "hypothesis_summary": hypotheses,
        "open_future_paper_protocol": bool(
            oof_report["gate"]["passed"]
            and calibration_report.get("evaluated")
            and calibration_report.get("gate", {}).get("passed")
        ),
        "detector_inference_used": False,
        "roi_used": False,
        "audit_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    atomic_json(RUNS / "report.json", report)
    lines = [
        "# COCO Relational Action-Value OOF 诊断结果",
        "",
        "本实验只使用 canonical train2017 fit=9,678 与冻结 calibration=571；"
        "动作空间为 Edge/Original/Q90，严格不含 ROI。Fixed-byte 是主指标，"
        "val2017、audit 与 registered final 均未访问。",
        "",
        f'OOF 冻结 gate：**{str(bool(oof_report["gate"]["passed"])).lower()}**。',
        "",
        *_markdown_table(rows, "fixed_byte"),
        "",
        *_markdown_table(rows, "fixed_call"),
        "",
        "## 相对 CurrentActionValue 的 Fixed-byte AP 变化",
        "",
        "| 方法 | 10% | 25% | 40% | 三点均值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS[1:]:
        points = [
            row
            for row in comparisons[method]["work_points"]
            if row["mode"] == "fixed_byte"
        ]
        deltas = [float(row["delta_ap50_95"]) for row in points]
        lines.append(
            f'| {method} | {deltas[0]:+.6f} | {deltas[1]:+.6f} | '
            f'{deltas[2]:+.6f} | {np.mean(deltas):+.6f} |'
        )
    if calibration_report.get("evaluated"):
        calibration_text = (
            "OOF gate 通过后，calibration 已按预注册流程一次性评测；"
            f'calibration gate={str(bool(calibration_report["gate"]["passed"])).lower()}。'
        )
    else:
        calibration_text = "OOF gate 未通过，因此 calibration 自动封存且未评测。"
    lines.extend(
        [
            "",
            "## 协议结论",
            "",
            calibration_text,
            "本轮只回答关系集合表征与跨图片动作排序是否值得继续，"
            "不是新的论文 final；是否开启新论文级协议以 report.json 的"
            " open_future_paper_protocol 为准。",
            "",
        ]
    )
    notes_path = ROOT / "notes/results_zh.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = notes_path.with_name(f".{notes_path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, notes_path)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "report_sha256": file_sha256(RUNS / "report.json"),
            "notes_sha256": file_sha256(notes_path),
            "audit_accessed": False,
            "final_accessed": False,
            "registered_final_accessed": False,
        },
    )
    return report


STAGES = {
    "preflight": run_preflight,
    "fit_rewards": lambda: run_rewards("fit"),
    "oof": run_oof,
    "oof_evaluation": run_oof_evaluation,
    "calibration": run_calibration,
    "report": run_report,
}


def run_all() -> dict[str, Any]:
    outputs = {}
    for name in (
        "preflight",
        "fit_rewards",
        "oof",
        "oof_evaluation",
        "calibration",
        "report",
    ):
        print(f"[stage] {name}", flush=True)
        outputs[name] = STAGES[name]()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    output = run_all() if args.stage == "all" else STAGES[args.stage]()
    print(json.dumps(output, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
