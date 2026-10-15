"""Resumable COCO Action-Frontier Path Distillation exploration."""

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


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL_PATH = ROOT / "configs/protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load("frontier_runner_core", ROOT / "scripts/core.py")
TRAINING = _load("frontier_runner_training", ROOT / "scripts/training.py")
REL_ROOT = REPO / PROTOCOL["source_experiments"]["fold_contained_oq"]
REL = _load(
    "frontier_relational_source_runner", REL_ROOT / "scripts/run_experiment.py"
)
ACTION_ROOT = REPO / "paper/experiments/20260805_coco_actionconditioned_valuerisk_oof_v1"
ACTION = _load("frontier_action_source", ACTION_ROOT / "scripts/action_core.py")
COMMON = (
    REPO
    / PROTOCOL["source_experiments"]["canonical"]
    / "runs/canonical_v1/common/coco"
)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
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


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _samples(role: str) -> list[dict[str, Any]]:
    values = read_json(COMMON / role / "samples.json")
    if not isinstance(values, list):
        raise ValueError(f"Unexpected sample document for {role}")
    return [dict(value) for value in values]


def run_preflight() -> dict[str, Any]:
    output = RUNS / "preflight/report.json"
    completion = RUNS / "preflight/completion.json"
    if completion.is_file():
        return read_json(output)
    CORE.validate_protocol(PROTOCOL)
    source_completions = [
        REL_ROOT / "runs/COMPLETED.json",
        REL_ROOT / "runs/oof/completion.json",
        REL_ROOT / "runs/rewards/fit/completion.json",
    ]
    missing = [str(path) for path in source_completions if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete source experiment: {missing}")
    role_ids: dict[str, list[int]] = {}
    for role, expected in PROTOCOL["roles"].items():
        values = _samples(role)
        ids = [int(value["image_id"]) for value in values]
        if len(ids) != int(expected) or len(ids) != len(set(ids)):
            raise RuntimeError(f"Canonical {role} population changed")
        manifest = read_json(COMMON / role / "source_manifest.json")
        if not {"edge", "original", "q90", "roi"} <= set(
            manifest["action_sources"]
        ):
            raise RuntimeError(f"Canonical {role} action set is incomplete")
        role_ids[role] = ids
    if set(role_ids["fit"]) & set(role_ids["calibration"]):
        raise RuntimeError("Fit and calibration overlap")
    folds = read_json(REL_ROOT / "runs/preflight/folds.json")["folds"]
    flat = [int(image_id) for fold in folds for image_id in fold]
    if len(folds) != 5 or set(flat) != set(role_ids["fit"]):
        raise RuntimeError("Frozen OOF folds changed")
    hashes = {
        str(path): file_sha256(path)
        for path in (
            PROTOCOL_PATH,
            ROOT / "scripts/core.py",
            ROOT / "scripts/training.py",
            *source_completions,
            REL_ROOT / "runs/oof/scores.pkl",
            REL_ROOT / "runs/rewards/fit/reward_map.pkl",
        )
    }
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "passed": True,
        "role_counts": {key: len(value) for key, value in role_ids.items()},
        "fit_calibration_overlap": 0,
        "fold_sizes": list(map(len, folds)),
        "source_hashes": hashes,
        "source_digest": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "detector_inference_required": False,
        "audit_accessed": False,
        "val2017_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    atomic_json(output, report)
    atomic_json(completion, {"complete": True, "source_digest": report["source_digest"]})
    return report


def _oq_reward_map(role: str) -> dict[int, dict[str, float]]:
    if role == "fit":
        path = REL_ROOT / "runs/rewards/fit/reward_map.pkl"
    else:
        path = REL_ROOT / f"runs/rewards/{role}/reward_map.pkl"
    with path.open("rb") as handle:
        values = pickle.load(handle)
    return {
        int(image_id): {str(action): float(reward) for action, reward in rewards.items()}
        for image_id, rewards in values.items()
    }


def run_diagnose() -> dict[str, Any]:
    completion = RUNS / "diagnose/completion.json"
    output = RUNS / "diagnose/report.json"
    if completion.is_file():
        return read_json(output)
    run_preflight()
    samples = _samples("fit")
    rewards = _oq_reward_map("fit")
    with (REL_ROOT / "runs/oof/scores.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    image_ids = list(map(int, artifact["image_ids"]))
    matrices = artifact["scores"]["CurrentActionValue"]
    scores = np.mean([np.asarray(value) for value in matrices.values()], axis=0)
    score_map = {
        image_id: {"original": float(scores[row, 1]), "q90": float(scores[row, 2])}
        for row, image_id in enumerate(image_ids)
    }
    original_total = sum(int(sample["payload_bytes"]["original"]) for sample in samples)
    rows = []
    for budget in PROTOCOL["budgets"]:
        capacity = float(budget) * original_total
        teacher = ACTION.schedule_fixed_byte(
            samples, rewards, capacity_bytes=capacity
        )
        learned = ACTION.schedule_fixed_byte(
            samples, score_map, capacity_bytes=capacity
        )
        teacher_selection = {
            int(key): str(value) for key, value in teacher["selection"].items()
        }
        learned_selection = {
            int(key): str(value) for key, value in learned["selection"].items()
        }
        teacher_value = sum(
            rewards[image_id][action]
            for image_id, action in teacher_selection.items()
        )
        learned_value = sum(
            rewards[image_id][action]
            for image_id, action in learned_selection.items()
        )
        common = set(teacher_selection) & set(learned_selection)
        union = set(teacher_selection) | set(learned_selection)
        exact = sum(
            teacher_selection[image_id] == learned_selection[image_id]
            for image_id in common
        )
        teacher_lambda = float(teacher["lambda"])
        margins, errors = [], []
        error_pairs: Counter[tuple[str, str]] = Counter()
        for sample in samples:
            image_id = int(sample["image_id"])
            options = [("edge", 0.0, 0.0)]
            for action in ("original", "q90"):
                if sample["available"].get(action, False):
                    options.append(
                        (
                            action,
                            float(rewards[image_id][action]),
                            float(sample["payload_bytes"][action]),
                        )
                    )
            truth = np.asarray(
                [reward - teacher_lambda * cost for _, reward, cost in options]
            )
            prediction = np.asarray(
                [
                    (0.0 if action == "edge" else score_map[image_id][action])
                    - teacher_lambda * cost
                    for action, _, cost in options
                ]
            )
            truth_index, prediction_index = int(truth.argmax()), int(prediction.argmax())
            ordered = np.argsort(truth)[::-1]
            margins.append(float(truth[ordered[0]] - truth[ordered[1]]))
            errors.append(truth_index != prediction_index)
            if truth_index != prediction_index:
                error_pairs[(options[truth_index][0], options[prediction_index][0])] += 1
        margin_array = np.asarray(margins)
        error_array = np.asarray(errors, dtype=bool)
        rows.append(
            {
                "budget": float(budget),
                "teacher_lambda": teacher_lambda,
                "teacher_reward": float(teacher_value),
                "learned_reward": float(learned_value),
                "reward_recovery": float(learned_value / max(1e-12, teacher_value)),
                "teacher_regret": float(teacher_value - learned_value),
                "membership_jaccard": len(common) / max(1, len(union)),
                "common_action_accuracy": exact / max(1, len(common)),
                "local_frontier_error_rate": float(error_array.mean()),
                "lowest_margin_quartile_error_rate": float(
                    error_array[margin_array <= np.quantile(margin_array, 0.25)].mean()
                ),
                "largest_error_pairs": [
                    {"teacher": pair[0], "learned": pair[1], "count": count}
                    for pair, count in error_pairs.most_common(6)
                ],
            }
        )
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": "fit_oof",
        "rows": rows,
        "observed_fact": "Current ActionValue has substantial byte-boundary regret despite higher action agreement on common selected images.",
        "mechanistic_hypothesis": "Training scalar rewards does not directly preserve the parametric action upper envelope used by the byte scheduler.",
        "final_accessed": False,
    }
    atomic_json(output, report)
    atomic_json(completion, {"complete": True})
    return report


_ROI_STATE: dict[str, Any] = {}


def _init_roi_worker(annotation_path: str) -> None:
    _ROI_STATE["metric"] = REL.CocoMetricBackend(Path(annotation_path))


def _roi_reward_worker(target_id: int) -> dict[str, Any]:
    target_id = int(target_id)
    data = _ROI_STATE["data"]
    context_ids = _ROI_STATE["contexts"][target_id]
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
        "cloud_predictions": data["actions"]["roi"].get(target_id, []),
    }
    row = REL.dual_metric_context_reward(
        _ROI_STATE["metric"], context_records, target
    )
    return {
        "target_image_id": target_id,
        "action_id": "roi",
        "reward": float(row["reward_ap50_95"]),
        "context_image_ids": list(map(int, context_ids)),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_roi_rewards(role: str = "fit") -> dict[str, Any]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(f"Forbidden reward role: {role}")
    root = RUNS / f"rewards/{role}"
    completion = root / "completion.json"
    output = root / "report.json"
    if completion.is_file():
        return read_json(output)
    run_preflight()
    data = REL.load_role(role)
    image_ids = [int(sample["image_id"]) for sample in data["samples"]]
    _, contexts = REL._reward_contexts(role, image_ids)
    checkpoint = root / "roi_rewards.jsonl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_jsonl(checkpoint)
    completed = {int(row["target_image_id"]) for row in existing}
    if len(completed) != len(existing) or not completed <= set(image_ids):
        raise RuntimeError("ROI reward checkpoint is invalid")
    pending = sorted(set(image_ids) - completed)
    _ROI_STATE.clear()
    _ROI_STATE.update({"data": data, "contexts": contexts})
    workers = min(int(PROTOCOL["reward"]["roi_workers"]), max(1, os.cpu_count() or 1))
    started = time.time()
    with checkpoint.open("a", encoding="utf-8") as handle:
        context = mp.get_context("fork")
        with context.Pool(
            processes=workers,
            initializer=_init_roi_worker,
            initargs=(str(COMMON / role / "gt.json"),),
        ) as pool:
            for count, row in enumerate(
                pool.imap_unordered(_roi_reward_worker, pending, chunksize=1),
                start=len(completed) + 1,
            ):
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                if count % 100 == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                    print(f"[roi-reward:{role}] {count}/{len(image_ids)}", flush=True)
        handle.flush()
        os.fsync(handle.fileno())
    rows = _read_jsonl(checkpoint)
    observed = {int(row["target_image_id"]) for row in rows}
    if observed != set(image_ids) or len(rows) != len(image_ids):
        raise RuntimeError("ROI reward matrix is incomplete")
    reward_map = {
        int(row["target_image_id"]): {"roi": float(row["reward"])} for row in rows
    }
    reward_path = root / "roi_reward_map.pkl"
    atomic_pickle(reward_path, reward_map)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": role,
        "images": len(image_ids),
        "rows": len(rows),
        "context_size": int(PROTOCOL["reward"]["context_size"]),
        "fold_contained": True,
        "reward_mean": float(np.mean([row["reward"] for row in rows])),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "fit_used_for_training": role == "fit",
        "calibration_used_for_training_or_selection": False,
        "final_accessed": False,
    }
    atomic_json(output, report)
    atomic_json(
        completion,
        {"complete": True, "reward_sha256": file_sha256(reward_path)},
    )
    _ROI_STATE.clear()
    return report


def _records_for_space(role: str, action_space: str) -> list[dict[str, Any]]:
    if action_space not in PROTOCOL["action_spaces"]:
        raise ValueError(f"Unknown action space: {action_space}")
    data = REL.load_role(role)
    records = ACTION.build_action_records(
        data["samples"], data["actions"]["edge"], category_ids=REL.category_ids(role)
    )
    filtered = CORE.filter_action_space_records(
        records, PROTOCOL["action_spaces"][action_space]
    )
    if [int(row["image_id"]) for row in filtered] != [
        int(sample["image_id"]) for sample in data["samples"]
    ]:
        raise RuntimeError(f"{role}/{action_space} record order changed")
    return filtered


def _reward_map_for_space(role: str, action_space: str) -> dict[int, dict[str, float]]:
    rewards = _oq_reward_map(role)
    if action_space == "OQR":
        roi_path = RUNS / f"rewards/{role}/roi_reward_map.pkl"
        if not roi_path.is_file():
            raise FileNotFoundError(f"ROI rewards are incomplete: {roi_path}")
        with roi_path.open("rb") as handle:
            roi_rewards = pickle.load(handle)
        for image_id, values in roi_rewards.items():
            rewards[int(image_id)].update(
                {str(action): float(value) for action, value in values.items()}
            )
    expected = {int(sample["image_id"]) for sample in _samples(role)}
    if set(rewards) != expected:
        raise RuntimeError(f"{role}/{action_space} reward population changed")
    return rewards


def _screen_tensor_data(
    role: str, action_space: str
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    records = _records_for_space(role, action_space)
    cost_scale_bytes = float(
        np.mean([float(record["full_payload_bytes"]) for record in records])
    )
    raw_reward, cost_ratio = TRAINING.reward_and_cost_tensors(
        records,
        _reward_map_for_space(role, action_space),
        max_actions=int(PROTOCOL["model"]["max_actions"]),
        cost_scale_bytes=cost_scale_bytes,
    )
    tensor_protocol = {
        "max_boxes": int(PROTOCOL["model"]["max_boxes"]),
        "max_actions": int(PROTOCOL["model"]["max_actions"]),
        "deploy_confidence": float(PROTOCOL["model"]["deploy_confidence"]),
    }
    data = TRAINING.strip_cost_inputs(
        REL._record_tensors(
            records, raw_reward.numpy().astype(np.float32), tensor_protocol
        )
    )
    data["raw_target"] = raw_reward.to(torch.float32)
    data["cost_ratio"] = cost_ratio.to(torch.float32)
    data["cost_scale_bytes"] = torch.full(
        (len(records),), cost_scale_bytes, dtype=torch.float32
    )
    if data["target"].shape != raw_reward.shape:
        raise RuntimeError(f"{role}/{action_space} tensor shape changed")
    return records, data


def _index_tensors(
    data: Mapping[str, torch.Tensor], indices: Sequence[int] | torch.Tensor
) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[index] for key, value in data.items()}


def _normalized_screen_data(
    data: Mapping[str, torch.Tensor], train_indices: Sequence[int]
) -> tuple[dict[str, torch.Tensor], float, float]:
    index = torch.as_tensor(train_indices, dtype=torch.long)
    cloud = data["action_mask"][index] & (data["action_type"][index] != 0)
    values = data["target"][index][cloud]
    reward_mean = float(values.mean())
    reward_std = max(1e-6, float(values.std(unbiased=False)))
    output = dict(data)
    output["target"] = (data["target"] - reward_mean) / reward_std
    return output, reward_mean, reward_std


def _screen_epoch(
    model: torch.nn.Module,
    method: str,
    data: Mapping[str, torch.Tensor],
    indices: Sequence[int],
    *,
    reward_mean: float,
    reward_std: float,
    optimizer: torch.optim.Optimizer | None,
    seed: int,
) -> dict[str, float]:
    model.train(optimizer is not None)
    ordered = torch.as_tensor(indices, dtype=torch.long)
    if optimizer is not None:
        generator = torch.Generator().manual_seed(int(seed))
        ordered = ordered[torch.randperm(len(ordered), generator=generator)]
    totals: Counter[str] = Counter()
    count_total = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    batch_size = int(PROTOCOL["model"]["batch_size"])
    with context:
        for start in range(0, len(ordered), batch_size):
            batch_indices = ordered[start : start + batch_size]
            batch = _index_tensors(data, batch_indices)
            result = TRAINING.training_objective(
                model,
                method,
                batch,
                reward_mean=reward_mean,
                reward_std=reward_std,
                config=PROTOCOL["model"],
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(PROTOCOL["model"]["gradient_clip"])
                )
                optimizer.step()
            count = len(batch_indices)
            count_total += count
            for key, value in result.items():
                totals[key] += float(value.detach()) * count
    return {key: value / max(1, count_total) for key, value in totals.items()}


def _fit_screen_model(
    method: str,
    data: Mapping[str, torch.Tensor],
    image_ids: Sequence[int],
    train_indices: Sequence[int],
    *,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    train_ids = [int(image_ids[index]) for index in train_indices]
    inner_ids, validation_ids = REL.deterministic_validation_split(
        train_ids,
        seed=int(seed),
        fraction=float(PROTOCOL["model"]["validation_fraction"]),
    )
    by_id = {int(image_id): index for index, image_id in enumerate(image_ids)}
    inner_train = [by_id[image_id] for image_id in inner_ids]
    validation = [by_id[image_id] for image_id in validation_ids]
    normalized, reward_mean, reward_std = _normalized_screen_data(
        data, inner_train
    )
    torch.manual_seed(int(seed))
    model = TRAINING.build_model(
        category_count=len(REL.category_ids("fit")),
        scene_dim=int(REL.SCENE_DIM),
        action_dim=int(REL.ACTION_DIM),
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
        train_metrics = _screen_epoch(
            model,
            method,
            normalized,
            inner_train,
            reward_mean=reward_mean,
            reward_std=reward_std,
            optimizer=optimizer,
            seed=int(seed) + epoch * 1_000_003,
        )
        validation_metrics = _screen_epoch(
            model,
            method,
            normalized,
            validation,
            reward_mean=reward_mean,
            reward_std=reward_std,
            optimizer=None,
            seed=int(seed),
        )
        validation_loss = float(validation_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "validation_loss": validation_loss,
                "validation_frontier_loss": float(
                    validation_metrics["frontier_loss"]
                ),
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_epoch, best_loss, stale = epoch, validation_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= int(PROTOCOL["model"]["patience"]):
            break
    model.load_state_dict(best_state)
    return model, {
        "selected_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "inner_train_images": len(inner_train),
        "validation_images": len(validation),
        "history": history,
    }


def _predict_screen_model(
    model: torch.nn.Module,
    data: Mapping[str, torch.Tensor],
    indices: Sequence[int],
    *,
    reward_mean: float,
    reward_std: float,
) -> np.ndarray:
    model.eval()
    outputs = []
    ordered = torch.as_tensor(indices, dtype=torch.long)
    batch_size = int(PROTOCOL["model"]["batch_size"])
    with torch.no_grad():
        for start in range(0, len(ordered), batch_size):
            batch = _index_tensors(data, ordered[start : start + batch_size])
            scores = model(**TRAINING.model_inputs(batch))
            outputs.append(scores.cpu().numpy())
    values = np.concatenate(outputs, axis=0).astype(np.float64)
    values = values * float(reward_std) + float(reward_mean)
    action_types = data["action_type"][ordered].numpy()
    action_mask = data["action_mask"][ordered].numpy()
    values[action_types == 0] = 0.0
    values[~action_mask] = 0.0
    if not np.isfinite(values).all():
        raise RuntimeError("Screen prediction contains non-finite values")
    return values


_SCREEN_STATE: dict[str, Any] = {}


def _train_screen_job(task: tuple[str, str, int, int]) -> dict[str, Any]:
    action_space, method, base_seed, fold_index = (
        str(task[0]),
        str(task[1]),
        int(task[2]),
        int(task[3]),
    )
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    root = RUNS / (
        f"screen/jobs/{action_space}/{method}/seed_{base_seed}/fold_{fold_index}"
    )
    completion = root / "completion.json"
    if completion.is_file():
        return read_json(root / "report.json")
    data = _SCREEN_STATE["data"][action_space]
    image_ids = _SCREEN_STATE["image_ids"][action_space]
    heldout_ids = set(map(int, _SCREEN_STATE["folds"][fold_index]))
    heldout = [
        index for index, image_id in enumerate(image_ids) if int(image_id) in heldout_ids
    ]
    heldout_indices = set(heldout)
    train = [index for index in range(len(image_ids)) if index not in heldout_indices]
    if len(heldout) != len(heldout_ids):
        raise RuntimeError(f"Fold {fold_index} population changed")
    job_seed = int(base_seed) + fold_index * 10_007
    started = time.time()
    model, fit = _fit_screen_model(
        method, data, image_ids, train, seed=job_seed
    )
    scores = _predict_screen_model(
        model,
        data,
        heldout,
        reward_mean=float(fit["reward_mean"]),
        reward_std=float(fit["reward_std"]),
    )
    score_path = root / "scores.npz"
    checkpoint_path = root / "checkpoint.pt"
    atomic_npz(
        score_path,
        image_ids=np.asarray([image_ids[index] for index in heldout], dtype=np.int64),
        scores=scores,
    )
    atomic_torch(
        checkpoint_path,
        {
            "action_space": action_space,
            "method": method,
            "seed": job_seed,
            "model_state": model.state_dict(),
            "fit": fit,
        },
    )
    report = {
        "action_space": action_space,
        "method": method,
        "base_seed": base_seed,
        "job_seed": job_seed,
        "fold": fold_index,
        "train_images": len(train),
        "heldout_images": len(heldout),
        "selected_epoch": int(fit["selected_epoch"]),
        "best_validation_loss": float(fit["best_validation_loss"]),
        "elapsed_seconds": time.time() - started,
        "score_sha256": file_sha256(score_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    atomic_json(root / "report.json", report)
    atomic_json(
        completion,
        {
            "complete": True,
            "score_sha256": report["score_sha256"],
            "checkpoint_sha256": report["checkpoint_sha256"],
        },
    )
    return report


def run_screen_training() -> dict[str, Any]:
    completion = RUNS / "screen/training_completion.json"
    report_path = RUNS / "screen/training_report.json"
    if completion.is_file():
        return read_json(report_path)
    run_preflight()
    run_roi_rewards("fit")
    spaces = tuple(PROTOCOL["action_spaces"])
    records_by_space: dict[str, list[dict[str, Any]]] = {}
    data_by_space: dict[str, dict[str, torch.Tensor]] = {}
    image_ids_by_space: dict[str, list[int]] = {}
    for action_space in spaces:
        records, data = _screen_tensor_data("fit", action_space)
        records_by_space[action_space] = records
        data_by_space[action_space] = data
        image_ids_by_space[action_space] = [
            int(record["image_id"]) for record in records
        ]
    folds = read_json(REL_ROOT / "runs/preflight/folds.json")["folds"]
    _SCREEN_STATE.clear()
    _SCREEN_STATE.update(
        {
            "data": data_by_space,
            "image_ids": image_ids_by_space,
            "folds": folds,
        }
    )
    seed = int(PROTOCOL["screen_seed"])
    tasks = [
        (action_space, method, seed, fold_index)
        for action_space in spaces
        for method in PROTOCOL["screen_methods"]
        for fold_index in range(int(PROTOCOL["fold_count"]))
    ]
    workers = min(int(PROTOCOL["cpu"]["screen_workers"]), len(tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        job_reports = list(
            pool.imap_unordered(_train_screen_job, tasks, chunksize=1)
        )
    scores: dict[str, dict[str, np.ndarray]] = {}
    for action_space in spaces:
        ids = image_ids_by_space[action_space]
        position = {image_id: index for index, image_id in enumerate(ids)}
        scores[action_space] = {}
        for method in PROTOCOL["screen_methods"]:
            matrix = np.full(
                (len(ids), int(PROTOCOL["model"]["max_actions"])),
                np.nan,
                dtype=np.float64,
            )
            for fold_index in range(int(PROTOCOL["fold_count"])):
                path = RUNS / (
                    f"screen/jobs/{action_space}/{method}/seed_{seed}/"
                    f"fold_{fold_index}/scores.npz"
                )
                with np.load(path, allow_pickle=False) as artifact:
                    for row, image_id in enumerate(map(int, artifact["image_ids"])):
                        matrix[position[image_id]] = artifact["scores"][row]
            if not np.isfinite(matrix).all():
                raise RuntimeError(f"Incomplete OOF matrix: {action_space}/{method}")
            scores[action_space][method] = matrix
    score_path = RUNS / "screen/scores.pkl"
    atomic_pickle(
        score_path,
        {
            "image_ids": image_ids_by_space,
            "scores": scores,
            "seed": seed,
            "method_semantics": {
                "CurrentActionValue": "scalar contextual reward regression",
                "PathDistill": "complete reward-minus-shadow-price path distillation",
                "BoundaryPathDistill": "path distillation with action-crossing emphasis",
            },
        },
    )
    source_hashes = {
        str(path): file_sha256(path)
        for path in (
            SCRIPT,
            ROOT / "scripts/core.py",
            ROOT / "scripts/training.py",
            RUNS / "preflight/report.json",
            RUNS / "rewards/fit/roi_reward_map.pkl",
            REL_ROOT / "runs/rewards/fit/reward_map.pkl",
        )
    }
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "jobs": len(tasks),
        "completed_jobs": len(job_reports),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "score_sha256": file_sha256(score_path),
        "source_hashes": source_hashes,
        "fit_oof_only": True,
        "calibration_accessed": False,
        "val2017_accessed": False,
        "final_accessed": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {
            "complete": True,
            "score_sha256": report["score_sha256"],
            "report_sha256": file_sha256(report_path),
        },
    )
    _SCREEN_STATE.clear()
    gc.collect()
    return report


def _score_map(
    records: Sequence[Mapping[str, Any]], values: np.ndarray
) -> dict[int, dict[str, float]]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (
        len(records),
        int(PROTOCOL["model"]["max_actions"]),
    ):
        raise ValueError("Score matrix and action records differ")
    output = {}
    for row, record in enumerate(records):
        output[int(record["image_id"])] = {
            str(action["action_id"]): float(matrix[row, column])
            for column, action in enumerate(record["actions"])
            if str(action["action_id"]) != "edge"
        }
    return output


_SCREEN_EVAL_STATE: dict[str, Any] = {}


def _evaluate_screen_task(task: Mapping[str, Any]) -> dict[str, Any]:
    data = _SCREEN_EVAL_STATE["data"]
    rewards = _SCREEN_EVAL_STATE["rewards"][str(task["action_space"])]
    selection = {int(key): str(value) for key, value in task["selection"].items()}
    predictions, _ = REL.canonical.mixed_predictions(data, selection, 0.001)
    official = data["evaluator"].evaluate(predictions)
    request_bytes = sum(
        int(data["sample_by_id"][image_id]["payload_bytes"][action])
        for image_id, action in selection.items()
    )
    selected_rewards = [
        float(rewards[image_id][action]) for image_id, action in selection.items()
    ]
    counts = Counter(selection.values())
    roi_ids = [image_id for image_id, action in selection.items() if action == "roi"]
    suboptimal_roi = sum(
        float(rewards[image_id]["roi"])
        < max(
            [0.0]
            + [
                float(value)
                for action, value in rewards[image_id].items()
                if action != "roi"
            ]
        )
        - 1e-12
        for image_id in roi_ids
    )
    capacity = float(task["capacity_bytes"])
    mode = str(task["mode"])
    expected_calls = int(
        math.floor(len(data["samples"]) * float(task["budget"]) + 0.5)
    )
    utilization = (
        len(selection) / max(1, expected_calls)
        if mode == "fixed_call"
        else request_bytes / max(1.0, capacity)
    )
    return {
        "role": "fit_oof",
        "action_space": str(task["action_space"]),
        "method": str(task["method"]),
        "seed": str(task.get("seed", PROTOCOL["screen_seed"])),
        "mode": mode,
        "budget": float(task["budget"]),
        "official_ap50_95": float(official["map50_95"]),
        "official_ap50": float(official["map50"]),
        "official_ap75": float(official["map75"]),
        "kb_per_image": request_bytes / 1024.0 / len(data["samples"]),
        "request_bytes": int(request_bytes),
        "capacity_bytes": capacity,
        "capacity_utilization": float(utilization),
        "capacity_ok": bool(request_bytes <= capacity + 1e-6)
        if mode == "fixed_byte"
        else bool(len(selection) == expected_calls),
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["roi"]),
        "roi_ratio": int(counts["roi"]) / max(1, len(selection)),
        "suboptimal_roi_rate": suboptimal_roi / max(1, len(roi_ids)),
        "harmful_action_rate": sum(value <= 0.0 for value in selected_rewards)
        / max(1, len(selected_rewards)),
        "selected_teacher_reward": float(sum(selected_rewards)),
        "teacher_reward_ceiling": float(task["teacher_value"]),
        "teacher_regret": float(task["teacher_value"] - sum(selected_rewards)),
        "teacher_selection_jaccard": float(task["teacher_selection_jaccard"]),
        "score_reward_spearman": float(task["score_reward_spearman"]),
        "scheduler_lambda": float(task["scheduler_lambda"]),
    }


def _screen_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = {
        (str(row["mode"]), float(row["budget"])): row
        for row in rows
        if row["action_space"] == "OQ"
        and row["method"] == "CurrentActionValue"
    }
    comparisons = {}
    for action_space in PROTOCOL["action_spaces"]:
        for method in PROTOCOL["screen_methods"]:
            if action_space == "OQ" and method == "CurrentActionValue":
                continue
            candidate_rows = [
                row
                for row in rows
                if row["action_space"] == action_space and row["method"] == method
            ]
            work_points = []
            for row in candidate_rows:
                reference = baseline[(str(row["mode"]), float(row["budget"]))]
                work_points.append(
                    {
                        "mode": str(row["mode"]),
                        "budget": float(row["budget"]),
                        "delta_ap": float(row["official_ap50_95"])
                        - float(reference["official_ap50_95"]),
                        "capacity_utilization": float(row["capacity_utilization"]),
                        "delta_teacher_regret": float(row["teacher_regret"])
                        - float(reference["teacher_regret"]),
                    }
                )
            gate = CORE.evaluate_screen_gate(work_points, PROTOCOL["gate"])
            mean_delta = float(
                np.mean(
                    [
                        row["delta_ap"]
                        for row in work_points
                        if row["mode"] == "fixed_byte"
                    ]
                )
            )
            label = f"{action_space}/{method}"
            comparisons[label] = {
                "action_space": action_space,
                "method": method,
                "versus": "OQ/CurrentActionValue",
                "work_points": work_points,
                "fixed_byte_mean_delta_ap": mean_delta,
                "eligible_innovation": method != "CurrentActionValue",
                "gate": gate,
            }
    passed = [
        (label, value)
        for label, value in comparisons.items()
        if value["eligible_innovation"] and value["gate"]["passed"]
    ]
    promoted = (
        max(passed, key=lambda item: item[1]["fixed_byte_mean_delta_ap"])[0]
        if passed
        else None
    )
    return {
        "passed": promoted is not None,
        "promoted_candidate": promoted,
        "comparisons": comparisons,
    }


def run_screen_evaluation() -> dict[str, Any]:
    completion = RUNS / "screen/evaluation_completion.json"
    report_path = RUNS / "screen/evaluation_report.json"
    if completion.is_file():
        return read_json(report_path)
    training = run_screen_training()
    score_path = RUNS / "screen/scores.pkl"
    if training["score_sha256"] != file_sha256(score_path):
        raise RuntimeError("Screen score artifact changed")
    with score_path.open("rb") as handle:
        artifact = pickle.load(handle)
    data = REL.load_role("fit")
    samples = data["samples"]
    original_total = sum(
        int(sample["payload_bytes"]["original"]) for sample in samples
    )
    rewards_by_space = {
        action_space: _reward_map_for_space("fit", action_space)
        for action_space in PROTOCOL["action_spaces"]
    }
    tasks = []
    for action_space in PROTOCOL["action_spaces"]:
        records = _records_for_space("fit", action_space)
        expected_ids = [int(record["image_id"]) for record in records]
        if list(map(int, artifact["image_ids"][action_space])) != expected_ids:
            raise RuntimeError(f"{action_space} score order changed")
        rewards = rewards_by_space[action_space]
        for method in PROTOCOL["screen_methods"]:
            score_map = _score_map(records, artifact["scores"][action_space][method])
            flat_score, flat_reward = [], []
            for image_id, action_scores in score_map.items():
                for action, score in action_scores.items():
                    flat_score.append(float(score))
                    flat_reward.append(float(rewards[image_id][action]))
            correlation = REL.spearman(flat_score, flat_reward)
            for budget in PROTOCOL["budgets"]:
                capacity = float(budget) * original_total
                for mode in PROTOCOL["modes"]:
                    learned = (
                        ACTION.schedule_fixed_byte(
                            samples, score_map, capacity_bytes=capacity
                        )
                        if mode == "fixed_byte"
                        else ACTION.schedule_fixed_call(
                            samples, score_map, budget=float(budget)
                        )
                    )
                    teacher = (
                        ACTION.schedule_fixed_byte(
                            samples, rewards, capacity_bytes=capacity
                        )
                        if mode == "fixed_byte"
                        else ACTION.schedule_fixed_call(
                            samples, rewards, budget=float(budget)
                        )
                    )
                    selection = {
                        int(image_id): str(action)
                        for image_id, action in learned["selection"].items()
                    }
                    teacher_selection = {
                        (int(image_id), str(action))
                        for image_id, action in teacher["selection"].items()
                    }
                    tasks.append(
                        {
                            "action_space": action_space,
                            "method": method,
                            "mode": mode,
                            "budget": float(budget),
                            "capacity_bytes": capacity,
                            "selection": selection,
                            "teacher_value": float(teacher["total_value"]),
                            "teacher_selection_jaccard": REL.jaccard(
                                set(selection.items()), teacher_selection
                            ),
                            "score_reward_spearman": correlation,
                            "scheduler_lambda": float(learned.get("lambda", 0.0)),
                        }
                    )
    _SCREEN_EVAL_STATE.clear()
    _SCREEN_EVAL_STATE.update({"data": data, "rewards": rewards_by_space})
    workers = min(int(PROTOCOL["cpu"]["evaluation_workers"]), len(tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        rows = list(pool.imap_unordered(_evaluate_screen_task, tasks, chunksize=1))
    expected_rows = (
        len(PROTOCOL["action_spaces"])
        * len(PROTOCOL["screen_methods"])
        * len(PROTOCOL["modes"])
        * len(PROTOCOL["budgets"])
    )
    if len(rows) != expected_rows or not all(row["capacity_ok"] for row in rows):
        raise RuntimeError("Screen evaluation matrix is incomplete or invalid")
    rows.sort(
        key=lambda row: (
            list(PROTOCOL["action_spaces"]).index(str(row["action_space"])),
            PROTOCOL["screen_methods"].index(str(row["method"])),
            PROTOCOL["modes"].index(str(row["mode"])),
            float(row["budget"]),
        )
    )
    gate = _screen_gate(rows)
    results_path = RUNS / "screen/results.csv"
    atomic_csv(results_path, rows)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": "fit_oof",
        "rows": len(rows),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "gate": gate,
        "observed_fact": (
            "The screen compares identical encoders under scalar-reward versus "
            "complete cost-frontier supervision in both OQ and OQR spaces."
        ),
        "calibration_accessed": False,
        "val2017_accessed": False,
        "final_accessed": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {
            "complete": True,
            "results_sha256": file_sha256(results_path),
            "report_sha256": file_sha256(report_path),
        },
    )
    _SCREEN_EVAL_STATE.clear()
    gc.collect()
    return report


def run_screen() -> dict[str, Any]:
    run_screen_training()
    return run_screen_evaluation()


def _assemble_oof_matrix(
    action_space: str,
    method: str,
    seed: int,
    image_ids: Sequence[int],
) -> np.ndarray:
    position = {int(image_id): index for index, image_id in enumerate(image_ids)}
    matrix = np.full(
        (len(image_ids), int(PROTOCOL["model"]["max_actions"])),
        np.nan,
        dtype=np.float64,
    )
    for fold_index in range(int(PROTOCOL["fold_count"])):
        path = RUNS / (
            f"screen/jobs/{action_space}/{method}/seed_{seed}/"
            f"fold_{fold_index}/scores.npz"
        )
        with np.load(path, allow_pickle=False) as artifact:
            for row, image_id in enumerate(map(int, artifact["image_ids"])):
                matrix[position[image_id]] = artifact["scores"][row]
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"Incomplete OOF matrix: {action_space}/{method}/{seed}")
    return matrix


def _confirm_methods() -> tuple[tuple[str, str], tuple[str, str]]:
    report = read_json(RUNS / "screen/evaluation_report.json")
    promoted = str(report["gate"]["promoted_candidate"])
    if promoted != "OQR/BoundaryPathDistill":
        raise RuntimeError(f"Unexpected screened candidate: {promoted}")
    return ("OQ", "CurrentActionValue"), ("OQR", "BoundaryPathDistill")


def run_confirm_training() -> dict[str, Any]:
    completion = RUNS / "confirm/training_completion.json"
    report_path = RUNS / "confirm/training_report.json"
    if completion.is_file():
        return read_json(report_path)
    run_screen()
    methods = _confirm_methods()
    spaces = sorted({action_space for action_space, _ in methods})
    data_by_space = {}
    ids_by_space = {}
    for action_space in spaces:
        records, data = _screen_tensor_data("fit", action_space)
        data_by_space[action_space] = data
        ids_by_space[action_space] = [int(row["image_id"]) for row in records]
    folds = read_json(REL_ROOT / "runs/preflight/folds.json")["folds"]
    _SCREEN_STATE.clear()
    _SCREEN_STATE.update(
        {"data": data_by_space, "image_ids": ids_by_space, "folds": folds}
    )
    seeds = tuple(map(int, PROTOCOL["confirm_seeds"]))
    tasks = [
        (action_space, method, seed, fold_index)
        for action_space, method in methods
        for seed in seeds
        for fold_index in range(int(PROTOCOL["fold_count"]))
    ]
    workers = min(int(PROTOCOL["cpu"]["confirm_workers"]), len(tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        job_reports = list(pool.imap_unordered(_train_screen_job, tasks, chunksize=1))
    scores = {}
    for action_space, method in methods:
        label = f"{action_space}/{method}"
        scores[label] = {
            seed: _assemble_oof_matrix(
                action_space, method, seed, ids_by_space[action_space]
            )
            for seed in seeds
        }
    score_path = RUNS / "confirm/scores.pkl"
    atomic_pickle(
        score_path,
        {
            "image_ids": ids_by_space,
            "scores": scores,
            "seeds": seeds,
            "baseline": "OQ/CurrentActionValue",
            "candidate": "OQR/BoundaryPathDistill",
        },
    )
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "jobs": len(tasks),
        "completed_jobs": len(job_reports),
        "new_jobs": sum(
            int(row["base_seed"]) != int(PROTOCOL["screen_seed"])
            for row in job_reports
        ),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "score_sha256": file_sha256(score_path),
        "screen_results_sha256": file_sha256(RUNS / "screen/results.csv"),
        "source_hashes": {
            str(path): file_sha256(path)
            for path in (
                PROTOCOL_PATH,
                SCRIPT,
                ROOT / "scripts/core.py",
                ROOT / "scripts/training.py",
            )
        },
        "fit_oof_only": True,
        "calibration_accessed": False,
        "val2017_accessed": False,
        "final_accessed": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {
            "complete": True,
            "score_sha256": report["score_sha256"],
            "report_sha256": file_sha256(report_path),
        },
    )
    _SCREEN_STATE.clear()
    gc.collect()
    return report


def _confirm_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline_label = ("OQ", "CurrentActionValue")
    candidate_label = ("OQR", "BoundaryPathDistill")
    work_points = []
    for seed in [*map(str, PROTOCOL["confirm_seeds"]), "ensemble"]:
        for mode in PROTOCOL["modes"]:
            for budget in PROTOCOL["budgets"]:
                baseline = next(
                    row
                    for row in rows
                    if row["action_space"] == baseline_label[0]
                    and row["method"] == baseline_label[1]
                    and str(row["seed"]) == seed
                    and row["mode"] == mode
                    and math.isclose(float(row["budget"]), float(budget))
                )
                candidate = next(
                    row
                    for row in rows
                    if row["action_space"] == candidate_label[0]
                    and row["method"] == candidate_label[1]
                    and str(row["seed"]) == seed
                    and row["mode"] == mode
                    and math.isclose(float(row["budget"]), float(budget))
                )
                work_points.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "budget": float(budget),
                        "delta_ap": float(candidate["official_ap50_95"])
                        - float(baseline["official_ap50_95"]),
                        "capacity_utilization": float(
                            candidate["capacity_utilization"]
                        ),
                    }
                )
    seed_rows = [row for row in work_points if row["seed"] != "ensemble"]
    ensemble_rows = [row for row in work_points if row["seed"] == "ensemble"]
    result = CORE.evaluate_confirm_gate(seed_rows, ensemble_rows, PROTOCOL["gate"])
    result["work_points"] = work_points
    return result


def run_confirm_evaluation() -> dict[str, Any]:
    completion = RUNS / "confirm/evaluation_completion.json"
    report_path = RUNS / "confirm/evaluation_report.json"
    if completion.is_file():
        return read_json(report_path)
    training = run_confirm_training()
    score_path = RUNS / "confirm/scores.pkl"
    if training["score_sha256"] != file_sha256(score_path):
        raise RuntimeError("Confirmation score artifact changed")
    with score_path.open("rb") as handle:
        artifact = pickle.load(handle)
    data = REL.load_role("fit")
    samples = data["samples"]
    original_total = sum(
        int(sample["payload_bytes"]["original"]) for sample in samples
    )
    methods = _confirm_methods()
    rewards_by_space = {
        action_space: _reward_map_for_space("fit", action_space)
        for action_space, _ in methods
    }
    tasks = []
    seeds = tuple(map(int, PROTOCOL["confirm_seeds"]))
    for action_space, method in methods:
        label = f"{action_space}/{method}"
        records = _records_for_space("fit", action_space)
        ids = [int(row["image_id"]) for row in records]
        if ids != list(map(int, artifact["image_ids"][action_space])):
            raise RuntimeError(f"Confirmation {label} ordering changed")
        matrices = {str(seed): artifact["scores"][label][seed] for seed in seeds}
        matrices["ensemble"] = np.mean(list(matrices.values()), axis=0)
        rewards = rewards_by_space[action_space]
        for seed, matrix in matrices.items():
            score_map = _score_map(records, matrix)
            flat_score, flat_reward = [], []
            for image_id, action_scores in score_map.items():
                for action, score in action_scores.items():
                    flat_score.append(float(score))
                    flat_reward.append(float(rewards[image_id][action]))
            correlation = REL.spearman(flat_score, flat_reward)
            for budget in PROTOCOL["budgets"]:
                capacity = float(budget) * original_total
                for mode in PROTOCOL["modes"]:
                    learned = (
                        ACTION.schedule_fixed_byte(
                            samples, score_map, capacity_bytes=capacity
                        )
                        if mode == "fixed_byte"
                        else ACTION.schedule_fixed_call(
                            samples, score_map, budget=float(budget)
                        )
                    )
                    teacher = (
                        ACTION.schedule_fixed_byte(
                            samples, rewards, capacity_bytes=capacity
                        )
                        if mode == "fixed_byte"
                        else ACTION.schedule_fixed_call(
                            samples, rewards, budget=float(budget)
                        )
                    )
                    selection = {
                        int(image_id): str(action)
                        for image_id, action in learned["selection"].items()
                    }
                    tasks.append(
                        {
                            "seed": seed,
                            "action_space": action_space,
                            "method": method,
                            "mode": mode,
                            "budget": float(budget),
                            "capacity_bytes": capacity,
                            "selection": selection,
                            "teacher_value": float(teacher["total_value"]),
                            "teacher_selection_jaccard": REL.jaccard(
                                set(selection.items()),
                                {
                                    (int(image_id), str(action))
                                    for image_id, action in teacher["selection"].items()
                                },
                            ),
                            "score_reward_spearman": correlation,
                            "scheduler_lambda": float(learned.get("lambda", 0.0)),
                        }
                    )
    _SCREEN_EVAL_STATE.clear()
    _SCREEN_EVAL_STATE.update({"data": data, "rewards": rewards_by_space})
    workers = min(int(PROTOCOL["cpu"]["evaluation_workers"]), len(tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        rows = list(pool.imap_unordered(_evaluate_screen_task, tasks, chunksize=1))
    expected = len(methods) * (len(seeds) + 1) * len(PROTOCOL["modes"]) * len(
        PROTOCOL["budgets"]
    )
    if len(rows) != expected or not all(row["capacity_ok"] for row in rows):
        raise RuntimeError("Confirmation evaluation matrix is invalid")
    rows.sort(
        key=lambda row: (
            [f"{a}/{m}" for a, m in methods].index(
                f"{row['action_space']}/{row['method']}"
            ),
            [*map(str, seeds), "ensemble"].index(str(row["seed"])),
            PROTOCOL["modes"].index(str(row["mode"])),
            float(row["budget"]),
        )
    )
    gate = _confirm_gate(rows)
    results_path = RUNS / "confirm/results.csv"
    atomic_csv(results_path, rows)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": "fit_oof",
        "baseline": "OQ/CurrentActionValue",
        "candidate": "OQR/BoundaryPathDistill",
        "rows": len(rows),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "gate": gate,
        "calibration_accessed": False,
        "val2017_accessed": False,
        "final_accessed": False,
    }
    atomic_json(report_path, report)
    atomic_json(
        completion,
        {
            "complete": True,
            "results_sha256": file_sha256(results_path),
            "report_sha256": file_sha256(report_path),
        },
    )
    _SCREEN_EVAL_STATE.clear()
    gc.collect()
    return report


def run_confirm() -> dict[str, Any]:
    run_confirm_training()
    return run_confirm_evaluation()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "preflight",
            "diagnose",
            "roi-rewards",
            "screen",
            "confirm",
            "all",
        ),
        default="all",
    )
    args = parser.parse_args()
    if args.stage in {"preflight", "all"}:
        print(json.dumps(run_preflight(), indent=2), flush=True)
    if args.stage in {"diagnose", "all"}:
        print(json.dumps(run_diagnose(), indent=2), flush=True)
    if args.stage in {"roi-rewards", "all"}:
        print(json.dumps(run_roi_rewards("fit"), indent=2), flush=True)
    if args.stage in {"screen", "all"}:
        print(json.dumps(run_screen(), indent=2), flush=True)
    if args.stage in {"confirm", "all"}:
        print(json.dumps(run_confirm(), indent=2), flush=True)


if __name__ == "__main__":
    main()
