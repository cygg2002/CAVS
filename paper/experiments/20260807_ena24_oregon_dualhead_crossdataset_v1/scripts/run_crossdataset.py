#!/usr/bin/env python3
"""Run cache-only DualHead experiments for canonical ENA24 and Oregon."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from crossdataset_core import (
    BUDGETS,
    TRACK_A_METHODS,
    TRACK_B_POLICIES,
    assert_capacity_safe,
    assert_same_membership,
    filter_actions,
    track_b_bootstrap_metadata,
    validate_protocol,
)


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL_PATH = ROOT / "configs/protocol.json"


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


PROTOCOL = read_json(PROTOCOL_PATH)
validate_protocol(PROTOCOL)
SOURCES = {
    name: REPO / path for name, path in PROTOCOL["source_experiments"].items()
}
CANONICAL_ROOT = SOURCES["canonical"]
NINERANKER_ROOT = SOURCES["nineranker_triaction"]


def _prepend(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


for module_root in (
    CANONICAL_ROOT / "scripts",
    REPO
    / "paper/experiments/20260722_ena24_hierarchical_triaction_paper_exploration/scripts",
    REPO / "paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1/scripts",
    REPO / "paper/experiments/20260804_coco_budgetset_value_cost_oof_v1/scripts",
    SOURCES["actionvalue"] / "scripts",
    SOURCES["external"] / "scripts",
):
    _prepend(module_root)

import run_multiranker_benchmark as canonical  # type: ignore  # noqa: E402
import run_exploration as tri  # type: ignore  # noqa: E402
from action_core import (  # type: ignore  # noqa: E402
    build_action_records,
    schedule_fixed_byte,
    schedule_fixed_call,
)
from baseline_core import detection_stats, oric_reward  # type: ignore  # noqa: E402
from oof_train import (  # type: ignore  # noqa: E402
    fit_dual_router,
    fit_value_router,
    predict_dual_value,
    predict_value_router,
)
import run_experiment as budgetset  # type: ignore  # noqa: E402

# This experiment is deliberately CPU-only even if a GPU becomes visible while
# the resumable job is running.
budgetset._device = lambda: torch.device("cpu")


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
    return path.is_file() and read_json(path).get("complete") is True


def canonical_common(dataset: str) -> Path:
    run = str(PROTOCOL["datasets"][dataset]["canonical_run"])
    return CANONICAL_ROOT / "runs" / run / "common" / dataset


def canonical_benchmark(dataset: str) -> Path:
    run = str(PROTOCOL["datasets"][dataset]["canonical_run"])
    return CANONICAL_ROOT / "runs" / run / "benchmark" / dataset


def dataset_run(dataset: str) -> Path:
    return RUNS / dataset


def configure_canonical(dataset: str) -> None:
    common = canonical_common(dataset)
    canonical.COMMON = common.parent
    tri.current.COMMON = common.parent
    tri.current.dataset_cost = tri.canonical_dataset_cost
    if Path(canonical.COMMON) / dataset != common:
        raise RuntimeError(f"Canonical cache binding failed for {dataset}")


def load_role(dataset: str, role: str) -> dict[str, Any]:
    expected = int(PROTOCOL["datasets"][dataset]["roles"][role])
    configure_canonical(dataset)
    data = canonical.load_role(dataset, role)
    if len(data["samples"]) != expected:
        raise RuntimeError(
            f"{dataset}/{role} population changed: {len(data['samples'])} != {expected}"
        )
    identities = [str(sample["identity"]) for sample in data["samples"]]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"Duplicate role-local identity in {dataset}/{role}")
    return data


def role_source_files(dataset: str, role: str) -> list[Path]:
    root = canonical_common(dataset) / role
    manifest_path = root / "source_manifest.json"
    manifest = read_json(manifest_path)
    paths = [manifest_path, root / "samples.json", root / "gt.json"]
    for action in ("edge", "original", "q90", "roi"):
        source = Path(manifest["action_sources"][action])
        if not source.is_file():
            raise FileNotFoundError(source)
        paths.append(source)
    return paths


def required_source_files(dataset: str) -> list[Path]:
    paths = {
        PROTOCOL_PATH,
        SCRIPT,
        ROOT / "scripts/crossdataset_core.py",
        CANONICAL_ROOT / "scripts/run_multiranker_benchmark.py",
        canonical_benchmark(dataset) / "models.json",
        canonical_benchmark(dataset) / "action_gate.json",
        NINERANKER_ROOT / f"runs/{dataset}/models/saferoi_gbdt.pkl",
        NINERANKER_ROOT / f"runs/{dataset}/threshold_choices.json",
        NINERANKER_ROOT / f"runs/{dataset}/results.csv",
        NINERANKER_ROOT / f"runs/{dataset}/selections.json",
        REPO
        / "paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1/scripts/run_experiment.py",
        REPO
        / "paper/experiments/20260804_coco_budgetset_value_cost_oof_v1/scripts/oof_train.py",
        SOURCES["actionvalue"] / "scripts/action_core.py",
        SOURCES["external"] / "scripts/baseline_core.py",
    }
    for role in ("fit", "calibration", "final"):
        paths.update(role_source_files(dataset, role))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen sources:\n" + "\n".join(sorted(missing)))
    lowered = "\n".join(str(path).lower() for path in paths)
    for forbidden in PROTOCOL["forbidden_source_substrings"]:
        if str(forbidden).lower() in lowered:
            raise RuntimeError(f"Forbidden Cloud-V2 source: {forbidden}")
    return sorted(path.resolve() for path in paths)


def capture_hashes(dataset: str) -> dict[str, str]:
    return {str(path): file_sha256(path) for path in required_source_files(dataset)}


def source_digest(hashes: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(hashes), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_preflight(dataset: str) -> dict[str, Any]:
    output = dataset_run(dataset) / "preflight"
    if complete(output / "completion.json"):
        return read_json(output / "report.json")
    roles = {role: load_role(dataset, role) for role in ("fit", "calibration", "final")}
    identity_sets = {
        role: {str(sample["identity"]) for sample in data["samples"]}
        for role, data in roles.items()
    }
    overlap = {
        "fit_calibration": len(identity_sets["fit"] & identity_sets["calibration"]),
        "fit_final": len(identity_sets["fit"] & identity_sets["final"]),
        "calibration_final": len(identity_sets["calibration"] & identity_sets["final"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Identity leakage across roles for {dataset}: {overlap}")
    choices = read_json(NINERANKER_ROOT / f"runs/{dataset}/threshold_choices.json")
    observed_threshold = float(choices["shared"]["fixed_call"]["threshold"])
    expected_threshold = float(
        PROTOCOL["datasets"][dataset]["gbdt_fixed_call_threshold"]
    )
    if not math.isclose(observed_threshold, expected_threshold, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"Frozen GBDT threshold drift for {dataset}")
    hashes = capture_hashes(dataset)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "dataset": dataset,
        "passed": True,
        "role_counts": {role: len(data["samples"]) for role, data in roles.items()},
        "identity_overlap": overlap,
        "gbdt_fixed_call_threshold": observed_threshold,
        "source_sha256": hashes,
        "source_digest": source_digest(hashes),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gpu_inference": False,
        "detector_inference": False,
        "track_a_final_accessed": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(
        output / "completion.json",
        {"complete": True, "source_digest": report["source_digest"]},
    )
    return report


def verify_hashes(dataset: str) -> dict[str, str]:
    report = read_json(dataset_run(dataset) / "preflight/report.json")
    observed = capture_hashes(dataset)
    if observed != report["source_sha256"]:
        raise RuntimeError(f"Frozen source changed after preflight for {dataset}")
    return observed


_STATS_STATE: dict[str, Any] = {}


def _stats_worker(image_id: int) -> tuple[int, dict[str, Any]]:
    data = _STATS_STATE["data"]
    sample = data["sample_by_id"][int(image_id)]
    target = data["gt_map"].get(int(image_id), [])
    output = {}
    for action in ("edge", "original", "q90"):
        if action != "edge" and not bool(sample["available"].get(action, False)):
            continue
        predictions = canonical.filter_conf(
            data["actions"][action].get(int(image_id), []),
            float(PROTOCOL["reward"]["confidence"]),
        )
        output[action] = detection_stats(
            predictions,
            target,
            tuple(map(float, PROTOCOL["reward"]["iou_thresholds"])),
        )
    return int(image_id), output


def fork_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("fork")
    except ValueError as error:
        raise RuntimeError("This cache-only pipeline requires multiprocessing fork") from error


def build_stats(dataset: str, role: str, workers: int) -> dict[str, Any]:
    output = dataset_run(dataset) / "stats"
    path = output / f"{role}.pkl"
    marker = output / f"{role}_completion.json"
    if complete(marker):
        return load_pickle(path)
    data = load_role(dataset, role)
    _STATS_STATE.clear()
    _STATS_STATE["data"] = data
    stats = {action: {} for action in ("edge", "original", "q90")}
    started = time.time()
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=fork_context()) as pool:
        for completed_count, (image_id, action_stats) in enumerate(
            pool.map(_stats_worker, data["image_ids"], chunksize=16), start=1
        ):
            for action, value in action_stats.items():
                stats[action][int(image_id)] = value
            if completed_count % 512 == 0 or completed_count == len(data["image_ids"]):
                print(
                    f"[{dataset}/stats/{role}] {completed_count}/{len(data['image_ids'])}",
                    flush=True,
                )
    payload = {
        "dataset": dataset,
        "role": role,
        "image_ids": list(map(int, data["image_ids"])),
        "identities": {
            int(sample["image_id"]): str(sample["identity"])
            for sample in data["samples"]
        },
        "stats": stats,
        "elapsed_seconds": time.time() - started,
    }
    atomic_pickle(path, payload)
    atomic_json(
        marker,
        {
            "complete": True,
            "images": len(data["image_ids"]),
            "sha256": file_sha256(path),
            "elapsed_seconds": payload["elapsed_seconds"],
        },
    )
    _STATS_STATE.clear()
    return payload


def deterministic_folds(samples: Sequence[Mapping[str, Any]], count: int, seed: int) -> list[list[int]]:
    ordered = sorted(
        range(len(samples)),
        key=lambda index: hashlib.sha256(
            f"{seed}:{samples[index]['identity']}".encode("utf-8")
        ).digest(),
    )
    folds = [[] for _ in range(int(count))]
    for position, index in enumerate(ordered):
        folds[position % int(count)].append(int(index))
    return folds


def _context_ids(
    candidate_ids: Sequence[int],
    identities: Mapping[int, str],
    target_identity: str,
    *,
    count: int,
    seed: int,
) -> list[int]:
    candidates = [
        int(image_id)
        for image_id in candidate_ids
        if str(identities[int(image_id)]) != str(target_identity)
    ]
    if not candidates:
        return []
    digest = hashlib.sha256(f"{seed}:{target_identity}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    take = min(int(count), len(candidates))
    indices = rng.choice(len(candidates), size=take, replace=False)
    return [candidates[int(index)] for index in indices]


_REWARD_STATE: dict[str, Any] = {}


def _reward_worker(item: tuple[int, str]) -> tuple[int, dict[str, float]]:
    image_id, identity = item
    context_ids = _context_ids(
        _REWARD_STATE["context_ids"],
        _REWARD_STATE["context_identities"],
        str(identity),
        count=int(PROTOCOL["reward"]["context_size"]),
        seed=int(PROTOCOL["reward"]["context_seed"]),
    )
    context = [_REWARD_STATE["context_stats"]["edge"][value] for value in context_ids]
    target_stats = _REWARD_STATE["target_stats"]
    edge = target_stats["edge"][int(image_id)]
    values = {}
    for action in PROTOCOL["reward"]["candidate_actions"]:
        cloud = target_stats[str(action)].get(int(image_id))
        if cloud is not None:
            values[str(action)] = float(oric_reward(context, edge, cloud)["reward"])
    return int(image_id), values


def compute_rewards(
    dataset: str,
    *,
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
    rewards = load_pickle(partial) if partial.is_file() else {}
    target_items = [
        (int(image_id), str(target_stats["identities"][int(image_id)]))
        for image_id in target_stats["image_ids"]
    ]
    remaining = [item for item in target_items if item[0] not in rewards]
    _REWARD_STATE.clear()
    _REWARD_STATE.update(
        {
            "context_ids": list(map(int, context_ids)),
            "context_identities": context_stats["identities"],
            "context_stats": context_stats["stats"],
            "target_stats": target_stats["stats"],
        }
    )
    checkpoint_interval = int(PROTOCOL["reward"]["checkpoint_interval"])
    started = time.time()
    if remaining:
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=fork_context()) as pool:
            for count, (image_id, values) in enumerate(
                pool.map(_reward_worker, remaining, chunksize=16), start=1
            ):
                rewards[int(image_id)] = values
                if count % checkpoint_interval == 0 or count == len(remaining):
                    atomic_pickle(partial, rewards)
                    print(
                        f"[{dataset}/rewards/{label}] {len(rewards)}/{len(target_items)}",
                        flush=True,
                    )
    if {item[0] for item in target_items} != set(rewards):
        raise RuntimeError(f"Incomplete reward cache: {dataset}/{label}")
    atomic_pickle(path, rewards)
    atomic_json(
        marker,
        {
            "complete": True,
            "images": len(rewards),
            "context_images": len(context_ids),
            "sha256": file_sha256(path),
            "elapsed_seconds": time.time() - started,
        },
    )
    partial.unlink(missing_ok=True)
    _REWARD_STATE.clear()
    return rewards


def run_rewards(dataset: str, workers: int) -> dict[str, Any]:
    output = dataset_run(dataset) / "rewards"
    if complete(output / "completion.json"):
        return read_json(output / "report.json")
    verify_hashes(dataset)
    fit_data = load_role(dataset, "fit")
    fit_stats = build_stats(dataset, "fit", workers)
    calibration_stats = build_stats(dataset, "calibration", workers)
    all_fit_ids = list(map(int, fit_stats["image_ids"]))
    full_rewards = compute_rewards(
        dataset,
        label="fit_full",
        context_stats=fit_stats,
        target_stats=fit_stats,
        context_ids=all_fit_ids,
        workers=workers,
    )
    folds = deterministic_folds(
        fit_data["samples"],
        int(PROTOCOL["track_a"]["fold_count"]),
        int(PROTOCOL["track_a"]["fold_seed"]),
    )
    fold_manifest = {}
    for fold_index, heldout_indices in enumerate(folds):
        heldout = set(heldout_indices)
        train_ids = [
            int(sample["image_id"])
            for index, sample in enumerate(fit_data["samples"])
            if index not in heldout
        ]
        fold_manifest[str(fold_index)] = {
            "train_identities": [
                str(sample["identity"])
                for index, sample in enumerate(fit_data["samples"])
                if index not in heldout
            ],
            "heldout_identities": [
                str(fit_data["samples"][index]["identity"])
                for index in heldout_indices
            ],
        }
        compute_rewards(
            dataset,
            label=f"fit_fold_{fold_index}",
            context_stats=fit_stats,
            target_stats=fit_stats,
            context_ids=train_ids,
            workers=workers,
        )
    calibration_rewards = compute_rewards(
        dataset,
        label="calibration",
        context_stats=fit_stats,
        target_stats=calibration_stats,
        context_ids=all_fit_ids,
        workers=workers,
    )
    atomic_json(output / "fold_manifest.json", fold_manifest)
    report = {
        "dataset": dataset,
        "fit_images": len(full_rewards),
        "calibration_images": len(calibration_rewards),
        "folds": len(folds),
        "fold_contained_context": True,
        "final_accessed": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(output / "completion.json", {"complete": True})
    return report


def category_ids(data: Mapping[str, Any]) -> list[int]:
    gt = read_json(Path(data["gt_path"]))
    values = [int(row["id"]) for row in gt["categories"]]
    if not values or len(values) != len(set(values)):
        raise RuntimeError("Canonical category IDs are missing or duplicated")
    return values


def set_protocol(max_actions: int) -> dict[str, Any]:
    return {
        "budgets": list(BUDGETS),
        "max_boxes": int(PROTOCOL["model"]["max_boxes"]),
        "max_actions": int(max_actions),
        "deploy_confidence": float(PROTOCOL["model"]["deploy_confidence"]),
        "model": dict(PROTOCOL["model"]),
        "resource_profile": "upload",
    }


def role_records(
    data: Mapping[str, Any], categories: Sequence[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    oq = build_action_records(
        data["samples"], data["actions"]["edge"], category_ids=categories
    )
    original = [filter_actions(record, ("edge", "original")) for record in oq]
    oq = [filter_actions(record, ("edge", "original", "q90")) for record in oq]
    return original, oq


def reward_matrix(
    records: Sequence[Mapping[str, Any]],
    rewards: Mapping[int, Mapping[str, float]],
    *,
    max_actions: int,
) -> np.ndarray:
    output = np.zeros((len(records), int(max_actions)), dtype=np.float32)
    for image_index, record in enumerate(records):
        image_id = int(record["image_id"])
        if image_id not in rewards:
            raise RuntimeError(f"Missing contextual reward for image {image_id}")
        for action_index, action in enumerate(record["actions"]):
            action_id = str(action["action_id"])
            if action_id == "edge":
                continue
            if action_id not in rewards[image_id]:
                raise RuntimeError(f"Missing reward for image/action {image_id}/{action_id}")
            output[image_index, action_index] = float(rewards[image_id][action_id])
    return output


def subset_tensor_map(
    values: Mapping[str, torch.Tensor], indices: Sequence[int]
) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[index] for key, value in values.items()}


def train_models(
    original_records: Sequence[Mapping[str, Any]],
    oq_records: Sequence[Mapping[str, Any]],
    original_rewards: np.ndarray,
    oq_rewards: np.ndarray,
    *,
    categories: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    original_protocol = set_protocol(2)
    oq_protocol = set_protocol(3)
    original_data = budgetset._record_tensors(
        original_records, original_rewards, original_protocol
    )
    oq_data = budgetset._record_tensors(oq_records, oq_rewards, oq_protocol)
    image_ids = [int(record["image_id"]) for record in original_records]
    cpu = torch.device("cpu")
    current = budgetset._fit_one_router(
        original_records,
        original_data,
        original_rewards,
        original_protocol,
        category_count=len(categories),
        seed=int(seed),
    )
    value = fit_value_router(
        image_ids,
        original_data,
        original_protocol,
        category_count=len(categories),
        seed=int(seed),
        device=cpu,
    )
    dual = fit_dual_router(
        original_records,
        original_data,
        original_rewards,
        original_protocol,
        category_count=len(categories),
        seed=int(seed),
        device=cpu,
    )
    action_value = fit_value_router(
        image_ids,
        oq_data,
        oq_protocol,
        category_count=len(categories),
        seed=int(seed),
        device=cpu,
    )
    return {
        "current": current,
        "value": value,
        "dual": dual,
        "action_value": action_value,
        "category_ids": list(map(int, categories)),
        "original_protocol": original_protocol,
        "oq_protocol": oq_protocol,
    }


def predict_models(
    artifact: Mapping[str, Any],
    original_records: Sequence[Mapping[str, Any]],
    oq_records: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    categories = list(map(int, artifact["category_ids"]))
    original_protocol = artifact["original_protocol"]
    oq_protocol = artifact["oq_protocol"]
    original_data = budgetset._record_tensors(
        original_records, None, original_protocol
    )
    oq_data = budgetset._record_tensors(oq_records, None, oq_protocol)
    return {
        "Current-BudgetSet-SharedAction": budgetset._predict_router(
            artifact["current"], original_data, original_protocol, len(categories)
        )[:, :, 1],
        "SetValue-SharedAction": predict_value_router(
            artifact["value"],
            original_data,
            original_protocol,
            category_count=len(categories),
        )[:, :, 1],
        "DualHead-SharedAction": predict_dual_value(
            artifact["dual"],
            original_records,
            original_data,
            original_protocol,
            category_count=len(categories),
        )[:, :, 1],
        "ActionValue-OQ-Joint": predict_value_router(
            artifact["action_value"],
            oq_data,
            oq_protocol,
            category_count=len(categories),
        ),
    }


_TRAIN_STATE: dict[str, Any] = {}


def _train_oof_job(item: tuple[int, int]) -> dict[str, Any]:
    fold_index, seed = map(int, item)
    torch.set_num_threads(int(PROTOCOL["workers"]["torch_threads_per_worker"]))
    dataset = str(_TRAIN_STATE["dataset"])
    output = dataset_run(dataset) / "models/oof" / f"fold_{fold_index}_seed_{seed}.pkl"
    marker = output.with_suffix(".completion.json")
    if complete(marker):
        return {"fold": fold_index, "seed": seed, "cached": True}
    heldout_indices = list(map(int, _TRAIN_STATE["folds"][fold_index]))
    heldout = set(heldout_indices)
    train_indices = [
        index for index in range(len(_TRAIN_STATE["original_records"])) if index not in heldout
    ]
    rewards = load_pickle(
        dataset_run(dataset) / f"rewards/fit_fold_{fold_index}.pkl"
    )
    original_reward = reward_matrix(
        _TRAIN_STATE["original_records"], rewards, max_actions=2
    )
    oq_reward = reward_matrix(_TRAIN_STATE["oq_records"], rewards, max_actions=3)
    artifact = train_models(
        [_TRAIN_STATE["original_records"][index] for index in train_indices],
        [_TRAIN_STATE["oq_records"][index] for index in train_indices],
        original_reward[train_indices],
        oq_reward[train_indices],
        categories=_TRAIN_STATE["category_ids"],
        seed=seed,
    )
    predictions = predict_models(
        artifact,
        [_TRAIN_STATE["original_records"][index] for index in heldout_indices],
        [_TRAIN_STATE["oq_records"][index] for index in heldout_indices],
    )
    payload = {
        "fold": fold_index,
        "seed": seed,
        "indices": heldout_indices,
        "scores": predictions,
        "selected_epochs": {
            key: int(artifact[name]["selected_epoch"])
            for key, name in (
                ("Current-BudgetSet", "current"),
                ("SetValue", "value"),
                ("DualHead", "dual"),
                ("ActionValue-OQ-Joint", "action_value"),
            )
        },
    }
    atomic_pickle(output, payload)
    atomic_json(
        marker,
        {"complete": True, "sha256": file_sha256(output)},
    )
    return {"fold": fold_index, "seed": seed, "cached": False}


def _train_full_job(seed: int) -> dict[str, Any]:
    seed = int(seed)
    torch.set_num_threads(int(PROTOCOL["workers"]["torch_threads_per_worker"]))
    dataset = str(_TRAIN_STATE["dataset"])
    output = dataset_run(dataset) / "models/full" / f"seed_{seed}.pkl"
    marker = output.with_suffix(".completion.json")
    if complete(marker):
        return {"seed": seed, "cached": True}
    rewards = load_pickle(dataset_run(dataset) / "rewards/fit_full.pkl")
    original_reward = reward_matrix(
        _TRAIN_STATE["original_records"], rewards, max_actions=2
    )
    oq_reward = reward_matrix(_TRAIN_STATE["oq_records"], rewards, max_actions=3)
    artifact = train_models(
        _TRAIN_STATE["original_records"],
        _TRAIN_STATE["oq_records"],
        original_reward,
        oq_reward,
        categories=_TRAIN_STATE["category_ids"],
        seed=seed,
    )
    artifact["dataset"] = dataset
    artifact["fit_images"] = len(_TRAIN_STATE["original_records"])
    artifact["source_digest"] = read_json(
        dataset_run(dataset) / "preflight/report.json"
    )["source_digest"]
    atomic_pickle(output, artifact)
    atomic_json(
        marker,
        {"complete": True, "sha256": file_sha256(output), "fit_only": True},
    )
    return {"seed": seed, "cached": False}


def _run_jobs(function, jobs: Sequence[Any], workers: int) -> list[Any]:
    if not jobs:
        return []
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=fork_context()) as pool:
        return list(pool.map(function, jobs, chunksize=1))


def run_train(dataset: str, workers: int) -> dict[str, Any]:
    output = dataset_run(dataset) / "models"
    if complete(output / "completion.json"):
        return read_json(output / "report.json")
    verify_hashes(dataset)
    fit = load_role(dataset, "fit")
    categories = category_ids(fit)
    original_records, oq_records = role_records(fit, categories)
    folds = deterministic_folds(
        fit["samples"],
        int(PROTOCOL["track_a"]["fold_count"]),
        int(PROTOCOL["track_a"]["fold_seed"]),
    )
    _TRAIN_STATE.clear()
    _TRAIN_STATE.update(
        {
            "dataset": dataset,
            "category_ids": categories,
            "original_records": original_records,
            "oq_records": oq_records,
            "folds": folds,
        }
    )
    seeds = list(map(int, PROTOCOL["model"]["seeds"]))
    oof_jobs = [
        (fold_index, seed)
        for fold_index in range(len(folds))
        for seed in seeds
    ]
    oof_reports = _run_jobs(_train_oof_job, oof_jobs, workers)
    full_reports = _run_jobs(_train_full_job, seeds, min(int(workers), len(seeds)))
    report = {
        "dataset": dataset,
        "fit_images": len(original_records),
        "category_count": len(categories),
        "folds": len(folds),
        "seeds": seeds,
        "oof_jobs": oof_reports,
        "full_jobs": full_reports,
        "cpu_only": True,
        "final_accessed": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(output / "completion.json", {"complete": True})
    _TRAIN_STATE.clear()
    return report


def ensemble_oof_scores(dataset: str) -> dict[str, np.ndarray]:
    path = dataset_run(dataset) / "scores/oof.pkl"
    marker = path.with_suffix(".completion.json")
    if complete(marker):
        return load_pickle(path)
    fit = load_role(dataset, "fit")
    count = len(fit["samples"])
    seeds = list(map(int, PROTOCOL["model"]["seeds"]))
    fold_count = int(PROTOCOL["track_a"]["fold_count"])
    per_seed = {
        method: (
            np.full((len(seeds), len(BUDGETS), count, 3), np.nan, dtype=np.float64)
            if method == "ActionValue-OQ-Joint"
            else np.full((len(seeds), len(BUDGETS), count), np.nan, dtype=np.float64)
        )
        for method in TRACK_A_METHODS
    }
    seed_index = {seed: index for index, seed in enumerate(seeds)}
    for fold in range(fold_count):
        for seed in seeds:
            artifact = load_pickle(
                dataset_run(dataset) / f"models/oof/fold_{fold}_seed_{seed}.pkl"
            )
            indices = list(map(int, artifact["indices"]))
            for method, values in artifact["scores"].items():
                per_seed[method][seed_index[seed]][:, indices, ...] = np.asarray(values)
    for method, values in per_seed.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete OOF scores for {dataset}/{method}")
    payload = {
        "ensemble": {method: values.mean(axis=0) for method, values in per_seed.items()},
        "per_seed": per_seed,
        "image_ids": [int(sample["image_id"]) for sample in fit["samples"]],
        "identities": [str(sample["identity"]) for sample in fit["samples"]],
    }
    atomic_pickle(path, payload)
    atomic_json(marker, {"complete": True, "sha256": file_sha256(path)})
    return payload


def score_full_models(dataset: str, role: str) -> dict[str, np.ndarray]:
    if role not in ("calibration", "final"):
        raise ValueError(role)
    path = dataset_run(dataset) / f"scores/{role}.pkl"
    marker = path.with_suffix(".completion.json")
    if complete(marker):
        return load_pickle(path)
    data = load_role(dataset, role)
    categories = category_ids(data)
    original_records, oq_records = role_records(data, categories)
    per_seed: dict[str, list[np.ndarray]] = {method: [] for method in TRACK_A_METHODS}
    for seed in map(int, PROTOCOL["model"]["seeds"]):
        artifact = load_pickle(dataset_run(dataset) / f"models/full/seed_{seed}.pkl")
        if artifact["category_ids"] != categories:
            raise RuntimeError(f"Category map changed for {dataset}/{role}")
        predicted = predict_models(artifact, original_records, oq_records)
        for method, values in predicted.items():
            per_seed[method].append(np.asarray(values, dtype=np.float64))
    payload = {
        "ensemble": {
            method: np.mean(np.stack(values, axis=0), axis=0)
            for method, values in per_seed.items()
        },
        "per_seed": {
            method: np.stack(values, axis=0) for method, values in per_seed.items()
        },
        "image_ids": [int(sample["image_id"]) for sample in data["samples"]],
        "identities": [str(sample["identity"]) for sample in data["samples"]],
    }
    atomic_pickle(path, payload)
    atomic_json(marker, {"complete": True, "sha256": file_sha256(path)})
    return payload


def frozen_models_and_base_actions(
    dataset: str, data: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[int, str]]:
    configure_canonical(dataset)
    models = tri.load_frozen_models(canonical_benchmark(dataset) / "models.json")
    gate = read_json(canonical_benchmark(dataset) / "action_gate.json")
    actions = canonical.chosen_actions(models["action_heads"], gate, data)
    if set(actions) != {int(sample["image_id"]) for sample in data["samples"]}:
        raise RuntimeError(f"Frozen SharedAction population changed for {dataset}")
    if any(action not in {"original", "q90"} for action in actions.values()):
        raise RuntimeError(f"SharedAction unexpectedly contains ROI for {dataset}")
    return models, actions


def policy_action_maps(
    dataset: str, data: Mapping[str, Any]
) -> tuple[dict[str, dict[int, str]], dict[str, Any]]:
    models, base = frozen_models_and_base_actions(dataset, data)
    model_path = NINERANKER_ROOT / f"runs/{dataset}/models/saferoi_gbdt.pkl"
    with model_path.open("rb") as handle:
        roi_model = pickle.load(handle)
    roi_scores = tri.predict_gbdt(roi_model, tri.profile_features(data))
    threshold = float(PROTOCOL["datasets"][dataset]["gbdt_fixed_call_threshold"])
    maps = {
        "SharedAction": base,
        "CostEligible-AllROI": tri.action_map_at_threshold(
            data, base, roi_scores, -math.inf
        ),
        "GBDT-Hierarchical": tri.action_map_at_threshold(
            data, base, roi_scores, threshold
        ),
    }
    return maps, models


def request_bytes(
    data: Mapping[str, Any], selection: Mapping[int, str]
) -> int:
    return sum(
        int(data["sample_by_id"][int(image_id)]["payload_bytes"][str(action)])
        for image_id, action in selection.items()
    )


def score_map_for_actions(
    data: Mapping[str, Any], values: np.ndarray, action_map: Mapping[int, str]
) -> dict[int, dict[str, float]]:
    if np.asarray(values).shape != (len(data["samples"]),):
        raise RuntimeError("Scalar score shape differs from role population")
    return {
        int(sample["image_id"]): {
            str(action_map[int(sample["image_id"])]): float(values[index])
        }
        for index, sample in enumerate(data["samples"])
    }


def score_map_for_oq(
    data: Mapping[str, Any], values: np.ndarray
) -> dict[int, dict[str, float]]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(data["samples"]), 3):
        raise RuntimeError(f"ActionValue score shape changed: {values.shape}")
    return {
        int(sample["image_id"]): {
            "original": float(values[index, 1]),
            "q90": float(values[index, 2]),
        }
        for index, sample in enumerate(data["samples"])
    }


def role_evaluation_rewards(dataset: str, role: str) -> dict[int, dict[str, float]]:
    if role == "calibration":
        return load_pickle(dataset_run(dataset) / "rewards/calibration.pkl")
    if role != "fit_oof":
        raise ValueError(role)
    fit = load_role(dataset, "fit")
    folds = deterministic_folds(
        fit["samples"],
        int(PROTOCOL["track_a"]["fold_count"]),
        int(PROTOCOL["track_a"]["fold_seed"]),
    )
    output: dict[int, dict[str, float]] = {}
    for fold_index, indices in enumerate(folds):
        rewards = load_pickle(
            dataset_run(dataset) / f"rewards/fit_fold_{fold_index}.pkl"
        )
        for index in indices:
            image_id = int(fit["samples"][int(index)]["image_id"])
            output[image_id] = dict(rewards[image_id])
    if len(output) != len(fit["samples"]):
        raise RuntimeError(f"OOF reward population changed for {dataset}")
    return output


def teacher_schedule(
    data: Mapping[str, Any],
    rewards: Mapping[int, Mapping[str, float]],
    *,
    mode: str,
    budget: float,
) -> dict[str, Any]:
    if mode == "fixed_call":
        return schedule_fixed_call(data["samples"], rewards, budget=float(budget))
    capacity = float(budget) * sum(
        float(sample["payload_bytes"]["original"]) for sample in data["samples"]
    )
    return schedule_fixed_byte(
        data["samples"], rewards, capacity_bytes=capacity
    )


def evaluate_track_a_selection(
    dataset: str,
    role: str,
    data: Mapping[str, Any],
    models: Mapping[str, Any],
    method: str,
    mode: str,
    budget: float,
    selection: Mapping[int, str],
    rewards: Mapping[int, Mapping[str, float]],
) -> dict[str, Any]:
    row = canonical.evaluate(
        method,
        mode,
        float(budget),
        data,
        dict(selection),
        dict(models),
    )
    used = request_bytes(data, selection)
    capacity = float(budget) * sum(
        float(sample["payload_bytes"]["original"]) for sample in data["samples"]
    )
    if mode == "fixed_byte":
        assert_capacity_safe(used, capacity)
    expected_count = int(math.floor(len(data["samples"]) * float(budget) + 0.5))
    if mode == "fixed_call" and len(selection) != expected_count:
        raise RuntimeError(
            f"Fixed-call count changed: {dataset}/{role}/{method}/{budget}"
        )
    selected_reward = sum(
        float(rewards[int(image_id)].get(str(action), 0.0))
        for image_id, action in selection.items()
    )
    teacher = teacher_schedule(data, rewards, mode=mode, budget=budget)
    row.update(
        {
            "dataset": dataset,
            "role": role,
            "method": method,
            "mode": mode,
            "nominal_budget": float(budget),
            "capacity_bytes": capacity,
            "used_bytes": used,
            "capacity_utilization": used / max(1.0, capacity),
            "selected_contextual_reward": selected_reward,
            "teacher_contextual_reward": float(teacher["total_value"]),
            "teacher_regret": float(teacher["total_value"]) - selected_reward,
            "result_status": "development_oof"
            if role == "fit_oof"
            else "frozen_calibration",
            "final_used": False,
        }
    )
    return row


def evaluate_track_a_role(
    dataset: str,
    role: str,
    scores: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    source_role = "fit" if role == "fit_oof" else "calibration"
    data = load_role(dataset, source_role)
    models, base_actions = frozen_models_and_base_actions(dataset, data)
    rewards = role_evaluation_rewards(dataset, role)
    rows = []
    saved: dict[str, dict[str, str]] = {}
    for method in TRACK_A_METHODS:
        for budget_index, budget in enumerate(BUDGETS):
            for mode in ("fixed_byte", "fixed_call"):
                if method == "ActionValue-OQ-Joint":
                    score_map = score_map_for_oq(data, scores[method][budget_index])
                    if mode == "fixed_call":
                        result = schedule_fixed_call(
                            data["samples"], score_map, budget=float(budget)
                        )
                    else:
                        capacity = float(budget) * sum(
                            float(sample["payload_bytes"]["original"])
                            for sample in data["samples"]
                        )
                        result = schedule_fixed_byte(
                            data["samples"], score_map, capacity_bytes=capacity
                        )
                    selection = result["selection"]
                else:
                    values = np.asarray(scores[method][budget_index])
                    if mode == "fixed_call":
                        selection = canonical.top_k_actions(
                            data, values, base_actions, float(budget)
                        )
                    else:
                        scalar = score_map_for_actions(data, values, base_actions)
                        capacity = float(budget) * sum(
                            float(sample["payload_bytes"]["original"])
                            for sample in data["samples"]
                        )
                        selection = schedule_fixed_byte(
                            data["samples"], scalar, capacity_bytes=capacity
                        )["selection"]
                key = f"{method}|{mode}|{budget:g}"
                saved[key] = {str(image_id): str(action) for image_id, action in selection.items()}
                rows.append(
                    evaluate_track_a_selection(
                        dataset,
                        role,
                        data,
                        models,
                        method,
                        mode,
                        budget,
                        selection,
                        rewards,
                    )
                )
    return rows, saved


def run_track_a(dataset: str) -> dict[str, Any]:
    output = dataset_run(dataset) / "track_a"
    if complete(output / "completion.json"):
        return read_json(output / "report.json")
    verify_hashes(dataset)
    oof = ensemble_oof_scores(dataset)
    calibration = score_full_models(dataset, "calibration")
    rows, selections = evaluate_track_a_role(
        dataset, "fit_oof", oof["ensemble"]
    )
    calibration_rows, calibration_selections = evaluate_track_a_role(
        dataset, "calibration", calibration["ensemble"]
    )
    rows.extend(calibration_rows)
    selections.update(
        {f"calibration|{key}": value for key, value in calibration_selections.items()}
    )
    expected = 2 * len(TRACK_A_METHODS) * 2 * len(BUDGETS)
    keys = {
        (
            row["role"],
            row["method"],
            row["mode"],
            float(row["nominal_budget"]),
        )
        for row in rows
    }
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError(f"Track A matrix incomplete for {dataset}")
    atomic_csv(output / "results.csv", rows)
    atomic_json(output / "selections.json", selections)
    report = {
        "dataset": dataset,
        "rows": len(rows),
        "methods": list(TRACK_A_METHODS),
        "roles": ["fit_oof", "calibration"],
        "final_accessed": False,
        "result_status": "development_oof_and_frozen_calibration",
    }
    atomic_json(output / "report.json", report)
    atomic_json(output / "completion.json", {"complete": True})
    return report


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def strongest_historical_row(
    dataset: str, policy: str, budget: float
) -> dict[str, str]:
    rows = read_csv(NINERANKER_ROOT / f"runs/{dataset}/results.csv")
    candidates = [
        row
        for row in rows
        if str(row.get("mode")) == "fixed_call"
        and str(row.get("action_policy")) == str(policy)
        and math.isclose(float(row.get("nominal_budget", "nan")), float(budget))
        and str(row.get("role")) == "final"
    ]
    if not candidates:
        raise RuntimeError(f"Missing historical comparator for {dataset}/{policy}/{budget}")
    return max(
        candidates,
        key=lambda row: (
            float(row["official_map50_95"]),
            -float(row["request_kb_per_image"]),
            str(row["method"]),
        ),
    )


def historical_selection(
    dataset: str, method: str, policy: str, budget: float
) -> dict[int, str]:
    selections = read_json(NINERANKER_ROOT / f"runs/{dataset}/selections.json")
    matches = []
    for key, value in selections.items():
        parts = str(key).split("|")
        if (
            len(parts) == 4
            and parts[0] == "fixed_call"
            and parts[1] == policy
            and parts[2] == method
            and math.isclose(float(parts[3]), float(budget))
        ):
            matches.append(value)
    if len(matches) != 1:
        raise RuntimeError(f"Historical selection is ambiguous: {dataset}/{method}/{policy}/{budget}")
    return {int(image_id): str(action) for image_id, action in matches[0].items()}


def run_track_b_core(dataset: str) -> dict[str, Any]:
    output = dataset_run(dataset) / "track_b"
    if complete(output / "core_completion.json"):
        return read_json(output / "core_report.json")
    verify_hashes(dataset)
    data = load_role(dataset, "final")
    scores = score_full_models(dataset, "final")["ensemble"][
        "DualHead-SharedAction"
    ]
    action_maps, models = policy_action_maps(dataset, data)
    rows = []
    selections: dict[str, dict[str, str]] = {}
    comparator_selections: dict[str, dict[str, str]] = {}
    for budget_index, budget in enumerate(BUDGETS):
        values = np.asarray(scores[budget_index], dtype=np.float64)
        memberships = {}
        for policy in TRACK_B_POLICIES:
            action_map = action_maps[policy]
            selection = canonical.top_k_actions(
                data, values, action_map, float(budget)
            )
            memberships[policy] = set(selection)
            expected_count = int(math.floor(len(data["samples"]) * budget + 0.5))
            if len(selection) != expected_count:
                raise RuntimeError(f"Track B fixed-call count drift for {dataset}/{budget}")
            key = f"{policy}|{budget:g}"
            selections[key] = {
                str(image_id): str(action) for image_id, action in selection.items()
            }
            row = canonical.evaluate(
                "DualHead-Value",
                "fixed_call",
                float(budget),
                data,
                selection,
                models,
            )
            comparator = strongest_historical_row(dataset, policy, budget)
            baseline_selection = historical_selection(
                dataset, comparator["method"], policy, budget
            )
            comparator_selections[key] = {
                str(image_id): str(action)
                for image_id, action in baseline_selection.items()
            }
            row.update(
                {
                    "dataset": dataset,
                    "role": "final",
                    "method": "DualHead-Value",
                    "mode": "fixed_call",
                    "nominal_budget": float(budget),
                    "action_policy": policy,
                    "comparator": str(comparator["method"]),
                    "comparator_map50_95": float(comparator["official_map50_95"]),
                    "delta_map50_95": float(row["official_map50_95"])
                    - float(comparator["official_map50_95"]),
                    "comparator_kb_per_image": float(comparator["request_kb_per_image"]),
                    "delta_kb_per_image": float(row["request_kb_per_image"])
                    - float(comparator["request_kb_per_image"]),
                    "result_status": str(
                        PROTOCOL["datasets"][dataset]["track_b_status"]
                    ),
                    "final_used_for_training_or_selection": False,
                }
            )
            rows.append(row)
        assert_same_membership(memberships)
    expected = len(TRACK_B_POLICIES) * len(BUDGETS)
    keys = {
        (row["action_policy"], float(row["nominal_budget"])) for row in rows
    }
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError(f"Track B matrix incomplete for {dataset}")
    atomic_csv(output / "results.csv", rows)
    atomic_json(output / "selections.json", selections)
    atomic_json(output / "comparator_selections.json", comparator_selections)
    report = {
        "dataset": dataset,
        "rows": len(rows),
        "method": "DualHead-Value",
        "policies": list(TRACK_B_POLICIES),
        "result_status": PROTOCOL["datasets"][dataset]["track_b_status"],
        "final_used_for_training_or_selection": False,
        "policy_membership_identical": True,
        "bootstrap_complete": False,
    }
    atomic_json(output / "core_report.json", report)
    atomic_json(output / "core_completion.json", {"complete": True})
    return report


_BOOTSTRAP_STATE: dict[str, Any] = {}


def _bootstrap_job(key: str) -> dict[str, Any]:
    dataset = str(_BOOTSTRAP_STATE["dataset"])
    output = dataset_run(dataset) / "track_b/bootstrap" / f"{key.replace('|', '_')}.json"
    marker = output.with_suffix(".completion.json")
    if complete(marker):
        return read_json(output)
    candidate = {
        int(image_id): str(action)
        for image_id, action in _BOOTSTRAP_STATE["candidate"][key].items()
    }
    baseline = {
        int(image_id): str(action)
        for image_id, action in _BOOTSTRAP_STATE["baseline"][key].items()
    }
    result = tri.paired_bootstrap(
        _BOOTSTRAP_STATE["data"],
        baseline_actions=baseline,
        candidate_actions=candidate,
        replicates=int(PROTOCOL["track_b"]["bootstrap_replicates"]),
        seed=int(PROTOCOL["track_b"]["bootstrap_seed"]),
        unit="image",
    )
    result.update(
        track_b_bootstrap_metadata(
            key, _BOOTSTRAP_STATE["core_index"][key]
        )
    )
    payload = {"dataset": dataset, "key": key, **result}
    atomic_json(output, payload)
    atomic_json(marker, {"complete": True, "sha256": file_sha256(output)})
    return payload


def run_bootstrap(dataset: str, workers: int) -> dict[str, Any]:
    output = dataset_run(dataset) / "track_b"
    if complete(output / "completion.json"):
        return read_json(output / "report.json")
    if not complete(output / "core_completion.json"):
        raise RuntimeError(f"Track B core must complete before bootstrap: {dataset}")
    data = load_role(dataset, "final")
    candidate = read_json(output / "selections.json")
    baseline = read_json(output / "comparator_selections.json")
    if set(candidate) != set(baseline):
        raise RuntimeError(f"Bootstrap comparison population changed for {dataset}")
    core_rows = read_csv(output / "results.csv")
    core_index = {
        f"{row['action_policy']}|{float(row['nominal_budget']):g}": row
        for row in core_rows
    }
    if set(core_index) != set(candidate) or len(core_index) != len(core_rows):
        raise RuntimeError(f"Bootstrap metadata population changed for {dataset}")
    _BOOTSTRAP_STATE.clear()
    _BOOTSTRAP_STATE.update(
        {
            "dataset": dataset,
            "data": data,
            "candidate": candidate,
            "baseline": baseline,
            "core_index": core_index,
        }
    )
    summaries = _run_jobs(
        _bootstrap_job,
        sorted(candidate),
        min(int(workers), len(candidate)),
    )
    atomic_csv(output / "paired_bootstrap.csv", summaries)
    core_report = read_json(output / "core_report.json")
    report = {**core_report, "bootstrap_complete": True, "bootstrap_rows": len(summaries)}
    atomic_json(output / "report.json", report)
    atomic_json(output / "completion.json", {"complete": True})
    _BOOTSTRAP_STATE.clear()
    return report


def run_dataset(dataset: str, stage: str, workers: int) -> dict[str, Any]:
    if dataset not in PROTOCOL["datasets"]:
        raise ValueError(dataset)
    if stage in ("preflight", "all"):
        run_preflight(dataset)
    if stage in ("rewards", "all"):
        if not complete(dataset_run(dataset) / "preflight/completion.json"):
            raise RuntimeError("Preflight must complete before rewards")
        run_rewards(dataset, workers)
    if stage in ("train", "all"):
        if not complete(dataset_run(dataset) / "rewards/completion.json"):
            raise RuntimeError("Rewards must complete before training")
        run_train(dataset, int(PROTOCOL["workers"]["training_per_dataset"]))
    if stage in ("track_a", "all"):
        if not complete(dataset_run(dataset) / "models/completion.json"):
            raise RuntimeError("Training must complete before Track A")
        run_track_a(dataset)
    if stage in ("track_b", "all"):
        if not complete(dataset_run(dataset) / "track_a/completion.json"):
            raise RuntimeError("Track A must complete before the post-hoc final extension")
        run_track_b_core(dataset)
    if stage in ("bootstrap", "all"):
        if not complete(dataset_run(dataset) / "track_b/core_completion.json"):
            raise RuntimeError("Track B core must complete before bootstrap")
        run_bootstrap(
            dataset, int(PROTOCOL["workers"]["bootstrap_per_dataset"])
        )
    state = {
        "dataset": dataset,
        "requested_stage": stage,
        "preflight": complete(dataset_run(dataset) / "preflight/completion.json"),
        "rewards": complete(dataset_run(dataset) / "rewards/completion.json"),
        "train": complete(dataset_run(dataset) / "models/completion.json"),
        "track_a": complete(dataset_run(dataset) / "track_a/completion.json"),
        "track_b_core": complete(dataset_run(dataset) / "track_b/core_completion.json"),
        "bootstrap": complete(dataset_run(dataset) / "track_b/completion.json"),
    }
    if all(state[key] for key in ("preflight", "rewards", "train", "track_a", "track_b_core", "bootstrap")):
        atomic_json(dataset_run(dataset) / "COMPLETED.json", {"complete": True, **state})
    atomic_json(dataset_run(dataset) / "pipeline_status.json", state)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(PROTOCOL["datasets"]), required=True)
    parser.add_argument(
        "--stage",
        choices=("preflight", "rewards", "train", "track_a", "track_b", "bootstrap", "all"),
        default="all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(PROTOCOL["workers"]["cache_per_dataset"]),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    state = run_dataset(arguments.dataset, arguments.stage, arguments.workers)
    print(json.dumps(state, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
