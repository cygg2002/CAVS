#!/usr/bin/env python3
"""Run the gated Budget-Set SharedAction + CostEligible diagnostic."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


SCRIPT = Path(__file__).resolve()
EXPERIMENT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = EXPERIMENT / "runs"
PROTOCOL_PATH = EXPERIMENT / "configs/protocol.json"
EXTERNAL = REPO / "paper/experiments/20260728_three_dataset_dcsb_oric_moric_external_baselines"

for path in (REPO, EXPERIMENT / "scripts", EXTERNAL / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_core import (  # noqa: E402
    build_output_feature,
    empirical_cdf,
    fit_reward_mlp,
    predict_reward_mlp,
)
from experiment_core import (  # noqa: E402
    ACTION_DIM,
    ACTION_TYPES,
    SCENE_DIM,
    BudgetSetRouter,
    assert_holdout_allowed,
    budget_set_loss,
    build_box_set,
    deployable_action_features,
    deployable_scene_features,
    deterministic_split,
    fixed_call_indices,
    select_costeligible,
    selection_mask,
    validate_protocol,
    validate_split_manifest,
)
from paper.experiments.CA_AORIC.ca_aoric import (  # noqa: E402
    CocoMetricBackend,
    action_mix,
    augment_full_region_actions,
    collect_coco_predictions,
    eligible_actions,
    filter_usable_actions,
    full_budget_capacity,
    selected_costs,
)
from paper.experiments.GAIR_v5.gair.cache import ShardStore  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def atomic_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty result table")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def source_fingerprint(protocol: Mapping[str, Any], store: ShardStore) -> dict[str, Any]:
    manifest_path = Path(protocol["source_cache"]) / "manifest.json"
    reward_path = Path(protocol["source_rewards"])
    return {
        "cache_fingerprint": str(store.manifest.get("fingerprint")),
        "cache_manifest_sha256": file_sha256(manifest_path),
        "reward_sha256": file_sha256(reward_path),
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
    }


def assert_fingerprint(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    if dict(expected) != dict(observed):
        raise RuntimeError("Immutable source fingerprint changed")


def _store(protocol: Mapping[str, Any]) -> ShardStore:
    store = ShardStore(
        protocol["source_cache"], expected_schema="gair_action_cache_v1", lru_size=4
    )
    if store.manifest.get("status") != "complete":
        raise RuntimeError("Source action cache is incomplete")
    if len(store) != int(protocol["expected_images"]):
        raise RuntimeError("Source action-cache population drifted")
    return store


def _prepared_record(record: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    usable = filter_usable_actions(
        dict(record), prediction_field=str(protocol["prediction_field"])
    )
    augmented = augment_full_region_actions(
        usable,
        nms_iou=float(protocol["nms_iou"]),
        max_composites=int(protocol["max_composites"]),
    )
    output = dict(augmented)
    output["actions"] = eligible_actions(
        augmented, max_actions_per_image=int(protocol["max_actions"])
    )
    return output


def _records_for_ids(
    store: ShardStore,
    image_ids: set[int],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for index in range(len(store)):
        record = store.record(index)
        if int(record["image_id"]) in image_ids:
            records.append(_prepared_record(record, protocol))
    records.sort(key=lambda record: int(record["image_id"]))
    if {int(record["image_id"]) for record in records} != set(map(int, image_ids)):
        raise RuntimeError("Requested record population is incomplete")
    return records


def _load_rewards(path: Path, allowed_ids: set[int]) -> dict[int, dict[str, float]]:
    rewards: dict[int, dict[str, float]] = {
        int(image_id): {} for image_id in allowed_ids
    }
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = int(row["image_id"])
            if image_id in allowed_ids:
                rewards[image_id][str(row["action_id"])] = float(row["reward"])
    return rewards


def _record_rewards(
    records: Sequence[Mapping[str, Any]], reward_map: Mapping[int, Mapping[str, float]]
) -> np.ndarray:
    max_actions = max(len(record["actions"]) for record in records)
    values = np.zeros((len(records), max_actions), dtype=np.float32)
    for image_index, record in enumerate(records):
        image_id = int(record["image_id"])
        for action_index, action in enumerate(record["actions"]):
            kind = str(action["action_type"])
            if kind == "edge":
                continue
            action_id = str(action["action_id"])
            if action_id not in reward_map[image_id]:
                raise RuntimeError(f"Missing reward for image={image_id} action={action_id}")
            values[image_index, action_index] = float(reward_map[image_id][action_id])
    return values


def _record_tensors(
    records: Sequence[Mapping[str, Any]],
    rewards: np.ndarray | None,
    protocol: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    max_boxes = int(protocol["max_boxes"])
    max_actions = int(protocol["max_actions"])
    boxes, classes, box_masks, scenes = [], [], [], []
    actions = torch.zeros((len(records), max_actions, ACTION_DIM), dtype=torch.float32)
    action_types = torch.zeros((len(records), max_actions), dtype=torch.long)
    action_masks = torch.zeros((len(records), max_actions), dtype=torch.bool)
    for image_index, record in enumerate(records):
        box, class_id, box_mask = build_box_set(
            record,
            max_boxes=max_boxes,
            confidence_floor=float(protocol["deploy_confidence"]),
        )
        boxes.append(box)
        classes.append(class_id)
        box_masks.append(box_mask)
        scenes.append(
            deployable_scene_features(
                record, deploy_confidence=float(protocol["deploy_confidence"])
            )
        )
        for action_index, action in enumerate(record["actions"]):
            actions[image_index, action_index] = deployable_action_features(action, record)
            action_types[image_index, action_index] = ACTION_TYPES[str(action["action_type"])]
            action_masks[image_index, action_index] = True
    output = {
        "boxes": torch.stack(boxes),
        "classes": torch.stack(classes),
        "box_mask": torch.stack(box_masks),
        "scene": torch.stack(scenes),
        "actions": actions,
        "action_type": action_types,
        "action_mask": action_masks,
    }
    if rewards is not None:
        target = torch.zeros((len(records), max_actions), dtype=torch.float32)
        target[:, : rewards.shape[1]] = torch.from_numpy(rewards)
        output["target"] = target
    return output


def _subset_tensors(data: Mapping[str, torch.Tensor], indices: Sequence[int]) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[index] for key, value in data.items()}


def _category_count(store: ShardStore) -> int:
    categories = list(store.manifest.get("metadata", {}).get("category_ids", ()))
    if not categories:
        raise RuntimeError("Action cache is missing category IDs")
    return len(categories)


def _model(protocol: Mapping[str, Any], category_count: int, seed: int) -> BudgetSetRouter:
    torch.manual_seed(int(seed))
    config = protocol["model"]
    return BudgetSetRouter(
        category_count=int(category_count),
        scene_dim=SCENE_DIM,
        action_dim=ACTION_DIM,
        class_embedding_dim=int(config["class_embedding_dim"]),
        box_hidden_dim=int(config["box_hidden_dim"]),
        action_hidden_dim=int(config["action_hidden_dim"]),
        joint_hidden_dim=int(config["joint_hidden_dim"]),
        dropout=float(config["dropout"]),
    )


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _to_device(data: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in data.items()}


def _teacher_masks(
    records: Sequence[Mapping[str, Any]],
    rewards: np.ndarray,
    budgets: Sequence[float],
    protocol: Mapping[str, Any],
) -> dict[float, np.ndarray]:
    capacity = full_budget_capacity(list(records), str(protocol["resource_profile"]))
    return {
        float(budget): selection_mask(
            records,
            select_costeligible(
                records,
                rewards,
                resource_profile=str(protocol["resource_profile"]),
                capacity=float(budget) * capacity,
            )["selected"],
            int(protocol["max_actions"]),
        )
        for budget in budgets
    }


def _epoch(
    model: BudgetSetRouter,
    data: Mapping[str, torch.Tensor],
    teacher_masks: Mapping[float, np.ndarray],
    protocol: Mapping[str, Any],
    *,
    seed: int,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    config = protocol["model"]
    losses = []
    metrics = {"reward_loss": 0.0, "within_loss": 0.0, "boundary_loss": 0.0}
    budgets = list(map(float, protocol["budgets"]))
    for offset, budget in enumerate(budgets):
        score = model(
            boxes=data["boxes"],
            classes=data["classes"],
            box_mask=data["box_mask"],
            scene=data["scene"],
            actions=data["actions"],
            action_type=data["action_type"],
            action_mask=data["action_mask"],
            budget=torch.full(
                (len(data["boxes"]),), budget, dtype=torch.float32, device=data["boxes"].device
            ),
        )
        result = budget_set_loss(
            score,
            data["target"],
            data["action_mask"],
            boundary_selected=torch.as_tensor(
                teacher_masks[budget], dtype=torch.bool, device=score.device
            ),
            boundary_pairs_count=int(config["pairs_per_budget"]),
            seed=int(seed) + offset * 1_000_003,
            reward_weight=float(config["reward_loss_weight"]),
            within_weight=float(config["within_action_weight"]),
            boundary_weight=float(config["boundary_weight"]),
        )
        losses.append(result["loss"])
        for key in metrics:
            metrics[key] += float(result[key]) / len(budgets)
    loss = torch.stack(losses).mean()
    if training:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return {"loss": float(loss.detach()), **metrics}


def _fit_one_router(
    records: Sequence[Mapping[str, Any]],
    data: Mapping[str, torch.Tensor],
    raw_rewards: np.ndarray,
    protocol: Mapping[str, Any],
    *,
    category_count: int,
    seed: int,
) -> dict[str, Any]:
    device = _device()
    image_ids = [int(record["image_id"]) for record in records]
    validation_count = max(
        1, int(round(len(records) * float(protocol["model"]["validation_fraction"])))
    )
    split = deterministic_split(
        image_ids, train_count=len(records) - validation_count, seed=int(seed)
    )
    index_by_id = {image_id: index for index, image_id in enumerate(image_ids)}
    fit_indices = [index_by_id[value] for value in split["train_image_ids"]]
    validation_indices = [index_by_id[value] for value in split["holdout_image_ids"]]
    fit_records = [records[index] for index in fit_indices]
    validation_records = [records[index] for index in validation_indices]
    fit_raw = raw_rewards[fit_indices]
    validation_raw = raw_rewards[validation_indices]

    valid_cloud = data["action_mask"] & (data["action_type"] != ACTION_TYPES["edge"])
    reward_values = data["target"][valid_cloud]
    reward_mean = float(reward_values.mean())
    reward_std = max(1e-6, float(reward_values.std(unbiased=False)))
    normalized = dict(data)
    normalized["target"] = (data["target"] - reward_mean) / reward_std
    fit_data = _to_device(_subset_tensors(normalized, fit_indices), device)
    validation_data = _to_device(_subset_tensors(normalized, validation_indices), device)
    fit_masks = _teacher_masks(fit_records, fit_raw, protocol["budgets"], protocol)
    validation_masks = _teacher_masks(
        validation_records, validation_raw, protocol["budgets"], protocol
    )

    model = _model(protocol, category_count, seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["model"]["learning_rate"]),
        weight_decay=float(protocol["model"]["weight_decay"]),
    )
    best_epoch, best_loss = 1, math.inf
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history = []
    for epoch in range(1, int(protocol["model"]["epochs"]) + 1):
        train_metrics = _epoch(
            model,
            fit_data,
            fit_masks,
            protocol,
            seed=int(seed) + epoch * 10_007,
            optimizer=optimizer,
        )
        with torch.no_grad():
            validation_metrics = _epoch(
                model,
                validation_data,
                validation_masks,
                protocol,
                seed=int(seed) + epoch * 10_007,
                optimizer=None,
            )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        print(
            f"[BudgetSet] seed={seed} epoch={epoch} "
            f"train={train_metrics['loss']:.6f} "
            f"validation={validation_metrics['loss']:.6f}",
            flush=True,
        )
        if validation_metrics["loss"] < best_loss - 1e-6:
            best_epoch = epoch
            best_loss = validation_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(protocol["model"]["patience"]):
            break

    final_model = _model(protocol, category_count, int(seed) + 1).to(device)
    all_data = _to_device(normalized, device)
    all_masks = _teacher_masks(records, raw_rewards, protocol["budgets"], protocol)
    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=float(protocol["model"]["learning_rate"]),
        weight_decay=float(protocol["model"]["weight_decay"]),
    )
    for epoch in range(1, best_epoch + 1):
        _epoch(
            final_model,
            all_data,
            all_masks,
            protocol,
            seed=int(seed) + 1 + epoch * 10_007,
            optimizer=optimizer,
        )
    return {
        "state_dict": {key: value.detach().cpu() for key, value in final_model.state_dict().items()},
        "seed": int(seed),
        "selected_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "history": history,
        "parameters": sum(parameter.numel() for parameter in final_model.parameters()),
        "device": str(device),
    }


def _predict_router(
    checkpoint: Mapping[str, Any],
    data: Mapping[str, torch.Tensor],
    protocol: Mapping[str, Any],
    category_count: int,
) -> np.ndarray:
    model = _model(protocol, category_count, int(checkpoint["seed"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    rows = []
    with torch.no_grad():
        for budget in map(float, protocol["budgets"]):
            score = model(
                boxes=data["boxes"],
                classes=data["classes"],
                box_mask=data["box_mask"],
                scene=data["scene"],
                actions=data["actions"],
                action_type=data["action_type"],
                action_mask=data["action_mask"],
                budget=torch.full((len(data["boxes"]),), budget),
            )
            raw = score.numpy() * float(checkpoint["reward_std"]) + float(
                checkpoint["reward_mean"]
            )
            raw[data["action_type"].numpy() == ACTION_TYPES["edge"]] = 0.0
            raw[~data["action_mask"].numpy()] = -1e9
            rows.append(raw)
    return np.asarray(rows, dtype=np.float64)


def _oric_features(
    records: Sequence[Mapping[str, Any]], category_ids: Sequence[int]
) -> np.ndarray:
    rows = []
    for record in records:
        predictions = []
        for box in record.get("edge_boxes", ()):
            class_index = int(box.get("cls", -1))
            if not 0 <= class_index < len(category_ids):
                continue
            x1, y1, x2, y2 = map(float, box["xyxy"])
            predictions.append(
                {
                    "category_id": int(category_ids[class_index]),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(box.get("conf", box.get("score", 0.0))),
                }
            )
        rows.append(
            build_output_feature(
                predictions,
                width=int(record["width"]),
                height=int(record["height"]),
                category_ids=category_ids,
                top_k=25,
            )
        )
    return np.asarray(rows, dtype=np.float64)


def _full_reward(records: Sequence[Mapping[str, Any]], rewards: np.ndarray) -> np.ndarray:
    values = []
    for image_index, record in enumerate(records):
        full_index = next(
            index
            for index, action in enumerate(record["actions"])
            if str(action["action_type"]) == "full"
        )
        values.append(float(rewards[image_index, full_index]))
    return np.asarray(values, dtype=np.float64)


def run_prepare() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    validate_protocol(protocol)
    completion_path = RUNS / "prepare/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "prepare/report.json")
    store = _store(protocol)
    image_ids = [int(store.record(index)["image_id"]) for index in range(len(store))]
    split = deterministic_split(
        image_ids,
        train_count=int(protocol["train_images"]),
        seed=int(protocol["split_seed"]),
    )
    validate_split_manifest(
        split,
        expected_images=int(protocol["expected_images"]),
        expected_train=int(protocol["train_images"]),
    )
    fingerprint = source_fingerprint(protocol, store)
    split.update({"protocol_id": protocol["protocol_id"], "source_fingerprint": fingerprint})
    split_path = RUNS / "prepare/split_manifest.json"
    atomic_json(split_path, split)
    report = {
        "protocol_id": protocol["protocol_id"],
        "train_images": len(split["train_image_ids"]),
        "holdout_images": len(split["holdout_image_ids"]),
        "holdout_metrics_accessed": False,
        "source_fingerprint": fingerprint,
    }
    atomic_json(RUNS / "prepare/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "split_sha256": file_sha256(split_path),
            "holdout_metrics_accessed": False,
        },
    )
    return report


def run_train() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    validate_protocol(protocol)
    completion_path = RUNS / "train/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "train/report.json")
    prepare = read_json(RUNS / "prepare/completion.json")
    if prepare.get("complete") is not True:
        raise RuntimeError("Prepare stage must complete before training")
    split = read_json(RUNS / "prepare/split_manifest.json")
    validate_split_manifest(
        split,
        expected_images=int(protocol["expected_images"]),
        expected_train=int(protocol["train_images"]),
    )
    store = _store(protocol)
    fingerprint = source_fingerprint(protocol, store)
    assert_fingerprint(split["source_fingerprint"], fingerprint)
    train_ids = set(map(int, split["train_image_ids"]))
    records = _records_for_ids(store, train_ids, protocol)
    reward_map = _load_rewards(Path(protocol["source_rewards"]), train_ids)
    rewards = _record_rewards(records, reward_map)
    data = _record_tensors(records, rewards, protocol)
    category_ids = list(map(int, store.manifest["metadata"]["category_ids"]))

    started = time.perf_counter()
    routers = [
        _fit_one_router(
            records,
            data,
            rewards,
            protocol,
            category_count=len(category_ids),
            seed=int(seed),
        )
        for seed in protocol["seeds"]
    ]
    features = _oric_features(records, category_ids)
    full_reward = _full_reward(records, rewards)
    baseline_config = dict(protocol["baseline_mlp"])
    oric = fit_reward_mlp(
        features, full_reward, np.ones_like(full_reward), baseline_config
    )
    moric_target = empirical_cdf(full_reward, full_reward)
    moric = fit_reward_mlp(
        features, moric_target, moric_target, baseline_config
    )
    artifact = {
        "routers": routers,
        "oric": oric,
        "moric": moric,
        "category_ids": category_ids,
        "train_image_ids": sorted(train_ids),
        "source_fingerprint": fingerprint,
    }
    checkpoint_path = RUNS / "train/checkpoints.pkl"
    atomic_pickle(checkpoint_path, artifact)
    report = {
        "protocol_id": protocol["protocol_id"],
        "train_images": len(records),
        "router_seeds": list(map(int, protocol["seeds"])),
        "selected_epochs": [int(item["selected_epoch"]) for item in routers],
        "validation_losses": [float(item["best_validation_loss"]) for item in routers],
        "router_parameters_each": [int(item["parameters"]) for item in routers],
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "reward_mean": float(rewards.mean()),
        "training_seconds": time.perf_counter() - started,
        "holdout_accessed": False,
        "source_fingerprint": fingerprint,
    }
    atomic_json(RUNS / "train/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "holdout_accessed": False,
            "source_fingerprint": fingerprint,
        },
    )
    return report


def _choice(
    record: Mapping[str, Any], action_index: int, value: float, resource_profile: str
) -> dict[str, Any]:
    actions = list(record["actions"])
    action = actions[int(action_index)]
    edge = next(item for item in actions if str(item["action_type"]) == "edge")
    if resource_profile == "upload":
        cost = float(action.get("payload_bytes", 0.0))
    elif resource_profile == "compute":
        cost = float(action.get("cloud_gflops", 0.0))
    else:
        cost = max(0.0, float(action.get("e2e_ms", 0.0)) - float(edge.get("e2e_ms", 0.0)))
    return {
        "image_id": int(record["image_id"]),
        "action_index": int(action_index),
        "action_id": str(action["action_id"]),
        "action_type": str(action["action_type"]),
        "action_record": action,
        "value": float(value),
        "cost": cost,
    }


def _fixed_call_selected(
    records: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    action_mask: np.ndarray,
    action_types: np.ndarray,
    *,
    budget: float,
    full_only: bool,
    resource_profile: str,
) -> list[dict[str, Any]]:
    selected = {
        image_index: action_index
        for image_index, action_index in fixed_call_indices(
            scores,
            action_mask,
            action_types,
            budget=budget,
            full_only=full_only,
        )
    }
    output = []
    for image_index, record in enumerate(records):
        action_index = selected.get(
            image_index,
            next(
                index
                for index, action in enumerate(record["actions"])
                if str(action["action_type"]) == "edge"
            ),
        )
        output.append(
            _choice(
                record,
                action_index,
                scores[image_index, action_index],
                resource_profile,
            )
        )
    return output


def _baseline_score_matrix(
    records: Sequence[Mapping[str, Any]], values: np.ndarray, max_actions: int
) -> np.ndarray:
    scores = np.full((len(records), int(max_actions)), -1e9, dtype=np.float64)
    for image_index, record in enumerate(records):
        for action_index, action in enumerate(record["actions"]):
            kind = str(action["action_type"])
            if kind == "edge":
                scores[image_index, action_index] = 0.0
            elif kind == "full":
                scores[image_index, action_index] = float(values[image_index])
    return scores


def _evaluate_selection(
    metric: CocoMetricBackend,
    records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    category_ids: list[int],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    official = metric.evaluate(
        collect_coco_predictions(
            records,
            selected,
            category_ids=category_ids,
            prediction_field=str(protocol["prediction_field"]),
        ),
        [int(record["image_id"]) for record in records],
    )
    return {
        **official,
        **selected_costs(selected, len(records)),
        **{f"action_{key}": value for key, value in action_mix(selected).items()},
    }


def _gate(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    primary = [row for row in rows if row["mode"] == "fixed_byte"]
    comparisons = []
    for budget in map(float, protocol["budgets"]):
        candidate = next(
            row
            for row in primary
            if row["method"] == "BudgetSet-SharedAction"
            and math.isclose(float(row["budget"]), budget)
        )
        controls = [
            row
            for row in primary
            if row["method"] in {"ORIC-MLP-Full", "MORIC-MLP-Full"}
            and math.isclose(float(row["budget"]), budget)
        ]
        baseline = max(controls, key=lambda row: float(row["map50_95"]))
        comparisons.append(
            {
                "budget": budget,
                "baseline": baseline["method"],
                "baseline_ap": float(baseline["map50_95"]),
                "candidate_ap": float(candidate["map50_95"]),
                "ap_gain": float(candidate["map50_95"]) - float(baseline["map50_95"]),
                "baseline_kb": float(baseline["upload_kb_per_image"]),
                "candidate_kb": float(candidate["upload_kb_per_image"]),
            }
        )
    success = protocol["success"]
    gains = [item["ap_gain"] for item in comparisons]
    gain_points = sum(value >= float(success["minimum_gain"]) for value in gains)
    primary_passed = (
        gain_points >= int(success["required_gain_points"])
        and min(gains) >= -float(success["maximum_other_loss"])
    )
    fixed = [row for row in rows if row["mode"] == "fixed_call" and math.isclose(float(row["budget"]), 0.1)]
    candidate = next(row for row in fixed if row["method"] == "BudgetSet-SharedAction")
    baseline = max(
        (row for row in fixed if row["method"] in {"ORIC-MLP-Full", "MORIC-MLP-Full"}),
        key=lambda row: float(row["map50_95"]),
    )
    secondary_passed = (
        float(candidate["map50_95"]) >= float(baseline["map50_95"])
        and float(candidate["upload_kb_per_image"])
        <= float(baseline["upload_kb_per_image"])
        + float(success["ten_percent_kb_increase_tolerance"])
    )
    return {
        "fixed_byte_comparisons": comparisons,
        "gain_points": gain_points,
        "primary_passed": primary_passed,
        "fixed_call_10": {
            "baseline": baseline["method"],
            "baseline_ap": baseline["map50_95"],
            "candidate_ap": candidate["map50_95"],
            "baseline_kb": baseline["upload_kb_per_image"],
            "candidate_kb": candidate["upload_kb_per_image"],
            "passed": secondary_passed,
        },
        "success": bool(primary_passed and secondary_passed),
    }


def run_holdout() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    validate_protocol(protocol)
    completion_path = RUNS / "holdout/completion.json"
    train_completion = read_json(RUNS / "train/completion.json")
    assert_holdout_allowed(train_completion, completion_path.is_file())
    split = read_json(RUNS / "prepare/split_manifest.json")
    validate_split_manifest(
        split,
        expected_images=int(protocol["expected_images"]),
        expected_train=int(protocol["train_images"]),
    )
    store = _store(protocol)
    fingerprint = source_fingerprint(protocol, store)
    assert_fingerprint(train_completion["source_fingerprint"], fingerprint)
    with (RUNS / "train/checkpoints.pkl").open("rb") as handle:
        checkpoints = pickle.load(handle)
    holdout_ids = set(map(int, split["holdout_image_ids"]))
    records = _records_for_ids(store, holdout_ids, protocol)
    data = _record_tensors(records, None, protocol)
    category_ids = list(map(int, checkpoints["category_ids"]))
    candidate = np.mean(
        [
            _predict_router(item, data, protocol, len(category_ids))
            for item in checkpoints["routers"]
        ],
        axis=0,
    )
    features = _oric_features(records, category_ids)
    baseline_values = {
        "ORIC-MLP-Full": predict_reward_mlp(checkpoints["oric"], features),
        "MORIC-MLP-Full": predict_reward_mlp(checkpoints["moric"], features),
    }
    baseline_scores = {
        name: _baseline_score_matrix(records, values, int(protocol["max_actions"]))
        for name, values in baseline_values.items()
    }
    action_mask = data["action_mask"].numpy()
    action_types = data["action_type"].numpy()
    metric = CocoMetricBackend(Path(protocol["annotations"]))
    full_capacity = full_budget_capacity(records, str(protocol["resource_profile"]))
    rows = []
    methods = [
        ("ORIC-MLP-Full", baseline_scores["ORIC-MLP-Full"], True),
        ("MORIC-MLP-Full", baseline_scores["MORIC-MLP-Full"], True),
    ]
    for budget_index, budget in enumerate(map(float, protocol["budgets"])):
        methods_for_budget = [
            *methods,
            ("BudgetSet-Full", candidate[budget_index], True),
            ("BudgetSet-SharedAction", candidate[budget_index], False),
        ]
        for method, scores, full_only in methods_for_budget:
            fixed_selected = _fixed_call_selected(
                records,
                scores,
                action_mask,
                action_types,
                budget=budget,
                full_only=full_only,
                resource_profile=str(protocol["resource_profile"]),
            )
            rows.append(
                {
                    "method": method,
                    "mode": "fixed_call",
                    "budget": budget,
                    **_evaluate_selection(
                        metric, records, fixed_selected, category_ids, protocol
                    ),
                }
            )
            byte_result = select_costeligible(
                records,
                scores,
                resource_profile=str(protocol["resource_profile"]),
                capacity=budget * full_capacity,
                full_only=full_only,
            )
            rows.append(
                {
                    "method": method,
                    "mode": "fixed_byte",
                    "budget": budget,
                    "capacity": budget * full_capacity,
                    "used_capacity": byte_result["total_cost"],
                    "lambda": byte_result["lambda"],
                    **_evaluate_selection(
                        metric, records, byte_result["selected"], category_ids, protocol
                    ),
                }
            )
    gate = _gate(rows, protocol)
    result_path = RUNS / "holdout/results.csv"
    atomic_rows(result_path, rows)
    report = {
        "protocol_id": protocol["protocol_id"],
        "scope": "transductive_reward_context_diagnostic_not_paper_final",
        "holdout_images": len(records),
        "holdout_evaluated_once": True,
        "val2017_final_accessed": False,
        "gate": gate,
        "rows": rows,
        "source_fingerprint": fingerprint,
    }
    report_path = RUNS / "holdout/report.json"
    atomic_json(report_path, report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "success": bool(gate["success"]),
            "results_sha256": file_sha256(result_path),
            "report_sha256": file_sha256(report_path),
            "val2017_final_accessed": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "holdout"), required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        report = run_prepare()
    elif args.stage == "train":
        report = run_train()
    else:
        report = run_holdout()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
