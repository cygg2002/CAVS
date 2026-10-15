"""Adapters that retrain every COCO-era scheduler on Cloud-V3 Oregon data."""

from __future__ import annotations

from collections import Counter
import copy
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
SOURCE = {
    "set": REPO / "paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1/scripts",
    "value": REPO / "paper/experiments/20260804_coco_budgetset_value_cost_oof_v1/scripts",
    "risk": REPO / "paper/experiments/20260805_coco_actionconditioned_valuerisk_oof_v1/scripts",
    "rel": REPO / "paper/experiments/20260806_coco_relational_actionvalue_oof_v1/scripts",
    "factor": REPO / "paper/experiments/20260806_coco_factorized_global_actionrank_oof_v1/scripts",
    "afpd": REPO / "paper/experiments/20260809_coco_frontier_aligned_actionvalue_exploration_v1/scripts",
    "occupancy": REPO / "paper/experiments/20260809_coco_frontier_occupancy_gate_exploration_v2/scripts",
    "external": REPO / "paper/experiments/20260728_three_dataset_dcsb_oric_moric_external_baselines/scripts",
}


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sources() -> dict[str, Any]:
    if hasattr(_sources, "cache"):
        return getattr(_sources, "cache")
    for path in SOURCE.values():
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    set_core = _load("experiment_core", SOURCE["set"] / "experiment_core.py")
    external = _load("baseline_core", SOURCE["external"] / "baseline_core.py")
    set_run = _load("oregon_cloudv3_set_runner", SOURCE["set"] / "run_experiment.py")
    sys.modules["run_experiment"] = set_run
    value_core = _load("oof_core", SOURCE["value"] / "oof_core.py")
    value_train = _load("oregon_cloudv3_value_train", SOURCE["value"] / "oof_train.py")
    risk = _load("oregon_cloudv3_value_risk", SOURCE["risk"] / "value_risk.py")
    rel = _load("oregon_cloudv3_rel_core", SOURCE["rel"] / "core.py")
    factor = _load("oregon_cloudv3_factor_core", SOURCE["factor"] / "core.py")
    afpd_core = _load("oregon_cloudv3_afpd_core", SOURCE["afpd"] / "core.py")
    afpd = _load("oregon_cloudv3_afpd_training", SOURCE["afpd"] / "training.py")
    occupancy_core = _load(
        "oregon_cloudv3_occupancy_core", SOURCE["occupancy"] / "core.py"
    )
    occupancy = _load(
        "oregon_cloudv3_occupancy_training",
        SOURCE["occupancy"] / "training.py",
    )
    result = {
        "set_core": set_core,
        "set_run": set_run,
        "value_core": value_core,
        "value_train": value_train,
        "risk": risk,
        "rel": rel,
        "factor": factor,
        "afpd_core": afpd_core,
        "afpd": afpd,
        "occupancy_core": occupancy_core,
        "occupancy": occupancy,
        "external": external,
    }
    setattr(_sources, "cache", result)
    return result


def tensor_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "max_boxes": int(protocol["model"]["max_boxes"]),
        "max_actions": int(protocol["model"]["max_actions"]),
        "deploy_confidence": float(protocol["model"]["deploy_confidence"]),
        "budgets": list(map(float, protocol["budgets"])),
        "resource_profile": "upload",
        "model": dict(protocol["model"]),
    }


def tensorize(
    records: Sequence[Mapping[str, Any]],
    rewards: np.ndarray | None,
    protocol: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    return _sources()["set_run"]._record_tensors(
        records, rewards, tensor_protocol(protocol)
    )


def subset(
    data: Mapping[str, torch.Tensor], indices: Sequence[int]
) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[index] for key, value in data.items()}


def _inner_split(count: int, seed: int, fraction: float) -> tuple[list[int], list[int]]:
    if count < 4:
        raise ValueError("at least four samples are required")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(count)
    validation_count = min(
        count - 2, max(1, int(round(count * float(fraction))))
    )
    return order[validation_count:].tolist(), order[:validation_count].tolist()


def _normalize(
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


def _model_inputs(data: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: data[key]
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


def _predict_static(
    model: torch.nn.Module,
    data: Mapping[str, torch.Tensor],
    *,
    reward_mean: float,
    reward_std: float,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(data["boxes"]), int(batch_size)):
            stop = min(len(data["boxes"]), start + int(batch_size))
            batch = {key: value[start:stop] for key, value in data.items()}
            output.append(model(**_model_inputs(batch)).cpu().numpy())
    values = np.concatenate(output).astype(np.float64)
    values = values * float(reward_std) + float(reward_mean)
    values[data["action_type"].numpy() == 0] = 0.0
    values[~data["action_mask"].numpy()] = -1e9
    return values


def _fit_batched(
    *,
    trainer: str,
    data: Mapping[str, torch.Tensor],
    protocol: Mapping[str, Any],
    category_count: int,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    source = _sources()
    model_config = protocol["model"]
    cost_free = (
        source["afpd"].strip_cost_inputs(data)
        if trainer in {"current_action", "afpd"}
        else source["rel"].strip_cost_inputs(data)
    )
    train, validation = _inner_split(
        len(cost_free["boxes"]),
        seed,
        float(model_config["validation_fraction"]),
    )
    normalized, reward_mean, reward_std = _normalize(cost_free, train)
    torch.manual_seed(int(seed))
    if trainer in {"current_action", "afpd"}:
        method = "CurrentActionValue" if trainer == "current_action" else "BoundaryPathDistill"
        model = source["afpd"].build_model(
            category_count=category_count,
            scene_dim=int(source["set_core"].SCENE_DIM),
            action_dim=int(source["set_core"].ACTION_DIM),
            config=model_config,
        )
    elif trainer in {"global", "relset", "relaction"}:
        method = {
            "global": "GlobalRank",
            "relset": "RelSetValue",
            "relaction": "RelActionRank",
        }[trainer]
        model = source["rel"].build_model(
            method,
            category_count=category_count,
            scene_dim=int(source["set_core"].SCENE_DIM),
            action_dim=int(source["set_core"].ACTION_DIM),
            config=model_config,
        )
    elif trainer == "factorized":
        method = "FactorizedGlobalActionRank"
        model = source["factor"].FactorizedGlobalActionRankRouter(
            category_count=category_count,
            scene_dim=int(source["set_core"].SCENE_DIM),
            action_dim=int(source["set_core"].ACTION_DIM),
            config=model_config,
        )
    else:
        raise ValueError(trainer)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, best_loss, stale = 1, math.inf, 0
    history = []

    def epoch(indices: Sequence[int], training: bool, epoch_seed: int) -> dict[str, float]:
        model.train(training)
        ordered = torch.as_tensor(indices, dtype=torch.long)
        if training:
            generator = torch.Generator().manual_seed(int(epoch_seed))
            ordered = ordered[torch.randperm(len(ordered), generator=generator)]
        totals: Counter[str] = Counter()
        samples = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for start in range(0, len(ordered), int(model_config["batch_size"])):
                index = ordered[start : start + int(model_config["batch_size"])]
                batch = {key: value[index] for key, value in normalized.items()}
                scores = model(**_model_inputs(batch))
                if trainer in {"current_action", "afpd"}:
                    result = source["afpd"].training_objective(
                        model,
                        method,
                        batch,
                        reward_mean=reward_mean,
                        reward_std=reward_std,
                        config=model_config,
                    )
                elif trainer in {"global", "relset", "relaction"}:
                    result = source["rel"].action_value_objective(
                        scores,
                        batch["target"],
                        batch["action_mask"],
                        batch["action_type"],
                        cross_image_weight=float(
                            model_config["cross_image_weight"]
                            if method in {"GlobalRank", "RelActionRank"}
                            else 0.0
                        ),
                        reward_weight=float(model_config["reward_loss_weight"]),
                        within_weight=float(model_config["within_action_weight"]),
                        minimum_target_gap=float(model_config["minimum_pair_target_gap"]),
                    )
                else:
                    result = source["factor"].factorized_objective(
                        model,
                        _model_inputs(batch),
                        batch["target"],
                        batch["action_mask"],
                        batch["action_type"],
                        config=model_config,
                    )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    result["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(model_config["gradient_clip"])
                    )
                    optimizer.step()
                count = len(index)
                samples += count
                for key, value in result.items():
                    totals[key] += float(value.detach()) * count
        return {key: value / max(1, samples) for key, value in totals.items()}

    for epoch_index in range(1, int(model_config["epochs"]) + 1):
        fit = epoch(train, True, int(seed) + epoch_index * 1_000_003)
        valid = epoch(validation, False, int(seed))
        value = float(valid["loss"])
        history.append(
            {"epoch": epoch_index, "train_loss": fit["loss"], "validation_loss": value}
        )
        if value < best_loss - 1e-6:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch, best_loss, stale = epoch_index, value, 0
        else:
            stale += 1
        if stale >= int(model_config["patience"]):
            break
    model.load_state_dict(best_state)
    return model, {
        "trainer": trainer,
        "method": method,
        "selected_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "history": history,
    }


def _frontier_data(
    records: Sequence[Mapping[str, Any]],
    data: Mapping[str, torch.Tensor],
    rewards: np.ndarray,
    protocol: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    source = _sources()
    scale = float(np.mean([record["full_payload_bytes"] for record in records]))
    reward_map = {
        int(record["image_id"]): {
            str(action["action_id"]): float(rewards[row, column])
            for column, action in enumerate(record["actions"])
            if str(action["action_id"]) != "edge"
        }
        for row, record in enumerate(records)
    }
    raw, cost = source["afpd"].reward_and_cost_tensors(
        list(records),
        reward_map,
        max_actions=int(protocol["model"]["max_actions"]),
        cost_scale_bytes=scale,
    )
    output = source["afpd"].strip_cost_inputs(data)
    output["raw_target"] = raw.to(torch.float32)
    output["target"] = raw.to(torch.float32)
    output["cost_ratio"] = cost.to(torch.float32)
    output["cost_scale_bytes"] = torch.full(
        (len(records),), scale, dtype=torch.float32
    )
    return output


def _fit_occupancy(
    *,
    cost_aware: bool,
    data: Mapping[str, torch.Tensor],
    protocol: Mapping[str, Any],
    category_count: int,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    source = _sources()
    model_config = protocol["model"]
    prices, price_mask = source["afpd_core"].frontier_probe_prices(
        data["raw_target"],
        data["cost_ratio"],
        data["action_mask"],
        relative_offset=float(model_config["frontier_relative_offset"]),
        maximum_price=float(model_config["frontier_maximum_price"]),
    )
    targets = source["afpd_core"].frontier_target_actions(
        data["raw_target"], data["cost_ratio"], data["action_mask"], prices
    )
    labels = source["occupancy_core"].frontier_occupancy_labels(
        targets, price_mask, data["action_mask"]
    )
    work = dict(data if cost_aware else source["afpd"].strip_cost_inputs(data))
    work["occupancy_target"] = labels
    train, validation = _inner_split(
        len(work["boxes"]), seed, float(model_config["validation_fraction"])
    )
    torch.manual_seed(int(seed))
    model = source["occupancy"].build_model(
        category_count=category_count,
        scene_dim=int(source["set_core"].SCENE_DIM),
        action_dim=int(source["set_core"].ACTION_DIM),
        config=model_config,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    best_state, best_epoch, best_loss, stale = copy.deepcopy(model.state_dict()), 1, math.inf, 0
    history = []

    def epoch(indices: Sequence[int], training: bool, epoch_seed: int) -> dict[str, float]:
        model.train(training)
        order = torch.as_tensor(indices, dtype=torch.long)
        if training:
            generator = torch.Generator().manual_seed(int(epoch_seed))
            order = order[torch.randperm(len(order), generator=generator)]
        total_loss = total_accuracy = total_roi = 0.0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for start in range(0, len(order), int(model_config["batch_size"])):
                index = order[start : start + int(model_config["batch_size"])]
                batch = {key: value[index] for key, value in work.items()}
                roi_count = int(
                    ((batch["action_type"] == 2) & batch["action_mask"]).sum()
                )
                if not roi_count:
                    continue
                result = source["occupancy"].occupancy_objective(model, batch)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    result["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                total_loss += float(result["loss"].detach()) * roi_count
                total_accuracy += float(result["accuracy"].detach()) * roi_count
                total_roi += roi_count
        return {
            "loss": total_loss / max(1.0, total_roi),
            "accuracy": total_accuracy / max(1.0, total_roi),
        }

    for epoch_index in range(1, int(model_config["epochs"]) + 1):
        fit = epoch(train, True, int(seed) + epoch_index * 1_000_003)
        valid = epoch(validation, False, int(seed))
        history.append({"epoch": epoch_index, "train": fit, "validation": valid})
        if valid["loss"] < best_loss - 1e-6:
            best_state, best_epoch, best_loss, stale = (
                copy.deepcopy(model.state_dict()),
                epoch_index,
                valid["loss"],
                0,
            )
        else:
            stale += 1
        if stale >= int(model_config["patience"]):
            break
    model.load_state_dict(best_state)
    return model, {
        "trainer": "occupancy_aware" if cost_aware else "occupancy_free",
        "selected_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "positive_roi_actions": int((labels & (data["action_type"] == 2)).sum()),
        "roi_actions": int(((data["action_type"] == 2) & data["action_mask"]).sum()),
        "history": history,
    }


def _predict_occupancy(
    model: torch.nn.Module,
    data: Mapping[str, torch.Tensor],
    protocol: Mapping[str, Any],
    *,
    cost_aware: bool,
) -> np.ndarray:
    source = _sources()
    work = data if cost_aware else source["afpd"].strip_cost_inputs(data)
    model.eval()
    rows = []
    batch_size = int(protocol["model"]["batch_size"])
    with torch.no_grad():
        for start in range(0, len(work["boxes"]), batch_size):
            batch = {
                key: value[start : start + batch_size] for key, value in work.items()
            }
            rows.append(torch.sigmoid(model(**_model_inputs(batch))).numpy())
    output = np.concatenate(rows).astype(np.float64)
    output[(work["action_type"].numpy() != 2) | ~work["action_mask"].numpy()] = 0.0
    return output


def fit_model(
    trainer: str,
    records: Sequence[Mapping[str, Any]],
    rewards: np.ndarray,
    protocol: Mapping[str, Any],
    *,
    category_count: int,
    seed: int,
) -> dict[str, Any]:
    """Fit one method family strictly on the records supplied by the caller."""
    torch.set_num_threads(1)
    source = _sources()
    config = tensor_protocol(protocol)
    data = tensorize(records, rewards, protocol)
    if trainer in {"current_action", "afpd", "occupancy_free", "occupancy_aware"}:
        data = _frontier_data(records, data, rewards, protocol)
    if trainer == "budgetset":
        source["set_run"]._device = lambda: torch.device("cpu")
        checkpoint = source["set_run"]._fit_one_router(
            records,
            data,
            rewards,
            config,
            category_count=category_count,
            seed=seed,
        )
        return {"kind": trainer, "checkpoint": checkpoint}
    if trainer == "value":
        checkpoint = source["value_train"].fit_value_router(
            [int(record["image_id"]) for record in records],
            data,
            config,
            category_count=category_count,
            seed=seed,
            device=torch.device("cpu"),
        )
        return {"kind": trainer, "checkpoint": checkpoint}
    if trainer == "dual":
        checkpoint = source["value_train"].fit_dual_router(
            records,
            data,
            rewards,
            config,
            category_count=category_count,
            seed=seed,
            device=torch.device("cpu"),
        )
        return {"kind": trainer, "checkpoint": checkpoint}
    if trainer == "risk":
        checkpoint = source["risk"].fit_value_risk_router(
            [int(record["image_id"]) for record in records],
            data,
            config,
            category_count=category_count,
            seed=seed,
            device=torch.device("cpu"),
        )
        return {"kind": trainer, "checkpoint": checkpoint}
    if trainer in {"global", "relset", "relaction", "factorized", "current_action", "afpd"}:
        model, metadata = _fit_batched(
            trainer=trainer,
            data=data,
            protocol=protocol,
            category_count=category_count,
            seed=seed,
        )
        return {
            "kind": trainer,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": metadata,
            "category_count": category_count,
        }
    if trainer in {"occupancy_free", "occupancy_aware"}:
        model, metadata = _fit_occupancy(
            cost_aware=trainer == "occupancy_aware",
            data=data,
            protocol=protocol,
            category_count=category_count,
            seed=seed,
        )
        return {
            "kind": trainer,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": metadata,
            "category_count": category_count,
        }
    raise ValueError(trainer)


def predict_model(
    artifact: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    category_count: int,
) -> np.ndarray:
    source = _sources()
    config = tensor_protocol(protocol)
    data = tensorize(records, None, protocol)
    kind = str(artifact["kind"])
    if kind == "budgetset":
        return source["set_run"]._predict_router(
            artifact["checkpoint"], data, config, category_count
        )
    if kind == "value":
        return source["value_train"].predict_value_router(
            artifact["checkpoint"], data, config, category_count=category_count
        )
    if kind == "dual":
        return source["value_train"].predict_dual_value(
            artifact["checkpoint"],
            records,
            data,
            config,
            category_count=category_count,
        )
    if kind == "risk":
        values = source["risk"].predict_value_risk_router(
            artifact["checkpoint"], data, config, category_count=category_count
        )["lower"]
        return np.repeat(values[None, :, :], len(protocol["budgets"]), axis=0)
    if kind in {"current_action", "afpd"}:
        data = source["afpd"].strip_cost_inputs(data)
        method = "CurrentActionValue" if kind == "current_action" else "BoundaryPathDistill"
        model = source["afpd"].build_model(
            category_count=category_count,
            scene_dim=int(source["set_core"].SCENE_DIM),
            action_dim=int(source["set_core"].ACTION_DIM),
            config=protocol["model"],
        )
    elif kind in {"global", "relset", "relaction"}:
        method = {"global": "GlobalRank", "relset": "RelSetValue", "relaction": "RelActionRank"}[kind]
        model = source["rel"].build_model(
            method,
            category_count=category_count,
            scene_dim=int(source["set_core"].SCENE_DIM),
            action_dim=int(source["set_core"].ACTION_DIM),
            config=protocol["model"],
        )
    elif kind == "factorized":
        model = source["factor"].FactorizedGlobalActionRankRouter(
            category_count=category_count,
            scene_dim=int(source["set_core"].SCENE_DIM),
            action_dim=int(source["set_core"].ACTION_DIM),
            config=protocol["model"],
        )
    elif kind in {"occupancy_free", "occupancy_aware"}:
        model = source["occupancy"].build_model(
            category_count=category_count,
            scene_dim=int(source["set_core"].SCENE_DIM),
            action_dim=int(source["set_core"].ACTION_DIM),
            config=protocol["model"],
        )
        model.load_state_dict(artifact["state_dict"])
        return _predict_occupancy(
            model,
            data,
            protocol,
            cost_aware=kind == "occupancy_aware",
        )[None, :, :]
    else:
        raise ValueError(kind)
    model.load_state_dict(artifact["state_dict"])
    values = _predict_static(
        model,
        data,
        reward_mean=float(artifact["metadata"]["reward_mean"]),
        reward_std=float(artifact["metadata"]["reward_std"]),
        batch_size=int(protocol["model"]["batch_size"]),
    )
    return np.repeat(values[None, :, :], len(protocol["budgets"]), axis=0)


def external_features(
    records: Sequence[Mapping[str, Any]],
    category_ids: Sequence[int],
    protocol: Mapping[str, Any],
) -> np.ndarray:
    source = _sources()["external"]
    rows = []
    for record in records:
        predictions = []
        for box in record["edge_boxes"]:
            x1, y1, x2, y2 = map(float, box["xyxy"])
            predictions.append(
                {
                    "category_id": int(category_ids[int(box["cls"])]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(box["conf"]),
                }
            )
        rows.append(
            source.build_output_feature(
                predictions,
                width=int(record["width"]),
                height=int(record["height"]),
                category_ids=category_ids,
                top_k=int(protocol["external_mlp"]["top_k"]),
            )
        )
    return np.asarray(rows, dtype=np.float64)


def fit_external(
    method: str,
    features: np.ndarray,
    targets: np.ndarray,
    protocol: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    source = _sources()["external"]
    config = dict(protocol["external_mlp"])
    config["seed"] = int(seed)
    raw = np.asarray(targets, dtype=np.float64)
    target = source.empirical_cdf(raw, raw) if method.startswith("MORIC") else raw
    checkpoint = source.fit_reward_mlp(
        features, target, np.ones_like(target), config
    )
    return {
        "kind": "external",
        "method": str(method),
        "checkpoint": checkpoint,
        "cdf_reference": np.sort(raw) if method.startswith("MORIC") else None,
    }


def predict_external(artifact: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    values = _sources()["external"].predict_reward_mlp(
        artifact["checkpoint"], features
    )
    return np.asarray(values, dtype=np.float64)
