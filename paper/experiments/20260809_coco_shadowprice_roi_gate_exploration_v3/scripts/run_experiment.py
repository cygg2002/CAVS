#!/usr/bin/env python3
"""Run the fit-OOF shadow-price-conditioned ROI-gate diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
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
V1_ROOT = REPO / PROTOCOL["source_v1"]
V2_ROOT = REPO / PROTOCOL["source_v2"]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load("price_gate_runner_core", ROOT / "scripts/core.py")
TRAINING = _load("price_gate_runner_training", ROOT / "scripts/training.py")
V2 = _load("price_gate_v2_runner", V2_ROOT / "scripts/run_experiment.py")
V1 = V2.V1
CORE.validate_protocol(PROTOCOL)


def run_preflight() -> dict[str, Any]:
    completion = RUNS / "preflight/completion.json"
    output = RUNS / "preflight/report.json"
    if completion.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    required = [
        V1_ROOT / "runs/confirm/evaluation_completion.json",
        V1_ROOT / "runs/confirm/scores.pkl",
        V2_ROOT / "runs/oof/completion.json",
        V2_ROOT / "runs/oof/probabilities.pkl",
        V2_ROOT / "runs/evaluation/completion.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete frozen source: {missing}")
    v1_report = json.loads(
        (V1_ROOT / "runs/confirm/evaluation_report.json").read_text(encoding="utf-8")
    )
    if not bool(v1_report["gate"]["passed"]):
        raise RuntimeError("The frozen AFPD source did not pass")
    data = V1.REL.load_role("fit")
    image_ids = [int(sample["image_id"]) for sample in data["samples"]]
    if len(image_ids) != 9678 or len(image_ids) != len(set(image_ids)):
        raise RuntimeError("Canonical fit population changed")
    folds = json.loads(
        (V1.REL_ROOT / "runs/preflight/folds.json").read_text(encoding="utf-8")
    )["folds"]
    if len(folds) != 5 or {int(x) for fold in folds for x in fold} != set(image_ids):
        raise RuntimeError("Frozen OOF folds changed")
    sources = [
        PROTOCOL_PATH,
        SCRIPT,
        ROOT / "scripts/core.py",
        ROOT / "scripts/training.py",
        *required,
    ]
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "passed": True,
        "fit_images": len(image_ids),
        "fold_sizes": list(map(len, folds)),
        "source_hashes": {str(path): V2.file_sha256(path) for path in sources},
        "training_prices": list(map(float, PROTOCOL["training_prices"])),
        "detector_inference_required": False,
        "gpu_required": False,
        "calibration_accessed": False,
        "val2017_accessed": False,
        "final_accessed": False,
    }
    V2.atomic_json(output, report)
    V2.atomic_json(
        completion, {"complete": True, "report_sha256": V2.file_sha256(output)}
    )
    return report


def _build_data() -> tuple[
    list[dict[str, Any]],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    records = V1._records_for_space("fit", "OQR")
    rewards = V1._reward_map_for_space("fit", "OQR")
    scale = float(np.mean([record["full_payload_bytes"] for record in records]))
    raw_reward, cost_ratio = V1.TRAINING.reward_and_cost_tensors(
        records,
        rewards,
        max_actions=int(PROTOCOL["model"]["max_actions"]),
        cost_scale_bytes=scale,
    )
    tensor_protocol = {
        "max_boxes": int(PROTOCOL["model"]["max_boxes"]),
        "max_actions": int(PROTOCOL["model"]["max_actions"]),
        "deploy_confidence": float(PROTOCOL["model"]["deploy_confidence"]),
    }
    raw = V1.REL._record_tensors(records, None, tensor_protocol)
    prices = torch.as_tensor(PROTOCOL["training_prices"], dtype=torch.float64)
    roi = raw["action_mask"].to(torch.bool) & (raw["action_type"] == 2)
    available = torch.nonzero(roi.any(dim=1)).flatten()
    if not bool((roi[available].sum(dim=1) == 1).all()):
        raise RuntimeError("Canonical OQR rows must contain at most one ROI")
    roi_indices = roi.to(torch.long).argmax(dim=1)
    targets = CORE.roi_optimal_by_price(
        raw_reward,
        cost_ratio,
        raw["action_mask"],
        prices,
        roi_index=roi_indices,
    )
    return records, raw, raw_reward, cost_ratio, targets, scale


def _index_data(
    data: Mapping[str, torch.Tensor], indices: Sequence[int] | torch.Tensor
) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[index] for key, value in data.items()}


def _pair_epoch(
    model: torch.nn.Module,
    data: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    image_rows: Sequence[int],
    *,
    optimizer: torch.optim.Optimizer | None,
    seed: int,
) -> dict[str, float]:
    price_grid = torch.as_tensor(PROTOCOL["training_prices"], dtype=torch.float32)
    price_count = len(price_grid)
    rows = torch.as_tensor(image_rows, dtype=torch.long)
    pair_codes = (
        rows[:, None] * price_count
        + torch.arange(price_count, dtype=torch.long)[None, :]
    ).reshape(-1)
    model.train(optimizer is not None)
    if optimizer is not None:
        generator = torch.Generator().manual_seed(int(seed))
        pair_codes = pair_codes[torch.randperm(len(pair_codes), generator=generator)]
    totals: Counter[str] = Counter()
    total = 0
    batch_size = int(PROTOCOL["model"]["batch_size"])
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for start in range(0, len(pair_codes), batch_size):
            codes = pair_codes[start : start + batch_size]
            image_index = torch.div(codes, price_count, rounding_mode="floor")
            price_index = codes.remainder(price_count)
            result = TRAINING.conditional_roi_objective(
                model,
                _index_data(data, image_index),
                price_grid[price_index],
                targets[image_index, price_index],
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(PROTOCOL["model"]["gradient_clip"])
                )
                optimizer.step()
            count = len(codes)
            total += count
            for key, value in result.items():
                totals[key] += float(value.detach()) * count
    return {key: value / max(1, total) for key, value in totals.items()}


def _fit(
    data: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    image_ids: Sequence[int],
    train_rows: Sequence[int],
    *,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    roi = data["action_mask"].to(torch.bool) & (data["action_type"] == 2)
    train_rows = [int(row) for row in train_rows if bool(roi[row].any())]
    train_ids = [int(image_ids[row]) for row in train_rows]
    inner_ids, validation_ids = V1.REL.deterministic_validation_split(
        train_ids,
        seed=int(seed),
        fraction=float(PROTOCOL["model"]["validation_fraction"]),
    )
    by_id = {int(image_id): index for index, image_id in enumerate(image_ids)}
    inner = [by_id[image_id] for image_id in inner_ids]
    validation = [by_id[image_id] for image_id in validation_ids]
    torch.manual_seed(int(seed))
    model = TRAINING.build_model(
        category_count=len(V1.REL.category_ids("fit")),
        scene_dim=int(V1.REL.SCENE_DIM),
        action_dim=int(V1.REL.ACTION_DIM),
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
        train = _pair_epoch(
            model,
            data,
            targets,
            inner,
            optimizer=optimizer,
            seed=int(seed) + epoch * 1_000_003,
        )
        validation_metrics = _pair_epoch(
            model,
            data,
            targets,
            validation,
            optimizer=None,
            seed=int(seed),
        )
        loss = float(validation_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train["loss"]),
                "validation_loss": loss,
                "validation_accuracy": float(validation_metrics["accuracy"]),
            }
        )
        if loss < best_loss - 1e-6:
            best_epoch, best_loss, stale = epoch, loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= int(PROTOCOL["model"]["patience"]):
            break
    model.load_state_dict(best_state)
    return model, {
        "selected_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "inner_train_images": len(inner),
        "validation_images": len(validation),
        "history": history,
    }


def _predict_at_prices(
    model: torch.nn.Module,
    data: Mapping[str, torch.Tensor],
    rows: Sequence[int],
    prices: Sequence[float],
) -> np.ndarray:
    model.eval()
    image_rows = torch.as_tensor(rows, dtype=torch.long)
    price_values = torch.as_tensor(prices, dtype=torch.float32)
    outputs = []
    batch_size = int(PROTOCOL["model"]["batch_size"])
    with torch.no_grad():
        for price in price_values:
            price_output = []
            for start in range(0, len(image_rows), batch_size):
                indices = image_rows[start : start + batch_size]
                batch = _index_data(data, indices)
                logits = model(
                    **TRAINING.model_inputs(
                        batch, price.expand(len(indices))
                    )
                )
                roi = batch["action_mask"].to(torch.bool) & (batch["action_type"] == 2)
                if not bool((roi.sum(dim=1) == 1).all()):
                    raise RuntimeError("ROI prediction rows require exactly one ROI")
                price_output.append(torch.sigmoid(logits[roi]).cpu().numpy())
            outputs.append(np.concatenate(price_output))
    values = np.stack(outputs, axis=1).astype(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("Conditional ROI probability contains non-finite values")
    return values


def _frozen_inference_prices(scale: float) -> dict[str, float]:
    with (V1_ROOT / "runs/confirm/scores.pkl").open("rb") as handle:
        score_artifact = pickle.load(handle)
    ensemble = np.mean(
        list(score_artifact["scores"]["OQR/BoundaryPathDistill"].values()), axis=0
    )
    records = V1._records_for_space("fit", "OQR")
    score_map = V1._score_map(records, ensemble)
    data = V1.REL.load_role("fit")
    total_original = sum(
        int(sample["payload_bytes"]["original"]) for sample in data["samples"]
    )
    values = {"fixed_call": 0.0}
    for budget in PROTOCOL["budgets"]:
        scheduled = V1.ACTION.schedule_fixed_byte(
            data["samples"],
            score_map,
            capacity_bytes=float(budget) * total_original,
        )
        values[f"fixed_byte:{float(budget):.2f}"] = float(scheduled["lambda"]) * scale
    return values


_TRAIN_STATE: dict[str, Any] = {}


def _train_job(fold_index: int) -> dict[str, Any]:
    fold_index = int(fold_index)
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    root = RUNS / f"oof/jobs/fold_{fold_index}"
    completion = root / "completion.json"
    if completion.is_file():
        return json.loads((root / "report.json").read_text(encoding="utf-8"))
    data = _TRAIN_STATE["data"]
    targets = _TRAIN_STATE["targets"]
    image_ids = _TRAIN_STATE["image_ids"]
    heldout_ids = set(map(int, _TRAIN_STATE["folds"][fold_index]))
    roi = data["action_mask"].to(torch.bool) & (data["action_type"] == 2)
    heldout = [
        index
        for index, image_id in enumerate(image_ids)
        if int(image_id) in heldout_ids and bool(roi[index].any())
    ]
    train = [
        index
        for index, image_id in enumerate(image_ids)
        if int(image_id) not in heldout_ids and bool(roi[index].any())
    ]
    seed = int(PROTOCOL["seed"]) + fold_index * 10_007
    started = time.time()
    model, fit = _fit(data, targets, image_ids, train, seed=seed)
    inference_prices = list(_TRAIN_STATE["inference_prices"].values())
    probabilities = _predict_at_prices(model, data, heldout, inference_prices)
    probability_path = root / "probabilities.npz"
    checkpoint_path = root / "checkpoint.pt"
    V2.atomic_npz(
        probability_path,
        image_ids=np.asarray([image_ids[index] for index in heldout], dtype=np.int64),
        probabilities=probabilities,
    )
    V2.atomic_torch(
        checkpoint_path,
        {"seed": seed, "model_state": model.state_dict(), "fit": fit},
    )
    report = {
        "fold": fold_index,
        "seed": seed,
        "train_roi_images": len(train),
        "heldout_roi_images": len(heldout),
        "selected_epoch": int(fit["selected_epoch"]),
        "best_validation_loss": float(fit["best_validation_loss"]),
        "elapsed_seconds": time.time() - started,
        "probability_sha256": V2.file_sha256(probability_path),
        "checkpoint_sha256": V2.file_sha256(checkpoint_path),
    }
    V2.atomic_json(root / "report.json", report)
    V2.atomic_json(
        completion, {"complete": True, "probability_sha256": report["probability_sha256"]}
    )
    return report


def run_oof() -> dict[str, Any]:
    completion = RUNS / "oof/completion.json"
    output = RUNS / "oof/report.json"
    if completion.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    run_preflight()
    records, data, raw_reward, cost_ratio, targets, scale = _build_data()
    image_ids = [int(record["image_id"]) for record in records]
    folds = json.loads(
        (V1.REL_ROOT / "runs/preflight/folds.json").read_text(encoding="utf-8")
    )["folds"]
    inference_prices = _frozen_inference_prices(scale)
    _TRAIN_STATE.clear()
    _TRAIN_STATE.update(
        {
            "data": data,
            "targets": targets,
            "image_ids": image_ids,
            "folds": folds,
            "inference_prices": inference_prices,
        }
    )
    started = time.time()
    workers = min(int(PROTOCOL["cpu"]["training_workers"]), len(folds))
    with mp.get_context("fork").Pool(processes=workers) as pool:
        reports = list(pool.imap_unordered(_train_job, range(len(folds)), chunksize=1))
    roi_rows = torch.nonzero(
        (data["action_mask"].to(torch.bool) & (data["action_type"] == 2)).any(dim=1)
    ).flatten()
    roi_ids = {image_ids[int(row)] for row in roi_rows}
    price_names = list(inference_prices)
    probabilities: dict[str, dict[int, float]] = {name: {} for name in price_names}
    for fold_index in range(len(folds)):
        path = RUNS / f"oof/jobs/fold_{fold_index}/probabilities.npz"
        with np.load(path, allow_pickle=False) as artifact:
            for row, image_id in enumerate(artifact["image_ids"]):
                for column, name in enumerate(price_names):
                    probabilities[name][int(image_id)] = float(
                        artifact["probabilities"][row, column]
                    )
    if any(set(values) != roi_ids for values in probabilities.values()):
        raise RuntimeError("Conditional ROI OOF probabilities are incomplete")
    exact_prices = torch.as_tensor(list(inference_prices.values()), dtype=torch.float64)
    exact_targets = CORE.roi_optimal_by_price(
        raw_reward,
        cost_ratio,
        data["action_mask"],
        exact_prices,
        roi_index=(
            data["action_mask"].to(torch.bool) & (data["action_type"] == 2)
        ).to(torch.long).argmax(dim=1),
    )
    target_maps = {
        name: {image_ids[int(row)]: bool(exact_targets[int(row), column]) for row in roi_rows}
        for column, name in enumerate(price_names)
    }
    artifact_path = RUNS / "oof/probabilities.pkl"
    V2.atomic_pickle(
        artifact_path,
        {
            "probabilities": probabilities,
            "targets": target_maps,
            "inference_prices": inference_prices,
            "training_prices": list(map(float, PROTOCOL["training_prices"])),
            "threshold": float(PROTOCOL["probability_threshold"]),
            "shared_original_byte_scale": scale,
        },
    )
    prevalence = {
        name: sum(values.values()) / max(1, len(values))
        for name, values in target_maps.items()
    }
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "jobs": len(folds),
        "completed_jobs": len(reports),
        "workers": workers,
        "roi_images": len(roi_ids),
        "inference_prices": inference_prices,
        "target_prevalence": prevalence,
        "elapsed_seconds": time.time() - started,
        "probability_sha256": V2.file_sha256(artifact_path),
        "fit_oof_only": True,
        "calibration_accessed": False,
        "val2017_accessed": False,
        "final_accessed": False,
    }
    V2.atomic_json(output, report)
    V2.atomic_json(
        completion,
        {"complete": True, "probability_sha256": report["probability_sha256"]},
    )
    _TRAIN_STATE.clear()
    return report


def _classification_metrics(
    probabilities: Mapping[int, float], targets: Mapping[int, bool]
) -> dict[str, float | int]:
    threshold = float(PROTOCOL["probability_threshold"])
    predicted = {image_id for image_id, value in probabilities.items() if value >= threshold}
    positive = {image_id for image_id, value in targets.items() if value}
    common = predicted & positive
    correct = sum((image_id in predicted) == bool(targets[image_id]) for image_id in targets)
    return {
        "predicted_positive": len(predicted),
        "true_positive": len(positive),
        "precision": len(common) / max(1, len(predicted)),
        "recall": len(common) / max(1, len(positive)),
        "accuracy": correct / max(1, len(targets)),
    }


def run_evaluate() -> dict[str, Any]:
    completion = RUNS / "evaluation/completion.json"
    output = RUNS / "evaluation/report.json"
    if completion.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    oof = run_oof()
    probability_path = RUNS / "oof/probabilities.pkl"
    if V2.file_sha256(probability_path) != oof["probability_sha256"]:
        raise RuntimeError("OOF probability artifact changed")
    with probability_path.open("rb") as handle:
        conditional = pickle.load(handle)
    with (V2_ROOT / "runs/oof/probabilities.pkl").open("rb") as handle:
        any_frontier = pickle.load(handle)
    with (V1_ROOT / "runs/confirm/scores.pkl").open("rb") as handle:
        score_artifact = pickle.load(handle)
    ensemble = np.mean(
        list(score_artifact["scores"]["OQR/BoundaryPathDistill"].values()), axis=0
    )
    records = V1._records_for_space("fit", "OQR")
    ungated = V1._score_map(records, ensemble)
    data = V1.REL.load_role("fit")
    samples = data["samples"]
    rewards = V1._reward_map_for_space("fit", "OQR")
    total_original = sum(int(sample["payload_bytes"]["original"]) for sample in samples)
    tasks = []
    work_point_meta: dict[str, dict[str, Any]] = {}
    for budget in PROTOCOL["budgets"]:
        capacity = float(budget) * total_original
        for mode in PROTOCOL["modes"]:
            price_name = "fixed_call" if mode == "fixed_call" else f"fixed_byte:{float(budget):.2f}"
            ungated_schedule = (
                V1.ACTION.schedule_fixed_byte(samples, ungated, capacity_bytes=capacity)
                if mode == "fixed_byte"
                else V1.ACTION.schedule_fixed_call(samples, ungated, budget=float(budget))
            )
            policies = {
                "Ungated-AFPD": ungated,
                "CostAware-AnyFrontier-v2": CORE.gate_roi_scores(
                    ungated,
                    any_frontier["probabilities"]["CostAware-FrontierGate"],
                    threshold=float(PROTOCOL["probability_threshold"]),
                ),
                "PriceConditional-ROIGate": CORE.gate_roi_scores(
                    ungated,
                    conditional["probabilities"][price_name],
                    threshold=float(PROTOCOL["probability_threshold"]),
                ),
                "GT-LearnedPrice-Gate": CORE.gate_roi_scores(
                    ungated,
                    {image_id: float(value) for image_id, value in conditional["targets"][price_name].items()},
                    threshold=float(PROTOCOL["probability_threshold"]),
                ),
            }
            teacher = (
                V1.ACTION.schedule_fixed_byte(samples, rewards, capacity_bytes=capacity)
                if mode == "fixed_byte"
                else V1.ACTION.schedule_fixed_call(samples, rewards, budget=float(budget))
            )
            teacher_roi = [
                int(image_id)
                for image_id, action in teacher["selection"].items()
                if action == "roi"
            ]
            for policy, score_map in policies.items():
                scheduled = (
                    ungated_schedule
                    if policy == "Ungated-AFPD"
                    else V1.ACTION.schedule_fixed_byte(samples, score_map, capacity_bytes=capacity)
                    if mode == "fixed_byte"
                    else V1.ACTION.schedule_fixed_call(samples, score_map, budget=float(budget))
                )
                tasks.append(
                    {
                        "policy": policy,
                        "mode": mode,
                        "budget": float(budget),
                        "capacity_bytes": capacity,
                        "selection": scheduled["selection"],
                        "teacher_roi": teacher_roi,
                    }
                )
            work_point_meta[f"{mode}:{float(budget):.2f}"] = {
                "price_name": price_name,
                "normalized_price": float(conditional["inference_prices"][price_name]),
                "ungated_lambda_absolute": float(ungated_schedule.get("lambda", 0.0)),
                "classification": _classification_metrics(
                    conditional["probabilities"][price_name],
                    conditional["targets"][price_name],
                ),
            }
    V2._EVAL_STATE.clear()
    V2._EVAL_STATE.update({"data": data})
    workers = min(int(PROTOCOL["cpu"]["evaluation_workers"]), len(tasks))
    started = time.time()
    with mp.get_context("fork").Pool(processes=workers) as pool:
        rows = list(pool.imap_unordered(V2._eval_task, tasks, chunksize=1))
    if len(rows) != 24 or not all(row["capacity_ok"] for row in rows):
        raise RuntimeError("Evaluation matrix is invalid")
    order = [
        "Ungated-AFPD",
        "CostAware-AnyFrontier-v2",
        "PriceConditional-ROIGate",
        "GT-LearnedPrice-Gate",
    ]
    rows.sort(
        key=lambda row: (
            order.index(str(row["policy"])),
            PROTOCOL["modes"].index(str(row["mode"])),
            float(row["budget"]),
        )
    )
    comparisons = {}
    for method in order[1:]:
        points = []
        for mode in PROTOCOL["modes"]:
            for budget in PROTOCOL["budgets"]:
                baseline = next(
                    row for row in rows
                    if row["policy"] == "Ungated-AFPD"
                    and row["mode"] == mode
                    and math.isclose(float(row["budget"]), float(budget))
                )
                candidate = next(
                    row for row in rows
                    if row["policy"] == method
                    and row["mode"] == mode
                    and math.isclose(float(row["budget"]), float(budget))
                )
                points.append(
                    {
                        "mode": mode,
                        "budget": float(budget),
                        "delta_ap": float(candidate["official_ap50_95"])
                        - float(baseline["official_ap50_95"]),
                        "capacity_utilization": float(candidate["capacity_utilization"]),
                    }
                )
        comparisons[method] = {
            "work_points": points,
            "gate": V1.CORE.evaluate_screen_gate(points, PROTOCOL["gate"]),
        }
    candidate_gate = comparisons["PriceConditional-ROIGate"]["gate"]
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": "fit_oof",
        "rows": len(rows),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "work_point_meta": work_point_meta,
        "comparisons": comparisons,
        "passed": bool(candidate_gate["passed"]),
        "promoted_method": "PriceConditional-ROIGate" if candidate_gate["passed"] else None,
        "threshold": float(PROTOCOL["probability_threshold"]),
        "threshold_selected_on_oof": False,
        "calibration_accessed": False,
        "val2017_accessed": False,
        "final_accessed": False,
    }
    result_path = RUNS / "evaluation/results.csv"
    V2.atomic_csv(result_path, rows)
    V2.atomic_json(output, report)
    V2.atomic_json(
        completion,
        {
            "complete": True,
            "results_sha256": V2.file_sha256(result_path),
            "report_sha256": V2.file_sha256(output),
        },
    )
    V2._EVAL_STATE.clear()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("preflight", "oof", "evaluate", "all"), default="all"
    )
    args = parser.parse_args()
    if args.stage in {"preflight", "all"}:
        print(json.dumps(run_preflight(), indent=2), flush=True)
    if args.stage in {"oof", "all"}:
        print(json.dumps(run_oof(), indent=2), flush=True)
    if args.stage in {"evaluate", "all"}:
        print(json.dumps(run_evaluate(), indent=2), flush=True)


if __name__ == "__main__":
    main()
