#!/usr/bin/env python3
"""Run dual-conditioned ROI frontier-residual learning on canonical fit OOF."""

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
V3_ROOT = REPO / PROTOCOL["source_v3"]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load("residual_runner_core", ROOT / "scripts/core.py")
TRAINING = _load("residual_runner_training", ROOT / "scripts/training.py")
V3 = _load("residual_v3_runner", V3_ROOT / "scripts/run_experiment.py")
V2, V1 = V3.V2, V3.V1
CORE.validate_protocol(PROTOCOL)


def run_preflight() -> dict[str, Any]:
    completion = RUNS / "preflight/completion.json"
    output = RUNS / "preflight/report.json"
    if completion.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    required = [
        V1_ROOT / "runs/confirm/evaluation_completion.json",
        V1_ROOT / "runs/confirm/scores.pkl",
        V2_ROOT / "runs/oof/probabilities.pkl",
        V3_ROOT / "runs/evaluation/completion.json",
        V3_ROOT / "runs/oof/probabilities.pkl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete frozen source: {missing}")
    v3_report = json.loads(
        (V3_ROOT / "runs/evaluation/report.json").read_text(encoding="utf-8")
    )
    if bool(v3_report["passed"]):
        raise RuntimeError("v4 is only justified after the frozen v3 gate failed")
    data = V1.REL.load_role("fit")
    image_ids = [int(sample["image_id"]) for sample in data["samples"]]
    folds = json.loads(
        (V1.REL_ROOT / "runs/preflight/folds.json").read_text(encoding="utf-8")
    )["folds"]
    if len(image_ids) != 9678 or len(set(image_ids)) != 9678:
        raise RuntimeError("Canonical fit population changed")
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


def _frozen_value_tensor(
    records: Sequence[Mapping[str, Any]],
    score_map: Mapping[int, Mapping[str, float]],
    *,
    max_actions: int,
) -> torch.Tensor:
    values = torch.zeros((len(records), int(max_actions)), dtype=torch.float64)
    for row, record in enumerate(records):
        image_id = int(record["image_id"])
        for column, action in enumerate(record["actions"]):
            action_id = str(action["action_id"])
            if action_id != "edge":
                values[row, column] = float(score_map[image_id][action_id])
    return values


def _roi_indices(data: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    roi = data["action_mask"].to(torch.bool) & (data["action_type"] == 2)
    available = roi.any(dim=1)
    if not bool((roi[available].sum(dim=1) == 1).all()):
        raise RuntimeError("Every available row must contain exactly one ROI")
    return roi.to(torch.long).argmax(dim=1), available


def _state_grid(
    values: torch.Tensor,
    costs: torch.Tensor,
    action_mask: torch.Tensor,
    roi_indices: torch.Tensor,
    prices: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    margins, features = [], []
    for price in prices:
        state = CORE.frontier_state(
            values,
            costs,
            action_mask,
            torch.full((len(values),), float(price), dtype=values.dtype),
            roi_index=roi_indices,
        )
        margins.append(state["margin"])
        features.append(
            torch.stack(
                (
                    state["margin"],
                    state["cost_gap"],
                    state["competitor_index"].to(values.dtype)
                    / max(1, values.shape[1] - 1),
                ),
                dim=1,
            )
        )
    return torch.stack(margins, dim=1), torch.stack(features, dim=1)


def _build_data() -> dict[str, Any]:
    records, data, true_reward, cost_ratio, _, scale = V3._build_data()
    with (V1_ROOT / "runs/confirm/scores.pkl").open("rb") as handle:
        score_artifact = pickle.load(handle)
    ensemble = np.mean(
        list(score_artifact["scores"]["OQR/BoundaryPathDistill"].values()), axis=0
    )
    score_map = V1._score_map(records, ensemble)
    frozen_values = _frozen_value_tensor(
        records, score_map, max_actions=int(PROTOCOL["model"]["max_actions"])
    )
    roi_indices, roi_available = _roi_indices(data)
    training_prices = list(map(float, PROTOCOL["training_prices"]))
    true_margin, _ = _state_grid(
        true_reward, cost_ratio, data["action_mask"], roi_indices, training_prices
    )
    frozen_margin, frontier_features = _state_grid(
        frozen_values, cost_ratio, data["action_mask"], roi_indices, training_prices
    )
    inference_prices = V3._frozen_inference_prices(scale)
    inference_margin, inference_features = _state_grid(
        frozen_values,
        cost_ratio,
        data["action_mask"],
        roi_indices,
        list(inference_prices.values()),
    )
    true_inference_margin, _ = _state_grid(
        true_reward,
        cost_ratio,
        data["action_mask"],
        roi_indices,
        list(inference_prices.values()),
    )
    return {
        "records": records,
        "data": data,
        "image_ids": [int(record["image_id"]) for record in records],
        "score_map": score_map,
        "frozen_values": frozen_values,
        "cost_ratio": cost_ratio,
        "roi_indices": roi_indices,
        "roi_available": roi_available,
        "training_true_margin": true_margin,
        "training_frozen_margin": frozen_margin,
        "training_features": frontier_features,
        "inference_prices": inference_prices,
        "inference_frozen_margin": inference_margin,
        "inference_true_margin": true_inference_margin,
        "inference_features": inference_features,
        "scale": scale,
    }


def _index_data(
    data: Mapping[str, torch.Tensor], indices: Sequence[int] | torch.Tensor
) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[index] for key, value in data.items()}


def _pair_epoch(
    model: torch.nn.Module,
    bundle: Mapping[str, Any],
    image_rows: Sequence[int],
    *,
    optimizer: torch.optim.Optimizer | None,
    seed: int,
) -> dict[str, float]:
    price_grid = torch.as_tensor(PROTOCOL["training_prices"], dtype=torch.float32)
    price_count = len(price_grid)
    rows = torch.as_tensor(image_rows, dtype=torch.long)
    codes = (
        rows[:, None] * price_count
        + torch.arange(price_count, dtype=torch.long)[None, :]
    ).reshape(-1)
    model.train(optimizer is not None)
    if optimizer is not None:
        generator = torch.Generator().manual_seed(int(seed))
        codes = codes[torch.randperm(len(codes), generator=generator)]
    totals: Counter[str] = Counter()
    total = 0
    batch_size = int(PROTOCOL["model"]["batch_size"])
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for start in range(0, len(codes), batch_size):
            batch_codes = codes[start : start + batch_size]
            image_index = torch.div(batch_codes, price_count, rounding_mode="floor")
            price_index = batch_codes.remainder(price_count)
            result = TRAINING.residual_objective(
                model,
                _index_data(bundle["data"], image_index),
                price_grid[price_index],
                bundle["training_features"][image_index, price_index].to(torch.float32),
                bundle["training_frozen_margin"][image_index, price_index],
                bundle["training_true_margin"][image_index, price_index],
                sign_loss_weight=float(PROTOCOL["model"]["sign_loss_weight"]),
                sign_temperature=float(PROTOCOL["model"]["sign_temperature"]),
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(PROTOCOL["model"]["gradient_clip"])
                )
                optimizer.step()
            count = len(batch_codes)
            total += count
            for key, value in result.items():
                totals[key] += float(value.detach()) * count
    return {key: value / max(1, total) for key, value in totals.items()}


def _fit(
    bundle: Mapping[str, Any], train_rows: Sequence[int], *, seed: int
) -> tuple[torch.nn.Module, dict[str, Any]]:
    train_rows = [int(row) for row in train_rows if bool(bundle["roi_available"][row])]
    image_ids = bundle["image_ids"]
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
            bundle,
            inner,
            optimizer=optimizer,
            seed=int(seed) + epoch * 1_000_003,
        )
        validation_metrics = _pair_epoch(
            model, bundle, validation, optimizer=None, seed=int(seed)
        )
        loss = float(validation_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train["loss"]),
                "validation_loss": loss,
                "validation_sign_accuracy": float(validation_metrics["sign_accuracy"]),
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


def _predict_residuals(
    model: torch.nn.Module, bundle: Mapping[str, Any], rows: Sequence[int]
) -> np.ndarray:
    model.eval()
    image_rows = torch.as_tensor(rows, dtype=torch.long)
    prices = torch.as_tensor(list(bundle["inference_prices"].values()), dtype=torch.float32)
    outputs = []
    batch_size = int(PROTOCOL["model"]["batch_size"])
    with torch.no_grad():
        for column, price in enumerate(prices):
            values = []
            for start in range(0, len(image_rows), batch_size):
                indices = image_rows[start : start + batch_size]
                batch = _index_data(bundle["data"], indices)
                scores = model(
                    **TRAINING.augmented_inputs(
                        batch,
                        price.expand(len(indices)),
                        bundle["inference_features"][indices, column].to(torch.float32),
                    )
                )
                roi = batch["action_mask"].to(torch.bool) & (batch["action_type"] == 2)
                if not bool((roi.sum(dim=1) == 1).all()):
                    raise RuntimeError("Residual prediction rows require exactly one ROI")
                values.append(scores[roi].cpu().numpy())
            outputs.append(np.concatenate(values))
    result = np.stack(outputs, axis=1).astype(np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("ROI residual contains non-finite values")
    return result


_TRAIN_STATE: dict[str, Any] = {}


def _train_job(fold_index: int) -> dict[str, Any]:
    fold_index = int(fold_index)
    torch.set_num_threads(int(PROTOCOL["cpu"]["torch_threads_per_worker"]))
    root = RUNS / f"screen/jobs/fold_{fold_index}"
    completion = root / "completion.json"
    if completion.is_file():
        return json.loads((root / "report.json").read_text(encoding="utf-8"))
    bundle = _TRAIN_STATE["bundle"]
    heldout_ids = set(map(int, _TRAIN_STATE["folds"][fold_index]))
    heldout = [
        index for index, image_id in enumerate(bundle["image_ids"])
        if int(image_id) in heldout_ids and bool(bundle["roi_available"][index])
    ]
    train = [
        index for index, image_id in enumerate(bundle["image_ids"])
        if int(image_id) not in heldout_ids and bool(bundle["roi_available"][index])
    ]
    seed = int(PROTOCOL["screen_seed"]) + fold_index * 10_007
    started = time.time()
    model, fit = _fit(bundle, train, seed=seed)
    residuals = _predict_residuals(model, bundle, heldout)
    residual_path = root / "residuals.npz"
    checkpoint_path = root / "checkpoint.pt"
    V2.atomic_npz(
        residual_path,
        image_ids=np.asarray([bundle["image_ids"][index] for index in heldout], dtype=np.int64),
        residuals=residuals,
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
        "residual_sha256": V2.file_sha256(residual_path),
        "checkpoint_sha256": V2.file_sha256(checkpoint_path),
    }
    V2.atomic_json(root / "report.json", report)
    V2.atomic_json(
        completion, {"complete": True, "residual_sha256": report["residual_sha256"]}
    )
    return report


def _regression_metrics(
    predicted: Mapping[int, float], target: Mapping[int, float], frozen: Mapping[int, float]
) -> dict[str, float]:
    ids = sorted(target)
    pred = np.asarray([float(frozen[i]) + float(predicted[i]) for i in ids])
    truth = np.asarray([float(target[i]) for i in ids])
    mae = float(np.mean(np.abs(pred - truth)))
    sign_accuracy = float(np.mean((pred >= 0.0) == (truth >= 0.0)))
    correlation = float(np.corrcoef(pred, truth)[0, 1]) if np.std(pred) > 0 else 0.0
    return {"margin_mae": mae, "sign_accuracy": sign_accuracy, "pearson": correlation}


def run_screen() -> dict[str, Any]:
    completion = RUNS / "screen/completion.json"
    output = RUNS / "screen/report.json"
    if completion.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    run_preflight()
    bundle = _build_data()
    folds = json.loads(
        (V1.REL_ROOT / "runs/preflight/folds.json").read_text(encoding="utf-8")
    )["folds"]
    _TRAIN_STATE.clear()
    _TRAIN_STATE.update({"bundle": bundle, "folds": folds})
    workers = min(int(PROTOCOL["cpu"]["training_workers"]), len(folds))
    started = time.time()
    with mp.get_context("fork").Pool(processes=workers) as pool:
        reports = list(pool.imap_unordered(_train_job, range(len(folds)), chunksize=1))
    roi_rows = torch.nonzero(bundle["roi_available"]).flatten()
    roi_ids = {bundle["image_ids"][int(row)] for row in roi_rows}
    price_names = list(bundle["inference_prices"])
    residual_maps: dict[str, dict[int, float]] = {name: {} for name in price_names}
    for fold_index in range(len(folds)):
        path = RUNS / f"screen/jobs/fold_{fold_index}/residuals.npz"
        with np.load(path, allow_pickle=False) as artifact:
            for row, image_id in enumerate(artifact["image_ids"]):
                for column, name in enumerate(price_names):
                    residual_maps[name][int(image_id)] = float(artifact["residuals"][row, column])
    if any(set(values) != roi_ids for values in residual_maps.values()):
        raise RuntimeError("OOF residual maps are incomplete")
    true_maps, frozen_maps, gt_residual_maps, diagnostics = {}, {}, {}, {}
    for column, name in enumerate(price_names):
        true_maps[name] = {
            bundle["image_ids"][int(row)]: float(bundle["inference_true_margin"][int(row), column])
            for row in roi_rows
        }
        frozen_maps[name] = {
            bundle["image_ids"][int(row)]: float(bundle["inference_frozen_margin"][int(row), column])
            for row in roi_rows
        }
        gt_residual_maps[name] = {
            image_id: true_maps[name][image_id] - frozen_maps[name][image_id]
            for image_id in roi_ids
        }
        diagnostics[name] = _regression_metrics(
            residual_maps[name], true_maps[name], frozen_maps[name]
        )
    artifact_path = RUNS / "screen/residuals.pkl"
    V2.atomic_pickle(
        artifact_path,
        {
            "residuals": residual_maps,
            "gt_residuals": gt_residual_maps,
            "true_margins": true_maps,
            "frozen_margins": frozen_maps,
            "inference_prices": bundle["inference_prices"],
            "shared_original_byte_scale": bundle["scale"],
        },
    )
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "jobs": len(folds),
        "completed_jobs": len(reports),
        "workers": workers,
        "roi_images": len(roi_ids),
        "inference_prices": bundle["inference_prices"],
        "diagnostics": diagnostics,
        "elapsed_seconds": time.time() - started,
        "residual_sha256": V2.file_sha256(artifact_path),
        "fit_oof_only": True,
        "calibration_accessed": False,
        "val2017_accessed": False,
        "final_accessed": False,
    }
    V2.atomic_json(output, report)
    V2.atomic_json(
        completion, {"complete": True, "residual_sha256": report["residual_sha256"]}
    )
    _TRAIN_STATE.clear()
    return report


def run_evaluate() -> dict[str, Any]:
    completion = RUNS / "evaluation/completion.json"
    output = RUNS / "evaluation/report.json"
    if completion.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    screen = run_screen()
    residual_path = RUNS / "screen/residuals.pkl"
    if V2.file_sha256(residual_path) != screen["residual_sha256"]:
        raise RuntimeError("Residual artifact changed")
    with residual_path.open("rb") as handle:
        residual_artifact = pickle.load(handle)
    with (V3_ROOT / "runs/oof/probabilities.pkl").open("rb") as handle:
        v3_artifact = pickle.load(handle)
    with (V2_ROOT / "runs/oof/probabilities.pkl").open("rb") as handle:
        v2_artifact = pickle.load(handle)
    bundle = _build_data()
    ungated = bundle["score_map"]
    data = V1.REL.load_role("fit")
    samples = data["samples"]
    rewards = V1._reward_map_for_space("fit", "OQR")
    total_original = sum(int(sample["payload_bytes"]["original"]) for sample in samples)
    tasks = []
    for budget in PROTOCOL["budgets"]:
        capacity = float(budget) * total_original
        for mode in PROTOCOL["modes"]:
            price_name = "fixed_call" if mode == "fixed_call" else f"fixed_byte:{float(budget):.2f}"
            policies = {
                "Ungated-AFPD": ungated,
                "CostAware-AnyFrontier-v2": V3.CORE.gate_roi_scores(
                    ungated,
                    v2_artifact["probabilities"]["CostAware-FrontierGate"],
                    threshold=0.5,
                ),
                "PriceConditional-Gate-v3": V3.CORE.gate_roi_scores(
                    ungated, v3_artifact["probabilities"][price_name], threshold=0.5
                ),
                "DualConditioned-ROIResidual": CORE.rectify_roi_scores(
                    ungated, residual_artifact["residuals"][price_name]
                ),
                "GT-ROIResidual": CORE.rectify_roi_scores(
                    ungated, residual_artifact["gt_residuals"][price_name]
                ),
            }
            teacher = (
                V1.ACTION.schedule_fixed_byte(samples, rewards, capacity_bytes=capacity)
                if mode == "fixed_byte"
                else V1.ACTION.schedule_fixed_call(samples, rewards, budget=float(budget))
            )
            teacher_roi = [
                int(image_id) for image_id, action in teacher["selection"].items()
                if action == "roi"
            ]
            for policy, scores in policies.items():
                scheduled = (
                    V1.ACTION.schedule_fixed_byte(samples, scores, capacity_bytes=capacity)
                    if mode == "fixed_byte"
                    else V1.ACTION.schedule_fixed_call(samples, scores, budget=float(budget))
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
    V2._EVAL_STATE.clear()
    V2._EVAL_STATE.update({"data": data})
    workers = min(int(PROTOCOL["cpu"]["evaluation_workers"]), len(tasks))
    started = time.time()
    with mp.get_context("fork").Pool(processes=workers) as pool:
        rows = list(pool.imap_unordered(V2._eval_task, tasks, chunksize=1))
    if len(rows) != 30 or not all(row["capacity_ok"] for row in rows):
        raise RuntimeError("Evaluation matrix is invalid")
    order = [
        "Ungated-AFPD",
        "CostAware-AnyFrontier-v2",
        "PriceConditional-Gate-v3",
        "DualConditioned-ROIResidual",
        "GT-ROIResidual",
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
                    if row["policy"] == "Ungated-AFPD" and row["mode"] == mode
                    and math.isclose(float(row["budget"]), float(budget))
                )
                candidate = next(
                    row for row in rows
                    if row["policy"] == method and row["mode"] == mode
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
    candidate = comparisons["DualConditioned-ROIResidual"]
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": "fit_oof",
        "rows": len(rows),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "residual_diagnostics": screen["diagnostics"],
        "comparisons": comparisons,
        "passed": bool(candidate["gate"]["passed"]),
        "promoted_to_three_seed": bool(candidate["gate"]["passed"]),
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
        "--stage", choices=("preflight", "screen", "evaluate", "all"), default="all"
    )
    args = parser.parse_args()
    if args.stage in {"preflight", "all"}:
        print(json.dumps(run_preflight(), indent=2), flush=True)
    if args.stage in {"screen", "all"}:
        print(json.dumps(run_screen(), indent=2), flush=True)
    if args.stage in {"evaluate", "all"}:
        print(json.dumps(run_evaluate(), indent=2), flush=True)


if __name__ == "__main__":
    main()

