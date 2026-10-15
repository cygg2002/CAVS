#!/usr/bin/env python3
"""Run the development-only COCO BudgetSet value/cost OOF diagnostic."""

from __future__ import annotations

import copy
import csv
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
RUNS = ROOT / "runs"
PROTOCOL_PATH = ROOT / "configs/protocol.json"
SOURCE = REPO / "paper/experiments/20260804_coco_budgetset_registered_paper_v1"
SOURCE_SCRIPTS = SOURCE / "scripts"
DIAGNOSTIC_SCRIPTS = (
    REPO
    / "paper/experiments/20260804_coco_budget_sharedaction_costeligible_v1/scripts"
)
EXTERNAL_SCRIPTS = (
    REPO
    / "paper/experiments/20260728_three_dataset_dcsb_oric_moric_external_baselines/scripts"
)
for module_root in (
    REPO,
    ROOT / "scripts",
    SOURCE_SCRIPTS,
    DIAGNOSTIC_SCRIPTS,
    EXTERNAL_SCRIPTS,
    REPO / "paper/experiments/CA_AORIC",
):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from baseline_core import (  # noqa: E402
    build_output_feature,
    empirical_cdf,
    fit_reward_mlp,
    predict_reward_mlp,
)
from experiment_core import select_costeligible  # noqa: E402
from oof_core import (  # noqa: E402
    candidate_gate,
    deterministic_folds,
    ranking_diagnostics,
)
from oof_train import (  # noqa: E402
    fit_dual_router,
    fit_value_router,
    predict_dual_value,
    predict_value_router,
)
from paper_core import select_fixed_call  # noqa: E402
from run_experiment import (  # noqa: E402
    _fit_one_router,
    _predict_router,
    _record_tensors,
)
from run_paper import (  # noqa: E402
    _category_ids,
    _load_reward_rows,
    _reward_vectors,
    _role_records,
    _score_matrix,
    _selected_predictions,
    _selection_metrics,
    _set_protocol,
)


EXTERNAL_METHODS = (
    "ORIC-AP50",
    "MORIC-AP50",
    "ORIC-AP5095",
    "MORIC-AP5095",
)
CANDIDATE_METHODS = ("Current-BudgetSet", "SetValue", "DualHead")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def source_fingerprint() -> dict[str, str]:
    role = str(read_json(PROTOCOL_PATH)["source_role"])
    files = {
        "diagnostic_protocol": PROTOCOL_PATH,
        "source_protocol": SOURCE / "configs/protocol.json",
        "source_manifest": SOURCE / "runs/prepare/split_manifest.json",
        "edge_predictions": SOURCE
        / f"runs/inference/{role}/edge/predictions.json",
        "cloud_predictions": SOURCE
        / f"runs/inference/{role}/cloud/predictions.json",
        "contextual_rewards": SOURCE / "runs/rewards/contextual_rewards.jsonl",
        "oof_core": ROOT / "scripts/oof_core.py",
        "oof_train": ROOT / "scripts/oof_train.py",
        "oof_runner": ROOT / "scripts/run_oof.py",
        "source_runner": SOURCE / "scripts/run_paper.py",
        "budgetset_core": DIAGNOSTIC_SCRIPTS / "experiment_core.py",
        "budgetset_runner": DIAGNOSTIC_SCRIPTS / "run_experiment.py",
        "external_core": EXTERNAL_SCRIPTS / "baseline_core.py",
    }
    return {name: file_sha256(path) for name, path in files.items()}


def _subset_data(
    data: Mapping[str, torch.Tensor], indices: Sequence[int]
) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    return {key: value[index] for key, value in data.items()}


def _external_features(records: Sequence[Mapping[str, Any]], source_protocol: dict) -> np.ndarray:
    category_ids = _category_ids(source_protocol)
    top_k = int(source_protocol["baseline_mlp"]["top_k"])
    return np.asarray(
        [
            build_output_feature(
                record["edge_predictions"],
                width=int(record["width"]),
                height=int(record["height"]),
                category_ids=category_ids,
                top_k=top_k,
            )
            for record in records
        ],
        dtype=np.float64,
    )


def _train_external_fold(
    features: np.ndarray,
    ap50: np.ndarray,
    ap5095: np.ndarray,
    train_indices: Sequence[int],
    target_indices: Sequence[int],
    source_protocol: dict,
    fold_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    config = dict(source_protocol["baseline_mlp"])
    config.pop("top_k")
    config["seed"] = int(config["seed"]) + int(fold_index) * 1009
    x_train = features[train_indices]
    x_target = features[target_indices]
    y50 = ap50[train_indices]
    y5095 = ap5095[train_indices]
    cdf50 = empirical_cdf(y50, y50)
    cdf5095 = empirical_cdf(y5095, y5095)
    checkpoints = {
        "ORIC-AP50": fit_reward_mlp(x_train, y50, np.ones_like(y50), config),
        "MORIC-AP50": fit_reward_mlp(x_train, cdf50, cdf50, config),
        "ORIC-AP5095": fit_reward_mlp(
            x_train, y5095, np.ones_like(y5095), config
        ),
        "MORIC-AP5095": fit_reward_mlp(x_train, cdf5095, cdf5095, config),
    }
    return (
        {name: predict_reward_mlp(model, x_target) for name, model in checkpoints.items()},
        {name: int(model["selected_epoch"]) for name, model in checkpoints.items()},
    )


def _mean_candidate_scores(values: Sequence[np.ndarray]) -> np.ndarray:
    result = np.mean(np.asarray(values, dtype=np.float64), axis=0)
    if result.ndim != 3 or result.shape[2] != 2:
        raise RuntimeError(f"Unexpected candidate score shape: {result.shape}")
    return result[:, :, 1]


def train_fold(
    fold_index: int,
    folds: Sequence[Sequence[int]],
    records: list[dict],
    data: Mapping[str, torch.Tensor],
    raw_rewards: np.ndarray,
    ap50: np.ndarray,
    ap5095: np.ndarray,
    features: np.ndarray,
    source_protocol: dict,
    diagnostic_protocol: dict,
    fingerprint: Mapping[str, str],
) -> dict:
    output_path = RUNS / f"folds/fold_{fold_index:02d}.pkl"
    target_ids = list(map(int, folds[fold_index]))
    if output_path.is_file():
        with output_path.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("source_fingerprint") != dict(fingerprint):
            raise RuntimeError(f"Fold {fold_index} source fingerprint changed")
        if list(map(int, cached.get("target_ids", ()))) != target_ids:
            raise RuntimeError(f"Fold {fold_index} target IDs changed")
        print(f"[fold {fold_index}] verified cached result", flush=True)
        return cached

    index_by_id = {int(record["image_id"]): index for index, record in enumerate(records)}
    target_indices = [index_by_id[image_id] for image_id in target_ids]
    target_set = set(target_indices)
    train_indices = [index for index in range(len(records)) if index not in target_set]
    train_records = [records[index] for index in train_indices]
    target_records = [records[index] for index in target_indices]
    train_data = _subset_data(data, train_indices)
    target_data = _subset_data(data, target_indices)
    train_rewards = raw_rewards[train_indices]
    category_count = len(_category_ids(source_protocol))
    set_protocol = _set_protocol(source_protocol)
    seeds = list(map(int, source_protocol["set_model"]["seeds"]))
    started = time.perf_counter()

    current_predictions, current_epochs = [], []
    for seed in seeds:
        checkpoint = _fit_one_router(
            train_records,
            train_data,
            train_rewards,
            set_protocol,
            category_count=category_count,
            seed=seed,
        )
        current_predictions.append(
            _predict_router(checkpoint, target_data, set_protocol, category_count)
        )
        current_epochs.append(int(checkpoint["selected_epoch"]))
        del checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    value_predictions, value_epochs = [], []
    train_image_ids = [int(record["image_id"]) for record in train_records]
    for seed in seeds:
        checkpoint = fit_value_router(
            train_image_ids,
            train_data,
            set_protocol,
            category_count=category_count,
            seed=seed,
        )
        value_predictions.append(
            predict_value_router(
                checkpoint, target_data, set_protocol, category_count=category_count
            )
        )
        value_epochs.append(int(checkpoint["selected_epoch"]))
        del checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dual_predictions, dual_epochs = [], []
    for seed in seeds:
        checkpoint = fit_dual_router(
            train_records,
            train_data,
            train_rewards,
            set_protocol,
            category_count=category_count,
            seed=seed,
        )
        dual_predictions.append(
            predict_dual_value(
                checkpoint,
                target_records,
                target_data,
                set_protocol,
                category_count=category_count,
            )
        )
        dual_epochs.append(int(checkpoint["selected_epoch"]))
        del checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    external, external_epochs = _train_external_fold(
        features,
        ap50,
        ap5095,
        train_indices,
        target_indices,
        source_protocol,
        fold_index,
    )
    budget_count = len(diagnostic_protocol["budgets"])
    scores = {
        "Current-BudgetSet": _mean_candidate_scores(current_predictions),
        "SetValue": _mean_candidate_scores(value_predictions),
        "DualHead": _mean_candidate_scores(dual_predictions),
    }
    for method, values in external.items():
        scores[method] = np.repeat(values[None, :], budget_count, axis=0)
    artifact = {
        "fold_index": int(fold_index),
        "target_ids": target_ids,
        "scores": scores,
        "selected_epochs": {
            "Current-BudgetSet": current_epochs,
            "SetValue": value_epochs,
            "DualHead": dual_epochs,
            **external_epochs,
        },
        "seconds": time.perf_counter() - started,
        "source_fingerprint": dict(fingerprint),
        "non_development_metrics_accessed": False,
    }
    atomic_pickle(output_path, artifact)
    print(
        f"[fold {fold_index}] complete in {artifact['seconds'] / 60.0:.1f} min",
        flush=True,
    )
    return artifact


def assemble_scores(
    artifacts: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    budget_count: int,
) -> dict[str, np.ndarray]:
    index_by_id = {int(record["image_id"]): index for index, record in enumerate(records)}
    output = {
        method: np.full((int(budget_count), len(records)), np.nan, dtype=np.float64)
        for method in methods
    }
    seen: set[int] = set()
    for artifact in artifacts:
        target_ids = list(map(int, artifact["target_ids"]))
        indices = [index_by_id[image_id] for image_id in target_ids]
        if seen & set(target_ids):
            raise RuntimeError("An image received more than one OOF prediction")
        seen.update(target_ids)
        for method in methods:
            values = np.asarray(artifact["scores"][method], dtype=np.float64)
            if values.shape != (int(budget_count), len(indices)):
                raise RuntimeError(f"Bad OOF score shape for {method}: {values.shape}")
            output[method][:, indices] = values
    expected = {int(record["image_id"]) for record in records}
    if seen != expected or any(not np.isfinite(values).all() for values in output.values()):
        raise RuntimeError("OOF score population is incomplete")
    return output


def evaluate(
    records: list[dict],
    scores: Mapping[str, np.ndarray],
    ap5095: np.ndarray,
    source_protocol: dict,
    diagnostic_protocol: dict,
) -> tuple[list[dict], dict, list[dict]]:
    from ca_aoric import CocoMetricBackend

    image_ids = [int(record["image_id"]) for record in records]
    payload = np.asarray([float(record["payload_bytes"]) for record in records])
    index_by_id = {image_id: index for index, image_id in enumerate(image_ids)}
    full_capacity = float(payload.sum())
    metric = CocoMetricBackend(Path(source_protocol["paths"]["annotations"]))
    rows, selections, diagnostics = [], {}, []
    for budget_index, budget in enumerate(map(float, diagnostic_protocol["budgets"])):
        teacher_ids = select_fixed_call(image_ids, ap5095, budget=budget)
        teacher_indices = sorted(index_by_id[image_id] for image_id in teacher_ids)
        for method, method_scores in scores.items():
            values = method_scores[budget_index]
            fixed_ids = select_fixed_call(image_ids, values, budget=budget)
            fixed_indices = sorted(index_by_id[image_id] for image_id in fixed_ids)
            diagnostics.append(
                {
                    "method": method,
                    "budget": budget,
                    **ranking_diagnostics(
                        scores=values,
                        rewards=ap5095,
                        payload_bytes=payload,
                        selected_indices=fixed_indices,
                        teacher_indices=teacher_indices,
                    ),
                }
            )
            byte_result = select_costeligible(
                records,
                _score_matrix(values),
                resource_profile="upload",
                capacity=budget * full_capacity,
                full_only=True,
            )
            if float(byte_result["total_cost"]) > budget * full_capacity + 1e-6:
                raise RuntimeError(f"Fixed-byte capacity exceeded for {method}")
            byte_ids = {
                int(item["image_id"])
                for item in byte_result["selected"]
                if str(item["action_type"]) == "full"
            }
            for mode, selected_ids, extra in (
                ("fixed_call", fixed_ids, {}),
                (
                    "fixed_byte",
                    byte_ids,
                    {
                        "capacity_bytes": budget * full_capacity,
                        "used_capacity_bytes": float(byte_result["total_cost"]),
                        "lambda": float(byte_result["lambda"]),
                    },
                ),
            ):
                official = metric.evaluate(
                    _selected_predictions(records, selected_ids), image_ids
                )
                key = f"{method}|{mode}|{budget:g}"
                selections[key] = sorted(map(int, selected_ids))
                rows.append(
                    {
                        "method": method,
                        "mode": mode,
                        "budget": budget,
                        **official,
                        **_selection_metrics(records, selected_ids, source_protocol),
                        **extra,
                    }
                )
        for mode, teacher_selected in (
            ("fixed_call", teacher_ids),
            (
                "fixed_byte",
                {
                    int(item["image_id"])
                    for item in select_costeligible(
                        records,
                        _score_matrix(ap5095),
                        resource_profile="upload",
                        capacity=budget * full_capacity,
                        full_only=True,
                    )["selected"]
                    if str(item["action_type"]) == "full"
                },
            ),
        ):
            rows.append(
                {
                    "method": "Teacher-ContextGain",
                    "mode": mode,
                    "budget": budget,
                    **metric.evaluate(
                        _selected_predictions(records, teacher_selected), image_ids
                    ),
                    **_selection_metrics(records, teacher_selected, source_protocol),
                }
            )
    return rows, selections, diagnostics


def summarize(rows: Sequence[Mapping[str, Any]], diagnostic_protocol: dict) -> dict:
    comparators, comparisons, gates = {}, [], {}
    for budget in map(float, diagnostic_protocol["budgets"]):
        external = [
            row
            for row in rows
            if row["method"] in EXTERNAL_METHODS
            and row["mode"] == "fixed_call"
            and float(row["budget"]) == budget
        ]
        strongest = max(
            external,
            key=lambda row: (
                float(row["map50_95"]),
                -float(row["upload_kb_per_image"]),
                str(row["method"]),
            ),
        )
        comparators[f"fixed_call:{budget:g}"] = str(strongest["method"])
        for candidate in CANDIDATE_METHODS:
            row = next(
                item
                for item in rows
                if item["method"] == candidate
                and item["mode"] == "fixed_call"
                and float(item["budget"]) == budget
            )
            comparisons.append(
                {
                    "candidate": candidate,
                    "budget": budget,
                    "comparator": strongest["method"],
                    "candidate_ap": float(row["map50_95"]),
                    "comparator_ap": float(strongest["map50_95"]),
                    "ap_gain": float(row["map50_95"])
                    - float(strongest["map50_95"]),
                }
            )
    for candidate in CANDIDATE_METHODS:
        gains = [
            row["ap_gain"] for row in comparisons if row["candidate"] == candidate
        ]
        gates[candidate] = candidate_gate(
            gains,
            minimum_gain=float(diagnostic_protocol["minimum_gain"]),
            required_gain_points=int(diagnostic_protocol["required_gain_points"]),
            maximum_other_loss=float(diagnostic_protocol["maximum_other_loss"]),
        )
    return {
        "comparators": comparators,
        "comparisons": comparisons,
        "gates": gates,
    }


def write_chinese_report(report: Mapping[str, Any], diagnostics: Sequence[Mapping[str, Any]]) -> None:
    rows = report["rows"]
    lines = [
        "# COCO BudgetSet Value/Cost OOF 诊断结果",
        "",
        "> 仅使用 5,000 张 development；模型预测为 5-fold OOF。冻结 contextual reward 的 context 可能跨 fold，因此这是机制诊断，不是论文 final。",
        "",
        "## Fixed-call 主对比",
        "",
        "| 方法 | 10% AP / KB | 25% AP / KB | 40% AP / KB | Gate |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in (*EXTERNAL_METHODS, *CANDIDATE_METHODS):
        values = []
        for budget in (0.1, 0.25, 0.4):
            row = next(
                item
                for item in rows
                if item["method"] == method
                and item["mode"] == "fixed_call"
                and float(item["budget"]) == budget
            )
            values.append(
                f"{float(row['map50_95']):.6f} / {float(row['upload_kb_per_image']):.2f}"
            )
        gate = report["summary"]["gates"].get(method, {}).get("passed", "—")
        lines.append(f"| {method} | {values[0]} | {values[1]} | {values[2]} | {gate} |")
    lines.extend(
        [
            "",
            "## 目标/成本诊断",
            "",
            "| 方法 | 预算 | gain Spearman | payload Spearman | Teacher Top-K recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in CANDIDATE_METHODS:
        for budget in (0.1, 0.25, 0.4):
            row = next(
                item
                for item in diagnostics
                if item["method"] == method and float(item["budget"]) == budget
            )
            lines.append(
                f"| {method} | {budget:.0%} | {row['reward_spearman']:.4f} | "
                f"{row['payload_spearman']:.4f} | {row['topk_recall']:.2%} |"
            )
    lines.extend(
        [
            "",
            f"结论字段：`{json.dumps(report['summary']['gates'], ensure_ascii=False)}`",
            "",
            "本实验没有访问非 development 指标，也没有改变已注册实验。",
        ]
    )
    path = ROOT / "notes/results_zh.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    diagnostic_protocol = read_json(PROTOCOL_PATH)
    source_protocol = read_json(SOURCE / "configs/protocol.json")
    role = str(diagnostic_protocol["source_role"])
    if role != "development":
        raise RuntimeError("The OOF diagnostic is development-only")
    fingerprint = source_fingerprint()
    records = _role_records(role, source_protocol)
    if len(records) != int(diagnostic_protocol["expected_images"]):
        raise RuntimeError("Development population drifted")
    reward_rows = _load_reward_rows(SOURCE / "runs/rewards/contextual_rewards.jsonl")
    ap50, ap5095 = _reward_vectors(records, reward_rows)
    raw_rewards = np.stack((np.zeros_like(ap5095), ap5095), axis=1).astype(
        np.float32
    )
    set_protocol = _set_protocol(source_protocol)
    data = _record_tensors(records, raw_rewards, set_protocol)
    features = _external_features(records, source_protocol)
    image_ids = [int(record["image_id"]) for record in records]
    folds = deterministic_folds(
        image_ids,
        fold_count=int(diagnostic_protocol["fold_count"]),
        seed=int(diagnostic_protocol["fold_seed"]),
    )
    atomic_json(
        RUNS / "folds/manifest.json",
        {
            "folds": folds,
            "source_fingerprint": fingerprint,
            "non_development_metrics_accessed": False,
        },
    )
    artifacts = [
        train_fold(
            index,
            folds,
            records,
            data,
            raw_rewards,
            ap50,
            ap5095,
            features,
            source_protocol,
            diagnostic_protocol,
            fingerprint,
        )
        for index in range(len(folds))
    ]
    scores = assemble_scores(
        artifacts,
        records,
        diagnostic_protocol["methods"],
        len(diagnostic_protocol["budgets"]),
    )
    rows, selections, diagnostics = evaluate(
        records, scores, ap5095, source_protocol, diagnostic_protocol
    )
    summary = summarize(rows, diagnostic_protocol)
    report = {
        "protocol_id": diagnostic_protocol["protocol_id"],
        "images": len(records),
        "folds": len(folds),
        "rows": rows,
        "diagnostics": diagnostics,
        "summary": summary,
        "fold_seconds": [float(artifact["seconds"]) for artifact in artifacts],
        "source_fingerprint": fingerprint,
        "reward_context_may_cross_oof_folds": True,
        "non_development_metrics_accessed": False,
    }
    atomic_csv(RUNS / "results.csv", rows)
    atomic_json(RUNS / "selections.json", selections)
    atomic_json(RUNS / "report.json", report)
    write_chinese_report(report, diagnostics)
    atomic_json(
        RUNS / "COMPLETED.json",
        {
            "complete": True,
            "report_sha256": file_sha256(RUNS / "report.json"),
            "results_sha256": file_sha256(RUNS / "results.csv"),
            "non_development_metrics_accessed": False,
        },
    )
    print(json.dumps(summary["gates"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
