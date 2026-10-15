#!/usr/bin/env python3
"""Run the frozen Current-OQ/OQR attribution and one-thread CPU profile."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import platform
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("current_oqr_strict_core", ROOT / "scripts/core.py")
SOURCES = {name: REPO / path for name, path in PROTOCOL["sources"].items()}
FORCED = load_module(
    "current_oqr_strict_source",
    SOURCES["current_fullval"] / "scripts/run_experiment.py",
)
SOURCE = FORCED.SOURCE
CORE.validate_protocol(PROTOCOL)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def source_paths() -> list[Path]:
    current_runs = SOURCES["current_fullval"] / "runs"
    paths = [
        ROOT / "configs/protocol.json",
        ROOT / "scripts/core.py",
        SCRIPT,
        SOURCES["current_fullval"] / "configs/protocol.json",
        SOURCES["current_definition"] / "configs/protocol.json",
        current_runs / "final/scores.pkl",
        current_runs / "final/evaluation/results.csv",
        SOURCES["full_benchmark"] / "runs/coco/point/final_results.csv",
    ]
    for method in PROTOCOL["current_methods"]:
        for seed in PROTOCOL["seeds"]:
            paths.append(
                current_runs
                / f"calibration/action/jobs/{method}/seed_{seed}/checkpoint.pt"
            )
    return paths


def run_preflight() -> dict[str, Any]:
    report_path = RUNS / "preflight/report.json"
    completion = RUNS / "preflight/completion.json"
    if completion.is_file():
        return read_json(report_path)
    paths = source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    with (SOURCES["current_fullval"] / "runs/final/scores.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    if set(artifact["action_scores"]) < set(PROTOCOL["current_methods"]):
        raise RuntimeError("Current score artifact is incomplete")
    for method in PROTOCOL["current_methods"]:
        CORE.ensemble_scores(artifact["action_scores"][method], PROTOCOL["seeds"])
    data = SOURCE.load_role("final")
    if len(data["samples"]) != int(PROTOCOL["roles"]["final"]):
        raise RuntimeError("Canonical final population changed")
    for method in PROTOCOL["current_methods"]:
        action_space = "OQ" if method.endswith("-OQ") else "OQR"
        records = SOURCE.records_for_role("final", action_space)
        ids = list(map(int, artifact["image_ids"][action_space]))
        if [int(record["image_id"]) for record in records] != ids:
            raise RuntimeError(f"Score/image order changed for {method}")
    checkpoint_meta = []
    for method in PROTOCOL["current_methods"]:
        action_space = "OQ" if method.endswith("-OQ") else "OQR"
        for seed in PROTOCOL["seeds"]:
            path = (
                SOURCES["current_fullval"]
                / f"runs/calibration/action/jobs/{method}/seed_{seed}/checkpoint.pt"
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if (
                checkpoint.get("label") != method
                or checkpoint.get("action_space") != action_space
                or checkpoint.get("training_method") != "CurrentActionValue"
                or int(checkpoint.get("fit_images", -1)) != 9678
                or int(checkpoint.get("seed", -1)) != int(seed)
            ):
                raise RuntimeError(f"Checkpoint contract changed: {method}/{seed}")
            checkpoint_meta.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "parameters": CORE.parameter_count(checkpoint["model_state"]),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    if len({row["parameters"] for row in checkpoint_meta}) != 1:
        raise RuntimeError("Current checkpoint architectures differ")
    frozen = read_csv(
        SOURCES["full_benchmark"] / "runs/coco/point/final_results.csv"
    )
    for method in PROTOCOL["frozen_comparators"]:
        for mode in ("fixed_call", "fixed_byte"):
            for budget in PROTOCOL["budgets"]:
                if not any(
                    row["method"] == method
                    and row["mode"] == mode
                    and math.isclose(float(row["budget"]), float(budget))
                    for row in frozen
                ):
                    raise RuntimeError(
                        f"Missing frozen comparator row: {method}/{mode}/{budget}"
                    )
    hashes = {str(path.relative_to(REPO)): file_sha256(path) for path in paths}
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "passed": True,
        "status": PROTOCOL["status"],
        "final_images": len(data["samples"]),
        "checkpoint_meta": checkpoint_meta,
        "source_hashes": hashes,
        "source_digest": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "final_is_hidden": False,
        "final_used_for_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(report_path)})
    return report


def evaluate_selection(
    data: Mapping[str, Any],
    *,
    method: str,
    variant: str,
    mode: str,
    budget: float,
    selection: Mapping[int, str],
    capacity_bytes: float,
) -> dict[str, Any]:
    chosen = {int(key): str(value) for key, value in selection.items()}
    predictions, _ = SOURCE.V1.REL.canonical.mixed_predictions(data, chosen, 0.001)
    metric = data["evaluator"].evaluate(predictions)
    request_bytes = sum(
        int(data["sample_by_id"][image_id]["payload_bytes"][action])
        for image_id, action in chosen.items()
    )
    if mode == "fixed_byte" and request_bytes > capacity_bytes + 1e-6:
        raise RuntimeError("Fixed-byte capacity exceeded")
    expected_calls = int(math.floor(len(data["samples"]) * budget + 0.5))
    if mode == "original_only_fixed_call" and len(chosen) != expected_calls:
        raise RuntimeError("Original-only Fixed-call count changed")
    counts = Counter(chosen.values())
    return {
        "method": method,
        "variant": variant,
        "mode": mode,
        "budget": float(budget),
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "kb_per_image": request_bytes / 1024.0 / len(data["samples"]),
        "request_bytes": int(request_bytes),
        "capacity_bytes": float(capacity_bytes),
        "capacity_utilization": (
            request_bytes / max(1.0, capacity_bytes)
            if mode == "fixed_byte"
            else len(chosen) / max(1, expected_calls)
        ),
        "selected_images": len(chosen),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["roi"]),
        "final_is_hidden": False,
        "final_used_for_selection": False,
    }


def frozen_comparator_rows() -> list[dict[str, Any]]:
    rows = read_csv(SOURCES["full_benchmark"] / "runs/coco/point/final_results.csv")
    output = []
    for row in rows:
        if row.get("method") not in PROTOCOL["frozen_comparators"]:
            continue
        if row.get("mode") not in {"fixed_call", "fixed_byte"}:
            continue
        if not any(
            math.isclose(float(row.get("budget", -1)), float(budget))
            for budget in PROTOCOL["budgets"]
        ):
            continue
        if row.get("policy") not in {"Original-only", "SharedAction-NoROI"}:
            continue
        output.append(
            {
                "method": row["method"],
                "policy": row["policy"],
                "mode": row["mode"],
                "budget": float(row["budget"]),
                "official_ap50_95": float(row["official_map50_95"]),
                "official_ap50": float(row["official_map50"]),
                "official_ap75": float(row["official_map75"]),
                "kb_per_image": float(row["total_kb_per_image"]),
                "selected_images": int(row["selected_images"]),
                "source": "frozen_full_scheduler_benchmark",
                "source_tier": row["source_tier"],
            }
        )
    return sorted(
        output,
        key=lambda row: (row["method"], row["policy"], row["mode"], row["budget"]),
    )


def _row(
    rows: Sequence[Mapping[str, Any]], method: str, variant: str, mode: str, budget: float
) -> Mapping[str, Any]:
    return next(
        row
        for row in rows
        if row["method"] == method
        and str(row["variant"]) == variant
        and row["mode"] == mode
        and math.isclose(float(row["budget"]), float(budget))
    )


def run_evaluation() -> dict[str, Any]:
    report_path = RUNS / "evaluation/report.json"
    completion = RUNS / "evaluation/completion.json"
    if completion.is_file():
        return read_json(report_path)
    preflight = run_preflight()
    with (SOURCES["current_fullval"] / "runs/final/scores.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    data = SOURCE.load_role("final")
    samples = data["samples"]
    original_total = sum(int(row["payload_bytes"]["original"]) for row in samples)
    current_rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, str]] = {}
    started = time.time()
    for method in PROTOCOL["current_methods"]:
        action_space = "OQ" if method.endswith("-OQ") else "OQR"
        records = SOURCE.records_for_role("final", action_space)
        ids = [int(record["image_id"]) for record in records]
        matrices = {
            str(seed): np.asarray(artifact["action_scores"][method][seed], dtype=np.float64)
            for seed in PROTOCOL["seeds"]
        }
        matrices["ensemble"] = CORE.ensemble_scores(
            artifact["action_scores"][method], PROTOCOL["seeds"]
        )
        for variant, matrix in matrices.items():
            score_map = SOURCE.V1._score_map(records, matrix)
            original_scores = {
                image_id: float(score_map[image_id]["original"]) for image_id in ids
            }
            for budget in map(float, PROTOCOL["budgets"]):
                capacity = budget * original_total
                fixed_byte = SOURCE._schedule(
                    samples,
                    score_map,
                    mode="fixed_byte",
                    budget=budget,
                    capacity_bytes=capacity,
                )
                fixed_key = f"{method}|{variant}|fixed_byte|{budget:g}"
                selections[fixed_key] = {
                    str(image_id): action for image_id, action in fixed_byte.items()
                }
                current_rows.append(
                    evaluate_selection(
                        data,
                        method=method,
                        variant=variant,
                        mode="fixed_byte",
                        budget=budget,
                        selection=fixed_byte,
                        capacity_bytes=capacity,
                    )
                )
                original = CORE.original_only_selection(ids, original_scores, budget=budget)
                original_key = (
                    f"{method}|{variant}|original_only_fixed_call|{budget:g}"
                )
                selections[original_key] = {
                    str(image_id): action for image_id, action in original.items()
                }
                current_rows.append(
                    evaluate_selection(
                        data,
                        method=method,
                        variant=variant,
                        mode="original_only_fixed_call",
                        budget=budget,
                        selection=original,
                        capacity_bytes=capacity,
                    )
                )
    expected_rows = 2 * 4 * 3 * 2
    if len(current_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} Current rows, got {len(current_rows)}")
    frozen_source = read_csv(
        SOURCES["current_fullval"] / "runs/final/evaluation/results.csv"
    )
    reproduction = []
    for method in PROTOCOL["current_methods"]:
        for budget in map(float, PROTOCOL["budgets"]):
            observed = _row(current_rows, method, "ensemble", "fixed_byte", budget)
            expected = next(
                row
                for row in frozen_source
                if row["method"] == method
                and row["seed"] == "ensemble"
                and row["mode"] == "fixed_byte"
                and math.isclose(float(row["budget"]), budget)
            )
            difference = float(observed["official_ap50_95"]) - float(
                expected["official_ap50_95"]
            )
            if abs(difference) > 1e-9:
                raise RuntimeError(
                    f"Frozen AP reproduction failed: {method}/{budget}/{difference}"
                )
            reproduction.append(
                {"method": method, "budget": budget, "delta_ap": difference}
            )
    attribution = {}
    for variant in [*map(str, PROTOCOL["seeds"]), "ensemble"]:
        attribution[variant] = {}
        for budget in map(float, PROTOCOL["budgets"]):
            oq = _row(current_rows, "Current-OQ", variant, "fixed_byte", budget)
            oqr = _row(current_rows, "Current-OQR", variant, "fixed_byte", budget)
            attribution[variant][f"{budget:g}"] = {
                "delta_ap": float(oqr["official_ap50_95"])
                - float(oq["official_ap50_95"]),
                "delta_selected_images": int(oqr["selected_images"])
                - int(oq["selected_images"]),
                "roi_count": int(oqr["roi_count"]),
            }
    comparators = frozen_comparator_rows()
    atomic_csv(RUNS / "evaluation/current_results.csv", current_rows)
    atomic_csv(RUNS / "evaluation/frozen_comparators.csv", comparators)
    atomic_json(RUNS / "evaluation/selections.json", selections)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "status": PROTOCOL["status"],
        "current_rows": len(current_rows),
        "frozen_comparator_rows": len(comparators),
        "elapsed_seconds": time.time() - started,
        "reproduction": reproduction,
        "roi_attribution": attribution,
        "source_digest": preflight["source_digest"],
        "final_is_hidden": False,
        "final_used_for_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(report_path)})
    return report


def tensor_batch(data: Mapping[str, torch.Tensor], size: int) -> dict[str, torch.Tensor]:
    indices = list(range(int(size)))
    indexed = SOURCE.V1._index_tensors(data, indices)
    return SOURCE.V1.TRAINING.model_inputs(indexed)


def measure_ns(
    function: Callable[[], Any], *, warmup: int, iterations: int
) -> dict[str, float | int]:
    with torch.inference_mode():
        for _ in range(int(warmup)):
            function()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            values = []
            for _ in range(int(iterations)):
                started = time.perf_counter_ns()
                function()
                values.append(time.perf_counter_ns() - started)
        finally:
            if gc_was_enabled:
                gc.enable()
    return CORE.latency_summary_ns(values)


def run_profile() -> dict[str, Any]:
    report_path = RUNS / "profile/report.json"
    completion = RUNS / "profile/completion.json"
    if completion.is_file():
        return read_json(report_path)
    run_preflight()
    torch.set_num_threads(int(PROTOCOL["profile"]["torch_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    records, tensors = SOURCE.prediction_tensors("final", "OQR")
    if len(records) != int(PROTOCOL["roles"]["final"]):
        raise RuntimeError("Profile tensor population changed")
    batch_size = min(int(PROTOCOL["profile"]["batch_size"]), len(records))
    batch1 = tensor_batch(tensors, 1)
    batched = tensor_batch(tensors, batch_size)
    models = []
    checkpoints = []
    for seed in PROTOCOL["seeds"]:
        checkpoint_path = (
            SOURCES["current_fullval"]
            / f"runs/calibration/action/jobs/Current-OQR/seed_{seed}/checkpoint.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = SOURCE.V1.TRAINING.build_model(
            category_count=len(SOURCE.category_ids("fit")),
            scene_dim=int(SOURCE.V1.REL.SCENE_DIM),
            action_dim=int(SOURCE.V1.REL.ACTION_DIM),
            config=SOURCE.V1.PROTOCOL["model"],
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)
        checkpoints.append(
            {
                "seed": int(seed),
                "parameters": CORE.parameter_count(checkpoint["model_state"]),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "checkpoint_sha256": file_sha256(checkpoint_path),
            }
        )
    config = SOURCE.V1.PROTOCOL["model"]
    flops = CORE.current_linear_flops(
        max_boxes=int(config["max_boxes"]),
        action_count=4,
        class_embedding_dim=int(config["class_embedding_dim"]),
        box_hidden_dim=int(config["box_hidden_dim"]),
        action_hidden_dim=int(config["action_hidden_dim"]),
        joint_hidden_dim=int(config["joint_hidden_dim"]),
        scene_dim=int(SOURCE.V1.REL.SCENE_DIM),
        action_dim=int(SOURCE.V1.REL.ACTION_DIM),
        multiply_add_flops=int(PROTOCOL["profile"]["linear_multiply_add_flops"]),
    )
    model_profiles = []
    for seed, model in zip(PROTOCOL["seeds"], models):
        batch1_latency = measure_ns(
            lambda model=model: model(**batch1),
            warmup=int(PROTOCOL["profile"]["warmup_iterations"]),
            iterations=int(PROTOCOL["profile"]["batch1_iterations"]),
        )
        batch_latency = measure_ns(
            lambda model=model: model(**batched),
            warmup=max(10, int(PROTOCOL["profile"]["warmup_iterations"]) // 10),
            iterations=int(PROTOCOL["profile"]["batched_iterations"]),
        )
        model_profiles.append(
            {
                "variant": str(seed),
                "models": 1,
                "parameters": checkpoints[len(model_profiles)]["parameters"],
                "linear_gflops_per_image": flops["linear_gflops"],
                "batch1": batch1_latency,
                "batch": batch_latency
                | {
                    "batch_size": batch_size,
                    "images_per_second_at_median": batch_size
                    / max(1e-12, float(batch_latency["median_ms"]) / 1000.0),
                },
            }
        )

    def ensemble_forward(inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([model(**inputs) for model in models], dim=0).mean(dim=0)

    ensemble_batch1 = measure_ns(
        lambda: ensemble_forward(batch1),
        warmup=int(PROTOCOL["profile"]["warmup_iterations"]),
        iterations=int(PROTOCOL["profile"]["batch1_iterations"]),
    )
    ensemble_batch = measure_ns(
        lambda: ensemble_forward(batched),
        warmup=max(10, int(PROTOCOL["profile"]["warmup_iterations"]) // 10),
        iterations=int(PROTOCOL["profile"]["batched_iterations"]),
    )
    model_profiles.append(
        {
            "variant": "ensemble",
            "models": len(models),
            "parameters": sum(row["parameters"] for row in checkpoints),
            "linear_gflops_per_image": flops["linear_gflops"] * len(models),
            "batch1": ensemble_batch1,
            "batch": ensemble_batch
            | {
                "batch_size": batch_size,
                "images_per_second_at_median": batch_size
                / max(1e-12, float(ensemble_batch["median_ms"]) / 1000.0),
            },
        }
    )
    with (SOURCES["current_fullval"] / "runs/final/scores.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    ensemble_matrix = CORE.ensemble_scores(
        artifact["action_scores"]["Current-OQR"], PROTOCOL["seeds"]
    )
    score_map = SOURCE.V1._score_map(records, ensemble_matrix)
    samples = SOURCE.load_role("final")["samples"]
    original_total = sum(int(row["payload_bytes"]["original"]) for row in samples)
    scheduler = {}
    for budget in map(float, PROTOCOL["budgets"]):
        capacity = budget * original_total
        latency = measure_ns(
            lambda capacity=capacity: SOURCE.V1.ACTION.schedule_fixed_byte(
                samples, score_map, capacity_bytes=capacity
            ),
            warmup=1,
            iterations=int(PROTOCOL["profile"]["scheduler_iterations"]),
        )
        scheduler[f"{budget:g}"] = latency | {
            "images": len(samples),
            "microseconds_per_image_at_median": float(latency["median_ms"])
            * 1000.0
            / len(samples),
        }
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "checkpoint_meta": checkpoints,
        "analytic_linear_flops": flops,
        "model_profiles": model_profiles,
        "scheduler_profiles": scheduler,
        "excluded_from_gflops_and_latency": [
            "Edge detector",
            "JPEG encoding",
            "feature construction",
            "network transfer",
            "cloud detector",
            "ROI fusion",
            "LayerNorm/SiLU/masking FLOPs",
        ],
        "ensemble_execution": "serial_one_cpu_thread",
    }
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(report_path)})
    return report


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def run_report() -> dict[str, Any]:
    evaluation = run_evaluation()
    profile = run_profile()
    current_rows = read_csv(RUNS / "evaluation/current_results.csv")
    comparator_rows = read_csv(RUNS / "evaluation/frozen_comparators.csv")
    budgets = list(map(float, PROTOCOL["budgets"]))
    ranking_methods = []
    for method in PROTOCOL["current_methods"]:
        ranking_methods.append(
            (
                method,
                [
                    float(
                        _row(
                            current_rows,
                            method,
                            "ensemble",
                            "original_only_fixed_call",
                            budget,
                        )["official_ap50_95"]
                    )
                    for budget in budgets
                ],
            )
        )
    for method in PROTOCOL["frozen_comparators"]:
        values = []
        for budget in budgets:
            row = next(
                value
                for value in comparator_rows
                if value["method"] == method
                and value["policy"] == "Original-only"
                and value["mode"] == "fixed_call"
                and math.isclose(float(value["budget"]), budget)
            )
            values.append(float(row["official_ap50_95"]))
        ranking_methods.append((method, values))
    fixed_rows = []
    for method in PROTOCOL["current_methods"]:
        values = [
            float(
                _row(current_rows, method, "ensemble", "fixed_byte", budget)[
                    "official_ap50_95"
                ]
            )
            for budget in budgets
        ]
        fixed_rows.append((method, values))
    for method in PROTOCOL["frozen_comparators"]:
        values = []
        policies = []
        for budget in budgets:
            candidates = [
                row
                for row in comparator_rows
                if row["method"] == method
                and row["mode"] == "fixed_byte"
                and math.isclose(float(row["budget"]), budget)
            ]
            best = max(candidates, key=lambda row: float(row["official_ap50_95"]))
            values.append(float(best["official_ap50_95"]))
            policies.append(best["policy"])
        fixed_rows.append((f"{method}（逐点最佳冻结策略）", values))
    lines = [
        "# Current-OQR 严格归因与 CPU 部署开销结果",
        "",
        "> 证据等级：COCO val2017 已消费后的 post-hoc reuse；没有重新训练、调参或选择比较器，不是新的隐藏 final。",
        "",
        "## 1. Original-only Fixed-call：纯图片排序",
        "",
        markdown_table(
            ["方法", "10% AP", "25% AP", "40% AP"],
            [
                [name, *[f"{value:.6f}" for value in values]]
                for name, values in ranking_methods
            ],
        ),
        "",
        "这里强制所有被选图片执行 Original，因此不包含 Q90/ROI 成本收益。Current 使用其 Original 动作价值作为图片分数。",
        "",
        "## 2–3. Fixed-byte：Current-OQ 与完整 Current-OQR",
        "",
        markdown_table(
            ["方法", "10% AP", "25% AP", "40% AP"],
            [
                [name, *[f"{value:.6f}" for value in values]]
                for name, values in fixed_rows
            ],
        ),
        "",
        "外部与 Confidence 行逐预算取其两个既有冻结策略中的较高点，仅作描述性系统比较；Current-OQ 是更接近的 No-ROI 动作受控比较。",
        "",
        "## 4. ROI 归因：Current-OQR − Current-OQ",
        "",
        markdown_table(
            ["模型", "10% ΔAP", "25% ΔAP", "40% ΔAP"],
            [
                [
                    variant,
                    *[
                        f"{evaluation['roi_attribution'][variant][f'{budget:g}']['delta_ap']:+.6f}"
                        for budget in budgets
                    ],
                ]
                for variant in [*map(str, PROTOCOL["seeds"]), "ensemble"]
            ],
        ),
        "",
        "## 5. Current-OQR 单模型与三种子 ensemble 开销",
        "",
        markdown_table(
            ["模型", "参数量", "线性 GFLOPs/图", "batch=1 median/p95 ms", "batch=192 图/s"],
            [
                [
                    row["variant"],
                    f"{row['parameters']:,}",
                    f"{row['linear_gflops_per_image']:.9f}",
                    f"{row['batch1']['median_ms']:.6f} / {row['batch1']['p95_ms']:.6f}",
                    f"{row['batch']['images_per_second_at_median']:.1f}",
                ]
                for row in profile["model_profiles"]
            ],
        ),
        "",
        "### 5,000 张全局 Fixed-byte 调度器",
        "",
        markdown_table(
            ["预算", "median/p95 ms", "median µs/图"],
            [
                [
                    budget,
                    f"{profile['scheduler_profiles'][f'{budget:g}']['median_ms']:.3f} / {profile['scheduler_profiles'][f'{budget:g}']['p95_ms']:.3f}",
                    f"{profile['scheduler_profiles'][f'{budget:g}']['microseconds_per_image_at_median']:.3f}",
                ]
                for budget in budgets
            ],
        ),
        "",
        "GFLOPs 只统计 Linear 的乘加；模型延迟只包含已构造 tensor 的 selector 前向。Edge/Cloud detector、JPEG、网络、特征构造和 ROI 融合均未计入。三模型 ensemble 在一个 CPU 线程串行执行。",
        "",
        "## 结论边界",
        "",
        "- Original-only Fixed-call 回答纯图片排序，不能代表通信优化。",
        "- Current-OQ 对外部/Confidence 是主要 No-ROI 系统对照。",
        "- Current-OQR 相对 Current-OQ 才能归因 ROI；Current-OQR 对外部的总差距不能全部归因于 ROI。",
        "- 当前结果只支持 post-hoc 点估计与机制归因；正式论文确认仍需要新互斥测试集。",
    ]
    output = ROOT / "notes/RESULTS_ZH.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "complete": True,
        "evaluation_report": str((RUNS / "evaluation/report.json").relative_to(ROOT)),
        "profile_report": str((RUNS / "profile/report.json").relative_to(ROOT)),
        "results_zh": str(output.relative_to(ROOT)),
        "results_sha256": file_sha256(output),
    }
    atomic_json(RUNS / "REPORT.json", report)
    atomic_json(RUNS / "COMPLETED.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("preflight", "evaluate", "profile", "report", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.stage == "preflight":
        result = run_preflight()
    elif args.stage == "evaluate":
        result = run_evaluation()
    elif args.stage == "profile":
        result = run_profile()
    else:
        result = run_report()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
