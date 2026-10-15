#!/usr/bin/env python3
"""Evaluate native Original-only ORIC/MORIC and heuristic baselines on COCO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("native_baseline_core", ROOT / "scripts/core.py")
CORE.validate_protocol(PROTOCOL)
COMPLETION_ROOT = REPO / PROTOCOL["source_completion_experiment"]
AC_ROOT = REPO / PROTOCOL["source_ac_experiment"]
SOURCE = load_module("native_baseline_completion_source", COMPLETION_ROOT / "scripts/run_experiment.py")
AC = load_module("native_baseline_ac_source", AC_ROOT / "scripts/run_experiment.py")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
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


def score_path(method: str, seed: int) -> Path:
    return (
        COMPLETION_ROOT
        / f"runs/coco/training/jobs/{method}/seed_{seed}/full/scores.npz"
    )


def source_paths() -> dict[str, Path]:
    paths = {
        "completion_protocol": COMPLETION_ROOT / "configs/protocol.json",
        "completion_script": COMPLETION_ROOT / "scripts/run_experiment.py",
        "completion_preflight": COMPLETION_ROOT / "runs/coco/preflight/report.json",
        "completion_results": COMPLETION_ROOT / "runs/coco/point/results.csv",
        "ac_protocol": AC_ROOT / "configs/protocol.json",
        "ac_script": AC_ROOT / "scripts/run_experiment.py",
        "ac_results": AC_ROOT / "runs/evaluation/results.csv",
        "ac_selections": AC_ROOT / "runs/evaluation/selections.json",
    }
    for label, method in (("oric", "ORIC-AP5095"), ("moric", "MORIC-AP5095")):
        for seed in map(int, PROTOCOL["model_seeds"]):
            paths[f"{label}_{seed}"] = score_path(method, seed)
    return paths


def run_preflight() -> dict[str, Any]:
    output = RUNS / "preflight"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    paths = source_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    observed = {name: file_sha256(path) for name, path in paths.items()}
    if observed != PROTOCOL["source_hashes"]:
        changed = {
            name: {"expected": PROTOCOL["source_hashes"].get(name), "observed": value}
            for name, value in observed.items()
            if PROTOCOL["source_hashes"].get(name) != value
        }
        raise RuntimeError(f"frozen source changed: {changed}")
    report = read_json(paths["completion_preflight"])
    if report["role_counts"] != PROTOCOL["roles"]:
        raise RuntimeError("COCO role populations changed")
    if report["cloud_model_sha256"] != PROTOCOL["cloud_model_sha256"]:
        raise RuntimeError("Cloud model changed")
    ac_rows = read_csv(paths["ac_results"])
    points = {
        f"{float(row['budget']):g}": float(row["official_ap50_95"])
        for row in ac_rows
        if row["method"] == "CAVS-OQR"
        and row["role"] == "final"
        and row["mode"] == "fixed_byte"
    }
    for budget, expected in PROTOCOL["cavs_expected_fixed_byte"].items():
        if abs(points[budget] - float(expected)) > 1e-12:
            raise RuntimeError(f"CAVS-OQR point changed: {budget}")
    result = {
        "complete": True,
        "runtime_domain": PROTOCOL["runtime_domain"],
        "role_counts": report["role_counts"],
        "cloud_model_sha256": report["cloud_model_sha256"],
        "cavs_fixed_byte": points,
        "source_sha256": observed,
        "rknn_artifacts_read": 0,
        "detector_inference_required": False,
        "source_experiments_read_only": True,
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(output / "report.json", result)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(output / "report.json")})
    return result


def selection_key(role: str, method: str, mode: str, budget: float, seed: str | int) -> str:
    return f"{role}|{method}|{mode}|{budget:g}|seed={seed}"


def evaluate(
    role: str,
    method: str,
    mode: str,
    budget: float,
    seed: str | int,
    selection: Mapping[int, str],
) -> dict[str, Any]:
    CORE.validate_native_selection(selection)
    row = SOURCE.evaluate_selection(
        "coco",
        role,
        selection,
        method=method,
        policy="Original-only",
        mode=mode,
        budget=budget,
        seed=seed,
        fusion="Reversible",
    )
    row.update(
        {
            "runtime_domain": PROTOCOL["runtime_domain"],
            "action_space": "Edge/Original",
            "rknn_artifacts_read": 0,
        }
    )
    return row


def point_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate(
        str(task["role"]),
        str(task["method"]),
        str(task["mode"]),
        float(task["budget"]),
        task["seed"],
        {int(image_id): str(action) for image_id, action in task["selection"].items()},
    )


def run_point() -> dict[str, Any]:
    output = RUNS / "point"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    run_preflight()
    adapter = SOURCE.adapter("coco")
    roles = ("calibration", "final")
    samples = {role: adapter.load_role(role)["samples"] for role in roles}
    actions = {role: CORE.original_action_map(samples[role]) for role in roles}
    external: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for report_method, source_method in PROTOCOL["source_methods"].items():
        external[report_method] = {}
        for role in roles:
            single, ensemble = SOURCE.ensemble_external_scores("coco", source_method, role)
            external[report_method][role] = {**single, "ensemble": ensemble}
    scalars = SOURCE.UNIFIED.scalar_scores(adapter)
    tasks: list[dict[str, Any]] = []
    selections: dict[str, dict[str, str]] = {}
    random_selections: dict[str, dict[str, str]] = {}

    def add(
        role: str,
        method: str,
        mode: str,
        budget: float,
        seed: str | int,
        selection: Mapping[int, str],
        *,
        random: bool = False,
    ) -> None:
        CORE.validate_native_selection(selection)
        key = selection_key(role, method, mode, budget, seed)
        target = random_selections if random else selections
        target[key] = {str(image_id): str(action) for image_id, action in selection.items()}
        tasks.append(
            {
                "role": role,
                "method": method,
                "mode": mode,
                "budget": budget,
                "seed": seed,
                "selection": dict(selection),
            }
        )

    for role in roles:
        for method in ("ORIC-Original", "MORIC-Original"):
            for seed, values in external[method][role].items():
                reference = external[method]["calibration"][seed]
                for budget in map(float, PROTOCOL["budgets"]):
                    for mode in PROTOCOL["modes"]:
                        selection = SOURCE._scalar_selection(
                            "coco", samples[role], values, reference, actions[role],
                            mode=mode, budget=budget,
                        )
                        add(role, method, mode, budget, seed, selection)
        confidence = np.asarray(scalars[role]["Confidence"], dtype=np.float64)
        cal_confidence = np.asarray(scalars["calibration"]["Confidence"], dtype=np.float64)
        for budget_index, budget in enumerate(map(float, PROTOCOL["budgets"])):
            for mode in PROTOCOL["modes"]:
                selection = SOURCE._scalar_selection(
                    "coco", samples[role], confidence[budget_index], cal_confidence[budget_index],
                    actions[role], mode=mode, budget=budget,
                )
                add(role, "Confidence-Original", mode, budget, "fixed", selection)
            for offset in range(int(PROTOCOL["bootstrap"]["random_seeds"])):
                seed = int(PROTOCOL["bootstrap"]["random_base_seed"]) + offset
                values = np.random.default_rng(seed).random(len(samples[role]))
                reference = np.random.default_rng(seed).random(len(samples["calibration"]))
                for mode in PROTOCOL["modes"]:
                    selection = SOURCE._scalar_selection(
                        "coco", samples[role], values, reference, actions[role],
                        mode=mode, budget=budget,
                    )
                    add(role, "Random-Original", mode, budget, seed, selection, random=True)

    workers = min(int(PROTOCOL["cpu"]["evaluation_workers"]), len(tasks))
    with mp.get_context("fork").Pool(workers) as pool:
        evaluated = list(pool.imap_unordered(point_worker, tasks, chunksize=1))
    rows = [row for row in evaluated if row["method"] != "Random-Original"]
    random_rows = [row for row in evaluated if row["method"] == "Random-Original"]
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in random_rows:
        key = (str(row["role"]), str(row["mode"]), float(row["budget"]))
        grouped.setdefault(key, []).append(row)
    for values in grouped.values():
        if len(values) != int(PROTOCOL["bootstrap"]["random_seeds"]):
            raise RuntimeError("Random-50 point population incomplete")
        item = dict(values[0])
        for name in (
            "official_ap50_95", "official_ap50", "official_ap75", "kb_per_image",
            "request_bytes", "capacity_utilization", "selected_images", "original_count",
            "q90_count", "roi_count",
        ):
            item[name] = float(np.mean([float(row[name]) for row in values]))
        item["seed"] = "mean50"
        item["random_seed_count"] = int(PROTOCOL["bootstrap"]["random_seeds"])
        rows.append(item)
    if len(rows) != CORE.expected_point_rows(PROTOCOL):
        raise RuntimeError(f"native point matrix incomplete: {len(rows)}")
    low = [
        row for row in rows
        if row["mode"] == "fixed_byte" and float(row["capacity_utilization"]) < 0.98
    ]
    if low:
        raise RuntimeError(f"fixed-byte utilization below 98%: {low[0]}")

    ac_rows = read_csv(AC_ROOT / "runs/evaluation/results.csv")
    anchors = [
        {**row, "action_space": "Edge/Original/Q90/Current@640"}
        for row in ac_rows
        if row["method"] == "CAVS-OQR" and row["role"] == "final"
    ]
    if len(anchors) != len(PROTOCOL["modes"]) * len(PROTOCOL["budgets"]):
        raise RuntimeError("CAVS-OQR anchor matrix incomplete")
    ac_selections = read_json(AC_ROOT / "runs/evaluation/selections.json")
    for mode in PROTOCOL["modes"]:
        for budget in map(float, PROTOCOL["budgets"]):
            source = f"final|CAVS-OQR|{mode}|{budget:g}"
            selections[selection_key("final", "CAVS-OQR", mode, budget, "ensemble")] = ac_selections[source]

    atomic_csv(output / "results.csv", [*rows, *anchors])
    atomic_csv(output / "random_seed_results.csv", random_rows)
    atomic_json(output / "selections.json", selections)
    atomic_json(output / "random_selections.json", random_selections)
    report = {
        "complete": True,
        "native_rows": len(rows),
        "anchor_rows": len(anchors),
        "random_seed_rows": len(random_rows),
        "evaluation_workers": workers,
        "results_sha256": file_sha256(output / "results.csv"),
        "selections_sha256": file_sha256(output / "selections.json"),
        "random_selections_sha256": file_sha256(output / "random_selections.json"),
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "results_sha256": report["results_sha256"]})
    return report


_BOOT: dict[str, Any] = {}


def bootstrap_path(job: Mapping[str, Any]) -> tuple[Path, Path]:
    token = hashlib.sha256(str(job["id"]).encode()).hexdigest()[:20]
    path = RUNS / f"bootstrap/jobs/{token}.json"
    return path, path.with_suffix(".partial.json")


def bootstrap_rng(job: Mapping[str, Any]) -> np.random.Generator:
    token = hashlib.sha256(
        f"{PROTOCOL['bootstrap']['seed']}:{job['id']}".encode()
    ).digest()
    return np.random.default_rng(int.from_bytes(token[:8], "big"))


def bootstrap_job(job: Mapping[str, Any]) -> dict[str, Any]:
    path, partial = bootstrap_path(job)
    if path.is_file():
        return read_json(path)
    values = list(map(float, read_json(partial)["values"])) if partial.is_file() else []
    rng = bootstrap_rng(job)
    image_ids = _BOOT["image_ids"]
    for _ in values:
        rng.integers(0, len(image_ids), size=len(image_ids))
    total = int(PROTOCOL["bootstrap"]["replicates"])
    left = _BOOT["cavs"][float(job["budget"])]
    comparators = _BOOT["comparators"][float(job["budget"])][str(job["right"])]
    for replicate in range(len(values), total):
        sampled = [image_ids[int(index)] for index in rng.integers(0, len(image_ids), size=len(image_ids))]
        right_ap = []
        left_ap = None
        for right in comparators:
            gt, candidate, comparator = AC.BOOTSTRAP_CORE._bootstrap_remap(
                _BOOT["ground_truth"], left, right, sampled
            )
            if left_ap is None:
                left_ap = float(AC.BOOTSTRAP_CORE._in_memory_ap(gt, candidate))
            right_ap.append(float(AC.BOOTSTRAP_CORE._in_memory_ap(gt, comparator)))
        values.append(float(left_ap - np.mean(right_ap)))
        interval = 25 if job["kind"] == "nested_random" else 100
        if (replicate + 1) % interval == 0:
            atomic_json(partial, {"values": values})
    array = np.asarray(values, dtype=np.float64)
    result = {
        **dict(job),
        "replicates": len(values),
        "mean_delta": float(array.mean()),
        "median_delta": float(np.median(array)),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "probability_gt_zero": float(np.mean(array > 0.0)),
        "random_seed_count": len(comparators) if job["kind"] == "nested_random" else "",
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(path, result)
    partial.unlink(missing_ok=True)
    return result


def run_bootstrap() -> dict[str, Any]:
    output = RUNS / "bootstrap"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    run_point()
    adapter = SOURCE.adapter("coco")
    data = adapter.load_role("final")
    stored = read_json(RUNS / "point/selections.json")
    random_stored = read_json(RUNS / "point/random_selections.json")

    def load_selection(method: str, budget: float, seed: str | int, *, random: bool = False) -> dict[int, str]:
        key = selection_key("final", method, "fixed_byte", budget, seed)
        source = random_stored if random else stored
        return {int(image_id): str(action) for image_id, action in source[key].items()}

    cavs = {}
    comparators: dict[float, dict[str, list[Any]]] = {}
    for budget in map(float, PROTOCOL["budgets"]):
        cavs[budget] = SOURCE.mixed_predictions(
            "coco", "final", load_selection("CAVS-OQR", budget, "ensemble"), "MAUR"
        )
        comparators[budget] = {}
        for method in ("ORIC-Original", "MORIC-Original"):
            comparators[budget][method] = [
                SOURCE.mixed_predictions(
                    "coco", "final", load_selection(method, budget, "ensemble"), "Reversible"
                )
            ]
        comparators[budget]["Confidence-Original"] = [
            SOURCE.mixed_predictions(
                "coco", "final", load_selection("Confidence-Original", budget, "fixed"), "Reversible"
            )
        ]
        comparators[budget]["Random-Original"] = [
            SOURCE.mixed_predictions(
                "coco", "final", load_selection("Random-Original", budget, seed, random=True), "Reversible"
            )
            for seed in range(
                int(PROTOCOL["bootstrap"]["random_base_seed"]),
                int(PROTOCOL["bootstrap"]["random_base_seed"])
                + int(PROTOCOL["bootstrap"]["random_seeds"]),
            )
        ]
    _BOOT.clear()
    _BOOT.update(
        ground_truth=read_json(adapter.gt_path("final")),
        image_ids=[int(sample["image_id"]) for sample in data["samples"]],
        cavs=cavs,
        comparators=comparators,
    )
    jobs = CORE.bootstrap_jobs(PROTOCOL)
    workers = min(int(PROTOCOL["cpu"]["bootstrap_workers"]), len(jobs))
    with mp.get_context("fork").Pool(workers) as pool:
        results = list(pool.imap_unordered(bootstrap_job, jobs, chunksize=1))
    _BOOT.clear()
    results.sort(key=lambda row: (float(row["budget"]), str(row["right"])))
    atomic_csv(output / "results.csv", results)
    report = {
        "complete": True,
        "jobs": len(results),
        "replicates": int(PROTOCOL["bootstrap"]["replicates"]),
        "nested_random": True,
        "results_sha256": file_sha256(output / "results.csv"),
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "jobs": len(results)})
    return report


def run_report() -> dict[str, Any]:
    output = RUNS / "report"
    completion = output / "completion.json"
    if is_complete(completion):
        return read_json(output / "report.json")
    run_bootstrap()
    rows = read_csv(RUNS / "point/results.csv")
    boot = read_csv(RUNS / "bootstrap/results.csv")
    final = {
        (row["method"], float(row["budget"])): float(row["official_ap50_95"])
        for row in rows
        if row["role"] == "final"
        and row["mode"] == "fixed_byte"
        and row["seed"] in {"ensemble", "fixed", "mean50"}
    }
    methods = ["CAVS-OQR", *PROTOCOL["native_methods"]]
    lines = [
        "# COCO 原生 ORIC/MORIC 与启发式基线补全",
        "",
        "> PyTorch 论文缓存域。ORIC/MORIC/Confidence/Random 仅允许 Edge/Original；CAVS-OQR 使用完整 OQR 动作，因此这是系统级比较，不是动作空间匹配消融。",
        "",
        "## Fixed-byte 主结果",
        "",
        "| 方法 | 动作空间 | 10% | 25% | 40% |",
        "|---|---|---:|---:|---:|",
    ]
    for method in methods:
        space = "OQR" if method == "CAVS-OQR" else "Original-only"
        lines.append(
            f"| {method} | {space} | {final[(method, 0.1)]:.6f} | "
            f"{final[(method, 0.25)]:.6f} | {final[(method, 0.4)]:.6f} |"
        )
    lines.extend([
        "",
        "## ORIC/MORIC 三种子单模型",
        "",
        "| 方法 | 预算 | Mean ± SD | Ensemble |",
        "|---|---:|---:|---:|",
    ])
    for method in ("ORIC-Original", "MORIC-Original"):
        for budget in map(float, PROTOCOL["budgets"]):
            values = np.asarray([
                float(row["official_ap50_95"])
                for row in rows
                if row["role"] == "final" and row["mode"] == "fixed_byte"
                and row["method"] == method and row["seed"] in {str(seed) for seed in PROTOCOL["model_seeds"]}
                and abs(float(row["budget"]) - budget) < 1e-12
            ])
            if len(values) != len(PROTOCOL["model_seeds"]):
                raise RuntimeError(f"single-model matrix incomplete: {method}/{budget}")
            lines.append(
                f"| {method} | {budget:g} | {values.mean():.6f} ± {values.std(ddof=1):.6f} | "
                f"{final[(method, budget)]:.6f} |"
            )
    lines.extend([
        "",
        "## Paired bootstrap",
        "",
        "| 对比 | 预算 | AP差 | 95% CI | P(>0) |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in boot:
        lines.append(
            f"| CAVS-OQR - {row['right']} | {float(row['budget']):g} | "
            f"{float(row['mean_delta']):+.6f} | "
            f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}] | "
            f"{float(row['probability_gt_zero']):.3f} |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- `ORIC-Original`/`MORIC-Original` 是论文标量学习器的统一系统重实现，不是原作者发布的官方数值。",
        "- 原生表回答完整 CAVS 系统是否优于外部方法；OQR action-conditioned 表回答在同动作空间下学习器本身是否更强。",
        "- Random 的 bootstrap 在每次图像重采样内重新计算 50 套冻结随机策略的 AP 均值。",
        "- COCO val2017 已多次用于诊断，属于 post-hoc paper-grade final，不是新的 hidden test。",
    ])
    output.mkdir(parents=True, exist_ok=True)
    (output / "RESULTS_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "complete": True,
        "point_results_sha256": file_sha256(RUNS / "point/results.csv"),
        "bootstrap_results_sha256": file_sha256(RUNS / "bootstrap/results.csv"),
        "report_sha256": file_sha256(output / "RESULTS_ZH.md"),
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(completion, {"complete": True, "report_sha256": report["report_sha256"]})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "point", "bootstrap", "report"), required=True)
    args = parser.parse_args()
    functions = {
        "preflight": run_preflight,
        "point": run_point,
        "bootstrap": run_bootstrap,
        "report": run_report,
    }
    print(json.dumps(functions[args.stage](), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
