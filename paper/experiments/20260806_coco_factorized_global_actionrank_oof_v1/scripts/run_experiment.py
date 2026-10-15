#!/usr/bin/env python3
"""Run the cache-only COCO Factorized Global Action Rank diagnostic."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
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
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
RELATIONAL_ROOT = REPO / PROTOCOL["source_experiments"]["relational_source"]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load_module("factorized_actionrank_core", ROOT / "scripts/core.py")
BASE = _load_module(
    "factorized_actionrank_relational_runner",
    RELATIONAL_ROOT / "scripts/run_experiment.py",
)
CORE.validate_protocol(PROTOCOL)

SOURCE_BUILD_MODEL = BASE.build_model
SOURCE_OBJECTIVE = BASE._objective
SOURCE_RUN_REWARDS = BASE.run_rewards
SOURCE_LOAD_REWARD_MAP = BASE._load_reward_map
SOURCE_FILES = BASE._source_files


def _bind_experiment() -> None:
    sources = {
        name: REPO / path for name, path in PROTOCOL["source_experiments"].items()
    }
    BASE.SCRIPT = SCRIPT
    BASE.ROOT = ROOT
    BASE.RUNS = RUNS
    BASE.PROTOCOL_PATH = PROTOCOL_PATH
    BASE.PROTOCOL = PROTOCOL
    BASE.SOURCES = sources
    BASE.CANONICAL_ROOT = sources["canonical"]
    BASE.COMMON = BASE.CANONICAL_ROOT / PROTOCOL["canonical_common"]
    BASE.ACTIONVALUE_SCRIPTS = sources["actionvalue"] / "scripts"
    BASE.SET_SCRIPTS = sources["set_model"] / "scripts"
    BASE.VALUE_SCRIPTS = sources["value_model"] / "scripts"
    BASE.REWARD_SCRIPTS = sources["reward_core"] / "scripts"
    BASE.CANONICAL_SCRIPTS = BASE.CANONICAL_ROOT / "scripts"
    BASE.METHODS = CORE.METHODS
    BASE.GLOBAL_METHODS = frozenset(("GlobalRank", "RelActionRank"))
    BASE.EXPECTED_ACTIONS = CORE.EXPECTED_ACTIONS
    BASE.EXPECTED_BUDGETS = CORE.EXPECTED_BUDGETS
    BASE.EXPECTED_MODES = CORE.EXPECTED_MODES
    BASE.validate_protocol = CORE.validate_protocol
    BASE.evaluate_candidate_gate = CORE.evaluate_candidate_gate
    BASE.should_run_calibration = CORE.should_run_calibration
    BASE.validate_result_matrix = CORE.validate_result_matrix
    BASE._ROLE_CACHE.clear()


_bind_experiment()


def build_model(
    method: str,
    *,
    category_count: int,
    scene_dim: int,
    action_dim: int,
    config: Mapping[str, Any],
) -> torch.nn.Module:
    if method == "FactorizedGlobalActionRank":
        return CORE.FactorizedGlobalActionRankRouter(
            category_count=category_count,
            scene_dim=scene_dim,
            action_dim=action_dim,
            config=config,
        )
    if method not in CORE.REUSED_METHODS:
        raise ValueError(f"Unknown method: {method}")
    return SOURCE_BUILD_MODEL(
        method,
        category_count=category_count,
        scene_dim=scene_dim,
        action_dim=action_dim,
        config=config,
    )


def objective(
    model: torch.nn.Module,
    method: str,
    data: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if method != "FactorizedGlobalActionRank":
        return SOURCE_OBJECTIVE(model, method, data, indices)
    inputs = BASE._model_inputs(data, indices)
    return CORE.factorized_objective(
        model,
        inputs,
        data["target"][indices],
        data["action_mask"][indices],
        data["action_type"][indices],
        config=PROTOCOL["model"],
    )


BASE.build_model = build_model
BASE._objective = objective


def _source_reward_paths() -> dict[str, Path]:
    root = RELATIONAL_ROOT / "runs/rewards/fit"
    return {
        "completion": root / "completion.json",
        "report": root / "report.json",
        "map": root / "reward_map.pkl",
        "rows": root / "rewards.jsonl",
    }


def _source_score_paths() -> dict[str, Path]:
    root = RELATIONAL_ROOT / "runs/oof"
    return {
        "completion": root / "completion.json",
        "report": root / "report.json",
        "scores": root / "scores.pkl",
        "folds": RELATIONAL_ROOT / "runs/preflight/folds.json",
    }


def _verify_source_reward() -> dict[str, Any]:
    paths = _source_reward_paths()
    completion = BASE.read_json(paths["completion"])
    if completion.get("complete") is not True:
        raise RuntimeError("Frozen fit reward source is incomplete")
    if completion.get("map_sha256") != BASE.file_sha256(paths["map"]):
        raise RuntimeError("Frozen fit reward map changed")
    report = BASE.read_json(paths["report"])
    if int(report["images"]) != 9678 or int(report["cross_fold_violations"]) != 0:
        raise RuntimeError("Frozen fit reward protocol changed")
    return report


def run_rewards(role: str) -> dict[str, Any]:
    if role != "fit":
        return SOURCE_RUN_REWARDS(role)
    completion_path = RUNS / "rewards/fit/completion.json"
    if completion_path.is_file():
        return BASE.read_json(RUNS / "rewards/fit/report.json")
    source = _verify_source_reward()
    report = {
        **source,
        "protocol_id": PROTOCOL["protocol_id"],
        "status": "reused_read_only",
        "source_experiment": str(RELATIONAL_ROOT),
        "source_map_sha256": BASE.file_sha256(_source_reward_paths()["map"]),
        "calibration_used_for_training_or_selection": False,
    }
    BASE.atomic_json(RUNS / "rewards/fit/report.json", report)
    BASE.atomic_json(
        completion_path,
        {
            "complete": True,
            "status": "reused_read_only",
            "source_map_sha256": report["source_map_sha256"],
            "report_sha256": BASE.file_sha256(RUNS / "rewards/fit/report.json"),
        },
    )
    return report


def load_reward_map(role: str) -> dict[int, dict[str, float]]:
    if role != "fit":
        return SOURCE_LOAD_REWARD_MAP(role)
    _verify_source_reward()
    values = BASE.load_pickle(_source_reward_paths()["map"])
    return {
        int(image_id): {
            str(action): float(reward) for action, reward in actions.items()
        }
        for image_id, actions in values.items()
    }


BASE.run_rewards = run_rewards
BASE._load_reward_map = load_reward_map


def source_files() -> list[Path]:
    paths = list(SOURCE_FILES())
    paths.extend(_source_reward_paths().values())
    paths.extend(_source_score_paths().values())
    paths.append(RELATIONAL_ROOT / "scripts/core.py")
    paths.append(RELATIONAL_ROOT / "scripts/run_experiment.py")
    return list(dict.fromkeys(map(Path, paths)))


BASE._source_files = source_files


def _load_frozen_scores(image_ids: Sequence[int]) -> dict[str, dict[int, np.ndarray]]:
    paths = _source_score_paths()
    completion = BASE.read_json(paths["completion"])
    if completion.get("complete") is not True:
        raise RuntimeError("Frozen OOF source is incomplete")
    if completion.get("score_sha256") != BASE.file_sha256(paths["scores"]):
        raise RuntimeError("Frozen OOF score source changed")
    artifact = BASE.load_pickle(paths["scores"])
    if list(map(int, artifact["image_ids"])) != list(map(int, image_ids)):
        raise RuntimeError("Frozen OOF score image ordering changed")
    output: dict[str, dict[int, np.ndarray]] = {}
    for method in CORE.REUSED_METHODS:
        output[method] = {}
        for seed in PROTOCOL["seeds"]:
            values = np.asarray(artifact["scores"][method][int(seed)], dtype=np.float64)
            if values.shape != (len(image_ids), 3) or not np.isfinite(values).all():
                raise RuntimeError(f"Invalid frozen scores: {method}/{seed}")
            output[method][int(seed)] = values.copy()
    return output


def run_oof() -> dict[str, Any]:
    completion_path = RUNS / "oof/completion.json"
    if completion_path.is_file():
        return BASE.read_json(RUNS / "oof/report.json")
    BASE._verify_preflight()
    run_rewards("fit")
    records, data = BASE._role_tensors("fit")
    image_ids = [int(record["image_id"]) for record in records]
    folds = [
        list(map(int, fold))
        for fold in BASE.read_json(RUNS / "preflight/folds.json")["folds"]
    ]
    source_folds = [
        list(map(int, fold))
        for fold in BASE.read_json(_source_score_paths()["folds"])["folds"]
    ]
    if folds != source_folds:
        raise RuntimeError("Current and frozen OOF folds differ")
    scores = _load_frozen_scores(image_ids)
    BASE._ROLE_CACHE.pop("fit", None)
    gc.collect()
    BASE._TRAIN_STATE.clear()
    BASE._TRAIN_STATE.update({"image_ids": image_ids, "data": data, "folds": folds})
    tasks = [
        ("FactorizedGlobalActionRank", int(seed), fold_index)
        for seed in PROTOCOL["seeds"]
        for fold_index in range(int(PROTOCOL["fold_count"]))
    ]
    workers = min(int(PROTOCOL["cpu"]["training_workers"]), len(tasks))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        reports = list(pool.imap_unordered(BASE._train_oof_job, tasks, chunksize=1))
    candidate_scores = {
        int(seed): np.full((len(image_ids), 3), np.nan, dtype=np.float64)
        for seed in PROTOCOL["seeds"]
    }
    id_to_index = {image_id: index for index, image_id in enumerate(image_ids)}
    for method, seed, fold_index in tasks:
        path = RUNS / f"oof/jobs/{method}/seed_{seed}/fold_{fold_index}/scores.npz"
        with np.load(path, allow_pickle=False) as artifact:
            heldout_ids = list(map(int, artifact["image_ids"]))
            values = np.asarray(artifact["scores"], dtype=np.float64)
        indices = [id_to_index[image_id] for image_id in heldout_ids]
        target = candidate_scores[int(seed)]
        if np.isfinite(target[indices]).any():
            raise RuntimeError("A factorized OOF image was predicted more than once")
        target[indices] = values
    for seed, values in candidate_scores.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete factorized OOF scores: {seed}")
    scores["FactorizedGlobalActionRank"] = candidate_scores
    score_path = RUNS / "oof/scores.pkl"
    BASE.atomic_pickle(score_path, {"image_ids": image_ids, "scores": scores})
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "new_jobs": len(tasks),
        "trained_methods": list(CORE.TRAINED_METHODS),
        "reused_methods": list(CORE.REUSED_METHODS),
        "seeds": list(PROTOCOL["seeds"]),
        "folds": int(PROTOCOL["fold_count"]),
        "images": len(image_ids),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "selected_epoch_summary": {
            str(seed): [
                int(row["selected_epoch"])
                for row in reports
                if int(row["seed"]) == int(seed)
            ]
            for seed in PROTOCOL["seeds"]
        },
        "source_score_sha256": BASE.file_sha256(_source_score_paths()["scores"]),
        "score_sha256": BASE.file_sha256(score_path),
        "reward_context_may_cross_oof_folds": False,
        "calibration_used": False,
    }
    BASE.atomic_json(RUNS / "oof/report.json", report)
    BASE.atomic_json(
        completion_path,
        {
            "complete": True,
            "score_sha256": report["score_sha256"],
            "report_sha256": BASE.file_sha256(RUNS / "oof/report.json"),
            "calibration_used": False,
        },
    )
    BASE._TRAIN_STATE.clear()
    gc.collect()
    return report


def run_oof_evaluation() -> dict[str, Any]:
    completion = BASE.read_json(RUNS / "oof/completion.json")
    score_path = RUNS / "oof/scores.pkl"
    if completion.get("score_sha256") != BASE.file_sha256(score_path):
        raise RuntimeError("OOF score cache changed")
    _, report = BASE.evaluate_scores(
        "fit", BASE.load_pickle(score_path), output_root=RUNS / "oof_evaluation"
    )
    return report


def _ensemble_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["seed"] == "ensemble"]


def _workpoint(
    rows: Sequence[Mapping[str, Any]], method: str, mode: str, budget: float
) -> Mapping[str, Any]:
    found = [
        row
        for row in rows
        if row["method"] == method
        and row["mode"] == mode
        and math.isclose(float(row["budget"]), budget, abs_tol=1e-12)
    ]
    if len(found) != 1:
        raise RuntimeError(f"Missing result: {method}/{mode}/{budget:g}")
    return found[0]


def run_report() -> dict[str, Any]:
    completion_path = RUNS / "COMPLETED.json"
    if completion_path.is_file():
        return BASE.read_json(RUNS / "report.json")
    oof = BASE.read_json(RUNS / "oof_evaluation/report.json")
    calibration = BASE.read_json(RUNS / "calibration/report.json")
    rows = _ensemble_rows(RUNS / "oof_evaluation/results.csv")
    comparisons = {}
    for method in CORE.METHODS[1:]:
        points = []
        for mode in CORE.EXPECTED_MODES:
            for budget in CORE.EXPECTED_BUDGETS:
                baseline = _workpoint(rows, "CurrentActionValue", mode, budget)
                candidate = _workpoint(rows, method, mode, budget)
                points.append(
                    {
                        "mode": mode,
                        "budget": budget,
                        "delta_ap50_95": float(candidate["official_ap50_95"])
                        - float(baseline["official_ap50_95"]),
                        "original_count": int(candidate["original_count"]),
                        "q90_count": int(candidate["q90_count"]),
                        "selected_images": int(candidate["selected_images"]),
                    }
                )
        comparisons[method] = points
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "result_type": PROTOCOL["result_type"],
        "dataset": "coco",
        "oof_gate": oof["gate"],
        "calibration": calibration,
        "comparisons": comparisons,
        "open_future_paper_protocol": bool(
            oof["gate"]["passed"]
            and calibration.get("evaluated")
            and calibration.get("gate", {}).get("passed")
        ),
        "fit_rewards_reused_read_only": True,
        "baseline_oof_scores_reused_read_only": True,
        "detector_inference_used": False,
        "roi_used": False,
        "audit_accessed": False,
        "final_accessed": False,
        "registered_final_accessed": False,
    }
    BASE.atomic_json(RUNS / "report.json", report)
    lines = [
        "# COCO Factorized Global Action Rank OOF 诊断结果",
        "",
        "本实验训练集为 canonical train2017 fit=9,678，动作仅为 Edge/Original/Q90。"
        "上一轮 fold-contained reward 与三个基线 OOF 分数只读复用；"
        "本轮只训练 FactorizedGlobalActionRank。",
        "",
        f'OOF gate：**{str(bool(oof["gate"]["passed"])).lower()}**。',
        "",
        "## Ensemble AP50-95 / KB每图",
        "",
        "| 模式 | 方法 | 10% | 25% | 40% |",
        "|---|---|---:|---:|---:|",
    ]
    for mode in ("fixed_byte", "fixed_call"):
        for method in CORE.METHODS:
            cells = []
            for budget in CORE.EXPECTED_BUDGETS:
                row = _workpoint(rows, method, mode, budget)
                cells.append(
                    f'{float(row["official_ap50_95"]):.6f} / '
                    f'{float(row["kb_per_image"]):.2f}'
                )
            lines.append(f"| {mode} | {method} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Factorized 相对 CurrentActionValue",
            "",
            "| 模式 | 10% | 25% | 40% |",
            "|---|---:|---:|---:|",
        ]
    )
    for mode in ("fixed_byte", "fixed_call"):
        values = [
            next(
                point
                for point in comparisons["FactorizedGlobalActionRank"]
                if point["mode"] == mode
                and math.isclose(float(point["budget"]), budget)
            )["delta_ap50_95"]
            for budget in CORE.EXPECTED_BUDGETS
        ]
        lines.append(
            f'| {mode} | {values[0]:+.6f} | {values[1]:+.6f} | '
            f'{values[2]:+.6f} |'
        )
    if calibration.get("evaluated"):
        status = (
            "OOF gate通过后 calibration 已一次性评测；"
            f'calibration gate={str(bool(calibration["gate"]["passed"])).lower()}。'
        )
    else:
        status = "OOF gate未通过，calibration自动封存且未评测。"
    lines.extend(["", "## 协议结论", "", status, ""])
    notes = ROOT / "notes/results_zh.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    temporary = notes.with_name(f".{notes.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, notes)
    BASE.atomic_json(
        completion_path,
        {
            "complete": True,
            "report_sha256": BASE.file_sha256(RUNS / "report.json"),
            "notes_sha256": BASE.file_sha256(notes),
            "audit_accessed": False,
            "final_accessed": False,
            "registered_final_accessed": False,
        },
    )
    return report


STAGES = {
    "preflight": BASE.run_preflight,
    "fit_rewards": lambda: run_rewards("fit"),
    "oof": run_oof,
    "oof_evaluation": run_oof_evaluation,
    "calibration": BASE.run_calibration,
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
