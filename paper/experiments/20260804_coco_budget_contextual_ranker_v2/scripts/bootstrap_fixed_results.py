#!/usr/bin/env python3
"""Paired-bootstrap the already-frozen COCO final selections."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np


SCRIPT = Path(__file__).resolve()
EXPERIMENT = SCRIPT.parents[1]
EXPERIMENTS = SCRIPT.parents[2]
REPO = SCRIPT.parents[4]
RUNS = EXPERIMENT / "runs"
EXTERNAL = EXPERIMENTS / "20260728_three_dataset_dcsb_oric_moric_external_baselines"
FROZEN = EXPERIMENTS / "20260720_frozen_detector_multiranker_contextm2e"
COMMON = FROZEN / "runs/canonical_v1/common"
SOURCE_RUN = EXTERNAL / "runs/coco"

for path in (EXTERNAL / "scripts", FROZEN / "scripts", REPO / "paper/experiments/RCMAR_v4"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_core import predict_reward_mlp  # noqa: E402
from final_paper_evaluation_core import (  # noqa: E402
    build_bootstrap_coco_sample,
    deterministic_bootstrap_indices,
)
from ranker_core import predict_budget_scores  # noqa: E402
from resume_coco_parallel_bootstrap import in_memory_ap  # noqa: E402
from run_dataset import _role_features, load_frozen_benchmark  # noqa: E402
from run_multiranker_benchmark import mixed_predictions  # noqa: E402
from runner_core import (  # noqa: E402
    assert_hashes_unchanged,
    atomic_write_json,
    atomic_write_rows,
    capture_hashes,
    file_sha256,
    role_source_paths,
)


_STATE: dict[str, Any] = {}


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def summarize_bootstrap(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Bootstrap values must be one finite vector")
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
        "probability_gt_zero": float(np.mean(array > 0.0)),
    }


def selection_overlap(left: set[int], right: set[int]) -> dict[str, float]:
    if not left or not right:
        raise ValueError("Selection overlap requires two non-empty sets")
    overlap = len(left & right)
    return {
        "jaccard": overlap / len(left | right),
        "overlap_over_smaller": overlap / min(len(left), len(right)),
    }


def _worker(item: tuple[float, int, np.ndarray]) -> tuple[float, int, float, float]:
    budget, replicate, indices = item
    state = _STATE[budget]
    source_ids = [_STATE["image_ids"][int(index)] for index in indices]
    sampled_gt, prediction_sets = build_bootstrap_coco_sample(
        _STATE["ground_truth"],
        state["prediction_sets"],
        sampled_image_ids=source_ids,
    )
    baseline_ap = in_memory_ap(sampled_gt, prediction_sets["baseline"])
    candidate_ap = in_memory_ap(sampled_gt, prediction_sets["candidate"])
    kb_delta = (
        sum(state["candidate_cost"][image_id] for image_id in source_ids)
        - sum(state["baseline_cost"][image_id] for image_id in source_ids)
    ) / len(source_ids) / 1024.0
    return budget, replicate, candidate_ap - baseline_ap, kb_delta


def _actions(data: dict[str, Any], scores: np.ndarray, budget: float) -> dict[int, str]:
    action_map = {
        int(sample["image_id"]): "original" for sample in data["samples"]
    }
    count = int(np.floor(len(scores) * float(budget) + 0.5))
    order = np.argsort(-np.asarray(scores), kind="mergesort")[:count]
    return {
        int(data["samples"][index]["image_id"]): action_map[
            int(data["samples"][index]["image_id"])
        ]
        for index in order
    }


def _best_baselines() -> dict[float, str]:
    rows = []
    with (RUNS / "final/results.csv").open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    return {
        budget: max(
            (
                row
                for row in rows
                if row["method"] in {"ORIC-MLP", "MORIC-MLP"}
                and float(row["nominal_budget"]) == budget
            ),
            key=lambda row: float(row["official_map50_95"]),
        )["method"]
        for budget in (0.1, 0.25, 0.4)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    output = RUNS / "final/bootstrap"
    completion = output / "completion.json"
    if completion.is_file():
        print(json.dumps(read_json(output / "report.json"), indent=2))
        return

    source_paths = [
        SCRIPT,
        EXPERIMENT / "configs/protocol.json",
        EXPERIMENT / "scripts/ranker_core.py",
        RUNS / "calibration/ensemble.pkl",
        RUNS / "calibration/gate.json",
        RUNS / "final/results.csv",
        RUNS / "final/completion.json",
        SOURCE_RUN / "rewards/fit_artifacts.npz",
        SOURCE_RUN / "evaluation/oric_mlp.pkl",
        SOURCE_RUN / "evaluation/moric_mlp.pkl",
    ]
    source_paths.extend(role_source_paths(COMMON, "coco", "final"))
    before = capture_hashes(source_paths)

    budgets = [0.1, 0.25, 0.4]
    with np.load(SOURCE_RUN / "rewards/fit_artifacts.npz") as artifact:
        category_ids = list(map(int, artifact["category_ids"]))
    benchmark = load_frozen_benchmark(COMMON)
    benchmark.COMMON = COMMON
    final = benchmark.load_role("coco", "final")
    features = _role_features(final, category_ids, top_k=25)
    checkpoints = load_pickle(RUNS / "calibration/ensemble.pkl")
    candidate_scores = np.mean(
        [predict_budget_scores(checkpoint, features, budgets) for checkpoint in checkpoints],
        axis=0,
    )
    external_scores = {
        method: predict_reward_mlp(
            load_pickle(SOURCE_RUN / f"evaluation/{method.split('-')[0].lower()}_mlp.pkl"),
            features,
        )
        for method in ("ORIC-MLP", "MORIC-MLP")
    }
    baseline_methods = _best_baselines()
    image_ids = list(map(int, final["image_ids"]))
    ground_truth = read_json(Path(final["gt_path"]))
    sample_by_id = final["sample_by_id"]

    global _STATE
    _STATE = {"image_ids": image_ids, "ground_truth": ground_truth}
    overlap = {}
    for row, budget in enumerate(budgets):
        baseline_method = baseline_methods[budget]
        baseline_actions = _actions(final, external_scores[baseline_method], budget)
        candidate_actions = _actions(final, candidate_scores[row], budget)
        baseline_predictions, _ = mixed_predictions(final, baseline_actions, 0.001)
        candidate_predictions, _ = mixed_predictions(final, candidate_actions, 0.001)
        _STATE[budget] = {
            "prediction_sets": {
                "baseline": baseline_predictions,
                "candidate": candidate_predictions,
            },
            "baseline_cost": {
                image_id: int(sample_by_id[image_id]["payload_bytes"]["original"])
                if image_id in baseline_actions
                else 0
                for image_id in image_ids
            },
            "candidate_cost": {
                image_id: int(sample_by_id[image_id]["payload_bytes"]["original"])
                if image_id in candidate_actions
                else 0
                for image_id in image_ids
            },
            "baseline_method": baseline_method,
        }
        overlap[str(budget)] = selection_overlap(
            set(baseline_actions), set(candidate_actions)
        )

    sampled = deterministic_bootstrap_indices(
        len(image_ids), replicates=int(args.replicates), seed=20260804
    )
    tasks = [
        (budget, replicate, indices)
        for budget in budgets
        for replicate, indices in enumerate(sampled)
    ]
    context = mp.get_context("fork")
    with context.Pool(processes=max(1, int(args.workers))) as pool:
        results = list(pool.imap_unordered(_worker, tasks, chunksize=1))

    rows = []
    for budget in budgets:
        selected = sorted(
            (item for item in results if item[0] == budget), key=lambda item: item[1]
        )
        ap = summarize_bootstrap(np.asarray([item[2] for item in selected]))
        kb = summarize_bootstrap(np.asarray([item[3] for item in selected]))
        rows.append(
            {
                "budget": budget,
                "baseline": _STATE[budget]["baseline_method"],
                "replicates": int(args.replicates),
                "ap_delta_mean": ap["mean"],
                "ap_delta_ci_low": ap["ci_low"],
                "ap_delta_ci_high": ap["ci_high"],
                "ap_probability_gt_zero": ap["probability_gt_zero"],
                "kb_delta_mean": kb["mean"],
                "kb_delta_ci_low": kb["ci_low"],
                "kb_delta_ci_high": kb["ci_high"],
                **overlap[str(budget)],
            }
        )
    assert_hashes_unchanged(before)
    results_path = output / "results.csv"
    atomic_write_rows(results_path, rows, RUNS)
    report = {
        "fixed_final_selections": True,
        "strategy_refit_in_bootstrap": False,
        "bootstrap_unit": "image",
        "replicates": int(args.replicates),
        "rows": rows,
        "source_sha256": before,
    }
    report_path = output / "report.json"
    atomic_write_json(report_path, report, RUNS)
    atomic_write_json(
        completion,
        {
            "results_sha256": file_sha256(results_path),
            "report_sha256": file_sha256(report_path),
            "complete": True,
        },
        RUNS,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
