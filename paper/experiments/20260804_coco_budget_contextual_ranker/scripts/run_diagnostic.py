#!/usr/bin/env python3
"""Run the gated COCO budget-conditioned ranker diagnostic."""

from __future__ import annotations

import argparse
import json
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
PROTOCOL_PATH = EXPERIMENT / "configs/protocol.json"
EXTERNAL = EXPERIMENTS / "20260728_three_dataset_dcsb_oric_moric_external_baselines"
FROZEN = EXPERIMENTS / "20260720_frozen_detector_multiranker_contextm2e"

for path in (EXTERNAL / "scripts", FROZEN / "scripts", REPO / "paper/experiments/RCMAR_v4"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_core import predict_reward_mlp  # noqa: E402
from ranker_core import (  # noqa: E402
    calibration_gate,
    fit_budget_ranker,
    predict_budget_scores,
    validate_protocol,
)
from run_dataset import _role_features, load_frozen_benchmark  # noqa: E402
from runner_core import (  # noqa: E402
    assert_hashes_unchanged,
    atomic_write_json,
    atomic_write_pickle,
    atomic_write_rows,
    capture_hashes,
    file_sha256,
    role_source_paths,
)


COMMON = FROZEN / "runs/canonical_v1/common"
BENCHMARK = FROZEN / "runs/canonical_v1/benchmark/coco"
SOURCE_RUN = EXTERNAL / "runs/coco"


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def roles_for_stage(stage: str) -> tuple[str, ...]:
    if stage == "calibration":
        return ("calibration",)
    if stage == "final":
        return ("final",)
    raise ValueError(f"Unknown stage: {stage}")


def assert_final_allowed(gate: dict[str, Any]) -> None:
    if gate.get("passed") is not True or gate.get("frozen") is not True:
        raise RuntimeError("Final is sealed because the frozen calibration gate did not pass")


def _load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _source_paths(role: str) -> list[Path]:
    paths = [
        PROTOCOL_PATH,
        SCRIPT,
        SCRIPT.parent / "ranker_core.py",
        EXTERNAL / "scripts/baseline_core.py",
        EXTERNAL / "scripts/run_dataset.py",
        EXTERNAL / "scripts/runner_core.py",
        SOURCE_RUN / "rewards/fit_artifacts.npz",
        SOURCE_RUN / "rewards/completion.json",
        SOURCE_RUN / "evaluation/oric_mlp.pkl",
        SOURCE_RUN / "evaluation/moric_mlp.pkl",
        SOURCE_RUN / "evaluation/completion.json",
        BENCHMARK / "models.json",
    ]
    paths.extend(role_source_paths(COMMON, "coco", role))
    return paths


def _load_fit_artifact() -> tuple[np.ndarray, np.ndarray, list[int]]:
    with np.load(SOURCE_RUN / "rewards/fit_artifacts.npz") as artifact:
        features = np.asarray(artifact["features"], dtype=np.float32)
        rewards = np.asarray(artifact["rewards"], dtype=np.float64)
        category_ids = list(map(int, artifact["category_ids"]))
    if features.shape[0] != 9678 or rewards.shape != (9678,):
        raise RuntimeError("Frozen COCO fit artifact population changed")
    return features, rewards, category_ids


def _external_checkpoints() -> dict[str, Any]:
    return {
        "ORIC-MLP": _load_pickle(SOURCE_RUN / "evaluation/oric_mlp.pkl"),
        "MORIC-MLP": _load_pickle(SOURCE_RUN / "evaluation/moric_mlp.pkl"),
    }


def _original_rows(
    benchmark: Any,
    data: dict[str, Any],
    frozen_models: dict[str, Any],
    method_scores: dict[str, np.ndarray],
    budgets: list[float],
) -> list[dict[str, Any]]:
    action_map = {
        int(sample["image_id"]): "original" for sample in data["samples"]
    }
    rows = []
    for method, scores_by_budget in method_scores.items():
        values = np.asarray(scores_by_budget, dtype=np.float64)
        if values.ndim == 1:
            values = np.repeat(values[None, :], len(budgets), axis=0)
        if values.shape != (len(budgets), len(data["samples"])):
            raise RuntimeError(f"Unexpected score matrix for {method}: {values.shape}")
        for budget, scores in zip(budgets, values):
            selected = benchmark.top_k_actions(data, scores, action_map, budget)
            row = benchmark.evaluate(
                method, "fixed_call", budget, data, selected, frozen_models
            )
            row.update(
                {
                    "payload_policy": "Original",
                    "diagnostic_phase": "objective_only_ap50_reward",
                }
            )
            rows.append(row)
    return rows


def _ap_by_budget(rows: list[dict[str, Any]], method: str) -> dict[float, float]:
    return {
        float(row["nominal_budget"]): float(row["official_map50_95"])
        for row in rows
        if row["method"] == method
    }


def _best_external_by_budget(rows: list[dict[str, Any]]) -> dict[float, float]:
    budgets = sorted({float(row["nominal_budget"]) for row in rows})
    return {
        budget: max(
            float(row["official_map50_95"])
            for row in rows
            if row["method"] in {"ORIC-MLP", "MORIC-MLP"}
            and float(row["nominal_budget"]) == budget
        )
        for budget in budgets
    }


def run_calibration() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    validate_protocol(protocol)
    completion_path = RUNS / "calibration/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "calibration/report.json")
    before = capture_hashes(_source_paths("calibration"))
    fit_features, fit_rewards, category_ids = _load_fit_artifact()
    budgets = list(map(float, protocol["budgets"]))
    checkpoints = [
        fit_budget_ranker(
            fit_features,
            fit_rewards,
            budgets,
            protocol["model"],
            seed=int(seed),
        )
        for seed in protocol["seeds"]
    ]

    benchmark = load_frozen_benchmark(COMMON)
    benchmark.COMMON = COMMON
    calibration = benchmark.load_role("coco", "calibration")
    calibration_features = _role_features(calibration, category_ids, top_k=25)
    candidate_scores = np.mean(
        [
            predict_budget_scores(checkpoint, calibration_features, budgets)
            for checkpoint in checkpoints
        ],
        axis=0,
    )
    external_scores = {
        method: predict_reward_mlp(checkpoint, calibration_features)
        for method, checkpoint in _external_checkpoints().items()
    }
    frozen_models = read_json(BENCHMARK / "models.json")
    rows = _original_rows(
        benchmark,
        calibration,
        frozen_models,
        {
            **external_scores,
            "BudgetContext-AP50-Listwise": candidate_scores,
        },
        budgets,
    )
    candidate_ap = _ap_by_budget(rows, "BudgetContext-AP50-Listwise")
    baseline_ap = _best_external_by_budget(rows)
    gate_config = protocol["gate"]
    gate = calibration_gate(
        candidate_ap,
        baseline_ap,
        minimum_mean_ap_gain=float(gate_config["minimum_mean_ap_gain"]),
        maximum_budget_ap_loss=float(gate_config["maximum_budget_ap_loss"]),
    )
    gate.update(
        {
            "frozen": True,
            "candidate": "BudgetContext-AP50-Listwise",
            "baseline": "best_of_ORIC_MLP_and_MORIC_MLP",
            "source_reward": protocol["source_reward"],
        }
    )
    assert_hashes_unchanged(before)
    root = RUNS / "calibration"
    results_path = root / "results.csv"
    model_path = root / "ensemble.pkl"
    gate_path = root / "gate.json"
    atomic_write_rows(results_path, rows, RUNS)
    atomic_write_pickle(model_path, checkpoints, RUNS)
    atomic_write_json(gate_path, gate, RUNS)
    report = {
        "protocol_id": protocol["protocol_id"],
        "role": "calibration",
        "fit_images": len(fit_rewards),
        "calibration_images": len(calibration["samples"]),
        "selected_epochs": [int(item["selected_epoch"]) for item in checkpoints],
        "fit_validation_ndcg": [
            float(item["fit_validation_ndcg"]) for item in checkpoints
        ],
        "candidate_ap": {str(key): value for key, value in candidate_ap.items()},
        "baseline_ap": {str(key): value for key, value in baseline_ap.items()},
        "gate": gate,
        "source_sha256": before,
    }
    report_path = root / "report.json"
    atomic_write_json(report_path, report, RUNS)
    atomic_write_json(
        completion_path,
        {
            "stage": "calibration",
            "gate_sha256": file_sha256(gate_path),
            "model_sha256": file_sha256(model_path),
            "results_sha256": file_sha256(results_path),
            "report_sha256": file_sha256(report_path),
            "complete": True,
        },
        RUNS,
    )
    return report


def _validated_calibration_bundle() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = RUNS / "calibration"
    completion = read_json(root / "completion.json")
    expected = {
        "gate_sha256": file_sha256(root / "gate.json"),
        "model_sha256": file_sha256(root / "ensemble.pkl"),
        "results_sha256": file_sha256(root / "results.csv"),
        "report_sha256": file_sha256(root / "report.json"),
    }
    if completion.get("complete") is not True or any(
        completion.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("Calibration bundle is incomplete or changed")
    gate = read_json(root / "gate.json")
    assert_final_allowed(gate)
    return gate, _load_pickle(root / "ensemble.pkl")


def run_final() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    validate_protocol(protocol)
    completion_path = RUNS / "final/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "final/report.json")
    gate, checkpoints = _validated_calibration_bundle()
    before = capture_hashes(
        [
            *_source_paths("final"),
            RUNS / "calibration/gate.json",
            RUNS / "calibration/ensemble.pkl",
            RUNS / "calibration/completion.json",
        ]
    )
    _, _, category_ids = _load_fit_artifact()
    budgets = list(map(float, protocol["budgets"]))
    benchmark = load_frozen_benchmark(COMMON)
    benchmark.COMMON = COMMON
    final = benchmark.load_role("coco", "final")
    features = _role_features(final, category_ids, top_k=25)
    candidate_scores = np.mean(
        [predict_budget_scores(checkpoint, features, budgets) for checkpoint in checkpoints],
        axis=0,
    )
    external_scores = {
        method: predict_reward_mlp(checkpoint, features)
        for method, checkpoint in _external_checkpoints().items()
    }
    rows = _original_rows(
        benchmark,
        final,
        read_json(BENCHMARK / "models.json"),
        {**external_scores, gate["candidate"]: candidate_scores},
        budgets,
    )
    candidate_ap = _ap_by_budget(rows, gate["candidate"])
    baseline_ap = _best_external_by_budget(rows)
    candidate_rows = {
        float(row["nominal_budget"]): row
        for row in rows
        if row["method"] == gate["candidate"]
    }
    baseline_rows = {
        budget: max(
            (
                row
                for row in rows
                if row["method"] in {"ORIC-MLP", "MORIC-MLP"}
                and float(row["nominal_budget"]) == budget
            ),
            key=lambda row: float(row["official_map50_95"]),
        )
        for budget in budgets
    }
    comparisons = {
        str(budget): {
            "candidate_ap": candidate_ap[budget],
            "baseline_ap": baseline_ap[budget],
            "ap_gain": candidate_ap[budget] - baseline_ap[budget],
            "candidate_kb": float(candidate_rows[budget]["request_kb_per_image"]),
            "baseline_kb": float(baseline_rows[budget]["request_kb_per_image"]),
            "pareto_win": candidate_ap[budget] > baseline_ap[budget]
            and float(candidate_rows[budget]["request_kb_per_image"])
            <= float(baseline_rows[budget]["request_kb_per_image"]) + 1e-12,
        }
        for budget in budgets
    }
    success = all(item["pareto_win"] for item in comparisons.values())
    assert_hashes_unchanged(before)
    root = RUNS / "final"
    results_path = root / "results.csv"
    atomic_write_rows(results_path, rows, RUNS)
    report = {
        "protocol_id": protocol["protocol_id"],
        "role": "final",
        "final_evaluated_once": True,
        "candidate": gate["candidate"],
        "comparisons": comparisons,
        "success_all_budgets": success,
        "source_sha256": before,
    }
    report_path = root / "report.json"
    atomic_write_json(report_path, report, RUNS)
    atomic_write_json(
        completion_path,
        {
            "stage": "final",
            "success_all_budgets": success,
            "results_sha256": file_sha256(results_path),
            "report_sha256": file_sha256(report_path),
            "complete": True,
        },
        RUNS,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("calibration", "final"))
    args = parser.parse_args()
    report = run_calibration() if args.stage == "calibration" else run_final()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
