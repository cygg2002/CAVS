#!/usr/bin/env python3
"""Evaluate set-valued consensus over frozen AFPD OOF frontiers."""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
V1 = REPO / "paper/experiments/20260809_coco_frontier_aligned_actionvalue_exploration_v1"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())
RUNS = ROOT / "runs"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


CORE = _load("frontier_consensus_core", ROOT / "scripts/core.py")
SOURCE = _load("frontier_consensus_v1_runner", V1 / "scripts/run_experiment.py")


def _frontier_membership(
    values: torch.Tensor,
    costs: torch.Tensor,
    availability: torch.Tensor,
) -> torch.Tensor:
    prices, price_mask = SOURCE.CORE.frontier_probe_prices(
        values,
        costs,
        availability,
        relative_offset=float(PROTOCOL["relative_offset"]),
        maximum_price=float(PROTOCOL["maximum_price"]),
    )
    targets = SOURCE.CORE.frontier_target_actions(
        values, costs, availability, prices
    )
    membership = (
        F.one_hot(targets, num_classes=values.shape[1]).to(torch.bool)
        & price_mask[:, :, None]
    ).any(dim=1)
    return membership & availability.to(torch.bool)


def _action_columns(records: list[dict[str, Any]], width: int) -> dict[str, np.ndarray]:
    output = {
        action: np.zeros((len(records), width), dtype=bool)
        for action in PROTOCOL["action_space"]
    }
    allowed = set(output)
    for row, record in enumerate(records):
        names = [str(action["action_id"]) for action in record["actions"]]
        if not names or names[0] != "edge" or len(names) != len(set(names)):
            raise RuntimeError(f"Invalid action IDs for image {record['image_id']}")
        if not set(names).issubset(allowed):
            raise RuntimeError(f"Unexpected action for image {record['image_id']}")
        for column, name in enumerate(names):
            output[name][row, column] = True
    return output


_EVAL_STATE: dict[str, Any] = {}


def _evaluate_task(task: Mapping[str, Any]) -> dict[str, Any]:
    data = _EVAL_STATE["data"]
    rewards = _EVAL_STATE["rewards"]
    selection = {int(key): str(value) for key, value in task["selection"].items()}
    predictions, _ = SOURCE.REL.canonical.mixed_predictions(data, selection, 0.001)
    official = data["evaluator"].evaluate(predictions)
    request_bytes = sum(
        int(data["sample_by_id"][image_id]["payload_bytes"][action])
        for image_id, action in selection.items()
    )
    counts = Counter(selection.values())
    expected_calls = int(
        math.floor(len(data["samples"]) * float(task["budget"]) + 0.5)
    )
    mode = str(task["mode"])
    capacity = float(task["capacity_bytes"])
    utilization = (
        len(selection) / max(1, expected_calls)
        if mode == "fixed_call"
        else request_bytes / max(1.0, capacity)
    )
    selected_rewards = [
        float(rewards[image_id][action]) for image_id, action in selection.items()
    ]
    return {
        "role": "fit_oof",
        "method": str(task["method"]),
        "mode": mode,
        "budget": float(task["budget"]),
        "official_ap50_95": float(official["map50_95"]),
        "official_ap50": float(official["map50"]),
        "official_ap75": float(official["map75"]),
        "kb_per_image": request_bytes / 1024.0 / len(data["samples"]),
        "request_bytes": int(request_bytes),
        "capacity_bytes": capacity,
        "capacity_utilization": float(utilization),
        "capacity_ok": bool(request_bytes <= capacity + 1e-6)
        if mode == "fixed_byte"
        else bool(len(selection) == expected_calls),
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["roi"]),
        "selected_teacher_reward": float(sum(selected_rewards)),
        "teacher_value_ceiling": float(task["teacher_value"]),
        "teacher_regret": float(task["teacher_value"] - sum(selected_rewards)),
        "scheduler_lambda": float(task.get("scheduler_lambda", 0.0)),
    }


def run() -> dict[str, Any]:
    completion = RUNS / "evaluation/completion.json"
    report_path = RUNS / "evaluation/report.json"
    if completion.is_file():
        return json.loads(report_path.read_text())
    if int(PROTOCOL["expected_images"]) != 9678:
        raise RuntimeError("Canonical fit population changed")
    if PROTOCOL["calibration_accessed"] or PROTOCOL["val2017_accessed"] or PROTOCOL["final_accessed"]:
        raise RuntimeError("A sealed role was enabled")
    source_report = json.loads((V1 / "runs/confirm/evaluation_report.json").read_text())
    if not bool(source_report["gate"]["passed"]):
        raise RuntimeError("Frozen AFPD source did not pass its confirmation gate")
    score_path = V1 / "runs/confirm/scores.pkl"
    with score_path.open("rb") as handle:
        artifact = pickle.load(handle)
    if tuple(map(int, artifact["seeds"])) != tuple(map(int, PROTOCOL["seeds"])):
        raise RuntimeError("AFPD seed set changed")
    label = "OQR/BoundaryPathDistill"
    matrices = [
        np.asarray(artifact["scores"][label][int(seed)], dtype=np.float64)
        for seed in PROTOCOL["seeds"]
    ]
    mean_scores = np.mean(matrices, axis=0)
    records, tensors = SOURCE._screen_tensor_data("fit", "OQR")
    image_ids = [int(record["image_id"]) for record in records]
    if image_ids != list(map(int, artifact["image_ids"]["OQR"])):
        raise RuntimeError("AFPD OOF score order changed")
    availability = tensors["action_mask"].to(torch.bool)
    seed_membership = []
    for matrix in matrices:
        seed_membership.append(
            _frontier_membership(
                torch.as_tensor(matrix, dtype=torch.float32),
                tensors["cost_ratio"],
                availability,
            ).cpu().numpy()
        )
    seed_membership_array = np.stack(seed_membership, axis=0)
    columns = _action_columns(records, mean_scores.shape[1])
    mandatory = columns["edge"] | columns["original"]
    consensus = CORE.majority_mask(
        seed_membership_array,
        availability.cpu().numpy(),
        minimum_votes=int(PROTOCOL["minimum_seed_votes"]),
        mandatory_columns=mandatory,
    )
    vote_counts = seed_membership_array.sum(axis=0)
    voted_consensus = (
        vote_counts >= int(PROTOCOL["minimum_seed_votes"])
    ) & availability.cpu().numpy()
    score_maps = {
        "AFPD-Mean": SOURCE._score_map(records, mean_scores),
        "ROI-FCS-2of3": CORE.prune_score_map(
            records, mean_scores, consensus, roi_only=True
        ),
        "AltAction-FCS-2of3": CORE.prune_score_map(
            records, mean_scores, consensus, roi_only=False
        ),
    }
    if tuple(score_maps) != tuple(PROTOCOL["methods"]):
        raise RuntimeError("Frozen method matrix changed")

    teacher_membership = _frontier_membership(
        tensors["raw_target"], tensors["cost_ratio"], availability
    ).cpu().numpy()
    audit = {}
    available_np = availability.cpu().numpy()
    for action, action_columns in columns.items():
        possible = available_np & action_columns
        predicted = voted_consensus & action_columns
        retained = consensus & action_columns
        truth = teacher_membership & action_columns
        true_positive = int((predicted & truth).sum())
        audit[action] = {
            "available": int(possible.sum()),
            "teacher_frontier": int(truth.sum()),
            "voted_frontier_consensus": int(predicted.sum()),
            "retained_by_policy": int(retained.sum()),
            "true_positive": true_positive,
            "precision": true_positive / max(1, int(predicted.sum())),
            "recall": true_positive / max(1, int(truth.sum())),
            "vote_histogram": {
                str(votes): int(
                    (
                        (vote_counts == votes)
                        & action_columns
                    ).sum()
                )
                for votes in range(len(matrices) + 1)
            },
        }

    data = SOURCE.REL.load_role("fit")
    samples = data["samples"]
    rewards = SOURCE._reward_map_for_space("fit", "OQR")
    original_total = sum(
        int(sample["payload_bytes"]["original"]) for sample in samples
    )
    tasks = []
    for method, score_map in score_maps.items():
        for budget in PROTOCOL["budgets"]:
            capacity = float(budget) * original_total
            for mode in PROTOCOL["modes"]:
                learned = (
                    SOURCE.ACTION.schedule_fixed_byte(
                        samples, score_map, capacity_bytes=capacity
                    )
                    if mode == "fixed_byte"
                    else SOURCE.ACTION.schedule_fixed_call(
                        samples, score_map, budget=float(budget)
                    )
                )
                teacher = (
                    SOURCE.ACTION.schedule_fixed_byte(
                        samples, rewards, capacity_bytes=capacity
                    )
                    if mode == "fixed_byte"
                    else SOURCE.ACTION.schedule_fixed_call(
                        samples, rewards, budget=float(budget)
                    )
                )
                tasks.append(
                    {
                        "method": method,
                        "mode": mode,
                        "budget": float(budget),
                        "capacity_bytes": capacity,
                        "selection": {
                            int(image_id): str(action)
                            for image_id, action in learned["selection"].items()
                        },
                        "teacher_value": float(teacher["total_value"]),
                        "scheduler_lambda": float(learned.get("lambda", 0.0)),
                    }
                )
    _EVAL_STATE.clear()
    _EVAL_STATE.update({"data": data, "rewards": rewards})
    started = time.time()
    context = mp.get_context("fork")
    workers = min(int(PROTOCOL["evaluation_workers"]), len(tasks))
    with context.Pool(processes=workers) as pool:
        rows = list(pool.imap_unordered(_evaluate_task, tasks, chunksize=1))
    rows.sort(
        key=lambda row: (
            PROTOCOL["methods"].index(row["method"]),
            PROTOCOL["modes"].index(row["mode"]),
            float(row["budget"]),
        )
    )
    if len(rows) != 18 or not all(row["capacity_ok"] for row in rows):
        raise RuntimeError("Evaluation matrix is incomplete or violates capacity")
    baseline = {
        (row["mode"], float(row["budget"])): float(row["official_ap50_95"])
        for row in rows
        if row["method"] == "AFPD-Mean"
    }
    gates = {}
    for method in PROTOCOL["methods"][1:]:
        candidate = {
            (row["mode"], float(row["budget"])): float(row["official_ap50_95"])
            for row in rows
            if row["method"] == method
        }
        gates[method] = CORE.screen_gate(baseline, candidate, PROTOCOL["gate"])

    results_path = RUNS / "evaluation/results.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    SOURCE.atomic_csv(results_path, rows)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": "fit_oof",
        "rows": len(rows),
        "workers": workers,
        "elapsed_seconds": time.time() - started,
        "frontier_consensus_audit": audit,
        "gates": gates,
        "passed": any(value["passed"] for value in gates.values()),
        "source_score_sha256": SOURCE.file_sha256(score_path),
        "access": {
            "fit": True,
            "calibration": False,
            "audit": False,
            "val2017": False,
            "final": False,
        },
        "interpretation_boundary": (
            "This is a fit-only OOF mechanism screen. Generic ensemble and "
            "set-valued uncertainty ideas are established prior art."
        ),
    }
    SOURCE.atomic_json(report_path, report)
    SOURCE.atomic_json(
        completion,
        {
            "complete": True,
            "passed": report["passed"],
            "results_sha256": SOURCE.file_sha256(results_path),
            "report_sha256": SOURCE.file_sha256(report_path),
        },
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
