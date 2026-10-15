#!/usr/bin/env python3
"""Audit learned ROI choices against the true byte-price frontier."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import pickle

import numpy as np
import torch

import run_experiment as runner


ROOT = Path(__file__).resolve().parents[1]


def _true_local_action(
    sample: dict,
    rewards: dict[str, float],
    shadow_price: float,
) -> tuple[str, float, dict[str, float]]:
    utilities = {"edge": 0.0}
    for action, reward in rewards.items():
        payload = sample["payload_bytes"].get(action)
        if bool(sample["available"].get(action, False)) and payload is not None:
            utilities[action] = float(reward) - float(shadow_price) * float(payload)
    action = max(
        utilities,
        key=lambda name: (
            utilities[name],
            -float(sample["payload_bytes"].get(name, 0) or 0),
            name,
        ),
    )
    return action, float(utilities[action]), utilities


def run() -> dict:
    completion = ROOT / "runs/confirm/roi_frontier_audit_completion.json"
    output = ROOT / "runs/confirm/roi_frontier_audit.json"
    if completion.is_file():
        return runner.read_json(output)
    confirm = runner.read_json(ROOT / "runs/confirm/evaluation_report.json")
    if not bool(confirm["gate"]["passed"]):
        raise RuntimeError("The three-seed candidate did not pass")
    with (ROOT / "runs/confirm/scores.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    label = "OQR/BoundaryPathDistill"
    matrices = artifact["scores"][label]
    ensemble = np.mean(list(matrices.values()), axis=0)
    records, tensors = runner._screen_tensor_data("fit", "OQR")
    image_ids = [int(record["image_id"]) for record in records]
    if image_ids != list(map(int, artifact["image_ids"]["OQR"])):
        raise RuntimeError("Confirmation and ROI record order differ")
    score_map = runner._score_map(records, ensemble)
    rewards = runner._reward_map_for_space("fit", "OQR")
    data = runner.REL.load_role("fit")
    samples = data["samples"]
    sample_by_id = {int(sample["image_id"]): sample for sample in samples}
    total_original = sum(
        int(sample["payload_bytes"]["original"]) for sample in samples
    )
    scale = total_original / len(samples)

    prices, price_mask = runner.CORE.frontier_probe_prices(
        tensors["raw_target"],
        tensors["cost_ratio"],
        tensors["action_mask"],
        relative_offset=float(runner.PROTOCOL["model"]["frontier_relative_offset"]),
        maximum_price=float(runner.PROTOCOL["model"]["frontier_maximum_price"]),
    )
    targets = runner.CORE.frontier_target_actions(
        tensors["raw_target"],
        tensors["cost_ratio"],
        tensors["action_mask"],
        prices,
    )
    roi_frontier = {}
    frontier_actions = {}
    for row, image_id in enumerate(image_ids):
        active_columns = set(
            map(int, targets[row][price_mask[row]].detach().cpu().tolist())
        )
        actions = [str(action["action_id"]) for action in records[row]["actions"]]
        active = {actions[column] for column in active_columns}
        frontier_actions[image_id] = active
        roi_frontier[image_id] = "roi" in active

    available_roi = {
        image_id
        for image_id, sample in sample_by_id.items()
        if bool(sample["available"].get("roi", False))
    }
    rows = []
    for budget in runner.PROTOCOL["budgets"]:
        capacity = float(budget) * total_original
        teacher = runner.ACTION.schedule_fixed_byte(
            samples, rewards, capacity_bytes=capacity
        )
        learned = runner.ACTION.schedule_fixed_byte(
            samples, score_map, capacity_bytes=capacity
        )
        teacher_selection = {
            int(image_id): str(action)
            for image_id, action in teacher["selection"].items()
        }
        learned_selection = {
            int(image_id): str(action)
            for image_id, action in learned["selection"].items()
        }
        teacher_roi = {
            image_id for image_id, action in teacher_selection.items() if action == "roi"
        }
        learned_roi = {
            image_id for image_id, action in learned_selection.items() if action == "roi"
        }
        common_roi = teacher_roi & learned_roi
        local_optimal = 0
        utility_gaps = []
        local_action_counts: Counter[str] = Counter()
        for image_id in learned_roi:
            best_action, best_utility, utilities = _true_local_action(
                sample_by_id[image_id],
                rewards[image_id],
                float(teacher["lambda"]),
            )
            local_action_counts[best_action] += 1
            roi_utility = float(utilities["roi"])
            gap = best_utility - roi_utility
            utility_gaps.append(gap)
            local_optimal += int(gap <= 1e-12)
        rows.append(
            {
                "budget": float(budget),
                "teacher_lambda_absolute": float(teacher["lambda"]),
                "teacher_lambda_normalized": float(teacher["lambda"]) * scale,
                "teacher_roi_count": len(teacher_roi),
                "learned_roi_count": len(learned_roi),
                "exact_roi_precision": len(common_roi) / max(1, len(learned_roi)),
                "exact_roi_recall": len(common_roi) / max(1, len(teacher_roi)),
                "learned_roi_true_local_utility_accuracy": local_optimal
                / max(1, len(learned_roi)),
                "learned_roi_on_any_true_frontier_rate": sum(
                    roi_frontier[image_id] for image_id in learned_roi
                )
                / max(1, len(learned_roi)),
                "learned_roi_mean_true_utility_gap": float(
                    np.mean(utility_gaps) if utility_gaps else 0.0
                ),
                "learned_roi_p90_true_utility_gap": float(
                    np.quantile(utility_gaps, 0.9) if utility_gaps else 0.0
                ),
                "true_local_action_for_learned_roi": dict(local_action_counts),
            }
        )
    report = {
        "protocol_id": runner.PROTOCOL["protocol_id"],
        "role": "fit_oof",
        "candidate": label,
        "available_roi_images": len(available_roi),
        "roi_on_any_true_frontier_images": sum(
            roi_frontier[image_id] for image_id in available_roi
        ),
        "roi_on_any_true_frontier_rate": sum(
            roi_frontier[image_id] for image_id in available_roi
        )
        / max(1, len(available_roi)),
        "rows": rows,
        "important_correction": (
            "The earlier suboptimal_roi_rate compared raw reward only. This audit "
            "uses true reward-minus-byte-price utility and exact teacher ROI overlap."
        ),
        "calibration_accessed": False,
        "val2017_accessed": False,
        "final_accessed": False,
    }
    runner.atomic_json(output, report)
    runner.atomic_json(
        completion,
        {"complete": True, "report_sha256": runner.file_sha256(output)},
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
