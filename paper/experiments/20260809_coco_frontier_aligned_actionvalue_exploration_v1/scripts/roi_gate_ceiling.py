#!/usr/bin/env python3
"""GT-aware go/no-go ceiling for a frontier-occupancy ROI gate."""

from __future__ import annotations

from collections import Counter
import json
import multiprocessing as mp
from pathlib import Path
import pickle

import numpy as np

import run_experiment as runner
from roi_frontier_audit import _true_local_action


ROOT = Path(__file__).resolve().parents[1]
_STATE = {}


def _evaluate(task: dict) -> dict:
    selection = {int(key): str(value) for key, value in task["selection"].items()}
    predictions, _ = runner.REL.canonical.mixed_predictions(
        _STATE["data"], selection, 0.001
    )
    metric = _STATE["data"]["evaluator"].evaluate(predictions)
    counts = Counter(selection.values())
    return {
        "policy": task["policy"],
        "budget": float(task["budget"]),
        "official_ap50_95": float(metric["map50_95"]),
        "official_ap50": float(metric["map50"]),
        "official_ap75": float(metric["map75"]),
        "selected_images": len(selection),
        "original_count": int(counts["original"]),
        "q90_count": int(counts["q90"]),
        "roi_count": int(counts["roi"]),
        "request_bytes": int(task["request_bytes"]),
        "capacity_bytes": float(task["capacity_bytes"]),
    }


def run() -> dict:
    completion = ROOT / "runs/confirm/roi_gate_ceiling_completion.json"
    output = ROOT / "runs/confirm/roi_gate_ceiling.json"
    if completion.is_file():
        return runner.read_json(output)
    with (ROOT / "runs/confirm/scores.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    label = "OQR/BoundaryPathDistill"
    ensemble = np.mean(list(artifact["scores"][label].values()), axis=0)
    records, tensors = runner._screen_tensor_data("fit", "OQR")
    score_map = runner._score_map(records, ensemble)
    rewards = runner._reward_map_for_space("fit", "OQR")
    data = runner.REL.load_role("fit")
    samples = data["samples"]
    sample_by_id = {int(sample["image_id"]): sample for sample in samples}
    total_original = sum(
        int(sample["payload_bytes"]["original"]) for sample in samples
    )

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
    for row, record in enumerate(records):
        actions = [str(action["action_id"]) for action in record["actions"]]
        active = {
            actions[int(column)]
            for column in targets[row][price_mask[row]].detach().cpu().tolist()
        }
        roi_frontier[int(record["image_id"])] = "roi" in active

    tasks = []
    for budget in runner.PROTOCOL["budgets"]:
        capacity = float(budget) * total_original
        teacher = runner.ACTION.schedule_fixed_byte(
            samples, rewards, capacity_bytes=capacity
        )
        policies = {"learned_ungated": score_map}
        policies["gt_any_frontier_gate"] = {
            image_id: {
                action: value
                for action, value in values.items()
                if action != "roi" or roi_frontier[image_id]
            }
            for image_id, values in score_map.items()
        }
        utility_allowed = {}
        for image_id, values in score_map.items():
            best_action, _, _ = _true_local_action(
                sample_by_id[image_id], rewards[image_id], float(teacher["lambda"])
            )
            utility_allowed[image_id] = {
                action: value
                for action, value in values.items()
                if action != "roi" or best_action == "roi"
            }
        policies["gt_budget_utility_gate"] = utility_allowed
        for policy, values in policies.items():
            scheduled = runner.ACTION.schedule_fixed_byte(
                samples, values, capacity_bytes=capacity
            )
            tasks.append(
                {
                    "policy": policy,
                    "budget": float(budget),
                    "capacity_bytes": capacity,
                    "request_bytes": int(scheduled["total_cost_bytes"]),
                    "selection": scheduled["selection"],
                }
            )
    _STATE.update({"data": data})
    context = mp.get_context("fork")
    with context.Pool(processes=min(9, len(tasks))) as pool:
        rows = list(pool.imap_unordered(_evaluate, tasks, chunksize=1))
    rows.sort(key=lambda row: (row["budget"], row["policy"]))
    deltas = []
    for budget in runner.PROTOCOL["budgets"]:
        baseline = next(
            row
            for row in rows
            if row["budget"] == float(budget)
            and row["policy"] == "learned_ungated"
        )
        for policy in ("gt_any_frontier_gate", "gt_budget_utility_gate"):
            candidate = next(
                row
                for row in rows
                if row["budget"] == float(budget) and row["policy"] == policy
            )
            deltas.append(
                {
                    "policy": policy,
                    "budget": float(budget),
                    "delta_ap50_95": float(candidate["official_ap50_95"])
                    - float(baseline["official_ap50_95"]),
                    "delta_roi_count": int(candidate["roi_count"])
                    - int(baseline["roi_count"]),
                }
            )
    report = {
        "protocol_id": runner.PROTOCOL["protocol_id"],
        "role": "fit_oof",
        "candidate": label,
        "rows": rows,
        "deltas": deltas,
        "gt_aware_non_deployable": True,
        "decision": (
            "Train an occupancy gate only if the any-frontier oracle has a "
            "meaningful AP-safe ceiling over the confirmed ungated candidate."
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
    _STATE.clear()
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
