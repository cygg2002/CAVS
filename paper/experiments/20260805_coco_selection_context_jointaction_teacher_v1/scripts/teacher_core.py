"""Pure contracts for the Selection-Context JointAction teacher diagnostic."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
sys.path.insert(0, str(REPO / "paper/experiments/GAIR_v5"))

from gair.selector import select_actions_for_budget  # noqa: E402


ALLOWED_ROLES = {"fit", "calibration"}
REQUIRED_ACTIONS = ("edge", "original", "q90", "roi")
CANDIDATE_ACTIONS = ("original", "q90", "roi")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject protocol drift that could invalidate the diagnostic."""
    if set(protocol.get("roles", {})) != ALLOWED_ROLES:
        raise ValueError("Only fit and calibration roles are allowed")
    if int(protocol["roles"]["fit"]) != 9678:
        raise ValueError("Canonical COCO fit must contain 9,678 images")
    if int(protocol["roles"]["calibration"]) != 571:
        raise ValueError("Canonical COCO calibration must contain 571 images")
    if tuple(protocol.get("actions", ())) != REQUIRED_ACTIONS:
        raise ValueError("The frozen action space changed")
    if tuple(protocol.get("candidate_actions", ())) != CANDIDATE_ACTIONS:
        raise ValueError("The frozen candidate action space changed")
    if tuple(map(float, protocol.get("budgets", ()))) != (0.1, 0.25, 0.4):
        raise ValueError("The frozen byte budgets changed")
    if protocol.get("primary_mode") != "fixed_byte":
        raise ValueError("This diagnostic is fixed-byte only")
    if protocol.get("metric") != "official_ap50_95":
        raise ValueError("The primary metric must remain Official AP50-95")
    if protocol.get("fit_only_context") is not True:
        raise ValueError("Context images must be fit-only")
    if protocol.get("calibration_teacher_only") is not True:
        raise ValueError("Calibration labels are teacher-only")
    for flag in ("audit_accessed", "final_accessed", "registered_final_accessed"):
        if protocol.get(flag) is not False:
            raise ValueError(f"{flag} must remain false")
    reward = protocol.get("reward", {})
    if int(reward.get("context_size", -1)) != 32:
        raise ValueError("The frozen context size changed")
    if tuple(map(float, reward.get("iou_thresholds", ()))) != tuple(
        round(0.5 + 0.05 * index, 2) for index in range(10)
    ):
        raise ValueError("The contextual reward must use IoU 0.50:0.95")


def build_context_ids(
    fit_image_ids: Sequence[int],
    target_image_id: int,
    *,
    context_size: int,
    seed: int,
) -> list[int]:
    """Choose a deterministic fit-only context for any target image."""
    target = int(target_image_id)
    candidates = sorted({int(image_id) for image_id in fit_image_ids if int(image_id) != target})
    count = min(max(0, int(context_size)), len(candidates))
    material = f"{int(seed)}:{target}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return sorted(random.Random(local_seed).sample(candidates, count))


def schedule_joint_actions(
    samples: Sequence[Mapping[str, Any]],
    rewards: Mapping[int, Mapping[str, float]],
    *,
    candidate_actions: Sequence[str],
    capacity_bytes: float,
) -> dict[str, Any]:
    """Select at most one available cloud action per image under a byte cap."""
    if float(capacity_bytes) < 0.0:
        raise ValueError("capacity_bytes must be non-negative")
    groups: list[list[dict[str, Any]]] = []
    image_ids: list[int] = []
    for sample in samples:
        image_id = int(sample["image_id"])
        image_ids.append(image_id)
        group: list[dict[str, Any]] = [
            {"action_id": "edge", "image_id": image_id, "value": 0.0, "cost": 0.0}
        ]
        for action in candidate_actions:
            available = bool(sample.get("available", {}).get(action, False))
            payload = sample.get("payload_bytes", {}).get(action)
            value = rewards.get(image_id, {}).get(action)
            if not available or payload is None or value is None or not math.isfinite(float(value)):
                continue
            group.append(
                {
                    "action_id": str(action),
                    "image_id": image_id,
                    "value": float(value),
                    "cost": float(payload),
                }
            )
        groups.append(group)
    scheduled = select_actions_for_budget(groups, float(capacity_bytes))
    chosen = scheduled["selected"]
    if len(chosen) != len(image_ids):
        raise RuntimeError("The scheduler did not return one choice per image")
    selection = {
        int(item["image_id"]): str(item["action_id"])
        for item in chosen
        if str(item["action_id"]) != "edge"
    }
    total_cost = float(scheduled["total_cost"])
    if total_cost > float(capacity_bytes) + 1e-6:
        raise RuntimeError("Joint-action schedule exceeded its byte capacity")
    return {
        "selection": selection,
        "total_cost_bytes": total_cost,
        "total_value": float(scheduled["total_value"]),
        "lambda": float(scheduled["lambda"]),
        "selected_images": len(selection),
    }


def selection_overlap(
    left: Mapping[int, str], right: Mapping[int, str]
) -> dict[str, float | int]:
    left_images = set(map(int, left))
    right_images = set(map(int, right))
    common_images = left_images & right_images
    image_union = left_images | right_images
    left_actions = {(int(image_id), str(action)) for image_id, action in left.items()}
    right_actions = {(int(image_id), str(action)) for image_id, action in right.items()}
    common_actions = left_actions & right_actions
    action_union = left_actions | right_actions
    return {
        "common_selected_images": len(common_images),
        "selected_image_jaccard": len(common_images) / max(1, len(image_union)),
        "common_exact_actions": len(common_actions),
        "exact_action_jaccard": len(common_actions) / max(1, len(action_union)),
    }


def evaluate_gate(
    budget_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    expected = tuple(map(float, protocol["budgets"]))
    observed = tuple(sorted(float(row["budget"]) for row in budget_rows))
    if observed != tuple(sorted(expected)):
        raise ValueError("Gate rows do not cover every frozen budget exactly once")
    gate = protocol["gate"]
    minimum = float(gate["minimum_gain"])
    margin = float(gate["remaining_noninferiority_margin"])
    gains = [float(row["delta_ap"]) for row in budget_rows]
    gain_points = sum(value >= minimum for value in gains)
    all_noninferior = all(value >= -margin for value in gains)
    capacity_ok = all(bool(row["capacity_ok"]) for row in budget_rows)
    passed = (
        gain_points >= int(gate["required_gain_points"])
        and all_noninferior
        and capacity_ok
    )
    return {
        "passed": bool(passed),
        "gain_points": int(gain_points),
        "required_gain_points": int(gate["required_gain_points"]),
        "all_noninferior": bool(all_noninferior),
        "capacity_ok": bool(capacity_ok),
        "minimum_gain": minimum,
        "noninferiority_margin": margin,
    }
