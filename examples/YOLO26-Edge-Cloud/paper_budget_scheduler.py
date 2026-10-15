# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Paper-compatible fixed-byte scheduling for the Current+MAUR demo."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


ACTION_IDS = ("edge", "original", "q90", "current@640")


def _choice(group: list[dict[str, Any]], multiplier: float) -> dict[str, Any]:
    """Match the paper scheduler's value-cost choice and tie-breaking."""
    return max(
        group,
        key=lambda action: (
            float(action["value"]) - multiplier * float(action["cost"]),
            -float(action["cost"]),
            str(action["action_id"]),
        ),
    )


def select_actions_for_budget(
    actions_by_frame: list[list[dict[str, Any]]],
    budget_bytes: float,
    *,
    iterations: int = 80,
) -> dict[str, Any]:
    """Run the same multiple-choice Lagrangian scheduler used by the paper experiments."""
    if budget_bytes < 0.0 or not actions_by_frame or any(not group for group in actions_by_frame):
        raise ValueError("Budget must be non-negative and every frame must have actions")
    for group in actions_by_frame:
        if any(float(action["cost"]) < 0.0 for action in group):
            raise ValueError("Action costs must be non-negative")
        identifiers = [str(action["action_id"]) for action in group]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Action IDs must be unique within each frame")

    def choose_all(multiplier: float) -> list[dict[str, Any]]:
        return [_choice(group, multiplier) for group in actions_by_frame]

    unconstrained = choose_all(0.0)
    if sum(float(item["cost"]) for item in unconstrained) <= float(budget_bytes):
        selected, multiplier = unconstrained, 0.0
    else:
        ratios = [
            max(0.0, float(action["value"])) / max(1e-12, float(action["cost"]))
            for group in actions_by_frame
            for action in group
            if float(action["cost"]) > 0.0
        ]
        low, high = 0.0, max(ratios, default=1.0) + 1.0
        for _ in range(max(1, int(iterations))):
            middle = (low + high) / 2.0
            if sum(float(item["cost"]) for item in choose_all(middle)) > float(budget_bytes):
                low = middle
            else:
                high = middle
        multiplier = high
        selected = choose_all(multiplier)

    used = sum(float(item["cost"]) for item in selected)
    while True:
        best = None
        for frame_index, (group, current) in enumerate(zip(actions_by_frame, selected)):
            for alternative in group:
                extra_cost = float(alternative["cost"]) - float(current["cost"])
                extra_value = float(alternative["value"]) - float(current["value"])
                if extra_cost <= 0.0 or extra_value <= 0.0 or used + extra_cost > budget_bytes + 1e-9:
                    continue
                candidate = (
                    extra_value / extra_cost,
                    extra_value,
                    -extra_cost,
                    str(alternative["action_id"]),
                    frame_index,
                    alternative,
                )
                if best is None or candidate[:4] > best[:4]:
                    best = candidate
        if best is None:
            break
        frame_index, alternative = best[4], best[5]
        used += float(alternative["cost"]) - float(selected[frame_index]["cost"])
        selected[frame_index] = alternative
    return {
        "selected": [dict(item) for item in selected],
        "total_cost": float(used),
        "total_value": sum(float(item["value"]) for item in selected),
        "lambda": float(multiplier),
    }


def summarize_fixed_byte_plan(
    *,
    actions: Mapping[int, str],
    request_bytes: int,
    original_total_bytes: int,
    capacity_bytes: float,
) -> dict[str, Any]:
    """Report byte usage and upload membership as separate quantities."""
    counts = Counter(str(action) for action in actions.values())
    frames = len(actions)
    uploads = frames - int(counts["edge"])
    return {
        "frames": frames,
        "uploads": uploads,
        "request_bytes": int(request_bytes),
        "original_total_bytes": int(original_total_bytes),
        "capacity_bytes": float(capacity_bytes),
        "byte_ratio": float(request_bytes / original_total_bytes) if original_total_bytes else 0.0,
        "frame_upload_ratio": float(uploads / frames) if frames else 0.0,
        "capacity_utilization": float(request_bytes / capacity_bytes) if capacity_bytes else 0.0,
        "action_counts": {action: int(counts[action]) for action in ACTION_IDS},
    }


def schedule_fixed_byte(candidates: Sequence[Mapping[str, Any]], budget: float) -> dict[str, Any]:
    """Freeze one Current+MAUR action per frame under a global raw-JPEG byte budget."""
    fraction = float(budget)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fixed-byte budget must be in (0, 1]")
    if not candidates:
        raise ValueError("fixed-byte scheduling requires at least one frame")

    groups: list[list[dict[str, Any]]] = []
    original_total = 0
    seen_frames: set[int] = set()
    payload_by_frame: dict[int, dict[str, int]] = {}
    for candidate in candidates:
        frame_id = int(candidate["frame_id"])
        if frame_id in seen_frames:
            raise ValueError(f"duplicate frame id: {frame_id}")
        seen_frames.add(frame_id)
        scores = {str(key): float(value) for key, value in candidate["scores"].items()}
        payloads = {str(key): int(value) for key, value in candidate["payload_bytes"].items()}
        if "original" not in payloads or payloads["original"] <= 0:
            raise ValueError(f"frame {frame_id} has no positive Original payload")
        original_total += payloads["original"]
        payload_by_frame[frame_id] = payloads

        group: list[dict[str, Any]] = []
        for action_id in ACTION_IDS:
            if action_id not in scores or action_id not in payloads:
                continue
            group.append(
                {
                    "frame_id": frame_id,
                    "action_id": action_id,
                    "value": scores[action_id],
                    "cost": payloads[action_id],
                }
            )
        if not group or group[0]["action_id"] != "edge":
            raise ValueError(f"frame {frame_id} requires the Edge fallback")
        groups.append(group)

    capacity = fraction * original_total
    result = select_actions_for_budget(groups, capacity)
    actions = {int(item["frame_id"]): str(item["action_id"]) for item in result["selected"]}
    request_bytes = sum(payload_by_frame[frame_id][action] for frame_id, action in actions.items())
    if request_bytes > capacity + 1e-6:
        raise RuntimeError(f"fixed-byte capacity exceeded: {request_bytes}>{capacity}")

    return {
        "budget": fraction,
        "actions": actions,
        "lambda": float(result["lambda"]),
        "total_value": float(result["total_value"]),
        **summarize_fixed_byte_plan(
            actions=actions,
            request_bytes=request_bytes,
            original_total_bytes=original_total,
            capacity_bytes=capacity,
        ),
    }
