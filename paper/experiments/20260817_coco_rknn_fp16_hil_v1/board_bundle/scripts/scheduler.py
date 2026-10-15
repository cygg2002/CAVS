"""Paper-compatible multiple-choice fixed-byte scheduling."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


ACTION_IDS = ("edge", "original", "q90", "current@640")


def _choice(group: list[dict[str, Any]], multiplier: float) -> dict[str, Any]:
    return max(group, key=lambda action: (float(action["value"]) - multiplier * float(action["cost"]), -float(action["cost"]), str(action["action_id"])))


def _select(groups: list[list[dict[str, Any]]], capacity: float) -> tuple[list[dict[str, Any]], float]:
    choose = lambda multiplier: [_choice(group, multiplier) for group in groups]
    selected = choose(0.0)
    if sum(float(item["cost"]) for item in selected) <= capacity:
        return selected, 0.0
    ratios = [max(0.0, float(row["value"])) / max(1e-12, float(row["cost"])) for group in groups for row in group if float(row["cost"]) > 0]
    low, high = 0.0, max(ratios, default=1.0) + 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if sum(float(item["cost"]) for item in choose(middle)) > capacity:
            low = middle
        else:
            high = middle
    selected = choose(high)
    used = sum(float(item["cost"]) for item in selected)
    while True:
        best = None
        for frame_index, (group, current) in enumerate(zip(groups, selected)):
            for alternative in group:
                extra_cost = float(alternative["cost"]) - float(current["cost"])
                extra_value = float(alternative["value"]) - float(current["value"])
                if extra_cost <= 0.0 or extra_value <= 0.0 or used + extra_cost > capacity + 1e-9:
                    continue
                candidate = (extra_value / extra_cost, extra_value, -extra_cost, str(alternative["action_id"]), frame_index, alternative)
                if best is None or candidate[:4] > best[:4]:
                    best = candidate
        if best is None:
            break
        frame_index, alternative = best[4], best[5]
        used += float(alternative["cost"]) - float(selected[frame_index]["cost"])
        selected[frame_index] = alternative
    return selected, high


def schedule_fixed_byte(candidates: Sequence[Mapping[str, Any]], budget: float) -> dict[str, Any]:
    """Choose exactly one action per image under the raw-JPEG capacity fraction."""
    if not 0.0 < float(budget) <= 1.0 or not candidates:
        raise ValueError("fixed-byte budget must be in (0, 1] with non-empty candidates")
    groups: list[list[dict[str, Any]]] = []
    payloads_by_frame: dict[int, dict[str, int]] = {}
    original_total, seen = 0, set()
    for candidate in candidates:
        frame_id = int(candidate["frame_id"])
        if frame_id in seen:
            raise ValueError(f"duplicate frame id: {frame_id}")
        seen.add(frame_id)
        payloads = {str(key): int(value) for key, value in candidate["payload_bytes"].items()}
        scores = {str(key): float(value) for key, value in candidate["scores"].items()}
        if payloads.get("original", 0) <= 0 or payloads.get("edge") != 0:
            raise ValueError(f"invalid payload contract for frame {frame_id}")
        group = [{"frame_id": frame_id, "action_id": action, "value": scores[action], "cost": payloads[action]} for action in ACTION_IDS if action in scores and action in payloads]
        if not group or group[0]["action_id"] != "edge":
            raise ValueError(f"frame {frame_id} requires the Edge fallback")
        groups.append(group)
        payloads_by_frame[frame_id] = payloads
        original_total += payloads["original"]
    capacity = float(budget) * original_total
    selected, multiplier = _select(groups, capacity)
    actions = {int(row["frame_id"]): str(row["action_id"]) for row in selected}
    request_bytes = sum(payloads_by_frame[frame][action] for frame, action in actions.items())
    if request_bytes > capacity + 1e-6:
        raise RuntimeError("fixed-byte capacity exceeded")
    counts = Counter(actions.values())
    return {
        "budget": float(budget), "actions": actions, "lambda": float(multiplier),
        "total_value": sum(float(row["value"]) for row in selected), "request_bytes": request_bytes,
        "original_total_bytes": original_total, "capacity_bytes": capacity,
        "capacity_utilization": request_bytes / capacity if capacity else 0.0,
        "frame_upload_ratio": (len(actions) - counts["edge"]) / len(actions),
        "action_counts": {action: int(counts[action]) for action in ACTION_IDS},
    }

