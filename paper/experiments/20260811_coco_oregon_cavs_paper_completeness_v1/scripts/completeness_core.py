"""Pure contracts and exact small-set audit helpers."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any, Mapping, Sequence


def validate_isolated_sources(paths: Sequence[str], protocol: Mapping[str, Any]) -> None:
    forbidden = tuple(str(value).lower() for value in protocol.get("forbidden_paths", ()))
    for path in paths:
        lowered = str(path).lower()
        if any(token in lowered for token in forbidden):
            raise ValueError(f"forbidden source path: {path}")


def action_mask(name: str, protocol: Mapping[str, Any]) -> set[str]:
    values = protocol["action_masks"][str(name)]
    allowed = {str(value) for value in values}
    if not allowed <= {"edge", "original", "q90"} or "edge" not in allowed:
        raise ValueError(f"invalid action mask: {name}")
    return allowed


def bootstrap_job_count(dataset: str, protocol: Mapping[str, Any]) -> int:
    units = ("image",) if dataset == "coco" else ("event", "image")
    return 4 * len(protocol["budgets"]) * len(units)


def event_groups(samples: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        if "event_id" not in sample:
            raise ValueError("event bootstrap requires event_id")
        groups[str(sample["event_id"])].append(int(sample["image_id"]))
    return [groups[key] for key in sorted(groups)]


def exact_oq_objective(
    values: Mapping[int, Mapping[str, float]],
    costs: Mapping[int, Mapping[str, float]],
    *,
    capacity: float,
) -> dict[str, Any]:
    """Enumerate the exact edge/original/q90 choice for a small audit subset."""
    ids = sorted(int(value) for value in values)
    best = {"objective": 0.0, "used_bytes": 0.0, "actions": {}}
    for choices in product(("edge", "original", "q90"), repeat=len(ids)):
        used = sum(float(costs[i].get(a, 0.0)) for i, a in zip(ids, choices))
        if used > float(capacity) + 1e-9:
            continue
        objective = sum(float(values[i].get(a, 0.0)) for i, a in zip(ids, choices))
        if objective > float(best["objective"]) + 1e-12:
            best = {
                "objective": float(objective),
                "used_bytes": float(used),
                "actions": {str(i): a for i, a in zip(ids, choices)},
            }
    return best


def heuristic_objective(actions: Mapping[int, str], values: Mapping[int, Mapping[str, float]]) -> float:
    return float(sum(float(values[int(i)].get(str(a), 0.0)) for i, a in actions.items()))
