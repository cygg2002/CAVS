"""Pure data/reward helpers for the Cloud‑V3 Oregon roles."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping, Sequence


def rebuild_observable_action_features(
    features: Sequence[float],
    *,
    original_bytes: int,
    q90_bytes: int,
    roi_payloads: Sequence[int],
    roi_areas: Sequence[float],
) -> list[float]:
    base = list(map(float, features))
    if len(base) != 49:
        raise ValueError(f"expected 49 Edge/FPN features, got {len(base)}")
    original = max(1, int(original_bytes))
    q90 = int(q90_bytes)
    payloads = [int(value) for value in roi_payloads if int(value) > 0]
    areas = [float(value) for value in roi_areas if float(value) > 0.0]
    return [
        *base,
        q90 / original,
        min(payloads, default=original) / original,
        min(areas, default=0.0),
        float(0 < q90 < original),
        float(bool(payloads)),
    ]


def proxy_best_action(
    qualities: Mapping[str, float], payloads: Mapping[str, int]
) -> str:
    if not qualities:
        raise ValueError("at least one action is required")
    missing = set(qualities) - set(payloads)
    if missing:
        raise ValueError(f"missing payloads for {sorted(missing)}")
    return min(
        map(str, qualities),
        key=lambda action: (
            -float(qualities[action]),
            int(payloads[action]),
            action,
        ),
    )


def fold_contained_contexts(
    folds: Sequence[Sequence[int]], *, context_size: int, seed: int
) -> dict[int, list[int]]:
    output: dict[int, list[int]] = {}
    for fold_index, fold in enumerate(folds):
        values = sorted(map(int, fold))
        for target in values:
            candidates = [value for value in values if value != target]
            count = min(max(0, int(context_size)), len(candidates))
            token = hashlib.sha256(
                f"{int(seed)}:{fold_index}:{target}".encode("utf-8")
            ).digest()
            rng = random.Random(int.from_bytes(token[:8], "big"))
            output[target] = sorted(rng.sample(candidates, count))
    return output


def reward_plan(protocol: Mapping[str, Any]) -> dict[str, int]:
    actions = 2 + len(protocol["roi_actions"])
    return {
        role: int(protocol["roles"][role]) * actions
        for role in ("fit", "calibration")
    }
