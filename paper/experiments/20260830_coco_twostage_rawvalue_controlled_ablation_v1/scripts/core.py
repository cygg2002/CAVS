"""Pure helpers for the raw-value TwoStage controlled ablation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


DEFAULT_ACTION_ORDER = ("original", "q90")


def raw_projected_score_map(
    action_scores: Mapping[int, Mapping[str, float]],
    action_order: Sequence[str] = DEFAULT_ACTION_ORDER,
) -> dict[int, dict[str, float]]:
    """Project each image to one cloud action while preserving its raw value.

    Ties follow ``action_order``. The Edge action is deliberately excluded: the
    downstream Fixed-byte solver adds Edge as the zero-cost fallback.
    """

    projected: dict[int, dict[str, float]] = {}
    for image_id, per_action in action_scores.items():
        candidates = [
            (action, float(per_action[action]))
            for action in action_order
            if action in per_action and math.isfinite(float(per_action[action]))
        ]
        if not candidates:
            raise ValueError(f"image {image_id} has no finite cloud-action score")

        chosen_action, chosen_value = candidates[0]
        for action, value in candidates[1:]:
            if value > chosen_value:
                chosen_action, chosen_value = action, value
        projected[int(image_id)] = {chosen_action: chosen_value}

    return projected
