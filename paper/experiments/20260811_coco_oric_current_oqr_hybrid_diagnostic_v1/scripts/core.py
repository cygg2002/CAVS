"""Pure contracts for the frozen ORIC–Current OQR hybrid diagnostic."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


EXPECTED_METHODS = (
    "ORIC-Current-OQ",
    "ORIC-Current-OQR",
    "Current-OQ",
    "Current-OQR",
)
EXPECTED_MODES = ("fixed_byte", "fixed_call")
EXPECTED_BUDGETS = (0.1, 0.25, 0.4)
EXPECTED_COMPARISONS = (
    ("ORIC-Current-OQ", "Current-OQ"),
    ("ORIC-Current-OQR", "Current-OQR"),
    ("ORIC-Current-OQR", "ORIC-Current-OQ"),
    ("Current-OQR", "Current-OQ"),
)


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, int]:
    if protocol.get("status") != "posthoc_final_reuse_hybrid_diagnostic":
        raise ValueError("Evidence status changed")
    if protocol.get("dataset") != "coco":
        raise ValueError("Only canonical COCO is allowed")
    if protocol.get("roles") != {"fit": 9678, "calibration": 571, "final": 5000}:
        raise ValueError("Canonical role counts changed")
    if tuple(protocol.get("methods", ())) != EXPECTED_METHODS:
        raise ValueError("Method matrix changed")
    if tuple(protocol.get("modes", ())) != EXPECTED_MODES:
        raise ValueError("Budget modes changed")
    if tuple(map(float, protocol.get("budgets", ()))) != EXPECTED_BUDGETS:
        raise ValueError("Budget grid changed")
    comparisons = tuple(tuple(item) for item in protocol.get("comparisons", ()))
    if comparisons != EXPECTED_COMPARISONS:
        raise ValueError("Comparison matrix changed")
    expected_actions = {
        "ORIC-Current-OQ": ("original", "q90"),
        "ORIC-Current-OQR": ("original", "q90", "roi"),
        "Current-OQ": ("original", "q90"),
        "Current-OQR": ("original", "q90", "roi"),
    }
    observed_actions = {
        method: tuple(protocol.get("action_spaces", {}).get(method, ()))
        for method in EXPECTED_METHODS
    }
    if observed_actions != expected_actions:
        raise ValueError("Action spaces changed")
    if protocol.get("hybrid_definition") != (
        "oric_original_anchor_plus_current_action_advantage"
    ):
        raise ValueError("Hybrid definition changed")
    for key in (
        "detector_inference_required",
        "retraining_allowed",
        "tuning_allowed",
        "final_is_hidden",
        "final_used_for_selection",
        "confirmatory",
    ):
        if protocol.get(key) is not False:
            raise ValueError(f"{key} must remain false")
    replicates = int(protocol.get("bootstrap", {}).get("replicates", 0))
    if replicates != 1000:
        raise ValueError("Bootstrap replicate count changed")
    return {
        "aggregate_rows": len(EXPECTED_METHODS)
        * len(EXPECTED_MODES)
        * len(EXPECTED_BUDGETS),
        "bootstrap_comparisons": len(EXPECTED_COMPARISONS)
        * len(EXPECTED_BUDGETS),
    }


def compose_hybrid_scores(
    image_ids: Sequence[int],
    oric_scores: Mapping[int, float],
    current_scores: Mapping[int, Mapping[str, float]],
    *,
    actions: Sequence[str],
) -> dict[int, dict[str, float]]:
    ids = list(map(int, image_ids))
    if len(ids) != len(set(ids)):
        raise ValueError("Image IDs must be unique")
    expected = set(ids)
    if set(map(int, oric_scores)) != expected or set(map(int, current_scores)) != expected:
        raise ValueError("ORIC, Current, and image IDs must align one-to-one")
    action_ids = tuple(map(str, actions))
    if not action_ids or action_ids[0] != "original" or len(action_ids) != len(set(action_ids)):
        raise ValueError("Actions must be unique and Original-anchored")
    output: dict[int, dict[str, float]] = {}
    for image_id in ids:
        anchor = float(oric_scores[image_id])
        current = current_scores[image_id]
        if "original" not in current:
            raise ValueError(f"Current Original score is missing: {image_id}")
        original = float(current["original"])
        available_actions = tuple(action for action in action_ids if action in current)
        values = {
            action: anchor + float(current[action]) - original
            for action in available_actions
        }
        if not math.isfinite(anchor) or not all(map(math.isfinite, values.values())):
            raise ValueError("Hybrid scores must be finite")
        output[image_id] = values
    return output


def assert_capacity(request_bytes: float, capacity_bytes: float) -> None:
    if float(request_bytes) > float(capacity_bytes) + 1e-6:
        raise ValueError("Fixed-byte capacity exceeded")
