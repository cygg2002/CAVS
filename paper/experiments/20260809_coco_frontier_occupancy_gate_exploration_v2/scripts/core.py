"""Pure contracts for the post-hoc frontier-occupancy diagnostic."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


EXPECTED_METHODS = ("CostFree-FrontierGate", "CostAware-FrontierGate")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco" or dict(protocol.get("roles", {})) != {
        "fit": 9678
    }:
        raise ValueError("The v2 diagnostic is canonical COCO fit-only")
    if tuple(protocol.get("methods", ())) != EXPECTED_METHODS:
        raise ValueError("The frontier-gate method matrix changed")
    if tuple(map(float, protocol.get("budgets", ()))) != (0.1, 0.25, 0.4):
        raise ValueError("The budget grid changed")
    if tuple(protocol.get("modes", ())) != ("fixed_byte", "fixed_call"):
        raise ValueError("The evaluation modes changed")
    if not 0.0 < float(protocol.get("probability_threshold", -1.0)) < 1.0:
        raise ValueError("The fixed occupancy threshold is invalid")
    if protocol.get("fit_only_training") is not True:
        raise ValueError("Training must remain fit-only")
    if protocol.get("threshold_selected_on_oof") is not False:
        raise ValueError("OOF results cannot select the gate threshold")
    for flag in (
        "calibration_accessed",
        "val2017_accessed",
        "audit_accessed",
        "final_accessed",
        "registered_final_accessed",
    ):
        if protocol.get(flag) is not False:
            raise ValueError(f"Forbidden role flag changed: {flag}")


def frontier_occupancy_labels(
    target_actions: torch.Tensor,
    price_mask: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Mark cloud actions that are optimal anywhere on a valid price path."""

    if target_actions.ndim != 2 or price_mask.shape != target_actions.shape:
        raise ValueError("Target path and price mask differ")
    if action_mask.ndim != 2 or action_mask.shape[0] != target_actions.shape[0]:
        raise ValueError("Action mask has the wrong population")
    labels = torch.zeros_like(action_mask, dtype=torch.bool)
    batch = torch.arange(target_actions.shape[0], device=target_actions.device)
    for probe in range(target_actions.shape[1]):
        valid = price_mask[:, probe].to(torch.bool)
        if bool(valid.any()):
            labels[batch[valid], target_actions[valid, probe]] = True
    labels &= action_mask.to(torch.bool)
    labels[:, 0] = False
    return labels


def gate_roi_scores(
    scores: Mapping[int, Mapping[str, float]],
    roi_probabilities: Mapping[int, float],
    *,
    threshold: float,
) -> dict[int, dict[str, float]]:
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("ROI threshold must lie in (0, 1)")
    output = {}
    for image_id_raw, values_raw in scores.items():
        image_id = int(image_id_raw)
        values = {str(action): float(value) for action, value in values_raw.items()}
        if "roi" in values:
            if image_id not in roi_probabilities:
                raise ValueError(f"Missing ROI occupancy probability: {image_id}")
            probability = float(roi_probabilities[image_id])
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"Invalid ROI occupancy probability: {image_id}")
            if probability < float(threshold):
                values.pop("roi")
        output[image_id] = values
    return output
