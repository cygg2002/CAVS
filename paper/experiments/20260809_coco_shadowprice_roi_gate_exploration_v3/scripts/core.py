"""Pure contracts for the shadow-price-conditioned ROI gate."""

from __future__ import annotations

from typing import Any, Mapping

import torch


EXPECTED_METHODS = ("PriceConditional-ROIGate",)
EXPECTED_PRICES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco" or dict(protocol.get("roles", {})) != {
        "fit": 9678
    }:
        raise ValueError("The v3 diagnostic is canonical COCO fit-only")
    if tuple(protocol.get("methods", ())) != EXPECTED_METHODS:
        raise ValueError("The price-gate method matrix changed")
    if tuple(map(float, protocol.get("budgets", ()))) != (0.1, 0.25, 0.4):
        raise ValueError("The budget grid changed")
    if tuple(protocol.get("modes", ())) != ("fixed_byte", "fixed_call"):
        raise ValueError("The evaluation modes changed")
    if tuple(map(float, protocol.get("training_prices", ()))) != EXPECTED_PRICES:
        raise ValueError("The frozen training-price grid changed")
    if not 0.0 < float(protocol.get("probability_threshold", -1.0)) < 1.0:
        raise ValueError("The fixed ROI threshold is invalid")
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


def roi_optimal_by_price(
    rewards: torch.Tensor,
    costs: torch.Tensor,
    action_mask: torch.Tensor,
    prices: torch.Tensor,
    *,
    roi_index: int | torch.Tensor,
) -> torch.Tensor:
    """Return whether ROI is utility-optimal at each normalized price."""

    if rewards.ndim != 2 or costs.shape != rewards.shape or action_mask.shape != rewards.shape:
        raise ValueError("Reward, cost, and action masks must share [image, action]")
    if prices.ndim != 1 or not len(prices):
        raise ValueError("Prices must be a non-empty vector")
    if isinstance(roi_index, torch.Tensor):
        indices = roi_index.to(device=rewards.device, dtype=torch.long)
        if indices.shape != (rewards.shape[0],):
            raise ValueError("Per-image ROI indices must have shape [image]")
        if bool(((indices < 0) | (indices >= rewards.shape[1])).any()):
            raise ValueError("A per-image ROI index is out of range")
    else:
        if not 0 <= int(roi_index) < rewards.shape[1]:
            raise ValueError("ROI index is out of range")
        indices = torch.full(
            (rewards.shape[0],),
            int(roi_index),
            dtype=torch.long,
            device=rewards.device,
        )
    utility = rewards[:, None, :] - prices[None, :, None] * costs[:, None, :]
    valid = action_mask[:, None, :].to(torch.bool)
    utility = utility.masked_fill(~valid, -torch.inf)
    maximum = utility.amax(dim=2)
    gather_index = indices[:, None, None].expand(-1, len(prices), 1)
    roi_utility = utility.gather(2, gather_index).squeeze(2)
    roi_available = valid.expand(-1, len(prices), -1).gather(2, gather_index).squeeze(2)
    return roi_available & (roi_utility >= maximum - 1e-12)


def gate_roi_scores(
    scores: Mapping[int, Mapping[str, float]],
    roi_probabilities: Mapping[int, float],
    *,
    threshold: float,
) -> dict[int, dict[str, float]]:
    """Remove ROI when the price-conditioned probability is below threshold."""

    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("ROI threshold must lie in (0, 1)")
    output: dict[int, dict[str, float]] = {}
    for image_id_raw, values_raw in scores.items():
        image_id = int(image_id_raw)
        values = {str(action): float(value) for action, value in values_raw.items()}
        if "roi" in values:
            if image_id not in roi_probabilities:
                raise ValueError(f"Missing conditional ROI probability: {image_id}")
            probability = float(roi_probabilities[image_id])
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"Invalid conditional ROI probability: {image_id}")
            if probability < float(threshold):
                values.pop("roi")
        output[image_id] = values
    return output
