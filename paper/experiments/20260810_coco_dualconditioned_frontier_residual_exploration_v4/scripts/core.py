"""Pure contracts for dual-conditioned frontier residual rectification."""

from __future__ import annotations

from typing import Any, Mapping

import torch


EXPECTED_METHODS = ("DualConditioned-ROIResidual",)
EXPECTED_PRICES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco" or dict(protocol.get("roles", {})) != {
        "fit": 9678
    }:
        raise ValueError("The v4 diagnostic is canonical COCO fit-only")
    if tuple(protocol.get("methods", ())) != EXPECTED_METHODS:
        raise ValueError("The residual method matrix changed")
    if tuple(map(float, protocol.get("budgets", ()))) != (0.1, 0.25, 0.4):
        raise ValueError("The budget grid changed")
    if tuple(protocol.get("modes", ())) != ("fixed_byte", "fixed_call"):
        raise ValueError("The evaluation modes changed")
    if tuple(map(float, protocol.get("training_prices", ()))) != EXPECTED_PRICES:
        raise ValueError("The frozen training-price grid changed")
    if int(protocol.get("model", {}).get("frontier_state_dim", -1)) != 3:
        raise ValueError("Frontier state must remain three-dimensional")
    if protocol.get("fit_only_training") is not True:
        raise ValueError("Training must remain fit-only")
    if protocol.get("hyperparameters_selected_on_oof") is not False:
        raise ValueError("OOF cannot tune residual hyperparameters")
    for flag in (
        "calibration_accessed",
        "val2017_accessed",
        "audit_accessed",
        "final_accessed",
        "registered_final_accessed",
    ):
        if protocol.get(flag) is not False:
            raise ValueError(f"Forbidden role flag changed: {flag}")


def _indices(
    roi_index: int | torch.Tensor, *, rows: int, actions: int, device: torch.device
) -> torch.Tensor:
    if isinstance(roi_index, torch.Tensor):
        values = roi_index.to(device=device, dtype=torch.long)
        if values.shape != (rows,):
            raise ValueError("Per-image ROI indices must have shape [image]")
    else:
        values = torch.full((rows,), int(roi_index), dtype=torch.long, device=device)
    if bool(((values < 0) | (values >= actions)).any()):
        raise ValueError("An ROI index is out of range")
    return values


def frontier_state(
    values: torch.Tensor,
    costs: torch.Tensor,
    action_mask: torch.Tensor,
    prices: torch.Tensor,
    *,
    roi_index: int | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Describe ROI relative to the best non-ROI alternative at each price."""

    if values.ndim != 2 or costs.shape != values.shape or action_mask.shape != values.shape:
        raise ValueError("Value, cost, and action masks must share [image, action]")
    if prices.shape != (values.shape[0],):
        raise ValueError("Prices must have one value per image")
    indices = _indices(
        roi_index,
        rows=values.shape[0],
        actions=values.shape[1],
        device=values.device,
    )
    valid = action_mask.to(torch.bool)
    utility = values - prices[:, None].to(values.dtype) * costs
    utility = utility.masked_fill(~valid, -torch.inf)
    alternatives = valid.clone()
    alternatives.scatter_(1, indices[:, None], False)
    alternative_utility = utility.masked_fill(~alternatives, -torch.inf)
    competitor_index = alternative_utility.argmax(dim=1)
    competitor_utility = alternative_utility.gather(
        1, competitor_index[:, None]
    ).squeeze(1)
    roi_utility = utility.gather(1, indices[:, None]).squeeze(1)
    roi_cost = costs.gather(1, indices[:, None]).squeeze(1)
    competitor_cost = costs.gather(1, competitor_index[:, None]).squeeze(1)
    roi_available = valid.gather(1, indices[:, None]).squeeze(1)
    if not bool(torch.isfinite(competitor_utility).all()):
        raise ValueError("Every row needs a valid non-ROI alternative")
    return {
        "margin": torch.where(
            roi_available, roi_utility - competitor_utility, torch.zeros_like(roi_utility)
        ),
        "cost_gap": torch.where(
            roi_available, roi_cost - competitor_cost, torch.zeros_like(roi_cost)
        ),
        "competitor_index": competitor_index,
        "roi_available": roi_available,
    }


def rectify_roi_scores(
    scores: Mapping[int, Mapping[str, float]],
    roi_residuals: Mapping[int, float],
) -> dict[int, dict[str, float]]:
    """Add a residual to ROI only; preserve all other action values exactly."""

    output: dict[int, dict[str, float]] = {}
    for image_id_raw, values_raw in scores.items():
        image_id = int(image_id_raw)
        values = {str(action): float(value) for action, value in values_raw.items()}
        if "roi" in values:
            if image_id not in roi_residuals:
                raise ValueError(f"Missing ROI residual: {image_id}")
            residual = float(roi_residuals[image_id])
            if not torch.isfinite(torch.tensor(residual)):
                raise ValueError(f"Non-finite ROI residual: {image_id}")
            values["roi"] += residual
        output[image_id] = values
    return output

