"""Pure contracts and frontier-path losses for the COCO diagnostic."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F


EXPECTED_ROLES = {"fit": 9678, "calibration": 571}
EXPECTED_BUDGETS = (0.1, 0.25, 0.4)
EXPECTED_MODES = ("fixed_byte", "fixed_call")
EXPECTED_METHODS = (
    "CurrentActionValue",
    "PathDistill",
    "BoundaryPathDistill",
)


def filter_action_space_records(
    records: Sequence[Mapping[str, Any]], allowed_actions: Sequence[str]
) -> list[dict[str, Any]]:
    """Return record copies restricted to one frozen action space."""

    allowed = tuple(map(str, allowed_actions))
    if not allowed or allowed[0] != "edge" or len(allowed) != len(set(allowed)):
        raise ValueError("An action space must contain unique actions starting with edge")
    output = []
    for record in records:
        actions = [
            dict(action)
            for action in record["actions"]
            if str(action["action_id"]) in allowed
        ]
        observed = tuple(str(action["action_id"]) for action in actions)
        if not observed or observed[0] != "edge":
            raise ValueError(f"Image {record.get('image_id')} has no Edge action")
        if any(action not in allowed for action in observed):
            raise RuntimeError("Action filtering failed")
        output.append({**dict(record), "actions": actions})
    return output


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("dataset") != "coco":
        raise ValueError("This exploration is COCO-only")
    if dict(protocol.get("roles", {})) != EXPECTED_ROLES:
        raise ValueError("Canonical development roles changed")
    spaces = protocol.get("action_spaces", {})
    if tuple(spaces.get("OQ", ())) != ("edge", "original", "q90"):
        raise ValueError("OQ action space changed")
    if tuple(spaces.get("OQR", ())) != (
        "edge",
        "original",
        "q90",
        "roi",
    ):
        raise ValueError("OQR action space changed")
    if tuple(protocol.get("screen_methods", ())) != EXPECTED_METHODS:
        raise ValueError("Screen method matrix changed")
    if tuple(map(float, protocol.get("budgets", ()))) != EXPECTED_BUDGETS:
        raise ValueError("Evaluation budgets changed")
    if tuple(protocol.get("modes", ())) != EXPECTED_MODES:
        raise ValueError("Evaluation modes changed")
    if int(protocol.get("fold_count", 0)) != 5:
        raise ValueError("Five OOF folds are required")
    if protocol.get("primary_mode") != "fixed_byte":
        raise ValueError("Fixed-byte must remain primary")
    if protocol.get("fit_only_parameter_selection") is not True:
        raise ValueError("Selection must remain fit-only")
    if protocol.get("calibration_used_for_training_or_selection") is not False:
        raise ValueError("Calibration cannot train or select")
    if protocol.get("detector_inference_required") is not False:
        raise ValueError("Detector inference is outside scope")
    for key in (
        "audit_accessed",
        "val2017_accessed",
        "final_accessed",
        "registered_final_accessed",
    ):
        if protocol.get(key) is not False:
            raise ValueError(f"Forbidden role flag changed: {key}")


def _check_frontier_tensors(
    values: torch.Tensor,
    costs: torch.Tensor,
    action_mask: torch.Tensor,
    prices: torch.Tensor,
) -> None:
    if values.ndim != 2 or values.shape != costs.shape:
        raise ValueError("Values and costs must be equal [image, action] tensors")
    if action_mask.shape != values.shape:
        raise ValueError("Action mask shape differs")
    if prices.ndim not in (1, 2) or not prices.numel():
        raise ValueError("Prices must be a non-empty vector or per-image matrix")
    if prices.ndim == 2 and prices.shape[0] != values.shape[0]:
        raise ValueError("Per-image prices have the wrong population")
    if bool((costs < 0).any()) or bool((prices < 0).any()):
        raise ValueError("Costs and prices must be non-negative")
    if not bool(action_mask.to(torch.bool).any(dim=1).all()):
        raise ValueError("Every image requires one available action")


def frontier_utilities(
    values: torch.Tensor,
    costs: torch.Tensor,
    action_mask: torch.Tensor,
    prices: torch.Tensor,
) -> torch.Tensor:
    """Return the parametric action lines V(a)-price*C(a)."""

    _check_frontier_tensors(values, costs, action_mask, prices)
    expanded_prices = (
        prices[None, :, None] if prices.ndim == 1 else prices[:, :, None]
    )
    utility = values[:, None, :] - expanded_prices * costs[:, None, :]
    return utility.masked_fill(~action_mask[:, None, :].to(torch.bool), -torch.inf)


def frontier_target_actions(
    rewards: torch.Tensor,
    costs: torch.Tensor,
    action_mask: torch.Tensor,
    prices: torch.Tensor,
) -> torch.Tensor:
    """Exact teacher action along each image's shadow-price path."""

    return frontier_utilities(rewards, costs, action_mask, prices).argmax(dim=-1)


def boundary_weights(
    top_two_margin: torch.Tensor,
    *,
    scale: float,
    boost: float,
) -> torch.Tensor:
    if float(scale) <= 0.0 or float(boost) < 0.0:
        raise ValueError("Boundary scale must be positive and boost non-negative")
    return 1.0 + float(boost) * torch.exp(
        -top_two_margin.clamp_min(0.0) / float(scale)
    )


def frontier_probe_prices(
    rewards: torch.Tensor,
    costs: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    relative_offset: float,
    maximum_price: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Probe just before/after every valid pairwise action-line crossing."""

    placeholder = torch.tensor([0.0], dtype=rewards.dtype, device=rewards.device)
    _check_frontier_tensors(rewards, costs, action_mask, placeholder)
    if not 0.0 < float(relative_offset) < 1.0:
        raise ValueError("Relative offset must lie in (0, 1)")
    if float(maximum_price) <= 0.0:
        raise ValueError("Maximum price must be positive")
    action_count = rewards.shape[1]
    left, right = torch.triu_indices(
        action_count, action_count, offset=1, device=rewards.device
    )
    reward_difference = rewards[:, left] - rewards[:, right]
    cost_difference = costs[:, left] - costs[:, right]
    pair_available = (
        action_mask[:, left].to(torch.bool)
        & action_mask[:, right].to(torch.bool)
        & (cost_difference.abs() > 1e-12)
    )
    safe_difference = torch.where(
        cost_difference.abs() > 1e-12,
        cost_difference,
        torch.ones_like(cost_difference),
    )
    crossing = reward_difference / safe_difference
    valid = (
        pair_available
        & torch.isfinite(crossing)
        & (crossing > 0.0)
        & (crossing < float(maximum_price))
    )
    crossing = torch.where(
        valid,
        crossing.clamp(min=0.0, max=float(maximum_price)),
        torch.zeros_like(crossing),
    )
    low = crossing * (1.0 - float(relative_offset))
    high = crossing * (1.0 + float(relative_offset))
    batch = rewards.shape[0]
    endpoints = torch.tensor(
        [0.0, float(maximum_price)], dtype=rewards.dtype, device=rewards.device
    ).expand(batch, 2)
    endpoint_mask = torch.ones_like(endpoints, dtype=torch.bool)
    prices = torch.cat((endpoints, low, high), dim=1)
    price_mask = torch.cat((endpoint_mask, valid, valid), dim=1)
    return prices, price_mask


def frontier_path_loss(
    predicted_values: torch.Tensor,
    rewards: torch.Tensor,
    costs: torch.Tensor,
    action_mask: torch.Tensor,
    prices: torch.Tensor,
    *,
    price_mask: torch.Tensor | None = None,
    temperature: float,
    boundary_scale: float,
    boundary_boost: float,
) -> torch.Tensor:
    """Distill the complete parametric upper-envelope action path.

    The learner still emits one static value per action.  Known cost and a
    shadow price are applied analytically, so payload never becomes a learned
    value feature.
    """

    if predicted_values.shape != rewards.shape:
        raise ValueError("Predicted and teacher values differ in shape")
    if float(temperature) <= 0.0:
        raise ValueError("Temperature must be positive")
    teacher_utility = frontier_utilities(rewards, costs, action_mask, prices)
    predicted_utility = frontier_utilities(
        predicted_values, costs, action_mask, prices
    )
    target = teacher_utility.argmax(dim=-1)
    action_count = rewards.shape[1]
    losses = F.cross_entropy(
        (predicted_utility / float(temperature)).reshape(-1, action_count),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(target)
    top_two = teacher_utility.topk(k=min(2, action_count), dim=-1).values
    margin = (
        top_two[..., 0] - top_two[..., 1]
        if action_count >= 2
        else torch.full_like(top_two[..., 0], math.inf)
    )
    weights = boundary_weights(
        margin, scale=float(boundary_scale), boost=float(boundary_boost)
    )
    if price_mask is not None:
        if price_mask.shape != losses.shape:
            raise ValueError("Price mask shape differs from the path")
        weights = weights * price_mask.to(dtype=weights.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


def evaluate_screen_gate(
    rows: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        (mode, budget) for mode in EXPECTED_MODES for budget in EXPECTED_BUDGETS
    }
    observed = {(str(row["mode"]), float(row["budget"])) for row in rows}
    if len(rows) != len(expected) or observed != expected:
        raise ValueError("Gate rows do not cover the frozen work points")
    byte = [row for row in rows if str(row["mode"]) == "fixed_byte"]
    call = [row for row in rows if str(row["mode"]) == "fixed_call"]
    minimum = float(gate["minimum_fixed_byte_gain"])
    maximum_other = float(gate["maximum_other_loss"])
    maximum_call = float(gate["maximum_fixed_call_loss"])
    gain_points = sum(float(row["delta_ap"]) >= minimum for row in byte)
    byte_safe = all(float(row["delta_ap"]) >= -maximum_other for row in byte)
    call_safe = all(float(row["delta_ap"]) >= -maximum_call for row in call)
    minimum_capacity = float(gate.get("minimum_capacity_utilization", 0.0))
    capacity_sufficient = all(
        float(row.get("capacity_utilization", 1.0)) >= minimum_capacity
        for row in byte
    )
    passed = (
        gain_points >= int(gate["required_fixed_byte_gain_points"])
        and byte_safe
        and call_safe
        and capacity_sufficient
    )
    return {
        "passed": bool(passed),
        "fixed_byte_gain_points": int(gain_points),
        "fixed_byte_noninferior": bool(byte_safe),
        "fixed_call_noninferior": bool(call_safe),
        "capacity_sufficient": bool(capacity_sufficient),
    }


def evaluate_confirm_gate(
    seed_rows: Sequence[Mapping[str, Any]],
    ensemble_rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Require an ensemble point gate plus repeatable per-seed byte gains."""

    ensemble = evaluate_screen_gate(ensemble_rows, gate)
    minimum = float(gate["minimum_fixed_byte_gain"])
    required_seeds = int(gate["minimum_supporting_seeds"])
    support = {}
    for budget in EXPECTED_BUDGETS:
        rows = [
            row
            for row in seed_rows
            if str(row["mode"]) == "fixed_byte"
            and math.isclose(float(row["budget"]), budget)
        ]
        seeds = {str(row["seed"]) for row in rows}
        if len(rows) != len(seeds) or len(rows) < required_seeds:
            raise ValueError(f"Incomplete seed support at budget {budget}")
        support[str(budget)] = {
            "supporting_seeds": sum(float(row["delta_ap"]) >= minimum for row in rows),
            "seed_deltas": {
                str(row["seed"]): float(row["delta_ap"]) for row in rows
            },
        }
    supported_points = sum(
        value["supporting_seeds"] >= required_seeds for value in support.values()
    )
    passed = bool(
        ensemble["passed"]
        and supported_points >= int(gate["required_fixed_byte_gain_points"])
    )
    return {
        "passed": passed,
        "ensemble_gate": ensemble,
        "seed_support": support,
        "supported_fixed_byte_points": int(supported_points),
    }
