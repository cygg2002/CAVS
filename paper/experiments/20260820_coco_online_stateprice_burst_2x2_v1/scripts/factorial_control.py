"""Zero-buffer controllers for the online price × byte-envelope factorial."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BudgetEnvelope:
    mode: str
    total_capacity: int
    total_frames: int
    budget: Fraction | None = None
    initial_burst_bytes: int = 0

    @classmethod
    def strict_prefix(
        cls,
        *,
        budget: Fraction,
        total_capacity: int,
        total_frames: int,
    ) -> "BudgetEnvelope":
        if budget <= 0 or budget > 1:
            raise ValueError("budget must be in (0, 1]")
        return cls._validated(
            mode="strict_prefix",
            budget=budget,
            total_capacity=total_capacity,
            total_frames=total_frames,
            initial_burst_bytes=0,
        )

    @classmethod
    def paced_burst(
        cls,
        *,
        total_capacity: int,
        total_frames: int,
        initial_burst_bytes: int,
    ) -> "BudgetEnvelope":
        return cls._validated(
            mode="paced_burst",
            budget=None,
            total_capacity=total_capacity,
            total_frames=total_frames,
            initial_burst_bytes=initial_burst_bytes,
        )

    @classmethod
    def _validated(
        cls,
        *,
        mode: str,
        budget: Fraction | None,
        total_capacity: int,
        total_frames: int,
        initial_burst_bytes: int,
    ) -> "BudgetEnvelope":
        capacity, frames, burst = map(
            int, (total_capacity, total_frames, initial_burst_bytes)
        )
        if capacity <= 0 or frames <= 0:
            raise ValueError("total capacity and frames must be positive")
        if burst < 0 or burst > capacity:
            raise ValueError("initial burst must lie in [0, total_capacity]")
        return cls(
            mode=mode,
            budget=budget,
            total_capacity=capacity,
            total_frames=frames,
            initial_burst_bytes=burst,
        )

    def limit(self, *, position: int, cumulative_original_bytes: int) -> int:
        position = int(position)
        cumulative = int(cumulative_original_bytes)
        if position <= 0 or position > self.total_frames or cumulative < 0:
            raise ValueError("invalid envelope position or observed bytes")
        if self.mode == "strict_prefix":
            assert self.budget is not None
            return cumulative * self.budget.numerator // self.budget.denominator
        if self.mode == "paced_burst":
            remainder = self.total_capacity - self.initial_burst_bytes
            return self.initial_burst_bytes + remainder * position // self.total_frames
        raise RuntimeError(f"unsupported envelope mode: {self.mode}")


@dataclass(frozen=True)
class DemandEntry:
    price: float
    mean_normalized_bytes: float
    mean_objective: float


@dataclass(frozen=True)
class EmpiricalDemandCurve:
    entries: tuple[DemandEntry, ...]

    def for_target_normalized_bytes(self, target: float) -> DemandEntry:
        target = float(target)
        if not math.isfinite(target) or target < 0:
            raise ValueError("target normalized bytes must be finite and non-negative")
        feasible = [
            entry
            for entry in self.entries
            if entry.mean_normalized_bytes <= target + 1e-12
        ]
        if not feasible:
            return min(self.entries, key=lambda entry: entry.mean_normalized_bytes)
        return max(
            feasible,
            key=lambda entry: (
                entry.mean_objective,
                -entry.mean_normalized_bytes,
                -entry.price,
            ),
        )


@dataclass(frozen=True)
class ImmediateDecision:
    position: int
    frame_id: int
    action: str
    payload_bytes: int
    original_bytes: int
    cumulative_original_bytes: int
    spent_bytes: int
    capacity_limit_bytes: int
    price: float
    net_value: float


def _validated_actions(action_ids: Sequence[str]) -> tuple[str, ...]:
    actions = tuple(map(str, action_ids))
    if not actions or actions[0] != "edge" or "original" not in actions:
        raise ValueError("action_ids must begin with edge and include original")
    if len(actions) != len(set(actions)):
        raise ValueError("action_ids must be unique")
    return actions


def _validated_candidates(
    record: Mapping[str, Any], action_ids: Sequence[str]
) -> tuple[int, dict[str, int], dict[str, float]]:
    try:
        frame_id = int(record["image_id"])
        payloads = {str(key): int(value) for key, value in record["payload_bytes"].items()}
        scores = {str(key): float(value) for key, value in record["scores"].items()}
    except (KeyError, AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("record candidates are invalid") from error
    if payloads.get("edge") != 0 or scores.get("edge") is None:
        raise ValueError("record requires zero-byte Edge")
    if payloads.get("original", 0) <= 0 or set(payloads) != set(scores):
        raise ValueError("record action candidates differ or lack Original")
    if not set(payloads) <= set(action_ids):
        raise ValueError("record contains an action outside the contract")
    if any(value < 0 for value in payloads.values()):
        raise ValueError("payload bytes must be non-negative")
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("scores must be finite")
    scores["edge"] = 0.0
    return frame_id, payloads, scores


def _action_for_price(
    *,
    payloads: Mapping[str, int],
    scores: Mapping[str, float],
    action_ids: Sequence[str],
    median_original_bytes: float,
    price: float,
    feasible: Sequence[str] | None = None,
) -> str:
    candidates = feasible if feasible is not None else [
        action for action in action_ids if action in payloads
    ]

    def key(action: str) -> tuple[float, int, int]:
        net = scores[action] - price * payloads[action] / median_original_bytes
        return net, -payloads[action], -action_ids.index(action)

    return max(candidates, key=key)


def _candidate_prices(
    records: Sequence[Mapping[str, Any]],
    *,
    action_ids: Sequence[str],
    median_original_bytes: float,
) -> list[float]:
    breakpoints = {0.0}
    for record in records:
        _, payloads, scores = _validated_candidates(record, action_ids)
        available = [action for action in action_ids if action in payloads]
        for left_index, left in enumerate(available):
            for right in available[left_index + 1 :]:
                cost_delta = payloads[left] - payloads[right]
                if cost_delta == 0:
                    continue
                price = (
                    (scores[left] - scores[right])
                    * median_original_bytes
                    / cost_delta
                )
                if math.isfinite(price) and price >= 0:
                    breakpoints.add(float(price))
    ordered = sorted(breakpoints)
    prices = set(ordered)
    prices.update((left + right) / 2 for left, right in zip(ordered, ordered[1:]))
    terminal = (ordered[-1] if ordered else 0.0) + max(1.0, abs(ordered[-1]) * 0.01)
    prices.add(float(terminal))
    return sorted(prices)


def build_empirical_demand_curve(
    records: Sequence[Mapping[str, Any]],
    *,
    action_ids: Sequence[str],
    median_original_bytes: float,
) -> EmpiricalDemandCurve:
    actions = _validated_actions(action_ids)
    median = float(median_original_bytes)
    if not records or not math.isfinite(median) or median <= 0:
        raise ValueError("records and positive median bytes are required")
    validated = [_validated_candidates(record, actions)[1:] for record in records]
    entries = []
    for price in _candidate_prices(
        records, action_ids=actions, median_original_bytes=median
    ):
        total_bytes = 0
        total_objective = 0.0
        for payloads, scores in validated:
            action = _action_for_price(
                payloads=payloads,
                scores=scores,
                action_ids=actions,
                median_original_bytes=median,
                price=price,
            )
            total_bytes += payloads[action]
            total_objective += scores[action]
        entries.append(DemandEntry(
            price=float(price),
            mean_normalized_bytes=float(total_bytes / len(validated) / median),
            mean_objective=float(total_objective / len(validated)),
        ))
    return EmpiricalDemandCurve(tuple(entries))


class ImmediateController:
    """Decide one current record immediately; no future record is accepted."""

    def __init__(
        self,
        *,
        action_ids: Sequence[str],
        median_original_bytes: float,
        envelope: BudgetEnvelope,
        fixed_price: float | None,
        demand_curve: EmpiricalDemandCurve | None,
    ) -> None:
        self.action_ids = _validated_actions(action_ids)
        self.median_original_bytes = float(median_original_bytes)
        if not math.isfinite(self.median_original_bytes) or self.median_original_bytes <= 0:
            raise ValueError("median original bytes must be positive")
        if (fixed_price is None) == (demand_curve is None):
            raise ValueError("exactly one price policy is required")
        if fixed_price is not None and (
            not math.isfinite(float(fixed_price)) or float(fixed_price) < 0
        ):
            raise ValueError("fixed price must be finite and non-negative")
        self.envelope = envelope
        self.fixed_price = float(fixed_price) if fixed_price is not None else None
        self.demand_curve = demand_curve
        self.cumulative_original_bytes = 0
        self.spent_bytes = 0
        self.frames = 0

    @classmethod
    def fixed_price(cls, *, price: float, **kwargs: Any) -> "ImmediateController":
        return cls(fixed_price=price, demand_curve=None, **kwargs)

    @classmethod
    def state_aware(
        cls, *, demand_curve: EmpiricalDemandCurve, **kwargs: Any
    ) -> "ImmediateController":
        return cls(fixed_price=None, demand_curve=demand_curve, **kwargs)

    def _price(self) -> float:
        if self.fixed_price is not None:
            return self.fixed_price
        assert self.demand_curve is not None
        remaining_frames = self.envelope.total_frames - self.frames
        remaining_bytes = max(0, self.envelope.total_capacity - self.spent_bytes)
        target = remaining_bytes / remaining_frames / self.median_original_bytes
        return self.demand_curve.for_target_normalized_bytes(target).price

    def decide(self, record: Mapping[str, Any]) -> ImmediateDecision:
        if self.frames >= self.envelope.total_frames:
            raise RuntimeError("controller received more frames than configured")
        frame_id, payloads, scores = _validated_candidates(record, self.action_ids)
        position = self.frames + 1
        original = payloads["original"]
        cumulative = self.cumulative_original_bytes + original
        capacity = self.envelope.limit(
            position=position,
            cumulative_original_bytes=cumulative,
        )
        feasible = [
            action
            for action in self.action_ids
            if action in payloads and self.spent_bytes + payloads[action] <= capacity
        ]
        if "edge" not in feasible:
            raise RuntimeError("Edge must remain feasible")
        price = self._price()
        action = _action_for_price(
            payloads=payloads,
            scores=scores,
            action_ids=self.action_ids,
            median_original_bytes=self.median_original_bytes,
            price=price,
            feasible=feasible,
        )
        cost = payloads[action]
        self.cumulative_original_bytes = cumulative
        self.spent_bytes += cost
        self.frames = position
        if self.spent_bytes > capacity:
            raise RuntimeError("byte envelope exceeded")
        return ImmediateDecision(
            position=position,
            frame_id=frame_id,
            action=action,
            payload_bytes=cost,
            original_bytes=original,
            cumulative_original_bytes=cumulative,
            spent_bytes=self.spent_bytes,
            capacity_limit_bytes=capacity,
            price=float(price),
            net_value=float(scores[action] - price * cost / self.median_original_bytes),
        )
