"""Deterministic zero-lookahead byte admission for an image stream."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class OnlineDecision:
    frame_id: int
    action: str
    payload_bytes: int
    original_bytes: int
    cumulative_original_bytes: int
    spent_bytes: int
    credit_floor_bytes: int
    net_value: float
    capacity_utilization: float


def _budget_fraction(value: str | float | Fraction) -> Fraction:
    budget = value if isinstance(value, Fraction) else Fraction(str(value))
    if budget <= 0 or budget > 1:
        raise ValueError("budget must be in (0, 1]")
    return budget


class CausalByteController:
    """Select one current-frame action while enforcing every byte prefix."""

    SNAPSHOT_SCHEMA = 1

    def __init__(
        self,
        *,
        budget: str | float | Fraction,
        price: float,
        median_original_bytes: int | float,
        action_ids: Sequence[str],
    ) -> None:
        fraction = _budget_fraction(budget)
        actions = tuple(map(str, action_ids))
        if not actions or actions[0] != "edge" or len(actions) != len(set(actions)):
            raise ValueError("action_ids must be unique and begin with edge")
        if "original" not in actions:
            raise ValueError("action_ids must include original")
        if not math.isfinite(float(price)) or float(price) < 0:
            raise ValueError("price must be finite and non-negative")
        if not math.isfinite(float(median_original_bytes)) or float(median_original_bytes) <= 0:
            raise ValueError("median_original_bytes must be positive")
        self.budget_numerator = int(fraction.numerator)
        self.budget_denominator = int(fraction.denominator)
        self.price = float(price)
        self.median_original_bytes = float(median_original_bytes)
        self.action_ids = actions
        self.cumulative_original_bytes = 0
        self.spent_bytes = 0
        self.frames = 0

    @property
    def budget(self) -> Fraction:
        return Fraction(self.budget_numerator, self.budget_denominator)

    def _validate_candidates(
        self, payload_bytes: Mapping[str, int], scores: Mapping[str, float]
    ) -> tuple[dict[str, int], dict[str, float]]:
        payloads = {str(key): int(value) for key, value in payload_bytes.items()}
        values = {str(key): float(value) for key, value in scores.items()}
        if payloads.get("edge") != 0 or "edge" not in values:
            raise ValueError("each frame requires an Edge action with zero bytes")
        if payloads.get("original", 0) <= 0:
            raise ValueError("each frame requires positive Original bytes")
        if not set(payloads) <= set(self.action_ids):
            raise ValueError("payload contains an action outside the controller contract")
        if set(payloads) != set(values):
            raise ValueError("payload and score actions differ")
        if any(value < 0 for value in payloads.values()):
            raise ValueError("payload bytes must be non-negative")
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("action values must be finite")
        values["edge"] = 0.0
        return payloads, values

    def decide(
        self,
        frame_id: int,
        payload_bytes: Mapping[str, int],
        scores: Mapping[str, float],
    ) -> OnlineDecision:
        payloads, values = self._validate_candidates(payload_bytes, scores)
        original_bytes = payloads["original"]
        cumulative = self.cumulative_original_bytes + original_bytes
        earned_numerator = cumulative * self.budget_numerator
        feasible = [
            action
            for action in self.action_ids
            if action in payloads
            and (self.spent_bytes + payloads[action]) * self.budget_denominator
            <= earned_numerator
        ]
        if "edge" not in feasible:
            raise RuntimeError("Edge must always be feasible")

        def key(action: str) -> tuple[float, int, int]:
            net = values[action] - self.price * payloads[action] / self.median_original_bytes
            return net, -payloads[action], -self.action_ids.index(action)

        action = max(feasible, key=key)
        cost = payloads[action]
        net_value = values[action] - self.price * cost / self.median_original_bytes
        self.cumulative_original_bytes = cumulative
        self.spent_bytes += cost
        self.frames += 1
        if self.spent_bytes * self.budget_denominator > earned_numerator:
            raise RuntimeError("causal byte capacity exceeded")
        earned = earned_numerator / self.budget_denominator
        credit_floor = earned_numerator // self.budget_denominator - self.spent_bytes
        return OnlineDecision(
            frame_id=int(frame_id),
            action=action,
            payload_bytes=cost,
            original_bytes=original_bytes,
            cumulative_original_bytes=self.cumulative_original_bytes,
            spent_bytes=self.spent_bytes,
            credit_floor_bytes=int(credit_floor),
            net_value=float(net_value),
            capacity_utilization=float(self.spent_bytes / earned) if earned else 0.0,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.SNAPSHOT_SCHEMA,
            "budget_numerator": self.budget_numerator,
            "budget_denominator": self.budget_denominator,
            "price": self.price,
            "median_original_bytes": self.median_original_bytes,
            "action_ids": list(self.action_ids),
            "cumulative_original_bytes": self.cumulative_original_bytes,
            "spent_bytes": self.spent_bytes,
            "frames": self.frames,
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "CausalByteController":
        if int(snapshot.get("schema_version", -1)) != cls.SNAPSHOT_SCHEMA:
            raise ValueError("unsupported controller snapshot")
        controller = cls(
            budget=Fraction(
                int(snapshot["budget_numerator"]),
                int(snapshot["budget_denominator"]),
            ),
            price=float(snapshot["price"]),
            median_original_bytes=float(snapshot["median_original_bytes"]),
            action_ids=list(snapshot["action_ids"]),
        )
        controller.cumulative_original_bytes = int(snapshot["cumulative_original_bytes"])
        controller.spent_bytes = int(snapshot["spent_bytes"])
        controller.frames = int(snapshot["frames"])
        if min(
            controller.cumulative_original_bytes,
            controller.spent_bytes,
            controller.frames,
        ) < 0:
            raise ValueError("controller snapshot contains negative state")
        if (
            controller.spent_bytes * controller.budget_denominator
            > controller.cumulative_original_bytes * controller.budget_numerator
        ):
            raise ValueError("controller snapshot exceeds its causal capacity")
        return controller


def decision_dict(decision: OnlineDecision) -> dict[str, Any]:
    return asdict(decision)

