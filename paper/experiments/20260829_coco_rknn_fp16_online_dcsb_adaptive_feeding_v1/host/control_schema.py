"""Validation for calibration-frozen formal online controls."""

from __future__ import annotations

from fractions import Fraction
import math
from typing import Any, Mapping


CONTROL_SCHEMA = "cavs_online_control_v1"


def validate_sha256(value: object, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return digest


def budget_fraction(value: str | float | Fraction) -> Fraction:
    try:
        budget = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("budget must be a fraction in (0, 1]") from error
    if budget <= 0 or budget > 1:
        raise ValueError("budget must be a fraction in (0, 1]")
    return budget


def _finite(value: object, *, field: str, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{field} must be {qualifier}")
    return number


def validate_formal_control(
    control: Mapping[str, Any],
    *,
    expected_protocol_id: str | None = None,
    expected_method: str | None = None,
    expected_budget: str | float | Fraction | None = None,
    expected_source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate a control before a formal run and return a plain copy."""

    if not isinstance(control, Mapping):
        raise ValueError("control must be a JSON object")
    if str(control.get("schema")) != CONTROL_SCHEMA:
        raise ValueError("unsupported online control schema")
    if control.get("formal") is not True:
        raise ValueError("formal control must be explicitly marked formal")
    if str(control.get("source_split")) != "calibration":
        raise ValueError("formal control must be frozen from calibration")
    if control.get("final_used_for_tuning") is not False:
        raise ValueError("formal control must state that final was not used for tuning")

    protocol_id = str(control.get("protocol_id", ""))
    method = str(control.get("method", ""))
    if not protocol_id or not method:
        raise ValueError("formal control requires protocol_id and method")
    if expected_protocol_id is not None and protocol_id != str(expected_protocol_id):
        raise ValueError("control protocol does not match the run contract")
    if expected_method is not None and method != str(expected_method):
        raise ValueError("control method does not match the run contract")

    random_seed = control.get("random_seed")
    if random_seed is not None and (
        isinstance(random_seed, bool) or not isinstance(random_seed, int)
    ):
        raise ValueError("random_seed must be an integer or null")
    if method == "random" and random_seed is None:
        raise ValueError("random control must bind random_seed")

    numerator = int(control.get("budget_numerator", 0))
    denominator = int(control.get("budget_denominator", 0))
    try:
        budget = budget_fraction(Fraction(numerator, denominator))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("control budget fraction is invalid") from error
    if expected_budget is not None and budget != budget_fraction(expected_budget):
        raise ValueError("control budget does not match the run contract")
    if not math.isclose(float(control.get("budget")), float(budget), rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("control budget fields disagree")

    price = _finite(control.get("lambda"), field="lambda")
    if price < 0:
        raise ValueError("lambda must be non-negative")
    _finite(control.get("median_original_bytes"), field="median_original_bytes", positive=True)
    frames = int(control.get("calibration_frames", 0))
    if frames <= 0:
        raise ValueError("calibration_frames must be positive")

    objective = control.get("calibration_objective")
    if not isinstance(objective, Mapping) or not str(objective.get("name", "")):
        raise ValueError("formal control requires a declared calibration objective")
    if str(objective.get("direction")) != "maximize":
        raise ValueError("calibration objective direction must be maximize")
    _finite(objective.get("value"), field="calibration objective value")

    grid_raw = control.get("lambda_grid")
    if not isinstance(grid_raw, list) or not grid_raw:
        raise ValueError("lambda_grid must be a non-empty list")
    grid = [_finite(value, field="lambda_grid") for value in grid_raw]
    if any(value < 0 for value in grid) or grid != sorted(set(grid)):
        raise ValueError("lambda_grid must contain sorted unique non-negative values")
    if price not in grid:
        raise ValueError("chosen lambda is not present in lambda_grid")

    hashes = control.get("source_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("formal control requires source hashes")
    normalized_hashes = {
        str(name): validate_sha256(digest, field=f"source_hashes.{name}")
        for name, digest in hashes.items()
    }
    for required in ("calibration_records", "protocol"):
        if required not in normalized_hashes:
            raise ValueError(f"formal control source hashes omit {required}")
    if expected_source_hashes is not None:
        for name, digest in expected_source_hashes.items():
            expected = validate_sha256(digest, field=f"expected_source_hashes.{name}")
            if normalized_hashes.get(str(name)) != expected:
                raise ValueError(f"control source hash does not match: {name}")

    return dict(control)
