#!/usr/bin/env python3
"""Freeze a causal byte price using calibration records only."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BOARD_SCRIPTS = ROOT / "board_bundle" / "scripts"
if str(BOARD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BOARD_SCRIPTS))

from online_control import CausalByteController
from control_schema import CONTROL_SCHEMA, budget_fraction, validate_formal_control, validate_sha256


CALIBRATION_RECORDS_SCHEMA = "cavs_online_calibration_records_v1"
RESERVED_SOURCE_NAMES = {"calibration_records", "protocol"}
TIE_BREAK = ["objective_desc", "spent_bytes_asc", "lambda_asc"]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path, *, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read {field} JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _source_hashes(bundle: Mapping[str, Any]) -> dict[str, str]:
    hashes = bundle.get("source_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("calibration records require source_hashes")
    normalized: dict[str, str] = {}
    for name, digest in hashes.items():
        source_name = str(name)
        if not source_name or source_name in RESERVED_SOURCE_NAMES:
            raise ValueError(f"invalid calibration source name: {source_name}")
        normalized[source_name] = validate_sha256(digest, field=f"source_hashes.{source_name}")
    return normalized


def load_calibration_inputs(
    *,
    records_path: str | Path,
    protocol_path: str | Path,
    source_files: Mapping[str, str | Path],
) -> tuple[Mapping[str, Any], dict[str, str]]:
    """Load calibration inputs and verify every declared source against disk."""

    bundle = _read_json(records_path, field="calibration records")
    protocol = _read_json(protocol_path, field="protocol")
    if str(bundle.get("schema")) != CALIBRATION_RECORDS_SCHEMA:
        raise ValueError("unsupported calibration records schema")
    if str(bundle.get("source_split")) != "calibration":
        raise ValueError("formal controls may only use the calibration split")
    protocol_id = str(protocol.get("protocol_id", ""))
    if not protocol_id or str(bundle.get("protocol_id")) != protocol_id:
        raise ValueError("calibration records and protocol_id do not match")

    declared = _source_hashes(bundle)
    supplied = {str(name): Path(path) for name, path in source_files.items()}
    if set(supplied) != set(declared):
        missing = sorted(set(declared) - set(supplied))
        extra = sorted(set(supplied) - set(declared))
        raise ValueError(f"source file set differs from declared hashes: missing={missing}, extra={extra}")
    for name in sorted(declared):
        actual = file_sha256(supplied[name])
        if actual != declared[name]:
            raise ValueError(f"{name} SHA-256 does not match calibration records")

    hashes = dict(declared)
    hashes["calibration_records"] = file_sha256(records_path)
    hashes["protocol"] = file_sha256(protocol_path)
    return bundle, hashes


def _finite(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _lambda_grid(values: Sequence[float]) -> list[float]:
    normalized = []
    for value in values:
        price = _finite(value, field="lambda grid")
        normalized.append(0.0 if price == 0.0 else price)
    grid = sorted(set(normalized))
    if not grid or any(value < 0 for value in grid):
        raise ValueError("lambda grid must contain non-negative values")
    return grid


def _validated_bundle(bundle: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...], list[dict[str, Any]]]:
    if str(bundle.get("schema")) != CALIBRATION_RECORDS_SCHEMA:
        raise ValueError("unsupported calibration records schema")
    if str(bundle.get("source_split")) != "calibration":
        raise ValueError("formal controls may only use the calibration split")
    protocol_id = str(bundle.get("protocol_id", ""))
    method = str(bundle.get("method", ""))
    if not protocol_id or not method:
        raise ValueError("calibration records require protocol_id and method")
    _source_hashes(bundle)

    objective = bundle.get("calibration_objective")
    if not isinstance(objective, Mapping) or not str(objective.get("name", "")):
        raise ValueError("a calibration objective name must be declared")
    if str(objective.get("direction")) != "maximize":
        raise ValueError("calibration objective direction must be maximize")

    raw_actions = bundle.get("action_ids")
    if not isinstance(raw_actions, list):
        raise ValueError("action_ids must be a list")
    actions = tuple(map(str, raw_actions))
    if not actions or actions[0] != "edge" or "original" not in actions or len(actions) != len(set(actions)):
        raise ValueError("action_ids must be unique, begin with edge, and include original")

    raw_records = bundle.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("calibration records must be a non-empty list")
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"records[{index}] must be an object")
        raw_image_id = raw.get("image_id")
        if isinstance(raw_image_id, bool) or not isinstance(raw_image_id, int):
            raise ValueError("calibration image IDs must be integers")
        image_id = raw_image_id
        if image_id <= 0 or image_id in seen:
            raise ValueError("calibration image IDs must be positive and unique")
        seen.add(image_id)
        payload_raw = raw.get("payload_bytes")
        scores_raw = raw.get("scores")
        values_raw = raw.get("objective_values")
        if not all(isinstance(value, Mapping) for value in (payload_raw, scores_raw, values_raw)):
            raise ValueError(f"records[{index}] requires payload_bytes, scores, and objective_values")
        payloads: dict[str, int] = {}
        for name, value in payload_raw.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"records[{index}] payload bytes must be integers")
            payloads[str(name)] = value
        scores = {str(name): _finite(value, field="score") for name, value in scores_raw.items()}
        values = {str(name): _finite(value, field="objective value") for name, value in values_raw.items()}
        if set(payloads) != set(scores) or set(payloads) != set(values):
            raise ValueError(f"records[{index}] action fields differ")
        if not set(payloads) <= set(actions):
            raise ValueError(f"records[{index}] includes an undeclared action")
        if payloads.get("edge") != 0 or payloads.get("original", 0) <= 0:
            raise ValueError(f"records[{index}] requires zero-byte edge and positive-byte original")
        if any(value < 0 for value in payloads.values()):
            raise ValueError(f"records[{index}] contains negative payload bytes")
        records.append({
            "image_id": image_id,
            "payload_bytes": payloads,
            "scores": scores,
            "objective_values": values,
        })
    return protocol_id, method, actions, records


def _replay(
    records: Sequence[Mapping[str, Any]],
    *,
    actions: Sequence[str],
    budget: Fraction,
    price: float,
    median_original_bytes: float,
) -> dict[str, Any]:
    controller = CausalByteController(
        budget=budget,
        price=price,
        median_original_bytes=median_original_bytes,
        action_ids=actions,
    )
    objective_value = 0.0
    counts: Counter[str] = Counter()
    maximum_violation = 0
    for record in records:
        decision = controller.decide(
            int(record["image_id"]),
            record["payload_bytes"],
            record["scores"],
        )
        objective_value += float(record["objective_values"][decision.action])
        counts[decision.action] += 1
        earned_floor = (
            decision.cumulative_original_bytes * controller.budget_numerator
            // controller.budget_denominator
        )
        maximum_violation = max(maximum_violation, decision.spent_bytes - earned_floor)
    snapshot = controller.snapshot()
    earned = (
        snapshot["cumulative_original_bytes"] * controller.budget_numerator
        / controller.budget_denominator
    )
    return {
        "lambda": float(price),
        "objective_value": float(objective_value),
        "spent_bytes": int(snapshot["spent_bytes"]),
        "earned_bytes": float(earned),
        "capacity_utilization": float(snapshot["spent_bytes"] / earned) if earned else 0.0,
        "action_counts": {name: int(counts[name]) for name in sorted(counts)},
        "maximum_prefix_violation_bytes": int(maximum_violation),
    }


def calibrate_control(
    bundle: Mapping[str, Any],
    *,
    budget: str | float | Fraction,
    lambda_grid: Sequence[float],
    calibration_records_sha256: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Causally replay one calibration stream and freeze one formal control."""

    protocol_id, method, actions, records = _validated_bundle(bundle)
    random_seed = bundle.get("random_seed")
    if random_seed is not None and (
        isinstance(random_seed, bool) or not isinstance(random_seed, int)
    ):
        raise ValueError("random_seed must be an integer or null")
    if method == "random" and random_seed is None:
        raise ValueError("random calibration records must bind random_seed")
    fraction = budget_fraction(budget)
    grid = _lambda_grid(lambda_grid)
    median = float(statistics.median(record["payload_bytes"]["original"] for record in records))
    trials = [
        _replay(
            records,
            actions=actions,
            budget=fraction,
            price=price,
            median_original_bytes=median,
        )
        for price in grid
    ]
    chosen = max(
        trials,
        key=lambda row: (
            row["objective_value"],
            -row["spent_bytes"],
            -row["lambda"],
        ),
    )
    objective = bundle["calibration_objective"]
    source_hashes = dict(_source_hashes(bundle))
    source_hashes["calibration_records"] = validate_sha256(
        calibration_records_sha256, field="calibration_records_sha256"
    )
    source_hashes["protocol"] = validate_sha256(protocol_sha256, field="protocol_sha256")
    control = {
        "schema": CONTROL_SCHEMA,
        "formal": True,
        "protocol_id": protocol_id,
        "method": method,
        "budget": float(fraction),
        "budget_numerator": int(fraction.numerator),
        "budget_denominator": int(fraction.denominator),
        "lambda": float(chosen["lambda"]),
        "median_original_bytes": median,
        "calibration_frames": len(records),
        "calibration_objective": {
            "name": str(objective["name"]),
            "direction": "maximize",
            "value": float(chosen["objective_value"]),
        },
        "lambda_grid": grid,
        "tie_break": list(TIE_BREAK),
        "source_split": "calibration",
        "source_hashes": source_hashes,
        "random_seed": (
            int(bundle["random_seed"])
            if bundle.get("random_seed") is not None else None
        ),
        "final_used_for_tuning": False,
        "diagnostics": {
            key: value for key, value in chosen.items() if key not in {"lambda", "objective_value"}
        },
        "calibration_trials": trials,
    }
    validate_formal_control(control)
    return control


def _parse_sources(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in output:
            raise ValueError("each --source must be a unique NAME=PATH")
        output[name] = Path(path)
    return output


def _parse_grid(value: str) -> list[float]:
    try:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--lambda-grid must be comma-separated numbers") from error


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="Calibration-record JSON bundle")
    parser.add_argument("--protocol", required=True, help="Frozen protocol JSON")
    parser.add_argument("--method", required=True, help="Expected method identifier")
    parser.add_argument("--budget", required=True, help="Budget fraction, for example 0.1 or 1/10")
    parser.add_argument("--objective", required=True, help="Expected declared calibration objective")
    parser.add_argument("--lambda-grid", required=True, help="Comma-separated non-negative prices")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="File for each source hash declared by the calibration bundle",
    )
    parser.add_argument("--output", required=True, help="Destination formal-control JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle, hashes = load_calibration_inputs(
        records_path=args.records,
        protocol_path=args.protocol,
        source_files=_parse_sources(args.source),
    )
    if str(bundle.get("method")) != args.method:
        raise ValueError("--method does not match calibration records")
    objective = bundle.get("calibration_objective")
    if not isinstance(objective, Mapping) or str(objective.get("name")) != args.objective:
        raise ValueError("--objective does not match calibration records")
    control = calibrate_control(
        bundle,
        budget=args.budget,
        lambda_grid=_parse_grid(args.lambda_grid),
        calibration_records_sha256=hashes["calibration_records"],
        protocol_sha256=hashes["protocol"],
    )
    _write_json(args.output, control)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
