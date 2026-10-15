#!/usr/bin/env python3
"""Summarize byte, action, failure, throughput, and latency evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile / 100.0
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def value_summary(values: Sequence[float]) -> dict[str, float | int]:
    clean = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in clean):
        raise ValueError("timing values must be finite and non-negative")
    return {
        "samples": len(clean),
        "mean": float(statistics.fmean(clean)) if clean else 0.0,
        "p50": _percentile(clean, 50.0),
        "p95": _percentile(clean, 95.0),
        "p99": _percentile(clean, 99.0),
    }


def summarize_frames(
    rows: Sequence[Mapping[str, Any]],
    *,
    budget_numerator: int,
    budget_denominator: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty online run")
    if budget_numerator <= 0 or budget_denominator <= 0 or budget_numerator > budget_denominator:
        raise ValueError("invalid budget fraction")
    counts: Counter[str] = Counter()
    e2e_values: list[float] = []
    decision_values: list[float] = []
    execution_values: list[float] = []
    stage_values: dict[str, list[float]] = {}
    maximum_violation = 0
    previous_spent = 0
    previous_original = 0
    selected_payload = 0
    wire_request = 0
    wire_response = 0
    failures = 0
    previous_done: int | None = None
    first_start: int | None = None
    last_done: int | None = None

    for index, row in enumerate(rows):
        decision = row.get("decision")
        if not isinstance(decision, Mapping):
            raise ValueError(f"frame {index} has no decision")
        start = int(row["frame_start_ns"])
        decided = int(row["decision_ns"])
        done = int(row["frame_done_ns"])
        if not start <= decided <= done or (previous_done is not None and start < previous_done):
            raise RuntimeError(f"invalid causal event order at frame {index}")
        previous_done = done
        first_start = start if first_start is None else first_start
        last_done = done
        e2e_values.append((done - start) / 1_000_000.0)
        decision_values.append((decided - start) / 1_000_000.0)
        execution_values.append((done - decided) / 1_000_000.0)

        action = str(decision["action"])
        payload = int(decision["payload_bytes"])
        cumulative_original = int(decision["cumulative_original_bytes"])
        spent = int(decision["spent_bytes"])
        if min(payload, cumulative_original, spent) < 0:
            raise ValueError("byte counters must be non-negative")
        if cumulative_original < previous_original or spent < previous_spent:
            raise RuntimeError("online byte counters moved backwards")
        expected_spent = previous_spent + payload
        if spent != expected_spent:
            raise RuntimeError("selected payload does not match cumulative spent bytes")
        earned_floor = cumulative_original * budget_numerator // budget_denominator
        maximum_violation = max(maximum_violation, spent - earned_floor)
        previous_spent, previous_original = spent, cumulative_original
        selected_payload += payload
        counts[action] += 1

        cloud = row.get("cloud", {})
        if not isinstance(cloud, Mapping):
            raise ValueError("cloud audit record must be an object")
        wire_request += int(cloud.get("request_body_bytes", 0))
        wire_response += int(cloud.get("response_body_bytes", 0))
        failures += row.get("execution_status") == "fallback"
        timings = row.get("timings_ms", {})
        if not isinstance(timings, Mapping):
            raise ValueError("timings_ms must be an object")
        for name, value in timings.items():
            stage_values.setdefault(str(name), []).append(float(value))

    if maximum_violation > 0:
        raise RuntimeError(f"prefix byte budget exceeded by {maximum_violation} bytes")
    capacity_floor = previous_original * budget_numerator // budget_denominator
    wall_seconds = ((last_done or 0) - (first_start or 0)) / 1_000_000_000.0
    return {
        "schema": "cavs_online_system_summary_v1",
        "frames": len(rows),
        "action_counts": dict(sorted(counts.items())),
        "uploads": sum(count for action, count in counts.items() if action != "edge"),
        "cloud_failures": int(failures),
        "selected_payload_bytes": selected_payload,
        "cumulative_original_bytes": previous_original,
        "capacity_floor_bytes": capacity_floor,
        "capacity_utilization": selected_payload / capacity_floor if capacity_floor else 0.0,
        "maximum_prefix_violation_bytes": maximum_violation,
        "wire_request_body_bytes": wire_request,
        "wire_response_body_bytes": wire_response,
        "stream_wall_seconds": wall_seconds,
        "stream_fps": len(rows) / wall_seconds if wall_seconds > 0 else 0.0,
        "e2e_ms": value_summary(e2e_values),
        "decision_ready_ms": value_summary(decision_values),
        "execution_after_decision_ms": value_summary(execution_values),
        "timing_ms": {
            name: value_summary(values) for name, values in sorted(stage_values.items())
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid frames JSON at line {line_number}") from error
            if not isinstance(row, Mapping) or row.get("status") != "committed":
                raise ValueError(f"frame {line_number} is not committed")
            rows.append(dict(row))
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.run_contract.read_text(encoding="utf-8"))
    summary = summarize_frames(
        _read_jsonl(args.frames),
        budget_numerator=int(contract["budget_numerator"]),
        budget_denominator=int(contract["budget_denominator"]),
    )
    summary.update({
        "run_id": str(contract["run_id"]),
        "method": str(contract["method"]),
        "budget_numerator": int(contract["budget_numerator"]),
        "budget_denominator": int(contract["budget_denominator"]),
    })
    _atomic_json(args.output, summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
