#!/usr/bin/env python3
"""Replay fixed and adaptive CAVS controls on one frozen causal record stream."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BOARD_SCRIPTS = ROOT / "board_bundle" / "scripts"
if str(BOARD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BOARD_SCRIPTS))

from online_control import CausalByteController, PrimalDualByteController


CONTROL_SCHEMA = "cavs_online_primal_dual_control_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_control(control: Mapping[str, Any]) -> None:
    valid = (
        control.get("schema") == CONTROL_SCHEMA
        and control.get("controller") == "projected_primal_dual"
        and control.get("source_split") == "calibration"
        and control.get("final_used_for_tuning") is False
        and control.get("buffer_frames") == 0
        and control.get("zero_lookahead") is True
        and control.get("global_plan") is False
    )
    if not valid:
        raise ValueError("control is not a frozen zero-buffer causal control")


def _run(
    records: Sequence[Mapping[str, Any]],
    *,
    controller: CausalByteController,
) -> tuple[dict[str, Any], dict[str, str]]:
    objective = 0.0
    maximum_violation = 0
    counts: Counter[str] = Counter()
    selections: dict[str, str] = {}
    started = time.perf_counter_ns()
    for record in records:
        decision = controller.decide(
            int(record["image_id"]),
            record["payload_bytes"],
            record["scores"],
        )
        objective += float(record["scores"][decision.action])
        counts[decision.action] += 1
        selections[str(decision.frame_id)] = decision.action
        earned_floor = (
            decision.cumulative_original_bytes * controller.budget_numerator
            // controller.budget_denominator
        )
        maximum_violation = max(maximum_violation, decision.spent_bytes - earned_floor)
    elapsed_ns = time.perf_counter_ns() - started
    earned = float(controller.budget) * controller.cumulative_original_bytes
    return {
        "frames": len(records),
        "predicted_action_value_sum": float(objective),
        "spent_bytes": int(controller.spent_bytes),
        "earned_bytes": float(earned),
        "capacity_utilization": float(controller.spent_bytes / earned) if earned else 0.0,
        "action_counts": {name: int(counts[name]) for name in sorted(counts)},
        "maximum_prefix_violation_bytes": int(maximum_violation),
        "controller_elapsed_ns": int(elapsed_ns),
        "controller_ns_per_frame": float(elapsed_ns / len(records)) if records else 0.0,
        "final_price": float(controller.price),
    }, selections


def compare_replays(
    records: Sequence[Mapping[str, Any]],
    *,
    action_ids: Sequence[str],
    adaptive_control: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    _validate_control(adaptive_control)
    try:
        budget = Fraction(
            int(adaptive_control["budget_numerator"]),
            int(adaptive_control["budget_denominator"]),
        )
        initial_price = float(adaptive_control["lambda_initial"])
        median = float(adaptive_control["median_original_bytes"])
        step_size = float(adaptive_control["step_size"])
        lower = float(adaptive_control["lambda_minimum"])
        upper = float(adaptive_control["lambda_maximum"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("adaptive control parameters are invalid") from error
    fixed = CausalByteController(
        budget=budget,
        price=initial_price,
        median_original_bytes=median,
        action_ids=action_ids,
    )
    adaptive = PrimalDualByteController(
        budget=budget,
        initial_price=initial_price,
        step_size=step_size,
        minimum_price=lower,
        maximum_price=upper,
        median_original_bytes=median,
        action_ids=action_ids,
    )
    fixed_report, fixed_selections = _run(records, controller=fixed)
    adaptive_report, adaptive_selections = _run(records, controller=adaptive)
    transitions = Counter(
        f"{fixed_selections[image_id]}->{adaptive_selections[image_id]}"
        for image_id in fixed_selections
        if fixed_selections[image_id] != adaptive_selections[image_id]
    )
    report = {
        "complete": True,
        "frames": len(records),
        "method": str(adaptive_control.get("method")),
        "budget": float(budget),
        "controller": "projected_primal_dual",
        "zero_lookahead": True,
        "global_plan": False,
        "buffer_frames": 0,
        "step_size": step_size,
        "fixed": fixed_report,
        "adaptive": adaptive_report,
        "predicted_action_value_delta": (
            adaptive_report["predicted_action_value_sum"]
            - fixed_report["predicted_action_value_sum"]
        ),
        "changed_frames": sum(
            fixed_selections[key] != adaptive_selections[key]
            for key in fixed_selections
        ),
        "action_transitions": dict(sorted(transitions.items())),
        "final_used_for_tuning": False,
    }
    if step_size == 0.0 and fixed_selections != adaptive_selections:
        raise RuntimeError("zero-step adaptive replay differs from fixed control")
    return report, {"fixed": fixed_selections, "adaptive": adaptive_selections}


def load_records(frames_path: Path, sequence_path: Path) -> list[dict[str, Any]]:
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    expected = sequence.get("records") if isinstance(sequence, Mapping) else None
    if not isinstance(expected, list):
        raise ValueError("sequence must contain records")
    records = []
    with frames_path.open("r", encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            row = json.loads(line)
            if position >= len(expected):
                raise ValueError("frame log is longer than the sequence")
            item = expected[position]
            if (
                int(row.get("position", -1)) != position
                or int(row.get("image_id", -1)) != int(item["image_id"])
                or str(row.get("file_name", "")) != str(item["file_name"])
            ):
                raise ValueError("frame log and sequence order differ")
            payloads, scores = row.get("payload_bytes"), row.get("scores")
            if not isinstance(payloads, Mapping) or not isinstance(scores, Mapping):
                raise ValueError("frame record lacks current action candidates")
            records.append({
                "image_id": int(row["image_id"]),
                "payload_bytes": dict(payloads),
                "scores": dict(scores),
            })
    if len(records) != len(expected):
        raise ValueError("frame log is incomplete")
    return records


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    control = json.loads(args.control.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    records = load_records(args.frames, args.sequence)
    report, selections = compare_replays(
        records,
        action_ids=tuple(map(str, policy["action_ids"])),
        adaptive_control=control,
    )
    report["source_hashes"] = {
        "frames": file_sha256(args.frames),
        "sequence": file_sha256(args.sequence),
        "control": file_sha256(args.control),
        "policy": file_sha256(args.policy),
    }
    _atomic_json(args.output_dir / "report.json", report)
    _atomic_json(args.output_dir / "selections.json", selections)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
