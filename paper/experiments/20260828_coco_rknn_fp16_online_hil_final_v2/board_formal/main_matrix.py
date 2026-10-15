#!/usr/bin/env python3
"""Validate the frozen main matrix and hard-gate completed board runs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


MATRIX_SCHEMA = "cavs_online_paper_run_matrix_v2"
PROTOCOL_ID = "coco_rknn_fp16_causal_online_hil_final_v2"
RANDOM_SEEDS = (20260720, 20260721, 20260722, 20260723, 20260724)
METHOD_COUNTS = {
    "always_original": 1,
    "cavs_oq": 3,
    "cavs_oqr": 3,
    "confidence": 3,
    "edge_only": 1,
    "moric": 3,
    "oric": 3,
    "random": 15,
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: str | Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {field}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return dict(value)


def _asset(bundle: Path, relative: object, digest: object, *, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{field} path is invalid")
    root = bundle.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes board bundle") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(digest).lower():
        raise ValueError(f"{field} SHA-256 mismatch: {relative}")
    return path


def validate_main_matrix(
    matrix_path: str | Path,
    bundle_root: str | Path,
    protocol_path: str | Path,
) -> list[dict[str, Any]]:
    matrix = _object(matrix_path, field="run matrix")
    protocol = _object(protocol_path, field="protocol")
    bundle = Path(bundle_root)
    if matrix.get("schema") != MATRIX_SCHEMA:
        raise ValueError("unsupported run-matrix schema")
    if matrix.get("protocol_id") != PROTOCOL_ID or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("formal main matrix requires the final-v2 protocol")
    if str(matrix.get("protocol_sha256", "")) != file_sha256(protocol_path):
        raise ValueError("protocol SHA-256 differs from the run matrix")
    if int(matrix.get("expected_images", -1)) != 5000:
        raise ValueError("formal main matrix requires 5,000 images")
    if matrix.get("final_used_for_matrix_selection") is not False:
        raise ValueError("final data may not select the run matrix")
    if list(map(int, matrix.get("random_seeds", ()))) != list(RANDOM_SEEDS):
        raise ValueError("pre-registered Random seeds changed")
    if int(matrix.get("family_counts", {}).get("main", -1)) != 32:
        raise ValueError("run matrix does not declare 32 main runs")
    raw_runs = matrix.get("runs")
    if not isinstance(raw_runs, list):
        raise ValueError("run matrix has no run list")
    rows = [dict(row) for row in raw_runs if isinstance(row, Mapping) and row.get("family") == "main"]
    rows.sort(key=lambda row: int(row.get("run_index", -1)))
    if len(rows) != 32 or [int(row.get("run_index", -1)) for row in rows] != list(range(1, 33)):
        raise ValueError("main run indices must be exactly 1..32")
    keys = [str(row.get("run_key", "")) for row in rows]
    if not all(keys) or len(keys) != len(set(keys)):
        raise ValueError("main run keys must be non-empty and unique")
    if dict(sorted(Counter(str(row.get("method")) for row in rows).items())) != METHOD_COUNTS:
        raise ValueError("main method counts differ from the paper specification")
    random_rows = [row for row in rows if row.get("method") == "random"]
    random_pairs = Counter(
        (int(row.get("random_seed", -1)), int(row.get("budget_percent", -1)))
        for row in random_rows
    )
    expected_pairs = Counter((seed, budget) for seed in RANDOM_SEEDS for budget in (10, 25, 40))
    if random_pairs != expected_pairs:
        raise ValueError("Random seed/budget matrix changed")

    for row in rows:
        if (
            int(row.get("expected_images", -1)) != 5000
            or row.get("zero_lookahead") is not True
            or row.get("global_plan") is not False
            or int(row.get("buffer_frames", -1)) != 0
            or row.get("network_profile") != "native"
            or row.get("controller_variant") != "registered"
        ):
            raise ValueError(f"non-causal or non-native main row: {row.get('run_key')}")
        numerator = int(row.get("budget_numerator", 0))
        denominator = int(row.get("budget_denominator", 0))
        if denominator <= 0 or numerator <= 0 or numerator > denominator:
            raise ValueError(f"invalid budget: {row.get('run_key')}")
        _asset(bundle, row.get("policy"), row.get("policy_sha256"), field="policy")
        _asset(bundle, row.get("control"), row.get("control_sha256"), field="control")
    return rows


def _finite(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite")
    return number


def _valid_boxes(row: Mapping[str, Any], *, position: int) -> None:
    shape = row.get("shape")
    boxes = row.get("resolved_boxes")
    if not isinstance(shape, list) or len(shape) != 2 or not isinstance(boxes, list):
        raise ValueError(f"frame {position} has no shape/resolved boxes")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"frame {position} has an invalid shape")
    for box in boxes:
        if not isinstance(box, Mapping) or not isinstance(box.get("xyxy"), list) or len(box["xyxy"]) != 4:
            raise ValueError(f"frame {position} has an invalid box")
        x1, y1, x2, y2 = (_finite(value, field="box coordinate") for value in box["xyxy"])
        confidence = _finite(box.get("conf"), field="box confidence")
        class_id = int(box.get("cls", -1))
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"frame {position} has a degenerate or out-of-bounds box")
        if not (0.0 <= confidence <= 1.0) or not (0 <= class_id < 80):
            raise ValueError(f"frame {position} has an invalid score or class")


def gate_run(
    frames_path: str | Path,
    contract_path: str | Path,
    matrix_row: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _object(contract_path, field="run contract")
    expected = int(matrix_row["expected_images"])
    expected_contract = {
        "protocol_id": PROTOCOL_ID,
        "method": str(matrix_row["method"]),
        "budget_numerator": int(matrix_row["budget_numerator"]),
        "budget_denominator": int(matrix_row["budget_denominator"]),
        "image_count": expected,
        "random_seed": matrix_row.get("random_seed"),
        "zero_lookahead": True,
        "global_plan": False,
        "buffer_frames": 0,
        "policy_sha256": str(matrix_row["policy_sha256"]),
        "control_sha256": str(matrix_row["control_sha256"]),
    }
    for field, value in expected_contract.items():
        if contract.get(field) != value:
            raise ValueError(f"run contract differs from matrix: {field}")

    frames = Path(frames_path)
    count = 0
    payload_total = 0
    body_total = 0
    original_total = 0
    spent_sum = 0
    maximum_prefix_violation = 0
    action_counts: Counter[str] = Counter()
    last_spent = 0
    last_original = 0
    with frames.open("r", encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid frame JSON at position {position}") from exc
            if not isinstance(row, Mapping) or int(row.get("position", -1)) != position:
                raise ValueError("frame positions are not the canonical prefix")
            if row.get("status") != "committed":
                raise ValueError(f"frame {position} is not committed")
            decision = row.get("decision")
            controller = row.get("controller_state")
            cloud = row.get("cloud")
            if not all(isinstance(value, Mapping) for value in (decision, controller, cloud)):
                raise ValueError(f"frame {position} omits decision/controller/cloud data")
            action = str(decision.get("action", ""))
            payload = int(decision.get("payload_bytes", -1))
            original = int(decision.get("original_bytes", -1))
            spent = int(decision.get("spent_bytes", -1))
            cumulative = int(decision.get("cumulative_original_bytes", -1))
            if payload < 0 or original <= 0 or spent < last_spent or cumulative <= last_original:
                raise ValueError(f"frame {position} has non-monotonic byte accounting")
            expected_execution = "not_required" if action == "edge" else "ok"
            if row.get("execution_status") != expected_execution:
                raise ValueError(f"frame {position} has a failed execution")
            if (
                int(controller.get("budget_numerator", 0)) != expected_contract["budget_numerator"]
                or int(controller.get("budget_denominator", 0)) != expected_contract["budget_denominator"]
            ):
                raise ValueError(f"frame {position} changed the frozen budget")
            _valid_boxes(row, position=position)
            payload_total += payload
            body_total += int(cloud.get("request_body_bytes") or 0)
            original_total += original
            spent_sum += payload
            last_spent, last_original = spent, cumulative
            earned_floor = cumulative * expected_contract["budget_numerator"] // expected_contract["budget_denominator"]
            maximum_prefix_violation = max(maximum_prefix_violation, spent - earned_floor)
            action_counts[action] += 1
            count += 1
    if count != expected:
        raise ValueError(f"expected {expected} frames, found {count}")
    if last_original != original_total or last_spent != spent_sum or payload_total != spent_sum:
        raise ValueError("aggregate byte accounting differs from the final controller state")
    if maximum_prefix_violation > 0:
        raise ValueError(f"prefix budget violation: {maximum_prefix_violation} bytes")
    ratio = body_total / payload_total if payload_total else 1.0
    if ratio > 1.02:
        raise ValueError(f"HTTP body/JPEG payload ratio exceeds 1.02: {ratio:.6f}")
    return {
        "frames": count,
        "cloud_failures": 0,
        "maximum_prefix_violation_bytes": maximum_prefix_violation,
        "selected_payload_bytes": payload_total,
        "wire_request_body_bytes": body_total,
        "wire_to_payload_ratio": ratio,
        "action_counts": dict(sorted(action_counts.items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix_parser = subparsers.add_parser("validate-matrix")
    matrix_parser.add_argument("--matrix", type=Path, required=True)
    matrix_parser.add_argument("--bundle", type=Path, required=True)
    matrix_parser.add_argument("--protocol", type=Path, required=True)
    gate_parser = subparsers.add_parser("gate-run")
    gate_parser.add_argument("--matrix", type=Path, required=True)
    gate_parser.add_argument("--bundle", type=Path, required=True)
    gate_parser.add_argument("--protocol", type=Path, required=True)
    gate_parser.add_argument("--run-index", type=int, required=True)
    gate_parser.add_argument("--frames", type=Path, required=True)
    gate_parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = validate_main_matrix(args.matrix, args.bundle, args.protocol)
    if args.command == "validate-matrix":
        print(json.dumps({"main_runs": len(rows), "passed": True}, sort_keys=True))
        return 0
    row = next((value for value in rows if int(value["run_index"]) == args.run_index), None)
    if row is None:
        raise ValueError(f"unknown main run index: {args.run_index}")
    print(json.dumps(gate_run(args.frames, args.contract, row), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
