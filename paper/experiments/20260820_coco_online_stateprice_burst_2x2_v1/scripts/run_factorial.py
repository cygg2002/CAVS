#!/usr/bin/env python3
"""Freeze calibration controls and run one final zero-buffer factorial replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from factorial_control import build_empirical_demand_curve
from factorial_experiment import calibrate_budget, replay_factorial


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(specification: Mapping[str, Any]) -> Path:
    try:
        path = (ROOT / str(specification["path"])).resolve()
        expected = str(specification["sha256"]).lower()
    except (KeyError, TypeError) as error:
        raise ValueError("source specification is invalid") from error
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"source is missing or hash-mismatched: {path}")
    return path


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_final_records(frames_path: Path, sequence_path: Path) -> list[dict[str, Any]]:
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    expected = sequence.get("records") if isinstance(sequence, Mapping) else None
    if not isinstance(expected, list):
        raise ValueError("sequence lacks records")
    records = []
    with frames_path.open("r", encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if position >= len(expected):
                raise ValueError("final frame log is longer than sequence")
            row = json.loads(line)
            item = expected[position]
            if (
                int(row.get("position", -1)) != position
                or int(row.get("image_id", -1)) != int(item["image_id"])
                or str(row.get("file_name", "")) != str(item["file_name"])
            ):
                raise ValueError("final frame log and sequence order differ")
            payloads, scores = row.get("payload_bytes"), row.get("scores")
            if not isinstance(payloads, Mapping) or not isinstance(scores, Mapping):
                raise ValueError("final row lacks current-frame candidates")
            records.append({
                "image_id": int(row["image_id"]),
                "payload_bytes": dict(payloads),
                "scores": dict(scores),
                "objective_values": dict(scores),
            })
    if len(records) != len(expected):
        raise ValueError("final frame log is incomplete")
    return records


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "cavs_online_stateprice_burst_experiment_v1":
        raise ValueError("unsupported experiment configuration")
    sources = config.get("sources")
    budgets = config.get("budgets")
    if not isinstance(sources, Mapping) or not isinstance(budgets, Mapping):
        raise ValueError("configuration lacks sources or budgets")
    calibration_path = _source(sources["calibration_records"])
    sequence_path = _source(sources["sequence"])
    policy_path = _source(sources["policy"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if (
        calibration.get("source_split") != "calibration"
        or calibration.get("final_used_for_tuning") is not False
        or calibration.get("ground_truth_read") is not False
    ):
        raise ValueError("calibration records are not final-blind")
    actions = tuple(map(str, calibration["action_ids"]))
    if actions != tuple(map(str, policy["action_ids"])):
        raise ValueError("calibration and policy action contracts differ")
    records = calibration.get("records")
    if not isinstance(records, list) or len(records) != 571:
        raise ValueError("expected exactly 571 calibration records")
    medians = {
        float(json.loads(_source(spec["base_control"]).read_text(encoding="utf-8"))["median_original_bytes"])
        for spec in budgets.values()
    }
    if len(medians) != 1:
        raise ValueError("budget controls use different median-byte normalization")
    median = next(iter(medians))
    curve = build_empirical_demand_curve(
        records,
        action_ids=actions,
        median_original_bytes=median,
    )
    curve_output = ROOT / "configs" / "demand_curve.json"
    _atomic_json(curve_output, {
        "schema": "cavs_empirical_demand_curve_v1",
        "source_split": "calibration",
        "final_used_for_tuning": False,
        "median_original_bytes": median,
        "entries": [entry.__dict__ for entry in curve.entries],
        "source_hashes": {"calibration_records": file_sha256(calibration_path)},
    })
    curve_sha256 = file_sha256(curve_output)
    burst_grid = config.get("burst_fraction_grid")
    if not isinstance(burst_grid, list):
        raise ValueError("configuration lacks burst grid")
    frozen: dict[str, dict[str, Any]] = {}
    for label in ("10", "25", "40"):
        specification = budgets.get(label)
        if not isinstance(specification, Mapping):
            raise ValueError(f"missing budget configuration: {label}")
        base_path = _source(specification["base_control"])
        base = json.loads(base_path.read_text(encoding="utf-8"))
        control, calibration_report = calibrate_budget(
            records,
            action_ids=actions,
            base_control=base,
            demand_curve=curve,
            burst_fraction_grid=burst_grid,
        )
        source_hashes = {
            "calibration_records": file_sha256(calibration_path),
            "base_control": file_sha256(base_path),
            "demand_curve": curve_sha256,
            "policy": file_sha256(policy_path),
        }
        control["source_hashes"] = source_hashes
        calibration_report["source_hashes"] = source_hashes
        control_path = ROOT / "configs" / "controls" / f"{label}.json"
        calibration_output = ROOT / "runs" / "calibration" / label / "report.json"
        _atomic_json(control_path, control)
        _atomic_json(calibration_output, calibration_report)
        frozen[label] = {
            "control": control,
            "control_path": control_path,
            "expected_specification": specification["expected_selections"],
        }

    # The final record stream and frozen-final reference traces are not opened
    # until every budget control has been calibrated and persisted.
    final_frames_path = _source(sources["final_frames"])
    final_records = _load_final_records(final_frames_path, sequence_path)
    summary: dict[str, Any] = {
        "schema": "cavs_online_stateprice_burst_factorial_summary_v1",
        "complete": True,
        "final_used_for_tuning": False,
        "zero_lookahead": True,
        "global_plan": False,
        "buffer_frames": 0,
        "budgets": {},
    }
    for label in ("10", "25", "40"):
        item = frozen[label]
        control = item["control"]
        control_path = item["control_path"]
        expected_path = _source(item["expected_specification"])
        expected_bundle = json.loads(expected_path.read_text(encoding="utf-8"))
        expected = expected_bundle.get("fixed")
        if not isinstance(expected, Mapping):
            raise ValueError("reference selection bundle lacks fixed trace")
        final_report, selections = replay_factorial(
            final_records,
            action_ids=actions,
            frozen_control=control,
            demand_curve=curve,
            expected_fixed_prefix=expected,
        )
        final_report["source_hashes"] = {
            "final_frames": file_sha256(final_frames_path),
            "sequence": file_sha256(sequence_path),
            "policy": file_sha256(policy_path),
            "control": file_sha256(control_path),
            "expected_selections": file_sha256(expected_path),
            "demand_curve": curve_sha256,
        }
        final_dir = ROOT / "runs" / "final" / label
        _atomic_json(final_dir / "report.json", final_report)
        _atomic_json(final_dir / "selections.json", selections)
        summary["budgets"][label] = final_report
    _atomic_json(ROOT / "runs" / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run(args.config)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
