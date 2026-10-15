#!/usr/bin/env python3
"""Run calibration-only selection for one frozen TriAction dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from selector_core import (
    choose_provisional,
    choose_reference,
    extract_frozen_policy_candidates,
    validate_protocol,
)


SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
SOURCE_NAME = "20260723_three_dataset_nineranker_triaction_paper_exploration"


def source_root() -> Path:
    return REPO / "paper/experiments" / SOURCE_NAME


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in values for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_input_paths(dataset: str) -> list[Path]:
    root = source_root()
    return [
        root / "configs/protocol.json",
        root / f"runs/{dataset}/CORE_COMPLETED",
        root / f"runs/{dataset}/calibration_curve.csv",
        root / f"runs/{dataset}/threshold_choices.json",
    ]


def _number(value: Any) -> float:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in {"+Infinity", "Infinity", "inf", "+inf"}:
            return math.inf
        if normalized in {"-Infinity", "-inf"}:
            return -math.inf
    return float(value)


def _curve_rows(path: Path) -> list[dict[str, Any]]:
    numeric = {
        "ap",
        "ap_delta",
        "budget",
        "kb",
        "kb_delta",
        "lambda_value",
        "median_original_bytes",
        "online_threshold",
        "threshold",
        "worst_ap_delta",
    }
    rows: list[dict[str, Any]] = []
    for raw in read_csv(path):
        row: dict[str, Any] = dict(raw)
        for key in numeric:
            if key in row and str(row[key]).strip() != "":
                row[key] = _number(row[key])
        rows.append(row)
    return rows


def preflight(dataset: str, protocol: dict[str, Any]) -> dict[str, Any]:
    validate_protocol(protocol)
    paths = calibration_input_paths(dataset)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen calibration inputs:\n" + "\n".join(missing))
    source_protocol = read_json(source_root() / "configs/protocol.json")
    if tuple(source_protocol["constrained_rankers"]) != tuple(
        protocol["selectable_rankers"]
    ):
        raise RuntimeError("Selectable Rankers differ from the frozen source")
    if tuple(source_protocol["policies"]) != tuple(protocol["policies"]):
        raise RuntimeError("Action policies differ from the frozen source")
    source_points = {
        "fixed_call": list(map(float, source_protocol["primary_call_budgets"])),
        "online": list(map(float, source_protocol["online_targets"])),
        "fixed_byte": list(map(float, source_protocol["fixed_byte_budgets"])),
    }
    expected_points = {
        mode: list(map(float, budgets))
        for mode, budgets in protocol["work_points"].items()
    }
    if source_points != expected_points:
        raise RuntimeError("Registered work points differ from the frozen source")
    return {
        "dataset": dataset,
        "source_experiment": SOURCE_NAME,
        "calibration_input_sha256": {
            str(path.resolve()): file_sha256(path) for path in paths
        },
        "final_artifacts_read": False,
    }


def build_provisional_choices(
    *,
    dataset: str,
    protocol: dict[str, Any],
    calibration_curve: Path,
    threshold_choices: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curve = _curve_rows(calibration_curve)
    thresholds = read_json(threshold_choices)["shared"]
    rankers = tuple(map(str, protocol["selectable_rankers"]))
    policies = tuple(map(str, protocol["policies"]))
    ranker_order = rankers
    policy_order = tuple(map(str, protocol["policy_tie_break_order"]))
    tolerance = float(protocol["ap_tolerance"])
    choices: list[dict[str, Any]] = []
    candidate_table: list[dict[str, Any]] = []
    for mode, budgets in protocol["work_points"].items():
        gbdt_threshold = _number(thresholds[mode]["threshold"])
        for budget in map(float, budgets):
            candidates = extract_frozen_policy_candidates(
                curve,
                mode=mode,
                budget=budget,
                rankers=rankers,
                gbdt_threshold=gbdt_threshold,
            )
            for row in candidates:
                row.update(
                    {
                        "dataset": dataset,
                        "selection_source_role": "calibration",
                    }
                )
            reference = choose_reference(candidates, ranker_order=ranker_order)
            provisional = choose_provisional(
                candidates,
                reference=reference,
                mode=mode,
                ap_tolerance=tolerance,
                ranker_order=ranker_order,
                policy_order=policy_order,
            )
            choices.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "budget": budget,
                    "selection_source_role": "calibration",
                    "reference_ranker": reference["ranker"],
                    "reference_policy": reference["policy"],
                    "reference_ap": float(reference["ap"]),
                    "reference_kb": float(reference["kb"]),
                    "reference_threshold": float(reference["threshold"]),
                    "reference_lambda": reference.get("lambda_value"),
                    "reference_median_original_bytes": reference.get(
                        "median_original_bytes"
                    ),
                    "reference_online_threshold": reference.get(
                        "online_threshold"
                    ),
                    "provisional_ranker": provisional["ranker"],
                    "provisional_policy": provisional["policy"],
                    "provisional_ap": float(provisional["ap"]),
                    "provisional_kb": float(provisional["kb"]),
                    "provisional_threshold": float(provisional["threshold"]),
                    "provisional_lambda": provisional.get("lambda_value"),
                    "provisional_median_original_bytes": provisional.get(
                        "median_original_bytes"
                    ),
                    "provisional_online_threshold": provisional.get(
                        "online_threshold"
                    ),
                    "point_ap_delta": float(provisional["ap"])
                    - float(reference["ap"]),
                    "point_kb_delta": float(provisional["kb"])
                    - float(reference["kb"]),
                }
            )
            candidate_table.extend(candidates)
    if len(choices) != int(protocol["expected_work_points_per_dataset"]):
        raise RuntimeError(f"Unexpected selection count: {len(choices)}")
    expected_candidates = len(choices) * len(rankers) * len(policies)
    if len(candidate_table) != expected_candidates:
        raise RuntimeError(
            f"Unexpected candidate count: {len(candidate_table)} != {expected_candidates}"
        )
    return choices, candidate_table


def run_calibration(dataset: str, protocol: dict[str, Any]) -> None:
    output = EXP_ROOT / "runs" / dataset
    output.mkdir(parents=True, exist_ok=True)
    checked = preflight(dataset, protocol)
    before = checked["calibration_input_sha256"]
    source = source_root() / "runs" / dataset
    choices, candidates = build_provisional_choices(
        dataset=dataset,
        protocol=protocol,
        calibration_curve=source / "calibration_curve.csv",
        threshold_choices=source / "threshold_choices.json",
    )
    write_csv(output / "candidate_table.csv", candidates)
    write_json(output / "provisional_choices.json", choices)
    after = {
        str(path.resolve()): file_sha256(path)
        for path in calibration_input_paths(dataset)
    }
    if before != after:
        raise RuntimeError("A frozen calibration input changed during selection")
    provenance = {
        "protocol": protocol,
        "preflight": checked,
        "calibration_input_sha256_before": before,
        "calibration_input_sha256_after": after,
        "final_artifacts_read": False,
        "provisional_choices_sha256": file_sha256(
            output / "provisional_choices.json"
        ),
    }
    write_json(output / "calibration_provenance.json", provenance)
    (output / "PROVISIONAL_FROZEN").write_text("ok\n", encoding="ascii")
    print(json.dumps(choices, indent=2, ensure_ascii=False, allow_nan=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("coco", "ena24", "oregon"), required=True)
    parser.add_argument("--stage", choices=("preflight", "calibrate"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = read_json(EXP_ROOT / "configs/protocol.json")
    if args.stage == "preflight":
        print(json.dumps(preflight(args.dataset, protocol), indent=2, ensure_ascii=False))
    else:
        run_calibration(args.dataset, protocol)


if __name__ == "__main__":
    main()
