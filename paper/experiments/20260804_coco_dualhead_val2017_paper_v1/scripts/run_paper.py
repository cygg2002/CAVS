#!/usr/bin/env python3
"""Evaluate the frozen train2017 learner on all 5,000 COCO val2017 images."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
RUNS = ROOT / "runs"
PROTOCOL_PATH = ROOT / "configs/protocol.json"
SOURCE_RUNNER = (
    REPO
    / "paper/experiments/20260804_coco_dualhead_fixedbyte_registered_v1/scripts/run_paper.py"
)


def _load_source_runner():
    spec = importlib.util.spec_from_file_location("_dualhead_registered_source", SOURCE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source_runner()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def _validate_hash(path: Path, expected: str, label: str) -> None:
    observed = SOURCE.file_sha256(path)
    if observed != str(expected):
        raise RuntimeError(f"{label} hash mismatch: {observed} != {expected}")


def _canonical_paths(protocol: Mapping[str, Any]) -> dict[str, Path]:
    final = Path(protocol["paths"]["canonical_common"]) / "final"
    manifest = read_json(final / "source_manifest.json")
    return {
        "samples": final / "samples.json",
        "ground_truth": final / "gt.json",
        "edge_predictions": Path(manifest["action_sources"]["edge"]),
        "cloud_predictions": Path(manifest["action_sources"]["original"]),
    }


def _load_val_records(
    protocol: Mapping[str, Any], *, expected_images=5000
) -> tuple[list[dict], dict]:
    paths = _canonical_paths(protocol)
    for label, path in paths.items():
        _validate_hash(path, protocol["hashes"][label], label)
    samples = read_json(paths["samples"])
    if len(samples) != int(expected_images):
        raise RuntimeError(f"Expected {expected_images} val2017 images, got {len(samples)}")
    image_ids = [int(sample["image_id"]) for sample in samples]
    if len(set(image_ids)) != int(expected_images):
        raise RuntimeError("val2017 image IDs are not unique")
    sources = []
    for sample in samples:
        identity = str(sample["identity"])
        normalized = identity.replace("\\", "/")
        if "/train2017/" in normalized or "/val2017/" not in normalized:
            raise RuntimeError(f"Non-val2017 evaluation path: {identity}")
        sources.append(
            {
                "image_id": int(sample["image_id"]),
                "file_name": Path(identity).name,
                "path": identity,
                "width": int(sample["width"]),
                "height": int(sample["height"]),
                "payload_bytes": int(sample["payload_bytes"]["original"]),
            }
        )
    records = SOURCE._records_from_sources(
        sources,
        read_json(paths["edge_predictions"]),
        read_json(paths["cloud_predictions"]),
        protocol,
    )
    if [int(record["image_id"]) for record in records] != image_ids:
        raise RuntimeError("Canonical val2017 ordering changed")
    return records, read_json(paths["ground_truth"])


def _load_checkpoint(protocol: Mapping[str, Any]) -> dict:
    root = Path(protocol["paths"]["training_experiment"]) / "runs/train"
    completion = read_json(root / "completion.json")
    checkpoint = root / "checkpoints.pkl"
    checkpoint_sha256 = SOURCE.file_sha256(checkpoint)
    if completion.get("complete") is not True:
        raise RuntimeError("Frozen train2017 learner checkpoint is incomplete")
    if checkpoint_sha256 != protocol["hashes"]["checkpoint"]:
        raise RuntimeError("Frozen train2017 learner checkpoint changed")
    if completion.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("Training completion does not bind the checkpoint")
    with checkpoint.open("rb") as handle:
        return pickle.load(handle)


def run_prepare() -> dict:
    protocol = read_json(PROTOCOL_PATH)
    records, ground_truth = _load_val_records(protocol, expected_images=5000)
    artifact = _load_checkpoint(protocol)
    image_ids = [int(record["image_id"]) for record in records]
    gt_ids = {int(row["id"]) for row in ground_truth["images"]}
    if set(image_ids) != gt_ids:
        raise RuntimeError("val2017 samples and ground truth differ")
    report = {
        "protocol_id": protocol["protocol_id"],
        "train_role": protocol["train_role"],
        "test_role": "coco_val2017_full",
        "test_images": len(records),
        "all_paths_val2017": True,
        "checkpoint_sha256": protocol["hashes"]["checkpoint"],
        "development_ids_sha256": artifact["development_ids_sha256"],
        "val_used_for_training": False,
        "val_used_for_selection": False,
    }
    atomic_json(RUNS / "prepare/report.json", report)
    atomic_json(
        RUNS / "prepare/completion.json",
        {
            "complete": True,
            "report_sha256": SOURCE.file_sha256(RUNS / "prepare/report.json"),
            **report,
        },
    )
    return report


def _selected_predictions(
    records: Sequence[Mapping[str, Any]], selected_ids: set[int]
) -> list[dict]:
    return [
        copy.deepcopy(prediction)
        for record in records
        for prediction in (
            record["cloud_predictions"]
            if int(record["image_id"]) in selected_ids
            else record["edge_predictions"]
        )
    ]


def _point_evaluation(
    protocol: Mapping[str, Any], records: list[dict], artifact: Mapping[str, Any]
) -> tuple[list[dict], dict[str, list[int]]]:
    from ca_aoric import CocoMetricBackend

    scores = SOURCE._method_scores(records, protocol, artifact)
    image_ids = [int(record["image_id"]) for record in records]
    metric = CocoMetricBackend(Path(protocol["paths"]["annotations"]))
    full_capacity = sum(int(record["payload_bytes"]) for record in records)
    rows, selections = [], {}
    for budget_index, budget in enumerate(map(float, protocol["budgets"])):
        for method, budget_scores in scores.items():
            values = budget_scores[budget_index]
            fixed_ids = SOURCE.select_fixed_call(image_ids, values, budget=budget)
            byte_result = SOURCE.select_costeligible(
                records,
                SOURCE._score_matrix(values),
                resource_profile="upload",
                capacity=budget * full_capacity,
                full_only=True,
            )
            if float(byte_result["total_cost"]) > budget * full_capacity + 1e-6:
                raise RuntimeError(f"Fixed-byte capacity exceeded for {method}/{budget:g}")
            byte_ids = {
                int(item["image_id"])
                for item in byte_result["selected"]
                if str(item["action_type"]) == "full"
            }
            for mode, selected_ids, extra in (
                ("fixed_call", fixed_ids, {}),
                (
                    "fixed_byte",
                    byte_ids,
                    {
                        "capacity_bytes": budget * full_capacity,
                        "used_capacity_bytes": float(byte_result["total_cost"]),
                        "lambda": float(byte_result["lambda"]),
                    },
                ),
            ):
                selections[f"{method}|{mode}|{budget:g}"] = sorted(selected_ids)
                rows.append(
                    {
                        "method": method,
                        "mode": mode,
                        "budget": budget,
                        **metric.evaluate(
                            _selected_predictions(records, selected_ids), image_ids
                        ),
                        **SOURCE._selection_metrics(records, selected_ids, protocol),
                        **extra,
                    }
                )
    expected = len(protocol["methods"]) * len(protocol["modes"]) * len(protocol["budgets"])
    if len(rows) != expected:
        raise RuntimeError(f"Incomplete result matrix: {len(rows)} != {expected}")
    return rows, selections


def _result_row(
    rows: Sequence[Mapping[str, Any]], method: str, mode: str, budget: float
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["method"] == method
        and row["mode"] == mode
        and abs(float(row["budget"]) - float(budget)) < 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Missing result: {method}/{mode}/{budget:g}")
    return matches[0]


def run_evaluate() -> dict:
    protocol = read_json(PROTOCOL_PATH)
    completion_path = RUNS / "evaluation/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "evaluation/report.json")
    prepared = read_json(RUNS / "prepare/completion.json")
    if prepared.get("complete") is not True or prepared.get("test_images") != 5000:
        raise RuntimeError("Full val2017 prepare must complete first")
    records, ground_truth = _load_val_records(protocol, expected_images=5000)
    artifact = _load_checkpoint(protocol)
    output = RUNS / "evaluation"
    point_path = output / "point_report.json"
    selections_path = output / "selections.json"
    if point_path.is_file() and selections_path.is_file():
        rows = read_json(point_path)["rows"]
        selections = read_json(selections_path)
    else:
        rows, selections = _point_evaluation(protocol, records, artifact)
        atomic_csv(output / "results.csv", rows)
        atomic_json(selections_path, selections)
        atomic_json(
            point_path,
            {
                "protocol_id": protocol["protocol_id"],
                "test_role": "coco_val2017_full",
                "images": 5000,
                "rows": rows,
                "val_used_for_selection": False,
            },
        )

    image_ids = [int(record["image_id"]) for record in records]
    bootstrap_config = protocol["bootstrap"]
    intervals, comparisons, capacities = [], [], []
    for budget in map(float, protocol["budgets"]):
        comparator = protocol["frozen_comparators"][f"fixed_byte:{budget:g}"]
        candidate_key = f"DualHead|fixed_byte|{budget:g}"
        comparator_key = f"{comparator}|fixed_byte|{budget:g}"
        candidate_ids = set(map(int, selections[candidate_key]))
        comparator_ids = set(map(int, selections[comparator_key]))
        bootstrap_path = output / f"bootstrap_{budget:g}.json"
        if bootstrap_path.is_file():
            bootstrap = read_json(bootstrap_path)
            values = np.asarray(bootstrap["values"], dtype=np.float64)
            if len(values) != int(bootstrap_config["replicates"]):
                raise RuntimeError("Bootstrap replicate count changed")
        else:
            values, indices_sha256 = SOURCE._paired_bootstrap(
                ground_truth=ground_truth,
                image_ids=image_ids,
                candidate_predictions=_selected_predictions(records, candidate_ids),
                comparator_predictions=_selected_predictions(records, comparator_ids),
                replicates=int(bootstrap_config["replicates"]),
                seed=int(bootstrap_config["seed"]),
            )
            bootstrap = {
                "budget": budget,
                "candidate": "DualHead",
                "comparator": comparator,
                "replicates": len(values),
                "seed": int(bootstrap_config["seed"]),
                "indices_sha256": indices_sha256,
                "values": values.tolist(),
            }
            atomic_json(bootstrap_path, bootstrap)
        summary = SOURCE.summarize_bootstrap(
            values, confidence=float(bootstrap_config["confidence"])
        )
        intervals.append(
            {"budget": budget, "candidate": "DualHead", "comparator": comparator, **summary}
        )
        candidate = _result_row(rows, "DualHead", "fixed_byte", budget)
        baseline = _result_row(rows, comparator, "fixed_byte", budget)
        comparisons.append(
            {
                "budget": budget,
                "comparator": comparator,
                "candidate_ap": float(candidate["map50_95"]),
                "comparator_ap": float(baseline["map50_95"]),
                "ap_gain": float(candidate["map50_95"]) - float(baseline["map50_95"]),
            }
        )
        for method, row in (("DualHead", candidate), (comparator, baseline)):
            capacities.append(
                {
                    "budget": budget,
                    "method": method,
                    "used": float(row["used_capacity_bytes"]),
                    "capacity": float(row["capacity_bytes"]),
                }
            )
    gate = SOURCE.fixed_byte_final_gate(
        intervals,
        capacities,
        required_superior_points=2,
        noninferiority_margin=0.001,
    )
    report = {
        "protocol_id": protocol["protocol_id"],
        "train_role": protocol["train_role"],
        "test_role": "coco_val2017_full",
        "test_images": 5000,
        "primary_mode": "fixed_byte",
        "frozen_comparators": protocol["frozen_comparators"],
        "comparisons": comparisons,
        "bootstrap": intervals,
        "capacities": capacities,
        "gate": gate,
        "val_used_for_training": False,
        "val_used_for_selection": False,
    }
    atomic_json(output / "report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "passed": bool(gate["passed"]),
            "results_sha256": SOURCE.file_sha256(output / "results.csv"),
            "selections_sha256": SOURCE.file_sha256(selections_path),
            "report_sha256": SOURCE.file_sha256(output / "report.json"),
            "test_images": 5000,
            "val_used_for_selection": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "evaluate", "all"), required=True)
    args = parser.parse_args()
    if args.stage in {"prepare", "all"}:
        print(json.dumps(run_prepare(), indent=2, ensure_ascii=False))
    if args.stage in {"evaluate", "all"}:
        print(json.dumps(run_evaluate(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
