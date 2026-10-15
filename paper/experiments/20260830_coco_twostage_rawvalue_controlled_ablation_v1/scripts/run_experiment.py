#!/usr/bin/env python3
"""Run the clean raw-value TwoStage point ablation on frozen CAVS-OQ scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs/point"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("twostage_raw_core", ROOT / "scripts/core.py")
SOURCE_ROOT = REPO / PROTOCOL["source_experiment"]
BASE = load_module("twostage_raw_source", SOURCE_ROOT / "scripts/run_experiment.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty results")
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def source_rows() -> dict[tuple[str, float], dict[str, str]]:
    path = SOURCE_ROOT / "runs/coco/point/results.csv"
    rows = BASE.read_csv(path)
    output: dict[tuple[str, float], dict[str, str]] = {}
    for row in rows:
        if (
            row["dataset"] == PROTOCOL["dataset"]
            and row["role"] == PROTOCOL["role"]
            and row["method"] == PROTOCOL["source_method"]
            and row["policy"] == PROTOCOL["source_policy"]
            and row["mode"] == PROTOCOL["mode"]
        ):
            output[(row["seed"], float(row["budget"]))] = row
    expected = (len(PROTOCOL["seeds"]) + 1) * len(PROTOCOL["budgets"])
    if len(output) != expected:
        raise RuntimeError(f"source CAVS-OQ point matrix incomplete: {len(output)}/{expected}")
    return output


def run() -> dict[str, Any]:
    records = BASE.role_records(PROTOCOL["dataset"], PROTOCOL["role"], "OQ")
    samples = BASE.adapter(PROTOCOL["dataset"]).load_role(PROTOCOL["role"])["samples"]
    single, ensemble = BASE.ensemble_action_scores(
        PROTOCOL["dataset"], PROTOCOL["source_method"], PROTOCOL["role"]
    )
    values_by_seed: dict[str, np.ndarray] = {**single, "ensemble": ensemble}
    references = source_rows()
    results: list[dict[str, Any]] = []
    selections: dict[str, dict[str, str]] = {}

    for seed_token, matrices in values_by_seed.items():
        for budget_index, raw_budget in enumerate(PROTOCOL["budgets"]):
            budget = float(raw_budget)
            score_map = BASE.action_score_map(records, matrices[budget_index])
            projected = CORE.raw_projected_score_map(
                score_map, action_order=tuple(PROTOCOL["action_order"])
            )
            selection = BASE._joint_selection(
                PROTOCOL["dataset"],
                samples,
                projected,
                mode=PROTOCOL["mode"],
                budget=budget,
            )
            row = BASE.evaluate_selection(
                PROTOCOL["dataset"],
                PROTOCOL["role"],
                selection,
                method=PROTOCOL["method"],
                policy=PROTOCOL["policy"],
                mode=PROTOCOL["mode"],
                budget=budget,
                seed=seed_token,
                fusion=PROTOCOL["fusion"],
            )
            source = references[(str(seed_token), budget)]
            source_ap = float(source["official_ap50_95"])
            row["source_joint_ap50_95"] = source_ap
            row["joint_minus_twostage_ap50_95"] = source_ap - float(row["official_ap50_95"])
            row["score_transform"] = "none"
            row["source_result_sha256"] = sha256(
                SOURCE_ROOT / "runs/coco/point/results.csv"
            )
            results.append(row)
            key = f"budget={budget:g}|seed={seed_token}"
            selections[key] = {
                str(image_id): action for image_id, action in sorted(selection.items())
            }

    expected = (len(PROTOCOL["seeds"]) + 1) * len(PROTOCOL["budgets"])
    if len(results) != expected:
        raise RuntimeError(f"point matrix incomplete: {len(results)}/{expected}")
    results.sort(key=lambda row: (float(row["budget"]), str(row["seed"])))
    atomic_csv(RUNS / "results.csv", results)
    atomic_json(RUNS / "selections.json", selections)
    report = {
        "complete": True,
        "rows": len(results),
        "score_transform": "none",
        "results_sha256": sha256(RUNS / "results.csv"),
        "selections_sha256": sha256(RUNS / "selections.json"),
        "source_results_sha256": sha256(SOURCE_ROOT / "runs/coco/point/results.csv"),
        "source_scores_reused": True,
        "learner_retrained": False,
    }
    atomic_json(RUNS / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("point",), default="point")
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
