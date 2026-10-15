#!/usr/bin/env python3
"""Orchestrate isolated external-baseline stages across datasets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from runner_core import assert_output_path, atomic_write_json


SCRIPT = Path(__file__).resolve()
EXPERIMENT = SCRIPT.parents[1]
RUNS = EXPERIMENT / "runs"
DATASETS = ("coco", "ena24", "oregon")
STAGES = ("preflight", "rewards", "evaluate", "all")


def dataset_command(dataset: str, stage: str) -> list[str]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    return [
        sys.executable,
        str(EXPERIMENT / "scripts/run_dataset.py"),
        "--dataset",
        dataset,
        "--stage",
        stage,
    ]


def completion_path(dataset: str, stage: str) -> Path:
    if stage == "all":
        return RUNS / dataset / "completion.json"
    if stage == "evaluate":
        return RUNS / dataset / "evaluation/completion.json"
    return RUNS / dataset / stage / "completion.json"


def _run_one(dataset: str, stage: str) -> dict[str, Any]:
    log_path = assert_output_path(RUNS / "logs" / f"{dataset}_{stage}.log", RUNS)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            dataset_command(dataset, stage),
            cwd=EXPERIMENT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"{dataset}/{stage} failed with code {result.returncode}; see {log_path}"
        )
    completion = completion_path(dataset, stage)
    if not completion.is_file():
        raise RuntimeError(f"{dataset}/{stage} returned without completion: {completion}")
    return {
        "dataset": dataset,
        "stage": stage,
        "log": str(log_path),
        "completion": str(completion),
    }


def run_pipeline(
    datasets: list[str], stage: str, *, max_parallel_datasets: int
) -> dict[str, Any]:
    selected = list(dict.fromkeys(datasets))
    if not selected or any(dataset not in DATASETS for dataset in selected):
        raise ValueError(f"Invalid datasets: {datasets}")
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    workers = min(max(1, int(max_parallel_datasets)), len(selected))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        reports = list(executor.map(lambda dataset: _run_one(dataset, stage), selected))
    summary = {
        "stage": stage,
        "datasets": selected,
        "max_parallel_datasets": workers,
        "reports": reports,
        "complete": True,
    }
    atomic_write_json(RUNS / f"pipeline_{stage}_completion.json", summary, RUNS)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--max-parallel-datasets", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            run_pipeline(
                args.datasets,
                args.stage,
                max_parallel_datasets=args.max_parallel_datasets,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
