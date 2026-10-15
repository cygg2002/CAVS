#!/usr/bin/env python3
"""Paired bootstrap for AFPD versus independently retrained ActionValue."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
PARENT_ROOT = REPO / PROTOCOL["parent_experiment"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("afpd_same_run_core", ROOT / "scripts/core.py")
PARENT = load_module("afpd_same_run_parent", PARENT_ROOT / "scripts/run_experiment.py")
CORE.validate_protocol(PROTOCOL)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parent_marker() -> Path:
    return PARENT_ROOT / PROTOCOL["parent_completion"]


def wait_for_parent() -> None:
    marker = parent_marker()
    poll_seconds = int(PROTOCOL["wait_poll_seconds"])
    polls = 0
    while not CORE.parent_ready(marker):
        if polls % 10 == 0:
            print(f"[waiting] parent bootstrap incomplete: {marker}", flush=True)
        polls += 1
        time.sleep(poll_seconds)
    print(f"[ready] parent bootstrap complete: {marker}", flush=True)


def readiness() -> dict[str, Any]:
    selections_path = PARENT_ROOT / "runs/final/evaluation/selections.json"
    final_completion = PARENT_ROOT / "runs/final/completion.json"
    required = (
        PARENT_ROOT / "configs/protocol.json",
        PARENT_ROOT / "scripts/run_experiment.py",
        selections_path,
        final_completion,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    selections = read_json(selections_path)
    expected_keys = {
        spec[key]
        for spec in CORE.comparison_specs(PROTOCOL)
        for key in ("candidate_key", "comparator_key")
    }
    absent = sorted(expected_keys - set(selections))
    if absent:
        raise RuntimeError(f"Missing same-run selections: {absent}")
    data = PARENT.SOURCE.load_role("final")
    image_ids = [int(sample["image_id"]) for sample in data["samples"]]
    if len(image_ids) != 5000 or len(set(image_ids)) != 5000:
        raise RuntimeError("Canonical final image population changed")
    for key in expected_keys:
        selection = PARENT.SOURCE._selection_from_json(selections, key)
        CORE.validate_selection(selection, image_ids)
    return {
        "protocol_id": PROTOCOL["protocol_id"],
        "parent_bootstrap_complete": CORE.parent_ready(parent_marker()),
        "final_images": len(image_ids),
        "selection_keys": sorted(expected_keys),
        "confirmatory": False,
        "final_is_hidden": False,
        "final_used_for_selection": False,
    }


def run_preflight(*, allow_concurrent_parent: bool = False) -> dict[str, Any]:
    report_path = RUNS / "preflight/report.json"
    completion = RUNS / "preflight/completion.json"
    if completion.is_file():
        return read_json(report_path)
    report = readiness()
    if not report["parent_bootstrap_complete"] and not allow_concurrent_parent:
        raise RuntimeError("Parent bootstrap must complete before sidecar compute")
    report["concurrent_with_parent_bootstrap"] = bool(
        allow_concurrent_parent and not report["parent_bootstrap_complete"]
    )
    source_files = [
        ROOT / "configs/protocol.json",
        ROOT / "scripts/core.py",
        SCRIPT,
        PARENT_ROOT / "configs/protocol.json",
        PARENT_ROOT / "scripts/run_experiment.py",
        PARENT_ROOT / "runs/final/evaluation/selections.json",
        PARENT_ROOT / "runs/final/completion.json",
    ]
    if parent_marker().is_file():
        source_files.append(parent_marker())
    report["source_sha256"] = {str(path): file_sha256(path) for path in source_files}
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(report_path)})
    return report


def run_bootstrap(*, allow_concurrent_parent: bool = False) -> dict[str, Any]:
    report_path = RUNS / "bootstrap/report.json"
    completion = RUNS / "bootstrap/completion.json"
    if completion.is_file():
        return read_json(report_path)
    preflight = run_preflight(allow_concurrent_parent=allow_concurrent_parent)
    selections = read_json(PARENT_ROOT / "runs/final/evaluation/selections.json")
    data = PARENT.SOURCE.load_role("final")
    image_ids = [int(sample["image_id"]) for sample in data["samples"]]
    ground_truth = read_json(PARENT.SOURCE.COMMON / "final/gt.json")
    comparisons = []
    for spec in CORE.comparison_specs(PROTOCOL):
        budget_index = list(map(float, PROTOCOL["budgets"])).index(spec["budget"])
        output = RUNS / "bootstrap" / f"{spec['key'].replace('|', '_')}.json"
        if output.is_file():
            item = read_json(output)
            values = np.asarray(item["values"], dtype=np.float64)
        else:
            candidate = PARENT.SOURCE._selection_from_json(
                selections, spec["candidate_key"]
            )
            comparator = PARENT.SOURCE._selection_from_json(
                selections, spec["comparator_key"]
            )
            CORE.validate_selection(candidate, image_ids)
            CORE.validate_selection(comparator, image_ids)
            candidate_predictions, _ = PARENT.SOURCE.V1.REL.canonical.mixed_predictions(
                data, candidate, 0.001
            )
            comparator_predictions, _ = PARENT.SOURCE.V1.REL.canonical.mixed_predictions(
                data, comparator, 0.001
            )
            seed = int(PROTOCOL["bootstrap_seed"]) + budget_index * 101
            generator = np.random.default_rng(seed)
            indices = generator.integers(
                0,
                len(image_ids),
                size=(int(PROTOCOL["bootstrap_replicates"]), len(image_ids)),
                dtype=np.int32,
            )
            PARENT.SOURCE.BOOTSTRAP_CORE.configure_bootstrap(
                ground_truth, image_ids, candidate_predictions, comparator_predictions
            )
            values = np.empty(len(indices), dtype=np.float64)
            workers = min(
                int(PROTOCOL["bootstrap_workers"]),
                max(1, (os.cpu_count() or 1) - 2),
            )
            context = mp.get_context("fork")
            try:
                with context.Pool(processes=workers) as pool:
                    tasks = ((index, indices[index]) for index in range(len(indices)))
                    for count, (replicate, delta) in enumerate(
                        pool.imap_unordered(
                            PARENT.SOURCE.BOOTSTRAP_CORE.bootstrap_worker,
                            tasks,
                            chunksize=1,
                        ),
                        start=1,
                    ):
                        values[int(replicate)] = float(delta)
                        if count % 100 == 0 or count == len(indices):
                            print(
                                f"[bootstrap:{spec['key']}] {count}/{len(indices)}",
                                flush=True,
                            )
            finally:
                PARENT.SOURCE.BOOTSTRAP_CORE.clear_bootstrap()
            item = {
                "protocol_id": PROTOCOL["protocol_id"],
                **spec,
                "replicates": len(values),
                "seed": seed,
                "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                "resampling_unit": "image",
                "confirmatory": False,
                "final_is_hidden": False,
                "values": values.tolist(),
            }
            atomic_json(output, item)
        if len(values) != int(PROTOCOL["bootstrap_replicates"]):
            raise RuntimeError(f"Incomplete bootstrap values: {spec['key']}")
        comparisons.append(
            {key: value for key, value in item.items() if key != "values"}
            | PARENT.SOURCE._bootstrap_summary(values)
        )
    if len(comparisons) != int(PROTOCOL["expected_comparisons"]):
        raise RuntimeError("Comparison count changed")
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "status": PROTOCOL["status"],
        "comparisons": comparisons,
        "replicates_per_comparison": int(PROTOCOL["bootstrap_replicates"]),
        "confirmatory": False,
        "final_is_hidden": False,
        "final_used_for_selection": False,
        "concurrent_with_parent_bootstrap": bool(
            preflight.get("concurrent_with_parent_bootstrap", False)
        ),
    }
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "report_sha256": file_sha256(report_path)})
    atomic_json(RUNS / "COMPLETED.json", {"complete": True, "comparisons": len(comparisons)})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("status", "preflight", "bootstrap", "all"), default="all")
    parser.add_argument("--wait-for-parent", action="store_true")
    parser.add_argument("--allow-concurrent-parent", action="store_true")
    args = parser.parse_args()
    if args.wait_for_parent and args.allow_concurrent_parent:
        parser.error("--wait-for-parent and --allow-concurrent-parent are mutually exclusive")
    if args.stage == "status":
        print(json.dumps(readiness(), indent=2, ensure_ascii=False))
        return
    if args.wait_for_parent:
        wait_for_parent()
    if args.stage == "preflight":
        result = run_preflight(allow_concurrent_parent=args.allow_concurrent_parent)
    else:
        result = run_bootstrap(allow_concurrent_parent=args.allow_concurrent_parent)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
