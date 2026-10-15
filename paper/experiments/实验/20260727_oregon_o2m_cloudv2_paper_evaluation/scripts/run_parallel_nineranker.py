#!/usr/bin/env python3
"""Resume the Cloud-V2 nine-Ranker core and parallelize independent bootstraps."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable


SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]
DATASET = "oregon"
BOOTSTRAP_METADATA = {
    "comparison",
    "method",
    "mode",
    "nominal_budget",
    "action_pair_sha256",
    "bootstrap_seed",
    "reused_identical_action_pair",
}
_WORKER_STATE: dict[str, Any] = {}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def normalized_actions(actions: dict[Any, str]) -> dict[int, str]:
    return {int(image_id): str(action) for image_id, action in actions.items()}


def plan_workpoints(
    *,
    rankers: Iterable[str],
    budgets: Iterable[float],
    selections: dict[str, dict[Any, str]],
    digest_fn: Callable[[dict[int, str], dict[int, str]], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plan: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    budgets = tuple(float(raw_budget) for raw_budget in budgets)
    for ranker in rankers:
        for raw_budget in budgets:
            budget = float(raw_budget)
            baseline = normalized_actions(
                selections[f"fixed_call|SharedAction|{ranker}|{budget:.2f}"]
            )
            candidate = normalized_actions(
                selections[f"fixed_call|GBDT-Hierarchical|{ranker}|{budget:.2f}"]
            )
            digest = str(digest_fn(baseline, candidate))
            if baseline == candidate:
                kind = "identical"
            elif digest in seen:
                kind = "reuse"
            else:
                kind = "compute"
                seen.add(digest)
                jobs.append(
                    {
                        "digest": digest,
                        "baseline_actions": baseline,
                        "candidate_actions": candidate,
                    }
                )
            plan.append(
                {
                    "method": str(ranker),
                    "nominal_budget": budget,
                    "digest": digest,
                    "kind": kind,
                }
            )
    return plan, jobs


def zero_result(replicates: int, unit: str) -> dict[str, Any]:
    return {
        "replicates": int(replicates),
        "official_map50_95_delta_mean": 0.0,
        "official_map50_95_delta_ci_low": 0.0,
        "official_map50_95_delta_ci_high": 0.0,
        "official_map50_95_probability_gt_zero": 0.0,
        "request_kb_per_image_delta_mean": 0.0,
        "request_kb_per_image_delta_ci_low": 0.0,
        "request_kb_per_image_delta_ci_high": 0.0,
        "bootstrap_unit": str(unit),
        "paired": True,
        "strategy_refit_in_bootstrap": False,
    }


def bootstrap_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in BOOTSTRAP_METADATA}


def materialize_rows(
    plan: list[dict[str, Any]],
    computed: dict[str, dict[str, Any]],
    *,
    comparison: str,
    replicates: int,
    seed: int,
    unit: str,
) -> list[dict[str, Any]]:
    rows = []
    for point in plan:
        if point["kind"] == "identical":
            result = zero_result(replicates, unit)
        else:
            result = bootstrap_payload(computed[point["digest"]])
            if point["kind"] == "reuse":
                result["reused_identical_action_pair"] = True
        result.update(
            {
                "comparison": str(comparison),
                "method": point["method"],
                "mode": "fixed_call",
                "nominal_budget": float(point["nominal_budget"]),
                "action_pair_sha256": point["digest"],
                "bootstrap_seed": int(seed),
            }
        )
        rows.append(result)
    return rows


def import_path(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_registered_runner() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    sys.path.insert(0, str(SCRIPT.parent))
    pipeline = import_path(SCRIPT.parent / "run_cloudv2_pipeline.py", "cloudv2_parallel_pipeline")
    canonical_source, output = pipeline.prepare_adapter_layout()
    source_scripts = (
        pipeline.REPO
        / "paper/experiments/20260723_three_dataset_nineranker_triaction_paper_exploration/scripts"
    )
    sys.path.insert(0, str(source_scripts))
    runner = import_path(source_scripts / "run_dataset.py", "cloudv2_parallel_nineranker_runner")
    runner.EXP_ROOT = output
    runner.SOURCE_ROOT = canonical_source
    protocol = read_json(output / "configs/protocol.json")
    runner.configure_reused_runner(DATASET, output / "runs" / DATASET)
    checked = runner.preflight(DATASET, protocol)
    return runner, protocol, checked


def bootstrap_worker(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tri = _WORKER_STATE["tri"]
    result = tri.paired_bootstrap(
        _WORKER_STATE["final"],
        baseline_actions=job["baseline_actions"],
        candidate_actions=job["candidate_actions"],
        replicates=_WORKER_STATE["replicates"],
        seed=_WORKER_STATE["seed"],
        unit=_WORKER_STATE["unit"],
    )
    return str(job["digest"]), result


def run_parallel_bootstrap(
    runner: Any,
    protocol: dict[str, Any],
    checked: dict[str, Any],
    *,
    workers: int,
) -> None:
    output = runner.EXP_ROOT / "runs" / DATASET
    if not (output / "CORE_COMPLETED").is_file():
        raise RuntimeError("Parallel bootstrap requires a completed frozen core")
    runner.configure_reused_runner(DATASET, output)
    final = runner.tri.current.load_role(DATASET, "final")
    selections = read_json(output / "selections.json")
    identity = {
        "protocol_sha256": runner.file_sha256(runner.EXP_ROOT / "configs/protocol.json"),
        "selections_sha256": runner.file_sha256(output / "selections.json"),
        "dataset": DATASET,
    }
    plan, jobs = plan_workpoints(
        rankers=protocol["constrained_rankers"],
        budgets=map(float, protocol["primary_call_budgets"]),
        selections=selections,
        digest_fn=runner.tri.action_pair_digest,
    )
    computed: dict[str, dict[str, Any]] = {}
    serial_checkpoint = output / "bootstrap_checkpoint.json"
    if serial_checkpoint.is_file():
        checkpoint = read_json(serial_checkpoint)
        if {key: checkpoint.get(key) for key in identity} != identity:
            raise RuntimeError("Existing serial checkpoint does not match frozen inputs")
        for row in checkpoint.get("rows", []):
            computed[str(row["action_pair_sha256"])] = bootstrap_payload(row)

    cache_path = output / "parallel_bootstrap_cache.json"
    if cache_path.is_file():
        checkpoint = read_json(cache_path)
        if {key: checkpoint.get(key) for key in identity} != identity:
            raise RuntimeError("Parallel checkpoint does not match frozen inputs")
        computed.update(
            {
                str(digest): bootstrap_payload(result)
                for digest, result in checkpoint.get("computed", {}).items()
            }
        )

    pending = [job for job in jobs if job["digest"] not in computed]
    worker_count = min(max(1, int(workers)), len(pending) or 1, os.cpu_count() or 1)
    _WORKER_STATE.update(
        {
            "tri": runner.tri,
            "final": final,
            "replicates": int(protocol["bootstrap_replicates"]),
            "seed": int(protocol["bootstrap_seed_base"]),
            "unit": str(protocol["bootstrap_unit"]),
        }
    )
    if pending:
        context = mp.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count, mp_context=context
        ) as pool:
            futures = {pool.submit(bootstrap_worker, job): job["digest"] for job in pending}
            for future in concurrent.futures.as_completed(futures):
                digest, result = future.result()
                computed[digest] = result
                write_json(cache_path, {**identity, "computed": computed})
                print(
                    f"[{DATASET}-cloudv2-parallel] bootstrap {len(computed)}/{len(jobs)} unique pairs complete",
                    flush=True,
                )

    rows = materialize_rows(
        plan,
        computed,
        comparison=str(protocol["bootstrap_comparison"]),
        replicates=int(protocol["bootstrap_replicates"]),
        seed=int(protocol["bootstrap_seed_base"]),
        unit=str(protocol["bootstrap_unit"]),
    )
    runner.assert_bootstrap_rows_complete(
        rows,
        rankers=protocol["constrained_rankers"],
        budgets=protocol["primary_call_budgets"],
        comparison=str(protocol["bootstrap_comparison"]),
        replicates=int(protocol["bootstrap_replicates"]),
    )
    write_json(serial_checkpoint, {**identity, "rows": rows})
    write_csv(output / "paired_bootstrap.csv", rows)
    after = runner.snapshot_file_sha256(runner.frozen_input_paths(DATASET))
    if checked["frozen_input_sha256"] != after:
        raise RuntimeError("A frozen input changed during parallel bootstrap")
    summary = read_json(output / "summary.json")
    summary.update(
        {
            "status": "complete",
            "bootstrap_comparisons": len(rows),
            "bootstrap_replicates": int(protocol["bootstrap_replicates"]),
            "bootstrap_seed": int(protocol["bootstrap_seed_base"]),
            "bootstrap_execution": "parallel_fork_unique_action_pairs",
            "bootstrap_workers": worker_count,
        }
    )
    write_json(output / "summary.json", summary)
    (output / "COMPLETED").write_text("ok\n", encoding="ascii")


def run(stage: str, workers: int) -> None:
    status_path = EXP_ROOT / "runs/pipeline_status.json"
    runner, protocol, checked = load_registered_runner()
    try:
        if stage in {"core", "all"}:
            write_json(
                status_path,
                {"stage": "nineranker_core", "status": "running", "started_at": time.strftime("%F %T")},
            )
            runner.run_core(DATASET, protocol, checked)
        if stage in {"bootstrap", "all"}:
            write_json(
                status_path,
                {
                    "stage": "nineranker_parallel_bootstrap",
                    "status": "running",
                    "workers": int(workers),
                    "started_at": time.strftime("%F %T"),
                },
            )
            run_parallel_bootstrap(runner, protocol, checked, workers=workers)
    except Exception as error:
        write_json(
            status_path,
            {
                "stage": f"nineranker_parallel_{stage}",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": time.strftime("%F %T"),
            },
        )
        raise
    if stage in {"bootstrap", "all"}:
        (EXP_ROOT / "NINERANKER_COMPLETED").write_text("ok\n", encoding="ascii")
    write_json(
        status_path,
        {
            "stage": "nineranker_parallel",
            "status": "complete",
            "workers": int(workers),
            "completed_at": time.strftime("%F %T"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("core", "bootstrap", "all"), default="all")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.stage, args.workers)


if __name__ == "__main__":
    main()
