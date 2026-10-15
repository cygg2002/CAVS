#!/usr/bin/env python3
"""Run one registered bootstrap work point without writing dataset summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_bootstrap_sensitivity as base
from submission_extensions_core import (
    raw_checkpoint_was_written,
    runtime_manifest,
    select_unique_workpoint,
    summarize_bootstrap_prefixes,
    validated_parallel_workers,
)


SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]


def run(
    dataset: str,
    *,
    mode: str,
    budget: float,
    workers: int = 1,
    preflight_only: bool = False,
) -> dict[str, Any]:
    workers = validated_parallel_workers(workers)
    protocol = base.read_json(EXP_ROOT / "configs/extensions_protocol.json")
    if dataset not in protocol["datasets"]:
        raise ValueError(f"Unsupported dataset: {dataset}")

    paths = [*base.source_paths(dataset), SCRIPT]
    before = base.snapshot(paths)
    data, comparisons = base.source_comparisons(dataset)
    comparison = select_unique_workpoint(comparisons, mode=mode, budget=budget)
    expected = base.old_rows(dataset)[(mode, float(comparison["budget"]))]
    digest = base.action_pair_digest(
        comparison["baseline_actions"], comparison["candidate_actions"]
    )
    if digest != str(expected["action_pair_sha256"]):
        raise RuntimeError(f"Action reconstruction drift at {(mode, budget)}")

    budget = float(comparison["budget"])
    label = f"{mode}-{budget:.2f}"
    output = EXP_ROOT / "runs/bootstrap_sensitivity" / dataset
    raw_path = output / "raw" / f"{label}-{digest[:12]}.json"
    maximum = max(map(int, protocol["bootstrap_checkpoints"]))
    seed = int(protocol["bootstrap_seed"])
    identity = {
        "dataset": dataset,
        "mode": mode,
        "budget": budget,
        "action_pair_sha256": digest,
        "bootstrap_seed": seed,
        "maximum_replicates": maximum,
        "protocol_sha256": base.file_sha256(
            EXP_ROOT / "configs/extensions_protocol.json"
        ),
    }
    resumed_replicates = 0
    if raw_path.is_file():
        resumed_replicates = len(base.read_json(raw_path).get("ap_deltas", []))

    status = {
        "dataset": dataset,
        "mode": mode,
        "budget": budget,
        "action_pair_sha256": digest,
        "action_digest_reproduced": True,
        "bootstrap_seed": seed,
        "maximum_replicates": maximum,
        "raw_checkpoint": str(raw_path),
        "resumed_replicates": resumed_replicates,
        "identical_action_shortcut": comparison["baseline_actions"]
        == comparison["candidate_actions"],
        "workers": workers,
        "preflight": preflight_only,
    }
    if preflight_only:
        return status

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    ap_deltas, kb_deltas = base.raw_bootstrap(
        data,
        baseline_actions=comparison["baseline_actions"],
        candidate_actions=comparison["candidate_actions"],
        replicates=maximum,
        seed=seed,
        checkpoint_path=raw_path,
        identity=identity,
        label=label,
        workers=workers,
    )
    rows = summarize_bootstrap_prefixes(
        ap_deltas=ap_deltas,
        kb_deltas=kb_deltas,
        checkpoints=tuple(map(int, protocol["bootstrap_checkpoints"])),
        mode=mode,
        ap_tolerance=float(protocol["ap_tolerance"]),
    )
    base.reconcile_200(expected, rows[0])
    after = base.snapshot(paths)
    if before != after:
        raise RuntimeError("A frozen input changed during bootstrap shard execution")

    checkpoint_written = raw_checkpoint_was_written(
        checkpoint_exists=raw_path.is_file(),
        identical_action_shortcut=bool(status["identical_action_shortcut"]),
    )
    manifest = {
        **status,
        "preflight": False,
        "completed_replicates": len(ap_deltas),
        "raw_checkpoint_written": checkpoint_written,
        "raw_checkpoint_sha256": (
            base.file_sha256(raw_path) if checkpoint_written else None
        ),
        "input_sha256": before,
        "input_sha256_after": after,
        "final_artifacts_read": False,
        "pointwise_not_simultaneous": True,
        "percentile_interval_only": True,
        "runtime": runtime_manifest(),
        "summaries": rows,
    }
    shard_path = output / "shards" / f"{label}-{digest[:12]}.json"
    base.write_json(shard_path, manifest)
    return {**status, "completed_replicates": len(ap_deltas), "shard": str(shard_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("coco", "ena24", "oregon"), required=True)
    parser.add_argument(
        "--mode", choices=("fixed_call", "online", "fixed_byte"), required=True
    )
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.dataset,
                mode=args.mode,
                budget=args.budget,
                workers=args.workers,
                preflight_only=args.preflight_only,
            ),
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
