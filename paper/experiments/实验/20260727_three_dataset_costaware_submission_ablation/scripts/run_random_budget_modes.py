#!/usr/bin/env python3
"""Add 50-seed Random bands for Online and Fixed-byte budget semantics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from submission_extensions_core import (
    calibrated_online_threshold,
    random_byte_actions,
    runtime_manifest,
    select_online_actions,
    summarize_random_band,
)


SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
SOURCE_20 = REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e"
SOURCE_23 = REPO / "paper/experiments/20260723_three_dataset_nineranker_triaction_paper_exploration"
sys.path.insert(0, str(SOURCE_23 / "scripts"))
import run_dataset as nine  # type: ignore  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(map(dict, rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in values for field in row})
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


def role_inputs(dataset: str, role: str) -> list[Path]:
    root = nine.canonical_root(dataset) / "common" / dataset / role
    manifest_path = root / "source_manifest.json"
    manifest = read_json(manifest_path)
    return [
        manifest_path,
        root / "samples.json",
        root / "gt.json",
        *(Path(value) for value in manifest["action_sources"].values()),
    ]


def input_paths(dataset: str) -> list[Path]:
    sources = nine.source_paths(dataset)
    return [
        EXP_ROOT / "configs/extensions_protocol.json",
        SOURCE_23 / "configs/protocol.json",
        SOURCE_23 / "runs" / dataset / "CORE_COMPLETED",
        sources["models"],
        sources["action_gate"],
        *role_inputs(dataset, "calibration"),
        *role_inputs(dataset, "final"),
        SCRIPT,
        EXP_ROOT / "scripts/submission_extensions_core.py",
    ]


def snapshot(paths: Iterable[Path]) -> dict[str, str]:
    normalized = sorted({Path(path).resolve() for path in paths})
    missing = [str(path) for path in normalized if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Random-band inputs:\n" + "\n".join(missing))
    return {str(path): file_sha256(path) for path in normalized}


def numeric_row(
    row: Mapping[str, Any],
    *,
    dataset: str,
    mode: str,
    budget: float,
    seed_index: int,
    random_seed: int,
    online_threshold: float | None,
) -> dict[str, Any]:
    return {
        **dict(row),
        "dataset": dataset,
        "method": "Random",
        "mode": mode,
        "nominal_budget": budget,
        "action_policy": "SharedAction",
        "calibration_scope": "same_seed_calibration_only",
        "seed": seed_index,
        "random_seed": random_seed,
        "online_threshold": "" if online_threshold is None else online_threshold,
        "final_used_for_calibration": False,
        "diagnostic_only": True,
    }


def run(dataset: str, *, preflight_only: bool = False) -> list[dict[str, Any]]:
    protocol = read_json(EXP_ROOT / "configs/extensions_protocol.json")
    if dataset not in protocol["datasets"]:
        raise ValueError(f"Unsupported dataset: {dataset}")
    output = EXP_ROOT / "runs/random_budget_modes" / dataset
    output.mkdir(parents=True, exist_ok=True)
    paths = input_paths(dataset)
    before = snapshot(paths)
    identity = {
        "protocol_sha256": file_sha256(EXP_ROOT / "configs/extensions_protocol.json"),
        "input_sha256": before,
    }
    checkpoint_path = output / "checkpoint.json"
    if checkpoint_path.is_file():
        checkpoint = read_json(checkpoint_path)
        if {key: checkpoint.get(key) for key in identity} != identity:
            raise RuntimeError("Random-band checkpoint differs from frozen inputs")
        rows = list(checkpoint.get("rows", []))
    else:
        rows = []

    nine.configure_reused_runner(dataset, output)
    roles = {
        role: nine.tri.current.load_role(dataset, role)
        for role in ("calibration", "final")
    }
    observed_sizes = {role: len(data["samples"]) for role, data in roles.items()}
    expected_sizes = {
        role: int(nine.EXPECTED_ROLES[dataset][role])
        for role in ("calibration", "final")
    }
    if observed_sizes != expected_sizes:
        raise RuntimeError(
            f"Random-band role-size drift: {observed_sizes} != {expected_sizes}"
        )
    sources = nine.source_paths(dataset)
    models = nine.tri.load_frozen_models(sources["models"])
    gate = read_json(sources["action_gate"])
    actions = {
        role: nine.tri.current.chosen_actions(models["action_heads"], gate, roles[role])
        for role in ("calibration", "final")
    }
    if preflight_only:
        return [
            {
                "dataset": dataset,
                "inputs": len(before),
                "calibration_images": len(roles["calibration"]["samples"]),
                "final_images": len(roles["final"]["samples"]),
                "shared_actions_loaded": all(
                    len(actions[role]) == len(roles[role]["samples"])
                    for role in ("calibration", "final")
                ),
                "preflight": True,
            }
        ]
    expected_modes = {
        (mode, float(budget), seed)
        for seed in range(int(protocol["random_seeds"]))
        for mode, budgets in protocol["work_points"].items()
        for budget in budgets
    }
    completed = {
        (str(row["mode"]), float(row["nominal_budget"]), int(row["seed"]))
        for row in rows
    }
    unknown = completed - expected_modes
    if unknown:
        raise RuntimeError(f"Random-band checkpoint has unknown rows: {sorted(unknown)}")

    for seed_index in range(int(protocol["random_seeds"])):
        random_seed = int(protocol["random_seed_base"]) + seed_index
        calibration_scores = nine.tri.current.random_scores(
            roles["calibration"]["samples"], random_seed
        )
        final_scores = nine.tri.current.random_scores(
            roles["final"]["samples"], random_seed
        )
        for target in map(float, protocol["work_points"]["online"]):
            key = ("online", target, seed_index)
            if key in completed:
                continue
            threshold = calibrated_online_threshold(calibration_scores, target)
            selected = select_online_actions(
                roles["final"]["samples"],
                scores=final_scores,
                action_map=actions["final"],
                threshold=threshold,
            )
            evaluated = nine.tri.current.evaluate(
                "Random", "online_seed", target, roles["final"], selected, models
            )
            rows.append(
                numeric_row(
                    evaluated,
                    dataset=dataset,
                    mode="online",
                    budget=target,
                    seed_index=seed_index,
                    random_seed=random_seed,
                    online_threshold=threshold,
                )
            )
            completed.add(key)
            write_json(checkpoint_path, {**identity, "rows": rows})

        for budget in map(float, protocol["work_points"]["fixed_byte"]):
            key = ("fixed_byte", budget, seed_index)
            if key in completed:
                continue
            selected = random_byte_actions(
                roles["final"]["samples"],
                scores=final_scores,
                action_map=actions["final"],
                budget=budget,
            )
            evaluated = nine.tri.current.evaluate(
                "Random", "fixed_byte_seed", budget, roles["final"], selected, models
            )
            rows.append(
                numeric_row(
                    evaluated,
                    dataset=dataset,
                    mode="fixed_byte",
                    budget=budget,
                    seed_index=seed_index,
                    random_seed=random_seed,
                    online_threshold=None,
                )
            )
            completed.add(key)
            write_json(checkpoint_path, {**identity, "rows": rows})
        write_csv(output / "random_seed_rows.csv", rows)
        print(f"[{dataset}-random-budget] seed {seed_index + 1}/50 complete", flush=True)

    if completed != expected_modes or len(rows) != len(expected_modes):
        raise RuntimeError(
            f"Random-band grid incomplete: {len(rows)} != {len(expected_modes)}"
        )
    summaries = []
    metrics = (
        "official_map50_95",
        "request_kb_per_image",
        "upload_ratio",
        "e2e_mean_ms_10mbps",
    )
    for mode, budgets in protocol["work_points"].items():
        for budget in map(float, budgets):
            selected_rows = [
                row
                for row in rows
                if row["mode"] == mode
                and math.isclose(float(row["nominal_budget"]), budget, abs_tol=1e-12)
            ]
            summaries.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "nominal_budget": budget,
                    "action_policy": "SharedAction",
                    **summarize_random_band(selected_rows, metrics=metrics),
                }
            )
    after = snapshot(paths)
    if before != after:
        raise RuntimeError("A frozen input changed during Random-band evaluation")
    write_csv(output / "random_seed_rows.csv", rows)
    write_csv(output / "random_bands.csv", summaries)
    write_json(
        output / "provenance.json",
        {
            **identity,
            "input_sha256_after": after,
            "final_used_for_selection": False,
            "final_interpretation": "already_seen_posthoc_random_band_only",
            "rows": len(rows),
            "runtime": runtime_manifest(),
        },
    )
    (output / "COMPLETED").write_text("ok\n", encoding="ascii")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("coco", "ena24", "oregon"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.dataset, preflight_only=args.preflight_only),
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
