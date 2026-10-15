#!/usr/bin/env python3
"""Freeze the complete, ordered COCO causal-online paper run matrix."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping


MATRIX_SCHEMA = "cavs_online_paper_run_matrix_v2"
ORDERING_SEED = 20260828
RANDOM_SEEDS = (20260720, 20260721, 20260722, 20260723, 20260724)
BUDGETS = ((10, Fraction(1, 10)), (25, Fraction(1, 4)), (40, Fraction(2, 5)))
NETWORK_PROFILES = {
    "native": {
        "mechanism": "none",
        "added_round_trip_delay_ms": 0,
        "uplink_mbps": None,
        "packet_loss_percent": 0.0,
    },
    "wan_20ms_20mbps": {
        "mechanism": "linux_tc_netem_tbf",
        "added_round_trip_delay_ms": 20,
        "uplink_mbps": 20,
        "packet_loss_percent": 0.0,
    },
    "wan_50ms_5mbps": {
        "mechanism": "linux_tc_netem_tbf",
        "added_round_trip_delay_ms": 50,
        "uplink_mbps": 5,
        "packet_loss_percent": 0.0,
    },
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return dict(value)


def _asset(bundle: Path, relative: str) -> dict[str, str]:
    root = bundle.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"run-matrix asset escapes board bundle: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "sha256": file_sha256(path)}


def _method_policy(protocol: Mapping[str, Any], method: str) -> str:
    methods = protocol.get("methods")
    if not isinstance(methods, Mapping) or not isinstance(methods.get(method), Mapping):
        raise ValueError(f"protocol omits method: {method}")
    policy = methods[method].get("policy")
    if not isinstance(policy, str) or not policy:
        raise ValueError(f"protocol method has no policy path: {method}")
    return policy


def _row(
    bundle: Path,
    protocol: Mapping[str, Any],
    *,
    family: str,
    method: str,
    budget_percent: int,
    budget: Fraction,
    control: str,
    network_profile: str = "native",
    random_seed: int | None = None,
    controller_variant: str = "registered",
) -> dict[str, Any]:
    policy_asset = _asset(bundle, _method_policy(protocol, method))
    control_asset = _asset(bundle, control)
    seed_suffix = f"-seed{random_seed}" if random_seed is not None else ""
    controller_suffix = "-static" if controller_variant == "static_lambda" else ""
    network_suffix = "" if network_profile == "native" else f"-{network_profile}"
    run_key = (
        f"{family}-{method}{seed_suffix}{controller_suffix}-"
        f"b{budget_percent:03d}{network_suffix}"
    )
    return {
        "run_key": run_key,
        "family": family,
        "method": method,
        "budget_percent": budget_percent,
        "budget_numerator": budget.numerator,
        "budget_denominator": budget.denominator,
        "random_seed": random_seed,
        "controller_variant": controller_variant,
        "network_profile": network_profile,
        "policy": policy_asset["path"],
        "policy_sha256": policy_asset["sha256"],
        "control": control_asset["path"],
        "control_sha256": control_asset["sha256"],
        "expected_images": int(protocol["dataset"]["expected_images"]),
        "zero_lookahead": True,
        "global_plan": False,
        "buffer_frames": 0,
    }


def _ordered(rows: list[dict[str, Any]], seed_offset: int) -> list[dict[str, Any]]:
    result = list(rows)
    random.Random(ORDERING_SEED + seed_offset).shuffle(result)
    return result


def build_run_matrix(
    bundle_root: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    bundle = Path(bundle_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    protocol = _load_object(protocol_file)
    if protocol.get("protocol_id") != "coco_rknn_fp16_causal_online_hil_final_v2":
        raise ValueError("run matrix requires the final-v2 protocol")
    if int(protocol.get("dataset", {}).get("expected_images", -1)) != 5000:
        raise ValueError("run matrix requires the frozen 5,000-image COCO sequence")
    if [Fraction(str(value)) for value in protocol.get("budgets", ())] != [
        value for _, value in BUDGETS
    ]:
        raise ValueError("protocol budgets must be exactly 10%, 25%, and 40%")

    main = [
        _row(
            bundle, protocol, family="main", method="edge_only",
            budget_percent=100, budget=Fraction(1, 1),
            control="controls/edge_only/100.json",
        ),
        _row(
            bundle, protocol, family="main", method="always_original",
            budget_percent=100, budget=Fraction(1, 1),
            control="controls/always_original/100.json",
        ),
    ]
    for method in ("cavs_oq", "cavs_oqr", "oric", "moric", "confidence"):
        for percent, budget in BUDGETS:
            main.append(_row(
                bundle, protocol, family="main", method=method,
                budget_percent=percent, budget=budget,
                control=f"controls/{method}/{percent}.json",
            ))
    for seed in RANDOM_SEEDS:
        for percent, budget in BUDGETS:
            main.append(_row(
                bundle, protocol, family="main", method="random",
                budget_percent=percent, budget=budget, random_seed=seed,
                control=f"controls/random_seed_{seed}/{percent}.json",
            ))

    ablation = []
    for method in ("cavs_oq", "cavs_oqr"):
        for percent, budget in BUDGETS:
            ablation.append(_row(
                bundle, protocol, family="controller_ablation", method=method,
                budget_percent=percent, budget=budget,
                control=f"controls/{method}_static/{percent}.json",
                controller_variant="static_lambda",
            ))

    network = []
    for profile in ("wan_20ms_20mbps", "wan_50ms_5mbps"):
        for method in ("cavs_oq", "cavs_oqr", "oric", "moric"):
            network.append(_row(
                bundle, protocol, family="network_sensitivity", method=method,
                budget_percent=25, budget=Fraction(1, 4),
                control=f"controls/{method}/25.json", network_profile=profile,
            ))

    families = {
        "main": _ordered(main, 0),
        "controller_ablation": _ordered(ablation, 1),
        "network_sensitivity": _ordered(network, 2),
    }
    runs: list[dict[str, Any]] = []
    for family in ("main", "controller_ablation", "network_sensitivity"):
        runs.extend(families[family])
    for index, row in enumerate(runs, start=1):
        row["run_index"] = index
    keys = [str(row["run_key"]) for row in runs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("run matrix contains duplicate run keys")
    counts = {name: len(rows) for name, rows in families.items()}
    if counts != {"main": 32, "controller_ablation": 6, "network_sensitivity": 8}:
        raise RuntimeError(f"paper run matrix is incomplete: {counts}")

    return {
        "schema": MATRIX_SCHEMA,
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_sha256": file_sha256(protocol_file),
        "expected_images": 5000,
        "ordering_seed": ORDERING_SEED,
        "random_seeds": list(RANDOM_SEEDS),
        "family_counts": dict(sorted(counts.items())),
        "network_profiles": NETWORK_PROFILES,
        "final_used_for_matrix_selection": False,
        "runs": runs,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = build_run_matrix(args.bundle_root, args.protocol)
    _atomic_json(args.output, matrix)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "runs": len(matrix["runs"]),
        "family_counts": matrix["family_counts"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
