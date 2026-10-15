#!/usr/bin/env python3
"""Create a new immutable main-32 bundle from completed policy artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


BUDGETS = (("10", "1/10"), ("25", "1/4"), ("40", "2/5"))
LAMBDA_GRID = [*(index / 100 for index in range(201)), 2.5, 3.0, 4.0, 5.0]
STEP_GRID = [0.0, 0.00025, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run(command: Sequence[str], *, cwd: Path) -> None:
    subprocess.run(list(command), cwd=cwd, check=True)


def _copy_policy(source: Path, destination: Path) -> None:
    if not (source / "manifest.json").is_file():
        raise FileNotFoundError(source / "manifest.json")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(temporary, destination)


def _copy_experiment(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for name in ("SPEC.md", "board_bundle", "board_formal", "host", "mac_server"):
        path = source / name
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            shutil.copytree(
                path,
                destination / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "runs"),
            )
        else:
            shutil.copy2(path, destination / name)


def _required_input(path: Path, *, kind: str, label: str) -> Path:
    resolved = path.resolve()
    valid = resolved.is_file() if kind == "file" else resolved.is_dir()
    if not valid:
        raise FileNotFoundError(f"{label}: {resolved}")
    return resolved


def _format_grid(values: Sequence[float]) -> str:
    return ",".join(format(value, ".12g") for value in values)


def finalize_bundle(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_experiment.resolve()
    output = args.output_experiment.resolve()
    calibration_candidates = _required_input(
        args.calibration_candidates, kind="file", label="calibration candidates"
    )
    calibration_report = _required_input(
        args.calibration_report, kind="file", label="calibration report"
    )
    calibration_images = _required_input(
        args.calibration_images, kind="directory", label="calibration images"
    )
    _copy_experiment(source, output)
    bundle = output / "board_bundle"
    host = output / "host"
    _copy_policy(args.cavs_oq_policy.resolve(), bundle / "policies/cavs_oq")
    _copy_policy(args.external_policy.resolve(), bundle / "policies/external")

    scripts = bundle / "scripts"
    sys.path.insert(0, str(scripts))
    from online_methods import build_method_runtime

    rknn_sha256 = "0915a99235c3be5723fcfeda45b06775ecccc2908cc5b490f093f23bbdede12d"
    cavs_manifest = bundle / "policies/cavs_oq/manifest.json"
    external_manifest = bundle / "policies/external/manifest.json"
    cavs_runtime = build_method_runtime(
        "cavs_oq", policy_path=cavs_manifest, expected_rknn_sha256=rknn_sha256
    )
    oric_runtime = build_method_runtime(
        "oric", policy_path=external_manifest, expected_rknn_sha256=rknn_sha256
    )
    moric_runtime = build_method_runtime(
        "moric", policy_path=external_manifest, expected_rknn_sha256=rknn_sha256
    )
    if tuple(cavs_runtime.action_ids) != ("edge", "original", "q90"):
        raise RuntimeError("CAVS-OQ export has the wrong action space")
    if tuple(oric_runtime.action_ids) != ("edge", "original") or tuple(moric_runtime.action_ids) != (
        "edge",
        "original",
    ):
        raise RuntimeError("external export has the wrong action space")

    protocol_path = bundle / "configs/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    built_in_policy = "scripts/online_methods.py"
    built_in_policy_sha256 = file_sha256(bundle / built_in_policy)
    protocol["methods"]["edge_only"] = {
        "policy": built_in_policy,
        "policy_sha256": built_in_policy_sha256,
        "actions": ["edge"],
        "scope": "deterministic_no_upload_system_anchor",
    }
    protocol["methods"]["always_original"] = {
        "policy": built_in_policy,
        "policy_sha256": built_in_policy_sha256,
        "actions": ["original"],
        "scope": "deterministic_full_upload_system_anchor",
    }
    protocol["methods"]["cavs_oq"] = {
        "policy": "policies/cavs_oq/manifest.json",
        "policy_sha256": file_sha256(cavs_manifest),
        "actions": ["edge", "original", "q90"],
        "scope": "rknn_fp16_three_seed_fresh_training",
    }
    for method in ("oric", "moric"):
        protocol["methods"][method] = {
            "policy": "policies/external/manifest.json",
            "policy_sha256": file_sha256(external_manifest),
            "actions": ["edge", "original"],
            "scope": "rknn_fp16_three_seed_ensemble",
        }
    protocol["controller"]["controls_by_method"] = {
        "cavs_oq": "controls/cavs_oq",
        "cavs_oqr": "controls/cavs_oqr",
    }
    expected_methods = {
        "edge_only",
        "always_original",
        "cavs_oq",
        "cavs_oqr",
        "oric",
        "moric",
        "confidence",
        "random",
    }
    missing_methods = expected_methods - set(protocol["methods"])
    if missing_methods:
        raise RuntimeError(f"main-32 protocol omits methods: {sorted(missing_methods)}")
    _atomic_json(protocol_path, protocol)

    records_root = host / "calibration_records"
    for method, policy in (
        ("cavs_oq", cavs_manifest),
        ("oric", external_manifest),
        ("moric", external_manifest),
    ):
        records = records_root / f"{method}.json"
        _run(
            (
                sys.executable,
                str(host / "build_calibration_records.py"),
                "--candidates",
                str(calibration_candidates),
                "--candidate-report",
                str(calibration_report),
                "--protocol",
                str(protocol_path),
                "--images",
                str(calibration_images),
                "--method",
                method,
                "--policy",
                str(policy),
                "--output",
                str(records),
            ),
            cwd=output,
        )
        for percent, budget in BUDGETS:
            static_dir = bundle / f"controls/{method}_static" if method == "cavs_oq" else bundle / f"controls/{method}"
            base_control = static_dir / f"{percent}.json"
            _run(
                (
                    sys.executable,
                    str(host / "calibrate_controls.py"),
                    "--records",
                    str(records),
                    "--protocol",
                    str(protocol_path),
                    "--method",
                    method,
                    "--budget",
                    budget,
                    "--objective",
                    "predicted_action_value",
                    "--lambda-grid",
                    _format_grid(LAMBDA_GRID),
                    "--source",
                    f"candidate_report={calibration_report}",
                    "--source",
                    f"candidates={calibration_candidates}",
                    "--source",
                    f"policy={policy}",
                    "--output",
                    str(base_control),
                ),
                cwd=output,
            )
            if method == "cavs_oq":
                _run(
                    (
                        sys.executable,
                        str(host / "calibrate_primal_dual.py"),
                        "--records",
                        str(records),
                        "--base-control",
                        str(base_control),
                        "--step-size-grid",
                        _format_grid(STEP_GRID),
                        "--minimum-price",
                        "0",
                        "--maximum-price",
                        "5",
                        "--output",
                        str(bundle / f"controls/cavs_oq/{percent}.json"),
                    ),
                    cwd=output,
                )

    built_in_policy_path = bundle / built_in_policy
    built_in_specs = [
        ("edge_only", None, "edge_only", (("100", "1"),)),
        ("always_original", None, "always_original", (("100", "1"),)),
        ("confidence", None, "confidence", BUDGETS),
        *[
            ("random", seed, f"random_seed_{seed}", BUDGETS)
            for seed in (20260720, 20260721, 20260722, 20260723, 20260724)
        ],
    ]
    for method, random_seed, output_name, budgets in built_in_specs:
        records = records_root / f"{output_name}.json"
        records_command = [
            sys.executable,
            str(host / "build_calibration_records.py"),
            "--candidates",
            str(calibration_candidates),
            "--candidate-report",
            str(calibration_report),
            "--protocol",
            str(protocol_path),
            "--images",
            str(calibration_images),
            "--method",
            method,
            "--policy",
            str(built_in_policy_path),
            "--output",
            str(records),
        ]
        if random_seed is not None:
            records_command.extend(("--random-seed", str(random_seed)))
        _run(records_command, cwd=output)
        for percent, budget in budgets:
            _run(
                (
                    sys.executable,
                    str(host / "calibrate_controls.py"),
                    "--records",
                    str(records),
                    "--protocol",
                    str(protocol_path),
                    "--method",
                    method,
                    "--budget",
                    budget,
                    "--objective",
                    "predicted_action_value",
                    "--lambda-grid",
                    _format_grid(LAMBDA_GRID),
                    "--source",
                    f"candidate_report={calibration_report}",
                    "--source",
                    f"candidates={calibration_candidates}",
                    "--source",
                    f"policy={built_in_policy_path}",
                    "--output",
                    str(bundle / f"controls/{output_name}/{percent}.json"),
                ),
                cwd=output,
            )

    bundle_manifest = bundle / "BUNDLE_MANIFEST.json"
    _run(
        (
            sys.executable,
            str(bundle / "scripts/verify_bundle.py"),
            "--root",
            str(bundle),
            "--manifest",
            str(bundle_manifest),
            "--write",
        ),
        cwd=output,
    )
    matrix = output / "RUN_MATRIX.json"
    _run(
        (
            sys.executable,
            str(host / "build_run_matrix.py"),
            "--bundle-root",
            str(bundle),
            "--protocol",
            str(protocol_path),
            "--output",
            str(matrix),
        ),
        cwd=output,
    )
    _run(
        (
            sys.executable,
            str(output / "board_formal/main_matrix.py"),
            "validate-matrix",
            "--matrix",
            str(matrix),
            "--bundle",
            str(bundle),
            "--protocol",
            str(protocol_path),
        ),
        cwd=output,
    )
    from online_runner import load_formal_control

    matrix_value = json.loads(matrix.read_text(encoding="utf-8"))
    main_rows = [row for row in matrix_value["runs"] if row.get("family") == "main"]
    for row in main_rows:
        load_formal_control(
            bundle / str(row["control"]),
            method=str(row["method"]),
            budget=f"{int(row['budget_numerator'])}/{int(row['budget_denominator'])}",
            policy_sha256=str(row["policy_sha256"]),
            random_seed=(
                int(row["random_seed"]) if row.get("random_seed") is not None else None
            ),
        )
    if len(main_rows) != 32:
        raise RuntimeError(f"main control-binding gate is incomplete: {len(main_rows)}/32")
    report = {
        "complete": True,
        "source_experiment": str(source),
        "output_experiment": str(output),
        "cavs_oq_manifest_sha256": file_sha256(cavs_manifest),
        "external_manifest_sha256": file_sha256(external_manifest),
        "protocol_sha256": file_sha256(protocol_path),
        "bundle_manifest_sha256": file_sha256(bundle_manifest),
        "run_matrix_sha256": file_sha256(matrix),
        "main_runs": 32,
        "validated_control_bindings": len(main_rows),
        "old_learner_checkpoints_reused": False,
        "final_used_for_training_or_control": False,
    }
    _atomic_json(output / "INTEGRATION_REPORT.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--output-experiment", type=Path, required=True)
    parser.add_argument("--cavs-oq-policy", type=Path, required=True)
    parser.add_argument("--external-policy", type=Path, required=True)
    parser.add_argument("--calibration-candidates", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--calibration-images", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = finalize_bundle(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
