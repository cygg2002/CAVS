#!/usr/bin/env python3
"""Evaluate frozen ORIC/MORIC with two frozen TriAction payload policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
from typing import Any, Iterable, Optional

import numpy as np


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
EXPERIMENT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
EXTERNAL = (
    REPO
    / "paper/experiments/20260728_three_dataset_dcsb_oric_moric_external_baselines"
)
TRIACTION = (
    REPO
    / "paper/experiments/20260723_three_dataset_nineranker_triaction_paper_exploration"
)
TRIACTION_ENGINE = (
    REPO
    / "paper/experiments/20260722_ena24_hierarchical_triaction_paper_exploration"
)
CANONICAL = (
    REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e"
)
RUNS = EXPERIMENT / "runs"
PROTOCOL_PATH = EXPERIMENT / "configs/protocol.json"
EXTERNAL_PROTOCOL_PATH = EXTERNAL / "configs/protocol.json"

sys.path.insert(0, str(EXTERNAL / "scripts"))
from baseline_core import predict_reward_mlp  # type: ignore  # noqa: E402
from run_dataset import _role_features  # type: ignore  # noqa: E402
from runner_core import (  # type: ignore  # noqa: E402
    assert_hashes_unchanged,
    assert_output_path,
    atomic_write_json,
    atomic_write_rows,
    capture_hashes,
    file_sha256,
    role_source_paths,
    validate_role_data,
)

sys.path.insert(0, str(TRIACTION_ENGINE / "scripts"))
import run_exploration as tri  # type: ignore  # noqa: E402

from extension_core import (  # noqa: E402
    MODES,
    POLICIES,
    RANKERS,
    assert_control_reproduction,
    assert_result_matrix_complete,
    assert_same_membership,
    frozen_policy_thresholds,
    validate_protocol,
)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_run_name(dataset: str) -> str:
    if dataset not in {"coco", "ena24", "oregon"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    return "canonical_oregon_event_v1" if dataset == "oregon" else "canonical_v1"


def layout(dataset: str) -> dict[str, Path]:
    canonical = CANONICAL / "runs" / source_run_name(dataset)
    return {
        "common": canonical / "common",
        "benchmark": canonical / "benchmark" / dataset,
        "external": EXTERNAL / "runs" / dataset,
        "triaction": TRIACTION / "runs" / dataset,
        "output": RUNS / dataset,
    }


def _required_source_paths(dataset: str) -> list[Path]:
    paths = layout(dataset)
    external = paths["external"]
    triaction = paths["triaction"]
    benchmark = paths["benchmark"]
    required = {
        PROTOCOL_PATH,
        EXTERNAL_PROTOCOL_PATH,
        SCRIPT,
        SCRIPT.parent / "extension_core.py",
        EXTERNAL / "scripts/baseline_core.py",
        EXTERNAL / "scripts/run_dataset.py",
        EXTERNAL / "scripts/runner_core.py",
        EXTERNAL / "configs/protocol.json",
        TRIACTION / "configs/protocol.json",
        external / "completion.json",
        external / "evaluation/completion.json",
        external / "evaluation/results.csv",
        external / "evaluation/frozen_controls.json",
        external / "evaluation/oric_mlp.pkl",
        external / "evaluation/moric_mlp.pkl",
        external / "rewards/fit_artifacts.npz",
        triaction / "CORE_COMPLETED",
        triaction / "threshold_choices.json",
        triaction / "models/saferoi_gbdt.pkl",
        TRIACTION / "scripts/evaluation_core.py",
        TRIACTION / "scripts/run_dataset.py",
        TRIACTION_ENGINE / "scripts/exploration_core.py",
        TRIACTION_ENGINE / "scripts/run_exploration.py",
        CANONICAL / "scripts/audit_ena24_roi_protocol.py",
        CANONICAL / "scripts/benchmark_core.py",
        CANONICAL / "scripts/evaluate_historical_saferoi.py",
        CANONICAL / "scripts/joint_action_scheduler_core.py",
        CANONICAL / "scripts/protocol_core.py",
        CANONICAL / "scripts/run_multiranker_benchmark.py",
        CANONICAL / "scripts/saferoi_gbdt_core.py",
        REPO / "paper/experiments/RCMAR_v4/rcmar/__init__.py",
        REPO / "paper/experiments/RCMAR_v4/rcmar/coco_eval.py",
        benchmark / "models.json",
        benchmark / "action_gate.json",
    }
    for role in ("calibration", "final"):
        required.update(role_source_paths(paths["common"], dataset, role))
    return sorted(required)


def _source_digest(hashes: dict[str, str]) -> str:
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_complete(path: Path, label: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise RuntimeError(f"{label} completion is missing: {path}")
    payload = read_json(path)
    if payload.get("complete") is not True:
        raise RuntimeError(f"{label} is not complete: {path}")
    return payload


def _configure_canonical(dataset: str, common: Path) -> None:
    tri.current.COMMON = Path(common)
    tri.current.dataset_cost = tri.canonical_dataset_cost
    if dataset == "coco":
        expected = 18.2
    else:
        expected = 18.0
    if float(tri.current.dataset_cost(dataset)["edge_ms"]) != expected:
        raise RuntimeError("Canonical dataset-cost binding failed")


def run_preflight(dataset: str, protocol: dict[str, Any]) -> dict[str, Any]:
    validate_protocol(protocol)
    try:
        import sklearn
        import torch
    except ImportError as error:
        raise RuntimeError(
            "The frozen GBDT and ORIC/MORIC checkpoints require sklearn and torch"
        ) from error
    paths = layout(dataset)
    output = assert_output_path(paths["output"], RUNS)
    external_completion = _require_complete(
        paths["external"] / "completion.json", "External baseline"
    )
    _require_complete(
        paths["external"] / "evaluation/completion.json",
        "External baseline evaluation",
    )
    if int(external_completion.get("rows", -1)) != 65:
        raise RuntimeError("Frozen external baseline result count changed")
    if not (paths["triaction"] / "CORE_COMPLETED").is_file():
        raise RuntimeError("Frozen TriAction core is incomplete")
    with (paths["triaction"] / "models/saferoi_gbdt.pkl").open("rb") as handle:
        frozen_gbdt = pickle.load(handle)
    if not hasattr(frozen_gbdt, "predict_proba"):
        raise RuntimeError("Frozen SafeROI GBDT checkpoint is invalid")

    before = capture_hashes(_required_source_paths(dataset))
    _configure_canonical(dataset, paths["common"])
    role_sizes: dict[str, int] = {}
    for role in ("calibration", "final"):
        data = tri.current.load_role(dataset, role)
        expected = int(protocol["expected_role_sizes"][dataset][role])
        validate_role_data(
            data, dataset=dataset, role=role, expected_count=expected
        )
        role_sizes[role] = len(data["samples"])
    assert_hashes_unchanged(before)

    report = {
        "passed": True,
        "dataset": dataset,
        "protocol_id": protocol["protocol_id"],
        "role_sizes": role_sizes,
        "source_sha256": before,
        "source_digest": _source_digest(before),
        "external_results_sha256": file_sha256(
            paths["external"] / "evaluation/results.csv"
        ),
        "gbdt_model_sha256": file_sha256(
            paths["triaction"] / "models/saferoi_gbdt.pkl"
        ),
        "threshold_choices_sha256": file_sha256(
            paths["triaction"] / "threshold_choices.json"
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "final_used_for_selection": False,
    }
    report_path = output / "preflight/report.json"
    atomic_write_json(report_path, report, RUNS)
    atomic_write_json(
        output / "preflight/completion.json",
        {
            "dataset": dataset,
            "report_sha256": file_sha256(report_path),
            "source_digest": report["source_digest"],
            "complete": True,
        },
        RUNS,
    )
    return report


def _load_frozen_scores(
    paths: dict[str, Path],
    roles: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    with np.load(paths["external"] / "rewards/fit_artifacts.npz") as artifact:
        category_ids = list(map(int, artifact["category_ids"]))
    external_protocol = read_json(EXTERNAL_PROTOCOL_PATH)
    features = {
        role: _role_features(
            data, category_ids, int(external_protocol["oric"]["top_k"])
        )
        for role, data in roles.items()
    }
    checkpoints = {}
    for method, filename in (
        ("ORIC-MLP", "oric_mlp.pkl"),
        ("MORIC-MLP", "moric_mlp.pkl"),
    ):
        with (paths["external"] / "evaluation" / filename).open("rb") as handle:
            checkpoints[method] = pickle.load(handle)
    scores = {
        role: {
            method: np.asarray(
                predict_reward_mlp(checkpoint, features[role]), dtype=np.float64
            )
            for method, checkpoint in checkpoints.items()
        }
        for role in roles
    }
    parameter_counts = {
        method: sum(
            int(value.numel()) for value in checkpoint.get("state_dict", {}).values()
        )
        for method, checkpoint in checkpoints.items()
    }
    if not all(parameter_counts.values()):
        raise RuntimeError("Frozen external MLP parameter accounting is missing")
    return scores, parameter_counts


def _row(
    *,
    dataset: str,
    method: str,
    policy: str,
    mode: str,
    budget: float,
    data: dict[str, Any],
    selected: dict[int, str],
    models: dict[str, Any],
    threshold: Optional[float],
    lambda_value: Optional[float] = None,
    online_threshold: Optional[float] = None,
    selector_params: int,
) -> dict[str, Any]:
    row = tri.current.evaluate(method, mode, budget, data, selected, models)
    row.update(
        {
            "dataset": dataset,
            "method": method,
            "payload_policy": policy,
            "action_policy": policy,
            "mode": mode,
            "nominal_budget": float(budget),
            "roi_threshold": threshold,
            "roi_threshold_source": (
                "frozen_shared_8ranker_calibration"
                if policy == "GBDT-Hierarchical"
                else "all_cost_eligible"
            ),
            "external_selector_frozen": True,
            "external_selector_runtime_measured": False,
            "selector_ms_per_image": None,
            "selector_params": int(selector_params),
            "estimated_e2e": False,
        }
    )
    if lambda_value is not None:
        row["lambda"] = float(lambda_value)
    if online_threshold is not None:
        row["online_threshold"] = float(online_threshold)
    for key in list(row):
        if key.startswith("e2e_"):
            row[key] = None
    return row


def _online_threshold(
    method: str,
    budget: float,
    calibration_scores: np.ndarray,
    frozen_controls: dict[str, Any],
    tolerance: float,
) -> float:
    values = [
        float(frozen_controls[f"{method}|{policy}|online|{budget}"]["threshold"])
        for policy in ("Original", "SharedAction")
    ]
    observed = float(tri.current.online_threshold(calibration_scores, budget))
    if max(abs(observed - value) for value in values) > float(tolerance):
        raise RuntimeError(
            f"Frozen Online threshold drift for {method}/{budget}: "
            f"observed={observed}, expected={values}"
        )
    return observed


def _evaluate_policies(
    *,
    dataset: str,
    protocol: dict[str, Any],
    roles: dict[str, dict[str, Any]],
    scores: dict[str, dict[str, np.ndarray]],
    models: dict[str, Any],
    action_maps: dict[str, dict[str, dict[str, dict[int, str]]]],
    thresholds: dict[str, dict[str, Optional[float]]],
    policies: Iterable[str],
    frozen_controls: dict[str, Any],
    verify_frozen_lambdas: bool,
    parameter_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration, final = roles["calibration"], roles["final"]
    external_protocol = read_json(EXTERNAL_PROTOCOL_PATH)
    tolerance = float(protocol["control_reproduction_tolerance"])
    rows: list[dict[str, Any]] = []
    controls: dict[str, Any] = {}

    for method in RANKERS:
        for budget in map(float, protocol["work_points"]["fixed_call"]):
            selections = {}
            for policy in policies:
                selected = tri.current.top_k_actions(
                    final,
                    scores["final"][method],
                    action_maps["final"]["fixed_call"][policy],
                    budget,
                )
                selections[policy] = selected
                rows.append(
                    _row(
                        dataset=dataset,
                        method=method,
                        policy=policy,
                        mode="fixed_call",
                        budget=budget,
                        data=final,
                        selected=selected,
                        models=models,
                        threshold=thresholds["fixed_call"][policy],
                        selector_params=parameter_counts[method],
                    )
                )
            assert_same_membership(selections)

        for budget in map(float, protocol["work_points"]["online"]):
            frame_threshold = _online_threshold(
                method,
                budget,
                scores["calibration"][method],
                frozen_controls,
                tolerance,
            )
            selections = {}
            for policy in policies:
                action_map = action_maps["final"]["online"][policy]
                selected = {
                    int(sample["image_id"]): action_map[int(sample["image_id"])]
                    for sample, score in zip(
                        final["samples"], scores["final"][method]
                    )
                    if float(score) >= frame_threshold
                }
                selections[policy] = selected
                rows.append(
                    _row(
                        dataset=dataset,
                        method=method,
                        policy=policy,
                        mode="online",
                        budget=budget,
                        data=final,
                        selected=selected,
                        models=models,
                        threshold=thresholds["online"][policy],
                        online_threshold=frame_threshold,
                        selector_params=parameter_counts[method],
                    )
                )
            assert_same_membership(selections)

        for policy in policies:
            calibration_map = action_maps["calibration"]["fixed_byte"][policy]
            final_map = action_maps["final"]["fixed_byte"][policy]
            for budget in map(float, protocol["work_points"]["fixed_byte"]):
                gate = tri.current.calibrate_lambda(
                    calibration,
                    scores["calibration"][method],
                    calibration_map,
                    budget,
                    external_protocol,
                )
                if verify_frozen_lambdas:
                    expected = frozen_controls[
                        f"{method}|{policy}|fixed_byte|{budget}"
                    ]
                    for key in ("lambda", "median_original_bytes"):
                        if abs(float(gate[key]) - float(expected[key])) > tolerance:
                            raise RuntimeError(
                                f"Frozen Fixed-byte {key} drift for "
                                f"{method}/{policy}/{budget}"
                            )
                priorities = tri.current.audit_priority(
                    scores["calibration"][method],
                    scores["final"][method],
                    final,
                    final_map,
                    gate["lambda"],
                    gate["median_original_bytes"],
                )
                selected = tri.current.byte_selection(
                    final, priorities, final_map, budget
                )
                rows.append(
                    _row(
                        dataset=dataset,
                        method=method,
                        policy=policy,
                        mode="fixed_byte",
                        budget=budget,
                        data=final,
                        selected=selected,
                        models=models,
                        threshold=thresholds["fixed_byte"][policy],
                        lambda_value=float(gate["lambda"]),
                        selector_params=parameter_counts[method],
                    )
                )
                controls[f"{method}|{policy}|fixed_byte|{budget}"] = {
                    "lambda": float(gate["lambda"]),
                    "median_original_bytes": float(
                        gate["median_original_bytes"]
                    ),
                }
    return rows, controls


def run_evaluate(dataset: str, protocol: dict[str, Any]) -> dict[str, Any]:
    validate_protocol(protocol)
    paths = layout(dataset)
    output = assert_output_path(paths["output"], RUNS)
    preflight_completion = _require_complete(
        output / "preflight/completion.json", "Extension preflight"
    )
    before = capture_hashes(_required_source_paths(dataset))
    if _source_digest(before) != preflight_completion["source_digest"]:
        raise RuntimeError("Frozen sources changed after preflight")

    _configure_canonical(dataset, paths["common"])
    roles = {
        role: tri.current.load_role(dataset, role)
        for role in ("calibration", "final")
    }
    for role, data in roles.items():
        validate_role_data(
            data,
            dataset=dataset,
            role=role,
            expected_count=int(protocol["expected_role_sizes"][dataset][role]),
        )
    scores, parameter_counts = _load_frozen_scores(paths, roles)
    models = tri.load_frozen_models(paths["benchmark"] / "models.json")
    gate = read_json(paths["benchmark"] / "action_gate.json")
    shared_actions = {
        role: tri.current.chosen_actions(models["action_heads"], gate, data)
        for role, data in roles.items()
    }

    with (paths["triaction"] / "models/saferoi_gbdt.pkl").open("rb") as handle:
        roi_model = pickle.load(handle)
    roi_scores = {
        role: tri.predict_gbdt(roi_model, tri.profile_features(data))
        for role, data in roles.items()
    }
    new_thresholds = frozen_policy_thresholds(
        read_json(paths["triaction"] / "threshold_choices.json")
    )
    new_action_maps = {
        role: {
            mode: {
                policy: tri.action_map_at_threshold(
                    data,
                    shared_actions[role],
                    roi_scores[role],
                    threshold,
                )
                for policy, threshold in mode_thresholds.items()
            }
            for mode, mode_thresholds in new_thresholds.items()
        }
        for role, data in roles.items()
    }

    original_actions = {
        role: {
            int(sample["image_id"]): "original" for sample in data["samples"]
        }
        for role, data in roles.items()
    }
    control_thresholds = {
        mode: {"Original": None, "SharedAction": None} for mode in MODES
    }
    control_action_maps = {
        role: {
            mode: {
                "Original": original_actions[role],
                "SharedAction": shared_actions[role],
            }
            for mode in MODES
        }
        for role in roles
    }
    frozen_controls = read_json(
        paths["external"] / "evaluation/frozen_controls.json"
    )
    control_rows, _ = _evaluate_policies(
        dataset=dataset,
        protocol=protocol,
        roles=roles,
        scores=scores,
        models=models,
        action_maps=control_action_maps,
        thresholds=control_thresholds,
        policies=("Original", "SharedAction"),
        frozen_controls=frozen_controls,
        verify_frozen_lambdas=True,
        parameter_counts=parameter_counts,
    )
    reproduction = assert_control_reproduction(
        control_rows,
        read_csv(paths["external"] / "evaluation/results.csv"),
        tolerance=float(protocol["control_reproduction_tolerance"]),
    )

    rows, new_controls = _evaluate_policies(
        dataset=dataset,
        protocol=protocol,
        roles=roles,
        scores=scores,
        models=models,
        action_maps=new_action_maps,
        thresholds=new_thresholds,
        policies=POLICIES,
        frozen_controls=frozen_controls,
        verify_frozen_lambdas=False,
        parameter_counts=parameter_counts,
    )
    assert_result_matrix_complete(rows, protocol)
    assert_hashes_unchanged(before)

    result_path = output / "evaluation/results.csv"
    atomic_write_rows(result_path, rows, RUNS)
    atomic_write_json(
        output / "evaluation/frozen_controls.json",
        {
            "online_threshold_source": str(
                paths["external"] / "evaluation/frozen_controls.json"
            ),
            "roi_threshold_source": str(
                paths["triaction"] / "threshold_choices.json"
            ),
            "roi_thresholds": new_thresholds,
            "fixed_byte": new_controls,
        },
        RUNS,
    )
    report = {
        "dataset": dataset,
        "rows": len(rows),
        "rankers": list(RANKERS),
        "payload_policies": list(POLICIES),
        "control_reproduction": reproduction,
        "source_sha256": before,
        "source_digest": _source_digest(before),
        "gbdt_thresholds_reselected": False,
        "external_rankers_retrained": False,
        "final_used_for_selection": False,
        "oregon_result_status": (
            protocol["oregon_result_status"]
            if dataset == "oregon"
            else "standard_final"
        ),
    }
    atomic_write_json(output / "evaluation/report.json", report, RUNS)
    atomic_write_json(
        output / "evaluation/completion.json",
        {
            "dataset": dataset,
            "rows": len(rows),
            "results_sha256": file_sha256(result_path),
            "complete": True,
        },
        RUNS,
    )
    atomic_write_json(
        output / "completion.json",
        {
            "dataset": dataset,
            "protocol_id": protocol["protocol_id"],
            "rows": len(rows),
            "results_sha256": file_sha256(result_path),
            "complete": True,
        },
        RUNS,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("coco", "ena24", "oregon"))
    parser.add_argument(
        "--stage", choices=("preflight", "evaluate", "all"), default="all"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL_PATH)
    if args.stage in {"preflight", "all"}:
        print(json.dumps(run_preflight(args.dataset, protocol), indent=2))
    if args.stage in {"evaluate", "all"}:
        print(json.dumps(run_evaluate(args.dataset, protocol), indent=2))


if __name__ == "__main__":
    main()
