#!/usr/bin/env python3
"""Run one dataset for the isolated DCSB and ORIC/MORIC baselines."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

from baseline_core import (
    assert_result_matrix_complete,
    area_threshold_grid,
    build_output_feature,
    context_indices,
    detection_stats,
    dcsb_decision_and_score,
    empirical_cdf,
    fit_dcsb,
    fit_reward_mlp,
    oric_reward,
    predict_reward_mlp,
    validate_protocol,
)
from runner_core import (
    assert_hashes_unchanged,
    assert_output_path,
    atomic_write_json,
    atomic_write_npz,
    atomic_write_pickle,
    atomic_write_rows,
    capture_hashes,
    file_sha256,
    role_source_paths,
    roles_for_stage,
    validate_role_data,
)


SCRIPT = Path(__file__).resolve()
EXPERIMENT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
OLD_EXPERIMENT = (
    REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e"
)
RUNS = EXPERIMENT / "runs"
PROTOCOL_PATH = EXPERIMENT / "configs/protocol.json"


def layout_for_dataset(dataset: str) -> dict[str, Path]:
    if dataset not in {"coco", "ena24", "oregon"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    canonical = (
        "canonical_oregon_event_v1" if dataset == "oregon" else "canonical_v1"
    )
    root = OLD_EXPERIMENT / "runs" / canonical
    return {
        "common": root / "common",
        "benchmark": root / "benchmark" / dataset,
        "output": RUNS / dataset,
    }


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_code_paths() -> tuple[Path, ...]:
    scripts = OLD_EXPERIMENT / "scripts"
    rcmar = REPO / "paper/experiments/RCMAR_v4/rcmar"
    return (
        SCRIPT,
        SCRIPT.parent / "baseline_core.py",
        SCRIPT.parent / "runner_core.py",
        EXPERIMENT / "configs/literature_sources.json",
        scripts / "run_multiranker_benchmark.py",
        scripts / "benchmark_core.py",
        scripts / "protocol_core.py",
        rcmar / "__init__.py",
        rcmar / "coco_eval.py",
    )


def load_frozen_benchmark(common_root: Path) -> Any:
    sys.dont_write_bytecode = True
    old_scripts = str(OLD_EXPERIMENT / "scripts")
    rcmar_root = str(REPO / "paper/experiments/RCMAR_v4")
    for path in (old_scripts, rcmar_root):
        if path not in sys.path:
            sys.path.insert(0, path)
    benchmark = importlib.import_module("run_multiranker_benchmark")
    benchmark.COMMON = Path(common_root)
    return benchmark


def run_preflight(
    dataset: str,
    protocol: dict[str, Any],
    *,
    layout: dict[str, Path] | None = None,
    benchmark: Any | None = None,
    code_paths: tuple[Path, ...] | None = None,
    output_root: Path = RUNS,
) -> dict[str, Any]:
    validate_protocol(protocol)
    paths = layout or layout_for_dataset(dataset)
    common = Path(paths["common"])
    benchmark_root = Path(paths["benchmark"])
    output = assert_output_path(paths["output"], output_root)
    source_paths = [PROTOCOL_PATH]
    source_paths.extend(source_code_paths() if code_paths is None else code_paths)
    for role in roles_for_stage("preflight"):
        source_paths.extend(role_source_paths(common, dataset, role))
    frozen_model = benchmark_root / "models.json"
    frozen_gate = benchmark_root / "action_gate.json"
    source_paths.extend((frozen_model, frozen_gate))
    before = capture_hashes(source_paths)

    frozen_models = read_json(frozen_model)
    gate = read_json(frozen_gate)
    if set(frozen_models.get("action_heads", {})) != {"q90", "roi"}:
        raise RuntimeError("Frozen SharedAction heads are missing")
    if "chosen" not in gate:
        raise RuntimeError("Frozen SharedAction gate is missing")
    loader = benchmark or load_frozen_benchmark(common)
    loader.COMMON = common
    torch_version = "injected-test-loader"
    if benchmark is None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required for ORIC/MORIC MLP training") from error
        torch_version = str(torch.__version__)
    role_sizes = {}
    for role in roles_for_stage("preflight"):
        data = loader.load_role(dataset, role)
        expected = int(protocol["expected_role_sizes"][dataset][role])
        validate_role_data(data, dataset=dataset, role=role, expected_count=expected)
        role_sizes[role] = len(data["samples"])

    assert_hashes_unchanged(before)
    report = {
        "dataset": dataset,
        "protocol_id": protocol["protocol_id"],
        "role_sizes": role_sizes,
        "common_root": str(common.resolve()),
        "frozen_benchmark_root": str(benchmark_root.resolve()),
        "source_sha256": before,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch_version,
        "passed": True,
    }
    report_path = output / "preflight/report.json"
    atomic_write_json(report_path, report, output_root)
    atomic_write_json(
        output / "preflight/completion.json",
        {
            "dataset": dataset,
            "stage": "preflight",
            "report_sha256": file_sha256(report_path),
            "source_files": len(before),
            "complete": True,
        },
        output_root,
    )
    return report


def _source_digest(hashes: dict[str, str]) -> str:
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_complete(path: Path, stage: str) -> None:
    if not Path(path).is_file() or read_json(path).get("complete") is not True:
        raise RuntimeError(f"{stage} must complete first: {path}")


def _validate_reward_bundle(
    rewards_root: Path,
    *,
    expected_source_digest: str,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    completion_path = Path(rewards_root) / "completion.json"
    artifact_path = Path(rewards_root) / "fit_artifacts.npz"
    dcsb_path = Path(rewards_root) / "dcsb_model.json"
    _require_complete(completion_path, "Rewards")
    completion = read_json(completion_path)
    expected = {
        "source_digest": str(expected_source_digest),
        "protocol_sha256": file_sha256(protocol_path),
        "artifact_sha256": file_sha256(artifact_path),
        "dcsb_model_sha256": file_sha256(dcsb_path),
    }
    observed = {key: completion.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"Reward completion does not match frozen inputs: "
            f"expected={expected}, observed={observed}"
        )
    with np.load(artifact_path) as artifact:
        artifact_source_digest = str(artifact["source_digest"].item())
    if artifact_source_digest != expected_source_digest:
        raise RuntimeError("Reward artifact source digest does not match frozen inputs")
    return completion


def _reward_for_index(
    index: int,
    edge_stats: list[Any],
    cloud_stats: list[Any],
    *,
    context_size: int,
    seed: int,
) -> dict[str, float]:
    context = context_indices(
        len(edge_stats), index, context_size=context_size, seed=seed
    )
    return oric_reward(
        [edge_stats[other] for other in context], edge_stats[index], cloud_stats[index]
    )


def _load_reward_checkpoint(
    path: Path, source_digest: str, expected_image_ids: list[int]
) -> tuple[list[float], list[float], list[float]]:
    with np.load(path) as saved:
        saved_digest = str(saved["source_digest"].item())
        image_ids = list(map(int, saved["image_ids"]))
        rewards = list(map(float, saved["rewards"]))
        edge_maps = list(map(float, saved["edge_maps"]))
        cloud_maps = list(map(float, saved["cloud_maps"]))
    lengths = {len(image_ids), len(rewards), len(edge_maps), len(cloud_maps)}
    if (
        saved_digest != source_digest
        or len(lengths) != 1
        or image_ids != expected_image_ids[: len(image_ids)]
    ):
        raise RuntimeError("Reward checkpoint does not match the current frozen inputs")
    return rewards, edge_maps, cloud_maps


def run_rewards(
    dataset: str,
    protocol: dict[str, Any],
    *,
    layout: dict[str, Path] | None = None,
    benchmark: Any | None = None,
    code_paths: tuple[Path, ...] | None = None,
    output_root: Path = RUNS,
) -> dict[str, Any]:
    validate_protocol(protocol)
    paths = layout or layout_for_dataset(dataset)
    common = Path(paths["common"])
    output = assert_output_path(paths["output"], output_root)
    _require_complete(output / "preflight/completion.json", "Preflight")
    source_paths = [PROTOCOL_PATH]
    source_paths.extend(source_code_paths() if code_paths is None else code_paths)
    source_paths.extend(role_source_paths(common, dataset, "fit"))
    before = capture_hashes(source_paths)
    digest = _source_digest(before)

    loader = benchmark or load_frozen_benchmark(common)
    loader.COMMON = common
    fit = loader.load_role(dataset, "fit")
    validate_role_data(
        fit,
        dataset=dataset,
        role="fit",
        expected_count=int(protocol["expected_role_sizes"][dataset]["fit"]),
    )
    gt = read_json(fit["gt_path"])
    category_ids = [int(category["id"]) for category in gt["categories"]]
    image_ids = list(map(int, fit["image_ids"]))
    sizes = {
        int(sample["image_id"]): (int(sample["width"]), int(sample["height"]))
        for sample in fit["samples"]
    }
    ground_truth_counts = {
        image_id: len(fit["gt_map"].get(image_id, [])) for image_id in image_ids
    }
    dcsb = fit_dcsb(
        image_ids=image_ids,
        edge_predictions=fit["actions"]["edge"],
        cloud_predictions=fit["actions"]["original"],
        image_sizes=sizes,
        ground_truth_counts=ground_truth_counts,
        detector_confidence=float(protocol["dcsb"]["detector_confidence"]),
        count_thresholds=protocol["dcsb"]["count_thresholds"],
        area_thresholds=area_threshold_grid(
            protocol["dcsb"]["area_threshold_start"],
            protocol["dcsb"]["area_threshold_stop"],
            protocol["dcsb"]["area_threshold_step"],
        ),
    )
    features = np.asarray(
        [
            build_output_feature(
                fit["actions"]["edge"].get(image_id, []),
                width=sizes[image_id][0],
                height=sizes[image_id][1],
                category_ids=category_ids,
                top_k=int(protocol["oric"]["top_k"]),
            )
            for image_id in image_ids
        ]
    )
    thresholds = protocol["oric"]["context_metric_iou_thresholds"]
    edge_stats = [
        detection_stats(
            fit["actions"]["edge"].get(image_id, []),
            fit["gt_map"].get(image_id, []),
            thresholds,
        )
        for image_id in image_ids
    ]
    cloud_stats = [
        detection_stats(
            fit["actions"]["original"].get(image_id, []),
            fit["gt_map"].get(image_id, []),
            thresholds,
        )
        for image_id in image_ids
    ]

    rewards_root = output / "rewards"
    checkpoint = rewards_root / "reward_checkpoint.npz"
    rewards: list[float] = []
    edge_maps: list[float] = []
    cloud_maps: list[float] = []
    if checkpoint.is_file():
        rewards, edge_maps, cloud_maps = _load_reward_checkpoint(
            checkpoint, digest, image_ids
        )

    interval = int(protocol["oric"]["checkpoint_interval"])
    workers = max(1, int(protocol["oric"]["reward_workers"]))
    for start in range(len(rewards), len(image_ids), interval):
        indices = list(range(start, min(start + interval, len(image_ids))))
        with ThreadPoolExecutor(max_workers=min(workers, len(indices))) as executor:
            results = list(
                executor.map(
                    lambda index: _reward_for_index(
                        index,
                        edge_stats,
                        cloud_stats,
                        context_size=int(protocol["oric"]["context_size"]),
                        seed=int(protocol["oric"]["context_seed"]),
                    ),
                    indices,
                )
            )
        rewards.extend(float(result["reward"]) for result in results)
        edge_maps.extend(float(result["edge_map"]) for result in results)
        cloud_maps.extend(float(result["cloud_map"]) for result in results)
        atomic_write_npz(
            checkpoint,
            output_root,
            source_digest=np.asarray(digest),
            image_ids=np.asarray(image_ids[: len(rewards)], dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float64),
            edge_maps=np.asarray(edge_maps, dtype=np.float64),
            cloud_maps=np.asarray(cloud_maps, dtype=np.float64),
        )

    assert_hashes_unchanged(before)
    artifact = rewards_root / "fit_artifacts.npz"
    atomic_write_npz(
        artifact,
        output_root,
        source_digest=np.asarray(digest),
        image_ids=np.asarray(image_ids, dtype=np.int64),
        category_ids=np.asarray(category_ids, dtype=np.int64),
        features=features,
        rewards=np.asarray(rewards, dtype=np.float64),
        edge_maps=np.asarray(edge_maps, dtype=np.float64),
        cloud_maps=np.asarray(cloud_maps, dtype=np.float64),
    )
    atomic_write_json(rewards_root / "dcsb_model.json", dcsb, output_root)
    dcsb_path = rewards_root / "dcsb_model.json"
    report = {
        "dataset": dataset,
        "fit_images": len(image_ids),
        "feature_dim": int(features.shape[1]),
        "context_size": min(int(protocol["oric"]["context_size"]), len(image_ids) - 1),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "reward_mean": float(np.mean(rewards)),
        "source_sha256": before,
    }
    atomic_write_json(rewards_root / "report.json", report, output_root)
    atomic_write_json(
        rewards_root / "completion.json",
        {
            "dataset": dataset,
            "stage": "rewards",
            "source_digest": digest,
            "protocol_sha256": file_sha256(PROTOCOL_PATH),
            "artifact_sha256": file_sha256(artifact),
            "dcsb_model_sha256": file_sha256(dcsb_path),
            "fit_images": len(image_ids),
            "complete": True,
        },
        output_root,
    )
    return report


def _role_features(
    data: dict[str, Any], category_ids: list[int], top_k: int
) -> np.ndarray:
    return np.asarray(
        [
            build_output_feature(
                data["actions"]["edge"].get(int(sample["image_id"]), []),
                width=int(sample["width"]),
                height=int(sample["height"]),
                category_ids=category_ids,
                top_k=top_k,
            )
            for sample in data["samples"]
        ]
    )


def _evaluate_row(
    benchmark: Any,
    frozen_models: dict[str, Any],
    *,
    method: str,
    payload_policy: str,
    mode: str,
    nominal_budget: float,
    data: dict[str, Any],
    actions: dict[int, str],
) -> dict[str, Any]:
    row = benchmark.evaluate(
        method, mode, nominal_budget, data, actions, frozen_models
    )
    row.update(
        {
            "method": method,
            "payload_policy": payload_policy,
            "mode": mode,
            "nominal_budget": float(nominal_budget),
            "external_method_payload": payload_policy == "Original",
            "controlled_payload_diagnostic": payload_policy == "SharedAction",
        }
    )
    return row


def run_evaluate(
    dataset: str,
    protocol: dict[str, Any],
    *,
    layout: dict[str, Path] | None = None,
    benchmark: Any | None = None,
    code_paths: tuple[Path, ...] | None = None,
    output_root: Path = RUNS,
    fit_model: Any = fit_reward_mlp,
    predict_model: Any = predict_reward_mlp,
) -> dict[str, Any]:
    validate_protocol(protocol)
    paths = layout or layout_for_dataset(dataset)
    common = Path(paths["common"])
    benchmark_root = Path(paths["benchmark"])
    output = assert_output_path(paths["output"], output_root)
    _require_complete(output / "preflight/completion.json", "Preflight")
    reward_artifact = output / "rewards/fit_artifacts.npz"
    dcsb_path = output / "rewards/dcsb_model.json"
    frozen_model_path = benchmark_root / "models.json"
    frozen_gate_path = benchmark_root / "action_gate.json"
    implementation_paths = list(
        source_code_paths() if code_paths is None else code_paths
    )
    reward_source_paths = [PROTOCOL_PATH, *implementation_paths]
    reward_source_paths.extend(role_source_paths(common, dataset, "fit"))
    reward_source_hashes = capture_hashes(reward_source_paths)
    reward_source_digest = _source_digest(reward_source_hashes)
    _validate_reward_bundle(
        output / "rewards",
        expected_source_digest=reward_source_digest,
    )
    source_paths = [
        *reward_source_paths,
        reward_artifact,
        dcsb_path,
        output / "rewards/completion.json",
        frozen_model_path,
        frozen_gate_path,
    ]
    for role in roles_for_stage("evaluate"):
        source_paths.extend(role_source_paths(common, dataset, role))
    before = capture_hashes(source_paths)

    with np.load(reward_artifact) as fit_artifact:
        fit_features = np.asarray(fit_artifact["features"], dtype=np.float64)
        fit_rewards = np.asarray(fit_artifact["rewards"], dtype=np.float64)
        category_ids = list(map(int, fit_artifact["category_ids"]))
    expected_fit = int(protocol["expected_role_sizes"][dataset]["fit"])
    if fit_features.shape[0] != expected_fit or fit_rewards.shape != (expected_fit,):
        raise RuntimeError("Fit reward artifact does not match the frozen population")
    oric_checkpoint = fit_model(
        fit_features,
        fit_rewards,
        np.ones_like(fit_rewards),
        protocol["mlp"],
    )
    moric_targets = empirical_cdf(fit_rewards, fit_rewards)
    moric_checkpoint = fit_model(
        fit_features,
        moric_targets,
        moric_targets,
        protocol["mlp"],
    )

    loader = benchmark or load_frozen_benchmark(common)
    loader.COMMON = common
    roles = {}
    for role in roles_for_stage("evaluate"):
        data = loader.load_role(dataset, role)
        validate_role_data(
            data,
            dataset=dataset,
            role=role,
            expected_count=int(protocol["expected_role_sizes"][dataset][role]),
        )
        roles[role] = data
    calibration, final = roles["calibration"], roles["final"]
    frozen_models = read_json(frozen_model_path)
    frozen_gate = read_json(frozen_gate_path)
    dcsb_model = read_json(dcsb_path)
    features = {
        role: _role_features(
            data, category_ids, int(protocol["oric"]["top_k"])
        )
        for role, data in roles.items()
    }
    scores = {
        role: {
            "ORIC-MLP": predict_model(oric_checkpoint, features[role]),
            "MORIC-MLP": predict_model(moric_checkpoint, features[role]),
        }
        for role in roles
    }
    native_decisions: dict[str, np.ndarray] = {}
    for role, data in roles.items():
        decisions = []
        adapted = []
        for sample in data["samples"]:
            image_id = int(sample["image_id"])
            decision, score = dcsb_decision_and_score(
                data["actions"]["edge"].get(image_id, []),
                int(sample["width"]),
                int(sample["height"]),
                dcsb_model,
            )
            decisions.append(decision)
            adapted.append(score)
        native_decisions[role] = np.asarray(decisions, dtype=np.int64)
        scores[role]["DCSB-Score-Adapted"] = np.asarray(adapted, dtype=np.float64)

    action_maps = {
        role: {
            "Original": {
                int(sample["image_id"]): "original" for sample in data["samples"]
            },
            "SharedAction": loader.chosen_actions(
                frozen_models["action_heads"], frozen_gate, data
            ),
        }
        for role, data in roles.items()
    }
    rows: list[dict[str, Any]] = []
    rows.append(
        _evaluate_row(
            loader,
            frozen_models,
            method="Edge-only",
            payload_policy="system",
            mode="system",
            nominal_budget=0.0,
            data=final,
            actions={},
        )
    )
    rows.append(
        _evaluate_row(
            loader,
            frozen_models,
            method="Cloud-only Original",
            payload_policy="system",
            mode="system",
            nominal_budget=0.0,
            data=final,
            actions=action_maps["final"]["Original"],
        )
    )
    rows.append(
        _evaluate_row(
            loader,
            frozen_models,
            method="SharedAction all",
            payload_policy="system",
            mode="system",
            nominal_budget=0.0,
            data=final,
            actions=action_maps["final"]["SharedAction"],
        )
    )

    for payload in protocol["payload_policies"]:
        final_map = action_maps["final"][payload]
        native_actions = {
            int(sample["image_id"]): final_map[int(sample["image_id"])]
            for sample, decision in zip(final["samples"], native_decisions["final"])
            if int(decision) == 1
        }
        native_row = _evaluate_row(
            loader,
            frozen_models,
            method="DCSB-Native",
            payload_policy=payload,
            mode="native_rule",
            nominal_budget=-1.0,
            data=final,
            actions=native_actions,
        )
        native_row["natural_upload_rate"] = len(native_actions) / len(final["samples"])
        rows.append(native_row)

    frozen_controls: dict[str, Any] = {}
    for method in protocol["continuous_methods"]:
        calibration_scores = scores["calibration"][method]
        final_scores = scores["final"][method]
        for payload in protocol["payload_policies"]:
            calibration_map = action_maps["calibration"][payload]
            final_map = action_maps["final"][payload]
            for budget in protocol["work_points"]["fixed_call"]:
                selected = loader.top_k_actions(final, final_scores, final_map, budget)
                rows.append(
                    _evaluate_row(
                        loader,
                        frozen_models,
                        method=method,
                        payload_policy=payload,
                        mode="fixed_call",
                        nominal_budget=budget,
                        data=final,
                        actions=selected,
                    )
                )
            for budget in protocol["work_points"]["online"]:
                threshold = loader.online_threshold(calibration_scores, budget)
                selected = {
                    int(sample["image_id"]): final_map[int(sample["image_id"])]
                    for sample, score in zip(final["samples"], final_scores)
                    if float(score) >= threshold
                }
                row = _evaluate_row(
                    loader,
                    frozen_models,
                    method=method,
                    payload_policy=payload,
                    mode="online",
                    nominal_budget=budget,
                    data=final,
                    actions=selected,
                )
                row["online_threshold"] = float(threshold)
                rows.append(row)
                frozen_controls[f"{method}|{payload}|online|{budget}"] = {
                    "threshold": float(threshold)
                }
            for budget in protocol["work_points"]["fixed_byte"]:
                selected_gate = loader.calibrate_lambda(
                    calibration,
                    calibration_scores,
                    calibration_map,
                    budget,
                    protocol,
                )
                priorities = loader.audit_priority(
                    calibration_scores,
                    final_scores,
                    final,
                    final_map,
                    selected_gate["lambda"],
                    selected_gate["median_original_bytes"],
                )
                selected = loader.byte_selection(final, priorities, final_map, budget)
                row = _evaluate_row(
                    loader,
                    frozen_models,
                    method=method,
                    payload_policy=payload,
                    mode="fixed_byte",
                    nominal_budget=budget,
                    data=final,
                    actions=selected,
                )
                row["lambda"] = float(selected_gate["lambda"])
                rows.append(row)
                frozen_controls[f"{method}|{payload}|fixed_byte|{budget}"] = {
                    "lambda": float(selected_gate["lambda"]),
                    "median_original_bytes": float(
                        selected_gate["median_original_bytes"]
                    ),
                }

    assert_result_matrix_complete(rows, protocol)
    external_methods = {
        *protocol["continuous_methods"],
        *protocol["native_methods"],
    }
    parameter_counts = {}
    for method, checkpoint in (
        ("ORIC-MLP", oric_checkpoint),
        ("MORIC-MLP", moric_checkpoint),
    ):
        state = checkpoint.get("state_dict", {})
        parameter_counts[method] = (
            sum(int(value.numel()) for value in state.values()) if state else None
        )
    parameter_counts.update({"DCSB-Native": 3, "DCSB-Score-Adapted": 3})
    for row in rows:
        if row["method"] not in external_methods:
            continue
        row["selector_params"] = parameter_counts[row["method"]]
        row["selector_ms_per_image"] = None
        row["external_selector_runtime_measured"] = False
        row["estimated_e2e"] = False
        for key in list(row):
            if key.startswith("e2e_"):
                row[key] = None
    assert_hashes_unchanged(before)
    evaluation_root = output / "evaluation"
    results_path = evaluation_root / "results.csv"
    atomic_write_rows(results_path, rows, output_root)
    atomic_write_pickle(evaluation_root / "oric_mlp.pkl", oric_checkpoint, output_root)
    atomic_write_pickle(evaluation_root / "moric_mlp.pkl", moric_checkpoint, output_root)
    atomic_write_json(
        evaluation_root / "frozen_controls.json", frozen_controls, output_root
    )
    report = {
        "dataset": dataset,
        "rows": len(rows),
        "methods": [
            *protocol["continuous_methods"],
            *protocol["native_methods"],
        ],
        "payload_policies": protocol["payload_policies"],
        "source_sha256": before,
        "final_used_for_selection": False,
        "calibration_used_for_model_training": False,
        "oregon_result_status": (
            protocol["oregon_result_status"] if dataset == "oregon" else "standard_final"
        ),
    }
    atomic_write_json(evaluation_root / "report.json", report, output_root)
    atomic_write_json(
        evaluation_root / "completion.json",
        {
            "dataset": dataset,
            "stage": "evaluate",
            "results_sha256": file_sha256(results_path),
            "rows": len(rows),
            "complete": True,
        },
        output_root,
    )
    atomic_write_json(
        output / "completion.json",
        {
            "dataset": dataset,
            "protocol_id": protocol["protocol_id"],
            "rows": len(rows),
            "complete": True,
        },
        output_root,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("coco", "ena24", "oregon"))
    parser.add_argument(
        "--stage", required=True, choices=("preflight", "rewards", "evaluate", "all")
    )
    args = parser.parse_args()
    protocol = read_json(PROTOCOL_PATH)
    if args.stage == "preflight":
        print(json.dumps(run_preflight(args.dataset, protocol), indent=2))
        return
    if args.stage == "rewards":
        print(json.dumps(run_rewards(args.dataset, protocol), indent=2))
        return
    if args.stage == "evaluate":
        print(json.dumps(run_evaluate(args.dataset, protocol), indent=2))
        return
    if args.stage == "all":
        run_preflight(args.dataset, protocol)
        run_rewards(args.dataset, protocol)
        print(json.dumps(run_evaluate(args.dataset, protocol), indent=2))
        return


if __name__ == "__main__":
    main()
