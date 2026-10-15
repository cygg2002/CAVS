#!/usr/bin/env python3
"""Run one isolated dataset in the frozen nine-Ranker TriAction protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Iterable

import numpy as np


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
SOURCE_ROOT = REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e"
TRIACTION_ROOT = REPO / "paper/experiments/20260722_ena24_hierarchical_triaction_paper_exploration"
sys.path.insert(0, str(TRIACTION_ROOT / "scripts"))
import run_exploration as tri  # type: ignore  # noqa: E402

from evaluation_core import (  # noqa: E402
    CONTROLLED_RANKER,
    POLICIES,
    assert_bootstrap_rows_complete,
    assert_canonical_rank_targets,
    assert_compatible_controlled_features,
    assert_fixed_byte_gates_complete,
    eligible_roi_ids,
    expected_matrix_counts,
    pca32_matrix,
    validate_protocol,
    verify_reproduction_rows,
)


CANONICAL_RANKERS = tuple(tri.current.RANKERS)
REPRODUCTION_METRICS = tri.REPRODUCTION_METRICS
EXPECTED_ROLES = {
    "coco": {"fit": 9678, "calibration": 571, "final": 5000},
    "ena24": {"fit": 1400, "calibration": 400, "final": 1758},
    "oregon": {"fit": 9923, "calibration": 4961, "final": 4962},
}


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_file_sha256(paths: Iterable[Path]) -> dict[str, str]:
    normalized = sorted({Path(path).resolve() for path in paths})
    missing = [str(path) for path in normalized if not path.is_file()]
    if missing:
        raise FileNotFoundError("Frozen inputs are missing:\n" + "\n".join(missing))
    return {str(path): file_sha256(path) for path in normalized}


def canonical_root(dataset: str) -> Path:
    run = "canonical_oregon_event_v1" if dataset == "oregon" else "canonical_v1"
    return SOURCE_ROOT / "runs" / run


def source_paths(dataset: str) -> dict[str, Path]:
    root = canonical_root(dataset)
    common = root / "common" / dataset
    benchmark = root / "benchmark" / dataset
    return {
        "protocol": EXP_ROOT / "configs/protocol.json",
        "source_protocol": SOURCE_ROOT / "configs/protocol.json",
        "models": benchmark / "models.json",
        "action_gate": benchmark / "action_gate.json",
        "canonical_results": benchmark / "results.csv",
        "pca32": common / "pca32.json",
        **{
            f"{role}_manifest": common / role / "source_manifest.json"
            for role in ("fit", "calibration", "final")
        },
    }


def frozen_input_paths(dataset: str) -> list[Path]:
    paths = set(source_paths(dataset).values())
    common = canonical_root(dataset) / "common" / dataset
    for role in ("fit", "calibration", "final"):
        manifest_path = common / role / "source_manifest.json"
        manifest = read_json(manifest_path)
        paths.add(manifest_path)
        paths.add(Path(manifest["samples"]))
        paths.add(Path(manifest["gt"]))
        paths.update(Path(path) for path in manifest["action_sources"].values())
        paths.update(Path(path) for path in manifest["source_paths"])
    paths.update(
        {
            SCRIPT,
            EXP_ROOT / "scripts/run_pipeline.py",
            EXP_ROOT / "scripts/evaluation_core.py",
            EXP_ROOT / "configs/protocol.json",
            TRIACTION_ROOT / "scripts/exploration_core.py",
            TRIACTION_ROOT / "scripts/run_exploration.py",
            REPO / "paper/experiments/RCMAR_v4/rcmar/coco_eval.py",
        }
    )
    for name in (
        "audit_ena24_roi_protocol.py",
        "benchmark_core.py",
        "evaluate_historical_saferoi.py",
        "final_paper_evaluation_core.py",
        "joint_action_scheduler_core.py",
        "protocol_core.py",
        "run_final_paper_evaluation.py",
        "run_multiranker_benchmark.py",
        "saferoi_gbdt_core.py",
    ):
        paths.add(SOURCE_ROOT / "scripts" / name)
    return sorted(paths)


def progress(dataset: str, message: str) -> None:
    print(f"[{dataset}-nine-ranker-triaction] {message}", flush=True)


def configure_reused_runner(dataset: str, output: Path) -> None:
    """Bind the validated ENA24 engine to one isolated dataset/output root."""
    tri.EXP_ROOT = EXP_ROOT
    tri.OUTPUT = output
    tri.SCRIPT = SCRIPT
    tri.POLICIES = POLICIES
    tri.progress = lambda message: progress(dataset, message)
    tri.current.COMMON = canonical_root(dataset) / "common"
    tri.current.dataset_cost = tri.canonical_dataset_cost

    original_evaluate = getattr(tri.current, "_nine_ranker_original_evaluate", None)
    if original_evaluate is None:
        original_evaluate = tri.current.evaluate
        tri.current._nine_ranker_original_evaluate = original_evaluate

    def evaluate_with_controlled(
        method: str,
        mode: str,
        nominal_budget: float,
        data: dict[str, Any],
        uploaded_actions: dict[int, str],
        models: dict[str, Any],
    ) -> dict[str, Any]:
        evaluation_method = "SemanticRank" if method == CONTROLLED_RANKER else method
        row = original_evaluate(
            evaluation_method, mode, nominal_budget, data, uploaded_actions, models
        )
        if method == CONTROLLED_RANKER:
            action_params = sum(
                len(head["weights"]) + 1 for head in models["action_heads"].values()
            )
            row["method"] = CONTROLLED_RANKER
            row["selector_ms_per_image"] = 0.08
            row["selector_params"] = action_params + 32
        return row

    tri.current.evaluate = evaluate_with_controlled
    tri.eligible_roi_ids = eligible_roi_ids

    def add_policy_metadata(
        row: dict[str, Any], policy: str, threshold: float
    ) -> dict[str, Any]:
        row["action_policy"] = policy
        row["roi_threshold"] = float(threshold)
        row["calibration_scope"] = "shared_8_rankers"
        return row

    tri.add_policy_metadata = add_policy_metadata


def preflight(dataset: str, protocol: dict[str, Any]) -> dict[str, Any]:
    validate_protocol(protocol)
    paths = source_paths(dataset)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Frozen {dataset} inputs are missing:\n" + "\n".join(missing)
        )
    hashes = snapshot_file_sha256(frozen_input_paths(dataset))
    return {
        "passed": True,
        "dataset": dataset,
        "expected_roles": EXPECTED_ROLES[dataset],
        "source_paths": {name: str(path) for name, path in paths.items()},
        "source_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "frozen_input_sha256": hashes,
    }


def load_roles(dataset: str) -> dict[str, dict[str, Any]]:
    roles = {
        role: tri.current.load_role(dataset, role)
        for role in ("fit", "calibration", "final")
    }
    observed = {role: len(data["samples"]) for role, data in roles.items()}
    if observed != EXPECTED_ROLES[dataset]:
        raise RuntimeError(f"Role-size drift for {dataset}: {observed}")
    for data in roles.values():
        assert_compatible_controlled_features(data["samples"])
        assert_canonical_rank_targets(data["samples"])
    return roles


def fit_controlled_32d(
    fit: dict[str, Any], source_config: dict[str, Any], seed: int
) -> dict[str, Any]:
    hp = source_config["ranker_hyperparameters"]
    frozen = read_json(EXP_ROOT / "configs/protocol.json")["controlled_32d"]
    expected_hp = {
        "semantic_epochs": int(frozen["epochs"]),
        "semantic_learning_rate": float(frozen["learning_rate"]),
        "semantic_l2": float(frozen["l2"]),
        "semantic_max_pairs": int(frozen["max_pairs"]),
    }
    observed_hp = {name: hp[name] for name in expected_hp}
    if observed_hp != expected_hp:
        raise RuntimeError(
            f"Canonical SemanticRank hyperparameters drifted: {observed_hp}"
        )
    if int(seed) != int(hp["train_seed"]):
        raise RuntimeError("Controlled-32D seed differs from canonical SemanticRank")
    targets = np.asarray([sample["target"] for sample in fit["samples"]])
    return tri.current.fit_pairwise_ranker(
        pca32_matrix(fit["samples"]),
        targets,
        epochs=int(hp["semantic_epochs"]),
        learning_rate=float(hp["semantic_learning_rate"]),
        ridge=float(hp["semantic_l2"]),
        max_pairs=int(hp["semantic_max_pairs"]),
        seed=int(seed),
    )


def score_roles(
    roles: dict[str, dict[str, Any]],
    models: dict[str, Any],
    controlled_model: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    output = {}
    for role in ("calibration", "final"):
        scores = tri.current.ranker_scores(models, roles[role])
        scores[CONTROLLED_RANKER] = tri.current.predict_pairwise_ranker(
            controlled_model, pca32_matrix(roles[role]["samples"])
        )
        output[role] = scores
    return output


def verify_ena24_controlled_anchor(
    rows: list[dict[str, Any]], tolerance: float
) -> dict[str, Any]:
    expected = {
        0.1: 0.7215224378210991,
        0.25: 0.7302054163983293,
        0.4: 0.7355056391481648,
    }
    actual = {
        float(row["nominal_budget"]): float(row["official_map50_95"])
        for row in rows
        if row["method"] == CONTROLLED_RANKER
        and row["mode"] == "fixed_call"
        and row["action_policy"] == "SharedAction"
        and float(row["nominal_budget"]) in expected
    }
    if set(actual) != set(expected):
        raise RuntimeError(f"ENA24 Controlled-32D anchor keys differ: {actual}")
    worst = max(abs(actual[key] - expected[key]) for key in expected)
    if worst > float(tolerance):
        raise RuntimeError(f"ENA24 Controlled-32D reproduction drift: {worst}")
    return {"passed": True, "rows": len(actual), "worst_difference": worst}


def verify_matrix_counts(
    rows: list[dict[str, Any]],
    per_ranker_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, int]:
    expected = expected_matrix_counts(protocol)
    observed = {
        "main": len(rows),
        "per_ranker_ablation": len(per_ranker_rows),
        "random_seed_rows": len(random_rows),
        "bootstrap_comparisons": expected["bootstrap_comparisons"],
    }
    if observed != expected:
        raise RuntimeError(f"Output matrix size drift: {observed} != {expected}")
    unique = {
        (
            row["method"],
            row["mode"],
            float(row["nominal_budget"]),
            row["action_policy"],
        )
        for row in rows
    }
    if len(unique) != len(rows):
        raise RuntimeError("Main result matrix contains duplicate work points")
    return observed


def run_core(dataset: str, protocol: dict[str, Any], checked: dict[str, Any]) -> None:
    output = EXP_ROOT / "runs" / dataset
    configure_reused_runner(dataset, output)
    for marker in (output / "COMPLETED", output / "CORE_COMPLETED"):
        marker.unlink(missing_ok=True)
    before = checked["frozen_input_sha256"]
    source_config = read_json(SOURCE_ROOT / "configs/protocol.json")
    roles = load_roles(dataset)
    profile_verification = {
        role: tri.verify_roi_profile(
            data,
            expected_protocol="merged_historical_cost_q90_v1",
            expected_profile="historical_cost",
        )
        for role, data in roles.items()
    }
    models = tri.load_frozen_models(checked["source_paths"]["models"])
    gate = read_json(checked["source_paths"]["action_gate"])
    base_actions = {
        role: tri.current.chosen_actions(models["action_heads"], gate, data)
        for role, data in roles.items()
    }
    controlled_model = fit_controlled_32d(
        roles["fit"], source_config, int(protocol["controlled_32d_seed"])
    )
    scores = score_roles(roles, models, controlled_model)

    epsilon = float(source_config["context_m2e"]["epsilon"])
    fit_labels = tri.roi_labels(roles["fit"], base_actions["fit"], epsilon)
    roi_model = tri.fit_gbdt(
        tri.profile_features(roles["fit"]),
        fit_labels,
        seed=int(protocol["gbdt_seed"]),
    )
    expected_gbdt = {
        "learning_rate": 0.05,
        "max_iter": 150,
        "max_depth": 3,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": int(protocol["gbdt_seed"]),
    }
    observed_gbdt = {
        name: roi_model.get_params()[name] for name in expected_gbdt
    }
    if observed_gbdt != expected_gbdt:
        raise RuntimeError(f"SafeROI GBDT parameters drifted: {observed_gbdt}")
    roi_scores = {
        role: tri.predict_gbdt(roi_model, tri.profile_features(data))
        for role, data in roles.items()
    }
    for role, data in roles.items():
        shape = tri.profile_features(data).shape
        if shape != (len(data["samples"]), int(protocol["saferoi_gbdt"]["feature_dim"])):
            raise RuntimeError(f"SafeROI feature shape drift for {role}: {shape}")
    score_by_id = {
        int(sample["image_id"]): float(score)
        for sample, score in zip(
            roles["calibration"]["samples"], roi_scores["calibration"]
        )
    }
    thresholds = tri.thresholds_for_fractions(
        score_by_id,
        tri.eligible_roi_ids(roles["calibration"], base_actions["calibration"]),
        map(float, protocol["threshold_fractions"]),
    )
    if math.inf not in thresholds or -math.inf not in thresholds:
        raise RuntimeError("ROI candidate grid lacks disabled/all-eligible controls")

    rankers = tuple(protocol["constrained_rankers"])
    tolerance = float(protocol["ap_tolerance"])
    fixed_call = tri.evaluate_fixed_call_candidates(
        roles["calibration"],
        scores["calibration"],
        base_actions["calibration"],
        roi_scores["calibration"],
        thresholds,
        rankers,
        map(float, protocol["primary_call_budgets"]),
    )
    progress(dataset, "Fixed-call calibration complete")
    online, frame_thresholds = tri.evaluate_online_candidates(
        roles["calibration"],
        scores["calibration"],
        base_actions["calibration"],
        roi_scores["calibration"],
        thresholds,
        rankers,
        map(float, protocol["online_targets"]),
    )
    progress(dataset, "Online calibration complete")
    fixed_byte = tri.evaluate_fixed_byte_candidates(
        roles["calibration"],
        scores["calibration"],
        base_actions["calibration"],
        roi_scores["calibration"],
        thresholds,
        rankers,
        map(float, protocol["fixed_byte_budgets"]),
        source_config,
    )
    for candidate in fixed_byte:
        assert_fixed_byte_gates_complete(
            candidate["points"], rankers, protocol["fixed_byte_budgets"]
        )
    progress(dataset, "Fixed-byte calibration complete")

    choices = {
        "fixed_call": tri.choose_cost_candidate(fixed_call, ap_tolerance=tolerance),
        "online": tri.choose_cost_candidate(online, ap_tolerance=tolerance),
        "fixed_byte": tri.choose_fixed_byte_candidate(
            fixed_byte, ap_tolerance=tolerance
        ),
    }
    per_ranker = {
        "fixed_call": tri.choose_per_ranker_candidates(
            fixed_call, rankers=rankers, ap_tolerance=tolerance, mode="cost"
        ),
        "online": tri.choose_per_ranker_candidates(
            online, rankers=rankers, ap_tolerance=tolerance, mode="cost"
        ),
        "fixed_byte": tri.choose_per_ranker_candidates(
            fixed_byte,
            rankers=rankers,
            ap_tolerance=tolerance,
            mode="fixed_byte",
        ),
    }
    write_json(output / "threshold_choices.json", {"shared": choices, "per_ranker": per_ranker})
    write_csv(
        output / "calibration_curve.csv",
        [
            *tri.flatten_candidates("fixed_call", fixed_call, tolerance),
            *tri.flatten_candidates("online", online, tolerance),
            *tri.flatten_candidates("fixed_byte", fixed_byte, tolerance),
        ],
    )
    (output / "models").mkdir(parents=True, exist_ok=True)
    write_json(output / "models/current_controlled_32d.json", controlled_model)
    with (output / "models/saferoi_gbdt.pkl").open("wb") as handle:
        pickle.dump(roi_model, handle)

    rows, random_rows, selections = tri.evaluate_final_matrix(
        roles["final"],
        scores,
        base_actions,
        roi_scores,
        choices,
        frame_thresholds,
        fixed_byte,
        models,
        protocol,
    )
    reproduction = verify_reproduction_rows(
        rows,
        read_csv(Path(checked["source_paths"]["canonical_results"])),
        canonical_rankers=CANONICAL_RANKERS,
        metrics=REPRODUCTION_METRICS,
        tolerance=float(protocol["reproduction_tolerance"]),
    )
    controlled_anchor = (
        verify_ena24_controlled_anchor(
            rows, float(protocol["reproduction_tolerance"])
        )
        if dataset == "ena24"
        else None
    )
    progress(dataset, "Canonical SharedAction reproduction passed")
    per_ranker_rows = tri.evaluate_per_ranker_ablation(
        roles["final"],
        scores,
        base_actions,
        roi_scores,
        per_ranker,
        frame_thresholds,
        models,
        protocol,
    )
    counts = verify_matrix_counts(rows, per_ranker_rows, random_rows, protocol)
    write_csv(output / "results.csv", rows)
    write_csv(output / "random_seeds.csv", random_rows)
    write_csv(output / "per_ranker_results.csv", per_ranker_rows)
    write_json(output / "selections.json", selections)
    canonical_rows = read_csv(Path(checked["source_paths"]["canonical_results"]))
    write_csv(
        output / "canonical_system_reference.csv",
        [
            row
            for row in canonical_rows
            if row.get("role") == "final" and row.get("mode") == "system"
        ],
    )

    after = snapshot_file_sha256(frozen_input_paths(dataset))
    if before != after:
        changed = sorted(
            path
            for path in set(before) | set(after)
            if before.get(path) != after.get(path)
        )
        raise RuntimeError("A frozen input changed during core:\n" + "\n".join(changed))
    summary = {
        "protocol_id": protocol["protocol_id"],
        "dataset": dataset,
        "status": "core_complete",
        "output_role": protocol["output_role"],
        "primary_ninth_ranker": CONTROLLED_RANKER,
        "legacy_in_primary_matrix": False,
        "final_used_for_selection": False,
        "canonical_experiment_replaced": False,
        "source_experiments_modified": False,
        "oregon_result_status": (
            protocol["oregon_result_status"] if dataset == "oregon" else None
        ),
        "canonical_reproduction": reproduction,
        "ena24_controlled_32d_reproduction": controlled_anchor,
        "frozen_shared_action_gate": {
            "chosen": gate.get("chosen"),
            "audit_attempts": gate.get("audit_attempts"),
        },
        "roi_profile_verification": profile_verification,
        "threshold_choices": choices,
        "per_ranker_threshold_choices": per_ranker,
        "controlled_32d": {
            "fit_role": "fit",
            "input_dim": 32,
            "target": "canonical max frozen-action gain",
            "parameters": len(controlled_model["weights"]),
        },
        "gbdt_metrics": {
            "fit": tri.binary_ranking_metrics(fit_labels, roi_scores["fit"]),
            "calibration": tri.binary_ranking_metrics(
                tri.roi_labels(
                    roles["calibration"], base_actions["calibration"], epsilon
                ),
                roi_scores["calibration"],
            ),
            "final_diagnostic_only": tri.binary_ranking_metrics(
                tri.roi_labels(roles["final"], base_actions["final"], epsilon),
                roi_scores["final"],
            ),
        },
        "gbdt_parameters": observed_gbdt,
        "gbdt_inference_cost_added_to_e2e": False,
        "row_counts": counts,
    }
    provenance = {
        "protocol": protocol,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "scikit_learn_version": importlib.metadata.version("scikit-learn"),
            "faster_coco_eval_version": importlib.metadata.version("faster-coco-eval"),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
        },
        "preflight": checked,
        "frozen_input_sha256_before": before,
        "frozen_input_sha256_after": after,
        "final_used_for_selection": False,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "provenance.json", provenance)
    (output / "CORE_COMPLETED").write_text("ok\n", encoding="ascii")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))


def run_bootstrap_stage(
    dataset: str, protocol: dict[str, Any], checked: dict[str, Any]
) -> None:
    output = EXP_ROOT / "runs" / dataset
    if not (output / "CORE_COMPLETED").is_file():
        raise RuntimeError(f"{dataset} core is not complete")
    configure_reused_runner(dataset, output)
    final = tri.current.load_role(dataset, "final")
    selections = read_json(output / "selections.json")
    checkpoint_path = output / "bootstrap_checkpoint.json"
    identity = {
        "protocol_sha256": file_sha256(EXP_ROOT / "configs/protocol.json"),
        "selections_sha256": file_sha256(output / "selections.json"),
        "dataset": dataset,
    }
    if checkpoint_path.is_file():
        checkpoint = read_json(checkpoint_path)
        if {key: checkpoint.get(key) for key in identity} != identity:
            raise RuntimeError("Bootstrap checkpoint does not match frozen inputs")
        rows = list(checkpoint.get("rows", []))
    else:
        rows = []
    completed = {
        (str(row["method"]), float(row["nominal_budget"])) for row in rows
    }
    comparison = str(protocol["bootstrap_comparison"])
    replicates = int(protocol["bootstrap_replicates"])
    seed = int(protocol["bootstrap_seed_base"])
    cache = {
        str(row["action_pair_sha256"]): dict(row)
        for row in rows
        if row.get("action_pair_sha256")
    }
    for ranker in protocol["constrained_rankers"]:
        for budget in map(float, protocol["primary_call_budgets"]):
            key = (str(ranker), budget)
            if key in completed:
                continue
            baseline = {
                int(image_id): action
                for image_id, action in selections[
                    f"fixed_call|SharedAction|{ranker}|{budget:.2f}"
                ].items()
            }
            candidate = {
                int(image_id): action
                for image_id, action in selections[
                    f"fixed_call|GBDT-Hierarchical|{ranker}|{budget:.2f}"
                ].items()
            }
            digest = tri.action_pair_digest(baseline, candidate)
            if baseline == candidate:
                result = {
                    "replicates": replicates,
                    "official_map50_95_delta_mean": 0.0,
                    "official_map50_95_delta_ci_low": 0.0,
                    "official_map50_95_delta_ci_high": 0.0,
                    "official_map50_95_probability_gt_zero": 0.0,
                    "request_kb_per_image_delta_mean": 0.0,
                    "request_kb_per_image_delta_ci_low": 0.0,
                    "request_kb_per_image_delta_ci_high": 0.0,
                    "bootstrap_unit": protocol["bootstrap_unit"],
                    "paired": True,
                    "strategy_refit_in_bootstrap": False,
                }
            elif digest in cache:
                result = dict(cache[digest])
                result["reused_identical_action_pair"] = True
            else:
                result = tri.paired_bootstrap(
                    final,
                    baseline_actions=baseline,
                    candidate_actions=candidate,
                    replicates=replicates,
                    seed=seed,
                    unit=str(protocol["bootstrap_unit"]),
                )
                cache[digest] = dict(result)
            result.update(
                {
                    "comparison": comparison,
                    "method": str(ranker),
                    "mode": "fixed_call",
                    "nominal_budget": budget,
                    "action_pair_sha256": digest,
                    "bootstrap_seed": seed,
                }
            )
            rows.append(result)
            completed.add(key)
            write_json(checkpoint_path, {**identity, "rows": rows})
            write_csv(output / "paired_bootstrap.csv", rows)
            progress(dataset, f"bootstrap {ranker} budget={budget:.2f} complete")
    assert_bootstrap_rows_complete(
        rows,
        rankers=protocol["constrained_rankers"],
        budgets=protocol["primary_call_budgets"],
        comparison=comparison,
        replicates=replicates,
    )
    after = snapshot_file_sha256(frozen_input_paths(dataset))
    if checked["frozen_input_sha256"] != after:
        raise RuntimeError("A frozen input changed during bootstrap")
    summary = read_json(output / "summary.json")
    summary["status"] = "complete"
    summary["bootstrap_comparisons"] = len(rows)
    summary["bootstrap_replicates"] = replicates
    summary["bootstrap_seed"] = seed
    write_json(output / "summary.json", summary)
    (output / "COMPLETED").write_text("ok\n", encoding="ascii")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(EXPECTED_ROLES), required=True)
    parser.add_argument("--stage", choices=("preflight", "core", "bootstrap"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = read_json(EXP_ROOT / "configs/protocol.json")
    configure_reused_runner(args.dataset, EXP_ROOT / "runs" / args.dataset)
    checked = preflight(args.dataset, protocol)
    progress(args.dataset, "preflight passed")
    if args.stage == "preflight":
        print(json.dumps(checked, indent=2, ensure_ascii=False))
    elif args.stage == "core":
        run_core(args.dataset, protocol, checked)
    else:
        run_bootstrap_stage(args.dataset, protocol, checked)


if __name__ == "__main__":
    main()
