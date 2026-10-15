#!/usr/bin/env python3
"""Run the frozen canonical COCO No-ROI full-ranker comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import platform
import sys
from typing import Any, Mapping

import numpy as np

from noroi_core import (
    BUDGETS,
    EXTERNAL_METHODS,
    HISTORICAL_METHODS,
    INTERNAL_METHODS,
    METHODS,
    POLICIES,
    assert_fixed_byte_capacity,
    assert_no_roi_actions,
    assert_result_matrix_complete,
    assert_result_rows_no_roi,
    assert_score_alignment,
    build_no_roi_action_maps,
    freeze_strongest,
    validate_protocol,
)


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL_PATH = ROOT / "configs/protocol.json"
_POINT_STATE: dict[str, Any] = {}


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


PROTOCOL = read_json(PROTOCOL_PATH)
validate_protocol(PROTOCOL)
SOURCES = {
    name: REPO / path for name, path in PROTOCOL["source_experiments"].items()
}
RANKER_ROOT = SOURCES["canonical_rankers"]
CANONICAL_ROOT = SOURCES["canonical_actions"]
HISTORICAL_ROOT = SOURCES["historical_rankers"]
EXTERNAL_ROOT = SOURCES["external_scheduler"]
COMMON = CANONICAL_ROOT / "runs/canonical_v1/common/coco"
BENCHMARK = CANONICAL_ROOT / "runs/canonical_v1/benchmark/coco"
HISTORICAL_RUN = HISTORICAL_ROOT / "runs/coco"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(RANKER_ROOT / "scripts"))
SOURCE_RUNNER = _load_module(
    "canonical9678_noroi_source_runner", RANKER_ROOT / "scripts/run_paper.py"
)
sys.path.insert(
    0,
    str(
        REPO
        / "paper/experiments/20260722_ena24_hierarchical_triaction_paper_exploration/scripts"
    ),
)
import run_exploration as tri  # type: ignore  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fields = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _require_complete(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} completion is missing: {path}")
    value = read_json(path)
    if value.get("complete") is not True:
        raise RuntimeError(f"{label} is incomplete: {path}")
    return value


def _configure_actions() -> None:
    tri.current.COMMON = COMMON.parent
    tri.current.dataset_cost = tri.canonical_dataset_cost
    if Path(tri.current.COMMON) / "coco" != COMMON:
        raise RuntimeError("Canonical COCO action-cache binding failed")


def _action_role(role: str) -> dict[str, Any]:
    _configure_actions()
    data = tri.current.load_role("coco", role)
    key = "registered_final" if role == "final" else role
    if len(data["samples"]) != int(PROTOCOL["roles"][key]):
        raise RuntimeError(f"Canonical {role} population changed")
    return data


def _manifest_files(role: str) -> list[Path]:
    manifest_path = COMMON / role / "source_manifest.json"
    manifest = read_json(manifest_path)
    paths = [manifest_path, COMMON / role / "samples.json", COMMON / role / "gt.json"]
    for action in ("edge", "original", "q90"):
        source = Path(manifest.get("action_sources", {}).get(action, ""))
        if not source.is_file():
            raise RuntimeError(f"Missing canonical {role}/{action} cache: {source}")
        expected = manifest.get("source_paths", {}).get(str(source))
        if expected is None or file_sha256(source) != str(expected):
            raise RuntimeError(f"Canonical {role}/{action} cache hash changed")
        paths.append(source)
    return paths


def required_source_files() -> list[Path]:
    paths = {
        PROTOCOL_PATH,
        SCRIPT,
        ROOT / "scripts/noroi_core.py",
        RANKER_ROOT / "configs/protocol.json",
        RANKER_ROOT / "scripts/run_paper.py",
        RANKER_ROOT / "scripts/canonical_core.py",
        RANKER_ROOT / "runs/train/checkpoints.pkl",
        RANKER_ROOT / "runs/train/completion.json",
        RANKER_ROOT / "runs/train/report.json",
        BENCHMARK / "models.json",
        BENCHMARK / "action_gate.json",
        HISTORICAL_RUN / "models/current_controlled_32d.json",
        HISTORICAL_RUN / "selections.json",
        HISTORICAL_RUN / "COMPLETED",
        HISTORICAL_ROOT / "configs/protocol.json",
        EXTERNAL_ROOT / "configs/protocol.json",
        CANONICAL_ROOT / "scripts/run_multiranker_benchmark.py",
        CANONICAL_ROOT / "scripts/benchmark_core.py",
        REPO
        / "paper/experiments/20260722_ena24_hierarchical_triaction_paper_exploration/scripts/run_exploration.py",
    }
    for role in ("calibration", "final"):
        paths.update(_manifest_files(role))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen sources:\n" + "\n".join(sorted(missing)))
    return sorted(path.resolve() for path in paths)


def capture_source_hashes() -> dict[str, str]:
    return {str(path): file_sha256(path) for path in required_source_files()}


def _source_digest(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(dict(hashes), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _verify_preflight_hashes() -> dict[str, str]:
    report = read_json(RUNS / "preflight/report.json")
    current = capture_source_hashes()
    if current != report["source_sha256"]:
        raise RuntimeError("Frozen source hashes changed after preflight")
    return current


def _policy_maps(data: dict[str, Any], models: dict[str, Any], gate: Mapping[str, Any]) -> dict[str, dict[int, str]]:
    q90_probability, _ = tri.current.action_probabilities(models["action_heads"], data)
    maps = build_no_roi_action_maps(
        data["samples"],
        q90_probability,
        q90_threshold=float(gate["chosen"]["q90_threshold"]),
    )
    for actions in maps.values():
        assert_no_roi_actions(actions)
    return maps


def run_preflight() -> dict[str, Any]:
    completion_path = RUNS / "preflight/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "preflight/report.json")
    train = _require_complete(
        RANKER_ROOT / "runs/train/completion.json", "Canonical ranker training"
    )
    train_report = read_json(RANKER_ROOT / "runs/train/report.json")
    checkpoint = RANKER_ROOT / "runs/train/checkpoints.pkl"
    if int(train_report.get("development_images", -1)) != 9678:
        raise RuntimeError("Canonical rankers were not trained on fit=9,678")
    if train.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise RuntimeError("Canonical ranker checkpoint hash changed")
    roles = {role: _action_role(role) for role in ("calibration", "final")}
    models = tri.load_frozen_models(BENCHMARK / "models.json")
    gate = read_json(BENCHMARK / "action_gate.json")
    action_counts = {}
    for role, data in roles.items():
        maps = _policy_maps(data, models, gate)
        action_counts[role] = {
            policy: {
                action: list(actions.values()).count(action)
                for action in ("original", "q90", "roi")
            }
            for policy, actions in maps.items()
        }
    hashes = capture_source_hashes()
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "passed": True,
        "result_status": PROTOCOL["result_status"],
        "role_counts": {role: len(data["samples"]) for role, data in roles.items()},
        "ranker_training_images": 9678,
        "methods": list(METHODS),
        "payload_policies": list(POLICIES),
        "action_counts": action_counts,
        "source_sha256": hashes,
        "source_digest": _source_digest(hashes),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rankers_retrained": False,
        "actions_retrained": False,
        "registered_final_used_for_selection": False,
    }
    atomic_json(RUNS / "preflight/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "report_sha256": file_sha256(RUNS / "preflight/report.json"),
            "source_digest": report["source_digest"],
        },
    )
    return report


def _verify_checkpoint_fingerprint(artifact: Mapping[str, Any], source_protocol: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    stored = dict(artifact.get("source_fingerprint", {}))
    current = SOURCE_RUNNER._source_fingerprint(source_protocol)
    if set(stored) != set(current):
        raise RuntimeError("Canonical checkpoint fingerprint keys changed")
    differences = {
        key: {"stored": str(stored[key]), "current": str(current[key])}
        for key in stored
        if str(stored[key]) != str(current[key])
    }
    if not set(differences).issubset({"canonical_core", "canonical_runner"}):
        raise RuntimeError(f"Canonical checkpoint scoring inputs changed: {differences}")
    return differences


def _historical_scores(data: dict[str, Any], models: dict[str, Any], controlled: Mapping[str, Any]) -> dict[str, np.ndarray]:
    values = tri.current.ranker_scores(models, data)
    values["Random"] = tri.current.random_scores(
        data["samples"], int(PROTOCOL["random_seed"])
    )
    pca32 = np.asarray([sample["pca32"] for sample in data["samples"]], dtype=np.float64)
    if pca32.shape != (len(data["samples"]), 32):
        raise RuntimeError(f"Historical controlled feature shape changed: {pca32.shape}")
    values["Current-Controlled-32D"] = tri.current.predict_pairwise_ranker(
        controlled, pca32
    )
    if set(values) != set(HISTORICAL_METHODS):
        raise RuntimeError(f"Historical score matrix changed: {sorted(values)}")
    return values


def _verify_historical_fixed_call_anchor(data: dict[str, Any], scores: Mapping[str, np.ndarray]) -> None:
    stored = read_json(HISTORICAL_RUN / "selections.json")
    for method in HISTORICAL_METHODS:
        if method == "Random":
            continue
        for budget in BUDGETS:
            selected = tri.current.top_k_actions(
                data,
                scores[method],
                {int(sample["image_id"]): "original" for sample in data["samples"]},
                budget,
            )
            key = f"fixed_call|SharedAction|{method}|{budget:.2f}"
            if set(selected) != set(map(int, stored[key])):
                raise RuntimeError(f"Historical fixed-call anchor differs: {key}")


def run_scores() -> dict[str, Any]:
    completion_path = RUNS / "scores/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "scores/report.json")
    _verify_preflight_hashes()
    source_protocol = SOURCE_RUNNER._protocol()
    checkpoint = RANKER_ROOT / "runs/train/checkpoints.pkl"
    with checkpoint.open("rb") as handle:
        artifact = pickle.load(handle)
    fingerprint_differences = _verify_checkpoint_fingerprint(artifact, source_protocol)
    historical_models = tri.load_frozen_models(BENCHMARK / "models.json")
    controlled = read_json(HISTORICAL_RUN / "models/current_controlled_32d.json")
    arrays: dict[str, np.ndarray] = {"methods": np.asarray(METHODS, dtype="U40")}
    role_counts = {}
    for role, canonical_alias in (("calibration", "calibration"), ("final", "registered_final")):
        data = _action_role(role)
        records = SOURCE_RUNNER._records_for_role(canonical_alias, source_protocol)
        score_ids = [int(record["image_id"]) for record in records]
        action_ids = [int(sample["image_id"]) for sample in data["samples"]]
        canonical = SOURCE_RUNNER.LEGACY._method_scores(records, source_protocol, artifact)
        historical = _historical_scores(data, historical_models, controlled)
        if role == "final":
            _verify_historical_fixed_call_anchor(data, historical)
        rows = []
        for method in METHODS:
            if method in historical:
                vector = np.asarray(historical[method], dtype=np.float64)
                rows.append(np.repeat(vector[None, :], len(BUDGETS), axis=0))
            else:
                rows.append(np.asarray(canonical[method], dtype=np.float64))
        matrix = np.stack(rows)
        assert_score_alignment(score_ids, action_ids, matrix)
        arrays[f"{role}_ids"] = np.asarray(action_ids, dtype=np.int64)
        arrays[f"{role}_scores"] = matrix
        role_counts[role] = len(action_ids)
    score_path = RUNS / "scores/frozen_scores.npz"
    atomic_npz(score_path, **arrays)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "methods": list(METHODS),
        "role_counts": role_counts,
        "score_shapes": {
            role: list(arrays[f"{role}_scores"].shape)
            for role in ("calibration", "final")
        },
        "canonical_checkpoint_sha256": file_sha256(checkpoint),
        "canonical_fingerprint_differences": fingerprint_differences,
        "historical_final_fixed_call_anchor_reproduced": True,
        "scores_sha256": file_sha256(score_path),
        "rankers_retrained": False,
        "registered_final_used_for_selection": False,
    }
    atomic_json(RUNS / "scores/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "scores_sha256": report["scores_sha256"],
            "report_sha256": file_sha256(RUNS / "scores/report.json"),
        },
    )
    return report


def _load_scores() -> dict[str, dict[str, np.ndarray]]:
    completion = _require_complete(RUNS / "scores/completion.json", "Score export")
    path = RUNS / "scores/frozen_scores.npz"
    if completion.get("scores_sha256") != file_sha256(path):
        raise RuntimeError("Frozen score cache changed")
    with np.load(path, allow_pickle=False) as value:
        methods = tuple(map(str, value["methods"].tolist()))
        if methods != METHODS:
            raise RuntimeError("Frozen score method ordering changed")
        return {
            role: {
                method: np.asarray(value[f"{role}_scores"][index], dtype=np.float64)
                for index, method in enumerate(METHODS)
            }
            for role in ("calibration", "final")
        }


def _evaluate_row(
    *,
    role: str,
    method: str,
    policy: str,
    mode: str,
    budget: float,
    data: dict[str, Any],
    selected: dict[int, str],
    models: dict[str, Any],
    lambda_value: float | None = None,
    capacity_bytes: float | None = None,
) -> dict[str, Any]:
    assert_no_roi_actions(selected)
    row = tri.current.evaluate(method, mode, budget, data, selected, models)
    row.update(
        {
            "role": role,
            "method": method,
            "payload_policy": policy,
            "action_policy": policy,
            "mode": mode,
            "nominal_budget": float(budget),
            "roi_disabled": True,
            "ranker_checkpoint_frozen": True,
            "selector_ms_per_image": None,
            "selector_params": None,
            "estimated_e2e": False,
        }
    )
    if lambda_value is not None:
        row["lambda"] = float(lambda_value)
    if capacity_bytes is not None:
        row["capacity_bytes"] = float(capacity_bytes)
        row["used_capacity_bytes"] = sum(
            float(data["sample_by_id"][image_id]["payload_bytes"][action])
            for image_id, action in selected.items()
        )
    for key in list(row):
        if key.startswith("e2e_"):
            row[key] = None
    assert_result_rows_no_roi([row])
    return row


def _point_worker(task: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if not _POINT_STATE:
        raise RuntimeError("Point worker state is not configured")
    role = str(task["role"])
    row = _evaluate_row(
        role=role,
        method=str(task["method"]),
        policy=str(task["policy"]),
        mode=str(task["mode"]),
        budget=float(task["budget"]),
        data=_POINT_STATE["roles"][role],
        selected={int(key): str(value) for key, value in task["selected"].items()},
        models=_POINT_STATE["models"],
        lambda_value=task.get("lambda_value"),
        capacity_bytes=task.get("capacity_bytes"),
    )
    return role, row


def _evaluate_tasks(
    tasks: list[dict[str, Any]],
    roles: Mapping[str, dict[str, Any]],
    models: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    _POINT_STATE.clear()
    _POINT_STATE.update({"roles": roles, "models": models})
    output: dict[str, list[dict[str, Any]]] = {"calibration": [], "final": []}
    workers = min(int(PROTOCOL["cpu_workers"]["point"]), max(1, (os.cpu_count() or 1) - 2))
    context = mp.get_context("fork")
    try:
        with context.Pool(processes=workers) as pool:
            for count, (role, row) in enumerate(
                pool.imap_unordered(_point_worker, tasks, chunksize=1), start=1
            ):
                output[role].append(row)
                if count % 16 == 0 or count == len(tasks):
                    print(f"[point] {count}/{len(tasks)}", flush=True)
    finally:
        _POINT_STATE.clear()
    return output


def _selection_key(method: str, policy: str, mode: str, budget: float) -> str:
    return f"{method}|{policy}|{mode}|{budget:g}"


def _serialize_selections(values: Mapping[str, Mapping[int, str]]) -> dict[str, dict[str, str]]:
    return {
        key: {str(image_id): str(action) for image_id, action in actions.items()}
        for key, actions in values.items()
    }


def run_point() -> dict[str, Any]:
    completion_path = RUNS / "point/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "point/report.json")
    _verify_preflight_hashes()
    scores = _load_scores()
    roles = {role: _action_role(role) for role in ("calibration", "final")}
    models = tri.load_frozen_models(BENCHMARK / "models.json")
    gate = read_json(BENCHMARK / "action_gate.json")
    action_maps = {
        role: _policy_maps(data, models, gate) for role, data in roles.items()
    }
    scheduler = read_json(EXTERNAL_ROOT / "configs/protocol.json")
    selections: dict[str, dict[str, dict[int, str]]] = {
        "calibration": {},
        "final": {},
    }
    tasks: list[dict[str, Any]] = []
    lambdas: dict[str, dict[str, float]] = {}
    for budget_index, budget in enumerate(BUDGETS):
        for method in METHODS:
            for role, data in roles.items():
                for policy in POLICIES:
                    selected = tri.current.top_k_actions(
                        data,
                        scores[role][method][budget_index],
                        action_maps[role][policy],
                        budget,
                    )
                    assert_no_roi_actions(selected)
                    key = _selection_key(method, policy, "fixed_call", budget)
                    selections[role][key] = selected
                    tasks.append(
                        {
                            "role": role,
                            "method": method,
                            "policy": policy,
                            "mode": "fixed_call",
                            "budget": budget,
                            "selected": selected,
                        }
                    )
    for budget_index, budget in enumerate(BUDGETS):
        for method in METHODS:
            for policy in POLICIES:
                calibrated = tri.current.calibrate_lambda(
                    roles["calibration"],
                    scores["calibration"][method][budget_index],
                    action_maps["calibration"][policy],
                    budget,
                    scheduler,
                )
                lambda_key = f"{method}|{policy}|{budget:g}"
                lambdas[lambda_key] = {
                    "lambda": float(calibrated["lambda"]),
                    "median_original_bytes": float(calibrated["median_original_bytes"]),
                    "teacher_gain": float(calibrated["teacher_gain"]),
                    "used_bytes": float(calibrated["used_bytes"]),
                    "selected_on": "calibration",
                }
                for role, data in roles.items():
                    priorities = tri.current.audit_priority(
                        scores["calibration"][method][budget_index],
                        scores[role][method][budget_index],
                        data,
                        action_maps[role][policy],
                        float(calibrated["lambda"]),
                        float(calibrated["median_original_bytes"]),
                    )
                    selected = tri.current.byte_selection(
                        data, priorities, action_maps[role][policy], budget
                    )
                    capacity = float(budget) * sum(
                        float(sample["payload_bytes"]["original"])
                        for sample in data["samples"]
                    )
                    assert_no_roi_actions(selected)
                    assert_fixed_byte_capacity(data["samples"], selected, capacity=capacity)
                    key = _selection_key(method, policy, "fixed_byte", budget)
                    selections[role][key] = selected
                    tasks.append(
                        {
                            "role": role,
                            "method": method,
                            "policy": policy,
                            "mode": "fixed_byte",
                            "budget": budget,
                            "selected": selected,
                            "lambda_value": float(calibrated["lambda"]),
                            "capacity_bytes": capacity,
                        }
                    )
    rows = _evaluate_tasks(tasks, roles, models)
    for role in rows:
        assert_result_matrix_complete(rows[role], PROTOCOL)
        assert_result_rows_no_roi(rows[role])
    calibration = rows["calibration"]
    comparators = {
        "external": freeze_strongest(
            calibration,
            candidates=EXTERNAL_METHODS,
            policies=POLICIES,
            budgets=BUDGETS,
        ),
        "historical": freeze_strongest(
            calibration,
            candidates=HISTORICAL_METHODS,
            policies=POLICIES,
            budgets=BUDGETS,
        ),
    }
    _verify_preflight_hashes()
    atomic_csv(RUNS / "point/calibration_results.csv", calibration)
    atomic_csv(RUNS / "point/final_results.csv", rows["final"])
    atomic_json(
        RUNS / "point/selections.json",
        {role: _serialize_selections(value) for role, value in selections.items()},
    )
    atomic_json(RUNS / "point/frozen_lambdas.json", lambdas)
    atomic_json(RUNS / "point/frozen_comparators.json", comparators)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "result_status": PROTOCOL["result_status"],
        "rows": rows["final"],
        "calibration_rows": len(calibration),
        "final_rows": len(rows["final"]),
        "comparators_selected_on": "calibration=571",
        "frozen_comparators": comparators,
        "rankers_retrained": False,
        "actions_retrained": False,
        "registered_final_used_for_selection": False,
    }
    atomic_json(RUNS / "point/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "rows": len(rows["final"]),
            "results_sha256": file_sha256(RUNS / "point/final_results.csv"),
            "selections_sha256": file_sha256(RUNS / "point/selections.json"),
            "comparators_sha256": file_sha256(RUNS / "point/frozen_comparators.json"),
            "registered_final_used_for_selection": False,
        },
    )
    return report


def _read_selection(stored: Mapping[str, Mapping[str, str]], key: str) -> dict[int, str]:
    actions = {int(image_id): str(action) for image_id, action in stored[key].items()}
    assert_no_roi_actions(actions)
    return actions


def _bootstrap_summary(values: np.ndarray, confidence: float) -> dict[str, float]:
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "ci_low": float(np.quantile(values, alpha)),
        "ci_high": float(np.quantile(values, 1.0 - alpha)),
        "probability_superior": float(np.mean(values > 0.0)),
    }


def _paired_bootstrap(
    *,
    ground_truth: dict,
    image_ids: list[int],
    candidate_predictions: list[dict],
    comparator_predictions: list[dict],
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0, len(image_ids), size=(int(replicates), len(image_ids)), dtype=np.int32
    )
    indices_sha256 = hashlib.sha256(indices.tobytes()).hexdigest()
    core = SOURCE_RUNNER.canonical_core
    core.configure_bootstrap(
        ground_truth, image_ids, candidate_predictions, comparator_predictions
    )
    values = np.empty(int(replicates), dtype=np.float64)
    workers = min(
        int(PROTOCOL["cpu_workers"]["bootstrap"]),
        max(1, (os.cpu_count() or 1) - 2),
    )
    context = mp.get_context("fork")
    try:
        with context.Pool(processes=workers) as pool:
            tasks = ((index, indices[index]) for index in range(int(replicates)))
            for count, (replicate, delta) in enumerate(
                pool.imap_unordered(core.bootstrap_worker, tasks, chunksize=1), start=1
            ):
                values[int(replicate)] = float(delta)
                if count % 50 == 0 or count == int(replicates):
                    print(f"[bootstrap] {count}/{replicates}", flush=True)
    finally:
        core.clear_bootstrap()
    return values, indices_sha256


def run_bootstrap() -> dict[str, Any]:
    completion_path = RUNS / "bootstrap/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "bootstrap/report.json")
    _verify_preflight_hashes()
    _require_complete(RUNS / "point/completion.json", "Point evaluation")
    final = _action_role("final")
    stored = read_json(RUNS / "point/selections.json")["final"]
    comparators = read_json(RUNS / "point/frozen_comparators.json")
    ground_truth = read_json(COMMON / "final/gt.json")
    image_ids = list(map(int, final["image_ids"]))
    config = PROTOCOL["bootstrap"]
    comparisons = []
    work = []
    for policy in POLICIES:
        for budget in BUDGETS:
            work.extend(
                [
                    (policy, budget, "strongest_external", comparators["external"][f"{policy}|{budget:g}"]),
                    (policy, budget, "strongest_historical", comparators["historical"][f"{policy}|{budget:g}"]),
                    (policy, budget, "SetValue", "SetValue"),
                ]
            )
    if len(work) != int(config["expected_comparisons"]):
        raise RuntimeError("Bootstrap comparison matrix changed")
    for index, (policy, budget, comparison_type, comparator) in enumerate(work, start=1):
        candidate_key = _selection_key("DualHead", policy, "fixed_byte", budget)
        comparator_key = _selection_key(comparator, policy, "fixed_byte", budget)
        slug = comparison_type.replace("-", "_")
        output = RUNS / "bootstrap" / f"{policy}_{budget:g}_{slug}.json"
        if output.is_file():
            item = read_json(output)
            values = np.asarray(item["values"], dtype=np.float64)
            if len(values) != int(config["replicates"]):
                raise RuntimeError("Bootstrap replicate count changed")
        else:
            candidate_actions = _read_selection(stored, candidate_key)
            comparator_actions = _read_selection(stored, comparator_key)
            candidate_predictions, _ = tri.current.mixed_predictions(
                final, candidate_actions, 0.001
            )
            comparator_predictions, _ = tri.current.mixed_predictions(
                final, comparator_actions, 0.001
            )
            values, indices_sha256 = _paired_bootstrap(
                ground_truth=ground_truth,
                image_ids=image_ids,
                candidate_predictions=candidate_predictions,
                comparator_predictions=comparator_predictions,
                replicates=int(config["replicates"]),
                seed=int(config["seed"]),
            )
            item = {
                "policy": policy,
                "budget": budget,
                "comparison_type": comparison_type,
                "candidate": "DualHead",
                "comparator": comparator,
                "replicates": len(values),
                "seed": int(config["seed"]),
                "indices_sha256": indices_sha256,
                "values": values.tolist(),
            }
            atomic_json(output, item)
        comparisons.append(
            {
                "policy": policy,
                "budget": budget,
                "comparison_type": comparison_type,
                "candidate": "DualHead",
                "comparator": comparator,
                "replicates": len(values),
                **_bootstrap_summary(values, float(config["confidence"])),
            }
        )
        print(f"[bootstrap-complete] {index}/{len(work)} {policy} {budget:g} {comparison_type}", flush=True)
    _verify_preflight_hashes()
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "result_status": PROTOCOL["result_status"],
        "comparisons": comparisons,
        "registered_final_used_for_selection": False,
    }
    atomic_json(RUNS / "bootstrap/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "comparisons": len(comparisons),
            "report_sha256": file_sha256(RUNS / "bootstrap/report.json"),
        },
    )
    return report


def _find_row(rows: list[dict[str, Any]], method: str, policy: str, mode: str, budget: float) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if str(row["method"]) == method
        and str(row["payload_policy"]) == policy
        and str(row["mode"]) == mode
        and abs(float(row["nominal_budget"]) - float(budget)) < 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Result lookup failed: {method}/{policy}/{mode}/{budget:g}")
    return matches[0]


def run_report() -> dict[str, Any]:
    completion_path = RUNS / "COMPLETED.json"
    if completion_path.is_file():
        return read_json(completion_path)
    point = read_json(RUNS / "point/report.json")
    bootstrap = read_json(RUNS / "bootstrap/report.json")
    rows = point["rows"]
    comparators = point["frozen_comparators"]
    lines = [
        "# COCO canonical fit=9,678 No-ROI 全 Ranker 对比",
        "",
        "> 状态：post-hoc canonical val2017 No-ROI 完整性实验，不是新的隐藏 final。",
        "",
        "## Fixed-byte：DualHead 对冻结最强基线",
        "",
        "| 策略 | 预算 | DualHead AP | 外部最强 / AP | 历史内部最强 / AP | KB/图 |",
        "|---|---:|---:|---|---|---:|",
    ]
    for policy in POLICIES:
        for budget in BUDGETS:
            external = comparators["external"][f"{policy}|{budget:g}"]
            historical = comparators["historical"][f"{policy}|{budget:g}"]
            dual = _find_row(rows, "DualHead", policy, "fixed_byte", budget)
            ext = _find_row(rows, external, policy, "fixed_byte", budget)
            hist = _find_row(rows, historical, policy, "fixed_byte", budget)
            lines.append(
                f"| {policy} | {budget:.0%} | {float(dual['official_map50_95']):.6f} | "
                f"{external} / {float(ext['official_map50_95']):.6f} | "
                f"{historical} / {float(hist['official_map50_95']):.6f} | "
                f"{float(dual['request_kb_per_image']):.2f} |"
            )
    lines.extend(
        [
            "",
            "## Paired bootstrap",
            "",
            "| 策略 | 预算 | 比较对象 | 均值 ΔAP | 95% CI | P(Δ>0) |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in bootstrap["comparisons"]:
        lines.append(
            f"| {row['policy']} | {float(row['budget']):.0%} | {row['comparator']} | "
            f"{float(row['mean_delta']):+.6f} | "
            f"[{float(row['ci_low']):+.6f}, {float(row['ci_high']):+.6f}] | "
            f"{float(row['probability_superior']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 协议核对",
            "",
            "- 16 个方法，2 个 No-ROI 策略，2 种预算模式，3 个预算，共 192 行。",
            "- 所有 `roi_ratio` 均为 0。",
            "- Fixed-byte lambda 和比较器仅由 571 张 calibration 冻结。",
            "- 所有学习器均复用 fit=9,678 的冻结 checkpoint；val2017 未用于选择。",
        ]
    )
    report_path = ROOT / "notes/results_zh.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    hashes = _verify_preflight_hashes()
    completion = {
        "complete": True,
        "protocol_id": PROTOCOL["protocol_id"],
        "result_status": PROTOCOL["result_status"],
        "final_rows": len(rows),
        "bootstrap_comparisons": len(bootstrap["comparisons"]),
        "source_digest": _source_digest(hashes),
        "results_sha256": file_sha256(RUNS / "point/final_results.csv"),
        "report_sha256": file_sha256(report_path),
        "source_experiments_modified": False,
        "registered_final_used_for_selection": False,
    }
    atomic_json(completion_path, completion)
    return completion


def run_all() -> dict[str, Any]:
    reports = {}
    for stage in ("preflight", "scores", "point", "bootstrap", "report"):
        print(f"[stage] {stage}", flush=True)
        reports[stage] = globals()[f"run_{stage}"]()
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("preflight", "scores", "point", "bootstrap", "report", "all"),
    )
    args = parser.parse_args()
    print(json.dumps(globals()[f"run_{args.stage}"](), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
