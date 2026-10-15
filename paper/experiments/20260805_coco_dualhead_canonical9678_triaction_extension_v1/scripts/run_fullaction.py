#!/usr/bin/env python3
"""Evaluate frozen canonical COCO rankers with the frozen TriAction policies."""

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

from fullaction_core import (
    METHODS,
    MODES,
    POLICIES,
    assert_fixed_call_membership,
    assert_result_matrix_complete,
    freeze_external_comparators,
    validate_checkpoint_fingerprint,
    validate_protocol,
    validate_source_artifact_state,
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
    name: REPO / path
    for name, path in PROTOCOL["source_experiments"].items()
}
RANKER_ROOT = SOURCES["rankers"]
CANONICAL_ROOT = SOURCES["canonical"]
TRIACTION_ROOT = SOURCES["triaction"]
EXTERNAL_ROOT = SOURCES["external_scheduler"]
COMMON = CANONICAL_ROOT / "runs/canonical_v1/common/coco"
BENCHMARK = CANONICAL_ROOT / "runs/canonical_v1/benchmark/coco"
TRIACTION_RUN = TRIACTION_ROOT / "runs/coco"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the frozen ranker runner normally enough for its multiprocessing worker
# module to remain importable, while keeping all paths pointed at its own runs.
sys.path.insert(0, str(RANKER_ROOT / "scripts"))
SOURCE_RUNNER = _load_module(
    "canonical9678_frozen_source_runner", RANKER_ROOT / "scripts/run_paper.py"
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
    expected = int(PROTOCOL["roles"]["registered_final" if role == "final" else role])
    if len(data["samples"]) != expected:
        raise RuntimeError(f"Canonical {role} population changed")
    return data


def _manifest_action_files(role: str) -> list[Path]:
    manifest_path = COMMON / role / "source_manifest.json"
    manifest = read_json(manifest_path)
    files = [manifest_path, COMMON / role / "samples.json", COMMON / role / "gt.json"]
    for action in ("edge", "original", "q90", "roi"):
        source = Path(manifest.get("action_sources", {}).get(action, ""))
        if not source.is_file():
            raise RuntimeError(f"Missing canonical {role}/{action} cache: {source}")
        expected_hash = manifest.get("source_paths", {}).get(str(source))
        # The canonical ROI action is a derived MAUR cache stored next to the
        # role data, while its mapped ROI parent is the file registered in
        # source_paths.  Freeze the derived cache in this preflight snapshot and
        # require all directly registered action sources to match immediately.
        if expected_hash is None and action != "roi":
            raise RuntimeError(f"Canonical {role}/{action} source is unregistered")
        if expected_hash is not None and file_sha256(source) != str(expected_hash):
            raise RuntimeError(f"Canonical {role}/{action} cache hash changed")
        if action == "roi" and source.name != "roi_maur_predictions.json":
            raise RuntimeError(f"Unexpected canonical {role}/roi derived cache")
        files.append(source)
    return files


def required_source_files() -> list[Path]:
    paths = {
        PROTOCOL_PATH,
        SCRIPT,
        ROOT / "scripts/fullaction_core.py",
        RANKER_ROOT / "configs/protocol.json",
        RANKER_ROOT / "scripts/run_paper.py",
        RANKER_ROOT / "scripts/canonical_core.py",
        RANKER_ROOT / "runs/train/checkpoints.pkl",
        RANKER_ROOT / "runs/train/completion.json",
        RANKER_ROOT / "runs/registered_final/point_report.json",
        RANKER_ROOT / "runs/registered_final/selections.json",
        BENCHMARK / "models.json",
        BENCHMARK / "action_gate.json",
        TRIACTION_ROOT / "configs/protocol.json",
        TRIACTION_RUN / "models/saferoi_gbdt.pkl",
        TRIACTION_RUN / "threshold_choices.json",
        EXTERNAL_ROOT / "configs/protocol.json",
        REPO
        / "paper/experiments/20260722_ena24_hierarchical_triaction_paper_exploration/scripts/run_exploration.py",
        CANONICAL_ROOT / "scripts/run_multiranker_benchmark.py",
        CANONICAL_ROOT / "scripts/saferoi_gbdt_core.py",
    }
    for role in ("calibration", "final"):
        paths.update(_manifest_action_files(role))
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


def run_preflight() -> dict[str, Any]:
    completion_path = RUNS / "preflight/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "preflight/report.json")
    train_completion = _require_complete(
        RANKER_ROOT / "runs/train/completion.json", "Source canonical training"
    )
    train_report = read_json(RANKER_ROOT / "runs/train/report.json")
    if int(train_report.get("development_images", -1)) != 9678:
        raise RuntimeError("Frozen ranker checkpoint was not trained on fit=9,678")
    checkpoint_matches = train_completion.get("checkpoint_sha256") == file_sha256(
        RANKER_ROOT / "runs/train/checkpoints.pkl"
    )
    validate_source_artifact_state(
        {
            "train_complete": bool(train_completion.get("complete")),
            "checkpoint_hash_matches": checkpoint_matches,
            "point_report_exists": (
                RANKER_ROOT / "runs/registered_final/point_report.json"
            ).is_file(),
            "selections_exist": (
                RANKER_ROOT / "runs/registered_final/selections.json"
            ).is_file(),
        }
    )
    roles = {role: _action_role(role) for role in ("calibration", "final")}
    hashes = capture_source_hashes()
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "passed": True,
        "result_status": PROTOCOL["result_status"],
        "role_counts": {role: len(data["samples"]) for role, data in roles.items()},
        "ranker_training_images": 9678,
        "ranker_checkpoint_sha256": train_completion["checkpoint_sha256"],
        "source_ranker_point_complete": True,
        "source_ranker_bootstrap_required_before_start": False,
        "source_sha256": hashes,
        "source_digest": _source_digest(hashes),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rankers_retrained": False,
        "action_models_retrained": False,
        "final_used_for_selection": False,
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


def _assert_source_selection_reproduction(
    *,
    role: str,
    records: list[dict[str, Any]],
    scores: Mapping[str, np.ndarray],
    protocol: Mapping[str, Any],
) -> None:
    """Prove the restricted checkpoint load reproduces frozen Original choices."""
    stored_path = (
        RANKER_ROOT / "runs/audit/selections.json"
        if role == "calibration"
        else RANKER_ROOT / "runs/registered_final/selections.json"
    )
    stored = read_json(stored_path)
    image_ids = [int(record["image_id"]) for record in records]
    full_capacity = sum(int(record["payload_bytes"]) for record in records)
    for budget_index, budget in enumerate(map(float, protocol["budgets"])):
        for method in METHODS:
            values = np.asarray(scores[method][budget_index], dtype=np.float64)
            fixed_ids = SOURCE_RUNNER.LEGACY.select_fixed_call(
                image_ids, values, budget=budget
            )
            byte_result = SOURCE_RUNNER.LEGACY.select_costeligible(
                records,
                SOURCE_RUNNER.LEGACY._score_matrix(values),
                resource_profile="upload",
                capacity=budget * full_capacity,
                full_only=True,
            )
            byte_ids = {
                int(item["image_id"])
                for item in byte_result["selected"]
                if str(item["action_type"]) == "full"
            }
            for mode, observed in (
                ("fixed_call", fixed_ids),
                ("fixed_byte", byte_ids),
            ):
                key = f"{method}|{mode}|{budget:g}"
                expected = set(map(int, stored[key]))
                if set(map(int, observed)) != expected:
                    raise RuntimeError(
                        f"Frozen source selection failed reproduction: {role}/{key}"
                    )


def run_scores() -> dict[str, Any]:
    completion_path = RUNS / "scores/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "scores/report.json")
    _verify_preflight_hashes()
    source_protocol = SOURCE_RUNNER._protocol()
    checkpoint_path = RANKER_ROOT / "runs/train/checkpoints.pkl"
    completion = _require_complete(
        RANKER_ROOT / "runs/train/completion.json", "Source canonical training"
    )
    if completion.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise RuntimeError("Frozen ranker checkpoint hash changed")
    with checkpoint_path.open("rb") as handle:
        artifact = pickle.load(handle)
    fingerprint_differences = validate_checkpoint_fingerprint(
        artifact["source_fingerprint"], SOURCE_RUNNER._source_fingerprint(source_protocol)
    )
    arrays: dict[str, np.ndarray] = {"methods": np.asarray(METHODS, dtype="U32")}
    role_alias = {"calibration": "calibration", "final": "registered_final"}
    role_counts = {}
    for role, alias in role_alias.items():
        records = SOURCE_RUNNER._records_for_role(alias, source_protocol)
        observed = SOURCE_RUNNER.LEGACY._method_scores(records, source_protocol, artifact)
        ids = np.asarray([int(row["image_id"]) for row in records], dtype=np.int64)
        action_ids = np.asarray(
            [int(row["image_id"]) for row in _action_role(role)["samples"]],
            dtype=np.int64,
        )
        if not np.array_equal(ids, action_ids):
            raise RuntimeError(f"Ranker/action image ordering differs for {role}")
        matrix = np.stack(
            [np.asarray(observed[method], dtype=np.float64) for method in METHODS]
        )
        expected_shape = (len(METHODS), 3, len(ids))
        if matrix.shape != expected_shape or not np.all(np.isfinite(matrix)):
            raise RuntimeError(
                f"Invalid {role} score matrix: {matrix.shape} != {expected_shape}"
            )
        arrays[f"{role}_ids"] = ids
        arrays[f"{role}_scores"] = matrix
        role_counts[role] = len(ids)
        _assert_source_selection_reproduction(
            role=role,
            records=records,
            scores={method: matrix[index] for index, method in enumerate(METHODS)},
            protocol=source_protocol,
        )
    score_path = RUNS / "scores/frozen_scores.npz"
    atomic_npz(score_path, **arrays)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "methods": list(METHODS),
        "role_counts": role_counts,
        "score_shapes": {
            role: list(arrays[f"{role}_scores"].shape)
            for role in role_alias
        },
        "checkpoint_sha256": file_sha256(
            RANKER_ROOT / "runs/train/checkpoints.pkl"
        ),
        "checkpoint_fingerprint_differences": fingerprint_differences,
        "source_original_selection_reproduced": True,
        "scores_sha256": file_sha256(score_path),
        "rankers_retrained": False,
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
    completion = _require_complete(
        RUNS / "scores/completion.json", "Frozen score export"
    )
    score_path = RUNS / "scores/frozen_scores.npz"
    if completion.get("scores_sha256") != file_sha256(score_path):
        raise RuntimeError("Frozen score cache changed")
    with np.load(score_path, allow_pickle=False) as value:
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


def _policy_assets(
    roles: Mapping[str, dict[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, dict[str, dict[int, str]]]],
    dict[str, dict[str, float | None]],
]:
    models = tri.load_frozen_models(BENCHMARK / "models.json")
    gate = read_json(BENCHMARK / "action_gate.json")
    shared = {
        role: tri.current.chosen_actions(models["action_heads"], gate, data)
        for role, data in roles.items()
    }
    sklearn_path = Path(PROTOCOL["runtime"]["sklearn_site_packages"])
    if not sklearn_path.is_dir():
        raise RuntimeError(f"Frozen scikit-learn runtime is missing: {sklearn_path}")
    if str(sklearn_path) not in sys.path:
        sys.path.append(str(sklearn_path))
    import sklearn  # type: ignore

    if sklearn.__version__ != str(PROTOCOL["runtime"]["sklearn_version"]):
        raise RuntimeError(
            f"scikit-learn version changed: {sklearn.__version__}"
        )
    with (TRIACTION_RUN / "models/saferoi_gbdt.pkl").open("rb") as handle:
        roi_model = pickle.load(handle)
    roi_scores = {
        role: tri.predict_gbdt(roi_model, tri.profile_features(data))
        for role, data in roles.items()
    }
    choices = read_json(TRIACTION_RUN / "threshold_choices.json")["shared"]
    thresholds = {
        mode: {
            "SharedAction": None,
            "CostEligible-AllROI": float("-inf"),
            "GBDT-Hierarchical": float(choices[mode]["threshold"]),
        }
        for mode in MODES
    }
    maps = {
        role: {
            mode: {
                "SharedAction": shared[role],
                "CostEligible-AllROI": tri.action_map_at_threshold(
                    data, shared[role], roi_scores[role], float("-inf")
                ),
                "GBDT-Hierarchical": tri.action_map_at_threshold(
                    data,
                    shared[role],
                    roi_scores[role],
                    float(thresholds[mode]["GBDT-Hierarchical"]),
                ),
            }
            for mode in MODES
        }
        for role, data in roles.items()
    }
    return models, maps, thresholds


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
    threshold: float | None,
    lambda_value: float | None = None,
) -> dict[str, Any]:
    row = tri.current.evaluate(method, mode, budget, data, selected, models)
    row.update(
        {
            "role": role,
            "method": method,
            "payload_policy": policy,
            "action_policy": policy,
            "mode": mode,
            "nominal_budget": float(budget),
            "roi_threshold": threshold,
            "roi_threshold_source": (
                "none_shared_action"
                if policy == "SharedAction"
                else "all_cost_eligible"
                if policy == "CostEligible-AllROI"
                else "frozen_shared_8ranker_calibration"
            ),
            "ranker_checkpoint_frozen": True,
            "selector_ms_per_image": None,
            "selector_params": None,
            "estimated_e2e": False,
        }
    )
    if lambda_value is not None:
        row["lambda"] = float(lambda_value)
    for key in list(row):
        if key.startswith("e2e_"):
            row[key] = None
    return row


def _point_worker(task: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if not _POINT_STATE:
        raise RuntimeError("Point-evaluation worker state is not configured")
    role = str(task["role"])
    return role, _evaluate_row(
        role=role,
        method=str(task["method"]),
        policy=str(task["policy"]),
        mode=str(task["mode"]),
        budget=float(task["budget"]),
        data=_POINT_STATE["roles"][role],
        selected=dict(task["selected"]),
        models=_POINT_STATE["models"],
        threshold=task["threshold"],
        lambda_value=task.get("lambda_value"),
    )


def _evaluate_point_tasks(
    tasks: list[dict[str, Any]],
    roles: Mapping[str, dict[str, Any]],
    models: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    _POINT_STATE.clear()
    _POINT_STATE.update({"roles": roles, "models": models})
    rows: dict[str, list[dict[str, Any]]] = {"calibration": [], "final": []}
    workers = int(PROTOCOL["cpu_workers"]["point"])
    context = mp.get_context("fork")
    try:
        with context.Pool(processes=workers) as pool:
            for completed, (role, row) in enumerate(
                pool.imap(_point_worker, tasks, chunksize=1), start=1
            ):
                rows[role].append(row)
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"[point] {completed}/{len(tasks)}", flush=True)
    finally:
        _POINT_STATE.clear()
    return rows


def _serialize_selections(
    selections: Mapping[str, Mapping[int, str]],
) -> dict[str, dict[str, str]]:
    return {
        key: {str(image_id): str(action) for image_id, action in actions.items()}
        for key, actions in selections.items()
    }


def _selection_key(method: str, policy: str, mode: str, budget: float) -> str:
    return f"{method}|{policy}|{mode}|{budget:g}"


def run_point() -> dict[str, Any]:
    completion_path = RUNS / "point/completion.json"
    if completion_path.is_file():
        return read_json(RUNS / "point/report.json")
    _verify_preflight_hashes()
    scores = _load_scores()
    roles = {role: _action_role(role) for role in ("calibration", "final")}
    models, action_maps, thresholds = _policy_assets(roles)
    scheduler = read_json(EXTERNAL_ROOT / "configs/protocol.json")
    tasks: list[dict[str, Any]] = []
    selections: dict[str, dict[str, dict[int, str]]] = {
        "calibration": {},
        "final": {},
    }
    lambdas: dict[str, dict[str, float]] = {}

    for budget_index, budget in enumerate(map(float, PROTOCOL["work_points"]["fixed_call"])):
        for method in METHODS:
            by_policy: dict[str, dict[int, str]] = {}
            for role, data in roles.items():
                role_by_policy = {}
                method_score = scores[role][method][budget_index]
                for policy in POLICIES:
                    selected = tri.current.top_k_actions(
                        data,
                        method_score,
                        action_maps[role]["fixed_call"][policy],
                        budget,
                    )
                    role_by_policy[policy] = selected
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
                            "threshold": thresholds["fixed_call"][policy],
                        }
                    )
                assert_fixed_call_membership(role_by_policy)
                if role == "final":
                    by_policy = role_by_policy
            assert_fixed_call_membership(by_policy)

    for budget_index, budget in enumerate(map(float, PROTOCOL["work_points"]["fixed_byte"])):
        for method in METHODS:
            for policy in POLICIES:
                calibration_map = action_maps["calibration"]["fixed_byte"][policy]
                final_map = action_maps["final"]["fixed_byte"][policy]
                calibrated = tri.current.calibrate_lambda(
                    roles["calibration"],
                    scores["calibration"][method][budget_index],
                    calibration_map,
                    budget,
                    scheduler,
                )
                lambda_key = f"{method}|{policy}|{budget:g}"
                lambdas[lambda_key] = {
                    "lambda": float(calibrated["lambda"]),
                    "median_original_bytes": float(calibrated["median_original_bytes"]),
                    "teacher_gain": float(calibrated["teacher_gain"]),
                    "used_bytes": float(calibrated["used_bytes"]),
                }
                for role, action_map in (
                    ("calibration", calibration_map),
                    ("final", final_map),
                ):
                    priorities = tri.current.audit_priority(
                        scores["calibration"][method][budget_index],
                        scores[role][method][budget_index],
                        roles[role],
                        action_map,
                        float(calibrated["lambda"]),
                        float(calibrated["median_original_bytes"]),
                    )
                    selected = tri.current.byte_selection(
                        roles[role], priorities, action_map, budget
                    )
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
                            "threshold": thresholds["fixed_byte"][policy],
                            "lambda_value": float(calibrated["lambda"]),
                        }
                    )

    rows = _evaluate_point_tasks(tasks, roles, models)
    assert_result_matrix_complete(rows["calibration"], PROTOCOL)
    assert_result_matrix_complete(rows["final"], PROTOCOL)
    comparators = freeze_external_comparators(rows["calibration"], PROTOCOL)
    _verify_preflight_hashes()
    atomic_csv(RUNS / "point/calibration_results.csv", rows["calibration"])
    atomic_csv(RUNS / "point/final_results.csv", rows["final"])
    atomic_json(
        RUNS / "point/selections.json",
        {
            role: _serialize_selections(value)
            for role, value in selections.items()
        },
    )
    atomic_json(RUNS / "point/frozen_lambdas.json", lambdas)
    atomic_json(RUNS / "point/frozen_comparators.json", comparators)
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "result_status": PROTOCOL["result_status"],
        "rows": rows["final"],
        "calibration_rows": len(rows["calibration"]),
        "final_rows": len(rows["final"]),
        "comparators_selected_on": "calibration=571",
        "frozen_comparators": comparators,
        "ranker_checkpoint_retrained": False,
        "action_models_retrained": False,
        "final_used_for_selection": False,
    }
    atomic_json(RUNS / "point/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "rows": len(rows["final"]),
            "results_sha256": file_sha256(RUNS / "point/final_results.csv"),
            "selections_sha256": file_sha256(RUNS / "point/selections.json"),
            "comparators_sha256": file_sha256(
                RUNS / "point/frozen_comparators.json"
            ),
            "final_used_for_selection": False,
        },
    )
    return report


def _read_selection(
    stored: Mapping[str, Mapping[str, str]], key: str
) -> dict[int, str]:
    return {int(image_id): str(action) for image_id, action in stored[key].items()}


def _bootstrap_summary(values: np.ndarray, confidence: float) -> dict[str, float]:
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "ci_low": float(np.quantile(values, alpha)),
        "ci_high": float(np.quantile(values, 1.0 - alpha)),
        "probability_superior": float(np.mean(values > 0.0)),
    }


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
    reports = []
    for policy_index, policy in enumerate(POLICIES):
        for budget_index, budget in enumerate(
            map(float, PROTOCOL["work_points"]["fixed_byte"])
        ):
            comparator = comparators[f"{policy}|{budget:g}"]
            candidate_key = _selection_key(
                PROTOCOL["primary_candidate"], policy, "fixed_byte", budget
            )
            comparator_key = _selection_key(
                comparator, policy, "fixed_byte", budget
            )
            output = RUNS / "bootstrap" / f"{policy}_{budget:g}.json"
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
                values, indices_sha256 = SOURCE_RUNNER._paired_bootstrap(
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
                    "candidate": PROTOCOL["primary_candidate"],
                    "comparator": comparator,
                    "replicates": len(values),
                    "seed": int(config["seed"]),
                    "indices_sha256": indices_sha256,
                    "values": values.tolist(),
                }
                atomic_json(output, item)
            reports.append(
                {
                    "policy": policy,
                    "budget": budget,
                    "candidate": PROTOCOL["primary_candidate"],
                    "comparator": comparator,
                    "replicates": len(values),
                    **_bootstrap_summary(values, float(config["confidence"])),
                }
            )
            print(
                f"[bootstrap-complete] {policy} {budget:g} "
                f"({policy_index * 3 + budget_index + 1}/9)",
                flush=True,
            )
    _verify_preflight_hashes()
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "result_status": PROTOCOL["result_status"],
        "comparisons": reports,
        "final_used_for_selection": False,
    }
    atomic_json(RUNS / "bootstrap/report.json", report)
    atomic_json(
        completion_path,
        {
            "complete": True,
            "comparisons": len(reports),
            "report_sha256": file_sha256(RUNS / "bootstrap/report.json"),
        },
    )
    return report


def run_report() -> dict[str, Any]:
    completion_path = RUNS / "COMPLETED.json"
    if completion_path.is_file():
        return read_json(completion_path)
    point = read_json(RUNS / "point/report.json")
    bootstrap = read_json(RUNS / "bootstrap/report.json")
    rows = point["rows"]
    comparators = point["frozen_comparators"]
    lines = [
        "# COCO canonical 9,678 TriAction 追加实验结果",
        "",
        "> 状态：post-hoc canonical val2017 动作扩展，不是新的隐藏 final。",
        "",
        "## Fixed-byte：DualHead 对冻结外部最强方法",
        "",
        "| 动作框架 | 预算 | DualHead AP | 外部方法 | 外部 AP | ΔAP | DualHead KB/图 |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for policy in POLICIES:
        for budget in map(float, PROTOCOL["work_points"]["fixed_byte"]):
            comparator = comparators[f"{policy}|{budget:g}"]
            def find(method: str) -> Mapping[str, Any]:
                matches = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["payload_policy"] == policy
                    and row["mode"] == "fixed_byte"
                    and abs(float(row["nominal_budget"]) - budget) < 1e-12
                ]
                if len(matches) != 1:
                    raise RuntimeError("Result lookup failed while writing report")
                return matches[0]
            candidate = find("DualHead")
            external = find(comparator)
            delta = float(candidate["official_map50_95"]) - float(
                external["official_map50_95"]
            )
            lines.append(
                f"| {policy} | {budget:.0%} | "
                f"{float(candidate['official_map50_95']):.6f} | {comparator} | "
                f"{float(external['official_map50_95']):.6f} | {delta:+.6f} | "
                f"{float(candidate['request_kb_per_image']):.2f} |"
            )
    lines.extend(
        [
            "",
            "## Paired bootstrap",
            "",
            "| 动作框架 | 预算 | 均值 ΔAP | 95% CI | P(Δ>0) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in bootstrap["comparisons"]:
        lines.append(
            f"| {row['policy']} | {float(row['budget']):.0%} | "
            f"{float(row['mean_delta']):+.6f} | "
            f"[{float(row['ci_low']):+.6f}, {float(row['ci_high']):+.6f}] | "
            f"{float(row['probability_superior']):.3f} |"
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
        "final_used_for_selection": False,
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
    result = globals()[f"run_{args.stage}"]()
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
