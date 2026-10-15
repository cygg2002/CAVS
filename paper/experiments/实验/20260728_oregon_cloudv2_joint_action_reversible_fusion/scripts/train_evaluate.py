"""Train, calibrate, and audit the Oregon Cloud-V2 joint action selector."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np


SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
SOURCE_SCRIPTS = (
    REPO
    / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts"
)
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(SOURCE_SCRIPTS))

from audit_ena24_roi_protocol import base_actions, q90_probabilities  # noqa: E402
from benchmark_core import image_quality  # noqa: E402
from final_paper_evaluation_core import (  # noqa: E402
    build_bootstrap_coco_sample,
    deterministic_bootstrap_indices,
)
from protocol_core import class_aware_nms, maur_candidates  # noqa: E402
import run_multiranker_benchmark as benchmark  # noqa: E402

from joint_core import (  # noqa: E402
    BASE_ACTIONS,
    action_feature,
    balanced_binary_weights,
    candidate_matrix,
    choose_ap_safe_configuration,
    choose_reversible_configuration,
    reversible_fusion,
    select_joint_actions,
    validate_roles,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_hash_inventory(
    observed: Mapping[str, str], expected: Mapping[str, str], *, label: str
) -> None:
    if dict(observed) != dict(expected):
        raise RuntimeError(f"{label} hash drift")


def finalize_stage(summary_path: Path, result: Any, marker_path: Path) -> None:
    write_json(summary_path, result)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("ok\n", encoding="ascii")


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


def analysis_code_hashes() -> dict[str, str]:
    paths = (
        SCRIPT,
        SCRIPT.parent / "joint_core.py",
        SCRIPT.parent / "run_experiment.py",
    )
    return {path.name: file_sha256(path) for path in paths}


def candidate_summary_hashes(
    config: Mapping[str, Any], role: str
) -> dict[str, str]:
    root = Path(config["output_root"]) / "runs/actions" / role
    return {
        f"{role}:{candidate['id']}": file_sha256(
            root
            / str(candidate["profile"])
            / f"imgsz_{int(candidate['imgsz'])}"
            / "summary.json"
        )
        for candidate in candidate_matrix(config["profiles"], config["imgsz"])
    }


def grouped_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in read_json(path):
        grouped[int(prediction["image_id"])].append(prediction)
    return dict(grouped)


def load_role(
    config: dict[str, Any], role: str
) -> tuple[dict[str, Any], dict[int, str], dict[str, Any]]:
    validate_roles(config["roles"])
    if role not in config["roles"] or role == "final":
        raise ValueError(f"unregistered role: {role}")
    canonical = (
        Path(config["source_cloudv2_root"])
        / "runs/canonical_oregon_event_v1"
    )
    benchmark.COMMON = canonical / "common"
    data = benchmark.load_role("oregon", role)
    model_root = canonical / "benchmark/oregon"
    models = read_json(model_root / "models.json")
    gate = read_json(model_root / "action_gate.json")
    threshold = float(gate["chosen"]["q90_threshold"])
    base = base_actions(
        data,
        q90_probabilities(models["action_heads"]["q90"], data),
        threshold,
    )
    return data, base, {
        "models_sha256": file_sha256(model_root / "models.json"),
        "action_gate_sha256": file_sha256(model_root / "action_gate.json"),
        "q90_threshold": threshold,
    }


def candidate_assets(
    config: dict[str, Any], role: str, candidate: Mapping[str, Any]
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    root = (
        Path(config["output_root"])
        / "runs/actions"
        / role
        / str(candidate["profile"])
    )
    geometry_path = root / "geometry/roi_manifest.json"
    prediction_root = root / f"imgsz_{int(candidate['imgsz'])}"
    required = [
        geometry_path,
        prediction_root / "COMPLETED",
        prediction_root / "summary.json",
        prediction_root / "roi_predictions_mapped.json",
    ]
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"incomplete candidate assets: {prediction_root}")
    geometry = read_json(geometry_path)
    summary = read_json(prediction_root / "summary.json")
    if summary["candidate_id"] != candidate["id"]:
        raise RuntimeError(f"candidate ID drift: {prediction_root}")
    if summary["geometry_signature"] != geometry["signature"]["sha256"]:
        raise RuntimeError(f"geometry signature drift: {prediction_root}")
    predictions_path = prediction_root / "roi_predictions_mapped.json"
    if file_sha256(predictions_path) != summary["prediction_sha256"]:
        raise RuntimeError(f"prediction hash drift: {prediction_root}")
    records = {
        int(image_id): dict(value)
        for image_id, value in geometry["records"].items()
    }
    return records, grouped_predictions(predictions_path)


def load_raw_candidates(
    config: dict[str, Any], role: str
) -> dict[str, dict[str, Any]]:
    raw = {}
    for candidate in candidate_matrix(config["profiles"], config["imgsz"]):
        records, predictions = candidate_assets(config, role, candidate)
        raw[candidate["id"]] = {
            "candidate": dict(candidate),
            "records": records,
            "predictions": predictions,
        }
    return raw


def fusion_id(parameters: Mapping[str, Any]) -> str:
    return (
        f"miou={float(parameters['match_iou']):.2f}|"
        f"margin={float(parameters['score_margin']):+.2f}|"
        f"add={float(parameters['add_threshold']):.2f}"
    )


def build_details(
    data: dict[str, Any],
    base: dict[int, str],
    raw: dict[str, dict[str, Any]],
    *,
    fusion: str,
    reversible_parameters: Mapping[str, Any] | None = None,
) -> dict[str, dict[int, dict[str, Any]]]:
    if fusion not in {"maur", "reversible"}:
        raise ValueError(f"unknown fusion: {fusion}")
    if fusion == "reversible" and reversible_parameters is None:
        raise ValueError("reversible fusion parameters are required")
    details: dict[str, dict[int, dict[str, Any]]] = {}
    for candidate_id, assets in raw.items():
        candidate = assets["candidate"]
        by_image = {}
        for sample in data["samples"]:
            image_id = int(sample["image_id"])
            record = assets["records"].get(image_id)
            if record is None:
                continue
            edge = data["actions"]["edge"].get(image_id, [])
            cloud = assets["predictions"].get(image_id, [])
            if fusion == "maur":
                predictions = class_aware_nms(
                    maur_candidates(edge, cloud, tuple(map(int, record["xyxy"]))),
                    iou_threshold=0.7,
                )
            else:
                predictions = reversible_fusion(
                    edge,
                    cloud,
                    match_iou=float(reversible_parameters["match_iou"]),
                    score_margin=float(reversible_parameters["score_margin"]),
                    add_threshold=float(reversible_parameters["add_threshold"]),
                    nms_iou=0.7,
                )
            base_action = base[image_id]
            by_image[image_id] = {
                "action_id": candidate_id,
                "profile": candidate["profile"],
                "imgsz": int(candidate["imgsz"]),
                "payload_bytes": int(record["bytes"]),
                "base_payload_bytes": int(sample["payload_bytes"][base_action]),
                "record": record,
                "predictions": predictions,
                "quality": float(
                    image_quality(data["gt_map"].get(image_id, []), predictions)
                ),
                "base_quality": float(sample["qualities"][base_action]),
            }
        details[candidate_id] = by_image
    return details


def action_predictions(
    data: dict[str, Any],
    details: dict[str, dict[int, dict[str, Any]]],
    image_id: int,
    action: str,
) -> list[dict[str, Any]]:
    if action in BASE_ACTIONS:
        return data["actions"][action].get(image_id, [])
    return details[action][image_id]["predictions"]


def flat_predictions(
    data: dict[str, Any],
    details: dict[str, dict[int, dict[str, Any]]],
    actions: Mapping[int, str],
) -> list[dict[str, Any]]:
    flat = []
    for image_id in data["image_ids"]:
        action = str(actions[int(image_id)])
        flat.extend(
            value
            for value in action_predictions(data, details, int(image_id), action)
            if float(value.get("score", 1.0)) >= 0.001
        )
    return flat


def evaluate_actions(
    data: dict[str, Any],
    base: dict[int, str],
    details: dict[str, dict[int, dict[str, Any]]],
    actions: Mapping[int, str],
) -> dict[str, Any]:
    metrics = data["evaluator"].evaluate(flat_predictions(data, details, actions))
    request_bytes = 0
    safe = []
    counts = Counter(map(str, actions.values()))
    sample_by_id = data["sample_by_id"]
    for image_id, action in actions.items():
        sample = sample_by_id[int(image_id)]
        if action in BASE_ACTIONS:
            request_bytes += int(sample["payload_bytes"][action])
        else:
            detail = details[action][int(image_id)]
            request_bytes += int(detail["payload_bytes"])
            safe.append(
                float(detail["quality"])
                >= float(detail["base_quality"]) - 0.02
            )
    count = len(data["samples"])
    return {
        "official_map50_95": float(metrics["map50_95"]),
        "official_map50": float(metrics["map50"]),
        "official_map75": float(metrics["map75"]),
        "request_kb_per_image": request_bytes / count / 1024.0,
        "original_ratio": counts["original"] / count,
        "q90_ratio": counts["q90"] / count,
        "roi_ratio": sum(count for action, count in counts.items() if "@" in action)
        / count,
        "roi_safe_precision": float(np.mean(safe)) if safe else 1.0,
        "action_counts": dict(sorted(counts.items())),
    }


def teacher_actions(
    data: dict[str, Any],
    base: dict[int, str],
    details: dict[str, dict[int, dict[str, Any]]],
    budget: float,
) -> dict[int, str]:
    choices = []
    for sample in data["samples"]:
        image_id = int(sample["image_id"])
        base_action = base[image_id]
        base_payload = int(sample["payload_bytes"][base_action])
        options = [
            value
            for by_image in details.values()
            if (value := by_image.get(image_id)) is not None
            and int(value["payload_bytes"]) < base_payload
        ]
        if not options:
            continue
        best = min(
            options,
            key=lambda value: (
                -float(value["quality"]),
                int(value["payload_bytes"]),
                str(value["action_id"]),
            ),
        )
        gain = float(best["quality"]) - float(best["base_quality"])
        choices.append((-gain, int(best["payload_bytes"]), image_id, best["action_id"]))
    actions = dict(base)
    maximum = round(float(budget) * len(data["samples"]))
    for _, _, image_id, action_id in sorted(choices)[:maximum]:
        actions[image_id] = str(action_id)
    return actions


def choose_reversible_fusion(
    config: dict[str, Any],
    data: dict[str, Any],
    base: dict[int, str],
    raw: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    empty_details: dict[str, dict[int, dict[str, Any]]] = {}
    baseline = evaluate_actions(data, base, empty_details, base)
    rows = []
    for parameters in config["fusion"]["reversible_grid"]:
        details = build_details(
            data,
            base,
            raw,
            fusion="reversible",
            reversible_parameters=parameters,
        )
        config_id = fusion_id(parameters)
        for budget in map(float, config["scheduler"]["roi_budgets"]):
            metrics = evaluate_actions(
                data, base, details, teacher_actions(data, base, details, budget)
            )
            rows.append(
                {
                    "config_id": config_id,
                    "budget": budget,
                    "map_delta_vs_base": float(metrics["official_map50_95"])
                    - float(baseline["official_map50_95"]),
                    **metrics,
                }
            )
        del details
    chosen = choose_reversible_configuration(rows)
    parameters = next(
        dict(value)
        for value in config["fusion"]["reversible_grid"]
        if fusion_id(value) == chosen["config_id"]
    )
    return {**chosen, "parameters": parameters}, rows


def action_rows(
    data: dict[str, Any],
    base: dict[int, str],
    details: dict[str, dict[int, dict[str, Any]]],
    *,
    epsilon: float,
) -> dict[str, Any]:
    features = []
    labels = []
    gains = []
    image_ids = []
    metadata = []
    for sample in data["samples"]:
        image_id = int(sample["image_id"])
        base_action = base[image_id]
        base_quality = float(sample["qualities"][base_action])
        base_payload = int(sample["payload_bytes"][base_action])
        for action_id in BASE_ACTIONS:
            quality = float(sample["qualities"][action_id])
            payload = int(sample["payload_bytes"][action_id])
            features.append(
                action_feature(
                    sample,
                    action_id=action_id,
                    payload_bytes=payload,
                    base_payload_bytes=base_payload,
                    record=None,
                )
            )
            labels.append(quality >= base_quality - float(epsilon))
            gains.append(quality - base_quality)
            image_ids.append(image_id)
            metadata.append(
                {
                    "image_id": image_id,
                    "action_id": action_id,
                    "payload_bytes": payload,
                    "quality": quality,
                    "base_quality": base_quality,
                }
            )
        for candidate_id, by_image in details.items():
            detail = by_image.get(image_id)
            if detail is None:
                continue
            quality = float(detail["quality"])
            features.append(
                action_feature(
                    sample,
                    action_id=candidate_id,
                    payload_bytes=int(detail["payload_bytes"]),
                    base_payload_bytes=base_payload,
                    record=detail["record"],
                )
            )
            labels.append(quality >= base_quality - float(epsilon))
            gains.append(quality - base_quality)
            image_ids.append(image_id)
            metadata.append(
                {
                    "image_id": image_id,
                    "action_id": candidate_id,
                    "payload_bytes": int(detail["payload_bytes"]),
                    "quality": quality,
                    "base_quality": base_quality,
                }
            )
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 70 or not np.isfinite(matrix).all():
        raise RuntimeError(f"invalid action feature matrix: {matrix.shape}")
    return {
        "features": matrix,
        "labels": np.asarray(labels, dtype=bool),
        "gains": np.asarray(gains, dtype=np.float32),
        "image_ids": np.asarray(image_ids, dtype=np.int64),
        "metadata": metadata,
    }


def fit_models(rows: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    hp = config["model"]
    common = {
        "learning_rate": float(hp["learning_rate"]),
        "max_iter": int(hp["max_iter"]),
        "max_depth": int(hp["max_depth"]),
        "max_leaf_nodes": int(hp["max_leaf_nodes"]),
        "min_samples_leaf": int(hp["min_samples_leaf"]),
        "l2_regularization": float(hp["l2_regularization"]),
        "early_stopping": False,
        "random_state": int(config["seed"]),
    }
    classifier = HistGradientBoostingClassifier(**common)
    classifier.fit(
        rows["features"],
        rows["labels"],
        sample_weight=np.asarray(
            balanced_binary_weights(rows["labels"], rows["image_ids"]),
            dtype=np.float64,
        ),
    )
    counts = Counter(map(int, rows["image_ids"]))
    gain_weights = np.asarray(
        [1.0 / counts[int(image_id)] for image_id in rows["image_ids"]],
        dtype=np.float64,
    )
    gain_weights /= gain_weights.sum()
    regressor = HistGradientBoostingRegressor(**common)
    regressor.fit(
        rows["features"], rows["gains"], sample_weight=gain_weights
    )
    return {"safe_classifier": classifier, "gain_regressor": regressor}


def predict_options(
    rows: dict[str, Any], models: dict[str, Any]
) -> dict[int, list[dict[str, Any]]]:
    classifier = models["safe_classifier"]
    probabilities = classifier.predict_proba(rows["features"])
    classes = list(map(bool, classifier.classes_))
    safe_scores = (
        probabilities[:, classes.index(True)]
        if True in classes
        else np.zeros(len(probabilities), dtype=np.float64)
    )
    predicted_gains = models["gain_regressor"].predict(rows["features"])
    options: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for metadata, safe, gain in zip(
        rows["metadata"], safe_scores, predicted_gains
    ):
        options[int(metadata["image_id"])].append(
            {
                **metadata,
                "safe_probability": float(safe),
                "predicted_gain": float(gain),
            }
        )
    return dict(options)


def calibration_rows(
    config: dict[str, Any],
    branch: str,
    data: dict[str, Any],
    base: dict[int, str],
    details: dict[str, dict[int, dict[str, Any]]],
    options: dict[int, list[dict[str, Any]]],
    budget: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_metrics = evaluate_actions(data, base, details, base)
    baseline = {
        "name": "base",
        "branch": branch,
        "budget": float(budget),
        "safe_threshold": None,
        "cost_weight": None,
        "base_mode": "frozen_q90_original",
        **baseline_metrics,
    }
    rows = [baseline]
    for base_mode in config["scheduler"]["base_modes"]:
        for threshold in map(float, config["scheduler"]["safe_thresholds"]):
            for cost_weight in map(float, config["scheduler"]["cost_weights"]):
                actions = select_joint_actions(
                    options,
                    frozen_base=base,
                    roi_budget=float(budget),
                    safe_threshold=threshold,
                    cost_weight=cost_weight,
                    base_mode=str(base_mode),
                )
                metrics = evaluate_actions(data, base, details, actions)
                rows.append(
                    {
                        "name": (
                            f"{branch}|{base_mode}|p={threshold:.2f}|"
                            f"cost={cost_weight:.4f}"
                        ),
                        "branch": branch,
                        "budget": float(budget),
                        "safe_threshold": threshold,
                        "cost_weight": cost_weight,
                        "base_mode": str(base_mode),
                        **metrics,
                    }
                )
    chosen = choose_ap_safe_configuration(
        rows,
        baseline_name="base",
        tolerance=float(config["evaluation"]["ap_tolerance"]),
    )
    return rows, chosen


def compact_choice(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "name",
        "branch",
        "budget",
        "safe_threshold",
        "cost_weight",
        "base_mode",
        "official_map50_95",
        "official_map50",
        "official_map75",
        "request_kb_per_image",
        "original_ratio",
        "q90_ratio",
        "roi_ratio",
        "roi_safe_precision",
        "action_counts",
    )
    return {key: row.get(key) for key in keys}


def validate_analysis_environment(config: Mapping[str, Any]) -> str:
    sklearn_version = importlib.metadata.version("scikit-learn")
    expected_version = str(config["execution"]["analysis_sklearn_version"])
    if sklearn_version != expected_version:
        raise RuntimeError(
            f"scikit-learn version drift: {sklearn_version} != {expected_version}"
        )
    expected_python = Path(config["execution"]["analysis_python"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"unexpected analysis interpreter: {sys.executable}")
    return sklearn_version


def primary_candidate_rows(
    baseline_metrics: Mapping[str, Any],
    *,
    budget: float,
    branch_choices: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        {
            "name": "base",
            "branch": "base",
            "budget": float(budget),
            "safe_threshold": None,
            "cost_weight": None,
            "base_mode": "frozen_q90_original",
            **dict(baseline_metrics),
        }
    ]
    rows.extend(
        dict(branch_choices[branch])
        for branch in ("maur", "reversible")
        if str(branch_choices[branch]["name"]) != "base"
    )
    return rows


def train_and_calibrate(config: dict[str, Any]) -> dict[str, Any]:
    validate_roles(config["roles"])
    output = Path(config["output_root"])
    if not (output / "runs/FIT_CACHE_COMPLETED").is_file():
        raise RuntimeError("calibration requires FIT_CACHE_COMPLETED")
    if (output / "runs/AUDIT_COMPLETED").exists():
        raise RuntimeError("cannot recalibrate after audit")
    calibration_summary = output / "runs/calibration/summary.json"
    if (output / "runs/CALIBRATION_COMPLETED").is_file():
        return read_json(calibration_summary)
    sklearn_version = validate_analysis_environment(config)
    preflight_path = output / "runs/preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("calibration requires frozen preflight inventory")
    preflight = read_json(preflight_path)
    if preflight.get("protocol_id") != config["protocol_id"]:
        raise RuntimeError("preflight protocol drift")
    expected_calibration_hashes = {
        key: value
        for key, value in preflight["frozen_cache_summary_sha256"].items()
        if str(key).startswith("calibration:")
    }
    validate_hash_inventory(
        candidate_summary_hashes(config, "calibration"),
        expected_calibration_hashes,
        label="calibration cache",
    )

    calibration, calibration_base, base_metadata = load_role(
        config, "calibration"
    )
    calibration_raw = load_raw_candidates(config, "calibration")
    reversible_choice, fusion_rows = choose_reversible_fusion(
        config, calibration, calibration_base, calibration_raw
    )
    write_csv(output / "runs/calibration/reversible_fusion_scan.csv", fusion_rows)

    fit, fit_base, _ = load_role(config, "fit")
    fit_raw = load_raw_candidates(config, "fit")
    fusion_parameters = {
        "maur": None,
        "reversible": reversible_choice["parameters"],
    }
    models = {}
    fit_stats = {}
    for branch, parameters in fusion_parameters.items():
        details = build_details(
            fit,
            fit_base,
            fit_raw,
            fusion=branch,
            reversible_parameters=parameters,
        )
        rows = action_rows(
            fit,
            fit_base,
            details,
            epsilon=float(config["model"]["epsilon"]),
        )
        models[branch] = fit_models(rows, config)
        fit_stats[branch] = {
            "rows": int(len(rows["labels"])),
            "images": int(len(set(map(int, rows["image_ids"])))),
            "safe_rate": float(np.mean(rows["labels"])),
            "gain_mean": float(np.mean(rows["gains"])),
            "gain_std": float(np.std(rows["gains"])),
            "feature_dim": int(rows["features"].shape[1]),
        }
        del rows, details

    model_path = output / "runs/models/joint_action_models.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(models, handle)

    all_rows = []
    branch_choices: dict[str, dict[str, Any]] = {
        "maur": {},
        "reversible": {},
    }
    calibration_details = {}
    for branch, parameters in fusion_parameters.items():
        details = build_details(
            calibration,
            calibration_base,
            calibration_raw,
            fusion=branch,
            reversible_parameters=parameters,
        )
        calibration_details[branch] = details
        rows = action_rows(
            calibration,
            calibration_base,
            details,
            epsilon=float(config["model"]["epsilon"]),
        )
        options = predict_options(rows, models[branch])
        for budget in map(float, config["scheduler"]["roi_budgets"]):
            attempts, chosen = calibration_rows(
                config,
                branch,
                calibration,
                calibration_base,
                details,
                options,
                budget,
            )
            all_rows.extend(attempts)
            branch_choices[branch][f"{budget:.2f}"] = compact_choice(chosen)

    baseline_metrics = evaluate_actions(
        calibration, calibration_base, calibration_details["maur"], calibration_base
    )
    primary_choices = {}
    for budget in map(float, config["scheduler"]["roi_budgets"]):
        candidates = primary_candidate_rows(
            baseline_metrics,
            budget=budget,
            branch_choices={
                branch: branch_choices[branch][f"{budget:.2f}"]
                for branch in ("maur", "reversible")
            },
        )
        primary_choices[f"{budget:.2f}"] = compact_choice(
            choose_ap_safe_configuration(
                candidates,
                baseline_name="base",
                tolerance=float(config["evaluation"]["ap_tolerance"]),
            )
        )

    write_csv(output / "runs/calibration/scheduler_grid.csv", all_rows)
    freeze = {
        "protocol_id": config["protocol_id"],
        "protocol_config_sha256": json_sha256(config),
        "analysis_code_sha256": analysis_code_hashes(),
        "preflight_sha256": file_sha256(preflight_path),
        "model_sha256": file_sha256(model_path),
        "model_path": str(model_path),
        "base_metadata": base_metadata,
        "reversible_fusion": reversible_choice,
        "fit_stats": fit_stats,
        "branch_choices": branch_choices,
        "primary_choices": primary_choices,
        "calibration_base": baseline_metrics,
        "calibration_inputs": {
            "fit_samples_sha256": file_sha256(
                Path(config["source_cloudv2_root"])
                / "runs/canonical_oregon_event_v1/common/oregon/fit/samples.json"
            ),
            "calibration_samples_sha256": file_sha256(
                Path(config["source_cloudv2_root"])
                / "runs/canonical_oregon_event_v1/common/oregon/calibration/samples.json"
            ),
        },
        "analysis_environment": {
            "python": sys.executable,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn_version,
        },
        "audit_used_for_selection": False,
        "registered_final_test_accessed": False,
    }
    freeze_path = output / "runs/CALIBRATION_FROZEN.json"
    write_json(freeze_path, freeze)
    result = {
        "stage": "calibrate",
        "status": "completed",
        "reversible_fusion": reversible_choice,
        "branch_choices": branch_choices,
        "primary_choices": primary_choices,
        "calibration_freeze_sha256": file_sha256(freeze_path),
        "audit_used_for_selection": False,
        "registered_final_test_accessed": False,
    }
    finalize_stage(
        calibration_summary, result, output / "runs/CALIBRATION_COMPLETED"
    )
    return result


def audit_frozen(config: dict[str, Any]) -> dict[str, Any]:
    validate_roles(config["roles"])
    validate_analysis_environment(config)
    output = Path(config["output_root"])
    freeze_path = output / "runs/CALIBRATION_FROZEN.json"
    if not freeze_path.is_file() or not (
        output / "runs/CALIBRATION_COMPLETED"
    ).is_file():
        raise RuntimeError("audit requires frozen calibration")
    final_report_path = output / "runs/audit/final_report.json"
    if (output / "runs/AUDIT_COMPLETED").is_file():
        return read_json(final_report_path)

    freeze = read_json(freeze_path)
    if freeze["protocol_id"] != config["protocol_id"]:
        raise RuntimeError("calibration protocol drift")
    if freeze["protocol_config_sha256"] != json_sha256(config):
        raise RuntimeError("protocol configuration drift")
    validate_hash_inventory(
        analysis_code_hashes(),
        freeze["analysis_code_sha256"],
        label="analysis code",
    )
    preflight_path = output / "runs/preflight.json"
    if file_sha256(preflight_path) != freeze["preflight_sha256"]:
        raise RuntimeError("preflight inventory drift")
    preflight = read_json(preflight_path)
    expected_audit_hashes = {
        key: value
        for key, value in preflight["frozen_cache_summary_sha256"].items()
        if str(key).startswith("audit:")
    }
    validate_hash_inventory(
        candidate_summary_hashes(config, "audit"),
        expected_audit_hashes,
        label="audit cache",
    )
    model_path = Path(freeze["model_path"])
    if file_sha256(model_path) != freeze["model_sha256"]:
        raise RuntimeError("frozen model hash drift")
    with model_path.open("rb") as handle:
        models = pickle.load(handle)

    audit, base, base_metadata = load_role(config, "audit")
    if base_metadata != freeze["base_metadata"]:
        raise RuntimeError("Q90/Original base metadata drift")
    raw = load_raw_candidates(config, "audit")
    parameters = {
        "maur": None,
        "reversible": freeze["reversible_fusion"]["parameters"],
    }
    branch_details = {}
    branch_options = {}
    for branch, fusion_parameters in parameters.items():
        details = build_details(
            audit,
            base,
            raw,
            fusion=branch,
            reversible_parameters=fusion_parameters,
        )
        rows = action_rows(
            audit,
            base,
            details,
            epsilon=float(config["model"]["epsilon"]),
        )
        branch_details[branch] = details
        branch_options[branch] = predict_options(rows, models[branch])
        del rows

    baseline = evaluate_actions(audit, base, branch_details["maur"], base)
    baseline_predictions = flat_predictions(audit, branch_details["maur"], base)
    baseline_class_ap = per_class_ap(audit, baseline_predictions)
    branch_results = []
    primary_results = []
    class_rows = []
    bootstrap_rows = []

    for budget in map(float, config["scheduler"]["roi_budgets"]):
        budget_key = f"{budget:.2f}"
        frozen_branch_actions = {}
        frozen_branch_metrics = {}
        for branch in ("maur", "reversible"):
            choice = freeze["branch_choices"][branch][budget_key]
            actions = actions_from_choice(
                choice,
                options=branch_options[branch],
                base=base,
                budget=budget,
            )
            metrics = evaluate_actions(
                audit, base, branch_details[branch], actions
            )
            row = {
                "scope": "branch",
                "branch": branch,
                "budget": budget,
                "calibration_choice": choice["name"],
                "map_delta_vs_base": float(metrics["official_map50_95"])
                - float(baseline["official_map50_95"]),
                "kb_delta_vs_base": float(metrics["request_kb_per_image"])
                - float(baseline["request_kb_per_image"]),
                **metrics,
            }
            branch_results.append(row)
            frozen_branch_actions[branch] = actions
            frozen_branch_metrics[branch] = metrics

        primary_choice = freeze["primary_choices"][budget_key]
        primary_branch = str(primary_choice["branch"])
        if str(primary_choice["name"]) == "base" or primary_branch == "base":
            primary_branch = "maur"
            primary_actions = dict(base)
            primary_metrics = dict(baseline)
        else:
            primary_actions = frozen_branch_actions[primary_branch]
            primary_metrics = frozen_branch_metrics[primary_branch]
        primary_row = {
            "scope": "primary",
            "branch": primary_branch,
            "budget": budget,
            "calibration_choice": primary_choice["name"],
            "map_delta_vs_base": float(primary_metrics["official_map50_95"])
            - float(baseline["official_map50_95"]),
            "kb_delta_vs_base": float(primary_metrics["request_kb_per_image"])
            - float(baseline["request_kb_per_image"]),
            **primary_metrics,
        }
        primary_results.append(primary_row)
        write_json(
            output / f"runs/audit/primary_actions_{budget_key}.json",
            {str(image_id): action for image_id, action in sorted(primary_actions.items())},
        )

        candidate_predictions = flat_predictions(
            audit, branch_details[primary_branch], primary_actions
        )
        candidate_class_ap = per_class_ap(audit, candidate_predictions)
        class_rows.extend(
            compare_class_ap(
                baseline_class_ap,
                candidate_class_ap,
                budget=budget,
                branch=primary_branch,
                minimum_instances=int(
                    config["evaluation"]["minimum_class_instances"]
                ),
                negative_delta=float(
                    config["evaluation"]["class_negative_delta"]
                ),
            )
        )
        bootstrap_rows.append(
            paired_bootstrap(
                config,
                audit,
                baseline_actions=base,
                candidate_actions=primary_actions,
                baseline_details=branch_details["maur"],
                candidate_details=branch_details[primary_branch],
                budget=budget,
                branch=primary_branch,
                checkpoint=output
                / f"runs/audit/bootstrap_{budget_key}_checkpoint.json",
            )
        )

    write_csv(output / "runs/audit/results.csv", [*branch_results, *primary_results])
    write_csv(output / "runs/audit/class_ap.csv", class_rows)
    write_csv(output / "runs/audit/paired_bootstrap.csv", bootstrap_rows)
    negative_groups = [row for row in class_rows if row["significant_negative"]]
    tolerance = float(config["evaluation"]["ap_tolerance"])
    successful_points = [
        row
        for row in primary_results
        if float(row["map_delta_vs_base"]) >= -tolerance
        and float(row["kb_delta_vs_base"]) < 0.0
    ]
    teacher_report_path = Path(config["source_scan_root"]) / "runs/final_report.json"
    result = {
        "protocol_id": config["protocol_id"],
        "status": "completed",
        "baseline": baseline,
        "branch_results": branch_results,
        "primary_results": primary_results,
        "paired_bootstrap": bootstrap_rows,
        "class_gate": {
            "minimum_instances": int(
                config["evaluation"]["minimum_class_instances"]
            ),
            "negative_delta": float(config["evaluation"]["class_negative_delta"]),
            "significant_negative_groups": negative_groups,
            "passed": not negative_groups,
        },
        "event_bootstrap": {
            "status": "unavailable",
            "reason": (
                "No role-specific event manifest exists; the global assignments "
                "file also contains sealed final identities and was not read."
            ),
        },
        "teacher_ceiling_reference": (
            {
                "path": str(teacher_report_path),
                "sha256": file_sha256(teacher_report_path),
                "frozen_multi_candidate": read_json(teacher_report_path)[
                    "frozen_multi_candidate"
                ],
            }
            if teacher_report_path.is_file()
            else None
        ),
        "successful_work_points": len(successful_points),
        "success": bool(successful_points),
        "calibration_freeze_sha256": file_sha256(freeze_path),
        "audit_used_for_selection": False,
        "registered_final_test_accessed": False,
    }
    finalize_stage(final_report_path, result, output / "runs/AUDIT_COMPLETED")
    return result


def actions_from_choice(
    choice: Mapping[str, Any],
    *,
    options: dict[int, list[dict[str, Any]]],
    base: dict[int, str],
    budget: float,
) -> dict[int, str]:
    if str(choice["name"]) == "base":
        return dict(base)
    return select_joint_actions(
        options,
        frozen_base=base,
        roi_budget=float(budget),
        safe_threshold=float(choice["safe_threshold"]),
        cost_weight=float(choice["cost_weight"]),
        base_mode=str(choice["base_mode"]),
    )


def per_class_ap(
    data: dict[str, Any], predictions: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    wrapper = data["evaluator"]
    result = wrapper.annotation.loadRes(predictions)
    if wrapper.faster:
        evaluator = wrapper.evaluator_type(
            wrapper.annotation,
            result,
            iouType="bbox",
            print_function=lambda *args, **kwargs: None,
        )
    else:
        evaluator = wrapper.evaluator_type(wrapper.annotation, result, iouType="bbox")
    evaluator.params.imgIds = list(map(int, data["image_ids"]))
    evaluator.evaluate()
    evaluator.accumulate()
    precision = np.asarray(evaluator.eval["precision"], dtype=np.float64)
    gt = read_json(data["gt_path"])
    names = {
        int(category["id"]): str(category["name"])
        for category in gt["categories"]
    }
    instances = Counter(
        int(annotation["category_id"]) for annotation in gt["annotations"]
    )
    output = {}
    for category_index, category_id in enumerate(map(int, evaluator.params.catIds)):
        values = precision[:, :, category_index, 0, -1]
        valid = values[values > -1.0]
        output[category_id] = {
            "category_id": category_id,
            "category_name": names.get(category_id, str(category_id)),
            "instances": int(instances[category_id]),
            "map50_95": float(valid.mean()) if valid.size else None,
        }
    return output


def compare_class_ap(
    baseline: Mapping[int, Mapping[str, Any]],
    candidate: Mapping[int, Mapping[str, Any]],
    *,
    budget: float,
    branch: str,
    minimum_instances: int,
    negative_delta: float,
) -> list[dict[str, Any]]:
    rows = []
    for category_id in sorted(set(baseline) | set(candidate)):
        base = baseline[category_id]
        value = candidate[category_id]
        base_ap = base["map50_95"]
        candidate_ap = value["map50_95"]
        delta = (
            float(candidate_ap) - float(base_ap)
            if base_ap is not None and candidate_ap is not None
            else None
        )
        rows.append(
            {
                "budget": float(budget),
                "branch": branch,
                "category_id": int(category_id),
                "category_name": str(base["category_name"]),
                "instances": int(base["instances"]),
                "base_map50_95": base_ap,
                "candidate_map50_95": candidate_ap,
                "delta_map50_95": delta,
                "significant_negative": bool(
                    int(base["instances"]) >= int(minimum_instances)
                    and delta is not None
                    and float(delta) < float(negative_delta)
                ),
            }
        )
    return rows


def action_request_bytes(
    data: dict[str, Any],
    details: dict[str, dict[int, dict[str, Any]]],
    actions: Mapping[int, str],
) -> dict[int, int]:
    result = {}
    for image_id, action in actions.items():
        if action in BASE_ACTIONS:
            result[int(image_id)] = int(
                data["sample_by_id"][int(image_id)]["payload_bytes"][action]
            )
        else:
            result[int(image_id)] = int(details[action][int(image_id)]["payload_bytes"])
    return result


def action_map_signature(actions: Mapping[int, str]) -> str:
    encoded = json.dumps(
        sorted((int(image_id), str(action)) for image_id, action in actions.items()),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def paired_bootstrap(
    config: dict[str, Any],
    data: dict[str, Any],
    *,
    baseline_actions: dict[int, str],
    candidate_actions: dict[int, str],
    baseline_details: dict[str, dict[int, dict[str, Any]]],
    candidate_details: dict[str, dict[int, dict[str, Any]]],
    budget: float,
    branch: str,
    checkpoint: Path,
) -> dict[str, Any]:
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    seed = int(config["evaluation"]["bootstrap_seed"]) + round(
        float(budget) * 1000
    )
    signature = hashlib.sha256(
        (
            action_map_signature(baseline_actions)
            + action_map_signature(candidate_actions)
            + str(replicates)
            + str(seed)
        ).encode("ascii")
    ).hexdigest()
    if action_map_signature(baseline_actions) == action_map_signature(
        candidate_actions
    ):
        return {
            "comparison": f"{branch} joint action - frozen Q90/Original base",
            "budget": float(budget),
            "branch": branch,
            "replicates": replicates,
            "official_map50_95_delta_mean": 0.0,
            "official_map50_95_delta_ci_low": 0.0,
            "official_map50_95_delta_ci_high": 0.0,
            "official_map50_95_probability_gt_zero": 0.0,
            "request_kb_per_image_delta_mean": 0.0,
            "request_kb_per_image_delta_ci_low": 0.0,
            "request_kb_per_image_delta_ci_high": 0.0,
            "bootstrap_unit": "image_diagnostic",
            "paired": True,
            "identical_action_maps": True,
            "strategy_refit_in_bootstrap": False,
        }
    state = {
        "signature": signature,
        "ap_deltas": [],
        "kb_deltas": [],
    }
    if checkpoint.is_file():
        observed = read_json(checkpoint)
        if observed.get("signature") != signature:
            raise RuntimeError(f"bootstrap checkpoint drift: {checkpoint}")
        state = observed

    baseline_predictions = flat_predictions(data, baseline_details, baseline_actions)
    candidate_predictions = flat_predictions(data, candidate_details, candidate_actions)
    baseline_cost = action_request_bytes(data, baseline_details, baseline_actions)
    candidate_cost = action_request_bytes(data, candidate_details, candidate_actions)
    ground_truth = read_json(data["gt_path"])
    image_ids = list(map(int, data["image_ids"]))
    sampled = deterministic_bootstrap_indices(
        len(image_ids), replicates=replicates, seed=seed
    )
    evaluator_type = type(data["evaluator"])
    start = len(state["ap_deltas"])
    with tempfile.TemporaryDirectory(prefix="oregon_joint_bootstrap_") as tmp:
        root = Path(tmp)
        for replicate in range(start, replicates):
            source_ids = [image_ids[int(index)] for index in sampled[replicate]]
            sampled_gt, prediction_sets = build_bootstrap_coco_sample(
                ground_truth,
                {
                    "baseline": baseline_predictions,
                    "candidate": candidate_predictions,
                },
                sampled_image_ids=source_ids,
            )
            gt_path = root / f"gt_{replicate:04d}.json"
            write_json(gt_path, sampled_gt)
            base_eval = evaluator_type(
                gt_path, list(range(1, len(source_ids) + 1))
            )
            candidate_eval = evaluator_type(
                gt_path, list(range(1, len(source_ids) + 1))
            )
            base_ap = float(
                base_eval.evaluate(prediction_sets["baseline"])["map50_95"]
            )
            candidate_ap = float(
                candidate_eval.evaluate(prediction_sets["candidate"])["map50_95"]
            )
            state["ap_deltas"].append(candidate_ap - base_ap)
            state["kb_deltas"].append(
                (
                    sum(candidate_cost[image_id] for image_id in source_ids)
                    - sum(baseline_cost[image_id] for image_id in source_ids)
                )
                / len(source_ids)
                / 1024.0
            )
            if (replicate + 1) % 10 == 0 or replicate + 1 == replicates:
                write_json(checkpoint, state)
                print(
                    f"[bootstrap] budget={budget:.2f} "
                    f"completed={replicate + 1}/{replicates}",
                    flush=True,
                )
            gt_path.unlink(missing_ok=True)

    ap = np.asarray(state["ap_deltas"], dtype=np.float64)
    kb = np.asarray(state["kb_deltas"], dtype=np.float64)
    return {
        "comparison": f"{branch} joint action - frozen Q90/Original base",
        "budget": float(budget),
        "branch": branch,
        "replicates": replicates,
        "official_map50_95_delta_mean": float(ap.mean()),
        "official_map50_95_delta_ci_low": float(np.quantile(ap, 0.025)),
        "official_map50_95_delta_ci_high": float(np.quantile(ap, 0.975)),
        "official_map50_95_probability_gt_zero": float(np.mean(ap > 0.0)),
        "request_kb_per_image_delta_mean": float(kb.mean()),
        "request_kb_per_image_delta_ci_low": float(np.quantile(kb, 0.025)),
        "request_kb_per_image_delta_ci_high": float(np.quantile(kb, 0.975)),
        "bootstrap_unit": "image_diagnostic",
        "paired": True,
        "strategy_refit_in_bootstrap": False,
    }
