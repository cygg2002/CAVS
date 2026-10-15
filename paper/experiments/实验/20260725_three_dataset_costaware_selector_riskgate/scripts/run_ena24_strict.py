#!/usr/bin/env python3
"""Run the strict ENA24 grouped selector/calibration/risk-gate stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Any, Iterable, Mapping

import numpy as np

from ena24_strict_core import (
    apply_strict_risk_gate,
    build_near_duplicate_split_manifest,
    dhash256,
    file_sha256,
    require_strict_provisional_marker,
    subset_coco_gt,
)
from selector_core import (
    choose_provisional,
    choose_reference,
    extract_frozen_policy_candidates,
    validate_candidate_grid,
    validate_protocol,
)


SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
OUTPUT = EXP_ROOT / "runs/ena24"
SOURCE_20 = (
    REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e"
)
SOURCE_22 = (
    REPO / "paper/experiments/20260722_ena24_hierarchical_triaction_paper_exploration"
)
SOURCE_23 = (
    REPO / "paper/experiments/20260723_three_dataset_nineranker_triaction_paper_exploration"
)
CANONICAL = SOURCE_20 / "runs/canonical_v1"
COMMON = CANONICAL / "common/ena24"
SOURCE_RUN = SOURCE_23 / "runs/ena24"

STRICT_MARKER = "STRICT_PROVISIONAL_FROZEN"
STRICT_GATE_MARKER = "STRICT_RISK_GATE_COMPLETED"
LEGACY_MARKER = "PROVISIONAL_FROZEN"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in values for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _number(value: Any) -> float:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in {"inf", "+inf", "Infinity", "+Infinity"}:
            return math.inf
        if normalized in {"-inf", "-Infinity"}:
            return -math.inf
    return float(value)


def _numeric_rows(path: Path) -> list[dict[str, Any]]:
    numeric = {
        "ap",
        "ap_delta",
        "budget",
        "kb",
        "kb_delta",
        "lambda_value",
        "median_original_bytes",
        "online_threshold",
        "threshold",
        "worst_ap_delta",
    }
    rows: list[dict[str, Any]] = []
    for raw in read_csv(path):
        row: dict[str, Any] = dict(raw)
        for key in numeric:
            if key in row and str(row[key]).strip():
                row[key] = _number(row[key])
        rows.append(row)
    return rows


def strict_input_paths() -> list[Path]:
    """Static allowlist for strict stages; deliberately excludes final."""
    return [
        EXP_ROOT / "configs/protocol.json",
        SOURCE_23 / "configs/protocol.json",
        SOURCE_RUN / "CORE_COMPLETED",
        SOURCE_RUN / "models/current_controlled_32d.json",
        SOURCE_RUN / "models/saferoi_gbdt.pkl",
        SOURCE_20 / "configs/protocol.json",
        CANONICAL / "benchmark/ena24/models.json",
        CANONICAL / "benchmark/ena24/action_gate.json",
        COMMON / "calibration/source_manifest.json",
        COMMON / "calibration/samples.json",
        COMMON / "calibration/gt.json",
    ]


def _calibration_manifest_inputs() -> list[Path]:
    manifest_path = COMMON / "calibration/source_manifest.json"
    manifest = read_json(manifest_path)
    source_paths = manifest.get("source_paths", {})
    if isinstance(source_paths, Mapping):
        sources = list(source_paths)
    else:
        sources = list(source_paths)
    return [
        *strict_input_paths(),
        *map(Path, sources),
        *map(Path, manifest["action_sources"].values()),
        Path(manifest["samples"]),
        Path(manifest["gt"]),
        SCRIPT,
        EXP_ROOT / "scripts/ena24_strict_core.py",
        EXP_ROOT / "scripts/selector_core.py",
        SOURCE_23 / "scripts/run_dataset.py",
        SOURCE_22 / "scripts/run_exploration.py",
    ]


def snapshot_hashes(paths: Iterable[Path]) -> dict[str, str]:
    normalized = sorted({Path(path).resolve() for path in paths})
    missing = [str(path) for path in normalized if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing strict ENA24 inputs:\n" + "\n".join(missing))
    return {str(path): file_sha256(path) for path in normalized}


def supersede_legacy_provisional(output: Path) -> dict[str, Any]:
    """Invalidate the pre-audit full-calibration prototype marker."""
    output = Path(output)
    names = (
        "provisional_choices.json",
        "candidate_table.csv",
        "calibration_provenance.json",
        LEGACY_MARKER,
    )
    hashes = {
        name: file_sha256(output / name)
        for name in names
        if (output / name).is_file()
    }
    (output / LEGACY_MARKER).unlink(missing_ok=True)
    (output / STRICT_MARKER).unlink(missing_ok=True)
    (output / STRICT_GATE_MARKER).unlink(missing_ok=True)
    return {
        "status": "superseded_pre_audit_full_calibration_prototype",
        "legacy_marker_removed": True,
        "strict_marker_present": False,
        "artifact_sha256": hashes,
        "never_use_for_selection_or_gate": True,
    }


def finalize_identical_bootstrap(*, replicates: int) -> dict[str, Any]:
    return {
        "replicates": int(replicates),
        "official_map50_95_delta_mean": 0.0,
        "official_map50_95_delta_ci_low": 0.0,
        "official_map50_95_delta_ci_high": 0.0,
        "official_map50_95_probability_gt_zero": 0.0,
        "request_kb_per_image_delta_mean": 0.0,
        "request_kb_per_image_delta_ci_low": 0.0,
        "request_kb_per_image_delta_ci_high": 0.0,
        "bootstrap_unit": "image",
        "paired": True,
        "strategy_refit_in_bootstrap": False,
        "identical_action_shortcut": True,
    }


def _load_engine() -> tuple[Any, Any]:
    scripts = str(SOURCE_23 / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    nine = importlib.import_module("run_dataset")
    nine.configure_reused_runner("ena24", OUTPUT)
    return nine, nine.tri


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    validate_protocol(protocol)
    split = protocol["ena24_calibration_split"]
    if split.get("source_role") != "calibration":
        raise RuntimeError("Strict ENA24 source role must remain calibration")
    if split.get("near_duplicate_group_disjoint") is not True:
        raise RuntimeError("Near-duplicate group disjointness must remain enabled")
    if split.get("event_camera_disjoint") != "unknown":
        raise RuntimeError("ENA24 event/camera disjointness cannot be claimed")


def _source_records(calibration: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for sample in calibration["samples"]:
        path = Path(str(sample["identity"]))
        if not path.is_file():
            raise FileNotFoundError(f"ENA24 original identity image is missing: {path}")
        records.append(
            {
                "image_id": int(sample["image_id"]),
                "identity": str(path.resolve()),
                "class_ids": list(map(int, sample.get("class_ids", ()))),
                "dhash256": dhash256(path),
                "source_sha256": file_sha256(path),
            }
        )
    return records


def run_split(protocol: dict[str, Any]) -> dict[str, Any]:
    _validate_protocol(protocol)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    superseded = supersede_legacy_provisional(OUTPUT)
    write_json(OUTPUT / "superseded_pre_audit_prototype.json", superseded)
    for name in (
        "STRICT_SPLIT_READY",
        "strict_risk_gate_checkpoint.json",
        "strict_risk_gate.csv",
        "strict_frozen_choices.json",
    ):
        (OUTPUT / name).unlink(missing_ok=True)

    before = snapshot_hashes(_calibration_manifest_inputs())
    _, tri = _load_engine()
    calibration = tri.current.load_role("ena24", "calibration")
    expected = int(protocol["ena24_calibration_split"]["target_selector_images"]) + int(
        protocol["ena24_calibration_split"]["target_gate_images"]
    )
    if len(calibration["samples"]) != expected:
        raise RuntimeError(
            f"ENA24 calibration size drift: {len(calibration['samples'])} != {expected}"
        )
    split_config = protocol["ena24_calibration_split"]
    manifest = build_near_duplicate_split_manifest(
        _source_records(calibration),
        max_hamming=int(split_config["hamming_threshold"]),
        target_selector_images=int(split_config["target_selector_images"]),
        seed=int(split_config["seed"]),
    )
    manifest["source_role"] = "calibration"
    manifest["selector_role"] = str(split_config["selector_role"])
    manifest["gate_role"] = str(split_config["gate_role"])
    manifest["source_manifest_sha256"] = file_sha256(
        COMMON / "calibration/source_manifest.json"
    )
    write_json(OUTPUT / "strict_split_manifest.json", manifest)
    after = snapshot_hashes(_calibration_manifest_inputs())
    if before != after:
        raise RuntimeError("A frozen ENA24 input changed while creating the split")
    provenance = {
        "stage": "strict_split",
        "source_role": "calibration",
        "final_artifacts_read": False,
        "frozen_input_sha256_before": before,
        "frozen_input_sha256_after": after,
        "strict_split_manifest_sha256": file_sha256(
            OUTPUT / "strict_split_manifest.json"
        ),
        "superseded_pre_audit_prototype": superseded,
    }
    write_json(OUTPUT / "strict_split_provenance.json", provenance)
    (OUTPUT / "STRICT_SPLIT_READY").write_text("ok\n", encoding="ascii")
    return manifest


def _subset_role(
    calibration: Mapping[str, Any],
    image_ids: Iterable[int],
    *,
    role: str,
    tri: Any,
) -> dict[str, Any]:
    selected = set(map(int, image_ids))
    samples = [
        sample
        for sample in calibration["samples"]
        if int(sample["image_id"]) in selected
    ]
    if {int(sample["image_id"]) for sample in samples} != selected:
        raise RuntimeError(f"{role} samples do not match the frozen split")
    root = OUTPUT / "strict_roles" / role
    root.mkdir(parents=True, exist_ok=True)
    gt = read_json(Path(calibration["gt_path"]))
    gt_path = root / "gt.json"
    write_json(gt_path, subset_coco_gt(gt, selected))
    actions = {
        name: {
            image_id: predictions
            for image_id, predictions in action_rows.items()
            if int(image_id) in selected
        }
        for name, action_rows in calibration["actions"].items()
    }
    return {
        "dataset": "ena24",
        "role": role,
        "root": root,
        "samples": samples,
        "sample_by_id": {
            int(sample["image_id"]): sample for sample in samples
        },
        "image_ids": sorted(selected),
        "gt_path": gt_path,
        "gt_map": {
            int(image_id): rows
            for image_id, rows in calibration["gt_map"].items()
            if int(image_id) in selected
        },
        "actions": actions,
        "evaluator": tri.current.CocoEvaluator(gt_path, sorted(selected)),
    }


def _load_strict_roles(tri: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(OUTPUT / "strict_split_manifest.json")
    for image in manifest["images"]:
        path = Path(image["identity"])
        if file_sha256(path) != image["source_sha256"]:
            raise RuntimeError(f"ENA24 source image changed after split: {path}")
    calibration = tri.current.load_role("ena24", "calibration")
    selector = _subset_role(
        calibration,
        manifest["selector_image_ids"],
        role="near_duplicate_grouped_selector",
        tri=tri,
    )
    gate = _subset_role(
        calibration,
        manifest["gate_image_ids"],
        role="near_duplicate_grouped_gate",
        tri=tri,
    )
    if set(selector["image_ids"]) & set(gate["image_ids"]):
        raise RuntimeError("Selector and gate image IDs overlap")
    return selector, gate


def _load_frozen_state(
    nine: Any,
    tri: Any,
    selector: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    models = tri.load_frozen_models(
        CANONICAL / "benchmark/ena24/models.json"
    )
    controlled = read_json(SOURCE_RUN / "models/current_controlled_32d.json")
    action_gate = read_json(CANONICAL / "benchmark/ena24/action_gate.json")
    with (SOURCE_RUN / "models/saferoi_gbdt.pkl").open("rb") as handle:
        roi_model = pickle.load(handle)

    roles = {"selector": selector, "gate": gate}
    scores = {}
    base_actions = {}
    roi_scores = {}
    for role, data in roles.items():
        role_scores = tri.current.ranker_scores(models, data)
        role_scores["Current-Controlled-32D"] = (
            tri.current.predict_pairwise_ranker(
                controlled,
                np.asarray(
                    [sample["pca32"] for sample in data["samples"]],
                    dtype=np.float64,
                ),
            )
        )
        scores[role] = role_scores
        base_actions[role] = tri.current.chosen_actions(
            models["action_heads"], action_gate, data
        )
        roi_scores[role] = tri.predict_gbdt(
            roi_model, tri.profile_features(data)
        )
    return {
        "models": models,
        "scores": scores,
        "base_actions": base_actions,
        "roi_scores": roi_scores,
    }


def _strict_policy_candidates(
    *,
    protocol: Mapping[str, Any],
    flattened: list[dict[str, Any]],
    choices: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rankers = tuple(map(str, protocol["selectable_rankers"]))
    policies = tuple(map(str, protocol["policies"]))
    policy_order = tuple(map(str, protocol["policy_tie_break_order"]))
    tolerance = float(protocol["ap_tolerance"])
    rows = []
    candidate_table = []
    for mode, budgets in protocol["work_points"].items():
        gbdt_threshold = float(choices[mode]["threshold"])
        for budget in map(float, budgets):
            candidates = extract_frozen_policy_candidates(
                flattened,
                mode=mode,
                budget=budget,
                rankers=rankers,
                gbdt_threshold=gbdt_threshold,
            )
            validate_candidate_grid(
                candidates, rankers=rankers, policies=policies
            )
            for candidate in candidates:
                candidate.update(
                    {
                        "dataset": "ena24",
                        "selection_source_role": (
                            "near_duplicate_grouped_selector"
                        ),
                    }
                )
            reference = choose_reference(candidates, ranker_order=rankers)
            provisional = choose_provisional(
                candidates,
                reference=reference,
                mode=mode,
                ap_tolerance=tolerance,
                ranker_order=rankers,
                policy_order=policy_order,
            )
            rows.append(
                {
                    "dataset": "ena24",
                    "mode": mode,
                    "budget": budget,
                    "selection_source_role": (
                        "near_duplicate_grouped_selector"
                    ),
                    "reference": dict(reference),
                    "provisional": dict(provisional),
                    "point_ap_delta": float(provisional["ap"])
                    - float(reference["ap"]),
                    "point_kb_delta": float(provisional["kb"])
                    - float(reference["kb"]),
                }
            )
            candidate_table.extend(candidates)
    if len(rows) != int(protocol["expected_work_points_per_dataset"]):
        raise RuntimeError(f"Strict ENA24 choice count drift: {len(rows)}")
    expected = len(rows) * int(protocol["expected_candidates_per_work_point"])
    if len(candidate_table) != expected:
        raise RuntimeError(
            f"Strict ENA24 candidate count drift: {len(candidate_table)} != {expected}"
        )
    return rows, candidate_table


def run_calibrate(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    _validate_protocol(protocol)
    if not (OUTPUT / "STRICT_SPLIT_READY").is_file():
        raise RuntimeError("Run the strict ENA24 split stage first")
    supersede_legacy_provisional(OUTPUT)
    before = snapshot_hashes(_calibration_manifest_inputs())
    nine, tri = _load_engine()
    selector, gate = _load_strict_roles(tri)
    state = _load_frozen_state(nine, tri, selector, gate)
    rankers = tuple(map(str, protocol["selectable_rankers"]))
    score_by_id = {
        int(sample["image_id"]): float(score)
        for sample, score in zip(
            selector["samples"], state["roi_scores"]["selector"]
        )
    }
    thresholds = tri.thresholds_for_fractions(
        score_by_id,
        tri.eligible_roi_ids(
            selector, state["base_actions"]["selector"]
        ),
        map(
            float,
            read_json(SOURCE_23 / "configs/protocol.json")[
                "threshold_fractions"
            ],
        ),
    )
    if math.inf not in thresholds or -math.inf not in thresholds:
        raise RuntimeError("Strict ROI threshold grid lacks disabled/all controls")

    fixed_call = tri.evaluate_fixed_call_candidates(
        selector,
        state["scores"]["selector"],
        state["base_actions"]["selector"],
        state["roi_scores"]["selector"],
        thresholds,
        rankers,
        map(float, protocol["work_points"]["fixed_call"]),
    )
    online, frame_thresholds = tri.evaluate_online_candidates(
        selector,
        state["scores"]["selector"],
        state["base_actions"]["selector"],
        state["roi_scores"]["selector"],
        thresholds,
        rankers,
        map(float, protocol["work_points"]["online"]),
    )
    source_config = read_json(SOURCE_20 / "configs/protocol.json")
    fixed_byte = tri.evaluate_fixed_byte_candidates(
        selector,
        state["scores"]["selector"],
        state["base_actions"]["selector"],
        state["roi_scores"]["selector"],
        thresholds,
        rankers,
        map(float, protocol["work_points"]["fixed_byte"]),
        source_config,
    )
    tolerance = float(protocol["ap_tolerance"])
    choices = {
        "fixed_call": tri.choose_cost_candidate(
            fixed_call, ap_tolerance=tolerance
        ),
        "online": tri.choose_cost_candidate(
            online, ap_tolerance=tolerance
        ),
        "fixed_byte": tri.choose_fixed_byte_candidate(
            fixed_byte, ap_tolerance=tolerance
        ),
    }
    flattened = [
        *tri.flatten_candidates("fixed_call", fixed_call, tolerance),
        *tri.flatten_candidates("online", online, tolerance),
        *tri.flatten_candidates("fixed_byte", fixed_byte, tolerance),
    ]
    provisional, candidate_table = _strict_policy_candidates(
        protocol=protocol,
        flattened=flattened,
        choices=choices,
    )
    write_csv(OUTPUT / "strict_calibration_curve.csv", flattened)
    write_csv(OUTPUT / "strict_candidate_table.csv", candidate_table)
    write_json(
        OUTPUT / "strict_threshold_choices.json",
        {
            "shared": choices,
            "calibration_source_role": "near_duplicate_grouped_selector",
            "online_frame_thresholds": [
                {
                    "ranker": ranker,
                    "budget": budget,
                    "threshold": threshold,
                }
                for (ranker, budget), threshold in sorted(
                    frame_thresholds.items()
                )
            ],
            "fixed_byte_lambda_source_role": (
                "near_duplicate_grouped_selector"
            ),
        },
    )
    write_json(OUTPUT / "strict_provisional_choices.json", provisional)
    after = snapshot_hashes(_calibration_manifest_inputs())
    if before != after:
        raise RuntimeError("A frozen ENA24 input changed during strict calibration")
    provenance = {
        "stage": "strict_calibration",
        "selector_images": len(selector["samples"]),
        "gate_images_not_used_for_selection": len(gate["samples"]),
        "selection_source_role": "near_duplicate_grouped_selector",
        "recalibration_sources": {
            "roi_threshold": "near_duplicate_grouped_selector",
            "online_frame_threshold": "near_duplicate_grouped_selector",
            "fixed_byte_lambda": "near_duplicate_grouped_selector",
        },
        "frozen_fit_models_reused": True,
        "final_artifacts_read": False,
        "legacy_pre_audit_provisional_used": False,
        "frozen_input_sha256_before": before,
        "frozen_input_sha256_after": after,
        "strict_split_manifest_sha256": file_sha256(
            OUTPUT / "strict_split_manifest.json"
        ),
        "strict_provisional_choices_sha256": file_sha256(
            OUTPUT / "strict_provisional_choices.json"
        ),
    }
    write_json(OUTPUT / "strict_calibration_provenance.json", provenance)
    write_json(
        OUTPUT / STRICT_MARKER,
        {
            "status": "strict_provisional_frozen",
            "strict_provisional_choices_sha256": provenance[
                "strict_provisional_choices_sha256"
            ],
            "legacy_pre_audit_provisional_used": False,
        },
    )
    return provisional


def _candidate_lookup() -> dict[tuple[str, float, str, str], dict[str, Any]]:
    rows = _numeric_rows(OUTPUT / "strict_candidate_table.csv")
    return {
        (
            str(row["mode"]),
            float(row["budget"]),
            str(row["ranker"]),
            str(row["policy"]),
        ): row
        for row in rows
    }


def _policy_thresholds() -> dict[str, dict[str, float]]:
    shared = read_json(OUTPUT / "strict_threshold_choices.json")["shared"]
    return {
        mode: {
            "SharedAction": math.inf,
            "CostEligible-AllROI": -math.inf,
            "GBDT-Hierarchical": _number(shared[mode]["threshold"]),
        }
        for mode in ("fixed_call", "online", "fixed_byte")
    }


def _selected_actions(
    *,
    tri: Any,
    data: Mapping[str, Any],
    selector_scores: np.ndarray,
    gate_scores: np.ndarray,
    base_actions: Mapping[int, str],
    roi_scores: np.ndarray,
    mode: str,
    budget: float,
    candidate: Mapping[str, Any],
    threshold: float,
) -> dict[int, str]:
    action_map = tri.action_map_at_threshold(
        data, base_actions, roi_scores, threshold
    )
    if mode == "fixed_call":
        return tri.current.top_k_actions(data, gate_scores, action_map, budget)
    if mode == "online":
        return tri.selected_online(
            data,
            gate_scores,
            action_map,
            _number(candidate["online_threshold"]),
        )
    priorities = tri.current.audit_priority(
        selector_scores,
        gate_scores,
        data,
        action_map,
        _number(candidate["lambda_value"]),
        _number(candidate["median_original_bytes"]),
    )
    return tri.current.byte_selection(data, priorities, action_map, budget)


def _gate_identity() -> dict[str, Any]:
    return {
        "dataset": "ena24",
        "strict_split_manifest_sha256": file_sha256(
            OUTPUT / "strict_split_manifest.json"
        ),
        "strict_provisional_choices_sha256": file_sha256(
            OUTPUT / "strict_provisional_choices.json"
        ),
    }


def strict_gate_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    bootstrap = {
        key: value
        for key, value in row["bootstrap"].items()
        if key not in {"dataset", "mode", "budget", "nominal_budget", "comparison"}
    }
    return {
        **bootstrap,
        "dataset": row["dataset"],
        "mode": row["mode"],
        "budget": row["budget"],
        "reference_ranker": row["reference"]["ranker"],
        "reference_policy": row["reference"]["policy"],
        "provisional_ranker": row["provisional"]["ranker"],
        "provisional_policy": row["provisional"]["policy"],
        "frozen_ranker": row["frozen"]["ranker"],
        "frozen_policy": row["frozen"]["policy"],
        "gate_passed": row["gate_passed"],
        "used_reference_fallback": row["used_reference_fallback"],
    }


def run_risk_gate(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    _validate_protocol(protocol)
    require_strict_provisional_marker(OUTPUT)
    if (OUTPUT / LEGACY_MARKER).exists():
        raise RuntimeError("Superseded PROVISIONAL_FROZEN marker must be absent")
    marker = read_json(OUTPUT / STRICT_MARKER)
    expected_sha = file_sha256(OUTPUT / "strict_provisional_choices.json")
    if marker.get("strict_provisional_choices_sha256") != expected_sha:
        raise RuntimeError("Strict provisional marker does not match choices")

    before = snapshot_hashes(_calibration_manifest_inputs())
    nine, tri = _load_engine()
    selector, gate = _load_strict_roles(tri)
    state = _load_frozen_state(nine, tri, selector, gate)
    choices = read_json(OUTPUT / "strict_provisional_choices.json")
    candidates = _candidate_lookup()
    thresholds = _policy_thresholds()
    identity = _gate_identity()
    checkpoint_path = OUTPUT / "strict_risk_gate_checkpoint.json"
    if checkpoint_path.is_file():
        checkpoint = read_json(checkpoint_path)
        if {key: checkpoint.get(key) for key in identity} != identity:
            raise RuntimeError("Strict risk-gate checkpoint identity changed")
        rows = list(checkpoint.get("rows", []))
    else:
        rows = []
    completed = {
        (str(row["mode"]), float(row["budget"])) for row in rows
    }
    cached = {
        str(row["action_pair_sha256"]): dict(row["bootstrap"])
        for row in rows
        if row.get("action_pair_sha256")
    }
    replicates = int(protocol["bootstrap_replicates"])
    seed = int(protocol["bootstrap_seed"])

    for choice in choices:
        mode = str(choice["mode"])
        budget = float(choice["budget"])
        key = (mode, budget)
        if key in completed:
            continue
        reference = dict(choice["reference"])
        provisional = dict(choice["provisional"])
        reference_candidate = candidates[
            (mode, budget, reference["ranker"], reference["policy"])
        ]
        provisional_candidate = candidates[
            (mode, budget, provisional["ranker"], provisional["policy"])
        ]
        reference_actions = _selected_actions(
            tri=tri,
            data=gate,
            selector_scores=state["scores"]["selector"][
                reference["ranker"]
            ],
            gate_scores=state["scores"]["gate"][reference["ranker"]],
            base_actions=state["base_actions"]["gate"],
            roi_scores=state["roi_scores"]["gate"],
            mode=mode,
            budget=budget,
            candidate=reference_candidate,
            threshold=thresholds[mode][reference["policy"]],
        )
        provisional_actions = _selected_actions(
            tri=tri,
            data=gate,
            selector_scores=state["scores"]["selector"][
                provisional["ranker"]
            ],
            gate_scores=state["scores"]["gate"][provisional["ranker"]],
            base_actions=state["base_actions"]["gate"],
            roi_scores=state["roi_scores"]["gate"],
            mode=mode,
            budget=budget,
            candidate=provisional_candidate,
            threshold=thresholds[mode][provisional["policy"]],
        )
        action_digest = tri.action_pair_digest(
            reference_actions, provisional_actions
        )
        if reference_actions == provisional_actions:
            bootstrap = finalize_identical_bootstrap(replicates=replicates)
        elif action_digest in cached:
            bootstrap = dict(cached[action_digest])
            bootstrap["reused_identical_action_pair"] = True
        else:
            bootstrap = tri.paired_bootstrap(
                gate,
                baseline_actions=reference_actions,
                candidate_actions=provisional_actions,
                replicates=replicates,
                seed=seed,
                unit=str(protocol["bootstrap_unit"]),
            )
            cached[action_digest] = dict(bootstrap)
        decision = apply_strict_risk_gate(
            reference=reference,
            provisional=provisional,
            mode=mode,
            ap_delta_ci_low=float(
                bootstrap["official_map50_95_delta_ci_low"]
            ),
            kb_delta_ci_high=float(
                bootstrap["request_kb_per_image_delta_ci_high"]
            ),
            ap_tolerance=float(protocol["ap_tolerance"]),
        )
        row = {
            "dataset": "ena24",
            "mode": mode,
            "budget": budget,
            "gate_source_role": "near_duplicate_grouped_gate",
            "bootstrap_seed": seed,
            "bootstrap": bootstrap,
            "action_pair_sha256": action_digest,
            **decision,
        }
        rows.append(row)
        completed.add(key)
        write_json(checkpoint_path, {**identity, "rows": rows})
        write_json(OUTPUT / "strict_risk_gate.json", rows)

    expected = int(protocol["expected_work_points_per_dataset"])
    if len(rows) != expected or len(completed) != expected:
        raise RuntimeError(f"Strict risk-gate row count drift: {len(rows)}")
    frozen = [
        {
            "dataset": "ena24",
            "mode": row["mode"],
            "budget": row["budget"],
            "frozen": row["frozen"],
            "gate_passed": row["gate_passed"],
            "used_reference_fallback": row["used_reference_fallback"],
            "selection_source_role": "near_duplicate_grouped_selector",
            "gate_source_role": "near_duplicate_grouped_gate",
        }
        for row in rows
    ]
    write_json(OUTPUT / "strict_frozen_choices.json", frozen)
    write_csv(
        OUTPUT / "strict_risk_gate.csv",
        [strict_gate_csv_row(row) for row in rows],
    )
    after = snapshot_hashes(_calibration_manifest_inputs())
    if before != after:
        raise RuntimeError("A frozen ENA24 input changed during strict risk gate")
    provenance = {
        "stage": "strict_risk_gate",
        "selection_source_role": "near_duplicate_grouped_selector",
        "gate_source_role": "near_duplicate_grouped_gate",
        "bootstrap_replicates": replicates,
        "pointwise_not_simultaneous": True,
        "direct_reference_fallback_without_second_candidate": True,
        "final_artifacts_read": False,
        "legacy_pre_audit_provisional_used": False,
        "frozen_input_sha256_before": before,
        "frozen_input_sha256_after": after,
        **identity,
        "strict_frozen_choices_sha256": file_sha256(
            OUTPUT / "strict_frozen_choices.json"
        ),
    }
    write_json(OUTPUT / "strict_risk_gate_provenance.json", provenance)
    write_json(
        OUTPUT / STRICT_GATE_MARKER,
        {
            "status": "strict_risk_gate_complete",
            "strict_frozen_choices_sha256": provenance[
                "strict_frozen_choices_sha256"
            ],
        },
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("split", "calibrate", "risk-gate", "all"),
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = read_json(EXP_ROOT / "configs/protocol.json")
    if args.stage in {"split", "all"}:
        result = run_split(protocol)
    if args.stage in {"calibrate", "all"}:
        result = run_calibrate(protocol)
    if args.stage in {"risk-gate", "all"}:
        result = run_risk_gate(protocol)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()
