"""Pure data, scheduling, and gate contracts for the OOF diagnostic."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
sys.path.insert(0, str(REPO / "paper/experiments/GAIR_v5"))

from gair.selector import select_actions_for_budget  # noqa: E402


ACTIONS = ("edge", "original", "q90", "roi")
CLOUD_ACTIONS = ACTIONS[1:]
METHODS = (
    "CurrentDualHead-SharedAction",
    "ActionValue-Joint",
    "ActionValueRiskLCB-Joint",
)
SCORE_FIELDS = ("baseline", "value", "risk_mean", "risk_lower")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if set(protocol.get("roles", {})) != {"fit", "calibration"}:
        raise ValueError("Only fit and calibration roles are permitted")
    if int(protocol["roles"]["fit"]) != 9678 or int(
        protocol["roles"]["calibration"]
    ) != 571:
        raise ValueError("Canonical role populations changed")
    if int(protocol.get("fold_count", -1)) != 5:
        raise ValueError("The diagnostic requires five OOF folds")
    if tuple(protocol.get("actions", ())) != ACTIONS:
        raise ValueError("The frozen action space changed")
    if tuple(protocol.get("methods", ())) != METHODS:
        raise ValueError("The frozen method matrix changed")
    if tuple(map(float, protocol.get("budgets", ()))) != (0.1, 0.25, 0.4):
        raise ValueError("The frozen budgets changed")
    if tuple(protocol.get("modes", ())) != ("fixed_call", "fixed_byte"):
        raise ValueError("The frozen evaluation modes changed")
    if protocol.get("fit_only_training") is not True:
        raise ValueError("Training must remain fit-only")
    if protocol.get("calibration_used_for_training") is not False:
        raise ValueError("Calibration cannot train the models")
    for flag in ("audit_accessed", "final_accessed", "registered_final_accessed"):
        if protocol.get(flag) is not False:
            raise ValueError(f"{flag} must remain false")
    model = protocol.get("model", {})
    if tuple(map(int, model.get("seeds", ()))) != (20260805,):
        raise ValueError("This go/no-go diagnostic uses one frozen seed")
    if int(model.get("pairs_per_budget", -1)) != 4000:
        raise ValueError("The canonical boundary-pair count changed")
    if float(model.get("boundary_weight", -1.0)) != 1.0:
        raise ValueError("The canonical boundary-loss weight changed")
    if float(model.get("quantile", -1.0)) != 0.1:
        raise ValueError("The lower reward quantile changed")


def numeric_score_arrays(
    score_artifact: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Read only the frozen numeric score fields, ignoring checkpoint metadata."""
    scores = score_artifact.get("scores")
    if not isinstance(scores, Mapping):
        raise ValueError("Score artifact is missing its score mapping")
    missing = [name for name in SCORE_FIELDS if name not in scores]
    if missing:
        raise ValueError(f"Score artifact is missing fields: {missing}")
    return {
        name: np.asarray(scores[name], dtype=np.float64) for name in SCORE_FIELDS
    }


def deterministic_folds(
    image_ids: Iterable[int], *, fold_count: int, seed: int
) -> list[list[int]]:
    values = sorted(map(int, image_ids))
    if len(values) != len(set(values)):
        raise ValueError("Image IDs must be unique")
    if not 2 <= int(fold_count) <= len(values):
        raise ValueError("Fold count must create non-empty folds")
    ordered = sorted(
        values,
        key=lambda image_id: (
            hashlib.sha256(f"{int(seed)}:{image_id}".encode("ascii")).hexdigest(),
            image_id,
        ),
    )
    folds = [[] for _ in range(int(fold_count))]
    for index, image_id in enumerate(ordered):
        folds[index % len(folds)].append(image_id)
    return [sorted(fold) for fold in folds]


def _resource_costs(payload: int) -> dict[str, float]:
    return {
        "upload": float(payload),
        "compute": 1.0 if payload else 0.0,
        "e2e": float(payload),
    }


def build_action_records(
    samples: Sequence[Mapping[str, Any]],
    edge_predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    category_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Convert canonical caches to the frozen set-router record schema."""
    category_index = {int(category_id): index for index, category_id in enumerate(category_ids)}
    if len(category_index) != len(category_ids):
        raise ValueError("category_ids must be unique")
    records = []
    for sample in samples:
        image_id = int(sample["image_id"])
        width, height = int(sample["width"]), int(sample["height"])
        original_bytes = sample["payload_bytes"].get("original")
        if original_bytes is None:
            raise ValueError(f"Original is unavailable for image {image_id}")
        boxes = []
        for prediction in edge_predictions.get(image_id, ()):
            category_id = int(prediction["category_id"])
            if category_id not in category_index:
                raise ValueError(f"Unknown category {category_id}")
            x, y, box_width, box_height = map(float, prediction["bbox"])
            boxes.append(
                {
                    "xyxy": [x, y, x + box_width, y + box_height],
                    "cls": int(category_index[category_id]),
                    "conf": float(prediction.get("score", prediction.get("conf", 0.0))),
                }
            )
        actions = [
            {
                "action_id": "edge",
                "action_type": "edge",
                "payload_bytes": 0,
                "response_bytes": 0,
                "cloud_calls": 0,
                "resolutions": [],
                "region_uids": [],
                "resource_costs": _resource_costs(0),
            }
        ]
        for action_id, action_type in (("original", "full"), ("q90", "pair")):
            payload = sample["payload_bytes"].get(action_id)
            if not bool(sample["available"].get(action_id, False)) or payload is None:
                continue
            actions.append(
                {
                    "action_id": action_id,
                    "action_type": action_type,
                    "payload_bytes": int(payload),
                    "response_bytes": 0,
                    "cloud_calls": 1,
                    "resolutions": [640],
                    "region_uids": [],
                    "resource_costs": _resource_costs(int(payload)),
                }
            )
        regions = []
        roi_payload = sample["payload_bytes"].get("roi")
        if bool(sample["available"].get("roi", False)) and roi_payload is not None:
            ratio = min(1.0, max(0.0, float(sample.get("roi_area_ratio", 0.0))))
            side = math.sqrt(ratio)
            regions.append(
                {
                    "region_uid": "canonical_roi",
                    "crop_xyxy": [0.0, 0.0, side * width, side * height],
                    "area_ratio": ratio,
                    "source": "deployed_box",
                }
            )
            actions.append(
                {
                    "action_id": "roi",
                    "action_type": "roi",
                    "payload_bytes": int(roi_payload),
                    "response_bytes": 0,
                    "cloud_calls": 1,
                    "resolutions": [640],
                    "region_uids": ["canonical_roi"],
                    "resource_costs": _resource_costs(int(roi_payload)),
                }
            )
        records.append(
            {
                "image_id": image_id,
                "width": width,
                "height": height,
                "edge_boxes": boxes,
                "full_payload_bytes": int(original_bytes),
                "image_stats": {},
                "regions": regions,
                "actions": actions,
            }
        )
    return records


def _available_actions(
    sample: Mapping[str, Any], scores: Mapping[str, float]
) -> list[dict[str, Any]]:
    actions = []
    for action in CLOUD_ACTIONS:
        payload = sample["payload_bytes"].get(action)
        value = scores.get(action)
        if (
            bool(sample["available"].get(action, False))
            and payload is not None
            and value is not None
            and math.isfinite(float(value))
        ):
            actions.append(
                {
                    "action_id": action,
                    "image_id": int(sample["image_id"]),
                    "value": float(value),
                    "cost": float(payload),
                }
            )
    if not actions:
        raise ValueError(f"No cloud action for image {sample['image_id']}")
    return actions


def schedule_fixed_call(
    samples: Sequence[Mapping[str, Any]],
    scores: Mapping[int, Mapping[str, float]],
    *,
    budget: float,
) -> dict[str, Any]:
    count = int(math.floor(len(samples) * float(budget) + 0.5))
    candidates = []
    for sample in samples:
        image_id = int(sample["image_id"])
        best = max(
            _available_actions(sample, scores[image_id]),
            key=lambda item: (
                float(item["value"]),
                -float(item["cost"]),
                str(item["action_id"]),
            ),
        )
        candidates.append(best)
    selected = sorted(
        candidates,
        key=lambda item: (
            -float(item["value"]),
            float(item["cost"]),
            int(item["image_id"]),
        ),
    )[:count]
    return {
        "selection": {
            int(item["image_id"]): str(item["action_id"]) for item in selected
        },
        "selected_images": len(selected),
        "total_cost_bytes": sum(float(item["cost"]) for item in selected),
        "total_value": sum(float(item["value"]) for item in selected),
    }


def schedule_fixed_byte(
    samples: Sequence[Mapping[str, Any]],
    scores: Mapping[int, Mapping[str, float]],
    *,
    capacity_bytes: float,
) -> dict[str, Any]:
    groups = []
    for sample in samples:
        image_id = int(sample["image_id"])
        groups.append(
            [
                {
                    "action_id": "edge",
                    "image_id": image_id,
                    "value": 0.0,
                    "cost": 0.0,
                },
                *_available_actions(sample, scores[image_id]),
            ]
        )
    result = select_actions_for_budget(groups, float(capacity_bytes))
    selection = {
        int(item["image_id"]): str(item["action_id"])
        for item in result["selected"]
        if str(item["action_id"]) != "edge"
    }
    if float(result["total_cost"]) > float(capacity_bytes) + 1e-6:
        raise RuntimeError("Fixed-byte capacity exceeded")
    return {
        "selection": selection,
        "selected_images": len(selection),
        "total_cost_bytes": float(result["total_cost"]),
        "total_value": float(result["total_value"]),
        "lambda": float(result["lambda"]),
    }


def evaluate_gate(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        (mode, float(budget))
        for mode in protocol["modes"]
        for budget in protocol["budgets"]
    }
    observed = {(str(row["mode"]), float(row["budget"])) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("Gate rows do not cover the frozen work points")
    gate = protocol["gate"]
    byte = [row for row in rows if row["mode"] == "fixed_byte"]
    call = [row for row in rows if row["mode"] == "fixed_call"]
    byte_gains = [float(row["delta_ap"]) for row in byte]
    gain_points = sum(
        gain >= float(gate["minimum_fixed_byte_gain"]) for gain in byte_gains
    )
    byte_noninferior = all(
        gain >= -float(gate["maximum_other_loss"]) for gain in byte_gains
    )
    call_noninferior = all(
        float(row["delta_ap"]) >= -float(gate["maximum_fixed_call_loss"])
        for row in call
    )
    capacity_ok = all(bool(row["capacity_ok"]) for row in rows)
    passed = (
        gain_points >= int(gate["required_fixed_byte_gain_points"])
        and byte_noninferior
        and call_noninferior
        and capacity_ok
    )
    return {
        "passed": bool(passed),
        "fixed_byte_gain_points": int(gain_points),
        "fixed_byte_noninferior": bool(byte_noninferior),
        "fixed_call_noninferior": bool(call_noninferior),
        "capacity_ok": bool(capacity_ok),
    }
