"""Pure contracts for the Oregon Cloud-V2 joint action experiment."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
SOURCE_SCRIPTS = (
    REPO
    / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts"
)
sys.path.insert(0, str(SOURCE_SCRIPTS))

from protocol_core import class_aware_nms, iou_xywh  # noqa: E402


ROLES = ("fit", "calibration", "audit")
PROFILES = ("current", "moderate", "historical_cost", "fixed_1_8")
IMAGE_SIZES = (160, 320, 640)
BASE_ACTIONS = ("original", "q90")


def validate_roles(roles: Mapping[str, int]) -> tuple[str, ...]:
    names = tuple(map(str, roles))
    if "final" in names:
        raise ValueError("final must remain sealed")
    if names != ROLES:
        raise ValueError(f"roles must be {ROLES}, got {names}")
    if any(int(roles[name]) <= 0 for name in names):
        raise ValueError("role population must be positive")
    return names


def candidate_matrix(
    profiles: Iterable[str], image_sizes: Iterable[int]
) -> list[dict[str, Any]]:
    profile_values = tuple(map(str, profiles))
    size_values = tuple(map(int, image_sizes))
    if profile_values != PROFILES:
        raise ValueError(f"profiles must be {PROFILES}, got {profile_values}")
    if size_values != IMAGE_SIZES:
        raise ValueError(f"imgsz must be {IMAGE_SIZES}, got {size_values}")
    return [
        {"id": f"{profile}@{imgsz}", "profile": profile, "imgsz": imgsz}
        for profile in profile_values
        for imgsz in size_values
    ]


def action_feature(
    sample: Mapping[str, Any],
    *,
    action_id: str,
    payload_bytes: int,
    base_payload_bytes: int,
    record: Mapping[str, Any] | None,
) -> list[float]:
    observable = list(map(float, sample["action_features"]))
    if len(observable) != 54:
        raise ValueError(f"expected 54 observable features, got {len(observable)}")

    if action_id in BASE_ACTIONS:
        action_type = action_id
        profile = None
        imgsz = None
    else:
        profile, size = action_id.rsplit("@", maxsplit=1)
        imgsz = int(size)
        if profile not in PROFILES or imgsz not in IMAGE_SIZES:
            raise ValueError(f"unknown action: {action_id}")
        if record is None:
            raise ValueError(f"ROI action requires geometry: {action_id}")
        action_type = "roi"

    type_features = [float(action_type == value) for value in ("original", "q90", "roi")]
    profile_features = [float(profile == value) for value in PROFILES]
    size_features = [float(imgsz == value) for value in IMAGE_SIZES]

    original_payload = max(1, int(sample["payload_bytes"]["original"]))
    base_payload = max(1, int(base_payload_bytes))
    if record is None:
        area_ratio = log_aspect = magnification = 0.0
    else:
        crop_width = max(1, int(record["crop_width"]))
        crop_height = max(1, int(record["crop_height"]))
        image_width = max(1, int(record["image_width"]))
        image_height = max(1, int(record["image_height"]))
        area_ratio = float(record["area_ratio"])
        log_aspect = math.log(crop_width / crop_height)
        full_scale = 640.0 / max(image_width, image_height)
        crop_scale = float(imgsz) / max(crop_width, crop_height)
        magnification = crop_scale / full_scale

    dynamic = [
        1.0,
        float(payload_bytes) / original_payload,
        float(payload_bytes) / base_payload,
        area_ratio,
        log_aspect,
        magnification,
    ]
    result = [
        *observable,
        *type_features,
        *profile_features,
        *size_features,
        *dynamic,
    ]
    if len(result) != 70 or not all(math.isfinite(value) for value in result):
        raise RuntimeError("joint action feature contract failed")
    return result


def _with_source(prediction: Mapping[str, Any], source: str) -> dict[str, Any]:
    value = dict(prediction)
    value["_source"] = source
    return value


def reversible_fusion(
    edge_boxes: list[dict[str, Any]],
    cloud_boxes: list[dict[str, Any]],
    *,
    match_iou: float,
    score_margin: float,
    add_threshold: float,
    nms_iou: float = 0.7,
) -> list[dict[str, Any]]:
    if not 0.0 <= float(match_iou) <= 1.0:
        raise ValueError("match_iou must be in [0, 1]")
    if not 0.0 <= float(add_threshold) <= 1.0:
        raise ValueError("add_threshold must be in [0, 1]")

    pairs = []
    for edge_index, edge in enumerate(edge_boxes):
        for cloud_index, cloud in enumerate(cloud_boxes):
            if int(edge["category_id"]) != int(cloud["category_id"]):
                continue
            overlap = iou_xywh(edge["bbox"], cloud["bbox"])
            if overlap >= float(match_iou):
                pairs.append(
                    (
                        -overlap,
                        -float(cloud.get("score", 1.0)),
                        edge_index,
                        cloud_index,
                    )
                )
    paired_edge: dict[int, int] = {}
    paired_cloud: set[int] = set()
    for _, _, edge_index, cloud_index in sorted(pairs):
        if edge_index in paired_edge or cloud_index in paired_cloud:
            continue
        paired_edge[edge_index] = cloud_index
        paired_cloud.add(cloud_index)

    fused = []
    for edge_index, edge in enumerate(edge_boxes):
        cloud_index = paired_edge.get(edge_index)
        if cloud_index is None:
            fused.append(_with_source(edge, "edge"))
            continue
        cloud = cloud_boxes[cloud_index]
        if float(cloud.get("score", 1.0)) >= (
            float(edge.get("score", 1.0)) + float(score_margin)
        ):
            fused.append(_with_source(cloud, "roi"))
        else:
            fused.append(_with_source(edge, "edge"))

    for cloud_index, cloud in enumerate(cloud_boxes):
        if cloud_index in paired_cloud:
            continue
        if float(cloud.get("score", 1.0)) >= float(add_threshold):
            fused.append(_with_source(cloud, "roi"))
    return class_aware_nms(fused, iou_threshold=float(nms_iou))


def _option_score(
    option: Mapping[str, Any], *, original_payload: int, cost_weight: float
) -> float:
    saving = 1.0 - float(option["payload_bytes"]) / max(1, int(original_payload))
    return float(option["predicted_gain"]) + float(cost_weight) * saving


def select_joint_actions(
    options_by_image: Mapping[int, Iterable[Mapping[str, Any]]],
    *,
    frozen_base: Mapping[int, str],
    roi_budget: float,
    safe_threshold: float,
    cost_weight: float,
    base_mode: str,
) -> dict[int, str]:
    if base_mode not in {"frozen_q90_original", "joint_q90_original"}:
        raise ValueError(f"unknown base mode: {base_mode}")
    if not 0.0 <= float(roi_budget) <= 1.0:
        raise ValueError("roi_budget must be in [0, 1]")

    selected: dict[int, str] = {}
    roi_candidates = []
    for raw_image_id, raw_options in options_by_image.items():
        image_id = int(raw_image_id)
        options = [dict(value) for value in raw_options]
        by_action = {str(value["action_id"]): value for value in options}
        if set(BASE_ACTIONS) - set(by_action):
            raise ValueError(f"missing base actions for image {image_id}")
        original_payload = int(by_action["original"]["payload_bytes"])

        if base_mode == "frozen_q90_original":
            base_action = str(frozen_base[image_id])
            base_option = by_action[base_action]
        else:
            base_options = [by_action["original"]]
            if float(by_action["q90"]["safe_probability"]) >= float(safe_threshold):
                base_options.append(by_action["q90"])
            base_option = min(
                base_options,
                key=lambda value: (
                    -_option_score(
                        value,
                        original_payload=original_payload,
                        cost_weight=cost_weight,
                    ),
                    int(value["payload_bytes"]),
                    str(value["action_id"]),
                ),
            )
            base_action = str(base_option["action_id"])
        selected[image_id] = base_action
        base_score = _option_score(
            base_option,
            original_payload=original_payload,
            cost_weight=cost_weight,
        )

        eligible_roi = [
            value
            for value in options
            if "@" in str(value["action_id"])
            and float(value["safe_probability"]) >= float(safe_threshold)
            and int(value["payload_bytes"]) < int(base_option["payload_bytes"])
        ]
        if not eligible_roi:
            continue
        best_roi = min(
            eligible_roi,
            key=lambda value: (
                -_option_score(
                    value,
                    original_payload=original_payload,
                    cost_weight=cost_weight,
                ),
                int(value["payload_bytes"]),
                str(value["action_id"]),
            ),
        )
        roi_score = _option_score(
            best_roi,
            original_payload=original_payload,
            cost_weight=cost_weight,
        )
        advantage = roi_score - base_score
        if advantage > 0.0:
            roi_candidates.append(
                (
                    -advantage,
                    int(best_roi["payload_bytes"]),
                    image_id,
                    str(best_roi["action_id"]),
                )
            )

    maximum_roi = round(float(roi_budget) * len(options_by_image))
    for _, _, image_id, action_id in sorted(roi_candidates)[:maximum_roi]:
        selected[image_id] = action_id
    return selected


def choose_ap_safe_configuration(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline_name: str,
    tolerance: float,
) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    baselines = [row for row in values if str(row["name"]) == str(baseline_name)]
    if len(baselines) != 1:
        raise ValueError("calibration requires exactly one named baseline")
    minimum_ap = float(baselines[0]["official_map50_95"]) - float(tolerance)
    eligible = [row for row in values if float(row["official_map50_95"]) >= minimum_ap]
    return min(
        eligible,
        key=lambda row: (
            float(row["request_kb_per_image"]),
            -float(row["official_map50_95"]),
            str(row["name"]),
        ),
    )


def balanced_binary_weights(
    labels: Iterable[bool], image_ids: Iterable[int]
) -> list[float]:
    label_values = [bool(value) for value in labels]
    image_values = [int(value) for value in image_ids]
    if len(label_values) != len(image_values) or not label_values:
        raise ValueError("labels and image_ids must have equal non-zero length")
    counts = Counter(image_values)
    weights = [1.0 / counts[image_id] for image_id in image_values]
    class_totals = {
        label: sum(
            weight
            for weight, observed in zip(weights, label_values)
            if observed is label
        )
        for label in (False, True)
    }
    present = [label for label, total in class_totals.items() if total > 0.0]
    target = 1.0 / len(present)
    return [
        weight * target / class_totals[label]
        for weight, label in zip(weights, label_values)
    ]


def choose_reversible_configuration(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config_id"])].append(dict(row))
    if not grouped:
        raise ValueError("reversible fusion calibration is empty")
    budget_sets = {
        config_id: {float(row["budget"]) for row in values}
        for config_id, values in grouped.items()
    }
    expected = next(iter(budget_sets.values()))
    if any(values != expected for values in budget_sets.values()):
        raise ValueError("reversible fusion configs cover different budgets")
    summaries = []
    for config_id, values in grouped.items():
        deltas = [float(row["map_delta_vs_base"]) for row in values]
        summaries.append(
            {
                "config_id": config_id,
                "worst_map_delta": min(deltas),
                "mean_map_delta": sum(deltas) / len(deltas),
                "rows": sorted(values, key=lambda row: float(row["budget"])),
            }
        )
    return min(
        summaries,
        key=lambda value: (
            -float(value["worst_map_delta"]),
            -float(value["mean_map_delta"]),
            str(value["config_id"]),
        ),
    )
