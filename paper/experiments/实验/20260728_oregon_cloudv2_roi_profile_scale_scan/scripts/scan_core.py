"""Pure contracts for the Oregon Cloud-V2 ROI profile/scale scan."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


PROFILES = ("current", "moderate", "historical_cost", "fixed_1_8")
IMAGE_SIZES = (160, 320, 640)
DEV_ROLES = ("calibration", "audit")


def canonical_signature(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_roles(roles: Mapping[str, int]) -> tuple[str, ...]:
    names = tuple(map(str, roles))
    if "final" in names:
        raise ValueError("final must remain sealed")
    if names != DEV_ROLES:
        raise ValueError(f"roles must be {DEV_ROLES}, got {names}")
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


def roi_expansion(area_ratio: float, profile: str) -> float:
    area = max(0.0, float(area_ratio))
    if profile == "current":
        if area < 0.01:
            return 1.8
        if area < 0.10:
            return 1.6
        return 1.4
    if profile == "moderate":
        if area < 0.02:
            return 2.5
        if area < 0.04:
            return 2.0
        if area < 0.08:
            return 1.8
        return 1.4
    if profile == "historical_cost":
        if area < 0.02:
            return 4.0
        if area < 0.04:
            return 3.0
        if area < 0.08:
            return 2.5
        return 1.4
    if profile == "fixed_1_8":
        return 1.8
    raise ValueError(f"unknown ROI profile: {profile}")


def merged_roi_xyxy(
    width: int,
    height: int,
    boxes: list[dict[str, Any]],
    *,
    profile: str,
    confidence: float = 0.25,
    max_area_ratio: float = 0.65,
) -> tuple[int, int, int, int] | None:
    image_width = max(1, int(width))
    image_height = max(1, int(height))
    expanded = []
    for box in boxes:
        if float(box.get("score", box.get("conf", 0.0))) < float(confidence):
            continue
        x, y, box_width, box_height = map(float, box["bbox"])
        center_x = x + box_width / 2.0
        center_y = y + box_height / 2.0
        area_ratio = box_width * box_height / (image_width * image_height)
        factor = roi_expansion(area_ratio, profile)
        half_width = box_width * factor / 2.0
        half_height = box_height * factor / 2.0
        expanded.append(
            (
                max(0.0, center_x - half_width),
                max(0.0, center_y - half_height),
                min(float(image_width), center_x + half_width),
                min(float(image_height), center_y + half_height),
            )
        )
    if not expanded:
        return None
    x1 = max(0, int(min(value[0] for value in expanded)))
    y1 = max(0, int(min(value[1] for value in expanded)))
    x2 = min(image_width, int(max(value[2] for value in expanded) + 0.999999))
    y2 = min(image_height, int(max(value[3] for value in expanded) + 0.999999))
    if x2 <= x1 or y2 <= y1:
        return None
    crop_area = (x2 - x1) * (y2 - y1) / (image_width * image_height)
    return None if crop_area > float(max_area_ratio) else (x1, y1, x2, y2)


def best_candidate(
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    values = list(candidates)
    if not values:
        return None
    return min(
        values,
        key=lambda value: (
            -float(value["quality"]),
            int(value["payload_bytes"]),
            str(value["candidate_id"]),
        ),
    )


def ranked_candidate_actions(
    options_by_image: Mapping[int, Iterable[dict[str, Any]]], *, count: int
) -> dict[int, dict[str, Any]]:
    ranked = []
    for image_id, options in options_by_image.items():
        chosen = best_candidate(options)
        if chosen is None:
            continue
        value = dict(chosen)
        value["gain"] = float(value["quality"]) - float(value["base_quality"])
        ranked.append((int(image_id), value))
    ranked.sort(
        key=lambda item: (
            -float(item[1]["gain"]),
            int(item[0]),
            str(item[1]["candidate_id"]),
        )
    )
    return dict(ranked[: max(0, int(count))])


def select_ap_safe_point(
    rows: list[dict[str, Any]], *, tolerance: float
) -> dict[str, Any]:
    if not rows:
        raise ValueError("working-point rows are empty")
    baseline = max(
        (
            row
            for row in rows
            if float(row.get("roi_ratio", 0.0)) == 0.0
        ),
        key=lambda row: float(row["official_map50_95"]),
        default=rows[0],
    )
    minimum_ap = float(baseline["official_map50_95"]) - float(tolerance)
    eligible = [
        row for row in rows if float(row["official_map50_95"]) >= minimum_ap
    ]
    if not eligible:
        return baseline
    return min(
        eligible,
        key=lambda row: (
            float(row["request_kb_per_image"]),
            -float(row["official_map50_95"]),
            -float(row.get("roi_ratio", 0.0)),
        ),
    )


def with_baseline_deltas(
    point: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(point)
    result["map_delta_vs_base"] = (
        float(point["official_map50_95"])
        - float(baseline["official_map50_95"])
    )
    result["kb_delta_vs_base"] = (
        float(point["request_kb_per_image"])
        - float(baseline["request_kb_per_image"])
    )
    return result
