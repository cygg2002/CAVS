"""Pure cache/geometry helpers shared by Cloud‑V3 inference stages."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def map_crop_bbox(
    crop_xyxy: Sequence[float], roi_xyxy: Sequence[float]
) -> list[float]:
    if len(crop_xyxy) != 4 or len(roi_xyxy) != 4:
        raise ValueError("xyxy boxes require four coordinates")
    x1, y1, x2, y2 = map(float, crop_xyxy)
    offset_x, offset_y = map(float, roi_xyxy[:2])
    return [
        x1 + offset_x,
        y1 + offset_y,
        max(0.0, x2 - x1),
        max(0.0, y2 - y1),
    ]


def action_record(record: Mapping[str, Any], *, imgsz: int) -> dict[str, Any]:
    profile = str(record["profile"])
    xyxy = list(map(int, record["xyxy"]))
    width = int(record["image_width"])
    height = int(record["image_height"])
    if xyxy[0] < 0 or xyxy[1] < 0 or xyxy[2] > width or xyxy[3] > height:
        raise ValueError("crop geometry exceeds the source image")
    if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
        raise ValueError("crop geometry is empty")
    return {
        "action_id": f"{profile}@{int(imgsz)}",
        "image_id": int(record["image_id"]),
        "identity": str(record["identity"]),
        "profile": profile,
        "imgsz": int(imgsz),
        "xyxy": xyxy,
        "crop_width": int(record["crop_width"]),
        "crop_height": int(record["crop_height"]),
        "area_ratio": float(record["area_ratio"]),
        "payload_bytes": int(record["bytes"]),
        "path": str(record.get("path", "")),
    }


def cache_signature(
    *,
    role: str,
    action_id: str,
    model_sha: str,
    source_sha: str,
    settings: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "role": str(role),
            "action_id": str(action_id),
            "cloud_model_sha256": str(model_sha),
            "source_sha256": str(source_sha),
            "inference": dict(settings),
        }
    )


def planned_cache_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for role in protocol["roles"]:
        jobs.extend(
            {
                "kind": "geometry",
                "role": str(role),
                "profile": str(profile),
                "action_id": str(profile),
            }
            for profile in protocol["profiles"]
        )
        jobs.extend(
            {"kind": "inference", "role": str(role), "action_id": action_id}
            for action_id in ("original", "q90", *map(str, protocol["roi_actions"]))
        )
    return jobs
