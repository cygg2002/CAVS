"""Pure adapters from validated RKNN candidates to CAVS COCO records."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


ACTION_ID = "current@640"


def edge_predictions(
    candidate: Mapping[str, Any], category_ids: Sequence[int]
) -> list[dict[str, Any]]:
    image_id = int(candidate["image_id"])
    categories = list(map(int, category_ids))
    output = []
    for box in candidate["edge_boxes"]:
        class_index = int(box["cls"])
        if not 0 <= class_index < len(categories):
            raise RuntimeError(f"RKNN class index is outside COCO schema: {class_index}")
        x1, y1, x2, y2 = map(float, box["xyxy"])
        output.append(
            {
                "image_id": image_id,
                "category_id": categories[class_index],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(box["conf"]),
            }
        )
    return output


def deployment_sample(
    base: Mapping[str, Any], roi: Mapping[str, Any] | None
) -> dict[str, Any]:
    original_bytes = int(base["payload_bytes"]["original"])
    q90_bytes = int(base["payload_bytes"]["q90"])
    sample = deepcopy(dict(base))
    sample["payload_bytes"] = {"original": original_bytes, "q90": q90_bytes}
    sample["available"] = {
        "original": True,
        "q90": bool(base.get("available", {}).get("q90", 0 < q90_bytes < original_bytes)),
    }
    sample["actions"] = [
        {"action_id": "edge", "payload_bytes": 0, "available": True},
        {"action_id": "original", "payload_bytes": original_bytes, "available": True},
        {
            "action_id": "q90",
            "payload_bytes": q90_bytes,
            "available": sample["available"]["q90"],
        },
    ]
    if roi is not None:
        payload = int(roi["bytes"])
        sample["payload_bytes"][ACTION_ID] = payload
        sample["available"][ACTION_ID] = True
        sample["actions"].append(
            {
                "action_id": ACTION_ID,
                "payload_bytes": payload,
                "available": True,
                "profile": "current",
                "imgsz": 640,
                "xyxy": list(map(int, roi["xyxy"])),
                "crop_width": int(roi["crop_width"]),
                "crop_height": int(roi["crop_height"]),
                "area_ratio": float(roi["area_ratio"]),
            }
        )
    else:
        sample["available"][ACTION_ID] = False
    sample["rknn_edge_aligned"] = True
    return sample
