"""Pure record and job contracts for Cloud-V4 CurrentActionValue retraining."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


NEURAL_METHOD_SPECS: dict[str, tuple[str, str]] = {
    "CurrentActionValue-OQR12": ("OQR12", "current_action"),
}
EXTERNAL_METHODS: tuple[str, ...] = ()


def _resource_costs(payload: int) -> dict[str, float]:
    return {
        "upload": float(payload),
        "compute": float(payload > 0),
        "e2e": float(payload),
    }


def build_action_records(
    samples: Sequence[Mapping[str, Any]],
    edge_predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    category_ids: Sequence[int],
) -> list[dict[str, Any]]:
    category_index = {
        int(category_id): index for index, category_id in enumerate(category_ids)
    }
    if len(category_index) != len(category_ids):
        raise ValueError("category IDs must be unique")
    records = []
    for sample in samples:
        image_id = int(sample["image_id"])
        width, height = int(sample["width"]), int(sample["height"])
        full_payload = int(sample["payload_bytes"]["original"])
        edge_boxes = []
        for prediction in edge_predictions.get(image_id, ()):
            category_id = int(prediction["category_id"])
            if category_id not in category_index:
                raise ValueError(f"unknown category ID: {category_id}")
            x, y, box_width, box_height = map(float, prediction["bbox"])
            edge_boxes.append(
                {
                    "xyxy": [x, y, x + box_width, y + box_height],
                    "cls": int(category_index[category_id]),
                    "conf": float(prediction.get("score", 0.0)),
                }
            )
        regions = []
        actions = []
        for source in sample["actions"]:
            action_id = str(source["action_id"])
            if not bool(source.get("available", True)):
                continue
            payload = int(source.get("payload_bytes", 0))
            if action_id == "edge":
                action_type, resolutions, region_uids, cloud_calls = "edge", [], [], 0
            elif action_id == "original":
                action_type, resolutions, region_uids, cloud_calls = "full", [640], [], 1
            elif action_id == "q90":
                action_type, resolutions, region_uids, cloud_calls = "pair", [640], [], 1
            else:
                if "@" not in action_id:
                    raise ValueError(f"unknown action: {action_id}")
                action_type = "roi"
                scale = int(source["imgsz"])
                xyxy = list(map(float, source["xyxy"]))
                region_uid = action_id
                regions.append(
                    {
                        "region_uid": region_uid,
                        "crop_xyxy": xyxy,
                        "area_ratio": float(source["area_ratio"]),
                        "source": "deployed_box",
                        "profile": str(source["profile"]),
                        "imgsz": scale,
                    }
                )
                resolutions, region_uids, cloud_calls = [scale], [region_uid], 1
            actions.append(
                {
                    "action_id": action_id,
                    "action_type": action_type,
                    "payload_bytes": payload,
                    "response_bytes": 0,
                    "cloud_calls": cloud_calls,
                    "resolutions": resolutions,
                    "region_uids": region_uids,
                    "resource_costs": _resource_costs(payload),
                }
            )
        if not actions or actions[0]["action_id"] != "edge":
            raise RuntimeError(f"Edge must be the first action: {image_id}")
        records.append(
            {
                "image_id": image_id,
                "identity": str(sample["identity"]),
                "event_id": str(sample["event_id"]),
                "camera": str(sample["camera"]),
                "width": width,
                "height": height,
                "edge_boxes": edge_boxes,
                "edge_predictions": list(edge_predictions.get(image_id, ())),
                "full_payload_bytes": full_payload,
                "image_stats": {},
                "regions": regions,
                "actions": actions,
            }
        )
    return records


def action_space(name: str, all_actions: Sequence[str]) -> list[str]:
    values = list(map(str, all_actions))
    if name == "O":
        allowed = {"edge", "original"}
    elif name == "OQ":
        allowed = {"edge", "original", "q90"}
    elif name == "OQR12":
        allowed = set(values)
    else:
        raise ValueError(f"unknown action space: {name}")
    output = [value for value in values if value in allowed]
    if not output or output[0] != "edge":
        raise ValueError("every action space must start with Edge")
    return output


def filter_records(
    records: Sequence[Mapping[str, Any]], action_ids: Sequence[str]
) -> list[dict[str, Any]]:
    allowed = set(map(str, action_ids))
    output = []
    for source in records:
        actions = [dict(action) for action in source["actions"] if str(action["action_id"]) in allowed]
        used_regions = {
            str(uid) for action in actions for uid in action.get("region_uids", ())
        }
        regions = [
            dict(region)
            for region in source.get("regions", ())
            if str(region["region_uid"]) in used_regions
        ]
        if not actions or actions[0]["action_id"] != "edge":
            raise RuntimeError(f"filtered record has no Edge: {source['image_id']}")
        output.append({**dict(source), "actions": actions, "regions": regions})
    return output


def reward_matrix(
    records: Sequence[Mapping[str, Any]],
    reward_map: Mapping[int, Mapping[str, float]],
) -> np.ndarray:
    maximum = max(len(record["actions"]) for record in records)
    values = np.zeros((len(records), maximum), dtype=np.float32)
    for row, record in enumerate(records):
        image_id = int(record["image_id"])
        if image_id not in reward_map:
            raise ValueError(f"missing reward image: {image_id}")
        for column, action in enumerate(record["actions"]):
            action_id = str(action["action_id"])
            if action_id == "edge":
                continue
            if action_id not in reward_map[image_id]:
                raise ValueError(f"missing reward: {image_id}/{action_id}")
            values[row, column] = float(reward_map[image_id][action_id])
    return values


def training_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    folds: list[int | str] = [*range(int(protocol["fold_count"])), "full"]
    jobs = []
    for method in (*NEURAL_METHOD_SPECS, *EXTERNAL_METHODS):
        if method in NEURAL_METHOD_SPECS:
            space, trainer = NEURAL_METHOD_SPECS[method]
        else:
            space, trainer = "O", "external"
        for seed in map(int, protocol["model_seeds"]):
            for fold in folds:
                token = f"fold_{fold}" if fold != "full" else "full"
                jobs.append(
                    {
                        "id": f"{method}|seed={seed}|{token}",
                        "method": method,
                        "action_space": space,
                        "trainer": trainer,
                        "seed": seed,
                        "fold": fold,
                    }
                )
    return jobs
