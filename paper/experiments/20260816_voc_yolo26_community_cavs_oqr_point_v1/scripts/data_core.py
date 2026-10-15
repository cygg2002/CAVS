"""Small data and evaluation helpers for the VOC point benchmark."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
from typing import Any, Iterable, Mapping, Sequence


def reward_roles(role: str) -> tuple[str, ...]:
    value = str(role)
    if value == "all":
        return ("fit", "calibration")
    if value in {"fit", "calibration"}:
        return (value,)
    raise ValueError("contextual rewards are forbidden for final")


def voc_box_to_coco(
    xmin: float, ymin: float, xmax: float, ymax: float
) -> list[float]:
    x = float(xmin) - 1.0
    y = float(ymin) - 1.0
    width = max(0.0, float(xmax) - float(xmin))
    height = max(0.0, float(ymax) - float(ymin))
    return [x, y, width, height]


def voc_box_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    width: int,
    height: int,
) -> list[float]:
    box_width = max(0.0, float(xmax) - float(xmin))
    box_height = max(0.0, float(ymax) - float(ymin))
    center_x = (float(xmin) + float(xmax)) / 2.0 - 1.0
    center_y = (float(ymin) + float(ymax)) / 2.0 - 1.0
    return [
        center_x / max(1, int(width)),
        center_y / max(1, int(height)),
        box_width / max(1, int(width)),
        box_height / max(1, int(height)),
    ]


def group_predictions(
    rows: Iterable[Mapping[str, Any]], *, image_ids: Sequence[int] = ()
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {
        int(image_id): [] for image_id in image_ids
    }
    for source in rows:
        image_id = int(source["image_id"])
        grouped.setdefault(image_id, []).append(
            {
                "image_id": image_id,
                "category_id": int(source["category_id"]),
                "bbox": list(map(float, source["bbox"])),
                "score": float(source.get("score", 1.0)),
            }
        )
    return grouped


def source_basename_to_id(
    images: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for image in images:
        basename = Path(str(image["absolute_path"])).name
        if basename in mapping:
            raise RuntimeError(f"Source image basenames are not unique: {basename}")
        mapping[basename] = int(image["id"])
    return mapping


def fixed_call_count(population: int, budget: float) -> int:
    return int(math.floor(int(population) * float(budget) + 0.5))


def fold_contained_contexts(
    folds: Sequence[Sequence[int]], *, context_size: int, seed: int
) -> dict[int, list[int]]:
    output: dict[int, list[int]] = {}
    for fold_index, fold in enumerate(folds):
        values = sorted(map(int, fold))
        for target in values:
            candidates = [value for value in values if value != target]
            count = min(max(0, int(context_size)), len(candidates))
            token = hashlib.sha256(
                f"{int(seed)}:{fold_index}:{target}".encode("utf-8")
            ).digest()
            rng = random.Random(int.from_bytes(token[:8], "big"))
            output[target] = sorted(rng.sample(candidates, count))
    return output


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
            if not bool(source.get("available", True)):
                continue
            action_id = str(source["action_id"])
            payload = int(source.get("payload_bytes", 0))
            if action_id == "edge":
                action_type, resolutions, region_uids, cloud_calls = "edge", [], [], 0
            elif action_id == "original":
                action_type, resolutions, region_uids, cloud_calls = "full", [640], [], 1
            elif action_id == "q90":
                action_type, resolutions, region_uids, cloud_calls = "pair", [640], [], 1
            elif action_id == "current@640":
                action_type, resolutions, region_uids, cloud_calls = "roi", [640], [action_id], 1
                regions.append(
                    {
                        "region_uid": action_id,
                        "crop_xyxy": list(map(float, source["xyxy"])),
                        "area_ratio": float(source["area_ratio"]),
                        "source": "deployed_box",
                        "profile": "current",
                        "imgsz": 640,
                    }
                )
            else:
                raise ValueError(f"unknown action: {action_id}")
            actions.append(
                {
                    "action_id": action_id,
                    "action_type": action_type,
                    "payload_bytes": payload,
                    "response_bytes": 0,
                    "cloud_calls": cloud_calls,
                    "resolutions": resolutions,
                    "region_uids": region_uids,
                    "resource_costs": {
                        "upload": float(payload),
                        "compute": float(payload > 0),
                        "e2e": float(payload),
                    },
                }
            )
        if not actions or actions[0]["action_id"] != "edge":
            raise RuntimeError(f"Edge must be the first action: {image_id}")
        records.append(
            {
                "image_id": image_id,
                "identity": str(sample["identity"]),
                "event_id": str(sample.get("event_id", sample["identity"])),
                "camera": str(sample.get("camera", "voc")),
                "width": width,
                "height": height,
                "edge_boxes": edge_boxes,
                "edge_predictions": list(edge_predictions.get(image_id, ())),
                "full_payload_bytes": int(sample["payload_bytes"]["original"]),
                "image_stats": {},
                "regions": regions,
                "actions": actions,
            }
        )
    return records


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def atomic_pickle(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, target)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def atomic_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def flat_predictions(grouped: Mapping[int, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    return [dict(row) for image_id in sorted(grouped) for row in grouped[image_id]]


def ground_truth_by_image(gt: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in gt["annotations"]:
        grouped[int(row["image_id"])].append(
            {
                "category_id": int(row["category_id"]),
                "bbox": list(map(float, row["bbox"])),
            }
        )
    return dict(grouped)


def map_crop_predictions(
    predictions: Sequence[Mapping[str, Any]],
    geometry: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for source in predictions:
        image_id = int(source["image_id"])
        if image_id not in geometry:
            raise ValueError(f"missing crop geometry: {image_id}")
        x1, y1, _, _ = map(float, geometry[image_id]["xyxy"])
        x, y, width, height = map(float, source["bbox"])
        output.append(
            {
                "image_id": image_id,
                "category_id": int(source["category_id"]),
                "bbox": [x + x1, y + y1, width, height],
                "score": float(source.get("score", 1.0)),
            }
        )
    return output
