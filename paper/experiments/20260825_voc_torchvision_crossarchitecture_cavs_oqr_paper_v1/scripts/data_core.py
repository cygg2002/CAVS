"""Data, hashing and action-record helpers for the VOC experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
from typing import Any, Iterable, Mapping, Sequence


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


def group_predictions(
    rows: Iterable[Mapping[str, Any]], *, image_ids: Sequence[int] = ()
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {int(image_id): [] for image_id in image_ids}
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


def flat_predictions(grouped: Mapping[int, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    return [dict(row) for image_id in sorted(grouped) for row in grouped[image_id]]


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


def source_basename_to_id(images: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for image in images:
        name = Path(str(image["absolute_path"])).name
        if name in mapping:
            raise RuntimeError(f"source basename is not unique: {name}")
        mapping[name] = int(image["id"])
    return mapping


def build_action_records(
    samples: Sequence[Mapping[str, Any]],
    edge_predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    category_ids: Sequence[int],
    action_space: str,
) -> list[dict[str, Any]]:
    if action_space not in {"O", "OQ", "OQR"}:
        raise ValueError(action_space)
    category_index = {int(category_id): index for index, category_id in enumerate(category_ids)}
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
        actions = [
            {
                "action_id": "edge",
                "action_type": "edge",
                "payload_bytes": 0,
                "response_bytes": 0,
                "cloud_calls": 0,
                "resolutions": [],
                "region_uids": [],
                "resource_costs": {"upload": 0.0, "compute": 0.0, "e2e": 0.0},
            },
            {
                "action_id": "original",
                "action_type": "full",
                "payload_bytes": int(sample["payload_bytes"]["original"]),
                "response_bytes": 0,
                "cloud_calls": 1,
                "resolutions": [800],
                "region_uids": [],
                "resource_costs": {
                    "upload": float(sample["payload_bytes"]["original"]),
                    "compute": 1.0,
                    "e2e": float(sample["payload_bytes"]["original"]),
                },
            },
        ]
        if action_space in {"OQ", "OQR"} and bool(
            sample.get("available", {}).get("q90", False)
        ):
            q90_bytes = int(sample["payload_bytes"]["q90"])
            actions.append(
                {
                    "action_id": "q90",
                    "action_type": "pair",
                    "payload_bytes": q90_bytes,
                    "response_bytes": 0,
                    "cloud_calls": 1,
                "resolutions": [800],
                    "region_uids": [],
                    "resource_costs": {
                        "upload": float(q90_bytes),
                        "compute": 1.0,
                        "e2e": float(q90_bytes),
                    },
                }
            )
        regions = []
        if action_space == "OQR" and bool(
            sample.get("available", {}).get("current@official", False)
        ):
            roi = dict(sample["roi"])
            roi_bytes = int(sample["payload_bytes"]["current@official"])
            region_uid = f"current:{image_id}"
            regions.append(
                {
                    "region_uid": region_uid,
                    "crop_xyxy": list(map(float, roi["xyxy"])),
                    "source": "deployed_box",
                }
            )
            actions.append(
                {
                    "action_id": "current@official",
                    "action_type": "roi",
                    "payload_bytes": roi_bytes,
                    "response_bytes": 0,
                    "cloud_calls": 1,
                    "resolutions": [800],
                    "region_uids": [region_uid],
                    "resource_costs": {
                        "upload": float(roi_bytes),
                        "compute": 1.0,
                        "e2e": float(roi_bytes),
                    },
                }
            )
        records.append(
            {
                "image_id": image_id,
                "identity": str(sample["identity"]),
                "event_id": str(sample.get("event_id", sample["identity"])),
                "camera": str(sample.get("camera", str(sample["identity"]).split(":", 1)[0])),
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
