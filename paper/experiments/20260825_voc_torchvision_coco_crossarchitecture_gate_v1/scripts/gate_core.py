"""Pure contracts and stop-gate logic for the Torchvision VOC preflight."""

from __future__ import annotations

import hashlib
import json
import math
import copy
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_DETECTORS = {
    "edge": (
        "ssdlite320_mobilenet_v3_large",
        "SSDLite320_MobileNet_V3_Large_Weights.COCO_V1",
    ),
    "cloud": (
        "fasterrcnn_resnet50_fpn_v2",
        "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
    ),
}


def read_protocol(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("status") != "preregistered_preflight_gate_only":
        raise ValueError("preflight status changed")
    if list(map(str, protocol.get("allowed_roles", ()))) != ["fit", "calibration"]:
        raise ValueError("final and unknown roles are forbidden")
    if protocol.get("final_access_allowed") is not False:
        raise ValueError("final access must remain forbidden")
    roles = protocol.get("roles", {})
    if set(roles) != {"fit", "calibration"}:
        raise ValueError("final and unknown role contracts are forbidden")
    if {key: int(value.get("images", 0)) for key, value in roles.items()} != {
        "fit": 15629,
        "calibration": 922,
    }:
        raise ValueError("frozen VOC role counts changed")
    for role, contract in roles.items():
        for key in ("gt_sha256", "images_sha256", "identity_sha256"):
            if len(str(contract.get(key, ""))) != 64:
                raise ValueError(f"incomplete {role}/{key}")
    for role, expected in EXPECTED_DETECTORS.items():
        detector = protocol.get("detectors", {}).get(role, {})
        observed = (str(detector.get("builder", "")), str(detector.get("weights_enum", "")))
        if observed != expected:
            raise ValueError(f"frozen {role} detector changed")
        if str(detector.get("training_dataset", "")) != "COCO train2017":
            raise ValueError(f"{role} detector must be trained on COCO train2017")
    runtime = protocol.get("runtime", {})
    if int(runtime.get("batch_size", 0)) != 1:
        raise ValueError("single-image inference contract changed")
    if float(runtime.get("score_threshold", -1.0)) != 0.001:
        raise ValueError("score threshold changed")
    if int(runtime.get("detections_per_image", 0)) != 300:
        raise ValueError("maximum detections changed")
    mapping = protocol.get("category_mapping", {})
    if len(mapping) != 20 or len(set(map(str, mapping.values()))) != 20:
        raise ValueError("COCO-to-VOC mapping must be one-to-one over 20 classes")


def coco_to_voc_ids(
    category_mapping: Mapping[str, str], voc_category_names: Sequence[str]
) -> dict[str, int]:
    names = list(map(str, voc_category_names))
    if len(names) != 20 or len(names) != len(set(names)):
        raise ValueError("VOC category schema must contain 20 unique names")
    by_name = {name: index + 1 for index, name in enumerate(names)}
    mapped = {}
    for coco_name, voc_name in category_mapping.items():
        if str(voc_name) not in by_name:
            raise ValueError(f"unknown VOC category mapping: {coco_name}/{voc_name}")
        mapped[str(coco_name)] = int(by_name[str(voc_name)])
    if len(mapped) != 20 or set(category_mapping.values()) != set(names):
        raise ValueError("COCO-to-VOC mapping does not cover the exact VOC schema")
    return mapped


def _ordered(values: Iterable[str], seed: int) -> list[str]:
    return sorted(
        map(str, values),
        key=lambda value: (
            hashlib.sha256(f"{int(seed)}:{value}".encode("utf-8")).digest(),
            value,
        ),
    )


def deterministic_folds(
    identities: Sequence[str], *, count: int, seed: int
) -> list[list[str]]:
    values = list(map(str, identities))
    if len(values) != len(set(values)):
        raise ValueError("fold identities must be unique")
    if int(count) < 2 or int(count) > len(values):
        raise ValueError("invalid fold count")
    folds = [[] for _ in range(int(count))]
    for index, identity in enumerate(_ordered(values, seed)):
        folds[index % int(count)].append(identity)
    return [sorted(fold) for fold in folds]


def select_smoke_images(
    images: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
    category_ids: Iterable[int],
) -> list[dict[str, Any]]:
    if int(count) <= 0 or int(count) > len(images):
        raise ValueError("invalid smoke population")
    rows = [dict(row) for row in images]
    by_identity = {str(row["identity"]): row for row in rows}
    if len(by_identity) != len(rows):
        raise ValueError("smoke identities must be unique")
    ordered = [by_identity[key] for key in _ordered(by_identity, seed)]
    selected: dict[str, dict[str, Any]] = {}
    for category_id in map(int, category_ids):
        candidate = next(
            (row for row in ordered if category_id in set(map(int, row.get("category_ids", ())))),
            None,
        )
        if candidate is None:
            raise ValueError(f"smoke population cannot cover category {category_id}")
        selected[str(candidate["identity"])] = candidate
    for row in ordered:
        if len(selected) >= int(count):
            break
        selected.setdefault(str(row["identity"]), row)
    if len(selected) != int(count):
        raise RuntimeError("smoke selection count changed")
    return sorted(selected.values(), key=lambda row: int(row["id"]))


def sanitize_detections(
    *,
    image_id: int,
    width: int,
    height: int,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    scores: Sequence[float],
    label_names: Mapping[int, str],
    category_ids: Mapping[str, int],
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not (len(boxes) == len(labels) == len(scores)):
        raise ValueError("detection arrays are not aligned")
    audit = {
        "below_threshold": 0,
        "filtered_non_voc": 0,
        "nonfinite_boxes": 0,
        "nonfinite_scores": 0,
        "degenerate_boxes": 0,
        "kept": 0,
    }
    output = []
    for box, label, score in zip(boxes, labels, scores):
        value = float(score)
        if not math.isfinite(value):
            audit["nonfinite_scores"] += 1
            continue
        if value < float(threshold):
            audit["below_threshold"] += 1
            continue
        name = str(label_names.get(int(label), ""))
        if name not in category_ids:
            audit["filtered_non_voc"] += 1
            continue
        coordinates = list(map(float, box))
        if len(coordinates) != 4 or not all(map(math.isfinite, coordinates)):
            audit["nonfinite_boxes"] += 1
            continue
        x1, y1, x2, y2 = coordinates
        x1, x2 = min(max(x1, 0.0), float(width)), min(max(x2, 0.0), float(width))
        y1, y2 = min(max(y1, 0.0), float(height)), min(max(y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            audit["degenerate_boxes"] += 1
            continue
        output.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_ids[name]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": value,
            }
        )
        audit["kept"] += 1
    return output, audit


def bootstrap_remap(
    ground_truth: Mapping[str, Any],
    prediction_sets: Sequence[Sequence[Mapping[str, Any]]],
    sampled_image_ids: Sequence[int],
) -> tuple[dict[str, Any], tuple[list[dict[str, Any]], ...]]:
    """Give repeated bootstrap draws unique IDs so the VOC metric preserves multiplicity."""
    images = {int(row["id"]): row for row in ground_truth["images"]}
    annotations: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ground_truth["annotations"]:
        annotations[int(row["image_id"])].append(row)
    prediction_groups = []
    for predictions in prediction_sets:
        grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in predictions:
            grouped[int(row["image_id"])].append(row)
        prediction_groups.append(grouped)
    remapped_gt = {
        "images": [],
        "annotations": [],
        "categories": copy.deepcopy(ground_truth["categories"]),
    }
    remapped_predictions: list[list[dict[str, Any]]] = [[] for _ in prediction_sets]
    annotation_id = 1
    for new_id, source_id in enumerate(map(int, sampled_image_ids), start=1):
        if source_id not in images:
            raise ValueError(f"bootstrap sampled an unknown image: {source_id}")
        image = copy.deepcopy(images[source_id])
        image["id"] = new_id
        remapped_gt["images"].append(image)
        for source in annotations[source_id]:
            row = copy.deepcopy(source)
            row["id"] = annotation_id
            row["image_id"] = new_id
            annotation_id += 1
            remapped_gt["annotations"].append(row)
        for target, grouped in zip(remapped_predictions, prediction_groups):
            for source in grouped[source_id]:
                row = copy.deepcopy(source)
                row["image_id"] = new_id
                target.append(row)
    return remapped_gt, tuple(remapped_predictions)


def assess_endpoint_gate(
    protocol: Mapping[str, Any],
    *,
    fit: Mapping[str, Mapping[str, float]],
    calibration: Mapping[str, Mapping[str, float]],
    fit_fold_deltas: Sequence[float],
    bootstrap_ci: Sequence[float],
    original_bytes: int,
    q90_bytes: int,
) -> dict[str, Any]:
    gate = protocol["gate"]
    minimum_gap = float(gate["minimum_cloud_edge_map50_95"])
    fit_delta = float(fit["original"]["map50_95"]) - float(fit["edge"]["map50_95"])
    calibration_delta = float(calibration["original"]["map50_95"]) - float(
        calibration["edge"]["map50_95"]
    )
    q90_savings = 1.0 - float(q90_bytes) / max(1.0, float(original_bytes))
    q90_losses = [
        float(role["original"]["map50_95"]) - float(role["q90"]["map50_95"])
        for role in (fit, calibration)
    ]

    checks = []

    def add(name: str, passed: bool, value: Any, criterion: str) -> None:
        checks.append(
            {"name": name, "passed": bool(passed), "value": value, "criterion": criterion}
        )

    add("fit_cloud_edge_gap", fit_delta >= minimum_gap, fit_delta, f">={minimum_gap}")
    add(
        "calibration_cloud_edge_gap",
        calibration_delta >= minimum_gap,
        calibration_delta,
        f">={minimum_gap}",
    )
    add(
        "all_fit_folds_positive",
        len(fit_fold_deltas) == int(protocol["folds"]["count"])
        and all(float(value) > 0.0 for value in fit_fold_deltas),
        list(map(float, fit_fold_deltas)),
        "all > 0",
    )
    add(
        "calibration_bootstrap_ci_lower",
        len(bootstrap_ci) == 2
        and float(bootstrap_ci[0]) > float(gate["minimum_bootstrap_ci_lower"]),
        list(map(float, bootstrap_ci)),
        f"lower>{float(gate['minimum_bootstrap_ci_lower'])}",
    )
    add(
        "cloud_ap75_not_lower",
        all(float(role["original"]["map75"]) >= float(role["edge"]["map75"]) for role in (fit, calibration)),
        {
            "fit": float(fit["original"]["map75"]) - float(fit["edge"]["map75"]),
            "calibration": float(calibration["original"]["map75"])
            - float(calibration["edge"]["map75"]),
        },
        "both >= 0",
    )
    add(
        "q90_byte_savings",
        q90_savings >= float(gate["minimum_q90_byte_savings_fraction"]),
        q90_savings,
        f">={float(gate['minimum_q90_byte_savings_fraction'])}",
    )
    add(
        "q90_map_loss",
        max(q90_losses) <= float(gate["maximum_q90_map50_95_loss"]),
        q90_losses,
        f"max<={float(gate['maximum_q90_map50_95_loss'])}",
    )
    return {
        "proceed_to_reward": all(row["passed"] for row in checks),
        "checks": checks,
    }


def assess_calibration_preflight(
    protocol: Mapping[str, Any],
    *,
    metrics: Mapping[str, Mapping[str, float]],
    original_bytes: int,
    q90_bytes: int,
) -> dict[str, Any]:
    """Fail fast on calibration before spending GPU time on all fit images."""
    gate = protocol["gate"]
    cloud_gap = float(metrics["original"]["map50_95"]) - float(metrics["edge"]["map50_95"])
    ap75_gap = float(metrics["original"]["map75"]) - float(metrics["edge"]["map75"])
    q90_loss = float(metrics["original"]["map50_95"]) - float(metrics["q90"]["map50_95"])
    q90_savings = 1.0 - float(q90_bytes) / max(1.0, float(original_bytes))
    checks = [
        {
            "name": "calibration_cloud_edge_gap",
            "value": cloud_gap,
            "criterion": f">={float(gate['minimum_cloud_edge_map50_95'])}",
            "passed": cloud_gap >= float(gate["minimum_cloud_edge_map50_95"]),
        },
        {
            "name": "calibration_cloud_ap75_not_lower",
            "value": ap75_gap,
            "criterion": ">=0",
            "passed": ap75_gap >= 0.0,
        },
        {
            "name": "calibration_q90_map_loss",
            "value": q90_loss,
            "criterion": f"<={float(gate['maximum_q90_map50_95_loss'])}",
            "passed": q90_loss <= float(gate["maximum_q90_map50_95_loss"]),
        },
        {
            "name": "calibration_q90_byte_savings",
            "value": q90_savings,
            "criterion": f">={float(gate['minimum_q90_byte_savings_fraction'])}",
            "passed": q90_savings >= float(gate["minimum_q90_byte_savings_fraction"]),
        },
    ]
    return {"proceed_to_fit": all(row["passed"] for row in checks), "checks": checks}
