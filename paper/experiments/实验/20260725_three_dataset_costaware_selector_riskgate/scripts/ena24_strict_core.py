"""Pure helpers for the strict ENA24 selector/gate split."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from selector_core import (
    assign_group_disjoint_split,
    connected_hash_components,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash256(path: Path) -> int:
    """Return a 256-bit horizontal grayscale dHash.

    Resizing to 17 columns by 16 rows gives 16 horizontal comparisons in
    each of 16 rows. The first comparison becomes the most-significant bit.
    """
    with Image.open(path) as image:
        grayscale = image.convert("L").resize(
            (17, 16), resample=Image.Resampling.LANCZOS
        )
        pixels = list(grayscale.getdata())
    value = 0
    for row in range(16):
        offset = row * 17
        for column in range(16):
            value = (value << 1) | int(
                pixels[offset + column] < pixels[offset + column + 1]
            )
    return value


def build_near_duplicate_split_manifest(
    records: Iterable[Mapping[str, Any]],
    *,
    max_hamming: int,
    target_selector_images: int,
    seed: int,
) -> dict[str, Any]:
    """Build a deterministic whole-component selector/gate assignment."""
    normalized = sorted(
        (
            {
                "image_id": int(record["image_id"]),
                "identity": str(record["identity"]),
                "class_ids": sorted(set(map(int, record.get("class_ids", ())))),
                "dhash256": int(record["dhash256"]),
                "source_sha256": str(record["source_sha256"]),
            }
            for record in records
        ),
        key=lambda record: record["image_id"],
    )
    image_ids = [record["image_id"] for record in normalized]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Split records contain duplicate image IDs")
    components = connected_hash_components(
        {record["image_id"]: record["dhash256"] for record in normalized},
        max_hamming=int(max_hamming),
    )
    assigned = assign_group_disjoint_split(
        components,
        class_ids_by_image={
            record["image_id"]: record["class_ids"] for record in normalized
        },
        target_selector_images=int(target_selector_images),
        seed=int(seed),
    )
    selector_ids = set(map(int, assigned["selector_image_ids"]))
    gate_ids = set(map(int, assigned["gate_image_ids"]))
    if selector_ids & gate_ids or selector_ids | gate_ids != set(image_ids):
        raise RuntimeError("Near-duplicate split is not a disjoint full partition")

    component_rows = []
    component_by_image = {}
    partition_by_image = {}
    for index, component in enumerate(components):
        members = set(map(int, component))
        if members <= selector_ids:
            partition = "selector"
        elif members <= gate_ids:
            partition = "gate"
        else:
            raise RuntimeError("A near-duplicate component crosses the split")
        component_id = f"component_{index:04d}"
        component_rows.append(
            {
                "component_id": component_id,
                "partition": partition,
                "image_ids": sorted(members),
                "size": len(members),
            }
        )
        for image_id in members:
            component_by_image[image_id] = component_id
            partition_by_image[image_id] = partition

    image_rows = [
        {
            **record,
            "dhash256": f"{int(record['dhash256']):064x}",
            "component_id": component_by_image[record["image_id"]],
            "partition": partition_by_image[record["image_id"]],
        }
        for record in normalized
    ]
    return {
        "split_kind": "near_duplicate_group_disjoint",
        "event_camera_disjoint": "unknown",
        "hash_algorithm": "grayscale_horizontal_dhash_17x16_to_256bit",
        "hamming_threshold": int(max_hamming),
        "component_algorithm": "union_find_connected_components",
        "assignment_algorithm": "whole_component_rare_class_and_size_balance",
        "seed": int(seed),
        "target_selector_images": int(target_selector_images),
        "selector_image_ids": sorted(selector_ids),
        "gate_image_ids": sorted(gate_ids),
        "selector_images": len(selector_ids),
        "gate_images": len(gate_ids),
        "selector_class_counts": {
            str(key): int(value)
            for key, value in assigned["selector_class_counts"].items()
        },
        "gate_class_counts": {
            str(key): int(value)
            for key, value in assigned["gate_class_counts"].items()
        },
        "components": component_rows,
        "images": image_rows,
    }


def require_strict_provisional_marker(output: Path) -> None:
    marker = Path(output) / "STRICT_PROVISIONAL_FROZEN"
    if not marker.is_file():
        raise RuntimeError(
            "STRICT_PROVISIONAL_FROZEN is required; the legacy "
            "PROVISIONAL_FROZEN marker is superseded and must not be used"
        )


def apply_strict_risk_gate(
    *,
    reference: Mapping[str, Any],
    provisional: Mapping[str, Any],
    mode: str,
    ap_delta_ci_low: float,
    kb_delta_ci_high: float,
    ap_tolerance: float,
) -> dict[str, Any]:
    """Apply the pointwise AP gate and the mode-specific request-cost gate."""
    if mode not in {"fixed_call", "online", "fixed_byte"}:
        raise ValueError(f"Unsupported mode: {mode}")
    ap_passed = float(ap_delta_ci_low) >= -float(ap_tolerance)
    cost_passed = (
        True if mode == "fixed_byte" else float(kb_delta_ci_high) <= 0.0
    )
    passed = bool(ap_passed and cost_passed)
    return {
        "reference": dict(reference),
        "provisional": dict(provisional),
        "frozen": dict(provisional) if passed else dict(reference),
        "ap_delta_ci_low": float(ap_delta_ci_low),
        "kb_delta_ci_high": float(kb_delta_ci_high),
        "ap_gate_passed": bool(ap_passed),
        "cost_gate_applied": mode != "fixed_byte",
        "cost_gate_passed": bool(cost_passed),
        "gate_passed": passed,
        "used_reference_fallback": not passed,
        "fallback_rule": "reference_without_second_candidate",
    }


def subset_coco_gt(
    gt: Mapping[str, Any], image_ids: Iterable[int]
) -> dict[str, Any]:
    selected = set(map(int, image_ids))
    copied = {
        key: value
        for key, value in gt.items()
        if key not in {"images", "annotations"}
    }
    copied["images"] = [
        image for image in gt["images"] if int(image["id"]) in selected
    ]
    copied["annotations"] = [
        annotation
        for annotation in gt["annotations"]
        if int(annotation["image_id"]) in selected
    ]
    observed = {int(image["id"]) for image in copied["images"]}
    if observed != selected:
        raise ValueError(f"GT does not contain requested image IDs: {selected - observed}")
    return copied
