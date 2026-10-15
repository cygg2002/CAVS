#!/usr/bin/env python3
"""Build and validate the annotation-free image sequence used by the board."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA = "coco_images_only_sequence_v1"
RECORD_FIELDS = frozenset(("image_id", "file_name", "width", "height"))
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(records))).hexdigest()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _record_from_coco_image(image: Mapping[str, Any]) -> dict[str, Any]:
    missing = {"id", "file_name", "width", "height"} - set(image)
    if missing:
        raise ValueError(f"COCO image is missing fields: {sorted(missing)}")
    image_id = _positive_int(image["id"], "image_id")
    file_name = image["file_name"]
    if (
        not isinstance(file_name, str)
        or not file_name
        or Path(file_name).is_absolute()
        or ".." in Path(file_name).parts
    ):
        raise ValueError(f"invalid file_name for image_id {image_id}")
    return {
        "image_id": image_id,
        "file_name": file_name,
        "width": _positive_int(image["width"], f"width for image_id {image_id}"),
        "height": _positive_int(image["height"], f"height for image_id {image_id}"),
    }


def _validate_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise ValueError("manifest records must be a non-empty list")
    validated: list[dict[str, Any]] = []
    image_ids: set[int] = set()
    file_names: set[str] = set()
    for position, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"manifest record {position} must be an object")
        if set(raw) != RECORD_FIELDS:
            raise ValueError(
                f"manifest record {position} must contain exactly {sorted(RECORD_FIELDS)}"
            )
        image_id = _positive_int(raw["image_id"], f"image_id at position {position}")
        file_name = raw["file_name"]
        if (
            not isinstance(file_name, str)
            or not file_name
            or Path(file_name).is_absolute()
            or ".." in Path(file_name).parts
        ):
            raise ValueError(f"invalid file_name at position {position}")
        if image_id in image_ids:
            raise ValueError(f"duplicate image_id {image_id}")
        if file_name in file_names:
            raise ValueError(f"duplicate file_name {file_name}")
        image_ids.add(image_id)
        file_names.add(file_name)
        validated.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "width": _positive_int(raw["width"], f"width at position {position}"),
                "height": _positive_int(raw["height"], f"height at position {position}"),
            }
        )
    if [record["image_id"] for record in validated] != sorted(image_ids):
        raise ValueError("manifest records must be ordered by ascending image_id")
    return validated


def build_manifest(
    annotation_path: str | Path,
    *,
    split: str,
    dataset: str = "coco",
) -> dict[str, Any]:
    annotation_path = Path(annotation_path)
    try:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read COCO annotations {annotation_path}: {exc}") from exc
    if not isinstance(annotation, Mapping) or not isinstance(annotation.get("images"), list):
        raise ValueError("COCO annotations must contain an images list")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("dataset must be a non-empty string")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a non-empty string")

    raw_records = [_record_from_coco_image(image) for image in annotation["images"]]
    raw_records.sort(key=lambda record: record["image_id"])
    records = _validate_records(raw_records)
    return {
        "schema": SCHEMA,
        "dataset": dataset,
        "split": split,
        "images": len(records),
        "source_annotations_sha256": file_sha256(annotation_path),
        "sequence_sha256": records_sha256(records),
        "records": records,
    }


def validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("sequence manifest must be a JSON object")
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"sequence manifest schema must be {SCHEMA}")
    for field in ("dataset", "split"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"manifest {field} must be a non-empty string")
    records = _validate_records(manifest.get("records"))
    if manifest.get("images") != len(records):
        raise ValueError("manifest images count does not match records")
    source_hash = manifest.get("source_annotations_sha256")
    if not isinstance(source_hash, str) or not SHA256_RE.fullmatch(source_hash):
        raise ValueError("manifest source_annotations_sha256 is invalid")
    expected_sequence_hash = records_sha256(records)
    if manifest.get("sequence_sha256") != expected_sequence_hash:
        raise ValueError("manifest sequence_sha256 does not match records")
    return {
        "schema": SCHEMA,
        "dataset": manifest["dataset"],
        "split": manifest["split"],
        "images": len(records),
        "source_annotations_sha256": source_hash,
        "sequence_sha256": expected_sequence_hash,
        "records": records,
    }


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read sequence manifest {path}: {exc}") from exc
    return validate_manifest(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_manifest(
    annotation_path: str | Path,
    output_path: str | Path,
    *,
    split: str,
    dataset: str = "coco",
) -> dict[str, Any]:
    manifest = build_manifest(annotation_path, split=split, dataset=dataset)
    _atomic_json(Path(output_path), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--split", default="val2017")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_manifest(
        args.annotations,
        args.output,
        split=args.split,
        dataset=args.dataset,
    )
    print(
        f"wrote {manifest['images']} images to {args.output} "
        f"sequence_sha256={manifest['sequence_sha256']}"
    )


if __name__ == "__main__":
    main()
