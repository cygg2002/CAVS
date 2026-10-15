#!/usr/bin/env python3
"""Materialize hash-bound board input manifests from the frozen role cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import contracts


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "configs" / "protocol.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_role(role: str) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path]:
    source_manifest = contracts.role_manifest(PROTOCOL["canonical_root"], role)
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    samples_path = Path(str(source["samples"]))
    gt_path = Path(str(source["gt"]))
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    contracts.validate_population([int(row["image_id"]) for row in samples], role=role)
    return source, samples, source_manifest, gt_path


def materialize(output: Path, *, copy_images: bool, copy_roles: set[str]) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    identities: dict[str, set[str]] = {}
    for role in contracts.ROLES:
        source, samples, source_manifest, gt_path = read_role(role)
        records = []
        for sample in samples:
            identity = Path(str(sample["identity"])).resolve()
            if not identity.is_file():
                raise FileNotFoundError(identity)
            file_name = Path(str(sample.get("file_name") or identity.name)).name
            if file_name != identity.name:
                raise RuntimeError(f"sample filename mismatch: {role}/{sample['image_id']}")
            records.append(
                {
                    "image_id": int(sample["image_id"]),
                    "file_name": file_name,
                    "width": int(sample["width"]),
                    "height": int(sample["height"]),
                    "source_split": identity.parent.name,
                    "native_jpeg_bytes": int(identity.stat().st_size),
                }
            )
            identities.setdefault(identity.parent.name, set()).add(str(identity))
            if copy_images and role in copy_roles:
                destination = output / "images" / identity.parent.name / file_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_file() and sha256(destination) == sha256(identity):
                    continue
                temporary = destination.with_name(f".{destination.name}.tmp")
                shutil.copy2(identity, temporary)
                temporary.replace(destination)
        input_manifest = {
            "schema": "rknn_candidate_input_v1",
            "dataset": "coco",
            "role": role,
            "images": len(records),
            "source_manifest_sha256": sha256(source_manifest),
            "source_manifest": str(source_manifest),
            "source_gt": str(gt_path),
            "records": records,
        }
        atomic_json(output / "manifests" / f"{role}.json", input_manifest)
        roles[role] = {
            "images": len(records),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": input_manifest["source_manifest_sha256"],
            "input_manifest": str((output / "manifests" / f"{role}.json").resolve()),
        }
    overlap = {left: len(identities[left] & identities[right]) for left in identities for right in identities if left < right}
    summary = {
        "protocol_id": PROTOCOL["protocol_id"],
        "roles": roles,
        "identity_overlap_by_split": overlap,
        "copied_images": bool(copy_images),
        "copied_roles": sorted(copy_roles) if copy_images else [],
    }
    atomic_json(output / "input_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--copy-roles", nargs="+", choices=tuple(contracts.ROLES), default=("fit", "calibration"))
    args = parser.parse_args()
    print(json.dumps(materialize(args.output.resolve(), copy_images=args.copy_images, copy_roles=set(args.copy_roles)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
