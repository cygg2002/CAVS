#!/usr/bin/env python3
"""Verify official VOC archives and build frozen router roles with difficult flags."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import xml.etree.ElementTree as ET
from typing import Any, Mapping

from PIL import Image


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("voc_gluoncv_prepare_core", ROOT / "scripts/core.py")
DATA = load_module("voc_gluoncv_prepare_data", ROOT / "scripts/data_core.py")
METRIC = load_module("voc_gluoncv_prepare_metric", ROOT / "scripts/voc_metric.py")


def _validate_archive(path: Path, contract: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(contract["bytes"]):
        raise RuntimeError(f"VOC archive size changed: {path}")
    observed = DATA.file_sha256(path)
    if observed != str(contract["sha256"]):
        raise RuntimeError(f"VOC archive SHA-256 changed: {path}/{observed}")


def _extract_archive(path: Path, target: Path) -> None:
    with tarfile.open(path) as archive:
        root = target.resolve()
        members = archive.getmembers()
        for member in members:
            resolved = (target / member.name).resolve()
            if resolved != root and root not in resolved.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        archive.extractall(target, members=members)


def extract_archives() -> Path:
    output = RUNS / "data/raw"
    completion = output / "completion.json"
    devkit = output / "VOCdevkit"
    if completion.is_file():
        report = DATA.read_json(completion)
        if not (devkit / "VOC2007").is_dir() or not (devkit / "VOC2012").is_dir():
            raise RuntimeError("completed VOC extraction is incomplete")
        if report.get("archives") != {
            name: row["sha256"] for name, row in PROTOCOL["data"]["archives"].items()
        }:
            raise RuntimeError("completed VOC archive contract changed")
        return devkit
    archive_root = Path(PROTOCOL["data"]["archive_root"])
    archives = []
    for name, contract in PROTOCOL["data"]["archives"].items():
        path = archive_root / name
        _validate_archive(path, contract)
        archives.append(path)
    output.mkdir(parents=True, exist_ok=True)
    for path in archives:
        _extract_archive(path, output)
    if not (devkit / "VOC2007").is_dir() or not (devkit / "VOC2012").is_dir():
        raise RuntimeError("VOC archive extraction is incomplete")
    DATA.atomic_json(
        completion,
        {
            "complete": True,
            "archives": {name: row["sha256"] for name, row in PROTOCOL["data"]["archives"].items()},
        },
    )
    return devkit


def split_ids(devkit: Path, year: str, split: str) -> list[str]:
    path = devkit / f"VOC{year}/ImageSets/Main/{split}.txt"
    values = [line.strip().split()[0] for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise RuntimeError(f"duplicate VOC identifiers: {year}/{split}")
    return values


def source_records(devkit: Path) -> tuple[dict[str, dict[str, str]], list[str], list[str]]:
    records: dict[str, dict[str, str]] = {}
    training: list[str] = []
    for year, split in PROTOCOL["data"]["train_sources"]:
        for image_name in split_ids(devkit, str(year), str(split)):
            identity = f"{year}:{image_name}"
            if identity in records:
                raise RuntimeError(f"duplicate VOC identity: {identity}")
            records[identity] = {"year": str(year), "image_name": image_name}
            training.append(identity)
    year, split = PROTOCOL["data"]["final_source"]
    final: list[str] = []
    for image_name in split_ids(devkit, str(year), str(split)):
        identity = f"{year}:{image_name}"
        if identity in records:
            raise RuntimeError(f"VOC final overlaps trainval: {identity}")
        records[identity] = {"year": str(year), "image_name": image_name}
        final.append(identity)
    return records, sorted(training), sorted(final)


def image_id(year: str, image_name: str) -> int:
    return int(year) * 1_000_000 + int(image_name)


def parse_annotation(path: Path, category_index: Mapping[str, int]) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is None:
        raise RuntimeError(f"missing VOC size: {path}")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid VOC size: {path}/{width}x{height}")
    objects = []
    for item in root.findall("object"):
        name = item.findtext("name", "")
        if name not in category_index:
            raise RuntimeError(f"unknown VOC class: {path}/{name}")
        box = item.find("bndbox")
        if box is None:
            raise RuntimeError(f"missing VOC box: {path}/{name}")
        coordinates = [float(box.findtext(key, "0")) for key in ("xmin", "ymin", "xmax", "ymax")]
        xyxy = METRIC.voc_box_to_xyxy(*coordinates)
        bbox = METRIC.xyxy_to_xywh(xyxy)
        if bbox[2] <= 0.0 or bbox[3] <= 0.0:
            raise RuntimeError(f"degenerate VOC box: {path}/{coordinates}")
        objects.append(
            {
                "category_id": int(category_index[name]),
                "bbox": bbox,
                "difficult": int(item.findtext("difficult", "0")) == 1,
                "truncated": int(item.findtext("truncated", "0")) == 1,
            }
        )
    return {"width": width, "height": height, "objects": objects}


def _symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"staged image drift: {target}")
        return
    target.symlink_to(source.resolve())


def build_role(
    devkit: Path,
    role: str,
    identities: list[str],
    records: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    role_root = RUNS / f"data/roles/{role}"
    completion = role_root / "completion.json"
    gt_path = role_root / "gt.json"
    if completion.is_file():
        report = DATA.read_json(completion)
        if report.get("gt_sha256") != DATA.file_sha256(gt_path):
            raise RuntimeError(f"completed GT changed: {role}")
        return report
    categories = list(map(str, PROTOCOL["data"]["category_names"]))
    category_index = {name: index + 1 for index, name in enumerate(categories)}
    gt: dict[str, Any] = {
        "info": {
            "description": f"VOC {role}: difficult retained for metric-time ignore",
            "protocol_id": PROTOCOL["protocol_id"],
        },
        "licenses": [],
        "categories": [
            {"id": index + 1, "name": name, "supercategory": "object"}
            for index, name in enumerate(categories)
        ],
        "images": [],
        "annotations": [],
    }
    annotation_id = 1
    difficult_count = 0
    original_bytes = 0
    for identity in sorted(identities):
        source = records[identity]
        year, image_name = str(source["year"]), str(source["image_name"])
        original = devkit / f"VOC{year}/JPEGImages/{image_name}.jpg"
        annotation = devkit / f"VOC{year}/Annotations/{image_name}.xml"
        if not original.is_file() or not annotation.is_file():
            raise FileNotFoundError(f"{original} / {annotation}")
        parsed = parse_annotation(annotation, category_index)
        with Image.open(original) as image:
            if image.size != (int(parsed["width"]), int(parsed["height"])):
                raise RuntimeError(f"VOC XML/JPEG size mismatch: {identity}")
        staged_name = f"{year}_{image_name}.jpg"
        staged = RUNS / f"data/images/{role}/{staged_name}"
        _symlink(original, staged)
        current_id = image_id(year, image_name)
        gt["images"].append(
            {
                "id": current_id,
                "file_name": staged_name,
                "absolute_path": str(staged.resolve()),
                "source_path": str(original.resolve()),
                "width": int(parsed["width"]),
                "height": int(parsed["height"]),
                "identity": identity,
                "original_bytes": original.stat().st_size,
            }
        )
        original_bytes += original.stat().st_size
        for item in parsed["objects"]:
            difficult = bool(item["difficult"])
            difficult_count += int(difficult)
            bbox = list(map(float, item["bbox"]))
            gt["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": current_id,
                    "category_id": int(item["category_id"]),
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "difficult": int(difficult),
                    "ignore": int(difficult),
                    "truncated": int(bool(item["truncated"])),
                }
            )
            annotation_id += 1
    DATA.atomic_json(gt_path, gt)
    image_list = role_root / "images.txt"
    image_list.parent.mkdir(parents=True, exist_ok=True)
    image_list.write_text(
        "".join(f"{image['absolute_path']}\n" for image in gt["images"]),
        encoding="utf-8",
    )
    report = {
        "complete": True,
        "role": role,
        "images": len(gt["images"]),
        "annotations": len(gt["annotations"]),
        "difficult_annotations": difficult_count,
        "original_bytes": original_bytes,
        "gt_sha256": DATA.file_sha256(gt_path),
        "identity_sha256": hashlib.sha256(
            "\n".join(sorted(identities)).encode("utf-8")
        ).hexdigest(),
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(completion, report)
    return report


def run() -> dict[str, Any]:
    CORE.validate_protocol(PROTOCOL)
    devkit = extract_archives()
    records, training, final = source_records(devkit)
    split = CORE.deterministic_fit_calibration_split(
        training,
        fit_count=int(PROTOCOL["roles"]["fit"]),
        calibration_count=int(PROTOCOL["roles"]["calibration"]),
        seed=int(PROTOCOL["split_seed"]),
    )
    roles = {**split, "final": final}
    names = list(roles)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if set(roles[left]) & set(roles[right]):
                raise RuntimeError(f"VOC roles overlap: {left}/{right}")
    reports = {
        role: build_role(devkit, role, identities, records)
        for role, identities in roles.items()
    }
    for role, expected in PROTOCOL["roles"].items():
        if int(reports[role]["images"]) != int(expected):
            raise RuntimeError(f"role count changed: {role}/{reports[role]['images']}")
    result = {
        "complete": True,
        "roles": {role: int(value["images"]) for role, value in reports.items()},
        "ground_truth_semantics": "VOC difficult retained and ignored during matching",
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(RUNS / "data/report.json", result)
    DATA.atomic_json(RUNS / "data/completion.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

