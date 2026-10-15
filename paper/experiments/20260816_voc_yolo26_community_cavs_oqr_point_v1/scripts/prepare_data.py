#!/usr/bin/env python3
"""Extract canonical VOC archives and build deterministic router roles."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET
from typing import Any, Mapping

import yaml


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_module("voc_point_protocol", ROOT / "scripts/protocol_core.py")
DATA = load_module("voc_point_data", ROOT / "scripts/data_core.py")


def _extract_archive(path: Path, target: Path) -> None:
    with tarfile.open(path) as archive:
        root = target.resolve()
        for member in archive.getmembers():
            resolved = (target / member.name).resolve()
            if root not in resolved.parents and resolved != root:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        archive.extractall(target)


def extract_archives() -> Path:
    output = RUNS / "data/raw"
    completion = output / "completion.json"
    if completion.is_file():
        return output / "VOCdevkit"
    archive_root = Path(PROTOCOL["data"]["archive_root"])
    archives = [archive_root / name for name in PROTOCOL["data"]["archives"]]
    missing = [str(path) for path in archives if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    output.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        _extract_archive(archive, output)
    devkit = output / "VOCdevkit"
    if not (devkit / "VOC2007").is_dir() or not (devkit / "VOC2012").is_dir():
        raise RuntimeError("VOC archive extraction is incomplete")
    DATA.atomic_json(
        completion,
        {
            "complete": True,
            "archives": {str(path): DATA.file_sha256(path) for path in archives},
        },
    )
    return devkit


def split_ids(devkit: Path, year: str, split: str) -> list[str]:
    path = devkit / f"VOC{year}/ImageSets/Main/{split}.txt"
    values = [line.strip().split()[0] for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise RuntimeError(f"duplicate VOC identifiers: {year}/{split}")
    return values


def source_records(devkit: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    records: dict[str, dict[str, str]] = {}
    training = []
    for year, split in PROTOCOL["data"]["train_sources"]:
        for image_name in split_ids(devkit, str(year), str(split)):
            identity = f"{year}:{image_name}"
            if identity in records:
                raise RuntimeError(identity)
            records[identity] = {"year": str(year), "image_name": image_name}
            training.append(identity)
    year, split = PROTOCOL["data"]["final_source"]
    final = []
    for image_name in split_ids(devkit, str(year), str(split)):
        identity = f"{year}:{image_name}"
        if identity in records:
            raise RuntimeError(f"VOC final overlaps trainval: {identity}")
        records[identity] = {"year": str(year), "image_name": image_name}
        final.append(identity)
    return records, final


def image_id(year: str, image_name: str) -> int:
    return int(year) * 1_000_000 + int(image_name)


def parse_annotation(path: Path, category_index: Mapping[str, int]) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is None:
        raise RuntimeError(f"missing size: {path}")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    objects = []
    for item in root.findall("object"):
        name = item.findtext("name", "")
        if name not in category_index or int(item.findtext("difficult", "0")) == 1:
            continue
        box = item.find("bndbox")
        if box is None:
            continue
        coordinates = [float(box.findtext(key, "0")) for key in ("xmin", "ymin", "xmax", "ymax")]
        coco = DATA.voc_box_to_coco(*coordinates)
        if coco[2] <= 0 or coco[3] <= 0:
            continue
        objects.append(
            {
                "category_id": int(category_index[name]),
                "coco_bbox": coco,
                "yolo_bbox": DATA.voc_box_to_yolo(*coordinates, width=width, height=height),
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
    if completion.is_file():
        return DATA.read_json(completion)
    categories = list(map(str, PROTOCOL["data"]["category_names"]))
    category_index = {name: index for index, name in enumerate(categories)}
    gt = {
        "info": {"description": f"VOC {role} for CAVS point benchmark"},
        "licenses": [],
        "categories": [
            {"id": index, "name": name, "supercategory": "object"}
            for index, name in enumerate(categories)
        ],
        "images": [],
        "annotations": [],
    }
    annotation_id = 1
    for identity in sorted(identities):
        source = records[identity]
        year, image_name = str(source["year"]), str(source["image_name"])
        original = devkit / f"VOC{year}/JPEGImages/{image_name}.jpg"
        annotation = devkit / f"VOC{year}/Annotations/{image_name}.xml"
        if not original.is_file() or not annotation.is_file():
            raise FileNotFoundError(f"{original} / {annotation}")
        staged_name = f"{year}_{image_name}.jpg"
        staged = RUNS / f"data/images/{role}/{staged_name}"
        label = RUNS / f"data/labels/{role}/{Path(staged_name).stem}.txt"
        _symlink(original, staged)
        parsed = parse_annotation(annotation, category_index)
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(
            "".join(
                f"{item['category_id']} "
                + " ".join(f"{float(value):.10f}" for value in item["yolo_bbox"])
                + "\n"
                for item in parsed["objects"]
            ),
            encoding="ascii",
        )
        current_id = image_id(year, image_name)
        gt["images"].append(
            {
                "id": current_id,
                "file_name": staged_name,
                "absolute_path": str(staged.resolve()),
                "width": int(parsed["width"]),
                "height": int(parsed["height"]),
                "identity": identity,
            }
        )
        for item in parsed["objects"]:
            bbox = list(map(float, item["coco_bbox"]))
            gt["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": current_id,
                    "category_id": int(item["category_id"]),
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    gt_path = role_root / "gt.json"
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
        "gt_sha256": DATA.file_sha256(gt_path),
        "identity_sha256": __import__("hashlib").sha256(
            "\n".join(sorted(identities)).encode("utf-8")
        ).hexdigest(),
    }
    DATA.atomic_json(completion, report)
    return report


def run() -> dict[str, Any]:
    CORE.validate_protocol(PROTOCOL)
    devkit = extract_archives()
    records, final = source_records(devkit)
    training = sorted(set(records) - set(final))
    split = CORE.deterministic_fit_calibration_split(
        training,
        fit_count=int(PROTOCOL["roles"]["fit"]),
        calibration_count=int(PROTOCOL["roles"]["calibration"]),
        seed=int(PROTOCOL["split_seed"]),
    )
    roles = {**split, "final": sorted(final)}
    if any(set(roles[left]) & set(roles[right]) for left in roles for right in roles if left < right):
        raise RuntimeError("VOC roles overlap")
    reports = {
        role: build_role(devkit, role, identities, records)
        for role, identities in roles.items()
    }
    for role, expected in PROTOCOL["roles"].items():
        if int(reports[role]["images"]) != int(expected):
            raise RuntimeError(f"role count changed: {role}/{reports[role]['images']}")
    template = RUNS / "data/voc_template.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "path": str((RUNS / "data").resolve()),
                "train": str((RUNS / "data/roles/fit/images.txt").resolve()),
                "val": str((RUNS / "data/roles/final/images.txt").resolve()),
                "names": {index: name for index, name in enumerate(PROTOCOL["data"]["category_names"])},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    result = {
        "complete": True,
        "roles": {role: int(value["images"]) for role, value in reports.items()},
        "template": str(template.resolve()),
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

