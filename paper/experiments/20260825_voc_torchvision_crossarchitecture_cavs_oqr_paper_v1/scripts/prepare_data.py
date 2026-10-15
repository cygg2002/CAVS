#!/usr/bin/env python3
"""Import and verify the already frozen VOC router roles without re-splitting data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
from typing import Any, Mapping

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


def _source_experiment() -> Path:
    return (ROOT / str(PROTOCOL["source_data_experiment"])).resolve()


def validate_source_role(
    source_root: Path,
    role: str,
    contract: Mapping[str, Any],
    *,
    require_images: bool = True,
) -> dict[str, Any]:
    role_root = source_root / f"runs/data/roles/{role}"
    gt_path = role_root / "gt.json"
    images_path = role_root / "images.txt"
    if DATA.file_sha256(gt_path) != str(contract["gt_sha256"]):
        raise RuntimeError(f"ground-truth hash changed: {role}")
    if DATA.file_sha256(images_path) != str(contract["images_sha256"]):
        raise RuntimeError(f"image-list hash changed: {role}")
    gt = DATA.read_json(gt_path)
    images = [dict(row) for row in gt.get("images", ())]
    if len(images) != int(PROTOCOL["roles"][role]):
        raise RuntimeError(f"frozen role population changed: {role}")
    identities = [str(row["identity"]) for row in images]
    identity_sha256 = hashlib.sha256(
        "\n".join(sorted(identities)).encode("utf-8")
    ).hexdigest()
    if identity_sha256 != str(contract["identity_sha256"]):
        raise RuntimeError(f"identity hash changed: {role}")
    listed = [line.strip() for line in images_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_paths = [str(row["absolute_path"]) for row in images]
    if listed != expected_paths:
        raise RuntimeError(f"image-list order changed: {role}")
    if require_images:
        missing = [path for path in expected_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"missing {len(missing)} source images for {role}: {missing[0]}")
    return {
        "role": role,
        "images": len(images),
        "annotations": len(gt.get("annotations", ())),
        "difficult_annotations": sum(
            int(bool(row.get("difficult", row.get("ignore", 0))))
            for row in gt.get("annotations", ())
        ),
        "original_bytes": sum(int(row.get("original_bytes", 0)) for row in images),
        "gt_sha256": DATA.file_sha256(gt_path),
        "images_sha256": DATA.file_sha256(images_path),
        "identity_sha256": identity_sha256,
        "gt_path": str(gt_path.resolve()),
        "images_path": str(images_path.resolve()),
    }


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def import_role(role: str) -> dict[str, Any]:
    contract = PROTOCOL["role_contracts"][role]
    source = validate_source_role(_source_experiment(), role, contract)
    output = RUNS / f"data/roles/{role}"
    gt_path = output / "gt.json"
    images_path = output / "images.txt"
    completion = output / "completion.json"
    if completion.is_file():
        report = DATA.read_json(completion)
        if report.get("gt_sha256") != DATA.file_sha256(gt_path):
            raise RuntimeError(f"completed ground truth changed: {role}")
        if report.get("images_sha256") != DATA.file_sha256(images_path):
            raise RuntimeError(f"completed image list changed: {role}")
        return report
    _atomic_copy(Path(source["gt_path"]), gt_path)
    _atomic_copy(Path(source["images_path"]), images_path)
    report = {
        **{key: value for key, value in source.items() if not key.endswith("_path")},
        "complete": True,
        "source_experiment": str(_source_experiment()),
        "protocol_sha256": DATA.file_sha256(ROOT / "configs/protocol.json"),
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(completion, report)
    return report


def run() -> dict[str, Any]:
    CORE.validate_protocol(PROTOCOL)
    reports = {role: import_role(role) for role in PROTOCOL["roles"]}
    for role, expected in PROTOCOL["roles"].items():
        if int(reports[role]["images"]) != int(expected):
            raise RuntimeError(f"role count changed: {role}/{reports[role]['images']}")
    identities = {
        role: {
            str(row["identity"])
            for row in DATA.read_json(RUNS / f"data/roles/{role}/gt.json")["images"]
        }
        for role in PROTOCOL["roles"]
    }
    names = list(identities)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if identities[left] & identities[right]:
                raise RuntimeError(f"VOC roles overlap: {left}/{right}")
    result = {
        "complete": True,
        "roles": {role: int(value["images"]) for role, value in reports.items()},
        "source_experiment": str(_source_experiment()),
        "source_roles_reused_without_resplitting": True,
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
