#!/usr/bin/env python3
"""Build canonical COCO role bundles with twelve Reversible ROI actions."""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
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


CORE = load_module("coco_oqr12_dataset_core", ROOT / "scripts/core.py")
SOURCE_ROOT = Path(PROTOCOL["source_training_experiment"])
SOURCE = load_module(
    "coco_oqr12_common_source", SOURCE_ROOT / "scripts/build_common.py"
)
SOURCE.PROTOCOL = CORE.protocol_for_reused_fusion_source(PROTOCOL)
SOURCE.ROOT = ROOT
SOURCE.RUNS = RUNS


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def canonical_manifest(role: str) -> tuple[Path, dict[str, Any]]:
    path = Path(PROTOCOL["canonical_root"]) / f"common/coco/{role}/source_manifest.json"
    return path, read_json(path)


def q90_manifest_path(manifest: Mapping[str, Any]) -> Path:
    matches = [
        Path(path)
        for path in manifest["source_paths"]
        if str(path).endswith("q90_manifest.json")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one Q90 manifest, got {matches}")
    return matches[0]


def role_inputs(role: str) -> dict[str, Any]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(role)
    manifest_path, manifest = canonical_manifest(role)
    roi = {}
    for action_id in PROTOCOL["roi_actions"]:
        profile, scale = str(action_id).split("@", 1)
        roi[action_id] = {
            "manifest": RUNS / f"actions/{role}/roi/{profile}/geometry/roi_manifest.json",
            "predictions": RUNS
            / f"actions/{role}/roi/{profile}/imgsz_{scale}/roi_predictions_mapped.json",
        }
    return {
        "gt": Path(manifest["gt"]),
        "old_samples": Path(manifest["samples"]),
        "source_manifest": manifest_path,
        "q90_manifest": q90_manifest_path(manifest),
        "actions": {
            name: Path(manifest["action_sources"][name])
            for name in ("edge", "original", "q90")
        },
        "roi": roi,
    }


def action_file(role: str, action_id: str) -> Path:
    return RUNS / f"common/coco/{role}/actions/{action_id.replace('@', '__')}.json"


def _original_bytes(image: Mapping[str, Any], q90: Mapping[str, Any]) -> int:
    if q90.get("original_bytes") is not None:
        return int(q90["original_bytes"])
    return Path(str(image.get("absolute_path") or image["file_name"])).stat().st_size


def _q90_bytes(record: Mapping[str, Any]) -> int:
    for key in ("action_bytes", "compressed_bytes", "bytes"):
        if record.get(key) is not None:
            return int(record[key])
    raise KeyError("Q90 manifest has no byte field")


def build_role(role: str) -> dict[str, Any]:
    output = RUNS / f"common/coco/{role}"
    completion = output / "completion.json"
    if completion.is_file():
        report = read_json(completion)
        if report.get("cloud_model_sha256") != PROTOCOL["cloud_model"]["sha256"]:
            raise RuntimeError(f"common cache cloud hash drift: {role}")
        return report
    inputs = role_inputs(role)
    required = [
        inputs["gt"],
        inputs["old_samples"],
        inputs["source_manifest"],
        inputs["q90_manifest"],
        *inputs["actions"].values(),
    ]
    for paths in inputs["roi"].values():
        required.extend(paths.values())
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing[:5])

    gt = read_json(inputs["gt"])
    images = {int(image["id"]): image for image in gt["images"]}
    old_samples = read_json(inputs["old_samples"])
    if len(old_samples) != int(PROTOCOL["roles"][role]):
        raise RuntimeError(f"canonical Edge feature population changed: {role}")
    old_by_id = {int(sample["image_id"]): sample for sample in old_samples}
    q90_records = SOURCE.manifest_records(inputs["q90_manifest"])
    actions = {
        name: SOURCE.group_predictions(path) for name, path in inputs["actions"].items()
    }
    roi_assets = {
        action_id: {
            "records": SOURCE.manifest_records(paths["manifest"]),
            "raw": SOURCE.group_predictions(paths["predictions"]),
            "reversible": {},
        }
        for action_id, paths in inputs["roi"].items()
    }
    gt_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in gt["annotations"]:
        gt_map[int(annotation["image_id"])].append(
            {
                "category_id": int(annotation["category_id"]),
                "bbox": list(map(float, annotation["bbox"])),
            }
        )

    samples = []
    for image_id in sorted(images):
        image = images[image_id]
        old = old_by_id[image_id]
        identity = str(image.get("absolute_path") or image["file_name"])
        if str(old["identity"]) != identity:
            raise RuntimeError(f"Edge feature identity drift: {role}/{image_id}")
        q90_record = q90_records[image_id]
        original_bytes = _original_bytes(image, q90_record)
        q90_bytes = _q90_bytes(q90_record)
        payloads: dict[str, int] = {"original": original_bytes, "q90": q90_bytes}
        qualities: dict[str, float] = {
            action: float(
                SOURCE.BENCHMARK.image_quality(
                    gt_map[image_id], actions[action].get(image_id, [])
                )
            )
            for action in ("edge", "original", "q90")
        }
        available: dict[str, bool] = {
            "original": True,
            "q90": 0 < q90_bytes < original_bytes,
        }
        action_records = [
            {"action_id": "edge", "payload_bytes": 0, "available": True},
            {"action_id": "original", "payload_bytes": original_bytes, "available": True},
            {"action_id": "q90", "payload_bytes": q90_bytes, "available": available["q90"]},
        ]
        roi_payloads: list[int] = []
        roi_areas: list[float] = []
        for action_id in PROTOCOL["roi_actions"]:
            asset = roi_assets[action_id]
            record = asset["records"].get(image_id)
            if record is None:
                available[action_id] = False
                continue
            _, reversible = SOURCE._roi_fusions(
                actions["edge"].get(image_id, []),
                asset["raw"].get(image_id, []),
                record,
            )
            asset["reversible"][image_id] = reversible
            payload = int(record["bytes"])
            area = float(record["area_ratio"])
            payloads[action_id] = payload
            available[action_id] = True
            qualities[action_id] = float(
                SOURCE.BENCHMARK.image_quality(gt_map[image_id], reversible)
            )
            roi_payloads.append(payload)
            roi_areas.append(area)
            action_records.append(
                {
                    "action_id": action_id,
                    "payload_bytes": payload,
                    "available": True,
                    "profile": str(record["profile"]),
                    "imgsz": int(action_id.split("@", 1)[1]),
                    "xyxy": list(map(int, record["xyxy"])),
                    "crop_width": int(record["crop_width"]),
                    "crop_height": int(record["crop_height"]),
                    "area_ratio": area,
                }
            )
        gains = {
            action: float(quality) - float(qualities["edge"])
            for action, quality in qualities.items()
            if action != "edge" and available.get(action, False)
        }
        valid_qualities = {
            action: quality
            for action, quality in qualities.items()
            if action != "edge" and available.get(action, False)
        }
        valid_payloads = {action: payloads[action] for action in valid_qualities}
        features = list(map(float, old["features"]))
        samples.append(
            {
                "image_id": image_id,
                "identity": str(old["identity"]),
                "file_name": str(image["file_name"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "event_id": f"coco:{image_id}",
                "camera": "coco",
                "class_ids": list(map(int, old["class_ids"])),
                "confidence": float(old["confidence"]),
                "structured": list(map(float, old["structured"])),
                "pca32": list(map(float, old["pca32"])),
                "features": features,
                "action_features": SOURCE.DATA_CORE.rebuild_observable_action_features(
                    features,
                    original_bytes=original_bytes,
                    q90_bytes=q90_bytes,
                    roi_payloads=roi_payloads,
                    roi_areas=roi_areas,
                ),
                "qualities": qualities,
                "gains": gains,
                "target": max(gains.values(), default=0.0),
                "best_proxy_action": SOURCE.DATA_CORE.proxy_best_action(
                    valid_qualities, valid_payloads
                ),
                "payload_bytes": payloads,
                "available": available,
                "actions": action_records,
                "roi_area_ratio": min(roi_areas, default=0.0),
                "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
            }
        )

    action_sources = {
        name: str(Path(path).resolve()) for name, path in inputs["actions"].items()
    }
    for action_id, asset in roi_assets.items():
        path = action_file(role, action_id)
        SOURCE.atomic_json(path, SOURCE._flat(asset["reversible"]))
        action_sources[action_id] = str(path.resolve())
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "gt.json", gt)
    atomic_json(output / "samples.json", samples)
    source_hashes = {
        str(Path(path).resolve()): file_sha256(Path(path)) for path in required
    }
    manifest = {
        "dataset": "coco",
        "role": role,
        "images": len(samples),
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "action_sources": action_sources,
        "source_paths": source_hashes,
        "gt": str((output / "gt.json").resolve()),
        "samples": str((output / "samples.json").resolve()),
        "samples_sha256": file_sha256(output / "samples.json"),
        "real_roi_geometry": True,
        "primary_roi_fusion": "Reversible",
        "reversible_parameters": PROTOCOL["reversible_fusion"],
        "original_q90_predictions_reused": True,
        "roi_predictions_reused": False,
        "old_learner_checkpoints_reused": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "source_manifest.json", manifest)
    report = {
        "complete": True,
        "role": role,
        "images": len(samples),
        "actions": len(action_sources),
        "roi_actions": len(PROTOCOL["roi_actions"]),
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "samples_sha256": file_sha256(output / "samples.json"),
        "manifest_sha256": file_sha256(output / "source_manifest.json"),
        "final_used_for_training_or_selection": False,
    }
    atomic_json(completion, report)
    return report


def run_all() -> dict[str, Any]:
    if not (RUNS / "cloud_cache/completion.json").is_file():
        raise RuntimeError("COCO R12 prediction cache must complete first")
    roles = {role: build_role(role) for role in PROTOCOL["roles"]}
    report = {
        "complete": True,
        "roles": {role: int(value["images"]) for role, value in roles.items()},
        "actions": 15,
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "final_used_for_training_or_selection": False,
    }
    atomic_json(RUNS / "common/report.json", report)
    atomic_json(RUNS / "common/completion.json", {"complete": True})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=(*PROTOCOL["roles"], "all"), default="all")
    args = parser.parse_args()
    result = run_all() if args.role == "all" else build_role(args.role)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
