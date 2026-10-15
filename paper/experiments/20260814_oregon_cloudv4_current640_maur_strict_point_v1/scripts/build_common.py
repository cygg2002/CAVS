#!/usr/bin/env python3
"""Build Cloud-V4 role bundles with Current@640 and MAUR fusion."""

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
CANONICAL = Path(PROTOCOL["canonical_root"])
CANONICAL_SCRIPTS = (
    REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts"
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# joint_core imports the canonical protocol_core by its historical module name.
CANONICAL_PROTOCOL = load_module(
    "protocol_core", CANONICAL_SCRIPTS / "protocol_core.py"
)
BENCHMARK = load_module(
    "oregon_cloudv3_benchmark_core", CANONICAL_SCRIPTS / "benchmark_core.py"
)
JOINT = load_module(
    "oregon_cloudv3_joint_core",
    REPO
    / "paper/experiments/20260728_oregon_cloudv2_joint_action_reversible_fusion/scripts/joint_core.py",
)
DATA_CORE = load_module("oregon_cloudv3_data_core", ROOT / "scripts/data_core.py")


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


def group_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in read_json(path):
        row = {
            "image_id": int(raw["image_id"]),
            "category_id": int(raw["category_id"]),
            "bbox": list(map(float, raw["bbox"])),
            "score": float(raw.get("score", 1.0)),
        }
        output[row["image_id"]].append(row)
    return dict(output)


def manifest_records(path: Path) -> dict[int, dict[str, Any]]:
    payload = read_json(path)
    records = payload.get("records", payload)
    if isinstance(records, list):
        return {int(record["image_id"]): dict(record) for record in records}
    return {int(key): dict(value) for key, value in records.items()}


def action_file(role: str, action_id: str, *, maur: bool = False) -> Path:
    safe = action_id.replace("@", "__")
    family = "actions_maur" if maur else "actions"
    return RUNS / f"common/oregon/{role}/{family}/{safe}.json"


def role_inputs(role: str) -> dict[str, Any]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(role)
    raw_actions = {
        "edge": CANONICAL / f"actions/edge/{role}/predictions.json",
        "original": RUNS / f"actions/{role}/original/predictions.json",
        "q90": RUNS / f"actions/{role}/q90/q90_predictions.json",
    }
    geometries = {}
    for action_id in PROTOCOL["roi_actions"]:
        profile, scale = str(action_id).split("@")
        geometries[action_id] = {
            "manifest": RUNS
            / f"actions/{role}/roi/{profile}/geometry/roi_manifest.json",
            "predictions": RUNS
            / f"actions/{role}/roi/{profile}/imgsz_{scale}/roi_predictions_mapped.json",
        }
    return {
        "gt": CANONICAL / f"gt/oregon/{role}/gt.json",
        "old_samples": CANONICAL / f"common/oregon/{role}/samples.json",
        "fpn": CANONICAL / f"actions/fpn/{role}/global_fpn_features.json",
        "q90_manifest": CANONICAL / f"actions/q90/{role}/q90_manifest.json",
        "actions": raw_actions,
        "roi": geometries,
    }


def assignments_by_name() -> dict[str, dict[str, Any]]:
    canonical_protocol = read_json(Path(PROTOCOL["canonical_protocol"]))
    path = Path(canonical_protocol["datasets"]["oregon"]["assignments"])
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = str(row["master_name"])
        if name in output:
            raise RuntimeError(f"duplicate canonical assignment: {name}")
        output[name] = row
    return output


def _original_bytes(image: Mapping[str, Any], q90: Mapping[str, Any]) -> int:
    if q90.get("original_bytes") is not None:
        return int(q90["original_bytes"])
    source = Path(str(image.get("absolute_path") or image["file_name"]))
    return source.stat().st_size


def _q90_bytes(record: Mapping[str, Any]) -> int:
    for key in ("action_bytes", "compressed_bytes", "bytes"):
        if record.get(key) is not None:
            return int(record[key])
    raise KeyError("Q90 manifest has no byte field")


def _flat(grouped: Mapping[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [row for image_id in sorted(grouped) for row in grouped[image_id]]


def _roi_fusions(
    edge: list[dict[str, Any]],
    cloud: list[dict[str, Any]],
    record: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    maur = CANONICAL_PROTOCOL.class_aware_nms(
        CANONICAL_PROTOCOL.maur_candidates(
            edge, cloud, tuple(map(int, record["xyxy"]))
        ),
        iou_threshold=0.7,
    )
    params = PROTOCOL["reversible_fusion"]["primary_parameters"]
    reversible = JOINT.reversible_fusion(
        edge,
        cloud,
        match_iou=float(params["match_iou"]),
        score_margin=float(params["score_margin"]),
        add_threshold=float(params["add_threshold"]),
        nms_iou=0.7,
    )
    return maur, reversible


def build_role(role: str) -> dict[str, Any]:
    output = RUNS / f"common/oregon/{role}"
    completion = output / "completion.json"
    if completion.is_file():
        report = read_json(completion)
        if report.get("cloud_model_sha256") != PROTOCOL["cloud_model"]["sha256"]:
            raise RuntimeError(f"common cache cloud hash drift: {role}")
        return report
    inputs = role_inputs(role)
    required = [inputs["gt"], inputs["old_samples"], inputs["fpn"], inputs["q90_manifest"]]
    required.extend(inputs["actions"].values())
    for assets in inputs["roi"].values():
        required.extend(assets.values())
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing[:5])

    gt = read_json(inputs["gt"])
    images = {int(image["id"]): image for image in gt["images"]}
    old_samples = read_json(inputs["old_samples"])
    if len(old_samples) != int(PROTOCOL["roles"][role]):
        raise RuntimeError(f"old Edge feature population changed: {role}")
    old_by_id = {int(sample["image_id"]): sample for sample in old_samples}
    q90_records = manifest_records(inputs["q90_manifest"])
    assignments = assignments_by_name()
    actions = {name: group_predictions(path) for name, path in inputs["actions"].items()}
    roi_assets = {}
    for action_id, paths in inputs["roi"].items():
        roi_assets[action_id] = {
            "records": manifest_records(paths["manifest"]),
            "raw": group_predictions(paths["predictions"]),
            "maur": {},
            "reversible": {},
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
        if str(old["identity"]) != str(image.get("absolute_path") or image["file_name"]):
            raise RuntimeError(f"Edge feature identity drift: {role}/{image_id}")
        assignment = assignments.get(Path(str(image["file_name"])).name)
        if assignment is None:
            raise RuntimeError(f"missing event assignment: {image['file_name']}")
        q90_record = q90_records[image_id]
        original_bytes = _original_bytes(image, q90_record)
        q90_bytes = _q90_bytes(q90_record)
        payloads: dict[str, int] = {"original": original_bytes, "q90": q90_bytes}
        qualities: dict[str, float] = {
            "edge": float(BENCHMARK.image_quality(gt_map[image_id], actions["edge"].get(image_id, []))),
            "original": float(BENCHMARK.image_quality(gt_map[image_id], actions["original"].get(image_id, []))),
            "q90": float(BENCHMARK.image_quality(gt_map[image_id], actions["q90"].get(image_id, []))),
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
        roi_payloads, roi_areas = [], []
        for action_id in PROTOCOL["roi_actions"]:
            assets = roi_assets[action_id]
            record = assets["records"].get(image_id)
            if record is None:
                available[action_id] = False
                continue
            maur, reversible = _roi_fusions(
                actions["edge"].get(image_id, []), assets["raw"].get(image_id, []), record
            )
            assets["maur"][image_id] = maur
            assets["reversible"][image_id] = reversible
            payload = int(record["bytes"])
            payloads[action_id] = payload
            available[action_id] = True
            qualities[action_id] = float(BENCHMARK.image_quality(gt_map[image_id], maur))
            roi_payloads.append(payload)
            roi_areas.append(float(record["area_ratio"]))
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
                    "area_ratio": float(record["area_ratio"]),
                }
            )
        gains = {
            action_id: float(quality) - float(qualities["edge"])
            for action_id, quality in qualities.items()
            if action_id != "edge" and available.get(action_id, False)
        }
        valid_qualities = {
            action_id: quality
            for action_id, quality in qualities.items()
            if action_id != "edge" and available.get(action_id, False)
        }
        valid_payloads = {action_id: payloads[action_id] for action_id in valid_qualities}
        best_action = DATA_CORE.proxy_best_action(valid_qualities, valid_payloads)
        features = list(map(float, old["features"]))
        sample = {
            "image_id": image_id,
            "identity": str(old["identity"]),
            "file_name": str(image["file_name"]),
            "width": int(image["width"]),
            "height": int(image["height"]),
            "event_id": str(assignment["event_id"]),
            "camera": str(assignment["camera"]),
            "class_ids": list(map(int, old["class_ids"])),
            "confidence": float(old["confidence"]),
            "structured": list(map(float, old["structured"])),
            "pca32": list(map(float, old["pca32"])),
            "features": features,
            "action_features": DATA_CORE.rebuild_observable_action_features(
                features,
                original_bytes=original_bytes,
                q90_bytes=q90_bytes,
                roi_payloads=roi_payloads,
                roi_areas=roi_areas,
            ),
            "qualities": qualities,
            "gains": gains,
            "target": max(gains.values(), default=0.0),
            "best_proxy_action": best_action,
            "payload_bytes": payloads,
            "available": available,
            "actions": action_records,
            "roi_area_ratio": min(roi_areas, default=0.0),
            "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        }
        samples.append(sample)

    action_sources = {
        "edge": str(inputs["actions"]["edge"].resolve()),
        "original": str(inputs["actions"]["original"].resolve()),
        "q90": str(inputs["actions"]["q90"].resolve()),
    }
    maur_sources = {}
    for action_id, assets in roi_assets.items():
        primary_path = action_file(role, action_id)
        maur_path = action_file(role, action_id, maur=True)
        atomic_json(primary_path, _flat(assets["maur"]))
        atomic_json(maur_path, _flat(assets["maur"]))
        action_sources[action_id] = str(primary_path.resolve())
        maur_sources[action_id] = str(maur_path.resolve())

    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "gt.json", gt)
    atomic_json(output / "samples.json", samples)
    source_hashes = {str(Path(path).resolve()): file_sha256(Path(path)) for path in required}
    manifest = {
        "dataset": "oregon",
        "role": role,
        "images": len(samples),
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "action_sources": action_sources,
        "maur_action_sources": maur_sources,
        "source_paths": source_hashes,
        "gt": str((output / "gt.json").resolve()),
        "samples": str((output / "samples.json").resolve()),
        "samples_sha256": file_sha256(output / "samples.json"),
        "real_roi_geometry": True,
        "primary_roi_fusion": "MAUR",
        "reversible_parameters": PROTOCOL["reversible_fusion"]["primary_parameters"],
        "old_cloud_predictions_reused": False,
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
    cloud_completion = RUNS / "cloud_cache/completion.json"
    if not cloud_completion.is_file():
        raise RuntimeError("Cloud-V3 prediction cache must complete first")
    roles = {role: build_role(role) for role in PROTOCOL["roles"]}
    report = {
        "complete": True,
        "roles": {role: int(value["images"]) for role, value in roles.items()},
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "final_used_for_training_or_selection": False,
    }
    atomic_json(RUNS / "common/report.json", report)
    atomic_json(
        RUNS / "common/completion.json",
        {"complete": True, "report_sha256": file_sha256(RUNS / "common/report.json")},
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=(*PROTOCOL["roles"], "all"), default="all")
    args = parser.parse_args()
    result = run_all() if args.role == "all" else build_role(args.role)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
