#!/usr/bin/env python3
"""Build deployment-aligned Edge and Current@640+MAUR action bundles."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from PIL import Image

import backend_runtime


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
SOURCE_CURRENT = Path(PROTOCOL["source_current_experiment"])
SOURCE_TRAINING = Path(PROTOCOL["source_training_experiment"])


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DATA = load_module("rknn_backend_data", ROOT / "scripts/deployment_data.py")
CACHE = load_module(
    "rknn_backend_cloud_cache", SOURCE_TRAINING / "scripts/cache_actions.py"
)
MAUR = load_module(
    "rknn_backend_maur",
    REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/protocol_core.py",
)


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


def atomic_json_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("[")
        first = True
        for row in rows:
            if not first:
                handle.write(",")
            json.dump(dict(row), handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            first = False
        handle.write("]\n")
    os.replace(temporary, path)


def backend_root(backend: str) -> Path:
    return RUNS / "backends" / backend


def imported_candidates(backend: str, role: str) -> Path:
    return RUNS / "imported" / backend / role / "edge_candidates.json"


def imported_report(backend: str, role: str) -> Path:
    return RUNS / "imported" / backend / role / "report.json"


def source_root(role: str) -> Path:
    return SOURCE_CURRENT / f"runs/common/coco/{role}"


def source_manifest(role: str) -> dict[str, Any]:
    manifest = read_json(source_root(role) / "source_manifest.json")
    if manifest.get("cloud_model_sha256") != PROTOCOL["cloud_model_sha256"]:
        raise RuntimeError(f"source cloud model changed: {role}")
    return {
        **manifest,
        "validated_reused_action_sha256": backend_runtime.validate_reused_action_sources(
            manifest, ("original", "q90")
        ),
    }


def geometry_root(backend: str, role: str) -> Path:
    return backend_root(backend) / f"actions/{role}/roi/current/geometry"


def geometry_signature(backend: str, role: str) -> str:
    values = {
        "backend": backend,
        "role": role,
        "candidate_report_sha256": file_sha256(imported_report(backend, role)),
        "codec": PROTOCOL["codec"],
        "roi": PROTOCOL["roi"],
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()


def _encode_crop(
    candidate: Mapping[str, Any], base: Mapping[str, Any], crop_dir: Path, role: str
) -> tuple[str, dict[str, Any]] | None:
    xyxy = candidate.get("roi_xyxy")
    if xyxy is None:
        return None
    image_id = int(candidate["image_id"])
    source = Path(str(base["identity"])).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    x1, y1, x2, y2 = map(int, xyxy)
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
    target = crop_dir / f"{image_id:012d}_{digest}.jpg"
    if not target.is_file():
        with Image.open(source) as opened:
            expected = (int(candidate["width"]), int(candidate["height"]))
            if opened.size != expected:
                raise RuntimeError(f"image geometry changed: {image_id}/{opened.size}/{expected}")
            opened.crop((x1, y1, x2, y2)).convert("RGB").save(
                target,
                format="JPEG",
                quality=int(PROTOCOL["codec"]["quality"]),
                optimize=bool(PROTOCOL["codec"]["optimize"]),
                subsampling=int(PROTOCOL["codec"]["subsampling"]),
            )
    width, height = int(candidate["width"]), int(candidate["height"])
    record = {
        "image_id": image_id,
        "identity": str(source),
        "role": role,
        "file_name": str(base.get("file_name") or source.name),
        "source_path": str(source),
        "path": str(target.resolve()),
        "image_width": width,
        "image_height": height,
        "xyxy": [x1, y1, x2, y2],
        "crop_width": x2 - x1,
        "crop_height": y2 - y1,
        "area_ratio": (x2 - x1) * (y2 - y1) / float(width * height),
        "bytes": target.stat().st_size,
        "profile": "current",
    }
    return str(image_id), record


def prepare_geometry(backend: str, role: str) -> dict[str, Any]:
    output = geometry_root(backend, role)
    manifest_path = output / "roi_manifest.json"
    signature = geometry_signature(backend, role)
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("signature") != signature:
            raise RuntimeError(f"geometry signature changed: {backend}/{role}")
        return manifest
    candidates = read_json(imported_candidates(backend, role))
    bases = {
        int(row["image_id"]): row
        for row in read_json(source_root(role) / "samples.json")
    }
    if set(bases) != {int(row["image_id"]) for row in candidates}:
        raise RuntimeError(f"candidate/source population changed: {backend}/{role}")
    crop_dir = output / "roi_crops" / signature[:16]
    crop_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=24) as executor:
        encoded = executor.map(
            lambda row: _encode_crop(row, bases[int(row["image_id"])], crop_dir, role),
            candidates,
        )
        records = dict(value for value in encoded if value is not None)
    manifest = {
        "protocol": "coco_rknn_current640_geometry_v1",
        "backend": backend,
        "role": role,
        "profile": "current",
        "signature": signature,
        "expected_images": len(candidates),
        "roi_candidates": len(records),
        "records": dict(sorted(records.items(), key=lambda item: int(item[0]))),
        "cloud_model_used": False,
        "roi_source": "raw_rknn_edge_boxes",
    }
    atomic_json(manifest_path, manifest)
    atomic_json(output / "completion.json", {"complete": True, "signature": signature})
    return manifest


def bind_cloud_cache(backend: str) -> None:
    root = backend_root(backend)
    adapted = dict(PROTOCOL)
    adapted["roles"] = dict(PROTOCOL["roles"])
    adapted["profiles"] = ["current"]
    adapted["imgsz"] = [640]
    adapted["geometry"] = {
        "edge_conf": float(PROTOCOL["roi"]["confidence"]),
        "max_area_ratio": float(PROTOCOL["roi"]["max_area_ratio"]),
        "jpeg_quality": int(PROTOCOL["codec"]["quality"]),
        "encode_workers": 24,
    }
    adapted["cloud_model"] = dict(PROTOCOL["cloud_model"])
    adapted["resource_gate"] = {
        "minimum_free_gpu_mib": 0,
        "maximum_gpu_utilization_percent": 100,
        "poll_seconds": 5,
    }
    CACHE.ROOT = ROOT
    CACHE.RUNS = root
    CACHE.PROTOCOL = adapted
    CACHE.EXPECTED_CLOUD_SHA = str(PROTOCOL["cloud_model_sha256"])
    CACHE.prepare_geometry = lambda role, profile: prepare_geometry(backend, str(role))
    CACHE.role_paths = lambda role: {"gt": source_root(str(role)) / "gt.json"}


def run_cloud_cache(backend: str, role: str) -> dict[str, Any]:
    bind_cloud_cache(backend)
    prepare_geometry(backend, role)
    return CACHE.infer_roi(role, "current", 640)


def group_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_json(path):
        output[int(row["image_id"])].append(dict(row))
    return dict(output)


def flat(grouped: Mapping[int, list[dict[str, Any]]]) -> Iterable[dict[str, Any]]:
    for image_id in sorted(grouped):
        yield from grouped[image_id]


def build_common(backend: str, role: str) -> dict[str, Any]:
    output = backend_root(backend) / f"common/coco/{role}"
    completion = output / "completion.json"
    manifest = source_manifest(role)
    if completion.is_file():
        existing = read_json(output / "source_manifest.json")
        for action_id in ("original", "q90"):
            if Path(existing["action_sources"][action_id]).resolve() != Path(
                manifest["action_sources"][action_id]
            ).resolve():
                raise RuntimeError(f"reused action source changed: {backend}/{role}/{action_id}")
        return read_json(completion)
    roi_summary = run_cloud_cache(backend, role)
    gt = read_json(source_root(role) / "gt.json")
    category_ids = [int(row["id"]) for row in gt["categories"]]
    candidates = read_json(imported_candidates(backend, role))
    edge = {
        int(row["image_id"]): DATA.edge_predictions(row, category_ids)
        for row in candidates
    }
    edge_path = output / "actions/edge.json"
    atomic_json_rows(edge_path, flat(edge))
    geometry = read_json(geometry_root(backend, role) / "roi_manifest.json")["records"]
    raw_path = backend_root(backend) / f"actions/{role}/roi/current/imgsz_640/roi_predictions_mapped.json"
    raw = group_predictions(raw_path)
    eligible = set(map(int, geometry))
    maur = {}
    for image_id in sorted(eligible):
        candidates_for_image = MAUR.maur_candidates(
            edge.get(image_id, []),
            raw.get(image_id, []),
            tuple(map(int, geometry[str(image_id)]["xyxy"])),
            duplicate_iou=float(PROTOCOL["maur"]["duplicate_iou"]),
        )
        maur[image_id] = MAUR.class_aware_nms(
            candidates_for_image, iou_threshold=float(PROTOCOL["maur"]["nms_iou"])
        )
    maur_path = output / "actions/current__640__maur.json"
    atomic_json_rows(maur_path, flat(maur))
    base_samples = read_json(source_root(role) / "samples.json")
    samples = []
    for base in base_samples:
        image_id = int(base["image_id"])
        roi = geometry.get(str(image_id))
        sample = DATA.deployment_sample(base, roi)
        for key in ("qualities", "gains", "target", "best_proxy_action", "action_features"):
            sample.pop(key, None)
        sample["rknn_backend"] = backend
        sample["rknn_model_sha256"] = PROTOCOL["rknn_models"][backend]["sha256"]
        samples.append(sample)
    action_sources = {
        "edge": str(edge_path.resolve()),
        "original": str(Path(manifest["action_sources"]["original"]).resolve()),
        "q90": str(Path(manifest["action_sources"]["q90"]).resolve()),
        DATA.ACTION_ID: str(maur_path.resolve()),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "gt.json", gt)
    atomic_json(output / "samples.json", samples)
    source = {
        "dataset": "coco",
        "role": role,
        "images": len(samples),
        "backend": backend,
        "rknn_model_sha256": PROTOCOL["rknn_models"][backend]["sha256"],
        "cloud_model_sha256": PROTOCOL["cloud_model_sha256"],
        "action_sources": action_sources,
        "gt": str((output / "gt.json").resolve()),
        "samples": str((output / "samples.json").resolve()),
        "candidate_report_sha256": file_sha256(imported_report(backend, role)),
        "geometry_signature": geometry_signature(backend, role),
        "roi_prediction_signature": roi_summary["signature"],
        "original_q90_predictions_reused": True,
        "reused_action_sha256": manifest["validated_reused_action_sha256"],
        "reused_source_manifest_sha256": file_sha256(
            source_root(role) / "source_manifest.json"
        ),
        "roi_predictions_reused": False,
        "old_learner_checkpoints_reused": False,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output / "source_manifest.json", source)
    report = {
        "complete": True,
        "backend": backend,
        "role": role,
        "images": len(samples),
        "roi_candidates": len(eligible),
        "edge_predictions": sum(map(len, edge.values())),
        "maur_predictions": sum(map(len, maur.values())),
        "cloud_model_sha256": PROTOCOL["cloud_model_sha256"],
    }
    atomic_json(completion, report)
    return report


def run_backend(backend: str, roles: Iterable[str]) -> dict[str, Any]:
    reports = {role: build_common(backend, role) for role in roles}
    result = {"complete": True, "backend": backend, "roles": reports}
    completed = backend_runtime.completed_common_roles(ROOT, backend)
    if completed is not None:
        atomic_json(
            backend_root(backend) / "common/completion.json",
            {"complete": True, "backend": backend, "roles": completed},
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=PROTOCOL["backends"], required=True)
    parser.add_argument("--role", choices=(*PROTOCOL["roles"], "all"), default="all")
    args = parser.parse_args()
    roles = tuple(PROTOCOL["roles"]) if args.role == "all" else (args.role,)
    print(json.dumps(run_backend(args.backend, roles), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
