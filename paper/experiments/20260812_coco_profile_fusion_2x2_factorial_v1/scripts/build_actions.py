#!/usr/bin/env python3
"""Build MAUR actions from the exact Current/Historical ROI source caches."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
SOURCE_ROOT = REPO / PROTOCOL["source_current_experiment"]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("factorial_action_core", ROOT / "scripts/core.py")
CORE.validate_protocol(PROTOCOL)
MAUR = load_module(
    "factorial_maur_core",
    REPO
    / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/protocol_core.py",
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


def flat(grouped: Mapping[int, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    return [dict(row) for image_id in sorted(grouped) for row in grouped[image_id]]


def source_manifest(role: str) -> tuple[Path, dict[str, Any]]:
    path = SOURCE_ROOT / f"runs/common/coco/{role}/source_manifest.json"
    return path, read_json(path)


def source_paths(role: str, profile: str) -> dict[str, Path]:
    manifest_path, manifest = source_manifest(role)
    root = SOURCE_ROOT / f"runs/actions/{role}/roi/{profile}"
    action = f"{profile}@{int(PROTOCOL['scale'])}"
    return {
        "manifest": manifest_path,
        "edge": Path(manifest["action_sources"]["edge"]),
        "reversible": Path(manifest["action_sources"][action]),
        "geometry": root / "geometry/roi_manifest.json",
        "raw": root / f"imgsz_{int(PROTOCOL['scale'])}/roi_predictions_mapped.json",
    }


def output_path(role: str, profile: str) -> Path:
    return RUNS / f"actions/{role}/{profile}__640_maur.json"


def build_action(role: str, profile: str) -> dict[str, Any]:
    if role not in PROTOCOL["roles"] or profile not in PROTOCOL["profiles"]:
        raise ValueError(f"invalid action request: {role}/{profile}")
    paths = source_paths(role, profile)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    sources = {str(path.resolve()): file_sha256(path) for path in paths.values()}
    output = output_path(role, profile)
    completion = output.with_suffix(".completion.json")
    if output.is_file() and completion.is_file():
        report = read_json(completion)
        if report.get("source_sha256") == sources and report.get(
            "output_sha256"
        ) == file_sha256(output):
            return report
    manifest = read_json(paths["manifest"])
    if manifest.get("cloud_model_sha256") != PROTOCOL["cloud_model_sha256"]:
        raise RuntimeError(f"wrong Cloud cache: {role}/{profile}")
    geometry = manifest_records(paths["geometry"])
    raw = group_predictions(paths["raw"])
    edge = group_predictions(paths["edge"])
    reversible = group_predictions(paths["reversible"])
    sample_path = Path(manifest["samples"])
    samples = read_json(sample_path)
    expected = {
        int(sample["image_id"])
        for sample in samples
        if bool(sample.get("available", {}).get(f"{profile}@640", False))
    }
    if set(geometry) != expected:
        raise RuntimeError(
            f"geometry/availability drift: {role}/{profile}/"
            f"missing={len(expected - set(geometry))}/extra={len(set(geometry) - expected)}"
        )
    if not set(raw) <= expected or not set(reversible) <= expected:
        raise RuntimeError(f"prediction population exceeds eligibility: {role}/{profile}")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for image_id in sorted(expected):
        candidates = MAUR.maur_candidates(
            edge.get(image_id, []),
            raw.get(image_id, []),
            tuple(map(int, geometry[image_id]["xyxy"])),
            duplicate_iou=0.5,
        )
        grouped[image_id] = MAUR.class_aware_nms(candidates, iou_threshold=0.7)
    atomic_json(output, flat(grouped))
    report = {
        "complete": True,
        "role": role,
        "profile": profile,
        "action_id": f"{profile}@640",
        "eligible_images": len(expected),
        "predictions": sum(map(len, grouped.values())),
        "same_geometry_as_reversible": True,
        "same_payload_as_reversible": True,
        "source_sha256": sources,
        "output_sha256": file_sha256(output),
        "cloud_model_sha256": PROTOCOL["cloud_model_sha256"],
        "detector_inference_performed": False,
        "final_used_for_training_gate_or_selection": False,
    }
    atomic_json(completion, report)
    return report


def run_roles(roles: Sequence[str]) -> dict[str, Any]:
    reports = {
        f"{role}|{profile}": build_action(role, profile)
        for role in roles
        for profile in PROTOCOL["profiles"]
    }
    report = {
        "complete": True,
        "roles": list(roles),
        "actions": len(reports),
        "reports": reports,
        "detector_inference_performed": False,
    }
    if set(roles) == set(PROTOCOL["roles"]):
        token = "all"
    elif tuple(roles) == ("final",):
        token = "final"
    else:
        token = "prefinal"
    atomic_json(RUNS / f"actions/{token}_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roles", choices=("prefinal", "final", "all"), default="prefinal"
    )
    args = parser.parse_args()
    roles = {
        "prefinal": ("fit", "calibration"),
        "final": ("final",),
        "all": ("fit", "calibration", "final"),
    }[args.roles]
    print(json.dumps(run_roles(roles), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
