#!/usr/bin/env python3
"""Encode Q90 and build isolated, resumable Edge/Original/Q90 detector caches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


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


DATA = load_module("voc_gluoncv_action_data", ROOT / "scripts/data_core.py")
INFERENCE = load_module("voc_gluoncv_action_inference", ROOT / "scripts/inference_core.py")


def parse_free_gpu_mib(output: str) -> int:
    values = [int(line.strip().split()[0]) for line in output.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no visible GPU")
    return min(values)


def free_gpu_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_free_gpu_mib(result.stdout)


def validate_record_population(
    records: Sequence[Mapping[str, Any]], image_ids: Sequence[int]
) -> None:
    expected = set(map(int, image_ids))
    values = [int(row["image_id"]) for row in records]
    observed = set(values)
    if len(values) != len(observed) or observed != expected:
        raise RuntimeError(
            f"inference record population changed: duplicate={len(values) - len(observed)}, "
            f"missing={len(expected - observed)}, unknown={len(observed - expected)}"
        )


def gt_path(role: str) -> Path:
    if role not in PROTOCOL["roles"]:
        raise ValueError(role)
    return RUNS / f"data/roles/{role}/gt.json"


def encode_q90(role: str) -> dict[str, Any]:
    output = RUNS / f"actions/{role}/q90"
    completion = output / "encode_completion.json"
    manifest_path = output / "q90_manifest.json"
    if completion.is_file():
        report = DATA.read_json(completion)
        if report.get("manifest_sha256") != DATA.file_sha256(manifest_path):
            raise RuntimeError(f"completed Q90 manifest changed: {role}")
        manifest = DATA.read_json(manifest_path)
        records = list(manifest.get("records", ()))
        if len(records) != int(PROTOCOL["roles"][role]):
            raise RuntimeError(f"completed Q90 population changed: {role}")
        upgraded = False
        for row in records:
            path = Path(str(row["path"]))
            if not path.is_file() or path.stat().st_size != int(row["action_bytes"]):
                raise RuntimeError(f"completed Q90 file changed: {role}/{path}")
            observed = DATA.file_sha256(path)
            if "action_sha256" not in row:
                row["action_sha256"] = observed
                upgraded = True
            elif str(row["action_sha256"]) != observed:
                raise RuntimeError(f"completed Q90 content hash changed: {role}/{path}")
        if upgraded:
            manifest["records"] = records
            DATA.atomic_json(manifest_path, manifest)
            report["manifest_sha256"] = DATA.file_sha256(manifest_path)
            report["per_image_content_hashes"] = True
            DATA.atomic_json(completion, report)
        return report
    gt = DATA.read_json(gt_path(role))
    images = sorted(gt["images"], key=lambda row: int(row["id"]))
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    config = PROTOCOL["inference"]

    def encode(image: Mapping[str, Any]) -> dict[str, Any]:
        source = Path(str(image["absolute_path"])).resolve()
        target = image_dir / str(image["file_name"])
        if target.is_file():
            observed = target.stat().st_size
        else:
            observed = INFERENCE.encode_q90(
                source,
                target,
                quality=int(config["jpeg_quality"]),
                optimize=bool(config["jpeg_optimize"]),
                subsampling=int(config["jpeg_subsampling"]),
            )
        return {
            "image_id": int(image["id"]),
            "identity": str(image["identity"]),
            "path": str(target.resolve()),
            "original_path": str(source),
            "original_bytes": source.stat().st_size,
            "action_bytes": int(observed),
            "action_sha256": DATA.file_sha256(target),
            "available": 0 < int(observed) < source.stat().st_size,
        }

    workers = min(24, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(encode, images))
    if len(records) != int(PROTOCOL["roles"][role]):
        raise RuntimeError(f"Q90 population changed: {role}/{len(records)}")
    manifest = {
        "protocol_id": PROTOCOL["protocol_id"],
        "role": role,
        "encoder": {
            "library": "Pillow",
            "quality": int(config["jpeg_quality"]),
            "optimize": bool(config["jpeg_optimize"]),
            "subsampling": int(config["jpeg_subsampling"]),
        },
        "records": records,
    }
    DATA.atomic_json(manifest_path, manifest)
    report = {
        "complete": True,
        "role": role,
        "images": len(records),
        "available": sum(bool(row["available"]) for row in records),
        "original_bytes": sum(int(row["original_bytes"]) for row in records),
        "q90_bytes": sum(int(row["action_bytes"]) for row in records),
        "per_image_content_hashes": True,
        "manifest_sha256": DATA.file_sha256(manifest_path),
    }
    DATA.atomic_json(completion, report)
    return report


def infer(role: str, action: str, *, device: int, enforce_memory_gate: bool) -> dict[str, Any]:
    if action not in {"edge", "original", "q90"}:
        raise ValueError(action)
    output = RUNS / f"actions/{role}/{action}"
    completion = output / "completion.json"
    predictions = output / "predictions.json"
    if completion.is_file():
        report = DATA.read_json(completion)
        if report.get("predictions_sha256") != DATA.file_sha256(predictions):
            raise RuntimeError(f"completed action predictions changed: {role}/{action}")
        records_path = output / "records.jsonl"
        if report.get("records_sha256") != DATA.file_sha256(records_path):
            raise RuntimeError(f"completed action records changed: {role}/{action}")
        return report
    image_map = None
    if action == "q90":
        encode_q90(role)
        image_map = output / "q90_manifest.json"
    if int(device) >= 0 and enforce_memory_gate:
        free = free_gpu_mib()
        minimum = int(PROTOCOL["inference"]["minimum_free_gpu_mib"])
        if free < minimum:
            raise RuntimeError(f"GPU memory gate failed: {free} MiB < {minimum} MiB")
    command = [
        sys.executable,
        str(ROOT / "scripts/gluon_infer.py"),
        "--gt-json",
        str(gt_path(role)),
        "--role",
        "edge" if action == "edge" else "cloud",
        "--action",
        action,
        "--output",
        str(output),
        "--device",
        str(int(device)),
    ]
    if image_map is not None:
        command.extend(("--image-map", str(image_map)))
    subprocess.run(command, check=True, cwd=ROOT)
    gt = DATA.read_json(gt_path(role))
    records = [
        json.loads(line)
        for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_record_population(records, [int(row["id"]) for row in gt["images"]])
    return DATA.read_json(completion)


def run(role: str, action: str, *, device: int, enforce_memory_gate: bool) -> dict[str, Any]:
    roles = list(PROTOCOL["roles"]) if role == "all" else [role]
    actions = ["edge", "original", "q90"] if action == "all" else [action]
    reports: dict[str, Any] = {}
    for current_role in roles:
        for current_action in actions:
            key = f"{current_role}/{current_action}"
            reports[key] = infer(
                current_role,
                current_action,
                device=device,
                enforce_memory_gate=enforce_memory_gate,
            )
    report = {"complete": True, "actions": reports}
    if role == "all" and action == "all":
        DATA.atomic_json(RUNS / "actions/report.json", report)
        DATA.atomic_json(RUNS / "actions/completion.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=(*PROTOCOL["roles"], "all"), default="all")
    parser.add_argument("--action", choices=("edge", "original", "q90", "all"), default="all")
    parser.add_argument("--device", type=int, default=int(PROTOCOL["inference"]["device"]))
    parser.add_argument(
        "--encode-only",
        action="store_true",
        help="Encode and freeze Q90 files without starting detector inference.",
    )
    parser.add_argument(
        "--no-memory-gate",
        action="store_true",
        help="Smoke/debug only; formal runs must keep the frozen GPU memory gate.",
    )
    args = parser.parse_args()
    if args.encode_only:
        roles = list(PROTOCOL["roles"]) if args.role == "all" else [args.role]
        result = {role: encode_q90(role) for role in roles}
        print(json.dumps({"complete": True, "roles": result}, indent=2, ensure_ascii=False))
        return
    print(
        json.dumps(
            run(
                args.role,
                args.action,
                device=int(args.device),
                enforce_memory_gate=not bool(args.no_memory_gate),
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
