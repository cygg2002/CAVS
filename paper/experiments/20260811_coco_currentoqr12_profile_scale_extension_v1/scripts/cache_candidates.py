#!/usr/bin/env python3
"""Build isolated, resource-gated COCO Profile x Scale ROI caches."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


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


CORE = load_module("coco_oqr12_cache_core", ROOT / "scripts/core.py")
SOURCE_ROOT = Path(PROTOCOL["source_training_experiment"])
SOURCE = load_module(
    "coco_oqr12_cache_source", SOURCE_ROOT / "scripts/cache_actions.py"
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


def canonical_manifest(role: str) -> dict[str, Any]:
    path = (
        Path(PROTOCOL["canonical_root"])
        / f"common/coco/{role}/source_manifest.json"
    )
    return read_json(path)


def role_paths(role: str) -> dict[str, Path]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(role)
    manifest_path = (
        Path(PROTOCOL["canonical_root"])
        / f"common/coco/{role}/source_manifest.json"
    )
    manifest = read_json(manifest_path)
    return {
        "gt": Path(manifest["gt"]),
        "edge": Path(manifest["action_sources"]["edge"]),
        "samples": Path(manifest["samples"]),
        "source_manifest": manifest_path,
    }


def bind_source() -> None:
    """Bind the proven Oregon cache implementation to this isolated COCO run."""
    SOURCE.ROOT = ROOT
    SOURCE.RUNS = RUNS
    SOURCE.PROTOCOL = PROTOCOL
    SOURCE.EXPECTED_CLOUD_SHA = str(PROTOCOL["cloud_model"]["sha256"])
    SOURCE.role_paths = role_paths


bind_source()


def run_preflight() -> dict[str, Any]:
    output = RUNS / "preflight/cache_report.json"
    completion = RUNS / "preflight/cache_completion.json"
    if completion.is_file():
        return read_json(output)
    CORE.validate_protocol(PROTOCOL)
    model = Path(PROTOCOL["cloud_model"]["path"])
    if file_sha256(model) != str(PROTOCOL["cloud_model"]["sha256"]):
        raise RuntimeError("COCO cloud model hash changed")
    role_ids: dict[str, list[int]] = {}
    identities: dict[str, set[str]] = {}
    sources = [ROOT / "configs/protocol.json", ROOT / "scripts/core.py", SCRIPT, model]
    for role, expected in PROTOCOL["roles"].items():
        paths = role_paths(role)
        sources.extend(paths.values())
        samples = read_json(paths["samples"])
        ids = [int(sample["image_id"]) for sample in samples]
        if len(ids) != int(expected) or len(ids) != len(set(ids)):
            raise RuntimeError(f"canonical COCO population changed: {role}")
        role_ids[role] = ids
        identities[role] = {str(sample["identity"]) for sample in samples}
    roles = list(PROTOCOL["roles"])
    overlaps = {
        f"{left}_{right}": len(identities[left] & identities[right])
        for index, left in enumerate(roles)
        for right in roles[index + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError(f"COCO role identity leakage: {overlaps}")
    folds = read_json(Path(PROTOCOL["frozen_folds"]))["folds"]
    flat = [int(image_id) for fold in folds for image_id in fold]
    if len(folds) != int(PROTOCOL["fold_count"]) or set(flat) != set(role_ids["fit"]):
        raise RuntimeError("frozen COCO folds changed")
    sources.append(Path(PROTOCOL["frozen_folds"]))
    missing = [str(path) for path in sources if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing[:5])
    hashes = {str(Path(path).resolve()): file_sha256(Path(path)) for path in sources}
    report = {
        "complete": True,
        "role_counts": {role: len(ids) for role, ids in role_ids.items()},
        "identity_overlap": overlaps,
        "cache_jobs": len(CORE.cache_jobs(PROTOCOL)),
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "source_digest": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "source_hashes": hashes,
        "source_opened_read_only": True,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(output, report)
    atomic_json(completion, {"complete": True, "source_digest": report["source_digest"]})
    return report


def run_geometry() -> dict[str, Any]:
    completion = RUNS / "geometry/completion.json"
    report_path = RUNS / "geometry/report.json"
    if completion.is_file():
        return read_json(report_path)
    run_preflight()
    reports = []
    for role in PROTOCOL["roles"]:
        for profile in PROTOCOL["profiles"]:
            print(f"[geometry] {role}/{profile}", flush=True)
            manifest = SOURCE.prepare_geometry(str(role), str(profile))
            reports.append(
                {
                    "role": role,
                    "profile": profile,
                    "roi_candidates": int(manifest["roi_candidates"]),
                    "signature": str(manifest["signature"]),
                }
            )
    report = {"complete": True, "jobs": len(reports), "reports": reports}
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "jobs": len(reports)})
    return report


def gpu_status() -> dict[str, int]:
    return SOURCE.gpu_status()


def wait_for_gpu() -> dict[str, int]:
    state = CORE.GpuGateState(PROTOCOL["resource_gate"])
    while True:
        status = gpu_status()
        eligible = state.observe(
            free_mib=int(status["free_mib"]),
            utilization=int(status["utilization_percent"]),
        )
        print(
            "[gpu-gate] "
            f"free={status['free_mib']}MiB util={status['utilization_percent']}% "
            f"stable={state.consecutive}/{PROTOCOL['resource_gate']['stable_polls']}",
            flush=True,
        )
        if eligible:
            return status
        time.sleep(int(PROTOCOL["resource_gate"]["poll_seconds"]))


def run_inference() -> dict[str, Any]:
    completion = RUNS / "cloud_cache/completion.json"
    report_path = RUNS / "cloud_cache/report.json"
    if completion.is_file():
        return read_json(report_path)
    run_geometry()
    reports = []
    for index, job in enumerate(CORE.cache_jobs(PROTOCOL), 1):
        output = SOURCE.roi_output(job["role"], job["profile"], job["imgsz"])
        if not (output / "completion.json").is_file():
            wait_for_gpu()
        print(f"[roi {index}/36] {job['id']}", flush=True)
        result = SOURCE.infer_roi(job["role"], job["profile"], job["imgsz"])
        if result.get("cloud_model_sha256") != PROTOCOL["cloud_model"]["sha256"]:
            raise RuntimeError(f"wrong Cloud cache: {job['id']}")
        reports.append(
            {
                "id": job["id"],
                "roi_candidates": int(result["roi_candidates"]),
                "predictions": int(result["predictions"]),
                "signature": str(result["signature"]),
            }
        )
    if len(reports) != len(CORE.cache_jobs(PROTOCOL)):
        raise RuntimeError("COCO ROI cache matrix incomplete")
    report = {
        "complete": True,
        "jobs": len(reports),
        "reports": reports,
        "cloud_model_sha256": PROTOCOL["cloud_model"]["sha256"],
        "final_used_for_training_or_selection": False,
    }
    atomic_json(report_path, report)
    atomic_json(completion, {"complete": True, "jobs": len(reports)})
    return report


def run_all() -> dict[str, Any]:
    return {
        "preflight": run_preflight(),
        "geometry": run_geometry(),
        "inference": run_inference(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("preflight", "geometry", "inference", "all"), default="all"
    )
    args = parser.parse_args()
    result = {
        "preflight": run_preflight,
        "geometry": run_geometry,
        "inference": run_inference,
        "all": run_all,
    }[args.stage]()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
