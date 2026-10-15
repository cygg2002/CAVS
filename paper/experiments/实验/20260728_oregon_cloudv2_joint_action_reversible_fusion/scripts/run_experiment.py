#!/usr/bin/env python3
"""Run the isolated Oregon Cloud-V2 joint action experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]
sys.path.insert(0, str(SCRIPT.parent))

from fit_cache import (  # noqa: E402
    file_sha256,
    infer_fit_candidate,
    prepare_fit_geometry,
)
from joint_core import candidate_matrix, validate_roles  # noqa: E402


STAGES = ("preflight", "fit-cache", "calibrate", "audit")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_stage(stage: str) -> str:
    value = str(stage)
    if value == "final":
        raise ValueError("final is forbidden in this experiment")
    if value not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {value}")
    return value


def planned_fit_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = [
        {"kind": "geometry", "role": "fit", "profile": profile}
        for profile in config["profiles"]
    ]
    jobs.extend(
        {
            "kind": "inference",
            "role": "fit",
            "profile": candidate["profile"],
            "imgsz": int(candidate["imgsz"]),
            "candidate_id": candidate["id"],
        }
        for candidate in candidate_matrix(config["profiles"], config["imgsz"])
    )
    return jobs


def install_read_only_action_links(config: dict[str, Any]) -> dict[str, str]:
    source = Path(config["source_scan_root"]) / "runs/actions"
    output = Path(config["output_root"]) / "runs/actions"
    output.mkdir(parents=True, exist_ok=True)
    installed = {}
    for role in ("calibration", "audit"):
        target = (source / role).resolve()
        if not target.is_dir():
            raise FileNotFoundError(f"missing frozen {role} action cache: {target}")
        link = output / role
        if link.is_symlink():
            if link.resolve() != target:
                raise RuntimeError(f"action cache link drift: {link}")
        elif link.exists():
            raise RuntimeError(f"expected a read-only action cache symlink: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
        installed[role] = str(target)
    if (output / "final").exists() or (output / "final").is_symlink():
        raise RuntimeError("final action cache must not exist")
    return installed


def source_role_paths(config: dict[str, Any], role: str) -> dict[str, Path]:
    canonical = (
        Path(config["source_cloudv2_root"])
        / "runs/canonical_oregon_event_v1"
    )
    return {
        "gt": canonical / f"gt/oregon/{role}/gt.json",
        "edge": canonical / f"actions/edge/{role}/predictions.json",
        "samples": canonical / f"common/oregon/{role}/samples.json",
        "manifest": canonical / f"common/oregon/{role}/source_manifest.json",
    }


def gpu_status() -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    total, used, utilization = result.stdout.strip().splitlines()[0].split(",")
    return {
        "total_mib": int(total.strip()),
        "used_mib": int(used.strip()),
        "free_mib": int(total.strip()) - int(used.strip()),
        "utilization_percent": int(utilization.strip()),
    }


def wait_for_gpu(config: dict[str, Any]) -> None:
    gate = config["resource_gate"]
    while True:
        status = gpu_status()
        memory_ok = status["free_mib"] >= int(gate["minimum_free_gpu_mib"])
        utilization_ok = status["utilization_percent"] <= int(
            gate["maximum_gpu_utilization_percent"]
        )
        if memory_ok and utilization_ok:
            return
        print(
            "[gpu-gate] "
            f"free={status['free_mib']} MiB "
            f"util={status['utilization_percent']}%",
            flush=True,
        )
        time.sleep(int(gate["poll_seconds"]))


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    roles = validate_roles(config["roles"])
    candidates = candidate_matrix(config["profiles"], config["imgsz"])
    if config.get("registered_final_allowed") is not False:
        raise RuntimeError("registered final must remain forbidden")
    if config.get("final_used_for_selection") is not False:
        raise RuntimeError("final cannot be used for selection")
    output = Path(config["output_root"])
    if output.name != EXP_ROOT.name:
        raise RuntimeError(f"unexpected output experiment: {output}")
    model = Path(config["cloud_model"]["path"])
    if file_sha256(model) != config["cloud_model"]["sha256"]:
        raise RuntimeError("Cloud-V2 model hash mismatch")

    role_report = {}
    for role in roles:
        paths = source_role_paths(config, role)
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing {role} inputs: {missing}")
        samples = read_json(paths["samples"])
        expected = int(config["roles"][role])
        if len(samples) != expected:
            raise RuntimeError(
                f"{role} population changed: {len(samples)} != {expected}"
            )
        role_report[role] = {
            "images": expected,
            "samples_sha256": file_sha256(paths["samples"]),
            "gt_sha256": file_sha256(paths["gt"]),
            "edge_sha256": file_sha256(paths["edge"]),
        }

    scan_actions = Path(config["source_scan_root"]) / "runs/actions"
    frozen_cache_hashes = {}
    for role in ("calibration", "audit"):
        for candidate in candidates:
            root = (
                scan_actions
                / role
                / candidate["profile"]
                / f"imgsz_{candidate['imgsz']}"
            )
            required = [
                root / "COMPLETED",
                root / "summary.json",
                root / "roi_predictions_mapped.json",
            ]
            if not all(path.is_file() for path in required):
                raise RuntimeError(f"incomplete frozen candidate cache: {root}")
            frozen_cache_hashes[f"{role}:{candidate['id']}"] = file_sha256(
                root / "summary.json"
            )

    links = install_read_only_action_links(config)
    report = {
        "protocol_id": config["protocol_id"],
        "roles": role_report,
        "candidate_ids": [value["id"] for value in candidates],
        "cloud_model_sha256": file_sha256(model),
        "frozen_action_links": links,
        "frozen_cache_summary_sha256": frozen_cache_hashes,
        "gpu": gpu_status(),
        "audit_used_for_selection": False,
        "registered_final_test_accessed": False,
    }
    write_json(output / "runs/preflight.json", report)
    return report


def run_fit_cache(config: dict[str, Any]) -> dict[str, Any]:
    preflight(config)
    jobs = planned_fit_jobs(config)
    for index, job in enumerate(jobs, start=1):
        print(
            f"[{time.strftime('%F %T')}] fit-cache {index}/{len(jobs)}: {job}",
            flush=True,
        )
        if job["kind"] == "geometry":
            prepare_fit_geometry(config, job["profile"])
        else:
            wait_for_gpu(config)
            infer_fit_candidate(config, job["profile"], int(job["imgsz"]))
    marker = Path(config["output_root"]) / "runs/FIT_CACHE_COMPLETED"
    marker.write_text("ok\n", encoding="ascii")
    return {
        "stage": "fit-cache",
        "status": "completed",
        "jobs": len(jobs),
        "registered_final_test_accessed": False,
    }


def execute(config: dict[str, Any], stage: str) -> dict[str, Any]:
    validate_stage(stage)
    if stage == "preflight":
        return preflight(config)
    if stage == "fit-cache":
        return run_fit_cache(config)
    if stage == "calibrate":
        from train_evaluate import train_and_calibrate

        return train_and_calibrate(config)
    from train_evaluate import audit_frozen

    return audit_frozen(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=EXP_ROOT / "configs/protocol.json"
    )
    parser.add_argument("--stage", choices=STAGES, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    output = Path(config["output_root"])
    status = output / "runs/pipeline_status.json"
    write_json(
        status,
        {"stage": args.stage, "status": "running", "started_at": time.strftime("%F %T")},
    )
    try:
        result = execute(config, args.stage)
    except Exception as error:
        write_json(
            status,
            {
                "stage": args.stage,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": time.strftime("%F %T"),
            },
        )
        raise
    write_json(
        status,
        {"stage": args.stage, "status": "completed", "completed_at": time.strftime("%F %T")},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
