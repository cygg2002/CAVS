#!/usr/bin/env python3
"""Run the isolated, resumable Oregon Cloud-V2 profile/scale scan."""

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

from build_cache import file_sha256, infer_candidate, prepare_geometry  # noqa: E402
from evaluate_scan import (  # noqa: E402
    evaluate_role,
    finalize_audit,
    freeze_calibration,
    read_json,
)
from scan_core import candidate_matrix, validate_roles  # noqa: E402


STAGES = ("preflight", "calibration", "audit")


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


def planned_jobs(
    config: dict[str, Any], role: str
) -> list[dict[str, Any]]:
    validate_roles(config["roles"])
    if role not in config["roles"]:
        raise ValueError(role)
    jobs = [
        {"kind": "geometry", "role": role, "profile": profile}
        for profile in config["profiles"]
    ]
    jobs.extend(
        {
            "kind": "inference",
            "role": role,
            "profile": candidate["profile"],
            "imgsz": candidate["imgsz"],
            "candidate_id": candidate["id"],
        }
        for candidate in candidate_matrix(config["profiles"], config["imgsz"])
    )
    jobs.append({"kind": "evaluate", "role": role})
    return jobs


def gpu_memory() -> tuple[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    total, used = result.stdout.strip().splitlines()[0].split(",")
    return int(total.strip()), int(used.strip())


def wait_for_gpu(config: dict[str, Any]) -> None:
    gate = config["resource_gate"]
    while True:
        total, used = gpu_memory()
        free = total - used
        if free >= int(gate["minimum_free_gpu_mib"]):
            return
        print(
            f"[gpu-gate] free={free} MiB, "
            f"required={gate['minimum_free_gpu_mib']} MiB",
            flush=True,
        )
        time.sleep(int(gate["poll_seconds"]))


def source_paths(
    config: dict[str, Any], role: str
) -> dict[str, Path]:
    source = (
        Path(config["source_cloudv2_root"])
        / "runs/canonical_oregon_event_v1"
    )
    return {
        "gt": source / f"gt/oregon/{role}/gt.json",
        "edge": source / f"actions/edge/{role}/predictions.json",
        "common_samples": source / f"common/oregon/{role}/samples.json",
        "common_manifest": source / f"common/oregon/{role}/source_manifest.json",
        "models": source / "benchmark/oregon/models.json",
        "action_gate": source / "benchmark/oregon/action_gate.json",
    }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    roles = validate_roles(config["roles"])
    candidates = candidate_matrix(config["profiles"], config["imgsz"])
    if config.get("registered_final_allowed") is not False:
        raise RuntimeError("registered final must remain forbidden")
    if config.get("final_used_for_selection") is not False:
        raise RuntimeError("final cannot be used for selection")
    output = Path(config["output_root"])
    source = Path(config["source_cloudv2_root"])
    if output == source or source in output.parents:
        raise RuntimeError("output must be isolated from the frozen source")
    if output.name != EXP_ROOT.name:
        raise RuntimeError(f"unexpected output experiment: {output}")
    model = Path(config["cloud_model"]["path"])
    if file_sha256(model) != config["cloud_model"]["sha256"]:
        raise RuntimeError("Cloud-V2 model hash mismatch")
    role_report = {}
    for role in roles:
        paths = source_paths(config, role)
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing {role} inputs: {missing}")
        image_count = len(read_json(paths["gt"])["images"])
        sample_count = len(read_json(paths["common_samples"]))
        expected = int(config["roles"][role])
        if image_count != expected or sample_count != expected:
            raise RuntimeError(
                f"{role} population changed: "
                f"gt={image_count}, samples={sample_count}, expected={expected}"
            )
        role_report[role] = {
            "images": expected,
            "gt_sha256": file_sha256(paths["gt"]),
            "edge_sha256": file_sha256(paths["edge"]),
            "common_manifest_sha256": file_sha256(paths["common_manifest"]),
        }
    total, used = gpu_memory()
    report = {
        "protocol_id": config["protocol_id"],
        "roles": role_report,
        "candidate_ids": [value["id"] for value in candidates],
        "geometry_jobs_per_role": len(config["profiles"]),
        "inference_jobs_per_role": len(candidates),
        "cloud_model_sha256": file_sha256(model),
        "gpu_memory_mib": {"total": total, "used": used, "free": total - used},
        "source_experiment_modified": False,
        "audit_used_for_selection": False,
        "registered_final_test_accessed": False,
    }
    write_json(output / "runs/preflight.json", report)
    return report


def execute_job(config: dict[str, Any], job: dict[str, Any]) -> Any:
    kind = job["kind"]
    if kind == "geometry":
        return prepare_geometry(config, job["role"], job["profile"])
    if kind == "inference":
        wait_for_gpu(config)
        return infer_candidate(
            config, job["role"], job["profile"], int(job["imgsz"])
        )
    if kind == "evaluate":
        report = evaluate_role(config, job["role"])
        if job["role"] == "calibration":
            return freeze_calibration(config, report)
        freeze_path = (
            Path(config["output_root"]) / "runs/CALIBRATION_FROZEN.json"
        )
        if not freeze_path.is_file():
            raise RuntimeError("audit requires CALIBRATION_FROZEN.json")
        return finalize_audit(config, report, read_json(freeze_path))
    raise ValueError(kind)


def execute_stage(
    config: dict[str, Any], stage: str, *, dry_run: bool
) -> dict[str, Any]:
    validate_stage(stage)
    if dry_run:
        roles = () if stage == "preflight" else (stage,)
        return {
            "stage": stage,
            "dry_run": True,
            "jobs": [
                job
                for role in roles
                for job in planned_jobs(config, role)
            ],
            "registered_final_test_accessed": False,
        }
    report = preflight(config)
    if stage == "preflight":
        return report
    if stage == "audit" and not (
        Path(config["output_root"]) / "runs/CALIBRATION_FROZEN.json"
    ).is_file():
        raise RuntimeError("audit requires completed calibration")
    jobs = planned_jobs(config, stage)
    for index, job in enumerate(jobs, start=1):
        print(
            f"[{time.strftime('%F %T')}] "
            f"{stage} job {index}/{len(jobs)}: {job}",
            flush=True,
        )
        execute_job(config, job)
    marker = Path(config["output_root"]) / f"{stage.upper()}_COMPLETED"
    marker.write_text("ok\n", encoding="ascii")
    return {
        "stage": stage,
        "status": "completed",
        "jobs": len(jobs),
        "registered_final_test_accessed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXP_ROOT / "configs/protocol.json",
    )
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    output = Path(config["output_root"])
    status_path = output / "runs/pipeline_status.json"
    if not args.dry_run:
        write_json(
            status_path,
            {
                "stage": args.stage,
                "status": "running",
                "started_at": time.strftime("%F %T"),
            },
        )
    try:
        result = execute_stage(config, args.stage, dry_run=args.dry_run)
    except Exception as error:
        if not args.dry_run:
            write_json(
                status_path,
                {
                    "stage": args.stage,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_at": time.strftime("%F %T"),
                },
            )
        raise
    if not args.dry_run:
        write_json(
            status_path,
            {
                "stage": args.stage,
                "status": "completed",
                "completed_at": time.strftime("%F %T"),
            },
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
