#!/usr/bin/env python3
"""Run the locked, audit-gated registered paper pipeline in the background."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
EDGE_GPU_FREE_MIB = 4_000
CLOUD_GPU_FREE_MIB = 7_500
POLL_SECONDS = 60


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_stage(stage: str) -> None:
    print(f"[auto] stage={stage}", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_paper.py"), "--stage", stage],
        check=True,
    )


def inference_complete(role: str, detector: str) -> bool:
    return (RUNS / "inference" / role / detector / "completion.json").is_file()


def gpu_free_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no GPU memory values")
    return max(values)


def run_inference(
    role: str,
    detector: str,
    batch: int,
    minimum_gpu_free_mib: int,
) -> None:
    while not inference_complete(role, detector):
        try:
            free_mib = gpu_free_mib()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            print(f"[auto] GPU query failed: {error}", flush=True)
            time.sleep(POLL_SECONDS)
            continue
        if free_mib < int(minimum_gpu_free_mib):
            print(
                f"[auto] waiting: role={role} detector={detector} "
                f"free={free_mib} MiB required={minimum_gpu_free_mib} MiB",
                flush=True,
            )
            time.sleep(POLL_SECONDS)
            continue
        print(
            f"[auto] inference role={role} detector={detector} "
            f"batch={batch} free={free_mib} MiB",
            flush=True,
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/infer_original.py"),
                "--role",
                role,
                "--detector",
                detector,
                "--device",
                "0",
                "--batch",
                str(int(batch)),
            ],
            check=False,
        )
        if result.returncode and not inference_complete(role, detector):
            print(
                f"[auto] inference exited {result.returncode}; retrying in {POLL_SECONDS}s",
                flush=True,
            )
            time.sleep(POLL_SECONDS)


def main() -> None:
    lock_path = RUNS / "auto_pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another automatic paper pipeline is already running") from error

        run_stage("prepare")
        run_stage("train")
        run_inference("audit", "cloud", 1, CLOUD_GPU_FREE_MIB)
        run_inference("audit", "edge", 16, EDGE_GPU_FREE_MIB)
        run_stage("audit")

        audit_completion = read_json(RUNS / "audit/completion.json")
        if audit_completion.get("passed") is True:
            run_stage("open_final")
            run_inference("registered_final", "cloud", 1, CLOUD_GPU_FREE_MIB)
            run_inference("registered_final", "edge", 16, EDGE_GPU_FREE_MIB)
            run_stage("final")
        else:
            print("[auto] audit gate failed; registered final remains sealed", flush=True)


if __name__ == "__main__":
    main()
