#!/usr/bin/env python3
"""Run Gate 1–2 sequentially only after a conservative GPU-idle gate opens."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNTIME = load_module("voc_torchvision_wait_runtime", ROOT / "scripts/runtime_core.py")


def gpu_state() -> tuple[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if len(output) != 1:
        raise RuntimeError(f"expected one registered GPU, found {len(output)}")
    free_mib, utilization = (int(value.strip()) for value in output[0].split(","))
    return free_mib, utilization


def wait_for_gpu() -> None:
    contract = PROTOCOL["resource_gate"]
    required = int(contract["consecutive_polls"])
    consecutive = 0
    while consecutive < required:
        free_mib, utilization = gpu_state()
        opened = RUNTIME.gpu_gate_open(free_mib, utilization, contract)
        consecutive = consecutive + 1 if opened else 0
        print(
            json.dumps(
                {
                    "status": "gpu_gate_open_sample" if opened else "waiting_for_gpu",
                    "free_mib": free_mib,
                    "utilization_percent": utilization,
                    "consecutive": consecutive,
                    "required": required,
                }
            ),
            flush=True,
        )
        if consecutive < required:
            time.sleep(max(10, int(contract["poll_seconds"])))


def acquire_gpu(*, start_now: bool) -> None:
    if not start_now:
        wait_for_gpu()


def execute(python: str, *arguments: str) -> None:
    subprocess.run([python, *arguments], cwd=ROOT, check=True)


def inference_actions(python: str, role: str, stage: str, *, start_now: bool) -> None:
    for action in ("edge", "original", "q90"):
        completion = ROOT / f"runs/actions/{stage}/{role}/{action}/completion.json"
        if completion.is_file():
            RUNTIME.read_protocol_bound_json(
                completion, RUNTIME.file_sha256(PROTOCOL_PATH)
            )
            print(json.dumps({"status": "cached", "role": role, "stage": stage, "action": action}), flush=True)
            continue
        acquire_gpu(start_now=start_now)
        execute(
            python,
            "scripts/run_inference.py",
            "--role",
            role,
            "--stage",
            stage,
            "--action",
            action,
        )


def read_report(path: Path) -> dict:
    return RUNTIME.read_protocol_bound_json(path, RUNTIME.file_sha256(PROTOCOL_PATH))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="/home/111_wcy/anaconda3/envs/yolo26/bin/python")
    parser.add_argument(
        "--start-now",
        action="store_true",
        help="Bypass the conservative GPU-idle gate for this invocation.",
    )
    args = parser.parse_args()
    python = str(Path(args.python))
    read_report(ROOT / "runs/gate0/completion.json")

    inference_actions(python, "fit", "smoke", start_now=args.start_now)
    execute(python, "scripts/evaluate_gate.py", "--stage", "smoke")
    if read_report(ROOT / "runs/gate1/completion.json").get("proceed_to_full_endpoints") is not True:
        print(json.dumps({"complete": True, "stopped_at": "gate1", "final_accessed": False}), flush=True)
        return

    inference_actions(python, "calibration", "full", start_now=args.start_now)
    execute(python, "scripts/evaluate_gate.py", "--stage", "calibration")
    if read_report(ROOT / "runs/gate2a/completion.json").get("proceed_to_fit") is not True:
        print(json.dumps({"complete": True, "stopped_at": "gate2a", "final_accessed": False}), flush=True)
        return

    inference_actions(python, "fit", "full", start_now=args.start_now)
    execute(python, "scripts/evaluate_gate.py", "--stage", "full")
    result = read_report(ROOT / "runs/gate2/completion.json")
    print(
        json.dumps(
            {
                "complete": True,
                "stopped_at": "gate2",
                "proceed_to_reward": bool(result.get("proceed_to_reward")),
                "final_accessed": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
