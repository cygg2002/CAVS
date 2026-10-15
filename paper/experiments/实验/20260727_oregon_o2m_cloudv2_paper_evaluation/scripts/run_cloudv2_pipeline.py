#!/usr/bin/env python3
"""Run the isolated, resumable Oregon Cloud-V2 paper-level evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from cloudv2_core import ensure_symlink, gpu_has_capacity, stage_roles, validate_protocol


SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
CONFIG = EXP_ROOT / "configs/protocol.json"
CANONICAL_CONFIG = EXP_ROOT / "configs/canonical_protocol.json"
SOURCE_EXP = REPO / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e"
SOURCE_ROOT = SOURCE_EXP / "runs/canonical_oregon_event_v1"
CANONICAL_ROOT = EXP_ROOT / "runs/canonical_oregon_event_v1"
COMMON = CANONICAL_ROOT / "common"
BENCHMARK = CANONICAL_ROOT / "benchmark"
LOGS = EXP_ROOT / "logs"
PYTHON = Path("/home/111_wcy/anaconda3/envs/yolo26/bin/python")
YOLO_RUNTIME = Path("/home/111_wcy/work/yolo26/yoloe26")
FAIR = REPO / "paper/experiments/20260710_ena24_oracle_fairness/scripts"
SOURCE_SCRIPTS = SOURCE_EXP / "scripts"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment(config: dict[str, Any]) -> dict[str, str]:
    threads = str(config["resource_gate"]["cpu_threads"])
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
            "YOLO_AUTOINSTALL": "False",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(
                value
                for value in (str(YOLO_RUNTIME), str(REPO), env.get("PYTHONPATH", ""))
                if value
            ),
        }
    )
    return env


def run_command(
    name: str,
    command: list[str],
    *,
    outputs: tuple[Path, ...] = (),
    gpu: bool = False,
) -> None:
    if outputs and all(path.is_file() for path in outputs):
        return
    config = read_json(CONFIG)
    if gpu:
        wait_for_gpu(config)
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{name}.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%F %T')}] START {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=REPO,
            env=environment(config),
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: os.nice(int(config["resource_gate"]["nice"])),
            check=False,
        )
        log.write(f"[{time.strftime('%F %T')}] EXIT {result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"{name} failed; see {log_path}")
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"{name} omitted outputs: {missing}")


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
        if gpu_has_capacity(
            total_mib=total,
            used_mib=used,
            minimum_free_mib=int(gate["minimum_free_gpu_mib"]),
        ):
            return
        time.sleep(int(gate["poll_seconds"]))


def blocking_cpu_pids(config: dict[str, Any]) -> list[int]:
    needle = str(config["resource_gate"]["blocked_process_substring"])
    result = subprocess.run(
        ["pgrep", "-u", str(os.getuid()), "-f", needle],
        text=True,
        capture_output=True,
        check=False,
    )
    return [
        int(value)
        for value in result.stdout.split()
        if int(value) not in {os.getpid(), os.getppid()}
    ]


def wait_for_cpu(config: dict[str, Any]) -> None:
    while blocking_cpu_pids(config):
        time.sleep(int(config["resource_gate"]["poll_seconds"]))


def old_role(role: str) -> Path:
    return SOURCE_ROOT / f"common/oregon/{role}"


def prepare_role(config: dict[str, Any], role: str) -> None:
    old_actions = SOURCE_ROOT / "actions"
    ensure_symlink(SOURCE_ROOT / f"gt/oregon/{role}/gt.json", CANONICAL_ROOT / f"gt/oregon/{role}/gt.json")
    ensure_symlink(old_actions / f"edge/{role}/predictions.json", CANONICAL_ROOT / f"actions/edge/{role}/predictions.json")
    edge_manifest = old_actions / f"edge/{role}/predictions.manifest.json"
    if edge_manifest.is_file():
        ensure_symlink(edge_manifest, CANONICAL_ROOT / f"actions/edge/{role}/predictions.manifest.json")
    ensure_symlink(old_actions / f"fpn/{role}/global_fpn_features.json", CANONICAL_ROOT / f"actions/fpn/{role}/global_fpn_features.json")
    ensure_symlink(old_actions / f"q90/{role}", CANONICAL_ROOT / f"actions/q90/{role}")
    runtime = SOURCE_ROOT / f"runtime_data/{role}.yaml"
    if runtime.is_file():
        ensure_symlink(runtime, CANONICAL_ROOT / f"runtime_data/{role}.yaml")

    audit_root = Path(config["audit_reuse"]["root"])
    if role == "audit":
        ensure_symlink(
            audit_root / "new_o2m_finetune_predictions.json",
            CANONICAL_ROOT / "actions/original/audit/predictions.json",
        )
        ensure_symlink(
            audit_root / "new_o2m_finetune_predictions.manifest.json",
            CANONICAL_ROOT / "actions/original/audit/predictions.manifest.json",
        )
        ensure_symlink(
            audit_root / "payload_actions/new_q90",
            CANONICAL_ROOT / "actions/q90_predictions/audit",
        )
        ensure_symlink(
            audit_root / "payload_actions/new_roi_historical_cost",
            CANONICAL_ROOT / "actions/roi/audit",
        )
    else:
        roi = CANONICAL_ROOT / f"actions/roi/{role}"
        roi.mkdir(parents=True, exist_ok=True)
        ensure_symlink(old_actions / f"roi/{role}/roi_crops", roi / "roi_crops")


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    checked = validate_protocol(config)
    model = Path(config["new_cloud_model"]["path"])
    if sha256(model) != config["new_cloud_model"]["sha256"]:
        raise RuntimeError("new cloud model hash mismatch")
    if not PYTHON.is_file() or not YOLO_RUNTIME.is_dir():
        raise FileNotFoundError("frozen runtime is unavailable")
    role_report = {}
    for role, expected in checked["roles"].items():
        manifest = read_json(old_role(role) / "source_manifest.json")
        if int(manifest["images"]) != int(expected):
            raise RuntimeError(f"{role} source population changed")
        role_report[role] = {
            "images": int(expected),
            "source_manifest_sha256": sha256(old_role(role) / "source_manifest.json"),
        }
    report = {
        "protocol_sha256": sha256(CONFIG),
        "canonical_protocol_sha256": sha256(CANONICAL_CONFIG),
        "cloud_model_sha256": sha256(model),
        "roles": role_report,
        "active_blocking_cpu_pids": blocking_cpu_pids(config),
        "gpu_memory_mib": dict(zip(("total", "used"), gpu_memory())),
        "source_experiments_modified": False,
        "final_used_for_selection": False,
    }
    write_json(EXP_ROOT / "runs/preflight.json", report)
    return report


def original_inference(config: dict[str, Any], role: str) -> None:
    output = CANONICAL_ROOT / f"actions/original/{role}/predictions.json"
    model = config["new_cloud_model"]["path"]
    inference = config["inference"]
    run_command(
        f"original_{role}",
        [
            str(PYTHON),
            str(FAIR / "build_validator_prediction_cache.py"),
            "--data",
            str(CANONICAL_ROOT / f"runtime_data/{role}.yaml"),
            "--split",
            "val",
            "--gt-json",
            str(CANONICAL_ROOT / f"gt/oregon/{role}/gt.json"),
            "--imgsz",
            str(inference["imgsz"]),
            "--conf",
            str(inference["conf"]),
            "--iou",
            str(inference["iou"]),
            "--max-det",
            str(inference["max_det"]),
            "--device",
            str(inference["device"]),
            "--model",
            str(model),
            "--out-json",
            str(output),
            "--batch",
            str(inference["batch"]),
        ],
        outputs=(output, output.with_suffix(".manifest.json")),
        gpu=True,
    )


def q90_inference(config: dict[str, Any], role: str) -> None:
    output = CANONICAL_ROOT / f"actions/q90_predictions/{role}"
    inference = config["inference"]
    run_command(
        f"q90_{role}",
        [
            str(PYTHON),
            str(SOURCE_SCRIPTS / "infer_q90_cache.py"),
            "--gt-json",
            str(CANONICAL_ROOT / f"gt/oregon/{role}/gt.json"),
            "--manifest",
            str(CANONICAL_ROOT / f"actions/q90/{role}/q90_manifest.json"),
            "--cloud-model",
            str(config["new_cloud_model"]["path"]),
            "--expected-cloud-sha256",
            str(config["new_cloud_model"]["sha256"]),
            "--out-dir",
            str(output),
            "--batch",
            str(inference["batch"]),
            "--chunk-size",
            "256",
            "--device",
            str(inference["device"]),
        ],
        outputs=(output / "COMPLETED",),
        gpu=True,
    )


def roi_inference(config: dict[str, Any], role: str) -> None:
    output = CANONICAL_ROOT / f"actions/roi/{role}"
    inference = config["inference"]
    run_command(
        f"roi_{role}",
        [
            str(PYTHON),
            str(SOURCE_SCRIPTS / "build_roi_q90_cache.py"),
            "--gt-json",
            str(CANONICAL_ROOT / f"gt/oregon/{role}/gt.json"),
            "--edge-json",
            str(CANONICAL_ROOT / f"actions/edge/{role}/predictions.json"),
            "--cloud-model",
            str(config["new_cloud_model"]["path"]),
            "--expected-cloud-sha256",
            str(config["new_cloud_model"]["sha256"]),
            "--out-dir",
            str(output),
            "--profile",
            "historical_cost",
            "--quality",
            "90",
            "--device",
            str(inference["device"]),
            "--batch",
            str(inference["batch"]),
            "--chunk-size",
            "256",
            "--encode-workers",
            str(inference["encode_workers"]),
        ],
        outputs=(output / "COMPLETED",),
        gpu=True,
    )


def infer_roles(config: dict[str, Any], stage: str) -> None:
    for role in stage_roles(config, stage):
        prepare_role(config, role)
        if role != "audit":
            original_inference(config, role)
            q90_inference(config, role)
            roi_inference(config, role)


def import_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def source_spec(role: str) -> dict[str, Any]:
    return {
        "gt": CANONICAL_ROOT / f"gt/oregon/{role}/gt.json",
        "edge": CANONICAL_ROOT / f"actions/edge/{role}/predictions.json",
        "original": CANONICAL_ROOT / f"actions/original/{role}/predictions.json",
        "q90": CANONICAL_ROOT / f"actions/q90_predictions/{role}/q90_predictions.json",
        "q90_manifest": CANONICAL_ROOT / f"actions/q90/{role}/q90_manifest.json",
        "fpn": CANONICAL_ROOT / f"actions/fpn/{role}/global_fpn_features.json",
        "roi_dir": CANONICAL_ROOT / f"actions/roi/{role}",
        "ids": None,
    }


def build_common(config: dict[str, Any], stage: str) -> None:
    sys.path.insert(0, str(SOURCE_SCRIPTS))
    import build_common_cache as common

    common.OUTPUT = COMMON
    pca_path = COMMON / "oregon/pca32.json"
    pca = read_json(pca_path) if pca_path.is_file() else None
    for role in stage_roles(config, stage):
        target = COMMON / f"oregon/{role}"
        if all((target / name).is_file() for name in ("gt.json", "samples.json", "source_manifest.json")):
            continue
        _, _, state = common.build_role("oregon", role, source_spec(role), pca)
        pca = state["pca"]
        if role == "fit":
            write_json(pca_path, pca)
    summary = {
        role: len(read_json(COMMON / f"oregon/{role}/samples.json"))
        for role in ("fit", "calibration", "audit", "final")
        if (COMMON / f"oregon/{role}/samples.json").is_file()
    }
    write_json(COMMON / "summary.json", {"oregon": summary})
    (COMMON / f"{stage.upper()}_COMPLETED").write_text("ok\n")


def canonical_benchmark(config: dict[str, Any], stage: str) -> None:
    sys.path.insert(0, str(SOURCE_SCRIPTS))
    import run_canonical_multiranker as canonical
    import run_multiranker_benchmark as benchmark

    canonical.COMMON = COMMON
    canonical.OUTPUT = BENCHMARK
    canonical.CANONICAL_CONFIG = CANONICAL_CONFIG
    canonical.ALGORITHM_CONFIG = SOURCE_EXP / "configs/protocol.json"
    benchmark.COMMON = COMMON
    benchmark.OUTPUT = BENCHMARK
    benchmark.calibrate_actions = canonical.calibration_only_gate
    benchmark.dataset_cost = canonical.canonical_dataset_cost
    report = benchmark.run_dataset("oregon", canonical.merged_config())
    write_json(BENCHMARK / "cloudv2_summary.json", report)
    frozen_paths = (BENCHMARK / "oregon/models.json", BENCHMARK / "oregon/action_gate.json")
    current = {path.name: sha256(path) for path in frozen_paths}
    freeze_path = CANONICAL_ROOT / "DEV_FROZEN.json"
    if stage == "dev":
        write_json(
            freeze_path,
            {
                "frozen_at": time.strftime("%F %T"),
                "hashes": current,
                "final_artifacts_read": False,
            },
        )
    else:
        expected = read_json(freeze_path)["hashes"]
        if current != expected:
            raise RuntimeError("canonical models/action gate changed after final")
    (BENCHMARK / f"{stage.upper()}_COMPLETED").write_text("ok\n")


def prepare_adapter_layout() -> tuple[Path, Path]:
    canonical_source = EXP_ROOT / "canonical_source"
    nineranker = EXP_ROOT / "nineranker"
    ensure_symlink(SOURCE_EXP / "configs/protocol.json", canonical_source / "configs/protocol.json")
    ensure_symlink(SOURCE_SCRIPTS, canonical_source / "scripts")
    ensure_symlink(CANONICAL_ROOT, canonical_source / "runs/canonical_oregon_event_v1")
    source_nine = REPO / "paper/experiments/20260723_three_dataset_nineranker_triaction_paper_exploration"
    ensure_symlink(source_nine / "configs/protocol.json", nineranker / "configs/protocol.json")
    ensure_symlink(source_nine / "scripts", nineranker / "scripts")
    return canonical_source, nineranker


def run_nineranker(config: dict[str, Any], stage: str) -> None:
    wait_for_cpu(config)
    canonical_source, output = prepare_adapter_layout()
    scripts = REPO / "paper/experiments/20260723_three_dataset_nineranker_triaction_paper_exploration/scripts"
    sys.path.insert(0, str(scripts))
    module = import_from(scripts / "run_dataset.py", "cloudv2_nineranker_runner")
    module.EXP_ROOT = output
    module.SOURCE_ROOT = canonical_source
    protocol = read_json(output / "configs/protocol.json")
    module.configure_reused_runner("oregon", output / "runs/oregon")
    checked = module.preflight("oregon", protocol)
    if stage == "core":
        module.run_core("oregon", protocol, checked)
    else:
        module.run_bootstrap_stage("oregon", protocol, checked)


def run_selector(config: dict[str, Any], stage: str) -> None:
    wait_for_cpu(config)
    canonical_source, nineranker = prepare_adapter_layout()
    output = EXP_ROOT / "selector"
    source = REPO / "paper/experiments/20260725_three_dataset_costaware_selector_riskgate"
    ensure_symlink(source / "configs/protocol.json", output / "configs/protocol.json")
    sys.path.insert(0, str(source / "scripts"))
    if stage == "calibrate":
        module = import_from(source / "scripts/run_selector.py", "cloudv2_selector")
        module.EXP_ROOT = output
        module.source_root = lambda: nineranker
        module.SOURCE_NAME = "local_cloudv2_nineranker"
        module.run_calibration("oregon", read_json(output / "configs/protocol.json"))
    elif stage == "gate":
        module = import_from(source / "scripts/run_audit_gate.py", "cloudv2_audit_gate")
        module.EXP_ROOT = output
        module.SOURCE_20260720 = canonical_source
        module.SOURCE_20260723 = nineranker
        module.run("oregon")
    else:
        module = import_from(source / "scripts/finalize_source_dataset.py", "cloudv2_finalizer")
        module.EXP_ROOT = output
        module.SOURCE_ROOT = nineranker
        module.finalize("oregon")


def execute(stage: str) -> None:
    config = read_json(CONFIG)
    preflight(config)
    if stage == "preflight":
        return
    if stage in {"dev", "all"}:
        wait_for_cpu(config)
        infer_roles(config, "dev")
        wait_for_cpu(config)
        build_common(config, "dev")
        canonical_benchmark(config, "dev")
    if stage in {"final", "all"}:
        if not (CANONICAL_ROOT / "DEV_FROZEN.json").is_file():
            raise RuntimeError("final requires a frozen dev configuration")
        wait_for_cpu(config)
        infer_roles(config, "final")
        wait_for_cpu(config)
        build_common(config, "final")
        canonical_benchmark(config, "final")
    if stage in {"nineranker", "all"}:
        run_nineranker(config, "core")
        run_nineranker(config, "bootstrap")
    if stage in {"selector", "all"}:
        run_selector(config, "calibrate")
        run_selector(config, "gate")
        run_selector(config, "finalize")
    (EXP_ROOT / f"{stage.upper()}_COMPLETED").write_text("ok\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "dev", "final", "nineranker", "selector", "all"),
        default="all",
    )
    args = parser.parse_args()
    status_path = EXP_ROOT / "runs/pipeline_status.json"
    try:
        write_json(status_path, {"stage": args.stage, "status": "running", "started_at": time.strftime("%F %T")})
        execute(args.stage)
    except Exception as error:
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
    write_json(
        status_path,
        {"stage": args.stage, "status": "complete", "completed_at": time.strftime("%F %T")},
    )


if __name__ == "__main__":
    main()
