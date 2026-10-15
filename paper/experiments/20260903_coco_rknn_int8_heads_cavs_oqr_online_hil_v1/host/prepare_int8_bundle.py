#!/usr/bin/env python3
"""Build an isolated INT8-heads CAVS-OQR causal-online runtime bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


PROTOCOL_ID = "coco_rknn_int8_heads_causal_online_hil_v1"
RKNN_SHA256 = "9da11e627d5086c99b089a59cfe1c8089046704aeebd7bf27fc675410d9a2ad5"
CLOUD_SHA256 = "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare_bundle(*, source_bundle: Path, policy_source: Path, source_host: Path, output: Path) -> dict[str, str]:
    """Copy the proven causal runtime and bind it only to INT8-heads assets."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be a new empty directory: {output}")
    if not source_bundle.is_dir():
        raise FileNotFoundError(source_bundle)
    if not (policy_source / "manifest.json").is_file():
        raise FileNotFoundError(policy_source / "manifest.json")

    policy_manifest = _read_json(policy_source / "manifest.json")
    if policy_manifest.get("policy_id") != "cavs_current_maur_rknn_int8_heads_v1":
        raise ValueError("policy is not the separately trained INT8-heads CAVS-OQR policy")
    if policy_manifest.get("rknn_model_sha256") != RKNN_SHA256:
        raise ValueError("policy INT8 RKNN hash differs from the frozen experiment hash")
    if policy_manifest.get("cloud_model_sha256") != CLOUD_SHA256:
        raise ValueError("policy cloud model hash differs from the frozen cloud model hash")
    if policy_manifest.get("action_ids") != ["edge", "original", "q90", "current@640"]:
        raise ValueError("policy action space is not CAVS-OQR")

    output.mkdir(parents=True, exist_ok=False)
    bundle = output / "board_bundle"
    shutil.copytree(source_bundle, bundle, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "runs"))
    template_formal = Path(__file__).resolve().parents[1] / "board_formal"
    shutil.copytree(template_formal, output / "board_formal")
    manifest = bundle / "BUNDLE_MANIFEST.json"
    if manifest.exists():
        manifest.unlink()

    policies = bundle / "policies"
    shutil.rmtree(policies)
    shutil.copytree(policy_source, policies / "cavs_oqr")
    controls = bundle / "controls"
    if controls.exists():
        shutil.rmtree(controls)
    (controls / "cavs_oqr_static").mkdir(parents=True)
    (controls / "cavs_oqr").mkdir(parents=True)

    protocol_path = bundle / "configs" / "protocol.json"
    protocol = _read_json(protocol_path)
    protocol["protocol_id"] = PROTOCOL_ID
    protocol["experiment_variant"] = "int8_heads_binary_transport_immediate_upload_quantization_ablation_v1"
    protocol["source_experiment"] = "20260828_coco_rknn_fp16_online_hil_final_v2 runtime + 20260816 INT8-heads retrained policy"
    protocol["rknn_sha256"] = RKNN_SHA256
    protocol["models"] = {
        "int8_heads": {
            "path": "../yolo26n_coco_val/models/yolo26n-rk3588-int8-heads.rknn",
            "sha256": RKNN_SHA256,
        },
        "cloud_sha256": CLOUD_SHA256,
    }
    protocol["methods"] = {
        "cavs_oqr": {
            "policy": "policies/cavs_oqr/manifest.json",
            "policy_sha256": sha256(policies / "cavs_oqr" / "manifest.json"),
            "actions": ["edge", "original", "q90", "current@640"],
            "scope": "int8_heads_three_seed_policy_retrained_from_int8_board_candidates",
        }
    }
    protocol["controller"] = {
        "id": "projected_primal_dual",
        "update": "one update after every committed frame",
        "upload": "selected payload is sent before the next image is opened",
        "controls": "controls/cavs_oqr",
    }
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_contract = bundle / "scripts" / "run_contract.py"
    content = run_contract.read_text(encoding="utf-8")
    old = 'PROTOCOL_ID = "coco_rknn_fp16_causal_online_hil_final_v2"'
    if content.count(old) != 1:
        raise RuntimeError("source runtime protocol identifier changed unexpectedly")
    run_contract.write_text(content.replace(old, f'PROTOCOL_ID = "{PROTOCOL_ID}"'), encoding="utf-8")
    for test_path in (bundle / "tests").glob("test_*.py"):
        test_content = test_path.read_text(encoding="utf-8")
        test_path.write_text(test_content.replace(old.split(" = ", 1)[1], f'"{PROTOCOL_ID}"'), encoding="utf-8")

    runtime_host = output / "host"
    for name in ("build_calibration_records.py", "calibrate_controls.py", "calibrate_primal_dual.py", "control_schema.py"):
        _copy_required(source_host / name, runtime_host / name)

    record = {
        "schema": "cavs_int8_heads_online_hil_setup_v1",
        "protocol_id": PROTOCOL_ID,
        "source_bundle": str(source_bundle.resolve()),
        "source_bundle_runtime_sha256": sha256(source_bundle / "scripts" / "online_runner.py"),
        "policy_source": str(policy_source.resolve()),
        "policy_manifest_sha256": sha256(policies / "cavs_oqr" / "manifest.json"),
        "rknn_sha256": RKNN_SHA256,
        "cloud_sha256": CLOUD_SHA256,
        "actions": ["edge", "original", "q90", "current@640"],
        "final_used_for_control": False,
    }
    setup_path = output / "INT8_SETUP.json"
    setup_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"bundle": str(bundle), "setup": str(setup_path), "policy_sha256": record["policy_manifest_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--source-host", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_bundle(
        source_bundle=args.source_bundle,
        policy_source=args.policy,
        source_host=args.source_host,
        output=args.output,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
