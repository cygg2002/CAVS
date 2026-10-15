#!/usr/bin/env python3
"""Audit frozen data, official Torchvision weights, and runtime versions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from urllib.parse import urlparse
import sys


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/protocol.json"
RUNS = ROOT / "runs"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_module("voc_torchvision_audit_gate", ROOT / "scripts/gate_core.py")
RUNTIME = load_module("voc_torchvision_audit_runtime", ROOT / "scripts/runtime_core.py")


def _weights_contracts():
    import torch
    import torchvision
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        SSDLite320_MobileNet_V3_Large_Weights,
    )

    return torch, torchvision, {
        "edge": SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
        "cloud": FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
    }


def _downloaded_weight_path(torch_module, url: str) -> Path:
    return Path(torch_module.hub.get_dir()) / "checkpoints" / Path(urlparse(url).path).name


def run() -> dict:
    protocol = GATE.read_protocol(PROTOCOL_PATH)
    GATE.validate_protocol(protocol)
    if GATE.file_sha256(PROTOCOL_PATH) == "":
        raise RuntimeError("empty protocol hash")

    source_reports = {}
    for role in protocol["allowed_roles"]:
        data = RUNTIME.load_source_role(ROOT, protocol, role)
        source_reports[str(role)] = {
            "images": len(data["images"]),
            "gt_sha256": RUNTIME.file_sha256(data["paths"]["gt"]),
            "images_sha256": RUNTIME.file_sha256(data["paths"]["images"]),
            "identity_sha256": RUNTIME.text_sha256(
                "\n".join(sorted(str(row["identity"]) for row in data["images"]))
            ),
        }
    metric_path = RUNTIME.source_role_paths(ROOT, protocol, "fit")["metric"]
    metric_sha256 = RUNTIME.file_sha256(metric_path)
    if metric_sha256 != str(protocol["source_metric_sha256"]):
        raise RuntimeError("frozen VOC metric source changed")

    torch, torchvision, weights_by_role = _weights_contracts()
    if str(torch.__version__) != str(protocol["runtime"]["torch"]):
        raise RuntimeError(f"torch version changed: {torch.__version__}")
    if str(torchvision.__version__) != str(protocol["runtime"]["torchvision"]):
        raise RuntimeError(f"torchvision version changed: {torchvision.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for registered endpoint inference")

    voc_names = [row["name"] for row in RUNTIME.load_source_role(ROOT, protocol, "fit")["categories"]]
    weight_reports = {}
    for role, weights in weights_by_role.items():
        detector = protocol["detectors"][role]
        meta = weights.meta
        official_map = float(meta["_metrics"]["COCO-val2017"]["box_map"]) / 100.0
        if abs(official_map - float(detector["official_box_map"])) > 1e-12:
            raise RuntimeError(f"official box mAP changed: {role}/{official_map}")
        if int(meta["num_params"]) != int(detector["official_num_params"]):
            raise RuntimeError(f"official parameter count changed: {role}")
        if abs(float(meta["_ops"]) - float(detector["official_gflops"])) > 1e-12:
            raise RuntimeError(f"official operation count changed: {role}")
        GATE.coco_to_voc_ids(protocol["category_mapping"], voc_names)
        missing = sorted(set(protocol["category_mapping"]) - set(map(str, meta["categories"])))
        if missing:
            raise RuntimeError(f"weight categories cannot map to VOC: {role}/{missing}")
        weights.get_state_dict(progress=True, check_hash=True)
        weight_path = _downloaded_weight_path(torch, weights.url)
        if not weight_path.is_file():
            raise FileNotFoundError(weight_path)
        weight_reports[role] = {
            "weights_enum": str(weights),
            "url": str(weights.url),
            "path": str(weight_path),
            "sha256": RUNTIME.file_sha256(weight_path),
            "bytes": weight_path.stat().st_size,
            "official_box_map": official_map,
            "official_num_params": int(meta["num_params"]),
            "official_gflops": float(meta["_ops"]),
            "training_dataset": str(detector["training_dataset"]),
        }

    report = {
        "complete": True,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": RUNTIME.file_sha256(PROTOCOL_PATH),
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_device": torch.cuda.get_device_name(0),
        "source_roles": source_reports,
        "source_metric_sha256": metric_sha256,
        "weights": weight_reports,
        "allowed_roles": list(protocol["allowed_roles"]),
        "final_accessed": False,
        "proceed_to_smoke": True,
        "sources": list(protocol["sources"]),
    }
    RUNTIME.atomic_json(RUNS / "gate0/audit.json", report)
    RUNTIME.atomic_json(RUNS / "gate0/completion.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
