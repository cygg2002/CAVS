#!/usr/bin/env python3
"""Run resumable single-image Torchvision endpoint inference on fit/calibration only."""

from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sys
import time


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


GATE = load_module("voc_torchvision_infer_gate", ROOT / "scripts/gate_core.py")
RUNTIME = load_module("voc_torchvision_infer_runtime", ROOT / "scripts/runtime_core.py")
PROTOCOL = GATE.read_protocol(PROTOCOL_PATH)


def _models():
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        SSDLite320_MobileNet_V3_Large_Weights,
        fasterrcnn_resnet50_fpn_v2,
        ssdlite320_mobilenet_v3_large,
    )

    threshold = float(PROTOCOL["runtime"]["score_threshold"])
    maximum = int(PROTOCOL["runtime"]["detections_per_image"])
    return {
        "edge": (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
            {"score_thresh": threshold, "detections_per_img": maximum},
        ),
        "cloud": (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
            {"box_score_thresh": threshold, "box_detections_per_img": maximum},
        ),
    }


def _load_model(kind: str):
    import torch

    builder, weights, kwargs = _models()[kind]
    model = builder(weights=weights, progress=True, **kwargs)
    model.eval().to(torch.device(PROTOCOL["runtime"]["device"]))
    return model, weights


def _configure_runtime() -> None:
    import torch

    torch.manual_seed(int(PROTOCOL["smoke"]["seed"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _image_population(role_data: dict, stage: str) -> list[dict]:
    if stage == "full":
        return list(role_data["images"])
    if stage != "smoke" or role_data["role"] != "fit":
        raise ValueError("smoke is restricted to fit")
    return GATE.select_smoke_images(
        role_data["images"],
        count=int(PROTOCOL["smoke"]["images"]),
        seed=int(PROTOCOL["smoke"]["seed"]),
        category_ids=[int(row["id"]) for row in role_data["categories"]],
    )


def _input_image(image: dict, action: str):
    from PIL import Image

    path = Path(image["absolute_path"])
    original = Image.open(path).convert("RGB")
    if action != "q90":
        return original, (0 if action == "edge" else path.stat().st_size)
    buffer = BytesIO()
    q90 = PROTOCOL["jpeg_q90"]
    original.save(
        buffer,
        format="JPEG",
        quality=int(q90["quality"]),
        optimize=bool(q90["optimize"]),
        subsampling=int(q90["subsampling"]),
    )
    payload = buffer.getvalue()
    return Image.open(BytesIO(payload)).convert("RGB"), len(payload)


def _infer_one(model, weights, device, image: dict, action: str, label_names, category_ids):
    import torch

    pil_image, payload_bytes = _input_image(image, action)
    tensor = weights.transforms()(pil_image).to(device)
    torch.cuda.synchronize(device)
    before = time.perf_counter()
    with torch.inference_mode():
        prediction = model([tensor])[0]
    torch.cuda.synchronize(device)
    predictions, sanitization = GATE.sanitize_detections(
        image_id=int(image["id"]),
        width=int(image["width"]),
        height=int(image["height"]),
        boxes=prediction["boxes"].detach().cpu().tolist(),
        labels=prediction["labels"].detach().cpu().tolist(),
        scores=prediction["scores"].detach().cpu().tolist(),
        label_names=label_names,
        category_ids=category_ids,
        threshold=float(PROTOCOL["runtime"]["score_threshold"]),
    )
    return {
        "action": action,
        "image_id": int(image["id"]),
        "identity": str(image["identity"]),
        "protocol_sha256": RUNTIME.file_sha256(PROTOCOL_PATH),
        "payload_bytes": int(payload_bytes),
        "latency_ms": float((time.perf_counter() - before) * 1000.0),
        "predictions": predictions,
        "sanitization": sanitization,
    }


def _run_action(role: str, stage: str, action: str, limit: int | None) -> dict:
    import torch

    protocol_sha256 = RUNTIME.file_sha256(PROTOCOL_PATH)
    if not (RUNS / "gate0/completion.json").is_file():
        raise RuntimeError("Gate 0 audit must complete before inference")
    RUNTIME.read_protocol_bound_json(RUNS / "gate0/completion.json", protocol_sha256)
    role_data = RUNTIME.load_source_role(ROOT, PROTOCOL, role)
    images = _image_population(role_data, stage)
    expected_ids = {int(row["id"]) for row in images}
    output = RUNS / f"actions/{stage}/{role}/{action}"
    rows_path = output / "rows.jsonl"
    completion_path = output / "completion.json"
    if completion_path.is_file():
        report = RUNTIME.read_protocol_bound_json(completion_path, protocol_sha256)
        expected_contract = {"stage": stage, "role": role, "action": action}
        if any(str(report.get(key)) != str(value) for key, value in expected_contract.items()):
            raise RuntimeError(f"completed inference contract changed: {stage}/{role}/{action}")
        if report.get("rows_sha256") != RUNTIME.file_sha256(rows_path):
            raise RuntimeError(f"completed inference changed: {stage}/{role}/{action}")
        return report
    existing = RUNTIME.read_checkpoint(
        rows_path,
        action=action,
        expected_ids=expected_ids,
        protocol_sha256=protocol_sha256,
    )
    pending = [row for row in images if int(row["id"]) not in existing]
    if limit is not None:
        pending = pending[: max(0, int(limit))]
    kind = "edge" if action == "edge" else "cloud"
    model, weights = _load_model(kind)
    device = torch.device(PROTOCOL["runtime"]["device"])
    label_names = {index: str(name) for index, name in enumerate(weights.meta["categories"])}
    voc_names = [str(row["name"]) for row in role_data["categories"]]
    category_ids = GATE.coco_to_voc_ids(PROTOCOL["category_mapping"], voc_names)
    started = time.time()
    for index, image in enumerate(pending, 1):
        RUNTIME.append_jsonl(
            rows_path,
            _infer_one(model, weights, device, image, action, label_names, category_ids),
        )
        if index % 50 == 0 or index == len(pending):
            print(
                f"[{stage}/{role}/{action}] {len(existing) + index}/{len(images)}",
                flush=True,
            )
    rows = RUNTIME.read_checkpoint(
        rows_path,
        action=action,
        expected_ids=expected_ids,
        protocol_sha256=protocol_sha256,
    )
    complete = set(rows) == expected_ids
    counters = Counter()
    for row in rows.values():
        counters.update({key: int(value) for key, value in row["sanitization"].items()})
    determinism = {"images": 0, "exact_match": None}
    if complete and stage == "smoke":
        repeated = images[:2]
        matches = []
        for image in repeated:
            observed = _infer_one(model, weights, device, image, action, label_names, category_ids)
            reference = rows[int(image["id"])]
            matches.append(
                observed["payload_bytes"] == reference["payload_bytes"]
                and observed["predictions"] == reference["predictions"]
                and observed["sanitization"] == reference["sanitization"]
            )
        determinism = {"images": len(repeated), "exact_match": all(matches)}
    report = {
        "complete": complete,
        "stage": stage,
        "role": role,
        "action": action,
        "images": len(rows),
        "expected_images": len(images),
        "payload_bytes": sum(int(row["payload_bytes"]) for row in rows.values()),
        "mean_latency_ms": sum(float(row["latency_ms"]) for row in rows.values()) / max(1, len(rows)),
        "sanitization": dict(counters),
        "determinism": determinism,
        "elapsed_seconds_this_invocation": time.time() - started,
        "rows_sha256": RUNTIME.file_sha256(rows_path),
        "protocol_sha256": protocol_sha256,
        "final_accessed": False,
    }
    RUNTIME.atomic_json(output / "report.json", report)
    if complete:
        RUNTIME.atomic_json(completion_path, report)
    return report


def run(role: str, stage: str, action: str, limit: int | None) -> dict:
    GATE.validate_protocol(PROTOCOL)
    if role not in PROTOCOL["allowed_roles"]:
        raise ValueError(f"forbidden data role: {role}")
    if action not in {"edge", "original", "q90", "all"}:
        raise ValueError(action)
    _configure_runtime()
    actions = ("edge", "original", "q90") if action == "all" else (action,)
    reports = {value: _run_action(role, stage, value, limit) for value in actions}
    return {
        "complete": all(bool(row["complete"]) for row in reports.values()),
        "stage": stage,
        "role": role,
        "actions": reports,
        "final_accessed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("fit", "calibration"))
    parser.add_argument("--stage", required=True, choices=("smoke", "full"))
    parser.add_argument("--action", required=True, choices=("edge", "original", "q90", "all"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.role, args.stage, args.action, args.limit), ensure_ascii=False, indent=2))
