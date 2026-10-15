#!/usr/bin/env python3
"""Build resumable Cloud‑V3 Original/Q90/R12 prediction caches."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

from PIL import Image


os.environ.setdefault("YOLO_AUTOINSTALL", "False")
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
PROTOCOL_PATH = ROOT / "configs/protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
RUNS = ROOT / "runs"
sys.path.insert(0, str(SCRIPT.parent))

from cache_core import cache_signature, map_crop_bbox, planned_cache_jobs  # noqa: E402
from protocol_core import (  # noqa: E402
    EXPECTED_CLOUD_SHA,
    assert_source_boundary,
    validate_protocol,
    validate_role_isolation,
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module(
    "oregon_cloudv3_geometry_source",
    REPO
    / "paper/experiments/20260728_oregon_cloudv2_roi_profile_scale_scan/scripts/scan_core.py",
)
CANONICAL_PROTOCOL = load_module(
    "oregon_cloudv3_canonical_protocol",
    REPO
    / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/protocol_core.py",
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


def role_paths(role: str) -> dict[str, Path]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(f"forbidden role: {role}")
    canonical = Path(PROTOCOL["canonical_root"])
    return {
        "gt": canonical / f"gt/oregon/{role}/gt.json",
        "edge": canonical / f"actions/edge/{role}/predictions.json",
        "fpn": canonical / f"actions/fpn/{role}/global_fpn_features.json",
        "samples": canonical / f"common/oregon/{role}/samples.json",
        "source_manifest": canonical / f"common/oregon/{role}/source_manifest.json",
        "q90_manifest": canonical / f"actions/q90/{role}/q90_manifest.json",
        "runtime_yaml": canonical / f"runtime_data/{role}.yaml",
    }


def grouped_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_json(path):
        output[int(row["image_id"])].append(row)
    return dict(output)


def crop_name(image_id: int, identity: str) -> str:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{int(image_id):012d}_{digest}.jpg"


def _encode_geometry(
    image: Mapping[str, Any],
    boxes: list[dict[str, Any]],
    *,
    role: str,
    profile: str,
    crop_dir: Path,
) -> tuple[str, dict[str, Any]] | None:
    image_id = int(image["id"])
    width, height = int(image["width"]), int(image["height"])
    settings = PROTOCOL["geometry"]
    xyxy = SCAN.merged_roi_xyxy(
        width,
        height,
        boxes,
        profile=profile,
        confidence=float(settings["edge_conf"]),
        max_area_ratio=float(settings["max_area_ratio"]),
    )
    if xyxy is None:
        return None
    source = Path(str(image.get("absolute_path") or image["file_name"])).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    identity = str(source)
    target = crop_dir / crop_name(image_id, identity)
    x1, y1, x2, y2 = map(int, xyxy)
    if not target.is_file():
        with Image.open(source) as opened:
            if opened.size != (width, height):
                raise RuntimeError(
                    f"image dimensions changed for {identity}: {opened.size} != {(width, height)}"
                )
            opened.crop((x1, y1, x2, y2)).convert("RGB").save(
                target,
                format="JPEG",
                quality=int(settings["jpeg_quality"]),
                optimize=True,
                subsampling=2,
            )
    record = {
        "image_id": image_id,
        "identity": identity,
        "role": role,
        "file_name": str(image["file_name"]),
        "source_path": identity,
        "path": str(target.resolve()),
        "image_width": width,
        "image_height": height,
        "xyxy": [x1, y1, x2, y2],
        "crop_width": x2 - x1,
        "crop_height": y2 - y1,
        "area_ratio": (x2 - x1) * (y2 - y1) / max(1, width * height),
        "bytes": target.stat().st_size,
        "profile": profile,
    }
    return str(image_id), record


def geometry_root(role: str, profile: str) -> Path:
    return RUNS / f"actions/{role}/roi/{profile}/geometry"


def prepare_geometry(role: str, profile: str) -> dict[str, Any]:
    if profile not in PROTOCOL["profiles"]:
        raise ValueError(profile)
    paths = role_paths(role)
    output = geometry_root(role, profile)
    manifest_path = output / "roi_manifest.json"
    signature = cache_signature(
        role=role,
        action_id=f"geometry:{profile}",
        model_sha="edge-only-no-cloud-model",
        source_sha=file_sha256(paths["edge"]),
        settings={
            **PROTOCOL["geometry"],
            "gt_sha256": file_sha256(paths["gt"]),
            "protocol": "real_merged_roi_geometry_v1",
        },
    )
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("signature") != signature:
            raise RuntimeError(f"geometry signature drift: {manifest_path}")
        return manifest
    gt = read_json(paths["gt"])
    images = list(gt["images"])
    if len(images) != int(PROTOCOL["roles"][role]):
        raise RuntimeError(f"{role} GT population changed")
    edge = grouped_predictions(paths["edge"])
    crop_dir = output / "roi_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(
        max_workers=int(PROTOCOL["geometry"]["encode_workers"])
    ) as executor:
        encoded = executor.map(
            lambda image: _encode_geometry(
                image,
                edge.get(int(image["id"]), []),
                role=role,
                profile=profile,
                crop_dir=crop_dir,
            ),
            images,
        )
        records = dict(value for value in encoded if value is not None)
    records = dict(sorted(records.items(), key=lambda item: int(item[0])))
    identities = [str(record["identity"]) for record in records.values()]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"duplicate geometry identity in {role}/{profile}")
    manifest = {
        "protocol": "oregon_cloudv3_real_profile_geometry_v1",
        "role": role,
        "profile": profile,
        "signature": signature,
        "expected_images": len(images),
        "roi_candidates": len(records),
        "records": records,
        "cloud_model_used": False,
        "final_used_for_selection": False,
    }
    atomic_json(manifest_path, manifest)
    atomic_json(
        output / "completion.json",
        {"signature": signature, "manifest_sha256": file_sha256(manifest_path)},
    )
    return manifest


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


def wait_for_gpu() -> None:
    gate = PROTOCOL["resource_gate"]
    while True:
        status = gpu_status()
        if (
            status["free_mib"] >= int(gate["minimum_free_gpu_mib"])
            and status["utilization_percent"]
            <= int(gate["maximum_gpu_utilization_percent"])
        ):
            return
        print(
            "[gpu-gate] "
            f"free={status['free_mib']} MiB util={status['utilization_percent']}% "
            f"required_free={gate['minimum_free_gpu_mib']} MiB",
            flush=True,
        )
        time.sleep(int(gate["poll_seconds"]))


def run_command(command: list[str], *, completion: Path) -> None:
    if completion.is_file():
        return
    subprocess.run(command, cwd=REPO, check=True)
    if not completion.is_file():
        raise RuntimeError(f"command did not create completion: {completion}")


def infer_original(role: str) -> dict[str, Any]:
    paths = role_paths(role)
    output = RUNS / f"actions/{role}/original"
    predictions = output / "predictions.json"
    completion = output / "completion.json"
    signature = cache_signature(
        role=role,
        action_id="original",
        model_sha=EXPECTED_CLOUD_SHA,
        source_sha=file_sha256(paths["gt"]),
        settings={**PROTOCOL["inference"], "imgsz": 640, "source": "original"},
    )
    if completion.is_file():
        state = read_json(completion)
        if (
            state.get("signature") != signature
            or state.get("predictions_sha256") != file_sha256(predictions)
        ):
            raise RuntimeError(f"invalid Original completion: {completion}")
        return state
    output.mkdir(parents=True, exist_ok=True)
    validator = (
        REPO
        / "paper/experiments/20260710_ena24_oracle_fairness/scripts/build_validator_prediction_cache.py"
    )
    manifest = predictions.with_suffix(".manifest.json")
    subprocess.run(
        [
            sys.executable,
            str(validator),
            "--data",
            str(paths["runtime_yaml"]),
            "--split",
            "val",
            "--gt-json",
            str(paths["gt"]),
            "--model",
            str(PROTOCOL["cloud_model"]["path"]),
            "--out-json",
            str(predictions),
            "--imgsz",
            "640",
            "--conf",
            str(PROTOCOL["inference"]["conf"]),
            "--iou",
            str(PROTOCOL["inference"]["iou"]),
            "--max-det",
            str(PROTOCOL["inference"]["max_det"]),
            "--device",
            str(PROTOCOL["inference"]["device"]),
            "--batch",
            str(PROTOCOL["inference"]["batch"]),
        ],
        cwd=REPO,
        check=True,
    )
    if not predictions.is_file() or not manifest.is_file():
        raise RuntimeError(f"Original inference incomplete: {output}")
    state = {
        "signature": signature,
        "role": role,
        "action_id": "original",
        "cloud_model_sha256": file_sha256(Path(PROTOCOL["cloud_model"]["path"])),
        "predictions_sha256": file_sha256(predictions),
        "validator_manifest_sha256": file_sha256(manifest),
        "final_used_for_selection": False,
    }
    atomic_json(completion, state)
    return state


def infer_q90(role: str) -> dict[str, Any]:
    paths = role_paths(role)
    output = RUNS / f"actions/{role}/q90"
    source_script = (
        REPO
        / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts/infer_q90_cache.py"
    )
    command = [
        sys.executable,
        str(source_script),
        "--gt-json",
        str(paths["gt"]),
        "--manifest",
        str(paths["q90_manifest"]),
        "--cloud-model",
        str(PROTOCOL["cloud_model"]["path"]),
        "--expected-cloud-sha256",
        EXPECTED_CLOUD_SHA,
        "--out-dir",
        str(output),
        "--batch",
        str(PROTOCOL["inference"]["batch"]),
        "--chunk-size",
        str(PROTOCOL["inference"]["chunk_size"]),
        "--device",
        str(PROTOCOL["inference"]["device"]),
    ]
    if bool(PROTOCOL["inference"]["half"]):
        command.append("--half")
    run_command(command, completion=output / "COMPLETED")
    summary = read_json(output / "q90_summary.json")
    if summary.get("cloud_model_sha256") != EXPECTED_CLOUD_SHA:
        raise RuntimeError(f"Q90 cache used wrong cloud model: {output}")
    return summary


def roi_output(role: str, profile: str, imgsz: int) -> Path:
    return RUNS / f"actions/{role}/roi/{profile}/imgsz_{int(imgsz)}"


def infer_roi(role: str, profile: str, imgsz: int) -> dict[str, Any]:
    output = roi_output(role, profile, imgsz)
    completion = output / "completion.json"
    predictions_path = output / "roi_predictions_mapped.json"
    geometry = prepare_geometry(role, profile)
    settings = {**PROTOCOL["inference"], "imgsz": int(imgsz), "profile": profile}
    signature = cache_signature(
        role=role,
        action_id=f"{profile}@{int(imgsz)}",
        model_sha=EXPECTED_CLOUD_SHA,
        source_sha=str(geometry["signature"]),
        settings=settings,
    )
    if completion.is_file():
        state = read_json(completion)
        if (
            state.get("signature") != signature
            or state.get("predictions_sha256") != file_sha256(predictions_path)
        ):
            raise RuntimeError(f"invalid ROI completion: {completion}")
        return read_json(output / "summary.json")

    from ultralytics import YOLO

    gt = read_json(role_paths(role)["gt"])
    model_path = Path(PROTOCOL["cloud_model"]["path"])
    if file_sha256(model_path) != EXPECTED_CLOUD_SHA:
        raise RuntimeError("Cloud-V3 model hash changed before ROI inference")
    model = YOLO(str(model_path))
    category_map = CANONICAL_PROTOCOL.category_map_from_names(
        model.names, gt["categories"]
    )
    records = list(geometry["records"].values())
    chunk_size = int(PROTOCOL["inference"]["chunk_size"])
    chunk_dir = output / "prediction_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    total_chunks = (len(records) + chunk_size - 1) // chunk_size
    states = []
    for chunk_index in range(total_chunks):
        chunk = records[chunk_index * chunk_size : (chunk_index + 1) * chunk_size]
        chunk_predictions = chunk_dir / f"predictions_{chunk_index:05d}.json"
        chunk_state = chunk_dir / f"state_{chunk_index:05d}.json"
        chunk_signature = cache_signature(
            role=role,
            action_id=f"{profile}@{int(imgsz)}:chunk:{chunk_index}",
            model_sha=EXPECTED_CLOUD_SHA,
            source_sha=signature,
            settings={
                "paths": [record["path"] for record in chunk],
                "bytes": [int(record["bytes"]) for record in chunk],
                "category_map": category_map,
            },
        )
        if chunk_predictions.is_file() and chunk_state.is_file():
            state = read_json(chunk_state)
            if (
                state.get("signature") == chunk_signature
                and state.get("predictions_sha256") == file_sha256(chunk_predictions)
            ):
                states.append(state)
                continue
        source_list = chunk_dir / f"sources_{chunk_index:05d}.txt"
        source_list.write_text(
            "".join(f"{record['path']}\n" for record in chunk), encoding="utf-8"
        )
        by_path = {str(Path(record["path"]).resolve()): record for record in chunk}
        predictions: list[dict[str, Any]] = []
        for result in model.predict(
            source=str(source_list),
            imgsz=int(imgsz),
            conf=float(PROTOCOL["inference"]["conf"]),
            iou=float(PROTOCOL["inference"]["iou"]),
            max_det=int(PROTOCOL["inference"]["max_det"]),
            device=PROTOCOL["inference"]["device"],
            batch=int(PROTOCOL["inference"]["batch"]),
            half=bool(PROTOCOL["inference"]["half"]),
            rect=bool(PROTOCOL["inference"]["rect"]),
            end2end=bool(PROTOCOL["inference"]["end2end"]),
            verbose=False,
            stream=True,
        ):
            record = by_path[str(Path(result.path).resolve())]
            if result.boxes is None:
                continue
            for box, score, class_index in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                predictions.append(
                    {
                        "image_id": int(record["image_id"]),
                        "category_id": int(category_map[int(class_index)]),
                        "bbox": map_crop_bbox(box, record["xyxy"]),
                        "score": float(score),
                    }
                )
        atomic_json(chunk_predictions, predictions)
        state = {
            "signature": chunk_signature,
            "images": len(chunk),
            "predictions": len(predictions),
            "predictions_sha256": file_sha256(chunk_predictions),
        }
        atomic_json(chunk_state, state)
        states.append(state)

    merged: list[dict[str, Any]] = []
    for chunk_index, state in enumerate(states):
        path = chunk_dir / f"predictions_{chunk_index:05d}.json"
        if state["predictions_sha256"] != file_sha256(path):
            raise RuntimeError(f"ROI chunk changed after inference: {path}")
        merged.extend(read_json(path))
    atomic_json(predictions_path, merged)
    summary = {
        "protocol": "oregon_cloudv3_profile_scale_prediction_v1",
        "role": role,
        "action_id": f"{profile}@{int(imgsz)}",
        "profile": profile,
        "imgsz": int(imgsz),
        "signature": signature,
        "cloud_model_sha256": EXPECTED_CLOUD_SHA,
        "expected_images": int(PROTOCOL["roles"][role]),
        "roi_candidates": len(records),
        "generation_rate": len(records) / max(1, int(PROTOCOL["roles"][role])),
        "predictions": len(merged),
        "request_bytes": sum(int(record["bytes"]) for record in records),
        "predictions_sha256": file_sha256(predictions_path),
        "final_used_for_selection": False,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(
        completion,
        {
            "signature": signature,
            "summary_sha256": file_sha256(output / "summary.json"),
            "predictions_sha256": summary["predictions_sha256"],
        },
    )
    (output / "COMPLETED").write_text("ok\n", encoding="ascii")
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return summary


def _assignment_rows() -> list[dict[str, Any]]:
    canonical_protocol = read_json(Path(PROTOCOL["canonical_protocol"]))
    path = Path(canonical_protocol["datasets"]["oregon"]["assignments"])
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_preflight() -> dict[str, Any]:
    validate_protocol(PROTOCOL)
    model = Path(PROTOCOL["cloud_model"]["path"])
    if file_sha256(model) != EXPECTED_CLOUD_SHA:
        raise RuntimeError("Cloud-V3 epoch-28 best.pt hash mismatch")
    role_samples: dict[str, list[dict[str, Any]]] = {}
    source_paths: list[str] = []
    role_report = {}
    for role, expected in PROTOCOL["roles"].items():
        paths = role_paths(role)
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing)
        gt = read_json(paths["gt"])
        samples = read_json(paths["samples"])
        if len(gt["images"]) != int(expected) or len(samples) != int(expected):
            raise RuntimeError(f"canonical {role} population changed")
        role_samples[role] = [
            {"image_id": int(row["image_id"]), "identity": str(row["identity"])}
            for row in samples
        ]
        source_paths.extend(
            map(str, (paths["gt"], paths["edge"], paths["fpn"], paths["samples"], paths["q90_manifest"], paths["runtime_yaml"]))
        )
        role_report[role] = {
            "images": int(expected),
            "gt_sha256": file_sha256(paths["gt"]),
            "edge_sha256": file_sha256(paths["edge"]),
            "fpn_sha256": file_sha256(paths["fpn"]),
            "samples_sha256": file_sha256(paths["samples"]),
            "q90_manifest_sha256": file_sha256(paths["q90_manifest"]),
        }
    validate_role_isolation(role_samples)
    assert_source_boundary(source_paths)
    assignments = _assignment_rows()
    event_fields = sorted(
        {
            key
            for row in assignments[: min(100, len(assignments))]
            for key in row
            if "event" in key.lower() or "camera" in key.lower()
        }
    )
    if not event_fields:
        raise RuntimeError("canonical assignments do not expose event/camera grouping")
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "passed": True,
        "status": PROTOCOL["status"],
        "cloud_model_path": str(model),
        "cloud_model_sha256": file_sha256(model),
        "cloud_model_epoch": 28,
        "roles": role_report,
        "cache_jobs": len(planned_cache_jobs(PROTOCOL)),
        "event_assignment_rows": len(assignments),
        "event_assignment_fields": event_fields,
        "allowed_reused_artifacts": ["GT/manifests", "Edge predictions", "Edge FPN", "Q90 input images"],
        "old_cloud_predictions_reused": False,
        "old_learner_checkpoints_reused": False,
        "final_used_for_training_or_selection": False,
        "gpu": gpu_status(),
    }
    atomic_json(RUNS / "preflight/report.json", report)
    atomic_json(
        RUNS / "preflight/completion.json",
        {"complete": True, "report_sha256": file_sha256(RUNS / "preflight/report.json")},
    )
    return report


def run_geometry() -> dict[str, Any]:
    run_preflight()
    jobs = [job for job in planned_cache_jobs(PROTOCOL) if job["kind"] == "geometry"]
    for index, job in enumerate(jobs, start=1):
        print(f"[geometry {index}/{len(jobs)}] {job['role']} {job['profile']}", flush=True)
        prepare_geometry(job["role"], job["profile"])
    report = {"complete": True, "jobs": len(jobs), "cloud_model_used": False}
    atomic_json(RUNS / "geometry/report.json", report)
    return report


def run_inference() -> dict[str, Any]:
    run_geometry()
    jobs = [job for job in planned_cache_jobs(PROTOCOL) if job["kind"] == "inference"]
    summaries = []
    for index, job in enumerate(jobs, start=1):
        print(f"[inference {index}/{len(jobs)}] {job['role']} {job['action_id']}", flush=True)
        wait_for_gpu()
        action_id = str(job["action_id"])
        if action_id == "original":
            result = infer_original(job["role"])
        elif action_id == "q90":
            result = infer_q90(job["role"])
        else:
            profile, scale = action_id.split("@", 1)
            result = infer_roi(job["role"], profile, int(scale))
        summaries.append({"role": job["role"], "action_id": action_id, "result": result})
    report = {
        "complete": True,
        "jobs": len(jobs),
        "cloud_model_sha256": EXPECTED_CLOUD_SHA,
        "final_used_for_training_or_selection": False,
    }
    atomic_json(RUNS / "cloud_cache/report.json", report)
    atomic_json(
        RUNS / "cloud_cache/completion.json",
        {"complete": True, "report_sha256": file_sha256(RUNS / "cloud_cache/report.json")},
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "geometry", "inference", "all"), default="all")
    args = parser.parse_args()
    if args.stage == "preflight":
        result = run_preflight()
    elif args.stage == "geometry":
        result = run_geometry()
    else:
        result = run_inference()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
