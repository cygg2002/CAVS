#!/usr/bin/env python3
"""Build shared ROI geometry and scale-specific Cloud-V2 prediction caches."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image


SCRIPT = Path(__file__).resolve()
EXP_ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
SOURCE_SCRIPTS = (
    REPO
    / "paper/experiments/20260720_frozen_detector_multiranker_contextm2e/scripts"
)
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(SOURCE_SCRIPTS))

from protocol_core import audit_roi_records, category_map_from_names  # noqa: E402
from scan_core import (  # noqa: E402
    PROFILES,
    canonical_signature,
    merged_roi_xyxy,
    validate_roles,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in read_json(path):
        grouped[int(prediction["image_id"])].append(prediction)
    return dict(grouped)


def paths(config: dict[str, Any], role: str, profile: str) -> dict[str, Path]:
    validate_roles(config["roles"])
    if role not in config["roles"]:
        raise ValueError(f"role must be one of {tuple(config['roles'])}")
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    source = (
        Path(config["source_cloudv2_root"])
        / "runs/canonical_oregon_event_v1"
    )
    root = (
        Path(config["output_root"])
        / f"runs/actions/{role}/{profile}"
    )
    return {
        "gt": source / f"gt/oregon/{role}/gt.json",
        "edge": source / f"actions/edge/{role}/predictions.json",
        "geometry": root / "geometry",
        "model": Path(config["cloud_model"]["path"]),
    }


def source_path(image: dict[str, Any]) -> Path:
    candidate = Path(
        str(image.get("absolute_path") or image.get("file_name"))
    )
    if not candidate.is_file():
        raise FileNotFoundError(f"missing source image: {candidate}")
    return candidate.resolve()


def crop_name(image_id: int, source: Path) -> str:
    identity = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
    return f"{image_id:012d}_{identity}.jpg"


def geometry_signature(
    config: dict[str, Any], role: str, profile: str, source: dict[str, Path]
) -> dict[str, Any]:
    geometry = config["geometry"]
    inputs = {
        "role": role,
        "profile": profile,
        "gt_sha256": file_sha256(source["gt"]),
        "edge_sha256": file_sha256(source["edge"]),
        "edge_conf": float(geometry["edge_conf"]),
        "max_area_ratio": float(geometry["max_area_ratio"]),
        "jpeg_quality": int(geometry["jpeg_quality"]),
        "expected_images": int(config["roles"][role]),
    }
    return {"inputs": inputs, "sha256": canonical_signature(inputs)}


def encode_image(
    image: dict[str, Any],
    boxes: list[dict[str, Any]],
    *,
    crop_dir: Path,
    profile: str,
    edge_conf: float,
    max_area_ratio: float,
    jpeg_quality: int,
) -> tuple[str, dict[str, Any]] | None:
    image_id = int(image["id"])
    width = int(image["width"])
    height = int(image["height"])
    roi = merged_roi_xyxy(
        width,
        height,
        boxes,
        profile=profile,
        confidence=edge_conf,
        max_area_ratio=max_area_ratio,
    )
    if roi is None:
        return None
    source = source_path(image)
    target = crop_dir / crop_name(image_id, source)
    x1, y1, x2, y2 = roi
    if not target.is_file():
        with Image.open(source) as image_file:
            if image_file.size != (width, height):
                raise ValueError(
                    f"image size mismatch: {source} "
                    f"GT={(width, height)} actual={image_file.size}"
                )
            image_file.crop((x1, y1, x2, y2)).convert("RGB").save(
                target,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
                subsampling=2,
            )
    record = {
        "image_id": image_id,
        "file_name": str(image["file_name"]),
        "source_path": str(source),
        "path": str(target.resolve()),
        "image_width": width,
        "image_height": height,
        "xyxy": [x1, y1, x2, y2],
        "crop_width": x2 - x1,
        "crop_height": y2 - y1,
        "area_ratio": (x2 - x1) * (y2 - y1) / (width * height),
        "bytes": target.stat().st_size,
        "profile": profile,
    }
    return str(image_id), record


def prepare_geometry(
    config: dict[str, Any], role: str, profile: str
) -> dict[str, Any]:
    source = paths(config, role, profile)
    signature = geometry_signature(config, role, profile, source)
    manifest_path = source["geometry"] / "roi_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("signature", {}).get("sha256") != signature["sha256"]:
            raise RuntimeError(f"geometry signature drift: {manifest_path}")
        return manifest

    gt = read_json(source["gt"])
    images = list(gt["images"])
    if len(images) != int(config["roles"][role]):
        raise RuntimeError(
            f"{role} image count changed: {len(images)} != {config['roles'][role]}"
        )
    edge = grouped_predictions(source["edge"])
    geometry = config["geometry"]
    crop_dir = source["geometry"] / "roi_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=int(geometry["encode_workers"])) as executor:
        encoded = executor.map(
            lambda image: encode_image(
                image,
                edge.get(int(image["id"]), []),
                crop_dir=crop_dir,
                profile=profile,
                edge_conf=float(geometry["edge_conf"]),
                max_area_ratio=float(geometry["max_area_ratio"]),
                jpeg_quality=int(geometry["jpeg_quality"]),
            ),
            images,
        )
        records = dict(value for value in encoded if value is not None)
    records = dict(sorted(records.items(), key=lambda item: int(item[0])))
    expected_ids = {int(image["id"]) for image in images}
    audit = audit_roi_records(records, expected_ids=expected_ids, require_all=False)
    if not audit["passed"]:
        raise RuntimeError(f"ROI geometry audit failed: {audit}")
    manifest = {
        "protocol": "oregon_cloudv2_profile_geometry_v1",
        "role": role,
        "profile": profile,
        "signature": signature,
        "expected_image_ids": sorted(expected_ids),
        "records": records,
        "audit": audit,
        "registered_final_test_accessed": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def prediction_signature(
    config: dict[str, Any],
    geometry: dict[str, Any],
    imgsz: int,
    model: Path,
) -> dict[str, Any]:
    inference = config["inference"]
    model_hash = file_sha256(model)
    if model_hash != config["cloud_model"]["sha256"]:
        raise RuntimeError("Cloud-V2 model hash mismatch")
    inputs = {
        "geometry_sha256": geometry["signature"]["sha256"],
        "cloud_model_sha256": model_hash,
        "imgsz": int(imgsz),
        "conf": float(inference["conf"]),
        "iou": float(inference["iou"]),
        "max_det": int(inference["max_det"]),
        "rect": bool(inference["rect"]),
        "end2end": bool(inference["end2end"]),
    }
    return {"inputs": inputs, "sha256": canonical_signature(inputs)}


def completion_matches(output: Path, signature: str) -> bool:
    completion_path = output / "completion.json"
    summary_path = output / "summary.json"
    prediction_path = output / "roi_predictions_mapped.json"
    if not all(
        path.is_file()
        for path in (completion_path, summary_path, prediction_path)
    ):
        return False
    completion = read_json(completion_path)
    return (
        completion.get("signature") == signature
        and completion.get("summary_sha256") == file_sha256(summary_path)
        and completion.get("prediction_sha256")
        == file_sha256(prediction_path)
    )


def infer_candidate(
    config: dict[str, Any], role: str, profile: str, imgsz: int
) -> dict[str, Any]:
    if int(imgsz) not in tuple(map(int, config["imgsz"])):
        raise ValueError(f"unregistered imgsz: {imgsz}")
    source = paths(config, role, profile)
    geometry = prepare_geometry(config, role, profile)
    model_path = source["model"]
    signature = prediction_signature(config, geometry, imgsz, model_path)
    output = source["geometry"].parent / f"imgsz_{int(imgsz)}"
    completion = output / "completion.json"
    if completion.is_file():
        if not completion_matches(output, signature["sha256"]):
            raise RuntimeError(f"invalid prediction completion: {completion}")
        return read_json(output / "summary.json")

    from ultralytics import YOLO

    gt = read_json(source["gt"])
    model = YOLO(str(model_path))
    category_map = category_map_from_names(model.names, gt["categories"])
    records = list(geometry["records"].values())
    chunk_dir = output / "prediction_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    inference = config["inference"]
    chunk_size = int(inference["chunk_size"])
    total_chunks = (len(records) + chunk_size - 1) // chunk_size
    chunk_states = []
    for chunk_index in range(total_chunks):
        chunk = records[chunk_index * chunk_size : (chunk_index + 1) * chunk_size]
        predictions_path = chunk_dir / f"predictions_{chunk_index:05d}.json"
        state_path = chunk_dir / f"state_{chunk_index:05d}.json"
        chunk_hash = canonical_signature(
            {
                "prediction": signature["sha256"],
                "paths": [record["path"] for record in chunk],
                "bytes": [record["bytes"] for record in chunk],
                "category_map": category_map,
            }
        )
        if predictions_path.is_file() and state_path.is_file():
            state = read_json(state_path)
            if (
                state.get("signature") == chunk_hash
                and state.get("prediction_sha256")
                == file_sha256(predictions_path)
            ):
                chunk_states.append(state)
                continue
        sources = chunk_dir / f"sources_{chunk_index:05d}.txt"
        sources.write_text(
            "\n".join(record["path"] for record in chunk) + "\n",
            encoding="utf-8",
        )
        by_path = {str(Path(record["path"]).resolve()): record for record in chunk}
        predictions = []
        for result in model.predict(
            source=str(sources),
            imgsz=int(imgsz),
            conf=float(inference["conf"]),
            iou=float(inference["iou"]),
            max_det=int(inference["max_det"]),
            device=inference["device"],
            batch=int(inference["batch"]),
            half=bool(inference["half"]),
            rect=bool(inference["rect"]),
            end2end=bool(inference["end2end"]),
            verbose=False,
            stream=True,
        ):
            record = by_path[str(Path(result.path).resolve())]
            offset_x, offset_y = record["xyxy"][:2]
            if result.boxes is None:
                continue
            for box, score, class_index in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                x1, y1, x2, y2 = map(float, box)
                predictions.append(
                    {
                        "image_id": int(record["image_id"]),
                        "category_id": int(category_map[int(class_index)]),
                        "bbox": [
                            x1 + offset_x,
                            y1 + offset_y,
                            max(0.0, x2 - x1),
                            max(0.0, y2 - y1),
                        ],
                        "score": float(score),
                    }
                )
        write_json(predictions_path, predictions)
        state = {
            "signature": chunk_hash,
            "images": len(chunk),
            "predictions": len(predictions),
            "prediction_sha256": file_sha256(predictions_path),
        }
        write_json(state_path, state)
        chunk_states.append(state)

    merged = []
    for chunk_index, state in enumerate(chunk_states):
        predictions_path = chunk_dir / f"predictions_{chunk_index:05d}.json"
        if file_sha256(predictions_path) != state["prediction_sha256"]:
            raise RuntimeError(f"prediction chunk hash mismatch: {predictions_path}")
        merged.extend(read_json(predictions_path))
    predictions_path = output / "roi_predictions_mapped.json"
    write_json(predictions_path, merged)
    summary = {
        "protocol": "oregon_cloudv2_profile_scale_prediction_v1",
        "role": role,
        "candidate_id": f"{profile}@{int(imgsz)}",
        "profile": profile,
        "imgsz": int(imgsz),
        "signature": signature,
        "geometry_signature": geometry["signature"]["sha256"],
        "expected_images": len(geometry["expected_image_ids"]),
        "roi_candidates": len(records),
        "generation_rate": len(records) / max(1, len(geometry["expected_image_ids"])),
        "predictions": len(merged),
        "request_bytes": sum(int(record["bytes"]) for record in records),
        "prediction_sha256": file_sha256(predictions_path),
        "registered_final_test_accessed": False,
    }
    write_json(output / "summary.json", summary)
    write_json(
        completion,
        {
            "signature": signature["sha256"],
            "summary_sha256": file_sha256(output / "summary.json"),
            "prediction_sha256": summary["prediction_sha256"],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXP_ROOT / "configs/protocol.json",
    )
    parser.add_argument("--role", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--imgsz", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    result = (
        prepare_geometry(config, args.role, args.profile)
        if args.imgsz is None
        else infer_candidate(config, args.role, args.profile, args.imgsz)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
