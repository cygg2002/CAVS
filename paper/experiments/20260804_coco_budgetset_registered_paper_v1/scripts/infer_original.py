#!/usr/bin/env python3
"""Build resumable Original-image Edge and Cloud prediction caches."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_core import (  # noqa: E402
    batches,
    canonical_hash,
    category_index_to_id,
    chunk_signature,
    resolve_result_image_id,
    validate_prediction_population,
)
from run_paper import file_sha256, read_json, source_fingerprint  # noqa: E402


PROTOCOL_PATH = ROOT / "configs/protocol.json"
RUNS = ROOT / "runs"


def directory_sha256(root: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _prediction_rows(result: Any, image_id: int, category_map: dict[int, int]) -> list[dict[str, Any]]:
    if result.boxes is None:
        return []
    output = []
    for xyxy, confidence, class_id in zip(
        result.boxes.xyxy.detach().cpu().tolist(),
        result.boxes.conf.detach().cpu().tolist(),
        result.boxes.cls.detach().cpu().tolist(),
    ):
        x1, y1, x2, y2 = map(float, xyxy)
        output.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_map[int(class_id)]),
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(confidence),
            }
        )
    return output


def run_one(
    *,
    model: Any,
    model_hash: str,
    detector: str,
    role: str,
    records: list[dict[str, Any]],
    category_map: dict[int, int],
    protocol: dict[str, Any],
    protocol_hash: str,
    implementation_root: str,
    implementation_sha256: str,
    device: str,
    batch: int,
    chunk_size: int,
) -> dict[str, Any]:
    output_root = RUNS / "inference" / role / detector
    completion_path = output_root / "completion.json"
    if completion_path.is_file():
        report = read_json(output_root / "report.json")
        if (
            report.get("protocol_hash") == protocol_hash
            and report.get("detector_implementation_sha256") == implementation_sha256
        ):
            return report
        raise RuntimeError(f"Existing inference completion is not implementation-pinned: {role}/{detector}")
    chunk_root = output_root / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    detector_config = protocol["detector"]
    for chunk_index, offset in enumerate(range(0, len(records), chunk_size)):
        chunk = records[offset : offset + chunk_size]
        image_ids = [int(row["image_id"]) for row in chunk]
        paths = [str(row["path"]) for row in chunk]
        signature = chunk_signature(
            protocol_hash=protocol_hash,
            role=role,
            detector=detector,
            model_hash=model_hash,
            image_ids=image_ids,
            paths=paths,
        )
        predictions_path = chunk_root / f"predictions_{chunk_index:05d}.json"
        state_path = chunk_root / f"state_{chunk_index:05d}.json"
        if predictions_path.is_file() and state_path.is_file():
            state = read_json(state_path)
            if state.get("signature") == signature and state.get("sha256") == file_sha256(predictions_path):
                continue
        path_to_id = {str(Path(row["path"]).resolve()): int(row["image_id"]) for row in chunk}
        predictions = []
        seen_ids = set()
        indexed_paths = list(zip(paths, image_ids))
        for group in batches(indexed_paths, int(batch)):
            group_paths = [path for path, _ in group]
            group_ids = [image_id for _, image_id in group]
            stream = model.predict(
                source=group_paths,
                imgsz=int(detector_config["imgsz"]),
                conf=float(detector_config["confidence"]),
                iou=float(detector_config["iou"]),
                max_det=int(detector_config["max_det"]),
                rect=bool(detector_config["rect"]),
                end2end=bool(detector_config["end2end"]),
                device=device,
                batch=len(group),
                half=bool(detector_config["half"]),
                verbose=False,
                stream=True,
            )
            for result_index, result in enumerate(stream):
                image_id = resolve_result_image_id(
                    str(result.path),
                    path_to_id=path_to_id,
                    ordered_image_ids=group_ids,
                    result_index=result_index,
                )
                seen_ids.add(image_id)
                predictions.extend(_prediction_rows(result, image_id, category_map))
        if seen_ids != set(image_ids):
            raise RuntimeError(
                f"Inference result population mismatch: missing={sorted(set(image_ids) - seen_ids)[:10]}"
            )
        validate_prediction_population(predictions, expected_ids=set(image_ids))
        atomic_json(predictions_path, predictions)
        atomic_json(
            state_path,
            {
                "signature": signature,
                "images": len(chunk),
                "predictions": len(predictions),
                "sha256": file_sha256(predictions_path),
            },
        )
        print(
            f"[infer] detector={detector} role={role} chunk={chunk_index + 1}/"
            f"{(len(records) + chunk_size - 1) // chunk_size}",
            flush=True,
        )

    merged = []
    states = []
    total_chunks = (len(records) + chunk_size - 1) // chunk_size
    for chunk_index in range(total_chunks):
        predictions_path = chunk_root / f"predictions_{chunk_index:05d}.json"
        state_path = chunk_root / f"state_{chunk_index:05d}.json"
        state = read_json(state_path)
        if state["sha256"] != file_sha256(predictions_path):
            raise RuntimeError(f"Prediction chunk hash mismatch: {role}/{detector}/{chunk_index}")
        merged.extend(read_json(predictions_path))
        states.append(state)
    validate_prediction_population(merged, expected_ids={int(row["image_id"]) for row in records})
    prediction_path = output_root / "predictions.json"
    atomic_json(prediction_path, merged)
    report = {
        "role": role,
        "detector": detector,
        "images": len(records),
        "predictions": len(merged),
        "chunks": len(states),
        "model_sha256": model_hash,
        "protocol_hash": protocol_hash,
        "detector_implementation_root": implementation_root,
        "detector_implementation_sha256": implementation_sha256,
        "predictions_sha256": file_sha256(prediction_path),
    }
    atomic_json(output_root / "report.json", report)
    atomic_json(
        completion_path,
        {"complete": True, "report_sha256": file_sha256(output_root / "report.json"), **report},
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("all", "development", "audit", "registered_final"), default="all")
    parser.add_argument("--detector", choices=("all", "edge", "cloud"), default="all")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--smoke-images", type=int, default=0)
    args = parser.parse_args()

    protocol = read_json(PROTOCOL_PATH)
    prepared = read_json(RUNS / "prepare/completion.json")
    fingerprint = source_fingerprint(protocol)
    if prepared.get("source_fingerprint") != fingerprint:
        raise RuntimeError("Prepared/source fingerprint drift")
    split = read_json(RUNS / "prepare/split_manifest.json")
    annotations = read_json(Path(protocol["paths"]["annotations"]))
    if args.smoke_images > 0 and args.role == "all":
        raise ValueError("Smoke inference requires one explicit source role")
    roles = list(split["roles"]) if args.role == "all" else [args.role]
    detectors = ("edge", "cloud") if args.detector == "all" else (args.detector,)

    import ultralytics
    from ultralytics import YOLO

    implementation_root = Path(ultralytics.__file__).resolve().parent
    implementation_sha256 = directory_sha256(implementation_root)
    protocol_hash = canonical_hash(
        {
            "protocol": fingerprint["protocol"],
            "inference_script": file_sha256(Path(__file__)),
            "detector_implementation": implementation_sha256,
        }
    )

    reports = []
    for detector in detectors:
        model_path = Path(protocol["paths"][f"{detector}_model"])
        model_hash = fingerprint[f"{detector}_model"]
        model = YOLO(str(model_path))
        category_map = category_index_to_id(model.names, annotations["categories"])
        default_batch = int(protocol["detector"][f"{detector}_batch"])
        for source_role in roles:
            records = list(split["roles"][source_role])
            output_role = source_role
            if args.smoke_images > 0:
                records = records[: int(args.smoke_images)]
                output_role = f"smoke_{source_role}_{int(args.smoke_images)}"
            reports.append(
                run_one(
                    model=model,
                    model_hash=model_hash,
                    detector=detector,
                    role=output_role,
                    records=records,
                    category_map=category_map,
                    protocol=protocol,
                    protocol_hash=protocol_hash,
                    implementation_root=str(implementation_root),
                    implementation_sha256=implementation_sha256,
                    device=str(args.device),
                    batch=int(args.batch or default_batch),
                    chunk_size=int(args.chunk_size or protocol["detector"]["chunk_size"]),
                )
            )
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    print(json.dumps({"complete": True, "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
