#!/usr/bin/env python3
"""Verify every frozen COCO O/Q/R payload against regenerated real bytes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
sys.path.insert(0, str(SCRIPT.parent))

from inference_protocol import native_jpeg, paper_q90_jpeg, roi_record  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def audit_one(sample: dict[str, Any], images: Path) -> dict[str, Any]:
    image_id = int(sample["image_id"])
    path = images / str(sample["file_name"])
    expected = sample["payload_bytes"]
    original = len(native_jpeg(path))
    q90 = len(paper_q90_jpeg(path))
    record = roi_record(sample)
    roi = len(paper_q90_jpeg(path, crop_xyxy=record["xyxy"])) if record else None
    return {
        "image_id": image_id,
        "original": original,
        "expected_original": int(expected["original"]),
        "q90": q90,
        "expected_q90": int(expected["q90"]),
        "roi": roi,
        "expected_roi": int(expected["current@640"]) if record else None,
    }


def main() -> None:
    protocol = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path(protocol["dataset"]["images"]))
    parser.add_argument("--samples", type=Path, default=ROOT / protocol["paper_artifacts"]["samples"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/payload_audit/report.json")
    args = parser.parse_args()
    args.images = args.images.resolve()
    args.samples = args.samples.resolve()
    if sha256(args.samples) != protocol["paper_artifacts"]["samples_sha256"]:
        raise RuntimeError("frozen paper samples SHA-256 mismatch")
    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    if len(samples) != int(protocol["dataset"]["expected_images"]):
        raise RuntimeError(f"paper sample count changed: {len(samples)}")
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        rows = list(executor.map(lambda sample: audit_one(sample, args.images), samples))
    failures = [
        row
        for row in rows
        if row["original"] != row["expected_original"]
        or row["q90"] != row["expected_q90"]
        or row["roi"] != row["expected_roi"]
    ]
    report = {
        "complete": not failures,
        "images": len(rows),
        "roi_images": sum(row["roi"] is not None for row in rows),
        "original_total_bytes": sum(row["original"] for row in rows),
        "q90_total_bytes": sum(row["q90"] for row in rows),
        "roi_total_bytes": sum(row["roi"] or 0 for row in rows),
        "failures": len(failures),
        "failure_examples": failures[:20],
        "paper_samples_sha256": sha256(args.samples),
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise RuntimeError(f"payload byte parity failed for {len(failures)} images")
    if report["original_total_bytes"] != int(protocol["dataset"]["native_jpeg_total_bytes"]):
        raise RuntimeError("native JPEG total anchor changed")


if __name__ == "__main__":
    main()
