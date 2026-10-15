#!/usr/bin/env python3
"""Launch the frozen Oregon Cloud-V5 30-epoch fine-tune."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_protocol(*, require_new_destination: bool = True) -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    source = Path(protocol["source_checkpoint"])
    model_yaml = Path(protocol["model_yaml"])
    data_yaml = Path(protocol["data_yaml"])
    destination = Path(protocol["project"]) / protocol["name"]

    for path in (source, model_yaml, data_yaml):
        if not path.is_file():
            raise FileNotFoundError(path)
    actual_sha = sha256(source)
    if actual_sha != protocol["source_sha256"]:
        raise RuntimeError(f"source checkpoint hash mismatch: {actual_sha}")
    model_text = model_yaml.read_text()
    if "end2end: False" not in model_text and "end2end: false" not in model_text:
        raise RuntimeError("model YAML is not explicitly O2M-only")
    if require_new_destination and destination.exists():
        raise FileExistsError(destination)

    expected = {
        "epochs": 30,
        "batch": 8,
        "imgsz": 640,
        "optimizer": "auto",
        "seed": 0,
        "resume": False,
        "exist_ok": False,
    }
    train = protocol["train"]
    for key, value in expected.items():
        if train.get(key) != value:
            raise RuntimeError(
                f"frozen setting mismatch: {key}={train.get(key)!r}, expected {value!r}"
            )
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    protocol = load_and_validate_protocol()
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "experiment_id": protocol["experiment_id"],
                }
            )
        )
        return

    from ultralytics import YOLO

    model = YOLO(protocol["model_yaml"])
    model.train(
        data=protocol["data_yaml"],
        project=protocol["project"],
        name=protocol["name"],
        pretrained=protocol["source_checkpoint"],
        **protocol["train"],
    )


if __name__ == "__main__":
    main()
