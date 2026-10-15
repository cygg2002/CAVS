#!/usr/bin/env python3
"""Assemble observable VOC samples and independent O/OQ CAVS records."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
RUNS = ROOT / "runs"
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DATA = load_module("voc_gluoncv_common_data", ROOT / "scripts/data_core.py")


def _group(path: Path, image_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    return DATA.group_predictions(DATA.read_json(path), image_ids=image_ids)


def _q90_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = DATA.read_json(path)["records"]
    output = {int(row["image_id"]): dict(row) for row in rows}
    if len(output) != len(rows):
        raise RuntimeError(f"duplicate Q90 manifest image: {path}")
    return output


def build_role(role: str) -> dict[str, Any]:
    if role not in PROTOCOL["roles"]:
        raise ValueError(role)
    output = RUNS / f"common/{role}"
    completion = output / "completion.json"
    samples_path = output / "samples.json"
    o_path = output / "records_O.pkl"
    oq_path = output / "records_OQ.pkl"
    manifest_path = output / "source_manifest.json"
    if completion.is_file():
        report = DATA.read_json(completion)
        for key, path in (
            ("samples_sha256", samples_path),
            ("records_O_sha256", o_path),
            ("records_OQ_sha256", oq_path),
            ("manifest_sha256", manifest_path),
        ):
            if report.get(key) != DATA.file_sha256(path):
                raise RuntimeError(f"completed common artifact changed: {role}/{path.name}")
        return report
    gt_path = RUNS / f"data/roles/{role}/gt.json"
    action_root = RUNS / f"actions/{role}"
    paths = {
        "gt": gt_path,
        "edge": action_root / "edge/predictions.json",
        "original": action_root / "original/predictions.json",
        "q90": action_root / "q90/predictions.json",
        "q90_manifest": action_root / "q90/q90_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    gt = DATA.read_json(gt_path)
    images = {int(row["id"]): dict(row) for row in gt["images"]}
    image_ids = sorted(images)
    if len(image_ids) != int(PROTOCOL["roles"][role]):
        raise RuntimeError(f"common role population changed: {role}")
    predictions = {
        action: _group(paths[action], image_ids)
        for action in ("edge", "original", "q90")
    }
    q90 = _q90_rows(paths["q90_manifest"])
    if set(q90) != set(image_ids):
        raise RuntimeError(f"Q90 manifest population changed: {role}")
    samples = []
    for image_id in image_ids:
        image = images[image_id]
        encoded = q90[image_id]
        original_bytes = int(encoded["original_bytes"])
        q90_bytes = int(encoded["action_bytes"])
        q90_available = bool(encoded["available"]) and 0 < q90_bytes < original_bytes
        samples.append(
            {
                "image_id": image_id,
                "identity": str(image["identity"]),
                "event_id": str(image["identity"]),
                "camera": str(image["identity"]).split(":", 1)[0],
                "file_name": str(image["file_name"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "payload_bytes": {
                    "original": original_bytes,
                    "q90": q90_bytes,
                },
                "available": {"original": True, "q90": q90_available},
            }
        )
    category_ids = [int(row["id"]) for row in gt["categories"]]
    records_o = DATA.build_action_records(
        samples,
        predictions["edge"],
        category_ids=category_ids,
        action_space="O",
    )
    records_oq = DATA.build_action_records(
        samples,
        predictions["edge"],
        category_ids=category_ids,
        action_space="OQ",
    )
    if any(len(row["actions"]) != 2 for row in records_o):
        raise RuntimeError("O record contains a non-O action")
    if any(
        {str(action["action_id"]) for action in row["actions"]} - {"edge", "original", "q90"}
        for row in records_oq
    ):
        raise RuntimeError("OQ record contains an unknown action")
    DATA.atomic_json(samples_path, samples)
    DATA.atomic_pickle(o_path, records_o)
    DATA.atomic_pickle(oq_path, records_oq)
    manifest = {
        "complete": True,
        "role": role,
        "gt": str(gt_path.resolve()),
        "actions": {action: str(paths[action].resolve()) for action in ("edge", "original", "q90")},
        "q90_manifest": str(paths["q90_manifest"].resolve()),
        "source_sha256": {key: DATA.file_sha256(path) for key, path in paths.items()},
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(manifest_path, manifest)
    report = {
        "complete": True,
        "role": role,
        "images": len(samples),
        "q90_available": sum(bool(row["available"]["q90"]) for row in samples),
        "samples_sha256": DATA.file_sha256(samples_path),
        "records_O_sha256": DATA.file_sha256(o_path),
        "records_OQ_sha256": DATA.file_sha256(oq_path),
        "manifest_sha256": DATA.file_sha256(manifest_path),
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(output / "report.json", report)
    DATA.atomic_json(completion, report)
    return report


def run_all() -> dict[str, Any]:
    reports = {role: build_role(role) for role in PROTOCOL["roles"]}
    report = {
        "complete": True,
        "roles": {role: int(value["images"]) for role, value in reports.items()},
        "independent_action_spaces": ["O", "OQ"],
        "final_used_for_training_or_selection": False,
    }
    DATA.atomic_json(RUNS / "common/report.json", report)
    DATA.atomic_json(RUNS / "common/completion.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=(*PROTOCOL["roles"], "all"), default="all")
    args = parser.parse_args()
    result = run_all() if args.role == "all" else build_role(args.role)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
