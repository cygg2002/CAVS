#!/usr/bin/env python3
"""Import and normalize one backend's board candidate bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import candidate_io
import contracts


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "configs" / "protocol.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def board_role_paths(board_root: Path, *, backend: str, role: str) -> tuple[Path, Path]:
    input_path = board_root / "deployment_aligned_inputs" / "manifests" / f"{role}.json"
    candidate_root = (
        board_root
        / "cavs_oqr_edge_cloud"
        / "runs"
        / "deployment_aligned"
        / backend
        / role
    )
    return input_path, candidate_root


def import_backend(backend: str, board_root: Path, output_root: Path) -> dict[str, Any]:
    contracts.validate_backend(backend)
    expected_model_hash = str(PROTOCOL["rknn_models"][backend]["sha256"])
    role_outputs: dict[str, Any] = {}
    for role, expected_count in contracts.ROLES.items():
        input_path, candidate_root = board_role_paths(board_root, backend=backend, role=role)
        bundle_path = candidate_root / "candidate_bundle.json"
        candidates_path = candidate_root / "candidates.jsonl"
        if not input_path.is_file() or not bundle_path.is_file() or not candidates_path.is_file():
            raise FileNotFoundError(f"incomplete board bundle: {backend}/{role}")
        input_payload = json.loads(input_path.read_text(encoding="utf-8"))
        input_records = candidate_io.validate_input_manifest(
            input_payload, expected_role=role, expected_images=expected_count
        )
        native_bytes = {
            int(row["image_id"]): int(row.get("native_jpeg_bytes", 0))
            for row in input_payload["records"]
        }
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        candidate_io.validate_candidate_bundle(
            bundle,
            backend=backend,
            model_sha256=expected_model_hash,
            role=role,
            expected_images=expected_count,
        )
        if str(bundle["input_manifest_sha256"]) != sha256(input_path):
            raise RuntimeError(f"input manifest hash mismatch: {backend}/{role}")
        if str(bundle.get("candidates_sha256")) != sha256(candidates_path):
            raise RuntimeError(f"candidate JSONL hash mismatch: {backend}/{role}")
        if bool(bundle.get("scores_exported", True)):
            raise RuntimeError(f"policy scores were exported in candidate bundle: {backend}/{role}")
        if str(bundle.get("roi_source")) != "raw_rknn_edge_boxes":
            raise RuntimeError(f"candidate ROI provenance changed: {backend}/{role}")
        records = candidate_io.load_candidate_records(
            candidates_path,
            input_manifest=input_payload,
            backend=backend,
            model_sha256=expected_model_hash,
            role=role,
        )
        normalized = []
        for image_id in sorted(records):
            row = records[image_id]
            normalized.append(
                {
                    "image_id": image_id,
                    "file_name": row["file_name"],
                    "width": row["width"],
                    "height": row["height"],
                    "edge_boxes": row["edge_boxes"],
                    "roi_xyxy": row["roi_xyxy"],
                    "native_jpeg_bytes": native_bytes[image_id],
                }
            )
        destination = output_root / backend / role
        atomic_json(destination / "edge_candidates.json", normalized)
        report = {
            "complete": True,
            "backend": backend,
            "role": role,
            "images": len(normalized),
            "model_sha256": expected_model_hash,
            "input_manifest_sha256": sha256(input_path),
            "candidate_bundle_sha256": sha256(bundle_path),
            "candidate_jsonl_sha256": sha256(candidates_path),
            "scores_imported": False,
            "roi_source": "raw_rknn_edge_boxes",
        }
        atomic_json(destination / "report.json", report)
        role_outputs[role] = report
    summary = {"complete": True, "backend": backend, "roles": role_outputs}
    atomic_json(output_root / backend / "completion.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=candidate_io.BACKENDS, required=True)
    parser.add_argument("--board-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(import_backend(args.backend, args.board_root.resolve(), args.output_root.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
