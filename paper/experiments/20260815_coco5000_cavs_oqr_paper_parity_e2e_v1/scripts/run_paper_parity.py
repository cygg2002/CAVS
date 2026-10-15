#!/usr/bin/env python3
"""Run the corrected, resumable COCO CAVS-OQR paper-parity deployment."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
LEGACY_ROOT = REPO / "paper/experiments/20260814_coco5000_cavs_oqr_end_to_end_v1"
DEMO = REPO / "examples/YOLO26-Edge-Cloud"
sys.path[:0] = [str(SCRIPT.parent), str(DEMO), str(REPO)]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = load_module("cavs_oqr_legacy_runner", LEGACY_ROOT / "scripts/full_oqr.py")

from current_maur_uploader import current_roi_xyxy, load_current_maur_policy, predict_current_maur_scores  # noqa: E402
from inference_protocol import (  # noqa: E402
    ScaleDownDetectionPredictor,
    native_jpeg,
    paper_q90_jpeg,
    roi_record,
    validate_cloud_health,
)
from protocol import bytes_to_b64  # noqa: E402
from ultralytics import YOLO  # noqa: E402


PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
IMAGE_PATHS: dict[int, Path] = {}
PAPER_SAMPLES: dict[int, dict[str, Any]] = {}


def source_path(image_id: int) -> Path:
    try:
        return IMAGE_PATHS[int(image_id)]
    except KeyError as exc:
        raise KeyError(f"missing source path for image {image_id}") from exc


def paper_sample(image_id: int) -> dict[str, Any]:
    try:
        return PAPER_SAMPLES[int(image_id)]
    except KeyError as exc:
        raise KeyError(f"missing frozen paper sample for image {image_id}") from exc


def timed_edge_predict(model: YOLO, image: np.ndarray, args: argparse.Namespace) -> tuple[list[dict[str, Any]], float, float]:
    started = time.perf_counter()
    result = model.predict(
        source=image,
        predictor=ScaleDownDetectionPredictor,
        imgsz=int(args.edge_imgsz),
        conf=float(args.confidence),
        iou=float(args.iou),
        max_det=int(args.max_det),
        device=args.edge_device or None,
        rect=True,
        end2end=False,
        half=bool(args.edge_half),
        verbose=False,
    )[0]
    inference_ms = (time.perf_counter() - started) * 1000.0
    post_started = time.perf_counter()
    boxes = LEGACY.BASE.boxes_from_result(result)
    return boxes, inference_ms, (time.perf_counter() - post_started) * 1000.0


def candidate_oqr(
    image: np.ndarray,
    image_id: int,
    policy: Mapping[str, Any],
    edge_boxes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    started = time.perf_counter()
    path = source_path(image_id)
    sample = paper_sample(image_id)
    height, width = image.shape[:2]
    roi_cfg = policy["roi"]
    roi_xyxy = current_roi_xyxy(
        width,
        height,
        edge_boxes,
        confidence=float(roi_cfg["confidence"]),
        max_area_ratio=float(roi_cfg["max_area_ratio"]),
    )
    original = native_jpeg(path)
    q90 = paper_q90_jpeg(path, long_side=int(PROTOCOL["codec"]["long_side"]))
    expected = sample["payload_bytes"]
    if len(original) != int(expected["original"]):
        raise RuntimeError(f"native Original bytes changed: {image_id} {len(original)} != {expected['original']}")
    if len(q90) != int(expected["q90"]):
        raise RuntimeError(f"paper Q90 bytes changed: {image_id} {len(q90)} != {expected['q90']}")
    payloads = {"edge": 0, "original": len(original), "q90": len(q90)}
    frozen_roi = roi_record(sample)
    roi_geometry_match = bool(frozen_roi and roi_xyxy == list(map(int, frozen_roi["xyxy"])))
    if roi_xyxy is not None:
        roi = paper_q90_jpeg(path, crop_xyxy=roi_xyxy, long_side=int(PROTOCOL["codec"]["long_side"]))
        payloads["current@640"] = len(roi)
        if roi_geometry_match and len(roi) != int(expected["current@640"]):
            raise RuntimeError(f"paper ROI bytes changed: {image_id} {len(roi)} != {expected['current@640']}")
    scores = predict_current_maur_scores(
        policy,
        edge_boxes,
        width=width,
        height=height,
        roi_xyxy=roi_xyxy,
    )
    if not 0 < len(q90) < len(original):
        payloads.pop("q90", None)
        scores.pop("q90", None)
    return {
        "frame_id": int(image_id),
        "roi_xyxy": roi_xyxy,
        "payload_bytes": payloads,
        "scores": {key: float(value) for key, value in scores.items() if key in payloads},
        "candidate_ms": (time.perf_counter() - started) * 1000.0,
        "paper_parity": {
            "original_bytes_match": True,
            "q90_bytes_match": True,
            "roi_available_live": roi_xyxy is not None,
            "roi_available_paper": frozen_roi is not None,
            "roi_geometry_match": roi_geometry_match,
            "roi_bytes_match": bool(
                roi_geometry_match
                and payloads.get("current@640") == int(expected.get("current@640", -1))
            ),
        },
    }


def build_action_payload(
    image: np.ndarray,
    image_id: int,
    action: str,
    edge_boxes: Sequence[Mapping[str, Any]],
    roi_xyxy: Sequence[int] | None,
    score: float,
    cloud_imgsz: int,
) -> tuple[dict[str, Any], int, float]:
    if action not in {"original", "q90", "current@640"}:
        raise ValueError(f"not an upload action: {action}")
    if action == "current@640" and roi_xyxy is None:
        raise ValueError("Current@640 requires ROI geometry")
    started = time.perf_counter()
    path = source_path(image_id)
    if action == "original":
        jpeg = native_jpeg(path)
    else:
        jpeg = paper_q90_jpeg(
            path,
            crop_xyxy=roi_xyxy if action == "current@640" else None,
            long_side=int(PROTOCOL["codec"]["long_side"]),
        )
    encode_ms = (time.perf_counter() - started) * 1000.0
    metadata = {
        "task_id": f"coco5000-paper-parity-{image_id}-{action}",
        "frame_id": int(image_id),
        "timestamp": time.time(),
        "action_id": action,
        "payload_type": "roi" if action == "current@640" else "keyframe",
        "upload_score": float(score),
        "crop_xyxy": list(map(int, roi_xyxy)) if action == "current@640" else None,
        "orig_shape": list(image.shape[:2]),
        "edge_boxes": LEGACY.compact_boxes(edge_boxes),
        "payload_bytes": len(jpeg),
        "cloud_imgsz": int(cloud_imgsz),
    }
    return {"metadata": metadata, "image_b64": bytes_to_b64(jpeg)}, len(jpeg), encode_ms


def validate_contract(args: argparse.Namespace, policy: Mapping[str, Any], *, require_cloud: bool) -> dict[str, Any] | None:
    if LEGACY.BASE.file_sha256(args.edge_model) != args.edge_sha256:
        raise RuntimeError("edge model SHA-256 mismatch")
    if tuple(policy["action_ids"]) != ("edge", "original", "q90", "current@640"):
        raise RuntimeError("CAVS-OQR action contract changed")
    if not require_cloud:
        return None
    cloud = LEGACY.BASE.health(args.cloud_url, args.cloud_timeout)
    validate_cloud_health(
        cloud,
        cloud_sha256=args.cloud_sha256,
        imgsz=args.cloud_imgsz,
        conf=args.confidence,
        iou=args.iou,
        max_det=args.max_det,
    )
    return cloud


def parity_audit(records: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    rows = [record["candidate"]["paper_parity"] for record in records.values()]
    return {
        "images": len(rows),
        "native_original_matches": sum(bool(row["original_bytes_match"]) for row in rows),
        "q90_matches": sum(bool(row["q90_bytes_match"]) for row in rows),
        "paper_roi_available": sum(bool(row["roi_available_paper"]) for row in rows),
        "live_roi_available": sum(bool(row["roi_available_live"]) for row in rows),
        "roi_geometry_matches": sum(bool(row["roi_geometry_match"]) for row in rows),
        "roi_byte_matches": sum(bool(row["roi_bytes_match"]) for row in rows),
        "native_original_total_bytes": sum(int(record["candidate"]["payload_bytes"]["original"]) for record in records.values()),
    }


def frozen_plans(path: Path, image_ids: Sequence[int]) -> dict[str, dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(map(int, image_ids))
    output = {}
    for budget in PROTOCOL["budgets"]:
        label = str(int(float(budget) * 100))
        selected = payload[f"final|CAVS-Current-MAUR|fixed_byte|{float(budget):g}"]
        plan = {int(image_id): "edge" for image_id in image_ids}
        plan.update(
            {
                int(image_id): str(action)
                for image_id, action in selected.items()
                if int(image_id) in allowed
            }
        )
        output[label] = plan
    return output


def selection_audit(plans: Mapping[str, Mapping[int, str]], frozen: Mapping[str, Mapping[int, str]]) -> dict[str, Any]:
    report = {}
    for label, plan in plans.items():
        reference = frozen[label]
        ids = sorted(reference)
        matches = sum(str(plan.get(image_id, "edge")) == str(reference[image_id]) for image_id in ids)
        report[label] = {
            "images": len(ids),
            "exact_action_matches": matches,
            "action_agreement": matches / max(1, len(ids)),
            "live_counts": dict(Counter(str(plan.get(image_id, "edge")) for image_id in ids)),
            "paper_counts": dict(Counter(str(reference[image_id]) for image_id in ids)),
        }
    return report


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# COCO val2017 {summary['images']}张 CAVS-OQR 论文协议对齐真实端云结果",
        "",
        "> Mac 是端侧代理；远程 GPU 是云端。本结果不是 RK3588 延迟。",
        "",
        "- Original：原始 COCO JPEG，不重编码。",
        "- Q90/ROI：Pillow quality=90, optimize=True, subsampling=2。",
        "- Edge/Original/Q90：end2end=False、scaleup=False、multi_label=True；ROI 保留论文 FP16 predict 语义。",
        "",
        "| 方法 | AP50-95 | 论文点 | 差值 | AP50 | AP75 | 上传率 | JPEG KB/图 | E2E mean/P95 ms | 失败 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in LEGACY.METHODS:
        item = summary["methods"][method]
        images = max(1, int(item["images"]))
        metric = item["metrics"]
        theoretical = item.get("theoretical_ap50_95")
        delta = item.get("delta_ap50_95_vs_theoretical")
        theoretical_text = f"{float(theoretical):.6f}" if theoretical is not None else "-"
        delta_text = f"{float(delta):+.6f}" if delta is not None else "-"
        e2e = item["timings_ms"]["e2e_ms"]
        lines.append(
            f"| {method} | {metric['ap50_95']:.6f} | "
            f"{theoretical_text} | {delta_text} | "
            f"{metric['ap50']:.6f} | {metric['ap75']:.6f} | "
            f"{item['uploads'] / images:.2%} | {item['payload_bytes_total'] / images / 1024:.2f} | "
            f"{e2e['mean']:.2f}/{e2e['p95']:.2f} | {item['failures']} |"
        )
    lines.extend(["", "## 协议核对", "", "```json", json.dumps(summary.get("protocol_parity", {}), indent=2, ensure_ascii=False), "```"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    runtime = PROTOCOL["runtime"]
    dataset = PROTOCOL["dataset"]
    models = PROTOCOL["models"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("candidates", "requests", "evaluate", "all"), default="all")
    parser.add_argument("--images", default=dataset["images"])
    parser.add_argument("--annotations", default=dataset["annotations"])
    parser.add_argument("--expected-images", type=int, default=dataset["expected_images"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--edge-model", default=models["edge"])
    parser.add_argument("--edge-sha256", default=models["edge_sha256"])
    parser.add_argument("--cloud-sha256", default=models["cloud_sha256"])
    parser.add_argument("--policy", default=models["policy"])
    parser.add_argument("--edge-device", default=runtime["edge_device"])
    parser.add_argument("--edge-imgsz", type=int, default=runtime["edge_imgsz"])
    parser.add_argument("--edge-half", action=argparse.BooleanOptionalAction, default=runtime["edge_half"])
    parser.add_argument("--cloud-imgsz", type=int, default=runtime["cloud_imgsz"])
    parser.add_argument("--confidence", type=float, default=runtime["confidence"])
    parser.add_argument("--iou", type=float, default=runtime["iou"])
    parser.add_argument("--max-det", type=int, default=runtime["max_det"])
    parser.add_argument("--cloud-url", default=runtime["cloud_url"])
    parser.add_argument("--cloud-timeout", type=float, default=runtime["cloud_timeout_seconds"])
    parser.add_argument("--warmup-runs", type=int, default=runtime["warmup_runs"])
    parser.add_argument("--request-retries", type=int, default=runtime["request_retries"])
    parser.add_argument("--paper-samples", default=str(ROOT / PROTOCOL["paper_artifacts"]["samples"]))
    parser.add_argument("--paper-selections", default=str(ROOT / PROTOCOL["paper_artifacts"]["selections"]))
    parser.add_argument("--output", default=str(ROOT / "runs/formal"))
    parser.set_defaults(budgets=PROTOCOL["budgets"], theoretical=PROTOCOL["theoretical_current_maur"])
    return parser.parse_args()


def main() -> None:
    global IMAGE_PATHS, PAPER_SAMPLES
    args = parse_args()
    runs = Path(args.output).resolve()
    runs.mkdir(parents=True, exist_ok=True)
    paths = LEGACY.image_paths(args)
    IMAGE_PATHS = {LEGACY.BASE.parse_image_id(path): path for path in paths}
    all_samples = json.loads(Path(args.paper_samples).read_text(encoding="utf-8"))
    PAPER_SAMPLES = {int(sample["image_id"]): sample for sample in all_samples if int(sample["image_id"]) in IMAGE_PATHS}
    if len(PAPER_SAMPLES) != len(paths):
        raise RuntimeError(f"paper sample population mismatch: {len(PAPER_SAMPLES)} != {len(paths)}")
    policy = load_current_maur_policy(args.policy)

    LEGACY.BASE._timed_edge_predict = timed_edge_predict
    LEGACY.candidate_oqr = candidate_oqr
    LEGACY.build_action_payload = build_action_payload
    LEGACY.validate_contract = validate_contract
    LEGACY.render_report = render_report

    cloud_health = validate_contract(args, policy, require_cloud=args.stage in {"requests", "all"})
    records = LEGACY.load_latest_jsonl(runs / "candidates.jsonl", lambda row: int(row["image_id"]))
    if args.stage in {"candidates", "all"}:
        records = LEGACY.run_candidates(args, runs, paths, policy)
    if len(records) != len(paths):
        raise RuntimeError(f"candidate stage incomplete: {len(records)}/{len(paths)}")
    audit = parity_audit(records)
    expected_total = sum(int(PAPER_SAMPLES[image_id]["payload_bytes"]["original"]) for image_id in PAPER_SAMPLES)
    if audit["native_original_total_bytes"] != expected_total:
        raise RuntimeError("native capacity denominator differs from frozen paper samples")
    if not args.limit and expected_total != int(PROTOCOL["dataset"]["native_jpeg_total_bytes"]):
        raise RuntimeError("full COCO native capacity anchor changed")

    plans, plan_reports = LEGACY.build_plans(runs, records, args.budgets)
    frozen = frozen_plans(Path(args.paper_selections), sorted(records))
    selection = selection_audit(plans, frozen)
    LEGACY.atomic_json(runs / "parity_audit.json", {"costs": audit, "selections": selection})
    if args.stage == "candidates":
        LEGACY.update_status(runs, stage="candidates_complete", candidates_done=len(records), candidates_total=len(paths))
        return

    requests = LEGACY.load_latest_jsonl(runs / "requests.jsonl", lambda row: (int(row["image_id"]), str(row["action"])))
    if args.stage in {"requests", "all"}:
        requests = LEGACY.run_requests(args, runs, records, plans)
    if args.stage == "requests":
        return
    required = LEGACY.selected_request_keys(plans)
    if sum(requests.get(key, {}).get("status") == "ok" for key in required) != len(required):
        raise RuntimeError("request stage incomplete")
    summary = LEGACY.run_evaluation(
        args,
        runs,
        records,
        plans,
        plan_reports,
        requests,
        policy,
        cloud_health or validate_contract(args, policy, require_cloud=True),
        args.theoretical,
    )
    summary["protocol_parity"] = {"costs": audit, "selections": selection}
    LEGACY.atomic_json(runs / "summary.json", summary)
    (runs / "RESULTS_ZH.md").write_text(render_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
