#!/usr/bin/env python3
"""Run the resumable full-COCO CAVS-OQR real-system benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
BASE_ROOT = REPO / "paper/experiments/20260814_mac_remote_end_to_end_system_benchmark_v1"
DEMO = REPO / "examples/YOLO26-Edge-Cloud"
sys.path.insert(0, str(DEMO))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module("cavs_system_benchmark_base", BASE_ROOT / "scripts/benchmark.py")

from current_maur_uploader import (  # noqa: E402
    current_roi_xyxy,
    load_current_maur_policy,
    maur_fuse_boxes,
    predict_current_maur_scores,
)
from paper_budget_scheduler import schedule_fixed_byte  # noqa: E402
from protocol import bytes_to_b64, encode_jpeg  # noqa: E402
from ultralytics import YOLO  # noqa: E402


METHODS = ("edge", "cavs_oqr_10", "cavs_oqr_25", "cavs_oqr_40")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def load_latest_jsonl(path: Path, key: Callable[[Mapping[str, Any]], Hashable]) -> dict[Hashable, dict[str, Any]]:
    """Load append-only checkpoints, allowing a truncated final line after interruption."""
    if not path.is_file():
        return {}
    rows: dict[Hashable, dict[str, Any]] = {}
    size = path.stat().st_size
    with path.open(encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if handle.tell() == size:
                    break
                raise
            rows[key(row)] = row
    return rows


def compact_boxes(boxes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "xyxy": list(map(float, box["xyxy"])),
            "conf": float(box.get("conf", box.get("score", 0.0))),
            "cls": int(box["cls"]),
        }
        for box in boxes
    ]


def candidate_oqr(
    image: np.ndarray,
    image_id: int,
    policy: Mapping[str, Any],
    edge_boxes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Construct exact Current+MAUR O/Q/R scores and raw-JPEG costs."""
    started = time.perf_counter()
    height, width = image.shape[:2]
    roi_config = policy["roi"]
    roi_xyxy = current_roi_xyxy(
        width,
        height,
        edge_boxes,
        confidence=float(roi_config["confidence"]),
        max_area_ratio=float(roi_config["max_area_ratio"]),
    )
    original_bytes = len(encode_jpeg(image, 95))
    q90_bytes = len(encode_jpeg(image, 90))
    payloads = {"edge": 0, "original": original_bytes, "q90": q90_bytes}
    if roi_xyxy is not None:
        x1, y1, x2, y2 = roi_xyxy
        payloads["current@640"] = len(encode_jpeg(image[y1:y2, x1:x2], 90))
    scores = predict_current_maur_scores(
        policy,
        edge_boxes,
        width=width,
        height=height,
        roi_xyxy=roi_xyxy,
    )
    if not 0 < q90_bytes < original_bytes:
        payloads.pop("q90", None)
        scores.pop("q90", None)
    return {
        "frame_id": int(image_id),
        "roi_xyxy": roi_xyxy,
        "payload_bytes": payloads,
        "scores": {key: float(value) for key, value in scores.items() if key in payloads},
        "candidate_ms": (time.perf_counter() - started) * 1000.0,
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
    """Build the exact demo payload for one selected O/Q/R action."""
    if action not in {"original", "q90", "current@640"}:
        raise ValueError(f"not an upload action: {action}")
    is_roi = action == "current@640"
    if is_roi and roi_xyxy is None:
        raise ValueError("Current@640 requires ROI geometry")
    started = time.perf_counter()
    payload_image = image
    if is_roi:
        x1, y1, x2, y2 = map(int, roi_xyxy)
        payload_image = image[y1:y2, x1:x2]
    quality = 95 if action == "original" else 90
    jpeg = encode_jpeg(payload_image, quality)
    encode_ms = (time.perf_counter() - started) * 1000.0
    metadata = {
        "task_id": f"coco5000-{image_id}-{action}",
        "frame_id": int(image_id),
        "timestamp": time.time(),
        "payload_type": "roi" if is_roi else "keyframe",
        "reason": [f"cavs_oqr_{action}"],
        "upload_score": float(score),
        "crop_xyxy": list(map(int, roi_xyxy)) if is_roi else None,
        "orig_shape": list(image.shape[:2]),
        "edge_boxes": compact_boxes(edge_boxes),
        "target_edge_box": None,
        "payload_bytes": len(jpeg),
        "cloud_imgsz": int(cloud_imgsz),
    }
    return {"metadata": metadata, "image_b64": bytes_to_b64(jpeg)}, len(jpeg), encode_ms


def selected_request_keys(plans: Mapping[str, Mapping[int, str]]) -> set[tuple[int, str]]:
    return {
        (int(image_id), str(action))
        for plan in plans.values()
        for image_id, action in plan.items()
        if str(action) != "edge"
    }


def resolve_oqr_output(
    action: str,
    request_result: Mapping[str, Any] | None,
    edge_boxes: Sequence[Mapping[str, Any]],
    roi_xyxy: Sequence[int] | None,
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str, float]:
    """Resolve full replacement, MAUR ROI fusion, or Edge fallback."""
    if action == "edge":
        return compact_boxes(edge_boxes), "not_required", 0.0
    if request_result is None or request_result.get("status") != "ok":
        return compact_boxes(edge_boxes), "fallback", 0.0
    cloud_boxes = compact_boxes(request_result.get("cloud_boxes", []))
    if action != "current@640":
        return cloud_boxes, "ok", 0.0
    if roi_xyxy is None:
        raise RuntimeError("selected ROI action has no geometry")
    started = time.perf_counter()
    maur = policy["maur"]
    fused = maur_fuse_boxes(
        edge_boxes,
        cloud_boxes,
        roi_xyxy,
        duplicate_iou=float(maur["duplicate_iou"]),
        nms_iou=float(maur["nms_iou"]),
    )
    return compact_boxes(fused), "ok", (time.perf_counter() - started) * 1000.0


def update_status(runs: Path, **values: Any) -> None:
    atomic_json(runs / "status.json", {"updated_at": time.time(), **values})


def image_paths(args: argparse.Namespace) -> list[Path]:
    paths = sorted(Path(args.images).glob("*.jpg"))
    if args.limit:
        paths = paths[: int(args.limit)]
    expected = int(args.limit or args.expected_images)
    if len(paths) != expected:
        raise RuntimeError(f"COCO image count changed: {len(paths)} != {expected}")
    return paths


def validate_contract(args: argparse.Namespace, policy: Mapping[str, Any], *, require_cloud: bool) -> dict[str, Any] | None:
    if BASE.file_sha256(args.edge_model) != args.edge_sha256:
        raise RuntimeError("edge model SHA-256 mismatch")
    if tuple(policy["action_ids"]) != ("edge", "original", "q90", "current@640"):
        raise RuntimeError("CAVS-OQR action contract changed")
    if not require_cloud:
        return None
    cloud = BASE.health(args.cloud_url, args.cloud_timeout)
    if cloud.get("model_sha256") != args.cloud_sha256:
        raise RuntimeError("cloud model SHA-256 mismatch")
    if int(cloud.get("imgsz", -1)) != int(args.cloud_imgsz):
        raise RuntimeError("cloud imgsz mismatch")
    if float(cloud.get("conf", -1)) != float(args.confidence):
        raise RuntimeError("cloud confidence mismatch")
    return cloud


def run_candidates(args: argparse.Namespace, runs: Path, paths: Sequence[Path], policy: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    checkpoint = runs / "candidates.jsonl"
    completed = load_latest_jsonl(checkpoint, lambda row: int(row["image_id"]))
    expected_ids = {BASE.parse_image_id(path) for path in paths}
    if not set(completed) <= expected_ids:
        raise RuntimeError("candidate checkpoint contains images outside this run")
    missing = [path for path in paths if BASE.parse_image_id(path) not in completed]
    if not missing:
        return completed

    model = YOLO(args.edge_model)
    warmup = cv2.imread(str(missing[0]))
    if warmup is None:
        raise ValueError(missing[0])
    for _ in range(max(0, int(args.warmup_runs))):
        boxes, _, _ = BASE._timed_edge_predict(model, warmup, args)
        candidate_oqr(warmup, BASE.parse_image_id(missing[0]), policy, compact_boxes(boxes))

    for offset, path in enumerate(missing, start=1):
        decode_started = time.perf_counter()
        image = cv2.imread(str(path))
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        if image is None:
            raise ValueError(path)
        image_id = BASE.parse_image_id(path)
        boxes, edge_ms, post_ms = BASE._timed_edge_predict(model, image, args)
        boxes = compact_boxes(boxes)
        candidate = candidate_oqr(image, image_id, policy, boxes)
        row = {
            "image_id": image_id,
            "path": str(path),
            "shape": list(image.shape[:2]),
            "decode_ms": decode_ms,
            "edge_infer_ms": edge_ms,
            "edge_postprocess_ms": post_ms,
            "edge_boxes": boxes,
            "candidate": candidate,
        }
        append_jsonl(checkpoint, row)
        completed[image_id] = row
        done = len(completed)
        if offset == 1 or done % 25 == 0 or done == len(paths):
            update_status(runs, stage="candidates", candidates_done=done, candidates_total=len(paths))
            print(f"candidates {done}/{len(paths)} image={image_id}", flush=True)
    return completed


def build_plans(runs: Path, records: Mapping[int, Mapping[str, Any]], budgets: Sequence[float]) -> tuple[dict[str, dict[int, str]], dict[str, Any]]:
    candidates = [records[image_id]["candidate"] for image_id in sorted(records)]
    plans: dict[str, dict[int, str]] = {}
    reports: dict[str, Any] = {}
    for budget in map(float, budgets):
        label = str(int(round(budget * 100)))
        result = schedule_fixed_byte(candidates, budget)
        plans[label] = {int(image_id): str(action) for image_id, action in result["actions"].items()}
        reports[label] = {key: value for key, value in result.items() if key != "actions"}
    atomic_json(
        runs / "plans.json",
        {
            "plans": {label: {str(key): value for key, value in plan.items()} for label, plan in plans.items()},
            "reports": reports,
        },
    )
    return plans, reports


def run_requests(
    args: argparse.Namespace,
    runs: Path,
    records: Mapping[int, Mapping[str, Any]],
    plans: Mapping[str, Mapping[int, str]],
) -> dict[tuple[int, str], dict[str, Any]]:
    checkpoint = runs / "requests.jsonl"
    latest = load_latest_jsonl(checkpoint, lambda row: (int(row["image_id"]), str(row["action"])))
    required = sorted(selected_request_keys(plans))
    required_set = set(required)
    successful = {key for key, row in latest.items() if row.get("status") == "ok"}
    pending = [key for key in required if key not in successful]
    if pending:
        warm_image_id, warm_action = pending[0]
        warm_record = records[warm_image_id]
        warm_image = cv2.imread(warm_record["path"])
        if warm_image is None:
            raise ValueError(warm_record["path"])
        warm_candidate = warm_record["candidate"]
        warm_payload, _, _ = build_action_payload(
            warm_image,
            warm_image_id,
            warm_action,
            warm_record["edge_boxes"],
            warm_candidate.get("roi_xyxy"),
            float(warm_candidate["scores"][warm_action]),
            args.cloud_imgsz,
        )
        for _ in range(max(0, int(args.warmup_runs))):
            warm_response = BASE.post_instrumented(args.cloud_url, warm_payload, args.cloud_timeout)
            if warm_response.get("status") != "ok":
                raise RuntimeError(f"cloud warmup failed: {warm_response.get('error')}")
    for offset, (image_id, action) in enumerate(pending, start=1):
        record = records[image_id]
        image = cv2.imread(record["path"])
        if image is None:
            raise ValueError(record["path"])
        candidate = record["candidate"]
        payload, payload_bytes, encode_ms = build_action_payload(
            image,
            image_id,
            action,
            record["edge_boxes"],
            candidate.get("roi_xyxy"),
            float(candidate["scores"][action]),
            args.cloud_imgsz,
        )
        response: dict[str, Any] = {}
        for attempt in range(1, int(args.request_retries) + 1):
            response = BASE.post_instrumented(args.cloud_url, payload, args.cloud_timeout)
            if response.get("status") == "ok":
                break
        response.update(
            {
                "image_id": image_id,
                "action": action,
                "attempts": attempt,
                "payload_bytes": payload_bytes,
                "upload_encode_ms": encode_ms,
            }
        )
        append_jsonl(checkpoint, response)
        latest[(image_id, action)] = response
        success_count = sum(row.get("status") == "ok" for key, row in latest.items() if key in required_set)
        if offset == 1 or offset % 10 == 0 or offset == len(pending):
            update_status(
                runs,
                stage="requests",
                candidates_done=len(records),
                requests_successful=success_count,
                requests_required=len(required),
                requests_pending=max(0, len(required) - success_count),
            )
            print(
                f"requests {success_count}/{len(required)} image={image_id} action={action} "
                f"status={response.get('status')} rtt={float(response.get('roundtrip_ms', 0.0)):.1f}ms",
                flush=True,
            )
    return latest


def summarize_method(
    rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, float],
    theoretical: float | None,
) -> dict[str, Any]:
    timing_keys = (
        "decode_ms",
        "edge_infer_ms",
        "edge_postprocess_ms",
        "candidate_ms",
        "upload_encode_ms",
        "request_json_serialize_ms",
        "roundtrip_ms",
        "fusion_ms",
        "e2e_ms",
        "server_cloud_infer_ms",
        "transport_http_residual_ms",
    )
    images = max(1, len(rows))
    payload_total = sum(int(row["payload_bytes"]) for row in rows)
    result = {
        "images": len(rows),
        "uploads": sum(str(row["action"]) != "edge" for row in rows),
        "failures": sum(row["cloud_status"] == "fallback" for row in rows),
        "action_counts": dict(Counter(str(row["action"]) for row in rows)),
        "payload_bytes_total": payload_total,
        "request_body_bytes_total": sum(int(row["request_body_bytes"]) for row in rows),
        "response_body_bytes_total": sum(int(row["response_body_bytes"]) for row in rows),
        "metrics": dict(metrics),
        "timings_ms": {
            key: BASE.summarize_values([float(row.get(key, 0.0)) for row in rows])
            for key in timing_keys
        },
        "paper_payload_tx_ms_per_image_at_10mbps": payload_total * 8.0 / 10_000_000.0 * 1000.0 / images,
    }
    if theoretical is not None:
        result["theoretical_ap50_95"] = float(theoretical)
        result["delta_ap50_95_vs_theoretical"] = float(metrics["ap50_95"] - theoretical)
    return result


def write_predictions(path: Path, predictions: Sequence[Mapping[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(list(predictions), handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")


def run_evaluation(
    args: argparse.Namespace,
    runs: Path,
    records: Mapping[int, Mapping[str, Any]],
    plans: Mapping[str, Mapping[int, str]],
    plan_reports: Mapping[str, Any],
    requests: Mapping[tuple[int, str], Mapping[str, Any]],
    policy: Mapping[str, Any],
    cloud_health: Mapping[str, Any],
    theoretical: Mapping[str, float],
) -> dict[str, Any]:
    category_ids = list(map(int, policy["category_ids"]))
    image_ids = sorted(records)
    methods: dict[str, Any] = {}
    method_plans: dict[str, Mapping[int, str]] = {"edge": {image_id: "edge" for image_id in image_ids}}
    method_plans.update({f"cavs_oqr_{label}": plan for label, plan in plans.items()})

    for method in METHODS:
        plan = method_plans[method]
        predictions: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for image_id in image_ids:
            record = records[image_id]
            action = str(plan[image_id])
            request_result = requests.get((image_id, action)) if action != "edge" else None
            boxes, cloud_status, fusion_ms = resolve_oqr_output(
                action,
                request_result,
                record["edge_boxes"],
                record["candidate"].get("roi_xyxy"),
                policy,
            )
            predictions.extend(BASE.coco_predictions(image_id, boxes, category_ids))
            candidate_ms = float(record["candidate"]["candidate_ms"]) if method != "edge" else 0.0
            request_json_ms = float(request_result.get("request_json_serialize_ms", 0.0)) if request_result else 0.0
            roundtrip_ms = float(request_result.get("roundtrip_ms", 0.0)) if request_result else 0.0
            encode_ms = float(request_result.get("upload_encode_ms", 0.0)) if request_result else 0.0
            e2e_ms = (
                float(record["decode_ms"])
                + float(record["edge_infer_ms"])
                + float(record["edge_postprocess_ms"])
                + candidate_ms
                + encode_ms
                + request_json_ms
                + roundtrip_ms
                + fusion_ms
            )
            server = request_result.get("server_timings_ms", {}) if request_result else {}
            rows.append(
                {
                    "image_id": image_id,
                    "action": action,
                    "cloud_status": cloud_status,
                    "decode_ms": float(record["decode_ms"]),
                    "edge_infer_ms": float(record["edge_infer_ms"]),
                    "edge_postprocess_ms": float(record["edge_postprocess_ms"]),
                    "candidate_ms": candidate_ms,
                    "upload_encode_ms": encode_ms,
                    "request_json_serialize_ms": request_json_ms,
                    "roundtrip_ms": roundtrip_ms,
                    "fusion_ms": fusion_ms,
                    "e2e_ms": e2e_ms,
                    "server_cloud_infer_ms": float(server.get("cloud_infer", 0.0)),
                    "transport_http_residual_ms": float(request_result.get("transport_http_residual_ms", 0.0)) if request_result else 0.0,
                    "payload_bytes": int(request_result.get("payload_bytes", 0)) if request_result else 0,
                    "request_body_bytes": int(request_result.get("request_body_bytes", 0)) if request_result else 0,
                    "response_body_bytes": int(request_result.get("response_body_bytes", 0)) if request_result else 0,
                }
            )
        metrics = BASE.evaluate_coco(Path(args.annotations), image_ids, predictions)
        budget = method.rsplit("_", 1)[-1] if method != "edge" else None
        use_theoretical = not args.limit and len(image_ids) == int(args.expected_images)
        theoretical_value = (
            theoretical.get(str(float(budget) / 100.0)) if budget and use_theoretical else None
        )
        methods[method] = summarize_method(rows, metrics, theoretical_value)
        write_predictions(runs / f"predictions_{method}.json.gz", predictions)
        print(f"evaluated {method}: AP50-95={metrics['ap50_95']:.6f}", flush=True)
        del predictions, rows
        gc.collect()

    summary = {
        "complete": True,
        "benchmark_kind": "desktop_edge_proxy_full_coco5000_cavs_oqr",
        "images": len(image_ids),
        "edge_model_sha256": BASE.file_sha256(args.edge_model),
        "policy_sha256": BASE.file_sha256(args.policy),
        "cloud_health": dict(cloud_health),
        "plans": dict(plan_reports),
        "required_unique_requests": len(selected_request_keys(plans)),
        "successful_unique_requests": sum(
            requests.get(key, {}).get("status") == "ok" for key in selected_request_keys(plans)
        ),
        "methods": methods,
    }
    atomic_json(runs / "summary.json", summary)
    (runs / "RESULTS_ZH.md").write_text(render_report(summary), encoding="utf-8")
    update_status(runs, stage="complete", complete=True, images=len(image_ids))
    return summary


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# COCO val2017 {summary['images']}张 CAVS-OQR 真实端云系统结果",
        "",
        "> Mac 是端侧代理；远程共享 RTX 4090 是云端。本结果不是 RK3588 延迟。",
        "",
        f"- 图像：{summary['images']} 张。",
        f"- 唯一真实云请求：{summary['successful_unique_requests']}/{summary['required_unique_requests']} 成功。",
        f"- Cloud：`{summary['cloud_health'].get('model')}` / `{summary['cloud_health'].get('model_sha256')}`。",
        "- Original 使用 demo 的 Q95 重编码；ROI 使用 Current@640 Q90 + MAUR。",
        "",
        "| 方法 | AP50-95 | 理论点 | 差值 | AP50 | AP75 | 上传率 | JPEG KB/图 | HTTP Body KB/图 | E2E mean/P95 ms | 失败 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
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
            f"{theoretical_text} | {delta_text} | {metric['ap50']:.6f} | {metric['ap75']:.6f} | "
            f"{item['uploads'] / images:.2%} | {item['payload_bytes_total'] / images / 1024:.2f} | "
            f"{item['request_body_bytes_total'] / images / 1024:.2f} | {e2e['mean']:.2f}/{e2e['p95']:.2f} | "
            f"{item['failures']} |"
        )
    lines.extend(
        [
            "",
            "## 容量与动作",
            "",
            "| 预算 | JPEG容量利用率 | 动作分布 |",
            "|---:|---:|---|",
        ]
    )
    for label in ("10", "25", "40"):
        plan = summary["plans"][label]
        lines.append(
            f"| {label}% | {plan['capacity_utilization']:.4%} | `{plan['action_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 理论点来自缓存式论文评测；本轮经过真实 Q95/Q90/ROI 编码、HTTP、Cloud 和 MAUR。",
            "- 论文容量统计 JPEG payload；真实 HTTP Body 还包括 Base64 与 metadata。",
            "- E2E 是真实组件的串行阻塞求和，不包含全局两遍式窗口等待。",
            "- 共享 GPU 负载会显著放大 Cloud/RTT，但没有超时或失败时不会直接改变静态图像 AP。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    protocol = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("candidates", "requests", "evaluate", "all"), default="all")
    parser.add_argument("--images", default=protocol["dataset"]["images"])
    parser.add_argument("--annotations", default=protocol["dataset"]["annotations"])
    parser.add_argument("--expected-images", type=int, default=protocol["dataset"]["expected_images"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--edge-model", default=protocol["models"]["edge"])
    parser.add_argument("--edge-sha256", default=protocol["models"]["edge_sha256"])
    parser.add_argument("--cloud-sha256", default=protocol["models"]["cloud_sha256"])
    parser.add_argument("--policy", default=protocol["models"]["policy"])
    parser.add_argument("--edge-device", default=protocol["runtime"]["edge_device"])
    parser.add_argument("--edge-imgsz", type=int, default=protocol["runtime"]["edge_imgsz"])
    parser.add_argument("--cloud-imgsz", type=int, default=protocol["runtime"]["cloud_imgsz"])
    parser.add_argument("--confidence", type=float, default=protocol["runtime"]["confidence"])
    parser.add_argument("--cloud-url", default=protocol["runtime"]["cloud_url"])
    parser.add_argument("--cloud-timeout", type=float, default=protocol["runtime"]["cloud_timeout_seconds"])
    parser.add_argument("--warmup-runs", type=int, default=protocol["runtime"]["warmup_runs"])
    parser.add_argument("--request-retries", type=int, default=protocol["runtime"]["request_retries"])
    parser.add_argument("--output", default=str(ROOT / "runs/formal"))
    parser.set_defaults(budgets=protocol["budgets"], theoretical=protocol["theoretical_current_maur"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = Path(args.output).resolve()
    runs.mkdir(parents=True, exist_ok=True)
    paths = image_paths(args)
    policy = load_current_maur_policy(args.policy)
    require_cloud = args.stage in {"requests", "all"}
    cloud_health = validate_contract(args, policy, require_cloud=require_cloud)
    records = load_latest_jsonl(runs / "candidates.jsonl", lambda row: int(row["image_id"]))
    if args.stage in {"candidates", "all"}:
        records = run_candidates(args, runs, paths, policy)
    if len(records) != len(paths):
        raise RuntimeError(f"candidate stage incomplete: {len(records)}/{len(paths)}")
    plans, plan_reports = build_plans(runs, records, args.budgets)
    if args.stage == "candidates":
        update_status(runs, stage="candidates_complete", candidates_done=len(records), candidates_total=len(paths))
        return
    requests = load_latest_jsonl(
        runs / "requests.jsonl", lambda row: (int(row["image_id"]), str(row["action"]))
    )
    if args.stage in {"requests", "all"}:
        requests = run_requests(args, runs, records, plans)
    if args.stage == "requests":
        return
    if cloud_health is None:
        cloud_health = BASE.health(args.cloud_url, args.cloud_timeout)
    run_evaluation(
        args,
        runs,
        records,
        plans,
        plan_reports,
        requests,
        policy,
        cloud_health,
        args.theoretical,
    )


if __name__ == "__main__":
    main()
