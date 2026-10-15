#!/usr/bin/env python3
"""Run a reproducible Mac-to-remote CAVS end-to-end system benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
DEMO = REPO / "examples/YOLO26-Edge-Cloud"
sys.path.insert(0, str(DEMO))

from current_maur_uploader import load_current_maur_policy, predict_current_maur_scores  # noqa: E402
from paper_budget_scheduler import schedule_fixed_byte  # noqa: E402
from protocol import boxes_from_result, bytes_to_b64, encode_jpeg  # noqa: E402
from ultralytics import YOLO  # noqa: E402


METHODS = ("edge", "always_original", "always_q90", "cavs_oq_10", "cavs_oq_25", "cavs_oq_40")
QUALITY = {"original": 95, "q90": 90}


def file_sha256(path: str | Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_sample(paths: Iterable[Path], size: int, seed: int) -> list[Path]:
    """Select a stable, sorted subset independent of input traversal order."""
    ordered = sorted({Path(path).resolve() for path in paths})
    if not 0 < int(size) <= len(ordered):
        raise ValueError(f"sample size must be in [1, {len(ordered)}]")
    sampled = random.Random(int(seed)).sample(ordered, int(size))
    return sorted(sampled)


def percentile(values: Sequence[float], q: float) -> float:
    """Return a linearly interpolated percentile."""
    if not values:
        return 0.0
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("percentile q must be in [0, 1]")
    ordered = sorted(map(float, values))
    position = (len(ordered) - 1) * float(q)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize one latency or byte distribution."""
    rows = list(map(float, values))
    if not rows:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": len(rows),
        "mean": statistics.fmean(rows),
        "median": statistics.median(rows),
        "p95": percentile(rows, 0.95),
        "p99": percentile(rows, 0.99),
    }


def request_body(payload: Mapping[str, Any]) -> bytes:
    """Serialize exactly the JSON body sent over HTTP."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def action_plan_for_budget(candidates: Sequence[Mapping[str, Any]], budget: float) -> dict[str, Any]:
    """Run the frozen scheduler after masking the optional ROI action."""
    oq = []
    for candidate in candidates:
        oq.append(
            {
                "frame_id": int(candidate["frame_id"]),
                "scores": {
                    action: float(candidate["scores"][action])
                    for action in ("edge", "original", "q90")
                    if action in candidate["scores"]
                },
                "payload_bytes": {
                    action: int(candidate["payload_bytes"][action])
                    for action in ("edge", "original", "q90")
                    if action in candidate["payload_bytes"]
                },
            }
        )
    return schedule_fixed_byte(oq, float(budget))


def coco_predictions(
    image_id: int,
    boxes: Sequence[Mapping[str, Any]],
    category_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Convert UI-format boxes to COCO result rows."""
    rows = []
    for box in boxes:
        cls_id = int(box["cls"])
        if not 0 <= cls_id < len(category_ids):
            raise ValueError(f"class id outside category mapping: {cls_id}")
        x1, y1, x2, y2 = map(float, box["xyxy"])
        rows.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_ids[cls_id]),
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(box.get("conf", box.get("score", 0.0))),
            }
        )
    return rows


def resolve_output_boxes(
    action: str,
    request_result: Mapping[str, Any] | None,
    edge_boxes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Use a successful cloud result or preserve Edge output on any failure."""
    if action == "edge":
        return [dict(box) for box in edge_boxes], "not_required"
    if request_result is not None and request_result.get("status") == "ok":
        return [dict(box) for box in request_result.get("cloud_boxes", [])], "ok"
    return [dict(box) for box in edge_boxes], "fallback"


def evaluate_coco(
    annotation_path: Path,
    image_ids: Sequence[int],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Evaluate a fixed image subset with the standard COCO bbox metric."""
    from faster_coco_eval import COCO, COCOeval_faster

    ground_truth = COCO(annotation_path, print_function=lambda *_args, **_kwargs: None)
    detections = ground_truth.loadRes(list(predictions), min_score=0.0)
    evaluator = COCOeval_faster(
        ground_truth,
        detections,
        "bbox",
        print_function=lambda *_args, **_kwargs: None,
    )
    evaluator.params.imgIds = list(map(int, image_ids))
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "ap50_95": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]),
    }


def build_payload(
    image: np.ndarray,
    image_id: int,
    action: str,
    edge_boxes: Sequence[Mapping[str, Any]],
    cloud_imgsz: int,
) -> tuple[dict[str, Any], int, float]:
    """Encode one O/Q request and preserve the actual JPEG encode cost."""
    started = time.perf_counter()
    jpeg = encode_jpeg(image, QUALITY[action])
    encode_ms = (time.perf_counter() - started) * 1000.0
    metadata = {
        "task_id": f"system-benchmark-{image_id}-{action}",
        "frame_id": int(image_id),
        "timestamp": time.time(),
        "payload_type": "keyframe",
        "reason": [f"benchmark_{action}"],
        "upload_score": 0.0,
        "crop_xyxy": None,
        "orig_shape": list(image.shape[:2]),
        "edge_boxes": list(edge_boxes),
        "target_edge_box": None,
        "payload_bytes": len(jpeg),
        "cloud_imgsz": int(cloud_imgsz),
    }
    return {"metadata": metadata, "image_b64": bytes_to_b64(jpeg)}, len(jpeg), encode_ms


def post_instrumented(url: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    """Send one exact JSON request and return raw wire/timing telemetry."""
    serialize_started = time.perf_counter()
    body = request_body(payload)
    serialize_ms = (time.perf_counter() - serialize_started) * 1000.0
    request = urlrequest.Request(
        str(url),
        data=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlrequest.urlopen(request, timeout=float(timeout)) as response:
            response_body = response.read()
            http_status = int(response.status)
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        data = json.loads(response_body.decode("utf-8"))
        status = str(data.get("status", "error"))
        error_text = None
    except (urlerror.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        response_body = b""
        http_status = 0
        data = {"status": "error", "error": str(exc)[:200]}
        status = "error"
        error_text = str(exc)[:200]
    server = data.get("server_timings_ms", {}) if isinstance(data, dict) else {}
    server_total = float(server.get("pre_send_total", data.get("cloud_latency_ms", 0.0)))
    return {
        "status": status,
        "error": error_text or data.get("error"),
        "http_status": http_status,
        "request_json_serialize_ms": serialize_ms,
        "request_body_bytes": len(body),
        "response_body_bytes": len(response_body),
        "roundtrip_ms": roundtrip_ms,
        "transport_http_residual_ms": max(0.0, roundtrip_ms - server_total),
        "server_timings_ms": server,
        "cloud_boxes": list(data.get("cloud_boxes", [])) if status == "ok" else [],
        "model": data.get("model"),
    }


def parse_image_id(path: Path) -> int:
    """Read the canonical numeric COCO image id from its filename."""
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"non-numeric COCO image filename: {path.name}") from exc


def health(url: str, timeout: float) -> dict[str, Any]:
    """Read the isolated cloud service health contract."""
    health_url = str(url).rsplit("/", 1)[0] + "/health"
    started = time.perf_counter()
    with urlrequest.urlopen(health_url, timeout=float(timeout)) as response:
        body = response.read()
    payload = json.loads(body.decode("utf-8"))
    payload["health_roundtrip_ms"] = (time.perf_counter() - started) * 1000.0
    return payload


def _timed_edge_predict(model: YOLO, image: np.ndarray, args: argparse.Namespace) -> tuple[list[dict[str, Any]], float, float]:
    started = time.perf_counter()
    kwargs = {
        "source": image,
        "imgsz": int(args.edge_imgsz),
        "conf": float(args.confidence),
        "verbose": False,
    }
    if args.edge_device:
        kwargs["device"] = args.edge_device
    result = model.predict(**kwargs)[0]
    inference_ms = (time.perf_counter() - started) * 1000.0
    post_started = time.perf_counter()
    boxes = boxes_from_result(result)
    post_ms = (time.perf_counter() - post_started) * 1000.0
    return boxes, inference_ms, post_ms


def _candidate(image: np.ndarray, image_id: int, policy: Mapping[str, Any], boxes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    original = len(encode_jpeg(image, 95))
    q90 = len(encode_jpeg(image, 90))
    scores = predict_current_maur_scores(policy, boxes, width=image.shape[1], height=image.shape[0], roi_xyxy=None)
    elapsed = (time.perf_counter() - started) * 1000.0
    payloads = {"edge": 0, "original": original}
    filtered_scores = {"edge": float(scores["edge"]), "original": float(scores["original"])}
    if 0 < q90 < original:
        payloads["q90"] = q90
        filtered_scores["q90"] = float(scores["q90"])
    return {
        "frame_id": int(image_id),
        "scores": filtered_scores,
        "payload_bytes": payloads,
        "candidate_ms": elapsed,
    }


def _method_actions(records: Sequence[Mapping[str, Any]], plans: Mapping[str, Mapping[int, str]]) -> dict[str, dict[int, str]]:
    ids = [int(record["image_id"]) for record in records]
    actions = {
        "edge": {image_id: "edge" for image_id in ids},
        "always_original": {image_id: "original" for image_id in ids},
        "always_q90": {
            image_id: ("q90" if "q90" in record["candidate"]["payload_bytes"] else "original")
            for image_id, record in zip(ids, records)
        },
    }
    actions.update({name: dict(plan) for name, plan in plans.items()})
    return actions


def _e2e_row(record: Mapping[str, Any], action: str, request_result: Mapping[str, Any] | None, cavs: bool) -> dict[str, float]:
    decode_ms = float(record["decode_ms"])
    edge_ms = float(record["edge_infer_ms"])
    post_ms = float(record["edge_postprocess_ms"])
    decision_ms = float(record["candidate"]["candidate_ms"]) if cavs else 0.0
    upload_encode_ms = float(request_result.get("upload_encode_ms", 0.0)) if request_result else 0.0
    request_json_ms = float(request_result.get("request_json_serialize_ms", 0.0)) if request_result else 0.0
    roundtrip_ms = float(request_result.get("roundtrip_ms", 0.0)) if request_result else 0.0
    fusion_ms = float(request_result.get("fusion_ms", 0.0)) if request_result else 0.0
    total = decode_ms + edge_ms + post_ms + decision_ms + upload_encode_ms + request_json_ms + roundtrip_ms + fusion_ms
    return {
        "decode_ms": decode_ms,
        "edge_infer_ms": edge_ms,
        "edge_postprocess_ms": post_ms,
        "decision_ms": decision_ms,
        "upload_encode_ms": upload_encode_ms,
        "request_json_serialize_ms": request_json_ms,
        "roundtrip_ms": roundtrip_ms,
        "fusion_ms": fusion_ms,
        "e2e_ms": total,
        "uploaded": float(action != "edge"),
    }


def _summarize_method(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    timing_keys = (
        "decode_ms",
        "edge_infer_ms",
        "edge_postprocess_ms",
        "decision_ms",
        "upload_encode_ms",
        "request_json_serialize_ms",
        "roundtrip_ms",
        "fusion_ms",
        "e2e_ms",
        "transport_http_residual_ms",
        "server_request_read_ms",
        "server_json_parse_ms",
        "server_base64_decode_ms",
        "server_jpeg_decode_ms",
        "server_cloud_infer_ms",
        "server_cloud_postprocess_ms",
        "server_response_serialize_ms",
        "server_pre_send_total_ms",
    )
    images = max(1, len(rows))
    payload_total = sum(int(row["payload_bytes"]) for row in rows)
    return {
        "images": len(rows),
        "uploads": sum(int(row["action"] != "edge") for row in rows),
        "failures": sum(int(row["cloud_status"] != "ok") for row in rows if row["action"] != "edge"),
        "action_counts": dict(Counter(str(row["action"]) for row in rows)),
        "payload_bytes_total": payload_total,
        "request_body_bytes_total": sum(int(row["request_body_bytes"]) for row in rows),
        "response_body_bytes_total": sum(int(row["response_body_bytes"]) for row in rows),
        "paper_payload_tx_ms_per_image_at_10mbps": payload_total * 8.0 / 10_000_000.0 * 1000.0 / images,
        "timings_ms": {
            key: summarize_values([float(row.get(key, 0.0)) for row in rows])
            for key in timing_keys
        },
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Mac 端代理到远程云端的完整系统测试",
        "",
        "> 这是桌面端代理实测，不代表 Orange Pi 5 Ultra / RK3588 的端侧延迟。",
        "",
        "## 运行契约",
        "",
        f"- 样本：COCO val2017 确定性抽样 {summary['sample_size']} 张。",
        f"- Edge：`{summary['edge_model_sha256']}`。",
        f"- Cloud：`{summary['cloud_health'].get('model')}` / `{summary['cloud_health'].get('model_sha256')}`。",
        f"- 云端 GPU 状态：`{summary['cloud_health'].get('gpu_name', 'unknown')}`；本轮存在共享负载。",
        "- 动作：Edge / Original(Q95) / Q90；CAVS 使用全局 Fixed-byte 10%/25%/40%。",
        "- CAVS-OQ 为当前已导出的 Current+MAUR OQR 三种子部署包在推理时屏蔽 ROI；它不是另行导出的 dedicated OQ checkpoint。",
        "",
        "## 端到端结果",
        "",
        "| 方法 | AP50-95 | AP50 | AP75 | 上传率 | JPEG KB/图 | HTTP请求KB/图 | 阻塞E2E mean/P95 ms | RTT mean/P95 ms | 失败 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = summary["methods"][method]
        images = max(1, int(item["images"]))
        ap = item["metrics"]
        e2e = item["timings_ms"]["e2e_ms"]
        rtt = item["timings_ms"]["roundtrip_ms"]
        lines.append(
            f"| {method} | {ap['ap50_95']:.6f} | {ap['ap50']:.6f} | {ap['ap75']:.6f} | "
            f"{item['uploads'] / images:.1%} | {item['payload_bytes_total'] / images / 1024:.2f} | "
            f"{item['request_body_bytes_total'] / images / 1024:.2f} | {e2e['mean']:.2f}/{e2e['p95']:.2f} | "
            f"{rtt['mean']:.2f}/{rtt['p95']:.2f} | {item['failures']} |"
        )
    lines.extend(
        [
            "",
            "## 延迟分解",
            "",
            "| 方法 | 论文JPEG@10Mbps ms/图 | 服务端读请求 | Base64+JPEG解码 | Cloud infer | 传输/HTTP残差 | 响应KB/图 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        item = summary["methods"][method]
        timing = item["timings_ms"]
        timing_mean = lambda key: float(timing.get(key, {}).get("mean", 0.0))
        decode = timing_mean("server_base64_decode_ms") + timing_mean("server_jpeg_decode_ms")
        payload_tx = float(
            item.get(
                "paper_payload_tx_ms_per_image_at_10mbps",
                item["payload_bytes_total"] * 8.0 / 10_000_000.0 * 1000.0 / max(1, item["images"]),
            )
        )
        lines.append(
            f"| {method} | {payload_tx:.2f} | "
            f"{timing_mean('server_request_read_ms'):.2f} | {decode:.2f} | "
            f"{timing_mean('server_cloud_infer_ms'):.2f} | "
            f"{timing_mean('transport_http_residual_ms'):.2f} | "
            f"{item['response_body_bytes_total'] / max(1, item['images']) / 1024:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 理论与真实系统的区别",
            "",
            "1. 论文 Fixed-byte 约束统计原始 JPEG payload；真实 JSON+Base64 请求还包含约 4/3 Base64 膨胀和 metadata。表中的 HTTP 请求字节是实际发送 body，不能与论文 KB/图混用。",
            "2. 离线 AP 评测默认已有 Edge/Cloud 缓存；真实系统额外支付图像读取、端侧推理、两种候选 JPEG 编码、CAVS 打分、HTTP、云端排队/解码/推理、响应下载与融合。",
            "3. 表中阻塞 E2E 是逐项真实计时的串行求和，不包含全局 Fixed-byte 第一遍收集完整窗口后再执行第二遍所产生的窗口等待时间。",
            "4. 本轮模型、JPEG 质量、Cloud imgsz/conf 和动作调度与 demo 冻结部署合同一致；但它是100图子集与 OQR 部署包的 OQ-mask，不是5000图 full-val 论文点，AP不能直接横向相减。",
            "5. 当前远程 GPU 与论文 bootstrap/其他任务共享，RTT 和 Cloud 延迟是有竞争负载的观察值。空载结果和 RK3588 端侧结果仍需单独测量。",
            "",
            "## Fixed-byte 审计",
            "",
            "| 预算 | JPEG容量利用率 | HTTP body / JPEG payload | 动作分布 |",
            "|---:|---:|---:|---|",
        ]
    )
    for budget in ("10", "25", "40"):
        plan = summary["plans"][f"cavs_oq_{budget}"]
        method = summary["methods"][f"cavs_oq_{budget}"]
        payload = max(1, int(method["payload_bytes_total"]))
        lines.append(
            f"| {budget}% | {plan['capacity_utilization']:.4%} | "
            f"{method['request_body_bytes_total'] / payload:.3f}x | `{method['action_counts']}` |"
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the frozen benchmark and write all artifacts."""
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = deterministic_sample(Path(args.images).glob("*.jpg"), args.sample_size, args.sample_seed)
    annotation_path = Path(args.annotations).resolve()
    edge_path = Path(args.edge_model).resolve()
    policy_path = Path(args.policy).resolve()
    if file_sha256(edge_path) != args.edge_sha256:
        raise RuntimeError("edge model SHA-256 mismatch")
    policy = load_current_maur_policy(policy_path)
    cloud_health = health(args.cloud_url, args.cloud_timeout)
    if cloud_health.get("model_sha256") != args.cloud_sha256:
        raise RuntimeError("cloud model SHA-256 mismatch")
    if int(cloud_health.get("imgsz", -1)) != int(args.cloud_imgsz):
        raise RuntimeError("cloud image size mismatch")
    if float(cloud_health.get("conf", -1.0)) != float(args.confidence):
        raise RuntimeError("cloud confidence mismatch")

    edge_model = YOLO(edge_path)
    first = cv2.imread(str(images[0]))
    if first is None:
        raise ValueError(f"failed to decode warmup image: {images[0]}")
    for _ in range(max(0, int(args.warmup_runs))):
        boxes, _, _ = _timed_edge_predict(edge_model, first, args)
        predict_current_maur_scores(policy, boxes, width=first.shape[1], height=first.shape[0], roi_xyxy=None)
        payload, _, _ = build_payload(first, parse_image_id(images[0]), "q90", boxes, args.cloud_imgsz)
        response = post_instrumented(args.cloud_url, payload, args.cloud_timeout)
        if response["status"] != "ok":
            raise RuntimeError(f"cloud warmup failed: {response.get('error')}")

    records = []
    for index, path in enumerate(images, start=1):
        decode_started = time.perf_counter()
        image = cv2.imread(str(path))
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        if image is None:
            raise ValueError(f"failed to decode image: {path}")
        image_id = parse_image_id(path)
        boxes, edge_ms, post_ms = _timed_edge_predict(edge_model, image, args)
        candidate = _candidate(image, image_id, policy, boxes)
        records.append(
            {
                "image_id": image_id,
                "path": str(path),
                "decode_ms": decode_ms,
                "edge_infer_ms": edge_ms,
                "edge_postprocess_ms": post_ms,
                "edge_boxes": boxes,
                "candidate": candidate,
            }
        )
        print(f"candidate {index}/{len(images)} image={image_id}", flush=True)

    candidates = [record["candidate"] for record in records]
    plans = {}
    plan_reports = {}
    for budget in (0.1, 0.25, 0.4):
        name = f"cavs_oq_{int(round(budget * 100))}"
        report = action_plan_for_budget(candidates, budget)
        plans[name] = {int(key): str(value) for key, value in report["actions"].items()}
        plan_reports[name] = {key: value for key, value in report.items() if key != "actions"}
    actions_by_method = _method_actions(records, plans)

    required = {(int(record["image_id"]), action) for record in records for action in ("original", "q90")}
    requests: dict[tuple[int, str], dict[str, Any]] = {}
    record_by_id = {int(record["image_id"]): record for record in records}
    for index, (image_id, action) in enumerate(sorted(required), start=1):
        record = record_by_id[image_id]
        image = cv2.imread(record["path"])
        payload, payload_bytes, encode_ms = build_payload(
            image, image_id, action, record["edge_boxes"], args.cloud_imgsz
        )
        response = post_instrumented(args.cloud_url, payload, args.cloud_timeout)
        response["payload_bytes"] = payload_bytes
        response["upload_encode_ms"] = encode_ms
        response["fusion_ms"] = 0.0
        requests[(image_id, action)] = response
        print(
            f"request {index}/{len(required)} image={image_id} action={action} "
            f"status={response['status']} rtt={response['roundtrip_ms']:.1f}ms",
            flush=True,
        )

    category_ids = list(map(int, policy["category_ids"]))
    method_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    predictions: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for method in METHODS:
        cavs = method.startswith("cavs_")
        for record in records:
            image_id = int(record["image_id"])
            action = actions_by_method[method][image_id]
            request_result = requests.get((image_id, action)) if action != "edge" else None
            boxes, cloud_status = resolve_output_boxes(action, request_result, record["edge_boxes"])
            timing = _e2e_row(record, action, request_result, cavs)
            payload_bytes = int(request_result.get("payload_bytes", 0)) if request_result else 0
            row = {
                "method": method,
                "image_id": image_id,
                "action": action,
                "cloud_status": cloud_status,
                "payload_bytes": payload_bytes,
                "request_body_bytes": int(request_result.get("request_body_bytes", 0)) if request_result else 0,
                "response_body_bytes": int(request_result.get("response_body_bytes", 0)) if request_result else 0,
                **timing,
            }
            if request_result:
                row["transport_http_residual_ms"] = float(request_result["transport_http_residual_ms"])
                for key, value in request_result.get("server_timings_ms", {}).items():
                    row[f"server_{key}_ms"] = float(value)
            method_rows[method].append(row)
            predictions[method].extend(coco_predictions(image_id, boxes, category_ids))

    image_ids = [int(record["image_id"]) for record in records]
    summaries = {}
    for method in METHODS:
        item = _summarize_method(method_rows[method])
        item["metrics"] = evaluate_coco(annotation_path, image_ids, predictions[method])
        summaries[method] = item
        (output / f"predictions_{method}.json").write_text(
            json.dumps(predictions[method], ensure_ascii=False) + "\n", encoding="utf-8"
        )

    summary = {
        "complete": True,
        "benchmark_kind": "desktop_edge_proxy",
        "sample_size": len(records),
        "sample_seed": int(args.sample_seed),
        "image_ids": image_ids,
        "edge_model_sha256": file_sha256(edge_path),
        "policy_sha256": file_sha256(policy_path),
        "cloud_health": cloud_health,
        "plans": plan_reports,
        "methods": summaries,
    }
    (output / "image_records.json").write_text(
        json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "request_records.json").write_text(
        json.dumps(
            [
                {"image_id": image_id, "action": action, **value}
                for (image_id, action), value in sorted(requests.items())
            ],
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "method_rows.json").write_text(
        json.dumps(method_rows, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "RESULTS_ZH.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    protocol = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=protocol["dataset"]["images"])
    parser.add_argument("--annotations", default=protocol["dataset"]["annotations"])
    parser.add_argument("--sample-size", type=int, default=protocol["dataset"]["formal_sample_size"])
    parser.add_argument("--sample-seed", type=int, default=protocol["dataset"]["sample_seed"])
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
    parser.add_argument("--output", default=str(ROOT / "runs/formal"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
