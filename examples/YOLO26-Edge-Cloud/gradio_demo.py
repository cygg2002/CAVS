# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Gradio UI for the YOLO26 edge-cloud demo.

This UI is intended for interactive demonstration rather than strict latency
benchmarking. Browser webcam capture and Gradio event dispatch add overhead
that does not exist in the OpenCV-only webcam demo.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
from current_maur_uploader import (
    MAX_MANIFEST_BYTES,
    current_roi_xyxy,
    load_current_maur_policy,
    maur_fuse_boxes,
    predict_current_maur_scores,
    select_current_maur_action,
)
from demo_system import check_cloud_health, infer_url_to_health_url, render_demo_report
from edge_client import (
    make_payload,
    mark_temporal_memory_from_response,
    post_to_cloud,
    select_cloud_imgsz,
    temporal_skip_frame_upload,
    temporal_skip_roi_upload,
    update_temporal_tracks,
)
from evaluate_edge_cloud import apply_roi_cloud, class_aware_nms, full_frame_fusion_boxes
from learned_uploader import frame_features, image_context, load_policy, policy_decision, roi_features
from paper_budget_scheduler import schedule_fixed_byte
from protocol import boxes_from_result, encode_jpeg, expand_xyxy
from upload_strategies import adaptive_roi_gain
from webcam_demo import MODE_CLOUD, MODE_COLLAB, MODE_EDGE, cloud_request, draw_boxes, filter_confidence

from ultralytics import YOLO

MODEL_CACHE: dict[tuple[str, str], YOLO] = {}
POLICY_CACHE: dict[str, dict[str, Any]] = {}
RUN_SAVE_DIR: Path | None = None
DEFAULT_CURRENT_MAUR_MANIFEST = str(
    Path(__file__).resolve().parent / "policies" / "current_maur_v2" / "manifest.json"
)
VIDEO_SCHEDULER_PAPER = "Paper Fixed-byte"
VIDEO_SCHEDULER_ONLINE = "Online per-frame"

COCO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
COCO_CLASS_TO_ID = {name: idx for idx, name in enumerate(COCO_CLASS_NAMES)}


def cloud_timeout_arg(value: str) -> float:
    """Parse a cloud timeout that also fits the presentation UI controls."""
    timeout = float(value)
    if not math.isfinite(timeout) or not 0.05 <= timeout <= 10.0:
        raise argparse.ArgumentTypeError("cloud timeout must be between 0.05 and 10 seconds")
    return timeout


def parse_args() -> argparse.Namespace:
    """Parse launch arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-model", default="yolo26n.pt", help="Local edge model path.")
    parser.add_argument("--cloud-url", default="http://127.0.0.1:8765/infer", help="Remote cloud endpoint.")
    parser.add_argument(
        "--policy-path",
        default=DEFAULT_CURRENT_MAUR_MANIFEST,
        help="Default learned policy path.",
    )
    parser.add_argument("--device", default="", help="Ultralytics device string for edge inference.")
    parser.add_argument("--server-name", default="127.0.0.1", help="Gradio bind host.")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio bind port.")
    parser.add_argument("--save-dir", default="runs/edge_cloud/gradio_demo", help="Directory for UI event logs.")
    parser.add_argument("--max-file-size", default="512mb", help="Maximum uploaded image/video size.")
    parser.add_argument("--share", action="store_true", help="Enable public share link.")
    parser.add_argument("--inbrowser", action="store_true", help="Open the UI in a browser.")
    parser.add_argument("--imgsz", type=int, default=640, help="Edge inference size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Edge confidence threshold.")
    parser.add_argument("--cloud-conf", type=float, default=0.25, help="Displayed cloud confidence threshold.")
    parser.add_argument("--cloud-timeout", type=cloud_timeout_arg, default=8.0, help="Cloud timeout in seconds.")
    parser.add_argument("--cloud-imgsz", type=int, default=640, help="Requested cloud inference size.")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="Upload JPEG quality.")
    parser.add_argument("--classes", default="", help="Comma-separated class names or ids, e.g. person,car. Empty means all.")
    parser.add_argument("--roi-expand", type=float, default=1.35, help="Base ROI expansion.")
    parser.add_argument("--learned-frame-threshold", type=float, default=0.60, help="Frame upload threshold.")
    parser.add_argument("--learned-roi-threshold", type=float, default=0.08, help="ROI upload threshold.")
    parser.add_argument("--learned-cost-weight", type=float, default=0.02, help="Upload cost penalty.")
    parser.add_argument(
        "--online-fixed-byte-budget",
        type=float,
        choices=(0.1, 0.25, 0.4),
        default=0.1,
        help="Strict online raw-JPEG byte budget for camera input.",
    )
    parser.add_argument("--disable-online-budget", action="store_true", help="Disable the strict camera byte budget.")
    parser.add_argument("--learned-max-roi-per-frame", type=int, default=1, help="Max ROI uploads.")
    parser.add_argument("--learned-cloud-imgsz-low", type=int, default=416, help="Adaptive low cloud size.")
    parser.add_argument("--learned-cloud-imgsz-high", type=int, default=512, help="Adaptive high cloud size.")
    parser.add_argument("--learned-cloud-imgsz-switch", type=float, default=0.75, help="Adaptive size switch.")
    parser.add_argument("--temporal-iou", type=float, default=0.5, help="Temporal IoU.")
    parser.add_argument("--temporal-ttl-frames", type=int, default=45, help="Track TTL.")
    parser.add_argument("--temporal-stable-hits", type=int, default=3, help="Stable-hit threshold.")
    parser.add_argument("--temporal-reupload-gap", type=int, default=15, help="Re-upload gap.")
    parser.add_argument("--temporal-conf-drop", type=float, default=0.12, help="Confidence drop threshold.")
    parser.add_argument("--temporal-frame-stable-ratio", type=float, default=0.7, help="Frame stable ratio.")
    parser.add_argument("--roi-small-gain", type=float, default=1.6, help="Adaptive ROI gain for small objects.")
    parser.add_argument("--roi-edge-gain", type=float, default=1.45, help="Adaptive ROI gain for edge objects.")
    parser.add_argument("--roi-large-gain", type=float, default=1.15, help="Adaptive ROI gain for large objects.")
    parser.add_argument("--iou-match", type=float, default=0.5, help="Edge/cloud match IoU.")
    parser.add_argument("--fused-nms-iou", type=float, default=0.7, help="Final NMS IoU.")
    return parser.parse_args()


def asset_path(name: str) -> str:
    """Return a local UI asset path."""
    return str(Path(__file__).resolve().parent / "assets" / name)


def temporal_memory_active(requested: bool, input_kind: str) -> bool:
    """Enable temporal suppression only for genuine frame sequences."""
    return bool(requested) and str(input_kind) in {"camera", "video"}


def cumulative_summary(state: dict[str, Any], last_event: dict[str, Any]) -> dict[str, Any]:
    """Build cumulative presentation metrics from one Gradio session."""
    stats = Counter(state.get("stats", {}))
    edge = list(state.get("edge_latencies", []))
    cloud = list(state.get("cloud_latencies", []))
    rtt = list(state.get("roundtrip_latencies", []))
    e2e = list(state.get("e2e_latencies", []))
    timing_records = list(state.get("timing_records", []))
    timing_keys = sorted({key for item in timing_records for key in item})
    timing_avg = {}
    timing_p95 = {}
    for key in timing_keys:
        values = sorted(float(item[key]) for item in timing_records if isinstance(item.get(key), (int, float)))
        if not values:
            continue
        timing_avg[key] = round(statistics.mean(values), 3)
        timing_p95[key] = round(values[min(len(values) - 1, round((len(values) - 1) * 0.95))], 3)
    successful_uploads = int(stats["collab_full_ok"] + stats["collab_roi_ok"] + stats["cloud_only_ok"])
    return {
        "frames": int(stats["frames"]),
        "cloud_errors": int(stats["cloud_errors"]),
        "uploads": int(stats["upload_attempts"]),
        "successful_uploads": successful_uploads,
        "full_uploads": int(stats["collab_full_ok"] + stats["cloud_only_ok"]),
        "roi_uploads": int(stats["collab_roi_ok"]),
        "payload_kb_total": round(float(state.get("payload_kb_total", 0.0)), 3),
        "temporal_frame_suppressed": int(stats["temporal_frame_suppressed"]),
        "temporal_roi_suppressed": int(stats["temporal_roi_suppressed"]),
        "edge_latency_ms_avg": round(average_ms(edge), 3),
        "cloud_latency_ms_avg": round(average_ms(cloud), 3),
        "roundtrip_ms_avg": round(average_ms(rtt), 3),
        "e2e_ms_avg": round(average_ms(e2e), 3),
        "fps_avg": round(1000.0 / average_ms(e2e), 3) if average_ms(e2e) > 1e-6 else 0.0,
        "timing_avg_ms": timing_avg,
        "timing_p95_ms": timing_p95,
        "last_event": dict(last_event),
    }


def save_event(result: dict[str, Any], state: dict[str, Any]) -> None:
    """Persist lightweight UI telemetry and a human-readable report."""
    if RUN_SAVE_DIR is None:
        return
    RUN_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    event_path = RUN_SAVE_DIR / "events.jsonl"
    summary_path = RUN_SAVE_DIR / "summary.json"
    report_path = RUN_SAVE_DIR / "demo_report.md"
    record = dict(result)
    record["timestamp"] = time.time()
    with event_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = cumulative_summary(state, record)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(render_demo_report(summary), encoding="utf-8")


def default_class_selection(value: str) -> list[str]:
    """Parse launch class defaults into UI labels."""
    value = str(value or "").strip()
    if not value:
        return ["all"]
    selected: list[str] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if item.isdigit():
            cls_id = int(item)
            if 0 <= cls_id < len(COCO_CLASS_NAMES):
                selected.append(COCO_CLASS_NAMES[cls_id])
            continue
        if item in COCO_CLASS_TO_ID:
            selected.append(item)
    return selected or ["all"]


def parse_class_selection(selection: Any) -> list[int] | None:
    """Convert UI class selection to YOLO class ids."""
    if selection is None:
        return None
    if isinstance(selection, str):
        items = [selection]
    else:
        items = [str(item) for item in selection]
    items = [item.strip() for item in items if str(item).strip()]
    if not items or "all" in items:
        return None
    class_ids: list[int] = []
    for item in items:
        if item.isdigit():
            cls_id = int(item)
        else:
            cls_id = COCO_CLASS_TO_ID.get(item, -1)
        if 0 <= cls_id < len(COCO_CLASS_NAMES) and cls_id not in class_ids:
            class_ids.append(cls_id)
    return class_ids or None


def filter_classes(boxes: list[dict[str, Any]], class_ids: list[int] | None) -> list[dict[str, Any]]:
    """Filter boxes by selected class ids."""
    if class_ids is None:
        return boxes
    allowed = {int(cls_id) for cls_id in class_ids}
    return [box for box in boxes if int(box.get("cls", -1)) in allowed]


def cloud_request_timed(
    frame: np.ndarray,
    frame_id: int,
    cloud_url: str,
    jpeg_quality: int,
    cloud_timeout: float,
    payload_type: str,
    reason: list[str],
    edge_boxes: list[dict[str, Any]],
    score: float,
    crop_xyxy: list[int] | None = None,
    target_edge_box: dict[str, Any] | None = None,
    cloud_imgsz: int | None = None,
    class_ids: list[int] | None = None,
) -> tuple[dict[str, Any], int, dict[str, float]]:
    """Create one cloud request and expose encode/HTTP timings separately."""
    try:
        infer_url_to_health_url(cloud_url)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}, 0, {}
    encode_started = time.perf_counter()
    payload, payload_bytes = make_payload(
        frame,
        frame_id,
        payload_type,
        reason,
        edge_boxes,
        score,
        jpeg_quality,
        crop_xyxy=crop_xyxy,
        target_edge_box=target_edge_box,
        cloud_imgsz=cloud_imgsz,
    )
    if class_ids is not None:
        payload["metadata"]["classes"] = [int(cls_id) for cls_id in class_ids]
    encode_ms = (time.perf_counter() - encode_started) * 1000.0

    http_started = time.perf_counter()
    response = post_to_cloud(cloud_url, payload, cloud_timeout)
    http_client_ms = (time.perf_counter() - http_started) * 1000.0
    return (
        response,
        payload_bytes,
        {
            "jpeg_encode_ms": encode_ms,
            "http_client_block_ms": http_client_ms,
            "roundtrip_ms": float(response.get("roundtrip_ms", 0.0)),
            "cloud_reported_ms": float(response.get("cloud_latency_ms", 0.0)),
        },
    )


def add_timing(accumulator: dict[str, float], values: dict[str, float]) -> None:
    """Accumulate timing values for one frame."""
    for key, value in values.items():
        accumulator[key] = float(accumulator.get(key, 0.0)) + float(value)


def get_edge_model(model_path: str, device: str) -> YOLO:
    """Load and cache the edge model."""
    key = (str(Path(model_path).resolve()), str(device))
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = YOLO(model_path)
    return MODEL_CACHE[key]


def get_policy(policy_path: str) -> dict[str, Any] | None:
    """Load and cache a learned policy."""
    policy_path = str(policy_path).strip()
    if not policy_path:
        return None
    if policy_path not in POLICY_CACHE:
        path = Path(policy_path).expanduser()
        raw = path.read_bytes()
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ValueError("policy JSON is too large")
        payload = json.loads(raw)
        if isinstance(payload, dict) and payload.get("policy_type") == "cavs_current_maur":
            POLICY_CACHE[policy_path] = load_current_maur_policy(path)
        else:
            POLICY_CACHE[policy_path] = load_policy(path)
    return POLICY_CACHE[policy_path]


def new_state() -> dict[str, Any]:
    """Create fresh UI state."""
    return {
        "frame_id": 0,
        "temporal_tracks": [],
        "stats": Counter(),
        "edge_latencies": [],
        "cloud_latencies": [],
        "roundtrip_latencies": [],
        "e2e_latencies": [],
        "timing_records": [],
        "payload_kb_total": 0.0,
        "online_budget_credit_bytes": 0.0,
        "online_original_bytes_total": 0,
        "online_request_bytes_total": 0,
    }


def rgb_to_bgr(frame_rgb: np.ndarray | None) -> np.ndarray | None:
    """Convert Gradio RGB input to OpenCV BGR."""
    if frame_rgb is None:
        return None
    return cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR output to RGB."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def average_ms(values: list[float]) -> float:
    """Return the arithmetic mean or zero."""
    return float(statistics.mean(values)) if values else 0.0


def build_runtime_args(
    cloud_imgsz: int,
    conf: float,
    cloud_conf: float,
    enable_adaptive_cloud_imgsz: bool,
    learned_cloud_imgsz_low: int,
    learned_cloud_imgsz_high: int,
    learned_cloud_imgsz_switch: float,
    temporal_iou: float,
    temporal_ttl_frames: int,
    temporal_stable_hits: int,
    temporal_reupload_gap: int,
    temporal_conf_drop: float,
    temporal_frame_stable_ratio: float,
    roi_expand: float,
    roi_small_gain: float,
    roi_edge_gain: float,
    roi_large_gain: float,
    iou_match: float,
    fused_nms_iou: float,
) -> argparse.Namespace:
    """Build a small runtime namespace reused by helper functions."""
    return argparse.Namespace(
        cloud_imgsz=int(cloud_imgsz),
        enable_learned_adaptive_cloud_imgsz=bool(enable_adaptive_cloud_imgsz),
        learned_cloud_imgsz_low=int(learned_cloud_imgsz_low),
        learned_cloud_imgsz_high=int(learned_cloud_imgsz_high),
        learned_cloud_imgsz_switch=float(learned_cloud_imgsz_switch),
        temporal_iou=float(temporal_iou),
        temporal_ttl_frames=int(temporal_ttl_frames),
        temporal_stable_hits=int(temporal_stable_hits),
        temporal_reupload_gap=int(temporal_reupload_gap),
        temporal_conf_drop=float(temporal_conf_drop),
        temporal_frame_stable_ratio=float(temporal_frame_stable_ratio),
        roi_expand=float(roi_expand),
        roi_small_gain=float(roi_small_gain),
        roi_edge_gain=float(roi_edge_gain),
        roi_large_gain=float(roi_large_gain),
        iou_match=float(iou_match),
        fused_nms_iou=float(fused_nms_iou),
        conf=float(conf),
        cloud_conf=float(cloud_conf),
    )


def overlay_status(frame: np.ndarray, result: dict[str, Any]) -> None:
    """Render a compact telemetry overlay on the image."""
    panel_h = 122
    cv2.rectangle(frame, (0, 0), (520, panel_h), (18, 18, 18), thickness=-1)
    cv2.rectangle(frame, (0, 0), (520, panel_h), (55, 55, 55), thickness=1)
    lines = [
        f"mode={result['mode']}  detections={result['detections']}  tracks={result['temporal_tracks']}",
        (
            f"edge={result['edge_latency_ms']:.1f} ms  cloud={result['cloud_latency_ms']:.1f} ms  "
            f"rtt={result['roundtrip_ms']:.1f} ms  e2e={result['e2e_ms']:.1f} ms"
        ),
        f"fps={result['fps']:.1f}  upload={result['upload_type']}  payload={result['payload_kb']:.1f} KB",
        f"action={result.get('selected_action', 'n/a')}  decision={result['decision_reason']}  {result['upload_note']}",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line[:90], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
        y += 26


def warmup_demo(
    edge_model_path: str,
    device: str,
    cloud_url: str,
    cloud_timeout: float,
    imgsz: int,
    conf: float,
    jpeg_quality: int,
    cloud_imgsz: int,
) -> str:
    """Warm up one edge inference and one cloud request."""
    try:
        infer_url_to_health_url(cloud_url)
    except ValueError as exc:
        return f"Warmup blocked: {exc}"
    edge_model = get_edge_model(edge_model_path, device)
    dummy = np.zeros((int(imgsz), int(imgsz), 3), dtype=np.uint8)
    kwargs: dict[str, Any] = {"source": dummy, "imgsz": int(imgsz), "conf": float(conf), "verbose": False}
    if device:
        kwargs["device"] = device
    edge_model.predict(**kwargs)
    response, _ = cloud_request(
        dummy,
        0,
        cloud_url,
        int(jpeg_quality),
        float(cloud_timeout),
        "keyframe",
        ["warmup"],
        [],
        0.0,
        cloud_imgsz=int(cloud_imgsz),
    )
    if response.get("status") == "ok":
        return "Warmup finished"
    return f"Warmup failed: {response.get('error', 'unknown error')}"


def cloud_health_markdown(cloud_url: str, cloud_timeout: float) -> str:
    """Return a safe, concise connection status for the presentation UI."""
    health = check_cloud_health(cloud_url, min(3.0, max(0.1, float(cloud_timeout))))
    if health["ok"]:
        model = str(health.get("model", "unknown")).replace("`", "")
        model_sha256 = str(health.get("model_sha256", "unknown")).replace("`", "")[:12]
        device = str(health.get("device", "unknown")).replace("`", "")
        imgsz = str(health.get("imgsz", "unknown")).replace("`", "")
        conf = str(health.get("conf", "unknown")).replace("`", "")
        return (
            f"**云端状态：在线**　模型 `{model}`　SHA `{model_sha256}`　设备 `{device}`　"
            f"imgsz `{imgsz}`　conf `{conf}`　健康检查 `{health['latency_ms']:.1f} ms`"
        )
    return f"**云端状态：离线**　`{health.get('status', 'offline')}`　可继续使用“仅端侧”模式。"


def policy_summary_markdown(policy_path: str) -> str:
    """Describe the actual loaded learner instead of relying on a UI label."""
    try:
        policy = get_policy(policy_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return f"**学习器状态：不可用**　`{type(exc).__name__}`"
    if policy is None:
        return "**学习器状态：未配置**"
    if policy.get("kind") == "current_maur":
        seeds = ", ".join(map(str, policy["ensemble_seeds"]))
        return (
            f"**学习器：{policy['display_name']}**　动作 `Edge / Original / Q90 / current@640`　"
            f"融合 `MAUR`　三种子 `{seeds}`"
        )
    return "**学习器：Legacy Dual-Head**　兼容旧版 Frame / ROI 上传策略"


def reset_memory() -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Reset temporal memory and cumulative counters."""
    state = new_state()
    return {"status": "memory_reset"}, "Temporal memory reset", state


def stats_to_markdown(mode: str, result: dict[str, Any], state: dict[str, Any]) -> str:
    """Summarize cumulative demo state."""
    stats = Counter(state.get("stats", {}))
    edge_avg = average_ms(state.get("edge_latencies", []))
    cloud_avg = average_ms(state.get("cloud_latencies", []))
    rtt_avg = average_ms(state.get("roundtrip_latencies", []))
    e2e_avg = average_ms(state.get("e2e_latencies", []))
    fps_avg = 1000.0 / e2e_avg if e2e_avg > 1e-6 else 0.0
    uploads = int(stats["upload_attempts"])
    upload_ratio = uploads / max(1, int(stats["frames"]))
    online_original = int(state.get("online_original_bytes_total", 0))
    online_request = int(state.get("online_request_bytes_total", 0))
    online_budget_details = ""
    if online_original > 0:
        online_budget_details = (
            f"  \n**在线字节预算**　实际 `{online_request / online_original:.1%}`　"
            f"额度余额 `{float(state.get('online_budget_credit_bytes', 0.0)) / 1024.0:.1f} KB`"
        )
    if result.get("policy_kind") == "current_maur":
        decision_details = (
            f"**学习器**　`{result['policy_name']}`　动作 `{result['selected_action']}`　"
            f"预测价值 `{result['learner_expected_gain']:.3f}`　代价后效用 "
            f"`{result['learner_decision_score']:.3f}`  \n"
            f"**四动作分数**　`{result.get('action_scores', {})}`  \n"
        )
    else:
        decision_details = (
            f"**决策细节**　原因 `{result['decision_reason']}`　P(收益) "
            f"`{result['learner_probability']:.3f}`　预期收益 `{result['learner_expected_gain']:.3f}`　"
            f"决策分数 `{result['learner_decision_score']:.3f}`  \n"
        )
    return (
        f"### 当前状态 · `{mode}`\n\n"
        f"**本帧**　端侧 `{result['edge_latency_ms']:.1f} ms`　云端 `{result['cloud_latency_ms']:.1f} ms`　"
        f"RTT `{result['roundtrip_ms']:.1f} ms`, e2e `{result['e2e_ms']:.1f} ms`, fps `{result['fps']:.1f}`  \n"
        f"**学习器决策**　`{result['upload_type']}`　上传 `{result['payload_kb']:.1f} KB`　`{result['upload_note']}`  \n"
        f"{decision_details}"
        f"**累计平均**　端侧 `{edge_avg:.1f} ms`　云端 `{cloud_avg:.1f} ms`　RTT `{rtt_avg:.1f} ms`　"
        f"e2e `{e2e_avg:.1f} ms`, fps `{fps_avg:.1f}`  \n"
        f"**累计流量**　`{float(state.get('payload_kb_total', 0.0)):.1f} KB`　上传任务 `{uploads}`　"
        f"任务/帧 `{upload_ratio:.1%}`　云端错误 `{stats['cloud_errors']}`"
        f"{online_budget_details}"
    )


def prepare_current_maur_candidate(
    frame: np.ndarray,
    policy: dict[str, Any],
    policy_edge_boxes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the paper O/Q/R values and exact raw-JPEG costs without selecting an action."""
    height, width = frame.shape[:2]
    roi_config = policy["roi"]
    roi_xyxy = current_roi_xyxy(
        width,
        height,
        policy_edge_boxes,
        confidence=float(roi_config["confidence"]),
        max_area_ratio=float(roi_config["max_area_ratio"]),
    )
    original_bytes = len(encode_jpeg(frame, 95))
    q90_bytes = len(encode_jpeg(frame, 90))
    payloads = {"edge": 0, "original": original_bytes, "q90": q90_bytes}
    if roi_xyxy is not None:
        x1, y1, x2, y2 = roi_xyxy
        payloads["current@640"] = len(encode_jpeg(frame[y1:y2, x1:x2], 90))

    scores = predict_current_maur_scores(
        policy,
        policy_edge_boxes,
        width=width,
        height=height,
        roi_xyxy=roi_xyxy,
    )
    if not 0 < q90_bytes < original_bytes:
        payloads.pop("q90")
        scores.pop("q90", None)
    return {"roi_xyxy": roi_xyxy, "payload_bytes": payloads, "scores": scores}


def run_current_maur_action(
    frame: np.ndarray,
    frame_id: int,
    policy: dict[str, Any],
    policy_edge_boxes: list[dict[str, Any]],
    display_edge_boxes: list[dict[str, Any]],
    cloud_url: str,
    cloud_timeout: float,
    cloud_conf: float,
    cost_weight: float,
    class_ids: list[int] | None,
    *,
    use_temporal_memory: bool,
    frame_temporal_stable: bool,
    temporal_tracks: list[dict[str, Any]],
    runtime_args: argparse.Namespace,
    forced_action: str | None = None,
    online_budget_fraction: float | None = None,
    online_budget_credit_bytes: float = 0.0,
) -> dict[str, Any]:
    """Select and execute one CAVS OQR action, using MAUR for Current ROI fusion."""
    roi_config = policy["roi"]
    candidate = prepare_current_maur_candidate(frame, policy, policy_edge_boxes)
    roi_xyxy = candidate["roi_xyxy"]
    payloads = candidate["payload_bytes"]
    scores = candidate["scores"]
    online_decision = select_current_maur_action(
        scores,
        payloads,
        full_payload_bytes=int(payloads["original"]),
        cost_weight=float(cost_weight),
    )
    decision = online_decision
    accrued_credit: float | None = None
    if forced_action is not None:
        forced = str(forced_action)
        if forced not in scores or forced not in payloads:
            raise ValueError(f"paper action is unavailable: {forced}")
        decision = {
            **online_decision,
            "action": forced,
            "score": float(scores[forced]),
            "utility": float(online_decision["utilities"][forced]),
        }
    elif online_budget_fraction is not None:
        fraction = float(online_budget_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("online fixed-byte budget must be in (0, 1]")
        accrued_credit = max(0.0, float(online_budget_credit_bytes)) + fraction * float(payloads["original"])
        feasible = [
            action
            for action in online_decision["utilities"]
            if float(payloads[action]) <= accrued_credit + 1e-9
        ]
        chosen = max(
            feasible,
            key=lambda action: (
                float(online_decision["utilities"][action]),
                -tuple(payloads).index(action),
            ),
        )
        if chosen != online_decision["action"]:
            decision = {
                **online_decision,
                "action": chosen,
                "score": float(scores[chosen]),
                "utility": float(online_decision["utilities"][chosen]),
            }
    model_action = str(decision["action"])
    action = model_action
    stat_updates: Counter[str] = Counter()
    if use_temporal_memory and frame_temporal_stable and action != "edge":
        action = "edge"
        stat_updates["temporal_frame_suppressed"] += 1
    if online_budget_fraction is not None and model_action != online_decision["action"]:
        stat_updates["online_budget_suppressed"] += 1
    remaining_credit = None
    if accrued_credit is not None:
        remaining_credit = max(0.0, accrued_credit - float(payloads[action]))

    common = {
        "display_boxes": [dict(box) for box in display_edge_boxes],
        "upload_type": "none" if action == "edge" else action,
        "upload_note": (
            f"CAVS={model_action} V={decision['score']:.3f} U={decision['utility']:.3f}"
        ),
        "decision_reason": (
            "temporal_frame_suppressed"
            if action != model_action
            else (
                f"paper_fixed_byte_{action}"
                if forced_action is not None
                else (
                    f"online_fixed_byte_{action}"
                    if online_budget_fraction is not None
                    else f"cavs_current_maur_{action.replace('@', '_').replace('.', '_')}"
                )
            )
        ),
        "selected_action": action,
        "model_selected_action": model_action,
        "action_scores": {key: round(float(value), 5) for key, value in decision["scores"].items()},
        "action_utilities": {key: round(float(value), 5) for key, value in decision["utilities"].items()},
        "value": float(decision["score"]),
        "utility": float(decision["utility"]),
        "payload_kb": 0.0,
        "cloud_status": "not_required",
        "cloud_latency_ms": 0.0,
        "roundtrip_ms": 0.0,
        "timings": {},
        "stat_updates": stat_updates,
        "original_payload_bytes": int(payloads["original"]),
        "online_budget_fraction": online_budget_fraction,
        "online_budget_credit_bytes": remaining_credit,
    }
    if action == "edge":
        if action == model_action:
            common["upload_note"] += "; cloud not selected"
        return common

    stat_updates["upload_attempts"] += 1
    is_roi = action == "current@640"
    quality = 90 if action in {"q90", "current@640"} else 95
    response, payload_bytes, request_timings = cloud_request_timed(
        frame,
        frame_id,
        cloud_url,
        quality,
        float(cloud_timeout),
        "roi" if is_roi else "keyframe",
        [f"cavs_current_maur_{action}"],
        display_edge_boxes,
        float(decision["score"]),
        crop_xyxy=roi_xyxy if is_roi else None,
        cloud_imgsz=int(roi_config["imgsz"]),
        class_ids=class_ids,
    )
    common["payload_kb"] = payload_bytes / 1024.0
    common["cloud_status"] = str(response.get("status", "error"))
    common["timings"] = request_timings
    if response.get("status") != "ok":
        stat_updates["cloud_errors"] += 1
        common["upload_note"] = str(response.get("error", "cloud_error"))[:80]
        return common

    raw_cloud_boxes = filter_classes(list(response.get("cloud_boxes", [])), class_ids)
    if is_roi:
        maur_config = policy["maur"]
        fused = maur_fuse_boxes(
            policy_edge_boxes,
            raw_cloud_boxes,
            roi_xyxy,
            duplicate_iou=float(maur_config["duplicate_iou"]),
            nms_iou=float(maur_config["nms_iou"]),
        )
        common["display_boxes"] = filter_confidence(fused, float(cloud_conf))
        stat_updates["collab_roi_ok"] += 1
    else:
        common["display_boxes"] = filter_confidence(raw_cloud_boxes, float(cloud_conf))
        stat_updates["collab_full_ok"] += 1
    common["cloud_latency_ms"] = float(response.get("cloud_latency_ms", 0.0))
    common["roundtrip_ms"] = float(response.get("roundtrip_ms", 0.0))
    if use_temporal_memory:
        mark_temporal_memory_from_response(
            temporal_tracks,
            response.get("request_metadata", {}),
            raw_cloud_boxes,
            frame_id,
            runtime_args,
        )
    return common


def process_frame(
    frame_rgb: np.ndarray | None,
    mode: str,
    cloud_url: str,
    edge_model_path: str,
    device: str,
    policy_path: str,
    imgsz: int,
    conf: float,
    cloud_conf: float,
    cloud_timeout: float,
    cloud_imgsz: int,
    jpeg_quality: int,
    class_selection: Any,
    learned_frame_threshold: float,
    learned_roi_threshold: float,
    learned_cost_weight: float,
    learned_max_roi_per_frame: int,
    enable_adaptive_cloud_imgsz: bool,
    learned_cloud_imgsz_low: int,
    learned_cloud_imgsz_high: int,
    learned_cloud_imgsz_switch: float,
    enable_temporal_memory: bool,
    temporal_iou: float,
    temporal_ttl_frames: int,
    temporal_stable_hits: int,
    temporal_reupload_gap: int,
    temporal_conf_drop: float,
    temporal_frame_stable_ratio: float,
    enable_adaptive_roi: bool,
    roi_expand: float,
    roi_small_gain: float,
    roi_edge_gain: float,
    roi_large_gain: float,
    iou_match: float,
    fused_nms_iou: float,
    enforce_online_fixed_byte: bool,
    online_fixed_byte_budget: Any,
    state: dict[str, Any],
    *,
    input_kind: str = "camera",
) -> tuple[np.ndarray | None, dict[str, Any], str, dict[str, Any]]:
    """Process a single webcam/uploaded frame."""
    if frame_rgb is None:
        return None, {"status": "waiting_for_frame"}, "Waiting for frame", state or new_state()

    frame_started = time.perf_counter()
    timings: dict[str, float] = {}
    section_started = time.perf_counter()
    runtime_args = build_runtime_args(
        cloud_imgsz,
        conf,
        cloud_conf,
        enable_adaptive_cloud_imgsz,
        learned_cloud_imgsz_low,
        learned_cloud_imgsz_high,
        learned_cloud_imgsz_switch,
        temporal_iou,
        temporal_ttl_frames,
        temporal_stable_hits,
        temporal_reupload_gap,
        temporal_conf_drop,
        temporal_frame_stable_ratio,
        roi_expand,
        roi_small_gain,
        roi_edge_gain,
        roi_large_gain,
        iou_match,
        fused_nms_iou,
    )
    timings["runtime_args_ms"] = (time.perf_counter() - section_started) * 1000.0

    section_started = time.perf_counter()
    frame = rgb_to_bgr(frame_rgb)
    assert frame is not None
    timings["input_convert_ms"] = (time.perf_counter() - section_started) * 1000.0

    state = dict(state or new_state())
    paper_forced_action = state.pop("paper_forced_action", None)
    state["frame_id"] = int(state.get("frame_id", 0)) + 1
    frame_id = int(state["frame_id"])
    temporal_tracks = list(state.get("temporal_tracks", []))
    stats = Counter(state.get("stats", {}))
    edge_latencies = list(state.get("edge_latencies", []))
    cloud_latencies = list(state.get("cloud_latencies", []))
    roundtrip_latencies = list(state.get("roundtrip_latencies", []))
    e2e_latencies = list(state.get("e2e_latencies", []))
    timing_records = list(state.get("timing_records", []))

    section_started = time.perf_counter()
    edge_model = get_edge_model(edge_model_path, device)
    policy = get_policy(policy_path)
    class_ids = parse_class_selection(class_selection)
    timings["model_policy_cache_ms"] = (time.perf_counter() - section_started) * 1000.0

    is_current_maur = bool(policy and policy.get("kind") == "current_maur")
    deploy_confidence = float(policy.get("deploy_confidence", conf)) if is_current_maur else float(conf)
    predict_kwargs: dict[str, Any] = {
        "source": frame,
        "imgsz": int(imgsz),
        "conf": min(float(conf), deploy_confidence),
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device
    if class_ids is not None:
        predict_kwargs["classes"] = class_ids
    edge_started = time.perf_counter()
    edge_result = edge_model.predict(**predict_kwargs)[0]
    edge_latency_ms = (time.perf_counter() - edge_started) * 1000.0
    timings["edge_infer_ms"] = edge_latency_ms
    section_started = time.perf_counter()
    policy_edge_boxes = filter_classes(boxes_from_result(edge_result), class_ids)
    edge_boxes = filter_confidence(policy_edge_boxes, float(conf))
    timings["edge_postprocess_ms"] = (time.perf_counter() - section_started) * 1000.0
    edge_latencies.append(edge_latency_ms)
    stats["frames"] += 1

    use_temporal_memory = temporal_memory_active(enable_temporal_memory, input_kind)
    box_track_map: dict[int, dict[str, Any]] = {}
    frame_temporal_stable = False
    section_started = time.perf_counter()
    if use_temporal_memory:
        temporal_tracks, box_track_map = update_temporal_tracks(
            temporal_tracks,
            edge_boxes,
            frame_id,
            float(temporal_iou),
            int(temporal_ttl_frames),
        )
        frame_temporal_stable = temporal_skip_frame_upload(edge_boxes, box_track_map, frame_id, runtime_args)
    timings["temporal_memory_ms"] = (time.perf_counter() - section_started) * 1000.0

    display_boxes = [dict(box) for box in edge_boxes]
    upload_type = "none"
    upload_note = "local"
    learner_probability = 0.0
    learner_expected_gain = 0.0
    learner_decision_score = 0.0
    selected_action = "edge"
    model_selected_action = "edge"
    action_scores: dict[str, float] = {}
    action_utilities: dict[str, float] = {}
    policy_name = str(policy.get("display_name", "legacy-dual-head")) if policy else "unavailable"
    decision_reason = "edge_only"
    cloud_status = "not_required"
    payload_kb = 0.0
    cloud_latency_ms = 0.0
    roundtrip_ms = 0.0

    if mode == MODE_CLOUD:
        stats["upload_attempts"] += 1
        decision_reason = "cloud_only"
        response, payload_bytes, request_timings = cloud_request_timed(
            frame,
            frame_id,
            cloud_url,
            int(jpeg_quality),
            float(cloud_timeout),
            "keyframe",
            ["cloud_only"],
            edge_boxes,
            1.0,
            cloud_imgsz=int(cloud_imgsz),
            class_ids=class_ids,
        )
        add_timing(timings, request_timings)
        payload_kb = payload_bytes / 1024.0
        upload_type = "full"
        upload_note = "cloud_only"
        cloud_status = str(response.get("status", "error"))
        if response.get("status") == "ok":
            section_started = time.perf_counter()
            display_boxes = filter_classes(filter_confidence(response.get("cloud_boxes", []), float(cloud_conf)), class_ids)
            timings["fusion_postprocess_ms"] = (time.perf_counter() - section_started) * 1000.0
            cloud_latency_ms = float(response.get("cloud_latency_ms", 0.0))
            roundtrip_ms = float(response.get("roundtrip_ms", 0.0))
            cloud_latencies.append(cloud_latency_ms)
            roundtrip_latencies.append(roundtrip_ms)
            stats["cloud_only_ok"] += 1
        else:
            stats["cloud_errors"] += 1
            upload_note = str(response.get("error", "cloud_error"))[:80]

    elif mode == MODE_COLLAB:
        decision_started = time.perf_counter()
        context = image_context(frame, edge_boxes)
        estimate_started = time.perf_counter()
        full_frame_kb = len(encode_jpeg(frame, int(jpeg_quality))) / 1024.0
        timings["payload_estimate_encode_ms"] = (time.perf_counter() - estimate_started) * 1000.0
        if is_current_maur:
            online_budget_fraction = (
                parse_paper_budget(online_fixed_byte_budget)
                if bool(enforce_online_fixed_byte) and input_kind == "camera"
                else None
            )
            current_result = run_current_maur_action(
                frame,
                frame_id,
                policy,
                policy_edge_boxes,
                edge_boxes,
                cloud_url,
                float(cloud_timeout),
                float(cloud_conf),
                float(learned_cost_weight),
                class_ids,
                use_temporal_memory=use_temporal_memory,
                frame_temporal_stable=frame_temporal_stable,
                temporal_tracks=temporal_tracks,
                runtime_args=runtime_args,
                forced_action=str(paper_forced_action) if paper_forced_action is not None else None,
                online_budget_fraction=online_budget_fraction,
                online_budget_credit_bytes=float(state.get("online_budget_credit_bytes", 0.0)),
            )
            if online_budget_fraction is not None:
                state["online_budget_credit_bytes"] = float(current_result["online_budget_credit_bytes"])
                state["online_original_bytes_total"] = int(state.get("online_original_bytes_total", 0)) + int(
                    current_result["original_payload_bytes"]
                )
                state["online_request_bytes_total"] = int(state.get("online_request_bytes_total", 0)) + int(
                    round(float(current_result["payload_kb"]) * 1024.0)
                )
            timings["decision_ms"] = (time.perf_counter() - decision_started) * 1000.0
            add_timing(timings, current_result["timings"])
            display_boxes = current_result["display_boxes"]
            upload_type = str(current_result["upload_type"])
            upload_note = str(current_result["upload_note"])
            decision_reason = str(current_result["decision_reason"])
            selected_action = str(current_result["selected_action"])
            model_selected_action = str(current_result["model_selected_action"])
            action_scores = dict(current_result["action_scores"])
            action_utilities = dict(current_result["action_utilities"])
            learner_expected_gain = float(current_result["value"])
            learner_decision_score = float(current_result["utility"])
            payload_kb = float(current_result["payload_kb"])
            cloud_status = str(current_result["cloud_status"])
            cloud_latency_ms = float(current_result["cloud_latency_ms"])
            roundtrip_ms = float(current_result["roundtrip_ms"])
            stats.update(current_result["stat_updates"])
            if cloud_latency_ms > 0.0:
                cloud_latencies.append(cloud_latency_ms)
            if roundtrip_ms > 0.0:
                roundtrip_latencies.append(roundtrip_ms)
        elif policy is None:
            upload_note = "no_policy"
            decision_reason = "policy_unavailable"
            timings["decision_ms"] = (time.perf_counter() - decision_started) * 1000.0
        else:
            frame_eval = policy_decision(
                policy,
                "frame",
                frame_features(context, policy.get("frame_features")),
                full_frame_kb,
                float(learned_cost_weight),
            )
            learner_probability = float(frame_eval["probability"])
            learner_expected_gain = float(frame_eval["expected_gain"])
            learner_decision_score = float(frame_eval["decision"])
            selected_action = "legacy"
            model_selected_action = "legacy"
            if use_temporal_memory and frame_temporal_stable:
                stats["temporal_frame_suppressed"] += 1
                upload_note = "temporal_frame_skip"
                decision_reason = "temporal_frame_suppressed"
                timings["decision_ms"] = (time.perf_counter() - decision_started) * 1000.0
            elif frame_eval["probability"] >= float(learned_frame_threshold) and frame_eval["decision"] > 0.0:
                stats["upload_attempts"] += 1
                decision_reason = "learned_full_frame"
                timings["decision_ms"] = (time.perf_counter() - decision_started) * 1000.0
                response, payload_bytes, request_timings = cloud_request_timed(
                    frame,
                    frame_id,
                    cloud_url,
                    int(jpeg_quality),
                    float(cloud_timeout),
                    "keyframe",
                    ["learned_full_frame"],
                    edge_boxes,
                    float(frame_eval["probability"]),
                    cloud_imgsz=select_cloud_imgsz(
                        float(frame_eval["probability"]),
                        runtime_args,
                        adaptive=bool(enable_adaptive_cloud_imgsz),
                    ),
                    class_ids=class_ids,
                )
                add_timing(timings, request_timings)
                payload_kb = payload_bytes / 1024.0
                upload_type = "full"
                upload_note = (
                    f"frame p={frame_eval['probability']:.2f} "
                    f"g={frame_eval['expected_gain']:.2f}"
                )
                cloud_status = str(response.get("status", "error"))
                if response.get("status") == "ok":
                    cloud_boxes = filter_classes(filter_confidence(response.get("cloud_boxes", []), float(cloud_conf)), class_ids)
                    section_started = time.perf_counter()
                    display_boxes = full_frame_fusion_boxes(
                        edge_boxes,
                        cloud_boxes,
                        float(iou_match),
                        float(fused_nms_iou),
                        float(cloud_conf),
                        float(conf),
                    )
                    timings["fusion_postprocess_ms"] = (time.perf_counter() - section_started) * 1000.0
                    cloud_latency_ms = float(response.get("cloud_latency_ms", 0.0))
                    roundtrip_ms = float(response.get("roundtrip_ms", 0.0))
                    cloud_latencies.append(cloud_latency_ms)
                    roundtrip_latencies.append(roundtrip_ms)
                    stats["collab_full_ok"] += 1
                    if use_temporal_memory:
                        mark_temporal_memory_from_response(
                            temporal_tracks,
                            response.get("request_metadata", {}),
                            response.get("cloud_boxes", []),
                            frame_id,
                            runtime_args,
                        )
                else:
                    stats["cloud_errors"] += 1
                    upload_note = str(response.get("error", "cloud_error"))[:80]
            else:
                final_boxes = [dict(box) for box in edge_boxes]
                candidates: list[tuple[float, float, float, int, dict[str, Any], list[int]]] = []
                for edge_index, box in enumerate(edge_boxes):
                    track = box_track_map.get(edge_index)
                    if use_temporal_memory and temporal_skip_roi_upload(
                        box,
                        track,
                        frame_id,
                        float(temporal_iou),
                        int(temporal_stable_hits),
                        int(temporal_reupload_gap),
                        float(temporal_conf_drop),
                    ):
                        stats["temporal_roi_suppressed"] += 1
                        continue
                    roi_gain = (
                        adaptive_roi_gain(
                            box,
                            frame.shape[:2],
                            base_gain=float(roi_expand),
                            small_gain=float(roi_small_gain),
                            edge_gain=float(roi_edge_gain),
                            large_gain=float(roi_large_gain),
                        )
                        if enable_adaptive_roi
                        else float(roi_expand)
                    )
                    crop_xyxy = expand_xyxy(box["xyxy"], frame.shape[:2], roi_gain)
                    x1, y1, x2, y2 = crop_xyxy
                    crop_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, float(frame.shape[0] * frame.shape[1]))
                    estimated_kb = max(1.0, full_frame_kb * crop_ratio)
                    roi_eval = policy_decision(
                        policy,
                        "roi",
                        roi_features(box, frame.shape[:2], context, policy.get("roi_features")),
                        estimated_kb,
                        float(learned_cost_weight),
                    )
                    if roi_eval["probability"] >= float(learned_roi_threshold) and roi_eval["decision"] > 0.0:
                        candidates.append(
                            (
                                float(roi_eval["decision"]),
                                float(roi_eval["probability"]),
                                float(roi_eval["expected_gain"]),
                                edge_index,
                                box,
                                crop_xyxy,
                            )
                        )
                candidates.sort(key=lambda item: item[0], reverse=True)
                chosen = candidates[: max(1, int(learned_max_roi_per_frame))]
                timings["decision_ms"] = (time.perf_counter() - decision_started) * 1000.0
                if not chosen:
                    upload_note = "edge_kept"
                    decision_reason = "below_upload_threshold"
                for decision_score, probability, expected_gain, edge_index, box, crop_xyxy in chosen:
                    stats["upload_attempts"] += 1
                    decision_reason = "learned_roi"
                    learner_probability = probability
                    learner_expected_gain = expected_gain
                    learner_decision_score = decision_score
                    response, payload_bytes, request_timings = cloud_request_timed(
                        frame,
                        frame_id,
                        cloud_url,
                        int(jpeg_quality),
                        float(cloud_timeout),
                        "roi",
                        ["learned_roi"],
                        edge_boxes,
                        probability,
                        crop_xyxy=crop_xyxy,
                        target_edge_box=box,
                        cloud_imgsz=select_cloud_imgsz(
                            probability,
                            runtime_args,
                            adaptive=bool(enable_adaptive_cloud_imgsz),
                        ),
                        class_ids=class_ids,
                    )
                    add_timing(timings, request_timings)
                    payload_kb += payload_bytes / 1024.0
                    upload_type = "roi"
                    upload_note = f"roi p={probability:.2f} g={expected_gain:.2f}"
                    cloud_status = str(response.get("status", "error"))
                    if response.get("status") == "ok":
                        cloud_boxes = filter_classes(filter_confidence(response.get("cloud_boxes", []), float(cloud_conf)), class_ids)
                        cloud_ms = float(response.get("cloud_latency_ms", 0.0))
                        rtt_ms = float(response.get("roundtrip_ms", 0.0))
                        cloud_latency_ms = max(cloud_latency_ms, cloud_ms)
                        roundtrip_ms = max(roundtrip_ms, rtt_ms)
                        cloud_latencies.append(cloud_ms)
                        roundtrip_latencies.append(rtt_ms)
                        section_started = time.perf_counter()
                        final_boxes = apply_roi_cloud(
                            final_boxes,
                            edge_index,
                            cloud_boxes,
                            float(iou_match),
                            float(cloud_conf),
                            float(cloud_conf),
                            float(conf),
                        )
                        timings["fusion_postprocess_ms"] = timings.get("fusion_postprocess_ms", 0.0) + (
                            time.perf_counter() - section_started
                        ) * 1000.0
                        stats["collab_roi_ok"] += 1
                        if use_temporal_memory:
                            mark_temporal_memory_from_response(
                                temporal_tracks,
                                response.get("request_metadata", {}),
                                response.get("cloud_boxes", []),
                                frame_id,
                                runtime_args,
                            )
                    else:
                        stats["cloud_errors"] += 1
                        upload_note = str(response.get("error", "cloud_error"))[:80]
                section_started = time.perf_counter()
                display_boxes = class_aware_nms(final_boxes, float(fused_nms_iou))
                timings["fusion_postprocess_ms"] = timings.get("fusion_postprocess_ms", 0.0) + (
                    time.perf_counter() - section_started
                ) * 1000.0

    timings["pre_draw_server_ms"] = (time.perf_counter() - frame_started) * 1000.0
    section_started = time.perf_counter()
    canvas = frame.copy()
    if mode == MODE_EDGE:
        draw_boxes(canvas, display_boxes, (0, 255, 0), "")
    elif mode == MODE_CLOUD:
        draw_boxes(canvas, display_boxes, (0, 0, 255), "")
    else:
        draw_boxes(canvas, display_boxes, (0, 255, 255), "")
    timings["draw_boxes_ms"] = (time.perf_counter() - section_started) * 1000.0

    pre_overlay_ms = (time.perf_counter() - frame_started) * 1000.0
    pre_overlay_fps = 1000.0 / pre_overlay_ms if pre_overlay_ms > 1e-6 else 0.0
    result_payload = {
        "mode": mode,
        "frame_id": frame_id,
        "input_kind": input_kind,
        "temporal_memory_active": use_temporal_memory,
        "detections": len(display_boxes),
        "temporal_tracks": len(temporal_tracks),
        "edge_latency_ms": round(edge_latency_ms, 3),
        "cloud_latency_ms": round(cloud_latency_ms, 3),
        "roundtrip_ms": round(roundtrip_ms, 3),
        "e2e_ms": round(pre_overlay_ms, 3),
        "fps": round(pre_overlay_fps, 3),
        "upload_type": upload_type,
        "payload_kb": round(payload_kb, 3),
        "upload_note": upload_note,
        "decision_reason": decision_reason,
        "policy_name": policy_name,
        "policy_kind": "current_maur" if is_current_maur else "legacy",
        "selected_action": selected_action,
        "model_selected_action": model_selected_action,
        "action_scores": action_scores,
        "action_utilities": action_utilities,
        "learner_cost_weight": round(float(learned_cost_weight), 4),
        "learner_probability": round(learner_probability, 4),
        "learner_expected_gain": round(learner_expected_gain, 4),
        "learner_decision_score": round(learner_decision_score, 4),
        "cloud_status": cloud_status,
        "cloud_errors": int(stats["cloud_errors"]),
        "classes": "all" if class_ids is None else [COCO_CLASS_NAMES[idx] for idx in class_ids],
    }
    section_started = time.perf_counter()
    overlay_status(canvas, result_payload)
    timings["overlay_ms"] = (time.perf_counter() - section_started) * 1000.0
    section_started = time.perf_counter()
    output_rgb = bgr_to_rgb(canvas)
    timings["output_convert_ms"] = (time.perf_counter() - section_started) * 1000.0

    e2e_ms = (time.perf_counter() - frame_started) * 1000.0
    timings["server_function_total_ms"] = e2e_ms
    e2e_latencies.append(e2e_ms)
    fps = 1000.0 / e2e_ms if e2e_ms > 1e-6 else 0.0
    result_payload["e2e_ms"] = round(e2e_ms, 3)
    result_payload["fps"] = round(fps, 3)
    result_payload["timings_ms"] = {key: round(value, 3) for key, value in sorted(timings.items())}

    timing_records.append(dict(result_payload["timings_ms"]))
    state["stats"] = dict(stats)
    state["temporal_tracks"] = temporal_tracks
    state["edge_latencies"] = edge_latencies
    state["cloud_latencies"] = cloud_latencies
    state["roundtrip_latencies"] = roundtrip_latencies
    state["e2e_latencies"] = e2e_latencies
    state["timing_records"] = timing_records
    state["payload_kb_total"] = float(state.get("payload_kb_total", 0.0)) + payload_kb

    log_started = time.perf_counter()
    save_event(result_payload, state)
    result_payload["timings_ms"]["log_write_ms"] = round((time.perf_counter() - log_started) * 1000.0, 3)
    return output_rgb, result_payload, stats_to_markdown(mode, result_payload, state), state


def process_any_frame(
    webcam_rgb: np.ndarray | None,
    upload_rgb: np.ndarray | None,
    *args: Any,
) -> tuple[np.ndarray | None, dict[str, Any], str, dict[str, Any]]:
    """Prefer the uploaded image when present; otherwise use the live webcam frame."""
    frame_rgb = upload_rgb if upload_rgb is not None else webcam_rgb
    input_kind = "image" if upload_rgb is not None else "camera"
    return process_frame(frame_rgb, *args, input_kind=input_kind)


def video_path_from_input(video_file: Any) -> str | None:
    """Extract a file path from Gradio video input variants."""
    if video_file is None:
        return None
    if isinstance(video_file, str):
        return video_file
    if isinstance(video_file, (tuple, list)) and video_file:
        return str(video_file[0])
    if isinstance(video_file, dict):
        for key in ("video", "path", "name"):
            if video_file.get(key):
                return str(video_file[key])
    return None


def paper_video_runtime_args(frame_args: list[Any]) -> list[Any]:
    """Freeze the process_frame arguments that define the paper Current+MAUR runtime."""
    if len(frame_args) != 36:
        raise ValueError(f"paper video requires 36 frame arguments, received {len(frame_args)}")
    runtime = list(frame_args)
    runtime[0] = MODE_COLLAB
    runtime[5] = 640
    runtime[9] = 640
    runtime[11] = ["all"]
    runtime[16] = False
    runtime[20] = False
    runtime[27] = False
    runtime[34] = False
    return runtime


def parse_paper_budget(value: Any) -> float:
    """Convert a paper budget control value to its fixed-byte fraction."""
    text = str(value).strip()
    if text.endswith("%"):
        text = text[:-1]
        fraction = float(text) / 100.0
    else:
        fraction = float(text)
    if fraction not in {0.1, 0.25, 0.4}:
        raise ValueError("Paper Fixed-byte budget must be 10%, 25%, or 40%")
    return fraction


def collect_paper_video_candidates(
    video_file: Any,
    frame_args: list[Any],
    *,
    max_frames: int,
    stride: int,
) -> list[dict[str, Any]]:
    """Run the first video pass and collect paper O/Q/R values and exact JPEG costs."""
    video_path = video_path_from_input(video_file)
    if not video_path:
        raise ValueError("Waiting for video")
    runtime = paper_video_runtime_args(frame_args)
    edge_model = get_edge_model(str(runtime[2]), str(runtime[3]))
    policy = get_policy(str(runtime[4]))
    if not policy or policy.get("kind") != "current_maur":
        raise ValueError("Paper Fixed-byte requires the Current+MAUR policy")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    candidates: list[dict[str, Any]] = []
    read_index = 0
    try:
        while len(candidates) < max(1, int(max_frames)):
            ok, frame = cap.read()
            if not ok:
                break
            read_index += 1
            if (read_index - 1) % max(1, int(stride)) != 0:
                continue
            deploy_confidence = float(policy.get("deploy_confidence", runtime[6]))
            predict_kwargs: dict[str, Any] = {
                "source": frame,
                "imgsz": 640,
                "conf": min(float(runtime[6]), deploy_confidence),
                "verbose": False,
            }
            if runtime[3]:
                predict_kwargs["device"] = str(runtime[3])
            edge_result = edge_model.predict(**predict_kwargs)[0]
            candidate = prepare_current_maur_candidate(frame, policy, boxes_from_result(edge_result))
            candidates.append({"frame_id": len(candidates) + 1, **candidate})
    finally:
        cap.release()
    if not candidates:
        raise ValueError("No video frame was available for paper scheduling")
    return candidates


def process_video_file(
    video_file: Any,
    *args: Any,
    forced_actions: dict[int, str] | None = None,
    paper_plan: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any], str, dict[str, Any]]:
    """Process an uploaded video and return an annotated video path."""
    video_path = video_path_from_input(video_file)
    if not video_path:
        return None, {"status": "waiting_for_video"}, "Waiting for video", args[-1] if args else new_state()

    max_frames = max(1, int(args[-3]))
    stride = max(1, int(args[-2]))
    state = args[-1] or new_state()
    starting_payload_bytes = int(round(float(state.get("payload_kb_total", 0.0)) * 1024.0))
    frame_args = args[:-3] + (state,)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {"status": "video_open_failed", "path": video_path}, f"Could not open video: {video_path}", state

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_fps = max(1.0, float(fps) / float(stride))
    save_dir = RUN_SAVE_DIR or Path("runs/edge_cloud/gradio_demo")
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"annotated_video_{int(time.time())}.mp4"
    writer = None
    processed = 0
    read_idx = 0
    read_times: list[float] = []
    write_times: list[float] = []
    convert_times: list[float] = []
    video_started = time.perf_counter()
    last_result: dict[str, Any] = {"status": "no_frame_processed"}
    last_status = "No frame processed"

    try:
        while processed < max_frames:
            read_started = time.perf_counter()
            ok, frame_bgr = cap.read()
            read_times.append((time.perf_counter() - read_started) * 1000.0)
            if not ok:
                break
            read_idx += 1
            if (read_idx - 1) % stride != 0:
                continue
            convert_started = time.perf_counter()
            frame_rgb = bgr_to_rgb(frame_bgr)
            convert_times.append((time.perf_counter() - convert_started) * 1000.0)
            if forced_actions is not None:
                state = dict(state)
                state["paper_forced_action"] = forced_actions.get(processed + 1, "edge")
                frame_args = args[:-3] + (state,)
            annotated_rgb, last_result, last_status, state = process_frame(
                frame_rgb, *frame_args, input_kind="video"
            )
            frame_args = args[:-3] + (state,)
            if annotated_rgb is None:
                continue
            annotated_bgr = rgb_to_bgr(annotated_rgb)
            if writer is None:
                h, w = annotated_bgr.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), out_fps, (w, h))
            write_started = time.perf_counter()
            writer.write(annotated_bgr)
            write_times.append((time.perf_counter() - write_started) * 1000.0)
            processed += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    last_result = dict(last_result)
    last_result["video_frames_processed"] = processed
    last_result["video_stride"] = stride
    last_result["video_output"] = str(out_path)
    last_result["video_total_ms"] = round((time.perf_counter() - video_started) * 1000.0, 3)
    last_result["video_read_ms_avg"] = round(average_ms(read_times), 3)
    last_result["video_input_convert_ms_avg"] = round(average_ms(convert_times), 3)
    last_result["video_write_ms_avg"] = round(average_ms(write_times), 3)
    status = f"Processed {processed} video frames. Output: {out_path}"
    if paper_plan is not None:
        actual_request_bytes = int(
            round(float(state.get("payload_kb_total", 0.0)) * 1024.0)
        ) - starting_payload_bytes
        capacity_bytes = float(paper_plan["capacity_bytes"])
        if actual_request_bytes > capacity_bytes + 1.0:
            raise RuntimeError(
                f"Paper Fixed-byte capacity exceeded during execution: {actual_request_bytes}>{capacity_bytes}"
            )
        paper_result = {
            **paper_plan,
            "actual_request_bytes": actual_request_bytes,
            "actual_capacity_utilization": actual_request_bytes / capacity_bytes if capacity_bytes else 0.0,
        }
        last_result["paper_fixed_byte"] = paper_result
        state = dict(state)
        state["paper_fixed_byte"] = paper_result
        counts = paper_plan["action_counts"]
        status += (
            f"\nPaper Fixed-byte {float(paper_plan['budget']):.0%}: "
            f"{actual_request_bytes / 1024.0:.1f}/{capacity_bytes / 1024.0:.1f} KB, "
            f"Edge/O/Q/R={counts['edge']}/{counts['original']}/{counts['q90']}/{counts['current@640']}"
        )
    return str(out_path) if processed else None, last_result, status + "\n\n" + last_status, state


def process_video_dispatch(video_file: Any, *args: Any) -> tuple[str | None, dict[str, Any], str, dict[str, Any]]:
    """Dispatch an uploaded video to the exact paper scheduler or the legacy online path."""
    if len(args) < 5:
        return process_video_file(video_file, *args)
    frame_args = list(args[:-5])
    scheduler_mode, budget_value, max_frames, stride, state = args[-5:]
    if str(scheduler_mode) != VIDEO_SCHEDULER_PAPER:
        return process_video_file(video_file, *frame_args, max_frames, stride, state)
    try:
        runtime = paper_video_runtime_args(frame_args)
        candidates = collect_paper_video_candidates(
            video_file,
            runtime,
            max_frames=max(1, int(max_frames)),
            stride=max(1, int(stride)),
        )
        plan = schedule_fixed_byte(candidates, parse_paper_budget(budget_value))
        return process_video_file(
            video_file,
            *runtime,
            max_frames,
            stride,
            new_state(),
            forced_actions=plan["actions"],
            paper_plan=plan,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        clean_state = state or new_state()
        return None, {"status": "paper_video_failed", "error": str(exc)}, str(exc), clean_state


def build_app(args: argparse.Namespace) -> gr.Blocks:
    """Build the Gradio app."""
    global RUN_SAVE_DIR
    RUN_SAVE_DIR = Path(args.save_dir)
    edge_stem = Path(args.edge_model).stem
    edge_label = f"YOLO{edge_stem[4:]}" if edge_stem.lower().startswith("yolo") else edge_stem
    css = """
    .edge-cloud-actions button {
        min-height: 48px !important;
        padding: 8px 14px !important;
        white-space: nowrap !important;
        font-weight: 600 !important;
    }
    .edge-cloud-actions img {
        width: 20px !important;
        height: 20px !important;
        object-fit: contain !important;
    }
    .edge-cloud-title h1 {
        margin: 0 0 4px 0;
        font-size: 24px;
    }
    .edge-cloud-panel {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
    }
    .edge-cloud-topology {
        border-left: 4px solid #0f766e;
        padding: 8px 12px;
        background: var(--block-background-fill);
    }
    .edge-cloud-status {
        border: 1px solid var(--border-color-primary);
        border-radius: 8px;
        padding: 8px 12px;
    }
    """
    with gr.Blocks(title="YOLO26 Edge-Cloud Demo", css=css) as demo:
        gr.Markdown(
            "# YOLO26 端云上传学习器演示\n本机运行 CAVS-OQR 上传决策，服务器只处理被选中的整图或 Current ROI。",
            elem_classes=["edge-cloud-title"],
        )
        gr.Markdown(
            f"**本机摄像头/文件**　→　**{edge_label} 端侧检测**　→　**CAVS-OQR 四动作决策**　→　"
            "**远程 YOLO26x 复核**　→　**Current@640 + MAUR 融合**",
            elem_classes=["edge-cloud-topology"],
        )
        policy_status_md = gr.Markdown(
            policy_summary_markdown(args.policy_path),
            elem_classes=["edge-cloud-status"],
            sanitize_html=True,
        )
        session_state = gr.State(new_state())

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Camera"):
                        webcam_frame = gr.Image(
                            label="Camera",
                            sources=["webcam"],
                            type="numpy",
                            streaming=True,
                            height=320,
                        )
                    with gr.Tab("Image"):
                        upload_frame = gr.Image(label="Upload Image", sources=["upload"], type="numpy", height=320)
                    with gr.Tab("Video"):
                        video_file = gr.Video(label="Upload Video", sources=["upload"], height=260)
                        video_scheduler = gr.Radio(
                            [VIDEO_SCHEDULER_PAPER, VIDEO_SCHEDULER_ONLINE],
                            value=VIDEO_SCHEDULER_PAPER,
                            label="Video Scheduler",
                            info="Paper模式先全局调度再执行；Online模式逐帧使用代价lambda。",
                        )
                        video_budget = gr.Radio(
                            ["10%", "25%", "40%"],
                            value="10%",
                            label="Fixed-byte Budget",
                        )
                        with gr.Row():
                            video_max_frames = gr.Slider(1, 600, value=120, step=1, label="Max Frames")
                            video_stride = gr.Slider(1, 10, value=1, step=1, label="Stride")
                        video_btn = gr.Button("Run Video", variant="primary", icon=asset_path("play.svg"))
                with gr.Row(elem_classes=["edge-cloud-actions"]):
                    run_btn = gr.Button("运行当前输入", variant="primary", icon=asset_path("play.svg"))
                    warmup_btn = gr.Button("预热模型", icon=asset_path("bolt.svg"))
                    reset_btn = gr.Button("重置统计", icon=asset_path("reset.svg"))

                mode = gr.Radio(
                    choices=[("仅端侧", MODE_EDGE), ("仅云端", MODE_CLOUD), ("学习器协同", MODE_COLLAB)],
                    value=MODE_COLLAB,
                    label="演示模式",
                    info="协同模式由学习器逐帧选择动作；选 Edge 时云端在线但不会产生请求。",
                )
                with gr.Row():
                    cloud_url = gr.Textbox(value=args.cloud_url, label="云端 /infer 地址", scale=4)
                    health_btn = gr.Button("检查云端", scale=1)
                cloud_status_md = gr.Markdown(
                    cloud_health_markdown(args.cloud_url, args.cloud_timeout),
                    elem_classes=["edge-cloud-status"],
                    sanitize_html=True,
                )

                with gr.Accordion("CAVS Current+MAUR 工作点", open=True):
                    conf = gr.Slider(0.05, 0.95, value=args.conf, step=0.01, label="Edge Conf")
                    cloud_conf = gr.Slider(0.05, 0.95, value=args.cloud_conf, step=0.01, label="Cloud Conf")
                    enforce_online_budget = gr.Checkbox(
                        value=not args.disable_online_budget,
                        label="Camera Hard Byte Budget",
                    )
                    online_budget = gr.Radio(
                        ["10%", "25%", "40%"],
                        value=f"{args.online_fixed_byte_budget * 100:g}%",
                        label="Online Fixed-byte Budget",
                    )
                    cost_weight = gr.Slider(
                        0.0,
                        0.2,
                        value=args.learned_cost_weight,
                        step=0.005,
                        label="在线上传代价 λ",
                        info="动作效用 = CAVS 预测价值 − λ × 相对上传字节。",
                    )

                with gr.Accordion("旧版 Dual-Head 兼容参数", open=False):
                    frame_thr = gr.Slider(0.0, 1.0, value=args.learned_frame_threshold, step=0.01, label="Frame Threshold")
                    roi_thr = gr.Slider(0.0, 1.0, value=args.learned_roi_threshold, step=0.01, label="ROI Threshold")
                    max_roi = gr.Slider(1, 3, value=args.learned_max_roi_per_frame, step=1, label="Max ROI Per Frame")

                with gr.Accordion("高级运行参数", open=False):
                    edge_model = gr.Textbox(value=args.edge_model, label="端侧模型")
                    device = gr.Textbox(value=args.device, label="端侧设备")
                    policy_path = gr.Textbox(value=args.policy_path, label="上传学习器清单")
                    imgsz = gr.Slider(320, 960, value=args.imgsz, step=32, label="Edge ImgSz")
                    cloud_timeout = gr.Slider(0.05, 10.0, value=args.cloud_timeout, step=0.05, label="Cloud Timeout (s)")
                    cloud_imgsz = gr.Slider(320, 960, value=args.cloud_imgsz, step=32, label="Cloud ImgSz")
                    jpeg_quality = gr.Slider(40, 95, value=args.jpeg_quality, step=1, label="JPEG Quality")
                    class_selector = gr.Dropdown(
                        choices=["all", *COCO_CLASS_NAMES],
                        value=default_class_selection(args.classes),
                        multiselect=True,
                        label="Classes",
                    )
                    enable_adaptive_cloud = gr.Checkbox(value=False, label="Adaptive Cloud ImgSz")
                    low_imgsz = gr.Slider(320, 768, value=args.learned_cloud_imgsz_low, step=32, label="Cloud ImgSz Low")
                    high_imgsz = gr.Slider(320, 960, value=args.learned_cloud_imgsz_high, step=32, label="Cloud ImgSz High")
                    switch_thr = gr.Slider(0.0, 1.0, value=args.learned_cloud_imgsz_switch, step=0.01, label="Cloud ImgSz Switch")

                with gr.Accordion("Temporal / ROI", open=False):
                    enable_temporal = gr.Checkbox(value=True, label="Temporal Memory")
                    temporal_iou = gr.Slider(0.1, 0.9, value=args.temporal_iou, step=0.01, label="Temporal IoU")
                    temporal_ttl = gr.Slider(5, 120, value=args.temporal_ttl_frames, step=1, label="Temporal TTL")
                    temporal_hits = gr.Slider(1, 10, value=args.temporal_stable_hits, step=1, label="Stable Hits")
                    temporal_gap = gr.Slider(1, 60, value=args.temporal_reupload_gap, step=1, label="Reupload Gap")
                    temporal_drop = gr.Slider(0.0, 0.5, value=args.temporal_conf_drop, step=0.01, label="Confidence Drop")
                    temporal_ratio = gr.Slider(0.1, 1.0, value=args.temporal_frame_stable_ratio, step=0.01, label="Frame Stable Ratio")
                    enable_adaptive_roi = gr.Checkbox(value=True, label="Adaptive ROI")
                    roi_expand = gr.Slider(1.0, 2.5, value=args.roi_expand, step=0.05, label="ROI Expand")
                    roi_small_gain = gr.Slider(1.0, 3.0, value=args.roi_small_gain, step=0.05, label="ROI Small Gain")
                    roi_edge_gain = gr.Slider(1.0, 3.0, value=args.roi_edge_gain, step=0.05, label="ROI Edge Gain")
                    roi_large_gain = gr.Slider(1.0, 2.5, value=args.roi_large_gain, step=0.05, label="ROI Large Gain")
                    iou_match = gr.Slider(0.1, 0.9, value=args.iou_match, step=0.01, label="Fusion IoU")
                    fused_nms = gr.Slider(0.1, 0.95, value=args.fused_nms_iou, step=0.01, label="Fused NMS IoU")

            with gr.Column(scale=1):
                output_frame = gr.Image(label="融合检测结果", type="numpy", height=440)
                status_md = gr.Markdown(elem_classes=["edge-cloud-panel"], sanitize_html=True)
                with gr.Accordion("逐帧完整指标", open=False):
                    status_json = gr.JSON(label="当前帧 JSON")
                with gr.Accordion("视频输出", open=False):
                    output_video = gr.Video(label="标注视频", height=220)

        event_inputs = [
            webcam_frame,
            upload_frame,
            mode,
            cloud_url,
            edge_model,
            device,
            policy_path,
            imgsz,
            conf,
            cloud_conf,
            cloud_timeout,
            cloud_imgsz,
            jpeg_quality,
            class_selector,
            frame_thr,
            roi_thr,
            cost_weight,
            max_roi,
            enable_adaptive_cloud,
            low_imgsz,
            high_imgsz,
            switch_thr,
            enable_temporal,
            temporal_iou,
            temporal_ttl,
            temporal_hits,
            temporal_gap,
            temporal_drop,
            temporal_ratio,
            enable_adaptive_roi,
            roi_expand,
            roi_small_gain,
            roi_edge_gain,
            roi_large_gain,
            iou_match,
            fused_nms,
            enforce_online_budget,
            online_budget,
            session_state,
        ]
        event_outputs = [output_frame, status_json, status_md, session_state]
        video_inputs = [
            video_file,
            *event_inputs[2:-1],
            video_scheduler,
            video_budget,
            video_max_frames,
            video_stride,
            session_state,
        ]
        video_outputs = [output_video, status_json, status_md, session_state]

        run_btn.click(process_any_frame, inputs=event_inputs, outputs=event_outputs, show_progress="minimal", queue=False)
        upload_frame.change(process_any_frame, inputs=event_inputs, outputs=event_outputs, show_progress="hidden", queue=False)
        video_btn.click(process_video_dispatch, inputs=video_inputs, outputs=video_outputs, show_progress="minimal", queue=True)
        webcam_frame.stream(
            process_any_frame,
            inputs=event_inputs,
            outputs=event_outputs,
            show_progress="hidden",
            trigger_mode="always_last",
            concurrency_limit=1,
            queue=False,
        )
        warmup_btn.click(
            warmup_demo,
            inputs=[edge_model, device, cloud_url, cloud_timeout, imgsz, conf, jpeg_quality, cloud_imgsz],
            outputs=status_md,
            show_progress="minimal",
            queue=False,
        )
        reset_btn.click(reset_memory, outputs=[status_json, status_md, session_state], show_progress="hidden", queue=False)
        health_btn.click(
            cloud_health_markdown,
            inputs=[cloud_url, cloud_timeout],
            outputs=cloud_status_md,
            show_progress="hidden",
            queue=False,
        )
        policy_path.change(
            policy_summary_markdown,
            inputs=[policy_path],
            outputs=policy_status_md,
            show_progress="hidden",
            queue=False,
        )

    return demo


def main() -> None:
    """Launch the Gradio app."""
    args = parse_args()
    demo = build_app(args)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=args.inbrowser,
        show_api=False,
        enable_monitoring=False,
        max_file_size=args.max_file_size,
    )


if __name__ == "__main__":
    main()
