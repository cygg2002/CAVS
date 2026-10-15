# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""YOLO26s edge client with selective asynchronous cloud verification."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import cv2
import requests
from ultralytics import YOLO

from fusion import fuse_detections
from learned_uploader import (
    frame_features,
    image_context,
    load_policy,
    policy_decision,
    roi_features,
)
from protocol import box_iou, boxes_from_result, bytes_to_b64, crop_image, encode_jpeg, expand_xyxy
from upload_strategies import (
    adaptive_roi_gain,
    boxes_in_region,
    low_conf_proposals,
    parse_grid,
    should_complexity_probe,
    stability_probe_score,
    tile_candidates,
)

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def percentile(values: list[float], q: float) -> float:
    """Return a percentile value without requiring NumPy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def edge_area_signal(xyxy: list[float], shape: tuple[int, int]) -> float:
    """Signal if a box is near the image boundary."""
    h, w = shape[:2]
    margin_x, margin_y = w * 0.04, h * 0.04
    x1, y1, x2, y2 = xyxy
    return 1.0 if x1 <= margin_x or y1 <= margin_y or x2 >= w - margin_x or y2 >= h - margin_y else 0.0


def small_object_signal(xyxy: list[float], shape: tuple[int, int]) -> float:
    """Signal if a box covers a small fraction of the frame."""
    h, w = shape[:2]
    x1, y1, x2, y2 = xyxy
    area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, float(w * h))
    return max(0.0, min(1.0, (0.04 - area_ratio) / 0.04))


def low_conf_signal(conf: float, target: float = 0.65) -> float:
    """Signal if an edge detection is below the desired confidence target."""
    return max(0.0, min(1.0, (target - float(conf)) / target))


def temporal_signal(box: dict[str, Any], previous_boxes: list[dict[str, Any]]) -> float:
    """Signal if a detection has no stable same-class predecessor."""
    from protocol import box_iou

    same_cls = [prev for prev in previous_boxes if int(prev["cls"]) == int(box["cls"])]
    if not same_cls:
        return 1.0
    best = max(box_iou(box["xyxy"], prev["xyxy"]) for prev in same_cls)
    return max(0.0, min(1.0, (0.5 - best) / 0.5))


def detection_upload_score(
    box: dict[str, Any],
    shape: tuple[int, int],
    previous_boxes: list[dict[str, Any]],
    use_temporal: bool = True,
) -> tuple[float, list[str]]:
    """Compute an interpretable detection uncertainty score."""
    signals = {
        "low_conf": low_conf_signal(float(box["conf"])),
        "small_object": small_object_signal(box["xyxy"], shape),
        "edge_area": edge_area_signal(box["xyxy"], shape),
        "temporal_instability": temporal_signal(box, previous_boxes) if use_temporal else 0.0,
        "probe": 0.0,
    }
    score = (
        0.35 * signals["low_conf"]
        + 0.20 * signals["small_object"]
        + 0.15 * signals["edge_area"]
        + 0.20 * signals["temporal_instability"]
        + 0.10 * signals["probe"]
    )
    reasons = [name for name, value in signals.items() if value >= 0.5]
    return round(float(score), 4), reasons or ["mixed_uncertainty"]


def post_to_cloud(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Send one upload payload to the cloud verifier."""
    started = time.perf_counter()
    metadata = payload["metadata"]
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        data = response.json()
        data["http_status"] = response.status_code
        data["roundtrip_ms"] = round(roundtrip_ms, 3)
        data["request_metadata"] = metadata
        return data
    except Exception as exc:  # noqa: BLE001 - keep the edge loop alive on network failures
        return {
            "status": "error",
            "error": str(exc),
            "roundtrip_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "request_metadata": metadata,
        }


def edge_boxes_for_fusion(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Use only the ROI target box for ROI fusion; use all edge boxes for full-frame probes."""
    if metadata.get("payload_type") in {"roi", "proposal"} and metadata.get("target_edge_box"):
        return [metadata["target_edge_box"]]
    if metadata.get("payload_type") == "tile" and metadata.get("crop_xyxy"):
        return boxes_in_region(metadata.get("edge_boxes", []), metadata["crop_xyxy"])
    return metadata.get("edge_boxes", [])


def make_payload(
    frame,
    frame_id: int,
    payload_type: str,
    reason: list[str],
    edge_boxes: list[dict[str, Any]],
    score: float,
    jpeg_quality: int,
    crop_xyxy: list[int] | None = None,
    target_edge_box: dict[str, Any] | None = None,
    cloud_imgsz: int | None = None,
) -> tuple[dict[str, Any], int]:
    """Create a cloud upload payload and return it with raw JPEG byte length."""
    image = crop_image(frame, crop_xyxy) if crop_xyxy else frame
    image_bytes = encode_jpeg(image, jpeg_quality)
    metadata = {
        "task_id": uuid.uuid4().hex,
        "frame_id": frame_id,
        "timestamp": time.time(),
        "payload_type": payload_type,
        "reason": reason,
        "upload_score": score,
        "crop_xyxy": crop_xyxy,
        "orig_shape": list(frame.shape[:2]),
        "edge_boxes": edge_boxes,
        "target_edge_box": target_edge_box,
        "payload_bytes": len(image_bytes),
    }
    if cloud_imgsz:
        metadata["cloud_imgsz"] = int(cloud_imgsz)
    return {"metadata": metadata, "image_b64": bytes_to_b64(image_bytes)}, len(image_bytes)


def select_cloud_imgsz(score: float, args: argparse.Namespace, adaptive: bool = False) -> int | None:
    """Select a per-request cloud inference size, if the client wants to override the server default."""
    if adaptive and args.enable_learned_adaptive_cloud_imgsz:
        if score >= args.learned_cloud_imgsz_switch:
            return int(args.learned_cloud_imgsz_high)
        return int(args.learned_cloud_imgsz_low)
    return int(args.cloud_imgsz) if args.cloud_imgsz > 0 else None


def draw_boxes(frame, boxes: list[dict[str, Any]], color: tuple[int, int, int], label_prefix: str) -> None:
    """Draw boxes in-place for optional preview video."""
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box["xyxy"]]
        label = f"{label_prefix}:{box.get('name', box.get('cls'))} {float(box.get('conf', 0.0)):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def normalize_capture_source(source: str) -> int | str:
    """Interpret bare integers as webcam device ids instead of filenames."""
    text = str(source).strip()
    if text.isdigit() and not Path(text).exists():
        return int(text)
    return source


def match_track_index(
    tracks: list[dict[str, Any]],
    box: dict[str, Any],
    iou_threshold: float,
    require_same_class: bool = True,
    exclude: set[int] | None = None,
) -> tuple[int | None, float]:
    """Find the best cached track for a detection."""
    exclude = exclude or set()
    best_idx = None
    best_iou = 0.0
    for idx, track in enumerate(tracks):
        if idx in exclude:
            continue
        if require_same_class and int(track.get("cls", -1)) != int(box.get("cls", -2)):
            continue
        iou = box_iou(track.get("xyxy", track.get("verified_xyxy", box["xyxy"])), box["xyxy"])
        if iou >= iou_threshold and iou > best_iou:
            best_idx = idx
            best_iou = iou
    return best_idx, best_iou


def update_temporal_tracks(
    tracks: list[dict[str, Any]],
    edge_boxes: list[dict[str, Any]],
    frame_id: int,
    iou_threshold: float,
    ttl_frames: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Update lightweight IoU tracks from current edge boxes."""
    tracks = [track for track in tracks if frame_id - int(track.get("last_seen_frame", frame_id)) <= ttl_frames]
    assignments: dict[int, dict[str, Any]] = {}
    used: set[int] = set()
    for edge_index, box in enumerate(edge_boxes):
        track_idx, _ = match_track_index(tracks, box, iou_threshold, require_same_class=True, exclude=used)
        if track_idx is None:
            track = {
                "track_id": uuid.uuid4().hex[:8],
                "cls": int(box["cls"]),
                "xyxy": [float(v) for v in box["xyxy"]],
                "last_seen_frame": frame_id,
                "last_conf": float(box.get("conf", 0.0)),
                "stable_hits": 1,
                "cloud_verified": False,
                "last_cloud_frame": -10**9,
                "last_cloud_conf": 0.0,
                "verified_xyxy": [float(v) for v in box["xyxy"]],
                "verified_cls": int(box["cls"]),
            }
            tracks.append(track)
            assignments[edge_index] = track
            continue
        track = tracks[track_idx]
        used.add(track_idx)
        if frame_id - int(track.get("last_seen_frame", frame_id)) <= 2:
            track["stable_hits"] = int(track.get("stable_hits", 1)) + 1
        else:
            track["stable_hits"] = 1
        track["xyxy"] = [float(v) for v in box["xyxy"]]
        track["cls"] = int(box["cls"])
        track["last_seen_frame"] = frame_id
        track["last_conf"] = float(box.get("conf", 0.0))
        assignments[edge_index] = track
    return tracks, assignments


def temporal_skip_roi_upload(
    box: dict[str, Any],
    track: dict[str, Any] | None,
    frame_id: int,
    iou_threshold: float,
    stable_hits: int,
    reupload_gap: int,
    conf_drop_threshold: float,
) -> bool:
    """Return True when a recently cloud-verified stable target does not need another upload."""
    if not track or not bool(track.get("cloud_verified", False)):
        return False
    if int(track.get("stable_hits", 0)) < stable_hits:
        return False
    if frame_id - int(track.get("last_cloud_frame", -10**9)) > reupload_gap:
        return False
    if int(track.get("verified_cls", track.get("cls", -1))) != int(box.get("cls", -2)):
        return False
    if box_iou(track.get("verified_xyxy", track.get("xyxy", box["xyxy"])), box["xyxy"]) < iou_threshold:
        return False
    baseline_conf = float(track.get("last_cloud_conf", track.get("last_conf", 0.0)))
    return baseline_conf - float(box.get("conf", 0.0)) <= conf_drop_threshold


def temporal_skip_frame_upload(
    edge_boxes: list[dict[str, Any]],
    box_tracks: dict[int, dict[str, Any]],
    frame_id: int,
    args: argparse.Namespace,
) -> bool:
    """Suppress redundant full-frame uploads when the visible scene is already cloud-verified and stable."""
    if not edge_boxes:
        return False
    stable = 0
    for edge_index, box in enumerate(edge_boxes):
        if temporal_skip_roi_upload(
            box,
            box_tracks.get(edge_index),
            frame_id,
            args.temporal_iou,
            args.temporal_stable_hits,
            args.temporal_reupload_gap,
            args.temporal_conf_drop,
        ):
            stable += 1
    return (stable / max(1, len(edge_boxes))) >= args.temporal_frame_stable_ratio


def mark_temporal_memory_from_response(
    tracks: list[dict[str, Any]],
    metadata: dict[str, Any],
    cloud_boxes: list[dict[str, Any]],
    frame_id: int,
    args: argparse.Namespace,
) -> None:
    """Promote matching tracks to cloud-verified state and seed new tracks for cloud-only misses."""
    matched_cloud: set[int] = set()
    target_boxes = edge_boxes_for_fusion(metadata)
    for edge_box in target_boxes:
        track_idx, _ = match_track_index(tracks, edge_box, args.temporal_iou, require_same_class=False)
        if track_idx is None:
            continue
        track = tracks[track_idx]
        best_cloud_idx = None
        best_iou = 0.0
        for cloud_idx, cloud_box in enumerate(cloud_boxes):
            iou = box_iou(edge_box["xyxy"], cloud_box["xyxy"])
            if iou >= args.temporal_iou and iou > best_iou:
                best_iou = iou
                best_cloud_idx = cloud_idx
        if best_cloud_idx is not None:
            cloud_box = cloud_boxes[best_cloud_idx]
            track["verified_xyxy"] = [float(v) for v in cloud_box["xyxy"]]
            track["verified_cls"] = int(cloud_box["cls"])
            track["last_cloud_conf"] = float(cloud_box.get("conf", track.get("last_conf", 0.0)))
            matched_cloud.add(best_cloud_idx)
        else:
            track["verified_xyxy"] = [float(v) for v in edge_box["xyxy"]]
            track["verified_cls"] = int(edge_box["cls"])
            track["last_cloud_conf"] = float(edge_box.get("conf", track.get("last_conf", 0.0)))
        track["cloud_verified"] = True
        track["last_cloud_frame"] = frame_id
        track["stable_hits"] = max(int(track.get("stable_hits", 1)), int(args.temporal_stable_hits))
    for cloud_idx, cloud_box in enumerate(cloud_boxes):
        if cloud_idx in matched_cloud:
            continue
        tracks.append(
            {
                "track_id": uuid.uuid4().hex[:8],
                "cls": int(cloud_box["cls"]),
                "xyxy": [float(v) for v in cloud_box["xyxy"]],
                "last_seen_frame": frame_id,
                "last_conf": float(cloud_box.get("conf", 0.0)),
                "stable_hits": int(args.temporal_stable_hits),
                "cloud_verified": True,
                "last_cloud_frame": frame_id,
                "last_cloud_conf": float(cloud_box.get("conf", 0.0)),
                "verified_xyxy": [float(v) for v in cloud_box["xyxy"]],
                "verified_cls": int(cloud_box["cls"]),
            }
        )


def annotate_status(frame, lines: list[str]) -> None:
    """Draw compact status text in-place."""
    y = 20
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        y += 22


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26s.pt", help="Edge detector model.")
    parser.add_argument("--source", default=r"F:\IDE\codex\yoloe-new\datasets\bear.mp4", help="Video source path.")
    parser.add_argument("--cloud-url", default="http://127.0.0.1:8765/infer", help="Cloud verifier endpoint.")
    parser.add_argument("--save-dir", default="runs/edge_cloud/bear", help="Directory for logs and outputs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Edge inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Edge confidence threshold.")
    parser.add_argument("--device", default="", help="Ultralytics device string, e.g. 0 or cpu.")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum frames to process. Use 0 for all frames.")
    parser.add_argument("--vid-stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--upload-threshold", type=float, default=0.45, help="Detection upload threshold.")
    parser.add_argument("--max-uploads-per-sec", type=int, default=5, help="Token-bucket upload limit.")
    parser.add_argument("--max-roi-per-frame", type=int, default=2, help="Maximum ROI uploads per processed frame.")
    parser.add_argument("--queue-size", type=int, default=16, help="Maximum pending cloud requests.")
    parser.add_argument("--workers", type=int, default=2, help="Async upload worker threads.")
    parser.add_argument("--cloud-timeout", type=float, default=3.0, help="Cloud request timeout in seconds.")
    parser.add_argument("--cloud-imgsz", type=int, default=0, help="Requested cloud inference size. Use 0 for server default.")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality for uploaded payloads.")
    parser.add_argument("--roi-expand", type=float, default=1.35, help="ROI expansion factor.")
    parser.add_argument("--keyframe-interval", type=int, default=30, help="Periodic full-frame probe interval.")
    parser.add_argument("--empty-probe-frames", type=int, default=15, help="Consecutive empty detections before probe.")
    parser.add_argument("--enable-complexity-probe", action="store_true", help="Upload full frames for complex sparse scenes.")
    parser.add_argument("--complexity-threshold", type=float, default=0.18, help="Complex-scene edge density threshold.")
    parser.add_argument("--empty-complexity-threshold", type=float, default=0.08, help="Empty-scene edge density threshold.")
    parser.add_argument("--complexity-max-detections", type=int, default=1, help="Max detections for sparse complexity probe.")
    parser.add_argument("--complexity-coverage-threshold", type=float, default=0.02, help="Max coverage for low-coverage probe.")
    parser.add_argument("--enable-learned-policy", action="store_true", help="Use a trained lightweight upload policy.")
    parser.add_argument("--learned-policy-model", default="", help="Path to learned upload policy JSON.")
    parser.add_argument("--learned-roi-threshold", type=float, default=0.50, help="Minimum learned ROI utility probability.")
    parser.add_argument("--learned-frame-threshold", type=float, default=0.50, help="Minimum learned full-frame utility probability.")
    parser.add_argument("--learned-cost-weight", type=float, default=0.02, help="Penalty per 100 KB in learned policy decision score.")
    parser.add_argument("--learned-max-roi-per-frame", type=int, default=1, help="Maximum learned ROI uploads per frame.")
    parser.add_argument("--enable-learned-adaptive-cloud-imgsz", action="store_true", help="Choose cloud inference size from learned utility probability.")
    parser.add_argument("--learned-cloud-imgsz-low", type=int, default=416, help="Cloud inference size for marginal learned uploads.")
    parser.add_argument("--learned-cloud-imgsz-high", type=int, default=512, help="Cloud inference size for high-utility learned uploads.")
    parser.add_argument("--learned-cloud-imgsz-switch", type=float, default=0.75, help="Use high cloud size above this learned probability.")
    parser.add_argument("--enable-stability-probe", action="store_true", help="Upload frames whose edge detections are unstable across input scales.")
    parser.add_argument("--stability-imgsz", type=int, default=384, help="Secondary edge inference size for scale-stability probing.")
    parser.add_argument("--stability-threshold", type=float, default=0.45, help="Minimum scale-instability score for full-frame upload.")
    parser.add_argument("--stability-iou", type=float, default=0.5, help="IoU threshold for scale-stability matching.")
    parser.add_argument("--enable-lowconf-proposal", action="store_true", help="Upload suppressed low-confidence edge proposals.")
    parser.add_argument("--proposal-conf", type=float, default=0.05, help="Internal edge confidence for proposal mining.")
    parser.add_argument("--proposal-score-threshold", type=float, default=0.55, help="Minimum low-confidence proposal score.")
    parser.add_argument("--max-proposals-per-frame", type=int, default=2, help="Maximum suppressed proposals to upload.")
    parser.add_argument("--enable-tile-probe", action="store_true", help="Upload high-complexity uncovered tiles.")
    parser.add_argument("--tile-grid", default="2x2", help="Tile grid as COLSxROWS, e.g. 2x2.")
    parser.add_argument("--max-tiles-per-frame", type=int, default=1, help="Maximum tile uploads per frame/image.")
    parser.add_argument("--tile-score-threshold", type=float, default=0.12, help="Minimum tile upload score.")
    parser.add_argument("--enable-adaptive-roi", action="store_true", help="Adapt ROI expansion by object scale and edge proximity.")
    parser.add_argument("--roi-small-gain", type=float, default=1.6, help="ROI gain for small objects.")
    parser.add_argument("--roi-edge-gain", type=float, default=1.45, help="ROI gain for edge-near objects.")
    parser.add_argument("--roi-large-gain", type=float, default=1.15, help="ROI gain for large objects.")
    parser.add_argument("--iou-match", type=float, default=0.5, help="IoU threshold for edge/cloud matching.")
    parser.add_argument("--save-video", action="store_true", help="Save an edge preview video.")
    parser.add_argument("--show", action="store_true", help="Show a live preview window for webcam or video demos.")
    parser.add_argument("--enable-temporal-memory", action="store_true", help="Use lightweight track memory to suppress repeated uploads in videos.")
    parser.add_argument("--temporal-iou", type=float, default=0.5, help="IoU threshold for temporal track association.")
    parser.add_argument("--temporal-ttl-frames", type=int, default=45, help="Drop cached tracks after this many unseen frames.")
    parser.add_argument("--temporal-stable-hits", type=int, default=3, help="Track hits required before a cloud-verified target is considered stable.")
    parser.add_argument("--temporal-reupload-gap", type=int, default=15, help="Minimum frames before re-uploading a stable cloud-verified target.")
    parser.add_argument("--temporal-conf-drop", type=float, default=0.12, help="Re-upload when edge confidence drops more than this from the last cloud-confirmed confidence.")
    parser.add_argument("--temporal-frame-stable-ratio", type=float, default=0.7, help="Suppress full-frame uploads when this fraction of current boxes are stable and cloud-verified.")
    return parser.parse_args()


def collect_image_paths(source: str) -> list[Path]:
    """Collect image paths from a single image file, a manifest, or an image directory."""
    path = Path(source)
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if path.is_file() and path.suffix.lower() in {".txt", ".lst"}:
        image_paths: list[Path] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            entry_path = Path(entry)
            if not entry_path.is_absolute():
                entry_path = (path.parent / entry_path).resolve()
            image_paths.append(entry_path)
        return image_paths
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return []


def iter_frames(source: str, image_paths: list[Path], cap: cv2.VideoCapture | None, vid_stride: int):
    """Yield frame_id and BGR image from either image files or a video capture."""
    stride = max(1, vid_stride)
    if image_paths:
        for frame_id, image_path in enumerate(image_paths):
            if frame_id % stride != 0:
                continue
            frame = cv2.imread(str(image_path))
            if frame is not None:
                yield frame_id, frame
        return

    raw_frame_id = -1
    while cap is not None:
        ok, frame = cap.read()
        if not ok:
            break
        raw_frame_id += 1
        if raw_frame_id % stride != 0:
            continue
        yield raw_frame_id, frame


def main() -> None:
    """Run the edge client."""
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    events_path = save_dir / "events.jsonl"
    summary_path = save_dir / "summary.json"

    image_paths = collect_image_paths(args.source)
    cap = None
    if image_paths:
        first_frame = cv2.imread(str(image_paths[0]))
        if first_frame is None:
            raise FileNotFoundError(f"Could not read first image source: {image_paths[0]}")
        fps = 30.0
        height, width = first_frame.shape[:2]
        print(f"Using image source with {len(image_paths)} images: {args.source}")
    else:
        cap = cv2.VideoCapture(normalize_capture_source(args.source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video or image source: {args.source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.save_video:
        preview_path = save_dir / "preview.mp4"
        writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / max(1, args.vid_stride), (width, height))

    print(f"Loading edge model: {args.model}")
    model = YOLO(args.model)
    learned_policy = None
    if args.enable_learned_policy:
        if not args.learned_policy_model:
            raise ValueError("--learned-policy-model is required when --enable-learned-policy is set")
        learned_policy = load_policy(args.learned_policy_model)

    stats = Counter()
    edge_latencies: list[float] = []
    cloud_latencies: list[float] = []
    cloud_roundtrips: list[float] = []
    payload_sizes: list[int] = []
    upload_times: deque[float] = deque()
    pending = set()
    previous_boxes: list[dict[str, Any]] = []
    consecutive_empty = 0
    temporal_tracks: list[dict[str, Any]] = []
    frame_cache: dict[int, dict[str, Any]] = {}
    frame_cache_order: deque[int] = deque()
    latest_cloud_preview = None

    def cache_frame(frame_id: int, frame, edge_boxes: list[dict[str, Any]]) -> None:
        frame_cache[frame_id] = {"frame": frame.copy(), "edge_boxes": [dict(box) for box in edge_boxes]}
        frame_cache_order.append(frame_id)
        while len(frame_cache_order) > 32:
            stale = frame_cache_order.popleft()
            frame_cache.pop(stale, None)

    def can_upload() -> bool:
        now = time.time()
        while upload_times and now - upload_times[0] > 1.0:
            upload_times.popleft()
        return len(upload_times) < args.max_uploads_per_sec and len(pending) < args.queue_size

    def submit_payload(executor: ThreadPoolExecutor, payload: dict[str, Any], payload_bytes: int) -> None:
        if not can_upload():
            stats["uploads_skipped_limit"] += 1
            return
        upload_times.append(time.time())
        payload_sizes.append(payload_bytes)
        stats["uploads_submitted"] += 1
        stats[f"payload_{payload['metadata']['payload_type']}"] += 1
        requested_imgsz = payload["metadata"].get("cloud_imgsz")
        if requested_imgsz:
            stats[f"cloud_imgsz_{requested_imgsz}"] += 1
        future = executor.submit(post_to_cloud, args.cloud_url, payload, args.cloud_timeout)
        pending.add(future)

    def drain_done(timeout: float = 0.0) -> list[dict[str, Any]]:
        if not pending:
            return []
        done, _ = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
        records = []
        for future in done:
            pending.remove(future)
            records.append(future.result())
        return records

    with events_path.open("w", encoding="utf-8") as events_file, ThreadPoolExecutor(max_workers=args.workers) as executor:
        processed_frames = 0
        for raw_frame_id, frame in iter_frames(args.source, image_paths, cap, args.vid_stride):
            if args.max_frames and processed_frames >= args.max_frames:
                break
            processed_frames += 1
            stats["frames_processed"] += 1
            full_frame_submitted = False

            started = time.perf_counter()
            predict_conf = args.conf
            if args.enable_lowconf_proposal:
                predict_conf = min(args.conf, args.proposal_conf)
            predict_kwargs = {"source": frame, "imgsz": args.imgsz, "conf": predict_conf, "verbose": False}
            if args.device:
                predict_kwargs["device"] = args.device
            result = model.predict(**predict_kwargs)[0]
            edge_latency_ms = (time.perf_counter() - started) * 1000.0
            edge_latencies.append(edge_latency_ms)
            candidate_edge_boxes = boxes_from_result(result)
            edge_boxes = [box for box in candidate_edge_boxes if float(box.get("conf", 0.0)) >= args.conf]
            consecutive_empty = consecutive_empty + 1 if not edge_boxes else 0
            box_track_map: dict[int, dict[str, Any]] = {}
            frame_temporal_stable = False
            if args.enable_temporal_memory and not image_paths:
                temporal_tracks, box_track_map = update_temporal_tracks(
                    temporal_tracks, edge_boxes, raw_frame_id, args.temporal_iou, args.temporal_ttl_frames
                )
                stats["temporal_active_tracks"] = len(temporal_tracks)
                frame_temporal_stable = temporal_skip_frame_upload(edge_boxes, box_track_map, raw_frame_id, args)
            cache_frame(raw_frame_id, frame, edge_boxes)

            if learned_policy is not None:
                context = image_context(frame, edge_boxes)
                full_frame_kb = len(encode_jpeg(frame, args.jpeg_quality)) / 1024.0
                frame_eval = policy_decision(
                    learned_policy,
                    "frame",
                    frame_features(context, learned_policy.get("frame_features")),
                    full_frame_kb,
                    args.learned_cost_weight,
                )
                frame_prob = frame_eval["probability"]
                frame_expected_gain = frame_eval["expected_gain"]
                frame_decision = frame_eval["decision"]
                if args.enable_temporal_memory and not image_paths and frame_temporal_stable:
                    stats["temporal_frame_suppressed"] += 1
                elif frame_prob >= args.learned_frame_threshold and frame_decision > 0.0:
                    stats["full_frame_probe_submitted"] += 1
                    payload, payload_bytes = make_payload(
                        frame,
                        raw_frame_id,
                        "keyframe",
                        ["learned_full_frame"],
                        edge_boxes,
                        frame_prob,
                        args.jpeg_quality,
                        cloud_imgsz=select_cloud_imgsz(frame_prob, args, adaptive=True),
                    )
                    payload["metadata"]["learned_prob"] = round(float(frame_prob), 4)
                    payload["metadata"]["learned_expected_gain"] = round(float(frame_expected_gain), 4)
                    payload["metadata"]["learned_decision"] = round(float(frame_decision), 4)
                    submit_payload(executor, payload, payload_bytes)
                    full_frame_submitted = True

                if not full_frame_submitted:
                    learned_roi_candidates: list[
                        tuple[float, float, float, float, dict[str, Any], float, list[int]]
                    ] = []
                    for edge_index, box in enumerate(edge_boxes):
                        if args.enable_temporal_memory and not image_paths:
                            if temporal_skip_roi_upload(
                                box,
                                box_track_map.get(edge_index),
                                raw_frame_id,
                                args.temporal_iou,
                                args.temporal_stable_hits,
                                args.temporal_reupload_gap,
                                args.temporal_conf_drop,
                            ):
                                stats["temporal_roi_suppressed"] += 1
                                continue
                        roi_gain = (
                            adaptive_roi_gain(
                                box,
                                frame.shape[:2],
                                base_gain=args.roi_expand,
                                small_gain=args.roi_small_gain,
                                edge_gain=args.roi_edge_gain,
                                large_gain=args.roi_large_gain,
                            )
                            if args.enable_adaptive_roi
                            else args.roi_expand
                        )
                        crop_xyxy = expand_xyxy(box["xyxy"], frame.shape[:2], roi_gain)
                        x1, y1, x2, y2 = crop_xyxy
                        crop_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, float(frame.shape[0] * frame.shape[1]))
                        estimated_kb = max(1.0, full_frame_kb * crop_ratio)
                        roi_eval = policy_decision(
                            learned_policy,
                            "roi",
                            roi_features(box, frame.shape[:2], context, learned_policy.get("roi_features")),
                            estimated_kb,
                            args.learned_cost_weight,
                        )
                        prob = roi_eval["probability"]
                        expected_gain = roi_eval["expected_gain"]
                        learned_score = roi_eval["decision"]
                        if prob >= args.learned_roi_threshold and learned_score > 0.0:
                            learned_roi_candidates.append(
                                (learned_score, prob, expected_gain, estimated_kb, box, roi_gain, crop_xyxy)
                            )
                    learned_roi_candidates.sort(key=lambda item: item[0], reverse=True)
                    for learned_score, prob, expected_gain, estimated_kb, box, roi_gain, crop_xyxy in learned_roi_candidates[
                        : args.learned_max_roi_per_frame
                    ]:
                        payload, payload_bytes = make_payload(
                            frame,
                            raw_frame_id,
                            "roi",
                            ["learned_roi"],
                            edge_boxes,
                            prob,
                            args.jpeg_quality,
                            crop_xyxy=crop_xyxy,
                            target_edge_box=box,
                            cloud_imgsz=select_cloud_imgsz(prob, args, adaptive=True),
                        )
                        payload["metadata"]["roi_gain"] = round(float(roi_gain), 3)
                        payload["metadata"]["learned_prob"] = round(float(prob), 4)
                        payload["metadata"]["learned_expected_gain"] = round(float(expected_gain), 4)
                        payload["metadata"]["learned_decision"] = round(float(learned_score), 4)
                        payload["metadata"]["estimated_kb"] = round(float(estimated_kb), 3)
                        submit_payload(executor, payload, payload_bytes)

            if learned_policy is None and args.keyframe_interval > 0 and raw_frame_id % args.keyframe_interval == 0:
                if args.enable_temporal_memory and not image_paths and frame_temporal_stable:
                    stats["temporal_frame_suppressed"] += 1
                else:
                    stats["full_frame_probe_submitted"] += 1
                    payload, payload_bytes = make_payload(
                        frame,
                        raw_frame_id,
                        "keyframe",
                        ["periodic_keyframe"],
                        edge_boxes,
                        0.55,
                        args.jpeg_quality,
                        cloud_imgsz=select_cloud_imgsz(0.55, args),
                    )
                    submit_payload(executor, payload, payload_bytes)
                    full_frame_submitted = True

            if learned_policy is None and consecutive_empty >= args.empty_probe_frames:
                stats["full_frame_probe_submitted"] += 1
                payload, payload_bytes = make_payload(
                    frame,
                    raw_frame_id,
                    "keyframe",
                    ["empty_probe"],
                    edge_boxes,
                    0.65,
                    args.jpeg_quality,
                    cloud_imgsz=select_cloud_imgsz(0.65, args),
                )
                submit_payload(executor, payload, payload_bytes)
                full_frame_submitted = True
                consecutive_empty = 0

            if learned_policy is None and args.enable_complexity_probe and not full_frame_submitted:
                should_probe, complexity, reasons = should_complexity_probe(
                    frame,
                    edge_boxes,
                    complexity_threshold=args.complexity_threshold,
                    empty_complexity_threshold=args.empty_complexity_threshold,
                    max_detections=args.complexity_max_detections,
                    coverage_threshold=args.complexity_coverage_threshold,
                )
                if should_probe:
                    if args.enable_temporal_memory and not image_paths and frame_temporal_stable:
                        stats["temporal_frame_suppressed"] += 1
                    else:
                        stats["full_frame_probe_submitted"] += 1
                        payload, payload_bytes = make_payload(
                            frame,
                            raw_frame_id,
                            "keyframe",
                            reasons,
                            edge_boxes,
                            complexity,
                            args.jpeg_quality,
                            cloud_imgsz=select_cloud_imgsz(complexity, args),
                        )
                        submit_payload(executor, payload, payload_bytes)
                        full_frame_submitted = True

            if learned_policy is None and args.enable_stability_probe and not full_frame_submitted:
                stability_started = time.perf_counter()
                stability_kwargs = {"source": frame, "imgsz": args.stability_imgsz, "conf": args.conf, "verbose": False}
                if args.device:
                    stability_kwargs["device"] = args.device
                stability_result = model.predict(**stability_kwargs)[0]
                edge_latencies[-1] += (time.perf_counter() - stability_started) * 1000.0
                secondary_boxes = boxes_from_result(stability_result)
                stability_score, stability_reasons = stability_probe_score(edge_boxes, secondary_boxes, args.stability_iou)
                if stability_score >= args.stability_threshold:
                    if args.enable_temporal_memory and not image_paths and frame_temporal_stable:
                        stats["temporal_frame_suppressed"] += 1
                    else:
                        stats["full_frame_probe_submitted"] += 1
                        payload, payload_bytes = make_payload(
                            frame,
                            raw_frame_id,
                            "keyframe",
                            stability_reasons,
                            edge_boxes,
                            stability_score,
                            args.jpeg_quality,
                            cloud_imgsz=select_cloud_imgsz(stability_score, args),
                        )
                        payload["metadata"]["secondary_edge_boxes"] = secondary_boxes
                        submit_payload(executor, payload, payload_bytes)
                        full_frame_submitted = True

            candidates: list[tuple[float, list[str], dict[str, Any]]] = []
            for edge_index, box in enumerate(edge_boxes):
                if learned_policy is None and args.enable_temporal_memory and not image_paths:
                    if temporal_skip_roi_upload(
                        box,
                        box_track_map.get(edge_index),
                        raw_frame_id,
                        args.temporal_iou,
                        args.temporal_stable_hits,
                        args.temporal_reupload_gap,
                        args.temporal_conf_drop,
                    ):
                        stats["temporal_roi_suppressed"] += 1
                        continue
                score, reasons = detection_upload_score(box, frame.shape[:2], previous_boxes, use_temporal=not image_paths)
                if score >= args.upload_threshold:
                    candidates.append((score, reasons, box))
            candidates.sort(key=lambda item: item[0], reverse=True)
            for score, reasons, box in ([] if learned_policy is not None else candidates[: args.max_roi_per_frame]):
                roi_gain = (
                    adaptive_roi_gain(
                        box,
                        frame.shape[:2],
                        base_gain=args.roi_expand,
                        small_gain=args.roi_small_gain,
                        edge_gain=args.roi_edge_gain,
                        large_gain=args.roi_large_gain,
                    )
                    if args.enable_adaptive_roi
                    else args.roi_expand
                )
                crop_xyxy = expand_xyxy(box["xyxy"], frame.shape[:2], roi_gain)
                payload, payload_bytes = make_payload(
                    frame,
                    raw_frame_id,
                    "roi",
                    reasons,
                    edge_boxes,
                    score,
                    args.jpeg_quality,
                    crop_xyxy=crop_xyxy,
                    target_edge_box=box,
                    cloud_imgsz=select_cloud_imgsz(score, args),
                )
                payload["metadata"]["roi_gain"] = round(float(roi_gain), 3)
                submit_payload(executor, payload, payload_bytes)

            if learned_policy is None and args.enable_lowconf_proposal and not full_frame_submitted:
                proposals = low_conf_proposals(
                    candidate_edge_boxes,
                    edge_boxes,
                    frame.shape[:2],
                    min_conf=args.proposal_conf,
                    max_conf=args.conf,
                    max_proposals=args.max_proposals_per_frame,
                    score_threshold=args.proposal_score_threshold,
                )
                for proposal in proposals:
                    crop_xyxy = expand_xyxy(proposal["xyxy"], frame.shape[:2], args.roi_expand)
                    proposal_score = float(proposal.get("proposal_score", proposal.get("conf", 0.0)))
                    payload, payload_bytes = make_payload(
                        frame,
                        raw_frame_id,
                        "proposal",
                        proposal.get("proposal_reason", ["suppressed_low_conf"]),
                        edge_boxes,
                        proposal_score,
                        args.jpeg_quality,
                        crop_xyxy=crop_xyxy,
                        target_edge_box=proposal,
                        cloud_imgsz=select_cloud_imgsz(proposal_score, args),
                    )
                    submit_payload(executor, payload, payload_bytes)

            if learned_policy is None and args.enable_tile_probe and not full_frame_submitted:
                grid = parse_grid(args.tile_grid)
                for tile in tile_candidates(
                    frame,
                    edge_boxes,
                    grid=grid,
                    max_tiles=args.max_tiles_per_frame,
                    score_threshold=args.tile_score_threshold,
                ):
                    payload, payload_bytes = make_payload(
                        frame,
                        raw_frame_id,
                        "tile",
                        tile["reason"],
                        edge_boxes,
                        tile["score"],
                        args.jpeg_quality,
                        crop_xyxy=tile["xyxy"],
                        cloud_imgsz=select_cloud_imgsz(float(tile["score"]), args),
                    )
                    payload["metadata"]["tile"] = tile
                    submit_payload(executor, payload, payload_bytes)

            for response in drain_done(timeout=0.0):
                metadata = response.get("request_metadata", {})
                if response.get("status") == "ok":
                    fusion = fuse_detections(
                        edge_boxes_for_fusion(metadata),
                        response.get("cloud_boxes", []),
                        metadata.get("payload_type", ""),
                        args.iou_match,
                    )
                    response["fusion"] = fusion
                    for key, value in fusion["counts"].items():
                        stats[key] += value
                    cloud_latencies.append(float(response.get("cloud_latency_ms", 0.0)))
                    cloud_roundtrips.append(float(response.get("roundtrip_ms", 0.0)))
                    stats["uploads_completed"] += 1
                    if args.enable_temporal_memory and not image_paths:
                        mark_temporal_memory_from_response(
                            temporal_tracks,
                            metadata,
                            response.get("cloud_boxes", []),
                            int(metadata.get("frame_id", raw_frame_id)),
                            args,
                        )
                        stats["temporal_verified_responses"] += 1
                    if args.show:
                        cached = frame_cache.get(int(metadata.get("frame_id", -1)))
                        if cached is not None:
                            latest_cloud_preview = cached["frame"].copy()
                            draw_boxes(latest_cloud_preview, cached["edge_boxes"], (0, 255, 0), "edge")
                            draw_boxes(latest_cloud_preview, response.get("cloud_boxes", []), (0, 0, 255), "cloud")
                            annotate_status(
                                latest_cloud_preview,
                                [
                                    f"frame={metadata.get('frame_id', '?')} payload={metadata.get('payload_type', '?')}",
                                    f"cloud={response.get('cloud_latency_ms', 0.0):.1f}ms rtt={response.get('roundtrip_ms', 0.0):.1f}ms",
                                ],
                            )
                else:
                    stats["uploads_failed"] += 1
                events_file.write(json.dumps(response, ensure_ascii=False) + "\n")
                events_file.flush()

            preview = frame.copy()
            draw_boxes(preview, edge_boxes, (0, 255, 0), "edge")
            annotate_status(
                preview,
                [
                    f"frame={raw_frame_id} uploads={int(stats['uploads_submitted'])} pending={len(pending)}",
                    f"temporal={'on' if args.enable_temporal_memory and not image_paths else 'off'} stable={int(frame_temporal_stable)} tracks={len(temporal_tracks)}",
                ],
            )
            if writer:
                writer.write(preview)
            if args.show:
                cv2.imshow("YOLO26 EdgeCloud Live", preview)
                if latest_cloud_preview is not None:
                    cv2.imshow("YOLO26 EdgeCloud Cloud", latest_cloud_preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            previous_boxes = edge_boxes

        while pending:
            for response in drain_done(timeout=0.25):
                metadata = response.get("request_metadata", {})
                if response.get("status") == "ok":
                    fusion = fuse_detections(
                        edge_boxes_for_fusion(metadata),
                        response.get("cloud_boxes", []),
                        metadata.get("payload_type", ""),
                        args.iou_match,
                    )
                    response["fusion"] = fusion
                    for key, value in fusion["counts"].items():
                        stats[key] += value
                    cloud_latencies.append(float(response.get("cloud_latency_ms", 0.0)))
                    cloud_roundtrips.append(float(response.get("roundtrip_ms", 0.0)))
                    stats["uploads_completed"] += 1
                    if args.enable_temporal_memory and not image_paths:
                        mark_temporal_memory_from_response(
                            temporal_tracks,
                            metadata,
                            response.get("cloud_boxes", []),
                            int(metadata.get("frame_id", -1)),
                            args,
                        )
                        stats["temporal_verified_responses"] += 1
                    if args.show:
                        cached = frame_cache.get(int(metadata.get("frame_id", -1)))
                        if cached is not None:
                            latest_cloud_preview = cached["frame"].copy()
                            draw_boxes(latest_cloud_preview, cached["edge_boxes"], (0, 255, 0), "edge")
                            draw_boxes(latest_cloud_preview, response.get("cloud_boxes", []), (0, 0, 255), "cloud")
                            annotate_status(
                                latest_cloud_preview,
                                [
                                    f"frame={metadata.get('frame_id', '?')} payload={metadata.get('payload_type', '?')}",
                                    f"cloud={response.get('cloud_latency_ms', 0.0):.1f}ms rtt={response.get('roundtrip_ms', 0.0):.1f}ms",
                                ],
                            )
                else:
                    stats["uploads_failed"] += 1
                events_file.write(json.dumps(response, ensure_ascii=False) + "\n")
                events_file.flush()

    if cap is not None:
        cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    frames = max(1, int(stats["frames_processed"]))
    summary = {
        "source": args.source,
        "edge_model": args.model,
        "cloud_url": args.cloud_url,
        "edge_imgsz": int(args.imgsz),
        "cloud_imgsz_request": int(args.cloud_imgsz) if args.cloud_imgsz > 0 else None,
        "learned_adaptive_cloud_imgsz": bool(args.enable_learned_adaptive_cloud_imgsz),
        "learned_cloud_imgsz_low": int(args.learned_cloud_imgsz_low),
        "learned_cloud_imgsz_high": int(args.learned_cloud_imgsz_high),
        "learned_cloud_imgsz_switch": float(args.learned_cloud_imgsz_switch),
        "cloud_imgsz_request_counts": {
            str(key).replace("cloud_imgsz_", ""): int(value)
            for key, value in sorted(stats.items())
            if str(key).startswith("cloud_imgsz_")
        },
        "frames_processed": int(stats["frames_processed"]),
        "uploads_submitted": int(stats["uploads_submitted"]),
        "uploads_completed": int(stats["uploads_completed"]),
        "uploads_failed": int(stats["uploads_failed"]),
        "uploads_skipped_limit": int(stats["uploads_skipped_limit"]),
        "upload_ratio": round(float(stats["uploads_submitted"]) / frames, 4),
        "roi_uploads": int(stats["payload_roi"]),
        "keyframe_uploads": int(stats["payload_keyframe"]),
        "proposal_uploads": int(stats["payload_proposal"]),
        "tile_uploads": int(stats["payload_tile"]),
        "temporal_memory_enabled": bool(args.enable_temporal_memory and not image_paths),
        "temporal_roi_suppressed": int(stats["temporal_roi_suppressed"]),
        "temporal_frame_suppressed": int(stats["temporal_frame_suppressed"]),
        "temporal_verified_responses": int(stats["temporal_verified_responses"]),
        "temporal_active_tracks": int(len(temporal_tracks)),
        "avg_payload_kb": round((statistics.mean(payload_sizes) / 1024.0) if payload_sizes else 0.0, 3),
        "total_payload_mb": round(sum(payload_sizes) / (1024.0 * 1024.0), 3),
        "edge_latency_ms_avg": round(statistics.mean(edge_latencies), 3) if edge_latencies else 0.0,
        "edge_latency_ms_p95": round(percentile(edge_latencies, 0.95), 3),
        "cloud_latency_ms_avg": round(statistics.mean(cloud_latencies), 3) if cloud_latencies else 0.0,
        "cloud_latency_ms_p95": round(percentile(cloud_latencies, 0.95), 3),
        "roundtrip_ms_avg": round(statistics.mean(cloud_roundtrips), 3) if cloud_roundtrips else 0.0,
        "roundtrip_ms_p95": round(percentile(cloud_roundtrips, 0.95), 3),
        "confirmed": int(stats["confirmed"]),
        "missed_by_edge": int(stats["missed_by_edge"]),
        "possible_false_positive": int(stats["possible_false_positive"]),
        "class_corrected": int(stats["class_corrected"]),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Events: {events_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
