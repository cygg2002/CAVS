# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Real-time webcam edge/cloud demo with interactive mode switching.

Modes:
- `1`: edge-only (local YOLO26s)
- `2`: cloud-only (remote YOLO26x full-frame)
- `3`: selective edge-cloud collaboration

The demo is intentionally synchronous per frame so the overlays describe the
current frame truthfully. This makes latency comparisons easy to present on a
projected screen.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from edge_client import (
    annotate_status,
    make_payload,
    normalize_capture_source,
    post_to_cloud,
    select_cloud_imgsz,
    temporal_skip_frame_upload,
    temporal_skip_roi_upload,
    update_temporal_tracks,
    mark_temporal_memory_from_response,
)
from evaluate_edge_cloud import apply_roi_cloud, class_aware_nms, full_frame_fusion_boxes
from learned_uploader import frame_features, image_context, load_policy, policy_decision, roi_features
from protocol import boxes_from_result, crop_image, encode_jpeg, expand_xyxy, shift_boxes
from upload_strategies import adaptive_roi_gain


MODE_EDGE = "edge"
MODE_CLOUD = "cloud"
MODE_COLLAB = "collab"


def parse_args() -> argparse.Namespace:
    """Parse demo arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-model", default="yolo26s.pt", help="Local edge model.")
    parser.add_argument("--source", default="0", help="Camera index or video source.")
    parser.add_argument("--cloud-url", required=True, help="Remote cloud /infer endpoint.")
    parser.add_argument("--learned-policy-model", default="", help="Learned upload policy JSON.")
    parser.add_argument("--mode", choices=(MODE_EDGE, MODE_CLOUD, MODE_COLLAB), default=MODE_COLLAB, help="Initial display mode.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process. 0 means unlimited.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a display window.")
    parser.add_argument("--warmup", action="store_true", help="Warm up edge and cloud inference once before starting the live loop.")
    parser.add_argument("--imgsz", type=int, default=640, help="Edge inference size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Edge confidence threshold.")
    parser.add_argument("--cloud-conf", type=float, default=0.25, help="Displayed cloud confidence threshold.")
    parser.add_argument("--device", default="", help="Ultralytics device string for the edge model.")
    parser.add_argument("--cloud-timeout", type=float, default=3.0, help="Remote request timeout in seconds.")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="Upload JPEG quality.")
    parser.add_argument("--cloud-imgsz", type=int, default=512, help="Requested cloud inference size.")
    parser.add_argument("--roi-expand", type=float, default=1.35, help="Base ROI expansion.")
    parser.add_argument("--enable-adaptive-roi", action="store_true", help="Use adaptive ROI gains.")
    parser.add_argument("--roi-small-gain", type=float, default=1.6, help="ROI gain for small objects.")
    parser.add_argument("--roi-edge-gain", type=float, default=1.45, help="ROI gain for edge-near objects.")
    parser.add_argument("--roi-large-gain", type=float, default=1.15, help="ROI gain for large objects.")
    parser.add_argument("--learned-frame-threshold", type=float, default=0.70, help="Full-frame learned threshold.")
    parser.add_argument("--learned-roi-threshold", type=float, default=0.10, help="ROI learned threshold.")
    parser.add_argument("--learned-cost-weight", type=float, default=0.02, help="Payload penalty in the learned score.")
    parser.add_argument("--learned-max-roi-per-frame", type=int, default=1, help="Maximum ROIs to send in collaboration mode.")
    parser.add_argument("--enable-learned-adaptive-cloud-imgsz", action="store_true", help="Choose cloud size from learned utility.")
    parser.add_argument("--learned-cloud-imgsz-low", type=int, default=416, help="Cloud size for marginal learned uploads.")
    parser.add_argument("--learned-cloud-imgsz-high", type=int, default=512, help="Cloud size for high-utility learned uploads.")
    parser.add_argument("--learned-cloud-imgsz-switch", type=float, default=0.75, help="High cloud size cutoff.")
    parser.add_argument("--enable-temporal-memory", action="store_true", help="Suppress repeated uploads for stable video tracks.")
    parser.add_argument("--temporal-iou", type=float, default=0.5, help="Temporal track association IoU.")
    parser.add_argument("--temporal-ttl-frames", type=int, default=45, help="Track cache TTL in frames.")
    parser.add_argument("--temporal-stable-hits", type=int, default=3, help="Stable-hit threshold before suppression.")
    parser.add_argument("--temporal-reupload-gap", type=int, default=15, help="Frames before re-uploading a stable verified object.")
    parser.add_argument("--temporal-conf-drop", type=float, default=0.12, help="Confidence drop that re-enables upload.")
    parser.add_argument("--temporal-frame-stable-ratio", type=float, default=0.7, help="Suppress full-frame upload when enough targets are stable.")
    parser.add_argument("--iou-match", type=float, default=0.5, help="Edge/cloud fusion IoU threshold.")
    parser.add_argument("--fused-nms-iou", type=float, default=0.7, help="Final NMS IoU threshold.")
    parser.add_argument("--save-dir", default="runs/edge_cloud/webcam_demo", help="Optional output directory for the final summary.")
    return parser.parse_args()


def filter_confidence(boxes: list[dict[str, Any]], conf: float) -> list[dict[str, Any]]:
    """Filter boxes by confidence threshold."""
    return [box for box in boxes if float(box.get("conf", 0.0)) >= conf]


def draw_boxes(frame, boxes: list[dict[str, Any]], color: tuple[int, int, int], label_prefix: str) -> None:
    """Draw serializable boxes on a frame."""
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box["xyxy"]]
        name = box.get("name", box.get("cls"))
        label = f"{label_prefix}:{name} {float(box.get('conf', 0.0)):.2f}" if label_prefix else f"{name} {float(box.get('conf', 0.0)):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def cloud_request(
    frame,
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
) -> tuple[dict[str, Any], int]:
    """Send one blocking cloud request and return the response plus payload bytes."""
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
    return post_to_cloud(cloud_url, payload, cloud_timeout), payload_bytes


def main() -> None:
    """Run the interactive webcam demo."""
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_path = save_dir / "webcam_demo_summary.json"

    cap = cv2.VideoCapture(normalize_capture_source(args.source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open camera or video source: {args.source}")

    print(f"Loading edge model: {args.edge_model}")
    edge_model = YOLO(args.edge_model)
    learned_policy = load_policy(args.learned_policy_model) if args.learned_policy_model else None

    if args.warmup:
        warmup_frame = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        warm_kwargs = {"source": warmup_frame, "imgsz": args.imgsz, "conf": args.conf, "verbose": False}
        if args.device:
            warm_kwargs["device"] = args.device
        edge_model.predict(**warm_kwargs)
        cloud_request(
            warmup_frame,
            0,
            args.cloud_url,
            args.jpeg_quality,
            args.cloud_timeout,
            "keyframe",
            ["warmup"],
            [],
            0.0,
            cloud_imgsz=args.cloud_imgsz,
        )

    mode = args.mode
    frame_id = 0
    edge_latencies: list[float] = []
    cloud_latencies: list[float] = []
    roundtrip_latencies: list[float] = []
    e2e_latencies: list[float] = []
    fps_window: deque[float] = deque(maxlen=30)
    stats = Counter()
    temporal_tracks: list[dict[str, Any]] = []
    last_report_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if args.max_frames and frame_id > args.max_frames:
                break
            frame_started = time.perf_counter()
            stats["frames"] += 1

            edge_started = time.perf_counter()
            predict_kwargs = {"source": frame, "imgsz": args.imgsz, "conf": args.conf, "verbose": False}
            if args.device:
                predict_kwargs["device"] = args.device
            result = edge_model.predict(**predict_kwargs)[0]
            edge_latency_ms = (time.perf_counter() - edge_started) * 1000.0
            edge_latencies.append(edge_latency_ms)
            edge_boxes = filter_confidence(boxes_from_result(result), args.conf)

            box_track_map: dict[int, dict[str, Any]] = {}
            frame_temporal_stable = False
            if args.enable_temporal_memory:
                temporal_tracks, box_track_map = update_temporal_tracks(
                    temporal_tracks, edge_boxes, frame_id, args.temporal_iou, args.temporal_ttl_frames
                )
                frame_temporal_stable = temporal_skip_frame_upload(edge_boxes, box_track_map, frame_id, args)

            display_boxes = [dict(box) for box in edge_boxes]
            upload_type = "none"
            cloud_latency_ms = 0.0
            roundtrip_ms = 0.0
            payload_kb = 0.0
            upload_note = "local"

            if mode == MODE_CLOUD:
                response, payload_bytes = cloud_request(
                    frame,
                    frame_id,
                    args.cloud_url,
                    args.jpeg_quality,
                    args.cloud_timeout,
                    "keyframe",
                    ["cloud_only"],
                    edge_boxes,
                    1.0,
                    cloud_imgsz=args.cloud_imgsz,
                )
                payload_kb = payload_bytes / 1024.0
                upload_type = "full"
                upload_note = "cloud_only"
                if response.get("status") == "ok":
                    display_boxes = filter_confidence(response.get("cloud_boxes", []), args.cloud_conf)
                    cloud_latency_ms = float(response.get("cloud_latency_ms", 0.0))
                    roundtrip_ms = float(response.get("roundtrip_ms", 0.0))
                    cloud_latencies.append(cloud_latency_ms)
                    roundtrip_latencies.append(roundtrip_ms)
                    stats["cloud_only_ok"] += 1
                else:
                    stats["cloud_errors"] += 1
                    upload_note = str(response.get("error", "cloud_error"))[:60]

            elif mode == MODE_COLLAB:
                context = image_context(frame, edge_boxes)
                full_frame_kb = len(encode_jpeg(frame, args.jpeg_quality)) / 1024.0
                if learned_policy is None:
                    upload_note = "no_policy"
                else:
                    frame_eval = policy_decision(
                        learned_policy,
                        "frame",
                        frame_features(context, learned_policy.get("frame_features")),
                        full_frame_kb,
                        args.learned_cost_weight,
                    )
                    if args.enable_temporal_memory and frame_temporal_stable:
                        stats["temporal_frame_suppressed"] += 1
                        upload_note = "temporal_frame_skip"
                    elif (
                        frame_eval["probability"] >= args.learned_frame_threshold
                        and frame_eval["decision"] > 0.0
                    ):
                        response, payload_bytes = cloud_request(
                            frame,
                            frame_id,
                            args.cloud_url,
                            args.jpeg_quality,
                            args.cloud_timeout,
                            "keyframe",
                            ["learned_full_frame"],
                            edge_boxes,
                            frame_eval["probability"],
                            cloud_imgsz=select_cloud_imgsz(frame_eval["probability"], args, adaptive=True),
                        )
                        payload_kb = payload_bytes / 1024.0
                        upload_type = "full"
                        upload_note = f"frame p={frame_eval['probability']:.2f} g={frame_eval['expected_gain']:.2f}"
                        if response.get("status") == "ok":
                            cloud_boxes = filter_confidence(response.get("cloud_boxes", []), args.cloud_conf)
                            display_boxes = full_frame_fusion_boxes(
                                edge_boxes,
                                cloud_boxes,
                                args.iou_match,
                                args.fused_nms_iou,
                                args.cloud_conf,
                                args.conf,
                            )
                            cloud_latency_ms = float(response.get("cloud_latency_ms", 0.0))
                            roundtrip_ms = float(response.get("roundtrip_ms", 0.0))
                            cloud_latencies.append(cloud_latency_ms)
                            roundtrip_latencies.append(roundtrip_ms)
                            stats["collab_full_ok"] += 1
                            if args.enable_temporal_memory:
                                mark_temporal_memory_from_response(
                                    temporal_tracks,
                                    response.get("request_metadata", {}),
                                    response.get("cloud_boxes", []),
                                    frame_id,
                                    args,
                                )
                        else:
                            stats["cloud_errors"] += 1
                            upload_note = str(response.get("error", "cloud_error"))[:60]
                    else:
                        final_boxes = [dict(box) for box in edge_boxes]
                        candidates: list[tuple[float, float, float, int, dict[str, Any], float, list[int]]] = []
                        for edge_index, box in enumerate(edge_boxes):
                            if args.enable_temporal_memory and temporal_skip_roi_upload(
                                box,
                                box_track_map.get(edge_index),
                                frame_id,
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
                            if roi_eval["probability"] >= args.learned_roi_threshold and roi_eval["decision"] > 0.0:
                                candidates.append(
                                    (
                                        roi_eval["decision"],
                                        roi_eval["probability"],
                                        roi_eval["expected_gain"],
                                        edge_index,
                                        box,
                                        roi_gain,
                                        crop_xyxy,
                                    )
                                )
                        candidates.sort(key=lambda item: item[0], reverse=True)
                        chosen = candidates[: max(1, args.learned_max_roi_per_frame)]
                        if not chosen:
                            upload_note = "edge_kept"
                        for score, probability, expected_gain, edge_index, box, roi_gain, crop_xyxy in chosen:
                            response, payload_bytes = cloud_request(
                                frame,
                                frame_id,
                                args.cloud_url,
                                args.jpeg_quality,
                                args.cloud_timeout,
                                "roi",
                                ["learned_roi"],
                                edge_boxes,
                                probability,
                                crop_xyxy=crop_xyxy,
                                target_edge_box=box,
                                cloud_imgsz=select_cloud_imgsz(probability, args, adaptive=True),
                            )
                            payload_kb += payload_bytes / 1024.0
                            upload_type = "roi"
                            upload_note = f"roi p={probability:.2f} g={expected_gain:.2f}"
                            if response.get("status") == "ok":
                                cloud_boxes = filter_confidence(response.get("cloud_boxes", []), args.cloud_conf)
                                cloud_latency_ms = max(cloud_latency_ms, float(response.get("cloud_latency_ms", 0.0)))
                                roundtrip_ms = max(roundtrip_ms, float(response.get("roundtrip_ms", 0.0)))
                                cloud_latencies.append(float(response.get("cloud_latency_ms", 0.0)))
                                roundtrip_latencies.append(float(response.get("roundtrip_ms", 0.0)))
                                shifted = cloud_boxes
                                final_boxes = apply_roi_cloud(
                                    final_boxes,
                                    edge_index,
                                    shifted,
                                    args.iou_match,
                                    args.cloud_conf,
                                    args.cloud_conf,
                                    args.conf,
                                )
                                stats["collab_roi_ok"] += 1
                                if args.enable_temporal_memory:
                                    mark_temporal_memory_from_response(
                                        temporal_tracks,
                                        response.get("request_metadata", {}),
                                        response.get("cloud_boxes", []),
                                        frame_id,
                                        args,
                                    )
                            else:
                                stats["cloud_errors"] += 1
                                upload_note = str(response.get("error", "cloud_error"))[:60]
                        display_boxes = class_aware_nms(final_boxes, args.fused_nms_iou)

            # edge-only just reuses local boxes
            e2e_ms = (time.perf_counter() - frame_started) * 1000.0
            e2e_latencies.append(e2e_ms)
            fps_window.append(time.perf_counter())
            while len(fps_window) >= 2 and fps_window[-1] - fps_window[0] > 1.0:
                fps_window.popleft()
            fps = (len(fps_window) - 1) / max(1e-6, (fps_window[-1] - fps_window[0])) if len(fps_window) >= 2 else 0.0

            canvas = frame.copy()
            if mode == MODE_EDGE:
                draw_boxes(canvas, display_boxes, (0, 255, 0), "edge")
            elif mode == MODE_CLOUD:
                draw_boxes(canvas, display_boxes, (0, 0, 255), "cloud")
            else:
                draw_boxes(canvas, display_boxes, (0, 255, 255), "fused")
            annotate_status(
                canvas,
                [
                    f"mode={mode}  [1] edge  [2] cloud  [3] collab  [q] quit",
                    f"edge={edge_latency_ms:.1f}ms cloud={cloud_latency_ms:.1f}ms rtt={roundtrip_ms:.1f}ms e2e={e2e_ms:.1f}ms fps={fps:.1f}",
                    f"upload={upload_type} payload={payload_kb:.1f}KB note={upload_note}",
                    f"temporal={'on' if args.enable_temporal_memory else 'off'} tracks={len(temporal_tracks)}",
                ],
            )
            if not args.headless:
                cv2.imshow("YOLO26 Webcam Demo", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("1"):
                    mode = MODE_EDGE
                elif key == ord("2"):
                    mode = MODE_CLOUD
                elif key == ord("3"):
                    mode = MODE_COLLAB
                elif key in (ord("q"), 27):
                    break

            if time.perf_counter() - last_report_time > 1.0:
                last_report_time = time.perf_counter()
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    summary = {
        "frames": int(stats["frames"]),
        "edge_latency_ms_avg": round(statistics.mean(edge_latencies), 3) if edge_latencies else 0.0,
        "cloud_latency_ms_avg": round(statistics.mean(cloud_latencies), 3) if cloud_latencies else 0.0,
        "roundtrip_ms_avg": round(statistics.mean(roundtrip_latencies), 3) if roundtrip_latencies else 0.0,
        "e2e_latency_ms_avg": round(statistics.mean(e2e_latencies), 3) if e2e_latencies else 0.0,
        "cloud_errors": int(stats["cloud_errors"]),
        "collab_full_ok": int(stats["collab_full_ok"]),
        "collab_roi_ok": int(stats["collab_roi_ok"]),
        "cloud_only_ok": int(stats["cloud_only_ok"]),
        "temporal_frame_suppressed": int(stats["temporal_frame_suppressed"]),
        "temporal_roi_suppressed": int(stats["temporal_roi_suppressed"]),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
