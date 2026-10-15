# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate YOLO26 edge-cloud fusion on a YOLO-format detection dataset."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics, box_iou

from edge_client import collect_image_paths, detection_upload_score, edge_boxes_for_fusion, make_payload, percentile
from fusion import fuse_detections
from learned_uploader import (
    frame_features,
    image_context,
    load_policy,
    policy_decision,
    roi_features,
)
from protocol import b64_to_bytes, boxes_from_result, crop_image, decode_jpeg, encode_jpeg, expand_xyxy, shift_boxes
from upload_strategies import (
    adaptive_roi_gain,
    low_conf_proposals,
    parse_grid,
    should_complexity_probe,
    stability_probe_score,
    tile_candidates,
)


def load_coco_names() -> dict[int, str]:
    """Return COCO80 class names."""
    return {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
    }


def label_path_for_image(image_path: Path) -> Path:
    """Map a COCO-style image path to its YOLO label path."""
    parts = list(image_path.parts)
    if "images" not in parts:
        return image_path.with_suffix(".txt")
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_labels(image_path: Path, shape: tuple[int, int]) -> list[dict[str, Any]]:
    """Load YOLO-format bbox or segmentation labels and convert them to native xyxy boxes."""
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return []
    h, w = shape[:2]
    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(x) for x in line.split()]
        if len(values) < 5:
            continue
        cls = int(values[0])
        if len(values) == 5:
            _, xc, yc, bw, bh = values
            x1 = (xc - bw / 2.0) * w
            y1 = (yc - bh / 2.0) * h
            x2 = (xc + bw / 2.0) * w
            y2 = (yc + bh / 2.0) * h
        else:
            # YOLO segmentation labels are "cls x1 y1 x2 y2 ..."; use their tight bbox for detection metrics.
            coords = values[1:]
            if len(coords) < 6:
                continue
            xs = coords[0::2]
            ys = coords[1::2]
            x1 = min(xs) * w
            y1 = min(ys) * h
            x2 = max(xs) * w
            y2 = max(ys) * h
        x1 = max(0.0, min(float(w), x1))
        y1 = max(0.0, min(float(h), y1))
        x2 = max(0.0, min(float(w), x2))
        y2 = max(0.0, min(float(h), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        labels.append({"cls": int(cls), "xyxy": [x1, y1, x2, y2]})
    return labels


def match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    iouv: torch.Tensor,
) -> np.ndarray:
    """Match predictions to labels for each IoU threshold, following Ultralytics validator logic."""
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    if pred_classes.shape[0] == 0 or true_classes.shape[0] == 0:
        return correct
    correct_class = true_classes[:, None] == pred_classes
    iou = (iou * correct_class).cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def update_metrics(metrics: DetMetrics, preds: list[dict[str, Any]], labels: list[dict[str, Any]], im_name: str) -> None:
    """Update detection metrics with one image."""
    iouv = torch.linspace(0.5, 0.95, 10)
    pred_boxes = torch.tensor([p["xyxy"] for p in preds], dtype=torch.float32)
    pred_cls = torch.tensor([p["cls"] for p in preds], dtype=torch.float32)
    pred_conf = np.array([p["conf"] for p in preds], dtype=np.float32)
    target_boxes = torch.tensor([x["xyxy"] for x in labels], dtype=torch.float32)
    target_cls = torch.tensor([x["cls"] for x in labels], dtype=torch.float32)

    if len(preds) and len(labels):
        correct = match_predictions(pred_cls, target_cls, box_iou(target_boxes, pred_boxes), iouv)
    else:
        correct = np.zeros((len(preds), len(iouv)), dtype=bool)

    target_cls_np = target_cls.numpy().astype(np.float32) if len(labels) else np.zeros(0, dtype=np.float32)
    pred_cls_np = pred_cls.numpy().astype(np.float32) if len(preds) else np.zeros(0, dtype=np.float32)
    metrics.update_stats(
        {
            "tp": correct,
            "conf": pred_conf,
            "pred_cls": pred_cls_np,
            "target_cls": target_cls_np,
            "target_img": np.unique(target_cls_np),
            "im_name": im_name,
        }
    )


def fixed_threshold_counts(
    preds: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, int]:
    """Return class-aware TP, FP, and FN counts at one fixed operating point."""
    if preds and labels:
        pred_boxes = torch.tensor([p["xyxy"] for p in preds], dtype=torch.float32)
        pred_cls = torch.tensor([p["cls"] for p in preds], dtype=torch.float32)
        target_boxes = torch.tensor([x["xyxy"] for x in labels], dtype=torch.float32)
        target_cls = torch.tensor([x["cls"] for x in labels], dtype=torch.float32)
        correct = match_predictions(
            pred_cls,
            target_cls,
            box_iou(target_boxes, pred_boxes),
            torch.tensor([float(iou_threshold)], dtype=torch.float32),
        )
        tp = int(correct[:, 0].sum())
    else:
        tp = 0
    return {"tp": tp, "fp": len(preds) - tp, "fn": len(labels) - tp}


def fixed_threshold_metrics(counts: dict[str, int]) -> dict[str, int | float]:
    """Convert aggregate TP, FP, and FN counts to precision, recall, and F1."""
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def boxes_to_coco_predictions(
    image_id: int,
    boxes: list[dict[str, Any]],
    class_to_category: dict[int, int],
) -> list[dict[str, Any]]:
    """Convert internal xyxy detections to COCO prediction rows."""
    rows = []
    for box in boxes:
        class_index = int(box["cls"])
        if class_index not in class_to_category:
            raise KeyError(f"Unknown model class index for COCO export: {class_index}")
        x1, y1, x2, y2 = [float(value) for value in box["xyxy"]]
        rows.append(
            {
                "image_id": int(image_id),
                "category_id": int(class_to_category[class_index]),
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(box.get("conf", 0.0)),
            }
        )
    return rows


def metrics_to_dict(metrics: DetMetrics, save_dir: Path) -> dict[str, float]:
    """Process and return a compact metrics dictionary."""
    metrics.process(save_dir=save_dir, plot=False)
    result = {k: float(v) for k, v in metrics.results_dict.items()}
    metrics.clear_stats()
    return result


def class_aware_nms(boxes: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    """Run a simple class-aware NMS over fused boxes."""
    from protocol import box_iou as scalar_iou

    kept: list[dict[str, Any]] = []
    for box in sorted(boxes, key=lambda x: float(x.get("conf", 0.0)), reverse=True):
        if all(int(box["cls"]) != int(k["cls"]) or scalar_iou(box["xyxy"], k["xyxy"]) < iou_threshold for k in kept):
            kept.append(box)
    return kept


def full_frame_fusion_boxes(
    edge_boxes: list[dict[str, Any]],
    cloud_boxes: list[dict[str, Any]],
    match_iou_threshold: float,
    nms_iou_threshold: float,
    cloud_conf_threshold: float,
    edge_conf_threshold: float,
) -> list[dict[str, Any]]:
    """Fuse full-frame edge and cloud detections as a sanity upper-bound reference.

    This is intentionally conservative: matched boxes prefer the cloud class
    correction, same-class matches keep the higher-confidence box, unmatched
    detections from either side survive if they clear their own confidence gate.
    """
    from protocol import greedy_match

    matches, matched_edge, matched_cloud = greedy_match(edge_boxes, cloud_boxes, match_iou_threshold)
    fused: list[dict[str, Any]] = []
    for edge_idx, cloud_idx, _ in matches:
        edge_box = edge_boxes[edge_idx]
        cloud_box = cloud_boxes[cloud_idx]
        if int(edge_box["cls"]) == int(cloud_box["cls"]):
            fused.append(cloud_box if float(cloud_box.get("conf", 0.0)) >= float(edge_box.get("conf", 0.0)) else edge_box)
        else:
            fused.append(cloud_box if float(cloud_box.get("conf", 0.0)) >= cloud_conf_threshold else edge_box)
    for edge_idx, edge_box in enumerate(edge_boxes):
        if edge_idx not in matched_edge and float(edge_box.get("conf", 0.0)) >= edge_conf_threshold:
            fused.append(edge_box)
    for cloud_idx, cloud_box in enumerate(cloud_boxes):
        if cloud_idx not in matched_cloud and float(cloud_box.get("conf", 0.0)) >= cloud_conf_threshold:
            fused.append(cloud_box)
    return class_aware_nms(fused, nms_iou_threshold)


def oracle_full_frame_fusion_boxes(
    edge_boxes: list[dict[str, Any]],
    cloud_boxes: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    nms_iou_threshold: float,
) -> list[dict[str, Any]]:
    """Build a GT-aware oracle upper bound from the union of edge and cloud candidates.

    This is evaluation-only and not deployable. It answers a narrow question:
    if edge and cloud detections were both available for every image, how much
    correct signal exists in their union before any real fusion policy is chosen?
    """
    from protocol import box_iou as scalar_iou

    candidates = [("edge", idx, box) for idx, box in enumerate(edge_boxes)] + [
        ("cloud", idx, box) for idx, box in enumerate(cloud_boxes)
    ]
    used: set[tuple[str, int]] = set()
    oracle: list[dict[str, Any]] = []
    for label in labels:
        best_key = None
        best_box = None
        best_score = (-1.0, -1.0)
        for source, idx, box in candidates:
            key = (source, idx)
            if key in used or int(box["cls"]) != int(label["cls"]):
                continue
            iou = scalar_iou(box["xyxy"], label["xyxy"])
            if iou < 0.5:
                continue
            score = (float(iou), float(box.get("conf", 0.0)))
            if score > best_score:
                best_key = key
                best_box = box
                best_score = score
        if best_key is not None and best_box is not None:
            used.add(best_key)
            oracle.append(best_box)
    return class_aware_nms(oracle, nms_iou_threshold)


def resolve_end2end_modes(args: argparse.Namespace) -> tuple[bool, bool]:
    """Resolve separate edge/cloud modes while preserving the shared legacy option."""
    shared = bool(getattr(args, "end2end", False))
    edge = getattr(args, "edge_end2end", None)
    cloud = getattr(args, "cloud_end2end", None)
    return (shared if edge is None else bool(edge), shared if cloud is None else bool(cloud))


def cloud_predict(
    model: YOLO,
    image,
    args: argparse.Namespace,
    cloud_imgsz: int | None = None,
    count_compute: bool = True,
    conf: float | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Run cloud model on one image and return detections plus latency."""
    started = time.perf_counter()
    selected_imgsz = int(cloud_imgsz or getattr(args, "cloud_imgsz", args.imgsz))
    if count_compute:
        args._cloud_compute_uploads = int(getattr(args, "_cloud_compute_uploads", 0)) + 1
        args._cloud_compute_scale_sum = float(getattr(args, "_cloud_compute_scale_sum", 0.0)) + (
            selected_imgsz / max(1.0, float(args.imgsz))
        ) ** 2
    _, cloud_end2end = resolve_end2end_modes(args)
    kwargs = {
        "source": image,
        "imgsz": selected_imgsz,
        "conf": args.cloud_conf if conf is None else conf,
        "iou": getattr(args, "model_nms_iou", 0.7),
        "max_det": getattr(args, "max_det", 300),
        "end2end": cloud_end2end,
        "verbose": False,
    }
    if args.device:
        kwargs["device"] = args.device
    result = model.predict(**kwargs)[0]
    boxes = boxes_from_result(result)
    if count_compute:
        response_bytes = len(json.dumps({"boxes": boxes}, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        getattr(args, "_cloud_response_sizes", []).append(response_bytes)
    return boxes, (time.perf_counter() - started) * 1000.0


def make_evaluation_payload(args: argparse.Namespace, *payload_args, **payload_kwargs) -> tuple[dict[str, Any], int]:
    """Build one payload and record its JPEG/base64 serialization cost."""
    started = time.perf_counter()
    payload, payload_bytes = make_payload(*payload_args, **payload_kwargs)
    if getattr(args, "simulate_upload_codec", False):
        getattr(args, "_upload_encode_latencies", []).append((time.perf_counter() - started) * 1000.0)
    return payload, payload_bytes


def uploaded_image_for_cloud(image, args: argparse.Namespace, payload: dict[str, Any] | None = None):
    """Optionally decode the actual upload payload before cloud inference."""
    if not getattr(args, "simulate_upload_codec", False):
        return image
    if payload is None:
        encode_started = time.perf_counter()
        encoded = encode_jpeg(image, args.jpeg_quality)
        getattr(args, "_upload_encode_latencies", []).append((time.perf_counter() - encode_started) * 1000.0)
    else:
        encoded = b64_to_bytes(payload["image_b64"])
    decode_started = time.perf_counter()
    decoded = decode_jpeg(encoded)
    decode_ms = (time.perf_counter() - decode_started) * 1000.0
    getattr(args, "_upload_decode_latencies", []).append(decode_ms)
    return decoded


def prepare_cloud_only_input(image, args: argparse.Namespace) -> tuple[Any, int, float, float]:
    """Return the Cloud-only input plus JPEG payload and codec timing when enabled."""
    if not getattr(args, "simulate_cloud_only_codec", False):
        return image, 0, 0.0, 0.0
    encode_started = time.perf_counter()
    encoded = encode_jpeg(image, args.jpeg_quality)
    encode_ms = (time.perf_counter() - encode_started) * 1000.0
    decode_started = time.perf_counter()
    decoded = decode_jpeg(encoded)
    decode_ms = (time.perf_counter() - decode_started) * 1000.0
    return decoded, len(encoded), encode_ms, decode_ms


def str_to_bool(value) -> bool:
    """Parse shell-friendly bool values such as true/false and 1/0."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def learned_cloud_imgsz(probability: float, args: argparse.Namespace) -> int:
    """Choose cloud inference size for a learned full-frame upload."""
    if not getattr(args, "enable_learned_adaptive_cloud_imgsz", False):
        return int(getattr(args, "cloud_imgsz", args.imgsz))
    if probability >= getattr(args, "learned_cloud_imgsz_switch", 0.70):
        return int(getattr(args, "learned_cloud_imgsz_high", 512))
    return int(getattr(args, "learned_cloud_imgsz_low", 416))


def apply_roi_cloud(
    final_boxes: list[dict[str, Any]],
    target_index: int,
    cloud_boxes: list[dict[str, Any]],
    iou_threshold: float,
    cloud_conf_threshold: float,
    extra_cloud_conf_threshold: float | None = None,
    veto_edge_conf_threshold: float = 1.0,
) -> list[dict[str, Any]]:
    """Apply one ROI cloud result to current final boxes."""
    from protocol import box_iou as scalar_iou

    extra_cloud_conf_threshold = cloud_conf_threshold if extra_cloud_conf_threshold is None else extra_cloud_conf_threshold
    if target_index >= len(final_boxes):
        return final_boxes
    target = final_boxes[target_index]
    matches = [(scalar_iou(target["xyxy"], c["xyxy"]), i, c) for i, c in enumerate(cloud_boxes)]
    matches.sort(reverse=True, key=lambda x: x[0])
    used_cloud: set[int] = set()

    if matches and matches[0][0] >= iou_threshold:
        _, cloud_idx, cloud_box = matches[0]
        final_boxes[target_index] = cloud_box
        used_cloud.add(cloud_idx)
    else:
        if float(target.get("conf", 0.0)) < veto_edge_conf_threshold:
            final_boxes.pop(target_index)

    for cloud_idx, cloud_box in enumerate(cloud_boxes):
        if cloud_idx not in used_cloud and float(cloud_box.get("conf", 0.0)) >= extra_cloud_conf_threshold:
            final_boxes.append(cloud_box)
    return final_boxes


def apply_proposal_cloud(
    final_boxes: list[dict[str, Any]],
    proposal_box: dict[str, Any],
    cloud_boxes: list[dict[str, Any]],
    iou_threshold: float,
    cloud_conf_threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    """Accept a cloud result only when it confirms a suppressed edge proposal."""
    from protocol import box_iou as scalar_iou

    matches = [
        (scalar_iou(proposal_box["xyxy"], cloud_box["xyxy"]), float(cloud_box.get("conf", 0.0)), cloud_box)
        for cloud_box in cloud_boxes
        if float(cloud_box.get("conf", 0.0)) >= cloud_conf_threshold
    ]
    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if matches and matches[0][0] >= iou_threshold:
        final_boxes.append(matches[0][2])
        return final_boxes, 1
    return final_boxes, 0


def apply_tile_cloud(
    final_boxes: list[dict[str, Any]],
    cloud_boxes: list[dict[str, Any]],
    tile_xyxy: list[int],
    image_shape: tuple[int, int],
    cloud_conf_threshold: float,
    inner_margin_ratio: float = 0.03,
    min_area_ratio: float = 0.0,
) -> tuple[list[dict[str, Any]], int]:
    """Add conservative cloud tile detections without suppressing edge boxes prematurely.

    Tile crops are useful for recovering local misses, but they also create
    boundary artifacts because the cloud model sees incomplete context. The
    acceptance gate is therefore stricter than ROI review: confidence must be
    high, the box center must stay inside the tile interior, and tiny boxes can
    be rejected when requested.
    """
    h, w = image_shape[:2]
    tx1, ty1, tx2, ty2 = [float(v) for v in tile_xyxy]
    tile_w = max(1.0, tx2 - tx1)
    tile_h = max(1.0, ty2 - ty1)
    margin_x = tile_w * max(0.0, inner_margin_ratio)
    margin_y = tile_h * max(0.0, inner_margin_ratio)
    accepted = 0
    for cloud_box in cloud_boxes:
        if float(cloud_box.get("conf", 0.0)) < cloud_conf_threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in cloud_box["xyxy"]]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if not (tx1 + margin_x <= cx <= tx2 - margin_x and ty1 + margin_y <= cy <= ty2 - margin_y):
            continue
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, float(w * h))
        if area_ratio < min_area_ratio:
            continue
        final_boxes.append(cloud_box)
        accepted += 1
    return final_boxes, accepted


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=r"F:\IDE\trae\datasets\coco128", help="Dataset image directory.")
    parser.add_argument("--edge-model", default="yolo26s.pt", help="Edge model.")
    parser.add_argument("--cloud-model", default="yolo26x.pt", help="Cloud model.")
    parser.add_argument("--save-dir", default="runs/edge_cloud/coco128_eval", help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--cloud-imgsz", type=int, default=640, help="Cloud inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Edge deployment confidence threshold.")
    parser.add_argument("--cloud-conf", type=float, default=0.25, help="Cloud deployment confidence threshold.")
    parser.add_argument("--model-nms-iou", type=float, default=0.7, help="Edge/cloud model NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per model invocation.")
    parser.add_argument(
        "--end2end",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Ultralytics YOLO26 head mode; false selects end2many.",
    )
    parser.add_argument(
        "--edge-end2end",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
        help="Edge YOLO26 head mode; overrides --end2end for the edge model.",
    )
    parser.add_argument(
        "--cloud-end2end",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
        help="Cloud YOLO26 head mode; overrides --end2end for the cloud model.",
    )
    parser.add_argument(
        "--metric-conf",
        type=float,
        default=-1.0,
        help="Edge confidence threshold for mAP accounting; negative reuses --conf.",
    )
    parser.add_argument(
        "--metric-cloud-conf",
        type=float,
        default=-1.0,
        help="Cloud-only confidence threshold for mAP accounting; negative reuses --cloud-conf.",
    )
    parser.add_argument("--device", default="0", help="Ultralytics device string.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images; 0 means all.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Unmeasured edge/cloud warmup inferences.")
    parser.add_argument("--upload-threshold", type=float, default=0.45, help="Detection uncertainty upload threshold.")
    parser.add_argument("--max-roi-per-image", type=int, default=2, help="Max ROI uploads per image.")
    parser.add_argument("--keyframe-interval", type=int, default=30, help="Full-frame cloud probe interval.")
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
    parser.add_argument("--learned-max-roi-per-image", type=int, default=1, help="Maximum learned ROI uploads per image.")
    parser.add_argument("--enable-learned-adaptive-cloud-imgsz", action="store_true", help="Choose cloud imgsz from learned utility probability.")
    parser.add_argument("--learned-cloud-imgsz-low", type=int, default=416, help="Lower cloud imgsz for marginal learned uploads.")
    parser.add_argument("--learned-cloud-imgsz-high", type=int, default=512, help="Higher cloud imgsz for high-utility learned uploads.")
    parser.add_argument("--learned-cloud-imgsz-switch", type=float, default=0.70, help="Use high cloud imgsz above this learned probability.")
    parser.add_argument("--enable-stability-probe", action="store_true", help="Upload frames whose edge detections are unstable across input scales.")
    parser.add_argument("--stability-imgsz", type=int, default=384, help="Secondary edge inference size for scale-stability probing.")
    parser.add_argument("--stability-threshold", type=float, default=0.45, help="Minimum scale-instability score for full-frame upload.")
    parser.add_argument("--stability-iou", type=float, default=0.5, help="IoU threshold for scale-stability matching.")
    parser.add_argument("--enable-lowconf-proposal", action="store_true", help="Upload suppressed low-confidence edge proposals.")
    parser.add_argument("--proposal-conf", type=float, default=0.05, help="Internal edge confidence for proposal mining.")
    parser.add_argument("--proposal-score-threshold", type=float, default=0.55, help="Minimum low-confidence proposal score.")
    parser.add_argument("--max-proposals-per-image", type=int, default=2, help="Maximum suppressed proposals to upload.")
    parser.add_argument("--proposal-cloud-conf", type=float, default=0.35, help="Cloud confidence threshold for proposals.")
    parser.add_argument("--proposal-iou-match", type=float, default=0.25, help="IoU threshold for proposal/cloud confirmation.")
    parser.add_argument("--enable-tile-probe", action="store_true", help="Upload high-complexity uncovered tiles.")
    parser.add_argument("--tile-grid", default="2x2", help="Tile grid as COLSxROWS, e.g. 2x2.")
    parser.add_argument("--max-tiles-per-image", type=int, default=1, help="Maximum tile uploads per image.")
    parser.add_argument("--tile-score-threshold", type=float, default=0.12, help="Minimum tile upload score.")
    parser.add_argument("--tile-cloud-conf", type=float, default=0.45, help="Cloud confidence threshold for tile boxes.")
    parser.add_argument("--tile-inner-margin", type=float, default=0.03, help="Reject tile boxes centered near crop edges.")
    parser.add_argument("--tile-min-area-ratio", type=float, default=0.0, help="Minimum global area ratio for tile boxes.")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality used for upload byte accounting.")
    parser.add_argument("--simulate-upload-codec", action="store_true", help="Decode the JPEG upload before cloud inference.")
    parser.add_argument(
        "--simulate-cloud-only-codec",
        action="store_true",
        help="Evaluate Cloud-only on decoded JPEG pixels and record full-frame payload cost.",
    )
    parser.add_argument("--roi-expand", type=float, default=1.35, help="ROI expansion factor.")
    parser.add_argument("--enable-adaptive-roi", action="store_true", help="Adapt ROI expansion by object scale and edge proximity.")
    parser.add_argument("--roi-small-gain", type=float, default=2.0, help="ROI gain for small objects.")
    parser.add_argument("--roi-edge-gain", type=float, default=1.65, help="ROI gain for edge-near objects.")
    parser.add_argument("--roi-large-gain", type=float, default=1.15, help="ROI gain for large objects.")
    parser.add_argument("--roi-extra-cloud-conf", type=float, default=-1.0, help="Confidence for unmatched cloud boxes inside ROI; negative uses --cloud-conf.")
    parser.add_argument("--roi-veto-edge-conf", type=float, default=1.0, help="Only drop unmatched ROI edge boxes below this confidence.")
    parser.add_argument("--iou-match", type=float, default=0.5, help="Fusion IoU threshold.")
    parser.add_argument("--fused-nms-iou", type=float, default=0.7, help="Final class-aware NMS IoU threshold.")
    parser.add_argument("--eval-cloud-only", action="store_true", help="Also evaluate cloud model on every image.")
    parser.add_argument(
        "--export-cloud-only-coco",
        action="store_true",
        help="Save full Cloud-only official/deploy COCO prediction JSON for replay.",
    )
    parser.add_argument(
        "--coco-annotations",
        default="",
        help="COCO annotation JSON required by --export-cloud-only-coco.",
    )
    parser.add_argument(
        "--eval-full-frame-fusion-upper",
        action="store_true",
        help="Also evaluate a full-frame edge+cloud fused upper-bound reference on every image.",
    )
    return parser.parse_args()


def run_evaluation(
    args: argparse.Namespace,
    edge_model: YOLO | None = None,
    cloud_model: YOLO | None = None,
    print_summary: bool = True,
    eval_cloud_only: bool | None = None,
) -> dict[str, Any]:
    """Run edge-cloud evaluation and return the summary dictionary."""
    edge_end2end, cloud_end2end = resolve_end2end_modes(args)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    args._cloud_compute_uploads = 0
    args._cloud_compute_scale_sum = 0.0
    args._upload_encode_latencies = []
    args._upload_decode_latencies = []
    args._cloud_response_sizes = []
    image_paths = collect_image_paths(args.source)
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.source}")

    names = load_coco_names()
    edge_metrics = DetMetrics(names=names)
    edge_deploy_metrics = DetMetrics(names=names)
    eval_cloud_only = bool(getattr(args, "eval_cloud_only", False) if eval_cloud_only is None else eval_cloud_only)
    export_cloud_only_coco = bool(getattr(args, "export_cloud_only_coco", False))
    if export_cloud_only_coco and not eval_cloud_only:
        raise ValueError("--export-cloud-only-coco requires --eval-cloud-only")
    class_to_category: dict[int, int] = {}
    expected_export_image_ids: set[int] = set()
    cloud_only_official_predictions: list[dict[str, Any]] = []
    cloud_only_deploy_predictions: list[dict[str, Any]] = []
    if export_cloud_only_coco:
        annotations_path = Path(str(getattr(args, "coco_annotations", "")))
        if not annotations_path.is_file():
            raise FileNotFoundError(
                "--coco-annotations must point to an existing file when exporting Cloud-only predictions"
            )
        annotation_data = json.loads(annotations_path.read_text(encoding="utf-8"))
        category_ids = sorted(int(item["id"]) for item in annotation_data["categories"])
        class_to_category = {index: category_id for index, category_id in enumerate(category_ids)}
        expected_export_image_ids = {int(item["id"]) for item in annotation_data["images"]}
    cloud_only_metrics = DetMetrics(names=names) if eval_cloud_only else None
    cloud_only_deploy_metrics = DetMetrics(names=names) if eval_cloud_only else None
    full_frame_fusion_metrics = DetMetrics(names=names) if getattr(args, "eval_full_frame_fusion_upper", False) else None
    fused_metrics = DetMetrics(names=names)
    stats = Counter()
    payload_sizes: list[int] = []
    edge_latencies: list[float] = []
    cloud_latencies: list[float] = []
    cloud_only_latencies: list[float] = []
    cloud_only_payload_sizes: list[int] = []
    cloud_only_encode_latencies: list[float] = []
    cloud_only_decode_latencies: list[float] = []
    cloud_only_response_sizes: list[int] = []
    pipeline_latencies: list[float] = []
    policy_latencies_per_image: list[float] = []
    policy_call_latencies: list[float] = []
    uploaded_image_ids: set[int] = set()
    edge_operating_counts = Counter()
    cloud_operating_counts = Counter()
    fused_operating_counts = Counter()
    fusion_events = Counter()
    examples: list[dict[str, Any]] = []
    edge_metric_conf = args.conf if getattr(args, "metric_conf", -1.0) < 0 else float(args.metric_conf)
    cloud_metric_conf = args.cloud_conf if getattr(args, "metric_cloud_conf", -1.0) < 0 else float(args.metric_cloud_conf)

    if edge_model is None:
        print(f"Loading edge model: {args.edge_model}")
        edge_model = YOLO(args.edge_model)
    if cloud_model is None:
        print(f"Loading cloud model: {args.cloud_model}")
        cloud_model = YOLO(args.cloud_model)
    learned_policy = None
    if getattr(args, "enable_learned_policy", False):
        if not getattr(args, "learned_policy_model", ""):
            raise ValueError("--learned-policy-model is required when --enable-learned-policy is set")
        learned_policy = load_policy(args.learned_policy_model)

    warmup_image = cv2.imread(str(image_paths[0]))
    if warmup_image is not None:
        common_warmup = {
            "source": warmup_image,
            "conf": min(args.conf, edge_metric_conf),
            "iou": args.model_nms_iou,
            "max_det": args.max_det,
            "verbose": False,
        }
        if args.device:
            common_warmup["device"] = args.device
        for _ in range(max(0, int(args.warmup_runs))):
            edge_model.predict(imgsz=args.imgsz, end2end=edge_end2end, **common_warmup)
            cloud_model.predict(imgsz=args.cloud_imgsz, end2end=cloud_end2end, **common_warmup)

    events_path = save_dir / "fusion_events.jsonl"
    with events_path.open("w", encoding="utf-8") as events_file:
        for image_id, image_path in enumerate(image_paths):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to decode image: {image_path}")
            labels = load_labels(image_path, image.shape[:2])
            pipeline_started = time.perf_counter()
            policy_image_latency = 0.0

            started = time.perf_counter()
            predict_conf = min(args.conf, edge_metric_conf)
            if getattr(args, "enable_lowconf_proposal", False):
                predict_conf = min(args.conf, edge_metric_conf, getattr(args, "proposal_conf", 0.05))
            kwargs = {
                "source": image,
                "imgsz": args.imgsz,
                "conf": predict_conf,
                "iou": args.model_nms_iou,
                "max_det": args.max_det,
                "end2end": edge_end2end,
                "verbose": False,
            }
            if args.device:
                kwargs["device"] = args.device
            edge_result = edge_model.predict(**kwargs)[0]
            candidate_edge_boxes = boxes_from_result(edge_result)
            edge_latency = (time.perf_counter() - started) * 1000.0
            edge_latencies.append(edge_latency)
            edge_boxes = [box for box in candidate_edge_boxes if float(box.get("conf", 0.0)) >= args.conf]
            metric_edge_boxes = [box for box in candidate_edge_boxes if float(box.get("conf", 0.0)) >= edge_metric_conf]
            final_boxes = [dict(box) for box in edge_boxes]

            update_metrics(edge_metrics, metric_edge_boxes, labels, image_path.name)
            update_metrics(edge_deploy_metrics, edge_boxes, labels, image_path.name)
            edge_operating_counts.update(fixed_threshold_counts(edge_boxes, labels))
            cloud_metric_boxes = None
            if cloud_only_metrics is not None or full_frame_fusion_metrics is not None:
                cloud_only_input, cloud_payload_bytes, cloud_encode_ms, cloud_decode_ms = prepare_cloud_only_input(
                    image, args
                )
                cloud_candidate_conf = min(float(args.cloud_conf), float(cloud_metric_conf))
                cloud_candidate_boxes, cloud_only_latency = cloud_predict(
                    cloud_model,
                    cloud_only_input,
                    args,
                    count_compute=False,
                    conf=cloud_candidate_conf,
                )
                cloud_metric_boxes = [
                    box for box in cloud_candidate_boxes if float(box.get("conf", 0.0)) >= cloud_metric_conf
                ]
                cloud_deploy_boxes = [
                    box for box in cloud_candidate_boxes if float(box.get("conf", 0.0)) >= args.cloud_conf
                ]
                response_started = time.perf_counter()
                cloud_only_response_sizes.append(
                    len(
                        json.dumps(
                            {"boxes": cloud_deploy_boxes},
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ).encode("utf-8")
                    )
                )
                cloud_only_latency += (time.perf_counter() - response_started) * 1000.0
                cloud_only_latencies.append(cloud_only_latency)
                if cloud_payload_bytes:
                    cloud_only_payload_sizes.append(cloud_payload_bytes)
                    cloud_only_encode_latencies.append(cloud_encode_ms)
                    cloud_only_decode_latencies.append(cloud_decode_ms)
                if cloud_only_metrics is not None:
                    cloud_operating_counts.update(fixed_threshold_counts(cloud_deploy_boxes, labels))
            if cloud_only_metrics is not None and cloud_metric_boxes is not None:
                update_metrics(cloud_only_metrics, cloud_metric_boxes, labels, image_path.name)
                update_metrics(cloud_only_deploy_metrics, cloud_deploy_boxes, labels, image_path.name)
                if export_cloud_only_coco:
                    coco_image_id = int(image_path.stem)
                    if coco_image_id not in expected_export_image_ids:
                        raise ValueError(f"Image ID {coco_image_id} is missing from COCO annotations")
                    cloud_only_official_predictions.extend(
                        boxes_to_coco_predictions(coco_image_id, cloud_metric_boxes, class_to_category)
                    )
                    cloud_only_deploy_predictions.extend(
                        boxes_to_coco_predictions(coco_image_id, cloud_deploy_boxes, class_to_category)
                    )
            if full_frame_fusion_metrics is not None and cloud_metric_boxes is not None:
                full_frame_upper = oracle_full_frame_fusion_boxes(
                    metric_edge_boxes,
                    cloud_metric_boxes,
                    labels,
                    args.fused_nms_iou,
                )
                update_metrics(full_frame_fusion_metrics, full_frame_upper, labels, image_path.name)

            image_events: list[dict[str, Any]] = []
            full_frame_uploaded = False
            if learned_policy is not None:
                context = image_context(image, edge_boxes)
                full_frame_kb = len(encode_jpeg(image, args.jpeg_quality)) / 1024.0
                frame_schema = learned_policy.get("frame_features", None)
                policy_started = time.perf_counter()
                frame_eval = policy_decision(
                    learned_policy,
                    "frame",
                    frame_features(context, frame_schema),
                    full_frame_kb,
                    args.learned_cost_weight,
                )
                policy_elapsed = (time.perf_counter() - policy_started) * 1000.0
                policy_image_latency += policy_elapsed
                policy_call_latencies.append(policy_elapsed)
                stats["policy_frame_calls"] += 1
                frame_prob = frame_eval["probability"]
                frame_gain = frame_eval["expected_gain"]
                frame_decision = frame_eval["decision"]
                if frame_prob >= args.learned_frame_threshold and frame_decision > 0.0:
                    payload, payload_bytes = make_evaluation_payload(
                        args,
                        image,
                        image_id,
                        "keyframe",
                        ["learned_full_frame"],
                        edge_boxes,
                        frame_prob,
                        args.jpeg_quality,
                    )
                    payload_sizes.append(payload_bytes)
                    stats["payload_keyframe"] += 1
                    selected_cloud_imgsz = learned_cloud_imgsz(frame_prob, args)
                    cloud_boxes, cloud_latency = cloud_predict(
                        cloud_model, uploaded_image_for_cloud(image, args, payload), args, selected_cloud_imgsz
                    )
                    cloud_latencies.append(cloud_latency)
                    final_boxes = cloud_boxes
                    full_frame_uploaded = True
                    image_events.append(
                        {
                            "image": str(image_path),
                            "payload_type": "keyframe",
                            "reason": ["learned_full_frame"],
                            "learned_prob": round(float(frame_prob), 4),
                            "learned_expected_gain": round(float(frame_gain), 4),
                            "learned_decision": round(float(frame_decision), 4),
                            "cloud_imgsz": selected_cloud_imgsz,
                            "estimated_kb": round(float(full_frame_kb), 3),
                            "cloud_boxes": len(cloud_boxes),
                            "cloud_latency_ms": round(cloud_latency, 3),
                        }
                    )

                if not full_frame_uploaded:
                    learned_roi_candidates: list[tuple[float, float, float, int, dict[str, Any], float, list[int]]] = []
                    for edge_index, box in enumerate(edge_boxes):
                        roi_gain = (
                            adaptive_roi_gain(
                                box,
                                image.shape[:2],
                                base_gain=args.roi_expand,
                                small_gain=getattr(args, "roi_small_gain", 1.6),
                                edge_gain=getattr(args, "roi_edge_gain", 1.45),
                                large_gain=getattr(args, "roi_large_gain", 1.15),
                            )
                            if getattr(args, "enable_adaptive_roi", False)
                            else args.roi_expand
                        )
                        crop_xyxy = expand_xyxy(box["xyxy"], image.shape[:2], roi_gain)
                        x1, y1, x2, y2 = crop_xyxy
                        crop_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, float(image.shape[0] * image.shape[1]))
                        estimated_kb = max(1.0, full_frame_kb * crop_ratio)
                        roi_schema = learned_policy.get("roi_features", None)
                        policy_started = time.perf_counter()
                        roi_eval = policy_decision(
                            learned_policy,
                            "roi",
                            roi_features(box, image.shape[:2], context, roi_schema),
                            estimated_kb,
                            args.learned_cost_weight,
                        )
                        policy_elapsed = (time.perf_counter() - policy_started) * 1000.0
                        policy_image_latency += policy_elapsed
                        policy_call_latencies.append(policy_elapsed)
                        stats["policy_roi_calls"] += 1
                        prob = roi_eval["probability"]
                        expected_gain = roi_eval["expected_gain"]
                        learned_score = roi_eval["decision"]
                        if prob >= args.learned_roi_threshold and learned_score > 0.0:
                            learned_roi_candidates.append(
                                (learned_score, prob, expected_gain, estimated_kb, edge_index, box, roi_gain, crop_xyxy)
                            )
                    learned_roi_candidates.sort(key=lambda item: item[0], reverse=True)

                    for learned_score, prob, expected_gain, estimated_kb, edge_index, box, roi_gain, crop_xyxy in learned_roi_candidates[
                        : args.learned_max_roi_per_image
                    ]:
                        payload, payload_bytes = make_evaluation_payload(
                            args,
                            image,
                            image_id,
                            "roi",
                            ["learned_roi"],
                            edge_boxes,
                            prob,
                            args.jpeg_quality,
                            crop_xyxy=crop_xyxy,
                            target_edge_box=box,
                        )
                        payload["metadata"]["roi_gain"] = round(float(roi_gain), 3)
                        payload["metadata"]["learned_prob"] = round(float(prob), 4)
                        payload["metadata"]["learned_decision"] = round(float(learned_score), 4)
                        payload_sizes.append(payload_bytes)
                        stats["payload_roi"] += 1
                        crop = crop_image(image, crop_xyxy)
                        cloud_boxes, cloud_latency = cloud_predict(
                            cloud_model, uploaded_image_for_cloud(crop, args, payload), args
                        )
                        cloud_boxes = shift_boxes(cloud_boxes, (crop_xyxy[0], crop_xyxy[1]))
                        cloud_latencies.append(cloud_latency)
                        roi_extra_conf = args.cloud_conf if getattr(args, "roi_extra_cloud_conf", -1.0) < 0 else args.roi_extra_cloud_conf
                        final_boxes = apply_roi_cloud(
                            final_boxes,
                            edge_index,
                            cloud_boxes,
                            args.iou_match,
                            args.cloud_conf,
                            roi_extra_conf,
                            getattr(args, "roi_veto_edge_conf", 1.0),
                        )
                        fusion = fuse_detections(
                            edge_boxes_for_fusion(payload["metadata"]), cloud_boxes, "roi", args.iou_match, args.cloud_conf
                        )
                        fusion_events.update(fusion["counts"])
                        image_events.append(
                            {
                                "image": str(image_path),
                                "payload_type": "roi",
                                "reason": ["learned_roi"],
                                "learned_prob": round(float(prob), 4),
                                "learned_expected_gain": round(float(expected_gain), 4),
                                "learned_decision": round(float(learned_score), 4),
                                "estimated_kb": round(float(estimated_kb), 3),
                                "roi_gain": round(float(roi_gain), 3),
                                "crop_xyxy": crop_xyxy,
                                "cloud_boxes": len(cloud_boxes),
                                "cloud_latency_ms": round(cloud_latency, 3),
                                "fusion_counts": fusion["counts"],
                            }
                        )

                policy_latencies_per_image.append(policy_image_latency)

            if learned_policy is None and args.keyframe_interval > 0 and image_id % args.keyframe_interval == 0:
                payload, payload_bytes = make_evaluation_payload(
                    args,
                    image, image_id, "keyframe", ["periodic_keyframe"], edge_boxes, 0.55, args.jpeg_quality
                )
                payload_sizes.append(payload_bytes)
                stats["payload_keyframe"] += 1
                cloud_boxes, cloud_latency = cloud_predict(
                    cloud_model, uploaded_image_for_cloud(image, args, payload), args
                )
                cloud_latencies.append(cloud_latency)
                final_boxes = cloud_boxes
                full_frame_uploaded = True
                image_events.append(
                    {
                        "image": str(image_path),
                        "payload_type": "keyframe",
                        "reason": ["periodic_keyframe"],
                        "cloud_boxes": len(cloud_boxes),
                        "cloud_latency_ms": round(cloud_latency, 3),
                    }
                )

            if learned_policy is None and getattr(args, "enable_complexity_probe", False) and not full_frame_uploaded:
                should_probe, complexity, reasons = should_complexity_probe(
                    image,
                    edge_boxes,
                    complexity_threshold=args.complexity_threshold,
                    empty_complexity_threshold=args.empty_complexity_threshold,
                    max_detections=args.complexity_max_detections,
                    coverage_threshold=args.complexity_coverage_threshold,
                )
                if should_probe:
                    payload, payload_bytes = make_evaluation_payload(
                        args,
                        image, image_id, "keyframe", reasons, edge_boxes, complexity, args.jpeg_quality
                    )
                    payload_sizes.append(payload_bytes)
                    stats["payload_keyframe"] += 1
                    cloud_boxes, cloud_latency = cloud_predict(
                        cloud_model, uploaded_image_for_cloud(image, args, payload), args
                    )
                    cloud_latencies.append(cloud_latency)
                    final_boxes = cloud_boxes
                    full_frame_uploaded = True
                    image_events.append(
                        {
                            "image": str(image_path),
                            "payload_type": "keyframe",
                            "reason": reasons,
                            "complexity": round(complexity, 4),
                            "cloud_boxes": len(cloud_boxes),
                            "cloud_latency_ms": round(cloud_latency, 3),
                        }
                    )

            if learned_policy is None and getattr(args, "enable_stability_probe", False) and not full_frame_uploaded:
                stability_started = time.perf_counter()
                stability_kwargs = {
                    "source": image,
                    "imgsz": getattr(args, "stability_imgsz", 384),
                    "conf": args.conf,
                    "iou": args.model_nms_iou,
                    "max_det": args.max_det,
                    "end2end": edge_end2end,
                    "verbose": False,
                }
                if args.device:
                    stability_kwargs["device"] = args.device
                stability_result = edge_model.predict(**stability_kwargs)[0]
                edge_latencies[-1] += (time.perf_counter() - stability_started) * 1000.0
                secondary_boxes = boxes_from_result(stability_result)
                stability_score, stability_reasons = stability_probe_score(
                    edge_boxes,
                    secondary_boxes,
                    getattr(args, "stability_iou", 0.5),
                )
                if stability_score >= getattr(args, "stability_threshold", 0.45):
                    payload, payload_bytes = make_evaluation_payload(
                        args,
                        image,
                        image_id,
                        "keyframe",
                        stability_reasons,
                        edge_boxes,
                        stability_score,
                        args.jpeg_quality,
                    )
                    payload_sizes.append(payload_bytes)
                    stats["payload_keyframe"] += 1
                    cloud_boxes, cloud_latency = cloud_predict(
                        cloud_model, uploaded_image_for_cloud(image, args, payload), args
                    )
                    cloud_latencies.append(cloud_latency)
                    final_boxes = cloud_boxes
                    full_frame_uploaded = True
                    image_events.append(
                        {
                            "image": str(image_path),
                            "payload_type": "keyframe",
                            "reason": stability_reasons,
                            "stability_score": round(float(stability_score), 4),
                            "secondary_boxes": len(secondary_boxes),
                            "cloud_boxes": len(cloud_boxes),
                            "cloud_latency_ms": round(cloud_latency, 3),
                        }
                    )

            candidates: list[tuple[float, list[str], int, dict[str, Any]]] = []
            for edge_index, box in enumerate(edge_boxes):
                score, reasons = detection_upload_score(box, image.shape[:2], [], use_temporal=False)
                if score >= args.upload_threshold:
                    candidates.append((score, reasons, edge_index, box))
            candidates.sort(key=lambda item: item[0], reverse=True)

            for score, reasons, edge_index, box in ([] if learned_policy is not None else candidates[: args.max_roi_per_image]):
                roi_gain = (
                    adaptive_roi_gain(
                        box,
                        image.shape[:2],
                        base_gain=args.roi_expand,
                        small_gain=getattr(args, "roi_small_gain", 2.0),
                        edge_gain=getattr(args, "roi_edge_gain", 1.65),
                        large_gain=getattr(args, "roi_large_gain", 1.15),
                    )
                    if getattr(args, "enable_adaptive_roi", False)
                    else args.roi_expand
                )
                crop_xyxy = expand_xyxy(box["xyxy"], image.shape[:2], roi_gain)
                payload, payload_bytes = make_evaluation_payload(
                    args,
                    image,
                    image_id,
                    "roi",
                    reasons,
                    edge_boxes,
                    score,
                    args.jpeg_quality,
                    crop_xyxy=crop_xyxy,
                    target_edge_box=box,
                )
                payload_sizes.append(payload_bytes)
                stats["payload_roi"] += 1
                crop = crop_image(image, crop_xyxy)
                cloud_boxes, cloud_latency = cloud_predict(
                    cloud_model, uploaded_image_for_cloud(crop, args, payload), args
                )
                cloud_boxes = shift_boxes(cloud_boxes, (crop_xyxy[0], crop_xyxy[1]))
                cloud_latencies.append(cloud_latency)
                roi_extra_conf = args.cloud_conf if getattr(args, "roi_extra_cloud_conf", -1.0) < 0 else args.roi_extra_cloud_conf
                final_boxes = apply_roi_cloud(
                    final_boxes,
                    edge_index,
                    cloud_boxes,
                    args.iou_match,
                    args.cloud_conf,
                    roi_extra_conf,
                    getattr(args, "roi_veto_edge_conf", 1.0),
                )

                fusion = fuse_detections(
                    edge_boxes_for_fusion(payload["metadata"]), cloud_boxes, "roi", args.iou_match, args.cloud_conf
                )
                fusion_events.update(fusion["counts"])
                for event in fusion["events"]:
                    if len(examples) < 20:
                        examples.append(
                            {
                                "image": str(image_path),
                                "payload_type": "roi",
                                "reason": reasons,
                                "event": event["type"],
                            }
                        )
                image_events.append(
                    {
                        "image": str(image_path),
                        "payload_type": "roi",
                        "reason": reasons,
                        "score": score,
                        "roi_gain": round(float(roi_gain), 3),
                        "crop_xyxy": crop_xyxy,
                        "cloud_boxes": len(cloud_boxes),
                        "cloud_latency_ms": round(cloud_latency, 3),
                        "fusion_counts": fusion["counts"],
                    }
                )

            if learned_policy is None and getattr(args, "enable_lowconf_proposal", False) and not full_frame_uploaded:
                proposals = low_conf_proposals(
                    candidate_edge_boxes,
                    edge_boxes,
                    image.shape[:2],
                    min_conf=getattr(args, "proposal_conf", 0.05),
                    max_conf=args.conf,
                    max_proposals=getattr(args, "max_proposals_per_image", 2),
                    score_threshold=getattr(args, "proposal_score_threshold", 0.55),
                )
                for proposal in proposals:
                    crop_xyxy = expand_xyxy(proposal["xyxy"], image.shape[:2], args.roi_expand)
                    payload, payload_bytes = make_evaluation_payload(
                        args,
                        image,
                        image_id,
                        "proposal",
                        proposal.get("proposal_reason", ["suppressed_low_conf"]),
                        edge_boxes,
                        float(proposal.get("proposal_score", proposal.get("conf", 0.0))),
                        args.jpeg_quality,
                        crop_xyxy=crop_xyxy,
                        target_edge_box=proposal,
                    )
                    payload_sizes.append(payload_bytes)
                    stats["payload_proposal"] += 1
                    crop = crop_image(image, crop_xyxy)
                    cloud_boxes, cloud_latency = cloud_predict(
                        cloud_model, uploaded_image_for_cloud(crop, args, payload), args
                    )
                    cloud_boxes = shift_boxes(cloud_boxes, (crop_xyxy[0], crop_xyxy[1]))
                    cloud_latencies.append(cloud_latency)
                    proposal_cloud_conf = getattr(args, "proposal_cloud_conf", 0.35)
                    final_boxes, accepted_cloud_boxes = apply_proposal_cloud(
                        final_boxes,
                        proposal,
                        cloud_boxes,
                        getattr(args, "proposal_iou_match", 0.25),
                        proposal_cloud_conf,
                    )
                    fusion = fuse_detections(
                        [proposal],
                        cloud_boxes,
                        "proposal",
                        getattr(args, "proposal_iou_match", 0.25),
                        proposal_cloud_conf,
                    )
                    fusion_events.update(fusion["counts"])
                    for event in fusion["events"]:
                        if len(examples) < 20:
                            examples.append(
                                {
                                    "image": str(image_path),
                                    "payload_type": "proposal",
                                    "reason": proposal.get("proposal_reason", ["suppressed_low_conf"]),
                                    "event": event["type"],
                                }
                            )
                    image_events.append(
                        {
                            "image": str(image_path),
                            "payload_type": "proposal",
                            "reason": proposal.get("proposal_reason", ["suppressed_low_conf"]),
                            "score": proposal.get("proposal_score"),
                            "proposal_conf": proposal.get("conf"),
                            "crop_xyxy": crop_xyxy,
                            "cloud_boxes": len(cloud_boxes),
                            "accepted_cloud_boxes": accepted_cloud_boxes,
                            "proposal_cloud_conf": proposal_cloud_conf,
                            "cloud_latency_ms": round(cloud_latency, 3),
                            "fusion_counts": fusion["counts"],
                        }
                    )

            if learned_policy is None and getattr(args, "enable_tile_probe", False) and not full_frame_uploaded:
                for tile in tile_candidates(
                    image,
                    edge_boxes,
                    grid=parse_grid(args.tile_grid),
                    max_tiles=args.max_tiles_per_image,
                    score_threshold=args.tile_score_threshold,
                ):
                    payload, payload_bytes = make_evaluation_payload(
                        args,
                        image,
                        image_id,
                        "tile",
                        tile["reason"],
                        edge_boxes,
                        tile["score"],
                        args.jpeg_quality,
                        crop_xyxy=tile["xyxy"],
                    )
                    payload["metadata"]["tile"] = tile
                    payload_sizes.append(payload_bytes)
                    stats["payload_tile"] += 1
                    crop = crop_image(image, tile["xyxy"])
                    cloud_boxes, cloud_latency = cloud_predict(
                        cloud_model, uploaded_image_for_cloud(crop, args, payload), args
                    )
                    cloud_boxes = shift_boxes(cloud_boxes, (tile["xyxy"][0], tile["xyxy"][1]))
                    cloud_latencies.append(cloud_latency)
                    tile_conf = getattr(args, "tile_cloud_conf", 0.45)
                    final_boxes, accepted_cloud_boxes = apply_tile_cloud(
                        final_boxes,
                        cloud_boxes,
                        tile["xyxy"],
                        image.shape[:2],
                        tile_conf,
                        getattr(args, "tile_inner_margin", 0.03),
                        getattr(args, "tile_min_area_ratio", 0.0),
                    )
                    fusion = fuse_detections(
                        edge_boxes_for_fusion(payload["metadata"]), cloud_boxes, "tile", args.iou_match, tile_conf
                    )
                    fusion_events.update(fusion["counts"])
                    for event in fusion["events"]:
                        if len(examples) < 20:
                            examples.append(
                                {
                                    "image": str(image_path),
                                    "payload_type": "tile",
                                    "reason": tile["reason"],
                                    "event": event["type"],
                                }
                            )
                    image_events.append(
                        {
                            "image": str(image_path),
                            "payload_type": "tile",
                            "reason": tile["reason"],
                            "score": tile["score"],
                            "crop_xyxy": tile["xyxy"],
                            "cloud_boxes": len(cloud_boxes),
                            "accepted_cloud_boxes": accepted_cloud_boxes,
                            "tile_cloud_conf": tile_conf,
                            "cloud_latency_ms": round(cloud_latency, 3),
                            "fusion_counts": fusion["counts"],
                        }
                    )

            final_boxes = class_aware_nms(final_boxes, args.fused_nms_iou)[: args.max_det]
            fused_operating_counts.update(fixed_threshold_counts(final_boxes, labels))
            pipeline_latencies.append((time.perf_counter() - pipeline_started) * 1000.0)
            update_metrics(fused_metrics, final_boxes, labels, image_path.name)

            if image_events:
                uploaded_image_ids.add(image_id)
                for event in image_events:
                    events_file.write(json.dumps(event, ensure_ascii=False) + "\n")

    edge_results = metrics_to_dict(edge_metrics, save_dir)
    edge_deploy_results = metrics_to_dict(edge_deploy_metrics, save_dir)
    cloud_only_results = metrics_to_dict(cloud_only_metrics, save_dir) if cloud_only_metrics is not None else None
    cloud_only_deploy_results = (
        metrics_to_dict(cloud_only_deploy_metrics, save_dir) if cloud_only_deploy_metrics is not None else None
    )
    full_frame_fusion_results = (
        metrics_to_dict(full_frame_fusion_metrics, save_dir) if full_frame_fusion_metrics is not None else None
    )
    fused_results = metrics_to_dict(fused_metrics, save_dir)
    frames = max(1, len(image_paths))
    cloud_only_prediction_artifacts: dict[str, Any] = {}
    if export_cloud_only_coco:
        processed_export_ids = {int(path.stem) for path in image_paths}
        if not processed_export_ids <= expected_export_image_ids:
            raise RuntimeError("Cloud-only export contains image IDs outside the annotation set")
        if not args.max_images and processed_export_ids != expected_export_image_ids:
            missing = sorted(expected_export_image_ids - processed_export_ids)
            raise RuntimeError(
                f"Full Cloud-only export does not cover all annotation images; missing={missing[:5]}"
            )
        official_path = save_dir / "cloud_only_official_predictions.json"
        deploy_path = save_dir / "cloud_only_deploy_predictions.json"
        official_path.write_text(
            json.dumps(cloud_only_official_predictions, separators=(",", ":")),
            encoding="utf-8",
        )
        deploy_path.write_text(
            json.dumps(cloud_only_deploy_predictions, separators=(",", ":")),
            encoding="utf-8",
        )
        cloud_only_prediction_artifacts = {
            "official": str(official_path),
            "deploy": str(deploy_path),
            "processed_image_ids": len(processed_export_ids),
            "official_predictions": len(cloud_only_official_predictions),
            "deploy_predictions": len(cloud_only_deploy_predictions),
            "annotations": str(Path(args.coco_annotations).resolve()),
        }
    evaluation_config = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_") and (value is None or isinstance(value, (str, int, float, bool)))
    }
    summary = {
        "source": args.source,
        "images": len(image_paths),
        "edge_model": args.edge_model,
        "cloud_model": args.cloud_model,
        "imgsz": args.imgsz,
        "cloud_imgsz": getattr(args, "cloud_imgsz", args.imgsz),
        "adaptive_cloud_imgsz": bool(getattr(args, "enable_learned_adaptive_cloud_imgsz", False)),
        "cloud_compute_uploads": int(getattr(args, "_cloud_compute_uploads", 0)),
        "cloud_compute_scale_sum": round(float(getattr(args, "_cloud_compute_scale_sum", 0.0)), 6),
        "cloud_compute_scale_per_upload": round(
            float(getattr(args, "_cloud_compute_scale_sum", 0.0)) / max(1, int(getattr(args, "_cloud_compute_uploads", 0))),
            6,
        ),
        "edge_conf": args.conf,
        "cloud_conf": args.cloud_conf,
        "end2end": edge_end2end if edge_end2end == cloud_end2end else None,
        "head_mode": (
            ("end2end" if edge_end2end else "end2many") if edge_end2end == cloud_end2end else "mixed"
        ),
        "edge_end2end": edge_end2end,
        "cloud_end2end": cloud_end2end,
        "edge_head_mode": "end2end" if edge_end2end else "end2many",
        "cloud_head_mode": "end2end" if cloud_end2end else "end2many",
        "edge_metric_conf": edge_metric_conf,
        "cloud_metric_conf": cloud_metric_conf,
        "jpeg_quality": args.jpeg_quality,
        "simulate_upload_codec": bool(getattr(args, "simulate_upload_codec", False)),
        "simulate_cloud_only_codec": bool(getattr(args, "simulate_cloud_only_codec", False)),
        "warmup_runs": int(getattr(args, "warmup_runs", 0)),
        "evaluation_config": evaluation_config,
        "upload_threshold": args.upload_threshold,
        "max_roi_per_image": args.max_roi_per_image,
        "keyframe_interval": args.keyframe_interval,
        "uploads_submitted": int(
            stats["payload_roi"] + stats["payload_keyframe"] + stats["payload_proposal"] + stats["payload_tile"]
        ),
        "upload_ratio": round(
            float(stats["payload_roi"] + stats["payload_keyframe"] + stats["payload_proposal"] + stats["payload_tile"])
            / frames,
            4,
        ),
        "uploaded_images": len(uploaded_image_ids),
        "uploaded_image_ratio": round(len(uploaded_image_ids) / frames, 4),
        "roi_uploads": int(stats["payload_roi"]),
        "keyframe_uploads": int(stats["payload_keyframe"]),
        "proposal_uploads": int(stats["payload_proposal"]),
        "tile_uploads": int(stats["payload_tile"]),
        "avg_payload_kb": round((statistics.mean(payload_sizes) / 1024.0) if payload_sizes else 0.0, 3),
        "payload_kb_p95": round(percentile(payload_sizes, 0.95) / 1024.0, 3),
        "total_payload_bytes": int(sum(payload_sizes)),
        "total_payload_mb": round(sum(payload_sizes) / (1024.0 * 1024.0), 3),
        "total_response_bytes": int(sum(args._cloud_response_sizes)),
        "cloud_response_count": len(args._cloud_response_sizes),
        "avg_response_kb": round(
            (statistics.mean(args._cloud_response_sizes) / 1024.0) if args._cloud_response_sizes else 0.0,
            3,
        ),
        "response_kb_p95": round(percentile(args._cloud_response_sizes, 0.95) / 1024.0, 3),
        "edge_latency_ms_avg": round(statistics.mean(edge_latencies), 3) if edge_latencies else 0.0,
        "edge_latency_ms_p95": round(percentile(edge_latencies, 0.95), 3),
        "cloud_latency_ms_avg": round(statistics.mean(cloud_latencies), 3) if cloud_latencies else 0.0,
        "cloud_latency_ms_p95": round(percentile(cloud_latencies, 0.95), 3),
        "upload_encode_ms_avg": round(statistics.mean(args._upload_encode_latencies), 3)
        if args._upload_encode_latencies
        else 0.0,
        "upload_encode_ms_p95": round(percentile(args._upload_encode_latencies, 0.95), 3),
        "upload_decode_ms_avg": round(statistics.mean(args._upload_decode_latencies), 3)
        if args._upload_decode_latencies
        else 0.0,
        "upload_decode_ms_p95": round(percentile(args._upload_decode_latencies, 0.95), 3),
        "upload_encode_count": len(args._upload_encode_latencies),
        "upload_decode_count": len(args._upload_decode_latencies),
        "pipeline_latency_ms_avg": round(statistics.mean(pipeline_latencies), 3) if pipeline_latencies else 0.0,
        "pipeline_latency_ms_p95": round(percentile(pipeline_latencies, 0.95), 3),
        "policy_frame_calls": int(stats["policy_frame_calls"]),
        "policy_roi_calls": int(stats["policy_roi_calls"]),
        "policy_latency_ms_avg_per_source_image": round(statistics.mean(policy_latencies_per_image), 6)
        if policy_latencies_per_image
        else 0.0,
        "policy_latency_ms_p95_per_source_image": round(percentile(policy_latencies_per_image, 0.95), 6),
        "policy_latency_ms_avg_per_call": round(statistics.mean(policy_call_latencies), 6)
        if policy_call_latencies
        else 0.0,
        "policy_latency_ms_p95_per_call": round(percentile(policy_call_latencies, 0.95), 6),
        "cloud_only_total_payload_mb": round(sum(cloud_only_payload_sizes) / (1024.0 * 1024.0), 3),
        "cloud_only_total_payload_bytes": int(sum(cloud_only_payload_sizes)),
        "cloud_only_payload_count": len(cloud_only_payload_sizes),
        "cloud_only_total_response_bytes": int(sum(cloud_only_response_sizes)),
        "cloud_only_response_count": len(cloud_only_response_sizes),
        "cloud_only_avg_response_kb": round(
            (statistics.mean(cloud_only_response_sizes) / 1024.0) if cloud_only_response_sizes else 0.0,
            3,
        ),
        "cloud_only_response_kb_p95": round(percentile(cloud_only_response_sizes, 0.95) / 1024.0, 3),
        "cloud_only_avg_payload_kb": round(
            (statistics.mean(cloud_only_payload_sizes) / 1024.0) if cloud_only_payload_sizes else 0.0,
            3,
        ),
        "cloud_only_payload_kb_p95": round(percentile(cloud_only_payload_sizes, 0.95) / 1024.0, 3),
        "cloud_only_latency_ms_avg": round(statistics.mean(cloud_only_latencies), 3)
        if cloud_only_latencies
        else 0.0,
        "cloud_only_latency_ms_p95": round(percentile(cloud_only_latencies, 0.95), 3),
        "cloud_only_encode_ms_avg": round(statistics.mean(cloud_only_encode_latencies), 3)
        if cloud_only_encode_latencies
        else 0.0,
        "cloud_only_encode_ms_p95": round(percentile(cloud_only_encode_latencies, 0.95), 3),
        "cloud_only_decode_ms_avg": round(statistics.mean(cloud_only_decode_latencies), 3)
        if cloud_only_decode_latencies
        else 0.0,
        "cloud_only_decode_ms_p95": round(percentile(cloud_only_decode_latencies, 0.95), 3),
        "fusion_events": dict(fusion_events),
        "edge_only_metrics": edge_results,
        "edge_only_deploy_metrics": edge_deploy_results,
        "edge_only_operating_point": fixed_threshold_metrics(edge_operating_counts),
        "cloud_only_metrics": cloud_only_results,
        "cloud_only_deploy_metrics": cloud_only_deploy_results,
        "cloud_only_operating_point": fixed_threshold_metrics(cloud_operating_counts)
        if cloud_only_metrics is not None
        else None,
        "cloud_only_prediction_artifacts": cloud_only_prediction_artifacts,
        "full_frame_fusion_upper_metrics": full_frame_fusion_results,
        "edge_cloud_fused_metrics": fused_results,
        "edge_cloud_fused_deploy_metrics": fused_results,
        "edge_cloud_fused_operating_point": fixed_threshold_metrics(fused_operating_counts),
        "examples": examples,
    }
    summary_path = save_dir / "edge_cloud_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if print_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Summary: {summary_path}")
        print(f"Events: {events_path}")
    return summary


def main() -> None:
    """Run edge-cloud evaluation."""
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
