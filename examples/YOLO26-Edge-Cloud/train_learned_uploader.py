# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Train a lightweight learned upload policy from a YOLO-format dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from edge_client import collect_image_paths, detection_upload_score
from evaluate_edge_cloud import (
    apply_roi_cloud,
    class_aware_nms,
    cloud_predict,
    load_labels,
    str_to_bool,
)
from learned_uploader import fit_logistic, fit_value_regressor, frame_features, image_context, roi_features, save_policy
from protocol import boxes_from_result, crop_image, decode_jpeg, encode_jpeg, expand_xyxy, shift_boxes
from upload_strategies import adaptive_roi_gain


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=r"F:\IDE\trae\datasets\coco128", help="YOLO-format dataset image directory.")
    parser.add_argument("--edge-model", default="yolo26s.pt", help="Edge model.")
    parser.add_argument("--cloud-model", default="yolo26x.pt", help="Cloud model.")
    parser.add_argument("--save-dir", default="runs/edge_cloud/learned_policy", help="Output directory.")
    parser.add_argument("--policy-path", default="", help="Output policy JSON path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--cloud-imgsz", type=int, default=640, help="Cloud inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Edge deployment confidence threshold.")
    parser.add_argument("--cloud-conf", type=float, default=0.25, help="Cloud confidence threshold.")
    parser.add_argument("--model-nms-iou", type=float, default=0.7, help="Edge/cloud model NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per model invocation.")
    parser.add_argument(
        "--end2end",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
        help="YOLO26 head mode; false selects end2many. Omit to preserve the model default.",
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
        "--simulate-cloud-codec",
        action="store_true",
        help="Generate cloud utility labels from JPEG-decoded full-frame and ROI inputs.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality for codec-aware cloud labels.")
    parser.add_argument("--device", default="0", help="Ultralytics device string.")
    parser.add_argument("--edge-batch-size", type=int, default=8, help="Batch size for edge full-frame labeling inference.")
    parser.add_argument("--cloud-batch-size", type=int, default=4, help="Batch size for cloud full-frame labeling inference.")
    parser.add_argument("--roi-batch-size", type=int, default=8, help="Batch size for cloud ROI labeling inference.")
    parser.add_argument("--half", action="store_true", help="Use fp16 inference when supported by the backend.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images; 0 means all.")
    parser.add_argument("--max-roi-candidates", type=int, default=3, help="Maximum ROI candidates labeled per image.")
    parser.add_argument("--candidate-threshold", type=float, default=0.25, help="Minimum heuristic ROI score for labeling.")
    parser.add_argument("--roi-expand", type=float, default=1.35, help="Base ROI expansion.")
    parser.add_argument("--enable-adaptive-roi", action="store_true", help="Use adaptive ROI crops during labeling.")
    parser.add_argument("--roi-small-gain", type=float, default=1.6, help="ROI gain for small objects.")
    parser.add_argument("--roi-edge-gain", type=float, default=1.45, help="ROI gain for edge-near objects.")
    parser.add_argument("--roi-large-gain", type=float, default=1.15, help="ROI gain for large objects.")
    parser.add_argument("--roi-extra-cloud-conf", type=float, default=0.55, help="Unmatched cloud box confidence for utility simulation.")
    parser.add_argument("--roi-veto-edge-conf", type=float, default=0.25, help="Only drop unmatched ROI edge boxes below this confidence.")
    parser.add_argument("--iou-match", type=float, default=0.5, help="ROI/cloud match IoU.")
    parser.add_argument("--fused-nms-iou", type=float, default=0.7, help="Class-aware NMS IoU.")
    parser.add_argument("--fp-penalty", type=float, default=0.25, help="False-positive penalty for utility labels.")
    parser.add_argument("--utility-margin", type=float, default=0.0, help="Minimum utility gain for positive labels.")
    parser.add_argument("--epochs", type=int, default=800, help="Logistic regression epochs.")
    parser.add_argument("--lr", type=float, default=0.08, help="Logistic regression learning rate.")
    parser.add_argument("--l2", type=float, default=0.01, help="Logistic regression L2 penalty.")
    parser.add_argument("--value-epochs", type=int, default=500, help="Value regressor epochs.")
    parser.add_argument("--value-lr", type=float, default=0.05, help="Value regressor learning rate.")
    parser.add_argument("--value-l2", type=float, default=0.02, help="Value regressor L2 penalty.")
    parser.add_argument("--value-huber-delta", type=float, default=0.15, help="Huber delta for the value regressor.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split for the lightweight policy heads.")
    parser.add_argument("--feature-noise-std", type=float, default=0.02, help="Feature noise regularization for better cross-scene generalization.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/validation split.")
    parser.add_argument(
        "--gain-normalizer",
        choices=("labels", "none"),
        default="labels",
        help="Normalize utility gain by label count for better cross-dataset transfer.",
    )
    return parser.parse_args()


def detection_utility(preds: list[dict[str, Any]], labels: list[dict[str, Any]], fp_penalty: float = 0.25) -> float:
    """Object-level utility proxy: true positives minus a false-positive penalty."""
    from protocol import box_iou

    pairs: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(preds):
        for label_idx, label in enumerate(labels):
            if int(pred["cls"]) != int(label["cls"]):
                continue
            iou = box_iou(pred["xyxy"], label["xyxy"])
            if iou >= 0.5:
                pairs.append((iou, pred_idx, label_idx))
    pairs.sort(reverse=True)
    used_preds: set[int] = set()
    used_labels: set[int] = set()
    for _, pred_idx, label_idx in pairs:
        if pred_idx in used_preds or label_idx in used_labels:
            continue
        used_preds.add(pred_idx)
        used_labels.add(label_idx)
    tp = len(used_preds)
    fp = max(0, len(preds) - tp)
    return float(tp - fp_penalty * fp)


def normalize_gain(gain: float, labels: list[dict[str, Any]], mode: str) -> float:
    """Normalize utility gains so training targets transfer across scene densities."""
    if mode == "labels":
        return float(gain / max(1, len(labels)))
    return float(gain)


def batched_predict(
    model: YOLO,
    images: list[Any],
    imgsz: int,
    conf: float,
    device: str,
    batch_size: int,
    half: bool,
    model_nms_iou: float,
    max_det: int,
    end2end: bool | None,
) -> list[list[dict[str, Any]]]:
    """Run batched Ultralytics inference and return detection dicts per image."""
    if not images:
        return []
    kwargs = {
        "source": images,
        "imgsz": imgsz,
        "conf": conf,
        "iou": model_nms_iou,
        "max_det": max_det,
        "verbose": False,
        "batch": max(1, batch_size),
    }
    if end2end is not None:
        kwargs["end2end"] = end2end
    if device:
        kwargs["device"] = device
    if half:
        kwargs["half"] = True
    return [boxes_from_result(result) for result in model.predict(**kwargs)]


def prepare_cloud_inputs(images: list[Any], simulate_codec: bool, jpeg_quality: int) -> list[Any]:
    """Return the exact pixels used to generate cloud utility labels."""
    if not simulate_codec:
        return images
    return [decode_jpeg(encode_jpeg(image, jpeg_quality)) for image in images]


def resolve_end2end_modes(args: argparse.Namespace) -> tuple[bool | None, bool | None]:
    """Resolve separate edge/cloud modes while preserving the shared shorthand."""
    shared = getattr(args, "end2end", None)
    edge = getattr(args, "edge_end2end", None)
    cloud = getattr(args, "cloud_end2end", None)
    return (shared if edge is None else edge, shared if cloud is None else cloud)


def training_protocol(args: argparse.Namespace) -> dict[str, Any]:
    """Return the frozen detector and codec settings saved with the policy."""
    edge_end2end, cloud_end2end = resolve_end2end_modes(args)
    edge_head_mode = "model_default" if edge_end2end is None else ("end2end" if edge_end2end else "end2many")
    cloud_head_mode = "model_default" if cloud_end2end is None else ("end2end" if cloud_end2end else "end2many")
    shared_end2end = edge_end2end if edge_end2end == cloud_end2end else None
    head_mode = edge_head_mode if edge_head_mode == cloud_head_mode else "mixed"
    return {
        "edge_imgsz": args.imgsz,
        "cloud_imgsz": args.cloud_imgsz,
        "edge_conf": args.conf,
        "cloud_conf": args.cloud_conf,
        "model_nms_iou": args.model_nms_iou,
        "max_det": args.max_det,
        "end2end": shared_end2end,
        "head_mode": head_mode,
        "edge_end2end": edge_end2end,
        "cloud_end2end": cloud_end2end,
        "edge_head_mode": edge_head_mode,
        "cloud_head_mode": cloud_head_mode,
        "simulate_cloud_codec": args.simulate_cloud_codec,
        "jpeg_quality": args.jpeg_quality,
        "half": args.half,
    }


def main() -> None:
    """Train and save the learned upload policy."""
    args = parse_args()
    edge_end2end, cloud_end2end = resolve_end2end_modes(args)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    policy_path = Path(args.policy_path) if args.policy_path else save_dir / "learned_upload_policy.json"
    image_paths = collect_image_paths(args.source)
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.source}")

    print(f"Loading edge model: {args.edge_model}")
    edge_model = YOLO(args.edge_model)
    print(f"Loading cloud model: {args.cloud_model}")
    cloud_model = YOLO(args.cloud_model)

    roi_x: list[list[float]] = []
    roi_y: list[int] = []
    roi_value_y: list[float] = []
    frame_x: list[list[float]] = []
    frame_y: list[int] = []
    frame_value_y: list[float] = []
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()

    batch_stride = max(1, args.edge_batch_size)
    for batch_start in range(0, len(image_paths), batch_stride):
        batch_paths = image_paths[batch_start : batch_start + batch_stride]
        batch_records: list[dict[str, Any]] = []
        for image_path in batch_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            batch_records.append(
                {
                    "image_path": image_path,
                    "image": image,
                    "labels": load_labels(image_path, image.shape[:2]),
                }
            )
        if not batch_records:
            continue

        batch_images = [record["image"] for record in batch_records]
        edge_batches = batched_predict(
            edge_model,
            batch_images,
            args.imgsz,
            args.conf,
            args.device,
            args.edge_batch_size,
            args.half,
            args.model_nms_iou,
            args.max_det,
            edge_end2end,
        )
        cloud_full_images = prepare_cloud_inputs(batch_images, args.simulate_cloud_codec, args.jpeg_quality)
        cloud_full_batches = batched_predict(
            cloud_model,
            cloud_full_images,
            args.cloud_imgsz,
            args.cloud_conf,
            args.device,
            args.cloud_batch_size,
            args.half,
            args.model_nms_iou,
            args.max_det,
            cloud_end2end,
        )

        roi_jobs: list[dict[str, Any]] = []
        for record, edge_boxes, cloud_boxes_full in zip(batch_records, edge_batches, cloud_full_batches):
            image = record["image"]
            labels = record["labels"]
            image_path = record["image_path"]
            context = image_context(image, edge_boxes)
            edge_utility = detection_utility(class_aware_nms(edge_boxes, args.fused_nms_iou), labels, args.fp_penalty)
            frame_utility = detection_utility(class_aware_nms(cloud_boxes_full, args.fused_nms_iou), labels, args.fp_penalty)
            frame_gain_raw = frame_utility - edge_utility
            frame_gain = normalize_gain(frame_gain_raw, labels, args.gain_normalizer)
            frame_x.append(frame_features(context))
            frame_y.append(1 if frame_gain > args.utility_margin else 0)
            frame_value_y.append(frame_gain)
            samples.append(
                {
                    "image": str(image_path),
                    "task": "frame",
                    "gain": round(frame_gain, 4),
                    "gain_raw": round(frame_gain_raw, 4),
                    "label": frame_y[-1],
                    "edge_boxes": len(edge_boxes),
                    "cloud_boxes": len(cloud_boxes_full),
                }
            )

            candidates: list[tuple[float, list[str], int, dict[str, Any]]] = []
            for edge_index, box in enumerate(edge_boxes):
                score, reasons = detection_upload_score(box, image.shape[:2], [], use_temporal=False)
                if score >= args.candidate_threshold:
                    candidates.append((score, reasons, edge_index, box))
            candidates.sort(key=lambda item: item[0], reverse=True)

            for score, reasons, edge_index, box in candidates[: args.max_roi_candidates]:
                roi_gain_value = (
                    adaptive_roi_gain(
                        box,
                        image.shape[:2],
                        base_gain=args.roi_expand,
                        small_gain=args.roi_small_gain,
                        edge_gain=args.roi_edge_gain,
                        large_gain=args.roi_large_gain,
                    )
                    if args.enable_adaptive_roi
                    else args.roi_expand
                )
                crop_xyxy = expand_xyxy(box["xyxy"], image.shape[:2], roi_gain_value)
                roi_jobs.append(
                    {
                        "image_path": image_path,
                        "image_shape": image.shape[:2],
                        "labels": labels,
                        "context": context,
                        "edge_boxes": edge_boxes,
                        "edge_index": edge_index,
                        "edge_utility": edge_utility,
                        "box": box,
                        "score": score,
                        "reasons": reasons,
                        "crop_xyxy": crop_xyxy,
                        "crop": crop_image(image, crop_xyxy),
                    }
                )

        if roi_jobs:
            cloud_roi_inputs = prepare_cloud_inputs(
                [job["crop"] for job in roi_jobs],
                args.simulate_cloud_codec,
                args.jpeg_quality,
            )
            roi_cloud_batches = batched_predict(
                cloud_model,
                cloud_roi_inputs,
                args.cloud_imgsz,
                args.cloud_conf,
                args.device,
                args.roi_batch_size,
                args.half,
                args.model_nms_iou,
                args.max_det,
                cloud_end2end,
            )
            for job, cloud_boxes in zip(roi_jobs, roi_cloud_batches):
                crop_xyxy = job["crop_xyxy"]
                shifted_cloud_boxes = shift_boxes(cloud_boxes, (crop_xyxy[0], crop_xyxy[1]))
                final_boxes = apply_roi_cloud(
                    [dict(item) for item in job["edge_boxes"]],
                    job["edge_index"],
                    shifted_cloud_boxes,
                    args.iou_match,
                    args.cloud_conf,
                    args.roi_extra_cloud_conf,
                    args.roi_veto_edge_conf,
                )
                final_boxes = class_aware_nms(final_boxes, args.fused_nms_iou)
                roi_gain_raw = detection_utility(final_boxes, job["labels"], args.fp_penalty) - job["edge_utility"]
                roi_gain_utility = normalize_gain(roi_gain_raw, job["labels"], args.gain_normalizer)
                roi_x.append(roi_features(job["box"], job["image_shape"], job["context"]))
                roi_y.append(1 if roi_gain_utility > args.utility_margin else 0)
                roi_value_y.append(roi_gain_utility)
                samples.append(
                    {
                        "image": str(job["image_path"]),
                        "task": "roi",
                        "gain": round(roi_gain_utility, 4),
                        "gain_raw": round(roi_gain_raw, 4),
                        "label": roi_y[-1],
                        "score": round(job["score"], 4),
                        "reasons": job["reasons"],
                        "cloud_boxes": len(shifted_cloud_boxes),
                    }
                )

        processed_images = min(len(image_paths), batch_start + len(batch_records))
        if processed_images % 20 == 0 or processed_images == len(image_paths):
            elapsed = time.perf_counter() - started
            print(
                f"Labeled {processed_images}/{len(image_paths)} images, roi_samples={len(roi_y)}, "
                f"elapsed_min={elapsed / 60.0:.1f}",
                flush=True,
            )

    roi_model = fit_logistic(
        roi_x,
        roi_y,
        args.epochs,
        args.lr,
        args.l2,
        val_split=args.val_split,
        feature_noise_std=args.feature_noise_std,
        class_balance=True,
        seed=args.seed,
    )
    frame_model = fit_logistic(
        frame_x,
        frame_y,
        args.epochs,
        args.lr,
        args.l2,
        val_split=args.val_split,
        feature_noise_std=args.feature_noise_std,
        class_balance=True,
        seed=args.seed,
    )
    roi_value_model = fit_value_regressor(
        roi_x,
        roi_value_y,
        epochs=args.value_epochs,
        lr=args.value_lr,
        l2=args.value_l2,
        val_split=args.val_split,
        feature_noise_std=args.feature_noise_std,
        seed=args.seed,
        huber_delta=args.value_huber_delta,
    )
    frame_value_model = fit_value_regressor(
        frame_x,
        frame_value_y,
        epochs=args.value_epochs,
        lr=args.value_lr,
        l2=args.value_l2,
        val_split=args.val_split,
        feature_noise_std=args.feature_noise_std,
        seed=args.seed,
        huber_delta=args.value_huber_delta,
    )
    metadata = {
        "source": args.source,
        "edge_model": args.edge_model,
        "cloud_model": args.cloud_model,
        "images": len(image_paths),
        "roi_positive_rate": sum(roi_y) / max(1, len(roi_y)),
        "frame_positive_rate": sum(frame_y) / max(1, len(frame_y)),
        "roi_samples": len(roi_y),
        "frame_samples": len(frame_y),
        "fp_penalty": args.fp_penalty,
        "utility_margin": args.utility_margin,
        "gain_normalizer": args.gain_normalizer,
        "feature_noise_std": args.feature_noise_std,
        "val_split": args.val_split,
        "seed": args.seed,
        "training_objective": "probability_times_expected_gain_minus_cost",
        "training_seconds": round(time.perf_counter() - started, 3),
        "training_protocol": training_protocol(args),
    }
    save_policy(policy_path, roi_model, frame_model, metadata, roi_value_model=roi_value_model, frame_value_model=frame_value_model)
    (save_dir / "training_samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {
        "metadata": metadata,
        "roi_model": roi_model,
        "frame_model": frame_model,
        "roi_value_model": roi_value_model,
        "frame_value_model": frame_value_model,
        "policy_path": str(policy_path),
    }
    (save_dir / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
