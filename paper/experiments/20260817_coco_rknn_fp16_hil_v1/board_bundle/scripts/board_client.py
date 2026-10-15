#!/usr/bin/env python3
"""Resumable RK3588 CAVS-OQR client: candidates -> plan -> upload -> evaluate."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import cv2
import numpy as np

from cavs_numpy import current_roi_xyxy, load_policy, predict_scores
from fusion import maur_fuse_boxes
from protocol import coco_predictions, compact_boxes, encode_b64, native_jpeg, q90_jpeg
from scheduler import schedule_fixed_byte


ROOT = Path(__file__).resolve().parents[1]
ACTION_IDS = ("edge", "original", "q90", "current@640")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def latest_jsonl(path: Path, key: Callable[[Mapping[str, Any]], Any]) -> dict[Any, dict[str, Any]]:
    rows: dict[Any, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if isinstance(row, Mapping):
                    rows[key(row)] = dict(row)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
    return rows


def load_config() -> dict[str, Any]:
    return json.loads((ROOT / "configs" / "protocol.json").read_text(encoding="utf-8"))


def resolved(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def dataset_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    root = resolved(config["dataset"]["root"])
    return root, root / config["dataset"]["images"], root / config["dataset"]["annotations"]


def validate_policy_binding(
    policy: Mapping[str, Any], backend: str, expected_rknn_sha256: str
) -> None:
    if policy.get("backend") != backend:
        raise RuntimeError(f"policy backend mismatch: expected {backend}")
    if policy.get("rknn_model_sha256") != expected_rknn_sha256:
        raise RuntimeError("policy RKNN model mismatch")


def read_images(annotation_path: Path, images_dir: Path, limit: int) -> list[dict[str, Any]]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = sorted(annotation["images"], key=lambda row: row["file_name"])
    if limit:
        images = images[:limit]
    missing = [row["file_name"] for row in images if not (images_dir / row["file_name"]).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} image files; first: {missing[0]}")
    return images


def run_dir(args: argparse.Namespace) -> Path:
    return resolved(args.runs_root) / args.run_id / args.model


def stage_status(runs: Path, **values: Any) -> None:
    atomic_json(runs / "status.json", {"updated_at": time.time(), **values})


def ensure_run_contract(runs: Path, contract: Mapping[str, Any]) -> None:
    """Create one immutable run contract or verify an exact resume."""
    path = runs / "run_contract.json"
    expected = dict(contract)
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != expected:
            raise RuntimeError("run ID contract mismatch; use a fresh run ID")
        return
    artifacts = [item.name for item in runs.iterdir()]
    if artifacts:
        raise RuntimeError("run ID has artifacts but is missing its run contract")
    atomic_json(path, expected)


def imports_from_sibling() -> dict[str, Any]:
    sibling = (ROOT.parent / "yolo26n_coco_val").resolve()
    if not sibling.is_dir():
        raise FileNotFoundError(f"required sibling RKNN bundle is missing: {sibling}")
    if str(sibling) not in sys.path:
        sys.path.insert(0, str(sibling))
    from coco_val_rknn import NativeOutputCapture, core_mask, letterbox_val, scale_boxes, ultralytics_multilabel_nms
    from yolo26_decode import decode_rknn_outputs
    return {
        "NativeOutputCapture": NativeOutputCapture, "core_mask": core_mask, "letterbox_val": letterbox_val,
        "scale_boxes": scale_boxes, "ultralytics_multilabel_nms": ultralytics_multilabel_nms,
        "decode_rknn_outputs": decode_rknn_outputs,
    }


def validate_contract(args: argparse.Namespace, config: Mapping[str, Any], images: Sequence[Mapping[str, Any]]) -> tuple[Path, Path, Path]:
    _, images_dir, annotation_path = dataset_paths(config)
    model = resolved(config["models"][args.model]["path"])
    policy = resolved(config["models"]["policy"])
    if not model.is_file() or not policy.is_file() or not annotation_path.is_file():
        raise FileNotFoundError("missing RKNN model, policy, or annotation file")
    if sha256(model) != config["models"][args.model]["sha256"]:
        raise RuntimeError(f"{args.model} RKNN checksum changed")
    if sha256(policy) != config["models"]["policy_sha256"]:
        raise RuntimeError("frozen policy manifest checksum changed")
    validate_policy_binding(
        load_policy(policy), args.model, config["models"][args.model]["sha256"]
    )
    if not args.limit:
        if len(images) != int(config["dataset"]["expected_images"]):
            raise RuntimeError("formal run must contain exactly 5,000 COCO images")
        original_total = sum((images_dir / row["file_name"]).stat().st_size for row in images)
        if original_total != int(config["dataset"]["native_jpeg_total_bytes"]):
            raise RuntimeError(f"native JPEG denominator changed: {original_total}")
    return model, policy, annotation_path


def candidate_for_image(path: Path, image_id: int, shape: Sequence[int], edge_boxes: Sequence[Mapping[str, Any]], policy: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Build live O/Q/R scores and payload bytes from one raw RKNN Edge output."""
    started = time.perf_counter()
    height, width = int(shape[0]), int(shape[1])
    roi = current_roi_xyxy(width, height, edge_boxes, confidence=float(policy["roi"]["confidence"]), max_area_ratio=float(policy["roi"]["max_area_ratio"]))
    original = native_jpeg(path)
    q90 = q90_jpeg(path, long_side=int(config["codec"]["long_side"]))
    payloads: dict[str, int] = {"edge": 0, "original": len(original), "q90": len(q90)}
    if roi is not None:
        payloads["current@640"] = len(q90_jpeg(path, crop_xyxy=roi, long_side=int(config["codec"]["long_side"])))
    scores = predict_scores(policy, edge_boxes, width=width, height=height, roi_xyxy=roi)
    if not 0 < len(q90) < len(original):
        payloads.pop("q90")
        scores.pop("q90", None)
    return {"frame_id": int(image_id), "roi_xyxy": roi, "payload_bytes": payloads, "scores": {key: float(value) for key, value in scores.items() if key in payloads}, "candidate_ms": (time.perf_counter() - started) * 1000.0}


def run_candidates(args: argparse.Namespace, config: Mapping[str, Any], runs: Path, images: Sequence[Mapping[str, Any]], model_path: Path, policy_path: Path) -> dict[int, dict[str, Any]]:
    checkpoint = runs / "candidates.jsonl"
    completed = latest_jsonl(checkpoint, lambda row: int(row["image_id"]))
    wanted = {int(row["id"]) for row in images}
    if not set(completed) <= wanted:
        raise RuntimeError("candidate checkpoint contains a different image set")
    remaining = [row for row in images if int(row["id"]) not in completed]
    if not remaining:
        return completed
    policy = load_policy(policy_path)
    helpers = imports_from_sibling()
    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise RuntimeError("RKNNLite is required on the Orange Pi for candidates") from exc
    _, images_dir, _ = dataset_paths(config)
    capture = helpers["NativeOutputCapture"](runs / "rknn_runtime.log")
    capture.open()
    rknn = capture.call(RKNNLite, verbose=False)
    try:
        if capture.call(rknn.load_rknn, str(model_path)) != 0:
            raise RuntimeError("load_rknn failed")
        if capture.call(rknn.init_runtime, core_mask=helpers["core_mask"](RKNNLite, args.core)) != 0:
            raise RuntimeError("init_runtime failed")
        warmup = np.full((1, int(config["runtime"]["imgsz"]), int(config["runtime"]["imgsz"]), 3), 114, dtype=np.uint8)
        if capture.call(rknn.inference, inputs=[warmup]) is None:
            raise RuntimeError("RKNN warmup failed")
        for offset, item in enumerate(remaining, start=1):
            image_id, image_path = int(item["id"]), images_dir / item["file_name"]
            decode_started = time.perf_counter()
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            if image is None:
                raise ValueError(f"could not decode {image_path}")
            preprocess_started = time.perf_counter()
            padded, ratio, pad = helpers["letterbox_val"](image, int(config["runtime"]["imgsz"]))
            rknn_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)[None, ...])
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
            outputs, inference_ms = capture.timed_call(rknn.inference, inputs=[rknn_input])
            nms_started = time.perf_counter()
            decoded = helpers["decode_rknn_outputs"](outputs, imgsz=int(config["runtime"]["imgsz"]))
            boxes, scores, classes, raw_candidates = helpers["ultralytics_multilabel_nms"](decoded, float(config["runtime"]["conf"]), float(config["runtime"]["iou"]), int(config["runtime"]["max_det"]), int(config["runtime"]["max_nms"]))
            boxes = helpers["scale_boxes"](boxes, image.shape[:2], ratio, pad)
            edge_boxes = [{"xyxy": [float(value) for value in box], "conf": float(score), "cls": int(cls)} for box, score, cls in zip(boxes, scores, classes)]
            nms_ms = (time.perf_counter() - nms_started) * 1000.0
            candidate = candidate_for_image(image_path, image_id, image.shape[:2], edge_boxes, policy, config)
            record = {"image_id": image_id, "file_name": item["file_name"], "shape": list(image.shape[:2]), "edge_boxes": compact_boxes(edge_boxes), "raw_multilabel_candidates": int(raw_candidates), "decode_ms": decode_ms, "edge_preprocess_ms": preprocess_ms, "edge_inference_ms": inference_ms, "edge_nms_ms": nms_ms, "candidate": candidate}
            append_jsonl(checkpoint, record)
            completed[image_id] = record
            if offset == 1 or len(completed) % int(config["runtime"]["progress_every"]) == 0 or offset == len(remaining):
                stage_status(runs, stage="candidates", candidates_done=len(completed), candidates_total=len(images))
                print(f"candidates model={args.model} {len(completed)}/{len(images)} image={image_id}", flush=True)
    finally:
        try:
            capture.call(rknn.release)
        finally:
            capture.close()
    if set(completed) != wanted:
        raise RuntimeError("candidate run is incomplete")
    return completed


def build_plans(runs: Path, records: Mapping[int, Mapping[str, Any]], budgets: Sequence[float]) -> tuple[dict[str, dict[int, str]], dict[str, Any]]:
    candidates = [records[image_id]["candidate"] for image_id in sorted(records)]
    plans, reports = {}, {}
    for budget in map(float, budgets):
        label = str(int(round(budget * 100)))
        result = schedule_fixed_byte(candidates, budget)
        plans[label] = {int(frame): str(action) for frame, action in result["actions"].items()}
        reports[label] = {key: value for key, value in result.items() if key != "actions"}
    atomic_json(runs / "plans.json", {"plans": {label: {str(frame): action for frame, action in plan.items()} for label, plan in plans.items()}, "reports": reports})
    return plans, reports


def load_plans(runs: Path) -> tuple[dict[str, dict[int, str]], dict[str, Any]]:
    payload = json.loads((runs / "plans.json").read_text(encoding="utf-8"))
    return ({str(label): {int(frame): str(action) for frame, action in plan.items()} for label, plan in payload["plans"].items()}, dict(payload["reports"]))


def health_url(cloud_url: str) -> str:
    parsed = urlsplit(cloud_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("cloud URL must be an absolute HTTP(S) URL")
    return urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))


def http_json(url: str, *, body: bytes | None, timeout: float, token: str) -> tuple[dict[str, Any], int, float, int]:
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw, status = response.read(), int(response.status)
    except HTTPError as exc:
        raw, status = exc.read(), int(exc.code)
    except URLError as exc:
        return {"status": "error", "error": f"transport: {exc.reason}"}, 0, (time.perf_counter() - started) * 1000.0, 0
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = {"status": "error", "error": "invalid JSON response"}
    return dict(payload), status, (time.perf_counter() - started) * 1000.0, len(raw)


def validate_cloud(cloud_url: str, config: Mapping[str, Any], timeout: float, token: str) -> dict[str, Any]:
    health, status, _, _ = http_json(health_url(cloud_url), body=None, timeout=timeout, token=token)
    if status != 200 or health.get("status") != "ok":
        raise RuntimeError(f"cloud health failed: {health.get('error', status)}")
    if health.get("model_sha256") != config["models"]["cloud_sha256"]:
        raise RuntimeError("cloud model checksum mismatch")
    runtime = config["runtime"]
    for key in ("imgsz", "conf", "iou", "max_det"):
        if health.get(key) != runtime[key]:
            raise RuntimeError(f"cloud health mismatch for {key}")
    if health.get("action_protocols") != config["cloud_protocols"]:
        raise RuntimeError("cloud action protocol mismatch")
    if health.get("authentication_required") is not True:
        raise RuntimeError("cloud authentication is not enabled")
    return health


def selected_request_keys(plans: Mapping[str, Mapping[int, str]]) -> set[tuple[int, str]]:
    return {(int(frame), str(action)) for plan in plans.values() for frame, action in plan.items() if action != "edge"}


def _valid_cloud_response(
    response: Mapping[str, Any] | None,
    image_id: int,
    action: str,
    expected_cloud_sha256: str,
) -> bool:
    if not isinstance(response, Mapping):
        return False
    try:
        identity_matches = (
            response.get("status") == "ok"
            and int(response.get("frame_id", -1)) == int(image_id)
            and response.get("action_id") == action
            and response.get("model_sha256") == expected_cloud_sha256
            and isinstance(response.get("cloud_boxes"), list)
        )
    except (TypeError, ValueError, OverflowError):
        return False
    if not identity_matches:
        return False
    try:
        boxes = compact_boxes(response["cloud_boxes"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    for box in boxes:
        xyxy = box["xyxy"]
        if (
            len(xyxy) != 4
            or not all(math.isfinite(value) for value in xyxy)
            or xyxy[2] < xyxy[0]
            or xyxy[3] < xyxy[1]
            or not math.isfinite(box["conf"])
            or not 0.0 <= box["conf"] <= 1.0
            or not 0 <= box["cls"] < 80
        ):
            return False
    return True


def assert_required_requests_complete(
    plans: Mapping[str, Mapping[int, str]],
    requests: Mapping[tuple[int, str], Mapping[str, Any]],
    expected_cloud_sha256: str,
) -> int:
    required = sorted(selected_request_keys(plans))
    incomplete = [
        key
        for key in required
        if not _valid_cloud_response(
            requests.get(key), key[0], key[1], expected_cloud_sha256
        )
    ]
    if incomplete:
        first = incomplete[0]
        raise RuntimeError(
            f"cloud requests incomplete: {len(incomplete)}/{len(required)}; "
            f"first image={first[0]} action={first[1]}"
        )
    return len(required)


def build_payload(images_dir: Path, record: Mapping[str, Any], action: str, config: Mapping[str, Any]) -> tuple[bytes, int, float]:
    candidate = record["candidate"]
    path = images_dir / record["file_name"]
    started = time.perf_counter()
    if action == "original":
        jpeg = native_jpeg(path)
    elif action == "q90":
        jpeg = q90_jpeg(path, long_side=int(config["codec"]["long_side"]))
    elif action == "current@640":
        roi = candidate.get("roi_xyxy")
        if roi is None:
            raise ValueError("selected ROI has no geometry")
        jpeg = q90_jpeg(path, crop_xyxy=roi, long_side=int(config["codec"]["long_side"]))
    else:
        raise ValueError(f"not an upload action: {action}")
    if len(jpeg) != int(candidate["payload_bytes"][action]):
        raise RuntimeError(f"payload drift for image {record['image_id']} action={action}")
    metadata = {"task_id": f"rk3588-cavs-{record['image_id']}-{action}", "frame_id": int(record["image_id"]), "action_id": action, "payload_type": "roi" if action == "current@640" else "keyframe", "payload_bytes": len(jpeg), "crop_xyxy": candidate.get("roi_xyxy") if action == "current@640" else None, "orig_shape": list(record["shape"]), "cloud_imgsz": int(config["runtime"]["imgsz"])}
    body = json.dumps({"metadata": metadata, "image_b64": encode_b64(jpeg)}, separators=(",", ":")).encode("utf-8")
    return body, len(jpeg), (time.perf_counter() - started) * 1000.0


def run_uploads(args: argparse.Namespace, config: Mapping[str, Any], runs: Path, records: Mapping[int, Mapping[str, Any]], plans: Mapping[str, Mapping[int, str]]) -> dict[tuple[int, str], dict[str, Any]]:
    if not args.cloud_url:
        raise ValueError("--cloud-url is required for upload")
    token = os.environ.get(args.token_env, "")
    if not token:
        raise RuntimeError(f"runtime bearer token is required in {args.token_env}")
    cloud = validate_cloud(args.cloud_url, config, args.cloud_timeout, token)
    atomic_json(runs / "cloud_health.json", cloud)
    checkpoint = runs / "requests.jsonl"
    latest = latest_jsonl(checkpoint, lambda row: (int(row["image_id"]), str(row["action"])))
    required = sorted(selected_request_keys(plans))
    cloud_sha256 = str(config["models"]["cloud_sha256"])
    pending = [
        key
        for key in required
        if not _valid_cloud_response(latest.get(key), key[0], key[1], cloud_sha256)
    ]
    _, images_dir, _ = dataset_paths(config)
    for offset, (image_id, action) in enumerate(pending, start=1):
        body, payload_bytes, encode_ms = build_payload(images_dir, records[image_id], action, config)
        response: dict[str, Any] = {"status": "error", "error": "not attempted"}
        status, roundtrip_ms, response_bytes = 0, 0.0, 0
        for attempt in range(1, int(config["runtime"]["request_retries"]) + 1):
            response, status, roundtrip_ms, response_bytes = http_json(args.cloud_url, body=body, timeout=args.cloud_timeout, token=token)
            if status == 200 and _valid_cloud_response(
                response, image_id, action, cloud_sha256
            ):
                break
        server = response.get("server_timings_ms", {}) if isinstance(response.get("server_timings_ms"), Mapping) else {}
        response.update({"image_id": image_id, "action": action, "http_status": status, "attempts": attempt, "payload_bytes": payload_bytes, "request_body_bytes": len(body), "response_body_bytes": response_bytes, "upload_encode_ms": encode_ms, "roundtrip_ms": roundtrip_ms, "transport_http_residual_ms": max(0.0, roundtrip_ms - float(server.get("total_before_response", 0.0)))})
        append_jsonl(checkpoint, response)
        latest[(image_id, action)] = response
        successful = sum(
            _valid_cloud_response(latest.get(key), key[0], key[1], cloud_sha256)
            for key in required
        )
        if offset == 1 or offset % 10 == 0 or offset == len(pending):
            stage_status(runs, stage="upload", requests_successful=successful, requests_required=len(required), requests_pending=len(required) - successful)
            print(f"upload model={args.model} {successful}/{len(required)} image={image_id} action={action} status={response.get('status')}", flush=True)
    assert_required_requests_complete(plans, latest, cloud_sha256)
    return latest


def values_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()) if array.size else 0.0, "p50": float(np.percentile(array, 50)) if array.size else 0.0, "p95": float(np.percentile(array, 95)) if array.size else 0.0, "p99": float(np.percentile(array, 99)) if array.size else 0.0}


def evaluate_coco(annotation_path: Path, image_ids: Sequence[int], predictions: Sequence[Mapping[str, Any]], log_path: Path) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    prediction_path = log_path.with_suffix(".predictions.json")
    prediction_path.write_text(json.dumps(list(predictions), separators=(",", ":")), encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as stream:
        previous_out, previous_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = stream
        try:
            truth = COCO(str(annotation_path))
            detected = truth.loadRes(str(prediction_path))
            evaluator = COCOeval(truth, detected, "bbox")
            evaluator.params.imgIds = list(image_ids)
            evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        finally:
            sys.stdout, sys.stderr = previous_out, previous_err
    stats = evaluator.stats
    prediction_path.unlink(missing_ok=True)
    return {"AP50_95": float(stats[0]), "AP50": float(stats[1]), "AP75": float(stats[2]), "AP_small": float(stats[3]), "AP_medium": float(stats[4]), "AP_large": float(stats[5])}


def resolved_boxes(action: str, request: Mapping[str, Any] | None, record: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str, float]:
    edge = compact_boxes(record["edge_boxes"])
    if action == "edge":
        return edge, "not_required", 0.0
    if request is None or request.get("status") != "ok":
        return edge, "fallback", 0.0
    cloud = compact_boxes(request.get("cloud_boxes", []))
    if action != "current@640":
        return cloud, "ok", 0.0
    roi = record["candidate"].get("roi_xyxy")
    if roi is None:
        raise RuntimeError("ROI result lacks geometry")
    started = time.perf_counter()
    maur = policy["maur"]
    return maur_fuse_boxes(edge, cloud, roi, duplicate_iou=float(maur["duplicate_iou"]), nms_iou=float(maur["nms_iou"])), "ok", (time.perf_counter() - started) * 1000.0


def evaluate(args: argparse.Namespace, config: Mapping[str, Any], runs: Path, records: Mapping[int, Mapping[str, Any]], plans: Mapping[str, Mapping[int, str]], reports: Mapping[str, Any], policy_path: Path, annotation_path: Path) -> dict[str, Any]:
    policy = load_policy(policy_path)
    requests = latest_jsonl(runs / "requests.jsonl", lambda row: (int(row["image_id"]), str(row["action"])))
    successful_requests = assert_required_requests_complete(
        plans, requests, str(config["models"]["cloud_sha256"])
    )
    image_ids = sorted(records)
    methods: dict[str, Any] = {}
    all_plans = {"edge": {image_id: "edge" for image_id in image_ids}, **{f"cavs_oqr_{label}": plan for label, plan in plans.items()}}
    for method, plan in all_plans.items():
        predictions: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for image_id in image_ids:
            record, action = records[image_id], plan[image_id]
            request = requests.get((image_id, action)) if action != "edge" else None
            boxes, status, fusion_ms = resolved_boxes(action, request, record, policy)
            predictions.extend(coco_predictions(image_id, boxes))
            candidate_ms = float(record["candidate"]["candidate_ms"]) if method != "edge" else 0.0
            server = request.get("server_timings_ms", {}) if request else {}
            rows.append({"action": action, "status": status, "decode_ms": float(record["decode_ms"]), "edge_preprocess_ms": float(record["edge_preprocess_ms"]), "edge_inference_ms": float(record["edge_inference_ms"]), "edge_nms_ms": float(record["edge_nms_ms"]), "candidate_ms": candidate_ms, "upload_encode_ms": float(request.get("upload_encode_ms", 0.0)) if request else 0.0, "roundtrip_ms": float(request.get("roundtrip_ms", 0.0)) if request else 0.0, "fusion_ms": fusion_ms, "server_cloud_infer_ms": float(server.get("cloud_infer", 0.0)), "payload_bytes": int(request.get("payload_bytes", 0)) if request else 0, "request_body_bytes": int(request.get("request_body_bytes", 0)) if request else 0, "response_body_bytes": int(request.get("response_body_bytes", 0)) if request else 0})
        for row in rows:
            row["e2e_ms"] = sum(float(row[key]) for key in ("decode_ms", "edge_preprocess_ms", "edge_inference_ms", "edge_nms_ms", "candidate_ms", "upload_encode_ms", "roundtrip_ms", "fusion_ms"))
        metrics = evaluate_coco(annotation_path, image_ids, predictions, runs / f"coco_eval_{method}.log")
        with gzip.open(runs / f"predictions_{method}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(predictions, handle, ensure_ascii=False, separators=(",", ":"))
        methods[method] = {"metrics": metrics, "action_counts": dict(Counter(row["action"] for row in rows)), "uploads": sum(row["action"] != "edge" for row in rows), "failures": sum(row["status"] == "fallback" for row in rows), "payload_bytes_total": sum(row["payload_bytes"] for row in rows), "request_body_bytes_total": sum(row["request_body_bytes"] for row in rows), "response_body_bytes_total": sum(row["response_body_bytes"] for row in rows), "timing_ms": {key: values_summary([row[key] for row in rows]) for key in ("decode_ms", "edge_preprocess_ms", "edge_inference_ms", "edge_nms_ms", "candidate_ms", "upload_encode_ms", "roundtrip_ms", "fusion_ms", "server_cloud_infer_ms", "e2e_ms")}}
        print(f"evaluate model={args.model} method={method} AP50-95={metrics['AP50_95']:.6f}", flush=True)
    summary = {"complete": True, "protocol_id": config["protocol_id"], "board_model": args.model, "board_model_sha256": config["models"][args.model]["sha256"], "policy_sha256": config["models"]["policy_sha256"], "images": len(image_ids), "two_pass_global_scheduler": True, "paper_comparable_fixed_byte": not bool(args.limit), "plans": reports, "required_unique_requests": successful_requests, "successful_unique_requests": successful_requests, "methods": methods}
    atomic_json(runs / "summary.json", summary)
    stage_status(runs, stage="complete", complete=True, images=len(image_ids))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fp16",), required=True)
    parser.add_argument("--stage", choices=("candidates", "plan", "upload", "evaluate", "all"), default="all")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--core", choices=("auto", "0", "1", "2", "all"), default="0")
    parser.add_argument("--cloud-url", default="")
    parser.add_argument("--cloud-timeout", type=float, default=60.0)
    parser.add_argument("--token-env", default="CAVS_TOKEN")
    return parser.parse_args()


def main() -> None:
    args, config = parse_args(), load_config()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    _, images_dir, _ = dataset_paths(config)
    images = read_images(resolved(config["dataset"]["root"]) / config["dataset"]["annotations"], images_dir, args.limit)
    model_path, policy_path, annotation_path = validate_contract(args, config, images)
    runs = run_dir(args); runs.mkdir(parents=True, exist_ok=True)
    ensure_run_contract(runs, {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "protocol_config_sha256": sha256(ROOT / "configs" / "protocol.json"),
        "model": args.model,
        "model_sha256": config["models"][args.model]["sha256"],
        "policy_sha256": config["models"]["policy_sha256"],
        "cloud_model_sha256": config["models"]["cloud_sha256"],
        "limit": args.limit,
        "image_count": len(images),
        "budgets": list(config["budgets"]),
        "codec": dict(config["codec"]),
        "cloud_protocols": dict(config["cloud_protocols"]),
        "two_pass_global_scheduler": True,
    })
    records: dict[int, dict[str, Any]] | None = None
    plans: dict[str, dict[int, str]] | None = None
    reports: dict[str, Any] | None = None
    if args.stage in {"candidates", "all"}:
        records = run_candidates(args, config, runs, images, model_path, policy_path)
    if args.stage in {"plan", "upload", "evaluate", "all"}:
        records = records or latest_jsonl(runs / "candidates.jsonl", lambda row: int(row["image_id"]))
        if set(records) != {int(row["id"]) for row in images}:
            raise RuntimeError("plan/upload/evaluate requires complete candidate records")
    if args.stage in {"plan", "all"}:
        plans, reports = build_plans(runs, records, config["budgets"])
        stage_status(runs, stage="plan", plans_complete=True)
    if args.stage in {"upload", "evaluate", "all"}:
        if plans is None or reports is None:
            plans, reports = load_plans(runs)
    if args.stage in {"upload", "all"}:
        run_uploads(args, config, runs, records, plans)
    if args.stage in {"evaluate", "all"}:
        evaluate(args, config, runs, records, plans, reports, policy_path, annotation_path)


if __name__ == "__main__":
    main()
