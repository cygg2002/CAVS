#!/usr/bin/env python3
"""Bounded HTTP cloud inference service for the RK3588 CAVS-OQR experiment."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import time
from typing import Any, Mapping

from protocol import compact_boxes, decode_b64, decode_jpeg, shift_boxes


ACTION_PROTOCOLS = {
    "original": {"end2end": False, "rect": True, "scaleup": False, "half": False, "multi_label": True},
    "q90": {"end2end": False, "rect": True, "scaleup": False, "half": False, "multi_label": True},
    "current@640": {"end2end": False, "rect": True, "scaleup": True, "half": True, "multi_label": False},
}


class State:
    args: argparse.Namespace | None = None
    model: Any = None
    roi_model: Any = None


def require_runtime_token(value: str | None) -> str:
    token = str(value or "")
    if not token:
        raise RuntimeError("runtime bearer token is required")
    return token


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_inference_request(body: bytes, *, maximum_bytes: int) -> tuple[dict[str, Any], bytes]:
    """Validate JSON and base64 at the trust boundary before decoding JPEG pixels."""
    if not 0 < len(body) <= int(maximum_bytes):
        raise ValueError("invalid request length")
    request = json.loads(body.decode("utf-8"))
    if not isinstance(request, Mapping) or not isinstance(request.get("metadata"), Mapping):
        raise ValueError("request metadata is required")
    metadata = dict(request["metadata"])
    action = str(metadata.get("action_id", ""))
    if action not in ACTION_PROTOCOLS:
        raise ValueError("unsupported action")
    frame_id = int(metadata.get("frame_id", -1))
    if frame_id < 0:
        raise ValueError("invalid frame ID")
    payload = decode_b64(str(request.get("image_b64", "")))
    if len(payload) != int(metadata.get("payload_bytes", -1)):
        raise ValueError("payload byte count mismatch")
    crop = metadata.get("crop_xyxy")
    if action == "current@640":
        if not isinstance(crop, list) or len(crop) != 4 or any(int(value) < 0 for value in crop):
            raise ValueError("ROI requires a valid crop")
        if int(crop[2]) <= int(crop[0]) or int(crop[3]) <= int(crop[1]):
            raise ValueError("ROI crop has non-positive area")
    elif crop is not None:
        raise ValueError("full-image action must not carry a crop")
    return metadata, payload


def _predictor_type():
    """Build the validation-style predictor only on the cloud host."""
    from ultralytics.data.augment import LetterBox
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import nms, ops

    class ScaleDownDetectionPredictor(DetectionPredictor):
        def pre_transform(self, images):
            same_shapes = len({image.shape for image in images}) == 1
            letterbox = LetterBox(
                self.imgsz,
                auto=same_shapes and self.args.rect and (self.model.format == "pt" or (getattr(self.model, "dynamic", False) and self.model.format != "imx")),
                stride=self.model.stride,
                scaleup=False,
            )
            return [letterbox(image=image) for image in images]

        def postprocess(self, preds, image, original_images, **kwargs):
            outputs = nms.non_max_suppression(
                preds, self.args.conf, self.args.iou, self.args.classes, self.args.agnostic_nms,
                multi_label=True, max_det=self.args.max_det,
                nc=0 if self.args.task == "detect" else len(self.model.names),
                end2end=getattr(self.model, "end2end", False), rotated=self.args.task == "obb",
            )
            if not isinstance(original_images, list):
                original_images = ops.convert_torch2numpy_batch(original_images)[..., ::-1]
            return self.construct_results(outputs, image, original_images)

    return ScaleDownDetectionPredictor


def _result_boxes(result: Any) -> list[dict[str, Any]]:
    boxes = result.boxes
    return [
        {"xyxy": [float(value) for value in xyxy], "conf": float(conf), "cls": int(cls)}
        for xyxy, conf, cls in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist())
    ]


def shift_roi_boxes(
    boxes: list[dict[str, Any]], crop_xyxy: list[int]
) -> list[dict[str, Any]]:
    return shift_boxes(boxes, (int(crop_xyxy[0]), int(crop_xyxy[1])))


def health_payload() -> dict[str, Any]:
    args = State.args
    return {
        "status": "ok", "model": Path(args.model).name, "model_sha256": args.model_sha256,
        "device": str(args.device), "imgsz": int(args.imgsz), "conf": float(args.conf),
        "iou": float(args.iou), "max_det": int(args.max_det), "action_protocols": ACTION_PROTOCOLS,
        "authentication_required": bool(os.environ.get(args.token_env, "")),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "CAVSOQR/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        if State.args.verbose:
            super().log_message(fmt, *args)

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = os.environ.get(State.args.token_env, "")
        return bool(expected) and hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {expected}")

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send_json(200, health_payload())
        else:
            self._send_json(404, {"status": "not_found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/infer":
            self._send_json(404, {"status": "not_found"})
            return
        if not self._authorized():
            self._send_json(401, {"status": "error", "error": "unauthorized"})
            return
        started, timings = time.perf_counter(), {}
        try:
            length = int(self.headers.get("Content-Length", "0"))
            maximum = int(State.args.max_request_mb * 1024 * 1024)
            if not 0 < length <= maximum:
                raise ValueError("invalid request length")
            read_started = time.perf_counter()
            metadata, payload = parse_inference_request(self.rfile.read(length), maximum_bytes=maximum)
            timings["request_read_and_validate"] = (time.perf_counter() - read_started) * 1000.0
            decode_started = time.perf_counter()
            image = decode_jpeg(payload, maximum_pixels=int(State.args.max_image_pixels))
            timings["jpeg_decode"] = (time.perf_counter() - decode_started) * 1000.0
            infer_started = time.perf_counter()
            action = str(metadata["action_id"])
            kwargs = {"source": image, "imgsz": int(State.args.imgsz), "conf": float(State.args.conf), "iou": float(State.args.iou), "max_det": int(State.args.max_det), "device": str(State.args.device), "rect": True, "end2end": False, "verbose": False}
            if action == "current@640":
                if State.roi_model is None:
                    from ultralytics import YOLO
                    State.roi_model = YOLO(State.args.model)
                kwargs["half"] = True
                result = State.roi_model.predict(**kwargs)[0]
            else:
                kwargs["half"] = False
                result = State.model.predict(predictor=_predictor_type(), **kwargs)[0]
            timings["cloud_infer"] = (time.perf_counter() - infer_started) * 1000.0
            boxes = _result_boxes(result)
            if action == "current@640":
                boxes = shift_roi_boxes(boxes, metadata["crop_xyxy"])
            timings["total_before_response"] = (time.perf_counter() - started) * 1000.0
            self._send_json(200, {"status": "ok", "frame_id": metadata["frame_id"], "action_id": action, "cloud_boxes": compact_boxes(boxes), "server_timings_ms": timings, "model_sha256": State.args.model_sha256})
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"status": "error", "error": str(exc)[:160]})
        except Exception as exc:
            if State.args.verbose:
                print(f"inference error: {type(exc).__name__}: {exc}", flush=True)
            self._send_json(500, {"status": "error", "error": "inference_failed"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-request-mb", type=float, default=20.0)
    parser.add_argument("--max-image-pixels", type=int, default=100_000_000)
    parser.add_argument("--token-env", default="CAVS_TOKEN")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_runtime_token(os.environ.get(args.token_env))
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0 or args.max_det <= 0:
        raise ValueError("invalid inference thresholds")
    args.model, args.model_sha256 = str(model), sha256(model)
    from ultralytics import YOLO
    State.args, State.model = args, YOLO(args.model)
    print(json.dumps(health_payload(), ensure_ascii=False), flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
