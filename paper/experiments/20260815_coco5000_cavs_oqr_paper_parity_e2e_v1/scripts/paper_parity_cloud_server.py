#!/usr/bin/env python3
"""Action-aware HTTP Cloud reproducing the frozen CAVS-OQR inference protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
DEMO = REPO / "examples/YOLO26-Edge-Cloud"
sys.path.insert(0, str(DEMO))

from protocol import b64_to_bytes, boxes_from_result, decode_jpeg, shift_boxes  # noqa: E402
from inference_protocol import ScaleDownDetectionPredictor  # noqa: E402
from ultralytics import YOLO  # noqa: E402


ACTION_PROTOCOLS = {
    "original": {"end2end": False, "rect": True, "scaleup": False, "half": False, "multi_label": True},
    "q90": {"end2end": False, "rect": True, "scaleup": False, "half": False, "multi_label": True},
    "current@640": {"end2end": False, "rect": True, "scaleup": True, "half": True, "multi_label": False},
}


class State:
    args: argparse.Namespace | None = None
    full_model: YOLO | None = None
    roi_model: YOLO | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def roi_model() -> YOLO:
    if State.roi_model is None:
        State.roi_model = YOLO(State.args.model)
    return State.roi_model


def health_payload() -> dict[str, Any]:
    args = State.args
    return {
        "status": "ok",
        "model": Path(args.model).name,
        "model_sha256": args.model_sha256,
        "device": str(args.device),
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "max_det": int(args.max_det),
        "action_protocols": ACTION_PROTOCOLS,
        "roi_model_loaded": State.roi_model is not None,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "CAVSPaperParity/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        if State.args.verbose:
            super().log_message(fmt, *args)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self.send_json(200, health_payload()) if self.path.rstrip("/") == "/health" else self.send_json(404, {"status": "not_found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/infer":
            self.send_json(404, {"status": "not_found"})
            return
        request_started = time.perf_counter()
        timings: dict[str, float] = {}
        try:
            length = int(self.headers.get("Content-Length", "0"))
            maximum = int(State.args.max_request_mb * 1024 * 1024)
            if length <= 0 or length > maximum:
                raise ValueError(f"invalid request length: {length}")
            read_started = time.perf_counter()
            raw = self.rfile.read(length)
            timings["request_read"] = (time.perf_counter() - read_started) * 1000.0
            parse_started = time.perf_counter()
            request = json.loads(raw.decode("utf-8"))
            metadata = request["metadata"]
            action = str(metadata.get("action_id"))
            if action not in ACTION_PROTOCOLS:
                raise ValueError(f"unsupported action: {action}")
            timings["json_parse"] = (time.perf_counter() - parse_started) * 1000.0
            decode_started = time.perf_counter()
            image = decode_jpeg(b64_to_bytes(request["image_b64"]))
            timings["jpeg_decode"] = (time.perf_counter() - decode_started) * 1000.0

            infer_started = time.perf_counter()
            kwargs = {
                "source": image,
                "imgsz": int(State.args.imgsz),
                "conf": float(State.args.conf),
                "iou": float(State.args.iou),
                "max_det": int(State.args.max_det),
                "device": str(State.args.device),
                "rect": True,
                "end2end": False,
                "verbose": False,
            }
            if action == "current@640":
                kwargs["half"] = True
                result = roi_model().predict(**kwargs)[0]
            else:
                kwargs["half"] = False
                result = State.full_model.predict(predictor=ScaleDownDetectionPredictor, **kwargs)[0]
            timings["cloud_infer"] = (time.perf_counter() - infer_started) * 1000.0
            post_started = time.perf_counter()
            boxes = boxes_from_result(result)
            crop = metadata.get("crop_xyxy")
            if action == "current@640" and crop:
                boxes = shift_boxes(boxes, (int(crop[0]), int(crop[1])))
            timings["cloud_postprocess"] = (time.perf_counter() - post_started) * 1000.0
            timings["pre_send_total"] = (time.perf_counter() - request_started) * 1000.0
            self.send_json(
                200,
                {
                    "status": "ok",
                    "task_id": metadata.get("task_id"),
                    "frame_id": metadata.get("frame_id"),
                    "action_id": action,
                    "cloud_boxes": boxes,
                    "server_timings_ms": timings,
                    "model": Path(State.args.model).name,
                },
            )
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.send_json(400, {"status": "error", "error": str(exc)[:200]})
        except Exception as exc:  # keep the isolated service alive for resumable requests
            if State.args.verbose:
                print(f"inference error: {type(exc).__name__}: {exc}", flush=True)
            self.send_json(500, {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:160]}"})


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
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    args.model = str(model_path)
    args.model_sha256 = sha256(model_path)
    State.args = args
    State.full_model = YOLO(args.model)
    server = HTTPServer((args.host, args.port), Handler)
    print(json.dumps(health_payload(), ensure_ascii=False), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
