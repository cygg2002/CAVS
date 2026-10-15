#!/usr/bin/env python3
"""Isolated cloud server with component timing for the system benchmark."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
DEMO = REPO / "examples/YOLO26-Edge-Cloud"
sys.path.insert(0, str(DEMO))

from protocol import boxes_from_result, decode_jpeg, shift_boxes  # noqa: E402
from ultralytics import YOLO  # noqa: E402


class State:
    model: YOLO | None = None
    args: argparse.Namespace | None = None
    lock = threading.Lock()


def map_boxes_to_frame(boxes: list[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Map ROI detections back to full-image coordinates."""
    crop = metadata.get("crop_xyxy")
    if metadata.get("payload_type") == "roi" and crop:
        return shift_boxes(boxes, (int(crop[0]), int(crop[1])))
    return boxes


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_snapshot() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        ).strip().split(", ")
        return {
            "gpu_name": output[0],
            "gpu_memory_used_mb": int(output[1]),
            "gpu_memory_total_mb": int(output[2]),
            "gpu_utilization_percent": int(output[3]),
        }
    except Exception:
        return {}


class Handler(BaseHTTPRequestHandler):
    server_version = "CAVSBenchmarkCloud/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        if State.args.verbose:
            super().log_message(fmt, *args)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/health":
            self.send_json(404, {"status": "not_found"})
            return
        self.send_json(
            200,
            {
                "status": "ok",
                "pid": os.getpid(),
                "model": Path(State.args.model).name,
                "model_sha256": State.args.model_sha256,
                "device": State.args.device,
                "imgsz": State.args.imgsz,
                "conf": State.args.conf,
                **gpu_snapshot(),
            },
        )

    def do_POST(self) -> None:
        total_started = time.perf_counter()
        if self.path.rstrip("/") != "/infer":
            self.send_json(404, {"status": "not_found"})
            return
        timings: dict[str, float] = {}
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= int(State.args.max_request_mb * 1024 * 1024):
                raise ValueError("invalid request length")
            started = time.perf_counter()
            body = self.rfile.read(length)
            timings["request_read"] = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            request = json.loads(body.decode("utf-8"))
            timings["json_parse"] = (time.perf_counter() - started) * 1000.0
            metadata = request["metadata"]

            started = time.perf_counter()
            jpeg = base64.b64decode(request["image_b64"].encode("ascii"), validate=True)
            timings["base64_decode"] = (time.perf_counter() - started) * 1000.0
            if len(jpeg) != int(metadata["payload_bytes"]):
                raise ValueError("payload byte count mismatch")

            started = time.perf_counter()
            image = decode_jpeg(jpeg)
            timings["jpeg_decode"] = (time.perf_counter() - started) * 1000.0

            predict_kwargs = {
                "source": image,
                "imgsz": int(metadata.get("cloud_imgsz", State.args.imgsz)),
                "conf": float(State.args.conf),
                "verbose": False,
                "device": State.args.device,
            }
            started = time.perf_counter()
            with State.lock:
                result = State.model.predict(**predict_kwargs)[0]
            timings["cloud_infer"] = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            boxes = boxes_from_result(result)
            boxes = map_boxes_to_frame(boxes, metadata)
            timings["cloud_postprocess"] = (time.perf_counter() - started) * 1000.0
            timings["pre_send_total"] = (time.perf_counter() - total_started) * 1000.0
            payload = {
                "status": "ok",
                "frame_id": metadata.get("frame_id"),
                "payload_type": metadata.get("payload_type"),
                "cloud_boxes": boxes,
                "cloud_latency_ms": timings["pre_send_total"],
                "server_timings_ms": timings,
                "request_body_bytes": length,
                "model": Path(State.args.model).name,
            }
            serialize_started = time.perf_counter()
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            timings["response_serialize"] = (time.perf_counter() - serialize_started) * 1000.0
            self.send_json(200, payload)
        except Exception as exc:
            timings["pre_send_total"] = (time.perf_counter() - total_started) * 1000.0
            self.send_json(
                400,
                {
                    "status": "error",
                    "error": str(exc)[:200],
                    "cloud_latency_ms": timings["pre_send_total"],
                    "server_timings_ms": timings,
                },
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-request-mb", type=float, default=20.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = Path(args.model).resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    args.model_sha256 = file_sha256(model)
    State.args = args
    State.model = YOLO(model)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"status": "ready", "port": args.port, "model_sha256": args.model_sha256}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
