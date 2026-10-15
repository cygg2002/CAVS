# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Local HTTP cloud verifier for the YOLO26 edge-cloud detection demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from protocol import b64_to_bytes, boxes_from_result, decode_jpeg, shift_boxes

from ultralytics import YOLO


class CloudState:
    """Shared server state."""

    model: YOLO | None = None
    args: argparse.Namespace | None = None


def build_health_payload(args: argparse.Namespace | None) -> dict[str, Any]:
    """Expose the detector settings needed to verify the Current+MAUR runtime."""
    return {
        "status": "ok",
        "model": Path(args.model).name if args else None,
        "model_sha256": getattr(args, "model_sha256", None) if args else None,
        "device": (args.device or "auto") if args else None,
        "imgsz": args.imgsz if args else None,
        "conf": args.conf if args else None,
    }


def validate_request_length(value: str | None, max_bytes: int) -> int:
    """Validate Content-Length before allocating a request body."""
    try:
        length = int(value or "0")
    except ValueError as exc:
        raise ValueError("invalid Content-Length") from exc
    if length <= 0:
        raise ValueError("Content-Length must be positive")
    if length > int(max_bytes):
        raise ValueError(f"request too large; maximum is {int(max_bytes)} bytes")
    return length


def resolve_cloud_imgsz(requested: Any, default: int, minimum: int, maximum: int) -> int:
    """Validate per-request cloud resolution against configured compute bounds."""
    value = int(default) if requested in {None, ""} else int(requested)
    if not int(minimum) <= value <= int(maximum):
        raise ValueError(f"cloud_imgsz must be between {int(minimum)} and {int(maximum)}")
    return value


class CloudHandler(BaseHTTPRequestHandler):
    """Handle health checks and inference requests."""

    server_version = "YOLO26EdgeCloud/0.1"

    def log_message(self, fmt: str, *args) -> None:
        if CloudState.args and CloudState.args.verbose:
            super().log_message(fmt, *args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send_json(200, build_health_payload(CloudState.args))
        else:
            self._send_json(404, {"status": "not_found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/infer":
            self._send_json(404, {"status": "not_found"})
            return

        started = time.perf_counter()
        try:
            if "application/json" not in self.headers.get("Content-Type", ""):
                self._send_json(415, {"status": "error", "error": "Content-Type must be application/json"})
                return
            length = validate_request_length(
                self.headers.get("Content-Length"),
                int(CloudState.args.max_request_mb * 1024 * 1024),
            )
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            metadata = request.get("metadata", {})
            if not isinstance(metadata, dict):
                raise TypeError("metadata must be an object")
            image = decode_jpeg(b64_to_bytes(request["image_b64"]))
            cloud_imgsz = resolve_cloud_imgsz(
                metadata.get("cloud_imgsz"),
                CloudState.args.imgsz,
                CloudState.args.min_imgsz,
                CloudState.args.max_imgsz,
            )

            predict_kwargs = {
                "source": image,
                "imgsz": cloud_imgsz,
                "conf": CloudState.args.conf,
                "verbose": False,
            }
            if CloudState.args.device:
                predict_kwargs["device"] = CloudState.args.device
            class_ids = metadata.get("classes")
            if class_ids is not None:
                predict_kwargs["classes"] = [int(cls_id) for cls_id in class_ids]
            result = CloudState.model.predict(**predict_kwargs)[0]
            boxes = boxes_from_result(result)

            crop_xyxy = metadata.get("crop_xyxy")
            if metadata.get("payload_type") in {"roi", "proposal", "tile"} and crop_xyxy:
                boxes = shift_boxes(boxes, (int(crop_xyxy[0]), int(crop_xyxy[1])))

            self._send_json(
                200,
                {
                    "status": "ok",
                    "task_id": metadata.get("task_id"),
                    "frame_id": metadata.get("frame_id"),
                    "payload_type": metadata.get("payload_type"),
                    "cloud_imgsz": cloud_imgsz,
                    "cloud_boxes": boxes,
                    "cloud_latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "model": Path(CloudState.args.model).name,
                },
            )
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(
                400,
                {
                    "status": "error",
                    "error": str(exc)[:160],
                    "cloud_latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                },
            )
        except Exception as exc:  # noqa: BLE001 - keep the demo service alive after inference errors
            if CloudState.args.verbose:
                print(f"Cloud inference error: {exc}")
            self._send_json(
                500,
                {
                    "status": "error",
                    "error": "cloud inference failed",
                    "cloud_latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                },
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26x.pt", help="Cloud verifier model.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP bind port.")
    parser.add_argument("--imgsz", type=int, default=640, help="Cloud inference image size.")
    parser.add_argument("--conf", type=float, default=0.001, help="Cloud confidence threshold.")
    parser.add_argument("--device", default="", help="Ultralytics device string, e.g. 0 or cpu.")
    parser.add_argument("--min-imgsz", type=int, default=320, help="Minimum per-request inference size.")
    parser.add_argument("--max-imgsz", type=int, default=960, help="Maximum per-request inference size.")
    parser.add_argument("--max-request-mb", type=float, default=20.0, help="Maximum JSON request body size.")
    parser.add_argument("--verbose", action="store_true", help="Print HTTP access logs.")
    return parser.parse_args()


def main() -> None:
    """Start the cloud verifier service."""
    args = parse_args()
    if not 0 < args.min_imgsz <= args.imgsz <= args.max_imgsz:
        raise ValueError("Require 0 < min-imgsz <= imgsz <= max-imgsz")
    if args.max_request_mb <= 0:
        raise ValueError("--max-request-mb must be positive")
    model_path = Path(args.model).expanduser()
    args.model_sha256 = None
    if model_path.is_file():
        digest = hashlib.sha256()
        with model_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        args.model_sha256 = digest.hexdigest()
    CloudState.args = args
    print(f"Loading cloud model: {args.model}")
    CloudState.model = YOLO(args.model)
    server = HTTPServer((args.host, args.port), CloudHandler)
    print(f"Cloud verifier listening on http://{args.host}:{args.port}/infer")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping cloud verifier")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
