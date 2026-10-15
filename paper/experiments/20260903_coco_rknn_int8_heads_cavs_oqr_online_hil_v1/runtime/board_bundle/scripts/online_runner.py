#!/usr/bin/env python3
"""Strict one-frame-at-a-time RK3588 edge-cloud online runner."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
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
from urllib.request import HTTPRedirectHandler, Request, build_opener

import cv2
import numpy as np

from cavs_numpy import current_roi_xyxy
from fusion import maur_fuse_boxes
from online_control import CausalByteController, PrimalDualByteController
from online_methods import MethodRuntime, build_method_runtime, score_current_frame
from online_stream import execute_stream
from protocol import (
    WIRE_MEDIA_TYPE,
    compact_boxes,
    encode_inference_request,
    native_jpeg,
    q90_jpeg,
    sanitize_boxes,
    wire_contract,
)
from run_contract import (
    PROTOCOL_ID,
    append_committed_frame,
    build_run_contract,
    ensure_run_contract,
    exclusive_run_lock,
    file_sha256,
    restore_committed_prefix,
    verify_rknn_helpers_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_SCHEMA = "coco_images_only_sequence_v1"
CONTROL_SCHEMA = "cavs_online_control_v1"
PRIMAL_DUAL_CONTROL_SCHEMA = "cavs_online_primal_dual_control_v1"
RECORD_FIELDS = frozenset(("image_id", "file_name", "width", "height"))
FROZEN_CLOUD_PROTOCOLS = {
    "original": {
        "end2end": False, "rect": True, "scaleup": False,
        "half": False, "multi_label": True,
    },
    "q90": {
        "end2end": False, "rect": True, "scaleup": False,
        "half": False, "multi_label": True,
    },
    "current@640": {
        "end2end": False, "rect": True, "scaleup": True,
        "half": True, "multi_label": False,
    },
}


def _canonical_records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(records), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_images_only_sequence(path: str | Path) -> list[dict[str, Any]]:
    """Load only image identity and geometry; annotations are never accepted."""
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read images-only sequence manifest") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("images-only sequence manifest must be an object")
    if manifest.get("schema") != SEQUENCE_SCHEMA or not isinstance(manifest.get("records"), list):
        raise ValueError("unsupported images-only sequence manifest")
    if not isinstance(manifest.get("dataset"), str) or not manifest.get("dataset"):
        raise ValueError("images-only sequence dataset is required")
    if not isinstance(manifest.get("split"), str) or not manifest.get("split"):
        raise ValueError("images-only sequence split is required")
    records: list[dict[str, Any]] = []
    image_ids: set[int] = set()
    file_names: set[str] = set()
    for position, raw in enumerate(manifest["records"]):
        if not isinstance(raw, Mapping) or set(raw) != RECORD_FIELDS:
            raise ValueError(
                f"images-only record {position} must contain exactly {sorted(RECORD_FIELDS)}"
            )
        image_id, width, height = raw["image_id"], raw["width"], raw["height"]
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (image_id, width, height)):
            raise ValueError(f"invalid images-only geometry at position {position}")
        file_name = raw["file_name"]
        relative = Path(file_name) if isinstance(file_name, str) else Path("..")
        if (
            not isinstance(file_name, str)
            or not file_name
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in file_name
        ):
            raise ValueError(f"invalid images-only file name at position {position}")
        if image_id in image_ids or file_name in file_names:
            raise ValueError("images-only sequence identities must be unique")
        image_ids.add(image_id)
        file_names.add(file_name)
        records.append({
            "image_id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })
    if not records:
        raise ValueError("images-only sequence must not be empty")
    declared_count = manifest.get("images", manifest.get("record_count", len(records)))
    if declared_count != len(records):
        raise ValueError("images-only sequence count mismatch")
    source_hash = manifest.get("source_annotations_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(
        character not in "0123456789abcdef" for character in source_hash
    ):
        raise ValueError("images-only source binding is invalid")
    declared_hash = manifest.get("sequence_sha256")
    if declared_hash != _canonical_records_sha256(records):
        raise ValueError("images-only sequence hash mismatch")
    return records


def _valid_box(box: Mapping[str, Any]) -> bool:
    try:
        xyxy = box["xyxy"]
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
            return False
        coordinates = [float(value) for value in xyxy]
        confidence = float(box.get("conf", box.get("score")))
        category = box["cls"]
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (
        all(math.isfinite(value) for value in coordinates)
        and coordinates[2] > coordinates[0]
        and coordinates[3] > coordinates[1]
        and math.isfinite(confidence)
        and 0.0 <= confidence <= 1.0
        and isinstance(category, int)
        and not isinstance(category, bool)
        and 0 <= category < 80
    )


class _NoRedirectHandler(HTTPRedirectHandler):
    """Reject redirects so the bearer token never crosses an origin boundary."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "cloud redirect rejected", headers, fp)


_NO_REDIRECT_OPENER = build_opener(_NoRedirectHandler())


def validate_cloud_response(
    response: Mapping[str, Any],
    *,
    image_id: int,
    action: str,
    cloud_sha256: str,
    shape: Sequence[int],
) -> list[dict[str, Any]]:
    """Validate the complete remote identity and detection boundary."""
    try:
        identity_matches = (
            isinstance(response, Mapping)
            and response.get("status") == "ok"
            and int(response.get("frame_id", -1)) == int(image_id)
            and response.get("action_id") == action
            and response.get("model_sha256") == cloud_sha256
            and isinstance(response.get("cloud_boxes"), list)
        )
    except (TypeError, ValueError, OverflowError):
        identity_matches = False
    try:
        height, width = (int(shape[0]), int(shape[1]))
    except (IndexError, TypeError, ValueError):
        raise RuntimeError("invalid cloud response shape") from None
    if height <= 0 or width <= 0:
        raise RuntimeError("invalid cloud response shape")
    def valid_in_shape(box: Mapping[str, Any]) -> bool:
        if not _valid_box(box):
            return False
        x1, y1, x2, y2 = map(float, box["xyxy"])
        return x1 >= 0.0 and y1 >= 0.0 and x2 <= width and y2 <= height

    if not identity_matches or any(not isinstance(box, Mapping) or not valid_in_shape(box) for box in response.get("cloud_boxes", ())):
        raise RuntimeError("invalid cloud response")
    timings = response.get("server_timings_ms", {})
    if not isinstance(timings, Mapping):
        raise RuntimeError("invalid cloud response timings")
    try:
        valid_timings = all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in timings.values()
        )
    except (TypeError, ValueError, OverflowError):
        valid_timings = False
    if not valid_timings:
        raise RuntimeError("invalid cloud response timings")
    return compact_boxes(response["cloud_boxes"])


class CloudClient:
    """Authenticated bounded binary client; bearer material never enters records."""

    def __init__(
        self,
        url: str,
        *,
        token_env: str,
        cloud_sha256: str,
        timeout: float = 60.0,
        retries: int = 3,
        maximum_response_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        parsed = urlsplit(str(url))
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("cloud URL must be an absolute HTTP(S) URL without credentials")
        if parsed.path.rstrip("/") != "/infer":
            raise ValueError("cloud URL must target /infer")
        token = os.environ.get(str(token_env), "")
        if not token:
            raise RuntimeError(f"runtime bearer token is required in {token_env}")
        if len(cloud_sha256) != 64 or any(c not in "0123456789abcdef" for c in cloud_sha256):
            raise ValueError("invalid cloud model SHA-256")
        if not math.isfinite(float(timeout)) or timeout <= 0 or int(retries) <= 0:
            raise ValueError("invalid cloud timeout or retry count")
        self.url = urlunsplit(parsed)
        self.health_url = urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))
        self.cloud_sha256 = cloud_sha256
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.maximum_response_bytes = int(maximum_response_bytes)
        self._token = token

    def __repr__(self) -> str:
        return f"CloudClient(url={self.url!r}, cloud_sha256={self.cloud_sha256!r})"

    def public_contract(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "cloud_sha256": self.cloud_sha256,
            "timeout_seconds": self.timeout,
            "retries": self.retries,
            "transport": wire_contract(),
        }

    def _request(self, *, url: str, body: bytes | None) -> tuple[dict[str, Any], int, float, int]:
        headers = {"Accept": "application/json", "Authorization": f"Bearer {self._token}"}
        if body is not None:
            headers["Content-Type"] = WIRE_MEDIA_TYPE
        request = Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
        started = time.perf_counter()
        try:
            with _NO_REDIRECT_OPENER.open(request, timeout=self.timeout) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > self.maximum_response_bytes:
                    raise RuntimeError("cloud response exceeds size limit")
                raw = response.read(self.maximum_response_bytes + 1)
                status = int(response.status)
        except HTTPError as exc:
            status = int(exc.code)
            # A rejected redirect may have a response stream whose peer closes
            # immediately; it is enough to record the status and never follow it.
            raw = b"" if 300 <= status < 400 else exc.read(self.maximum_response_bytes + 1)
        if len(raw) > self.maximum_response_bytes:
            raise RuntimeError("cloud response exceeds size limit")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("cloud returned invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError("cloud returned a non-object response")
        return dict(payload), status, (time.perf_counter() - started) * 1000.0, len(raw)

    def validate_health(self) -> dict[str, Any]:
        try:
            health, status, _, _ = self._request(url=self.health_url, body=None)
        except URLError as exc:
            raise RuntimeError("cloud health transport failed") from exc
        expected = {
            "model_sha256": self.cloud_sha256,
            "imgsz": 640,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "action_protocols": FROZEN_CLOUD_PROTOCOLS,
            "authentication_required": True,
            "transport": wire_contract(),
        }
        if status != 200 or health.get("status") != "ok" or any(health.get(key) != value for key, value in expected.items()):
            raise RuntimeError("cloud health contract mismatch")
        return {"status": "ok", **expected}

    def infer(
        self,
        *,
        image_id: int,
        action: str,
        payload: bytes,
        roi_xyxy: Sequence[int] | None,
        shape: Sequence[int],
    ) -> dict[str, Any]:
        metadata = {
            "task_id": f"online-{int(image_id)}-{action}",
            "frame_id": int(image_id),
            "action_id": action,
            "payload_type": "roi" if action == "current@640" else "keyframe",
            "payload_bytes": len(payload),
            "crop_xyxy": list(map(int, roi_xyxy)) if action == "current@640" and roi_xyxy is not None else None,
            "orig_shape": list(map(int, shape)),
            "cloud_imgsz": 640,
        }
        body = encode_inference_request(metadata, payload)
        elapsed, response_bytes = 0.0, 0
        last_error = "request_failed"
        for attempt in range(1, self.retries + 1):
            attempt_started = time.perf_counter()
            try:
                response, status, roundtrip, response_size = self._request(url=self.url, body=body)
                elapsed += roundtrip
                response_bytes += response_size
                if status != 200:
                    last_error = f"http_{status}"
                    continue
                boxes = validate_cloud_response(
                    response, image_id=image_id, action=action,
                    cloud_sha256=self.cloud_sha256, shape=shape,
                )
                return {
                    "ok": True,
                    "cloud_boxes": boxes,
                    "attempts": attempt,
                    "roundtrip_ms": elapsed,
                    "request_body_bytes": len(body) * attempt,
                    "response_body_bytes": response_bytes,
                    "server_timings_ms": dict(response.get("server_timings_ms", {})),
                }
            except URLError:
                # Include connection failures in the reported per-frame RTT.
                elapsed += max(0.000001, (time.perf_counter() - attempt_started) * 1000.0)
                last_error = "transport_failed"
            except RuntimeError:
                last_error = "invalid_response"
        return {
            "ok": False,
            "error": last_error,
            "attempts": self.retries,
            "roundtrip_ms": elapsed,
            "request_body_bytes": len(body) * self.retries,
            "response_body_bytes": response_bytes,
        }


def _rknn_helpers(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise FileNotFoundError(f"RKNN helper bundle is missing: {directory}")
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    from coco_val_rknn import core_mask, letterbox_val, scale_boxes, ultralytics_multilabel_nms
    from yolo26_decode import decode_rknn_outputs
    return {
        "core_mask": core_mask,
        "letterbox_val": letterbox_val,
        "scale_boxes": scale_boxes,
        "ultralytics_multilabel_nms": ultralytics_multilabel_nms,
        "decode_rknn_outputs": decode_rknn_outputs,
    }


class RKNNFrameRuntime:
    """Open and finish exactly one image inside each observe call."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        helpers_dir: str | Path,
        core: str = "0",
        imgsz: int = 640,
        conf: float = 0.001,
        iou: float = 0.7,
        max_det: int = 300,
        max_nms: int = 30000,
        long_side: int = 640,
    ) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise RuntimeError("RKNNLite is required on the Orange Pi") from exc
        self.helpers = _rknn_helpers(Path(helpers_dir))
        self.imgsz, self.conf, self.iou = int(imgsz), float(conf), float(iou)
        self.max_det, self.max_nms, self.long_side = int(max_det), int(max_nms), int(long_side)
        self.rknn = RKNNLite(verbose=False)
        if self.rknn.load_rknn(str(model_path)) != 0:
            raise RuntimeError("load_rknn failed")
        if self.rknn.init_runtime(core_mask=self.helpers["core_mask"](RKNNLite, core)) != 0:
            raise RuntimeError("init_runtime failed")
        warmup = np.full((1, self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        if self.rknn.inference(inputs=[warmup]) is None:
            raise RuntimeError("RKNN warmup failed")

    def close(self) -> None:
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def observe(
        self,
        item: Mapping[str, Any],
        image_path: Path,
        method: MethodRuntime,
    ) -> dict[str, Any]:
        decode_started = time.perf_counter()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        if image is None:
            raise ValueError(f"could not decode current frame: {item['file_name']}")
        height, width = image.shape[:2]
        if width != int(item["width"]) or height != int(item["height"]):
            raise RuntimeError("current image geometry differs from sequence manifest")

        preprocess_started = time.perf_counter()
        padded, ratio, pad = self.helpers["letterbox_val"](image, self.imgsz)
        rknn_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)[None, ...])
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        infer_started = time.perf_counter()
        outputs = self.rknn.inference(inputs=[rknn_input])
        inference_ms = (time.perf_counter() - infer_started) * 1000.0
        if outputs is None:
            raise RuntimeError("RKNN inference failed")
        post_started = time.perf_counter()
        decoded = self.helpers["decode_rknn_outputs"](outputs, imgsz=self.imgsz)
        boxes, scores, classes, raw_candidates = self.helpers["ultralytics_multilabel_nms"](
            decoded, self.conf, self.iou, self.max_det, self.max_nms,
        )
        boxes = self.helpers["scale_boxes"](boxes, image.shape[:2], ratio, pad)
        edge_boxes = compact_boxes([
            {"xyxy": list(map(float, box)), "conf": float(score), "cls": int(category)}
            for box, score, category in zip(boxes, scores, classes)
        ])
        post_ms = (time.perf_counter() - post_started) * 1000.0

        candidate_started = time.perf_counter()
        enabled = set(method.action_ids)
        roi_xyxy: list[int] | None = None
        if "current@640" in enabled:
            roi_config = method.policy.get("roi", {}) if method.policy else {}
            roi_xyxy = current_roi_xyxy(
                width, height, edge_boxes,
                confidence=float(roi_config.get("confidence", 0.25)),
                max_area_ratio=float(roi_config.get("max_area_ratio", 0.65)),
            )
        original = native_jpeg(image_path)
        raw_payloads: dict[str, bytes] = {"original": original}
        if "q90" in enabled:
            q90 = q90_jpeg(image_path, long_side=self.long_side)
            if 0 < len(q90) < len(original):
                raw_payloads["q90"] = q90
        if "current@640" in enabled and roi_xyxy is not None:
            raw_payloads["current@640"] = q90_jpeg(
                image_path, crop_xyxy=roi_xyxy, long_side=self.long_side,
            )
        payload_bytes = {"edge": 0, **{name: len(value) for name, value in raw_payloads.items()}}
        frame_scores = score_current_frame(
            method, int(item["image_id"]), edge_boxes,
            width=width, height=height, roi_xyxy=roi_xyxy,
        )
        frame_scores = {name: float(frame_scores[name]) for name in payload_bytes if name in frame_scores}
        if set(frame_scores) != set(payload_bytes):
            raise RuntimeError("current-frame score and payload actions differ")
        candidate_ms = (time.perf_counter() - candidate_started) * 1000.0
        return {
            "frame_id": int(item["image_id"]),
            "shape": [height, width],
            "edge_boxes": edge_boxes,
            "raw_multilabel_candidates": int(raw_candidates),
            "roi_xyxy": roi_xyxy,
            "payload_bytes": payload_bytes,
            "payloads": raw_payloads,
            "scores": frame_scores,
            "timings_ms": {
                "decode": decode_ms,
                "edge_preprocess": preprocess_ms,
                "edge_inference": inference_ms,
                "edge_nms": post_ms,
                "candidate": candidate_ms,
            },
        }


def _execute_current_frame(
    method: MethodRuntime,
    cloud: CloudClient,
    observation: Mapping[str, Any],
    decision: Any,
) -> dict[str, Any]:
    height, width = map(int, observation["shape"])
    raw_edge_boxes = compact_boxes(observation["edge_boxes"])
    edge_boxes = sanitize_boxes(
        raw_edge_boxes, width=width, height=height, class_count=80,
    )
    action = str(decision.action)
    cloud_result: dict[str, Any] = {
        "ok": True, "attempts": 0, "roundtrip_ms": 0.0,
        "request_body_bytes": 0, "response_body_bytes": 0,
        "server_timings_ms": {},
    }
    fusion_ms = 0.0
    if action == "edge":
        resolved, execution_status = edge_boxes, "not_required"
    else:
        payload = observation["payloads"].get(action)
        if not isinstance(payload, bytes) or len(payload) != int(decision.payload_bytes):
            raise RuntimeError("selected current-frame payload is missing or changed")
        cloud_result = cloud.infer(
            image_id=int(observation["frame_id"]), action=action, payload=payload,
            roi_xyxy=observation.get("roi_xyxy"), shape=observation["shape"],
        )
        if not cloud_result.get("ok"):
            resolved, execution_status = edge_boxes, "fallback"
        else:
            cloud_boxes = compact_boxes(cloud_result["cloud_boxes"])
            if action == "current@640":
                roi = observation.get("roi_xyxy")
                if roi is None or method.policy is None:
                    raise RuntimeError("ROI action lacks its frozen fusion contract")
                started = time.perf_counter()
                maur = method.policy["maur"]
                resolved = maur_fuse_boxes(
                    edge_boxes, cloud_boxes, roi,
                    duplicate_iou=float(maur["duplicate_iou"]),
                    nms_iou=float(maur["nms_iou"]),
                )
                fusion_ms = (time.perf_counter() - started) * 1000.0
            else:
                resolved = cloud_boxes
            execution_status = "ok"
    raw_resolved = compact_boxes(resolved)
    resolved_boxes = sanitize_boxes(
        raw_resolved, width=width, height=height, class_count=80,
    )
    public_cloud = {
        key: cloud_result[key]
        for key in (
            "ok", "error", "attempts", "roundtrip_ms", "request_body_bytes",
            "response_body_bytes", "server_timings_ms",
        )
        if key in cloud_result
    }
    timings = dict(observation.get("timings_ms", {}))
    timings["cloud_roundtrip"] = float(cloud_result.get("roundtrip_ms", 0.0))
    timings["fusion"] = fusion_ms
    return {
        "shape": list(observation["shape"]),
        "edge_boxes": edge_boxes,
        "roi_xyxy": observation.get("roi_xyxy"),
        "raw_multilabel_candidates": int(observation.get("raw_multilabel_candidates", 0)),
        "payload_bytes": dict(observation["payload_bytes"]),
        "scores": dict(observation["scores"]),
        "resolved_boxes": resolved_boxes,
        "box_sanitization": {
            "edge_input": len(raw_edge_boxes),
            "edge_kept": len(edge_boxes),
            "edge_dropped": len(raw_edge_boxes) - len(edge_boxes),
            "resolved_input": len(raw_resolved),
            "resolved_kept": len(resolved_boxes),
            "resolved_dropped": len(raw_resolved) - len(resolved_boxes),
        },
        "execution_status": execution_status,
        "cloud": public_cloud,
        "timings_ms": timings,
    }


def run_online_sequence(
    *,
    sequence: Sequence[Mapping[str, Any]],
    images_dir: Path,
    method: MethodRuntime,
    controller: CausalByteController,
    frame_runtime: Any,
    cloud: Any,
    frames_path: Path,
    run_id: str,
    contract_sha256: str,
    on_commit: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run/resume a causal stream without opening a suffix frame early."""
    if tuple(controller.action_ids) != tuple(method.action_ids):
        raise ValueError("method and controller action contracts differ")
    controller, start_position, committed = restore_committed_prefix(
        frames_path, sequence, contract_sha256=contract_sha256,
        controller=controller, run_id=run_id,
    )
    records = list(committed)

    def observe(item: Mapping[str, Any]) -> Mapping[str, Any]:
        image_path = (images_dir / str(item["file_name"])).resolve()
        try:
            image_path.relative_to(images_dir.resolve())
        except ValueError as exc:
            raise RuntimeError("sequence image escapes the image root") from exc
        return frame_runtime.observe(item, image_path, method)

    def execute(item: Mapping[str, Any], observation: Mapping[str, Any], decision: Any) -> Mapping[str, Any]:
        return _execute_current_frame(method, cloud, observation, decision)

    def commit(record: dict[str, Any]) -> None:
        row = append_committed_frame(
            frames_path, record, run_id=run_id,
            contract_sha256=contract_sha256,
        )
        records.append(row)
        if on_commit is not None:
            on_commit(row)

    execute_stream(
        sequence, controller=controller, observe=observe, execute=execute,
        commit=commit, start_position=start_position,
    )
    counts = Counter(str(row["decision"]["action"]) for row in records)
    failures = sum(row.get("execution_status") == "fallback" for row in records)
    return {
        "complete": len(records) == len(sequence),
        "frames": len(records),
        "action_counts": dict(sorted(counts.items())),
        "cloud_failures": failures,
        "controller_state": controller.snapshot(),
    }


def load_formal_control(
    path: str | Path,
    *,
    method: str,
    budget: str,
    policy_sha256: str,
    random_seed: int | None = None,
) -> dict[str, Any]:
    try:
        control = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read formal online control") from exc
    if not isinstance(control, Mapping):
        raise ValueError("formal online control must be an object")
    expected_budget = Fraction(str(budget))
    try:
        actual_budget = Fraction(
            int(control.get("budget_numerator", 0)),
            int(control.get("budget_denominator", 0)),
        )
        price = float(control.get("lambda"))
        median = float(control.get("median_original_bytes"))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid formal online control") from exc
    hashes = control.get("source_hashes")
    objective = control.get("calibration_objective")
    grid = control.get("lambda_grid")
    bound_seed = control.get("random_seed")
    def valid_sha256(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    try:
        declared_budget_matches = math.isclose(
            float(control.get("budget")), float(actual_budget),
            rel_tol=0.0, abs_tol=1e-15,
        )
        objective_valid = (
            isinstance(objective, Mapping)
            and bool(str(objective.get("name", "")))
            and objective.get("direction") == "maximize"
            and math.isfinite(float(objective.get("value")))
        )
        grid_values = [float(value) for value in grid] if isinstance(grid, list) else []
        grid_valid = (
            bool(grid_values)
            and all(math.isfinite(value) and value >= 0.0 for value in grid_values)
            and grid_values == sorted(set(grid_values))
            and price in grid_values
        )
    except (TypeError, ValueError, OverflowError):
        declared_budget_matches = objective_valid = grid_valid = False
    schema = control.get("schema")
    adaptive_valid = True
    if schema == PRIMAL_DUAL_CONTROL_SCHEMA:
        try:
            lower = float(control.get("lambda_minimum"))
            upper = float(control.get("lambda_maximum"))
            initial = float(control.get("lambda_initial"))
            step_size = float(control.get("step_size"))
            step_grid = [float(value) for value in control.get("step_size_grid", [])]
            adaptive_valid = (
                control.get("controller") == "projected_primal_dual"
                and control.get("method") in {"cavs_oq", "cavs_oqr"}
                and control.get("buffer_frames") == 0
                and control.get("zero_lookahead") is True
                and control.get("global_plan") is False
                and all(math.isfinite(value) for value in (lower, upper, initial, step_size))
                and 0.0 <= lower <= initial <= upper
                and math.isclose(initial, price, rel_tol=0.0, abs_tol=1e-15)
                and step_size >= 0.0
                and bool(step_grid)
                and step_grid == sorted(set(step_grid))
                and all(math.isfinite(value) and value >= 0.0 for value in step_grid)
                and step_size in step_grid
                and isinstance(hashes, Mapping)
                and valid_sha256(hashes.get("base_control"))
            )
        except (TypeError, ValueError, OverflowError):
            adaptive_valid = False
    elif schema != CONTROL_SCHEMA:
        adaptive_valid = False
    valid = (
        schema in {CONTROL_SCHEMA, PRIMAL_DUAL_CONTROL_SCHEMA}
        and adaptive_valid
        and control.get("formal") is True
        and control.get("source_split") == "calibration"
        and control.get("final_used_for_tuning") is False
        and control.get("protocol_id") == PROTOCOL_ID
        and control.get("method") == method
        and actual_budget == expected_budget
        and declared_budget_matches
        and math.isfinite(price) and price >= 0.0
        and math.isfinite(median) and median > 0.0
        and isinstance(control.get("calibration_frames"), int)
        and not isinstance(control.get("calibration_frames"), bool)
        and int(control["calibration_frames"]) > 0
        and objective_valid
        and grid_valid
        and isinstance(hashes, Mapping)
        and hashes.get("policy") == policy_sha256
        and all(valid_sha256(value) for value in hashes.values())
        and valid_sha256(hashes.get("calibration_records"))
        and valid_sha256(hashes.get("protocol"))
        and (
            (method != "random" and bound_seed is None)
            or (
                method == "random"
                and isinstance(bound_seed, int)
                and not isinstance(bound_seed, bool)
                and random_seed is not None
                and int(bound_seed) == int(random_seed)
            )
        )
    )
    if not valid:
        raise ValueError("formal control is not calibration-frozen or does not match this run")
    return dict(control)


def build_controller_from_control(
    control: Mapping[str, Any],
    *,
    action_ids: Sequence[str],
) -> CausalByteController:
    budget = Fraction(
        int(control["budget_numerator"]), int(control["budget_denominator"])
    )
    common = {
        "budget": budget,
        "median_original_bytes": float(control["median_original_bytes"]),
        "action_ids": action_ids,
    }
    if control.get("schema") == PRIMAL_DUAL_CONTROL_SCHEMA:
        return PrimalDualByteController(
            **common,
            initial_price=float(control["lambda_initial"]),
            step_size=float(control["step_size"]),
            minimum_price=float(control["lambda_minimum"]),
            maximum_price=float(control["lambda_maximum"]),
        )
    return CausalByteController(**common, price=float(control["lambda"]))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--random-seed", type=int, default=20260720)
    parser.add_argument("--budget", required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--rknn", type=Path, required=True)
    parser.add_argument("--rknn-helpers-dir", type=Path, required=True)
    parser.add_argument(
        "--rknn-helpers-manifest",
        type=Path,
        default=ROOT / "configs" / "rknn_helpers_manifest.json",
    )
    parser.add_argument("--bundle-manifest", type=Path)
    parser.add_argument("--cloud-url", required=True)
    parser.add_argument("--cloud-sha256", required=True)
    parser.add_argument("--token-env", default="CAVS_TOKEN")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--core", choices=("auto", "0", "1", "2", "all"), default="0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cloud-timeout", type=float, default=60.0)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit < 0 or args.progress_every <= 0:
        raise ValueError("limit must be non-negative and progress interval positive")
    sequence = load_images_only_sequence(args.sequence)
    if args.limit:
        sequence = sequence[: args.limit]
    missing = [row["file_name"] for row in sequence if not (args.images_dir / row["file_name"]).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} sequence images; first={missing[0]}")
    rknn_sha256 = file_sha256(args.rknn)
    verify_rknn_helpers_manifest(args.rknn_helpers_manifest, args.rknn_helpers_dir)
    policy_sha256 = file_sha256(args.policy)
    method = build_method_runtime(
        args.method, policy_path=args.policy,
        random_seed=args.random_seed,
        expected_rknn_sha256=rknn_sha256,
    )
    control = load_formal_control(
        args.control, method=method.method, budget=args.budget,
        policy_sha256=policy_sha256,
        random_seed=args.random_seed if method.method == "random" else None,
    )
    cloud = CloudClient(
        args.cloud_url, token_env=args.token_env,
        cloud_sha256=args.cloud_sha256, timeout=args.cloud_timeout,
        retries=args.request_retries,
    )
    cloud.validate_health()
    contract = build_run_contract(
        run_id=args.run_id, method=method.method, budget=args.budget,
        sequence_path=args.sequence, policy_path=args.policy,
        control_path=args.control, rknn_path=args.rknn,
        cloud_sha256=args.cloud_sha256, action_ids=method.action_ids,
        random_seed=args.random_seed if method.method == "random" else None,
        bundle_manifest_path=args.bundle_manifest,
        rknn_helpers_manifest_path=args.rknn_helpers_manifest,
    )
    contract["image_count"] = len(sequence)
    contract["sequence_records_sha256"] = _canonical_records_sha256(sequence)
    contract["cloud_endpoint"] = cloud.public_contract()
    contract["controller"] = {
        "schema": str(control["schema"]),
        "id": str(control.get("controller", "fixed_shadow_price")),
        "initial_price": float(control.get("lambda_initial", control["lambda"])),
        "step_size": float(control.get("step_size", 0.0)),
        "buffer_frames": 0,
    }
    run_dir = args.runs_root.resolve() / args.run_id
    contract_sha256 = ensure_run_contract(run_dir, contract)
    controller = build_controller_from_control(control, action_ids=method.action_ids)
    runtime = RKNNFrameRuntime(
        args.rknn, helpers_dir=args.rknn_helpers_dir, core=args.core,
    )
    try:
        def progress(row: Mapping[str, Any]) -> None:
            done = int(row["position"]) + 1
            if done == 1 or done % args.progress_every == 0 or done == len(sequence):
                print(
                    f"online method={method.method} budget={args.budget} "
                    f"frames={done}/{len(sequence)} action={row['decision']['action']} "
                    f"status={row['execution_status']}",
                    flush=True,
                )

        with exclusive_run_lock(run_dir):
            summary = run_online_sequence(
                sequence=sequence, images_dir=args.images_dir.resolve(), method=method,
                controller=controller, frame_runtime=runtime, cloud=cloud,
                frames_path=run_dir / "frames.jsonl", run_id=args.run_id,
                contract_sha256=contract_sha256, on_commit=progress,
            )
    finally:
        runtime.close()
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
