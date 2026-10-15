from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
from urllib.error import URLError
from unittest.mock import patch
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from online_control import CausalByteController, PrimalDualByteController
from online_methods import MethodRuntime
from online_runner import (
    _execute_current_frame,
    build_controller_from_control,
    CloudClient,
    load_formal_control,
    load_images_only_sequence,
    run_online_sequence,
    validate_cloud_response,
)
from protocol import WIRE_MEDIA_TYPE, decode_inference_request, wire_contract


class FakeFrameRuntime:
    def __init__(self, events: list[tuple[str, int]]) -> None:
        self.events = events

    def observe(self, item, image_path, method):
        image_id = int(item["image_id"])
        self.events.append(("open", image_id))
        return {
            "frame_id": image_id,
            "shape": [int(item["height"]), int(item["width"])],
            "edge_boxes": [{"xyxy": [1.0, 1.0, 5.0, 5.0], "conf": 0.9, "cls": 0}],
            "roi_xyxy": None,
            "payload_bytes": {"edge": 0, "original": 100},
            "payloads": {"original": b"j" * 100},
            "scores": {"edge": 0.0, "original": 1.0},
            "timings_ms": {"edge_total": 1.0, "candidate": 0.1},
        }

    def close(self):
        return None


class FakeCloud:
    def __init__(self, events: list[tuple[str, int]], *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail

    def infer(self, *, image_id, action, payload, roi_xyxy, shape):
        self.events.append(("cloud", int(image_id)))
        if self.fail:
            return {
                "ok": False,
                "error": "transport_failed",
                "attempts": 3,
                "roundtrip_ms": 3.0,
                "request_body_bytes": 10,
                "response_body_bytes": 0,
            }
        return {
            "ok": True,
            "cloud_boxes": [{"xyxy": [2.0, 2.0, 6.0, 6.0], "conf": 0.8, "cls": 0}],
            "attempts": 1,
            "roundtrip_ms": 2.0,
            "request_body_bytes": 10,
            "response_body_bytes": 20,
            "server_timings_ms": {"cloud_infer": 1.5},
        }


class OnlineRunnerTest(unittest.TestCase):
    def _manifest(self, records):
        digest = hashlib.sha256(json.dumps(
            records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        return {
            "schema": "coco_images_only_sequence_v1",
            "dataset": "coco",
            "split": "val2017",
            "images": len(records),
            "source_annotations_sha256": "f" * 64,
            "sequence_sha256": digest,
            "records": records,
        }

    def test_loads_hash_pinned_host_images_only_manifest(self) -> None:
        records = [
            {"image_id": 1, "file_name": "0001.jpg", "width": 100, "height": 80},
        ]
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "sequence.json"
            manifest.write_text(json.dumps(self._manifest(records)), encoding="utf-8")
            self.assertEqual(load_images_only_sequence(manifest), records)

    def test_images_only_manifest_rejects_annotations_and_path_escape(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "sequence.json"
            manifest.write_text(json.dumps(self._manifest([{
                    "image_id": 1, "file_name": "../secret.jpg", "width": 100,
                    "height": 80, "annotations": [],
                }])), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "images-only"):
                load_images_only_sequence(manifest)

    def test_images_only_manifest_requires_record_hash(self) -> None:
        records = [
            {"image_id": 1, "file_name": "0001.jpg", "width": 100, "height": 80},
        ]
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "sequence.json"
            payload = self._manifest(records)
            payload["sequence_sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_images_only_sequence(manifest)

    def test_one_frame_is_committed_before_next_image_is_opened(self) -> None:
        events: list[tuple[str, int]] = []
        sequence = [
            {"image_id": 1, "file_name": "0001.jpg", "width": 100, "height": 80},
            {"image_id": 2, "file_name": "0002.jpg", "width": 100, "height": 80},
        ]
        with TemporaryDirectory() as directory:
            frames = Path(directory) / "frames.jsonl"
            run_online_sequence(
                sequence=sequence,
                images_dir=Path(directory),
                method=MethodRuntime("confidence", ("edge", "original")),
                controller=CausalByteController(
                    budget="1", price=0.0, median_original_bytes=100,
                    action_ids=("edge", "original"),
                ),
                frame_runtime=FakeFrameRuntime(events),
                cloud=FakeCloud(events),
                frames_path=frames,
                run_id="online-confidence-100",
                contract_sha256="a" * 64,
                on_commit=lambda row: events.append(("commit", int(row["image_id"]))),
            )

            self.assertEqual(events, [
                ("open", 1), ("cloud", 1), ("commit", 1),
                ("open", 2), ("cloud", 2), ("commit", 2),
            ])
            rows = [json.loads(line) for line in frames.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["status"] for row in rows], ["committed", "committed"])
            self.assertEqual([row["resolved_boxes"][0]["conf"] for row in rows], [0.8, 0.8])

    def test_cloud_failure_commits_edge_fallback_and_resume_skips_prefix(self) -> None:
        sequence = [
            {"image_id": 1, "file_name": "0001.jpg", "width": 100, "height": 80},
            {"image_id": 2, "file_name": "0002.jpg", "width": 100, "height": 80},
        ]
        with TemporaryDirectory() as directory:
            frames = Path(directory) / "frames.jsonl"
            first_events: list[tuple[str, int]] = []
            run_online_sequence(
                sequence=sequence[:1], images_dir=Path(directory),
                method=MethodRuntime("confidence", ("edge", "original")),
                controller=CausalByteController(
                    budget="1", price=0.0, median_original_bytes=100,
                    action_ids=("edge", "original"),
                ),
                frame_runtime=FakeFrameRuntime(first_events), cloud=FakeCloud(first_events, fail=True),
                frames_path=frames, run_id="online", contract_sha256="a" * 64,
            )
            resumed_events: list[tuple[str, int]] = []
            run_online_sequence(
                sequence=sequence, images_dir=Path(directory),
                method=MethodRuntime("confidence", ("edge", "original")),
                controller=CausalByteController(
                    budget="1", price=0.0, median_original_bytes=100,
                    action_ids=("edge", "original"),
                ),
                frame_runtime=FakeFrameRuntime(resumed_events), cloud=FakeCloud(resumed_events),
                frames_path=frames, run_id="online", contract_sha256="a" * 64,
            )
            rows = [json.loads(line) for line in frames.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(resumed_events, [("open", 2), ("cloud", 2)])
            self.assertEqual(rows[0]["execution_status"], "fallback")
            self.assertEqual(rows[0]["resolved_boxes"][0]["conf"], 0.9)
            self.assertEqual(len(rows), 2)

    def test_failed_frame_is_never_appended_as_committed(self) -> None:
        class ExplodingCloud:
            def infer(self, **kwargs):
                raise RuntimeError("cloud process stopped")

        sequence = [
            {"image_id": 1, "file_name": "0001.jpg", "width": 100, "height": 80},
        ]
        with TemporaryDirectory() as directory:
            frames = Path(directory) / "frames.jsonl"
            with self.assertRaisesRegex(RuntimeError, "cloud process stopped"):
                run_online_sequence(
                    sequence=sequence, images_dir=Path(directory),
                    method=MethodRuntime("confidence", ("edge", "original")),
                    controller=CausalByteController(
                        budget="1", price=0.0, median_original_bytes=100,
                        action_ids=("edge", "original"),
                    ),
                    frame_runtime=FakeFrameRuntime([]), cloud=ExplodingCloud(),
                    frames_path=frames, run_id="online",
                    contract_sha256="a" * 64,
                )
            self.assertFalse(frames.exists())

    def test_cloud_response_requires_exact_identity_model_and_valid_boxes(self) -> None:
        valid = {
            "status": "ok", "frame_id": 17, "action_id": "q90",
            "model_sha256": "c" * 64, "cloud_boxes": [],
            "server_timings_ms": {"cloud_infer": 1.0},
        }
        self.assertEqual(
            validate_cloud_response(
                valid, image_id=17, action="q90", cloud_sha256="c" * 64,
                shape=(80, 100),
            ),
            [],
        )
        for changed in (
            {**valid, "frame_id": 18},
            {**valid, "model_sha256": "d" * 64},
            {**valid, "cloud_boxes": [{"xyxy": [0, 0, -1, 2], "conf": 0.9, "cls": 0}]},
            {**valid, "cloud_boxes": [{"xyxy": [0, 0, 0, 2], "conf": 0.9, "cls": 0}]},
            {**valid, "cloud_boxes": [{"xyxy": [0, 0, 101, 2], "conf": 0.9, "cls": 0}]},
            {**valid, "cloud_boxes": [{"xyxy": [0, 0, 1, 2], "conf": 1.1, "cls": 0}]},
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid cloud response"):
                validate_cloud_response(
                    changed, image_id=17, action="q90", cloud_sha256="c" * 64,
                    shape=(80, 100),
                )

    def test_binary_cloud_request_is_metered_without_base64_expansion(self) -> None:
        class CaptureHandler(BaseHTTPRequestHandler):
            body = b""
            content_type = None

            def log_message(self, format, *args):
                pass

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                type(self).body = self.rfile.read(length)
                type(self).content_type = self.headers.get("Content-Type")
                metadata, _ = decode_inference_request(
                    type(self).body, maximum_bytes=length
                )
                response = json.dumps({
                    "status": "ok",
                    "frame_id": metadata["frame_id"],
                    "action_id": metadata["action_id"],
                    "model_sha256": "c" * 64,
                    "cloud_boxes": [],
                    "server_timings_ms": {"cloud_infer": 1.0},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

        server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        payload = b"j" * 20_000
        with patch.dict(os.environ, {"ONLINE_SECRET": "top-secret"}, clear=True):
            client = CloudClient(
                f"http://127.0.0.1:{server.server_port}/infer",
                token_env="ONLINE_SECRET", cloud_sha256="c" * 64, retries=1,
            )
            result = client.infer(
                image_id=1, action="original", payload=payload,
                roi_xyxy=None, shape=(80, 100),
            )

        self.assertTrue(result["ok"])
        self.assertEqual(CaptureHandler.content_type, WIRE_MEDIA_TYPE)
        self.assertEqual(result["request_body_bytes"], len(CaptureHandler.body))
        self.assertLess(result["request_body_bytes"] / len(payload), 1.02)

    def test_health_contract_requires_the_exact_v2_transport(self) -> None:
        health = {
            "status": "ok",
            "model_sha256": "c" * 64,
            "imgsz": 640,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "action_protocols": {
                "original": {"end2end": False, "rect": True, "scaleup": False, "half": False, "multi_label": True},
                "q90": {"end2end": False, "rect": True, "scaleup": False, "half": False, "multi_label": True},
                "current@640": {"end2end": False, "rect": True, "scaleup": True, "half": True, "multi_label": False},
            },
            "authentication_required": True,
            "transport": wire_contract(),
        }
        with patch.dict(os.environ, {"ONLINE_SECRET": "top-secret"}, clear=True):
            client = CloudClient(
                "http://127.0.0.1:8767/infer", token_env="ONLINE_SECRET",
                cloud_sha256="c" * 64,
            )
        with patch.object(client, "_request", return_value=(health, 200, 1.0, 10)):
            self.assertEqual(client.validate_health()["transport"], wire_contract())
        stale = dict(health)
        stale.pop("transport")
        with patch.object(client, "_request", return_value=(stale, 200, 1.0, 10)):
            with self.assertRaisesRegex(RuntimeError, "health contract"):
                client.validate_health()

    def test_final_edge_and_fallback_boxes_are_clipped_and_degenerate_boxes_dropped(self) -> None:
        observation = {
            "frame_id": 1,
            "shape": [80, 100],
            "edge_boxes": [
                {"xyxy": [-2.0, -3.0, 5.0, 6.0], "conf": 0.9, "cls": 0},
                {"xyxy": [10.0, 10.0, 10.0, 20.0], "conf": 0.8, "cls": 0},
                {"xyxy": [1.0, 1.0, 3.0, 3.0], "conf": float("nan"), "cls": 0},
                {"xyxy": [95.0, 70.0, 110.0, 90.0], "conf": 0.7, "cls": 1},
            ],
            "roi_xyxy": None,
            "payload_bytes": {"edge": 0, "original": 100},
            "payloads": {"original": b"j" * 100},
            "scores": {"edge": 1.0, "original": 0.0},
            "timings_ms": {},
        }
        result = _execute_current_frame(
            MethodRuntime("confidence", ("edge", "original")),
            FakeCloud([]), observation,
            SimpleNamespace(action="edge", payload_bytes=0),
        )
        self.assertEqual(
            result["resolved_boxes"],
            [
                {"xyxy": [0.0, 0.0, 5.0, 6.0], "conf": 0.9, "cls": 0},
                {"xyxy": [95.0, 70.0, 100.0, 80.0], "conf": 0.7, "cls": 1},
            ],
        )
        self.assertEqual(result["box_sanitization"]["edge_dropped"], 2)

    def test_cloud_token_is_required_from_environment_and_never_stored(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "runtime bearer token"):
                CloudClient(
                    "http://127.0.0.1:8767/infer", token_env="ONLINE_SECRET",
                    cloud_sha256="c" * 64,
                )
        with patch.dict(os.environ, {"ONLINE_SECRET": "top-secret"}, clear=True):
            client = CloudClient(
                "http://127.0.0.1:8767/infer", token_env="ONLINE_SECRET",
                cloud_sha256="c" * 64,
            )
            self.assertNotIn("top-secret", repr(client))
            self.assertNotIn("top-secret", json.dumps(client.public_contract()))

    def test_authenticated_request_does_not_follow_redirect(self) -> None:
        class RedirectHandler(BaseHTTPRequestHandler):
            captures = []

            def log_message(self, format, *args):
                pass

            def do_POST(self):
                if self.path == "/infer":
                    self.send_response(302)
                    self.send_header("Location", "/capture")
                    self.end_headers()
                    return
                type(self).captures.append(self.headers.get("Authorization"))
                body = b'{"status":"error"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with patch.dict(os.environ, {"ONLINE_SECRET": "top-secret"}, clear=True):
            client = CloudClient(
                f"http://127.0.0.1:{server.server_port}/infer",
                token_env="ONLINE_SECRET", cloud_sha256="c" * 64, retries=1,
            )
            result = client.infer(
                image_id=1, action="original", payload=b"jpeg",
                roi_xyxy=None, shape=(80, 100),
            )
        self.assertFalse(result["ok"])
        self.assertEqual(RedirectHandler.captures, [])

    def test_failed_attempt_latency_is_not_reported_as_zero(self) -> None:
        with patch.dict(os.environ, {"ONLINE_SECRET": "top-secret"}, clear=True):
            client = CloudClient(
                "http://127.0.0.1:8767/infer", token_env="ONLINE_SECRET",
                cloud_sha256="c" * 64, retries=1,
            )
        with patch.object(client, "_request", side_effect=URLError("offline")):
            result = client.infer(
                image_id=1, action="original", payload=b"jpeg",
                roi_xyxy=None, shape=(80, 100),
            )
        self.assertFalse(result["ok"])
        self.assertGreater(result["roundtrip_ms"], 0.0)

    def test_formal_control_must_be_calibration_frozen_and_policy_bound(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.json"
            policy.write_text("{}\n", encoding="utf-8")
            policy_sha256 = hashlib.sha256(policy.read_bytes()).hexdigest()
            control = {
                "schema": "cavs_online_control_v1",
                "formal": True,
                "protocol_id": "coco_rknn_int8_heads_causal_online_hil_v1",
                "method": "cavs_oqr",
                "budget": 0.1,
                "budget_numerator": 1,
                "budget_denominator": 10,
                "lambda": 0.25,
                "median_original_bytes": 100.0,
                "calibration_frames": 571,
                "calibration_objective": {
                    "name": "teacher_contextual_ap50_95",
                    "direction": "maximize",
                    "value": 1.0,
                },
                "lambda_grid": [0.0, 0.25, 0.5],
                "source_split": "calibration",
                "source_hashes": {
                    "policy": policy_sha256,
                    "calibration_records": "a" * 64,
                    "protocol": "b" * 64,
                },
                "final_used_for_tuning": False,
            }
            path = root / "control.json"
            path.write_text(json.dumps(control), encoding="utf-8")
            loaded = load_formal_control(
                path, method="cavs_oqr", budget="0.1",
                policy_sha256=policy_sha256,
            )
            self.assertEqual(loaded["lambda"], 0.25)

            control["final_used_for_tuning"] = True
            path.write_text(json.dumps(control), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "calibration-frozen"):
                load_formal_control(
                    path, method="cavs_oqr", budget="0.1",
                    policy_sha256=policy_sha256,
                )

    def test_random_control_rejects_a_different_runtime_seed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.py"
            policy.write_text("# random score\n", encoding="utf-8")
            policy_sha256 = hashlib.sha256(policy.read_bytes()).hexdigest()
            control = {
                "schema": "cavs_online_control_v1", "formal": True,
                "protocol_id": "coco_rknn_int8_heads_causal_online_hil_v1",
                "method": "random", "random_seed": 20260720,
                "budget": 0.1, "budget_numerator": 1, "budget_denominator": 10,
                "lambda": 0.5, "median_original_bytes": 100.0,
                "calibration_frames": 571,
                "calibration_objective": {
                    "name": "predicted_action_value", "direction": "maximize",
                    "value": 1.0,
                },
                "lambda_grid": [0.5], "source_split": "calibration",
                "source_hashes": {
                    "policy": policy_sha256, "calibration_records": "a" * 64,
                    "protocol": "b" * 64,
                },
                "final_used_for_tuning": False,
            }
            path = root / "control.json"
            path.write_text(json.dumps(control), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "calibration-frozen"):
                load_formal_control(
                    path, method="random", budget="0.1",
                    policy_sha256=policy_sha256, random_seed=20260721,
                )

    def test_adaptive_control_is_validated_and_builds_zero_buffer_controller(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.json"
            policy.write_text("{}\n", encoding="utf-8")
            policy_sha256 = hashlib.sha256(policy.read_bytes()).hexdigest()
            control = {
                "schema": "cavs_online_primal_dual_control_v1",
                "controller": "projected_primal_dual",
                "formal": True,
                "protocol_id": "coco_rknn_int8_heads_causal_online_hil_v1",
                "method": "cavs_oqr",
                "budget": 0.25,
                "budget_numerator": 1,
                "budget_denominator": 4,
                "lambda": 0.3,
                "lambda_initial": 0.3,
                "lambda_minimum": 0.0,
                "lambda_maximum": 5.0,
                "step_size": 0.001,
                "step_size_grid": [0.0, 0.001],
                "median_original_bytes": 100.0,
                "calibration_frames": 571,
                "calibration_objective": {
                    "name": "predicted_action_value", "direction": "maximize",
                    "value": 2.0,
                },
                "lambda_grid": [0.0, 0.3],
                "source_split": "calibration",
                "source_hashes": {
                    "policy": policy_sha256,
                    "calibration_records": "a" * 64,
                    "protocol": "b" * 64,
                    "base_control": "c" * 64,
                },
                "buffer_frames": 0,
                "zero_lookahead": True,
                "global_plan": False,
                "final_used_for_tuning": False,
            }
            path = root / "control.json"
            path.write_text(json.dumps(control), encoding="utf-8")
            loaded = load_formal_control(
                path, method="cavs_oqr", budget="0.25",
                policy_sha256=policy_sha256,
            )
            controller = build_controller_from_control(
                loaded, action_ids=("edge", "original", "q90", "current@640")
            )
            self.assertIsInstance(controller, PrimalDualByteController)
            self.assertEqual(controller.step_size, 0.001)

            control["method"] = "cavs_oq"
            path.write_text(json.dumps(control), encoding="utf-8")
            oq_loaded = load_formal_control(
                path, method="cavs_oq", budget="0.25",
                policy_sha256=policy_sha256,
            )
            oq_controller = build_controller_from_control(
                oq_loaded, action_ids=("edge", "original", "q90")
            )
            self.assertIsInstance(oq_controller, PrimalDualByteController)

            control["method"] = "cavs_oqr"
            control["buffer_frames"] = 1
            path.write_text(json.dumps(control), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "calibration-frozen"):
                load_formal_control(
                    path, method="cavs_oqr", budget="0.25",
                    policy_sha256=policy_sha256,
                )

            control["buffer_frames"] = 0
            control["source_hashes"] = None
            path.write_text(json.dumps(control), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "calibration-frozen"):
                load_formal_control(
                    path, method="cavs_oqr", budget="0.25",
                    policy_sha256=policy_sha256,
                )


if __name__ == "__main__":
    unittest.main()
