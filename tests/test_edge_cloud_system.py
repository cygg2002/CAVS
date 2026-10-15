# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the local-edge/remote-cloud demonstration system."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import pytest

DEMO_DIR = Path(__file__).resolve().parents[1] / "examples" / "YOLO26-Edge-Cloud"
sys.path.insert(0, str(DEMO_DIR))

import gradio_demo
from cloud_server import build_health_payload, resolve_cloud_imgsz, validate_request_length
from demo_system import (
    check_cloud_health,
    infer_url_to_health_url,
    render_demo_report,
    run_preflight,
)
from gradio_demo import (
    build_app,
    cloud_health_markdown,
    cumulative_summary,
    save_event,
    temporal_memory_active,
)
from gradio_demo import parse_args as parse_gradio_args


class HealthHandler(BaseHTTPRequestHandler):
    """Return a deterministic cloud health response for integration tests."""

    def log_message(self, _format: str, *_args) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/health":
            body = json.dumps(
                {
                    "status": "ok",
                    "model": "test-cloud.pt",
                    "model_sha256": "f" * 64,
                    "device": "cpu",
                    "imgsz": 640,
                    "conf": 0.001,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


@pytest.fixture
def health_server():
    """Yield a localhost health server and close it after the test."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), HealthHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/infer"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_infer_url_to_health_url_validates_external_input():
    """Only absolute HTTP(S) endpoints without embedded credentials are accepted."""
    assert infer_url_to_health_url("http://127.0.0.1:8765/infer") == "http://127.0.0.1:8765/health"
    assert infer_url_to_health_url("https://cloud.example.test/api/infer") == "https://cloud.example.test/api/health"

    with pytest.raises(ValueError, match="HTTP"):
        infer_url_to_health_url("file:///tmp/infer")
    with pytest.raises(ValueError, match="credentials"):
        infer_url_to_health_url("http://user:secret@cloud.example.test/infer")


def test_check_cloud_health_uses_real_http_boundary(health_server):
    """The health probe returns a concise, presentation-ready status."""
    result = check_cloud_health(health_server, timeout=1.0)

    assert result["ok"] is True
    assert result["status"] == "online"
    assert result["model"] == "test-cloud.pt"
    assert result["model_sha256"] == "f" * 64
    assert result["device"] == "cpu"
    assert result["imgsz"] == "640"
    assert result["conf"] == "0.001"
    assert result["latency_ms"] >= 0.0
    assert result["health_url"].endswith("/health")

    rendered = cloud_health_markdown(health_server, 1.0)
    assert "云端状态：在线" in rendered
    assert "ffffffffffff" in rendered
    assert "640" in rendered
    assert "0.001" in rendered


def test_preflight_allows_edge_only_when_cloud_is_offline(tmp_path):
    """Missing cloud connectivity is a warning unless strict cloud mode is requested."""
    model = tmp_path / "yolo26s.pt"
    policy = tmp_path / "learned_upload_policy.json"
    model.write_bytes(b"model")
    policy.write_text(json.dumps({"version": 2}), encoding="utf-8")
    offline_url = "http://127.0.0.1:1/infer"

    relaxed = run_preflight(model, policy, offline_url, cloud_timeout=0.05, require_cloud=False)
    strict = run_preflight(model, policy, offline_url, cloud_timeout=0.05, require_cloud=True)

    assert relaxed["ok"] is True
    assert relaxed["cloud"]["ok"] is False
    assert strict["ok"] is False
    assert any(check["name"] == "cloud" and check["required"] for check in strict["checks"])


def test_preflight_rejects_current_maur_manifest_without_seed_assets(tmp_path):
    """A manifest alone is not a usable learner; every hash-pinned seed must exist."""
    model = tmp_path / "yolo26s.pt"
    model.write_bytes(b"model")
    source = DEMO_DIR / "policies" / "current_maur_v2" / "manifest.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    result = run_preflight(model, manifest, "http://127.0.0.1:1/infer", cloud_timeout=0.05)

    assert result["ok"] is False
    assert any(check["name"] == "policy" and not check["ok"] for check in result["checks"])


def test_preflight_rejects_edge_model_that_does_not_match_current_maur(tmp_path):
    """Current+MAUR may run only with the exact YOLO26n detector used for its training records."""
    model = tmp_path / "yolo26s.pt"
    model.write_bytes(b"not-the-trained-edge-model")
    manifest = DEMO_DIR / "policies" / "current_maur_v2" / "manifest.json"

    result = run_preflight(model, manifest, "http://127.0.0.1:1/infer", cloud_timeout=0.05)

    edge_check = next(check for check in result["checks"] if check["name"] == "edge_model")
    assert result["ok"] is False
    assert edge_check["ok"] is False
    assert "YOLO26n" in edge_check["detail"]


def test_preflight_rejects_cloud_model_that_does_not_match_current_maur(tmp_path, monkeypatch):
    """An online Current+MAUR cloud must expose the checkpoint and inference threshold used by the paper."""
    model = tmp_path / "yolo26n.pt"
    model.write_bytes(b"edge")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "policy_type": "cavs_current_maur",
                "edge_model": {"name": "yolo26n.pt", "sha256": "e" * 64},
                "cloud_model_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("demo_system._file_check", lambda name, *_args: {"name": name, "ok": True, "required": True, "detail": "ok"})
    monkeypatch.setattr("demo_system._verify_current_maur_edge_model", lambda *_args: None)
    monkeypatch.setattr(
        "demo_system.check_cloud_health",
        lambda *_args: {
            "ok": True,
            "status": "online",
            "model": "wrong.pt",
            "model_sha256": "d" * 64,
            "device": "0",
            "imgsz": "640",
            "conf": "0.25",
            "health_url": "http://127.0.0.1:8765/health",
            "latency_ms": 1.0,
        },
    )

    result = run_preflight(model, manifest, "http://127.0.0.1:8765/infer")

    contract = next(check for check in result["checks"] if check["name"] == "cloud_contract")
    assert result["ok"] is False
    assert contract["required"] is True
    assert "SHA256 mismatch" in contract["detail"]
    assert "conf=0.001" in contract["detail"]


def test_render_demo_report_summarizes_upload_and_latency_metrics():
    """The generated report exposes the values needed during a live demonstration."""
    report = render_demo_report(
        {
            "frames": 20,
            "uploads": 8,
            "full_uploads": 3,
            "roi_uploads": 5,
            "payload_kb_total": 256.5,
            "edge_latency_ms_avg": 21.2,
            "cloud_latency_ms_avg": 35.4,
            "roundtrip_ms_avg": 51.8,
            "e2e_ms_avg": 78.1,
            "fps_avg": 12.8,
            "cloud_errors": 1,
        }
    )

    assert "上传率 | 40.00%" in report
    assert "整图 / ROI | 3 / 5" in report
    assert "累计上传 | 256.50 KB" in report
    assert "端侧 / 云端 / RTT / E2E | 21.20 / 35.40 / 51.80 / 78.10 ms" in report
    assert "云端错误 | 1" in report


def test_cumulative_summary_counts_payload_and_attempted_uploads():
    """Cumulative telemetry includes bytes even when a cloud request fails."""
    state = {
        "stats": {"frames": 4, "upload_attempts": 3, "collab_full_ok": 1, "collab_roi_ok": 1, "cloud_errors": 1},
        "payload_kb_total": 96.25,
        "edge_latencies": [10.0, 20.0],
        "cloud_latencies": [30.0, 40.0],
        "roundtrip_latencies": [45.0, 55.0],
        "e2e_latencies": [70.0, 80.0],
        "timing_records": [],
    }

    summary = cumulative_summary(state, {"upload_note": "edge_kept"})

    assert summary["uploads"] == 3
    assert summary["successful_uploads"] == 2
    assert summary["payload_kb_total"] == 96.25
    assert summary["edge_latency_ms_avg"] == 15.0


def test_save_event_writes_json_and_human_report(tmp_path, monkeypatch):
    """Each processed frame refreshes both machine- and human-readable outputs."""
    monkeypatch.setattr("gradio_demo.RUN_SAVE_DIR", tmp_path)
    state = {
        "stats": {"frames": 1, "upload_attempts": 1, "collab_full_ok": 1},
        "payload_kb_total": 12.5,
        "edge_latencies": [10.0],
        "cloud_latencies": [20.0],
        "roundtrip_latencies": [25.0],
        "e2e_latencies": [40.0],
        "timing_records": [],
    }

    save_event({"upload_note": "frame p=0.80 g=0.40"}, state)

    assert (tmp_path / "events.jsonl").is_file()
    assert json.loads((tmp_path / "summary.json").read_text())["payload_kb_total"] == 12.5
    assert "端云上传学习器演示报告" in (tmp_path / "demo_report.md").read_text()


def test_cloud_server_bounds_request_size_and_compute():
    """Untrusted HTTP metadata cannot request unbounded memory or GPU work."""
    assert validate_request_length("1024", max_bytes=2048) == 1024
    assert resolve_cloud_imgsz(None, default=512, minimum=320, maximum=960) == 512
    assert resolve_cloud_imgsz("640", default=512, minimum=320, maximum=960) == 640

    with pytest.raises(ValueError, match="Content-Length"):
        validate_request_length("0", max_bytes=2048)
    with pytest.raises(ValueError, match="too large"):
        validate_request_length("4096", max_bytes=2048)
    with pytest.raises(ValueError, match="cloud_imgsz"):
        resolve_cloud_imgsz(2048, default=512, minimum=320, maximum=960)


def test_cloud_health_payload_reports_the_actual_runtime_contract():
    """The UI can distinguish the Current+MAUR cloud from an unknown legacy service."""
    args = type(
        "Args",
        (),
        {
            "model": "/models/yolo26x.pt",
            "model_sha256": "9" * 64,
            "device": "0",
            "imgsz": 640,
            "conf": 0.001,
        },
    )()

    assert build_health_payload(args) == {
        "status": "ok",
        "model": "yolo26x.pt",
        "model_sha256": "9" * 64,
        "device": "0",
        "imgsz": 640,
        "conf": 0.001,
    }


def test_gradio_accepts_short_positive_cloud_timeout(monkeypatch):
    """A CLI timeout accepted by the launcher must also fit the UI slider bounds."""
    monkeypatch.setattr(sys, "argv", ["gradio_demo.py", "--cloud-timeout", "0.1"])

    args = parse_gradio_args()
    app = build_app(args)
    sliders = [component for component in app.blocks.values() if getattr(component, "label", "") == "Cloud Timeout (s)"]

    assert len(sliders) == 1
    assert sliders[0].minimum <= args.cloud_timeout <= sliders[0].maximum


def test_static_images_do_not_enter_temporal_memory():
    """Repeated clicks on one uploaded image must not look like a stable video track."""
    assert temporal_memory_active(True, "camera") is True
    assert temporal_memory_active(True, "video") is True
    assert temporal_memory_active(True, "image") is False
    assert temporal_memory_active(False, "camera") is False


def test_uploaded_image_dispatch_marks_the_input_as_static(monkeypatch):
    """The Gradio Image tab passes an explicit static source kind to frame processing."""
    observed = {}

    def fake_process(frame, *args, input_kind):
        observed["frame"] = frame
        observed["input_kind"] = input_kind
        return frame, {}, "ok", {}

    monkeypatch.setattr(gradio_demo, "process_frame", fake_process)
    uploaded = np.zeros((8, 8, 3), dtype=np.uint8)

    gradio_demo.process_any_frame(None, uploaded)

    assert observed["frame"] is uploaded
    assert observed["input_kind"] == "image"


def test_processed_video_uses_browser_playable_h264(tmp_path, monkeypatch):
    """The Gradio output must use H.264 rather than the browser-incompatible mp4v codec."""
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 32))
    assert writer.isOpened()
    for value in (0, 64):
        writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
    writer.release()

    monkeypatch.setattr(gradio_demo, "RUN_SAVE_DIR", tmp_path)
    monkeypatch.setattr(
        gradio_demo,
        "process_frame",
        lambda frame, *_args, input_kind: (frame, {"input_kind": input_kind}, "ok", {}),
    )

    output, result, _status, _state = gradio_demo.process_video_file(str(source), 2, 1, {})

    capture = cv2.VideoCapture(output)
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)).lower()
    assert result["video_frames_processed"] == 2
    assert codec in {"h264", "avc1"}


def test_paper_video_runtime_disables_nonpaper_adaptations():
    """The reproducibility path cannot silently use temporal or adaptive deployment extensions."""
    runtime = gradio_demo.paper_video_runtime_args(
        [
            "Edge Only",
            "http://127.0.0.1:8765/infer",
            "yolo26n.pt",
            "cpu",
            "manifest.json",
            512,
            0.25,
            0.25,
            8.0,
            512,
            80,
            ["person"],
            0.6,
            0.08,
            0.02,
            1,
            True,
            416,
            512,
            0.75,
            True,
            0.5,
            45,
            3,
            15,
            0.12,
            0.7,
            True,
            1.35,
            1.6,
            1.45,
            1.15,
            0.5,
            0.7,
            True,
            "25%",
        ]
    )

    assert runtime[0] == gradio_demo.MODE_COLLAB
    assert runtime[5] == 640
    assert runtime[9] == 640
    assert runtime[11] == ["all"]
    assert runtime[16] is False
    assert runtime[20] is False
    assert runtime[27] is False
    assert runtime[34] is False


def test_paper_video_dispatch_freezes_global_actions_before_execution(monkeypatch):
    """The Paper mode sends a complete fixed-byte plan into the second video pass."""
    frame_args = [None] * 36
    frame_args[0] = "Collaboration"
    candidates = [
        {
            "frame_id": 1,
            "scores": {"edge": 0.0, "original": 0.5, "q90": 0.4},
            "payload_bytes": {"edge": 0, "original": 1000, "q90": 400},
        },
        {
            "frame_id": 2,
            "scores": {"edge": 0.0, "original": 0.2, "q90": 0.1},
            "payload_bytes": {"edge": 0, "original": 1000, "q90": 400},
        },
    ]
    observed = {}
    monkeypatch.setattr(gradio_demo, "collect_paper_video_candidates", lambda *_args, **_kwargs: candidates)

    def fake_process(video_file, *args, forced_actions=None, paper_plan=None):
        observed["video_file"] = video_file
        observed["forced_actions"] = forced_actions
        observed["paper_plan"] = paper_plan
        return "out.mp4", {}, "ok", {}

    monkeypatch.setattr(gradio_demo, "process_video_file", fake_process)

    output = gradio_demo.process_video_dispatch(
        "input.mp4",
        *frame_args,
        "Paper Fixed-byte",
        "25%",
        120,
        1,
        {},
    )

    assert output[0] == "out.mp4"
    assert observed["video_file"] == "input.mp4"
    assert observed["forced_actions"] == observed["paper_plan"]["actions"]
    assert observed["paper_plan"]["request_bytes"] <= observed["paper_plan"]["capacity_bytes"]


def test_gradio_defaults_to_current_maur_policy(monkeypatch):
    """The presentation must match the detector and learner used by Current+MAUR training."""
    monkeypatch.setattr(sys, "argv", ["gradio_demo.py"])

    args = parse_gradio_args()

    assert args.edge_model == "yolo26n.pt"
    assert args.policy_path.endswith("policies/current_maur_v2/manifest.json")
    assert args.cloud_timeout == 8.0
    assert args.cloud_imgsz == 640
    assert args.jpeg_quality == 90
