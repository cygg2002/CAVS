# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Preflight, cloud health, and report helpers for the edge-cloud demonstration system."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

DEFAULT_POLICY_PATH = "examples/YOLO26-Edge-Cloud/policies/current_maur_v2/manifest.json"


def infer_url_to_health_url(infer_url: str) -> str:
    """Convert a validated cloud /infer URL to its sibling /health endpoint."""
    parsed = urlsplit(str(infer_url).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Cloud URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Cloud URL must not contain embedded credentials")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Cloud URL contains an invalid port") from exc
    path = parsed.path.rstrip("/")
    if not path.endswith("/infer"):
        raise ValueError("Cloud URL path must end with /infer")
    health_path = f"{path[:-len('/infer')]}/health"
    return urlunsplit((parsed.scheme, parsed.netloc, health_path, "", ""))


def _safe_text(value: Any, default: str = "unknown", limit: int = 160) -> str:
    """Return one bounded line suitable for status panels and reports."""
    text = str(value if value is not None else default).splitlines()[0].strip()
    return (text or default)[:limit]


def check_cloud_health(cloud_url: str, timeout: float = 1.5) -> dict[str, Any]:
    """Probe the cloud verifier and return a stable presentation-facing status."""
    started = time.perf_counter()
    try:
        health_url = infer_url_to_health_url(cloud_url)
    except ValueError as exc:
        return {
            "ok": False,
            "status": "invalid_url",
            "error": _safe_text(exc),
            "health_url": "",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    try:
        response = requests.get(health_url, timeout=max(0.05, float(timeout)), allow_redirects=False)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        if response.status_code != 200:
            return {
                "ok": False,
                "status": "http_error",
                "http_status": response.status_code,
                "error": f"health returned HTTP {response.status_code}",
                "health_url": health_url,
                "latency_ms": latency_ms,
            }
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            return {
                "ok": False,
                "status": "unhealthy",
                "error": "health response did not report status=ok",
                "health_url": health_url,
                "latency_ms": latency_ms,
            }
        return {
            "ok": True,
            "status": "online",
            "model": _safe_text(payload.get("model")),
            "model_sha256": _safe_text(payload.get("model_sha256")),
            "device": _safe_text(payload.get("device")),
            "imgsz": _safe_text(payload.get("imgsz")),
            "conf": _safe_text(payload.get("conf")),
            "health_url": health_url,
            "latency_ms": latency_ms,
        }
    except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "status": "offline",
            "error": _safe_text(exc),
            "health_url": health_url,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }


def _file_check(name: str, path: str | Path, kind: str) -> dict[str, Any]:
    """Validate a required local model or policy file."""
    file_path = Path(path).expanduser()
    ok = file_path.is_file() and file_path.stat().st_size > 0
    detail = f"{file_path} ({file_path.stat().st_size / 1024**2:.2f} MB)" if ok else f"missing: {file_path}"
    if ok and kind == "policy":
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            ok = isinstance(payload, dict)
            if not ok:
                detail = f"invalid policy object: {file_path}"
            elif payload.get("policy_type") == "cavs_current_maur":
                from current_maur_uploader import load_current_maur_policy

                policy = load_current_maur_policy(file_path)
                detail = (
                    f"{policy['display_name']}, seeds={','.join(map(str, policy['ensemble_seeds']))}"
                )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            ok = False
            detail = f"invalid policy JSON: {file_path}"
    return {"name": name, "ok": ok, "required": True, "detail": detail}


def _sha256(path: str | Path) -> str:
    """Hash a local model without loading executable model data."""
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_current_maur_edge_model(
    edge_model: str | Path,
    policy_path: str | Path,
    edge_check: dict[str, Any],
    policy_check: dict[str, Any],
) -> None:
    """Bind Current+MAUR to the exact detector that generated its training records."""
    if not edge_check["ok"] or not policy_check["ok"]:
        return
    payload = json.loads(Path(policy_path).expanduser().read_text(encoding="utf-8"))
    if payload.get("policy_type") != "cavs_current_maur":
        return
    expected = payload["edge_model"]
    expected_name = str(expected["name"])
    expected_sha256 = str(expected["sha256"]).lower()
    actual_path = Path(edge_model).expanduser()
    actual_name = actual_path.name
    actual_sha256 = _sha256(actual_path)
    if actual_name != expected_name or actual_sha256 != expected_sha256:
        edge_check["ok"] = False
        edge_check["detail"] = (
            f"Current+MAUR requires YOLO26n ({expected_name}, sha256={expected_sha256}), "
            f"got {actual_name} (sha256={actual_sha256})"
        )
        return
    edge_check["detail"] += "; Current+MAUR YOLO26n checksum verified"


def _current_maur_cloud_contract(
    policy_path: str | Path,
    policy_check: dict[str, Any],
    cloud: dict[str, Any],
) -> dict[str, Any] | None:
    """Verify the online cloud checkpoint and detector threshold used to create the frozen rewards."""
    if not policy_check["ok"]:
        return None
    payload = json.loads(Path(policy_path).expanduser().read_text(encoding="utf-8"))
    if payload.get("policy_type") != "cavs_current_maur":
        return None
    if not cloud["ok"]:
        return {
            "name": "cloud_contract",
            "ok": False,
            "required": False,
            "detail": "not verified because the cloud is offline",
        }

    expected_sha256 = str(payload.get("cloud_model_sha256", "")).lower()
    actual_sha256 = str(cloud.get("model_sha256", "")).lower()
    problems = []
    if actual_sha256 != expected_sha256:
        problems.append(f"cloud model SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    try:
        cloud_conf = float(cloud.get("conf"))
    except (TypeError, ValueError):
        cloud_conf = float("nan")
    if cloud_conf != 0.001:
        problems.append(f"Current+MAUR requires cloud conf=0.001, got {cloud.get('conf', 'unknown')}")
    return {
        "name": "cloud_contract",
        "ok": not problems,
        "required": True,
        "detail": "; ".join(problems) if problems else "Current+MAUR cloud checkpoint and conf verified",
    }


def run_preflight(
    edge_model: str | Path,
    policy_path: str | Path,
    cloud_url: str,
    cloud_timeout: float = 1.5,
    require_cloud: bool = False,
) -> dict[str, Any]:
    """Check local runtime assets and cloud connectivity before launching the UI."""
    required_modules = ("cv2", "gradio", "requests", "torch", "ultralytics")
    missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
    edge_check = _file_check("edge_model", edge_model, "model")
    policy_check = _file_check("policy", policy_path, "policy")
    _verify_current_maur_edge_model(edge_model, policy_path, edge_check, policy_check)
    checks = [
        {
            "name": "python",
            "ok": sys.version_info >= (3, 9),
            "required": True,
            "detail": sys.version.split()[0],
        },
        {
            "name": "dependencies",
            "ok": not missing_modules,
            "required": True,
            "detail": "ready" if not missing_modules else f"missing: {', '.join(missing_modules)}",
        },
        edge_check,
        policy_check,
    ]
    cloud = check_cloud_health(cloud_url, cloud_timeout)
    checks.append(
        {
            "name": "cloud",
            "ok": bool(cloud["ok"]),
            "required": bool(require_cloud),
            "detail": (
                f"online, {cloud.get('model', 'unknown')}, {cloud['latency_ms']:.1f} ms"
                if cloud["ok"]
                else f"offline: {cloud.get('error', cloud.get('status', 'unknown'))}"
            ),
        }
    )
    cloud_contract = _current_maur_cloud_contract(policy_path, policy_check, cloud)
    if cloud_contract is not None:
        checks.append(cloud_contract)
    return {
        "ok": all(check["ok"] or not check["required"] for check in checks),
        "checks": checks,
        "cloud": cloud,
    }


def render_preflight(result: dict[str, Any]) -> str:
    """Render preflight checks as concise terminal text."""
    lines = ["Edge-cloud demo preflight"]
    for check in result.get("checks", []):
        marker = "OK" if check.get("ok") else ("FAIL" if check.get("required") else "WARN")
        lines.append(f"[{marker}] {check.get('name', 'check')}: {check.get('detail', '')}")
    lines.append("Ready to launch." if result.get("ok") else "Preflight failed.")
    return "\n".join(lines)


def render_demo_report(summary: dict[str, Any]) -> str:
    """Render one human-readable Markdown report from accumulated UI metrics."""
    frames = int(summary.get("frames", 0))
    uploads = int(summary.get("uploads", 0))
    upload_ratio = uploads / frames if frames else 0.0
    return "\n".join(
        [
            "# 端云上传学习器演示报告",
            "",
            "| 指标 | 结果 |",
            "| --- | ---: |",
            f"| 处理帧数 | {frames} |",
            f"| 上传任务 | {uploads} |",
            f"| 上传率 | {upload_ratio:.2%} |",
            f"| 整图 / ROI | {int(summary.get('full_uploads', 0))} / {int(summary.get('roi_uploads', 0))} |",
            f"| 累计上传 | {float(summary.get('payload_kb_total', 0.0)):.2f} KB |",
            (
                "| 端侧 / 云端 / RTT / E2E | "
                f"{float(summary.get('edge_latency_ms_avg', 0.0)):.2f} / "
                f"{float(summary.get('cloud_latency_ms_avg', 0.0)):.2f} / "
                f"{float(summary.get('roundtrip_ms_avg', 0.0)):.2f} / "
                f"{float(summary.get('e2e_ms_avg', 0.0)):.2f} ms |"
            ),
            f"| 平均 FPS | {float(summary.get('fps_avg', 0.0)):.2f} |",
            f"| 云端错误 | {int(summary.get('cloud_errors', 0))} |",
            "",
            "> 上传率按上传任务数 / 处理帧数计算；同一帧可能包含多个 ROI 上传任务。",
            "",
        ]
    )


def write_demo_report(save_dir: str | Path) -> Path:
    """Read summary.json and write demo_report.md in the same run directory."""
    run_dir = Path(save_dir)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise TypeError("summary.json must contain an object")
    report_path = run_dir / "demo_report.md"
    report_path.write_text(render_demo_report(summary), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="Validate local assets and cloud connectivity.")
    preflight.add_argument("--edge-model", default="yolo26n.pt")
    preflight.add_argument(
        "--policy-path",
        default=DEFAULT_POLICY_PATH,
    )
    preflight.add_argument("--cloud-url", default="http://127.0.0.1:8765/infer")
    preflight.add_argument("--cloud-timeout", type=float, default=1.5)
    preflight.add_argument("--require-cloud", action="store_true")
    preflight.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    report = subparsers.add_parser("report", help="Regenerate demo_report.md from summary.json.")
    report.add_argument("--save-dir", default="runs/edge_cloud/gradio_demo")
    return parser.parse_args()


def main() -> None:
    """Run preflight checks or generate a report."""
    args = parse_args()
    if args.command == "report":
        print(write_demo_report(args.save_dir))
        return
    result = run_preflight(args.edge_model, args.policy_path, args.cloud_url, args.cloud_timeout, args.require_cloud)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else render_preflight(result))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
