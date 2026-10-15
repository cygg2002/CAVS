#!/usr/bin/env python3
"""Record real-system feasibility without manufacturing measured-E2E results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from selector_core import validate_measured_e2e_claim


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
EDGE_CLIENT = Path("examples/YOLO26-Edge-Cloud/edge_client.py")
CLOUD_SERVER = Path("examples/YOLO26-Edge-Cloud/cloud_server.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_record(repo_root: Path, relative_path: Path) -> dict[str, Any]:
    path = repo_root / relative_path
    return {
        "path": relative_path.as_posix(),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def contract_rejection_reason() -> str:
    incomplete_claim = {
        "latency_source": "not_measured",
        "edge_host": "local-mac",
        "cloud_host": "fzu111",
    }
    try:
        validate_measured_e2e_claim(incomplete_claim)
    except ValueError as error:
        return str(error)
    raise RuntimeError("Incomplete system metadata unexpectedly passed the E2E contract")


def build_feasibility_report(repo_root: Path) -> dict[str, Any]:
    """Build a deterministic report from already established preflight facts."""
    edge_client = implementation_record(repo_root, EDGE_CLIENT)
    cloud_server = implementation_record(repo_root, CLOUD_SERVER)
    missing_code = [
        name
        for name, record in (
            ("edge_client", edge_client),
            ("cloud_server", cloud_server),
        )
        if not record["exists"]
    ]
    blockers = [
        "quiet_or_exclusive_cloud_gpu",
        "frozen_edge_and_cloud_checkpoint_sha256",
        "frozen_payload_replay_manifest",
        "raw_component_and_end_to_end_logs",
    ]
    if missing_code:
        blockers.append("existing_edge_cloud_implementation_paths")

    return {
        "schema_version": 1,
        "audit_date": "2026-07-25",
        "artifact_kind": "system_measurement_feasibility",
        "fact_scope": (
            "Established preflight facts only; this audit does not execute a "
            "latency, throughput, energy, or network benchmark."
        ),
        "topology": {
            "distinct_hosts_feasible": True,
            "edge": {
                "alias": "local-mac",
                "role": "edge",
                "processor": "Apple M2 Pro",
            },
            "cloud": {
                "alias": "fzu111",
                "role": "cloud",
                "accelerator": "NVIDIA GeForce RTX 4090",
                "availability": "shared_busy",
                "availability_is_transient": True,
            },
        },
        "network": {
            "distinct_host_path_feasible": True,
            "bandwidth_measured_for_this_experiment": False,
            "rtt_measured_for_this_experiment": False,
            "controlled_shaping": {
                "status": "privilege_required",
                "profiles_executed": False,
            },
        },
        "implementation": {
            "edge_client": edge_client,
            "cloud_server": cloud_server,
            "component_timing_contract": [
                "edge_detector_inference",
                "payload_encoding",
                "network_transfer_round_trip",
                "payload_decoding",
                "cloud_detector_inference",
                "end_to_end",
            ],
            "frozen_instrumented_replay_executed": False,
        },
        "measurement_inputs": {
            "edge_model_sha256": None,
            "cloud_model_sha256": None,
            "payload_replay_manifest_sha256": None,
            "raw_log_sha256": None,
            "sample_count": 0,
        },
        "claim_gate": {
            "status": "blocked",
            "measured_e2e_claim_allowed": False,
            "contract_function": "selector_core.validate_measured_e2e_claim",
            "contract_rejection": contract_rejection_reason(),
            "blocking_requirements": blockers,
        },
        "paper_claims": {
            "allowed": [
                "A distinct Apple M2 Pro edge and RTX 4090 cloud path is feasible.",
                "Reusable edge-client and cloud-server code is present.",
                "A frozen measured run remains pending.",
            ],
            "forbidden_until_gate_passes": [
                "Measured end-to-end latency for the new three-dataset controller.",
                "Measured throughput or energy for the new controller.",
                "Controlled-network conclusions for unexecuted shaping profiles.",
            ],
        },
        "security": {
            "credentials_stored": False,
            "remote_endpoint_address_stored": False,
        },
    }


def render_status_note(report: Mapping[str, Any]) -> str:
    edge = report["topology"]["edge"]
    cloud = report["topology"]["cloud"]
    edge_client = report["implementation"]["edge_client"]
    cloud_server = report["implementation"]["cloud_server"]
    rejection = report["claim_gate"]["contract_rejection"]
    return f"""# 真实系统测量状态

## 当前结论

真实异机链路在硬件和代码层面可行，但本实验**不能声称已实测 E2E**。
本文件只记录 2026-07-25 已核实的可行性事实，不包含延迟、吞吐或能耗测量值。

## 已核实条件

- 端侧为 `{edge["processor"]}`，云端为 `{cloud["accelerator"]}`，两者是不同主机。
- 云端主机别名为 `{cloud["alias"]}`；GPU 当前为共享且繁忙的瞬时状态，不适合作为论文级计时环境。
- 现有端侧客户端：`{edge_client["path"]}`（存在：`{str(edge_client["exists"]).lower()}`）。
- 现有云端服务端：`{cloud_server["path"]}`（存在：`{str(cloud_server["exists"]).lower()}`）。
- 真实网络链路可建立；1/5/10/20 Mbps 等受控限速需要系统权限，本次没有执行网络整形。
- 审计文件不保存登录凭据或远程地址。

## 为什么尚未进行论文级实测

必须先同时满足：

1. 获得空闲或独占的 RTX 4090 计时窗口，避免其他进程污染尾延迟；
2. 冻结端侧和云端模型权重，并记录各自 SHA-256；
3. 冻结 Original/Q90/ROI 的 payload 回放清单及其 SHA-256；
4. 对端侧检测、编码、传输往返、解码、云端检测和总 E2E 分阶段记录原始日志；
5. 对自然网络状态进行实测，或在取得系统权限后执行并记录受控带宽/RTT配置。

当前 measured-E2E 契约按预期拒绝该声明：

```text
{rejection}
```

## 下一次可执行测量

在云 GPU 空闲后，冻结权重与 payload 回放；先做 warm-up，再对同一回放序列重复计时，
保存逐样本原始日志、环境信息和文件哈希。只有
`selector_core.validate_measured_e2e_claim` 对完整元数据校验通过后，才允许在论文中使用
“实测 E2E”表述。在此之前，既有 `E2E@10Mbps` 仍只能称为估算值。
"""


def write_artifacts(
    report: Mapping[str, Any], output_root: Path
) -> tuple[Path, Path]:
    json_path = output_root / "runs/system/feasibility.json"
    note_path = output_root / "notes/system_measurement_status_zh.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    note_path.write_text(render_status_note(report), encoding="utf-8")
    return json_path, note_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    args = parser.parse_args()

    paths = write_artifacts(
        build_feasibility_report(args.repo_root.resolve()),
        args.output_root.resolve(),
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
