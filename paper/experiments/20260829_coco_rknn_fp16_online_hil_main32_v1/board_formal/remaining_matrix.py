#!/usr/bin/env python3
"""Validate the frozen controller-ablation and network-sensitivity rows."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from main_matrix import MATRIX_SCHEMA, PROTOCOL_ID, file_sha256, gate_run


FAMILY_COUNTS = {"controller_ablation": 6, "network_sensitivity": 8}
REMAINING_INDICES = tuple(range(33, 47))
NETWORK_PROFILES = {
    "native": {
        "mechanism": "none",
        "added_round_trip_delay_ms": 0,
        "uplink_mbps": None,
        "packet_loss_percent": 0.0,
    },
    "wan_20ms_20mbps": {
        "mechanism": "linux_tc_netem_tbf",
        "added_round_trip_delay_ms": 20,
        "uplink_mbps": 20,
        "packet_loss_percent": 0.0,
    },
    "wan_50ms_5mbps": {
        "mechanism": "linux_tc_netem_tbf",
        "added_round_trip_delay_ms": 50,
        "uplink_mbps": 5,
        "packet_loss_percent": 0.0,
    },
}


def _object(path: str | Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {field}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return dict(value)


def _asset(bundle: Path, relative: object, digest: object, *, field: str) -> None:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{field} path is invalid")
    root = bundle.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes board bundle") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(digest).lower():
        raise ValueError(f"{field} SHA-256 mismatch: {relative}")


def _profile(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("network profile must be an object")
    return {
        "mechanism": value.get("mechanism"),
        "added_round_trip_delay_ms": value.get("added_round_trip_delay_ms"),
        "uplink_mbps": value.get("uplink_mbps"),
        "packet_loss_percent": value.get("packet_loss_percent"),
    }


def validate_remaining_matrix(
    matrix_path: str | Path,
    bundle_root: str | Path,
    protocol_path: str | Path,
) -> list[dict[str, Any]]:
    """Return the immutable rows 33..46 after checking their complete design."""
    matrix = _object(matrix_path, field="run matrix")
    protocol = _object(protocol_path, field="protocol")
    bundle = Path(bundle_root)
    if matrix.get("schema") != MATRIX_SCHEMA:
        raise ValueError("unsupported run-matrix schema")
    if matrix.get("protocol_id") != PROTOCOL_ID or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("remaining matrix requires the final-v2 protocol")
    if str(matrix.get("protocol_sha256", "")) != file_sha256(protocol_path):
        raise ValueError("protocol SHA-256 differs from the run matrix")
    if int(matrix.get("expected_images", -1)) != 5000:
        raise ValueError("remaining matrix requires 5,000 images")
    if matrix.get("final_used_for_matrix_selection") is not False:
        raise ValueError("final data may not select the run matrix")
    counts = matrix.get("family_counts")
    if not isinstance(counts, Mapping) or any(int(counts.get(key, -1)) != value for key, value in FAMILY_COUNTS.items()):
        raise ValueError("remaining family counts differ from the paper specification")
    profiles = matrix.get("network_profiles")
    if not isinstance(profiles, Mapping) or {
        name: _profile(profiles.get(name)) for name in NETWORK_PROFILES
    } != NETWORK_PROFILES:
        raise ValueError("network profiles differ from the frozen specification")

    raw_runs = matrix.get("runs")
    if not isinstance(raw_runs, list):
        raise ValueError("run matrix has no run list")
    rows = [
        dict(row)
        for row in raw_runs
        if isinstance(row, Mapping) and row.get("family") in FAMILY_COUNTS
    ]
    rows.sort(key=lambda row: int(row.get("run_index", -1)))
    indices = [int(row.get("run_index", -1)) for row in rows]
    if indices != list(REMAINING_INDICES):
        raise ValueError("remaining run indices must be exactly 33..46")
    if Counter(str(row.get("family")) for row in rows) != Counter(FAMILY_COUNTS):
        raise ValueError("remaining family counts are invalid")
    keys = [str(row.get("run_key", "")) for row in rows]
    if not all(keys) or len(keys) != len(set(keys)):
        raise ValueError("remaining run keys must be non-empty and unique")

    for row in rows:
        if (
            int(row.get("expected_images", -1)) != 5000
            or row.get("zero_lookahead") is not True
            or row.get("global_plan") is not False
            or int(row.get("buffer_frames", -1)) != 0
            or row.get("random_seed") is not None
        ):
            raise ValueError(f"non-causal remaining row: {row.get('run_key')}")
        numerator = int(row.get("budget_numerator", 0))
        denominator = int(row.get("budget_denominator", 0))
        if denominator <= 0 or numerator <= 0 or numerator > denominator:
            raise ValueError(f"invalid budget: {row.get('run_key')}")
        _asset(bundle, row.get("policy"), row.get("policy_sha256"), field="policy")
        _asset(bundle, row.get("control"), row.get("control_sha256"), field="control")

    controller = [row for row in rows if row["family"] == "controller_ablation"]
    expected_controller = Counter(
        (method, budget)
        for method in ("cavs_oq", "cavs_oqr")
        for budget in (10, 25, 40)
    )
    if Counter((row.get("method"), int(row.get("budget_percent", -1))) for row in controller) != expected_controller:
        raise ValueError("controller-ablation methods or budgets changed")
    if any(row.get("controller_variant") != "static_lambda" or row.get("network_profile") != "native" for row in controller):
        raise ValueError("controller-ablation semantics changed")

    network = [row for row in rows if row["family"] == "network_sensitivity"]
    expected_network = Counter(
        (method, profile)
        for method in ("cavs_oq", "cavs_oqr", "oric", "moric")
        for profile in ("wan_20ms_20mbps", "wan_50ms_5mbps")
    )
    if Counter((row.get("method"), row.get("network_profile")) for row in network) != expected_network:
        raise ValueError("network-sensitivity methods or profiles changed")
    if any(
        row.get("controller_variant") != "registered"
        or int(row.get("budget_percent", -1)) != 25
        for row in network
    ):
        raise ValueError("network-sensitivity controller or budget changed")
    return rows


def network_commands(
    interface: str,
    profile_name: str,
    profile: Mapping[str, object],
) -> list[list[str]]:
    """Build the exact root qdisc commands for one frozen uplink profile."""
    if not interface or any(character.isspace() for character in interface):
        raise ValueError("network interface is invalid")
    if profile_name == "native":
        if profile and _profile(profile) != NETWORK_PROFILES["native"]:
            raise ValueError("native network profile changed")
        return []
    normalized = _profile(profile)
    if normalized.get("mechanism") != "linux_tc_netem_tbf":
        raise ValueError("network shaping mechanism must be linux_tc_netem_tbf")
    rate = normalized.get("uplink_mbps")
    delay = normalized.get("added_round_trip_delay_ms")
    loss = normalized.get("packet_loss_percent")
    if not isinstance(rate, int) or rate <= 0 or not isinstance(delay, int) or delay < 0:
        raise ValueError("network rate and delay must be non-negative integers")
    if not isinstance(loss, (int, float)) or float(loss) < 0:
        raise ValueError("network loss is invalid")
    loss_text = f"{float(loss):g}%"
    return [
        [
            "tc", "qdisc", "replace", "dev", interface, "root", "handle", "1:",
            "tbf", "rate", f"{rate}mbit", "burst", "16kb", "latency", "2000ms",
        ],
        [
            "tc", "qdisc", "replace", "dev", interface, "parent", "1:1",
            "handle", "10:", "netem", "delay", f"{delay}ms", "loss", loss_text,
            "limit", "10000",
        ],
    ]


def gate_remaining_run(
    frames_path: str | Path,
    contract_path: str | Path,
    matrix_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the same causal/byte/box hard gate used by the main 32 runs."""
    return gate_run(frames_path, contract_path, matrix_row)
