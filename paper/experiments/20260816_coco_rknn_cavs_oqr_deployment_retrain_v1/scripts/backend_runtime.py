"""Pure runtime contracts for one deployment-aligned RKNN backend."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import contracts


ACTION_SPACE = ["edge", "original", "q90", "current@640"]


def backend_runs(root: str | Path, backend: str) -> Path:
    return Path(root) / "runs" / "backends" / contracts.validate_backend(backend)


def completed_common_roles(
    root: str | Path, backend: str
) -> dict[str, dict[str, Any]] | None:
    base = backend_runs(root, backend) / "common/coco"
    reports: dict[str, dict[str, Any]] = {}
    for role in contracts.ROLES:
        path = base / role / "completion.json"
        if not path.is_file():
            return None
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("complete"):
            return None
        reports[role] = report
    return reports


def validate_reused_action_sources(
    manifest: Mapping[str, Any], action_ids: Sequence[str]
) -> dict[str, str]:
    recorded = dict(manifest.get("source_paths", {}))
    output: dict[str, str] = {}
    for action_id in map(str, action_ids):
        path = Path(str(manifest["action_sources"][action_id])).resolve()
        expected = recorded.get(str(path))
        if expected is None:
            raise RuntimeError(f"missing recorded source hash: {action_id}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(f"reused action source hash changed: {action_id}")
        output[action_id] = actual
    return output


def adapt_protocol(protocol: Mapping[str, Any], backend: str) -> dict[str, Any]:
    value = contracts.validate_backend(backend)
    adapted = deepcopy(dict(protocol))
    adapted["backends"] = [value]
    adapted["model_seeds"] = list(map(int, protocol["seeds"]))
    adapted["roi_actions"] = ["current@640"]
    adapted["action_space"] = list(ACTION_SPACE)
    adapted["fusion"] = "MAUR"
    adapted["cloud_model"] = {
        **dict(protocol["cloud_model"]),
        "sha256": str(protocol["cloud_model_sha256"]),
    }
    adapted["cpu"] = {
        **dict(protocol["cpu"]),
        "maximum_workers": int(protocol["cpu"]["training_workers_per_backend"]),
    }
    return adapted


def training_jobs(protocol: Mapping[str, Any], backend: str) -> list[dict[str, Any]]:
    value = contracts.validate_backend(backend)
    method = f"CAVS-OQR-DeploymentAligned-{value}"
    return [
        {
            **job,
            "method": method,
            "trainer": "current_action",
            "action_space": list(ACTION_SPACE),
        }
        for job in contracts.training_jobs(protocol)
        if job["backend"] == value
    ]


def _plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def collapse_static_scores(values: Any, expected_budgets: int) -> Any:
    ndim = getattr(values, "ndim", None)
    if ndim == 2:
        return values
    plain = _plain(values)
    if ndim is None and plain and plain[0] and not isinstance(plain[0][0], Sequence):
        return values
    if len(plain) != int(expected_budgets):
        raise ValueError(f"unexpected score shape: {getattr(values, 'shape', None)}")
    first = plain[0]
    if any(candidate != first for candidate in plain[1:]):
        raise ValueError("budget-conditioned scores require budget-specific evaluation")
    return values[0]


def score_map(
    records: Sequence[Mapping[str, Any]], values: Any
) -> dict[int, dict[str, float]]:
    rows = _plain(values)
    if len(rows) != len(records):
        raise ValueError("score/image population changed")
    output: dict[int, dict[str, float]] = {}
    for record, row in zip(records, rows):
        actions = list(record["actions"])
        if len(row) < len(actions):
            raise ValueError(f"score/action population changed: {record['image_id']}")
        output[int(record["image_id"])] = {
            str(action["action_id"]): float(row[index])
            for index, action in enumerate(actions)
            if str(action["action_id"]) != "edge"
        }
    return output
