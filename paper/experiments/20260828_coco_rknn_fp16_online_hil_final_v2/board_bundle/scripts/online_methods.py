"""Current-frame score adapters for CAVS and scalar online baselines."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from baseline_numpy import confidence_score, external_score, load_policy as load_external_policy
from cavs_numpy import load_policy as load_cavs_policy, predict_scores


CAVS_METHODS = {
    "cavs_o": ("edge", "original"),
    "cavs_oq": ("edge", "original", "q90"),
    "cavs_oqr": ("edge", "original", "q90", "current@640"),
}
EXTERNAL_METHODS = {
    "oric": "oric",
    "moric": "moric",
}
REFERENCE_METHODS = {"edge_only", "always_original"}


@dataclass(frozen=True)
class MethodRuntime:
    method: str
    action_ids: tuple[str, ...]
    policy: Mapping[str, Any] | None = None
    external_model: str | None = None
    random_seed: int | None = None


def build_method_runtime(
    method: str,
    *,
    policy_path: str | Path | None = None,
    random_seed: int = 20260720,
    expected_rknn_sha256: str | None = None,
) -> MethodRuntime:
    name = str(method).lower()
    if name in CAVS_METHODS:
        if policy_path is None:
            raise ValueError(f"{name} requires a CAVS policy manifest")
        policy = load_cavs_policy(policy_path)
        actions = tuple(map(str, policy["action_ids"]))
        if actions != CAVS_METHODS[name]:
            raise ValueError(f"CAVS policy action mismatch: {name}/{actions}")
        if expected_rknn_sha256 and policy.get("rknn_model_sha256") != expected_rknn_sha256:
            raise ValueError("CAVS policy RKNN model mismatch")
        return MethodRuntime(name, actions, policy=policy)
    if name in EXTERNAL_METHODS:
        if policy_path is None:
            raise ValueError(f"{name} requires an external policy manifest")
        policy = load_external_policy(policy_path)
        if expected_rknn_sha256 and policy.get("rknn_model_sha256") != expected_rknn_sha256:
            raise ValueError("external policy RKNN model mismatch")
        return MethodRuntime(
            name,
            ("edge", "original"),
            policy=policy,
            external_model=EXTERNAL_METHODS[name],
        )
    if name == "confidence":
        return MethodRuntime(name, ("edge", "original"))
    if name in REFERENCE_METHODS:
        return MethodRuntime(name, ("edge", "original"))
    if name == "random":
        return MethodRuntime(name, ("edge", "original"), random_seed=int(random_seed))
    raise ValueError(f"unsupported online method: {method}")


def _random_score(seed: int, image_id: int) -> float:
    digest = hashlib.sha256(f"{int(seed)}:{int(image_id)}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def score_current_frame(
    runtime: MethodRuntime,
    image_id: int,
    edge_boxes: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    roi_xyxy: Sequence[int] | None,
) -> dict[str, float]:
    if runtime.method in CAVS_METHODS:
        assert runtime.policy is not None
        values = predict_scores(
            runtime.policy,
            edge_boxes,
            width=int(width),
            height=int(height),
            roi_xyxy=roi_xyxy,
        )
        return {action: float(values[action]) for action in runtime.action_ids if action in values}
    if runtime.method == "edge_only":
        score = -1_000_000.0
    elif runtime.method == "always_original":
        score = 1_000_000.0
    elif runtime.method == "confidence":
        score = confidence_score(edge_boxes, threshold=0.25)
    elif runtime.method == "random":
        assert runtime.random_seed is not None
        score = _random_score(runtime.random_seed, int(image_id))
    elif runtime.method in EXTERNAL_METHODS:
        assert runtime.policy is not None and runtime.external_model is not None
        score = external_score(
            runtime.policy,
            runtime.external_model,
            edge_boxes,
            width=int(width),
            height=int(height),
        )
    else:
        raise RuntimeError(f"unhandled method: {runtime.method}")
    return {"edge": 0.0, "original": float(score)}
