"""Reuse the frozen profile/scale cache builder for the fit role only."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any, Mapping


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
SOURCE = (
    REPO
    / "paper/experiments/20260728_oregon_cloudv2_roi_profile_scale_scan/scripts/build_cache.py"
)


def _validate_fit_roles(roles: Mapping[str, int]) -> tuple[str, ...]:
    names = tuple(map(str, roles))
    if "final" in names:
        raise ValueError("final must remain sealed")
    if names != ("fit",):
        raise ValueError(f"fit cache roles must be ('fit',), got {names}")
    if int(roles["fit"]) <= 0:
        raise ValueError("fit population must be positive")
    return names


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location(
        "oregon_profile_scale_fit_builder", SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen cache builder: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_roles = _validate_fit_roles
    return module


BUILDER = _load_builder()
file_sha256 = BUILDER.file_sha256


def fit_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["roles"] = {"fit": int(config["roles"]["fit"])}
    return value


def prepare_fit_geometry(config: dict[str, Any], profile: str) -> dict[str, Any]:
    return BUILDER.prepare_geometry(fit_config(config), "fit", profile)


def infer_fit_candidate(
    config: dict[str, Any], profile: str, imgsz: int
) -> dict[str, Any]:
    return BUILDER.infer_candidate(fit_config(config), "fit", profile, int(imgsz))
