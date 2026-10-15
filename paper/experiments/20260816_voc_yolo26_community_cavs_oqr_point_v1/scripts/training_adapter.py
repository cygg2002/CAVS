"""Thin adapter over the frozen CAVS and ORIC/MORIC training implementations."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
SOURCE = (
    REPO
    / "paper/experiments/20260810_oregon_cloudv3_full_scheduler_afpd_paper_v1/scripts/training_adapter.py"
)


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("voc_point_frozen_training_source", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SOURCE = _load()
fit_model = _SOURCE.fit_model
predict_model = _SOURCE.predict_model
external_features = _SOURCE.external_features
fit_external = _SOURCE.fit_external
predict_external = _SOURCE.predict_external
