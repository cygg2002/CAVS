#!/usr/bin/env python3
"""Run the existing inspected HTTP benchmark with the retrained policy."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
SOURCE = REPO / "paper/experiments/20260815_coco5000_cavs_oqr_paper_parity_e2e_v1/scripts/run_paper_parity.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_module("perframe_live_base", SOURCE)
PROTOCOL = json.loads((ROOT / "configs/live_protocol.json").read_text(encoding="utf-8"))
for key in ("policy",):
    value = Path(PROTOCOL["models"][key])
    if not value.is_absolute():
        PROTOCOL["models"][key] = str((ROOT / value).resolve())
for key in ("samples", "selections", "results", "theoretical"):
    value = Path(PROTOCOL["paper_artifacts"][key])
    if not value.is_absolute():
        PROTOCOL["paper_artifacts"][key] = str((ROOT / value).resolve())
PROTOCOL["theoretical_current_maur"] = json.loads(
    Path(PROTOCOL["paper_artifacts"]["theoretical"]).read_text(encoding="utf-8")
)
RUNNER.ROOT = ROOT
RUNNER.PROTOCOL = PROTOCOL


if __name__ == "__main__":
    RUNNER.main()

