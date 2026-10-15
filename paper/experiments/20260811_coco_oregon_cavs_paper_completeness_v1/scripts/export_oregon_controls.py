#!/usr/bin/env python3
"""Export the two sklearn-containing Cloud-V3 controls to portable caches."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
OREGON = REPO / "paper/experiments/20260810_oregon_cloudv3_full_scheduler_afpd_paper_v1"
OUT = ROOT / "runs/preflight"
sys.path.insert(0, str(OREGON / "scripts"))
spec = importlib.util.spec_from_file_location("cavs_control_export", OREGON / "scripts/evaluate.py")
module = importlib.util.module_from_spec(spec)
sys.modules["cavs_control_export"] = module
assert spec.loader is not None
spec.loader.exec_module(module)
controls = module.load_pickle(OREGON / "runs/evaluation/controls/artifact.pkl")
scores = np.asarray(controls["joint_scores"]["final"]["Raw-ActionValue-OQ"], dtype=np.float64)[:, :, :3]
maps = {
    role: {str(image_id): str(action) for image_id, action in controls["action_maps"][role]["SharedAction-NoROI"].items()}
    for role in ("calibration", "final")
}
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / "oregon_raw_cavs_scores.npz", scores=scores)
temporary = OUT / f".oregon_noroi_maps.{os.getpid()}.tmp"
temporary.write_text(json.dumps(maps, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, OUT / "oregon_noroi_maps.json")
print(json.dumps({"scores_shape": list(scores.shape), "roles": {key: len(value) for key, value in maps.items()}}, indent=2))
