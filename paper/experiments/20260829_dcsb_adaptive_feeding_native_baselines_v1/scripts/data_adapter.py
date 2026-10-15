"""Read-only adapters for the three frozen paper data protocols."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
UNIFIED_ROOT = (
    REPO / "paper/experiments/20260812_coco_oregon_cavs_oqr_unified_paper_v1"
)
VOC_ROOT = (
    REPO / "paper/experiments/20260825_voc_torchvision_crossarchitecture_cavs_oqr_paper_v1"
)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FrozenDataset:
    """Expose only samples, Edge/Original predictions, GT, and official AP."""

    def __init__(self, dataset: str) -> None:
        if dataset not in {"coco", "oregon", "voc"}:
            raise ValueError(dataset)
        self.dataset = str(dataset)
        self._cache: dict[str, dict[str, Any]] = {}
        if dataset in {"coco", "oregon"}:
            self.source = load_module(
                f"dcsb_af_unified_{dataset}", UNIFIED_ROOT / "scripts/run_experiment.py"
            )
            self.adapter = self.source.DatasetAdapter(dataset)
        else:
            self.source = load_module(
                "dcsb_af_voc_point", VOC_ROOT / "scripts/evaluate_point.py"
            )
            self.adapter = None

    def role(self, role: str) -> dict[str, Any]:
        if role not in {"fit", "calibration", "final"}:
            raise ValueError(role)
        if role not in self._cache:
            if self.dataset in {"coco", "oregon"}:
                data = self.adapter.load_role(role)
                gt_path = self.adapter.gt_path(role)
                manifest_path = self.adapter.common_manifest_path(role)
            else:
                data = self.source._role(role)
                gt_path = Path(data["gt"])
                manifest_path = VOC_ROOT / f"runs/common/{role}/source_manifest.json"
            gt = read_json(gt_path)
            category_ids = [int(row["id"]) for row in gt["categories"]]
            self._cache[role] = {
                "samples": list(data["samples"]),
                "image_ids": [int(row["image_id"]) for row in data["samples"]],
                "actions": {
                    "edge": data["actions"]["edge"],
                    "original": data["actions"]["original"],
                },
                "gt": gt,
                "gt_path": Path(gt_path),
                "manifest_path": Path(manifest_path),
                "manifest": read_json(manifest_path),
                "category_ids": category_ids,
                "raw": data,
            }
        return self._cache[role]

    def evaluate(self, role: str, selection: Mapping[int, str]) -> dict[str, float]:
        data = self.role(role)
        if set(selection.values()) - {"original"}:
            raise ValueError("DCSB/Adaptive Feeding may only upload Original images")
        if self.dataset in {"coco", "oregon"}:
            predictions = self.adapter.mixed_predictions(role, selection)
            metric = self.adapter.evaluator(role).evaluate(predictions)
        else:
            metric = self.source._evaluate(data["raw"], selection)
        return {
            "map50_95": float(metric["map50_95"]),
            "map50": float(metric["map50"]),
            "map75": float(metric["map75"]),
            **(
                {
                    "voc07_map50": float(metric["voc07_map50"]),
                    "voc12_map50": float(metric["voc12_map50"]),
                }
                if "voc07_map50" in metric
                else {}
            ),
        }

    def source_paths(self) -> list[Path]:
        paths = []
        for role in ("fit", "calibration", "final"):
            data = self.role(role)
            paths.extend((data["manifest_path"], data["gt_path"]))
        return paths

