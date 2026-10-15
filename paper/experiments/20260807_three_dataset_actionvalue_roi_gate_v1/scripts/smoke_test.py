#!/usr/bin/env python3
"""Exercise the four-action tensor and scheduler on 32 frozen cache records."""

from __future__ import annotations

import argparse
import json

import numpy as np

import run_roi_gate as experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("coco", "ena24", "oregon"), required=True)
    arguments = parser.parse_args()
    full = experiment.load_role(arguments.dataset, "fit_oof")
    image_ids = list(map(int, full["image_ids"][:32]))
    data = dict(full)
    data["samples"] = full["samples"][:32]
    data["image_ids"] = image_ids
    data["sample_by_id"] = {
        int(sample["image_id"]): sample for sample in data["samples"]
    }
    categories = experiment.category_ids(arguments.dataset, full)
    records = experiment.records_for_role(arguments.dataset, data, categories)
    raw = np.zeros((len(records), 4), dtype=np.float32)
    tensors = experiment.LEGACY.budgetset._record_tensors(
        records, raw, experiment.model_protocol()
    )
    scores = {
        int(sample["image_id"]): {
            "original": float(index + 1),
            "q90": float(index + 2),
            "roi": float(index + 3),
        }
        for index, sample in enumerate(data["samples"])
    }
    scheduled = experiment.schedule(data, scores, "fixed_byte", 0.1)
    action_sets = sorted(
        {tuple(str(action["action_id"]) for action in record["actions"]) for record in records}
    )
    if not any("roi" in actions for actions in action_sets):
        raise RuntimeError("Smoke subset contains no canonical ROI action")
    if tuple(tensors["actions"].shape) != (32, 4, 18):
        raise RuntimeError(f"Unexpected action tensor shape: {tensors['actions'].shape}")
    capacity = 0.1 * sum(
        float(sample["payload_bytes"]["original"]) for sample in data["samples"]
    )
    if float(scheduled["total_cost_bytes"]) > capacity + 1e-6:
        raise RuntimeError("Smoke scheduler exceeded fixed-byte capacity")
    print(
        json.dumps(
            {
                "dataset": arguments.dataset,
                "records": len(records),
                "action_sets": action_sets,
                "tensor_shape": list(tensors["actions"].shape),
                "selected_images": int(scheduled["selected_images"]),
                "used_bytes": float(scheduled["total_cost_bytes"]),
                "capacity_bytes": capacity,
                "passed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
