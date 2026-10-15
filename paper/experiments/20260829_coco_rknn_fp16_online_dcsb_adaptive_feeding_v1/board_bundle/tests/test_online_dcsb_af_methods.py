from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from online_methods import build_method_runtime, score_current_frame  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OnlineDcsbAdaptiveFeedingMethodTests(unittest.TestCase):
    def _policy(self, root: Path) -> Path:
        model = root / "adaptive_feeding.npz"
        np.savez(
            model,
            coefficient=np.ones(330, dtype=np.float64),
            intercept=np.asarray([0.0], dtype=np.float64),
        )
        reference = root / "calibration_references.npz"
        np.savez(
            reference,
            dcsb_margin=np.asarray([-1.0, 0.5]),
            adaptive_feeding_margin=np.asarray([-1.0, 1.0]),
        )
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "policy_type": "dcsb_adaptive_feeding_original_only_rknn",
                    "rknn_model_sha256": "1" * 64,
                    "category_ids": list(range(1, 81)),
                    "top_k": 50,
                    "adaptive_feeding": {"file": model.name, "sha256": sha256(model)},
                    "calibration_references": {
                        "file": reference.name,
                        "sha256": sha256(reference),
                    },
                    "dcsb": {
                        "confidence_threshold": 0.2,
                        "detector_confidence": 0.5,
                        "count_threshold": 2,
                        "area_threshold": 0.2,
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_builds_all_four_original_only_runtime_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._policy(Path(directory))
            for method in (
                "dcsb_native",
                "dcsb_budget",
                "adaptive_feeding_native",
                "adaptive_feeding_budget",
            ):
                runtime = build_method_runtime(
                    method,
                    policy_path=manifest,
                    expected_rknn_sha256="1" * 64,
                )
                self.assertEqual(runtime.action_ids, ("edge", "original"))
                self.assertEqual(runtime.method, method)

    def test_budget_variant_scores_the_current_frame_without_future_state(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._policy(Path(directory))
            runtime = build_method_runtime(
                "dcsb_budget",
                policy_path=manifest,
                expected_rknn_sha256="1" * 64,
            )
            scores = score_current_frame(
                runtime,
                image_id=1,
                edge_boxes=[
                    {"xyxy": [0, 0, 50, 50], "conf": 0.8, "cls": 0},
                    {"xyxy": [0, 0, 10, 10], "conf": 0.3, "cls": 0},
                ],
                width=100,
                height=100,
                roi_xyxy=None,
            )
            self.assertEqual(set(scores), {"edge", "original"})
            self.assertGreater(scores["original"], 0.0)


if __name__ == "__main__":
    unittest.main()
