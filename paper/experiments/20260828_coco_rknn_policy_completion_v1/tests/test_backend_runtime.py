from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "rknn_backend_runtime", SCRIPTS / "backend_runtime.py"
)
assert SPEC and SPEC.loader
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


class BackendRuntimeTests(unittest.TestCase):
    def test_adapts_one_isolated_oq_backend(self) -> None:
        adapted = RUNTIME.adapt_protocol(PROTOCOL, "fp16")

        self.assertEqual(adapted["backends"], ["fp16"])
        self.assertEqual(adapted["model_seeds"], [20260809, 20260810, 20260811])
        self.assertEqual(adapted["roi_actions"], [])
        self.assertEqual(
            adapted["action_space"],
            ["edge", "original", "q90"],
        )
        self.assertEqual(adapted["fusion"], "none")
        self.assertEqual(adapted["cloud_model"]["sha256"], PROTOCOL["cloud_model_sha256"])
        self.assertEqual(adapted["model"]["max_actions"], 15)

    def test_builds_exactly_18_fresh_jobs_per_backend(self) -> None:
        jobs = RUNTIME.training_jobs(PROTOCOL, "fp16")

        self.assertEqual(len(jobs), 18)
        self.assertEqual({job["backend"] for job in jobs}, {"fp16"})
        self.assertEqual({tuple(job["action_space"]) for job in jobs}, {("edge", "original", "q90")})
        self.assertEqual({job["trainer"] for job in jobs}, {"current_action"})
        self.assertEqual(
            {job["fold"] for job in jobs},
            {0, 1, 2, 3, 4, "full"},
        )

    def test_backend_run_roots_never_overlap(self) -> None:
        fp16 = RUNTIME.backend_runs(ROOT, "fp16")
        int8 = RUNTIME.backend_runs(ROOT, "int8-heads")

        self.assertNotEqual(fp16, int8)
        self.assertEqual(fp16, ROOT / "runs/backends/fp16")
        self.assertEqual(int8, ROOT / "runs/backends/int8-heads")

    def test_collapses_only_redundant_static_budget_scores(self) -> None:
        static = [[0.0, 1.0], [2.0, 3.0]]
        repeated = [[list(row) for row in static] for _ in range(3)]

        self.assertEqual(RUNTIME.collapse_static_scores(repeated, 3), static)
        repeated[2][0][0] += 1.0
        with self.assertRaisesRegex(ValueError, "budget-conditioned"):
            RUNTIME.collapse_static_scores(repeated, 3)

    def test_score_map_excludes_edge_and_preserves_action_alignment(self) -> None:
        records = [
            {
                "image_id": 7,
                "actions": [
                    {"action_id": "edge"},
                    {"action_id": "original"},
                    {"action_id": "q90"},
                ],
            }
        ]

        result = RUNTIME.score_map(records, [[0.0, 0.7, 0.5]])

        self.assertEqual(result, {7: {"original": 0.7, "q90": 0.5}})

    def test_common_completion_requires_all_three_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for role in ("fit", "calibration"):
                path = root / f"runs/backends/fp16/common/coco/{role}/completion.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"complete": True, "role": role}), encoding="utf-8")

            self.assertIsNone(RUNTIME.completed_common_roles(root, "fp16"))
            path = root / "runs/backends/fp16/common/coco/final/completion.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"complete": True, "role": "final"}), encoding="utf-8")

            reports = RUNTIME.completed_common_roles(root, "fp16")
            self.assertEqual(set(reports or {}), {"fit", "calibration", "final"})

    def test_reused_action_sources_must_match_recorded_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.json"
            path.write_text("[]\n", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {
                "action_sources": {"original": str(path)},
                "source_paths": {str(path.resolve()): digest},
            }

            self.assertEqual(
                RUNTIME.validate_reused_action_sources(manifest, ("original",)),
                {"original": digest},
            )
            path.write_text("[{}]\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash changed"):
                RUNTIME.validate_reused_action_sources(manifest, ("original",))


if __name__ == "__main__":
    unittest.main()
