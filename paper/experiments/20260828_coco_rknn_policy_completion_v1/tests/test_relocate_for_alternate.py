from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from relocate_for_alternate import relocate_experiment


OLD_ROOT = "/home/111_wcy/work/yolo26/ultralytics-main"


class RelocateForAlternateTest(unittest.TestCase):
    def test_relocates_paths_and_reuses_frozen_common_rewards(self) -> None:
        with TemporaryDirectory() as directory:
            mirror = Path(directory) / "ultralytics-main"
            experiment = mirror / "paper/experiments/policy_completion"
            source_backend = mirror / "paper/experiments/source_backend/runs/backends/fp16"
            source_current = mirror / "paper/experiments/source_current"
            source_training = mirror / "paper/experiments/source_training"
            folds = mirror / "paper/experiments/folds.json"
            cloud = mirror / "yolo26x.pt"

            (experiment / "configs").mkdir(parents=True)
            (source_backend / "common/coco/fit").mkdir(parents=True)
            (source_backend / "rewards/fit").mkdir(parents=True)
            (source_current / "runs/common/coco/fit").mkdir(parents=True)
            source_training.mkdir(parents=True)
            folds.parent.mkdir(parents=True, exist_ok=True)
            folds.write_text("{}\n", encoding="utf-8")
            cloud.write_bytes(b"model")

            protocol = {
                "canonical_root": f"{OLD_ROOT}/paper/experiments/canonical",
                "source_current_experiment": f"{OLD_ROOT}/paper/experiments/source_current",
                "source_training_experiment": f"{OLD_ROOT}/paper/experiments/source_training",
                "source_backend_root": f"{OLD_ROOT}/paper/experiments/source_backend/runs/backends/fp16",
                "frozen_folds": f"{OLD_ROOT}/paper/experiments/folds.json",
                "cloud_model": {"path": f"{OLD_ROOT}/yolo26x.pt"},
            }
            (experiment / "configs/protocol.json").write_text(
                json.dumps(protocol), encoding="utf-8"
            )
            source_manifest = {
                "action_sources": {
                    "original": f"{OLD_ROOT}/paper/experiments/original.json"
                },
                "source_paths": {
                    f"{OLD_ROOT}/paper/experiments/original.json": "a" * 64
                },
            }
            for path in (
                source_backend / "common/coco/fit/source_manifest.json",
                source_current / "runs/common/coco/fit/source_manifest.json",
            ):
                path.write_text(json.dumps(source_manifest), encoding="utf-8")
            (source_backend / "rewards/fit/reward_map.pkl").write_bytes(b"reward")

            result = relocate_experiment(
                mirror_root=mirror,
                experiment=experiment,
                source_backend=source_backend,
                source_current=source_current,
                source_training=source_training,
                folds=folds,
                cloud_model=cloud,
                old_root=OLD_ROOT,
            )

            copied = experiment / "runs/backends/fp16"
            self.assertTrue((copied / "common/coco/fit/source_manifest.json").is_file())
            self.assertTrue((copied / "rewards/fit/reward_map.pkl").is_file())
            updated = json.loads((experiment / "configs/protocol.json").read_text())
            self.assertEqual(updated["source_backend_root"], str(copied.resolve()))
            self.assertEqual(updated["source_current_experiment"], str(source_current.resolve()))
            self.assertEqual(updated["source_training_experiment"], str(source_training.resolve()))
            self.assertEqual(updated["frozen_folds"], str(folds.resolve()))
            self.assertEqual(updated["cloud_model"]["path"], str(cloud.resolve()))
            relocated_manifest = (
                copied / "common/coco/fit/source_manifest.json"
            ).read_text(encoding="utf-8")
            self.assertNotIn(OLD_ROOT, relocated_manifest)
            self.assertIn(str(mirror.resolve()), relocated_manifest)
            self.assertGreaterEqual(result["rewritten_json_files"], 2)


if __name__ == "__main__":
    unittest.main()
