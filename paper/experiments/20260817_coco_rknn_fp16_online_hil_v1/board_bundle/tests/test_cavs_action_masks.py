from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cavs_numpy import load_policy, predict_scores


class CavsActionMaskTest(unittest.TestCase):
    def _masked_policy(self, actions, method):
        source = ROOT / "policies" / "cavs_oqr"
        temporary = tempfile.TemporaryDirectory()
        target = Path(temporary.name)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        manifest["policy_type"] = "cavs_action_value"
        manifest["method"] = method
        manifest["action_ids"] = list(actions)
        manifest["profile"] = "none"
        manifest["fusion"] = "none"
        for row in manifest["seeds"]:
            shutil.copy2(source / row["file"], target / row["file"])
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return temporary, target / "manifest.json"

    def test_oq_policy_returns_only_trained_actions(self) -> None:
        temporary, path = self._masked_policy(
            ("edge", "original", "q90"), "CAVS-OQ"
        )
        self.addCleanup(temporary.cleanup)
        policy = load_policy(path)
        scores = predict_scores(
            policy, [], width=640, height=480, roi_xyxy=[0, 0, 100, 100]
        )
        self.assertEqual(tuple(scores), ("edge", "original", "q90"))

    def test_noncanonical_action_subset_is_rejected(self) -> None:
        temporary, path = self._masked_policy(
            ("edge", "q90", "original"), "CAVS-OQ"
        )
        self.addCleanup(temporary.cleanup)
        with self.assertRaises(ValueError):
            load_policy(path)


if __name__ == "__main__":
    unittest.main()
