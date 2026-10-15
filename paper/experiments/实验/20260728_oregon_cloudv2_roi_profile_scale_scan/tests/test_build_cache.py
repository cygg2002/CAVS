from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/build_cache.py"


def load_cache():
    spec = importlib.util.spec_from_file_location("build_cache", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildCacheTest(unittest.TestCase):
    def test_completion_rejects_corrupted_prediction_file(self):
        cache = load_cache()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "roi_predictions_mapped.json"
            summary = root / "summary.json"
            prediction.write_text("[]")
            summary.write_text("{}")
            completion = {
                "signature": "expected",
                "summary_sha256": cache.file_sha256(summary),
                "prediction_sha256": cache.file_sha256(prediction),
            }
            (root / "completion.json").write_text(json.dumps(completion))
            self.assertTrue(cache.completion_matches(root, "expected"))
            prediction.write_text("[{\"corrupted\": true}]")
            self.assertFalse(cache.completion_matches(root, "expected"))


if __name__ == "__main__":
    unittest.main()
