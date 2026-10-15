import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("afpd_same_run_core", ROOT / "scripts/core.py")
CORE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CORE)
PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))


class ProtocolTests(unittest.TestCase):
    def test_exact_six_comparison_matrix(self):
        CORE.validate_protocol(PROTOCOL)
        rows = CORE.comparison_specs(PROTOCOL)
        self.assertEqual(6, len(rows))
        self.assertEqual(6, len({row["key"] for row in rows}))
        self.assertEqual(
            {"Current-OQ", "Current-OQR"},
            {row["comparator"] for row in rows},
        )
        self.assertEqual({0.1, 0.25, 0.4}, {row["budget"] for row in rows})

    def test_protocol_rejects_confirmatory_or_training_flags(self):
        changed = dict(PROTOCOL, confirmatory=True)
        with self.assertRaises(ValueError):
            CORE.validate_protocol(changed)
        changed = dict(PROTOCOL, training_allowed=True)
        with self.assertRaises(ValueError):
            CORE.validate_protocol(changed)

    def test_parent_ready_requires_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "completion.json"
            self.assertFalse(CORE.parent_ready(marker))
            marker.write_text('{"complete": false}', encoding="utf-8")
            self.assertFalse(CORE.parent_ready(marker))
            marker.write_text('{"complete": true}', encoding="utf-8")
            self.assertTrue(CORE.parent_ready(marker))

    def test_selection_validation_requires_subset_ids_and_known_actions(self):
        expected = [1, 2, 3]
        CORE.validate_selection({1: "q90", 3: "roi"}, expected)
        with self.assertRaises(ValueError):
            CORE.validate_selection({1: "edge", 4: "q90"}, expected)
        with self.assertRaises(ValueError):
            CORE.validate_selection({1: "edge", 2: "bad", 3: "roi"}, expected)


if __name__ == "__main__":
    unittest.main()
