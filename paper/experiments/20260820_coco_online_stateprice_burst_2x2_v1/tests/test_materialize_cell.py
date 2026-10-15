from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from materialize_cell import load_cell_selections


class MaterializeCellTest(unittest.TestCase):
    def test_loads_only_the_requested_frozen_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selections.json"
            path.write_text(json.dumps({"a": {"1": "edge"}, "b": {"1": "q90"}}))
            self.assertEqual(load_cell_selections(path, "b"), {"1": "q90"})

    def test_rejects_an_unknown_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selections.json"
            path.write_text(json.dumps({"a": {"1": "edge"}}))
            with self.assertRaises(ValueError):
                load_cell_selections(path, "missing")


if __name__ == "__main__":
    unittest.main()
