from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ranker_core import validate_protocol  # noqa: E402


class ProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(
            (ROOT / "configs/protocol.json").read_text(encoding="utf-8")
        )

    def test_frozen_protocol_is_valid(self) -> None:
        validate_protocol(self.protocol)
        self.assertEqual(self.protocol["budgets"], [0.1, 0.25, 0.4])
        self.assertEqual(len(self.protocol["seeds"]), 3)

    def test_protocol_rejects_final_selection_or_calibration_training(self) -> None:
        changed = dict(self.protocol)
        changed["final_used_for_selection"] = True
        with self.assertRaises(ValueError):
            validate_protocol(changed)
        changed = dict(self.protocol)
        changed["calibration_used_for_model_training"] = True
        with self.assertRaises(ValueError):
            validate_protocol(changed)


if __name__ == "__main__":
    unittest.main()
