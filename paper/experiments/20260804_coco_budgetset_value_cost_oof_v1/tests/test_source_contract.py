from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SourceContractTest(unittest.TestCase):
    def test_protocol_is_development_only(self):
        protocol = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["source_role"], "development")
        self.assertFalse(protocol["audit_accessed"])
        self.assertFalse(protocol["registered_final_accessed"])
        self.assertEqual(protocol["expected_images"], 5000)
        self.assertEqual(protocol["fold_count"], 5)

    def test_runner_never_loads_non_development_role(self):
        runner = ROOT / "scripts/run_oof.py"
        if not runner.is_file():
            self.fail("run_oof.py has not been implemented")
        source = runner.read_text(encoding="utf-8")
        self.assertNotIn('_role_records("audit"', source)
        self.assertNotIn('_role_records("registered_final"', source)
        self.assertNotIn('/audit/', source)
        self.assertNotIn('/registered_final/', source)


if __name__ == "__main__":
    unittest.main()
