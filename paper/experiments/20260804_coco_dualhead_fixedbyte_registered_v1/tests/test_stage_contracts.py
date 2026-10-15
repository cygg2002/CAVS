from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StageContractTest(unittest.TestCase):
    def test_protocol_makes_fixed_byte_primary_and_final_inference_gated(self):
        protocol = json.loads((ROOT / "configs/protocol.json").read_text())
        self.assertEqual(protocol["primary_mode"], "fixed_byte")
        self.assertTrue(protocol["registered_final_inference_after_audit_only"])
        self.assertFalse(protocol["registered_final_used_for_selection"])

    def test_training_source_never_loads_new_evaluation_roles(self):
        runner = ROOT / "scripts/run_paper.py"
        if not runner.is_file():
            self.fail("run_paper.py is not implemented")
        tree = ast.parse(runner.read_text())
        source = ast.unparse(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_train"))
        self.assertNotIn("_role_records('audit'", source)
        self.assertNotIn("_role_records('registered_final'", source)

    def test_auto_pipeline_opens_and_infers_final_only_after_passed_audit(self):
        pipeline = ROOT / "scripts/auto_pipeline.py"
        if not pipeline.is_file():
            self.fail("auto_pipeline.py is not implemented")
        source = pipeline.read_text()
        self.assertIn("audit_completion.get(\"passed\") is True", source)
        self.assertIn('run_stage("open_final")', source)
        self.assertIn('run_inference("registered_final"', source)
        self.assertLess(source.index('run_stage("open_final")'), source.index('run_inference("registered_final"'))

    def test_final_stage_uses_frozen_fixed_byte_comparators(self):
        runner = ROOT / "scripts/run_paper.py"
        if not runner.is_file():
            self.fail("run_paper.py is not implemented")
        source = runner.read_text()
        self.assertIn("frozen_comparators.json", source)
        self.assertIn('f"fixed_byte:{budget:g}"', source)


if __name__ == "__main__":
    unittest.main()
