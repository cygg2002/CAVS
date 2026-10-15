from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StageContractTest(unittest.TestCase):
    def test_protocol_declares_final_as_selection_forbidden_and_once_only(self):
        source = (ROOT / "configs/protocol.json").read_text()
        self.assertIn('"registered_final_used_for_selection": false', source)
        self.assertIn('"registered_final_access_once": true', source)

    def test_runner_source_never_uses_final_for_training(self):
        runner = ROOT / "scripts/run_paper.py"
        if not runner.is_file():
            self.fail("run_paper.py is not implemented")
        tree = ast.parse(runner.read_text())
        train_source = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_train"
        )
        text = ast.unparse(train_source)
        self.assertNotIn("_role_records('registered_final'", text)
        self.assertNotIn("_role_records('audit'", text)

    def test_final_stage_reads_frozen_comparators(self):
        runner = ROOT / "scripts/run_paper.py"
        if not runner.is_file():
            self.fail("run_paper.py is not implemented")
        source = runner.read_text()
        self.assertIn('frozen_comparators.json', source)
        self.assertIn('assert_final_allowed', source)

    def test_final_marks_access_before_reading_registered_metrics(self):
        runner = ROOT / "scripts/run_paper.py"
        tree = ast.parse(runner.read_text())
        final_source = ast.unparse(
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "run_final"
            )
        )
        self.assertNotIn("NotImplementedError", final_source)
        self.assertIn("accessed.json", final_source)
        self.assertIn("_evaluate_role('registered_final'", final_source)
        self.assertLess(
            final_source.index("accessed.json"),
            final_source.index("_evaluate_role('registered_final'"),
        )

    def test_inference_chunks_bind_the_implementation_hash(self):
        source = (ROOT / "scripts/infer_original.py").read_text()
        self.assertIn('"inference_script": file_sha256(Path(__file__))', source)
        self.assertIn('"detector_implementation": implementation_sha256', source)
        self.assertIn('sys.path.insert(0, str(REPO))', source)

    def test_auto_pipeline_keeps_final_behind_the_audit_gate(self):
        source = (ROOT / "scripts/auto_pipeline.py").read_text()
        tree = ast.parse(source)
        main_source = ast.unparse(
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main"
            )
        )
        self.assertIn("audit_completion.get('passed') is True", main_source)
        self.assertIn("run_stage('final')", main_source)
        self.assertLess(
            main_source.index("audit_completion.get('passed') is True"),
            main_source.index("run_stage('final')"),
        )


if __name__ == "__main__":
    unittest.main()
