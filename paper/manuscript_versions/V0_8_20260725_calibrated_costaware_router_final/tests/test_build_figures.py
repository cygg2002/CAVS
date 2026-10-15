from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import tempfile
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import build_figures  # noqa: E402


class FigureEvidenceTests(unittest.TestCase):
    def test_frozen_evidence_contract_matches_completed_experiment(self) -> None:
        evidence = build_figures.load_evidence()

        self.assertEqual(
            {dataset: len(rows) for dataset, rows in evidence["final"].items()},
            {"coco": 10, "ena24": 10, "oregon": 10},
        )
        self.assertEqual(
            evidence["gate_summary"],
            {
                "coco": {"non_identical": 0, "identical": 0, "fallback": 10},
                "ena24": {"non_identical": 3, "identical": 5, "fallback": 2},
                "oregon": {"non_identical": 4, "identical": 1, "fallback": 5},
            },
        )

    def test_oracle_regret_summary_is_recomputed_from_rows(self) -> None:
        evidence = build_figures.load_evidence()

        expected = {
            "coco": (0.003815118478294144, 0.011312115729398164),
            "ena24": (0.008835639104639349, 0.013186877990319035),
            "oregon": (0.003980941472952371, 0.017063295008976798),
        }
        for dataset, (mean_value, max_value) in expected.items():
            actual = evidence["regret"][dataset]
            self.assertAlmostEqual(actual["mean"], mean_value, places=12)
            self.assertAlmostEqual(actual["max"], max_value, places=12)

    def test_all_four_figures_are_nonblank_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = build_figures.build_all(Path(directory))

            self.assertEqual(len(outputs), 4)
            for path in outputs:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 10_000)
                self.assertTrue(build_figures.has_visual_content(path))

    def test_architecture_uses_reviewed_drawio_export(self) -> None:
        reviewed = PACKAGE / "figures" / "drawio" / "Fig1_calibrated_costaware_router.png"

        with tempfile.TemporaryDirectory() as directory:
            generated = build_figures.build_all(Path(directory))[0]

            self.assertEqual(
                hashlib.sha256(generated.read_bytes()).digest(),
                hashlib.sha256(reviewed.read_bytes()).digest(),
            )


if __name__ == "__main__":
    unittest.main()
