from __future__ import annotations

from pathlib import Path
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
MANUSCRIPT = PACKAGE / "JRTIP_MANUSCRIPT_V0_8_ZH_FINAL.md"


class ManuscriptContractTests(unittest.TestCase):
    def test_scientific_placeholders_are_removed(self) -> None:
        text = MANUSCRIPT.read_text(encoding="utf-8")
        forbidden = (
            "待冻结 final",
            "冻结 final 完成后",
            "【待填",
            "TBD",
            "TODO",
        )
        for token in forbidden:
            self.assertNotIn(token, text)

    def test_frozen_evidence_is_stated_without_overclaiming(self) -> None:
        text = MANUSCRIPT.read_text(encoding="utf-8")
        required = (
            "非平凡通过只有 7 个",
            "17 个回退",
            "0.003815",
            "0.008836",
            "0.003981",
            "不构成 30 个工作点上的同时统计保证",
            "尚未执行",
            "Estimated E2E",
            "10.1109/SEC62691.2024.00021",
        )
        for token in required:
            self.assertIn(token, text)

    def test_all_figure_sources_exist(self) -> None:
        for number, name in enumerate(
            (
                "Fig1_calibrated_costaware_architecture.png",
                "Fig2_frozen_ap_request_cost.png",
                "Fig3_risk_gate_outcomes.png",
                "Fig4_frozen_action_composition.png",
            ),
            start=1,
        ):
            self.assertIn(f"[图 {number}：", MANUSCRIPT.read_text(encoding="utf-8"))
            self.assertTrue((PACKAGE / "figures" / name).is_file())


if __name__ == "__main__":
    unittest.main()
