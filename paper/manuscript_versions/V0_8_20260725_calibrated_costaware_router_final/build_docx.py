#!/usr/bin/env python3
"""Build the editable V0.8 Chinese JRTIP manuscript."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
BASE_BUILDER = REPO / "paper/tools/build_jrtip_v06_docx.py"
SOURCE = PACKAGE / "JRTIP_MANUSCRIPT_V0_8_ZH_FINAL.md"
OUTPUT = PACKAGE / "JRTIP_MANUSCRIPT_V0_8_ZH_FINAL.docx"


def load_base_builder():
    spec = importlib.util.spec_from_file_location("jrtip_v06_builder", BASE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Word builder: {BASE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_builder(base) -> None:
    # The legacy builder uses the Windows family name "宋体". On macOS the
    # installed family is exposed as "宋体-简"; using the actual family name
    # prevents LibreOffice/PDF conversion from dropping Chinese glyphs.
    original_set_run_font = base.set_run_font

    def set_run_font(run, size, bold=None, italic=None, name="Times New Roman"):
        original_set_run_font(run, size, bold=bold, italic=italic, name=name)
        if name != "Cambria Math":
            run._element.rPr.rFonts.set(base.qn("w:eastAsia"), "宋体-简")

    base.set_run_font = set_run_font

    original_setup_document = base.setup_document

    def setup_document():
        document = original_setup_document()
        document.styles["Normal"]._element.rPr.rFonts.set(
            base.qn("w:eastAsia"), "宋体-简"
        )
        return document

    base.setup_document = setup_document

    figures = PACKAGE / "figures"
    base.FIGURE_FILES_ZH = {
        1: figures / "Fig1_calibrated_costaware_architecture.png",
        2: figures / "Fig2_frozen_ap_request_cost.png",
        3: figures / "Fig3_risk_gate_outcomes.png",
        4: figures / "Fig4_frozen_action_composition.png",
    }
    base.FIGURE_CAPTIONS_ZH = {
        1: "校准式成本感知端云协同架构；在线阶段只执行一个冻结配置，离线阶段完成候选选择、风险门禁和一次性 final 评测",
        2: "三数据集冻结配置的 Official AP 与请求 KB/image 工作点；各面板独立缩放，颜色区分预算模式",
        3: "三数据集风险门禁结果；通过点区分非同动作比较与 gate 上动作完全相同的平凡通过",
        4: "冻结配置的 Edge、Original、Q90 和 ROI 动作组成；比例按全部输入图像计算",
    }

    equations = {
        r"\mathcal A=\{\mathrm{Original},\mathrm{Q90},\mathrm{ROI}\}.": "A = {Original, Q90, ROI}",
        r"V_i(a;\lambda)=\mathbb E[\Delta U_i(a)\mid z_i]-\lambda c_i(a),": "V_i(a; λ) = E[ΔU_i(a) | z_i] − λc_i(a)",
        r"a_i^*=\arg\max_{a\in\{\mathrm{Edge}\}\cup\mathcal A}V_i(a;\lambda).": "a_i* = arg max_(a ∈ {Edge} ∪ A) V_i(a; λ)",
        r"\theta=(r,\pi,m,B,\tau,\lambda),": "θ = (r, π, m, B, τ, λ)",
        r"\mathcal U_{call}=\operatorname{TopK}(\{s_i^{(r)}\}_{i=1}^{N},K).": "U_call = TopK({s_i^(r)}_(i=1)^N, K)",
        r"\mathcal U_{online}=\{i:s_i^{(r)}\ge t_{r,B}\}.": "U_online = {i : s_i^(r) ≥ t_(r,B)}",
        r"\sum_{i\in\mathcal U}c_i(a_i)\le C_B.": "Σ_(i ∈ U) c_i(a_i) ≤ C_B",
        r"C(\theta;\mathcal S)=\frac{1}{|\mathcal S|}\sum_{i\in\mathcal S}\frac{c_i(a_i)}{1024}": "C(θ; S) = (1 / |S|) Σ_(i ∈ S) c_i(a_i) / 1024",
        r"\min_{\theta\in\Theta}C(\theta;\mathcal D^{sel})\quad\mathrm{s.t.}\quad Q(\theta;\mathcal D^{sel})\ge Q(\theta_{ref};\mathcal D^{sel})-\epsilon.": "min_(θ ∈ Θ) C(θ; D_sel),  s.t.  Q(θ; D_sel) ≥ Q(θ_ref; D_sel) − ε",
        r"\max_{\theta\in\Theta}Q(\theta;\mathcal D^{sel})\quad\mathrm{s.t.}\quad C(\theta;\mathcal D^{sel})\le C_B,\;Q(\theta;\mathcal D^{sel})\ge Q(\theta_{ref};\mathcal D^{sel})-\epsilon.": "max_(θ ∈ Θ) Q(θ; D_sel),  s.t.  C(θ; D_sel) ≤ C_B and Q(θ; D_sel) ≥ Q(θ_ref; D_sel) − ε",
        r"\Delta Q^{(b)}=Q(\theta_{prov};\mathcal D_b^{gate})-Q(\theta_{ref};\mathcal D_b^{gate}),": "ΔQ^(b) = Q(θ_prov; D_gate^(b)) − Q(θ_ref; D_gate^(b))",
        r"\Delta C^{(b)}=C(\theta_{prov};\mathcal D_b^{gate})-C(\theta_{ref};\mathcal D_b^{gate}).": "ΔC^(b) = C(θ_prov; D_gate^(b)) − C(θ_ref; D_gate^(b))",
        r"\theta_{frozen}=\begin{cases}\theta_{prov},&\text{门禁通过},\\\theta_{ref},&\text{门禁失败}.\end{cases}": "θ_frozen = θ_prov if gate passes; otherwise θ_ref",
    }
    for latex, linear in equations.items():
        base.UNICODE_EQUATIONS[base.equation_key(latex)] = linear

    def add_full_width_figure(document, number, caption_text, figure_path):
        if not figure_path.is_file():
            raise FileNotFoundError(figure_path)
        base.switch_columns(document, 1)
        paragraph = document.add_paragraph()
        paragraph.alignment = base.WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run()
        run.add_picture(str(figure_path), width=base.Mm(170))

        caption = document.add_paragraph()
        caption.alignment = base.WD_ALIGN_PARAGRAPH.JUSTIFY
        caption.paragraph_format.space_before = base.Pt(1)
        caption.paragraph_format.space_after = base.Pt(3)
        caption.paragraph_format.first_line_indent = base.Mm(0)
        prefix = caption.add_run(f"图 {number} ")
        base.set_run_font(prefix, 7.7, bold=True)
        body = caption.add_run(caption_text)
        base.set_run_font(body, 7.7)
        base.switch_columns(document, 2)

    base.add_figure = add_full_width_figure


def main() -> None:
    base = load_base_builder()
    configure_builder(base)
    base.build_main(SOURCE, OUTPUT)
    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()
