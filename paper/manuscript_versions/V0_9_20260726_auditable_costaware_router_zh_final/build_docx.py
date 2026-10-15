#!/usr/bin/env python3
"""Build the submission-ready V0.9 Chinese manuscript Word file.

Reuses the shared JRTIP builder but fixes every V0.8 typesetting defect that
was confirmed in REVIEW_REPORT_V0_8_20260726_ZH.md:

1. inline-math LaTeX residue (\\theta, \\pi, \\epsilon, \\lfloor, \\mathcal D, ...);
2. equation font Cambria Math missing on this host (round-glyph substitution);
3. equation (13) rendered with English conditions in a Chinese manuscript;
4. tables without numbered captions (V0.9 Markdown now carries **表 N** blocks,
   and this builder additionally accepts appendix captions such as **表 B1**);
5. numeric table columns centered instead of right-aligned; tiny 6.2 pt font.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
BASE_BUILDER = REPO / "paper/tools/build_jrtip_v06_docx.py"
SOURCE = PACKAGE / "MANUSCRIPT_V0_9_ZH_FINAL.md"
OUTPUT = PACKAGE / "MANUSCRIPT_V0_9_ZH_FINAL.docx"

TABLE_CAPTION_RE = re.compile(r"^\*\*(?:Table|表)\s*[A-Z]?\d+")

# Applied before the base clean_inline so no LaTeX command survives the
# backslash-stripping fallback. Longest keys are replaced first.
INLINE_MATH_REPLACEMENTS = {
    r"\mathcal D": "D",
    r"\mathcal{D}": "D",
    r"\mathcal U": "U",
    r"\mathcal{U}": "U",
    r"\mathcal S": "S",
    r"\mathcal{S}": "S",
    r"\mathcal A": "A",
    r"\mathcal{A}": "A",
    r"\mathbb E": "E",
    r"\mathbb{E}": "E",
    r"\theta": "θ",
    r"\pi": "π",
    r"\epsilon": "ε",
    r"\lambda": "λ",
    r"\tau": "τ",
    r"\sigma": "σ",
    r"\Delta": "Δ",
    r"\Theta": "Θ",
    r"\lfloor": "⌊",
    r"\rfloor": "⌋",
    r"\lceil": "⌈",
    r"\rceil": "⌉",
    r"\notin": "∉",
    r"\setminus": "∖",
    r"\subseteq": "⊆",
    r"\le": "≤",
    r"\ge": "≥",
    r"\ne": "≠",
    r"\in": "∈",
    r"\cup": "∪",
    r"\cap": "∩",
    r"\dots": "…",
    r"\equiv": "≡",
    r"\cdot": "·",
    r"\mid": "|",
    r"\pm": "±",
    r"\times": "×",
    r"\arg": "arg ",
    r"\sum": "Σ",
}


def load_base_builder():
    spec = importlib.util.spec_from_file_location("jrtip_v06_builder", BASE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Word builder: {BASE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_builder(base) -> None:
    # --- macOS CJK family name -------------------------------------------------
    original_set_run_font = base.set_run_font

    def set_run_font(run, size, bold=None, italic=None, name="Times New Roman"):
        original_set_run_font(run, size, bold=bold, italic=italic, name=name)
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

    # --- inline math: no LaTeX command may survive ------------------------------
    original_clean_inline = base.clean_inline

    def clean_inline(value: str) -> str:
        for source in sorted(INLINE_MATH_REPLACEMENTS, key=len, reverse=True):
            value = value.replace(source, INLINE_MATH_REPLACEMENTS[source])
        value = value.replace("⌊ ", "⌊").replace(" ⌋", "⌋")
        return original_clean_inline(value)

    base.clean_inline = clean_inline

    # --- figures ---------------------------------------------------------------
    figures = PACKAGE / "figures"
    base.FIGURE_FILES_ZH = {
        1: figures / "Fig1_calibrated_costaware_architecture.png",
        2: figures / "Fig2_frozen_ap_request_cost.png",
        3: figures / "Fig3_risk_gate_outcomes.png",
        4: figures / "Fig4_frozen_action_composition.png",
    }
    base.FIGURE_CAPTIONS_ZH = {
        1: "可审计成本感知端云协同架构。上：在线部署泳道依次执行端侧检测（YOLO26n）、样本排序（8 个候选之一）、"
           "预算接纳（Fixed-call/Online/Fixed-byte）、载荷选择（Original/Q90/ROI）与云端复核融合（YOLO26x，IoU 0.5）；"
           "未接纳图像直接输出端侧结果。下：离线校准泳道仅使用拟合集、选择集与门禁集完成配置选择、"
           "200 次配对 bootstrap 风险门禁与冻结回退；虚线为一次性部署，终评不产生任何反向路径",
        2: "三数据集冻结配置的 Official AP 与请求 KB/图工作点。各面板独立缩放；同一预算模式的工作点按预算升序连线，"
           "形状区分模式（圆=Fixed-call，方=Online，三角=Fixed-byte）。灰点为系统参考：Edge=Edge-only，"
           "Cloud-Orig=Cloud-only Original，Q90=Always Q90，ROI-fb=Always ROI fallback，Shared-all=共享动作头全上传",
        3: "三数据集风险门禁结果构成（纵轴为工作点数，每域 10 个）。通过点区分非同动作比较与门禁集上动作完全相同的平凡通过",
        4: "冻结配置的 Edge、Original、Q90 和 ROI 动作组成。比例按全部输入图像计算，并标注实际上传率与 ROI 比例；"
           "FC=Fixed-call，ON=Online，FB=Fixed-byte",
    }

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

    # --- display equations -------------------------------------------------------
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
        r"\max_{\theta\in\Theta}Q(\theta;\mathcal D^{sel})\quad\mathrm{s.t.}\quad C(\theta;\mathcal D^{sel})\le C_B,\;Q(\theta;\mathcal D^{sel})\ge Q(\theta_{ref};\mathcal D^{sel})-\epsilon.": "max_(θ ∈ Θ) Q(θ; D_sel),  s.t.  C(θ; D_sel) ≤ C_B 且 Q(θ; D_sel) ≥ Q(θ_ref; D_sel) − ε",
        r"\Delta Q^{(b)}=Q(\theta_{prov};\mathcal D_b^{gate})-Q(\theta_{ref};\mathcal D_b^{gate}),": "ΔQ^(b) = Q(θ_prov; D_gate^(b)) − Q(θ_ref; D_gate^(b))",
        r"\Delta C^{(b)}=C(\theta_{prov};\mathcal D_b^{gate})-C(\theta_{ref};\mathcal D_b^{gate}).": "ΔC^(b) = C(θ_prov; D_gate^(b)) − C(θ_ref; D_gate^(b))",
        r"\theta_{frozen}=\begin{cases}\theta_{prov},&\text{门禁通过},\\\theta_{ref},&\text{门禁失败}.\end{cases}": "θ_frozen = θ_prov（门禁通过）；θ_ref（门禁失败）",
    }
    for latex, linear in equations.items():
        base.UNICODE_EQUATIONS[base.equation_key(latex)] = linear

    # Cambria Math does not exist on this host; Times New Roman does and keeps
    # the equations serif-consistent with the body (review finding: round-glyph
    # substitution in the V0.8 PDF).
    def add_equation(document, latex, number):
        normalized = " ".join(line.strip() for line in latex.splitlines())
        equation = base.UNICODE_EQUATIONS.get(
            base.equation_key(normalized), base.clean_inline(normalized)
        )
        paragraph = document.add_paragraph()
        paragraph.alignment = base.WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = base.Pt(2)
        paragraph.paragraph_format.space_after = base.Pt(2)
        run = paragraph.add_run(f"{equation}    ({number})")
        base.set_run_font(run, 9, name="Times New Roman")

    base.add_equation = add_equation

    # --- tables: markdown alignment markers + larger minimum font ---------------
    def add_table(document, caption_text, lines, full_width=True):
        keep_together = bool(
            caption_text and re.search(r"(?:Table|表)\s*[57]\b", caption_text)
        )
        if full_width:
            base.switch_columns(document, 1)
        if caption_text:
            caption = document.add_paragraph()
            caption.paragraph_format.keep_with_next = True
            caption.paragraph_format.space_after = base.Pt(2)
            caption.paragraph_format.first_line_indent = base.Mm(0)
            base.add_inline_runs(caption, caption_text.replace("**", ""), 7.8)

        alignments: list[str] = []
        if len(lines) >= 2:
            for spec in lines[1].strip().strip("|").split("|"):
                spec = spec.strip()
                if spec.endswith(":") and spec.startswith(":"):
                    alignments.append("center")
                elif spec.endswith(":"):
                    alignments.append("right")
                else:
                    alignments.append("left")

        rows = [
            [base.strip_cell_markup(cell) for cell in line.strip().strip("|").split("|")]
            for line in lines
        ]
        rows = [row for index, row in enumerate(rows) if index != 1]
        column_count = max(len(row) for row in rows)
        table = document.add_table(rows=len(rows), cols=column_count)
        table.alignment = base.WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True
        table.allow_autofit = True
        alignment_map = {
            "left": base.WD_ALIGN_PARAGRAPH.LEFT,
            "center": base.WD_ALIGN_PARAGRAPH.CENTER,
            "right": base.WD_ALIGN_PARAGRAPH.RIGHT,
        }
        for row_index, values in enumerate(rows):
            row = table.rows[row_index]
            if row_index == 0:
                base.set_repeat_table_header(row)
            for column_index in range(column_count):
                cell = row.cells[column_index]
                cell.vertical_alignment = base.WD_CELL_VERTICAL_ALIGNMENT.CENTER
                value = values[column_index] if column_index < len(values) else ""
                paragraph = cell.paragraphs[0]
                if row_index == 0:
                    paragraph.alignment = (
                        base.WD_ALIGN_PARAGRAPH.LEFT
                        if column_index == 0
                        else base.WD_ALIGN_PARAGRAPH.CENTER
                    )
                else:
                    spec = (
                        alignments[column_index]
                        if column_index < len(alignments)
                        else "left"
                    )
                    paragraph.alignment = alignment_map[spec]
                paragraph.paragraph_format.space_before = base.Pt(0)
                paragraph.paragraph_format.space_after = base.Pt(0)
                paragraph.paragraph_format.line_spacing_rule = base.WD_LINE_SPACING.SINGLE
                paragraph.paragraph_format.keep_with_next = (
                    keep_together and row_index < len(rows) - 1
                )
                run = paragraph.add_run(value)
                base.set_run_font(
                    run, 6.8 if column_count >= 7 else 7.2, bold=row_index == 0
                )
                border = {"val": "single", "sz": "4", "color": "A6A6A6"}
                base.set_cell_border(
                    cell, top=border, bottom=border, left=border, right=border
                )
                if row_index == 0:
                    shading = base.OxmlElement("w:shd")
                    shading.set(base.qn("w:fill"), "D9E2F3")
                    cell._tc.get_or_add_tcPr().append(shading)
        document.add_paragraph().paragraph_format.space_after = base.Pt(1)
        if full_width:
            base.switch_columns(document, 2)

    base.add_table = add_table


def build_main(base, source: Path, output: Path) -> None:
    """Copy of the base build_main with appendix-aware table captions."""
    blocks = base.split_blocks(source.read_text(encoding="utf-8"))
    document = base.setup_document()
    figure_captions = base.FIGURE_CAPTIONS_ZH
    figure_files = base.FIGURE_FILES_ZH
    in_body = False
    equation_number = 0
    pending_table_caption = None
    author_index = 0

    for block in blocks:
        if block.startswith("# "):
            base.add_title(document, block[2:].strip())
            continue
        if block.startswith("## Abstract") or block.startswith("## 摘要"):
            heading = "Abstract" if "Abstract" in block else "摘要"
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.space_before = base.Pt(4)
            paragraph.paragraph_format.space_after = base.Pt(1)
            run = paragraph.add_run(heading)
            base.set_run_font(run, 9, bold=True)
            continue
        if block.startswith("**Keywords:**") or block.startswith("**关键词：**"):
            base.add_body(document, block, size=8.5, first_line=False)
            continue
        if block.startswith("## "):
            heading = block[3:].strip()
            if re.match(r"^(1 |1\.|1 引言|1 Introduction)", heading) and not in_body:
                base.switch_columns(document, 2)
                in_body = True
            base.add_heading(document, heading, 1)
            continue
        if block.startswith("### "):
            base.add_heading(document, block[4:].strip(), 2)
            continue
        if TABLE_CAPTION_RE.match(block):
            pending_table_caption = block
            continue
        if base.is_table_block(block):
            base.add_table(document, pending_table_caption, block.splitlines(), full_width=True)
            pending_table_caption = None
            continue
        figure_match = re.search(r"\[(?:Fig\.|图)\s*(\d+)", block)
        if figure_match and block.startswith("**["):
            number = int(figure_match.group(1))
            base.add_figure(document, number, figure_captions[number], figure_files[number])
            continue
        if block.startswith("\\[") and block.endswith("\\]"):
            equation_number += 1
            base.add_equation(document, block[2:-2].strip(), equation_number)
            continue
        if re.match(r"^(\d+\.|-)\s+", block):
            base.add_list_item(document, block, ordered=block[0].isdigit())
            continue
        if not in_body and not block.startswith("##"):
            if author_index < 3:
                base.add_author_line(document, block, author_index)
                author_index += 1
            else:
                base.add_body(document, block, size=8.5)
            continue
        base.add_body(
            document,
            block,
            size=7.6 if block.startswith("[") and re.match(r"^\[\d+\]", block) else 8.7,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)


def main() -> None:
    base = load_base_builder()
    configure_builder(base)
    build_main(base, SOURCE, OUTPUT)
    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()
