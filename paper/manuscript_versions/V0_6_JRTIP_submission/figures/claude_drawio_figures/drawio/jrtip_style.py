"""jrtip_style.py — shared visual constants for JRTIP V0.6 figures.

Implements FIGURE_DRAWING_SPEC_ZH.md unified rules: white bg, Arial, black-gray
lines, low-saturation accent fills, thin strokes, small corner radius, no
gradients/shadows/3D. Figure numbers/titles live only in manuscript text.

All builders take `size` (pt) and `sw` (stroke width pt) so the SAME palette can
render both a roomy on-screen version and a print-scale version (body 8pt,
sw 1.0) that stays readable when fit into a 174 mm double-column slot.
"""
FONT = "Arial"
INK = "#262626"
LINE = "#595959"
RED = "#C00000"

CAT = {
    "edge":     ("#DCE6F1", "#8FAADC"),
    "queue":    ("#DDEBF7", "#8FAADC"),
    "cloud":    ("#FCE4D6", "#E8A87C"),
    "fusion":   ("#E2F0D9", "#A3C293"),
    "offline":  ("#E7E6E6", "#A6A6A6"),
    "io":       ("#F2F2F2", "#808080"),
    "decision": ("#FFF6DA", "#BF9000"),
    "deadline": ("#FFFFFF", RED),       # red border, NO red fill (spec)
}


def box(cat, size=9, bold=False, sw=1.2):
    fill, stroke = CAT[cat]
    return ("rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
            "fillColor=%s;strokeColor=%s;strokeWidth=%s;"
            "fontFamily=%s;fontSize=%d;fontColor=%s;%s"
            "align=center;verticalAlign=middle;" % (
                fill, stroke, sw, FONT, size, INK, "fontStyle=1;" if bold else ""))


def diamond(cat="decision", size=9, sw=1.2):
    fill, stroke = CAT[cat]
    return ("rhombus;whiteSpace=wrap;html=1;"
            "fillColor=%s;strokeColor=%s;strokeWidth=%s;"
            "fontFamily=%s;fontSize=%d;fontColor=%s;" % (
                fill, stroke, sw, FONT, size, INK))


def title(size=10, color="#1F3864"):
    return ("text;html=1;align=center;verticalAlign=middle;fillColor=none;strokeColor=none;"
            "fontFamily=%s;fontSize=%d;fontStyle=1;fontColor=%s;" % (FONT, size, color))


def note(size=8, color=INK):
    return ("text;html=1;align=center;verticalAlign=middle;fillColor=none;strokeColor=none;"
            "fontFamily=%s;fontSize=%d;fontColor=%s;" % (FONT, size, color))


def band(cat, size=9):
    fill, stroke = CAT[cat]
    return ("rounded=0;whiteSpace=wrap;html=1;"
            "fillColor=%s;strokeColor=%s;strokeWidth=1.0;"
            "fontFamily=%s;fontSize=%d;fontStyle=1;fontColor=#1F3864;"
            "align=center;verticalAlign=middle;" % (fill, stroke, FONT, size))


# --- edge fragments (APPENDED to the skill base edge style -> they win) ---
def main_flow(sw=1.2, size=8):
    return "edgeStyle=orthogonalEdgeStyle;strokeWidth=%s;fontFamily=%s;fontSize=%d;fontColor=%s;" % (sw, FONT, size, INK)


def ctrl_flow(sw=1.0, size=8):
    return "edgeStyle=orthogonalEdgeStyle;strokeWidth=%s;fontFamily=%s;fontSize=%d;fontColor=%s;" % (sw, FONT, size, INK)


def offline_dashed(sw=1.0, size=8):
    return "edgeStyle=orthogonalEdgeStyle;dashed=1;dashPattern=8 4;strokeWidth=%s;fontFamily=%s;fontSize=%d;fontColor=%s;" % (sw, FONT, size, INK)


def late_dashdot(sw=1.0, size=8):
    return "edgeStyle=orthogonalEdgeStyle;dashed=1;dashPattern=1 2 4 2;strokeWidth=%s;fontFamily=%s;fontSize=%d;fontColor=%s;" % (sw, FONT, size, RED)
