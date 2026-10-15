#!/usr/bin/env python3
"""Build editable draw.io sources for the JRTIP workflow figures."""

from __future__ import annotations

import html
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper/manuscript_versions/V0_6_JRTIP_submission/figures/drawio"

EDGE_FILL = "#DCE6F1"
TRANSPORT_FILL = "#DDEBF7"
CLOUD_FILL = "#FCE4D6"
OUTPUT_FILL = "#E2F0D9"
TRAIN_FILL = "#E7E6E6"
WHITE = "#FFFFFF"
LINE = "#404040"
MUTED_LINE = "#A6A6A6"
RED = "#C00000"
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")


def cell_style(fill: str, *, bold: bool = False, dashed: bool = False, font_size: int = 20) -> str:
    return (
        "rounded=1;arcSize=6;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={LINE};strokeWidth=1.2;"
        f"fontFamily=Arial;fontSize={font_size};align=center;verticalAlign=middle;"
        f"fontStyle={1 if bold else 0};"
        f"dashed={1 if dashed else 0};"
    )


def header_style(fill: str) -> str:
    return (
        "rounded=0;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={LINE};strokeWidth=1.2;"
        "fontFamily=Arial;fontSize=24;fontStyle=1;"
        "align=center;verticalAlign=middle;"
    )


def background_style() -> str:
    return (
        "rounded=0;whiteSpace=wrap;html=1;"
        f"fillColor={WHITE};strokeColor={MUTED_LINE};strokeWidth=1;"
        "pointerEvents=0;"
    )


def diamond_style(fill: str, *, red_border: bool = False, font_size: int = 19) -> str:
    stroke = RED if red_border else LINE
    return (
        "rhombus;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=1.4;"
        f"fontFamily=Arial;fontSize={font_size};fontStyle=1;"
        "align=center;verticalAlign=middle;"
    )


def edge_style(*, dashed: bool = False, dash_pattern: str | None = None, red: bool = False) -> str:
    stroke = RED if red else LINE
    style = (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
        "jettySize=auto;html=1;endArrow=block;endFill=1;"
        f"strokeColor={stroke};strokeWidth={1.4 if not dashed else 1.2};"
        "fontFamily=Arial;fontSize=16;labelBackgroundColor=#FFFFFF;"
    )
    if dashed:
        style += "dashed=1;"
    if dash_pattern:
        style += f"dashPattern={dash_pattern};"
    return style


class Diagram:
    def __init__(self, name: str, width: int, height: int) -> None:
        self.name = name
        self.width = width
        self.height = height
        self.vertices: dict[str, dict[str, object]] = {}
        self.edges: list[dict[str, object]] = []
        self.mxfile = ET.Element(
            "mxfile",
            {
                "host": "Electron",
                "agent": "Codex draw.io generator",
                "version": "24.7.17",
                "type": "device",
            },
        )
        diagram = ET.SubElement(self.mxfile, "diagram", {"id": name.lower().replace(" ", "-"), "name": name})
        self.model = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": "1600",
                "dy": "900",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(width),
                "pageHeight": str(height),
                "math": "0",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.next_id = 2

    def vertex(self, value: str, x: int, y: int, width: int, height: int, style: str, *, id_: str | None = None) -> str:
        cell_id = id_ or f"v{self.next_id}"
        self.next_id += 1
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {"id": cell_id, "value": value, "style": style, "vertex": "1", "parent": "1"},
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {"x": str(x), "y": str(y), "width": str(width), "height": str(height), "as": "geometry"},
        )
        self.vertices[cell_id] = {
            "value": value,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "style": style,
        }
        return cell_id

    def edge(
        self,
        source: str,
        target: str,
        *,
        label: str = "",
        style: str | None = None,
        points: list[tuple[int, int]] | None = None,
    ) -> str:
        cell_id = f"e{self.next_id}"
        self.next_id += 1
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": label,
                "style": style or edge_style(),
                "edge": "1",
                "parent": "1",
                "source": source,
                "target": target,
            },
        )
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if points:
            point_array = ET.SubElement(geometry, "Array", {"as": "points"})
            for x, y in points:
                ET.SubElement(point_array, "mxPoint", {"x": str(x), "y": str(y)})
        self.edges.append(
            {
                "source": source,
                "target": target,
                "label": label,
                "style": style or edge_style(),
                "points": points or [],
            }
        )
        return cell_id

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.indent(self.mxfile, space="  ")
        ET.ElementTree(self.mxfile).write(path, encoding="utf-8", xml_declaration=True)

    def replace_text(self, vertices: dict[str, str], edge_labels: dict[str, str]) -> None:
        """Replace visible labels before exporting a language variant."""
        cells = {cell.get("id"): cell for cell in self.root.findall("mxCell")}
        for cell_id, value in vertices.items():
            if cell_id not in self.vertices or cell_id not in cells:
                raise KeyError(f"Unknown vertex id: {cell_id}")
            self.vertices[cell_id]["value"] = value
            cells[cell_id].set("value", value)
        for edge, cell in zip(self.edges, [c for c in self.root.findall("mxCell") if c.get("edge") == "1"]):
            label = str(edge["label"])
            if label in edge_labels:
                translated = edge_labels[label]
                edge["label"] = translated
                cell.set("value", translated)

    @staticmethod
    def _style_value(style: str, key: str, default: str) -> str:
        match = re.search(rf"(?:^|;){re.escape(key)}=([^;]+)", style)
        return match.group(1) if match else default

    def _edge_polyline(self, edge: dict[str, object]) -> list[tuple[float, float]]:
        source = self.vertices[str(edge["source"])]
        target = self.vertices[str(edge["target"])]
        sx0 = float(source["x"])
        sy0 = float(source["y"])
        sw = float(source["width"])
        sh = float(source["height"])
        tx0 = float(target["x"])
        ty0 = float(target["y"])
        tw = float(target["width"])
        th = float(target["height"])
        scx, scy = sx0 + sw / 2, sy0 + sh / 2
        tcx, tcy = tx0 + tw / 2, ty0 + th / 2

        manual = [(float(x), float(y)) for x, y in edge["points"]]
        if manual:
            if abs(manual[0][0] - scx) >= abs(manual[0][1] - scy):
                start = (sx0 + sw if manual[0][0] > scx else sx0, scy)
            else:
                start = (scx, sy0 + sh if manual[0][1] > scy else sy0)
            if abs(manual[-1][0] - tcx) >= abs(manual[-1][1] - tcy):
                end = (tx0 if manual[-1][0] < tcx else tx0 + tw, tcy)
            else:
                end = (tcx, ty0 if manual[-1][1] < tcy else ty0 + th)
            return [start, *manual, end]

        dx, dy = tcx - scx, tcy - scy
        if abs(dx) >= abs(dy):
            start = (sx0 + sw if dx >= 0 else sx0, scy)
            end = (tx0 if dx >= 0 else tx0 + tw, tcy)
            mid = (start[0] + end[0]) / 2
            return [start, (mid, start[1]), (mid, end[1]), end]
        start = (scx, sy0 + sh if dy >= 0 else sy0)
        end = (tcx, ty0 if dy >= 0 else ty0 + th)
        mid = (start[1] + end[1]) / 2
        return [start, (start[0], mid), (end[0], mid), end]

    def save_svg(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        svg = ET.Element(
            "svg",
            {
                "xmlns": "http://www.w3.org/2000/svg",
                "xmlns:xhtml": "http://www.w3.org/1999/xhtml",
                "viewBox": f"0 0 {self.width} {self.height}",
                "width": str(self.width),
                "height": str(self.height),
            },
        )
        defs = ET.SubElement(svg, "defs")
        for marker_id, color in [("arrow", LINE), ("arrow-red", RED)]:
            marker = ET.SubElement(
                defs,
                "marker",
                {
                    "id": marker_id,
                    "markerWidth": "8",
                    "markerHeight": "8",
                    "refX": "7",
                    "refY": "3",
                    "orient": "auto",
                    "markerUnits": "strokeWidth",
                },
            )
            ET.SubElement(marker, "path", {"d": "M0,0 L0,6 L8,3 z", "fill": color})
        ET.SubElement(svg, "rect", {"x": "0", "y": "0", "width": str(self.width), "height": str(self.height), "fill": WHITE})

        background_ids = [
            id_
            for id_, vertex in self.vertices.items()
            if "pointerEvents=0" in str(vertex["style"])
        ]
        for id_ in background_ids:
            self._append_svg_vertex(svg, self.vertices[id_])

        for edge in self.edges:
            points = self._edge_polyline(edge)
            style = str(edge["style"])
            color = self._style_value(style, "strokeColor", LINE)
            dashed = self._style_value(style, "dashed", "0") == "1"
            dash_pattern = self._style_value(style, "dashPattern", "7 5")
            line = ET.SubElement(
                svg,
                "polyline",
                {
                    "points": " ".join(f"{x:.1f},{y:.1f}" for x, y in points),
                    "fill": "none",
                    "stroke": color,
                    "stroke-width": self._style_value(style, "strokeWidth", "1.4"),
                    "marker-end": f"url(#{'arrow-red' if color == RED else 'arrow'})",
                },
            )
            if dashed:
                line.set("stroke-dasharray", dash_pattern)
            label = str(edge["label"])
            if label:
                middle = points[len(points) // 2]
                self._append_svg_text(svg, label, middle[0] - 75, middle[1] - 20, 150, 40, 16, False)

        for id_, vertex in self.vertices.items():
            if id_ not in background_ids:
                self._append_svg_vertex(svg, vertex)

        ET.indent(svg, space="  ")
        ET.ElementTree(svg).write(path, encoding="utf-8", xml_declaration=True)

    @staticmethod
    def _plain_text(value: str) -> str:
        value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
        value = re.sub(r"<[^>]+>", "", value)
        return html.unescape(value)

    @staticmethod
    def _draw_dashed_line(
        draw: ImageDraw.ImageDraw,
        points: list[tuple[float, float]],
        fill: str,
        width: int,
        dash: tuple[int, int],
    ) -> None:
        for start, end in zip(points, points[1:]):
            x1, y1 = start
            x2, y2 = end
            length = math.hypot(x2 - x1, y2 - y1)
            if length == 0:
                continue
            ux, uy = (x2 - x1) / length, (y2 - y1) / length
            cursor = 0.0
            while cursor < length:
                dash_end = min(cursor + dash[0], length)
                draw.line(
                    (
                        x1 + ux * cursor,
                        y1 + uy * cursor,
                        x1 + ux * dash_end,
                        y1 + uy * dash_end,
                    ),
                    fill=fill,
                    width=width,
                )
                cursor += dash[0] + dash[1]

    @staticmethod
    def _draw_arrow_head(
        draw: ImageDraw.ImageDraw,
        before: tuple[float, float],
        tip: tuple[float, float],
        fill: str,
        scale: int,
    ) -> None:
        angle = math.atan2(tip[1] - before[1], tip[0] - before[0])
        length = 10 * scale
        half_width = 4.5 * scale
        base_x = tip[0] - length * math.cos(angle)
        base_y = tip[1] - length * math.sin(angle)
        normal_x = -math.sin(angle)
        normal_y = math.cos(angle)
        draw.polygon(
            [
                tip,
                (base_x + half_width * normal_x, base_y + half_width * normal_y),
                (base_x - half_width * normal_x, base_y - half_width * normal_y),
            ],
            fill=fill,
        )

    @staticmethod
    def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        lines: list[str] = []
        for raw_line in text.splitlines() or [""]:
            if not raw_line:
                lines.append("")
                continue
            current = ""
            for char in raw_line:
                candidate = current + char
                if current and draw.textlength(candidate, font=font) > max_width:
                    lines.append(current.rstrip())
                    current = char.lstrip()
                else:
                    current = candidate
            if current:
                lines.append(current)
        return lines

    def _draw_text_pillow(
        self,
        draw: ImageDraw.ImageDraw,
        value: str,
        box: tuple[int, int, int, int],
        font_size: int,
        bold: bool,
        scale: int,
        *,
        align_left: bool = False,
    ) -> None:
        x1, y1, x2, y2 = box
        font = ImageFont.truetype(str(FONT_PATH), font_size * scale)
        text = self._plain_text(value)
        lines = self._wrap_text(draw, text, font, max(10, x2 - x1 - 12 * scale))
        line_height = int(font_size * scale * 1.28)
        total_height = line_height * len(lines)
        y = y1 + max(0, (y2 - y1 - total_height) // 2)
        for line in lines:
            text_width = draw.textlength(line, font=font)
            x = x1 + 6 * scale if align_left else x1 + (x2 - x1 - text_width) / 2
            draw.text(
                (x, y),
                line,
                fill="#222222",
                font=font,
                stroke_width=1 if bold else 0,
                stroke_fill="#222222",
            )
            y += line_height

    def save_png_pdf(self, png_path: Path, pdf_path: Path, *, scale: int = 3) -> None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (self.width * scale, self.height * scale), WHITE)
        draw = ImageDraw.Draw(image)

        background_ids = [
            id_
            for id_, vertex in self.vertices.items()
            if "pointerEvents=0" in str(vertex["style"])
        ]
        for id_ in background_ids:
            self._draw_vertex_pillow(draw, self.vertices[id_], scale)

        for edge in self.edges:
            raw_points = self._edge_polyline(edge)
            points = [(x * scale, y * scale) for x, y in raw_points]
            style = str(edge["style"])
            color = self._style_value(style, "strokeColor", LINE)
            line_width = max(2, int(float(self._style_value(style, "strokeWidth", "1.4")) * scale))
            dashed = self._style_value(style, "dashed", "0") == "1"
            if dashed:
                self._draw_dashed_line(draw, points, color, line_width, (8 * scale, 5 * scale))
            else:
                draw.line(points, fill=color, width=line_width, joint="curve")
            self._draw_arrow_head(draw, points[-2], points[-1], color, scale)
            label = str(edge["label"])
            if label:
                middle = points[len(points) // 2]
                label_box = (
                    int(middle[0] - 72 * scale),
                    int(middle[1] - 17 * scale),
                    int(middle[0] + 72 * scale),
                    int(middle[1] + 17 * scale),
                )
                draw.rectangle(label_box, fill=WHITE)
                self._draw_text_pillow(draw, label, label_box, 16, False, scale)

        for id_, vertex in self.vertices.items():
            if id_ not in background_ids:
                self._draw_vertex_pillow(draw, vertex, scale)

        effective_dpi = self.width * scale / (174 / 25.4)
        image.save(png_path, dpi=(effective_dpi, effective_dpi))
        image.save(pdf_path, "PDF", resolution=effective_dpi)

    def _draw_vertex_pillow(self, draw: ImageDraw.ImageDraw, vertex: dict[str, object], scale: int) -> None:
        x = int(float(vertex["x"]) * scale)
        y = int(float(vertex["y"]) * scale)
        width = int(float(vertex["width"]) * scale)
        height = int(float(vertex["height"]) * scale)
        style = str(vertex["style"])
        fill = self._style_value(style, "fillColor", WHITE)
        stroke = self._style_value(style, "strokeColor", LINE)
        stroke_width = max(1, int(float(self._style_value(style, "strokeWidth", "1.2")) * scale))
        box = (x, y, x + width, y + height)
        if "rhombus" in style:
            draw.polygon(
                [(x + width // 2, y), (x + width, y + height // 2), (x + width // 2, y + height), (x, y + height // 2)],
                fill=fill,
                outline=None if stroke == "none" else stroke,
                width=stroke_width,
            )
        elif "rounded=1" in style:
            draw.rounded_rectangle(
                box,
                radius=7 * scale,
                fill=None if fill == "none" else fill,
                outline=None if stroke == "none" else stroke,
                width=stroke_width,
            )
        else:
            draw.rectangle(
                box,
                fill=None if fill == "none" else fill,
                outline=None if stroke == "none" else stroke,
                width=stroke_width,
            )
        value = str(vertex["value"])
        if value:
            font_size = int(float(self._style_value(style, "fontSize", "14")))
            bold = self._style_value(style, "fontStyle", "0") == "1" or "<b>" in value
            align_left = self._style_value(style, "align", "center") == "left"
            self._draw_text_pillow(draw, value, box, font_size, bold, scale, align_left=align_left)

    def _append_svg_vertex(self, svg: ET.Element, vertex: dict[str, object]) -> None:
        x = float(vertex["x"])
        y = float(vertex["y"])
        width = float(vertex["width"])
        height = float(vertex["height"])
        style = str(vertex["style"])
        value = str(vertex["value"])
        fill = self._style_value(style, "fillColor", WHITE)
        stroke = self._style_value(style, "strokeColor", LINE)
        stroke_width = self._style_value(style, "strokeWidth", "1.2")
        no_stroke = stroke == "none"
        if "rhombus" in style:
            points = f"{x + width / 2},{y} {x + width},{y + height / 2} {x + width / 2},{y + height} {x},{y + height / 2}"
            shape = ET.SubElement(svg, "polygon", {"points": points, "fill": fill})
        else:
            shape = ET.SubElement(
                svg,
                "rect",
                {
                    "x": str(x),
                    "y": str(y),
                    "width": str(width),
                    "height": str(height),
                    "rx": "7" if "rounded=1" in style else "0",
                    "fill": "none" if fill == "none" else fill,
                },
            )
        shape.set("stroke", "none" if no_stroke else stroke)
        shape.set("stroke-width", stroke_width)
        if self._style_value(style, "dashed", "0") == "1":
            shape.set("stroke-dasharray", "7 5")
        if value:
            font_size = int(float(self._style_value(style, "fontSize", "14")))
            bold = self._style_value(style, "fontStyle", "0") == "1"
            align_left = self._style_value(style, "align", "center") == "left"
            self._append_svg_text(svg, value, x, y, width, height, font_size, bold, align_left=align_left)

    @staticmethod
    def _append_svg_text(
        svg: ET.Element,
        value: str,
        x: float,
        y: float,
        width: float,
        height: float,
        font_size: int,
        bold: bool,
        *,
        align_left: bool = False,
    ) -> None:
        foreign = ET.SubElement(
            svg,
            "foreignObject",
            {"x": str(x), "y": str(y), "width": str(width), "height": str(height)},
        )
        body = ET.SubElement(
            foreign,
            "{http://www.w3.org/1999/xhtml}div",
            {
                "style": (
                    "width:100%;height:100%;box-sizing:border-box;padding:4px;"
                    "display:flex;align-items:center;"
                    f"justify-content:{'flex-start' if align_left else 'center'};"
                    f"text-align:{'left' if align_left else 'center'};"
                    f"font-family:Arial,sans-serif;font-size:{font_size}px;"
                    f"font-weight:{'700' if bold else '400'};line-height:1.22;color:#222;"
                    "white-space:pre-line;"
                )
            },
        )
        plain_text = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
        plain_text = re.sub(r"<[^>]+>", "", plain_text)
        body.text = html.unescape(plain_text)


def build_figure_1() -> Path:
    diagram = Diagram("Fig1 online architecture", 1600, 830)

    columns = [
        (25, 35, 275, 760, "低功耗摄像头 / 端侧", EDGE_FILL),
        (315, 35, 285, 760, "CAFRU 上传决策", EDGE_FILL),
        (615, 35, 285, 760, "传输与实时调度", TRANSPORT_FILL),
        (915, 35, 255, 760, "云端复核", CLOUD_FILL),
        (1185, 35, 385, 760, "融合与输出", OUTPUT_FILL),
    ]
    for index, (x, y, width, height, title, fill) in enumerate(columns):
        diagram.vertex("", x, y, width, height, background_style(), id_=f"col{index}")
        diagram.vertex(title, x, y, width, 48, header_style(fill), id_=f"head{index}")

    capture = diagram.vertex("视频帧采集", 60, 115, 205, 54, cell_style(EDGE_FILL), id_="capture")
    decode = diagram.vertex("解码 / NV12-RGB 转换", 60, 200, 205, 58, cell_style(EDGE_FILL), id_="decode")
    edge_yolo = diagram.vertex("<b>YOLO26s</b><br>端侧检测", 60, 300, 205, 68, cell_style(EDGE_FILL), id_="edge_yolo")
    edge_boxes = diagram.vertex("B_e = {class,<br>confidence, box}", 60, 405, 205, 70, cell_style(WHITE, font_size=17), id_="edge_boxes")

    frame_feat = diagram.vertex("帧级特征<br><b>15-D x_f</b>", 335, 125, 115, 72, cell_style(EDGE_FILL, font_size=17), id_="frame_feat")
    roi_feat = diagram.vertex("ROI 级特征<br><b>15-D x_r,i</b>", 465, 125, 115, 72, cell_style(EDGE_FILL, font_size=17), id_="roi_feat")
    heads = diagram.vertex("概率头 + 收益头<br>p_f, ĝ_f, p_r,i, ĝ_r,i", 345, 235, 225, 78, cell_style(EDGE_FILL, bold=True, font_size=17), id_="heads")
    score = diagram.vertex("收益 - 代价评分<br>S = p · max(0, ĝ) - λC", 345, 345, 225, 78, cell_style(WHITE, font_size=17), id_="score")
    decision = diagram.vertex("整帧 / ROI<br>/ 不上传", 390, 485, 135, 105, diamond_style(EDGE_FILL, font_size=16), id_="decision")

    jpeg = diagram.vertex("JPEG 编码", 650, 120, 215, 54, cell_style(TRANSPORT_FILL), id_="jpeg")
    queue = diagram.vertex("<b>有界队列 Q</b><br>keep-latest / drop-new", 650, 215, 215, 78, cell_style(TRANSPORT_FILL, font_size=18), id_="queue")
    deadline = diagram.vertex("deadline<br>准入检查", 680, 335, 155, 82, diamond_style(WHITE, red_border=True, font_size=16), id_="deadline")
    http = diagram.vertex("HTTP multipart 请求", 650, 485, 215, 56, cell_style(TRANSPORT_FILL), id_="http")
    runtime = diagram.vertex(
        "运行时状态<br>payload<br>queue length<br>result age",
        690,
        625,
        135,
        105,
        cell_style(WHITE, dashed=True, font_size=16),
        id_="runtime",
    )

    parse = diagram.vertex("请求解析与解码", 940, 125, 205, 56, cell_style(CLOUD_FILL), id_="parse")
    cloud_yolo = diagram.vertex("<b>YOLO26x</b><br>云端复核", 940, 235, 205, 68, cell_style(CLOUD_FILL, bold=True), id_="cloud_yolo")
    remap = diagram.vertex("ROI 坐标回映射<br><font color=\"#666666\">仅 ROI 请求</font>", 940, 355, 205, 74, cell_style(CLOUD_FILL, font_size=18), id_="remap")
    json_result = diagram.vertex("JSON 检测结果<br>+ frame id / timing", 940, 485, 205, 72, cell_style(WHITE, font_size=17), id_="json")

    age_gate = diagram.vertex("result age<br>≤ deadline?", 1235, 115, 140, 100, diamond_style(WHITE, red_border=True, font_size=17), id_="age_gate")
    stale = diagram.vertex("过期结果丢弃", 1410, 130, 125, 64, cell_style(WHITE, font_size=17), id_="stale")
    fusion = diagram.vertex("ROI 内 IoU 匹配<br>端侧框保留 / 云端框替换或补充", 1220, 300, 315, 88, cell_style(OUTPUT_FILL, font_size=17), id_="fusion")
    nms = diagram.vertex("类别感知 NMS", 1265, 445, 225, 58, cell_style(OUTPUT_FILL), id_="nms")
    output = diagram.vertex("<b>带框视频 / 事件结果</b>", 1265, 575, 225, 68, cell_style(OUTPUT_FILL, bold=True), id_="output")

    diagram.edge(capture, decode)
    diagram.edge(decode, edge_yolo)
    diagram.edge(edge_yolo, edge_boxes)
    diagram.edge(decode, frame_feat, points=[(290, 228), (290, 158)])
    diagram.edge(edge_boxes, frame_feat, points=[(290, 434), (325, 434), (325, 158)])
    diagram.edge(edge_boxes, roi_feat, points=[(290, 434), (300, 434), (300, 475), (590, 475), (590, 158)])
    diagram.edge(frame_feat, heads)
    diagram.edge(roi_feat, heads)
    diagram.edge(heads, score)
    diagram.edge(score, decision)

    diagram.edge(
        decision,
        jpeg,
        label="整帧 / ROI",
        points=[(600, 537), (600, 95), (757, 95)],
    )
    diagram.edge(jpeg, queue)
    diagram.edge(queue, deadline)
    diagram.edge(deadline, http, label="接纳")
    diagram.edge(runtime, queue, label="状态反馈", style=edge_style(dashed=True))
    diagram.edge(http, parse, points=[(900, 513), (900, 153)])
    diagram.edge(parse, cloud_yolo)
    diagram.edge(cloud_yolo, remap)
    diagram.edge(remap, json_result)
    diagram.edge(json_result, age_gate, points=[(1175, 521), (1175, 165)])
    diagram.edge(age_gate, stale, label="否", style=edge_style(dashed=True, dash_pattern="8 4 2 4", red=True))
    diagram.edge(age_gate, fusion, label="是")
    diagram.edge(
        decision,
        fusion,
        label="不上云：复用 B_e",
        points=[(600, 610), (1175, 610), (1175, 344)],
    )
    diagram.edge(fusion, nms)
    diagram.edge(nms, output)

    diagram.vertex("实线：在线数据流", 60, 735, 200, 38, "shape=label;html=1;fontFamily=Arial;fontSize=16;align=left;strokeColor=none;fillColor=none;", id_="legend1")
    diagram.vertex("虚线：运行时状态", 340, 735, 200, 38, "shape=label;html=1;fontFamily=Arial;fontSize=16;align=left;strokeColor=none;fillColor=none;", id_="legend2")
    diagram.vertex("点划线红色：迟到 / 丢弃", 650, 750, 230, 38, "shape=label;html=1;fontFamily=Arial;fontSize=16;align=left;strokeColor=none;fillColor=none;", id_="legend3")

    path = OUTPUT / "Fig1_online_edge_cloud_architecture.drawio"
    diagram.save(path)
    diagram.save_svg(path.with_suffix(".svg"))
    export_base = OUTPUT.parent / "Fig1_online_edge_cloud_architecture"
    diagram.save_png_pdf(export_base.with_suffix(".png"), export_base.with_suffix(".pdf"))
    diagram.replace_text(
        {
            "head0": "Edge Camera",
            "head1": "CAFRU Admission",
            "head2": "Transport / Scheduling",
            "head3": "Cloud Review",
            "head4": "Fusion & Output",
            "capture": "Video Frame Capture",
            "decode": "Decode /<br>NV12-to-RGB",
            "edge_yolo": "<b>YOLO26s</b><br>Edge Detection",
            "frame_feat": "Frame<br>Features<br><b>15-D x_f</b>",
            "roi_feat": "ROI<br>Features<br><b>15-D x_r,i</b>",
            "heads": "Probability +<br>Gain Heads<br>p_f, ĝ_f, p_r,i, ĝ_r,i",
            "score": "Utility-Cost Score<br>S = p · max(0, ĝ) - λC",
            "decision": "Full Frame / ROI<br>/ Local Only",
            "jpeg": "JPEG Encoding",
            "queue": "<b>Bounded Queue Q</b><br>keep-latest / drop-new",
            "deadline": "Admission<br>Check",
            "http": "HTTP Multipart<br>Request",
            "runtime": "Runtime State<br>payload<br>queue length<br>result age",
            "parse": "Request Parsing<br>and Decoding",
            "cloud_yolo": "<b>YOLO26x</b><br>Cloud Review",
            "remap": "ROI Coordinate<br>Remapping<br><font color=\"#666666\">ROI requests only</font>",
            "json": "JSON Detections<br>+ frame id / timing",
            "stale": "Discard Stale Result",
            "fusion": "IoU Match within ROI<br>Retain / Replace / Add Boxes",
            "nms": "Class-Aware NMS",
            "output": "<b>Annotated Video<br>/ Events</b>",
            "legend1": "Solid: online data flow",
            "legend2": "Dashed: runtime state",
            "legend3": "Red dash-dot: stale / discard",
        },
        {
            "整帧 / ROI": "Full Frame / ROI",
            "接纳": "Admit",
            "状态反馈": "Runtime State",
            "否": "No",
            "是": "Yes",
            "不上云：复用 B_e": "Local: Reuse B_e",
        },
    )
    english_path = OUTPUT / "Fig1_online_edge_cloud_architecture_en.drawio"
    diagram.save(english_path)
    diagram.save_svg(english_path.with_suffix(".svg"))
    english_base = OUTPUT.parent / "Fig1_online_edge_cloud_architecture_en"
    diagram.save_png_pdf(english_base.with_suffix(".png"), english_base.with_suffix(".pdf"))
    return path


def build_figure_2() -> Path:
    diagram = Diagram("Fig2 offline training and online decision", 1600, 860)

    diagram.vertex("", 25, 35, 1550, 350, background_style(), id_="offline_bg")
    diagram.vertex("离线训练阶段", 25, 35, 1550, 48, header_style(TRAIN_FILL), id_="offline_header")
    diagram.vertex("", 25, 425, 1550, 400, background_style(), id_="online_bg")
    diagram.vertex("在线部署阶段（参数冻结，不在线训练）", 25, 425, 1550, 48, header_style(EDGE_FILL), id_="online_header")

    data = diagram.vertex(
        "<b>COCO train2017</b><br>人工标注 Y",
        55,
        135,
        165,
        72,
        cell_style(TRAIN_FILL, bold=True, font_size=18),
        id_="train_data",
    )
    edge_pred = diagram.vertex(
        "YOLO26s<br>端侧预测 B_e",
        270,
        105,
        170,
        65,
        cell_style(EDGE_FILL, font_size=18),
        id_="train_edge",
    )
    cloud_pred = diagram.vertex(
        "YOLO26x<br>整帧 / ROI 复核",
        270,
        210,
        170,
        68,
        cell_style(CLOUD_FILL, font_size=18),
        id_="train_cloud",
    )
    utility = diagram.vertex(
        "目标级效用<br>U = TP - αFP<br>before / after review",
        490,
        130,
        185,
        92,
        cell_style(WHITE, font_size=17),
        id_="utility",
    )
    labels = diagram.vertex(
        "<b>收益监督</b><br>g = U_after - U_before<br>z = 1[g &gt; 0]",
        720,
        130,
        190,
        92,
        cell_style(TRAIN_FILL, bold=True, font_size=17),
        id_="gain_labels",
    )
    frame_features = diagram.vertex(
        "15-D 帧特征 x_f",
        960,
        105,
        170,
        58,
        cell_style(EDGE_FILL, font_size=17),
        id_="train_frame_features",
    )
    roi_features = diagram.vertex(
        "15-D ROI 特征 x_r,i",
        960,
        205,
        170,
        58,
        cell_style(EDGE_FILL, font_size=17),
        id_="train_roi_features",
    )
    split = diagram.vertex(
        "90% 训练 / 10% 验证<br>seed = 42",
        960,
        290,
        170,
        60,
        cell_style(WHITE, dashed=True, font_size=16),
        id_="split",
    )
    fit_heads = diagram.vertex(
        "<b>四个轻量线性头</b><br>概率头：balanced logistic + L2<br>收益头：Huber linear + L2",
        1175,
        120,
        225,
        118,
        cell_style(TRAIN_FILL, bold=True, font_size=16),
        id_="fit_heads",
    )
    export = diagram.vertex(
        "导出参数<br><b>JSON / C++ 常量</b>",
        1415,
        140,
        135,
        80,
        cell_style(TRAIN_FILL, bold=True, font_size=15),
        id_="export",
    )

    current = diagram.vertex(
        "当前帧 I_t<br>+ 端侧框 B_e",
        60,
        545,
        180,
        72,
        cell_style(EDGE_FILL, font_size=18),
        id_="current",
    )
    online_features = diagram.vertex(
        "计算 x_f 与 {x_r,i}<br>图像 / 检测统计",
        285,
        545,
        190,
        72,
        cell_style(EDGE_FILL, font_size=17),
        id_="online_features",
    )
    frozen_heads = diagram.vertex(
        "<b>四头冻结推理</b><br>p_f, ĝ_f, p_r,i, ĝ_r,i",
        525,
        545,
        195,
        72,
        cell_style(EDGE_FILL, bold=True, font_size=17),
        id_="frozen_heads",
    )
    scores = diagram.vertex(
        "收益 - payload 代价<br>S_f = p_f max(0, ĝ_f) - λ_b C_f<br>S_r = p_r max(0, ĝ_r) - λ_b C_r",
        765,
        525,
        235,
        110,
        cell_style(WHITE, font_size=15),
        id_="scores",
    )
    decision = diagram.vertex(
        "阈值满足?<br>队列可用?",
        1045,
        525,
        150,
        110,
        diamond_style(EDGE_FILL, font_size=16),
        id_="online_decision",
    )
    full_frame = diagram.vertex(
        "上传整帧",
        1250,
        500,
        150,
        58,
        cell_style(TRANSPORT_FILL, bold=True, font_size=18),
        id_="full_frame",
    )
    roi_upload = diagram.vertex(
        "上传最高分 ROI",
        1250,
        590,
        150,
        58,
        cell_style(TRANSPORT_FILL, bold=True, font_size=17),
        id_="roi_upload",
    )
    skip = diagram.vertex(
        "不上云<br>复用端侧结果",
        1250,
        680,
        150,
        68,
        cell_style(OUTPUT_FILL, bold=True, font_size=17),
        id_="skip",
    )
    scheduler_style = cell_style(WHITE, dashed=True, font_size=16).replace(f"strokeColor={LINE}", f"strokeColor={RED}")
    scheduler = diagram.vertex(
        "运行时调度器<br>queue length / deadline<br>result age<br><b>可否决上传或丢弃迟到结果</b>",
        785,
        690,
        270,
        96,
        scheduler_style,
        id_="scheduler",
    )
    boundary = diagram.vertex(
        "网络状态不是当前学习头输入<br>payload 为显式代价；delay 为调度变量",
        1070,
        755,
        360,
        52,
        "shape=label;html=1;fontFamily=Arial;fontSize=14;align=left;strokeColor=none;fillColor=none;",
        id_="boundary_note",
    )

    diagram.edge(data, edge_pred)
    diagram.edge(data, cloud_pred)
    diagram.edge(edge_pred, utility)
    diagram.edge(cloud_pred, utility)
    diagram.edge(utility, labels)
    diagram.edge(edge_pred, frame_features, points=[(460, 137), (930, 137)])
    diagram.edge(edge_pred, roi_features, points=[(460, 137), (460, 246), (930, 246)])
    diagram.edge(labels, fit_heads, label="g, z 监督")
    diagram.edge(frame_features, fit_heads)
    diagram.edge(roi_features, fit_heads)
    diagram.edge(split, fit_heads)
    diagram.edge(fit_heads, export)

    diagram.edge(current, online_features)
    diagram.edge(online_features, frozen_heads)
    diagram.edge(frozen_heads, scores)
    diagram.edge(scores, decision)
    diagram.edge(decision, full_frame, label="帧分数通过")
    diagram.edge(decision, roi_upload, label="ROI 通过")
    diagram.edge(decision, skip, label="均不满足")
    diagram.edge(scheduler, decision, label="运行时门控", style=edge_style(dashed=True, red=True))
    diagram.edge(
        export,
        frozen_heads,
        label="离线部署",
        style=edge_style(dashed=True),
        points=[(1490, 395), (625, 395), (625, 520)],
    )

    path = OUTPUT / "Fig2_cafru_offline_training_online_decision.drawio"
    diagram.save(path)
    diagram.save_svg(path.with_suffix(".svg"))
    export_base = OUTPUT.parent / "Fig2_cafru_offline_training_online_decision"
    diagram.save_png_pdf(export_base.with_suffix(".png"), export_base.with_suffix(".pdf"))
    diagram.replace_text(
        {
            "offline_header": "Offline Training",
            "online_header": "Online Deployment (Frozen Parameters)",
            "train_data": "<b>COCO train2017</b><br>Manual Labels Y",
            "train_edge": "YOLO26s<br>Edge Output B_e",
            "train_cloud": "YOLO26x<br>Frame / ROI",
            "utility": "Object Utility<br>U = TP - αFP<br>before / after",
            "gain_labels": "<b>Gain Labels</b><br>g = U_after - U_before<br>z = 1[g &gt; 0]",
            "train_frame_features": "15-D Frame<br>Features x_f",
            "train_roi_features": "15-D ROI<br>Features x_r,i",
            "split": "Train 90% / Val. 10%<br>seed = 42",
            "fit_heads": "<b>Four Linear Heads</b><br>probability:<br>balanced logistic + L2<br>gain: Huber linear + L2",
            "export": "Export<br><b>JSON / C++</b>",
            "current": "Frame I_t<br>+ Edge Boxes B_e",
            "online_features": "Compute x_f, {x_r,i}<br>Image / Box Stats",
            "frozen_heads": "<b>Frozen Four-Head<br>Inference</b><br>p_f, ĝ_f, p_r,i, ĝ_r,i",
            "scores": "Utility - Payload Cost<br>S_f = p_f max(0, ĝ_f) - λ_b C_f<br>S_r = p_r max(0, ĝ_r) - λ_b C_r",
            "online_decision": "Thresholds?<br>Queue?",
            "full_frame": "Full Frame",
            "roi_upload": "Top ROI",
            "skip": "Local Only<br>Use Edge",
            "scheduler": "Runtime Scheduler<br>queue / deadline / age<br><b>Reject overflow / stale</b>",
            "boundary_note": "Network state is not learned<br>payload: explicit cost; delay: runtime gate",
        },
        {
            "g, z 监督": "g, z labels",
            "帧分数通过": "Frame",
            "ROI 通过": "ROI",
            "均不满足": "Local",
            "运行时门控": "Runtime",
            "离线部署": "Offline export",
        },
    )
    english_path = OUTPUT / "Fig2_cafru_offline_training_online_decision_en.drawio"
    diagram.save(english_path)
    diagram.save_svg(english_path.with_suffix(".svg"))
    english_base = OUTPUT.parent / "Fig2_cafru_offline_training_online_decision_en"
    diagram.save_png_pdf(english_base.with_suffix(".png"), english_base.with_suffix(".pdf"))
    return path


def build_figure_5() -> Path:
    diagram = Diagram("Fig5 T41 bounded queue and deadline", 1600, 850)

    diagram.vertex("", 25, 35, 1275, 235, background_style(), id_="main_bg")
    diagram.vertex("T41 主采集 / 检测线程", 25, 35, 1275, 48, header_style(EDGE_FILL), id_="main_header")
    diagram.vertex("", 25, 300, 1275, 300, background_style(), id_="worker_bg")
    diagram.vertex("T41 上传工作线程", 25, 300, 1275, 48, header_style(TRANSPORT_FILL), id_="worker_header")
    diagram.vertex("", 25, 630, 1275, 180, background_style(), id_="cloud_bg")
    diagram.vertex("云端服务", 25, 630, 1275, 48, header_style(CLOUD_FILL), id_="cloud_header")

    capture = diagram.vertex("采集帧 t", 55, 125, 125, 58, cell_style(EDGE_FILL, font_size=18), id_="m_capture")
    edge_yolo = diagram.vertex("YOLO26s", 220, 125, 125, 58, cell_style(EDGE_FILL, bold=True, font_size=19), id_="m_yolo")
    admission = diagram.vertex(
        "CAFRU / 规则准入",
        385,
        125,
        155,
        58,
        cell_style(EDGE_FILL, font_size=17),
        id_="m_admission",
    )
    capacity = diagram.vertex(
        "队列有容量?",
        575,
        105,
        145,
        100,
        diamond_style(WHITE, red_border=True, font_size=17),
        id_="m_capacity",
    )
    enqueue = diagram.vertex(
        "写入候选任务",
        755,
        105,
        145,
        58,
        cell_style(TRANSPORT_FILL, font_size=17),
        id_="m_enqueue",
    )
    local_skip = diagram.vertex(
        "本地跳过上传",
        755,
        190,
        145,
        50,
        cell_style(OUTPUT_FILL, font_size=16),
        id_="m_skip",
    )
    edge_output = diagram.vertex(
        "立即输出端侧结果",
        940,
        125,
        155,
        58,
        cell_style(OUTPUT_FILL, font_size=17),
        id_="m_edge_output",
    )
    next_frame = diagram.vertex(
        "进入下一帧",
        1130,
        125,
        130,
        58,
        cell_style(EDGE_FILL, font_size=17),
        id_="m_next",
    )

    take = diagram.vertex(
        "取队首 / 保留最新<br>keep-latest / drop-new",
        80,
        385,
        190,
        72,
        cell_style(TRANSPORT_FILL, font_size=16),
        id_="w_take",
    )
    jpeg = diagram.vertex(
        "JPEG Q70 编码",
        310,
        390,
        150,
        62,
        cell_style(TRANSPORT_FILL, font_size=17),
        id_="w_jpeg",
    )
    timestamp = diagram.vertex(
        "记录 submit<br>timestamp",
        500,
        385,
        150,
        72,
        cell_style(TRANSPORT_FILL, font_size=16),
        id_="w_timestamp",
    )
    http = diagram.vertex(
        "HTTP multipart<br>上传",
        690,
        385,
        150,
        72,
        cell_style(TRANSPORT_FILL, font_size=17),
        id_="w_http",
    )
    receive = diagram.vertex(
        "接收 cloud boxes",
        880,
        390,
        150,
        62,
        cell_style(TRANSPORT_FILL, font_size=17),
        id_="w_receive",
    )
    age = diagram.vertex(
        "result_age =<br>return_time - frame_time",
        1070,
        380,
        185,
        82,
        cell_style(WHITE, font_size=15),
        id_="w_age",
    )
    gate = diagram.vertex(
        "result_age<br>≤ deadline D?",
        900,
        485,
        155,
        100,
        diamond_style(WHITE, red_border=True, font_size=16),
        id_="w_gate",
    )
    timely = diagram.vertex(
        "及时结果<br>进入融合缓存",
        690,
        500,
        155,
        68,
        cell_style(OUTPUT_FILL, bold=True, font_size=16),
        id_="w_timely",
    )
    stale = diagram.vertex(
        "过期结果丢弃",
        1080,
        510,
        150,
        58,
        cell_style(WHITE, font_size=17),
        id_="w_stale",
    )

    cloud_decode = diagram.vertex(
        "请求解码",
        690,
        710,
        150,
        58,
        cell_style(CLOUD_FILL, font_size=18),
        id_="c_decode",
    )
    cloud_yolo = diagram.vertex(
        "YOLO26x",
        880,
        710,
        150,
        58,
        cell_style(CLOUD_FILL, bold=True, font_size=19),
        id_="c_yolo",
    )
    cloud_return = diagram.vertex(
        "返回框 + server timing",
        1070,
        705,
        185,
        68,
        cell_style(CLOUD_FILL, font_size=16),
        id_="c_return",
    )

    metric_style = cell_style(WHITE, font_size=15).replace(f"strokeColor={LINE}", f"strokeColor={MUTED_LINE}")
    metrics = diagram.vertex(
        "<b>实时指标定义</b><br><br>"
        "scheduling-loop FPS<br>= processed frames<br>/ wall time<br><br>"
        "timely-frame ratio<br>= timely frames<br>/ processed frames<br><br>"
        "effective FPS@D<br>= timely frames<br>/ wall time<br><br>"
        "drop ratio<br>= discarded<br>/ upload candidates",
        1330,
        105,
        240,
        600,
        metric_style,
        id_="metrics",
    )
    note = diagram.vertex(
        "云端返回成功 ≠ 实时有效<br>deadline 内结果才进入融合",
        1330,
        735,
        240,
        65,
        "shape=label;html=1;fontFamily=Arial;fontSize=16;fontStyle=1;align=center;strokeColor=none;fillColor=none;",
        id_="deadline_note",
    )

    diagram.edge(capture, edge_yolo)
    diagram.edge(edge_yolo, admission)
    diagram.edge(admission, capacity)
    diagram.edge(capacity, enqueue, label="是")
    diagram.edge(capacity, local_skip, label="否")
    diagram.edge(enqueue, edge_output)
    diagram.edge(local_skip, edge_output)
    diagram.edge(edge_output, next_frame)
    diagram.edge(next_frame, capture, label="连续视频", points=[(1280, 154), (1280, 245), (45, 245), (45, 154)])

    diagram.edge(enqueue, take, label="异步队列", points=[(825, 280), (175, 280), (175, 360)])
    diagram.edge(take, jpeg)
    diagram.edge(jpeg, timestamp)
    diagram.edge(timestamp, http)
    diagram.edge(http, cloud_decode, label="请求", points=[(765, 610), (765, 685)])
    diagram.edge(cloud_decode, cloud_yolo)
    diagram.edge(cloud_yolo, cloud_return)
    diagram.edge(cloud_return, receive, label="返回", points=[(1280, 739), (1280, 340), (955, 340), (955, 365)])
    diagram.edge(receive, age)
    diagram.edge(age, gate)
    diagram.edge(gate, timely, label="及时")
    diagram.edge(gate, stale, label="过期", style=edge_style(dashed=True, dash_pattern="8 4 2 4", red=True))

    path = OUTPUT / "Fig5_t41_bounded_queue_deadline.drawio"
    diagram.save(path)
    diagram.save_svg(path.with_suffix(".svg"))
    export_base = OUTPUT.parent / "Fig5_t41_bounded_queue_deadline"
    diagram.save_png_pdf(export_base.with_suffix(".png"), export_base.with_suffix(".pdf"))
    diagram.replace_text(
        {
            "main_header": "T41 Capture / Detection Thread",
            "worker_header": "T41 Upload Worker",
            "cloud_header": "Cloud Service",
            "m_capture": "Frame t",
            "m_admission": "CAFRU / Rule<br>Admission",
            "m_capacity": "Queue<br>Capacity?",
            "m_enqueue": "Enqueue",
            "m_skip": "Skip Upload",
            "m_edge_output": "Emit Edge Result",
            "m_next": "Next Frame",
            "w_take": "Take Head / Latest<br>keep-latest / drop-new",
            "w_jpeg": "JPEG Q70",
            "w_timestamp": "Submit<br>Timestamp",
            "w_http": "HTTP Multipart<br>Upload",
            "w_receive": "Cloud Boxes",
            "w_timely": "Timely Result<br>to Fusion Cache",
            "w_stale": "Discard Stale",
            "c_decode": "Decode Request",
            "c_return": "Return Boxes<br>+ Server Timing",
            "metrics": "<b>Real-Time Metrics</b><br><br>"
            "scheduling-loop FPS<br>= processed frames<br>/ wall time<br><br>"
            "timely-frame ratio<br>= timely frames<br>/ processed frames<br><br>"
            "effective FPS@D<br>= timely frames<br>/ wall time<br><br>"
            "drop ratio<br>= discarded<br>/ upload candidates",
            "deadline_note": "Cloud return ≠ timely<br>Only age ≤ D enters fusion",
        },
        {
            "是": "Yes",
            "否": "No",
            "连续视频": "Stream",
            "异步队列": "Async Queue",
            "请求": "Request",
            "返回": "Return",
            "及时": "Timely",
            "过期": "Stale",
        },
    )
    english_path = OUTPUT / "Fig5_t41_bounded_queue_deadline_en.drawio"
    diagram.save(english_path)
    diagram.save_svg(english_path.with_suffix(".svg"))
    english_base = OUTPUT.parent / "Fig5_t41_bounded_queue_deadline_en"
    diagram.save_png_pdf(english_base.with_suffix(".png"), english_base.with_suffix(".pdf"))
    return path


def main() -> None:
    for path in [build_figure_1(), build_figure_2(), build_figure_5()]:
        print(path)


if __name__ == "__main__":
    main()
