#!/usr/bin/env python3
"""Figure 1 (V0.9): auditable cost-aware edge-cloud architecture, drawio-as-code.

Two filled swimlanes: online deployment (single frozen config) on top,
offline calibration + risk gate below. V0.9 terminology (选择集/门禁集/终评).
No in-figure legend — semantics live in the caption.
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/.claude/skills/drawio/scripts"))
import drawio
import print_style as P

BODY, SW = 8, 1.0
W = 76
XCEN = [48, 132, 216, 300, 384, 468]
Y_ON, Y_OFF = 66, 226

d = drawio.Diagram("fig1_v09_architecture", page=(520, 300), preset="paper")


def place(key, cat, col, y, lines, *, diamond=False, h=48, store=False, bold=False):
    if diamond:
        w, hh = W, 46
        st = P.diamond(cat, size=BODY, sw=SW)
    elif store:
        w, hh = W, h
        st = P.store(cat, size=BODY, sw=SW)
    else:
        w, hh = W, h
        st = P.box(cat, size=BODY, sw=SW, bold=bold)
    d.node(key, lines[0], lines=lines[1:] or None, style=st, w=w, h=hh)
    d.nodes[key].x = XCEN[col] - w / 2.0
    d.nodes[key].y = y - hh / 2.0


# --- online lane -------------------------------------------------------------
place("inp", "io", 0, Y_ON, ["输入图像"], h=40)
place("det", "edge", 1, Y_ON, ["端侧检测器", "YOLO26n"], bold=True, h=40)
place("rank", "edge", 2, Y_ON, ["样本排序器", "8 选 1·分数 s_i"], h=40)
place("bud", "queue", 3, Y_ON, ["预算调度", "三种预算模式"], h=40)
place("act", "transport", 4, Y_ON, ["载荷策略", "Original·Q90·ROI"], h=40)
place("cloud", "cloud", 5, Y_ON, ["云端复核融合", "YOLO26x"], bold=True, h=40)

# --- offline lane ------------------------------------------------------------
place("cache", "offline", 0, Y_OFF, ["冻结缓存", "24 候选/工作点"], store=True, h=46)
place("sel", "offline", 1, Y_OFF, ["配置选择·选择集", "θ_ref → θ_prov"], h=40)
place("gate", "offline", 2, Y_OFF, ["风险门禁·门禁集", "200×配对 bootstrap"], h=40)
place("pass", "decision", 3, Y_OFF, ["门禁", "通过?"], diamond=True)
place("frz", "offline", 4, Y_OFF, ["冻结配置", "θ_prov 或回退 θ_ref"], bold=True, h=40)
place("fin", "fusion", 5, Y_OFF, ["终评·一次评测", "oracle 仅诊断"], h=40)

# bypass as a small terminal fed by a short vertical arrow from the scheduler
d.node("byp", "未接纳：输出端侧结果", style=P.terminal("io", size=7, sw=SW), w=110, h=20)
d.nodes["byp"].x = XCEN[3] - 55
d.nodes["byp"].y = 108

# --- zones -------------------------------------------------------------------
d.group("zon", "在线部署（单个冻结配置）", members=("inp", "det", "rank", "bud", "act", "cloud", "byp"),
        fill=P.zone_fill("edge"), stroke=P.zone_stroke("edge"), font_size=9, pad=10)
d.group("zoff", "离线校准与风险门禁（仅用拟合集 / 选择集 / 门禁集）",
        members=("cache", "sel", "gate", "pass", "frz", "fin"),
        fill=P.zone_fill("offline"), stroke=P.zone_stroke("offline"), font_size=9, pad=10)

# --- edges -------------------------------------------------------------------
H = dict(exit=("right", "center"), entry=("left", "center"))
for a, b in [("inp", "det"), ("det", "rank"), ("rank", "bud"), ("bud", "act"), ("act", "cloud")]:
    d.edge(a, b, style=P.main_flow(sw=SW, size=BODY), **H)
# bypass: not admitted -> edge-only output
d.edge("bud", "byp", style=P.late_dashdot(sw=SW, size=7),
       exit=("center", "bottom"), entry=("center", "top"))

for a, b in [("cache", "sel"), ("sel", "gate"), ("gate", "pass"), ("pass", "frz"), ("frz", "fin")]:
    d.edge(a, b, style=P.ctrl_flow(sw=SW, size=BODY), **H)

# deploy: frozen config -> online lane (same column 4 -> straight vertical)
d.edge("frz", "act", label="部署",
       style=P.offline_dashed(sw=SW, size=BODY),
       exit=("center", "top"), entry=("center", "bottom"))

out = d.to_drawio(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_v09_architecture.drawio"))
report = d.validate()
print(report.text())
assert report.ok, report.text()
print("emitted:", out)
