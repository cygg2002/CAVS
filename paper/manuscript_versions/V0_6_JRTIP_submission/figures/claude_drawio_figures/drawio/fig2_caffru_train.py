"""Fig 2 (print, grid) — CAFRU 离线训练与在线决策.

Grid-aligned: supervision spine + parallel predictors share rows so the main
flow is straight horizontals/verticals. Two heads branches aligned. Dashed
'离线部署' export -> frozen-heads boundary; online never returns to training.
Print scale (8 pt). Legend to caption.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/.claude/skills/drawio/scripts"))
import drawio
import jrtip_style as J

BODY, HEAD, EDGE, SW = 8, 9, 8, 1.0
W, DW, DH = 78, 78, 52
d = drawio.Diagram("Fig2_caffru_train_online", page=(560, 420), preset="color")
E = d.edge
SOL = J.main_flow(sw=SW, size=EDGE); CTRL = J.ctrl_flow(sw=SW, size=EDGE)
DASH = J.offline_dashed(sw=SW, size=EDGE)

d.node("lane_off", "离线训练阶段 (offline training)", style=J.band("offline", size=HEAD), w=540, h=14)
d.nodes["lane_off"].x, d.nodes["lane_off"].y = 8, 12
d.node("lane_on", "在线部署阶段 (online deployment)", style=J.band("edge", size=HEAD), w=540, h=14)
d.nodes["lane_on"].x, d.nodes["lane_on"].y = 8, 230

def n(k, label, cat, x, y, w=W, h=28, bold=False, sub=None, diamond=False):
    st = J.diamond(cat, size=BODY, sw=SW) if diamond else J.box(cat, size=BODY, bold=bold, sw=SW)
    hh = DH if diamond else h
    d.node(k, label, subtitle=sub, style=st, w=(DW if diamond else w), h=hh)
    d.nodes[k].x, d.nodes[k].y = x, y

OFF, ON = "offline", "edge"
# ---- offline: supervision spine (row 40) + parallel (row 74) + branches (118/152) ----
n("o1", "COCO train2017", OFF, 8, 40, w=86, h=40, sub="+ 人工标注")
n("o2", "YOLO26s 预测 B_e", OFF, 100, 40)
n("o3", "YOLO26x 预测 B_c", OFF, 100, 74)
n("o4", "端侧效用 U(B_e)", OFF, 188, 40)
n("o5", "云端效用 U(B_c)", OFF, 188, 74)
n("o6", "收益标签 g", OFF, 276, 40, h=40, sub="U_after−U_before")
n("o7", "正收益 z", OFF, 364, 40, h=40, sub="1[g > 0]")
n("o8",  "帧特征 x_f", OFF, 100, 118, sub="(15 维)")
n("o9",  "帧概率头 p_f", OFF, 188, 118)
n("o10", "帧收益头 g_f", OFF, 276, 118)
n("o11", "ROI 特征 x_r", OFF, 100, 152, sub="(15 维)")
n("o12", "ROI 概率头 p_r", OFF, 188, 152)
n("o13", "ROI 收益头 g_r", OFF, 276, 152)
n("o14", "90/10 划分", OFF, 364, 118, h=40, sub="seed = 42")
n("o15", "逻辑回归 + L2", OFF, 452, 110, h=40, sub="类别平衡")
n("o16", "Huber + L2", OFF, 452, 156)
n("o17", "导出参数", OFF, 452, 190, h=34, bold=True, sub="JSON / C++")

# ---- online lane (row 268) ----
n("m1", "当前帧 I_t + B_e", ON, 8, 268, w=92)
n("m2", "计算 x_f, {x_r}", ON, 106, 268)
n("m3", "四个线性头", ON, 194, 268, h=40, bold=True, sub="(frozen)")
n("m4", "帧分数 S_f", ON, 282, 268)
n("m5", "ROI 分数 S_r,i", ON, 370, 268)
n("m6", "上传选择", "decision", 458, 260, diamond=True)
n("m7", "运行时调度器", ON, 458, 324, h=40, sub="queue/deadline/age")

# ---- edges ----
# supervision spine (straight horizontals)
E("o1","o2", style=SOL); E("o2","o4", style=SOL); E("o4","o6", style=SOL); E("o6","o7", style=SOL)
# parallel predictors/utilities
E("o1","o3", style=SOL, exit=("bottom","center"), entry=("top","center"))
E("o3","o5", style=SOL); E("o5","o6", style=SOL, exit=("right","center"), entry=("left","center"))
# features drop straight down from predictions
E("o2","o8", style=CTRL, exit=("bottom","center"), entry=("top","center"))
E("o3","o11", style=CTRL, exit=("bottom","center"), entry=("top","center"))
# branch internals (straight horizontals)
E("o8","o9", style=SOL); E("o9","o10", style=SOL)
E("o11","o12", style=SOL); E("o12","o13", style=SOL)
# supervision (z->prob head, g->gain head) — g->g_f is a clean vertical (same col)
E("o6","o10", style=CTRL, label="监督 g", exit=("bottom","center"), entry=("top","center"))
E("o7","o12", style=CTRL, label="监督 z", exit=("bottom","center"), entry=("top","center"))
# branches -> split -> trainers -> export
E("o10","o14", style=SOL); E("o13","o14", style=SOL, exit=("right","center"), entry=("left","center"))
E("o14","o15", style=SOL); E("o14","o16", style=SOL, exit=("bottom","center"), entry=("top","center"))
E("o15","o17", style=SOL, exit=("bottom","left"), entry=("top","left"))
E("o16","o17", style=SOL, exit=("bottom","right"), entry=("top","right"))
# online spine (straight horizontals)
E("m1","m2", style=SOL); E("m2","m3", style=SOL); E("m3","m4", style=SOL)
E("m4","m5", style=SOL); E("m5","m6", style=SOL); E("m6","m7", style=CTRL, label="可否决/丢弃")
# offline export -> online frozen heads (dashed boundary)
E("o17","m3", color=J.LINE, style=DASH, label="离线部署 (frozen)",
  exit=("bottom","center"), entry=("top","center"))

# boundary note (compact)
d.node("nb", "边界:概率/收益头输入为图像与检测统计特征;payload 与 delay 是显式代价/调度变量(非训练进模型);在线路径不返回训练集或优化器。",
       style=J.note(EDGE, "#7F7F7F"), w=540, h=24)
maxy = max(nn.y + nn.h for kk, nn in d.nodes.items() if kk != "nb")
d.nodes["nb"].x = 8; d.nodes["nb"].y = maxy + 8

out = os.path.join(HERE, "Fig2_caffru_train_online.drawio")
d.to_drawio(out)
rep = d.validate()
print(rep.text())
assert rep.ok, "VALIDATE FAILED"
print("wrote", out)
