"""Fig 5 (print, vertical lanes) — T41 有界队列与 deadline 处理.

Three VERTICAL swimlanes (main thread | upload worker | cloud), flow top->bottom.
Cross-lane connections (enqueue / upload / return) on shared rows -> straight
horizontals. Dash-dot red = stale (late) return dropped. Metric box at bottom.
Print scale (8 pt). Legend to caption.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/.claude/skills/drawio/scripts"))
import drawio
import jrtip_style as J

BODY, HEAD, EDGE, SW = 8, 9, 8, 1.0
W, DW, DH = 140, 140, 46
XCEN = [88, 248, 408]                                  # main | worker | cloud
ROWS = [60, 108, 156, 204, 252, 300, 348, 396, 444, 492, 540, 588]  # step 48

d = drawio.Diagram("Fig5_t41_queue_deadline", page=(500, 700), preset="color")
E = d.edge
SOL = J.main_flow(sw=SW, size=EDGE); CTRL = J.ctrl_flow(sw=SW, size=EDGE)
LATE = J.late_dashdot(sw=SW, size=EDGE)
LB = "labelBackgroundColor=#ffffff"

for i, lab in enumerate(["T41 主线程", "T41 上传工作线程", "云端服务"]):
    cat = ["edge", "queue", "cloud"][i]
    d.node("lane%d" % i, lab, style=J.band(cat, size=HEAD), w=W, h=16)
    d.nodes["lane%d" % i].x = XCEN[i] - W / 2.0
    d.nodes["lane%d" % i].y = 18

def place(key, label, cat, col, row, *, h=34, bold=False, diamond=False, sub=None, size=BODY):
    cx, cy = XCEN[col], ROWS[row]
    w = DW if diamond else W
    hh = DH if diamond else h
    st = J.diamond(cat, size=size, sw=SW) if diamond else J.box(cat, size=size, bold=bold, sw=SW)
    d.node(key, label, subtitle=sub, style=st, w=w, h=hh)
    d.nodes[key].x = cx - w / 2.0
    d.nodes[key].y = cy - hh / 2.0

# main lane (col 0)
place("a1", "采集帧 t",         "edge", 0, 0)
place("a2", "YOLO26s",          "edge", 0, 1, bold=True)
place("a3", "CAFRU / 规则准入",  "edge", 0, 2)
place("a4", "队列有容量 ?",     "decision", 0, 3, diamond=True)
place("a5", "写入任务 / 本地跳过", "edge", 0, 4)
place("a6", "立即输出端侧结果",  "fusion", 0, 5)
# worker lane (col 1)
place("b1", "取队首 / 保留最新", "queue", 1, 4)
place("b2", "JPEG Q70 编码",     "queue", 1, 5)
place("b3", "HTTP 上传",         "queue", 1, 6)
place("b4", "接收 cloud boxes",  "queue", 1, 8)
place("b5", "result_age",        "queue", 1, 9, sub="return−frame_time", size=EDGE)
place("b6", "result_age ≤ deadline ?", "deadline", 1, 10, diamond=True)
place("b7", "及时 → 融合缓存",   "fusion", 1, 11)
place("b8", "过期结果丢弃",      "io", 2, 11)
# cloud lane (col 2)
place("c1", "解码",             "cloud", 2, 6)
place("c2", "YOLO26x",          "cloud", 2, 7, bold=True)
place("c3", "返回框 + timing",   "cloud", 2, 8)

def ev(a, b, **kw): kw.setdefault("exit", ("bottom","center")); kw.setdefault("entry", ("top","center")); return E(a, b, **kw)
def eh(a, b, **kw): kw.setdefault("exit", ("right","center")); kw.setdefault("entry", ("left","center")); return E(a, b, **kw)

ev("a1","a2", style=SOL); ev("a2","a3", style=SOL); ev("a3","a4", style=SOL)
ev("a4","a5", style=SOL, label="有容量"); ev("a5","a6", style=SOL)
eh("a5","b1", style=SOL+";"+LB+";verticalLabelPosition=top", label="入队")  # main -> worker (L->R)
ev("b1","b2", style=SOL); ev("b2","b3", style=SOL); ev("b3","b4", style=SOL)
ev("b4","b5", style=SOL); ev("b5","b6", style=SOL)
eh("b3","c1", style=SOL+";"+LB, label="上传")             # worker -> cloud (L->R)
ev("c1","c2", style=SOL); ev("c2","c3", style=SOL)
E("c3","b4", style=SOL+";"+LB, label="返回",               # cloud -> worker (R->L): exit LEFT, entry RIGHT
  exit=("left","center"), entry=("right","center"))
ev("b6","b7", style=SOL, label="及时")
E("b6","b8", color=J.RED, style=LATE, label="过期",
  exit=("right","center"), entry=("left","center"))

d.node("metrics",
       "main-loop FPS = processed / wall time\n"
       "deadline-hit ratio = timely / returned\n"
       "effective FPS@D = deadline hits / wall time\n"
       "drop ratio = discarded / upload candidates",
       style=J.box("io", size=EDGE, sw=SW), w=300, h=64)
d.nodes["metrics"].x = 100
d.nodes["metrics"].y = ROWS[11] + DH / 2.0 + 14

out = os.path.join(HERE, "Fig5_t41_queue_deadline.drawio")
d.to_drawio(out)
rep = d.validate()
print(rep.text())
assert rep.ok, "VALIDATE FAILED"
print("wrote", out)
