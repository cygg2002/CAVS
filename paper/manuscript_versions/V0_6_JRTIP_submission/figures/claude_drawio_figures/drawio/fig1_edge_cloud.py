"""Fig 1 (publication) — CAFRU online edge-cloud architecture with async feedback.

TWO-ROW "snake" layout (double-column friendly, ~650px ≈ 172mm wide so 9pt text
stays ~8.5pt when fit into a JRTIP double-column slot — the single-row cascade was
~990px and scaled to ~6pt, unreadable).

  Top row, left → right:   Edge Device ─→ Async Transport
                                       │ (clean vertical fold down the right side)
  Bottom row, right → left: Cloud Reviewer ─→ Fresh Fusion

Every cross-lane handoff is still a straight HORIZONTAL arrow; the only fold is
the single Transport↓Cloud vertical on the right (not a long U/S detour). The
bottom Cloud→Fusion handoff runs right-to-left — the lane numbers (1→2→3→4) guide
the reader along the snake. The long "boxes + frame_id + timestamp" label now has
the full cross-column width to sit in.

Skip branch: CAFRU → "Skip output / edge boxes" (gray box) + side note "bypass
transport and cloud" — no long line. Stale branch: fr → Drop stale result, dark
gray dashed. Shapes: one ellipse (input), diamonds (decisions), rounded rects
(everything else; drop gray, out green). Labels lifted above the connector line.

Run:  python3 fig1_edge_cloud.py            # emit + validate + render PNG + PDF
      python3 fig1_edge_cloud.py --review    # ...then ask gpt-5.5 to critique
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/.claude/skills/drawio/scripts"))
import drawio
import print_style as P

NODE, LABEL, HEAD, SW = 9, 8, 10, 1.0

# Two columns: LEFT = Edge (top) / Fusion (bottom);  RIGHT = Transport (top) / Cloud (bottom).
# Transport sits directly above Cloud so the row fold is a clean vertical arrow.
# Gap tuned so the bottom Cloud->Fusion handoff (~105px) fits the long label.
LEFT_CX, RIGHT_CX = 215, 455

# Top-row rows (Edge has 4 nodes; CAFRU aligns with jpeg for a horizontal handoff).
T1, T2, T3, T4 = 56, 114, 172, 230      # frame / det / CAFRU=jpeg / SkipOut=queue
# Bottom-row rows (a larger top gap gives the queue->cloud FOLD room to breathe,
# so its label lands in the clear inter-row gap, not on the Cloud zone title).
B1, B2, B3, B4 = 340, 398, 456, 514    # cloud / remap=fr / fu / out
DROP_CX = 65                            # drop sits LEFT of fr (clears the remap->fr arrow)

d = drawio.Diagram("Fig1_edge_cloud_architecture", page=(600, 600), preset="color")


def place(key, label, style, cx, cy, w, h):
    d.node(key, label, style=style, w=w, h=h)
    d.nodes[key].x = cx - w / 2.0
    d.nodes[key].y = cy - h / 2.0


# ---- nodes ---------------------------------------------------------------
# Top-left — Edge Device (ellipse = the single input source)
place("frame", "Video frame I_t", P.terminal("io", size=NODE, sw=SW),
      LEFT_CX, T1, 90, 28)
place("det", "Edge detector\nYOLO26s", P.box("edge", size=NODE, sw=SW, bold=True),
      LEFT_CX, T2, 110, 36)
place("cafru", "CAFRU decision\nskip / frame / ROI",
      P.diamond("decision", size=NODE, sw=SW), LEFT_CX, T3, 140, 50)
place("skipout", "Skip output\nedge boxes", P.box("offline", size=NODE, sw=SW),
      LEFT_CX, T4, 124, 40)

# Top-right — Async Transport (jpeg aligns with CAFRU)
place("jpeg", "JPEG encoding\nframe or ROI crop",
      P.box("transport", size=NODE, sw=SW), RIGHT_CX, T3, 130, 36)
place("queue", "Bounded async queue\nkeep-latest / drop-new",
      P.box("queue", size=NODE, sw=SW), RIGHT_CX, T4, 140, 44)

# Bottom-right — Cloud Reviewer (cloud aligns under queue for the vertical fold)
place("cloud", "Cloud reviewer\nYOLO26x", P.box("cloud", size=NODE, sw=SW),
      RIGHT_CX, B1, 120, 36)
place("remap", "ROI remapping\nif ROI upload", P.box("cloud", size=NODE, sw=SW),
      RIGHT_CX, B2, 130, 36)

# Bottom-left — Fresh Fusion (fr aligns with remap; drop left of fr)
place("fr", "Freshness check\nage ≤ deadline",
      P.diamond("deadline", size=NODE, sw=SW), LEFT_CX, B2, 140, 50)
place("drop", "Drop stale result", P.box("offline", size=NODE, sw=SW),
      DROP_CX, B2, 100, 36)
place("fu", "Class-aware fusion\nIoU matching / replacement",
      P.box("fusion", size=NODE, sw=SW, bold=True), LEFT_CX, B3, 150, 44)
place("out", "Final boxes\nevent output",
      P.box("fusion", size=NODE, sw=SW, bold=True), LEFT_CX, B4, 140, 40)

# Side note annotating the Skip output node (floats below it, not in any group).
d.node("skipnote", "bypass transport and cloud",
       style=P.note(size=8, color="#7F7F7F"), w=180, h=16)
d.nodes["skipnote"].x = LEFT_CX - 90
d.nodes["skipnote"].y = T4 + 38

# ---- 4 swimlane zones (content-sized, one per quadrant) ------------------
def zone(key, label, members, cat):
    d.group(key, label, members=tuple(members),
            fill=P.zone_fill(cat), stroke=P.zone_stroke(cat),
            font_size=HEAD, pad=14)

zone("zedge",  "1) Edge Device",      ["frame", "det", "cafru", "skipout"], "edge")
zone("ztrans", "2) Async Transport",  ["jpeg", "queue"],                    "transport")
zone("zcloud", "3) Cloud Reviewer",   ["cloud", "remap"],                   "cloud")
zone("zfu",    "4) Fresh Fusion",     ["fr", "drop", "fu", "out"],          "fusion")

# ---- edges ---------------------------------------------------------------
SOL = P.main_flow(sw=SW, size=LABEL)               # dark-gray solid main flow
STALE = ("edgeStyle=orthogonalEdgeStyle;dashed=1;dashPattern=6 3;strokeWidth=1.0;"
         "strokeColor=#595959;fontFamily=Arial;fontSize=%d;fontColor=#595959;"
         "labelBackgroundColor=#FFFFFF;" % LABEL)  # dark-gray dashed discard
DN = dict(exit=("bottom", "center"), entry=("top", "center"))


def dn(a, b, **kw):   # vertical within-lane
    kw.setdefault("style", SOL)
    return d.edge(a, b, exit=DN["exit"], entry=DN["entry"], **kw)


def lr(a, b, **kw):   # horizontal left-to-right (top handoff)
    kw.setdefault("style", SOL)
    return d.edge(a, b, exit=("right", "center"), entry=("left", "center"), **kw)


def rl(a, b, **kw):   # horizontal right-to-left (bottom handoff)
    kw.setdefault("style", SOL)
    return d.edge(a, b, exit=("left", "center"), entry=("right", "center"), **kw)


# Top row: Edge chain + skip branch + handoff to Transport
dn("frame", "det")
dn("det", "cafru")
dn("cafru", "skipout", label="skip", label_dx=18)
lr("cafru", "jpeg", label="frame / ROI JPEG", label_dy=-12)
dn("jpeg", "queue")

# The fold: queue (top-right) straight DOWN into cloud (bottom-right). label_dy
# keeps the label in the clear inter-row gap (above the Cloud zone title).
dn("queue", "cloud", label="JPEG + metadata", label_dy=-10)

# Bottom row: Cloud chain + handoff (right-to-left) to Fusion + Fusion chain
dn("cloud", "remap")
rl("remap", "fr", label="boxes +\nframe_id + timestamp", label_dy=-14)
dn("fr", "fu", label="fresh", label_dx=18)
dn("fu", "out")

# Stale branch (dark-gray dashed, short leftward to the discard box beside fr)
d.edge("fr", "drop", label="stale", style=STALE, label_dy=-12,
       exit=("left", "center"), entry=("right", "center"))

# ---- emit + validate + render -------------------------------------------
out = os.path.join(HERE, "Fig1_edge_cloud_architecture.drawio")
d.to_drawio(out)
rep = d.validate()
print(rep.text())
assert rep.ok, "VALIDATE FAILED"

png = os.path.join(os.path.dirname(HERE), "Fig1_edge_cloud_architecture.png")
d.render_png(png)
print("wrote", out)
print("wrote", png)

pdf = os.path.join(os.path.dirname(HERE), "Fig1_edge_cloud_architecture.pdf")
drawio._run_render(out, pdf, "pdf")
print("wrote", pdf)

if "--review" in sys.argv:
    import ask_vlm
    print("\n===== gpt-5.5 critique =====")
    ask_vlm.ask(png)
