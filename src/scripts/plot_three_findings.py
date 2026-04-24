#!/usr/bin/env python3
"""Generate three-panel figure illustrating the three principal findings."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "paperNeurIPS" / "figures" / "three-findings.pdf"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))
fig.subplots_adjust(wspace=0.35, left=0.04, right=0.97, top=0.88, bottom=0.08)

# ─── Colors ───
C_HUMAN = "#2E86AB"      # blue for human/explicit
C_SYNTH = "#E8505B"      # red for synthesized/compiler
C_USED  = "#2E86AB"
C_WASTE = "#D4D4D4"      # grey for unused
C_INFRA = "#E8505B"      # red for infrastructure
C_MATH  = "#2E86AB"      # blue for mathematical

# ═══════════════════════════════════════════════════════
# Panel (a): Human plan vs compiler graph
# ═══════════════════════════════════════════════════════
ax1.set_title("(a) Human plan vs compiler", fontsize=11, fontweight="bold", pad=10)

# Left side: human blueprint (4 nodes, 4 edges)
human_x = [1.0, 0.3, 1.7, 1.0]
human_y = [3.2, 1.8, 1.8, 0.5]
human_labels = ["length_\nappend", "length_\nmap", "map_\nmap", "List.\nlength"]

for x, y, lab in zip(human_x, human_y, human_labels):
    ax1.add_patch(plt.Rectangle((x-0.38, y-0.28), 0.76, 0.56,
                  facecolor="#E8F4FD", edgecolor=C_HUMAN, linewidth=1.2, zorder=2))
    ax1.text(x, y, lab, ha="center", va="center", fontsize=6, family="monospace", zorder=3)

# Blueprint edges
for (x1,y1), (x2,y2) in [((1.0,2.92),(0.3,2.08)), ((1.0,2.92),(1.0,0.78)),
                           ((0.3,1.52),(1.0,0.78)), ((1.7,1.52),(1.0,0.78))]:
    ax1.annotate("", xy=(x2,y2), xytext=(x1,y1),
                 arrowprops=dict(arrowstyle="->", color=C_HUMAN, lw=1.2))

ax1.text(1.0, 3.9, "Blueprint", ha="center", fontsize=9, fontweight="bold", color=C_HUMAN)
ax1.text(1.0, 3.55, "4 nodes, 4 edges", ha="center", fontsize=7, color="gray")

# Right side: compiler graph (7 nodes, 10 edges)
comp_x = [4.2, 3.5, 4.9, 4.2, 3.2, 5.2, 4.2]
comp_y = [3.2, 1.8, 1.8, 0.8, 0.2, 0.2, -0.5]
comp_labels = ["length_\nappend", "length_\nmap", "map_\nmap", "List.\nlength",
               "List.rec", "Nat.add", "List.map"]
comp_colors = [C_HUMAN]*4 + [C_SYNTH]*3  # first 4 = from blueprint, last 3 = compiler-added

for x, y, lab, c in zip(comp_x, comp_y, comp_labels, comp_colors):
    ax1.plot(x, y, "o", color=c, markersize=7, zorder=3)
    ax1.text(x, y+0.32, lab, ha="center", va="bottom", fontsize=5.5, family="monospace", color=c)

# Compiler edges (explicit)
explicit_edges = [((4.2,3.0),(3.5,2.0)), ((4.2,3.0),(4.2,1.0)),
                  ((3.5,1.6),(4.2,1.0)), ((4.9,1.6),(4.2,1.0))]
for (x1,y1),(x2,y2) in explicit_edges:
    ax1.annotate("", xy=(x2,y2), xytext=(x1,y1),
                 arrowprops=dict(arrowstyle="->", color=C_HUMAN, lw=1.0))

# Compiler edges (synthesized - dashed)
synth_edges = [((4.2,3.0),(5.2,0.4)), ((4.2,3.0),(3.2,0.4)),
               ((4.2,3.0),(4.2,-0.3)), ((3.5,1.6),(4.2,-0.3)),
               ((4.9,1.6),(3.2,0.4)), ((4.2,0.6),(3.2,0.4))]
for (x1,y1),(x2,y2) in synth_edges:
    ax1.annotate("", xy=(x2,y2), xytext=(x1,y1),
                 arrowprops=dict(arrowstyle="->", color=C_SYNTH, lw=0.8, linestyle="dashed"))

ax1.text(4.2, 3.9, "Compiler graph", ha="center", fontsize=9, fontweight="bold", color=C_SYNTH)
ax1.text(4.2, 3.55, "7 nodes, 10 edges", ha="center", fontsize=7, color="gray")

# Legend
ax1.plot([], [], "o", color=C_HUMAN, markersize=5, label="Human-written (25.8%)")
ax1.plot([], [], "o", color=C_SYNTH, markersize=5, label="Synthesized (74.2%)")
ax1.legend(loc="lower center", fontsize=7, frameon=False, ncol=1,
           bbox_to_anchor=(0.5, -0.12))

ax1.set_xlim(-0.3, 5.8)
ax1.set_ylim(-1.0, 4.2)
ax1.set_aspect("equal")
ax1.axis("off")

# ═══════════════════════════════════════════════════════
# Panel (b): Over-importing (median 1.6% utilization)
# ═══════════════════════════════════════════════════════
ax2.set_title("(b) Over-importing", fontsize=11, fontweight="bold", pad=10)

# Draw two module boxes
# Module A (importer)
ax2.add_patch(mpatches.FancyBboxPatch((0.3, 1.5), 1.8, 2.5, boxstyle="round,pad=0.1",
              facecolor="#E8F4FD", edgecolor=C_HUMAN, linewidth=1.5))
ax2.text(1.2, 3.7, "Module A", ha="center", fontsize=9, fontweight="bold", color=C_HUMAN)
ax2.text(1.2, 3.35, "(importer)", ha="center", fontsize=7, color="gray")

# A few declarations in A
for i, y in enumerate([2.9, 2.5, 2.1, 1.7]):
    ax2.add_patch(plt.Rectangle((0.5, y-0.12), 1.4, 0.24,
                  facecolor="white", edgecolor=C_HUMAN, linewidth=0.6))
    ax2.text(1.2, y, f"thm_{i+1}", ha="center", fontsize=6, family="monospace")

# Module B (imported)
ax2.add_patch(mpatches.FancyBboxPatch((3.5, 0.0), 1.8, 4.0, boxstyle="round,pad=0.1",
              facecolor="#FFF5F5", edgecolor=C_SYNTH, linewidth=1.5))
ax2.text(4.4, 3.7, "Module B", ha="center", fontsize=9, fontweight="bold", color=C_SYNTH)
ax2.text(4.4, 3.35, "(imported)", ha="center", fontsize=7, color="gray")

# Many declarations in B, only 1 used
b_decls = 12
used_idx = 2  # which one is actually used
for i in range(b_decls):
    y = 3.0 - i * 0.25
    color = C_USED if i == used_idx else C_WASTE
    ec = C_HUMAN if i == used_idx else "#AAAAAA"
    ax2.add_patch(plt.Rectangle((3.7, y-0.09), 1.4, 0.18,
                  facecolor=color if i == used_idx else "#F0F0F0",
                  edgecolor=ec, linewidth=0.5 if i != used_idx else 1.2))

# Arrow from A to the used declaration in B
used_y = 3.0 - used_idx * 0.25
ax2.annotate("", xy=(3.7, used_y), xytext=(1.9, 2.5),
             arrowprops=dict(arrowstyle="->, head_width=0.15", color=C_HUMAN, lw=1.5))

# Import arrow (thick, covers whole module)
ax2.annotate("", xy=(3.5, 2.0), xytext=(2.1, 2.0),
             arrowprops=dict(arrowstyle="->, head_width=0.2", color=C_SYNTH, lw=2.5,
                           linestyle="dashed"))
ax2.text(2.8, 2.25, "import", ha="center", fontsize=7, color=C_SYNTH, fontstyle="italic")

# Stats
ax2.text(2.8, -0.3, "Median utilization: 1.6%", ha="center", fontsize=9, fontweight="bold")
ax2.text(2.8, -0.65, "1 of ~60 declarations used per import", ha="center", fontsize=7, color="gray")

ax2.set_xlim(-0.1, 5.8)
ax2.set_ylim(-1.0, 4.2)
ax2.axis("off")

# ═══════════════════════════════════════════════════════
# Panel (c): Centrality ≠ mathematical depth
# ═══════════════════════════════════════════════════════
ax3.set_title("(c) Centrality vs depth", fontsize=11, fontweight="bold", pad=10)

# Bar chart: top declarations by in-degree
names = ["OfNat.\nofNat", "Eq.refl", "HAdd.\nhAdd", "HSMul.\nhSMul", "...",
         "Sylow", "Chinese\nRem."]
values = [89936, 69579, 55074, 36380, 0, 119, 66]
colors = [C_INFRA, C_INFRA, C_INFRA, C_INFRA, "white", C_MATH, C_MATH]
edge_colors = [C_INFRA, C_INFRA, C_INFRA, C_INFRA, "white", C_MATH, C_MATH]

# Use log scale for visibility
log_vals = [np.log10(max(v, 1)) for v in values]
log_vals[4] = 0  # gap

bars = ax3.barh(range(len(names)-1, -1, -1), log_vals, color=colors,
                edgecolor=edge_colors, linewidth=0.8, height=0.7)

ax3.set_yticks(range(len(names)-1, -1, -1))
ax3.set_yticklabels(names, fontsize=7, family="monospace")

# Add value labels
for i, (v, lv) in enumerate(zip(values, log_vals)):
    if v > 0:
        label = f"{v:,}"
        ax3.text(lv + 0.1, len(names)-1-i, label, va="center", fontsize=6.5)

ax3.set_xlabel("In-degree (log₁₀)", fontsize=8)
ax3.set_xlim(0, 5.5)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.tick_params(axis="x", labelsize=7)

# Legend
infra_patch = mpatches.Patch(color=C_INFRA, label="Type-system infrastructure")
math_patch = mpatches.Patch(color=C_MATH, label="Mathematical content")
ax3.legend(handles=[infra_patch, math_patch], loc="lower right", fontsize=7, frameon=False)

# Annotation
ax3.text(2.75, -1.2, "Top hubs are compiler artifacts, not deep theorems",
         ha="center", fontsize=7, color="gray", fontstyle="italic")

plt.savefig(OUT, bbox_inches="tight", dpi=300)
print(f"Saved to {OUT}")
plt.close()
