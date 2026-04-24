#!/usr/bin/env python3
"""Generate containment decay curve for the paper."""

from pathlib import Path

from plot_style import setup_style, COLORS, FIGSIZE_SINGLE

COLORS = setup_style()
import matplotlib.pyplot as plt

# Data from namespace_graph.py output
# Namespace depths (computed on all 8,436,366 G_thm edges)
ns_depths =       [1,     2,     3,     4,     5,     6]
ns_containment =  [22.2,  14.2,  12.8,  12.7,  12.6,  12.6]

# File-system levels (computed on 2,506,738 file-mapped edges)
topdir_containment = 48.9   # 32 top-level directories
file_containment   = 15.6   # 7,225 file modules

# Declaration-level baseline (from §4 Table 5: 9.6% same-module)
decl_baseline = 9.6

OUTDIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 4))

# x positions: 0=topdir, 1..6=namespace depths, 7=file
x_ns = list(range(1, 7))

# Green line: namespace depth series
ax.plot(x_ns, ns_containment, "o-", color=COLORS["tertiary"], linewidth=1.8,
        markersize=6, label="Namespace depth $k$", zorder=3)

# Blue squares: file-system levels
ax.plot(0, topdir_containment, "s", color=COLORS["primary"], markersize=9,
        markeredgewidth=0.8, markeredgecolor=COLORS["primary"],
        label="File-system level", zorder=4)
ax.plot(7, file_containment, "s", color=COLORS["primary"], markersize=9,
        markeredgewidth=0.8, markeredgecolor=COLORS["primary"], zorder=4)

# Annotations for blue squares
ax.annotate("Top-level dir\n(32 dirs)", (0, topdir_containment),
            textcoords="offset points", xytext=(50, 8),
            ha="center", color=COLORS["primary"],
            arrowprops=dict(arrowstyle="-", color=COLORS["primary"], lw=0.6))
ax.annotate("File module\n(7,225 files)", (7, file_containment),
            textcoords="offset points", xytext=(-50, 12),
            ha="center", color=COLORS["primary"],
            arrowprops=dict(arrowstyle="-", color=COLORS["primary"], lw=0.6))

# Gray dashed baseline
ax.axhline(y=decl_baseline, color=COLORS["grey"], linestyle="--", linewidth=1,
           zorder=1)
ax.text(6.8, decl_baseline - 0.6,
        "same-file baseline (9.6%)",
        color=COLORS["grey"], va="top", ha="right")

# Axis labels
ax.set_xlabel("Granularity level")
ax.set_ylabel("Containment (%)")
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
ax.set_xticklabels(["Top\ndir", "$k{=}1$", "$k{=}2$", "$k{=}3$",
                     "$k{=}4$", "$k{=}5$", "$k{=}6$", "File"])
ax.set_ylim(0, 56)
ax.set_xlim(-0.6, 8.0)
ax.legend(loc="upper right", framealpha=0.9)

# Light grid
ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
out = OUTDIR / "containment_curve.pdf"
fig.savefig(out)
plt.close()
print(f"Saved {out}")
