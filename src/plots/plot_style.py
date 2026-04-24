"""Unified plotting style for all paper figures.

Usage in any script under src/plots/ or src/scripts/:
    from plot_style import setup_style, COLORS, FIGSIZE_SINGLE, FIGSIZE_DOUBLE, FIGSIZE_TRIPLE
    from plot_style import TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS, ANNOT_FS
"""

import matplotlib
import matplotlib.pyplot as plt


# ── Unified color palette ────────────────────────────────────────────
COLORS = {
    "primary": "#2E86AB",       # steel blue  (in-degree, random removal, main data)
    "secondary": "#E94F37",     # coral red   (out-degree, targeted removal, contrast)
    "tertiary": "#2E8B57",      # sea green   (namespace-level, containment)
    "quaternary": "#F5A623",    # gold        (power-law fits, reference lines)
    "grey": "#888888",          # grey        (baselines, annotations)
}

# ── Unified font sizes (single source of truth) ─────────────────────
# Figures are typically placed in 0.32\textwidth subfigures (~5 cm),
# so they shrink to ~40%.  Use large sizes so text stays readable.
TITLE_FS = 22       # axes / figure titles
LABEL_FS = 20       # axis labels (xlabel, ylabel)
TICK_FS = 18        # tick labels
LEGEND_FS = 18      # legend text
ANNOT_FS = 11       # heatmap cell annotations

# Larger sizes for heatmaps (bigger figures, scaled down more in LaTeX)
HEATMAP_TITLE_FS = 38
HEATMAP_LABEL_FS = 22
HEATMAP_TICK_FS = 15

# ── Standard figure sizes (width, height) in inches ──────────────────
FIGSIZE_SINGLE = (5, 3.8)      # single-column / standalone (compact for 0.32\textwidth)
FIGSIZE_DOUBLE = (12, 5)       # double-column / two subplots side by side
FIGSIZE_TRIPLE = (15, 4.5)     # three subplots side by side
FIGSIZE_HEATMAP = (14, 10)     # heatmap (needs room for labels)
FIGSIZE_HEATMAP_WIDE = (28, 14)  # side-by-side heatmaps


def setup_style():
    """Apply unified rcParams for all paper figures."""
    matplotlib.use("Agg")

    # ── Font: match LaTeX Computer Modern ──
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["text.usetex"] = False

    # ── Font sizes (driven by module-level constants) ──
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = TITLE_FS
    plt.rcParams["axes.labelsize"] = LABEL_FS
    plt.rcParams["xtick.labelsize"] = TICK_FS
    plt.rcParams["ytick.labelsize"] = TICK_FS
    plt.rcParams["legend.fontsize"] = LEGEND_FS

    # ── Figure output defaults ──
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.format"] = "pdf"
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"

    # ── Clean style ──
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = False
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.edgecolor"] = "0.8"

    return COLORS
