#!/usr/bin/env python3
"""Replot all paper figures from cached intermediate data.

This script does NO heavy computation (no networkx, no HuggingFace, no community).
It reads pre-computed data from src/plots/cache/ and renders figures using matplotlib.
Typical runtime: < 5 seconds for all 24 figures.

Usage:
    python src/plots/replot_all.py           # replot all
    python src/plots/replot_all.py degree    # replot only degree distribution figures
"""

import json
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from plot_style import (
    setup_style, COLORS,
    FIGSIZE_SINGLE, FIGSIZE_DOUBLE, FIGSIZE_TRIPLE,
    FIGSIZE_HEATMAP, FIGSIZE_HEATMAP_WIDE,
    TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS, ANNOT_FS,
    HEATMAP_TITLE_FS, HEATMAP_LABEL_FS, HEATMAP_TICK_FS,
)

COLORS = setup_style()

CACHE_DIR = Path(__file__).resolve().parent / "cache"
FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"


# =====================================================================
#  Degree distributions
# =====================================================================
def plot_degree_dist(csv_path, out_name, color, title_prefix,
                     alpha_in=None, xmin_in=None,
                     alpha_out=None, xmin_out=None):
    """Plot in/out degree distribution from cached degree sequences."""
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    for ax, col, label, alpha, xmin, scatter_alpha in [
        (axes[0], "in_degree", "In-degree", alpha_in, xmin_in, 0.7),
        (axes[1], "out_degree", "Out-degree", alpha_out, xmin_out, 0.5),
    ]:
        degrees = df[col].dropna().astype(int)
        degrees = degrees[degrees > 0]
        counts = Counter(degrees)
        degs = sorted(counts.keys())
        freqs = [counts[k] for k in degs]

        ax.scatter(degs, freqs, s=12, color=color, alpha=scatter_alpha)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"{label} distribution")
        ax.grid(True, alpha=0.3, which="both")

        if alpha and xmin:
            k_arr = np.array(degs, dtype=float)
            mask = k_arr >= xmin
            if mask.any():
                k_ref = k_arr[mask]
                freq_at_xmin = counts.get(int(xmin),
                    counts.get(min(k for k in degs if k >= xmin), 1))
                ref_line = freq_at_xmin * (k_ref / xmin) ** (-alpha)
                ax.plot(k_ref, ref_line, color=COLORS["quaternary"],
                        linestyle="--", linewidth=1, alpha=0.6,
                        label=f"$k^{{-{alpha:.2f}}}$")
                ax.legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / out_name, bbox_inches="tight")
    plt.close()


# =====================================================================
#  DAG structure (layer widths)
# =====================================================================
def plot_dag_single(csv_path, out_name, color, title_suffix):
    """Plot DAG layer width bar chart."""
    df = pd.read_csv(csv_path)
    if "width_raw" in df.columns and "width_tr" in df.columns:
        # Module level: two subplots (raw + TR)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        raw = df["width_raw"].dropna().astype(int).tolist()
        tr = df["width_tr"].dropna().astype(int).tolist()

        ax1.bar(range(len(raw)), raw, color=color, edgecolor="none", width=1.0)
        ax1.set_xlabel("Topological layer")
        ax1.set_ylabel("Number of modules")
        ax1.set_title(rf"DAG width by topological layer ($G_{{\mathrm{{module}}}}$, {len(raw)} layers)")
        ax1.set_xlim(-1, len(raw))

        ax2.bar(range(len(tr)), tr, color=color, edgecolor="none", width=1.0, alpha=0.7)
        ax2.set_xlabel("Topological layer")
        ax2.set_ylabel("Number of modules")
        ax2.set_title(rf"DAG width by topological layer ($G_{{\mathrm{{module}}}}^{{-}}$, {len(tr)} layers)")
        ax2.set_xlim(-1, len(tr))
    else:
        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
        widths = df["width"].astype(int).tolist()
        ax.bar(range(len(widths)), widths, color=color, edgecolor="none", width=1.0)
        ax.set_xlabel("Topological layer")
        ax.set_ylabel(f"Number of {title_suffix}")
        ax.set_title(rf"DAG width by topological layer ({len(widths)} layers)")
        ax.set_xlim(-1, len(widths))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / out_name, bbox_inches="tight")
    plt.close()


# =====================================================================
#  Centrality scatter plots
# =====================================================================
def plot_centrality(csv_path, out_prefix, color, title_prefix):
    """Plot 3 centrality scatter plots from cached data."""
    df = pd.read_csv(csv_path)
    indeg = df["in_degree"].values.astype(float)
    pr = df["pagerank"].values.astype(float)
    betw = df["betweenness"].values.astype(float)

    panels = [
        (indeg, pr, "In-degree", "PageRank", "indeg_pr"),
        (indeg, betw, "In-degree", "Betweenness", "indeg_betw"),
        (betw, pr, "Betweenness", "PageRank", "betw_pr"),
    ]

    for x, y, xlabel, ylabel, suffix in panels:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        mask = (x > 0) & (y > 0)
        ax.scatter(x[mask], y[mask], s=3, alpha=0.3, color=color, edgecolors="none",
                   rasterized=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=HEATMAP_LABEL_FS)
        ax.set_ylabel(ylabel, fontsize=HEATMAP_LABEL_FS)
        ax.tick_params(labelsize=HEATMAP_TICK_FS)
        ax.grid(True, alpha=0.2, which="both")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"{out_prefix}_{suffix}.pdf", bbox_inches="tight",
                    dpi=300)
        plt.close()


# =====================================================================
#  Robustness curves
# =====================================================================
def plot_robustness(csv_path, out_name, color, title):
    """Plot robustness curve from cached data."""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["fraction_removed"] * 100, df["random_gcc"],
            "o-", color=color, markersize=5, linewidth=1.5,
            label="Random removal")
    ax.plot(df["fraction_removed"] * 100, df["targeted_gcc"],
            "s--", color=color, alpha=0.5, markersize=5, linewidth=1.5,
            label="Targeted removal (by PageRank)")
    ax.set_xlabel("Fraction of nodes removed (%)")
    ax.set_ylabel("Largest WCC / Total nodes")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, 55, 5))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / out_name, bbox_inches="tight")
    plt.close()


# =====================================================================
#  Namespace heatmaps
# =====================================================================
def plot_namespace_heatmap(csv_path, out_name, title):
    """Plot namespace import heatmap from cached matrix."""
    df = pd.read_csv(csv_path, index_col=0)
    labels = list(df.columns)
    matrix = df.values.astype(float)

    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
    # Use log(1+count) for color scale
    im = ax.imshow(np.log1p(matrix), aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=HEATMAP_TICK_FS)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=HEATMAP_TICK_FS)
    ax.set_title(title, fontsize=HEATMAP_TITLE_FS)
    fig.colorbar(im, ax=ax, shrink=0.6, label="log(1 + count)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / out_name, bbox_inches="tight")
    plt.close()


# =====================================================================
#  Community heatmaps
# =====================================================================
def plot_community_heatmap(json_path, out_name, level_name, cmap="YlOrRd"):
    """Plot community x category heatmap from cached contingency data."""
    data = json.loads(Path(json_path).read_text())
    matrix = np.array(data["matrix"])
    row_labels = data["row_labels"]
    col_labels = data["col_labels"]
    nmi = data["nmi"]

    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
    norm = mcolors.LogNorm(vmin=max(1, matrix[matrix > 0].min()),
                           vmax=matrix.max()) if matrix.max() > 0 else None
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=HEATMAP_TICK_FS)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=HEATMAP_TICK_FS)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if v > 0:
                color = "white" if v > matrix.max() * 0.4 else "black"
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center",
                        fontsize=ANNOT_FS, color=color)

    ax.set_title(f"{level_name} communities vs. categories (NMI = {nmi:.2f})",
                 fontsize=HEATMAP_TITLE_FS)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Count")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / out_name, bbox_inches="tight")
    plt.close()


# =====================================================================
#  Containment curve (hardcoded small data)
# =====================================================================
def plot_containment():
    """Plot containment decay curve (data is small, hardcoded)."""
    ns_depths = [1, 2, 3, 4, 5, 6]
    ns_containment = [22.2, 14.2, 12.8, 12.7, 12.6, 12.6]
    topdir_containment = 48.9
    file_containment = 15.6
    decl_baseline = 9.6

    fig, ax = plt.subplots(figsize=(12, 4))
    x_ns = list(range(1, 7))
    ax.plot(x_ns, ns_containment, "o-", color=COLORS["tertiary"], linewidth=1.8,
            markersize=6, label="Namespace depth $k$", zorder=3)
    ax.plot(0, topdir_containment, "s", color=COLORS["primary"], markersize=9,
            markeredgewidth=0.8, markeredgecolor=COLORS["primary"],
            label="File-system level", zorder=4)
    ax.plot(7, file_containment, "s", color=COLORS["primary"], markersize=9,
            markeredgewidth=0.8, markeredgecolor=COLORS["primary"], zorder=4)
    ax.annotate("Top-level dir\n(32 dirs)", (0, topdir_containment),
                textcoords="offset points", xytext=(50, 8),
                ha="center", color=COLORS["primary"],
                arrowprops=dict(arrowstyle="-", color=COLORS["primary"], lw=0.6))
    ax.annotate("File module\n(7,225 files)", (7, file_containment),
                textcoords="offset points", xytext=(-50, 12),
                ha="center", color=COLORS["primary"],
                arrowprops=dict(arrowstyle="-", color=COLORS["primary"], lw=0.6))
    ax.axhline(y=decl_baseline, color=COLORS["grey"], linestyle="--", linewidth=1, zorder=1)
    ax.text(6.8, decl_baseline - 0.6, "same-file baseline (9.6%)",
            color=COLORS["grey"], va="top", ha="right")
    ax.set_xlabel("Granularity level")
    ax.set_ylabel("Containment (%)")
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(["Top\ndir", "$k{=}1$", "$k{=}2$", "$k{=}3$",
                         "$k{=}4$", "$k{=}5$", "$k{=}6$", "File"])
    ax.set_ylim(0, 56)
    ax.set_xlim(-0.6, 8.0)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "containment_curve.pdf", bbox_inches="tight")
    plt.close()


# =====================================================================
#  Main
# =====================================================================
def main():
    t0 = time.time()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    def _try(name, fn, *args, **kwargs):
        nonlocal count
        try:
            fn(*args, **kwargs)
            count += 1
            print(f"  OK  {name}")
        except FileNotFoundError as e:
            print(f"  SKIP {name} (cache missing: {e.filename})")
        except Exception as e:
            print(f"  FAIL {name}: {e}")

    print("Replotting all figures from cache...\n")

    # Containment (no cache needed)
    _try("containment_curve.pdf", plot_containment)

    # Degree distributions
    _try("degree_distribution.pdf", plot_degree_dist,
         CACHE_DIR / "module_degree_dist.csv", "degree_distribution.pdf",
         COLORS["primary"], "Module")
    _try("thm_degree_distribution.pdf", plot_degree_dist,
         CACHE_DIR / "thm_degree_dist.csv", "thm_degree_distribution.pdf",
         COLORS["secondary"], "Declaration",
         alpha_in=1.78, xmin_in=20)
    _try("ns_degree_distribution.pdf", plot_degree_dist,
         CACHE_DIR / "ns_degree_dist.csv", "ns_degree_distribution.pdf",
         COLORS["tertiary"], "Namespace",
         alpha_in=1.60, xmin_in=4, alpha_out=1.90, xmin_out=13)

    # DAG structure
    _try("dag_structure.pdf", plot_dag_single,
         CACHE_DIR / "module_dag_layers.csv", "dag_structure.pdf",
         COLORS["primary"], "modules")
    _try("thm_dag_structure.pdf", plot_dag_single,
         CACHE_DIR / "thm_dag_layers.csv", "thm_dag_structure.pdf",
         COLORS["secondary"], "declarations")
    _try("ns_dag_structure.pdf", plot_dag_single,
         CACHE_DIR / "ns_dag_layers.csv", "ns_dag_structure.pdf",
         COLORS["tertiary"], "super-nodes")

    # Centrality scatter
    _try("module_centrality_*.pdf", plot_centrality,
         CACHE_DIR / "module_centrality.csv", "module_centrality",
         COLORS["primary"], r"$G_{\mathrm{module}}$")
    _try("thm_centrality_*.pdf", plot_centrality,
         CACHE_DIR / "thm_centrality.csv", "thm_centrality",
         COLORS["secondary"], r"$G_{\mathrm{thm}}$")
    _try("ns_centrality_*.pdf", plot_centrality,
         CACHE_DIR / "ns_centrality.csv", "ns_centrality",
         COLORS["tertiary"], r"$G_{\mathrm{ns}}^{(2)}$")

    # Robustness
    _try("module_robustness_curve.pdf", plot_robustness,
         CACHE_DIR / "module_robustness.csv", "module_robustness_curve.pdf",
         COLORS["primary"], r"Network robustness: $G_{\mathrm{module}}$")
    _try("thm_robustness_curve.pdf", plot_robustness,
         CACHE_DIR / "thm_robustness.csv", "thm_robustness_curve.pdf",
         COLORS["secondary"], r"Network robustness: $G_{\mathrm{thm}}$")
    _try("ns_robustness_curve.pdf", plot_robustness,
         CACHE_DIR / "ns_robustness.csv", "ns_robustness_curve.pdf",
         COLORS["tertiary"], r"Network robustness: $G_{\mathrm{ns}}^{(2)}$")

    # Namespace heatmaps
    _try("namespace_heatmap_raw.pdf", plot_namespace_heatmap,
         CACHE_DIR / "namespace_heatmap_raw.csv", "namespace_heatmap_raw.pdf",
         "Raw")
    _try("namespace_heatmap_tr.pdf", plot_namespace_heatmap,
         CACHE_DIR / "namespace_heatmap_tr.csv", "namespace_heatmap_tr.pdf",
         "Transitive Reduction")

    # Community heatmaps
    _try("community_module_heatmap.pdf", plot_community_heatmap,
         CACHE_DIR / "community_module_heatmap.json", "community_module_heatmap.pdf",
         "Module", "Blues")
    _try("community_decl_heatmap.pdf", plot_community_heatmap,
         CACHE_DIR / "community_decl_heatmap.json", "community_decl_heatmap.pdf",
         "Declaration", "Reds")
    _try("community_ns_heatmap.pdf", plot_community_heatmap,
         CACHE_DIR / "community_ns_heatmap.json", "community_ns_heatmap.pdf",
         "Namespace", "Greens")

    elapsed = time.time() - t0
    print(f"\nDone: {count} figures in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
