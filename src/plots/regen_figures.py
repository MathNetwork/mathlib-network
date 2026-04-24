#!/usr/bin/env python3
"""
Regenerate 4 figures to match unified plot style.
"""

import time
from pathlib import Path
from collections import Counter

from plot_style import setup_style, COLORS, FIGSIZE_SINGLE, FIGSIZE_DOUBLE, FIGSIZE_TRIPLE

COLORS = setup_style()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datasets import load_dataset

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"


def load_and_build():
    print("Loading data from HuggingFace...")
    t0 = time.time()
    nodes_ds = load_dataset(
        "MathNetwork/MathlibGraph",
        data_files="mathlib_nodes.csv",
        split="train",
    )
    edges_ds = load_dataset(
        "MathNetwork/MathlibGraph",
        data_files="mathlib_edges.csv",
        split="train",
    )
    nodes_df = nodes_ds.to_pandas()
    edges_df = edges_ds.to_pandas()
    print(f"  Downloaded in {time.time() - t0:.1f}s")

    print("Building graph...")
    t0 = time.time()
    G = nx.DiGraph()
    for _, row in nodes_df.iterrows():
        G.add_node(row["name"], kind=row["kind"], module=row["module"])
    node_set = set(G.nodes)
    for _, row in edges_df.iterrows():
        if row["source"] in node_set and row["target"] in node_set:
            G.add_edge(row["source"], row["target"])
    print(f"  Built in {time.time() - t0:.1f}s  ({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)")
    return G


def plot_degree_distribution(G, alpha_in, xmin_in, alpha_out, xmin_out):
    """Degree distribution scatter plot -- matching unified style."""
    in_degrees = [d for _, d in G.in_degree() if d > 0]
    out_degrees = [d for _, d in G.out_degree() if d > 0]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
    # Both panels use red (secondary) for declaration-level G_thm
    for ax, degrees, title, alpha, xmin, alpha_scatter in [
        (axes[0], in_degrees, "In-degree distribution", alpha_in, xmin_in, 0.7),
        (axes[1], out_degrees, "Out-degree distribution", alpha_out, xmin_out, 0.5),
    ]:
        counts = Counter(degrees)
        degs = sorted(counts.keys())
        freqs = [counts[k] for k in degs]

        ax.scatter(degs, freqs, s=12, color=COLORS["secondary"], alpha=alpha_scatter)
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Fix labels
        if "In" in title:
            ax.set_xlabel("In-degree")
        else:
            ax.set_xlabel("Out-degree")
        ax.set_ylabel("Count")
        ax.set_title(title)

        # Power law reference line
        k_arr = np.array(degs, dtype=float)
        mask = k_arr >= xmin
        if mask.any():
            k_ref = k_arr[mask]
            freq_at_xmin = counts.get(int(xmin), counts.get(min(k for k in degs if k >= xmin), 1))
            ref_line = freq_at_xmin * (k_ref / xmin) ** (-alpha)
            ax.plot(k_ref, ref_line, color=COLORS["quaternary"], linestyle="--",
                    linewidth=1, alpha=0.6,
                    label=f"$k^{{-{alpha:.2f}}}$")
            ax.legend()

        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "thm_degree_distribution.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: thm_degree_distribution.pdf")


def plot_robustness_from_df(df):
    """Robustness curve -- matching unified style."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["fraction_removed"] * 100, df["random_wcc_ratio"],
            "o-", color=COLORS["secondary"], markersize=4, linewidth=1.5,
            label="Random removal")
    ax.plot(df["fraction_removed"] * 100, df["targeted_wcc_ratio"],
            "s--", color=COLORS["secondary"], markersize=4, linewidth=1.5, alpha=0.5,
            label="Targeted removal (by PageRank)")

    ax.set_xlabel("Fraction of nodes removed (%)")
    ax.set_ylabel("Largest WCC / Total nodes")
    ax.set_title(r"Network robustness: $G_{\mathrm{thm}}$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, 55, 5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "thm_robustness_curve.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: thm_robustness_curve.pdf")


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Power-law fit parameters (from paper appendix-decl.tex)
    alpha_in, xmin_in = 1.78, 20
    alpha_out, xmin_out = 2.5, 50  # out-degree is sharply bounded; reference line only

    # Load graph and plot degree distribution
    G = load_and_build()
    print("\nGenerating degree distribution plot...")
    plot_degree_distribution(G, alpha_in, xmin_in, alpha_out, xmin_out)

    # Compute and plot robustness (instead of reading CSV)
    print("Generating robustness plot...")
    from plot_robustness_curves import robustness_curve
    fractions = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.15, 0.17, 0.20,
                 0.25, 0.30, 0.40, 0.50]
    random_gcc, targeted_gcc = robustness_curve(G, fractions)
    x = [0.0] + list(fractions)
    df = pd.DataFrame({
        "fraction_removed": x,
        "random_wcc_ratio": random_gcc,
        "targeted_wcc_ratio": targeted_gcc,
    })
    df.to_csv(OUTPUT_DIR.parent.parent / "src" / "output" / "robustness_data.csv", index=False)
    plot_robustness_from_df(df)

    print(f"\nDone in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
