#!/usr/bin/env python3
"""Plot degree distribution of G_ns^(2) on log-log axes.

Style matches the module-level (full_analysis.py) and declaration-level
degree distribution figures used in 7.1.1 and 7.1.2.

Output: paper/analysis/ns_degree_distribution.pdf
"""

import time
from collections import Counter
from pathlib import Path

from plot_style import setup_style, COLORS, FIGSIZE_SINGLE, FIGSIZE_DOUBLE, FIGSIZE_TRIPLE

COLORS = setup_style()

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

DEPTH = 2
OUTDIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


def ns_at_depth(name: str, k: int) -> str:
    parts = name.split(".")
    if len(parts) <= k:
        return ".".join(parts[:-1]) if len(parts) > 1 else "_root_"
    return ".".join(parts[:k])


def main():
    # ── Load data ──
    print("Loading HuggingFace data ...")
    t0 = time.time()
    nodes_ds = load_dataset(
        "MathNetwork/MathlibGraph",
        data_files="mathlib_nodes.csv", split="train",
    )
    edges_ds = load_dataset(
        "MathNetwork/MathlibGraph",
        data_files="mathlib_edges.csv", split="train",
    )
    node_names = set(nodes_ds.to_pandas()["name"].dropna())
    edges_df = edges_ds.to_pandas()
    print(f"  Loaded {len(node_names):,} declarations, "
          f"{len(edges_df):,} edges ({time.time()-t0:.1f}s)")

    # ── Build G_ns at depth k ──
    print(f"Building G_ns at depth {DEPTH} ...")
    t0 = time.time()

    decl_to_ns = {name: ns_at_depth(name, DEPTH) for name in node_names}

    # Count unique namespace-pair edges and track all namespace nodes
    all_ns = set(decl_to_ns.values())
    edge_set = set()
    for _, row in edges_df.iterrows():
        s, t = row["source"], row["target"]
        if s not in decl_to_ns or t not in decl_to_ns:
            continue
        ns_s, ns_t = decl_to_ns[s], decl_to_ns[t]
        if ns_s != ns_t:
            edge_set.add((ns_s, ns_t))

    # Compute degree sequences
    in_counter = Counter()
    out_counter = Counter()
    for ns in all_ns:
        in_counter[ns] = 0
        out_counter[ns] = 0
    for ns_s, ns_t in edge_set:
        out_counter[ns_s] += 1
        in_counter[ns_t] += 1

    in_vals = list(in_counter.values())
    out_vals = list(out_counter.values())
    print(f"  Nodes: {len(all_ns):,}  Edges: {len(edge_set):,}  "
          f"({time.time()-t0:.1f}s)")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # --- In-degree (left panel) --- green for namespace level
    ax = axes[0]
    deg_count = Counter(v for v in in_vals if v > 0)
    degs = sorted(deg_count.keys())
    counts = [deg_count[d] for d in degs]
    ax.scatter(degs, counts, s=12, color=COLORS["tertiary"], alpha=0.7)

    # Power-law reference line
    in_arr = np.array([v for v in in_vals if v >= 4])  # x_min = 4
    if len(in_arr) > 0:
        alpha = 1.603  # from power-law fit
        k_ref = np.logspace(np.log10(4), np.log10(max(degs)), 50)
        c0 = deg_count.get(4, deg_count.get(5, 1))
        ref_line = c0 * (k_ref / 4) ** (-alpha)
        ax.plot(k_ref, ref_line, color=COLORS["quaternary"], linewidth=1.5,
                linestyle="--", alpha=0.8,
                label=rf"$k^{{-{alpha:.2f}}}$ ($x_{{\min}}=4$)")
        ax.legend(loc="upper right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("In-degree")
    ax.set_ylabel("Count")
    ax.set_title(r"In-degree distribution of $G_{\mathrm{ns}}^{(2)}$")
    ax.grid(True, alpha=0.3, which="both")

    # --- Out-degree (right panel) --- green for namespace level (lighter)
    ax = axes[1]
    deg_count = Counter(v for v in out_vals if v > 0)
    degs = sorted(deg_count.keys())
    counts = [deg_count[d] for d in degs]
    ax.scatter(degs, counts, s=12, color=COLORS["tertiary"], alpha=0.5)

    # Power-law reference line
    out_arr = np.array([v for v in out_vals if v >= 13])  # x_min = 13
    if len(out_arr) > 0:
        alpha_out = 1.903  # from power-law fit
        k_ref = np.logspace(np.log10(13), np.log10(max(degs)), 50)
        c0 = deg_count.get(13, deg_count.get(14, 1))
        ref_line = c0 * (k_ref / 13) ** (-alpha_out)
        ax.plot(k_ref, ref_line, color=COLORS["quaternary"], linewidth=1.5,
                linestyle="--", alpha=0.8,
                label=rf"$k^{{-{alpha_out:.2f}}}$ ($x_{{\min}}=13$)")
        ax.legend(loc="upper right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Out-degree")
    ax.set_ylabel("Count")
    ax.set_title(r"Out-degree distribution of $G_{\mathrm{ns}}^{(2)}$")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out_path = OUTDIR / "ns_degree_distribution.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
