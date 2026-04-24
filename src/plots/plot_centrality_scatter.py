#!/usr/bin/env python3
"""Generate centrality scatter plots for declaration and namespace levels.

Output:
  paper/analysis/thm_centrality_*.pdf
  paper/analysis/ns_centrality_*.pdf
"""

import time
from collections import Counter, defaultdict
from pathlib import Path

from plot_style import setup_style, COLORS, FIGSIZE_SINGLE, FIGSIZE_DOUBLE, FIGSIZE_TRIPLE

COLORS = setup_style()

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from datasets import load_dataset

OUTDIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


def plot_centrality_separate(in_deg, pagerank, betweenness, nodes,
                             title_prefix, color, out_prefix):
    """Plot 3 separate full-width centrality scatter plots."""
    ids = list(nodes)
    indeg = np.array([in_deg.get(n, 0) for n in ids], dtype=float)
    pr = np.array([pagerank.get(n, 0) for n in ids], dtype=float)
    betw = np.array([betweenness.get(n, 0) for n in ids], dtype=float)

    panels = [
        (indeg, pr, "In-degree", "PageRank", "indeg_pr"),
        (indeg, betw, "In-degree", "Betweenness", "indeg_betw"),
        (betw, pr, "Betweenness", "PageRank", "betw_pr"),
    ]

    for x, y, xlabel, ylabel, suffix in panels:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        mask = (x > 0) & (y > 0)
        ax.scatter(x[mask], y[mask], s=3, alpha=0.3, color=color, edgecolors="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xlabel} vs {ylabel} ({title_prefix})")
        ax.grid(True, alpha=0.2, which="both")
        plt.tight_layout()
        out_path = OUTDIR / f"{out_prefix}_{suffix}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path}")


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
    print(f"  Loaded {len(node_names):,} decls, {len(edges_df):,} edges "
          f"({time.time()-t0:.1f}s)")

    # ================================================================
    # 1. Declaration-level centrality scatter
    # ================================================================
    print("\n=== Declaration-level centrality ===")
    print("Building G_thm ...")
    t0 = time.time()
    G_thm = nx.DiGraph()
    G_thm.add_nodes_from(node_names)
    for _, row in edges_df.iterrows():
        s, t = row["source"], row["target"]
        if s in node_names and t in node_names and s != t:
            G_thm.add_edge(s, t)
    print(f"  {G_thm.number_of_nodes():,} nodes, {G_thm.number_of_edges():,} edges "
          f"({time.time()-t0:.1f}s)")

    # In-degree
    print("  Computing in-degree ...")
    thm_indeg = dict(G_thm.in_degree())

    # PageRank
    print("  Computing PageRank ...")
    t0 = time.time()
    thm_pr = nx.pagerank(G_thm, alpha=0.85, max_iter=100, tol=1e-6)
    print(f"    ({time.time()-t0:.1f}s)")

    # Betweenness (sampled)
    print("  Computing betweenness (k=500) ...")
    t0 = time.time()
    thm_betw = nx.betweenness_centrality(G_thm, k=500, seed=42)
    print(f"    ({time.time()-t0:.1f}s)")

    # Plot — 3 separate files
    plot_centrality_separate(
        thm_indeg, thm_pr, thm_betw, node_names,
        title_prefix=r"$G_{\mathrm{thm}}$",
        color=COLORS["secondary"],
        out_prefix="thm_centrality",
    )

    # ================================================================
    # 2. Namespace-level centrality scatter
    # ================================================================
    print("\n=== Namespace-level centrality ===")

    def ns_at_depth(name, k=2):
        parts = name.split(".")
        if len(parts) <= k:
            return ".".join(parts[:-1]) if len(parts) > 1 else "_root_"
        return ".".join(parts[:k])

    print("Building G_ns^(2) ...")
    t0 = time.time()
    decl_to_ns = {name: ns_at_depth(name) for name in node_names}
    all_ns = set(decl_to_ns.values())

    # Weighted edges
    edge_weights = Counter()
    for _, row in edges_df.iterrows():
        s, t = row["source"], row["target"]
        if s not in decl_to_ns or t not in decl_to_ns:
            continue
        ns_s, ns_t = decl_to_ns[s], decl_to_ns[t]
        if ns_s != ns_t:
            edge_weights[(ns_s, ns_t)] += 1

    G_ns = nx.DiGraph()
    G_ns.add_nodes_from(all_ns)
    for (s, t), w in edge_weights.items():
        G_ns.add_edge(s, t, weight=w)
    print(f"  {G_ns.number_of_nodes():,} nodes, {G_ns.number_of_edges():,} edges "
          f"({time.time()-t0:.1f}s)")

    # In-degree (unweighted)
    ns_indeg = dict(G_ns.in_degree())

    # PageRank (weighted)
    print("  Computing PageRank ...")
    t0 = time.time()
    ns_pr = nx.pagerank(G_ns, alpha=0.85, weight="weight")
    print(f"    ({time.time()-t0:.1f}s)")

    # Betweenness (sampled)
    print("  Computing betweenness (k=300) ...")
    t0 = time.time()
    ns_betw = nx.betweenness_centrality(G_ns, k=300, weight="weight", seed=42)
    print(f"    ({time.time()-t0:.1f}s)")

    # Plot — 3 separate files
    plot_centrality_separate(
        ns_indeg, ns_pr, ns_betw, all_ns,
        title_prefix=r"$G_{\mathrm{ns}}^{(2)}$",
        color=COLORS["tertiary"],
        out_prefix="ns_centrality",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
