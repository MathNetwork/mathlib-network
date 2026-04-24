#!/usr/bin/env python3
"""
Run cascade analysis and robustness analysis.
Optimized: uses subgraph views instead of full graph copies.
Outputs to output/.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset


OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Color scheme
INDIGO = "#2C3E6B"
GOLD = "#C9A96E"
IVORY = "#F5F0E8"


def load_and_build():
    """Load from HuggingFace, build DiGraph, compute PageRank."""
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

    print("Computing PageRank...")
    t0 = time.time()
    pagerank = nx.pagerank(G, alpha=0.85)
    print(f"  PageRank done in {time.time() - t0:.1f}s")

    return G, pagerank


def cascade_analysis(G, pagerank, top_n=30):
    """
    Cascade impact analysis using subgraph views (no graph copy).
    For each top-PR node, measure how much the largest WCC shrinks.
    """
    print("\n" + "=" * 60)
    print("  CASCADE ANALYSIS")
    print("=" * 60)

    all_nodes = set(G.nodes)
    original_wcc_size = len(max(nx.weakly_connected_components(G), key=len))
    print(f"  Original largest WCC: {original_wcc_size:,}")

    pr_sorted = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    t0 = time.time()

    for i, (node, pr_score) in enumerate(pr_sorted, 1):
        remaining = all_nodes - {node}
        sub = G.subgraph(remaining)
        if sub.number_of_nodes() > 0:
            new_wcc = len(max(nx.weakly_connected_components(sub), key=len))
        else:
            new_wcc = 0

        impact = original_wcc_size - new_wcc
        impact_pct = 100.0 * impact / original_wcc_size
        kind = G.nodes[node].get("kind", "?")

        results.append({
            "rank": i,
            "name": node,
            "kind": kind,
            "pagerank": pr_score,
            "wcc_after_removal": new_wcc,
            "wcc_reduction": impact,
            "wcc_reduction_pct": impact_pct,
        })

        short = node if len(node) <= 45 else node[:42] + "..."
        print(f"  {i:3d}. [{kind:8s}] {short:45s}  WCC-{impact:>5,} ({impact_pct:5.2f}%)  [{time.time()-t0:.0f}s]")

    # Sort by impact and print summary
    results.sort(key=lambda x: x["wcc_reduction"], reverse=True)
    print(f"\n  Top 10 by cascade impact:")
    for i, r in enumerate(results[:10], 1):
        short = r["name"] if len(r["name"]) <= 45 else r["name"][:42] + "..."
        print(f"  {i:3d}. {short:45s}  WCC reduced by {r['wcc_reduction']:,} ({r['wcc_reduction_pct']:.2f}%)")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "cascade_top30.csv", index=False)
    print(f"\n  Saved: cascade_top30.csv  ({time.time()-t0:.1f}s total)")
    return results


def robustness_analysis(G, pagerank, max_frac=0.20, step=0.01):
    """
    Robustness analysis: random vs targeted node removal.
    Uses subgraph views instead of graph copies.
    Caps at 20% removal to keep runtime reasonable.
    """
    print("\n" + "=" * 60)
    print("  ROBUSTNESS ANALYSIS")
    print("=" * 60)

    n_nodes = G.number_of_nodes()
    all_nodes = set(G.nodes)
    fracs = np.arange(0, max_frac + step / 2, step)

    # Pre-sort nodes
    pr_sorted = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    targeted_order = [n for n, _ in pr_sorted]

    random_order = list(G.nodes())
    np.random.seed(42)
    np.random.shuffle(random_order)

    results_random = []
    results_targeted = []

    t0 = time.time()
    print(f"  Fractions: 0% to {max_frac*100:.0f}% in {step*100:.0f}% steps ({len(fracs)} points)")

    for fi, frac in enumerate(fracs):
        n_remove = int(frac * n_nodes)

        # Random removal — subgraph view
        remaining_random = all_nodes - set(random_order[:n_remove])
        if remaining_random:
            sub_r = G.subgraph(remaining_random)
            wcc_r = len(max(nx.weakly_connected_components(sub_r), key=len))
        else:
            wcc_r = 0
        results_random.append(wcc_r / n_nodes)

        # Targeted removal — subgraph view
        remaining_targeted = all_nodes - set(targeted_order[:n_remove])
        if remaining_targeted:
            sub_t = G.subgraph(remaining_targeted)
            wcc_t = len(max(nx.weakly_connected_components(sub_t), key=len))
        else:
            wcc_t = 0
        results_targeted.append(wcc_t / n_nodes)

        elapsed = time.time() - t0
        if frac > 0 and (frac * 100) % 5 < step * 100 + 0.001:
            print(f"  {frac*100:5.1f}% removed: Random={results_random[-1]:.4f}  Targeted={results_targeted[-1]:.4f}  [{elapsed:.0f}s]")

    print(f"\n  Completed in {time.time()-t0:.1f}s")

    # Save data
    rob_df = pd.DataFrame({
        "fraction_removed": fracs,
        "random_wcc_ratio": results_random,
        "targeted_wcc_ratio": results_targeted,
    })
    rob_df.to_csv(OUTPUT_DIR / "robustness_data.csv", index=False)
    print(f"  Saved: robustness_data.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(IVORY)
    ax.set_facecolor(IVORY)

    ax.plot(fracs * 100, results_random, c=INDIGO, linewidth=2,
            label="Random removal", marker="o", markersize=3)
    ax.plot(fracs * 100, results_targeted, c=GOLD, linewidth=2,
            label="Targeted removal (by PageRank)", marker="s", markersize=3)

    ax.set_xlabel("Fraction of nodes removed (%)")
    ax.set_ylabel("Largest WCC / Total nodes")
    ax.set_title("Network Robustness: Random vs Targeted Attack", color=INDIGO)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_frac * 100)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "robustness_curve.png", dpi=150, facecolor=IVORY)
    plt.close()
    print(f"  Saved: robustness_curve.png")

    return {"random": results_random, "targeted": results_targeted}


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    G, pagerank = load_and_build()
    cascade_analysis(G, pagerank, top_n=30)
    robustness_analysis(G, pagerank, max_frac=0.20, step=0.01)

    print(f"\n{'=' * 60}")
    print(f"  ALL DONE in {time.time() - start:.1f}s")
    print(f"{'=' * 60}")
    print(f"\n  Output files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        if f.is_file():
            sz = f.stat().st_size
            if sz > 1024 * 1024:
                print(f"    {f.name}: {sz / 1024 / 1024:.1f} MB")
            elif sz > 1024:
                print(f"    {f.name}: {sz / 1024:.1f} KB")
            else:
                print(f"    {f.name}: {sz} B")


if __name__ == "__main__":
    main()
