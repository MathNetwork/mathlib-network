#!/usr/bin/env python3
"""
Run betweenness centrality (sampled approximation).
Outputs to output/.
"""

import time
from pathlib import Path

import pandas as pd
import networkx as nx
from datasets import load_dataset


OUTPUT_DIR = Path(__file__).parent.parent / "output"


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


def run_betweenness(G, k=500):
    print(f"\n{'=' * 60}")
    print(f"  BETWEENNESS CENTRALITY (sampled, k={k})")
    print(f"{'=' * 60}")

    t0 = time.time()
    print(f"  Computing (this may take several minutes)...")
    bc = nx.betweenness_centrality(G, k=k, seed=42)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    bc_sorted = sorted(bc.items(), key=lambda x: x[1], reverse=True)

    # Print top 30
    print(f"\n  Top 30 bridge nodes (betweenness centrality):")
    rows = []
    for i, (node, score) in enumerate(bc_sorted[:50], 1):
        kind = G.nodes[node].get("kind", "?")
        module = G.nodes[node].get("module", "?")
        if i <= 30:
            short = node if len(node) <= 50 else node[:47] + "..."
            print(f"  {i:3d}. {score:.6f}  [{kind:8s}]  {short}")
        rows.append({
            "rank": i,
            "name": node,
            "kind": kind,
            "module": module,
            "betweenness": score,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "betweenness_top50.csv", index=False)
    print(f"\n  Saved: betweenness_top50.csv")

    # Summary stats
    import numpy as np
    vals = list(bc.values())
    print(f"\n  Summary:")
    print(f"    mean   = {np.mean(vals):.8f}")
    print(f"    median = {np.median(vals):.8f}")
    print(f"    std    = {np.std(vals):.8f}")
    print(f"    max    = {np.max(vals):.6f}")
    print(f"    nodes with BC > 0: {sum(1 for v in vals if v > 0):,} / {len(vals):,}")

    return bc


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    G = load_and_build()
    run_betweenness(G, k=500)

    print(f"\n{'=' * 60}")
    print(f"  DONE in {time.time() - start:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
