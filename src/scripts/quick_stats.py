#!/usr/bin/env python3
"""
Lightweight quick statistics for the Mathlib dependency graph.
Skips expensive analyses (PageRank, community detection, cascade).
"""

import time
from collections import Counter

import numpy as np
import networkx as nx
from datasets import load_dataset


def load_data():
    """Load from HuggingFace and build DiGraph."""
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
    print(f"  nodes_df columns: {list(nodes_df.columns)}")
    print(f"  edges_df columns: {list(edges_df.columns)}")
    print(f"  nodes_df shape: {nodes_df.shape}")
    print(f"  edges_df shape: {edges_df.shape}")
    return nodes_df, edges_df


def build_graph(nodes_df, edges_df):
    """Build NetworkX DiGraph."""
    print("\nBuilding graph...")
    t0 = time.time()
    G = nx.DiGraph()

    for _, row in nodes_df.iterrows():
        G.add_node(row["name"], kind=row["kind"], module=row["module"])

    node_set = set(G.nodes)
    skipped = 0
    for _, row in edges_df.iterrows():
        if row["source"] in node_set and row["target"] in node_set:
            G.add_edge(row["source"], row["target"])
        else:
            skipped += 1

    print(f"  Built in {time.time() - t0:.1f}s")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    if skipped:
        print(f"  Skipped edges (missing endpoint): {skipped:,}")
    return G


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def basic_stats(G):
    """1. Basic statistics."""
    section("1. BASIC STATISTICS")
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"  Nodes:   {n:,}")
    print(f"  Edges:   {m:,}")
    print(f"  Density: {nx.density(G):.8f}")

    # Declaration type distribution
    kinds = [G.nodes[v].get("kind", "unknown") for v in G.nodes]
    kind_counts = Counter(kinds)
    print(f"\n  Declaration types ({len(kind_counts)} kinds):")
    for kind, count in kind_counts.most_common():
        print(f"    {kind:20s}  {count:>8,}  ({count/n*100:5.1f}%)")


def degree_stats(G):
    """2. Degree distribution summary."""
    section("2. DEGREE DISTRIBUTION")
    in_deg = np.array([d for _, d in G.in_degree()])
    out_deg = np.array([d for _, d in G.out_degree()])

    for label, arr in [("In-degree", in_deg), ("Out-degree", out_deg)]:
        print(f"\n  {label}:")
        print(f"    mean   = {arr.mean():.2f}")
        print(f"    median = {np.median(arr):.1f}")
        print(f"    std    = {arr.std():.2f}")
        print(f"    max    = {arr.max()}")
        print(f"    min    = {arr.min()}")
        print(f"    zero-degree nodes = {(arr == 0).sum():,}")


def top_in_degree(G, k=20):
    """3. Top-k nodes by in-degree (most cited)."""
    section("3. TOP 20 BY IN-DEGREE (most cited)")
    top = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:k]
    for i, (node, deg) in enumerate(top, 1):
        kind = G.nodes[node].get("kind", "?")
        mod = G.nodes[node].get("module", "?")
        # Truncate long names
        short_name = node if len(node) <= 55 else node[:52] + "..."
        print(f"  {i:3d}. [{kind:8s}] {short_name:55s}  in={deg:5d}  mod={mod}")


def top_out_degree(G, k=20):
    """4. Top-k nodes by out-degree (most dependencies)."""
    section("4. TOP 20 BY OUT-DEGREE (most dependencies)")
    top = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:k]
    for i, (node, deg) in enumerate(top, 1):
        kind = G.nodes[node].get("kind", "?")
        mod = G.nodes[node].get("module", "?")
        short_name = node if len(node) <= 55 else node[:52] + "..."
        print(f"  {i:3d}. [{kind:8s}] {short_name:55s}  out={deg:5d}  mod={mod}")


def module_edge_analysis(G):
    """5. Intra-module vs cross-module edges."""
    section("5. MODULE EDGE ANALYSIS")
    intra = 0
    cross = 0
    missing = 0
    for u, v in G.edges():
        mod_u = G.nodes[u].get("module")
        mod_v = G.nodes[v].get("module")
        if not mod_u or not mod_v:
            missing += 1
            continue
        if mod_u == mod_v:
            intra += 1
        else:
            cross += 1

    total = intra + cross
    print(f"  Intra-module edges: {intra:>10,}  ({intra/total*100:.1f}%)")
    print(f"  Cross-module edges: {cross:>10,}  ({cross/total*100:.1f}%)")
    if missing:
        print(f"  Missing module info: {missing:>8,}")

    # How many distinct modules?
    modules = set(G.nodes[v].get("module", "") for v in G.nodes)
    modules.discard("")
    modules.discard(None)
    print(f"  Distinct modules:   {len(modules):>10,}")


def connected_components(G):
    """6. Connected components (weakly)."""
    section("6. WEAKLY CONNECTED COMPONENTS")
    t0 = time.time()
    wccs = list(nx.weakly_connected_components(G))
    n = G.number_of_nodes()
    sizes = sorted([len(c) for c in wccs], reverse=True)
    print(f"  Computed in {time.time() - t0:.1f}s")
    print(f"  Number of WCCs: {len(wccs):,}")
    print(f"  Largest WCC:    {sizes[0]:,} nodes ({sizes[0]/n*100:.1f}%)")
    if len(sizes) > 1:
        print(f"  2nd largest:    {sizes[1]:,} nodes")
    if len(sizes) > 2:
        print(f"  3rd largest:    {sizes[2]:,} nodes")
    # Size distribution of small components
    small = [s for s in sizes if s < 10]
    print(f"  Components with <10 nodes: {len(small):,}")
    singleton = sizes.count(1)
    print(f"  Singleton components:      {singleton:,}")


def main():
    start = time.time()

    nodes_df, edges_df = load_data()
    G = build_graph(nodes_df, edges_df)

    basic_stats(G)
    degree_stats(G)
    top_in_degree(G)
    top_out_degree(G)
    module_edge_analysis(G)
    connected_components(G)

    print(f"\n{'='*60}")
    print(f"  DONE in {time.time() - start:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
