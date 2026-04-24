#!/usr/bin/env python3
"""Enrich mathlib_nodes.csv with 8 precomputed columns.

Adds: namespace_depth2, namespace_depth3, in_degree, out_degree,
      pagerank, betweenness, community_id, dag_layer.

Reads from HuggingFace cache, writes to data/release/mathlib_nodes.csv.
Parameters match populate_cache.py (the canonical pipeline).
"""

import time
import sys
from collections import defaultdict, deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ns_at_depth(name: str, k: int = 2) -> str:
    """Namespace at depth k (from populate_cache.py:67-71)."""
    parts = name.split(".")
    if len(parts) <= k:
        return ".".join(parts[:-1]) if len(parts) > 1 else "_root_"
    return ".".join(parts[:k])


def compute_dag_layers(G):
    """Kahn's algorithm returning per-node layer dict.

    Adapted from populate_cache.py:117-150 (topological_layers_fast).
    Nodes in cycles get layer = -1.
    """
    in_deg = defaultdict(int)
    adj = defaultdict(list)
    for u, v in G.edges():
        adj[u].append(v)
        in_deg[v] += 1
    for n in G.nodes():
        if n not in in_deg:
            in_deg[n] = 0

    layer = {}
    queue = deque()
    for n in G.nodes():
        if in_deg[n] == 0:
            layer[n] = 0
            queue.append(n)

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            in_deg[v] -= 1
            new_layer = layer[u] + 1
            if v in layer:
                layer[v] = max(layer[v], new_layer)
            else:
                layer[v] = new_layer
            if in_deg[v] == 0:
                queue.append(v)

    # Nodes not reached (in cycles) get -1
    for n in G.nodes():
        if n not in layer:
            layer[n] = -1
    return layer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    HF_DIR = Path.home() / ".cache/huggingface/hub/datasets--MathNetwork--MathlibGraph/snapshots/bc4173ec3beda64713ae81f602ce224491c61703"
    OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "release"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and deduplicate nodes
    print("Loading nodes ...")
    nodes_df = pd.read_csv(HF_DIR / "mathlib_nodes.csv")
    print(f"  Raw rows: {len(nodes_df):,}")
    nodes_df = nodes_df.drop_duplicates(subset="name", keep="first").reset_index(drop=True)
    print(f"  After dedup: {len(nodes_df):,}")
    node_set = set(nodes_df["name"].dropna())

    # 2. Load edges and build graph
    print("Loading edges ...")
    edges_df = pd.read_csv(HF_DIR / "mathlib_edges.csv")
    print(f"  {len(edges_df):,} edges")

    print("Building G_thm ...")
    t0 = time.time()
    G = nx.DiGraph()
    G.add_nodes_from(node_set)
    src = edges_df["source"].values
    tgt = edges_df["target"].values
    for i in range(len(src)):
        s, t = src[i], tgt[i]
        if s != t and s in node_set and t in node_set:
            G.add_edge(s, t)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges ({time.time()-t0:.1f}s)")

    # 3. Namespace columns
    print("Computing namespaces ...")
    nodes_df["namespace_depth2"] = nodes_df["name"].map(lambda n: ns_at_depth(n, 2))
    nodes_df["namespace_depth3"] = nodes_df["name"].map(lambda n: ns_at_depth(n, 3))

    # 4. Degree
    print("Computing degree ...")
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    nodes_df["in_degree"] = nodes_df["name"].map(in_deg).fillna(0).astype(int)
    nodes_df["out_degree"] = nodes_df["name"].map(out_deg).fillna(0).astype(int)

    # 5. PageRank (alpha=0.85, matches populate_cache.py:394)
    print("Computing PageRank ...")
    t0 = time.time()
    pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    print(f"  ({time.time()-t0:.1f}s)")
    nodes_df["pagerank"] = nodes_df["name"].map(pr).fillna(0.0)

    # 6. Betweenness (k=500, seed=42, matches populate_cache.py:398)
    print("Computing betweenness (k=500) ...")
    t0 = time.time()
    bc = nx.betweenness_centrality(G, k=500, seed=42)
    print(f"  ({time.time()-t0:.1f}s)")
    nodes_df["betweenness"] = nodes_df["name"].map(bc).fillna(0.0)

    # 7. Louvain communities (resolution=1.0, random_state=42, matches populate_cache.py:431)
    print("Computing Louvain communities ...")
    t0 = time.time()
    import community as community_louvain
    G_und = G.to_undirected()
    partition = community_louvain.best_partition(G_und, random_state=42)
    n_communities = len(set(partition.values()))
    modularity = community_louvain.modularity(partition, G_und)
    print(f"  {n_communities} communities, modularity={modularity:.4f} ({time.time()-t0:.1f}s)")
    nodes_df["community_id"] = nodes_df["name"].map(partition).fillna(-1).astype(int)

    # 8. DAG layer (Kahn's algorithm; nodes in cycles get -1)
    print("Computing DAG layers ...")
    t0 = time.time()
    layer_map = compute_dag_layers(G)
    assigned = [v for v in layer_map.values() if v >= 0]
    n_cycle = sum(1 for v in layer_map.values() if v < 0)
    max_layer = max(assigned) if assigned else 0
    print(f"  {max_layer + 1} layers, max depth {max_layer}, {n_cycle} nodes in cycles ({time.time()-t0:.1f}s)")
    nodes_df["dag_layer"] = nodes_df["name"].map(layer_map).fillna(-1).astype(int)

    # 9. Save
    out_path = OUT_DIR / "mathlib_nodes.csv"
    nodes_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"  {len(nodes_df):,} rows, {len(nodes_df.columns)} columns")
    print(f"  Columns: {list(nodes_df.columns)}")

    # 10. Verification
    print("\n=== Verification ===")
    ok = True

    # Row count
    if len(nodes_df) != 308129:
        print(f"  FAIL: row count {len(nodes_df):,} != 308,129")
        ok = False
    else:
        print(f"  OK: row count = {len(nodes_df):,}")

    # Null check on new columns
    new_cols = ["namespace_depth2", "namespace_depth3", "in_degree", "out_degree",
                "pagerank", "betweenness", "community_id", "dag_layer"]
    for col in new_cols:
        n_null = nodes_df[col].isna().sum()
        if n_null > 0:
            print(f"  WARN: {col} has {n_null} nulls")
            ok = False

    # Eq.refl in-degree
    eq_refl = nodes_df.loc[nodes_df["name"] == "Eq.refl"]
    if not eq_refl.empty:
        eq_deg = eq_refl["in_degree"].iloc[0]
        print(f"  Eq.refl in_degree = {eq_deg:,} (expected ~69,580)")

    # Max in-degree
    max_in_row = nodes_df.loc[nodes_df["in_degree"].idxmax()]
    print(f"  Max in_degree: {max_in_row['name']} = {max_in_row['in_degree']:,} (expected OfNat.ofNat ~89,936)")

    # DAG depth
    print(f"  DAG max layer = {max_layer} (expected ~84)")

    # Communities
    print(f"  Communities = {n_communities}, modularity = {modularity:.4f} (expected ~22, ~0.4757)")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")

    if ok:
        print("Phase 1 PASSED")
    else:
        print("Phase 1 completed with warnings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
