#!/usr/bin/env python3
"""Build mathlib_namespaces_k2.csv with per-namespace precomputed metrics.

Columns: namespace, decl_count, in_degree, out_degree, edge_weight_sum,
         pagerank, betweenness, community_id, cross_ns_ratio.

Reuses Phase 1 enriched nodes for namespace labels. Builds weighted
namespace graph following populate_cache.py:496-522 (build_ns_graph).
"""

import time
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

HF_DIR = Path.home() / ".cache/huggingface/hub/datasets--MathNetwork--MathlibGraph/snapshots/bc4173ec3beda64713ae81f602ce224491c61703"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "release"


def ns_at_depth(name: str, k: int = 2) -> str:
    parts = name.split(".")
    if len(parts) <= k:
        return ".".join(parts[:-1]) if len(parts) > 1 else "_root_"
    return ".".join(parts[:k])


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load nodes (deduped) and edges ----
    print("Loading data ...")
    nodes_df = pd.read_csv(HF_DIR / "mathlib_nodes.csv")
    nodes_df = nodes_df.drop_duplicates(subset="name", keep="first")
    edges_df = pd.read_csv(HF_DIR / "mathlib_edges.csv")
    print(f"  {len(nodes_df):,} nodes, {len(edges_df):,} edges")

    # ---- 2. Build namespace graph (matches populate_cache.py:496-522) ----
    print("Building G_ns^(2) ...")
    t0 = time.time()
    node_names = set(nodes_df["name"].dropna())
    decl_to_ns = {name: ns_at_depth(name, 2) for name in node_names}
    all_ns = set(decl_to_ns.values())

    # decl_count per namespace
    ns_decl_count = Counter(decl_to_ns.values())

    # Count total edges per declaration (for cross_ns_ratio)
    ns_total_edges = Counter()    # total edges touching this ns
    ns_cross_edges = Counter()    # cross-namespace edges touching this ns

    edge_weights = Counter()
    edge_set = set()

    src_arr = edges_df["source"].values
    tgt_arr = edges_df["target"].values
    for i in range(len(src_arr)):
        s, t = src_arr[i], tgt_arr[i]
        if s == t:
            continue
        ns_s = decl_to_ns.get(s)
        ns_t = decl_to_ns.get(t)
        if ns_s is None or ns_t is None:
            continue
        ns_total_edges[ns_s] += 1
        ns_total_edges[ns_t] += 1
        if ns_s != ns_t:
            edge_weights[(ns_s, ns_t)] += 1
            edge_set.add((ns_s, ns_t))
            ns_cross_edges[ns_s] += 1
            ns_cross_edges[ns_t] += 1

    G_ns = nx.DiGraph()
    G_ns.add_nodes_from(all_ns)
    for (s, t), w in edge_weights.items():
        G_ns.add_edge(s, t, weight=w)
    print(f"  {G_ns.number_of_nodes():,} nodes, {G_ns.number_of_edges():,} edges ({time.time()-t0:.1f}s)")

    # ---- 3. Degree (unweighted, from edge_set — matches cache_ns_degree) ----
    print("Computing metrics ...")
    in_counter = Counter()
    out_counter = Counter()
    for s, t in edge_set:
        out_counter[s] += 1
        in_counter[t] += 1

    # Edge weight sum (weighted degree)
    weighted_deg = Counter()
    for (s, t), w in edge_weights.items():
        weighted_deg[s] += w
        weighted_deg[t] += w

    # ---- 4. PageRank (weighted, matches populate_cache.py:565) ----
    t0 = time.time()
    pr = nx.pagerank(G_ns, alpha=0.85, weight="weight")
    print(f"  PageRank ({time.time()-t0:.1f}s)")

    # ---- 5. Betweenness (k=300, weighted, matches populate_cache.py:569) ----
    t0 = time.time()
    bc = nx.betweenness_centrality(G_ns, k=300, weight="weight", seed=42)
    print(f"  Betweenness k=300 ({time.time()-t0:.1f}s)")

    # ---- 6. DAG layers (handle cycles via condensation) ----
    t0 = time.time()
    is_dag = nx.is_directed_acyclic_graph(G_ns)
    print(f"  Is DAG: {is_dag}")

    layer_map = {}
    if not is_dag:
        condensed = nx.condensation(G_ns)
        # condensed.nodes have 'members' attribute = set of original nodes
        scc_layer = {}
        for layer_idx, gen in enumerate(nx.topological_generations(condensed)):
            for scc_node in gen:
                scc_layer[scc_node] = layer_idx
        # Map back to original nodes
        for scc_node, data in condensed.nodes(data=True):
            for orig_node in data["members"]:
                layer_map[orig_node] = scc_layer[scc_node]
        max_layer = max(scc_layer.values()) if scc_layer else 0
        print(f"  Condensed: {condensed.number_of_nodes()} super-nodes, {max_layer + 1} layers ({time.time()-t0:.1f}s)")
    else:
        for layer_idx, gen in enumerate(nx.topological_generations(G_ns)):
            for n in gen:
                layer_map[n] = layer_idx
        max_layer = max(layer_map.values()) if layer_map else 0
        print(f"  {max_layer + 1} layers ({time.time()-t0:.1f}s)")

    # ---- 7. Louvain (weighted undirected, matches populate_cache.py:601-609) ----
    t0 = time.time()
    import community as community_louvain
    from sklearn.metrics import normalized_mutual_info_score

    G_und = nx.Graph()
    G_und.add_nodes_from(G_ns.nodes())
    for u, v, d in G_ns.edges(data=True):
        w = d.get("weight", 1)
        if G_und.has_edge(u, v):
            G_und[u][v]["weight"] += w
        else:
            G_und.add_edge(u, v, weight=w)
    partition = community_louvain.best_partition(G_und, random_state=42)
    modularity_score = community_louvain.modularity(partition, G_und)
    n_communities = len(set(partition.values()))

    # NMI with top-level prefix
    nodes_list = sorted(all_ns)
    comm_labels = [partition.get(n, -1) for n in nodes_list]
    pfx_labels = [n.split(".")[0] if "." in n else n for n in nodes_list]
    nmi = normalized_mutual_info_score(pfx_labels, comm_labels)
    print(f"  Louvain: {n_communities} communities, modularity={modularity_score:.4f}, NMI={nmi:.4f} ({time.time()-t0:.1f}s)")

    # ---- 8. Cross-namespace ratio per namespace ----
    cross_ns_ratio = {}
    for ns in all_ns:
        total = ns_total_edges.get(ns, 0)
        cross = ns_cross_edges.get(ns, 0)
        cross_ns_ratio[ns] = cross / total if total > 0 else 0.0

    # Overall cross-ns ratio
    total_all = sum(ns_total_edges.values())
    cross_all = sum(ns_cross_edges.values())
    # Each cross-ns edge is counted twice (once per endpoint), same for total
    global_cross_ratio = cross_all / total_all if total_all > 0 else 0.0

    # ---- 9. Assemble DataFrame ----
    namespaces = sorted(all_ns)
    df = pd.DataFrame({
        "namespace": namespaces,
        "decl_count": [ns_decl_count.get(ns, 0) for ns in namespaces],
        "in_degree": [in_counter.get(ns, 0) for ns in namespaces],
        "out_degree": [out_counter.get(ns, 0) for ns in namespaces],
        "edge_weight_sum": [weighted_deg.get(ns, 0) for ns in namespaces],
        "pagerank": [pr.get(ns, 0.0) for ns in namespaces],
        "betweenness": [bc.get(ns, 0.0) for ns in namespaces],
        "community_id": [partition.get(ns, -1) for ns in namespaces],
        "cross_ns_ratio": [cross_ns_ratio.get(ns, 0.0) for ns in namespaces],
    })

    out_path = OUT_DIR / "mathlib_namespaces_k2.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # ---- 10. Verification ----
    print("\n=== Verification ===")
    print(f"  Row count: {len(df)} (expected ~10,097)")
    print(f"  DAG layers: {max_layer + 1}, max depth {max_layer} (expected ~8)")
    print(f"  Louvain: {n_communities} communities, modularity={modularity_score:.4f}")
    print(f"  NMI (namespace vs community): {nmi:.4f} (expected ~0.34)")
    print(f"  Global cross-ns ratio: {global_cross_ratio:.4f} (expected ~0.509)")
    print(f"  Sum decl_count: {df['decl_count'].sum():,} (expected ~308,129)")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")
    print("Phase 3 DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
