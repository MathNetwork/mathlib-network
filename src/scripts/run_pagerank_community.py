#!/usr/bin/env python3
"""
Run PageRank and Community Detection only.
Outputs CSV files to output/.
"""

import time
from pathlib import Path
from collections import Counter

import pandas as pd
import networkx as nx
from datasets import load_dataset


OUTPUT_DIR = Path(__file__).parent.parent / "output"


def load_and_build():
    """Load from HuggingFace, build DiGraph."""
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


def run_pagerank(G):
    """Compute PageRank, save top 50 to CSV."""
    print("\n" + "=" * 60)
    print("  PAGERANK")
    print("=" * 60)

    t0 = time.time()
    pr = nx.pagerank(G, alpha=0.85)
    print(f"  Computed in {time.time() - t0:.1f}s")

    pr_sorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)

    # Print top 20
    print("\n  Top 20 by PageRank:")
    rows = []
    for i, (node, score) in enumerate(pr_sorted[:50], 1):
        kind = G.nodes[node].get("kind", "?")
        module = G.nodes[node].get("module", "?")
        if i <= 20:
            short = node if len(node) <= 50 else node[:47] + "..."
            print(f"  {i:3d}. {score:.6f}  [{kind:8s}]  {short}")
        rows.append({
            "rank": i,
            "name": node,
            "kind": kind,
            "module": module,
            "pagerank": score,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "pagerank_top50.csv", index=False)
    print(f"\n  Saved: {OUTPUT_DIR / 'pagerank_top50.csv'}")
    return pr


def run_hits(G):
    """Compute HITS, save top 50 to CSV."""
    print("\n" + "=" * 60)
    print("  HITS (Hubs & Authorities)")
    print("=" * 60)

    t0 = time.time()
    try:
        hubs, authorities = nx.hits(G, max_iter=200, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        print("  HITS did not converge in 200 iterations, using partial result...")
        hubs, authorities = nx.hits(G, max_iter=200, tol=1e-3)
    print(f"  Computed in {time.time() - t0:.1f}s")

    auth_sorted = sorted(authorities.items(), key=lambda x: x[1], reverse=True)

    print("\n  Top 20 Authorities:")
    rows = []
    for i, (node, auth_score) in enumerate(auth_sorted[:50], 1):
        kind = G.nodes[node].get("kind", "?")
        hub_score = hubs.get(node, 0)
        if i <= 20:
            short = node if len(node) <= 50 else node[:47] + "..."
            print(f"  {i:3d}. auth={auth_score:.6f}  hub={hub_score:.6f}  [{kind:8s}]  {short}")
        rows.append({
            "authority_rank": i,
            "name": node,
            "kind": kind,
            "authority_score": auth_score,
            "hub_score": hub_score,
        })

    hub_sorted = sorted(hubs.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 20 Hubs:")
    for i, (node, hub_score) in enumerate(hub_sorted[:20], 1):
        kind = G.nodes[node].get("kind", "?")
        short = node if len(node) <= 50 else node[:47] + "..."
        print(f"  {i:3d}. hub={hub_score:.6f}  [{kind:8s}]  {short}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "hits_top50.csv", index=False)
    print(f"\n  Saved: {OUTPUT_DIR / 'hits_top50.csv'}")
    return hubs, authorities


def run_community(G, pagerank):
    """Louvain community detection on largest WCC."""
    import community as community_louvain

    print("\n" + "=" * 60)
    print("  COMMUNITY DETECTION (LOUVAIN)")
    print("=" * 60)

    # Get largest WCC
    t0 = time.time()
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    G_wcc = G.subgraph(largest_wcc).copy()
    G_undirected = G_wcc.to_undirected()
    print(f"  Largest WCC: {len(largest_wcc):,} nodes")
    print(f"  Undirected edges: {G_undirected.number_of_edges():,}")

    print("  Running Louvain...")
    partition = community_louvain.best_partition(G_undirected)
    elapsed = time.time() - t0
    print(f"  Computed in {elapsed:.1f}s")

    n_communities = len(set(partition.values()))
    print(f"  Found {n_communities} communities")

    # Modularity
    modularity = community_louvain.modularity(partition, G_undirected)
    print(f"  Modularity: {modularity:.4f}")

    # Community size distribution
    comm_sizes = Counter(partition.values())
    sizes = sorted(comm_sizes.values(), reverse=True)
    print(f"\n  Size distribution:")
    print(f"    max={sizes[0]:,}  median={sizes[len(sizes)//2]:,}  min={sizes[-1]}")
    print(f"    Communities with >1000 nodes: {sum(1 for s in sizes if s > 1000)}")
    print(f"    Communities with >100 nodes:  {sum(1 for s in sizes if s > 100)}")

    # Top 15 communities with representatives
    top_communities = comm_sizes.most_common(15)
    community_details = []

    print(f"\n  Top 15 communities:")
    for comm_id, size in top_communities:
        comm_nodes = [n for n, c in partition.items() if c == comm_id]
        # Sort by PageRank
        comm_nodes_pr = sorted(
            [(n, pagerank.get(n, 0)) for n in comm_nodes],
            key=lambda x: x[1],
            reverse=True,
        )
        top5 = comm_nodes_pr[:5]

        # What kinds dominate?
        kinds = Counter(G.nodes[n].get("kind", "?") for n in comm_nodes)
        kind_str = ", ".join(f"{k}:{v}" for k, v in kinds.most_common(3))

        # What modules dominate?
        modules = Counter(str(G.nodes[n].get("module", "?")) for n in comm_nodes)
        # Get top-level module (first part before '.')
        top_modules = Counter()
        for n in comm_nodes:
            mod = str(G.nodes[n].get("module", ""))
            top_mod = mod.split(".")[0] if mod else "?"
            top_modules[top_mod] += 1
        top_mod_str = ", ".join(f"{k}({v})" for k, v in top_modules.most_common(5))

        print(f"\n  Community {comm_id} — {size:,} nodes")
        print(f"    Kinds: {kind_str}")
        print(f"    Top modules: {top_mod_str}")
        print(f"    Representatives (by PageRank):")
        for node, pr in top5:
            kind = G.nodes[node].get("kind", "?")
            short = node if len(node) <= 55 else node[:52] + "..."
            print(f"      [{kind:8s}] {short}  (PR={pr:.6f})")

        community_details.append({
            "community_id": comm_id,
            "size": size,
            "top_nodes": [n for n, _ in top5],
            "dominant_kinds": kind_str,
            "dominant_modules": top_mod_str,
        })

    # Save full partition
    comm_data = []
    for node, comm_id in partition.items():
        comm_data.append({
            "name": node,
            "kind": G.nodes[node].get("kind", "?"),
            "module": G.nodes[node].get("module", "?"),
            "community": comm_id,
        })
    df = pd.DataFrame(comm_data)
    df.to_csv(OUTPUT_DIR / "communities.csv", index=False)
    print(f"\n  Saved: {OUTPUT_DIR / 'communities.csv'} ({len(df):,} rows)")

    # Save community summary
    summary_rows = []
    for comm_id, size in comm_sizes.most_common():
        comm_nodes = [n for n, c in partition.items() if c == comm_id]
        top_pr = sorted(
            [(n, pagerank.get(n, 0)) for n in comm_nodes],
            key=lambda x: x[1], reverse=True,
        )[:3]
        summary_rows.append({
            "community_id": comm_id,
            "size": size,
            "rep1": top_pr[0][0] if len(top_pr) > 0 else "",
            "rep2": top_pr[1][0] if len(top_pr) > 1 else "",
            "rep3": top_pr[2][0] if len(top_pr) > 2 else "",
        })
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "community_summary.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'community_summary.csv'}")

    return partition, community_details


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    G = load_and_build()

    # 1. PageRank
    pagerank = run_pagerank(G)

    # 2. HITS
    run_hits(G)

    # 3. Community detection
    run_community(G, pagerank)

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
