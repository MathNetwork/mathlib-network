#!/usr/bin/env python3
"""
Centrality analysis: PageRank, HITS, Betweenness.

Part II: Structure - Centrality Measures
"""

from pathlib import Path

import pandas as pd
import networkx as nx


def pagerank_hits_analysis(
    G: nx.DiGraph,
    output_dir: Path = None
) -> tuple[dict, dict, dict]:
    """
    Compute PageRank and HITS scores.

    Args:
        G: NetworkX DiGraph
        output_dir: Directory to save CSV files

    Returns:
        (pagerank_dict, hubs_dict, authorities_dict)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data"

    print("\n" + "="*60)
    print("PAGERANK + HITS")
    print("="*60)

    # PageRank
    print("\nComputing PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85)
    pr_sorted = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:50]

    pr_data = []
    print("\nPageRank Top 10:")
    for i, (node_id, score) in enumerate(pr_sorted[:10], 1):
        name = G.nodes[node_id].get('name', node_id)
        kind = G.nodes[node_id].get('kind', '?')
        print(f"  {i:2d}. {score:.6f} [{kind:10s}] {name}")
        pr_data.append({'rank': i, 'id': node_id, 'name': name, 'kind': kind, 'score': score})

    for i, (node_id, score) in enumerate(pr_sorted[10:50], 11):
        name = G.nodes[node_id].get('name', node_id)
        kind = G.nodes[node_id].get('kind', '?')
        pr_data.append({'rank': i, 'id': node_id, 'name': name, 'kind': kind, 'score': score})

    pd.DataFrame(pr_data).to_csv(output_dir / "pagerank_top50.csv", index=False)
    print("Saved: pagerank_top50.csv")

    # HITS
    print("\nComputing HITS...")
    hubs, authorities = nx.hits(G, max_iter=100)

    auth_sorted = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:50]

    hits_data = []
    print("\nHITS Authority Top 10:")
    for i, (node_id, score) in enumerate(auth_sorted[:10], 1):
        name = G.nodes[node_id].get('name', node_id)
        kind = G.nodes[node_id].get('kind', '?')
        print(f"  {i:2d}. {score:.6f} [{kind:10s}] {name}")

    for i, (node_id, score) in enumerate(auth_sorted, 1):
        name = G.nodes[node_id].get('name', node_id)
        kind = G.nodes[node_id].get('kind', '?')
        hub_score = hubs.get(node_id, 0)
        hits_data.append({
            'authority_rank': i,
            'id': node_id,
            'name': name,
            'kind': kind,
            'authority_score': score,
            'hub_score': hub_score
        })

    pd.DataFrame(hits_data).to_csv(output_dir / "hits_top50.csv", index=False)
    print("Saved: hits_top50.csv")

    return pagerank, hubs, authorities


def betweenness_analysis(
    G: nx.DiGraph,
    k: int = 2000,
    output_dir: Path = None
) -> dict:
    """
    Compute betweenness centrality (sampled for large graphs).

    Betweenness centrality identifies "bridge" nodes that lie on
    many shortest paths between other nodes.

    Args:
        G: NetworkX DiGraph
        k: Number of samples for approximation
        output_dir: Directory to save CSV files

    Returns:
        Dictionary mapping node_id -> betweenness score
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data"

    print("\n" + "="*60)
    print("BETWEENNESS CENTRALITY (BRIDGE ANALYSIS)")
    print("="*60)

    print(f"Computing betweenness centrality (sampled, k={k})...")
    n_nodes = G.number_of_nodes()
    k = min(k, n_nodes)

    betweenness = nx.betweenness_centrality(G, k=k)
    bc_sorted = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:30]

    bc_data = []
    print("\nBetweenness Top 10:")
    for i, (node_id, score) in enumerate(bc_sorted[:10], 1):
        name = G.nodes[node_id].get('name', node_id)
        kind = G.nodes[node_id].get('kind', '?')
        print(f"  {i:2d}. {score:.6f} [{kind:10s}] {name}")

    for i, (node_id, score) in enumerate(bc_sorted, 1):
        name = G.nodes[node_id].get('name', node_id)
        kind = G.nodes[node_id].get('kind', '?')
        bc_data.append({'rank': i, 'id': node_id, 'name': name, 'kind': kind, 'score': score})

    pd.DataFrame(bc_data).to_csv(output_dir / "betweenness_top30.csv", index=False)
    print("\nSaved: betweenness_top30.csv")

    return betweenness
