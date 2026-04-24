#!/usr/bin/env python3
"""
Community detection using Louvain algorithm.

Part II: Structure - Community Structure
"""

from pathlib import Path
from collections import Counter

import pandas as pd
import networkx as nx


def community_detection(
    G: nx.DiGraph,
    largest_wcc: set,
    pagerank: dict,
    output_dir: Path = None
) -> tuple[dict, list]:
    """
    Detect communities using Louvain algorithm.

    Args:
        G: NetworkX DiGraph
        largest_wcc: Set of nodes in largest weakly connected component
        pagerank: Dictionary of PageRank scores
        output_dir: Directory to save CSV files

    Returns:
        (partition_dict, community_details_list)
    """
    import community as community_louvain

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data"

    print("\n" + "="*60)
    print("COMMUNITY DETECTION (LOUVAIN)")
    print("="*60)

    # Work on largest WCC, convert to undirected
    G_wcc = G.subgraph(largest_wcc).copy()
    G_undirected = G_wcc.to_undirected()

    print(f"Running Louvain on largest WCC ({len(largest_wcc):,} nodes)...")
    partition = community_louvain.best_partition(G_undirected)

    n_communities = len(set(partition.values()))
    print(f"Found {n_communities} communities")

    # Community sizes
    comm_sizes = Counter(partition.values())
    top_communities = comm_sizes.most_common(10)

    print("\nTop 10 communities by size:")
    community_details = []

    for comm_id, size in top_communities:
        # Get nodes in this community
        comm_nodes = [n for n, c in partition.items() if c == comm_id]
        # Sort by PageRank
        comm_nodes_pr = [(n, pagerank.get(n, 0)) for n in comm_nodes]
        comm_nodes_pr.sort(key=lambda x: x[1], reverse=True)
        top5 = comm_nodes_pr[:5]

        print(f"\n  Community {comm_id} (size: {size:,})")
        print(f"    Representatives (by PageRank):")
        for node_id, pr in top5:
            name = G.nodes[node_id].get('name', node_id)
            kind = G.nodes[node_id].get('kind', '?')
            print(f"      [{kind:10s}] {name}")

        community_details.append({
            'community_id': comm_id,
            'size': size,
            'top_nodes': [G.nodes[n].get('name', n) for n, _ in top5]
        })

    # Save all community assignments
    comm_data = [{'id': n, 'name': G.nodes[n].get('name', n),
                  'kind': G.nodes[n].get('kind', '?'), 'community': c}
                 for n, c in partition.items()]
    pd.DataFrame(comm_data).to_csv(output_dir / "communities.csv", index=False)
    print("\nSaved: communities.csv")

    return partition, community_details
