#!/usr/bin/env python3
"""
Basic graph statistics and data loading.

Part II: Structure - Descriptive Analysis
"""

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent.parent / "data"


def load_graph(
    nodes_file: str = "mathlib_math_nodes.csv",
    edges_file: str = "mathlib_math_edges.csv",
    data_dir: Path = None
) -> tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
    """
    Load graph from CSV files.

    Args:
        nodes_file: Name of nodes CSV file
        edges_file: Name of edges CSV file
        data_dir: Data directory (defaults to src/data/)

    Returns:
        (G, nodes_df, edges_df)
    """
    if data_dir is None:
        data_dir = get_data_dir()

    print("Loading graph...")

    nodes_df = pd.read_csv(data_dir / nodes_file)
    edges_df = pd.read_csv(data_dir / edges_file)

    G = nx.DiGraph()

    for _, row in nodes_df.iterrows():
        G.add_node(row['id'], name=row['name'], module=row['module'], kind=row['kind'])

    for _, row in edges_df.iterrows():
        if row['source'] in G.nodes and row['target'] in G.nodes:
            G.add_edge(row['source'], row['target'])

    print(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, nodes_df, edges_df


def basic_statistics(G: nx.DiGraph) -> tuple[dict, set]:
    """
    Compute basic graph statistics.

    Args:
        G: NetworkX DiGraph

    Returns:
        (stats_dict, largest_wcc_nodes)
    """
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)

    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    avg_in = np.mean(in_degrees)
    avg_out = np.mean(out_degrees)

    n_wcc = nx.number_weakly_connected_components(G)
    n_scc = nx.number_strongly_connected_components(G)
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    largest_wcc_ratio = len(largest_wcc) / n_nodes

    stats = {
        'nodes': n_nodes,
        'edges': n_edges,
        'density': density,
        'avg_in_degree': avg_in,
        'avg_out_degree': avg_out,
        'weakly_connected_components': n_wcc,
        'strongly_connected_components': n_scc,
        'largest_wcc_size': len(largest_wcc),
        'largest_wcc_ratio': largest_wcc_ratio,
    }

    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v:,}")

    return stats, largest_wcc
