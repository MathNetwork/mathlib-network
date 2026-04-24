#!/usr/bin/env python3
"""
Cascade and robustness analysis.

Part IV: Dynamics - Network Resilience
Analyzes how the network responds to node removal.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Color scheme
INDIGO = '#2C3E6B'
GOLD = '#C9A96E'
IVORY = '#F5F0E8'


def cascade_analysis(
    G: nx.DiGraph,
    pagerank: dict,
    top_n: int = 30,
    output_dir: Path = None
) -> list[dict]:
    """
    Analyze cascade impact of removing top nodes.

    Measures how much the largest weakly connected component shrinks
    when individual high-PageRank nodes are removed.

    Args:
        G: NetworkX DiGraph
        pagerank: Dictionary of PageRank scores
        top_n: Number of top nodes to analyze
        output_dir: Directory to save CSV files

    Returns:
        List of dicts with cascade results
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data"

    print("\n" + "="*60)
    print("CASCADE ANALYSIS")
    print("="*60)

    # Get top nodes by PageRank
    pr_sorted = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Original largest WCC size
    original_wcc = len(max(nx.weakly_connected_components(G), key=len))

    cascade_results = []
    print("\nAnalyzing cascade impact (removing each node)...")

    for i, (node_id, pr_score) in enumerate(pr_sorted, 1):
        # Create graph without this node
        G_removed = G.copy()
        G_removed.remove_node(node_id)

        # New largest WCC
        if G_removed.number_of_nodes() > 0:
            new_wcc = len(max(nx.weakly_connected_components(G_removed), key=len))
        else:
            new_wcc = 0

        impact = original_wcc - new_wcc
        impact_pct = 100 * impact / original_wcc

        name = G.nodes[node_id].get('name', node_id)
        kind = G.nodes[node_id].get('kind', '?')

        cascade_results.append({
            'rank': i,
            'id': node_id,
            'name': name,
            'kind': kind,
            'pagerank': pr_score,
            'wcc_reduction': impact,
            'wcc_reduction_pct': impact_pct
        })

        if i <= 10:
            print(f"  {i:2d}. {name[:40]:40s} -> WCC reduced by {impact:,} ({impact_pct:.2f}%)")

    # Sort by impact
    cascade_results.sort(key=lambda x: x['wcc_reduction'], reverse=True)

    print("\nTop 10 by cascade impact:")
    for i, r in enumerate(cascade_results[:10], 1):
        print(f"  {i:2d}. {r['name'][:40]:40s} -> {r['wcc_reduction']:,} nodes ({r['wcc_reduction_pct']:.2f}%)")

    pd.DataFrame(cascade_results).to_csv(output_dir / "cascade_top30.csv", index=False)
    print("\nSaved: cascade_top30.csv")

    return cascade_results


def robustness_analysis(
    G: nx.DiGraph,
    pagerank: dict,
    max_removal_frac: float = 0.5,
    step: float = 0.01,
    output_dir: Path = None
) -> dict:
    """
    Analyze network robustness under node removal.

    Compares random removal vs targeted removal (by PageRank).

    Args:
        G: NetworkX DiGraph
        pagerank: Dictionary of PageRank scores
        max_removal_frac: Maximum fraction of nodes to remove
        step: Step size for removal fraction
        output_dir: Directory to save plots

    Returns:
        Dictionary with 'random' and 'targeted' robustness curves
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data"

    print("\n" + "="*60)
    print("ROBUSTNESS ANALYSIS")
    print("="*60)

    n_nodes = G.number_of_nodes()
    removal_fractions = np.arange(0, max_removal_frac + step, step)

    # Sort nodes by PageRank (for targeted attack)
    pr_sorted = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    pr_nodes = [n for n, _ in pr_sorted]

    # Random order
    random_nodes = list(G.nodes())
    np.random.seed(42)
    np.random.shuffle(random_nodes)

    results_random = []
    results_targeted = []

    print("Computing robustness curves...")

    for frac in removal_fractions:
        n_remove = int(frac * n_nodes)

        # Random removal
        G_random = G.copy()
        G_random.remove_nodes_from(random_nodes[:n_remove])
        if G_random.number_of_nodes() > 0:
            wcc_random = len(max(nx.weakly_connected_components(G_random), key=len))
        else:
            wcc_random = 0
        results_random.append(wcc_random / n_nodes)

        # Targeted removal (by PageRank)
        G_targeted = G.copy()
        G_targeted.remove_nodes_from(pr_nodes[:n_remove])
        if G_targeted.number_of_nodes() > 0:
            wcc_targeted = len(max(nx.weakly_connected_components(G_targeted), key=len))
        else:
            wcc_targeted = 0
        results_targeted.append(wcc_targeted / n_nodes)

        if frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
            print(f"  {frac*100:.0f}% removed: Random={results_random[-1]:.3f}, Targeted={results_targeted[-1]:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(IVORY)
    ax.set_facecolor(IVORY)

    ax.plot(removal_fractions * 100, results_random, c=INDIGO, linewidth=2,
            label='Random removal', marker='o', markersize=3)
    ax.plot(removal_fractions * 100, results_targeted, c=GOLD, linewidth=2,
            label='Targeted removal (by PageRank)', marker='s', markersize=3)

    ax.set_xlabel('Fraction of nodes removed (%)')
    ax.set_ylabel('Largest WCC / Total nodes')
    ax.set_title('Network Robustness Analysis', color=INDIGO)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_removal_frac * 100)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "robustness_curve.png", dpi=150, facecolor=IVORY)
    plt.close()
    print("\nSaved: robustness_curve.png")

    return {'random': results_random, 'targeted': results_targeted}
