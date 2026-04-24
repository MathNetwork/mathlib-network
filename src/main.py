#!/usr/bin/env python3
"""
Main entry point for MathlibGraph analysis.

Runs the full analysis pipeline:
1. Load graph from HuggingFace
2. Basic statistics
3. Degree distribution analysis
4. PageRank and HITS
5. Community detection
6. Cascade analysis
7. Robustness analysis
8. Betweenness centrality
9. Generate report
"""

import time
import signal
from pathlib import Path
from contextlib import contextmanager

from datasets import load_dataset

from analysis.structure import (
    basic_statistics,
    degree_distribution_analysis,
    pagerank_hits_analysis,
    betweenness_analysis,
    community_detection,
)
from analysis.dynamics import cascade_analysis, robustness_analysis


def load_data_from_huggingface():
    """
    Load MathlibGraph dataset from HuggingFace.

    Returns:
        tuple: (nodes_df, edges_df) as pandas DataFrames
    """
    nodes_ds = load_dataset("MathNetwork/MathlibGraph", data_files="mathlib_nodes.csv", split="train")
    edges_ds = load_dataset("MathNetwork/MathlibGraph", data_files="mathlib_edges.csv", split="train")
    nodes_df = nodes_ds.to_pandas()
    edges_df = edges_ds.to_pandas()
    return nodes_df, edges_df


TIMEOUT_SECONDS = 600  # 10 minutes per analysis


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds):
    """Context manager for timeout."""
    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def generate_report(
    stats: dict,
    degree_results: dict,
    community_details: list,
    cascade_results: list,
    robustness: dict,
    output_dir: Path
):
    """Generate comprehensive markdown report."""
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)

    md = []
    md.append("# Mathlib Dependency Network: Full Analysis Report")
    md.append("")
    md.append("Comprehensive analysis of theorem-level dependencies in Mathlib4.")
    md.append("")
    md.append("---")
    md.append("")

    # 1. Basic Statistics
    md.append("## 1. Basic Statistics")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Nodes | {stats['nodes']:,} |")
    md.append(f"| Edges | {stats['edges']:,} |")
    md.append(f"| Density | {stats['density']:.6f} |")
    md.append(f"| Avg In-degree | {stats['avg_in_degree']:.2f} |")
    md.append(f"| Avg Out-degree | {stats['avg_out_degree']:.2f} |")
    md.append(f"| Weakly Connected Components | {stats['weakly_connected_components']:,} |")
    md.append(f"| Strongly Connected Components | {stats['strongly_connected_components']:,} |")
    md.append(f"| Largest WCC Size | {stats['largest_wcc_size']:,} ({stats['largest_wcc_ratio']*100:.1f}%) |")
    md.append("")

    # 2. Degree Distribution
    if degree_results.get('in_degree'):
        md.append("## 2. Degree Distribution & Scale-Free Analysis")
        md.append("")
        md.append("![Degree Distribution](degree_distribution.png)")
        md.append("")
        md.append("### Power Law Fit")
        md.append("")
        md.append("| Metric | In-degree | Out-degree |")
        md.append("|--------|-----------|------------|")
        md.append(f"| Alpha (exponent) | {degree_results['in_degree']['alpha']:.3f} | {degree_results['out_degree']['alpha']:.3f} |")
        md.append(f"| xmin | {degree_results['in_degree']['xmin']} | {degree_results['out_degree']['xmin']} |")
        md.append(f"| vs Lognormal (R) | {degree_results['in_degree']['vs_lognormal_R']:.3f} | {degree_results['out_degree']['vs_lognormal_R']:.3f} |")
        md.append(f"| vs Exponential (R) | {degree_results['in_degree']['vs_exponential_R']:.3f} | {degree_results['out_degree']['vs_exponential_R']:.3f} |")
        md.append("")
        md.append("> R > 0 means power law is better fit; R < 0 means alternative is better.")
        md.append("")

    # 3. PageRank & HITS
    md.append("## 3. PageRank & HITS")
    md.append("")
    md.append("See `pagerank_top50.csv` and `hits_top50.csv` for full rankings.")
    md.append("")

    # 4. Communities
    if community_details:
        md.append("## 4. Community Structure")
        md.append("")
        md.append("Using Louvain algorithm on the largest weakly connected component.")
        md.append("")
        md.append("### Top 10 Communities")
        md.append("")
        for comm in community_details[:10]:
            md.append(f"**Community {comm['community_id']}** ({comm['size']:,} nodes)")
            md.append("")
            md.append("Representatives: " + ", ".join(f"`{n}`" for n in comm['top_nodes'][:3]))
            md.append("")

    # 5. Cascade Analysis
    if cascade_results:
        md.append("## 5. Cascade Analysis")
        md.append("")
        md.append("Impact of removing individual high-PageRank nodes on network connectivity.")
        md.append("")
        md.append("| Rank | Theorem | WCC Reduction | % |")
        md.append("|------|---------|---------------|---|")
        for r in sorted(cascade_results, key=lambda x: x['wcc_reduction'], reverse=True)[:10]:
            md.append(f"| {r['rank']} | `{r['name'][:30]}` | {r['wcc_reduction']:,} | {r['wcc_reduction_pct']:.2f}% |")
        md.append("")

    # 6. Robustness
    if robustness.get('random'):
        md.append("## 6. Network Robustness")
        md.append("")
        md.append("![Robustness Curve](robustness_curve.png)")
        md.append("")
        md.append("| Removal % | Random | Targeted (PageRank) |")
        md.append("|-----------|--------|---------------------|")
        for frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
            idx = int(frac * 100)
            if idx < len(robustness['random']):
                md.append(f"| {frac*100:.0f}% | {robustness['random'][idx]:.3f} | {robustness['targeted'][idx]:.3f} |")
        md.append("")

    # 7. Betweenness
    md.append("## 7. Bridge Nodes (Betweenness Centrality)")
    md.append("")
    md.append("See `betweenness_top30.csv` for full rankings.")
    md.append("")

    # Write report
    with open(output_dir / "full_analysis_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("Saved: full_analysis_report.md")


def load_graph_from_dataframes(nodes_df, edges_df):
    """
    Build NetworkX DiGraph from pandas DataFrames.

    Args:
        nodes_df: DataFrame with columns (name, kind, module)
        edges_df: DataFrame with columns (source, target)

    Returns:
        nx.DiGraph
    """
    import networkx as nx

    G = nx.DiGraph()

    for _, row in nodes_df.iterrows():
        G.add_node(row['name'], kind=row['kind'], module=row['module'])

    node_set = set(G.nodes)
    for _, row in edges_df.iterrows():
        if row['source'] in node_set and row['target'] in node_set:
            G.add_edge(row['source'], row['target'])

    print(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def main():
    start_time = time.time()

    # Output directory for analysis results
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Load data from HuggingFace
    print("Loading data from HuggingFace...")
    nodes_df, edges_df = load_data_from_huggingface()

    # Build graph
    G = load_graph_from_dataframes(nodes_df, edges_df)

    # 1. Basic statistics
    stats, largest_wcc = basic_statistics(G)

    # 2. Degree distribution
    try:
        with timeout(TIMEOUT_SECONDS):
            degree_results = degree_distribution_analysis(G, output_dir=output_dir)
    except TimeoutError as e:
        print(f"  SKIPPED: {e}")
        degree_results = {'in_degree': {}, 'out_degree': {}}

    # 3. PageRank & HITS
    try:
        with timeout(TIMEOUT_SECONDS):
            pagerank, hubs, authorities = pagerank_hits_analysis(G, output_dir=output_dir)
    except TimeoutError as e:
        print(f"  SKIPPED: {e}")
        import networkx as nx
        pagerank = nx.pagerank(G)
        hubs, authorities = {}, {}

    # 4. Community detection
    try:
        with timeout(TIMEOUT_SECONDS):
            partition, community_details = community_detection(G, largest_wcc, pagerank, output_dir=output_dir)
    except TimeoutError as e:
        print(f"  SKIPPED: {e}")
        partition, community_details = {}, []

    # 5. Cascade analysis
    try:
        with timeout(TIMEOUT_SECONDS):
            cascade_results = cascade_analysis(G, pagerank, output_dir=output_dir)
    except TimeoutError as e:
        print(f"  SKIPPED: {e}")
        cascade_results = []

    # 6. Robustness analysis
    try:
        with timeout(TIMEOUT_SECONDS):
            robustness = robustness_analysis(G, pagerank, output_dir=output_dir)
    except TimeoutError as e:
        print(f"  SKIPPED: {e}")
        robustness = {'random': [], 'targeted': []}

    # 7. Betweenness centrality
    try:
        with timeout(TIMEOUT_SECONDS):
            betweenness = betweenness_analysis(G, output_dir=output_dir)
    except TimeoutError as e:
        print(f"  SKIPPED: {e}")
        betweenness = {}

    # Generate report
    generate_report(stats, degree_results, community_details, cascade_results, robustness, output_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE in {elapsed:.1f} seconds")
    print(f"{'='*60}")
    print(f"\nOutput files in {output_dir}/:")
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024*1024:
                print(f"  {f.name}: {size/1024/1024:.1f} MB")
            elif size > 1024:
                print(f"  {f.name}: {size/1024:.1f} KB")
            else:
                print(f"  {f.name}: {size} B")


if __name__ == "__main__":
    main()
