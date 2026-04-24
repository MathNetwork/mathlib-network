#!/usr/bin/env python3
"""
Full structural analysis of the namespace dependency graph G_ns at depth 2.

Analyses:
1. Degree distribution (in/out) with power-law fitting
2. PageRank top 20
3. Betweenness centrality top 20 (sampled)
4. Louvain community detection + modularity
5. Community–top-level-directory NMI
6. Robustness: random vs targeted (by PageRank) node removal

Output: terminal + output/namespace_full_analysis.txt
"""

import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from datasets import load_dataset
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

OUTPUT_DIR = Path("output")
DEPTH = 2  # primary analysis depth


def ns_at_depth(name: str, k: int) -> str:
    """Truncate a declaration's fully qualified name to its depth-k namespace."""
    parts = name.split(".")
    if len(parts) <= k:
        # name has fewer than k+1 components → use full parent
        return ".".join(parts[:-1]) if len(parts) > 1 else "_root_"
    return ".".join(parts[:k])


def top_level_dir(name: str) -> str:
    """Extract the top-level component of a declaration name."""
    parts = name.split(".")
    return parts[0] if parts else "_root_"


def main():
    lines = []

    def out(s="", end="\n"):
        print(s, end=end)
        if end == "\n":
            lines.append(s)
        else:
            if lines:
                lines[-1] += s
            else:
                lines.append(s)

    out("=" * 70)
    out(f"  NAMESPACE GRAPH FULL ANALYSIS (G_ns at depth {DEPTH})")
    out("=" * 70)

    # ── Load data ──
    out("\nLoading data ...")
    t0 = time.time()

    nodes_ds = load_dataset(
        "MathNetwork/MathlibGraph",
        data_files="mathlib_nodes.csv", split="train",
    )
    edges_ds = load_dataset(
        "MathNetwork/MathlibGraph",
        data_files="mathlib_edges.csv", split="train",
    )
    nodes_df = nodes_ds.to_pandas()
    edges_df = edges_ds.to_pandas()

    node_names = set(nodes_df["name"].dropna())
    out(f"  Declarations: {len(node_names):,}  "
        f"Declaration edges: {len(edges_df):,}  ({time.time()-t0:.1f}s)")

    # ── Build G_ns at depth k ──
    out(f"\nBuilding G_ns at depth {DEPTH} ...")
    t0 = time.time()

    # Map each declaration to its depth-k namespace
    decl_to_ns = {}
    for name in node_names:
        decl_to_ns[name] = ns_at_depth(name, DEPTH)

    # Also record top-level directory for NMI later
    decl_to_topdir = {}
    for name in node_names:
        decl_to_topdir[name] = top_level_dir(name)

    # Build weighted directed graph
    edge_weights = Counter()
    for _, row in edges_df.iterrows():
        s, t = row["source"], row["target"]
        if s not in decl_to_ns or t not in decl_to_ns:
            continue
        ns_s = decl_to_ns[s]
        ns_t = decl_to_ns[t]
        if ns_s != ns_t:
            edge_weights[(ns_s, ns_t)] += 1

    G = nx.DiGraph()
    all_ns = set(decl_to_ns.values())
    G.add_nodes_from(all_ns)
    for (s, t), w in edge_weights.items():
        G.add_edge(s, t, weight=w)

    out(f"  Namespaces (nodes): {G.number_of_nodes():,}")
    out(f"  Edges (unique ns pairs): {G.number_of_edges():,}")
    out(f"  Total edge weight: {sum(edge_weights.values()):,}")
    out(f"  Time: {time.time()-t0:.1f}s")

    # ── 1. Degree Distribution ──
    out(f"\n{'─'*70}")
    out("  1. DEGREE DISTRIBUTION")
    out(f"{'─'*70}")

    in_degrees = [G.in_degree(n) for n in G.nodes()]
    out_degrees = [G.out_degree(n) for n in G.nodes()]
    w_in_degrees = [G.in_degree(n, weight="weight") for n in G.nodes()]
    w_out_degrees = [G.out_degree(n, weight="weight") for n in G.nodes()]

    out(f"\n  Unweighted (unique edge count):")
    out(f"  {'':>20} {'In-degree':>12} {'Out-degree':>12}")
    out(f"  {'-'*44}")
    out(f"  {'Mean':>20} {np.mean(in_degrees):>12.2f} {np.mean(out_degrees):>12.2f}")
    out(f"  {'Median':>20} {np.median(in_degrees):>12.1f} {np.median(out_degrees):>12.1f}")
    out(f"  {'Std':>20} {np.std(in_degrees):>12.2f} {np.std(out_degrees):>12.2f}")
    out(f"  {'Max':>20} {max(in_degrees):>12,} {max(out_degrees):>12,}")
    out(f"  {'Zero-degree':>20} "
        f"{sum(1 for d in in_degrees if d == 0):>12,} "
        f"{sum(1 for d in out_degrees if d == 0):>12,}")

    out(f"\n  Weighted (sum of declaration-level edges):")
    out(f"  {'':>20} {'In-degree':>12} {'Out-degree':>12}")
    out(f"  {'-'*44}")
    out(f"  {'Mean':>20} {np.mean(w_in_degrees):>12.1f} {np.mean(w_out_degrees):>12.1f}")
    out(f"  {'Median':>20} {np.median(w_in_degrees):>12.1f} {np.median(w_out_degrees):>12.1f}")
    out(f"  {'Max':>20} {max(w_in_degrees):>12,} {max(w_out_degrees):>12,}")

    # Top 10 by in-degree
    in_deg_sorted = sorted(G.nodes(), key=lambda n: G.in_degree(n), reverse=True)
    out(f"\n  Top 10 by unweighted in-degree:")
    out(f"  {'Rank':<6} {'Namespace':<40} {'In-deg':>8} {'W-in-deg':>10}")
    out(f"  {'-'*64}")
    for i, n in enumerate(in_deg_sorted[:10], 1):
        out(f"  {i:<6} {n:<40} {G.in_degree(n):>8,} "
            f"{G.in_degree(n, weight='weight'):>10,}")

    # Top 10 by out-degree
    out_deg_sorted = sorted(G.nodes(), key=lambda n: G.out_degree(n), reverse=True)
    out(f"\n  Top 10 by unweighted out-degree:")
    out(f"  {'Rank':<6} {'Namespace':<40} {'Out-deg':>8} {'W-out-deg':>10}")
    out(f"  {'-'*64}")
    for i, n in enumerate(out_deg_sorted[:10], 1):
        out(f"  {i:<6} {n:<40} {G.out_degree(n):>8,} "
            f"{G.out_degree(n, weight='weight'):>10,}")

    # ── 2. Power-law fitting ──
    out(f"\n{'─'*70}")
    out("  2. POWER-LAW FITTING")
    out(f"{'─'*70}")

    try:
        import powerlaw

        nonzero_in = [d for d in in_degrees if d > 0]
        nonzero_out = [d for d in out_degrees if d > 0]

        for label, data in [("In-degree", nonzero_in), ("Out-degree", nonzero_out)]:
            fit = powerlaw.Fit(data, discrete=True, verbose=False)
            out(f"\n  {label}:")
            out(f"    alpha = {fit.alpha:.3f},  x_min = {fit.xmin}")
            out(f"    sigma = {fit.sigma:.4f}")
            R_ln, p_ln = fit.distribution_compare("power_law", "lognormal")
            R_exp, p_exp = fit.distribution_compare("power_law", "exponential")
            out(f"    vs lognormal:    R = {R_ln:+.2f},  p = {p_ln:.4f}")
            out(f"    vs exponential:  R = {R_exp:+.2f},  p = {p_exp:.4f}")
    except ImportError:
        out("  [powerlaw library not available, skipping]")

    # ── 3. PageRank ──
    out(f"\n{'─'*70}")
    out("  3. PAGERANK TOP 20")
    out(f"{'─'*70}")
    t0 = time.time()

    pr = nx.pagerank(G, alpha=0.85, weight="weight")
    pr_sorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)

    out(f"\n  {'Rank':<6} {'Namespace':<40} {'PageRank':>12}")
    out(f"  {'-'*58}")
    for i, (ns, score) in enumerate(pr_sorted[:20], 1):
        out(f"  {i:<6} {ns:<40} {score:>12.6f}")
    out(f"  Time: {time.time()-t0:.1f}s")

    # ── 4. Betweenness Centrality ──
    out(f"\n{'─'*70}")
    out("  4. BETWEENNESS CENTRALITY TOP 20 (sampled, k=300)")
    out(f"{'─'*70}")
    t0 = time.time()

    bc = nx.betweenness_centrality(G, k=min(300, G.number_of_nodes()),
                                    weight="weight", seed=42)
    bc_sorted = sorted(bc.items(), key=lambda x: x[1], reverse=True)

    out(f"\n  {'Rank':<6} {'Namespace':<40} {'Betweenness':>12}")
    out(f"  {'-'*58}")
    for i, (ns, score) in enumerate(bc_sorted[:20], 1):
        out(f"  {i:<6} {ns:<40} {score:>12.6f}")
    out(f"  Time: {time.time()-t0:.1f}s")

    # ── 5. Louvain Community Detection ──
    out(f"\n{'─'*70}")
    out("  5. LOUVAIN COMMUNITY DETECTION")
    out(f"{'─'*70}")
    t0 = time.time()

    G_undirected = G.to_undirected()
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G_undirected, weight="weight",
                                                      random_state=42)
    except ImportError:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G_undirected, weight="weight", seed=42)
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i

    n_communities = len(set(partition.values()))
    comm_sizes = Counter(partition.values())

    # Modularity
    from networkx.algorithms.community.quality import modularity as nx_modularity
    comm_sets = defaultdict(set)
    for node, cid in partition.items():
        comm_sets[cid].add(node)
    mod = nx_modularity(G_undirected, comm_sets.values(), weight="weight")

    out(f"\n  Communities: {n_communities}")
    out(f"  Modularity: {mod:.4f}")
    out(f"  Communities with >100 nodes: "
        f"{sum(1 for c in comm_sizes.values() if c > 100)}")
    out(f"  Communities with >10 nodes: "
        f"{sum(1 for c in comm_sizes.values() if c > 10)}")

    # Top communities
    out(f"\n  Top 10 communities by size:")
    out(f"  {'ID':<6} {'Size':>8} {'Top namespaces (by node count)':<50}")
    out(f"  {'-'*64}")
    for cid, csize in comm_sizes.most_common(10):
        # Find top namespaces in this community by top-level prefix
        ns_in_comm = [n for n, c in partition.items() if c == cid]
        # Just show a few representative namespace names
        representatives = sorted(ns_in_comm,
                                  key=lambda n: G.in_degree(n, weight="weight"),
                                  reverse=True)[:3]
        rep_str = ", ".join(representatives)
        out(f"  {cid:<6} {csize:>8,} {rep_str:<50}")

    out(f"  Time: {time.time()-t0:.1f}s")

    # ── 6. NMI with top-level directory ──
    out(f"\n{'─'*70}")
    out("  6. COMMUNITY–TOP-LEVEL-DIRECTORY NMI")
    out(f"{'─'*70}")

    # For each namespace, determine its "top-level directory" = first component
    ns_topdir = {}
    for ns in G.nodes():
        parts = ns.split(".")
        ns_topdir[ns] = parts[0] if parts else "_root_"

    # Compute NMI between community labels and top-dir labels
    nodes_list = list(G.nodes())
    comm_labels = [partition.get(n, -1) for n in nodes_list]
    topdir_labels = [ns_topdir.get(n, "_root_") for n in nodes_list]

    nmi = normalized_mutual_info_score(topdir_labels, comm_labels)
    ari = adjusted_rand_score(topdir_labels, comm_labels)

    out(f"\n  NMI(community, top-level dir): {nmi:.4f}")
    out(f"  ARI(community, top-level dir): {ari:.4f}")

    # ── 7. Robustness ──
    out(f"\n{'─'*70}")
    out("  7. ROBUSTNESS: RANDOM vs TARGETED NODE REMOVAL")
    out(f"{'─'*70}")
    t0 = time.time()

    n_nodes = G.number_of_nodes()

    # Get targeted removal order (by PageRank)
    targeted_order = [ns for ns, _ in pr_sorted]

    # Random removal order (fixed seed)
    rng = np.random.RandomState(42)
    random_order = list(G.nodes())
    rng.shuffle(random_order)

    out(f"\n  {'Removal %':>12} {'Random GCC':>14} {'Targeted GCC':>14} {'Gap':>8}")
    out(f"  {'-'*48}")

    for pct in [1, 5, 10, 15, 20, 30, 40, 50]:
        n_remove = int(n_nodes * pct / 100)

        # Random
        G_rand = G.copy()
        G_rand.remove_nodes_from(random_order[:n_remove])
        if G_rand.number_of_nodes() > 0:
            gcc_rand = max(len(c) for c in nx.weakly_connected_components(G_rand))
            gcc_rand_frac = gcc_rand / n_nodes
        else:
            gcc_rand_frac = 0.0

        # Targeted
        G_targ = G.copy()
        G_targ.remove_nodes_from(targeted_order[:n_remove])
        if G_targ.number_of_nodes() > 0:
            gcc_targ = max(len(c) for c in nx.weakly_connected_components(G_targ))
            gcc_targ_frac = gcc_targ / n_nodes
        else:
            gcc_targ_frac = 0.0

        gap = gcc_rand_frac - gcc_targ_frac
        out(f"  {pct:>10}%  {gcc_rand_frac:>13.3f}  {gcc_targ_frac:>13.3f}  "
            f"{gap:>7.3f}")

    out(f"  Time: {time.time()-t0:.1f}s")

    # ── Connectivity summary ──
    out(f"\n{'─'*70}")
    out("  8. CONNECTIVITY SUMMARY")
    out(f"{'─'*70}")

    wccs = list(nx.weakly_connected_components(G))
    gcc_size = max(len(c) for c in wccs)
    out(f"\n  Weakly connected components: {len(wccs):,}")
    out(f"  Largest WCC: {gcc_size:,} / {n_nodes:,} "
        f"({100*gcc_size/n_nodes:.1f}%)")
    out(f"  Isolated nodes (no edges): "
        f"{sum(1 for n in G.nodes() if G.degree(n) == 0):,}")

    # ── Write output ──
    OUTPUT_DIR.mkdir(exist_ok=True)
    outpath = OUTPUT_DIR / "namespace_full_analysis.txt"
    outpath.write_text("\n".join(lines) + "\n")
    out(f"\nOutput written to {outpath}")


if __name__ == "__main__":
    main()
