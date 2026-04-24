#!/usr/bin/env python3
"""
Supplementary analyses for the Mathlib dependency graph.

1. Degree distribution by declaration type (kind)
2. Community vs namespace alignment (NMI, ARI)
3. Cross-module edge breakdown (same-module / same-namespace / cross-namespace)
4. Zero-citation theorems by namespace
"""

import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from datasets import load_dataset


OUTPUT_DIR = Path(__file__).parent.parent / "output"


class Tee:
    """Write to both stdout and a file."""
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()


def load_and_build():
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
        G.add_node(row["name"], kind=row["kind"], module=str(row["module"]))
    node_set = set(G.nodes)
    for _, row in edges_df.iterrows():
        if row["source"] in node_set and row["target"] in node_set:
            G.add_edge(row["source"], row["target"])
    print(f"  Built in {time.time() - t0:.1f}s  ({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)\n")
    return G


def top_ns(module_str):
    """Extract top-level namespace from module string."""
    if not module_str or module_str == "nan" or module_str == "None":
        return "(root)"
    return module_str.split(".")[0]


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ─────────────────────────────────────────────────────────────────────
# 1. Degree distribution by declaration type
# ─────────────────────────────────────────────────────────────────────
def analysis_1_degree_by_kind(G):
    section("1. DEGREE DISTRIBUTION BY DECLARATION TYPE")

    kinds = {}
    for node in G.nodes:
        k = G.nodes[node].get("kind", "unknown")
        if k not in kinds:
            kinds[k] = {"in": [], "out": []}
        kinds[k]["in"].append(G.in_degree(node))
        kinds[k]["out"].append(G.out_degree(node))

    # Summary table
    print(f"\n  {'Kind':15s} {'Count':>8s}  {'In-mean':>8s} {'In-med':>7s} {'In-max':>7s} {'In-std':>8s}  {'Out-mean':>8s} {'Out-med':>7s} {'Out-max':>7s} {'Out-std':>8s}")
    print(f"  {'─' * 105}")

    for k in sorted(kinds.keys(), key=lambda x: -len(kinds[x]["in"])):
        ind = np.array(kinds[k]["in"])
        outd = np.array(kinds[k]["out"])
        n = len(ind)
        print(
            f"  {k:15s} {n:>8,}  "
            f"{ind.mean():>8.1f} {np.median(ind):>7.0f} {ind.max():>7,} {ind.std():>8.1f}  "
            f"{outd.mean():>8.1f} {np.median(outd):>7.0f} {outd.max():>7,} {outd.std():>8.1f}"
        )

    # Top 10 in-degree within each kind
    for k in sorted(kinds.keys(), key=lambda x: -len(kinds[x]["in"])):
        print(f"\n  Top 10 in-degree [{k}]:")
        nodes_of_kind = [(n, G.in_degree(n)) for n in G.nodes if G.nodes[n].get("kind") == k]
        nodes_of_kind.sort(key=lambda x: -x[1])
        for i, (node, deg) in enumerate(nodes_of_kind[:10], 1):
            short = node if len(node) <= 55 else node[:52] + "..."
            print(f"    {i:3d}. {short:55s}  in={deg:,}")


# ─────────────────────────────────────────────────────────────────────
# 2. Community vs namespace alignment (NMI, ARI)
# ─────────────────────────────────────────────────────────────────────
def analysis_2_community_alignment(G):
    section("2. COMMUNITY vs NAMESPACE ALIGNMENT")

    comm_path = OUTPUT_DIR / "communities.csv"
    if not comm_path.exists():
        print("  ERROR: communities.csv not found. Run community detection first.")
        return

    print(f"  Loading {comm_path}...")
    df = pd.read_csv(comm_path)
    print(f"  Rows: {len(df):,}")

    # Extract top-level namespace
    df["top_ns"] = df["module"].apply(lambda m: top_ns(str(m)))

    labels_community = df["community"].values
    labels_namespace = df["top_ns"].values

    print("  Computing NMI and ARI...")
    nmi = normalized_mutual_info_score(labels_namespace, labels_community)
    ari = adjusted_rand_score(labels_namespace, labels_community)

    print(f"\n  Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"  Adjusted Rand Index (ARI):           {ari:.4f}")
    print()
    print(f"  Interpretation:")
    print(f"    NMI ranges [0, 1]: 0 = independent, 1 = perfect alignment")
    print(f"    ARI ranges [-1, 1]: 0 = random, 1 = perfect, <0 = worse than random")

    # Per-community dominant namespace
    print(f"\n  Per-community dominant namespace (top 7 communities):")
    comm_sizes = df.groupby("community").size().sort_values(ascending=False)
    for comm_id in comm_sizes.index[:7]:
        sub = df[df["community"] == comm_id]
        ns_counts = sub["top_ns"].value_counts()
        total = len(sub)
        top3 = ns_counts.head(3)
        purity = ns_counts.iloc[0] / total
        top3_str = ", ".join(f"{ns}({cnt:,})" for ns, cnt in top3.items())
        print(f"    Community {comm_id:2d} ({total:>6,} nodes)  purity={purity:.2%}  top: {top3_str}")

    # Per-namespace dominant community
    print(f"\n  Per-namespace dominant community (top 15 namespaces):")
    ns_sizes = df.groupby("top_ns").size().sort_values(ascending=False)
    for ns in ns_sizes.index[:15]:
        sub = df[df["top_ns"] == ns]
        comm_counts = sub["community"].value_counts()
        total = len(sub)
        top_comm = comm_counts.index[0]
        purity = comm_counts.iloc[0] / total
        print(f"    {ns:35s} ({total:>6,} nodes)  {purity:.1%} in community {top_comm}")


# ─────────────────────────────────────────────────────────────────────
# 3. Cross-module edge breakdown
# ─────────────────────────────────────────────────────────────────────
def analysis_3_edge_breakdown(G):
    section("3. CROSS-MODULE EDGE BREAKDOWN")

    same_module = 0
    cross_module_same_ns = 0
    cross_namespace = 0
    missing = 0

    for u, v in G.edges():
        mod_u = G.nodes[u].get("module", "")
        mod_v = G.nodes[v].get("module", "")
        if not mod_u or mod_u == "nan" or not mod_v or mod_v == "nan":
            missing += 1
            continue

        if mod_u == mod_v:
            same_module += 1
        else:
            ns_u = mod_u.split(".")[0]
            ns_v = mod_v.split(".")[0]
            if ns_u == ns_v:
                cross_module_same_ns += 1
            else:
                cross_namespace += 1

    counted = same_module + cross_module_same_ns + cross_namespace
    total = counted + missing

    print(f"\n  Edge classification ({total:,} total edges):")
    print(f"    Same module:                     {same_module:>10,}  ({100*same_module/total:5.1f}%)")
    print(f"    Cross-module, same namespace:    {cross_module_same_ns:>10,}  ({100*cross_module_same_ns/total:5.1f}%)")
    print(f"    Cross-namespace:                 {cross_namespace:>10,}  ({100*cross_namespace/total:5.1f}%)")
    print(f"    Missing module info:             {missing:>10,}  ({100*missing/total:5.1f}%)")

    # Top cross-namespace pairs
    print(f"\n  Top 20 cross-namespace edge pairs:")
    ns_pairs = Counter()
    for u, v in G.edges():
        mod_u = G.nodes[u].get("module", "")
        mod_v = G.nodes[v].get("module", "")
        if not mod_u or mod_u == "nan" or not mod_v or mod_v == "nan":
            continue
        ns_u = mod_u.split(".")[0]
        ns_v = mod_v.split(".")[0]
        if ns_u != ns_v:
            pair = tuple(sorted([ns_u, ns_v]))
            ns_pairs[pair] += 1

    print(f"    {'Namespace A':25s} {'Namespace B':25s} {'Edges':>10s}")
    print(f"    {'─' * 65}")
    for (a, b), cnt in ns_pairs.most_common(20):
        print(f"    {a:25s} {b:25s} {cnt:>10,}")


# ─────────────────────────────────────────────────────────────────────
# 4. Zero-citation theorems by namespace
# ─────────────────────────────────────────────────────────────────────
def analysis_4_zero_citation(G):
    section("4. ZERO-CITATION THEOREMS BY NAMESPACE")

    # Get theorems with in-degree 0
    all_theorems = []
    zero_theorems = []
    for node in G.nodes:
        if G.nodes[node].get("kind") != "theorem":
            continue
        mod = G.nodes[node].get("module", "")
        ns = top_ns(mod)
        all_theorems.append(ns)
        if G.in_degree(node) == 0:
            zero_theorems.append(ns)

    total_thm = len(all_theorems)
    total_zero = len(zero_theorems)
    print(f"\n  Total theorems: {total_thm:,}")
    print(f"  Zero-citation theorems: {total_zero:,} ({100*total_zero/total_thm:.1f}%)")

    # By namespace
    ns_total = Counter(all_theorems)
    ns_zero = Counter(zero_theorems)

    rows = []
    for ns in ns_total:
        t = ns_total[ns]
        z = ns_zero.get(ns, 0)
        rows.append((ns, t, z, z / t if t > 0 else 0))

    # Sort by total theorems descending
    rows.sort(key=lambda x: -x[1])

    print(f"\n  Top 30 namespaces by theorem count:")
    print(f"    {'Namespace':30s} {'Total':>8s} {'Zero-cite':>10s} {'Zero-rate':>10s}")
    print(f"    {'─' * 62}")
    for ns, t, z, rate in rows[:30]:
        print(f"    {ns:30s} {t:>8,} {z:>10,} {rate:>9.1%}")

    # Sort by zero-rate (for namespaces with >= 100 theorems)
    print(f"\n  Highest zero-citation rate (namespaces with >=100 theorems):")
    rows_filtered = [(ns, t, z, rate) for ns, t, z, rate in rows if t >= 100]
    rows_filtered.sort(key=lambda x: -x[3])
    print(f"    {'Namespace':30s} {'Total':>8s} {'Zero-cite':>10s} {'Zero-rate':>10s}")
    print(f"    {'─' * 62}")
    for ns, t, z, rate in rows_filtered[:20]:
        print(f"    {ns:30s} {t:>8,} {z:>10,} {rate:>9.1%}")

    # Lowest zero-rate
    print(f"\n  Lowest zero-citation rate (namespaces with >=100 theorems):")
    rows_filtered.sort(key=lambda x: x[3])
    print(f"    {'Namespace':30s} {'Total':>8s} {'Zero-cite':>10s} {'Zero-rate':>10s}")
    print(f"    {'─' * 62}")
    for ns, t, z, rate in rows_filtered[:20]:
        print(f"    {ns:30s} {t:>8,} {z:>10,} {rate:>9.1%}")


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    tee = Tee(OUTPUT_DIR / "supplementary_stats.txt")
    sys.stdout = tee

    try:
        G = load_and_build()
        analysis_1_degree_by_kind(G)
        analysis_2_community_alignment(G)
        analysis_3_edge_breakdown(G)
        analysis_4_zero_citation(G)

        print(f"\n{'=' * 70}")
        print(f"  ALL DONE in {time.time() - start:.1f}s")
        print(f"{'=' * 70}")
    finally:
        sys.stdout = tee.stdout
        tee.close()

    print(f"\nOutput also saved to: {OUTPUT_DIR / 'supplementary_stats.txt'}")


if __name__ == "__main__":
    main()
