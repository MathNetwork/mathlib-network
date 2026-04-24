#!/usr/bin/env python3
"""Populate the cache directory with intermediate data for all figures.

This script runs ALL heavy computations (graph building, centrality, community
detection, etc.) and saves results as CSV/JSON in src/plots/cache/.

After running this once, use replot_all.py to regenerate figures in seconds.

Runtime: ~30-60 minutes (dominated by declaration-level graph iteration).
"""

import json
import re
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MATHLIB_ROOT = Path(__file__).resolve().parent.parent.parent / "mathlib4" / "Mathlib"


# =====================================================================
#  Shared helpers
# =====================================================================

def lean_path_to_module(path):
    rel = path.relative_to(MATHLIB_ROOT.parent)
    return str(rel).replace("/", ".").removesuffix(".lean")


def parse_imports(path):
    imports = []
    in_block_comment = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            was_in = in_block_comment > 0
            entered = False
            i = 0
            while i < len(stripped):
                if i + 1 < len(stripped) and stripped[i:i+2] == "/-":
                    in_block_comment += 1; entered = True; i += 2
                elif i + 1 < len(stripped) and stripped[i:i+2] == "-/":
                    in_block_comment = max(0, in_block_comment - 1); i += 2
                else:
                    i += 1
            if in_block_comment > 0 or was_in or entered:
                continue
            if not stripped or stripped.startswith("--"):
                continue
            if stripped.startswith("module"):
                continue
            m = re.match(r"^(?:public\s+)?(?:meta\s+)?import\s+([\w.]+)", stripped)
            if m:
                imports.append(m.group(1))
                continue
            break
    return imports


def ns_at_depth(name, k=2):
    parts = name.split(".")
    if len(parts) <= k:
        return ".".join(parts[:-1]) if len(parts) > 1 else "_root_"
    return ".".join(parts[:k])


def top_level_ns(module):
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def robustness_curve(G, fractions, seed=42):
    """Compute GCC fraction under random and targeted (PageRank) removal."""
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    pr = nx.pagerank(G, alpha=0.85)
    pr_sorted = sorted(pr, key=pr.get, reverse=True)
    rng = np.random.RandomState(seed)
    random_order = list(nodes)
    rng.shuffle(random_order)

    random_gcc = [1.0]
    targeted_gcc = [1.0]
    G_rand = G.copy()
    G_targ = G.copy()
    prev_frac = 0.0

    for frac in fractions:
        n_remove = int(round(frac * n))
        n_prev = int(round(prev_frac * n))
        to_remove = n_remove - n_prev
        if to_remove > 0:
            G_rand.remove_nodes_from(random_order[n_prev:n_remove])
            G_targ.remove_nodes_from(pr_sorted[n_prev:n_remove])
        if G_rand.number_of_nodes() > 0:
            gcc_rand = max(len(c) for c in nx.weakly_connected_components(G_rand)) / n
        else:
            gcc_rand = 0.0
        if G_targ.number_of_nodes() > 0:
            gcc_targ = max(len(c) for c in nx.weakly_connected_components(G_targ)) / n
        else:
            gcc_targ = 0.0
        random_gcc.append(gcc_rand)
        targeted_gcc.append(gcc_targ)
        prev_frac = frac

    return random_gcc, targeted_gcc


def topological_layers_fast(adj, nodes):
    """Compute topological layer widths for a DAG given adjacency list."""
    in_deg = defaultdict(int)
    for u in nodes:
        if u not in in_deg:
            in_deg[u] = 0
        for v in adj.get(u, []):
            in_deg[v] += 1

    layer = {}
    queue = deque()
    for n in nodes:
        if in_deg[n] == 0:
            layer[n] = 0
            queue.append(n)

    max_layer = 0
    while queue:
        u = queue.popleft()
        for v in adj.get(u, []):
            in_deg[v] -= 1
            new_layer = layer[u] + 1
            if v in layer:
                layer[v] = max(layer[v], new_layer)
            else:
                layer[v] = new_layer
            max_layer = max(max_layer, layer[v])
            if in_deg[v] == 0:
                queue.append(v)

    widths = [0] * (max_layer + 1)
    for n, l in layer.items():
        widths[l] += 1
    return widths


FRACTIONS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.15, 0.17, 0.20,
             0.25, 0.30, 0.40, 0.50]


# =====================================================================
#  1. MODULE LEVEL (from .lean files)
# =====================================================================

def build_module_graph():
    print("Building module graph from .lean files ...")
    t0 = time.time()
    G_raw = nx.DiGraph()
    file_modules = set()
    for f in sorted(MATHLIB_ROOT.rglob("*.lean")):
        mod = lean_path_to_module(f)
        file_modules.add(mod)
        G_raw.add_node(mod)
    for f in sorted(MATHLIB_ROOT.rglob("*.lean")):
        mod = lean_path_to_module(f)
        for imp in parse_imports(f):
            if imp.startswith("Mathlib.") and imp in file_modules:
                G_raw.add_edge(mod, imp)
    print(f"  Raw: {G_raw.number_of_nodes()} nodes, {G_raw.number_of_edges()} edges "
          f"({time.time()-t0:.1f}s)")

    print("  Computing transitive reduction ...")
    t0 = time.time()
    G_tr = nx.transitive_reduction(G_raw)
    G_tr.add_nodes_from(G_raw.nodes())
    print(f"  TR: {G_tr.number_of_nodes()} nodes, {G_tr.number_of_edges()} edges "
          f"({time.time()-t0:.1f}s)")
    return G_raw, G_tr


def cache_module_degree(G_raw):
    print("  Caching module_degree_dist.csv ...")
    in_deg = dict(G_raw.in_degree())
    out_deg = dict(G_raw.out_degree())
    df = pd.DataFrame({
        "in_degree": [in_deg[n] for n in G_raw.nodes()],
        "out_degree": [out_deg[n] for n in G_raw.nodes()],
    })
    df.to_csv(CACHE_DIR / "module_degree_dist.csv", index=False)


def cache_module_dag(G_raw, G_tr):
    print("  Caching module_dag_layers.csv ...")
    raw_layers = [len(g) for g in nx.topological_generations(G_raw)]
    tr_layers = [len(g) for g in nx.topological_generations(G_tr)]
    # Pad shorter to match longer
    max_len = max(len(raw_layers), len(tr_layers))
    raw_layers += [0] * (max_len - len(raw_layers))
    tr_layers += [0] * (max_len - len(tr_layers))
    df = pd.DataFrame({"width_raw": raw_layers, "width_tr": tr_layers})
    df.to_csv(CACHE_DIR / "module_dag_layers.csv", index=False)


def cache_module_centrality(G_raw):
    print("  Caching module_centrality.csv ...")
    print("    PageRank ...")
    t0 = time.time()
    pr = nx.pagerank(G_raw)
    print(f"      ({time.time()-t0:.1f}s)")
    print("    Betweenness ...")
    t0 = time.time()
    bc = nx.betweenness_centrality(G_raw)
    print(f"      ({time.time()-t0:.1f}s)")
    in_deg = dict(G_raw.in_degree())
    nodes = list(G_raw.nodes())
    df = pd.DataFrame({
        "in_degree": [in_deg[n] for n in nodes],
        "pagerank": [pr[n] for n in nodes],
        "betweenness": [bc[n] for n in nodes],
    })
    df.to_csv(CACHE_DIR / "module_centrality.csv", index=False)


def cache_module_robustness(G_raw):
    print("  Caching module_robustness.csv ...")
    t0 = time.time()
    rand_gcc, targ_gcc = robustness_curve(G_raw, FRACTIONS)
    print(f"    ({time.time()-t0:.1f}s)")
    x = [0.0] + list(FRACTIONS)
    df = pd.DataFrame({
        "fraction_removed": x,
        "random_gcc": rand_gcc,
        "targeted_gcc": targ_gcc,
    })
    df.to_csv(CACHE_DIR / "module_robustness.csv", index=False)


def cache_namespace_heatmaps(G_raw, G_tr):
    print("  Caching namespace_heatmap_raw.csv and namespace_heatmap_tr.csv ...")
    all_ns = sorted(set(top_level_ns(n) for n in G_raw.nodes()))

    for G, fname in [(G_raw, "namespace_heatmap_raw.csv"),
                     (G_tr, "namespace_heatmap_tr.csv")]:
        ns_matrix = defaultdict(lambda: defaultdict(int))
        for u, v in G.edges():
            ns_u = top_level_ns(u)
            ns_v = top_level_ns(v)
            ns_matrix[ns_u][ns_v] += 1

        matrix = np.zeros((len(all_ns), len(all_ns)), dtype=int)
        for i, ns_u in enumerate(all_ns):
            for j, ns_v in enumerate(all_ns):
                matrix[i, j] = ns_matrix[ns_u][ns_v]

        df = pd.DataFrame(matrix, index=all_ns, columns=all_ns)
        df.to_csv(CACHE_DIR / fname)


def cache_module_community(G_raw):
    print("  Caching community_module_heatmap.json ...")
    import community as community_louvain
    from sklearn.metrics import normalized_mutual_info_score

    t0 = time.time()
    G_und = G_raw.to_undirected()
    partition = community_louvain.best_partition(G_und, random_state=42)
    print(f"    Louvain: {len(set(partition.values()))} communities ({time.time()-t0:.1f}s)")

    nodes_list = list(partition.keys())
    comm_labels = [partition[n] for n in nodes_list]
    dir_labels = [top_level_ns(n) for n in nodes_list]
    nmi = normalized_mutual_info_score(dir_labels, comm_labels)

    # Build contingency matrix (same logic as plot_community_figures.py)
    comm_sizes = Counter(partition.values())
    top_comms = [c for c, _ in comm_sizes.most_common(10)]

    cat_counts = Counter()
    contingency = defaultdict(Counter)
    for n, c in partition.items():
        cat = top_level_ns(n)
        cat_counts[cat] += 1
        contingency[c][cat] += 1

    top_cats = [c for c, _ in cat_counts.most_common(15)]

    matrix = np.zeros((len(top_comms), len(top_cats)), dtype=int)
    for i, comm in enumerate(top_comms):
        for j, cat in enumerate(top_cats):
            matrix[i, j] = contingency[comm].get(cat, 0)

    # Diagonal-maximising reorder
    row_argmax = np.argmax(matrix, axis=1)
    row_order = np.argsort(row_argmax)
    matrix = matrix[row_order]
    top_comms = [top_comms[i] for i in row_order]

    col_seen = []
    col_remaining = list(range(len(top_cats)))
    for i in range(len(top_comms)):
        best_col = int(np.argmax(matrix[i]))
        if best_col in col_remaining:
            col_seen.append(best_col)
            col_remaining.remove(best_col)
    col_order = col_seen + col_remaining
    matrix = matrix[:, col_order]
    top_cats = [top_cats[i] for i in col_order]

    # Row labels
    row_labels = []
    for comm in top_comms:
        sz = comm_sizes[comm]
        dom = contingency[comm].most_common(1)[0][0]
        row_labels.append(f"C{comm} ({dom}, {sz:,})")

    data = {
        "matrix": matrix.tolist(),
        "row_labels": row_labels,
        "col_labels": top_cats,
        "nmi": round(nmi, 4),
    }
    (CACHE_DIR / "community_module_heatmap.json").write_text(
        json.dumps(data, indent=2))


# =====================================================================
#  2. DECLARATION LEVEL (from HuggingFace)
# =====================================================================

def load_hf_data():
    print("\nLoading HuggingFace data ...")
    t0 = time.time()
    from datasets import load_dataset
    nodes_ds = load_dataset("MathNetwork/MathlibGraph",
                            data_files="mathlib_nodes.csv", split="train")
    edges_ds = load_dataset("MathNetwork/MathlibGraph",
                            data_files="mathlib_edges.csv", split="train")
    nodes_df = nodes_ds.to_pandas()
    edges_df = edges_ds.to_pandas()
    node_names = set(nodes_df["name"].dropna())
    print(f"  {len(node_names):,} decls, {len(edges_df):,} edges ({time.time()-t0:.1f}s)")
    return node_names, edges_df


def build_thm_graph(node_names, edges_df):
    print("Building G_thm ...")
    t0 = time.time()
    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    for _, row in edges_df.iterrows():
        s, t = row["source"], row["target"]
        if s in node_names and t in node_names and s != t:
            G.add_edge(s, t)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
          f"({time.time()-t0:.1f}s)")
    return G


def cache_thm_degree(G):
    print("  Caching thm_degree_dist.csv ...")
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    df = pd.DataFrame({
        "in_degree": [in_deg[n] for n in G.nodes()],
        "out_degree": [out_deg[n] for n in G.nodes()],
    })
    df.to_csv(CACHE_DIR / "thm_degree_dist.csv", index=False)


def cache_thm_dag(node_names, edges_df):
    print("  Caching thm_dag_layers.csv ...")
    t0 = time.time()
    thm_adj = defaultdict(list)
    for _, row in edges_df.iterrows():
        s, t = row["source"], row["target"]
        if s in node_names and t in node_names and s != t:
            thm_adj[s].append(t)
    widths = topological_layers_fast(thm_adj, node_names)
    print(f"    {len(widths)} layers, max width {max(widths)} ({time.time()-t0:.1f}s)")
    df = pd.DataFrame({"width": widths})
    df.to_csv(CACHE_DIR / "thm_dag_layers.csv", index=False)


def cache_thm_centrality(G):
    print("  Caching thm_centrality.csv ...")
    print("    PageRank ...")
    t0 = time.time()
    pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    print(f"      ({time.time()-t0:.1f}s)")
    print("    Betweenness (k=500) ...")
    t0 = time.time()
    bc = nx.betweenness_centrality(G, k=500, seed=42)
    print(f"      ({time.time()-t0:.1f}s)")
    in_deg = dict(G.in_degree())
    nodes = list(G.nodes())
    df = pd.DataFrame({
        "in_degree": [in_deg[n] for n in nodes],
        "pagerank": [pr[n] for n in nodes],
        "betweenness": [bc[n] for n in nodes],
    })
    df.to_csv(CACHE_DIR / "thm_centrality.csv", index=False)


def cache_thm_robustness(G):
    print("  Caching thm_robustness.csv ...")
    t0 = time.time()
    rand_gcc, targ_gcc = robustness_curve(G, FRACTIONS)
    print(f"    ({time.time()-t0:.1f}s)")
    x = [0.0] + list(FRACTIONS)
    df = pd.DataFrame({
        "fraction_removed": x,
        "random_gcc": rand_gcc,
        "targeted_gcc": targ_gcc,
    })
    df.to_csv(CACHE_DIR / "thm_robustness.csv", index=False)


def cache_thm_community(G):
    print("  Caching community_decl_heatmap.json ...")
    import community as community_louvain
    from sklearn.metrics import normalized_mutual_info_score

    t0 = time.time()
    G_und = G.to_undirected()
    partition = community_louvain.best_partition(G_und, random_state=42)
    print(f"    Louvain: {len(set(partition.values()))} communities ({time.time()-t0:.1f}s)")

    def ns_depth1(name):
        parts = name.split(".")
        return parts[0] if parts else "_root_"

    nodes_list = list(partition.keys())
    comm_labels = [partition[n] for n in nodes_list]
    ns_labels = [ns_depth1(n) for n in nodes_list]
    nmi = normalized_mutual_info_score(ns_labels, comm_labels)

    comm_sizes = Counter(partition.values())
    top_comms = [c for c, _ in comm_sizes.most_common(10)]

    cat_counts = Counter()
    contingency = defaultdict(Counter)
    for n, c in partition.items():
        cat = ns_depth1(n)
        cat_counts[cat] += 1
        contingency[c][cat] += 1

    top_cats = [c for c, _ in cat_counts.most_common(15)]

    matrix = np.zeros((len(top_comms), len(top_cats)), dtype=int)
    for i, comm in enumerate(top_comms):
        for j, cat in enumerate(top_cats):
            matrix[i, j] = contingency[comm].get(cat, 0)

    row_argmax = np.argmax(matrix, axis=1)
    row_order = np.argsort(row_argmax)
    matrix = matrix[row_order]
    top_comms = [top_comms[i] for i in row_order]

    col_seen = []
    col_remaining = list(range(len(top_cats)))
    for i in range(len(top_comms)):
        best_col = int(np.argmax(matrix[i]))
        if best_col in col_remaining:
            col_seen.append(best_col)
            col_remaining.remove(best_col)
    col_order = col_seen + col_remaining
    matrix = matrix[:, col_order]
    top_cats = [top_cats[i] for i in col_order]

    row_labels = []
    for comm in top_comms:
        sz = comm_sizes[comm]
        dom = contingency[comm].most_common(1)[0][0]
        row_labels.append(f"C{comm} ({dom}, {sz:,})")

    data = {
        "matrix": matrix.tolist(),
        "row_labels": row_labels,
        "col_labels": top_cats,
        "nmi": round(nmi, 4),
    }
    (CACHE_DIR / "community_decl_heatmap.json").write_text(
        json.dumps(data, indent=2))


# =====================================================================
#  3. NAMESPACE LEVEL (aggregated from declarations)
# =====================================================================

def build_ns_graph(node_names, edges_df):
    print("\nBuilding G_ns^(2) ...")
    t0 = time.time()
    decl_to_ns = {name: ns_at_depth(name) for name in node_names}
    all_ns = set(decl_to_ns.values())

    # Weighted edges
    edge_weights = Counter()
    # Unweighted edge set (for degree dist)
    edge_set = set()

    for _, row in edges_df.iterrows():
        s, t = row["source"], row["target"]
        if s not in decl_to_ns or t not in decl_to_ns:
            continue
        ns_s, ns_t = decl_to_ns[s], decl_to_ns[t]
        if ns_s != ns_t:
            edge_weights[(ns_s, ns_t)] += 1
            edge_set.add((ns_s, ns_t))

    G_ns = nx.DiGraph()
    G_ns.add_nodes_from(all_ns)
    for (s, t), w in edge_weights.items():
        G_ns.add_edge(s, t, weight=w)
    print(f"  {G_ns.number_of_nodes():,} nodes, {G_ns.number_of_edges():,} edges "
          f"({time.time()-t0:.1f}s)")
    return G_ns, decl_to_ns, edge_set


def cache_ns_degree(G_ns, edge_set):
    print("  Caching ns_degree_dist.csv ...")
    # Unweighted degree from edge_set
    in_counter = Counter()
    out_counter = Counter()
    for s, t in edge_set:
        out_counter[s] += 1
        in_counter[t] += 1

    all_nodes = set(G_ns.nodes())
    df = pd.DataFrame({
        "in_degree": [in_counter.get(n, 0) for n in all_nodes],
        "out_degree": [out_counter.get(n, 0) for n in all_nodes],
    })
    df.to_csv(CACHE_DIR / "ns_degree_dist.csv", index=False)


def cache_ns_dag(G_ns):
    print("  Caching ns_dag_layers.csv ...")
    is_dag = nx.is_directed_acyclic_graph(G_ns)
    print(f"    Is DAG: {is_dag}")

    if not is_dag:
        condensed = nx.condensation(G_ns)
        scc_sizes = {}
        for node, data in condensed.nodes(data=True):
            scc_sizes[node] = len(data['members'])
        widths = [len(gen) for gen in nx.topological_generations(condensed)]
        print(f"    Condensed: {condensed.number_of_nodes()} super-nodes, {len(widths)} layers")
    else:
        widths = [len(gen) for gen in nx.topological_generations(G_ns)]

    df = pd.DataFrame({"width": widths})
    df.to_csv(CACHE_DIR / "ns_dag_layers.csv", index=False)


def cache_ns_centrality(G_ns):
    print("  Caching ns_centrality.csv ...")
    print("    PageRank ...")
    t0 = time.time()
    pr = nx.pagerank(G_ns, alpha=0.85, weight="weight")
    print(f"      ({time.time()-t0:.1f}s)")
    print("    Betweenness (k=300) ...")
    t0 = time.time()
    bc = nx.betweenness_centrality(G_ns, k=300, weight="weight", seed=42)
    print(f"      ({time.time()-t0:.1f}s)")
    in_deg = dict(G_ns.in_degree())
    nodes = list(G_ns.nodes())
    df = pd.DataFrame({
        "in_degree": [in_deg[n] for n in nodes],
        "pagerank": [pr[n] for n in nodes],
        "betweenness": [bc[n] for n in nodes],
    })
    df.to_csv(CACHE_DIR / "ns_centrality.csv", index=False)


def cache_ns_robustness(G_ns):
    print("  Caching ns_robustness.csv ...")
    t0 = time.time()
    rand_gcc, targ_gcc = robustness_curve(G_ns, FRACTIONS)
    print(f"    ({time.time()-t0:.1f}s)")
    x = [0.0] + list(FRACTIONS)
    df = pd.DataFrame({
        "fraction_removed": x,
        "random_gcc": rand_gcc,
        "targeted_gcc": targ_gcc,
    })
    df.to_csv(CACHE_DIR / "ns_robustness.csv", index=False)


def cache_ns_community(G_ns):
    print("  Caching community_ns_heatmap.json ...")
    import community as community_louvain
    from sklearn.metrics import normalized_mutual_info_score

    t0 = time.time()
    G_und = nx.Graph()
    G_und.add_nodes_from(G_ns.nodes())
    for u, v, d in G_ns.edges(data=True):
        w = d.get("weight", 1)
        if G_und.has_edge(u, v):
            G_und[u][v]["weight"] += w
        else:
            G_und.add_edge(u, v, weight=w)
    partition = community_louvain.best_partition(G_und, random_state=42)
    print(f"    Louvain: {len(set(partition.values()))} communities ({time.time()-t0:.1f}s)")

    def ns_prefix(ns):
        parts = ns.split(".")
        return parts[0] if parts else "_root_"

    nodes_list = list(partition.keys())
    comm_labels = [partition[n] for n in nodes_list]
    pfx_labels = [ns_prefix(n) for n in nodes_list]
    nmi = normalized_mutual_info_score(pfx_labels, comm_labels)

    comm_sizes = Counter(partition.values())
    top_comms = [c for c, _ in comm_sizes.most_common(10)]

    cat_counts = Counter()
    contingency = defaultdict(Counter)
    for n, c in partition.items():
        cat = ns_prefix(n)
        cat_counts[cat] += 1
        contingency[c][cat] += 1

    top_cats = [c for c, _ in cat_counts.most_common(15)]

    matrix = np.zeros((len(top_comms), len(top_cats)), dtype=int)
    for i, comm in enumerate(top_comms):
        for j, cat in enumerate(top_cats):
            matrix[i, j] = contingency[comm].get(cat, 0)

    row_argmax = np.argmax(matrix, axis=1)
    row_order = np.argsort(row_argmax)
    matrix = matrix[row_order]
    top_comms = [top_comms[i] for i in row_order]

    col_seen = []
    col_remaining = list(range(len(top_cats)))
    for i in range(len(top_comms)):
        best_col = int(np.argmax(matrix[i]))
        if best_col in col_remaining:
            col_seen.append(best_col)
            col_remaining.remove(best_col)
    col_order = col_seen + col_remaining
    matrix = matrix[:, col_order]
    top_cats = [top_cats[i] for i in col_order]

    row_labels = []
    for comm in top_comms:
        sz = comm_sizes[comm]
        dom = contingency[comm].most_common(1)[0][0]
        row_labels.append(f"C{comm} ({dom}, {sz:,})")

    data = {
        "matrix": matrix.tolist(),
        "row_labels": row_labels,
        "col_labels": top_cats,
        "nmi": round(nmi, 4),
    }
    (CACHE_DIR / "community_ns_heatmap.json").write_text(
        json.dumps(data, indent=2))


# =====================================================================
#  Main
# =====================================================================

def main():
    t_total = time.time()

    # ---- Module level ----
    print("=" * 60)
    print("MODULE LEVEL")
    print("=" * 60)
    G_raw, G_tr = build_module_graph()
    cache_module_degree(G_raw)
    cache_module_dag(G_raw, G_tr)
    cache_module_centrality(G_raw)
    cache_module_robustness(G_raw)
    cache_namespace_heatmaps(G_raw, G_tr)
    cache_module_community(G_raw)

    # ---- Declaration level ----
    print("\n" + "=" * 60)
    print("DECLARATION LEVEL")
    print("=" * 60)
    node_names, edges_df = load_hf_data()
    G_thm = build_thm_graph(node_names, edges_df)
    cache_thm_degree(G_thm)
    cache_thm_dag(node_names, edges_df)
    cache_thm_centrality(G_thm)
    cache_thm_robustness(G_thm)
    cache_thm_community(G_thm)

    # ---- Namespace level ----
    print("\n" + "=" * 60)
    print("NAMESPACE LEVEL")
    print("=" * 60)
    G_ns, decl_to_ns, edge_set = build_ns_graph(node_names, edges_df)
    cache_ns_degree(G_ns, edge_set)
    cache_ns_dag(G_ns)
    cache_ns_centrality(G_ns)
    cache_ns_robustness(G_ns)
    cache_ns_community(G_ns)

    # ---- Summary ----
    elapsed = time.time() - t_total
    print("\n" + "=" * 60)
    print(f"DONE in {elapsed:.0f}s")
    print("=" * 60)
    expected = [
        "module_degree_dist.csv", "thm_degree_dist.csv", "ns_degree_dist.csv",
        "module_dag_layers.csv", "thm_dag_layers.csv", "ns_dag_layers.csv",
        "module_centrality.csv", "thm_centrality.csv", "ns_centrality.csv",
        "module_robustness.csv", "thm_robustness.csv", "ns_robustness.csv",
        "namespace_heatmap_raw.csv", "namespace_heatmap_tr.csv",
        "community_module_heatmap.json", "community_decl_heatmap.json",
        "community_ns_heatmap.json",
    ]
    for f in expected:
        path = CACHE_DIR / f
        status = "OK" if path.exists() else "MISSING"
        size = path.stat().st_size if path.exists() else 0
        print(f"  [{status}] {f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
