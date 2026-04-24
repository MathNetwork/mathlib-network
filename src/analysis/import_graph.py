#!/usr/bin/env python3
"""Comprehensive analysis of the Mathlib import graph."""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from plot_style import setup_style, COLORS, FIGSIZE_SINGLE, FIGSIZE_DOUBLE, FIGSIZE_TRIPLE, FIGSIZE_HEATMAP, FIGSIZE_HEATMAP_WIDE

COLORS = setup_style()

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np

MATHLIB_ROOT = Path(__file__).resolve().parent.parent.parent / "mathlib4" / "Mathlib"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"

# ===================================================================
# Build the import graph
# ===================================================================

def lean_path_to_module(path: Path) -> str:
    rel = path.relative_to(MATHLIB_ROOT.parent)
    return str(rel).replace("/", ".").removesuffix(".lean")


def parse_imports(path: Path) -> list[str]:
    imports = []
    in_block_comment = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            was_in_comment = in_block_comment > 0
            entered_comment = False
            i = 0
            while i < len(stripped):
                if i + 1 < len(stripped) and stripped[i:i+2] == "/-":
                    in_block_comment += 1
                    entered_comment = True
                    i += 2
                elif i + 1 < len(stripped) and stripped[i:i+2] == "-/":
                    in_block_comment = max(0, in_block_comment - 1)
                    i += 2
                else:
                    i += 1
            if in_block_comment > 0 or was_in_comment or entered_comment:
                continue
            if not stripped or stripped.startswith("--"):
                continue
            if stripped.startswith("module"):
                continue
            m = re.match(
                r"^(?:public\s+)?(?:meta\s+)?import\s+([\w.]+)", stripped
            )
            if m:
                imports.append(m.group(1))
                continue
            break
    return imports


print("Building import graph from .lean files...")
G_raw = nx.DiGraph()
file_modules = set()

for lean_file in sorted(MATHLIB_ROOT.rglob("*.lean")):
    mod = lean_path_to_module(lean_file)
    file_modules.add(mod)
    G_raw.add_node(mod)

for lean_file in sorted(MATHLIB_ROOT.rglob("*.lean")):
    mod = lean_path_to_module(lean_file)
    for imp in parse_imports(lean_file):
        if imp.startswith("Mathlib.") and imp in file_modules:
            G_raw.add_edge(mod, imp)

print(f"  Raw graph: {G_raw.number_of_nodes()} nodes, {G_raw.number_of_edges()} edges")

print("  Computing transitive reduction...")
G_tr = nx.transitive_reduction(G_raw)
G_tr.add_nodes_from(G_raw.nodes())
print(f"  TR graph:  {G_tr.number_of_nodes()} nodes, {G_tr.number_of_edges()} edges")

results = {}

# ===================================================================
# 1. Degree distribution analysis
# ===================================================================
print("\n=== 1. Degree distribution ===")

def degree_stats(G, label):
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    in_vals = np.array(list(in_deg.values()))
    out_vals = np.array(list(out_deg.values()))

    top_in = sorted(in_deg.items(), key=lambda x: -x[1])[:20]
    top_out = sorted(out_deg.items(), key=lambda x: -x[1])[:20]

    stats = {
        "in_degree": {
            "mean": round(float(np.mean(in_vals)), 2),
            "median": round(float(np.median(in_vals)), 2),
            "max": int(np.max(in_vals)),
            "std": round(float(np.std(in_vals)), 2),
            "top_20": [{"module": m, "value": int(d)} for m, d in top_in],
        },
        "out_degree": {
            "mean": round(float(np.mean(out_vals)), 2),
            "median": round(float(np.median(out_vals)), 2),
            "max": int(np.max(out_vals)),
            "std": round(float(np.std(out_vals)), 2),
            "top_20": [{"module": m, "value": int(d)} for m, d in top_out],
        },
    }

    print(f"  [{label}] In-degree:  mean={stats['in_degree']['mean']}, "
          f"median={stats['in_degree']['median']}, max={stats['in_degree']['max']}, "
          f"std={stats['in_degree']['std']}")
    print(f"  [{label}] Out-degree: mean={stats['out_degree']['mean']}, "
          f"median={stats['out_degree']['median']}, max={stats['out_degree']['max']}, "
          f"std={stats['out_degree']['std']}")

    return stats, in_vals, out_vals


raw_deg_stats, raw_in, raw_out = degree_stats(G_raw, "raw")
tr_deg_stats, tr_in, tr_out = degree_stats(G_tr, "TR")

results["degree_distribution"] = {"raw": raw_deg_stats, "transitive_reduction": tr_deg_stats}

# Degree distribution plots (log-log)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for col, (in_vals, out_vals, label) in enumerate([
    (raw_in, raw_out, r"$G_{\mathrm{module}}$"),
    (tr_in, tr_out, r"$G_{\mathrm{module}}^{-}$"),
]):
    # In-degree (log-log) — blue for module level
    ax = axes[0][col]
    in_counter = Counter(in_vals)
    degs = sorted(in_counter.keys())
    degs = [d for d in degs if d > 0]
    counts = [in_counter[d] for d in degs]
    ax.scatter(degs, counts, s=12, color=COLORS["primary"], alpha=0.7)
    # Power-law reference line: fit gamma via MLE on in-degree >= k_min
    in_arr = np.array([v for v in in_vals if v >= 2])
    if len(in_arr) > 0:
        gamma = 1 + len(in_arr) / np.sum(np.log(in_arr / 1.5))
        k_ref = np.logspace(np.log10(2), np.log10(max(degs)), 50)
        # scale so the line passes through the empirical count at k=2
        c0 = in_counter.get(2, in_counter.get(3, 1))
        ref_line = c0 * (k_ref / 2) ** (-gamma)
        ax.plot(k_ref, ref_line, color=COLORS["quaternary"], linestyle="--",
                linewidth=1, alpha=0.6,
                label=rf"$k^{{-{gamma:.2f}}}$")
        ax.legend(loc="upper right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("In-degree")
    ax.set_ylabel("Count")
    ax.set_title(f"In-degree distribution ({label})")
    ax.grid(True, alpha=0.3, which="both")

    # Out-degree (log-log) — blue for module level (darker shade)
    ax = axes[1][col]
    out_counter = Counter(out_vals)
    degs = sorted(out_counter.keys())
    degs = [d for d in degs if d > 0]
    counts = [out_counter[d] for d in degs]
    ax.scatter(degs, counts, s=12, color=COLORS["primary"], alpha=0.5)
    # Power-law reference line for out-degree
    out_arr = np.array([v for v in out_vals if v >= 2])
    if len(out_arr) > 0:
        gamma_out = 1 + len(out_arr) / np.sum(np.log(out_arr / 1.5))
        k_ref = np.logspace(np.log10(2), np.log10(max(degs)), 50)
        c0 = out_counter.get(2, out_counter.get(3, 1))
        ref_line = c0 * (k_ref / 2) ** (-gamma_out)
        ax.plot(k_ref, ref_line, color=COLORS["quaternary"], linestyle="--",
                linewidth=1, alpha=0.6,
                label=rf"$k^{{-{gamma_out:.2f}}}$")
        ax.legend(loc="upper right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Out-degree")
    ax.set_ylabel("Count")
    ax.set_title(f"Out-degree distribution ({label})")
    ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "degree_distribution.pdf", bbox_inches="tight")
plt.close()
print("  Saved degree_distribution.pdf")

# ===================================================================
# 2. DAG path structure
# ===================================================================
print("\n=== 2. DAG path structure ===")

def dag_analysis(G, label):
    is_dag = nx.is_directed_acyclic_graph(G)
    print(f"  [{label}] Is DAG: {is_dag}")

    longest_path = nx.dag_longest_path(G)
    longest_len = len(longest_path) - 1

    layers = list(nx.topological_generations(G))
    layer_sizes = [len(layer) for layer in layers]

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    sources = [n for n, d in in_deg.items() if d == 0]
    sinks = [n for n, d in out_deg.items() if d == 0]

    stats = {
        "is_dag": is_dag,
        "longest_path_length": longest_len,
        "longest_path_start": longest_path[0] if longest_path else None,
        "longest_path_end": longest_path[-1] if longest_path else None,
        "longest_path": longest_path,
        "num_layers": len(layers),
        "layer_width_max": max(layer_sizes),
        "layer_width_mean": round(float(np.mean(layer_sizes)), 2),
        "layer_width_median": round(float(np.median(layer_sizes)), 2),
        "num_sources": len(sources),
        "num_sinks": len(sinks),
        "sources_sample": sorted(sources)[:10],
        "sinks_sample": sorted(sinks)[:10],
    }

    print(f"  [{label}] Longest path: {longest_len} edges")
    print(f"  [{label}] Layers: {len(layers)}, max width: {max(layer_sizes)}")
    print(f"  [{label}] Sources (in=0): {len(sources)}, Sinks (out=0): {len(sinks)}")

    return stats, layer_sizes


raw_dag_stats, raw_layers = dag_analysis(G_raw, "raw")
tr_dag_stats, tr_layers = dag_analysis(G_tr, "TR")

results["dag_structure"] = {"raw": raw_dag_stats, "transitive_reduction": tr_dag_stats}

# DAG layer width plots — blue for module level
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.bar(range(len(raw_layers)), raw_layers, color=COLORS["primary"], edgecolor="none", width=1.0)
ax1.set_xlabel("Topological layer")
ax1.set_ylabel("Number of modules")
ax1.set_title(rf"DAG width by topological layer ($G_{{\mathrm{{module}}}}$, {len(raw_layers)} layers)")
ax1.set_xlim(-1, len(raw_layers))

ax2.bar(range(len(tr_layers)), tr_layers, color=COLORS["primary"], edgecolor="none", width=1.0, alpha=0.7)
ax2.set_xlabel("Topological layer")
ax2.set_ylabel("Number of modules")
ax2.set_title(rf"DAG width by topological layer ($G_{{\mathrm{{module}}}}^{{-}}$, {len(tr_layers)} layers)")
ax2.set_xlim(-1, len(tr_layers))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dag_structure.pdf", bbox_inches="tight")
plt.close()
print("  Saved dag_structure.pdf")

# ===================================================================
# 3. Namespace cross-analysis
# ===================================================================
print("\n=== 3. Namespace cross-analysis ===")

def top_level_ns(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def namespace_analysis(G, label):
    intra = 0
    cross = 0
    ns_matrix = defaultdict(lambda: defaultdict(int))

    for u, v in G.edges():
        ns_u = top_level_ns(u)
        ns_v = top_level_ns(v)
        ns_matrix[ns_u][ns_v] += 1
        if ns_u == ns_v:
            intra += 1
        else:
            cross += 1

    total = intra + cross
    stats = {
        "intra_namespace_edges": intra,
        "cross_namespace_edges": cross,
        "intra_ratio": round(intra / total, 4) if total > 0 else 0,
        "cross_ratio": round(cross / total, 4) if total > 0 else 0,
    }

    # Top cross-namespace pairs
    cross_pairs = []
    for ns_u, targets in ns_matrix.items():
        for ns_v, count in targets.items():
            if ns_u != ns_v:
                cross_pairs.append({"from": ns_u, "to": ns_v, "count": count})
    cross_pairs.sort(key=lambda x: -x["count"])
    stats["top_cross_namespace_pairs"] = cross_pairs[:20]

    print(f"  [{label}] Intra: {intra} ({stats['intra_ratio']:.1%}), "
          f"Cross: {cross} ({stats['cross_ratio']:.1%})")

    return stats, ns_matrix


raw_ns_stats, raw_ns_matrix = namespace_analysis(G_raw, "raw")
tr_ns_stats, tr_ns_matrix = namespace_analysis(G_tr, "TR")

results["namespace_analysis"] = {"raw": raw_ns_stats, "transitive_reduction": tr_ns_stats}

# Heatmaps
all_ns = sorted(set(top_level_ns(n) for n in G_raw.nodes()))
ns_to_idx = {ns: i for i, ns in enumerate(all_ns)}

for ns_matrix, label, fname in [
    (raw_ns_matrix, "Raw", "namespace_heatmap_raw.pdf"),
    (tr_ns_matrix, "Transitive Reduction", "namespace_heatmap_tr.pdf"),
]:
    fig, ax = plt.subplots(figsize=(11, 9))
    matrix = np.zeros((len(all_ns), len(all_ns)), dtype=int)
    for ns_u, targets in ns_matrix.items():
        for ns_v, count in targets.items():
            if ns_u in ns_to_idx and ns_v in ns_to_idx:
                matrix[ns_to_idx[ns_u], ns_to_idx[ns_v]] = count

    im = ax.imshow(np.log1p(matrix), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(all_ns)))
    ax.set_yticks(range(len(all_ns)))
    ax.set_xticklabels(all_ns, rotation=90)
    ax.set_yticklabels(all_ns)
    ax.set_xlabel("Imported namespace")
    ax.set_ylabel("Importing namespace")
    ax.set_title(f"Namespace import counts ({label}, log scale)")
    plt.colorbar(im, ax=ax, label="log(1 + count)", shrink=0.8)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")
print("  (namespace_heatmap split into two files)")

# ===================================================================
# 4. Centrality analysis
# ===================================================================
print("\n=== 4. Centrality analysis ===")

def centrality_analysis(G, label):
    print(f"  [{label}] Computing PageRank...")
    pr = nx.pagerank(G)
    top_pr = sorted(pr.items(), key=lambda x: -x[1])[:20]

    print(f"  [{label}] Computing betweenness centrality...")
    bc = nx.betweenness_centrality(G)
    top_bc = sorted(bc.items(), key=lambda x: -x[1])[:20]

    in_deg = dict(G.in_degree())
    top_in = sorted(in_deg.items(), key=lambda x: -x[1])[:20]

    set_in = {m for m, _ in top_in}
    set_pr = {m for m, _ in top_pr}
    set_bc = {m for m, _ in top_bc}

    stats = {
        "top_20_in_degree": [{"module": m, "value": int(d)} for m, d in top_in],
        "top_20_pagerank": [{"module": m, "value": round(v, 6)} for m, v in top_pr],
        "top_20_betweenness": [{"module": m, "value": round(v, 6)} for m, v in top_bc],
        "overlap": {
            "in_degree_AND_pagerank": sorted(set_in & set_pr),
            "in_degree_AND_betweenness": sorted(set_in & set_bc),
            "pagerank_AND_betweenness": sorted(set_pr & set_bc),
            "all_three": sorted(set_in & set_pr & set_bc),
        },
    }

    print(f"  [{label}] Top PageRank: {top_pr[0][0]} ({top_pr[0][1]:.4f})")
    print(f"  [{label}] Top betweenness: {top_bc[0][0]} ({top_bc[0][1]:.6f})")
    print(f"  [{label}] Overlap (all 3): {stats['overlap']['all_three']}")

    return stats, pr, bc, in_deg


raw_cent_stats, raw_pr, raw_bc, raw_in_deg = centrality_analysis(G_raw, "raw")
tr_cent_stats, tr_pr, tr_bc, tr_in_deg = centrality_analysis(G_tr, "TR")

results["centrality"] = {"raw": raw_cent_stats, "transitive_reduction": tr_cent_stats}

# Centrality scatter plots — all blue, 3 separate full-width files

# 1. In-degree vs PageRank (Raw)
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
nodes = list(raw_pr.keys())
x = [raw_in_deg[n] for n in nodes]
y = [raw_pr[n] for n in nodes]
ax.scatter(x, y, s=3, alpha=0.3, color=COLORS["primary"])
ax.set_xlabel("In-degree")
ax.set_ylabel("PageRank")
ax.set_title(r"In-degree vs PageRank ($G_{\mathrm{module}}$)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "module_centrality_indeg_pr.pdf", bbox_inches="tight")
plt.close()
print("  Saved module_centrality_indeg_pr.pdf")

# 2. In-degree vs Betweenness (Raw)
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
x = [raw_in_deg[n] for n in nodes]
y = [raw_bc[n] for n in nodes]
ax.scatter(x, y, s=3, alpha=0.3, color=COLORS["primary"])
ax.set_xlabel("In-degree")
ax.set_ylabel("Betweenness")
ax.set_title(r"In-degree vs Betweenness ($G_{\mathrm{module}}$)")
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-5)
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "module_centrality_indeg_betw.pdf", bbox_inches="tight")
plt.close()
print("  Saved module_centrality_indeg_betw.pdf")

# 3. Betweenness vs PageRank (Raw)
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
x = [raw_bc[n] for n in nodes]
y = [raw_pr[n] for n in nodes]
ax.scatter(x, y, s=3, alpha=0.3, color=COLORS["primary"])
ax.set_xlabel("Betweenness")
ax.set_ylabel("PageRank")
ax.set_title(r"Betweenness vs PageRank ($G_{\mathrm{module}}$)")
ax.set_xscale("symlog", linthresh=1e-5); ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "module_centrality_betw_pr.pdf", bbox_inches="tight")
plt.close()
print("  Saved module_centrality_betw_pr.pdf")

# ===================================================================
# 5. Connectivity analysis
# ===================================================================
print("\n=== 5. Connectivity analysis ===")

def connectivity_analysis(G, label):
    wcc = list(nx.weakly_connected_components(G))
    num_wcc = len(wcc)
    largest_wcc = max(len(c) for c in wcc)

    print(f"  [{label}] Weakly connected components: {num_wcc}, largest: {largest_wcc}")

    # Remove top-5 in-degree nodes
    in_deg = dict(G.in_degree())
    top5_in = [m for m, _ in sorted(in_deg.items(), key=lambda x: -x[1])[:5]]
    G_no_in5 = G.copy()
    G_no_in5.remove_nodes_from(top5_in)
    wcc_no_in5 = list(nx.weakly_connected_components(G_no_in5))
    print(f"  [{label}] After removing top-5 in-degree: {len(wcc_no_in5)} components")

    # Remove top-5 betweenness nodes
    bc = nx.betweenness_centrality(G)
    top5_bc = [m for m, _ in sorted(bc.items(), key=lambda x: -x[1])[:5]]
    G_no_bc5 = G.copy()
    G_no_bc5.remove_nodes_from(top5_bc)
    wcc_no_bc5 = list(nx.weakly_connected_components(G_no_bc5))
    print(f"  [{label}] After removing top-5 betweenness: {len(wcc_no_bc5)} components")

    # Size distribution of components after removal
    sizes_in5 = sorted([len(c) for c in wcc_no_in5], reverse=True)
    sizes_bc5 = sorted([len(c) for c in wcc_no_bc5], reverse=True)

    return {
        "num_weakly_connected_components": num_wcc,
        "largest_component_size": largest_wcc,
        "removed_top5_in_degree": {
            "nodes_removed": top5_in,
            "num_components": len(wcc_no_in5),
            "component_sizes_top10": sizes_in5[:10],
        },
        "removed_top5_betweenness": {
            "nodes_removed": top5_bc,
            "num_components": len(wcc_no_bc5),
            "component_sizes_top10": sizes_bc5[:10],
        },
    }


raw_conn_stats = connectivity_analysis(G_raw, "raw")
tr_conn_stats = connectivity_analysis(G_tr, "TR")

results["connectivity"] = {"raw": raw_conn_stats, "transitive_reduction": tr_conn_stats}

# ===================================================================
# Save results
# ===================================================================
print("\n=== Saving results ===")

# Remove longest_path list (too long for JSON readability)
for key in ["raw", "transitive_reduction"]:
    if "longest_path" in results["dag_structure"][key]:
        path = results["dag_structure"][key]["longest_path"]
        results["dag_structure"][key]["longest_path_full"] = path
        results["dag_structure"][key]["longest_path_preview"] = (
            path[:5] + ["..."] + path[-5:] if len(path) > 10 else path
        )
        del results["dag_structure"][key]["longest_path"]

JSON_OUTPUT = Path(__file__).resolve().parent.parent / "output" / "import_graph_analysis.json"
with open(JSON_OUTPUT, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved import_graph_analysis.json")
print("\nDone!")
