"""Reusable functions for Mathlib import graph analysis."""

import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np


def lean_path_to_module(path: Path, mathlib_parent: Path) -> str:
    """Convert a .lean file path to a dotted module name."""
    rel = path.relative_to(mathlib_parent)
    return str(rel).replace("/", ".").removesuffix(".lean")


def parse_imports(path: Path) -> list[str]:
    """Extract imported module names from a Lean 4 source file header."""
    imports = []
    in_block_comment = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            was_in_comment = in_block_comment > 0
            entered_comment = False
            i = 0
            while i < len(stripped):
                if i + 1 < len(stripped) and stripped[i:i + 2] == "/-":
                    in_block_comment += 1
                    entered_comment = True
                    i += 2
                elif i + 1 < len(stripped) and stripped[i:i + 2] == "-/":
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


def build_import_graph(mathlib_root: Path) -> tuple[nx.DiGraph, set[str]]:
    """Build the raw Mathlib-internal import graph from .lean files."""
    G = nx.DiGraph()
    file_modules = set()

    for lean_file in sorted(mathlib_root.rglob("*.lean")):
        mod = lean_path_to_module(lean_file, mathlib_root.parent)
        file_modules.add(mod)
        G.add_node(mod)

    for lean_file in sorted(mathlib_root.rglob("*.lean")):
        mod = lean_path_to_module(lean_file, mathlib_root.parent)
        for imp in parse_imports(lean_file):
            if imp.startswith("Mathlib.") and imp in file_modules:
                G.add_edge(mod, imp)

    return G, file_modules


def top_level_ns(module: str) -> str:
    """Extract top-level namespace: Mathlib.Algebra.* -> Algebra."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def degree_stats(G: nx.DiGraph) -> dict:
    """Compute in-degree and out-degree statistics."""
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    in_vals = np.array(list(in_deg.values()))
    out_vals = np.array(list(out_deg.values()))

    top_in = sorted(in_deg.items(), key=lambda x: -x[1])[:20]
    top_out = sorted(out_deg.items(), key=lambda x: -x[1])[:20]

    return {
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


def dag_stats(G: nx.DiGraph) -> dict:
    """Compute DAG structure statistics."""
    is_dag = nx.is_directed_acyclic_graph(G)
    longest_path = nx.dag_longest_path(G)
    longest_len = len(longest_path) - 1

    layers = list(nx.topological_generations(G))
    layer_sizes = [len(layer) for layer in layers]

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    sources = [n for n, d in in_deg.items() if d == 0]
    sinks = [n for n, d in out_deg.items() if d == 0]

    return {
        "is_dag": is_dag,
        "longest_path_length": longest_len,
        "longest_path": longest_path,
        "num_layers": len(layers),
        "layer_sizes": layer_sizes,
        "layer_width_max": max(layer_sizes),
        "layer_width_mean": round(float(np.mean(layer_sizes)), 2),
        "layer_width_median": round(float(np.median(layer_sizes)), 2),
        "num_sources": len(sources),
        "num_sinks": len(sinks),
        "sources": sources,
        "sinks": sinks,
    }


def namespace_stats(G: nx.DiGraph) -> dict:
    """Compute namespace cross-import statistics."""
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
    return {
        "intra_namespace_edges": intra,
        "cross_namespace_edges": cross,
        "intra_ratio": round(intra / total, 4) if total > 0 else 0,
        "cross_ratio": round(cross / total, 4) if total > 0 else 0,
        "ns_matrix": dict(ns_matrix),
    }


def centrality_stats(G: nx.DiGraph) -> dict:
    """Compute centrality measures and top-20 lists."""
    pr = nx.pagerank(G)
    bc = nx.betweenness_centrality(G)
    in_deg = dict(G.in_degree())

    top_pr = sorted(pr.items(), key=lambda x: -x[1])[:20]
    top_bc = sorted(bc.items(), key=lambda x: -x[1])[:20]
    top_in = sorted(in_deg.items(), key=lambda x: -x[1])[:20]

    set_in = {m for m, _ in top_in}
    set_pr = {m for m, _ in top_pr}
    set_bc = {m for m, _ in top_bc}

    return {
        "top_20_in_degree": [{"module": m, "value": int(d)} for m, d in top_in],
        "top_20_pagerank": [{"module": m, "value": round(v, 6)} for m, v in top_pr],
        "top_20_betweenness": [{"module": m, "value": round(v, 6)} for m, v in top_bc],
        "overlap": {
            "in_degree_AND_pagerank": sorted(set_in & set_pr),
            "in_degree_AND_betweenness": sorted(set_in & set_bc),
            "pagerank_AND_betweenness": sorted(set_pr & set_bc),
            "all_three": sorted(set_in & set_pr & set_bc),
        },
        "pagerank": pr,
        "betweenness": bc,
    }


def connectivity_stats(G: nx.DiGraph) -> dict:
    """Compute connectivity and robustness statistics."""
    wcc = list(nx.weakly_connected_components(G))

    in_deg = dict(G.in_degree())
    top5_in = [m for m, _ in sorted(in_deg.items(), key=lambda x: -x[1])[:5]]
    G_no_in5 = G.copy()
    G_no_in5.remove_nodes_from(top5_in)
    wcc_no_in5 = list(nx.weakly_connected_components(G_no_in5))

    bc = nx.betweenness_centrality(G)
    top5_bc = [m for m, _ in sorted(bc.items(), key=lambda x: -x[1])[:5]]
    G_no_bc5 = G.copy()
    G_no_bc5.remove_nodes_from(top5_bc)
    wcc_no_bc5 = list(nx.weakly_connected_components(G_no_bc5))

    return {
        "num_weakly_connected_components": len(wcc),
        "largest_component_size": max(len(c) for c in wcc),
        "removed_top5_in_degree": {
            "nodes_removed": top5_in,
            "num_components": len(wcc_no_in5),
            "component_sizes_top10": sorted(
                [len(c) for c in wcc_no_in5], reverse=True
            )[:10],
        },
        "removed_top5_betweenness": {
            "nodes_removed": top5_bc,
            "num_components": len(wcc_no_bc5),
            "component_sizes_top10": sorted(
                [len(c) for c in wcc_no_bc5], reverse=True
            )[:10],
        },
    }
