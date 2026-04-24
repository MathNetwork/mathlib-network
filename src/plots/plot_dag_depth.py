#!/usr/bin/env python3
"""Compute and plot DAG depth (topological layer width) for G_thm and G_ns^(2).

Also regenerates the module-level DAG figure with fixed labels.

Output:
  paper/analysis/thm_dag_structure.pdf
  paper/analysis/ns_dag_structure.pdf
  paper/analysis/dag_structure.pdf  (module, regenerated)
"""

import re
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

from plot_style import setup_style, COLORS, FIGSIZE_SINGLE, FIGSIZE_DOUBLE, FIGSIZE_TRIPLE, FIGSIZE_HEATMAP, FIGSIZE_HEATMAP_WIDE

COLORS = setup_style()

import matplotlib.pyplot as plt
import numpy as np

OUTDIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


def topological_layers_fast(adj, nodes):
    """Compute topological layer widths for a DAG given adjacency list.

    adj[u] = list of v such that u -> v (u depends on v).
    Returns list of layer sizes, where layer 0 = sinks (out-degree 0).
    We reverse the convention: layer 0 = sources (in-degree 0), deepest = sinks.
    """
    # Compute in-degree
    in_deg = defaultdict(int)
    for u in nodes:
        if u not in in_deg:
            in_deg[u] = 0
        for v in adj.get(u, []):
            in_deg[v] += 1

    # BFS from sources (in_deg == 0)
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

    # Count widths
    widths = [0] * (max_layer + 1)
    for n, l in layer.items():
        widths[l] += 1
    return widths


def main():
    # ================================================================
    # 1. Declaration-level DAG (G_thm)
    # ================================================================
    print("=== Declaration-level DAG ===")
    print("Loading HuggingFace data ...")
    t0 = time.time()

    from datasets import load_dataset
    nodes_ds = load_dataset(
        "MathNetwork/MathlibGraph",
        data_files="mathlib_nodes.csv", split="train",
    )
    edges_ds = load_dataset(
        "MathNetwork/MathlibGraph",
        data_files="mathlib_edges.csv", split="train",
    )
    node_names = set(nodes_ds.to_pandas()["name"].dropna())
    edges_df = edges_ds.to_pandas()
    print(f"  Loaded {len(node_names):,} decls, {len(edges_df):,} edges "
          f"({time.time()-t0:.1f}s)")

    # Build adjacency list (source -> target)
    print("Building G_thm adjacency ...")
    t0 = time.time()
    thm_adj = defaultdict(list)
    for _, row in edges_df.iterrows():
        s, t = row["source"], row["target"]
        if s in node_names and t in node_names and s != t:
            thm_adj[s].append(t)
    print(f"  Built adjacency ({time.time()-t0:.1f}s)")

    # Compute layers
    print("Computing topological layers for G_thm ...")
    t0 = time.time()
    thm_widths = topological_layers_fast(thm_adj, node_names)
    print(f"  Layers: {len(thm_widths)}, max width: {max(thm_widths)}, "
          f"({time.time()-t0:.1f}s)")

    # Sources & sinks
    in_deg = Counter()
    for u, targets in thm_adj.items():
        for v in targets:
            in_deg[v] += 1
    sources = sum(1 for n in node_names if in_deg[n] == 0)
    sinks = sum(1 for n in node_names if len(thm_adj.get(n, [])) == 0)
    print(f"  Sources: {sources:,}, Sinks: {sinks:,}")

    # Plot
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    ax.bar(range(len(thm_widths)), thm_widths,
           color=COLORS["secondary"], edgecolor="none", width=1.0)
    ax.set_xlabel("Topological layer")
    ax.set_ylabel("Number of declarations")
    ax.set_title(rf"DAG width by topological layer "
                 rf"($G_{{\mathrm{{thm}}}}$, {len(thm_widths)} layers)")
    ax.set_xlim(-1, len(thm_widths))
    plt.tight_layout()
    fig.savefig(OUTDIR / "thm_dag_structure.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved thm_dag_structure.pdf")

    # ================================================================
    # 2. Namespace-level DAG (G_ns^(2))
    # ================================================================
    print("\n=== Namespace-level DAG ===")

    def ns_at_depth(name, k=2):
        parts = name.split(".")
        if len(parts) <= k:
            return ".".join(parts[:-1]) if len(parts) > 1 else "_root_"
        return ".".join(parts[:k])

    print("Building G_ns^(2) ...")
    t0 = time.time()
    decl_to_ns = {name: ns_at_depth(name) for name in node_names}
    all_ns = set(decl_to_ns.values())

    ns_adj = defaultdict(set)
    for _, row in edges_df.iterrows():
        s, t = row["source"], row["target"]
        if s not in decl_to_ns or t not in decl_to_ns:
            continue
        ns_s, ns_t = decl_to_ns[s], decl_to_ns[t]
        if ns_s != ns_t:
            ns_adj[ns_s].add(ns_t)

    # Convert sets to lists for the layer function
    ns_adj_list = {k: list(v) for k, v in ns_adj.items()}
    print(f"  Nodes: {len(all_ns):,}, Edges: {sum(len(v) for v in ns_adj.values()):,} "
          f"({time.time()-t0:.1f}s)")

    # Check for cycles (namespace graph may have them)
    import networkx as nx
    G_ns = nx.DiGraph()
    G_ns.add_nodes_from(all_ns)
    for s, targets in ns_adj.items():
        for t in targets:
            G_ns.add_edge(s, t)

    is_dag = nx.is_directed_acyclic_graph(G_ns)
    print(f"  Is DAG: {is_dag}")

    if not is_dag:
        # Condense SCCs
        print("  Condensing SCCs ...")
        condensed = nx.condensation(G_ns)
        num_sccs = condensed.number_of_nodes()

        # Map each super-node to its SCC size
        scc_sizes = {}
        for node, data in condensed.nodes(data=True):
            scc_sizes[node] = len(data['members'])

        # Collect per-layer info: super-node count and largest SCC size
        layer_supernodes = []   # number of super-nodes per layer
        layer_giant_ns = []     # namespaces in the giant SCC at that layer (0 if none)
        giant_layer = None
        for i, gen in enumerate(nx.topological_generations(condensed)):
            layer_supernodes.append(len(gen))
            biggest = max(scc_sizes[n] for n in gen)
            if biggest > 100:
                giant_layer = i
                layer_giant_ns.append(biggest)
            else:
                layer_giant_ns.append(0)

        ns_widths = layer_supernodes
        print(f"  Condensed: {num_sccs} super-nodes, {len(ns_widths)} layers")
    else:
        ns_widths = [len(gen) for gen in nx.topological_generations(G_ns)]
        giant_layer = None
        layer_giant_ns = [0] * len(ns_widths)

    print(f"  Layers: {len(ns_widths)}, max width: {max(ns_widths)}")

    # Plot: super-node counts per layer, with giant SCC annotated
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    bars = ax.bar(range(len(ns_widths)), ns_widths,
                  color=COLORS["tertiary"], edgecolor="none", width=1.0)
    ax.set_xlabel("Topological layer")
    ax.set_ylabel("Number of super-nodes (SCCs)")
    ax.set_title(rf"Condensed DAG by topological layer "
                 rf"($G_{{\mathrm{{ns}}}}^{{(2)}}$, {len(ns_widths)} layers, "
                 rf"{num_sccs:,} super-nodes)")
    ax.set_xlim(-1, len(ns_widths))

    # Annotate the giant SCC
    if giant_layer is not None:
        giant_ns = layer_giant_ns[giant_layer]
        bar_height = ns_widths[giant_layer]
        ax.annotate(
            f"Giant SCC\n({giant_ns:,} namespaces)",
            xy=(giant_layer, bar_height),
            xytext=(giant_layer + 1.2, bar_height + max(ns_widths) * 0.15),
            color=COLORS["tertiary"], ha="center",
            arrowprops=dict(arrowstyle="->", color=COLORS["tertiary"], lw=1.2),
        )

    plt.tight_layout()
    fig.savefig(OUTDIR / "ns_dag_structure.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved ns_dag_structure.pdf")

    # ================================================================
    # 3. Regenerate module-level DAG with fixed labels
    # ================================================================
    print("\n=== Module-level DAG (label fix) ===")
    MATHLIB_ROOT = Path(__file__).resolve().parent.parent.parent / "mathlib4" / "Mathlib"

    def lean_path_to_module(path):
        rel = path.relative_to(MATHLIB_ROOT.parent)
        return str(rel).replace("/", ".").removesuffix(".lean")

    def parse_imports(path):
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

    print("Building module graph ...")
    t0 = time.time()
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
    print(f"  Raw: {G_raw.number_of_nodes()} nodes, {G_raw.number_of_edges()} edges")

    print("  Computing TR ...")
    G_tr = nx.transitive_reduction(G_raw)
    G_tr.add_nodes_from(G_raw.nodes())
    print(f"  TR: {G_tr.number_of_nodes()} nodes, {G_tr.number_of_edges()} edges "
          f"({time.time()-t0:.1f}s)")

    raw_layers = [len(g) for g in nx.topological_generations(G_raw)]
    tr_layers = [len(g) for g in nx.topological_generations(G_tr)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.bar(range(len(raw_layers)), raw_layers,
            color=COLORS["primary"], edgecolor="none", width=1.0)
    ax1.set_xlabel("Topological layer")
    ax1.set_ylabel("Number of modules")
    ax1.set_title(rf"DAG width by topological layer "
                  rf"($G_{{\mathrm{{module}}}}$, {len(raw_layers)} layers)")
    ax1.set_xlim(-1, len(raw_layers))

    ax2.bar(range(len(tr_layers)), tr_layers,
            color=COLORS["primary"], edgecolor="none", width=1.0, alpha=0.7)
    ax2.set_xlabel("Topological layer")
    ax2.set_ylabel("Number of modules")
    ax2.set_title(rf"DAG width by topological layer "
                  rf"($G_{{\mathrm{{module}}}}^{{-}}$, {len(tr_layers)} layers)")
    ax2.set_xlim(-1, len(tr_layers))

    plt.tight_layout()
    fig.savefig(OUTDIR / "dag_structure.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved dag_structure.pdf")


if __name__ == "__main__":
    main()
