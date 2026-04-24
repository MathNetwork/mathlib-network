#!/usr/bin/env python3
"""
Build namespace-level and file-module-level dependency graphs from
declaration-level data, compute basic statistics, and compare with
the module import graph G_import from §3.

Steps:
  1. Load HuggingFace nodes/edges + file-module mapping
  2. Label each declaration with namespace and file module
  3. Aggregate declaration-level edges into G_ns and G_file
  4. Compute statistics and compare with G_import

Output: terminal + output/namespace_graph_stats.txt
"""

import re
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = Path("output")
FILEMAP_CSV = OUTPUT_DIR / "declaration_to_file_module.csv"
MATHLIB_DIR = Path("/tmp/mathlib4_thm_lemma")

# G_import reference numbers from §3
G_IMPORT_NODES = 7_563
G_IMPORT_EDGES = 23_570


def extract_namespace(name: str) -> str:
    """Parent namespace: 'Foo.Bar.baz' → 'Foo.Bar'."""
    if not isinstance(name, str):
        return ""
    idx = name.rfind(".")
    return name[:idx] if idx >= 0 else ""


def build_aggregated_graph(edges_df, node_to_label, label_name):
    """Aggregate declaration-level edges by a labeling function.

    Returns (graph_edges, intra_count, cross_count, unmapped_count)
    where graph_edges is {(src_label, tgt_label): weight}.
    """
    graph = defaultdict(int)
    intra = 0
    cross = 0
    unmapped = 0

    for src, tgt in zip(edges_df["source"], edges_df["target"]):
        src_label = node_to_label.get(src)
        tgt_label = node_to_label.get(tgt)
        if not src_label or not tgt_label:
            unmapped += 1
            continue
        if src_label == tgt_label:
            intra += 1
        else:
            cross += 1
            graph[(src_label, tgt_label)] += 1

    return dict(graph), intra, cross, unmapped


def graph_stats(graph_edges, all_nodes, label):
    """Compute basic statistics for an aggregated graph."""
    n_nodes = len(all_nodes)
    n_edges = len(graph_edges)
    total_weight = sum(graph_edges.values())

    # Degree distributions (unweighted: count distinct neighbours)
    out_neighbours = defaultdict(set)
    in_neighbours = defaultdict(set)
    out_weight = defaultdict(int)
    in_weight = defaultdict(int)

    for (s, t), w in graph_edges.items():
        out_neighbours[s].add(t)
        in_neighbours[t].add(s)
        out_weight[s] += w
        in_weight[t] += w

    out_deg = np.array([len(out_neighbours.get(n, set())) for n in all_nodes])
    in_deg = np.array([len(in_neighbours.get(n, set())) for n in all_nodes])
    out_w = np.array([out_weight.get(n, 0) for n in all_nodes])
    in_w = np.array([in_weight.get(n, 0) for n in all_nodes])

    max_possible = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
    density = n_edges / max_possible

    stats = {
        "label": label,
        "nodes": n_nodes,
        "edges_unweighted": n_edges,
        "edges_total_weight": total_weight,
        "density": density,
        "out_deg_mean": out_deg.mean(),
        "out_deg_median": np.median(out_deg),
        "out_deg_max": int(out_deg.max()),
        "in_deg_mean": in_deg.mean(),
        "in_deg_median": np.median(in_deg),
        "in_deg_max": int(in_deg.max()),
        "out_weight_mean": out_w.mean(),
        "out_weight_max": int(out_w.max()),
        "in_weight_mean": in_w.mean(),
        "in_weight_max": int(in_w.max()),
    }

    # Top 10 by weighted in-degree
    sorted_in = sorted(all_nodes, key=lambda n: in_weight.get(n, 0), reverse=True)
    stats["top_in"] = [(n, in_weight.get(n, 0), len(in_neighbours.get(n, set())))
                       for n in sorted_in[:10]]

    # Top 10 by weighted out-degree
    sorted_out = sorted(all_nodes, key=lambda n: out_weight.get(n, 0), reverse=True)
    stats["top_out"] = [(n, out_weight.get(n, 0), len(out_neighbours.get(n, set())))
                        for n in sorted_out[:10]]

    return stats


def format_stats(s, intra, cross, unmapped):
    """Format statistics as text lines."""
    lines = []
    total_decl_edges = intra + cross + unmapped
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  {s['label']}")
    lines.append(f"{'=' * 70}")
    lines.append(f"  Nodes:                  {s['nodes']:>10,}")
    lines.append(f"  Edges (unweighted):     {s['edges_unweighted']:>10,}")
    lines.append(f"  Edges (total weight):   {s['edges_total_weight']:>10,}")
    lines.append(f"  Density:                {s['density']:>13.6f}")
    lines.append(f"")
    lines.append(f"  Declaration-level edges mapped:")
    lines.append(f"    Intra (same unit):    {intra:>10,}  "
                 f"({100*intra/(intra+cross):.1f}% of mapped)" if intra + cross > 0 else "")
    lines.append(f"    Cross (different):    {cross:>10,}  "
                 f"({100*cross/(intra+cross):.1f}% of mapped)" if intra + cross > 0 else "")
    lines.append(f"    Unmapped:             {unmapped:>10,}")
    lines.append(f"")
    lines.append(f"  Out-degree (unweighted):  mean={s['out_deg_mean']:.1f}  "
                 f"median={s['out_deg_median']:.0f}  max={s['out_deg_max']}")
    lines.append(f"  In-degree  (unweighted):  mean={s['in_deg_mean']:.1f}  "
                 f"median={s['in_deg_median']:.0f}  max={s['in_deg_max']}")
    lines.append(f"  Out-weight (weighted):    mean={s['out_weight_mean']:.1f}  "
                 f"max={s['out_weight_max']}")
    lines.append(f"  In-weight  (weighted):    mean={s['in_weight_mean']:.1f}  "
                 f"max={s['in_weight_max']}")

    lines.append(f"\n  Top 10 by weighted in-degree:")
    lines.append(f"  {'Rank':<5} {'Node':<45} {'Weight':>8} {'In-deg':>7}")
    lines.append(f"  {'-'*67}")
    for i, (n, w, d) in enumerate(s["top_in"], 1):
        lines.append(f"  {i:<5} {n:<45} {w:>8,} {d:>7,}")

    lines.append(f"\n  Top 10 by weighted out-degree:")
    lines.append(f"  {'Rank':<5} {'Node':<45} {'Weight':>8} {'Out-deg':>7}")
    lines.append(f"  {'-'*67}")
    for i, (n, w, d) in enumerate(s["top_out"], 1):
        lines.append(f"  {i:<5} {n:<45} {w:>8,} {d:>7,}")

    return lines


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    def out_lines(ls):
        for l in ls:
            out(l)

    # ── Step 1: Load data ──────────────────────────────────────
    out("Step 1: Loading data ...")

    if not FILEMAP_CSV.exists():
        out(f"  ERROR: {FILEMAP_CSV} not found. Run build_file_mapping.py first.")
        return

    filemap_df = pd.read_csv(FILEMAP_CSV)
    fmap = dict(zip(filemap_df["declaration_full_name"], filemap_df["file_module"]))
    out(f"  File mapping: {len(fmap):,} declarations → file modules")

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
    out(f"  HuggingFace: {len(nodes_df):,} nodes, {len(edges_df):,} edges  ({time.time()-t0:.1f}s)")

    # ── Step 1b: Label each declaration ────────────────────────
    out("\nStep 1b: Labeling declarations ...")

    node_to_ns = {}       # name → namespace
    node_to_file = {}     # name → file module
    ns_match = 0
    file_match = 0

    for name in nodes_df["name"]:
        if not isinstance(name, str):
            continue
        ns = extract_namespace(name)
        if ns:
            node_to_ns[name] = ns
            ns_match += 1
        if name in fmap:
            node_to_file[name] = fmap[name]
            file_match += 1

    total = len(nodes_df)
    out(f"  Namespace assigned:   {ns_match:>10,}  ({100*ns_match/total:.1f}%)")
    out(f"  File module matched:  {file_match:>10,}  ({100*file_match/total:.1f}%)")

    all_namespaces = set(node_to_ns.values())
    all_file_modules = set(node_to_file.values())
    out(f"  Unique namespaces:    {len(all_namespaces):>10,}")
    out(f"  Unique file modules:  {len(all_file_modules):>10,}")

    # ── Step 2: Build aggregated graphs ────────────────────────
    out("\n" + "=" * 70)
    out("Step 2: Building aggregated graphs ...")
    out("=" * 70)

    out("\n  Building G_ns (namespace-level) ...")
    t0 = time.time()
    g_ns_edges, ns_intra, ns_cross, ns_unmapped = build_aggregated_graph(
        edges_df, node_to_ns, "namespace"
    )
    out(f"  Done in {time.time()-t0:.1f}s")

    out("\n  Building G_file (file-module-level) ...")
    t0 = time.time()
    g_file_edges, file_intra, file_cross, file_unmapped = build_aggregated_graph(
        edges_df, node_to_file, "file_module"
    )
    out(f"  Done in {time.time()-t0:.1f}s")

    # ── Step 3: G_ns statistics ────────────────────────────────
    ns_stats = graph_stats(g_ns_edges, all_namespaces, "G_ns (Namespace-level dependency graph)")
    out_lines(format_stats(ns_stats, ns_intra, ns_cross, ns_unmapped))

    # Sanity check: cross-namespace %
    if ns_intra + ns_cross > 0:
        cross_pct = 100 * ns_cross / (ns_intra + ns_cross)
        out(f"\n  Sanity check: cross-namespace edges = {cross_pct:.1f}%")
        out(f"  (§4 reports 50.9% cross-namespace at declaration level)")

    # ── Step 4: G_file statistics ──────────────────────────────
    file_stats = graph_stats(g_file_edges, all_file_modules, "G_file (File-module-level dependency graph)")
    out_lines(format_stats(file_stats, file_intra, file_cross, file_unmapped))

    # Sanity check: intra-module %
    if file_intra + file_cross > 0:
        intra_pct = 100 * file_intra / (file_intra + file_cross)
        out(f"\n  Sanity check: intra-module edges = {intra_pct:.1f}%")
        out(f"  (§4 reports 9.6% intra-module at declaration level)")

    # ── Step 5: Cross-graph comparison ─────────────────────────
    out("\n" + "=" * 70)
    out("Step 5: Three-graph comparison")
    out("=" * 70)

    # G_import reference
    g_import_density = G_IMPORT_EDGES / (G_IMPORT_NODES * (G_IMPORT_NODES - 1))
    g_import_avg_out = G_IMPORT_EDGES / G_IMPORT_NODES

    header = (f"  {'Metric':<30} {'G_import (§3)':>15} "
              f"{'G_file (agg)':>15} {'G_ns (ns)':>15}")
    sep = f"  {'-'*75}"

    out(f"\n{header}")
    out(sep)
    out(f"  {'Nodes':<30} {G_IMPORT_NODES:>15,} "
        f"{file_stats['nodes']:>15,} {ns_stats['nodes']:>15,}")
    out(f"  {'Edges (unweighted)':<30} {G_IMPORT_EDGES:>15,} "
        f"{file_stats['edges_unweighted']:>15,} {ns_stats['edges_unweighted']:>15,}")
    out(f"  {'Edges (total weight)':<30} {'—':>15} "
        f"{file_stats['edges_total_weight']:>15,} {ns_stats['edges_total_weight']:>15,}")
    out(f"  {'Density':<30} {g_import_density:>15.6f} "
        f"{file_stats['density']:>15.6f} {ns_stats['density']:>15.6f}")
    out(f"  {'Avg out-degree (unw)':<30} {g_import_avg_out:>15.1f} "
        f"{file_stats['out_deg_mean']:>15.1f} {ns_stats['out_deg_mean']:>15.1f}")
    out(f"  {'Median out-degree (unw)':<30} {'—':>15} "
        f"{file_stats['out_deg_median']:>15.0f} {ns_stats['out_deg_median']:>15.0f}")
    out(f"  {'Max out-degree (unw)':<30} {'—':>15} "
        f"{file_stats['out_deg_max']:>15,} {ns_stats['out_deg_max']:>15,}")
    out(f"  {'Avg in-degree (unw)':<30} {g_import_avg_out:>15.1f} "
        f"{file_stats['in_deg_mean']:>15.1f} {ns_stats['in_deg_mean']:>15.1f}")
    out(f"  {'Max in-degree (unw)':<30} {'—':>15} "
        f"{file_stats['in_deg_max']:>15,} {ns_stats['in_deg_max']:>15,}")

    # G_file vs G_import edge overlap
    out(f"\n  G_file vs G_import edge comparison:")
    out(f"    G_import has {G_IMPORT_NODES:,} nodes, G_file has {file_stats['nodes']:,} nodes")
    out(f"    G_import has {G_IMPORT_EDGES:,} edges, G_file has {file_stats['edges_unweighted']:,} edges (unweighted)")
    ratio = file_stats['edges_unweighted'] / G_IMPORT_EDGES if G_IMPORT_EDGES > 0 else 0
    out(f"    Ratio G_file/G_import edges: {ratio:.2f}x")
    out(f"    (G_import includes transitive edges from 'import' statements;")
    out(f"     G_file only counts direct declaration-level usage)")

    out("")
    out("=" * 70)

    # ── Step 6: G_import vs G_file edge-level comparison ──────
    out("\n" + "=" * 70)
    out("Step 6: G_import vs G_file edge-level comparison")
    out("=" * 70)

    # 6a: Extract G_import from Mathlib source
    out("\n  6a: Extracting G_import from Mathlib source ...")
    t0 = time.time()
    g_import_edges_set = set()   # set of (src_module, tgt_module)
    g_import_adj = defaultdict(set)  # adjacency list for transitive closure
    pat_import = re.compile(r"^(?:public\s+)?import\s+(\S+)")
    mathlib_root = MATHLIB_DIR
    mathlib_src = MATHLIB_DIR / "Mathlib"

    pat_meta_import = re.compile(r"^(?:public\s+)?(?:meta\s+)?import\s+(\S+)")
    import_file_count = 0
    for lean_file in sorted(mathlib_src.rglob("*.lean")):
        rel = lean_file.relative_to(mathlib_root)
        parts = list(rel.parts)
        parts[-1] = parts[-1].removesuffix(".lean")
        src_module = ".".join(parts)
        import_file_count += 1

        in_block = False
        with open(lean_file, "r", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                # Handle block comments
                if in_block:
                    if "-/" in stripped:
                        in_block = False
                    continue
                if stripped.startswith("/-"):
                    if "-/" not in stripped[2:]:
                        in_block = True
                    continue
                # Skip line comments and blank lines
                if not stripped or stripped.startswith("--"):
                    continue
                # Skip the `module` keyword line
                if stripped == "module":
                    continue
                # Match import lines (including "public meta import")
                m = pat_meta_import.match(stripped)
                if m:
                    tgt_module = m.group(1)
                    if src_module != tgt_module:
                        g_import_edges_set.add((src_module, tgt_module))
                        g_import_adj[src_module].add(tgt_module)
                else:
                    # First non-import, non-comment, non-blank line → header is done
                    break

    out(f"    Scanned {import_file_count:,} files")
    out(f"    G_import edges extracted: {len(g_import_edges_set):,}")
    out(f"    (§3 reports {G_IMPORT_EDGES:,} edges)")
    out(f"    Time: {time.time()-t0:.1f}s")

    # All G_import nodes
    g_import_nodes_set = set()
    for s, t in g_import_edges_set:
        g_import_nodes_set.add(s)
        g_import_nodes_set.add(t)
    out(f"    G_import nodes: {len(g_import_nodes_set):,}")

    # 6b: Compute transitive closure of G_import
    out("\n  6b: Computing transitive closure of G_import ...")
    t0 = time.time()

    def transitive_reachable(adj, src):
        """BFS to find all nodes reachable from src."""
        visited = set()
        queue = deque()
        for n in adj.get(src, set()):
            queue.append(n)
            visited.add(n)
        while queue:
            cur = queue.popleft()
            for n in adj.get(cur, set()):
                if n not in visited:
                    visited.add(n)
                    queue.append(n)
        return visited

    # Build transitive closure as a set of (src, tgt) pairs
    # Only need to check reachability for modules that appear in G_file
    g_import_trans = set()
    all_src_modules = set(s for s, _ in g_file_edges.keys())
    checked = 0
    for src in all_src_modules:
        reachable = transitive_reachable(g_import_adj, src)
        for tgt in reachable:
            g_import_trans.add((src, tgt))
        checked += 1

    out(f"    Checked {checked:,} source modules")
    out(f"    Transitive closure pairs: {len(g_import_trans):,}")
    out(f"    Time: {time.time()-t0:.1f}s")

    # 6c: Edge-level comparison
    out("\n  6c: Edge-level comparison ...")
    g_file_edges_set = set(g_file_edges.keys())

    # Category 1: Active imports — in G_import AND in G_file
    active_imports = g_import_edges_set & g_file_edges_set
    # Category 2: Unused imports — in G_import but NOT in G_file
    unused_imports = g_import_edges_set - g_file_edges_set
    # Category 3: Indirect usage — in G_file but NOT in G_import (direct)
    indirect_usage = g_file_edges_set - g_import_edges_set
    # Category 3b: of those indirect, how many reachable transitively?
    indirect_reachable = indirect_usage & g_import_trans
    indirect_unreachable = indirect_usage - g_import_trans

    n_import = len(g_import_edges_set)
    n_file = len(g_file_edges_set)

    out(f"\n  G_import edges (direct):           {n_import:>10,}")
    out(f"  G_file edges (unweighted):         {n_file:>10,}")
    out(f"")
    out(f"  ┌─ G_import breakdown ─────────────────────────────────────┐")
    out(f"  │ Active imports (in both):       {len(active_imports):>8,}  "
        f"({100*len(active_imports)/n_import:.1f}% of G_import) │")
    out(f"  │ Unused imports (G_imp only):    {len(unused_imports):>8,}  "
        f"({100*len(unused_imports)/n_import:.1f}% of G_import) │")
    out(f"  └──────────────────────────────────────────────────────────┘")
    out(f"")
    out(f"  ┌─ G_file breakdown ────────────────────────────────────────┐")
    out(f"  │ Direct import exists:           {len(active_imports):>8,}  "
        f"({100*len(active_imports)/n_file:.1f}% of G_file)   │")
    out(f"  │ Indirect usage (no direct imp): {len(indirect_usage):>8,}  "
        f"({100*len(indirect_usage)/n_file:.1f}% of G_file)   │")
    out(f"  │   ├─ Transitively reachable:    {len(indirect_reachable):>8,}  "
        f"({100*len(indirect_reachable)/n_file:.1f}% of G_file)   │")
    out(f"  │   └─ Not reachable:             {len(indirect_unreachable):>8,}  "
        f"({100*len(indirect_unreachable)/n_file:.1f}% of G_file)   │")
    out(f"  └───────────────────────────────────────────────────────────┘")

    out(f"\n  Interpretation:")
    out(f"    'Active imports' = file A imports file B AND A's declarations cite B's.")
    out(f"    'Unused imports' = file A imports B but no declaration in A cites one in B.")
    out(f"      These may serve transitive visibility or be genuinely redundant.")
    out(f"    'Indirect usage' = A's declarations cite B's but A does not directly import B.")
    out(f"      These reach B through transitive imports.")

    # 6d: Transitive redundancy re-evaluation
    out(f"\n  6d: Transitive redundancy re-evaluation ...")

    # §3's method: an import edge A→B is transitively redundant if B is reachable
    # from A via other import edges (i.e., removing A→B doesn't disconnect A from B).
    # We compute this for the full G_import.
    t0 = time.time()
    redundant_count = 0
    non_redundant_count = 0
    for src, tgt in g_import_edges_set:
        # Check if tgt is reachable from src via other edges (excluding direct src→tgt)
        other_neighbours = g_import_adj[src] - {tgt}
        if not other_neighbours:
            non_redundant_count += 1
            continue
        # BFS from other neighbours
        visited = set(other_neighbours)
        queue = deque(other_neighbours)
        found = False
        while queue:
            cur = queue.popleft()
            if cur == tgt:
                found = True
                break
            for n in g_import_adj.get(cur, set()):
                if n not in visited:
                    visited.add(n)
                    queue.append(n)
        if found:
            redundant_count += 1
        else:
            non_redundant_count += 1

    redundancy_rate = 100 * redundant_count / n_import if n_import > 0 else 0
    out(f"    Total G_import edges:         {n_import:>8,}")
    out(f"    Transitively redundant:       {redundant_count:>8,}  ({redundancy_rate:.1f}%)")
    out(f"    Non-redundant (essential):    {non_redundant_count:>8,}  ({100-redundancy_rate:.1f}%)")
    out(f"    §3 reports:                   17.5%")
    out(f"    Time: {time.time()-t0:.1f}s")

    # 6e: "Unused" imports breakdown — are they transitively redundant?
    out(f"\n  6e: Unused imports breakdown ...")
    unused_and_redundant = 0
    unused_and_essential = 0
    for src, tgt in unused_imports:
        other_neighbours = g_import_adj[src] - {tgt}
        if not other_neighbours:
            unused_and_essential += 1
            continue
        visited = set(other_neighbours)
        queue = deque(other_neighbours)
        found = False
        while queue:
            cur = queue.popleft()
            if cur == tgt:
                found = True
                break
            for n in g_import_adj.get(cur, set()):
                if n not in visited:
                    visited.add(n)
                    queue.append(n)
        if found:
            unused_and_redundant += 1
        else:
            unused_and_essential += 1

    out(f"    Unused imports total:         {len(unused_imports):>8,}")
    out(f"      Also transitively redund.: {unused_and_redundant:>8,}  "
        f"({100*unused_and_redundant/len(unused_imports):.1f}% of unused)" if unused_imports else "")
    out(f"      Essential (sole path):     {unused_and_essential:>8,}  "
        f"({100*unused_and_essential/len(unused_imports):.1f}% of unused)" if unused_imports else "")
    out(f"    Essential unused imports provide transitive visibility for other modules.")

    out("")
    out("=" * 70)

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    outpath = OUTPUT_DIR / "namespace_graph_stats.txt"
    outpath.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
