#!/usr/bin/env python3
"""Build mathlib_modules.csv with per-module precomputed metrics.

Columns: module, decl_count, in_degree, out_degree, pagerank, betweenness,
         dag_layer, community_id, cohesion, import_utilization_median.

Module graph from GEXF. Declaration-to-file mapping from jixia data.
"""

import json
import statistics
import time
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

HF_DIR = Path.home() / ".cache/huggingface/hub/datasets--MathNetwork--MathlibGraph/snapshots/bc4173ec3beda64713ae81f602ce224491c61703"
GEXF_PATH = Path(__file__).resolve().parents[2] / "mathlib4" / "mathlib_import_graph.gexf"
JIXIA_DIR = Path(__file__).resolve().parents[2] / "data" / "jixia_decls"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "release"


def build_decl_to_file_module():
    """Build declaration name -> file module mapping from jixia JSON files."""
    decl_to_mod = {}
    for f in JIXIA_DIR.glob("*.json"):
        file_module = f.stem.replace("_", ".")
        with open(f) as fh:
            decls = json.load(fh)
        for d in decls:
            name_parts = d.get("name", [])
            if isinstance(name_parts, list):
                full_name = ".".join(str(p) for p in name_parts)
            else:
                full_name = str(name_parts)
            decl_to_mod[full_name] = file_module
    return decl_to_mod


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load module import graph from GEXF ----
    print("Loading module graph from GEXF ...")
    G = nx.read_gexf(GEXF_PATH)
    modules = sorted(G.nodes())
    print(f"  {G.number_of_nodes()} modules, {G.number_of_edges()} import edges")

    # ---- 2. Build decl -> file module mapping from jixia ----
    print("Building decl-to-file mapping from jixia ...")
    t0 = time.time()
    decl_to_mod = build_decl_to_file_module()
    print(f"  {len(decl_to_mod):,} declarations mapped ({time.time()-t0:.1f}s)")

    # decl_count per file module
    mod_set = set(modules)
    decl_count = defaultdict(int)
    for name, mod in decl_to_mod.items():
        if mod in mod_set:
            decl_count[mod] += 1

    # ---- 3. Load edges for cohesion + utilization ----
    print("Loading edges ...")
    edges_df = pd.read_csv(HF_DIR / "mathlib_edges.csv")
    src_arr = edges_df["source"].values
    tgt_arr = edges_df["target"].values
    print(f"  {len(edges_df):,} edges")

    # ---- 4. Module graph metrics ----
    print("Computing module graph metrics ...")

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    # PageRank
    t0 = time.time()
    pr = nx.pagerank(G, alpha=0.85)
    print(f"  PageRank ({time.time()-t0:.1f}s)")

    # Betweenness (exact, matches populate_cache.py:218)
    t0 = time.time()
    bc = nx.betweenness_centrality(G)
    print(f"  Betweenness exact ({time.time()-t0:.1f}s)")

    # DAG layers
    t0 = time.time()
    layer_map = {}
    for layer_idx, gen in enumerate(nx.topological_generations(G)):
        for n in gen:
            layer_map[n] = layer_idx
    max_layer = max(layer_map.values()) if layer_map else 0
    print(f"  DAG: {max_layer + 1} layers, max depth {max_layer} ({time.time()-t0:.1f}s)")

    # Louvain communities (matches populate_cache.py:272)
    t0 = time.time()
    import community as community_louvain
    G_und = G.to_undirected()
    partition = community_louvain.best_partition(G_und, random_state=42)
    modularity_score = community_louvain.modularity(partition, G_und)
    n_communities = len(set(partition.values()))
    print(f"  Louvain: {n_communities} communities, modularity={modularity_score:.4f} ({time.time()-t0:.1f}s)")

    # ---- 5. Cohesion per module ----
    print("Computing cohesion ...")
    t0 = time.time()
    mod_internal = defaultdict(int)
    mod_external = defaultdict(int)

    for i in range(len(src_arr)):
        s, t = src_arr[i], tgt_arr[i]
        if s == t:
            continue
        sm = decl_to_mod.get(s)
        tm = decl_to_mod.get(t)
        if sm is None or tm is None:
            continue
        if sm == tm:
            mod_internal[sm] += 1
        else:
            mod_external[sm] += 1
            mod_external[tm] += 1

    cohesion = {}
    for m in modules:
        i_val = mod_internal.get(m, 0)
        e_val = mod_external.get(m, 0)
        total = i_val + e_val
        cohesion[m] = i_val / total if total > 0 else 0.0
    print(f"  ({time.time()-t0:.1f}s)")

    # ---- 6. Import utilization per module ----
    # util(m_i, m_j) = |refs(m_i) ∩ D_{m_j}| / |D_{m_j}|
    # Per-module: median of util(m, *) across all imports of m
    print("Computing import utilization ...")
    t0 = time.time()

    # Declarations per file module
    mod_decls = defaultdict(set)
    for name, mod in decl_to_mod.items():
        mod_decls[mod].add(name)

    # used[(m_i, m_j)] = set of declarations in m_j referenced by m_i
    used = defaultdict(set)
    for i in range(len(src_arr)):
        s, t = src_arr[i], tgt_arr[i]
        if s == t:
            continue
        sm = decl_to_mod.get(s)
        tm = decl_to_mod.get(t)
        if sm and tm and sm != tm:
            used[(sm, tm)].add(t)

    # Compute per-import-edge utilization, grouped by importer
    # GEXF edge direction: A -> B means "A is imported by B" (A is dependency)
    # So for import edge (importer=B, imported=A), GEXF has edge A -> B
    # util(B, A) = |refs(B) ∩ D_A| / |D_A|
    per_module_utils = defaultdict(list)
    all_utils = []
    for dep, importer in G.edges():  # dep -> importer means importer imports dep
        total_dep = len(mod_decls.get(dep, set()))
        if total_dep == 0:
            continue
        used_count = len(used.get((importer, dep), set()))
        util = used_count / total_dep
        per_module_utils[importer].append(util)
        all_utils.append(util)

    import_util_median = {}
    for m in modules:
        utils = per_module_utils.get(m, [])
        import_util_median[m] = statistics.median(utils) if utils else 0.0
    print(f"  ({time.time()-t0:.1f}s)")

    # ---- 7. Assemble DataFrame ----
    df = pd.DataFrame({
        "module": modules,
        "decl_count": [decl_count.get(m, 0) for m in modules],
        "in_degree": [in_deg.get(m, 0) for m in modules],
        "out_degree": [out_deg.get(m, 0) for m in modules],
        "pagerank": [pr.get(m, 0.0) for m in modules],
        "betweenness": [bc.get(m, 0.0) for m in modules],
        "dag_layer": [layer_map.get(m, -1) for m in modules],
        "community_id": [partition.get(m, -1) for m in modules],
        "cohesion": [cohesion.get(m, 0.0) for m in modules],
        "import_utilization_median": [import_util_median.get(m, 0.0) for m in modules],
    })

    out_path = OUT_DIR / "mathlib_modules.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # ---- 8. Verification ----
    print("\n=== Verification ===")

    print(f"  Row count: {len(df)} (expected ~7,563)")

    print(f"  DAG: {max_layer + 1} layers, max depth {max_layer} (expected ~154 layers / ~153 depth)")

    print(f"  Louvain: {n_communities} communities, modularity={modularity_score:.4f} (expected ~9 major, ~0.6395)")

    total_decls = df["decl_count"].sum()
    print(f"  Sum decl_count: {total_decls:,} (expected ~308,129)")

    if all_utils:
        global_median = statistics.median(all_utils)
        print(f"  Global import utilization median: {global_median:.4f} (expected ~0.016)")

    coh_vals = df["cohesion"].values
    coh_nonzero = coh_vals[coh_vals > 0]
    print(f"  Cohesion: mean={coh_vals.mean():.4f}, median={np.median(coh_vals):.4f} (over {len(coh_nonzero)} non-zero modules)")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")
    print("Phase 2 DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
