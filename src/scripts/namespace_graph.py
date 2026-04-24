#!/usr/bin/env python3
"""
Build the namespace-level dependency graph G_ns from declaration-level
HuggingFace data, and compute cross-boundary percentages at every
namespace depth.

Depth d truncates a declaration name to its first d dot-separated
components (the part before the (d+1)-th dot).  Depth "max" keeps
everything before the last dot (the parent namespace).

Output: terminal + output/namespace_depth_stats.txt
"""

import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = Path("output")
FILEMAP_CSV = OUTPUT_DIR / "declaration_to_file_module.csv"


def ns_at_depth(name: str, depth: int) -> str:
    """Truncate a fully qualified name to the first `depth` components.

    ns_at_depth("CategoryTheory.Category.Iso.mk", 1) → "CategoryTheory"
    ns_at_depth("CategoryTheory.Category.Iso.mk", 2) → "CategoryTheory.Category"
    ns_at_depth("CategoryTheory.Category.Iso.mk", 3) → "CategoryTheory.Category.Iso"
    ns_at_depth("Nat.add_comm", 1) → "Nat"
    ns_at_depth("True", 1) → "_root_"

    If the name has fewer than depth+1 components, returns the parent
    namespace (everything before the last dot), or "_root_" if no dot.
    """
    parts = name.split(".")
    if len(parts) <= 1:
        return "_root_"
    # Need at least depth+1 parts for the name to have depth components
    # as namespace (the last part is always the short name).
    if depth >= len(parts):
        # Fall back to parent namespace
        return ".".join(parts[:-1])
    return ".".join(parts[:depth])


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    # ── Load data ──────────────────────────────────────────────
    out("Loading HuggingFace data ...")
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
    out(f"  Nodes: {len(nodes_df):,}   Edges: {len(edges_df):,}  ({time.time()-t0:.1f}s)")

    # Load file mapping for file-level comparison
    node_to_file = {}
    if FILEMAP_CSV.exists():
        fmap_df = pd.read_csv(FILEMAP_CSV)
        node_to_file = dict(zip(fmap_df["declaration_full_name"],
                                fmap_df["file_module"]))
        out(f"  File mapping: {len(node_to_file):,} declarations")
    else:
        out(f"  WARNING: {FILEMAP_CSV} not found, file-level row will be skipped")

    # ── Precompute: split each name into parts ─────────────────
    out("\nPrecomputing name components ...")
    # Only keep the 308,129 Mathlib declarations (nodes_df has 317,655
    # rows but some names are NaN or non-string).
    all_names = [n for n in nodes_df["name"] if isinstance(n, str)]
    name_parts = {n: n.split(".") for n in all_names}

    # Find max depth: max number of dots in any name
    max_depth = max(len(p) - 1 for p in name_parts.values())
    out(f"  Declarations: {len(name_parts):,}")
    out(f"  Max dot-depth: {max_depth}  (i.e. up to {max_depth+1} components)")

    # Distribution of dot-depths
    depth_dist = defaultdict(int)
    for p in name_parts.values():
        depth_dist[len(p) - 1] += 1
    out(f"\n  Dot-depth distribution (depth = number of dots in name):")
    out(f"  {'Depth':<7} {'Count':>10} {'Cumul%':>8}")
    out(f"  {'-'*27}")
    cumul = 0
    total_n = len(name_parts)
    for d in range(max_depth + 1):
        cnt = depth_dist.get(d, 0)
        cumul += cnt
        out(f"  {d:<7} {cnt:>10,} {100*cumul/total_n:>7.1f}%")

    # ── For each depth, compute namespace labels ───────────────
    # Pre-build: for each depth d (1..max_depth), a dict name → ns_label
    # depth "max" is len(parts)-1, i.e. parent namespace
    out(f"\n{'='*78}")
    out(f"  Cross-boundary analysis by namespace depth")
    out(f"{'='*78}")

    # Convert edge arrays to Python lists for speed
    src_list = edges_df["source"].tolist()
    tgt_list = edges_df["target"].tolist()
    n_edges = len(src_list)

    results = []  # (label, unique_ns, intra, cross, cross_pct)

    # Depths 1 through max_depth
    for d in range(1, max_depth + 1):
        t0 = time.time()

        # Build name → ns mapping at this depth
        ns_map = {}
        for name, parts in name_parts.items():
            if len(parts) <= 1:
                ns_map[name] = "_root_"
            elif d >= len(parts):
                ns_map[name] = ".".join(parts[:-1])
            else:
                ns_map[name] = ".".join(parts[:d])

        unique_ns = len(set(ns_map.values()))

        intra = 0
        cross = 0
        for i in range(n_edges):
            s_ns = ns_map.get(src_list[i])
            t_ns = ns_map.get(tgt_list[i])
            if s_ns is None or t_ns is None:
                continue
            if s_ns == t_ns:
                intra += 1
            else:
                cross += 1

        mapped = intra + cross
        cross_pct = 100 * cross / mapped if mapped > 0 else 0
        elapsed = time.time() - t0

        is_max = (d == max_depth)
        label = f"{d} (max)" if is_max else str(d)
        results.append((label, unique_ns, intra, cross, cross_pct))

        if d <= 10 or d == max_depth or d % 5 == 0:
            print(f"  depth {d:>2}: {unique_ns:>6,} ns, "
                  f"cross {cross_pct:5.1f}%  ({elapsed:.1f}s)")

    # Top-level directory row (§3's ns(m): first component after "Mathlib.")
    if node_to_file:
        t0 = time.time()
        node_to_topdir = {}
        for name, fmod in node_to_file.items():
            # fmod like "Mathlib.Algebra.Group.Basic" → "Algebra"
            parts = fmod.split(".")
            if len(parts) >= 2 and parts[0] == "Mathlib":
                node_to_topdir[name] = parts[1]
            else:
                node_to_topdir[name] = parts[0]
        unique_dirs = len(set(node_to_topdir.values()))
        intra_d = 0
        cross_d = 0
        for i in range(n_edges):
            s_d = node_to_topdir.get(src_list[i])
            t_d = node_to_topdir.get(tgt_list[i])
            if s_d is None or t_d is None:
                continue
            if s_d == t_d:
                intra_d += 1
            else:
                cross_d += 1
        mapped_d = intra_d + cross_d
        cross_pct_d = 100 * cross_d / mapped_d if mapped_d > 0 else 0
        results.append(("topdir", unique_dirs, intra_d, cross_d, cross_pct_d))
        print(f"  topdir:  {unique_dirs:>6,} dirs,    "
              f"cross {cross_pct_d:5.1f}%  ({time.time()-t0:.1f}s)")

    # File-level row
    if node_to_file:
        t0 = time.time()
        unique_files = len(set(node_to_file.values()))
        intra_f = 0
        cross_f = 0
        for i in range(n_edges):
            s_f = node_to_file.get(src_list[i])
            t_f = node_to_file.get(tgt_list[i])
            if s_f is None or t_f is None:
                continue
            if s_f == t_f:
                intra_f += 1
            else:
                cross_f += 1
        mapped_f = intra_f + cross_f
        cross_pct_f = 100 * cross_f / mapped_f if mapped_f > 0 else 0
        results.append(("file", unique_files, intra_f, cross_f, cross_pct_f))
        print(f"  file:    {unique_files:>6,} modules, "
              f"cross {cross_pct_f:5.1f}%  ({time.time()-t0:.1f}s)")

    # ── Output table ───────────────────────────────────────────
    out(f"\n  {'Depth':<10} {'Unique ns':>10} {'Intra-ns':>14} "
        f"{'Cross-ns':>14} {'Cross %':>9}")
    out(f"  {'-'*59}")
    for label, unique_ns, intra, cross, cross_pct in results:
        out(f"  {label:<10} {unique_ns:>10,} {intra:>14,} "
            f"{cross:>14,} {cross_pct:>8.1f}%")

    # ── Interpretation ─────────────────────────────────────────
    out(f"\n  Reference:")
    out(f"    §3 G_import cross-namespace (top-level dir): 37.1%")
    out(f"    §4 G_thm   cross-module    (file-level):    90.4%")
    out(f"")
    out(f"  'topdir' = Mathlib top-level subdirectory, = §3's ns(m)")
    out(f"  Depth 1 = top-level name component (Nat, List, Set, ...)")
    out(f"  Depth max = parent namespace (everything before last dot)")
    out(f"  'file' = source file (from declaration_to_file_module.csv)")

    out(f"\n{'='*78}")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    outpath = OUTPUT_DIR / "namespace_depth_stats.txt"
    outpath.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
