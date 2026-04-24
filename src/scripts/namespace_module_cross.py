#!/usr/bin/env python3
"""
Cross-tabulation of namespace partitions vs file-module partitions.

For each declaration with a known file module, extract namespace at
depth 1 and depth 2, then measure:
  (a) how many files each namespace spans
  (b) how many namespaces each file contains
  (c) NMI between the two partitions
  (d) pairwise agreement: same-ns-same-file vs same-ns-diff-file

Output: terminal + output/namespace_module_cross.txt
"""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import normalized_mutual_info_score

OUTPUT_DIR = Path("output")
FILEMAP_CSV = OUTPUT_DIR / "declaration_to_file_module.csv"


def ns_at_depth(name: str, depth: int) -> str:
    parts = name.split(".")
    if len(parts) <= 1:
        return "_root_"
    if depth >= len(parts):
        return ".".join(parts[:-1])
    return ".".join(parts[:depth])


def analyze_depth(decls, depth, out_fn):
    """Analyze namespace-module cross-tabulation at a given depth.

    decls: list of (name, file_module) tuples
    """
    out = out_fn

    # Build labels
    ns_labels = []
    mod_labels = []
    for name, fmod in decls:
        ns_labels.append(ns_at_depth(name, depth))
        mod_labels.append(fmod)

    unique_ns = sorted(set(ns_labels))
    unique_mod = sorted(set(mod_labels))
    out(f"  Declarations:      {len(decls):>10,}")
    out(f"  Unique namespaces: {len(unique_ns):>10,}")
    out(f"  Unique modules:    {len(unique_mod):>10,}")

    # ── (a) How many files does each namespace span? ──────────
    out(f"\n  (a) Files per namespace:")
    ns_to_files = defaultdict(set)
    for ns, mod in zip(ns_labels, mod_labels):
        ns_to_files[ns].add(mod)

    file_counts = np.array([len(ns_to_files[ns]) for ns in unique_ns])
    out(f"      mean={file_counts.mean():.1f}  "
        f"median={np.median(file_counts):.0f}  "
        f"max={int(file_counts.max())}")

    # Distribution buckets
    for threshold in [1, 2, 5, 10, 50, 100]:
        n = int(np.sum(file_counts <= threshold))
        out(f"      namespaces spanning ≤{threshold:>3} files: "
            f"{n:>6,}  ({100*n/len(unique_ns):.1f}%)")

    top_ns = sorted(unique_ns, key=lambda ns: len(ns_to_files[ns]),
                    reverse=True)[:20]
    out(f"\n      Top 20 namespaces by file span:")
    out(f"      {'Rank':<5} {'Namespace':<45} {'Files':>6} {'Decls':>7}")
    out(f"      {'-'*65}")
    ns_decl_count = defaultdict(int)
    for ns in ns_labels:
        ns_decl_count[ns] += 1
    for i, ns in enumerate(top_ns, 1):
        out(f"      {i:<5} {ns:<45} {len(ns_to_files[ns]):>6,} "
            f"{ns_decl_count[ns]:>7,}")

    # ── (b) How many namespaces does each file contain? ───────
    out(f"\n  (b) Namespaces per file:")
    mod_to_ns = defaultdict(set)
    for ns, mod in zip(ns_labels, mod_labels):
        mod_to_ns[mod].add(ns)

    ns_counts = np.array([len(mod_to_ns[mod]) for mod in unique_mod])
    out(f"      mean={ns_counts.mean():.1f}  "
        f"median={np.median(ns_counts):.0f}  "
        f"max={int(ns_counts.max())}")

    for threshold in [1, 2, 3, 5, 10]:
        n = int(np.sum(ns_counts <= threshold))
        out(f"      files with ≤{threshold:>2} namespaces: "
            f"{n:>6,}  ({100*n/len(unique_mod):.1f}%)")

    top_mod = sorted(unique_mod, key=lambda m: len(mod_to_ns[m]),
                     reverse=True)[:20]
    out(f"\n      Top 20 files by namespace count:")
    out(f"      {'Rank':<5} {'Module':<55} {'NS':>4} {'Decls':>7}")
    out(f"      {'-'*73}")
    mod_decl_count = defaultdict(int)
    for mod in mod_labels:
        mod_decl_count[mod] += 1
    for i, mod in enumerate(top_mod, 1):
        out(f"      {i:<5} {mod:<55} {len(mod_to_ns[mod]):>4} "
            f"{mod_decl_count[mod]:>7,}")

    # ── (c) NMI ───────────────────────────────────────────────
    out(f"\n  (c) Normalized Mutual Information:")
    t0 = time.time()
    nmi = normalized_mutual_info_score(ns_labels, mod_labels)
    out(f"      NMI(namespace, module) = {nmi:.4f}  ({time.time()-t0:.1f}s)")
    out(f"      (1.0 = perfect alignment, 0.0 = independent)")

    # ── (d) Pairwise agreement ────────────────────────────────
    out(f"\n  (d) Pairwise agreement (sampled):")
    t0 = time.time()
    # Exact pairwise on all N*(N-1)/2 pairs is too expensive.
    # Instead, compute via contingency table.
    # Number of pairs in same NS: sum over ns of C(count_ns, 2)
    # Number of pairs in same module: sum over mod of C(count_mod, 2)
    # Number of pairs in same NS AND same module:
    #   sum over (ns, mod) of C(count_{ns,mod}, 2)

    # Build contingency
    contingency = defaultdict(int)
    for ns, mod in zip(ns_labels, mod_labels):
        contingency[(ns, mod)] += 1

    N = len(decls)

    def comb2(n):
        return n * (n - 1) // 2

    total_pairs = comb2(N)
    same_ns_pairs = sum(comb2(ns_decl_count[ns]) for ns in unique_ns)
    same_mod_pairs = sum(comb2(mod_decl_count[mod]) for mod in unique_mod)
    same_both_pairs = sum(comb2(c) for c in contingency.values())

    same_ns_diff_mod = same_ns_pairs - same_both_pairs
    same_mod_diff_ns = same_mod_pairs - same_both_pairs

    out(f"      Total declaration pairs:     {total_pairs:>15,}")
    out(f"      Same NS, same module:        {same_both_pairs:>15,}  "
        f"({100*same_both_pairs/total_pairs:.4f}%)")
    out(f"      Same NS, different module:   {same_ns_diff_mod:>15,}  "
        f"({100*same_ns_diff_mod/total_pairs:.4f}%)")
    out(f"      Same module, different NS:   {same_mod_diff_ns:>15,}  "
        f"({100*same_mod_diff_ns/total_pairs:.4f}%)")

    # Among same-NS pairs, what fraction is also same-module?
    if same_ns_pairs > 0:
        overlap = 100 * same_both_pairs / same_ns_pairs
        out(f"\n      Of all same-NS pairs, {overlap:.1f}% are also "
            f"same-module")
        out(f"      (= {100-overlap:.1f}% of same-NS pairs are split "
            f"across files)")
    if same_mod_pairs > 0:
        overlap2 = 100 * same_both_pairs / same_mod_pairs
        out(f"      Of all same-module pairs, {overlap2:.1f}% are also "
            f"same-NS")
        out(f"      (= {100-overlap2:.1f}% of same-module pairs span "
            f"multiple NS)")
    out(f"      Time: {time.time()-t0:.1f}s")


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    # ── Load data ──────────────────────────────────────────────
    out("Loading data ...")
    t0 = time.time()

    if not FILEMAP_CSV.exists():
        out(f"  ERROR: {FILEMAP_CSV} not found. Run build_file_mapping.py first.")
        return

    fmap_df = pd.read_csv(FILEMAP_CSV)
    node_to_file = dict(zip(fmap_df["declaration_full_name"],
                            fmap_df["file_module"]))

    nodes_ds = load_dataset(
        "MathNetwork/MathlibGraph",
        data_files="mathlib_nodes.csv", split="train",
    )
    nodes_df = nodes_ds.to_pandas()
    out(f"  Nodes: {len(nodes_df):,}   File mapping: {len(node_to_file):,}"
        f"  ({time.time()-t0:.1f}s)")

    # Build list of (name, file_module) for declarations with file mapping
    decls = []
    for name in nodes_df["name"]:
        if isinstance(name, str) and name in node_to_file:
            decls.append((name, node_to_file[name]))
    out(f"  Declarations with file mapping: {len(decls):,}")

    # ── Depth 1 ────────────────────────────────────────────────
    out(f"\n{'='*78}")
    out(f"  DEPTH 1 (first name component)")
    out(f"{'='*78}")
    analyze_depth(decls, 1, out)

    # ── Depth 2 ────────────────────────────────────────────────
    out(f"\n{'='*78}")
    out(f"  DEPTH 2 (first two name components)")
    out(f"{'='*78}")
    analyze_depth(decls, 2, out)

    out(f"\n{'='*78}")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    outpath = OUTPUT_DIR / "namespace_module_cross.txt"
    outpath.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
