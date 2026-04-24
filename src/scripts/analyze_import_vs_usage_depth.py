#!/usr/bin/env python3
"""Analyze import depth vs actual usage depth.

Core question: What is the relationship between the depth of modules
a file imports and the depth of modules its declarations actually use?

- import depth: depth of modules that module A directly imports
- usage depth: depth of modules containing declarations that A's
  declarations actually cite as premises in G_thm
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

# ── paths ──
OUTPUT_DIR = Path("output")
FILEMAP_CSV = OUTPUT_DIR / "declaration_to_file_module.csv"
MATHLIB_DIR = Path("/tmp/mathlib4_thm_lemma")
RESULT_FILE = OUTPUT_DIR / "import_vs_usage_depth.txt"


def module_depth(name: str) -> int:
    """Module depth = number of dot-separated components.
    Mathlib.Data.Nat.Basic → 4."""
    return len(name.split("."))


def top_dir(name: str) -> str:
    """Top-level directory: second component after 'Mathlib'.
    Mathlib.Algebra.Group.Defs → Algebra."""
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "Mathlib":
        return parts[1]
    return parts[0]


# ── 1. Load G_thm from HuggingFace ──
def load_thm_graph():
    print("Loading G_thm from HuggingFace …")
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
    print(f"  nodes: {len(nodes_df):,}  edges: {len(edges_df):,}")
    return nodes_df, edges_df


# ── 2. Load file mapping ──
def load_file_mapping():
    print(f"Loading file mapping from {FILEMAP_CSV} …")
    fmap_df = pd.read_csv(FILEMAP_CSV)
    node_to_file = dict(
        zip(fmap_df["declaration_full_name"], fmap_df["file_module"])
    )
    print(f"  mapped declarations: {len(node_to_file):,}")
    return node_to_file


# ── 3. Extract G_import edges from Mathlib source ──
def load_import_edges():
    print(f"Extracting G_import from {MATHLIB_DIR} …")
    mathlib_src = MATHLIB_DIR / "Mathlib"
    pat = re.compile(r"^(?:public\s+)?(?:meta\s+)?import\s+(\S+)")

    import_adj: dict[str, set[str]] = defaultdict(set)

    for lean_file in sorted(mathlib_src.rglob("*.lean")):
        rel = lean_file.relative_to(MATHLIB_DIR)
        parts = list(rel.parts)
        parts[-1] = parts[-1].removesuffix(".lean")
        src_module = ".".join(parts)

        in_block = False
        with open(lean_file, errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if in_block:
                    if "-/" in stripped:
                        in_block = False
                    continue
                if stripped.startswith("/-"):
                    if "-/" not in stripped[2:]:
                        in_block = True
                    continue
                if not stripped or stripped.startswith("--"):
                    continue
                if stripped == "module":
                    continue
                m = pat.match(stripped)
                if m:
                    tgt = m.group(1)
                    if tgt != src_module:
                        import_adj[src_module].add(tgt)
                else:
                    break

    n_edges = sum(len(v) for v in import_adj.values())
    print(f"  modules with imports: {len(import_adj):,}  edges: {n_edges:,}")
    return import_adj


# ── 4. Build per-module usage adjacency from G_thm ──
def build_usage_adj(edges_df, node_to_file):
    """For each module A, collect the set of *other* modules whose
    declarations are cited by A's declarations."""
    print("Building per-module usage adjacency …")
    usage_adj: dict[str, set[str]] = defaultdict(set)
    mapped = 0
    for src, tgt in zip(edges_df["source"], edges_df["target"]):
        src_mod = node_to_file.get(src)
        tgt_mod = node_to_file.get(tgt)
        if src_mod and tgt_mod and src_mod != tgt_mod:
            usage_adj[src_mod].add(tgt_mod)
            mapped += 1
    print(f"  cross-module declaration edges with mapping: {mapped:,}")
    print(f"  modules with outgoing usage: {len(usage_adj):,}")
    return usage_adj


# ── 5. Compute per-module statistics ──
def compute_per_module(import_adj, usage_adj):
    all_modules = set(import_adj.keys()) | set(usage_adj.keys())
    rows = []
    for mod in sorted(all_modules):
        imp_targets = import_adj.get(mod, set())
        use_targets = usage_adj.get(mod, set())
        if not imp_targets and not use_targets:
            continue

        imp_depths = [module_depth(t) for t in imp_targets] if imp_targets else []
        use_depths = [module_depth(t) for t in use_targets] if use_targets else []

        mean_imp = np.mean(imp_depths) if imp_depths else np.nan
        mean_use = np.mean(use_depths) if use_depths else np.nan
        diff = mean_use - mean_imp if imp_depths and use_depths else np.nan

        rows.append({
            "module": mod,
            "n_imports": len(imp_targets),
            "n_usage": len(use_targets),
            "mean_import_depth": mean_imp,
            "mean_usage_depth": mean_use,
            "depth_diff": diff,
            "top_dir": top_dir(mod),
        })

    df = pd.DataFrame(rows)
    return df


def fmt(x, decimals=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{decimals}f}"


def main():
    out_lines: list[str] = []

    def pr(s=""):
        print(s)
        out_lines.append(s)

    # Load data
    nodes_df, edges_df = load_thm_graph()
    node_to_file = load_file_mapping()
    import_adj = load_import_edges()
    usage_adj = build_usage_adj(edges_df, node_to_file)

    # Compute per-module stats
    df = compute_per_module(import_adj, usage_adj)
    valid = df.dropna(subset=["depth_diff"])

    pr("=" * 70)
    pr("IMPORT DEPTH vs USAGE DEPTH ANALYSIS")
    pr("=" * 70)

    # ── Global statistics ──
    pr("\n── Global Statistics ──")
    pr(f"Modules with both import and usage data: {len(valid):,}")
    pr(f"  mean(depth_diff)   = {valid['depth_diff'].mean():+.4f}")
    pr(f"  median(depth_diff) = {valid['depth_diff'].median():+.4f}")
    pr(f"  std(depth_diff)    = {valid['depth_diff'].std():.4f}")
    pr(f"  min(depth_diff)    = {valid['depth_diff'].min():+.4f}")
    pr(f"  max(depth_diff)    = {valid['depth_diff'].max():+.4f}")

    pos = (valid["depth_diff"] > 0).sum()
    neg = (valid["depth_diff"] < 0).sum()
    zero = (valid["depth_diff"] == 0).sum()
    pr(f"\n  depth_diff > 0 (usage deeper than imports): {pos:,} ({100*pos/len(valid):.1f}%)")
    pr(f"  depth_diff < 0 (usage shallower than imports): {neg:,} ({100*neg/len(valid):.1f}%)")
    pr(f"  depth_diff = 0: {zero:,} ({100*zero/len(valid):.1f}%)")

    pr(f"\n  mean(mean_import_depth) = {valid['mean_import_depth'].mean():.3f}")
    pr(f"  mean(mean_usage_depth)  = {valid['mean_usage_depth'].mean():.3f}")

    # ── Distribution of depth_diff ──
    pr("\n── Distribution of depth_diff ──")
    bins = [(-np.inf, -1.0), (-1.0, -0.5), (-0.5, -0.1), (-0.1, 0.1),
            (0.1, 0.5), (0.5, 1.0), (1.0, np.inf)]
    for lo, hi in bins:
        cnt = ((valid["depth_diff"] > lo) & (valid["depth_diff"] <= hi)).sum()
        pr(f"  ({lo:+.1f}, {hi:+.1f}]: {cnt:>5,} ({100*cnt/len(valid):5.1f}%)")

    # ── Top 5 modules: usage much deeper than imports ──
    pr("\n── Top 5: usage deeper than imports (largest depth_diff) ──")
    top5_pos = valid.nlargest(5, "depth_diff")
    for _, row in top5_pos.iterrows():
        mod = row["module"]
        pr(f"\n  {mod}")
        pr(f"    imports: {row['n_imports']:.0f} modules, mean depth {fmt(row['mean_import_depth'])}")
        pr(f"    usage:   {row['n_usage']:.0f} modules, mean depth {fmt(row['mean_usage_depth'])}")
        pr(f"    diff:    {row['depth_diff']:+.3f}")
        # Show sample imports and usages
        imp_set = import_adj.get(mod, set())
        use_set = usage_adj.get(mod, set())
        if imp_set:
            sample_imp = sorted(imp_set, key=module_depth)[:3]
            pr(f"    sample imports:  {', '.join(sample_imp)}")
        if use_set:
            sample_use = sorted(use_set, key=module_depth, reverse=True)[:3]
            pr(f"    sample usages (deepest): {', '.join(sample_use)}")

    # ── Top 5 modules: usage much shallower than imports ──
    pr("\n── Top 5: usage shallower than imports (smallest depth_diff) ──")
    top5_neg = valid.nsmallest(5, "depth_diff")
    for _, row in top5_neg.iterrows():
        mod = row["module"]
        pr(f"\n  {mod}")
        pr(f"    imports: {row['n_imports']:.0f} modules, mean depth {fmt(row['mean_import_depth'])}")
        pr(f"    usage:   {row['n_usage']:.0f} modules, mean depth {fmt(row['mean_usage_depth'])}")
        pr(f"    diff:    {row['depth_diff']:+.3f}")
        imp_set = import_adj.get(mod, set())
        use_set = usage_adj.get(mod, set())
        if imp_set:
            sample_imp = sorted(imp_set, key=module_depth, reverse=True)[:3]
            pr(f"    sample imports (deepest): {', '.join(sample_imp)}")
        if use_set:
            sample_use = sorted(use_set, key=module_depth)[:3]
            pr(f"    sample usages (shallowest): {', '.join(sample_use)}")

    # ── By top-level directory ──
    pr("\n── By top-level directory ──")
    mathlib_valid = valid[valid["top_dir"] != "Mathlib"].copy()
    grp = mathlib_valid.groupby("top_dir").agg(
        n_modules=("module", "count"),
        mean_imp=("mean_import_depth", "mean"),
        mean_use=("mean_usage_depth", "mean"),
        mean_diff=("depth_diff", "mean"),
        median_diff=("depth_diff", "median"),
    ).sort_values("mean_diff")

    # Filter to directories with >= 10 modules
    grp = grp[grp["n_modules"] >= 10]
    pr(f"{'Directory':<25} {'N':>5} {'MeanImp':>8} {'MeanUse':>8} {'MeanDiff':>9} {'MedDiff':>8}")
    pr("-" * 70)
    for d, row in grp.iterrows():
        pr(f"{d:<25} {row['n_modules']:>5.0f} {row['mean_imp']:>8.3f} {row['mean_use']:>8.3f} {row['mean_diff']:>+9.4f} {row['median_diff']:>+8.4f}")

    # ── Interpretation ──
    overall_diff = valid["depth_diff"].mean()
    pr("\n── Interpretation ──")
    if overall_diff > 0:
        pr(f"Overall mean depth_diff = {overall_diff:+.4f} > 0")
        pr("→ Actual usage reaches DEEPER modules than what is directly imported.")
        pr("  Imports serve as shallow entry points; transitive chains reach deeper.")
    elif overall_diff < 0:
        pr(f"Overall mean depth_diff = {overall_diff:+.4f} < 0")
        pr("→ Actual usage reaches SHALLOWER modules than what is directly imported.")
        pr("  Through deep imports, declarations access shallow infrastructure.")
    else:
        pr("Overall mean depth_diff ≈ 0: import and usage depths are balanced.")

    # ── Save ──
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nResults saved to {RESULT_FILE}")

    # Also save per-module CSV
    csv_path = OUTPUT_DIR / "import_vs_usage_depth_per_module.csv"
    df.to_csv(csv_path, index=False)
    print(f"Per-module data saved to {csv_path}")


if __name__ == "__main__":
    main()
