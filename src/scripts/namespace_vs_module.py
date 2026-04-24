#!/usr/bin/env python3
"""
Namespace vs File-Module diagnostic (v3).

Joins HuggingFace declaration data with the file-module mapping built from
Mathlib source, then:
  1. Reports match rate
  2. Compares name-derived namespace vs file-module path
  3. Recomputes module cohesion using real file modules
  4. Compares with the old namespace-based cohesion numbers
"""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = Path("output")
FILEMAP_CSV = OUTPUT_DIR / "declaration_to_file_module.csv"


# ── helpers ──────────────────────────────────────────────────

def extract_decl_namespace(name) -> str:
    """Parent namespace from declaration name."""
    if not isinstance(name, str):
        return ""
    idx = name.rfind(".")
    return name[:idx] if idx >= 0 else ""


def extract_file_namespace(file_module: str) -> str:
    """Strip 'Mathlib.' prefix and drop last component (filename).

    Mathlib.Data.Nat.Basic → Data.Nat
    """
    if file_module.startswith("Mathlib."):
        file_module = file_module[len("Mathlib."):]
    idx = file_module.rfind(".")
    return file_module[:idx] if idx >= 0 else file_module


def last_component(ns: str) -> str:
    idx = ns.rfind(".")
    return ns[idx + 1:] if idx >= 0 else ns


def classify(decl_ns: str, file_ns: str) -> int:
    """Three-level match: 1=suffix, 2=core-name, 3=containment, 0=none."""
    if not decl_ns and not file_ns:
        return 1
    if not decl_ns or not file_ns:
        return 0

    # Level 1: exact suffix match
    if decl_ns == file_ns or file_ns.endswith("." + decl_ns):
        return 1

    # Level 2: last component match
    if last_component(decl_ns) == last_component(file_ns):
        return 2

    # Level 3: any component of decl_ns in file_ns components
    file_parts = set(file_ns.split("."))
    if any(part in file_parts for part in decl_ns.split(".")):
        return 3

    return 0


# ── main ─────────────────────────────────────────────────────

def main():
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    # ── Load file mapping ──
    if not FILEMAP_CSV.exists():
        out("File mapping CSV not found. Run build_file_mapping.py first.")
        return

    out("Loading file mapping ...")
    filemap_df = pd.read_csv(FILEMAP_CSV)
    out(f"  {len(filemap_df):,} source declarations → file modules")

    # Build lookup: full_name → file_module
    # For duplicates keep last (shouldn't matter much)
    fmap_full = dict(zip(filemap_df["declaration_full_name"], filemap_df["file_module"]))

    # ── Load HuggingFace data ──
    out("\nLoading HuggingFace data ...")
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
    out(f"  {len(nodes_df):,} nodes, {len(edges_df):,} edges  ({time.time()-t0:.1f}s)")

    # ── Join: map each node to its file module ──
    out("\nJoining declarations with file modules ...")
    match_full = 0
    unmatched = 0
    node_file_module = {}  # graph node name → file_module

    for name in nodes_df["name"]:
        if not isinstance(name, str):
            unmatched += 1
            continue
        if name in fmap_full:
            node_file_module[name] = fmap_full[name]
            match_full += 1
        else:
            unmatched += 1

    total = len(nodes_df)
    out(f"  Matched (full name):    {match_full:>7,}  ({100*match_full/total:.1f}%)")
    out(f"  Unmatched:              {unmatched:>7,}  ({100*unmatched/total:.1f}%)")

    # ══════════════════════════════════════════════════════════
    # PART A: Namespace vs File Module comparison
    # ══════════════════════════════════════════════════════════
    out("\n" + "=" * 70)
    out("PART A: Declaration Namespace vs File-Module Path")
    out("=" * 70)

    # For matched nodes, compare decl_ns vs file_ns
    records = []
    for name, fmod in node_file_module.items():
        decl_ns = extract_decl_namespace(name)
        file_ns = extract_file_namespace(fmod)
        level = classify(decl_ns, file_ns)
        records.append((name, decl_ns, file_ns, fmod, level))

    match_df = pd.DataFrame(records, columns=["name", "decl_ns", "file_ns", "file_module", "level"])
    n = len(match_df)
    counts = {lv: (match_df["level"] == lv).sum() for lv in (1, 2, 3, 0)}

    out(f"\nMatched declarations analyzed: {n:,}")
    out(f"  Level 1 (exact suffix):     {counts[1]:>7,}  ({100*counts[1]/n:.1f}%)")
    out(f"  Level 2 (core-name):        {counts[2]:>7,}  ({100*counts[2]/n:.1f}%)")
    out(f"  Level 3 (containment):      {counts[3]:>7,}  ({100*counts[3]/n:.1f}%)")
    out(f"  ────────────────────────────────────")
    out(f"  Any match (1+2+3):          {counts[1]+counts[2]+counts[3]:>7,}  "
        f"({100*(counts[1]+counts[2]+counts[3])/n:.1f}%)")
    out(f"  No match (level 0):         {counts[0]:>7,}  ({100*counts[0]/n:.1f}%)")

    # Top 30 true mismatches
    nomatch = match_df[match_df["level"] == 0]
    out(f"\nTop 30 true mismatches (level 0, of {counts[0]:,}):")
    out("-" * 90)
    out(f"{'Declaration Name':<45} {'Decl NS':<22} {'File NS'}")
    out("-" * 90)
    for _, row in nomatch.head(30).iterrows():
        out(f"{row['name']:<45} {row['decl_ns']:<22} {row['file_ns']}")

    # Per top-level namespace mismatch rate
    out(f"\nPer top-level namespace mismatch rate (level 0, n >= 50):")
    out("-" * 70)
    match_df["top_ns"] = match_df["decl_ns"].apply(lambda s: s.split(".")[0] if s else "(root)")
    grp = match_df.groupby("top_ns").agg(
        total=("level", "size"),
        nomatch=("level", lambda s: (s == 0).sum()),
    )
    grp["rate"] = grp["nomatch"] / grp["total"]
    grp = grp[grp["total"] >= 50].sort_values("rate", ascending=False)
    out(f"{'Top NS':<30} {'Total':>7} {'NoMatch':>7} {'Rate':>7}")
    out("-" * 70)
    for ns, row in grp.head(30).iterrows():
        out(f"{ns:<30} {row['total']:>7,} {row['nomatch']:>7,} {row['rate']:>6.1%}")

    # ══════════════════════════════════════════════════════════
    # PART B: Cohesion comparison — namespace-based vs file-based
    # ══════════════════════════════════════════════════════════
    out("\n" + "=" * 70)
    out("PART B: Module Cohesion — Namespace-based vs File-based")
    out("=" * 70)

    # Old method: module = name.rsplit(".", 1)[0]  (= the dataset's "module" column)
    node_ns_module = dict(zip(nodes_df["name"], nodes_df["module"]))

    def compute_cohesion(node_to_mod: dict, label: str):
        mod_decl_count = defaultdict(int)
        mod_internal = defaultdict(int)
        mod_external = defaultdict(int)

        for name, mod in node_to_mod.items():
            if mod and isinstance(mod, str):
                mod_decl_count[mod] += 1

        edge_count = 0
        for _, row in edges_df.iterrows():
            src_mod = node_to_mod.get(row["source"])
            tgt_mod = node_to_mod.get(row["target"])
            if not src_mod or not tgt_mod or not isinstance(src_mod, str) or not isinstance(tgt_mod, str):
                continue
            edge_count += 1
            if src_mod == tgt_mod:
                mod_internal[src_mod] += 1
            else:
                mod_external[src_mod] += 1
                mod_external[tgt_mod] += 1

        cohesion = {}
        for mod in mod_decl_count:
            i = mod_internal[mod]
            e = mod_external[mod]
            total = i + e
            cohesion[mod] = i / total if total > 0 else 0.0

        vals = np.array(list(cohesion.values()))
        zero_count = np.sum(vals == 0.0)
        internal_total = sum(mod_internal.values())
        external_total = sum(mod_external.values())

        out(f"\n  [{label}]")
        out(f"  Modules:              {len(cohesion):,}")
        out(f"  Edges classified:     {edge_count:,}")
        out(f"    Intra-module:       {internal_total:,}  ({100*internal_total/edge_count:.1f}%)" if edge_count else "")
        out(f"    Cross-module:       {external_total:,}  ({100*external_total/edge_count:.1f}%)" if edge_count else "")
        out(f"  Cohesion mean:        {vals.mean():.4f}")
        out(f"  Cohesion median:      {np.median(vals):.4f}")
        out(f"  Cohesion std:         {vals.std():.4f}")
        out(f"  Cohesion max:         {vals.max():.4f}")
        out(f"  Zero-cohesion:        {zero_count:,} / {len(cohesion):,}  ({100*zero_count/len(cohesion):.1f}%)")

        return cohesion

    out("\nComputing cohesion with namespace-based modules (old method) ...")
    t0 = time.time()
    coh_ns = compute_cohesion(node_ns_module, "Namespace-based (old)")
    out(f"  Time: {time.time()-t0:.1f}s")

    out("\nComputing cohesion with file-based modules (new method) ...")
    t0 = time.time()
    coh_file = compute_cohesion(node_file_module, "File-based (new)")
    out(f"  Time: {time.time()-t0:.1f}s")

    # ── Side-by-side comparison ──
    out("\n" + "-" * 70)
    out("SIDE-BY-SIDE COMPARISON")
    out("-" * 70)

    ns_vals = np.array(list(coh_ns.values()))
    file_vals = np.array(list(coh_file.values()))
    ns_zero = np.sum(ns_vals == 0.0)
    file_zero = np.sum(file_vals == 0.0)

    out(f"{'Metric':<30} {'Namespace':>15} {'File-based':>15}")
    out("-" * 70)
    out(f"{'Modules':<30} {len(coh_ns):>15,} {len(coh_file):>15,}")
    out(f"{'Cohesion mean':<30} {ns_vals.mean():>15.4f} {file_vals.mean():>15.4f}")
    out(f"{'Cohesion median':<30} {np.median(ns_vals):>15.4f} {np.median(file_vals):>15.4f}")
    out(f"{'Cohesion max':<30} {ns_vals.max():>15.4f} {file_vals.max():>15.4f}")
    out(f"{'Zero-cohesion modules':<30} {ns_zero:>14,}  {file_zero:>14,} ")
    out(f"{'Zero-cohesion %':<30} {100*ns_zero/len(coh_ns):>14.1f}% {100*file_zero/len(coh_file):>14.1f}%")

    out("\n" + "-" * 70)
    out("VERDICT")
    out("-" * 70)
    diff_mean = abs(file_vals.mean() - ns_vals.mean())
    out(f"  Cohesion mean difference:  {diff_mean:.4f}")
    if diff_mean < 0.005:
        out("  The two methods produce VERY SIMILAR cohesion numbers.")
        out("  The namespace-based analysis in the paper is a reasonable proxy.")
    elif diff_mean < 0.02:
        out("  The two methods produce MODERATELY DIFFERENT cohesion numbers.")
        out("  The paper's conclusions hold qualitatively but numbers should be updated.")
    else:
        out("  The two methods produce SUBSTANTIALLY DIFFERENT cohesion numbers.")
        out("  The paper's cohesion analysis needs revision with file-based modules.")

    out("")
    out("=" * 70)

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    outpath = OUTPUT_DIR / "namespace_vs_module_v3.txt"
    outpath.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
