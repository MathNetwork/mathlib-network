#!/usr/bin/env python3
"""
Analyze depth patterns in the module import graph G_import.

For each module, compute its depth (number of dot-separated components).
Then for each import edge (source, target), compare source/target depths
to identify depth asymmetries (shallow→deep, deep→shallow, same-depth).

Output: terminal + output/module_depth_analysis.txt
"""

import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("output")
MATHLIB_DIR = Path("/tmp/mathlib4_thm_lemma")


def module_depth(name: str) -> int:
    """Depth = number of dot-separated components. Mathlib.Data.Nat.Basic → 4."""
    return len(name.split("."))


def main():
    lines = []

    def out(s="", end="\n"):
        print(s, end=end)
        if end == "\n":
            lines.append(s)
        else:
            if lines:
                lines[-1] += s
            else:
                lines.append(s)

    out("=" * 70)
    out("  MODULE DEPTH ANALYSIS (G_import)")
    out("=" * 70)

    # ── Extract G_import from Mathlib source ──
    out("\nExtracting G_import from Mathlib source ...")
    t0 = time.time()

    mathlib_src = MATHLIB_DIR / "Mathlib"
    if not mathlib_src.exists():
        out(f"  ERROR: {mathlib_src} not found.")
        return

    pat_import = re.compile(r"^(?:public\s+)?(?:meta\s+)?import\s+(\S+)")
    edges = []  # list of (src_module, tgt_module)
    all_modules = set()

    for lean_file in sorted(mathlib_src.rglob("*.lean")):
        rel = lean_file.relative_to(MATHLIB_DIR)
        parts = list(rel.parts)
        parts[-1] = parts[-1].removesuffix(".lean")
        src_module = ".".join(parts)
        all_modules.add(src_module)

        in_block = False
        with open(lean_file, "r", errors="replace") as f:
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
                m = pat_import.match(stripped)
                if m:
                    tgt_module = m.group(1)
                    if src_module != tgt_module:
                        edges.append((src_module, tgt_module))
                        all_modules.add(tgt_module)
                else:
                    break

    out(f"  Modules: {len(all_modules):,}")
    out(f"  Import edges: {len(edges):,}")
    out(f"  Time: {time.time()-t0:.1f}s")

    # ── 1. Module depth distribution ──
    out(f"\n{'─'*70}")
    out("  1. MODULE DEPTH DISTRIBUTION")
    out(f"{'─'*70}")

    depth_counter = Counter()
    for mod in all_modules:
        d = module_depth(mod)
        depth_counter[d] += 1

    depths = sorted(depth_counter.keys())
    out(f"\n  {'Depth':<10} {'Count':>8} {'Percentage':>12}")
    out(f"  {'-'*30}")
    for d in depths:
        pct = 100.0 * depth_counter[d] / len(all_modules)
        out(f"  {d:<10} {depth_counter[d]:>8,} {pct:>11.1f}%")

    all_depths = [module_depth(mod) for mod in all_modules]
    out(f"\n  Mean depth: {np.mean(all_depths):.2f}")
    out(f"  Median depth: {np.median(all_depths):.1f}")
    out(f"  Max depth: {max(all_depths)}")

    # ── 2. Edge depth patterns ──
    out(f"\n{'─'*70}")
    out("  2. IMPORT EDGE DEPTH PATTERNS")
    out(f"{'─'*70}")

    same_depth = 0
    src_deeper = 0  # source deeper than target (deep → shallow)
    tgt_deeper = 0  # target deeper than source (shallow → deep)
    depth_diffs = []
    src_depth_list = []
    tgt_depth_list = []

    for src, tgt in edges:
        sd = module_depth(src)
        td = module_depth(tgt)
        src_depth_list.append(sd)
        tgt_depth_list.append(td)
        diff = sd - td
        depth_diffs.append(diff)
        if diff == 0:
            same_depth += 1
        elif diff > 0:
            src_deeper += 1
        else:
            tgt_deeper += 1

    n = len(edges)
    out(f"\n  Total import edges: {n:,}")
    out(f"  Same depth:                {same_depth:>8,}  ({100*same_depth/n:.1f}%)")
    out(f"  Source deeper (deep→shallow): {src_deeper:>8,}  ({100*src_deeper/n:.1f}%)")
    out(f"  Target deeper (shallow→deep): {tgt_deeper:>8,}  ({100*tgt_deeper/n:.1f}%)")
    out(f"\n  Mean depth diff (src - tgt): {np.mean(depth_diffs):+.3f}")
    out(f"  Median depth diff:           {np.median(depth_diffs):+.1f}")
    out(f"  Mean source depth: {np.mean(src_depth_list):.2f}")
    out(f"  Mean target depth: {np.mean(tgt_depth_list):.2f}")

    # ── 3. Cross-tabulation: source depth × target depth ──
    out(f"\n{'─'*70}")
    out("  3. SOURCE DEPTH × TARGET DEPTH CROSS-TABLE")
    out(f"{'─'*70}")

    cross = Counter()
    for src, tgt in edges:
        sd = module_depth(src)
        td = module_depth(tgt)
        cross[(sd, td)] += 1

    max_d = min(max(depths), 7)  # cap display at 7
    out(f"\n  {'':>12}", end="")
    for td in range(1, max_d + 1):
        out(f"  tgt={td:>2}", end="")
    out("")
    for sd in range(1, max_d + 1):
        out(f"  src={sd:>2}  ", end="")
        for td in range(1, max_d + 1):
            c = cross.get((sd, td), 0)
            if c > 0:
                out(f"  {c:>6,}", end="")
            else:
                out(f"  {'·':>6}", end="")
        out("")

    # ── 4. Imports targeting shallow modules ──
    out(f"\n{'─'*70}")
    out("  4. IMPORTS TARGETING SHALLOW MODULES")
    out(f"{'─'*70}")

    for max_tgt_depth in [2, 3]:
        count = sum(1 for _, tgt in edges if module_depth(tgt) <= max_tgt_depth)
        out(f"\n  Edges targeting depth ≤ {max_tgt_depth}: "
            f"{count:,} / {n:,} ({100*count/n:.1f}%)")

    # What are the most-imported shallow modules?
    out(f"\n  Top 20 most-imported modules with depth ≤ 2:")
    tgt_counter = Counter()
    for _, tgt in edges:
        if module_depth(tgt) <= 2:
            tgt_counter[tgt] += 1
    out(f"  {'Rank':<6} {'Module':<45} {'In-degree':>10}")
    out(f"  {'-'*61}")
    for rank, (mod, cnt) in enumerate(tgt_counter.most_common(20), 1):
        out(f"  {rank:<6} {mod:<45} {cnt:>10,}")

    # ── 5. Depth profile by direction ──
    out(f"\n{'─'*70}")
    out("  5. DEPTH PROFILE OF SOURCES AND TARGETS")
    out(f"{'─'*70}")

    src_depths = Counter(module_depth(src) for src, _ in edges)
    tgt_depths = Counter(module_depth(tgt) for _, tgt in edges)

    out(f"\n  {'Depth':<8} {'As source':>12} {'(%)':>8} {'As target':>12} {'(%)':>8}")
    out(f"  {'-'*48}")
    for d in range(1, max_d + 1):
        sc = src_depths.get(d, 0)
        tc = tgt_depths.get(d, 0)
        out(f"  {d:<8} {sc:>12,} {100*sc/n:>7.1f}% {tc:>12,} {100*tc/n:>7.1f}%")

    # ── Write output ──
    OUTPUT_DIR.mkdir(exist_ok=True)
    outpath = OUTPUT_DIR / "module_depth_analysis.txt"
    outpath.write_text("\n".join(lines) + "\n")
    out(f"\nOutput written to {outpath}")


if __name__ == "__main__":
    main()
