#!/usr/bin/env python3
"""
Module-level containment decay analysis for G_import.

For each import edge (src_module, tgt_module), truncate both module names
to depth k (k = 1..5) and check whether the edge crosses the boundary.

This is the module-level analogue of the namespace containment decay in §4.

Output: terminal + output/module_containment_decay.txt
"""

import re
import time
from collections import Counter
from pathlib import Path

OUTPUT_DIR = Path("output")
MATHLIB_DIR = Path("/tmp/mathlib4_thm_lemma")


def truncate_name(name: str, depth: int) -> str:
    """Truncate a dot-separated module name to the first `depth` components."""
    parts = name.split(".")
    return ".".join(parts[:depth])


def module_depth(name: str) -> int:
    """Number of dot-separated components."""
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
    out("  MODULE-LEVEL CONTAINMENT DECAY (G_import)")
    out("=" * 70)

    # ── Extract G_import from Mathlib source ──
    out("\nExtracting G_import from Mathlib source ...")
    t0 = time.time()

    mathlib_src = MATHLIB_DIR / "Mathlib"
    if not mathlib_src.exists():
        out(f"  ERROR: {mathlib_src} not found.")
        return

    pat_import = re.compile(r"^(?:public\s+)?(?:meta\s+)?import\s+(\S+)")
    edges = []
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

    # ── Module-name depth distribution ──
    out(f"\n{'─'*70}")
    out("  1. MODULE-NAME DEPTH DISTRIBUTION")
    out(f"{'─'*70}")

    depth_counter = Counter()
    for mod in all_modules:
        d = module_depth(mod)
        depth_counter[d] += 1

    max_d = max(depth_counter.keys())
    out(f"\n  {'Depth':<10} {'Count':>8} {'Percentage':>12}")
    out(f"  {'-'*30}")
    for d in range(1, max_d + 1):
        c = depth_counter.get(d, 0)
        pct = 100.0 * c / len(all_modules)
        out(f"  {d:<10} {c:>8,} {pct:>11.1f}%")

    # ── Containment decay by depth ──
    out(f"\n{'─'*70}")
    out("  2. CONTAINMENT DECAY BY MODULE-NAME DEPTH")
    out(f"{'─'*70}")

    out(f"\n  {'Depth k':<10} {'Groups':>10} {'Intra':>10} {'Cross':>10} "
        f"{'Cross %':>10} {'Contain %':>12}")
    out(f"  {'-'*62}")

    for k in range(1, max_d + 1):
        groups = set()
        intra = 0
        cross = 0
        for src, tgt in edges:
            s_trunc = truncate_name(src, k)
            t_trunc = truncate_name(tgt, k)
            groups.add(s_trunc)
            groups.add(t_trunc)
            if s_trunc == t_trunc:
                intra += 1
            else:
                cross += 1
        total = intra + cross
        cross_pct = 100.0 * cross / total if total else 0
        contain_pct = 100.0 * intra / total if total else 0
        out(f"  {k:<10} {len(groups):>10,} {intra:>10,} {cross:>10,} "
            f"{cross_pct:>9.1f}% {contain_pct:>11.1f}%")

    # ── Full module-level (no truncation) ──
    out(f"\n  At full depth (no truncation):")
    intra_full = sum(1 for s, t in edges if s == t)
    cross_full = len(edges) - intra_full
    out(f"  Modules: {len(all_modules):,}  Intra: {intra_full:,}  "
        f"Cross: {cross_full:,}  Cross: {100*cross_full/len(edges):.1f}%")

    # ── Steepest drops ──
    out(f"\n{'─'*70}")
    out("  3. STEEPEST CONTAINMENT DROPS")
    out(f"{'─'*70}")

    prev_contain = None
    for k in range(1, max_d + 1):
        intra = sum(1 for s, t in edges
                    if truncate_name(s, k) == truncate_name(t, k))
        contain = 100.0 * intra / len(edges)
        if prev_contain is not None:
            drop = prev_contain - contain
            out(f"  Depth {k-1} → {k}: {prev_contain:.1f}% → {contain:.1f}%  "
                f"(drop {drop:+.1f}pp)")
        prev_contain = contain

    # ── Write output ──
    OUTPUT_DIR.mkdir(exist_ok=True)
    outpath = OUTPUT_DIR / "module_containment_decay.txt"
    outpath.write_text("\n".join(lines) + "\n")
    out(f"\nOutput written to {outpath}")


if __name__ == "__main__":
    main()
