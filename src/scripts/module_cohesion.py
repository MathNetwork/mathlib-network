"""Module cohesion analysis for the MathlibGraph dependency graph."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from main import load_data_from_huggingface


def main():
    print("Loading data from HuggingFace...")
    nodes_df, edges_df = load_data_from_huggingface()
    print(f"Loaded {len(nodes_df)} nodes, {len(edges_df)} edges")

    # Build node -> module lookup
    node_module = dict(zip(nodes_df["name"], nodes_df["module"]))

    # Per-module stats
    mod_decl_count = defaultdict(int)
    mod_internal = defaultdict(int)
    mod_external = defaultdict(int)

    for _, row in nodes_df.iterrows():
        mod_decl_count[row["module"]] += 1

    for _, row in edges_df.iterrows():
        src = row["source"]
        tgt = row["target"]
        src_mod = node_module.get(src)
        tgt_mod = node_module.get(tgt)
        if src_mod is None or tgt_mod is None:
            continue
        if src_mod == tgt_mod:
            mod_internal[src_mod] += 1
        else:
            mod_external[src_mod] += 1
            mod_external[tgt_mod] += 1

    # Compute cohesion per module
    cohesion = {}
    for mod in mod_decl_count:
        i = mod_internal[mod]
        e = mod_external[mod]
        total = i + e
        cohesion[mod] = i / total if total > 0 else 0.0

    # ---- Report ----
    import numpy as np

    vals = list(cohesion.values())
    vals_arr = np.array(vals)

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 70)
    out("MODULE COHESION ANALYSIS")
    out("=" * 70)
    out(f"Total modules: {len(cohesion)}")
    out()

    out("--- Global distribution ---")
    out(f"  Mean:   {vals_arr.mean():.4f}")
    out(f"  Median: {np.median(vals_arr):.4f}")
    out(f"  Std:    {vals_arr.std():.4f}")
    out(f"  Min:    {vals_arr.min():.4f}")
    out(f"  Max:    {vals_arr.max():.4f}")
    out()

    zero_count = sum(1 for v in vals if v == 0.0)
    out(f"Modules with cohesion=0: {zero_count} / {len(vals)} ({100*zero_count/len(vals):.1f}%)")
    out()

    # Top/bottom 20 with decl >= 10
    big = {m: c for m, c in cohesion.items() if mod_decl_count[m] >= 10}
    sorted_big = sorted(big.items(), key=lambda x: x[1], reverse=True)

    out("--- Top 20 highest cohesion (decl >= 10) ---")
    for m, c in sorted_big[:20]:
        out(f"  {c:.4f}  decl={mod_decl_count[m]:>5}  int={mod_internal[m]:>6}  ext={mod_external[m]:>6}  {m}")
    out()

    out("--- Top 20 lowest cohesion (decl >= 10) ---")
    for m, c in sorted_big[-20:]:
        out(f"  {c:.4f}  decl={mod_decl_count[m]:>5}  int={mod_internal[m]:>6}  ext={mod_external[m]:>6}  {m}")
    out()

    # By top-level namespace
    ns_cohesions = defaultdict(list)
    for m, c in cohesion.items():
        ms = str(m) if not isinstance(m, float) else ""
        ns = ms.split(".")[0] if ms else "(root)"
        ns_cohesions[ns].append(c)

    out("--- Average cohesion by top-level namespace ---")
    ns_sorted = sorted(ns_cohesions.items(), key=lambda x: np.mean(x[1]), reverse=True)
    for ns, cs in ns_sorted:
        out(f"  {np.mean(cs):.4f}  (n={len(cs):>5})  {ns}")
    out()

    os.makedirs("output", exist_ok=True)
    with open("output/module_cohesion.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved to output/module_cohesion.txt")


if __name__ == "__main__":
    main()
