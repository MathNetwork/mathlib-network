"""Build premise co-occurrence statistics for the MathlibGraph dependency graph."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter, defaultdict
from itertools import combinations
from main import load_data_from_huggingface

LIMIT = 50_000_000


def main():
    print("Loading data from HuggingFace...")
    nodes_df, edges_df = load_data_from_huggingface()
    print(f"Loaded {len(nodes_df)} nodes, {len(edges_df)} edges")

    # Group targets by source
    src_targets = defaultdict(list)
    for _, row in edges_df.iterrows():
        src_targets[row["source"]].append(row["target"])

    # Filter: only sources that are theorems
    thm_names = set(nodes_df.loc[nodes_df["kind"] == "theorem", "name"])

    cooccur = Counter()
    skipped = 0
    total_sources = 0

    for src, targets in src_targets.items():
        if src not in thm_names:
            continue
        total_sources += 1
        if len(targets) > 100:
            skipped += 1
            continue
        targets = sorted(set(targets))
        for a, b in combinations(targets, 2):
            pair = (a, b) if a < b else (b, a)
            cooccur[pair] += 1
            if len(cooccur) > LIMIT:
                print(f"STOPPED: Counter exceeded {LIMIT:,} entries.")
                print(f"  Processed {total_sources} sources so far.")
                return

    # ---- Report ----
    import numpy as np

    weights = list(cooccur.values())
    w_arr = np.array(weights) if weights else np.array([0])

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 70)
    out("PREMISE CO-OCCURRENCE STATISTICS")
    out("=" * 70)
    out(f"Theorem sources considered: {total_sources}")
    out(f"Skipped (>100 premises):    {skipped}")
    out(f"Unique co-occurrence pairs: {len(cooccur):,}")
    out()

    out("--- Weight distribution ---")
    out(f"  Mean:   {w_arr.mean():.4f}")
    out(f"  Median: {np.median(w_arr):.1f}")
    out(f"  Max:    {w_arr.max()}")
    out()

    for threshold in [2, 5, 10, 50, 100]:
        cnt = sum(1 for w in weights if w >= threshold)
        out(f"  Weight >= {threshold:>3}: {cnt:>10,} pairs")
    out()

    out("--- Top 20 co-occurrence pairs ---")
    for (a, b), w in cooccur.most_common(20):
        out(f"  {w:>6}  {a}  <->  {b}")
    out()

    os.makedirs("output", exist_ok=True)
    with open("output/cooccurrence_stats.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved to output/cooccurrence_stats.txt")


if __name__ == "__main__":
    main()
