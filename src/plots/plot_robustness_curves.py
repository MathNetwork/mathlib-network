#!/usr/bin/env python3
"""Generate robustness curve figures for module and namespace levels.

Output:
  paper/analysis/module_robustness_curve.pdf
  paper/analysis/ns_robustness_curve.pdf
"""

import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from plot_style import setup_style, COLORS, FIGSIZE_SINGLE, FIGSIZE_DOUBLE, FIGSIZE_TRIPLE

COLORS = setup_style()

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

OUTDIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


def robustness_curve(G, fractions, seed=42):
    """Compute GCC fraction under random and targeted (PageRank) removal."""
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    # PageRank for targeted removal
    pr = nx.pagerank(G, alpha=0.85)
    pr_sorted = sorted(pr, key=pr.get, reverse=True)

    # Random order
    rng = np.random.RandomState(seed)
    random_order = list(nodes)
    rng.shuffle(random_order)

    random_gcc = [1.0]
    targeted_gcc = [1.0]

    G_rand = G.copy()
    G_targ = G.copy()

    prev_frac = 0.0
    for frac in fractions:
        n_remove = int(round(frac * n))
        n_prev = int(round(prev_frac * n))
        to_remove = n_remove - n_prev

        if to_remove > 0:
            # Random
            remove_rand = random_order[n_prev:n_remove]
            G_rand.remove_nodes_from(remove_rand)

            # Targeted
            remove_targ = pr_sorted[n_prev:n_remove]
            G_targ.remove_nodes_from(remove_targ)

        # Measure GCC (weakly connected)
        if G_rand.number_of_nodes() > 0:
            gcc_rand = max(len(c) for c in nx.weakly_connected_components(G_rand)) / n
        else:
            gcc_rand = 0.0

        if G_targ.number_of_nodes() > 0:
            gcc_targ = max(len(c) for c in nx.weakly_connected_components(G_targ)) / n
        else:
            gcc_targ = 0.0

        random_gcc.append(gcc_rand)
        targeted_gcc.append(gcc_targ)
        prev_frac = frac

    return random_gcc, targeted_gcc


def plot_robustness(fractions, random_gcc, targeted_gcc, title, out_path, color):
    """Plot robustness curves using the level color."""
    x = [0.0] + list(fractions)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.array(x) * 100, random_gcc, "o-", color=color,
            markersize=5, linewidth=1.5, label="Random removal")
    ax.plot(np.array(x) * 100, targeted_gcc, "s--", color=color, alpha=0.5,
            markersize=5, linewidth=1.5, label="Targeted removal (by PageRank)")
    ax.set_xlabel("Fraction of nodes removed (%)")
    ax.set_ylabel("Largest WCC / Total nodes")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, 55, 5))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def main():
    fractions = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.15, 0.17, 0.20,
                 0.25, 0.30, 0.40, 0.50]

    # ================================================================
    # 1. Module-level robustness
    # ================================================================
    print("=== Module-level robustness ===")
    MATHLIB_ROOT = Path("mathlib4/Mathlib")

    def lean_path_to_module(path):
        rel = path.relative_to(MATHLIB_ROOT.parent)
        return str(rel).replace("/", ".").removesuffix(".lean")

    def parse_imports(path):
        imports = []
        in_block_comment = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                was_in_comment = in_block_comment > 0
                entered_comment = False
                i = 0
                while i < len(stripped):
                    if i + 1 < len(stripped) and stripped[i:i+2] == "/-":
                        in_block_comment += 1; entered_comment = True; i += 2
                    elif i + 1 < len(stripped) and stripped[i:i+2] == "-/":
                        in_block_comment = max(0, in_block_comment - 1); i += 2
                    else:
                        i += 1
                if in_block_comment > 0 or was_in_comment or entered_comment:
                    continue
                if not stripped or stripped.startswith("--"):
                    continue
                if stripped.startswith("module"):
                    continue
                m = re.match(
                    r"^(?:public\s+)?(?:meta\s+)?import\s+([\w.]+)", stripped
                )
                if m:
                    imports.append(m.group(1))
                    continue
                break
        return imports

    print("  Building module graph ...")
    t0 = time.time()
    G_mod = nx.DiGraph()
    file_modules = set()
    for lean_file in sorted(MATHLIB_ROOT.rglob("*.lean")):
        mod = lean_path_to_module(lean_file)
        file_modules.add(mod)
        G_mod.add_node(mod)
    for lean_file in sorted(MATHLIB_ROOT.rglob("*.lean")):
        mod = lean_path_to_module(lean_file)
        for imp in parse_imports(lean_file):
            if imp.startswith("Mathlib.") and imp in file_modules:
                G_mod.add_edge(mod, imp)
    print(f"    {G_mod.number_of_nodes()} nodes, {G_mod.number_of_edges()} edges "
          f"({time.time()-t0:.1f}s)")

    print("  Computing robustness curves ...")
    t0 = time.time()
    mod_rand, mod_targ = robustness_curve(G_mod, fractions)
    print(f"    ({time.time()-t0:.1f}s)")

    plot_robustness(
        fractions, mod_rand, mod_targ,
        r"Network robustness: $G_{\mathrm{module}}$",
        OUTDIR / "module_robustness_curve.pdf",
        color=COLORS["primary"],
    )

    # ================================================================
    # 2. Namespace-level robustness
    # ================================================================
    print("\n=== Namespace-level robustness ===")
    from datasets import load_dataset

    print("  Loading HuggingFace data ...")
    t0 = time.time()
    nodes_ds = load_dataset(
        "MathNetwork/MathlibGraph",
        data_files="mathlib_nodes.csv", split="train",
    )
    edges_ds = load_dataset(
        "MathNetwork/MathlibGraph",
        data_files="mathlib_edges.csv", split="train",
    )
    node_names = set(nodes_ds.to_pandas()["name"].dropna())
    edges_df = edges_ds.to_pandas()
    print(f"    ({time.time()-t0:.1f}s)")

    def ns_at_depth(name, k=2):
        parts = name.split(".")
        if len(parts) <= k:
            return ".".join(parts[:-1]) if len(parts) > 1 else "_root_"
        return ".".join(parts[:k])

    print("  Building G_ns^(2) ...")
    t0 = time.time()
    decl_to_ns = {name: ns_at_depth(name) for name in node_names}
    all_ns = set(decl_to_ns.values())

    edge_weights = Counter()
    for _, row in edges_df.iterrows():
        s, t = row["source"], row["target"]
        if s not in decl_to_ns or t not in decl_to_ns:
            continue
        ns_s, ns_t = decl_to_ns[s], decl_to_ns[t]
        if ns_s != ns_t:
            edge_weights[(ns_s, ns_t)] += 1

    G_ns = nx.DiGraph()
    G_ns.add_nodes_from(all_ns)
    for (s, t), w in edge_weights.items():
        G_ns.add_edge(s, t, weight=w)
    print(f"    {G_ns.number_of_nodes()} nodes, {G_ns.number_of_edges()} edges "
          f"({time.time()-t0:.1f}s)")

    print("  Computing robustness curves ...")
    t0 = time.time()
    ns_rand, ns_targ = robustness_curve(G_ns, fractions)
    print(f"    ({time.time()-t0:.1f}s)")

    plot_robustness(
        fractions, ns_rand, ns_targ,
        r"Network robustness: $G_{\mathrm{ns}}^{(2)}$",
        OUTDIR / "ns_robustness_curve.pdf",
        color=COLORS["tertiary"],
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
