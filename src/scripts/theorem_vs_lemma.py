#!/usr/bin/env python3
"""
Theorem vs Lemma: Do human importance judgments align with logical structure?

In Mathlib source code, developers choose `theorem` or `lemma` to declare propositions.
Lean 4 treats them identically at compile time, so lean4export records both as "theorem".
We recover the distinction from source code and cross-reference with G_thm metrics.

Steps:
  1. Clone Mathlib source (shallow) and extract theorem/lemma declarations
  2. Load HuggingFace graph data, compute in-degree and PageRank
  3. Join on declaration name, compare statistics between groups
"""

import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
from datasets import load_dataset

MATHLIB_DIR = Path("/tmp/mathlib4_thm_lemma")
OUTPUT_DIR = Path("output")


def clone_mathlib():
    """Shallow-clone Mathlib4 if not already present."""
    if (MATHLIB_DIR / "Mathlib").is_dir():
        print(f"Mathlib source already at {MATHLIB_DIR}")
        return
    print("Cloning Mathlib4 (shallow)...")
    t0 = time.time()
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/leanprover-community/mathlib4.git",
         str(MATHLIB_DIR)],
        check=True,
        capture_output=True,
    )
    print(f"  Cloned in {time.time() - t0:.1f}s")


def extract_declarations():
    """
    Scan all .lean files under Mathlib/ for theorem/lemma declarations,
    tracking namespace context to reconstruct the full Lean name.

    Returns dict: short_name -> "theorem" | "lemma"
    Also returns dict: full_lean_name -> "theorem" | "lemma"
    """
    print("\nExtracting theorem/lemma declarations from source...")
    t0 = time.time()
    mathlib_src = MATHLIB_DIR / "Mathlib"

    # Patterns
    pat_decl = re.compile(r"^(theorem|lemma)\s+([A-Za-z_][\w'.]*)")
    pat_ns_open = re.compile(r"^namespace\s+(\S+)")
    pat_ns_close = re.compile(r"^end\s+(\S+)")
    # Also match bare `end` (closes the most recent namespace/section)
    pat_end_bare = re.compile(r"^end\s*$")
    pat_section_open = re.compile(r"^section\s*(.*)")
    # `variable`, `open`, `set_option` etc. don't create scope

    decl_short = {}   # short name -> kind
    decl_full = {}    # full Lean name -> kind
    file_count = 0
    collision_count = 0

    for lean_file in sorted(mathlib_src.rglob("*.lean")):
        file_count += 1
        ns_stack = []  # stack of (kind, name) where kind is "namespace" or "section"

        with open(lean_file, "r", errors="replace") as f:
            for line in f:
                stripped = line.strip()

                # Skip comments
                if stripped.startswith("--"):
                    continue

                # Track namespace open
                m_ns = pat_ns_open.match(stripped)
                if m_ns:
                    ns_stack.append(("namespace", m_ns.group(1)))
                    continue

                # Track section open
                m_sec = pat_section_open.match(stripped)
                if m_sec and not pat_decl.match(stripped):
                    # `section Foo` or bare `section`
                    ns_stack.append(("section", m_sec.group(1).strip()))
                    continue

                # Track `end Foo` (closes matching namespace or section)
                m_end = pat_ns_close.match(stripped)
                if m_end and not stripped.startswith("end_"):
                    end_name = m_end.group(1)
                    # Pop stack until we find matching name
                    for i in range(len(ns_stack) - 1, -1, -1):
                        if ns_stack[i][1] == end_name:
                            ns_stack.pop(i)
                            break
                    continue

                # Bare `end`
                if pat_end_bare.match(stripped):
                    if ns_stack:
                        ns_stack.pop()
                    continue

                # Match theorem/lemma declaration
                m_decl = pat_decl.match(stripped)
                if m_decl:
                    kind = m_decl.group(1)
                    raw_name = m_decl.group(2)

                    # Build namespace prefix from stack (only namespace, not section)
                    ns_parts = [name for (scope, name) in ns_stack
                                if scope == "namespace" and name]
                    prefix = ".".join(ns_parts)

                    # If the raw_name already contains dots, it may be partially
                    # or fully qualified. The full Lean name is prefix.raw_name
                    if prefix:
                        full_name = f"{prefix}.{raw_name}"
                    else:
                        full_name = raw_name

                    decl_full[full_name] = kind

                    # Short name = last component after dots
                    short_name = raw_name.rsplit(".", 1)[-1] if "." in raw_name else raw_name
                    if short_name in decl_short and decl_short[short_name] != kind:
                        collision_count += 1
                    decl_short[short_name] = kind

    elapsed = time.time() - t0
    n_thm = sum(1 for v in decl_full.values() if v == "theorem")
    n_lem = sum(1 for v in decl_full.values() if v == "lemma")
    print(f"  Scanned {file_count} files in {elapsed:.1f}s")
    print(f"  Found {len(decl_full)} declarations: {n_thm} theorem, {n_lem} lemma")
    print(f"  Short-name collisions (different kind): {collision_count}")

    return decl_short, decl_full


def load_graph():
    """Load HuggingFace data and build DiGraph. Return (G, nodes_df)."""
    print("\nLoading data from HuggingFace...")
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
    print(f"  Loaded in {time.time() - t0:.1f}s: {len(nodes_df)} nodes, {len(edges_df)} edges")

    # Filter to theorem-kind nodes only (the ones that could be theorem or lemma in source)
    thm_nodes = nodes_df[nodes_df["kind"] == "theorem"]
    print(f"  Theorem-kind nodes in graph: {len(thm_nodes)}")

    print("  Building DiGraph...")
    G = nx.DiGraph()
    for _, row in nodes_df.iterrows():
        G.add_node(row["name"], kind=row["kind"], module=row["module"])
    node_set = set(G.nodes)
    for _, row in edges_df.iterrows():
        if row["source"] in node_set and row["target"] in node_set:
            G.add_edge(row["source"], row["target"])
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G, nodes_df


def join_and_analyze(G, nodes_df, decl_short, decl_full):
    """Join source declarations with graph metrics and compare."""
    print("\nMatching declarations...")

    # Get theorem-kind nodes from graph
    thm_nodes = nodes_df[nodes_df["kind"] == "theorem"]["name"].tolist()

    # Try matching strategies:
    # 1. Exact match on full name
    # 2. Match on short name (last component)
    matched = {}  # graph_name -> "theorem" | "lemma"
    match_full = 0
    match_short = 0
    unmatched = 0

    for name in thm_nodes:
        if name in decl_full:
            matched[name] = decl_full[name]
            match_full += 1
        else:
            # Try short name: take last dotted component
            short = name.rsplit(".", 1)[-1] if "." in name else name
            if short in decl_short:
                matched[name] = decl_short[short]
                match_short += 1
            else:
                unmatched += 1

    total = len(thm_nodes)
    print(f"  Theorem-kind nodes in graph: {total}")
    print(f"  Matched by full name:  {match_full}")
    print(f"  Matched by short name: {match_short}")
    print(f"  Unmatched:             {unmatched}")
    print(f"  Match rate:            {len(matched)/total*100:.1f}%")

    if len(matched) / total < 0.3:
        print("\n  WARNING: Match rate very low. Trying alternative strategy...")
        # Try stripping common prefixes
        decl_full_stripped = {}
        for k, v in decl_full.items():
            # Source names start with Mathlib., graph names might not
            if k.startswith("Mathlib."):
                decl_full_stripped[k[len("Mathlib."):]] = v
            decl_full_stripped[k] = v

        matched = {}
        match_full = 0
        match_short = 0

        for name in thm_nodes:
            if name in decl_full_stripped:
                matched[name] = decl_full_stripped[name]
                match_full += 1
            else:
                short = name.rsplit(".", 1)[-1] if "." in name else name
                if short in decl_short:
                    matched[name] = decl_short[short]
                    match_short += 1

        unmatched = total - len(matched)
        print(f"  After stripping Mathlib. prefix:")
        print(f"  Matched by full name:  {match_full}")
        print(f"  Matched by short name: {match_short}")
        print(f"  Unmatched:             {unmatched}")
        print(f"  Match rate:            {len(matched)/total*100:.1f}%")

    # Compute in-degree for all nodes
    in_deg = dict(G.in_degree())

    # Compute PageRank
    print("\n  Computing PageRank...")
    t0 = time.time()
    pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    print(f"  PageRank computed in {time.time() - t0:.1f}s")

    # Split into groups
    thm_group = {n: matched[n] for n in matched if matched[n] == "theorem"}
    lem_group = {n: matched[n] for n in matched if matched[n] == "lemma"}

    thm_indeg = [in_deg.get(n, 0) for n in thm_group]
    lem_indeg = [in_deg.get(n, 0) for n in lem_group]
    thm_pr = [pr.get(n, 0) for n in thm_group]
    lem_pr = [pr.get(n, 0) for n in lem_group]

    # Build report
    lines = []
    lines.append("=" * 70)
    lines.append("THEOREM vs LEMMA: Human Importance Judgment vs Logical Structure")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total theorem-kind nodes in G_thm:  {total}")
    lines.append(f"Matched to source declarations:     {len(matched)} ({len(matched)/total*100:.1f}%)")
    lines.append(f"  - Classified as `theorem`:        {len(thm_group)}")
    lines.append(f"  - Classified as `lemma`:          {len(lem_group)}")
    lines.append("")

    def stats_block(label, indeg_vals, pr_vals):
        n = len(indeg_vals)
        if n == 0:
            return [f"  {label}: no data"]
        indeg_arr = np.array(indeg_vals)
        pr_arr = np.array(pr_vals)
        zero_rate = np.mean(indeg_arr == 0) * 100
        block = [
            f"  {label} (n = {n}):",
            f"    In-degree   mean={indeg_arr.mean():.2f}  median={np.median(indeg_arr):.1f}  max={indeg_arr.max()}",
            f"    Zero-citation rate:  {zero_rate:.1f}%",
            f"    PageRank    mean={pr_arr.mean():.2e}  median={np.median(pr_arr):.2e}  max={pr_arr.max():.2e}",
        ]
        return block

    lines.append("-" * 70)
    lines.append("COMPARISON")
    lines.append("-" * 70)
    lines.extend(stats_block("`theorem`", thm_indeg, thm_pr))
    lines.append("")
    lines.extend(stats_block("`lemma`  ", lem_indeg, lem_pr))
    lines.append("")

    # Ratios
    if thm_indeg and lem_indeg:
        thm_mean = np.mean(thm_indeg)
        lem_mean = np.mean(lem_indeg)
        ratio = thm_mean / lem_mean if lem_mean > 0 else float("inf")
        thm_zero = np.mean(np.array(thm_indeg) == 0) * 100
        lem_zero = np.mean(np.array(lem_indeg) == 0) * 100
        thm_pr_mean = np.mean(thm_pr)
        lem_pr_mean = np.mean(lem_pr)
        pr_ratio = thm_pr_mean / lem_pr_mean if lem_pr_mean > 0 else float("inf")

        lines.append("-" * 70)
        lines.append("VERDICT")
        lines.append("-" * 70)
        lines.append(f"  In-degree ratio (theorem/lemma):   {ratio:.2f}x")
        lines.append(f"  Zero-citation: theorem={thm_zero:.1f}%  lemma={lem_zero:.1f}%")
        lines.append(f"  PageRank ratio (theorem/lemma):    {pr_ratio:.2f}x")
        lines.append("")

        align_count = 0
        if thm_mean > lem_mean:
            lines.append("  [✓] theorem has higher mean in-degree than lemma")
            align_count += 1
        else:
            lines.append("  [✗] theorem does NOT have higher mean in-degree")

        if thm_zero < lem_zero:
            lines.append("  [✓] theorem has lower zero-citation rate than lemma")
            align_count += 1
        else:
            lines.append("  [✗] theorem does NOT have lower zero-citation rate")

        if thm_pr_mean > lem_pr_mean:
            lines.append("  [✓] theorem has higher mean PageRank than lemma")
            align_count += 1
        else:
            lines.append("  [✗] theorem does NOT have higher mean PageRank")

        lines.append("")
        if align_count == 3:
            lines.append("  CONCLUSION: All three predictions confirmed.")
            lines.append("  Human importance judgments align with logical structure.")
        elif align_count >= 1:
            lines.append(f"  CONCLUSION: {align_count}/3 predictions confirmed.")
            lines.append("  Partial alignment between human judgment and logical structure.")
        else:
            lines.append("  CONCLUSION: No predictions confirmed.")
            lines.append("  Human importance judgments diverge from logical structure.")

    # Interesting deviations: high-degree lemmas and zero-degree theorems
    lines.append("")
    lines.append("-" * 70)
    lines.append("NOTABLE DEVIATIONS")
    lines.append("-" * 70)

    # Top lemmas by in-degree (underestimated importance)
    lem_with_deg = [(n, in_deg.get(n, 0)) for n in lem_group]
    lem_with_deg.sort(key=lambda x: -x[1])
    lines.append("")
    lines.append("  Top 10 lemmas by in-degree (human labeled 'auxiliary' but heavily cited):")
    for name, deg in lem_with_deg[:10]:
        lines.append(f"    {name:60s}  in-degree={deg}")

    # Zero-degree theorems (overestimated importance)
    thm_zero_list = [n for n in thm_group if in_deg.get(n, 0) == 0]
    lines.append("")
    lines.append(f"  Zero-citation theorems (human labeled 'main result' but never cited): "
                 f"{len(thm_zero_list)}")
    if thm_zero_list:
        for name in sorted(thm_zero_list)[:10]:
            lines.append(f"    {name}")
        if len(thm_zero_list) > 10:
            lines.append(f"    ... and {len(thm_zero_list) - 10} more")

    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print("\n" + report)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "theorem_vs_lemma.txt"
    out_path.write_text(report + "\n")
    print(f"\nReport saved to {out_path}")


def main():
    clone_mathlib()
    decl_short, decl_full = extract_declarations()
    G, nodes_df = load_graph()
    join_and_analyze(G, nodes_df, decl_short, decl_full)


if __name__ == "__main__":
    main()
