#!/usr/bin/env python3
"""Generate community visualisation figures for all three graph levels.

Output (6 files in paper/figures/):
  community_module_graph.pdf   -- module-level community aggregation graph
  community_module_heatmap.pdf -- module communities x top-level directories
  community_decl_graph.pdf     -- declaration-level community aggregation graph
  community_decl_heatmap.pdf   -- declaration communities x depth-1 namespaces
  community_ns_graph.pdf       -- namespace-level community aggregation graph
  community_ns_heatmap.pdf     -- namespace communities x depth-1 prefixes
"""

import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from plot_style import setup_style, COLORS, FIGSIZE_SINGLE, FIGSIZE_DOUBLE, FIGSIZE_TRIPLE, FIGSIZE_HEATMAP, FIGSIZE_HEATMAP_WIDE, ANNOT_FS, LABEL_FS

COLORS = setup_style()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score

OUTDIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# -- Shared palette --
TAB20 = plt.cm.tab20.colors


# =====================================================================
#  Helper: community aggregation graph
# =====================================================================
def plot_community_graph(G_dir, partition, level_name, modularity,
                         label_fn, out_path, top_n=15, top_edges=30,
                         min_pct=0.01):
    """Draw a community-aggregated graph.

    Parameters
    ----------
    G_dir : nx.DiGraph  -- the original directed graph
    partition : dict     -- node -> community id
    level_name : str     -- "Module" / "Declaration" / "Namespace"
    modularity : float
    label_fn : callable  -- (community_id, member_list) -> display label
    out_path : Path
    top_n : int          -- keep only the largest communities (rest -> "Other")
    top_edges : int      -- draw only the heaviest inter-community edges
    min_pct : float      -- communities below this fraction merged into "Other"
    """
    total_nodes = len(partition)

    # Community sizes
    comm_sizes = Counter(partition.values())
    # Apply both top_n AND min_pct filters
    min_size = max(1, int(total_nodes * min_pct))
    top_comms = [c for c, cnt in comm_sizes.most_common(top_n)
                 if cnt >= min_size]
    top_set = set(top_comms)

    # Remap small communities -> -1 ("Other")
    remap = {c: c if c in top_set else -1 for c in comm_sizes}
    part2 = {n: remap[c] for n, c in partition.items()}
    sizes2 = Counter(part2.values())

    # Drop "Other" if it's empty
    show_other = sizes2.get(-1, 0) > 0

    # Collect members per community (for labelling)
    members = defaultdict(list)
    for n, c in part2.items():
        members[c].append(n)

    # Inter-community edge weights
    edge_w = Counter()
    for u, v in G_dir.edges():
        cu, cv = part2.get(u), part2.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        edge_w[(cu, cv)] += 1

    # Build meta-graph
    MG = nx.Graph()
    for c in sizes2:
        MG.add_node(c, size=sizes2[c])
    for (cu, cv), w in edge_w.items():
        if MG.has_edge(cu, cv):
            MG[cu][cv]["weight"] += w
        else:
            MG.add_edge(cu, cv, weight=w)

    # Keep only top edges
    all_edges = sorted(MG.edges(data=True), key=lambda e: e[2]["weight"], reverse=True)
    keep = set()
    for u, v, d in all_edges[:top_edges]:
        keep.add((u, v))
    remove = [(u, v) for u, v, _ in all_edges if (u, v) not in keep and (v, u) not in keep]
    MG.remove_edges_from(remove)

    # Layout -- use kamada_kawai for better separation, fall back to spring
    try:
        pos = nx.kamada_kawai_layout(MG)
    except Exception:
        pos = nx.spring_layout(MG, seed=42, k=3.5, iterations=120)

    # Post-process: push overlapping nodes apart (simple repulsion pass)
    pos_arr = np.array([pos[n] for n in MG.nodes()])
    node_list = list(MG.nodes())
    for _ in range(50):
        moved = False
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                diff = pos_arr[i] - pos_arr[j]
                dist = np.linalg.norm(diff)
                # Min distance proportional to sqrt of combined sizes
                si = MG.nodes[node_list[i]]["size"]
                sj = MG.nodes[node_list[j]]["size"]
                min_d = 0.15 * (np.sqrt(si) + np.sqrt(sj)) / np.sqrt(max(s for s in sizes2.values()))
                min_d = max(min_d, 0.12)
                if dist < min_d and dist > 0:
                    push = (min_d - dist) / 2 * diff / dist
                    pos_arr[i] += push
                    pos_arr[j] -= push
                    moved = True
        if not moved:
            break
    pos = {node_list[i]: pos_arr[i] for i in range(len(node_list))}

    # Draw
    fig, ax = plt.subplots(figsize=(10, 10))

    node_sizes_raw = np.array([MG.nodes[n]["size"] for n in node_list], dtype=float)
    node_sizes = np.sqrt(node_sizes_raw) / np.sqrt(node_sizes_raw.max()) * 2500 + 150

    # Assign stable colours
    color_map = {}
    ci = 0
    for c in sorted(top_comms):
        color_map[c] = TAB20[ci % len(TAB20)]
        ci += 1
    color_map[-1] = (0.75, 0.75, 0.75)  # grey for "Other"
    node_colors = [color_map.get(n, (0.5, 0.5, 0.5)) for n in node_list]

    # Edges
    edge_list = list(MG.edges(data=True))
    if edge_list:
        weights = np.array([d["weight"] for _, _, d in edge_list], dtype=float)
        edge_widths = np.log1p(weights)
        edge_widths = edge_widths / edge_widths.max() * 4 + 0.3
        nx.draw_networkx_edges(MG, pos, edgelist=[(u, v) for u, v, _ in edge_list],
                               width=edge_widths, alpha=0.3, edge_color=COLORS["grey"], ax=ax)

    nx.draw_networkx_nodes(MG, pos, nodelist=node_list, node_size=node_sizes,
                           node_color=node_colors, edgecolors="black",
                           linewidths=0.5, alpha=0.85, ax=ax)

    # Labels -- offset to avoid node overlap
    labels = {}
    for n in node_list:
        if n == -1:
            labels[n] = f"Other ({sizes2[n]:,})"
        else:
            labels[n] = label_fn(n, members[n])

    # Draw labels with small offset above each node
    label_pos = {}
    for n in node_list:
        sz = MG.nodes[n]["size"]
        offset = 0.02 + 0.01 * np.sqrt(sz) / np.sqrt(max(s for s in sizes2.values()))
        label_pos[n] = (pos[n][0], pos[n][1] + offset)

    nx.draw_networkx_labels(MG, label_pos, labels, font_size=LABEL_FS,
                            font_weight="bold", ax=ax)

    ax.set_title(f"{level_name}-level Louvain communities (modularity = {modularity:.2f})",
                 pad=12)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# =====================================================================
#  Helper: community x category heatmap
# =====================================================================
def plot_community_heatmap(partition, category_fn, level_name, nmi_value,
                           out_path, top_comm=10, top_cat=15,
                           row_label_fn=None, cmap="YlOrRd"):
    """Draw a community x category heatmap with diagonal-maximising sort.

    Parameters
    ----------
    partition : dict     -- node -> community id
    category_fn : callable -- node -> category string
    level_name : str
    nmi_value : float
    out_path : Path
    top_comm : int       -- number of communities (rows)
    top_cat : int        -- number of categories (columns)
    row_label_fn : callable -- (comm_id, size, dominant_cat) -> row label
    cmap : str           -- colormap name
    """
    # Build contingency
    comm_sizes = Counter(partition.values())
    top_comms = [c for c, _ in comm_sizes.most_common(top_comm)]

    cat_counts = Counter()
    contingency = defaultdict(Counter)  # comm -> {cat -> count}
    for n, c in partition.items():
        cat = category_fn(n)
        cat_counts[cat] += 1
        contingency[c][cat] += 1

    top_cats = [c for c, _ in cat_counts.most_common(top_cat)]

    # Build matrix
    matrix = np.zeros((len(top_comms), len(top_cats)), dtype=int)
    for i, comm in enumerate(top_comms):
        for j, cat in enumerate(top_cats):
            matrix[i, j] = contingency[comm].get(cat, 0)

    # Diagonal-maximising reorder: for each row, find the column with max value
    # then sort rows by their argmax column index
    row_argmax = np.argmax(matrix, axis=1)
    row_order = np.argsort(row_argmax)
    matrix = matrix[row_order]
    top_comms = [top_comms[i] for i in row_order]

    # Also reorder columns by first appearance in row order
    col_seen = []
    col_remaining = list(range(len(top_cats)))
    for i in range(len(top_comms)):
        best_col = np.argmax(matrix[i])
        if best_col in col_remaining:
            col_seen.append(best_col)
            col_remaining.remove(best_col)
    col_order = col_seen + col_remaining
    matrix = matrix[:, col_order]
    top_cats = [top_cats[i] for i in col_order]

    # Row labels
    row_labels = []
    for comm in top_comms:
        sz = comm_sizes[comm]
        dom = contingency[comm].most_common(1)[0][0] if contingency[comm] else "?"
        if row_label_fn:
            row_labels.append(row_label_fn(comm, sz, dom))
        else:
            row_labels.append(f"C{comm} ({dom}, {sz:,})")

    # Plot — full width
    fig, ax = plt.subplots(figsize=(12, 7))
    norm = mcolors.LogNorm(vmin=max(1, matrix[matrix > 0].min()),
                           vmax=matrix.max()) if matrix.max() > 0 else None
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(top_cats)))
    ax.set_xticklabels(top_cats, rotation=45, ha="right")
    ax.set_yticks(range(len(top_comms)))
    ax.set_yticklabels(row_labels)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if v > 0:
                color = "white" if v > matrix.max() * 0.4 else "black"
                ax.text(j, i, f"{v:,}", ha="center", va="center",
                        fontsize=ANNOT_FS, color=color)

    ax.set_title(f"{level_name} communities vs. categories (NMI = {nmi_value:.2f})",
                 pad=10)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Count")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# =====================================================================
#  1. Module Level
# =====================================================================
def do_module_level():
    print("=" * 60)
    print("1. MODULE LEVEL")
    print("=" * 60)
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
                was_in = in_block_comment > 0
                entered = False
                i = 0
                while i < len(stripped):
                    if i + 1 < len(stripped) and stripped[i:i+2] == "/-":
                        in_block_comment += 1; entered = True; i += 2
                    elif i + 1 < len(stripped) and stripped[i:i+2] == "-/":
                        in_block_comment = max(0, in_block_comment - 1); i += 2
                    else:
                        i += 1
                if in_block_comment > 0 or was_in or entered:
                    continue
                if not stripped or stripped.startswith("--"):
                    continue
                if stripped.startswith("module"):
                    continue
                m = re.match(r"^(?:public\s+)?(?:meta\s+)?import\s+([\w.]+)", stripped)
                if m:
                    imports.append(m.group(1))
                    continue
                break
        return imports

    print("  Building G_module ...")
    t0 = time.time()
    G = nx.DiGraph()
    file_modules = set()
    for f in sorted(MATHLIB_ROOT.rglob("*.lean")):
        mod = lean_path_to_module(f)
        file_modules.add(mod)
        G.add_node(mod)
    for f in sorted(MATHLIB_ROOT.rglob("*.lean")):
        mod = lean_path_to_module(f)
        for imp in parse_imports(f):
            if imp.startswith("Mathlib.") and imp in file_modules:
                G.add_edge(mod, imp)
    print(f"    {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
          f"({time.time()-t0:.1f}s)")

    # Louvain on undirected
    print("  Running Louvain ...")
    t0 = time.time()
    G_und = G.to_undirected()
    partition = community_louvain.best_partition(G_und, random_state=42)
    modularity = community_louvain.modularity(partition, G_und)
    n_comms = len(set(partition.values()))
    print(f"    {n_comms} communities, modularity = {modularity:.3f} ({time.time()-t0:.1f}s)")

    # Top-level directory for each module
    def top_dir(mod):
        parts = mod.split(".")
        # Mathlib.Algebra.Group.Basic -> Algebra
        return parts[1] if len(parts) > 1 else "_root_"

    # NMI
    nodes_list = list(partition.keys())
    comm_labels = [partition[n] for n in nodes_list]
    dir_labels = [top_dir(n) for n in nodes_list]
    nmi = normalized_mutual_info_score(dir_labels, comm_labels)
    print(f"    NMI vs top-level directory = {nmi:.3f}")

    # 1a: aggregation graph
    def label_fn(comm_id, members):
        dirs = Counter(top_dir(m) for m in members)
        dom = dirs.most_common(1)[0][0]
        return f"{dom}\n({len(members):,})"

    plot_community_graph(G, partition, "Module", modularity, label_fn,
                         OUTDIR / "community_module_graph.pdf",
                         top_n=15, top_edges=30)

    # 1b: heatmap — blue for module level
    plot_community_heatmap(
        partition, top_dir, "Module", nmi,
        OUTDIR / "community_module_heatmap.pdf",
        top_comm=10, top_cat=15,
        row_label_fn=lambda c, sz, dom: f"C{c} ({dom}, {sz:,})",
        cmap="Blues",
    )

    return partition, modularity, nmi


# =====================================================================
#  2. Declaration Level
# =====================================================================
def do_declaration_level():
    print("\n" + "=" * 60)
    print("2. DECLARATION LEVEL")
    print("=" * 60)
    from datasets import load_dataset

    print("  Loading HuggingFace data ...")
    t0 = time.time()
    nodes_ds = load_dataset("MathNetwork/MathlibGraph",
                            data_files="mathlib_nodes.csv", split="train")
    edges_ds = load_dataset("MathNetwork/MathlibGraph",
                            data_files="mathlib_edges.csv", split="train")
    node_names = set(nodes_ds.to_pandas()["name"].dropna())
    edges_df = edges_ds.to_pandas()
    print(f"    {len(node_names):,} decls, {len(edges_df):,} edges ({time.time()-t0:.1f}s)")

    print("  Building G_thm ...")
    t0 = time.time()
    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    for _, row in edges_df.iterrows():
        s, t = row["source"], row["target"]
        if s in node_names and t in node_names and s != t:
            G.add_edge(s, t)
    print(f"    {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
          f"({time.time()-t0:.1f}s)")

    # Louvain on undirected (may take a few minutes)
    print("  Running Louvain (this may take a few minutes) ...")
    t0 = time.time()
    G_und = G.to_undirected()
    partition = community_louvain.best_partition(G_und, random_state=42)
    modularity = community_louvain.modularity(partition, G_und)
    n_comms = len(set(partition.values()))
    print(f"    {n_comms} communities, modularity = {modularity:.3f} ({time.time()-t0:.1f}s)")

    # Depth-1 namespace
    def ns_depth1(name):
        parts = name.split(".")
        return parts[0] if parts else "_root_"

    # NMI
    nodes_list = list(partition.keys())
    comm_labels = [partition[n] for n in nodes_list]
    ns_labels = [ns_depth1(n) for n in nodes_list]
    nmi = normalized_mutual_info_score(ns_labels, comm_labels)
    print(f"    NMI vs depth-1 namespace = {nmi:.3f}")

    # 2a: aggregation graph
    def label_fn(comm_id, members):
        nss = Counter(ns_depth1(m) for m in members)
        dom = nss.most_common(1)[0][0]
        return f"{dom}\n({len(members):,})"

    plot_community_graph(G, partition, "Declaration", modularity, label_fn,
                         OUTDIR / "community_decl_graph.pdf",
                         top_n=15, top_edges=30)

    # 2b: heatmap — red for declaration level
    plot_community_heatmap(
        partition, ns_depth1, "Declaration", nmi,
        OUTDIR / "community_decl_heatmap.pdf",
        top_comm=10, top_cat=15,
        row_label_fn=lambda c, sz, dom: f"C{c} ({dom}, {sz:,})",
        cmap="Reds",
    )

    return partition, modularity, nmi


# =====================================================================
#  3. Namespace Level
# =====================================================================
def do_namespace_level():
    print("\n" + "=" * 60)
    print("3. NAMESPACE LEVEL")
    print("=" * 60)
    from datasets import load_dataset

    print("  Loading HuggingFace data ...")
    t0 = time.time()
    nodes_ds = load_dataset("MathNetwork/MathlibGraph",
                            data_files="mathlib_nodes.csv", split="train")
    edges_ds = load_dataset("MathNetwork/MathlibGraph",
                            data_files="mathlib_edges.csv", split="train")
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

    G = nx.DiGraph()
    G.add_nodes_from(all_ns)
    for (s, t), w in edge_weights.items():
        G.add_edge(s, t, weight=w)
    print(f"    {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
          f"({time.time()-t0:.1f}s)")

    # Louvain on undirected weighted
    print("  Running Louvain ...")
    t0 = time.time()
    G_und = nx.Graph()
    G_und.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        if G_und.has_edge(u, v):
            G_und[u][v]["weight"] += w
        else:
            G_und.add_edge(u, v, weight=w)
    partition = community_louvain.best_partition(G_und, random_state=42)
    modularity = community_louvain.modularity(partition, G_und)
    n_comms = len(set(partition.values()))
    print(f"    {n_comms} communities, modularity = {modularity:.3f} ({time.time()-t0:.1f}s)")

    # Depth-1 prefix of namespace
    def ns_prefix(ns):
        parts = ns.split(".")
        return parts[0] if parts else "_root_"

    # NMI
    nodes_list = list(partition.keys())
    comm_labels = [partition[n] for n in nodes_list]
    pfx_labels = [ns_prefix(n) for n in nodes_list]
    nmi = normalized_mutual_info_score(pfx_labels, comm_labels)
    print(f"    NMI vs depth-1 prefix = {nmi:.3f}")

    # 3a: aggregation graph
    def label_fn(comm_id, members):
        pfxs = Counter(ns_prefix(m) for m in members)
        dom = pfxs.most_common(1)[0][0]
        return f"{dom}\n({len(members):,})"

    plot_community_graph(G, partition, "Namespace", modularity, label_fn,
                         OUTDIR / "community_ns_graph.pdf",
                         top_n=10, top_edges=30, min_pct=0.005)

    # 3b: heatmap — green for namespace level
    plot_community_heatmap(
        partition, ns_prefix, "Namespace", nmi,
        OUTDIR / "community_ns_heatmap.pdf",
        top_comm=10, top_cat=15,
        row_label_fn=lambda c, sz, dom: f"C{c} ({dom}, {sz:,})",
        cmap="Greens",
    )

    return partition, modularity, nmi


# =====================================================================
#  Main
# =====================================================================
def main():
    t_total = time.time()

    mod_part, mod_q, mod_nmi = do_module_level()
    decl_part, decl_q, decl_nmi = do_declaration_level()
    ns_part, ns_q, ns_nmi = do_namespace_level()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Module:      {len(set(mod_part.values())):>3} communities, "
          f"Q = {mod_q:.3f}, NMI = {mod_nmi:.3f}")
    print(f"  Declaration: {len(set(decl_part.values())):>3} communities, "
          f"Q = {decl_q:.3f}, NMI = {decl_nmi:.3f}")
    print(f"  Namespace:   {len(set(ns_part.values())):>3} communities, "
          f"Q = {ns_q:.3f}, NMI = {ns_nmi:.3f}")
    print(f"\n  Total time: {time.time()-t_total:.0f}s")
    print("  Done.")


if __name__ == "__main__":
    main()
