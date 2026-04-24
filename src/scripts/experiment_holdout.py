#!/usr/bin/env python3
"""Hold-out experiment for premise retrieval.

Tests whether network-feature-based premise retrieval survives information
leakage removal: features are recomputed on G_train (80% of edges) and
evaluated on held-out edges E_test (20%).

See docs/premise-retrieval-audit.md §7 for the full design.
"""

import json
import time
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

# ── Paths ──
ENRICHED = Path(__file__).resolve().parents[2] / "data" / "release" / "declaration" / "metrics.csv"
HF_DIR = Path.home() / ".cache/huggingface/hub/datasets--MathNetwork--MathlibGraph/snapshots/bc4173ec3beda64713ae81f602ce224491c61703"
EDGES_PATH = HF_DIR / "mathlib_edges.csv"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


# ═══════════════════════════════════════════════════════════════════
# Helpers (same algorithms as enrich_nodes.py, self-contained)
# ═══════════════════════════════════════════════════════════════════

def compute_dag_layers(G):
    """Kahn's algorithm. Nodes in cycles get layer = -1."""
    in_deg = defaultdict(int)
    adj = defaultdict(list)
    for u, v in G.edges():
        adj[u].append(v)
        in_deg[v] += 1
    for n in G.nodes():
        if n not in in_deg:
            in_deg[n] = 0

    layer = {}
    queue = deque()
    for n in G.nodes():
        if in_deg[n] == 0:
            layer[n] = 0
            queue.append(n)

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            in_deg[v] -= 1
            new_layer = layer[u] + 1
            if v in layer:
                layer[v] = max(layer[v], new_layer)
            else:
                layer[v] = new_layer
            if in_deg[v] == 0:
                queue.append(v)

    for n in G.nodes():
        if n not in layer:
            layer[n] = -1
    return layer


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_data():
    """Load nodes and edges. Returns (nodes_df, edges_df) with self-loops removed."""
    print("Loading nodes ...")
    nodes = pd.read_csv(ENRICHED)
    print(f"  {len(nodes):,} rows")

    print("Loading edges ...")
    edges = pd.read_csv(EDGES_PATH)
    n_self = (edges["source"] == edges["target"]).sum()
    edges = edges[edges["source"] != edges["target"]].copy()
    node_set = set(nodes["name"])
    edges = edges[edges["source"].isin(node_set) & edges["target"].isin(node_set)].copy()
    print(f"  {len(edges):,} edges (removed {n_self:,} self-loops)")
    return nodes, edges


def build_graph(node_set, edges_df):
    """Build NetworkX DiGraph from edges within node_set."""
    G = nx.DiGraph()
    G.add_nodes_from(node_set)
    src, tgt = edges_df["source"].values, edges_df["target"].values
    for i in range(len(src)):
        s, t = src[i], tgt[i]
        if s in node_set and t in node_set:
            G.add_edge(s, t)
    return G


def compute_features(G, nodes_df):
    """Compute all graph features on G. Returns updated nodes_df copy."""
    import community as community_louvain

    df = nodes_df.copy()

    # Degree
    print("  Computing degree ...")
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    df["in_degree"] = df["name"].map(in_deg).fillna(0).astype(int)
    df["out_degree"] = df["name"].map(out_deg).fillna(0).astype(int)

    # PageRank
    print("  Computing PageRank ...")
    t0 = time.time()
    pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    print(f"    ({time.time()-t0:.1f}s)")
    df["pagerank"] = df["name"].map(pr).fillna(0.0)

    # Betweenness (k=500)
    print("  Computing betweenness (k=500) ...")
    t0 = time.time()
    bc = nx.betweenness_centrality(G, k=500, seed=42)
    print(f"    ({time.time()-t0:.1f}s)")
    df["betweenness"] = df["name"].map(bc).fillna(0.0)

    # Louvain
    print("  Computing Louvain communities ...")
    t0 = time.time()
    G_und = G.to_undirected()
    partition = community_louvain.best_partition(G_und, random_state=42)
    n_comm = len(set(partition.values()))
    print(f"    {n_comm} communities ({time.time()-t0:.1f}s)")
    df["community_id"] = df["name"].map(partition).fillna(-1).astype(int)

    # DAG layers
    print("  Computing DAG layers ...")
    t0 = time.time()
    layer_map = compute_dag_layers(G)
    max_layer = max((v for v in layer_map.values() if v >= 0), default=0)
    print(f"    max depth {max_layer} ({time.time()-t0:.1f}s)")
    df["dag_layer"] = df["name"].map(layer_map).fillna(-1).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════════
# Hold-out split
# ═══════════════════════════════════════════════════════════════════

def hold_out_edges(edges_df, frac, seed):
    """Split edges into train (1-frac) and test (frac)."""
    rng = np.random.RandomState(seed)
    n = len(edges_df)
    idx = rng.permutation(n)
    n_test = int(frac * n)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return edges_df.iloc[train_idx].copy(), edges_df.iloc[test_idx].copy()


# ═══════════════════════════════════════════════════════════════════
# Feature extraction and retrieval (same logic as original)
# ═══════════════════════════════════════════════════════════════════

def extract_features_batch(goal_name, candidate_names, node_idx):
    """Vectorized feature extraction for (goal, candidate) pairs."""
    g = node_idx.loc[goal_name]
    cands = node_idx.loc[candidate_names]

    feats = pd.DataFrame(index=range(len(candidate_names)))
    feats["same_community"] = (cands["community_id"].values == g["community_id"]).astype(int)
    feats["same_namespace"] = (cands["namespace_depth2"].values == g["namespace_depth2"]).astype(int)

    g_layer = g["dag_layer"]
    c_layers = cands["dag_layer"].values
    feats["dag_layer_diff"] = np.abs(g_layer - c_layers)
    feats["dag_direction"] = (g_layer > c_layers).astype(int)
    feats["is_cycle_node_c"] = (c_layers == -1).astype(int)

    feats["pagerank_c"] = cands["pagerank"].values
    g_pr = g["pagerank"] if g["pagerank"] > 0 else 1e-10
    feats["pagerank_ratio"] = cands["pagerank"].values / g_pr
    feats["in_degree_c"] = cands["in_degree"].values
    feats["out_degree_c"] = cands["out_degree"].values
    feats["betweenness_c"] = cands["betweenness"].values

    g_mod = g.get("file_module", None)
    if pd.notna(g_mod):
        feats["same_module"] = (cands["file_module"].values == g_mod).astype(int)
    else:
        feats["same_module"] = 0

    return feats


def build_retrieval_problem(decl_name, premises_dict, node_idx, all_names, rng, neg_n=50):
    """Build one retrieval problem: positives + negatives.
    Same negative sampling as original (exclude only current decl's premises)."""
    true_prems = list(premises_dict.get(decl_name, set()))
    if len(true_prems) == 0:
        return None
    true_prems = [p for p in true_prems if p in node_idx.index]
    if len(true_prems) == 0:
        return None

    prem_set = set(true_prems)
    neg_candidates = [n for n in rng.choice(all_names, size=neg_n * 3, replace=False)
                      if n not in prem_set and n != decl_name][:neg_n]

    candidates = true_prems + neg_candidates
    labels = np.array([1] * len(true_prems) + [0] * len(neg_candidates))
    feats = extract_features_batch(decl_name, candidates, node_idx)
    return candidates, labels, feats


def compute_metrics(per_problem_scores):
    """Compute AUC, R@10, R@50, MRR from per-problem (labels, scores)."""
    from sklearn.metrics import roc_auc_score

    aucs, r10s, r50s, mrrs = [], [], [], []
    for labels, scores in per_problem_scores:
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        try:
            aucs.append(roc_auc_score(labels, scores))
        except ValueError:
            continue
        order = np.argsort(-scores)
        ranked_labels = labels[order]
        n_pos = labels.sum()
        r10s.append(ranked_labels[:10].sum() / n_pos)
        r50s.append(ranked_labels[:50].sum() / n_pos)
        pos_positions = np.where(ranked_labels == 1)[0]
        mrrs.append(1.0 / (pos_positions[0] + 1) if len(pos_positions) > 0 else 0.0)
    return {
        "AUC": float(np.mean(aucs)) if aucs else 0.0,
        "R@10": float(np.mean(r10s)) if r10s else 0.0,
        "R@50": float(np.mean(r50s)) if r50s else 0.0,
        "MRR": float(np.mean(mrrs)) if mrrs else 0.0,
        "n_problems": len(aucs),
    }


# ═══════════════════════════════════════════════════════════════════
# Diagnostics: compare G vs G_train features
# ═══════════════════════════════════════════════════════════════════

def compute_diagnostics(nodes_full, nodes_train):
    """Compare feature distributions between full graph and train graph."""
    from sklearn.metrics import normalized_mutual_info_score

    diag = {}
    idx_full = nodes_full.set_index("name")
    idx_train = nodes_train.set_index("name")
    common = idx_full.index.intersection(idx_train.index)

    for col in ["in_degree", "out_degree", "pagerank", "betweenness"]:
        corr = np.corrcoef(idx_full.loc[common, col].values,
                           idx_train.loc[common, col].values)[0, 1]
        diag[f"{col}_pearson_r"] = round(float(corr), 6)

    # Community NMI
    nmi = normalized_mutual_info_score(
        idx_full.loc[common, "community_id"].values,
        idx_train.loc[common, "community_id"].values
    )
    diag["community_nmi"] = round(float(nmi), 4)

    # DAG layer shift
    layer_diff = np.abs(idx_full.loc[common, "dag_layer"].values -
                        idx_train.loc[common, "dag_layer"].values)
    diag["dag_layer_mean_shift"] = round(float(np.mean(layer_diff)), 4)
    diag["dag_layer_max_shift"] = int(np.max(layer_diff))

    # WCC on train graph (computed by caller, passed via nodes_train metadata)
    # — will be added by run_one_split

    return diag


# ═══════════════════════════════════════════════════════════════════
# Main: one split
# ═══════════════════════════════════════════════════════════════════

def run_one_split(nodes, edges, seed, frac=0.2, train_cap=50000, test_cap=5000):
    """Run one complete hold-out experiment."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    print(f"\n{'='*60}")
    print(f"SPLIT seed={seed}, hold-out={frac*100:.0f}%")
    print(f"{'='*60}")

    t_split_start = time.time()

    # 1. Hold out edges
    print("\n[1] Holding out edges ...")
    edges_train, edges_test = hold_out_edges(edges, frac, seed)
    print(f"  Train edges: {len(edges_train):,}")
    print(f"  Test edges:  {len(edges_test):,}")

    # 2. Build G_train
    print("\n[2] Building G_train ...")
    node_set = set(nodes["name"])
    t0 = time.time()
    G_train = build_graph(node_set, edges_train)
    print(f"  {G_train.number_of_nodes():,} nodes, {G_train.number_of_edges():,} edges ({time.time()-t0:.1f}s)")

    # WCC diagnostics
    n_wcc = nx.number_weakly_connected_components(G_train)
    largest_wcc_size = max(len(c) for c in nx.weakly_connected_components(G_train))
    print(f"  WCC: {n_wcc:,} components, largest: {largest_wcc_size:,}")

    # 3. Recompute features on G_train
    print("\n[3] Recomputing features on G_train ...")
    t0 = time.time()
    nodes_train = compute_features(G_train, nodes)
    print(f"  Total feature computation: {time.time()-t0:.1f}s")

    # 4. Diagnostics: G vs G_train feature stability
    print("\n[4] Computing diagnostics ...")
    diag = compute_diagnostics(nodes, nodes_train)
    diag["n_wcc"] = n_wcc
    diag["largest_wcc_size"] = largest_wcc_size
    for k, v in diag.items():
        print(f"  {k}: {v}")

    # 5. Build premise lookup from TEST edges
    print("\n[5] Building test premise lookup ...")
    test_premises = edges_test.groupby("source")["target"].apply(set).to_dict()

    # Filter to theorems with at least 1 held-out premise
    theorem_names = set(nodes.loc[nodes["kind"] == "theorem", "name"])
    eligible_test = sorted([n for n in theorem_names
                            if n in test_premises and len(test_premises[n]) > 0])
    print(f"  Eligible test declarations (with held-out premises): {len(eligible_test):,}")

    if test_cap and len(eligible_test) > test_cap:
        rng_split = np.random.RandomState(seed)
        eligible_test = list(rng_split.choice(eligible_test, size=test_cap, replace=False))
        print(f"  Capped to {test_cap:,}")

    # 6. Build train premise lookup from TRAIN edges (for training LR)
    train_premises = edges_train.groupby("source")["target"].apply(set).to_dict()
    eligible_train = sorted([n for n in theorem_names
                             if n in train_premises and len(train_premises[n]) > 0])
    print(f"  Eligible train declarations: {len(eligible_train):,}")

    if train_cap and len(eligible_train) > train_cap:
        rng_split = np.random.RandomState(seed)
        eligible_train = list(rng_split.choice(eligible_train, size=train_cap, replace=False))

    # 7. Prepare feature index and candidate pool
    node_idx = nodes_train.set_index("name")
    all_names = list(nodes_train["name"])
    rng = np.random.RandomState(seed)

    feature_cols = ["same_community", "same_namespace", "dag_layer_diff",
                    "dag_direction", "is_cycle_node_c", "pagerank_c",
                    "pagerank_ratio", "in_degree_c", "out_degree_c",
                    "betweenness_c", "same_module"]
    net_cols = [c for c in feature_cols if c != "same_module"]

    # 8. Build train set
    print(f"\n[6] Building train set ({len(eligible_train):,} declarations) ...")
    t0 = time.time()
    train_X_list, train_y_list = [], []
    for i, d in enumerate(eligible_train):
        if i % 10000 == 0 and i > 0:
            print(f"  {i:,} / {len(eligible_train):,} ...")
        result = build_retrieval_problem(d, train_premises, node_idx, all_names, rng)
        if result is None:
            continue
        _, labels, feats = result
        train_X_list.append(feats)
        train_y_list.append(labels)

    train_X = pd.concat(train_X_list, ignore_index=True).fillna(0).replace([np.inf, -np.inf], 0)
    train_y = np.concatenate(train_y_list)
    print(f"  Train pairs: {train_X.shape[0]:,} ({time.time()-t0:.1f}s)")

    # 9. Train models
    print("\n[7] Training logistic regression ...")
    lr_net = LogisticRegression(max_iter=1000, random_state=42)
    lr_net.fit(train_X[net_cols], train_y)
    lr_all = LogisticRegression(max_iter=1000, random_state=42)
    lr_all.fit(train_X[feature_cols], train_y)

    # 10. Build test set
    print(f"\n[8] Building test set ({len(eligible_test):,} declarations) ...")
    t0 = time.time()
    test_problems = []
    for i, d in enumerate(eligible_test):
        if i % 5000 == 0 and i > 0:
            print(f"  {i:,} / {len(eligible_test):,} ...")
        result = build_retrieval_problem(d, test_premises, node_idx, all_names, rng)
        if result is None:
            continue
        candidates, labels, feats = result
        feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
        test_problems.append((d, candidates, labels, feats))
    print(f"  Test problems: {len(test_problems):,} ({time.time()-t0:.1f}s)")

    # 11. Evaluate
    print(f"\n[9] Evaluating ...")
    rng_eval = np.random.RandomState(seed)

    methods = {
        "Random": lambda f: rng_eval.rand(len(f)),
        "Same module": lambda f: f["same_module"].values.astype(float),
        "Same namespace": lambda f: f["same_namespace"].values.astype(float),
        "Same community": lambda f: f["same_community"].values.astype(float),
        "Network features": lambda f: lr_net.predict_proba(f[net_cols])[:, 1],
        "All features": lambda f: lr_all.predict_proba(f[feature_cols])[:, 1],
    }

    results = {}
    print(f"\n{'Method':<20s} {'AUC':>6s} {'R@10':>6s} {'R@50':>6s} {'MRR':>6s} {'n':>6s}")
    print("-" * 55)
    for name, scorer in methods.items():
        scored = []
        for d, candidates, labels, feats in test_problems:
            scores = scorer(feats)
            scored.append((labels, scores))
        m = compute_metrics(scored)
        print(f"{name:<20s} {m['AUC']:>6.3f} {m['R@10']:>6.3f} {m['R@50']:>6.3f} {m['MRR']:>6.3f} {m['n_problems']:>6d}")
        results[name] = m

    elapsed = time.time() - t_split_start
    print(f"\nSplit completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return {"results": results, "diagnostics": diag, "elapsed_s": round(elapsed, 1)}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data once
    nodes, edges = load_data()

    # Load original results for comparison
    orig_path = RESULTS_DIR / "premise_retrieval_results.json"
    if orig_path.exists():
        with open(orig_path) as f:
            orig_results = json.load(f)
    else:
        orig_results = None
        print("WARNING: No original results found for comparison.")

    # Run splits
    seeds = [42, 43, 44, 45, 46]
    all_splits = []

    for i, seed in enumerate(seeds):
        print(f"\n{'#'*60}")
        print(f"# SPLIT {i+1}/{len(seeds)}")
        print(f"{'#'*60}")
        split_result = run_one_split(nodes, edges, seed)
        split_result["seed"] = seed
        all_splits.append(split_result)

        # Incremental save after each split
        partial_path = RESULTS_DIR / "holdout_partial.json"
        with open(partial_path, "w") as f:
            json.dump({"completed_splits": all_splits}, f, indent=2)
        print(f"\n>>> Saved {len(all_splits)} split(s) to {partial_path}")

        # After first split, print timing estimate
        if i == 0:
            est_total = split_result["elapsed_s"] * len(seeds)
            print(f"\n>>> Estimated total time: {est_total/60:.0f} min ({est_total/3600:.1f} h)")

    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")

    method_names = list(all_splits[0]["results"].keys())
    metrics = ["AUC", "R@10", "R@50", "MRR"]

    agg = {}
    for method in method_names:
        agg[method] = {}
        for metric in metrics:
            vals = [s["results"][method][metric] for s in all_splits]
            agg[method][metric] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "values": [round(v, 4) for v in vals],
            }

    # Print comparison table
    print(f"\n{'Method':<20s} {'AUC(orig)':>10s} {'AUC(holdout)':>16s} {'Delta':>8s} {'R@10(holdout)':>16s}")
    print("-" * 75)
    for method in method_names:
        a = agg[method]["AUC"]
        r = agg[method]["R@10"]
        orig_auc = orig_results[method]["AUC"] if orig_results and method in orig_results else float("nan")
        delta = a["mean"] - orig_auc
        print(f"{method:<20s} {orig_auc:>10.3f} {a['mean']:>7.3f}±{a['std']:.3f} {delta:>+8.3f} {r['mean']:>7.3f}±{r['std']:.3f}")

    # Check if more splits needed (std > 0.02)
    max_std = max(agg[m]["AUC"]["std"] for m in method_names if m != "Random")
    if max_std > 0.02:
        print(f"\n>>> WARNING: max AUC std = {max_std:.4f} > 0.02. Consider running more splits.")

    # Aggregate diagnostics
    print(f"\n--- Diagnostics (averaged over {len(seeds)} splits) ---")
    diag_keys = list(all_splits[0]["diagnostics"].keys())
    for k in diag_keys:
        vals = [s["diagnostics"][k] for s in all_splits]
        print(f"  {k}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

    # Save everything
    output = {
        "config": {
            "frac": 0.2,
            "seeds": seeds,
            "train_cap": 50000,
            "test_cap": 5000,
        },
        "per_split": all_splits,
        "aggregated": agg,
        "original_results": orig_results,
    }

    out_path = RESULTS_DIR / "holdout_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Save summary CSV
    rows = []
    for method in method_names:
        row = {"Method": method}
        if orig_results and method in orig_results:
            row["AUC_original"] = round(orig_results[method]["AUC"], 4)
        for metric in metrics:
            row[f"{metric}_holdout_mean"] = agg[method][metric]["mean"]
            row[f"{metric}_holdout_std"] = agg[method][metric]["std"]
        if "AUC_original" in row:
            row["AUC_delta"] = round(agg[method]["AUC"]["mean"] - row["AUC_original"], 4)
        rows.append(row)
    csv_path = RESULTS_DIR / "holdout_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
