#!/usr/bin/env python3
"""Declaration-level hold-out experiment for premise retrieval.

Splits by declaration (same as original experiment), but recomputes all
graph features on G_train — the subgraph with test declarations' edges removed.

See docs/premise-retrieval-audit.md for the audit and design rationale.
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
# Helpers
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


# ═══════════════════════════════════════════════════════════════════
# Graph + feature computation
# ═══════════════════════════════════════════════════════════════════

def build_graph(node_set, edges_df):
    """Build NetworkX DiGraph from edges within node_set."""
    G = nx.DiGraph()
    G.add_nodes_from(node_set)
    src, tgt = edges_df["source"].values, edges_df["target"].values
    for i in range(len(src)):
        G.add_edge(src[i], tgt[i])
    return G


def compute_features(G, nodes_df):
    """Compute all graph features on G. Returns updated nodes_df copy."""
    import community as community_louvain

    df = nodes_df.copy()

    print("  Computing degree ...")
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    df["in_degree"] = df["name"].map(in_deg).fillna(0).astype(int)
    df["out_degree"] = df["name"].map(out_deg).fillna(0).astype(int)

    print("  Computing PageRank ...")
    t0 = time.time()
    pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    print(f"    ({time.time()-t0:.1f}s)")
    df["pagerank"] = df["name"].map(pr).fillna(0.0)

    print("  Computing betweenness (k=500) ...")
    t0 = time.time()
    bc = nx.betweenness_centrality(G, k=500, seed=42)
    print(f"    ({time.time()-t0:.1f}s)")
    df["betweenness"] = df["name"].map(bc).fillna(0.0)

    print("  Computing Louvain communities ...")
    t0 = time.time()
    G_und = G.to_undirected()
    partition = community_louvain.best_partition(G_und, random_state=42)
    n_comm = len(set(partition.values()))
    print(f"    {n_comm} communities ({time.time()-t0:.1f}s)")
    df["community_id"] = df["name"].map(partition).fillna(-1).astype(int)

    print("  Computing DAG layers ...")
    t0 = time.time()
    layer_map = compute_dag_layers(G)
    max_layer = max((v for v in layer_map.values() if v >= 0), default=0)
    print(f"    max depth {max_layer} ({time.time()-t0:.1f}s)")
    df["dag_layer"] = df["name"].map(layer_map).fillna(-1).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════════
# Feature extraction and retrieval
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
    """Build one retrieval problem: positives + negatives."""
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


def build_hard_problem(decl_name, premises_dict, node_idx, comm_members, rng, neg_n=50):
    """Build one retrieval problem with hard negatives (same community)."""
    true_prems = list(premises_dict.get(decl_name, set()))
    if len(true_prems) == 0:
        return None
    true_prems = [p for p in true_prems if p in node_idx.index]
    if len(true_prems) == 0:
        return None

    prem_set = set(true_prems)
    cid = node_idx.loc[decl_name, "community_id"]
    pool = [n for n in comm_members.get(cid, [])
            if n not in prem_set and n != decl_name]

    if len(pool) >= neg_n:
        neg_candidates = list(rng.choice(pool, size=neg_n, replace=False))
    else:
        neg_candidates = pool.copy()
        remaining = neg_n - len(neg_candidates)
        all_names = list(node_idx.index)
        extra = [n for n in rng.choice(all_names, size=remaining * 3, replace=False)
                 if n not in prem_set and n != decl_name and n not in set(neg_candidates)][:remaining]
        neg_candidates.extend(extra)

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
        "AUC": round(float(np.mean(aucs)), 4) if aucs else 0.0,
        "R@10": round(float(np.mean(r10s)), 4) if r10s else 0.0,
        "R@50": round(float(np.mean(r50s)), 4) if r50s else 0.0,
        "MRR": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
        "n_problems": len(aucs),
    }


# ═══════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════

def compute_diagnostics(nodes_full, nodes_train, test_decl_set):
    """Compare feature distributions between full graph and train graph."""
    from sklearn.metrics import normalized_mutual_info_score

    diag = {}
    idx_full = nodes_full.set_index("name")
    idx_train = nodes_train.set_index("name")

    # Only compare train declarations (test decls have zeroed features by design)
    train_names = idx_full.index.difference(pd.Index(test_decl_set))

    for col in ["in_degree", "out_degree", "pagerank", "betweenness"]:
        corr = np.corrcoef(idx_full.loc[train_names, col].values,
                           idx_train.loc[train_names, col].values)[0, 1]
        diag[f"{col}_pearson_r"] = round(float(corr), 6)

    # Community NMI (train declarations only)
    nmi = normalized_mutual_info_score(
        idx_full.loc[train_names, "community_id"].values,
        idx_train.loc[train_names, "community_id"].values
    )
    diag["community_nmi"] = round(float(nmi), 4)

    # DAG layer shift (train declarations only)
    layer_diff = np.abs(idx_full.loc[train_names, "dag_layer"].values -
                        idx_train.loc[train_names, "dag_layer"].values)
    diag["dag_layer_mean_shift"] = round(float(np.mean(layer_diff)), 4)

    # Test declaration stats on G_train
    test_names_in_idx = [n for n in test_decl_set if n in idx_train.index]
    diag["test_decl_mean_in_degree"] = round(float(
        idx_train.loc[test_names_in_idx, "in_degree"].mean()), 2)
    diag["test_decl_mean_out_degree"] = round(float(
        idx_train.loc[test_names_in_idx, "out_degree"].mean()), 2)
    diag["test_decl_mean_pagerank"] = float(
        idx_train.loc[test_names_in_idx, "pagerank"].mean())

    return diag


# ═══════════════════════════════════════════════════════════════════
# One split
# ═══════════════════════════════════════════════════════════════════

def run_one_split(nodes, edges, all_premises, seed, train_cap=20000, test_cap=5000):
    """Run one complete declaration-level hold-out experiment."""
    from sklearn.linear_model import LogisticRegression

    print(f"\n{'='*60}")
    print(f"SPLIT seed={seed}, declaration-level hold-out")
    print(f"{'='*60}")

    t_split_start = time.time()

    # 1. Find eligible theorems and split
    print("\n[1] Splitting declarations ...")
    theorem_names = set(nodes.loc[nodes["kind"] == "theorem", "name"])
    eligible = sorted([n for n in theorem_names
                       if n in all_premises and len(all_premises[n]) > 0])
    print(f"  Eligible theorems: {len(eligible):,}")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(eligible))
    n_train = int(0.8 * len(eligible))
    train_decls = set(eligible[i] for i in indices[:n_train])
    test_decls = set(eligible[i] for i in indices[n_train:])
    print(f"  Train declarations: {len(train_decls):,}")
    print(f"  Test declarations:  {len(test_decls):,}")

    # 2. Build G_train: remove ALL edges involving test declarations
    print("\n[2] Building G_train (removing test declaration edges) ...")
    t0 = time.time()
    edges_train = edges[~edges["source"].isin(test_decls) &
                        ~edges["target"].isin(test_decls)].copy()
    print(f"  Full graph edges:  {len(edges):,}")
    print(f"  G_train edges:     {len(edges_train):,}")
    print(f"  Removed:           {len(edges) - len(edges_train):,} ({100*(len(edges)-len(edges_train))/len(edges):.1f}%)")

    node_set = set(nodes["name"])
    G_train = build_graph(node_set, edges_train)
    print(f"  G_train: {G_train.number_of_nodes():,} nodes, {G_train.number_of_edges():,} edges ({time.time()-t0:.1f}s)")

    # WCC
    n_wcc = nx.number_weakly_connected_components(G_train)
    largest_wcc_size = max(len(c) for c in nx.weakly_connected_components(G_train))
    print(f"  WCC: {n_wcc:,} components, largest: {largest_wcc_size:,}")

    # 3. Recompute features on G_train
    print("\n[3] Recomputing features on G_train ...")
    t0 = time.time()
    nodes_train = compute_features(G_train, nodes)
    feat_time = time.time() - t0
    print(f"  Total feature computation: {feat_time:.1f}s")

    # 4. Diagnostics
    print("\n[4] Computing diagnostics ...")
    diag = compute_diagnostics(nodes, nodes_train, test_decls)
    diag["n_wcc"] = n_wcc
    diag["largest_wcc_size"] = largest_wcc_size
    diag["g_train_edges"] = len(edges_train)
    diag["edges_removed_pct"] = round(100 * (len(edges) - len(edges_train)) / len(edges), 2)
    for k, v in diag.items():
        print(f"  {k}: {v}")

    # 5. Prepare lookups
    node_idx = nodes_train.set_index("name")
    all_names = list(nodes_train["name"])

    feature_cols = ["same_community", "same_namespace", "dag_layer_diff",
                    "dag_direction", "is_cycle_node_c", "pagerank_c",
                    "pagerank_ratio", "in_degree_c", "out_degree_c",
                    "betweenness_c", "same_module"]
    net_cols = [c for c in feature_cols if c != "same_module"]

    # Build community -> members lookup (for hard negatives)
    comm_members = {}
    for name, row in node_idx.iterrows():
        cid = row["community_id"]
        if cid not in comm_members:
            comm_members[cid] = []
        comm_members[cid].append(name)

    # 6. Build train set
    train_list = sorted(train_decls)
    if train_cap and len(train_list) > train_cap:
        train_list = list(rng.choice(train_list, size=train_cap, replace=False))

    print(f"\n[5] Building train set ({len(train_list):,} declarations) ...")
    t0 = time.time()
    train_X_list, train_y_list = [], []
    for i, d in enumerate(train_list):
        if i % 1000 == 0 and i > 0:
            print(f"  {i:,} / {len(train_list):,} ...")
        result = build_retrieval_problem(d, all_premises, node_idx, all_names, rng)
        if result is None:
            continue
        _, labels, feats = result
        train_X_list.append(feats)
        train_y_list.append(labels)

    train_X = pd.concat(train_X_list, ignore_index=True).fillna(0).replace([np.inf, -np.inf], 0)
    train_y = np.concatenate(train_y_list)
    print(f"  Train pairs: {train_X.shape[0]:,} ({time.time()-t0:.1f}s)")

    # Also build hard-negative train set
    print(f"\n[6] Building hard-negative train set ({len(train_list):,} declarations) ...")
    t0 = time.time()
    hard_train_X_list, hard_train_y_list = [], []
    for i, d in enumerate(train_list):
        if i % 1000 == 0 and i > 0:
            print(f"  {i:,} / {len(train_list):,} ...")
        result = build_hard_problem(d, all_premises, node_idx, comm_members, rng)
        if result is None:
            continue
        _, labels, feats = result
        hard_train_X_list.append(feats)
        hard_train_y_list.append(labels)

    hard_train_X = pd.concat(hard_train_X_list, ignore_index=True).fillna(0).replace([np.inf, -np.inf], 0)
    hard_train_y = np.concatenate(hard_train_y_list)
    print(f"  Hard train pairs: {hard_train_X.shape[0]:,} ({time.time()-t0:.1f}s)")

    # 7. Train models
    print("\n[7] Training logistic regression ...")
    lr_net = LogisticRegression(max_iter=1000, random_state=42)
    lr_net.fit(train_X[net_cols], train_y)
    lr_all = LogisticRegression(max_iter=1000, random_state=42)
    lr_all.fit(train_X[feature_cols], train_y)

    lr_net_hard = LogisticRegression(max_iter=1000, random_state=42)
    lr_net_hard.fit(hard_train_X[net_cols], hard_train_y)
    lr_all_hard = LogisticRegression(max_iter=1000, random_state=42)
    lr_all_hard.fit(hard_train_X[feature_cols], hard_train_y)
    print("  Done.")

    # 8. Build test set (uniform negatives)
    test_list = sorted(test_decls)
    if test_cap and len(test_list) > test_cap:
        test_list = list(rng.choice(test_list, size=test_cap, replace=False))

    print(f"\n[8] Building test set ({len(test_list):,} declarations) ...")
    t0 = time.time()
    test_problems = []
    for i, d in enumerate(test_list):
        if i % 1000 == 0 and i > 0:
            print(f"  {i:,} / {len(test_list):,} ...")
        result = build_retrieval_problem(d, all_premises, node_idx, all_names, rng)
        if result is None:
            continue
        candidates, labels, feats = result
        feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
        test_problems.append((d, candidates, labels, feats))
    print(f"  Test problems: {len(test_problems):,} ({time.time()-t0:.1f}s)")

    # Build hard-negative test set
    print(f"\n[9] Building hard-negative test set ({len(test_list):,} declarations) ...")
    t0 = time.time()
    hard_test_problems = []
    for i, d in enumerate(test_list):
        if i % 1000 == 0 and i > 0:
            print(f"  {i:,} / {len(test_list):,} ...")
        result = build_hard_problem(d, all_premises, node_idx, comm_members, rng)
        if result is None:
            continue
        candidates, labels, feats = result
        feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
        hard_test_problems.append((d, candidates, labels, feats))
    print(f"  Hard test problems: {len(hard_test_problems):,} ({time.time()-t0:.1f}s)")

    # 9. Evaluate — uniform negatives
    print(f"\n[10] Evaluating (uniform negatives) ...")
    rng_eval = np.random.RandomState(seed)

    methods_uniform = {
        "Random": lambda f: rng_eval.rand(len(f)),
        "Same module": lambda f: f["same_module"].values.astype(float),
        "Same namespace": lambda f: f["same_namespace"].values.astype(float),
        "Same community": lambda f: f["same_community"].values.astype(float),
        "Network features": lambda f: lr_net.predict_proba(f[net_cols])[:, 1],
        "All features": lambda f: lr_all.predict_proba(f[feature_cols])[:, 1],
    }

    results_uniform = {}
    print(f"\n{'Method':<20s} {'AUC':>6s} {'R@10':>6s} {'R@50':>6s} {'MRR':>6s} {'n':>6s}")
    print("-" * 55)
    for name, scorer in methods_uniform.items():
        scored = [(labels, scorer(feats))
                  for d, candidates, labels, feats in test_problems]
        m = compute_metrics(scored)
        print(f"{name:<20s} {m['AUC']:>6.3f} {m['R@10']:>6.3f} {m['R@50']:>6.3f} {m['MRR']:>6.3f} {m['n_problems']:>6d}")
        results_uniform[name] = m

    # 10. Evaluate — hard negatives
    print(f"\n[11] Evaluating (hard negatives) ...")
    rng_eval_hard = np.random.RandomState(seed)

    methods_hard = {
        "Random": lambda f: rng_eval_hard.rand(len(f)),
        "Same module": lambda f: f["same_module"].values.astype(float),
        "Same namespace": lambda f: f["same_namespace"].values.astype(float),
        "Same community": lambda f: f["same_community"].values.astype(float),
        "Network features": lambda f: lr_net_hard.predict_proba(f[net_cols])[:, 1],
        "All features": lambda f: lr_all_hard.predict_proba(f[feature_cols])[:, 1],
    }

    results_hard = {}
    print(f"\n{'Method':<20s} {'AUC':>6s} {'R@10':>6s} {'R@50':>6s} {'MRR':>6s} {'n':>6s}")
    print("-" * 55)
    for name, scorer in methods_hard.items():
        scored = [(labels, scorer(feats))
                  for d, candidates, labels, feats in hard_test_problems]
        m = compute_metrics(scored)
        print(f"{name:<20s} {m['AUC']:>6.3f} {m['R@10']:>6.3f} {m['R@50']:>6.3f} {m['MRR']:>6.3f} {m['n_problems']:>6d}")
        results_hard[name] = m

    elapsed = time.time() - t_split_start
    print(f"\nSplit completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return {
        "seed": seed,
        "results_uniform": results_uniform,
        "results_hard": results_hard,
        "diagnostics": diag,
        "n_train_decls": len(train_decls),
        "n_test_decls": len(test_decls),
        "elapsed_s": round(elapsed, 1),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    nodes, edges = load_data()

    # Build premise lookup (from full graph — ground truth)
    print("\nBuilding premise lookup ...")
    all_premises = edges.groupby("source")["target"].apply(set).to_dict()
    print(f"  {len(all_premises):,} declarations with premises")

    # Load original results for comparison
    orig_path = RESULTS_DIR / "premise_retrieval_results.json"
    orig_hard_path = RESULTS_DIR / "premise_retrieval_hard_negatives.json"
    orig_results = None
    orig_hard = None
    if orig_path.exists():
        with open(orig_path) as f:
            orig_results = json.load(f)
    if orig_hard_path.exists():
        with open(orig_hard_path) as f:
            orig_hard = json.load(f)

    seeds = [42, 43, 44, 45, 46]
    all_splits = []
    partial_path = RESULTS_DIR / "holdout_decl_partial.json"

    for i, seed in enumerate(seeds):
        print(f"\n{'#'*60}")
        print(f"# SPLIT {i+1}/{len(seeds)}")
        print(f"{'#'*60}")

        split_result = run_one_split(nodes, edges, all_premises, seed)
        all_splits.append(split_result)

        # Incremental save
        with open(partial_path, "w") as f:
            json.dump({"completed_splits": all_splits}, f, indent=2)
        print(f"\n>>> Saved {len(all_splits)} split(s) to {partial_path}")

        if i == 0:
            est_total = split_result["elapsed_s"] * len(seeds)
            print(f">>> Estimated total time: {est_total/60:.0f} min ({est_total/3600:.1f} h)")

    # ── Aggregate ──
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")

    method_names = list(all_splits[0]["results_uniform"].keys())
    metrics = ["AUC", "R@10", "R@50", "MRR"]

    agg_uniform = {}
    agg_hard = {}
    for method in method_names:
        agg_uniform[method] = {}
        agg_hard[method] = {}
        for metric in metrics:
            vals_u = [s["results_uniform"][method][metric] for s in all_splits]
            vals_h = [s["results_hard"][method][metric] for s in all_splits]
            agg_uniform[method][metric] = {
                "mean": round(float(np.mean(vals_u)), 4),
                "std": round(float(np.std(vals_u)), 4),
                "values": [round(v, 4) for v in vals_u],
            }
            agg_hard[method][metric] = {
                "mean": round(float(np.mean(vals_h)), 4),
                "std": round(float(np.std(vals_h)), 4),
                "values": [round(v, 4) for v in vals_h],
            }

    # Print comparison table — uniform
    print(f"\n--- Uniform Negatives ---")
    print(f"{'Method':<20s} {'AUC(orig)':>10s} {'AUC(holdout)':>16s} {'Delta':>8s} {'R@10(holdout)':>16s}")
    print("-" * 75)
    for method in method_names:
        a = agg_uniform[method]["AUC"]
        r = agg_uniform[method]["R@10"]
        orig_auc = orig_results[method]["AUC"] if orig_results and method in orig_results else float("nan")
        delta = a["mean"] - orig_auc
        print(f"{method:<20s} {orig_auc:>10.3f} {a['mean']:>7.3f}±{a['std']:.3f} {delta:>+8.3f} {r['mean']:>7.3f}±{r['std']:.3f}")

    # Print comparison table — hard
    print(f"\n--- Hard Negatives ---")
    print(f"{'Method':<20s} {'AUC(orig)':>10s} {'AUC(holdout)':>16s} {'Delta':>8s} {'R@10(holdout)':>16s}")
    print("-" * 75)
    for method in method_names:
        a = agg_hard[method]["AUC"]
        r = agg_hard[method]["R@10"]
        orig_auc = orig_hard[method]["AUC"] if orig_hard and method in orig_hard else float("nan")
        delta = a["mean"] - orig_auc
        print(f"{method:<20s} {orig_auc:>10.3f} {a['mean']:>7.3f}±{a['std']:.3f} {delta:>+8.3f} {r['mean']:>7.3f}±{r['std']:.3f}")

    # Diagnostics summary
    print(f"\n--- Diagnostics (averaged over {len(seeds)} splits) ---")
    diag_keys = list(all_splits[0]["diagnostics"].keys())
    diag_agg = {}
    for k in diag_keys:
        vals = [s["diagnostics"][k] for s in all_splits]
        m, s = float(np.mean(vals)), float(np.std(vals))
        print(f"  {k}: mean={m:.4f}, std={s:.4f}")
        diag_agg[k] = {"mean": round(m, 4), "std": round(s, 4)}

    # Check std
    max_std = max(agg_uniform[m]["AUC"]["std"] for m in method_names if m != "Random")
    if max_std > 0.02:
        print(f"\n>>> WARNING: max AUC std = {max_std:.4f} > 0.02. Consider more splits.")

    # Save final results
    output = {
        "config": {
            "split_type": "declaration-level",
            "frac": 0.2,
            "seeds": seeds,
            "train_cap": 20000,
            "test_cap": 5000,
            "description": "Features recomputed on G_train with test declaration edges removed",
        },
        "per_split": all_splits,
        "aggregated_uniform": agg_uniform,
        "aggregated_hard": agg_hard,
        "diagnostics_aggregated": diag_agg,
        "original_results": orig_results,
        "original_hard_results": orig_hard,
    }

    out_path = RESULTS_DIR / "holdout_decl_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary CSV
    rows = []
    for method in method_names:
        row = {"Method": method}
        if orig_results and method in orig_results:
            row["AUC_orig_uniform"] = round(orig_results[method]["AUC"], 4)
        if orig_hard and method in orig_hard:
            row["AUC_orig_hard"] = round(orig_hard[method]["AUC"], 4)
        for metric in metrics:
            row[f"{metric}_holdout_uniform_mean"] = agg_uniform[method][metric]["mean"]
            row[f"{metric}_holdout_uniform_std"] = agg_uniform[method][metric]["std"]
            row[f"{metric}_holdout_hard_mean"] = agg_hard[method][metric]["mean"]
            row[f"{metric}_holdout_hard_std"] = agg_hard[method][metric]["std"]
        if "AUC_orig_uniform" in row:
            row["AUC_delta_uniform"] = round(agg_uniform[method]["AUC"]["mean"] - row["AUC_orig_uniform"], 4)
        if "AUC_orig_hard" in row:
            row["AUC_delta_hard"] = round(agg_hard[method]["AUC"]["mean"] - row["AUC_orig_hard"], 4)
        rows.append(row)

    csv_path = RESULTS_DIR / "holdout_decl_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
