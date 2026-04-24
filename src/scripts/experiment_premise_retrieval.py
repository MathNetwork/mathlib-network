#!/usr/bin/env python3
"""Network-Feature-Based Premise Retrieval experiment.

Validates that precomputed network features from the MathlibGraph dataset
provide useful signal for premise retrieval. See docs/experiment-premise-retrieval.md.
"""

import json
import time
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

# ── Paths ──
ENRICHED = Path(__file__).resolve().parents[2] / "data" / "release" / "declaration" / "metrics.csv"
HF_DIR = Path.home() / ".cache/huggingface/hub/datasets--MathNetwork--MathlibGraph/snapshots/bc4173ec3beda64713ae81f602ce224491c61703"
EDGES_PATH = HF_DIR / "mathlib_edges.csv"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def phase1():
    """Phase 1: Load data and report statistics."""
    print("=" * 60)
    print("PHASE 1: Data Loading and Validation")
    print("=" * 60)

    # Load enriched nodes
    t0 = time.time()
    print(f"\nLoading enriched nodes from {ENRICHED} ...")
    nodes = pd.read_csv(ENRICHED)
    print(f"  {len(nodes):,} rows, {len(nodes.columns)} columns ({time.time()-t0:.1f}s)")
    print(f"  Columns: {list(nodes.columns)}")

    # Load edges
    t0 = time.time()
    print(f"\nLoading edges from {EDGES_PATH} ...")
    edges = pd.read_csv(EDGES_PATH)
    print(f"  {len(edges):,} rows ({time.time()-t0:.1f}s)")

    # Filter self-loops
    self_loops = (edges["source"] == edges["target"]).sum()
    edges = edges[edges["source"] != edges["target"]].copy()
    print(f"  Self-loops removed: {self_loops:,}")
    print(f"  Edges after filter: {len(edges):,}")

    # Kind distribution
    print("\n--- Kind Distribution ---")
    kind_counts = nodes["kind"].value_counts()
    for kind, count in kind_counts.items():
        print(f"  {kind:<15s} {count:>8,} ({100*count/len(nodes):.1f}%)")

    # Theorem + definition counts (Lean4 doesn't have "lemma" — all are "theorem")
    theorem_kinds = ["theorem"]
    n_theorems = nodes[nodes["kind"].isin(theorem_kinds)].shape[0]
    print(f"\n  Theorems (proof declarations): {n_theorems:,}")

    # Build outgoing edge sets
    node_set = set(nodes["name"])
    edges_filtered = edges[edges["source"].isin(node_set) & edges["target"].isin(node_set)]
    print(f"\n  Edges within enriched node set: {len(edges_filtered):,}")

    out_degree = edges_filtered.groupby("source").size()
    theorems_with_edges = out_degree.reindex(
        nodes[nodes["kind"].isin(theorem_kinds)]["name"]
    ).dropna()

    print(f"\n--- Theorem Premise Statistics ---")
    print(f"  Theorems with outgoing edges: {len(theorems_with_edges):,} / {n_theorems:,}")
    print(f"  Mean premises per theorem:    {theorems_with_edges.mean():.1f}")
    print(f"  Median premises per theorem:  {theorems_with_edges.median():.0f}")
    print(f"  Max premises per theorem:     {theorems_with_edges.max():.0f}")
    print(f"  Min premises per theorem:     {theorems_with_edges.min():.0f}")

    # Verify enriched columns
    required = ["community_id", "pagerank", "dag_layer", "namespace_depth2",
                 "in_degree", "out_degree", "betweenness"]
    print(f"\n--- Enriched Column Check ---")
    for col in required:
        present = col in nodes.columns
        nulls = nodes[col].isna().sum() if present else "N/A"
        print(f"  {col:<20s} {'✓' if present else '✗'}  nulls={nulls}")

    # Additional useful columns
    optional = ["file_module", "is_tactic_proof", "is_instance", "kind"]
    print(f"\n--- Optional Columns ---")
    for col in optional:
        present = col in nodes.columns
        print(f"  {col:<20s} {'✓' if present else '✗'}")

    print(f"\n{'='*60}")
    print("Phase 1 COMPLETE — data loaded and validated.")
    print(f"{'='*60}")

    return nodes, edges_filtered


def phase2(nodes, edges, pilot_n=100):
    """Phase 2: Split, feature engineering, and pilot evaluation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 60)
    print("PHASE 2: Split, Features, and Pilot Run")
    print("=" * 60)

    # ── Step 1: Build premise lookup ──
    print("\nBuilding premise lookup ...")
    t0 = time.time()
    premises = edges.groupby("source")["target"].apply(set).to_dict()
    print(f"  {len(premises):,} declarations with premises ({time.time()-t0:.1f}s)")

    # Filter to theorems with premises
    theorem_names = set(nodes.loc[nodes["kind"] == "theorem", "name"])
    eligible = sorted([n for n in theorem_names if n in premises and len(premises[n]) > 0])
    print(f"  Eligible theorems (with premises): {len(eligible):,}")

    # ── Step 2: Train/Val/Test split ──
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(eligible))
    n_train = int(0.8 * len(eligible))
    n_val = int(0.1 * len(eligible))

    train_decls = [eligible[i] for i in indices[:n_train]]
    val_decls = [eligible[i] for i in indices[n_train:n_train + n_val]]
    test_decls = [eligible[i] for i in indices[n_train + n_val:]]

    print(f"\n--- Split ---")
    print(f"  Train: {len(train_decls):,} declarations")
    print(f"  Val:   {len(val_decls):,} declarations")
    print(f"  Test:  {len(test_decls):,} declarations")

    # ── Step 3: Build node feature lookup (vectorized) ──
    print("\nBuilding feature lookup ...")
    node_idx = nodes.set_index("name")
    # All declarations as candidate pool
    all_names = list(nodes["name"])

    def extract_features_batch(goal_name, candidate_names):
        """Extract features for all (goal, candidate) pairs."""
        g = node_idx.loc[goal_name]
        # Vectorized lookup for candidates
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

        # Baseline feature
        g_ns2 = g["namespace_depth2"]
        g_mod = g.get("file_module", None)
        if pd.notna(g_mod):
            feats["same_module"] = (cands["file_module"].values == g_mod).astype(int)
        else:
            feats["same_module"] = 0

        return feats

    def build_retrieval_problem(decl_name, neg_n=50):
        """Build one retrieval problem: positives + negatives."""
        true_prems = list(premises.get(decl_name, set()))
        if len(true_prems) == 0:
            return None

        # Filter to premises that exist in our node set
        true_prems = [p for p in true_prems if p in node_idx.index]
        if len(true_prems) == 0:
            return None

        # Sample negatives
        prem_set = set(true_prems)
        neg_candidates = [n for n in rng.choice(all_names, size=neg_n * 3, replace=False)
                          if n not in prem_set and n != decl_name][:neg_n]

        candidates = true_prems + neg_candidates
        labels = np.array([1] * len(true_prems) + [0] * len(neg_candidates))

        feats = extract_features_batch(decl_name, candidates)
        return candidates, labels, feats

    # ── Step 4: Pilot run on first 100 test declarations ──
    pilot_decls = test_decls[:pilot_n]
    print(f"\n--- Pilot Run ({pilot_n} declarations) ---")
    t0 = time.time()

    all_X = []
    all_y = []
    per_decl_results = []

    for i, d in enumerate(pilot_decls):
        result = build_retrieval_problem(d)
        if result is None:
            continue
        candidates, labels, feats = result
        all_X.append(feats)
        all_y.append(labels)
        per_decl_results.append((d, candidates, labels, feats))

    X_pilot = pd.concat(all_X, ignore_index=True)
    y_pilot = np.concatenate(all_y)
    print(f"  Built {len(per_decl_results)} retrieval problems ({time.time()-t0:.1f}s)")
    print(f"  Feature matrix: {X_pilot.shape}")
    print(f"  Positive samples: {y_pilot.sum():,} ({100*y_pilot.mean():.1f}%)")
    print(f"  Negative samples: {(1-y_pilot).sum():.0f} ({100*(1-y_pilot.mean()):.1f}%)")

    # Feature statistics
    print(f"\n--- Feature Statistics ---")
    print(f"  {'Feature':<20s} {'Mean':>8s} {'Std':>8s} {'NaN':>5s} {'Inf':>5s}")
    feature_cols = list(X_pilot.columns)
    for col in feature_cols:
        vals = X_pilot[col]
        n_nan = vals.isna().sum()
        n_inf = np.isinf(vals).sum() if vals.dtype != object else 0
        print(f"  {col:<20s} {vals.mean():>8.4f} {vals.std():>8.4f} {n_nan:>5d} {n_inf:>5d}")

    # Replace any NaN/Inf
    X_pilot = X_pilot.fillna(0)
    X_pilot = X_pilot.replace([np.inf, -np.inf], 0)

    # ── Step 5: Quick sanity check — train LR on pilot data ──
    print(f"\n--- Sanity Check (LR on pilot) ---")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_pilot, y_pilot)
    y_prob = lr.predict_proba(X_pilot)[:, 1]
    auc = roc_auc_score(y_pilot, y_prob)
    print(f"  Train AUC (on pilot, overfit expected): {auc:.4f}")

    if auc < 0.5:
        print("  ⚠ AUC < 0.5 — labels or features may be inverted!")
        return None

    # Per-feature AUC (univariate)
    print(f"\n--- Per-Feature AUC (univariate) ---")
    for col in feature_cols:
        try:
            feat_auc = roc_auc_score(y_pilot, X_pilot[col])
            print(f"  {col:<20s} AUC={feat_auc:.4f}")
        except ValueError:
            print(f"  {col:<20s} AUC=N/A (constant)")

    # LR feature weights
    print(f"\n--- LR Feature Weights ---")
    for col, w in sorted(zip(feature_cols, lr.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {col:<20s} {w:>+8.4f}")

    print(f"\n{'='*60}")
    print(f"Phase 2 COMPLETE — pilot validated, AUC={auc:.4f}")
    print(f"{'='*60}")

    return {
        "train_decls": train_decls,
        "val_decls": val_decls,
        "test_decls": test_decls,
        "premises": premises,
        "node_idx": node_idx,
        "all_names": all_names,
        "extract_features_batch": extract_features_batch,
        "build_retrieval_problem": build_retrieval_problem,
        "feature_cols": feature_cols,
    }


def phase3(nodes, edges, p2, test_n=None, bootstrap_n=100):
    """Phase 3: Full evaluation on test set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 60)
    print("PHASE 3: Full Run and Evaluation")
    print("=" * 60)

    train_decls = p2["train_decls"]
    test_decls = p2["test_decls"]
    premises = p2["premises"]
    node_idx = p2["node_idx"]
    build_rp = p2["build_retrieval_problem"]
    feature_cols = p2["feature_cols"]

    if test_n:
        test_decls = test_decls[:test_n]
        train_subset = train_decls[:test_n * 4]  # proportional
    else:
        train_subset = train_decls[:50000]  # cap train for speed

    # ── Build train set ──
    print(f"\nBuilding train set ({len(train_subset):,} declarations) ...")
    t0 = time.time()
    train_X_list, train_y_list = [], []
    for i, d in enumerate(train_subset):
        if i % 5000 == 0 and i > 0:
            print(f"  {i:,} / {len(train_subset):,} ...")
        result = build_rp(d)
        if result is None:
            continue
        _, labels, feats = result
        train_X_list.append(feats)
        train_y_list.append(labels)

    train_X = pd.concat(train_X_list, ignore_index=True).fillna(0).replace([np.inf, -np.inf], 0)
    train_y = np.concatenate(train_y_list)
    print(f"  Train set: {train_X.shape[0]:,} pairs ({time.time()-t0:.0f}s)")

    # ── Train models ──
    print("\nTraining logistic regression ...")
    # Network features only (1-9, no same_module)
    net_cols = [c for c in feature_cols if c != "same_module"]
    lr_net = LogisticRegression(max_iter=1000, random_state=42)
    lr_net.fit(train_X[net_cols], train_y)

    # All features (1-10)
    lr_all = LogisticRegression(max_iter=1000, random_state=42)
    lr_all.fit(train_X[feature_cols], train_y)
    print("  Done.")

    # ── Build test set ──
    print(f"\nBuilding test set ({len(test_decls):,} declarations) ...")
    t0 = time.time()
    test_problems = []
    for i, d in enumerate(test_decls):
        if i % 5000 == 0 and i > 0:
            print(f"  {i:,} / {len(test_decls):,} ...")
        result = build_rp(d)
        if result is None:
            continue
        candidates, labels, feats = result
        feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
        test_problems.append((d, candidates, labels, feats))
    print(f"  Test problems: {len(test_problems):,} ({time.time()-t0:.0f}s)")

    # ── Evaluation functions ──
    def compute_metrics(per_problem_scores):
        """Compute AUC, R@10, R@50, MRR from per-problem (labels, scores)."""
        aucs, r10s, r50s, mrrs = [], [], [], []
        for labels, scores in per_problem_scores:
            if labels.sum() == 0 or labels.sum() == len(labels):
                continue
            try:
                aucs.append(roc_auc_score(labels, scores))
            except ValueError:
                continue
            # Recall@k and MRR
            order = np.argsort(-scores)
            ranked_labels = labels[order]
            n_pos = labels.sum()
            r10s.append(ranked_labels[:10].sum() / n_pos)
            r50s.append(ranked_labels[:50].sum() / n_pos)
            # MRR: first positive
            pos_positions = np.where(ranked_labels == 1)[0]
            mrrs.append(1.0 / (pos_positions[0] + 1) if len(pos_positions) > 0 else 0.0)
        return np.mean(aucs), np.mean(r10s), np.mean(r50s), np.mean(mrrs)

    def score_method(test_problems, scorer_fn):
        """Score all test problems with a given method."""
        results = []
        for d, candidates, labels, feats in test_problems:
            scores = scorer_fn(feats)
            results.append((labels, scores))
        return results

    # ── Define methods ──
    rng = np.random.RandomState(42)

    methods = {
        "Random": lambda f: rng.rand(len(f)),
        "Same module": lambda f: f["same_module"].values.astype(float),
        "Same namespace": lambda f: f["same_namespace"].values.astype(float),
        "Same community": lambda f: f["same_community"].values.astype(float),
        "Network features": lambda f: lr_net.predict_proba(f[net_cols])[:, 1],
        "All features": lambda f: lr_all.predict_proba(f[feature_cols])[:, 1],
    }

    # ── Evaluate ──
    print("\n--- Results ---")
    print(f"{'Method':<20s} {'AUC':>6s} {'R@10':>6s} {'R@50':>6s} {'MRR':>6s}")
    print("-" * 50)

    all_results = {}
    for name, scorer in methods.items():
        scored = score_method(test_problems, scorer)
        auc, r10, r50, mrr = compute_metrics(scored)
        print(f"{name:<20s} {auc:>6.3f} {r10:>6.3f} {r50:>6.3f} {mrr:>6.3f}")
        all_results[name] = {"AUC": auc, "R@10": r10, "R@50": r50, "MRR": mrr}

    # ── Bootstrap confidence intervals ──
    print(f"\nBootstrapping ({bootstrap_n} iterations) ...")
    t0 = time.time()
    boot_results = {name: {"AUC": [], "R@10": [], "R@50": [], "MRR": []}
                    for name in methods}

    for b in range(bootstrap_n):
        idx = rng.choice(len(test_problems), size=len(test_problems), replace=True)
        boot_problems = [test_problems[i] for i in idx]
        for name, scorer in methods.items():
            scored = score_method(boot_problems, scorer)
            auc, r10, r50, mrr = compute_metrics(scored)
            boot_results[name]["AUC"].append(auc)
            boot_results[name]["R@10"].append(r10)
            boot_results[name]["R@50"].append(r50)
            boot_results[name]["MRR"].append(mrr)

    print(f"  ({time.time()-t0:.0f}s)")

    print(f"\n--- Results with 95% CI ---")
    print(f"{'Method':<20s} {'AUC':>14s} {'R@10':>14s} {'R@50':>14s} {'MRR':>14s}")
    print("-" * 80)
    for name in methods:
        parts = []
        for metric in ["AUC", "R@10", "R@50", "MRR"]:
            vals = boot_results[name][metric]
            m = np.mean(vals)
            lo, hi = np.percentile(vals, [2.5, 97.5])
            parts.append(f"{m:.3f}±{(hi-lo)/2:.3f}")
            all_results[name][f"{metric}_ci_lo"] = lo
            all_results[name][f"{metric}_ci_hi"] = hi
        print(f"{name:<20s} {parts[0]:>14s} {parts[1]:>14s} {parts[2]:>14s} {parts[3]:>14s}")

    # ── LR feature weights ──
    print(f"\n--- LR Feature Weights (all features model) ---")
    for col, w in sorted(zip(feature_cols, lr_all.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {col:<20s} {w:>+8.4f}")

    # ── Finding 3 analysis: infrastructure vs math content ──
    print(f"\n--- Finding 3 Analysis: Infrastructure vs Math Content ---")
    infra_kinds = {"inductive", "constructor"}
    infra_threshold = 10000

    for split_name, split_fn in [
        ("Infrastructure (in_deg>10K or infra kind)",
         lambda f, c: (f["in_degree_c"].values > infra_threshold) |
                       np.isin([node_idx.loc[cn, "kind"] if cn in node_idx.index else ""
                                for cn in c], list(infra_kinds))),
        ("Math content (rest)",
         lambda f, c: (f["in_degree_c"].values <= infra_threshold) &
                       ~np.isin([node_idx.loc[cn, "kind"] if cn in node_idx.index else ""
                                 for cn in c], list(infra_kinds))),
    ]:
        split_scored = []
        for d, candidates, labels, feats in test_problems:
            mask = split_fn(feats, candidates)
            if mask.sum() == 0 or labels[mask].sum() == 0 or labels[mask].sum() == mask.sum():
                continue
            scores = lr_all.predict_proba(feats[feature_cols])[:, 1]
            split_scored.append((labels[mask], scores[mask]))

        if split_scored:
            auc, r10, r50, mrr = compute_metrics(split_scored)
            print(f"  {split_name}: AUC={auc:.3f} R@10={r10:.3f} R@50={r50:.3f}")
        else:
            print(f"  {split_name}: insufficient data")

    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = RESULTS_DIR / "premise_retrieval_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {json_path}")

    # CSV table
    csv_path = RESULTS_DIR / "premise_retrieval_table.csv"
    rows = []
    for name in methods:
        r = all_results[name]
        rows.append({"Method": name, "AUC": r["AUC"], "R@10": r["R@10"],
                      "R@50": r["R@50"], "MRR": r["MRR"]})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

    print(f"\n{'='*60}")
    print("Phase 3 COMPLETE")
    print(f"{'='*60}")


def phase_hard_negatives(nodes, edges, p2, test_n=5000):
    """Hard negative experiment: negatives from same community."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 60)
    print("PHASE HARD NEGATIVES")
    print("=" * 60)

    train_decls = p2["train_decls"]
    test_decls = p2["test_decls"][:test_n]
    premises = p2["premises"]
    node_idx = p2["node_idx"]
    extract_features_batch = p2["extract_features_batch"]
    feature_cols = p2["feature_cols"]
    net_cols = [c for c in feature_cols if c != "same_module"]

    rng = np.random.RandomState(42)

    # Build community -> members lookup
    print("\nBuilding community lookup ...")
    comm_members = {}
    for name, row in node_idx.iterrows():
        cid = row["community_id"]
        if cid not in comm_members:
            comm_members[cid] = []
        comm_members[cid].append(name)
    print(f"  {len(comm_members)} communities")

    def build_hard_problem(decl_name, neg_n=50):
        """Negatives from same community as goal."""
        true_prems = list(premises.get(decl_name, set()))
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
            # Not enough in same community, pad with random
            neg_candidates = pool.copy()
            remaining = neg_n - len(neg_candidates)
            all_names = list(node_idx.index)
            extra = [n for n in rng.choice(all_names, size=remaining * 3, replace=False)
                     if n not in prem_set and n != decl_name and n not in set(neg_candidates)][:remaining]
            neg_candidates.extend(extra)

        candidates = true_prems + neg_candidates
        labels = np.array([1] * len(true_prems) + [0] * len(neg_candidates))
        feats = extract_features_batch(decl_name, candidates)
        return candidates, labels, feats

    # ── Build train set with hard negatives ──
    train_subset = train_decls[:20000]
    print(f"\nBuilding hard-negative train set ({len(train_subset):,} declarations) ...")
    t0 = time.time()
    train_X_list, train_y_list = [], []
    for i, d in enumerate(train_subset):
        if i % 5000 == 0 and i > 0:
            print(f"  {i:,} / {len(train_subset):,} ...")
        result = build_hard_problem(d)
        if result is None:
            continue
        _, labels, feats = result
        train_X_list.append(feats)
        train_y_list.append(labels)

    train_X = pd.concat(train_X_list, ignore_index=True).fillna(0).replace([np.inf, -np.inf], 0)
    train_y = np.concatenate(train_y_list)
    print(f"  Train set: {train_X.shape[0]:,} pairs ({time.time()-t0:.0f}s)")

    # ── Train models ──
    print("\nTraining logistic regression ...")
    lr_net = LogisticRegression(max_iter=1000, random_state=42)
    lr_net.fit(train_X[net_cols], train_y)
    lr_all = LogisticRegression(max_iter=1000, random_state=42)
    lr_all.fit(train_X[feature_cols], train_y)
    print("  Done.")

    # ── Build test set with hard negatives ──
    print(f"\nBuilding hard-negative test set ({len(test_decls):,} declarations) ...")
    t0 = time.time()
    test_problems = []
    for i, d in enumerate(test_decls):
        if i % 5000 == 0 and i > 0:
            print(f"  {i:,} / {len(test_decls):,} ...")
        result = build_hard_problem(d)
        if result is None:
            continue
        candidates, labels, feats = result
        feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
        test_problems.append((d, candidates, labels, feats))
    print(f"  Test problems: {len(test_problems):,} ({time.time()-t0:.0f}s)")

    # ── Evaluate ──
    def compute_metrics(per_problem_scores):
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
        return np.mean(aucs), np.mean(r10s), np.mean(r50s), np.mean(mrrs)

    methods = {
        "Random": lambda f: rng.rand(len(f)),
        "Same module": lambda f: f["same_module"].values.astype(float),
        "Same namespace": lambda f: f["same_namespace"].values.astype(float),
        "Same community": lambda f: f["same_community"].values.astype(float),
        "Network features": lambda f: lr_net.predict_proba(f[net_cols])[:, 1],
        "All features": lambda f: lr_all.predict_proba(f[feature_cols])[:, 1],
    }

    print("\n--- Hard Negative Results ---")
    print(f"{'Method':<20s} {'AUC':>6s} {'R@10':>6s} {'R@50':>6s} {'MRR':>6s}")
    print("-" * 50)

    hard_results = {}
    for name, scorer in methods.items():
        scored = []
        for d, candidates, labels, feats in test_problems:
            scores = scorer(feats)
            scored.append((labels, scores))
        auc, r10, r50, mrr = compute_metrics(scored)
        print(f"{name:<20s} {auc:>6.3f} {r10:>6.3f} {r50:>6.3f} {mrr:>6.3f}")
        hard_results[name] = {"AUC": auc, "R@10": r10, "R@50": r50, "MRR": mrr}

    # ── Load easy results and compare ──
    easy_path = RESULTS_DIR / "premise_retrieval_results.json"
    with open(easy_path) as f:
        easy_results = json.load(f)

    print("\n--- Comparison: Easy vs Hard Negatives ---")
    print(f"{'Method':<20s} {'AUC(easy)':>10s} {'AUC(hard)':>10s} {'R@10(easy)':>11s} {'R@10(hard)':>11s}")
    print("-" * 65)
    for name in methods:
        e = easy_results[name]
        h = hard_results[name]
        print(f"{name:<20s} {e['AUC']:>10.3f} {h['AUC']:>10.3f} {e['R@10']:>11.3f} {h['R@10']:>11.3f}")

    # ── LR weights ──
    print(f"\n--- LR Feature Weights (hard negatives, all features) ---")
    for col, w in sorted(zip(feature_cols, lr_all.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {col:<20s} {w:>+8.4f}")

    # ── Save ──
    hard_path = RESULTS_DIR / "premise_retrieval_hard_negatives.json"
    with open(hard_path, "w") as f:
        json.dump(hard_results, f, indent=2)
    print(f"\nSaved to {hard_path}")

    print(f"\n{'='*60}")
    print("HARD NEGATIVES COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    nodes, edges = phase1()
    p2 = phase2(nodes, edges, pilot_n=100)
    if p2:
        # phase3(nodes, edges, p2, test_n=5000, bootstrap_n=100)
        phase_hard_negatives(nodes, edges, p2, test_n=5000)
