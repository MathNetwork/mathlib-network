#!/usr/bin/env python3
"""Parameter sensitivity analysis for Louvain, betweenness, and PageRank."""

import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

ENRICHED = Path(__file__).resolve().parents[2] / "data" / "release" / "declaration" / "metrics.csv"
HF_DIR = Path.home() / ".cache/huggingface/hub/datasets--MathNetwork--MathlibGraph/snapshots/bc4173ec3beda64713ae81f602ce224491c61703"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def main():
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data ...")
    nodes = pd.read_csv(ENRICHED)
    edges = pd.read_csv(HF_DIR / "mathlib_edges.csv")
    edges = edges[edges["source"] != edges["target"]]
    node_set = set(nodes["name"].dropna())

    print("Building graph ...")
    t0 = time.time()
    G = nx.DiGraph()
    G.add_nodes_from(node_set)
    src, tgt = edges["source"].values, edges["target"].values
    for i in range(len(src)):
        s, t = src[i], tgt[i]
        if s != t and s in node_set and t in node_set:
            G.add_edge(s, t)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges ({time.time()-t0:.0f}s)")

    results = {}

    # ── Louvain resolution sensitivity ──
    print("\n=== Louvain Resolution Sensitivity ===")
    import community as community_louvain
    from sklearn.metrics import normalized_mutual_info_score

    G_und = G.to_undirected()
    node_idx = nodes.set_index("name")
    ns_labels_all = {n: n.split(".")[0] if "." in n else "_root_" for n in G_und.nodes()}

    louvain_results = []
    for res in [0.5, 0.8, 1.0, 1.5, 2.0]:
        t0 = time.time()
        partition = community_louvain.best_partition(G_und, resolution=res, random_state=42)
        mod = community_louvain.modularity(partition, G_und)
        n_comm = len(set(partition.values()))

        nodes_list = list(partition.keys())
        comm_labels = [partition[n] for n in nodes_list]
        ns_labels = [ns_labels_all.get(n, "_root_") for n in nodes_list]
        nmi = normalized_mutual_info_score(ns_labels, comm_labels)

        print(f"  res={res}: {n_comm} communities, modularity={mod:.4f}, NMI={nmi:.4f} ({time.time()-t0:.0f}s)")
        louvain_results.append({
            "resolution": res, "n_communities": n_comm,
            "modularity": round(mod, 4), "nmi": round(nmi, 4)
        })

    results["louvain"] = louvain_results

    # ── Betweenness k sensitivity ──
    print("\n=== Betweenness k Sensitivity ===")
    # Compute reference (k=500)
    bc_500 = nx.betweenness_centrality(G, k=500, seed=42)
    top10_500 = set(sorted(bc_500, key=bc_500.get, reverse=True)[:10])

    betw_results = []
    for k in [200, 500, 1000]:
        t0 = time.time()
        if k == 500:
            bc = bc_500
        else:
            bc = nx.betweenness_centrality(G, k=k, seed=42)
        top10 = set(sorted(bc, key=bc.get, reverse=True)[:10])
        jaccard = len(top10 & top10_500) / len(top10 | top10_500)
        print(f"  k={k}: top-10 Jaccard vs k=500: {jaccard:.3f} ({time.time()-t0:.0f}s)")
        betw_results.append({"k": k, "jaccard_vs_500": round(jaccard, 3),
                             "top10": sorted(top10)[:5]})

    results["betweenness"] = betw_results

    # ── PageRank α sensitivity ──
    print("\n=== PageRank α Sensitivity ===")
    pr_85 = nx.pagerank(G, alpha=0.85)
    top10_85 = set(sorted(pr_85, key=pr_85.get, reverse=True)[:10])

    pr_results = []
    for alpha in [0.80, 0.85, 0.90]:
        t0 = time.time()
        if alpha == 0.85:
            pr = pr_85
        else:
            pr = nx.pagerank(G, alpha=alpha)
        top10 = set(sorted(pr, key=pr.get, reverse=True)[:10])
        jaccard = len(top10 & top10_85) / len(top10 | top10_85)
        print(f"  α={alpha}: top-10 Jaccard vs α=0.85: {jaccard:.3f} ({time.time()-t0:.0f}s)")
        pr_results.append({"alpha": alpha, "jaccard_vs_085": round(jaccard, 3),
                           "top10": sorted(top10)[:5]})

    results["pagerank"] = pr_results

    # Save
    out_path = RESULTS_DIR / "sensitivity_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Total time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
