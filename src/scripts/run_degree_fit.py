#!/usr/bin/env python3
"""
Run degree distribution power law fitting.
Uses the powerlaw library for rigorous statistical testing.
Outputs plots and fit results to output/.
"""

import time
from pathlib import Path
from collections import Counter

import numpy as np
import networkx as nx
import pandas as pd
import powerlaw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset


OUTPUT_DIR = Path(__file__).parent.parent / "output"

INDIGO = "#2C3E6B"
GOLD = "#C9A96E"
IVORY = "#F5F0E8"
TEAL = "#3A7D7B"
CORAL = "#C75B4A"


def load_and_build():
    print("Loading data from HuggingFace...")
    t0 = time.time()
    nodes_ds = load_dataset(
        "MathNetwork/MathlibGraph",
        data_files="mathlib_nodes.csv",
        split="train",
    )
    edges_ds = load_dataset(
        "MathNetwork/MathlibGraph",
        data_files="mathlib_edges.csv",
        split="train",
    )
    nodes_df = nodes_ds.to_pandas()
    edges_df = edges_ds.to_pandas()
    print(f"  Downloaded in {time.time() - t0:.1f}s")

    print("Building graph...")
    t0 = time.time()
    G = nx.DiGraph()
    for _, row in nodes_df.iterrows():
        G.add_node(row["name"], kind=row["kind"], module=row["module"])
    node_set = set(G.nodes)
    for _, row in edges_df.iterrows():
        if row["source"] in node_set and row["target"] in node_set:
            G.add_edge(row["source"], row["target"])
    print(f"  Built in {time.time() - t0:.1f}s  ({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)")
    return G


def fit_and_report(degrees, label):
    """Fit power law and compare with alternatives."""
    print(f"\n  {label} (n={len(degrees):,} non-zero values)")
    print(f"  {'─' * 50}")

    t0 = time.time()
    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
    print(f"  Fit computed in {time.time() - t0:.1f}s")

    print(f"\n  Power law fit:")
    print(f"    alpha (exponent) = {fit.alpha:.4f}")
    print(f"    xmin             = {fit.xmin}")
    print(f"    sigma (std err)  = {fit.sigma:.4f}")
    n_tail = sum(1 for d in degrees if d >= fit.xmin)
    print(f"    n_tail (d>=xmin) = {n_tail:,} ({100*n_tail/len(degrees):.1f}%)")

    # Compare with alternative distributions
    # R > 0 means power law is better; R < 0 means alternative is better
    # p < 0.05 means the comparison is statistically significant
    comparisons = [
        ("lognormal", "Lognormal"),
        ("exponential", "Exponential"),
        ("stretched_exponential", "Stretched Exp"),
        ("truncated_power_law", "Truncated PL"),
    ]

    print(f"\n  Distribution comparisons (R>0 => power law better):")
    print(f"    {'Alternative':20s} {'R':>8s} {'p-value':>8s}  Verdict")
    print(f"    {'─'*60}")

    comp_results = {}
    for dist_name, display_name in comparisons:
        try:
            R, p = fit.distribution_compare("power_law", dist_name)
            if p < 0.05:
                if R > 0:
                    verdict = "power law WINS"
                else:
                    verdict = f"{display_name} WINS"
            else:
                verdict = "inconclusive"
            print(f"    {display_name:20s} {R:>8.3f} {p:>8.4f}  {verdict}")
            comp_results[dist_name] = {"R": R, "p": p}
        except Exception as e:
            print(f"    {display_name:20s}  (failed: {e})")
            comp_results[dist_name] = {"R": float("nan"), "p": float("nan")}

    return fit, comp_results


def plot_distributions(G, fit_in, fit_out):
    """Plot degree distributions with power law fits."""
    in_degrees = [d for _, d in G.in_degree() if d > 0]
    out_degrees = [d for _, d in G.out_degree() if d > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(IVORY)

    for ax, degrees, title, fit, color in [
        (axes[0], in_degrees, "In-degree Distribution", fit_in, INDIGO),
        (axes[1], out_degrees, "Out-degree Distribution", fit_out, TEAL),
    ]:
        ax.set_facecolor(IVORY)

        # Empirical CCDF
        counts = Counter(degrees)
        x_vals = sorted(counts.keys())
        y_vals = [counts[k] for k in x_vals]

        ax.scatter(x_vals, y_vals, c=color, alpha=0.5, s=15, zorder=2, label="Data")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree (k)")
        ax.set_ylabel("Frequency")
        ax.set_title(title, color=INDIGO)

        # Power law fit line
        x_fit = np.array([k for k in x_vals if k >= fit.xmin])
        if len(x_fit) > 0:
            y_fit = (x_fit / fit.xmin) ** (-fit.alpha) * counts.get(int(fit.xmin), 1)
            ax.plot(
                x_fit, y_fit, c=GOLD, linewidth=2.5, zorder=3,
                label=f"Power law (α={fit.alpha:.2f}, xmin={fit.xmin})",
            )
            ax.axvline(fit.xmin, color=CORAL, linestyle="--", alpha=0.6,
                       label=f"xmin={fit.xmin}")

        ax.legend()
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "degree_distribution.png", dpi=150, facecolor=IVORY)
    plt.close()
    print(f"\n  Saved: degree_distribution.png")

    # Also plot CCDF (complementary CDF) using powerlaw library
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(IVORY)

    for ax, degrees, title, fit, color in [
        (axes[0], in_degrees, "In-degree CCDF", fit_in, INDIGO),
        (axes[1], out_degrees, "Out-degree CCDF", fit_out, TEAL),
    ]:
        ax.set_facecolor(IVORY)
        fit.plot_ccdf(ax=ax, color=color, linewidth=2, label="Empirical CCDF")
        fit.power_law.plot_ccdf(ax=ax, color=GOLD, linewidth=2, linestyle="--",
                                label=f"Power law fit (α={fit.alpha:.2f})")
        try:
            fit.lognormal.plot_ccdf(ax=ax, color=CORAL, linewidth=1.5, linestyle=":",
                                    label="Lognormal fit")
        except Exception:
            pass
        ax.set_title(title, color=INDIGO)
        ax.legend()
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "degree_ccdf.png", dpi=150, facecolor=IVORY)
    plt.close()
    print(f"  Saved: degree_ccdf.png")


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    G = load_and_build()

    print(f"\n{'=' * 60}")
    print(f"  DEGREE DISTRIBUTION — POWER LAW FITTING")
    print(f"{'=' * 60}")

    in_degrees = [d for _, d in G.in_degree() if d > 0]
    out_degrees = [d for _, d in G.out_degree() if d > 0]

    fit_in, comp_in = fit_and_report(in_degrees, "IN-DEGREE")
    fit_out, comp_out = fit_and_report(out_degrees, "OUT-DEGREE")

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Metric':30s} {'In-degree':>12s} {'Out-degree':>12s}")
    print(f"  {'─' * 56}")
    print(f"  {'alpha':30s} {fit_in.alpha:>12.4f} {fit_out.alpha:>12.4f}")
    print(f"  {'xmin':30s} {fit_in.xmin:>12.0f} {fit_out.xmin:>12.0f}")
    print(f"  {'sigma':30s} {fit_in.sigma:>12.4f} {fit_out.sigma:>12.4f}")
    for dist_name, display_name in [
        ("lognormal", "vs Lognormal R"),
        ("exponential", "vs Exponential R"),
        ("truncated_power_law", "vs Truncated PL R"),
        ("stretched_exponential", "vs Stretched Exp R"),
    ]:
        r_in = comp_in.get(dist_name, {}).get("R", float("nan"))
        r_out = comp_out.get(dist_name, {}).get("R", float("nan"))
        print(f"  {display_name:30s} {r_in:>12.3f} {r_out:>12.3f}")

    # Save results to CSV
    rows = []
    for label, fit, comp in [("in_degree", fit_in, comp_in), ("out_degree", fit_out, comp_out)]:
        row = {
            "distribution": label,
            "alpha": fit.alpha,
            "xmin": fit.xmin,
            "sigma": fit.sigma,
        }
        for dist_name in ["lognormal", "exponential", "truncated_power_law", "stretched_exponential"]:
            c = comp.get(dist_name, {})
            row[f"vs_{dist_name}_R"] = c.get("R", float("nan"))
            row[f"vs_{dist_name}_p"] = c.get("p", float("nan"))
        rows.append(row)

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "powerlaw_fit_results.csv", index=False)
    print(f"\n  Saved: powerlaw_fit_results.csv")

    # Plot
    plot_distributions(G, fit_in, fit_out)

    print(f"\n{'=' * 60}")
    print(f"  DONE in {time.time() - start:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
