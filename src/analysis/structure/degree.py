#!/usr/bin/env python3
"""
Degree distribution analysis and scale-free property testing.

Part II: Structure - Degree Distribution
Addresses Q2 (Scaling Laws)
"""

from pathlib import Path
from collections import Counter

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Color scheme
INDIGO = '#2C3E6B'
GOLD = '#C9A96E'
IVORY = '#F5F0E8'


def degree_distribution_analysis(
    G: nx.DiGraph,
    output_dir: Path = None
) -> dict:
    """
    Analyze degree distribution and test for scale-free property.

    Uses powerlaw library to fit power law and compare with alternatives.

    Args:
        G: NetworkX DiGraph
        output_dir: Directory to save plots (defaults to data/)

    Returns:
        Dictionary with fit results for in-degree and out-degree
    """
    import powerlaw

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data"

    print("\n" + "="*60)
    print("DEGREE DISTRIBUTION + SCALE-FREE TEST")
    print("="*60)

    in_degrees = [d for n, d in G.in_degree() if d > 0]
    out_degrees = [d for n, d in G.out_degree() if d > 0]

    results = {}

    # Fit power law to in-degree
    print("\nIn-degree distribution:")
    fit_in = powerlaw.Fit(in_degrees, discrete=True, verbose=False)
    print(f"  Power law alpha: {fit_in.alpha:.3f}")
    print(f"  xmin: {fit_in.xmin}")

    # Compare distributions
    R_ln, p_ln = fit_in.distribution_compare('power_law', 'lognormal')
    R_exp, p_exp = fit_in.distribution_compare('power_law', 'exponential')
    print(f"  vs lognormal: R={R_ln:.3f}, p={p_ln:.3f}")
    print(f"  vs exponential: R={R_exp:.3f}, p={p_exp:.3f}")

    results['in_degree'] = {
        'alpha': fit_in.alpha,
        'xmin': fit_in.xmin,
        'vs_lognormal_R': R_ln,
        'vs_exponential_R': R_exp,
    }

    # Fit power law to out-degree
    print("\nOut-degree distribution:")
    fit_out = powerlaw.Fit(out_degrees, discrete=True, verbose=False)
    print(f"  Power law alpha: {fit_out.alpha:.3f}")
    print(f"  xmin: {fit_out.xmin}")

    R_ln, p_ln = fit_out.distribution_compare('power_law', 'lognormal')
    R_exp, p_exp = fit_out.distribution_compare('power_law', 'exponential')
    print(f"  vs lognormal: R={R_ln:.3f}, p={p_ln:.3f}")
    print(f"  vs exponential: R={R_exp:.3f}, p={p_exp:.3f}")

    results['out_degree'] = {
        'alpha': fit_out.alpha,
        'xmin': fit_out.xmin,
        'vs_lognormal_R': R_ln,
        'vs_exponential_R': R_exp,
    }

    # Plot degree distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(IVORY)

    for ax, degrees, title, fit in [
        (axes[0], in_degrees, 'In-degree Distribution', fit_in),
        (axes[1], out_degrees, 'Out-degree Distribution', fit_out)
    ]:
        ax.set_facecolor(IVORY)

        # Histogram
        counts = Counter(degrees)
        x = sorted(counts.keys())
        y = [counts[k] for k in x]

        ax.scatter(x, y, c=INDIGO, alpha=0.6, s=20, label='Data')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        ax.set_title(title, color=INDIGO)

        # Power law fit line
        x_fit = np.array(x)
        x_fit = x_fit[x_fit >= fit.xmin]
        if len(x_fit) > 0:
            y_fit = (x_fit ** (-fit.alpha)) * (x_fit[0] ** fit.alpha) * counts[x_fit[0]]
            ax.plot(x_fit, y_fit, c=GOLD, linewidth=2,
                   label=f'Power law (alpha={fit.alpha:.2f})')

        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "degree_distribution.png", dpi=150, facecolor=IVORY)
    plt.close()
    print("\nSaved: degree_distribution.png")

    return results
