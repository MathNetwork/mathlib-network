"""
Part IV: Dynamics - How does it grow and change?

Addresses Q3 (Phase Transitions): Can we detect Kline-style crises
as structural phase transitions in Mathlib's evolution?

Modules:
- cascade: Cascade and robustness analysis (node removal impact)
- temporal: Time series analysis of graph evolution (skeleton)
"""

from .cascade import cascade_analysis, robustness_analysis

__all__ = [
    "cascade_analysis",
    "robustness_analysis",
]
