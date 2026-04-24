"""
MathlibGraph: Network Analysis of Mathematical Knowledge

Extracts and analyzes the theorem-level dependency network of Mathlib4,
applying network science, spectral geometry, and statistical physics
methods to understand the structure of formalized mathematics.

Organized around five central questions:

Q1 (Universality): Do independent formalizations converge on similar structures?
Q2 (Scaling Laws): Does the graph obey power laws and small-world properties?
Q3 (Phase Transitions): Can we detect "crises" in mathematical evolution?
Q4 (Abstraction Gradients): Does abstraction correlate with geometric properties?
Q5 (Conway's Law): Does structure reflect mathematics or human organization?

Package structure:
- parser/: Extract dependency graphs using lean4export + lean-training-data
- analysis/: Analysis modules organized by question
  - structure/: Part II (Q2) - Descriptive network analysis
  - geometry/:  Part III (Q4) - Spectral and geometric analysis
  - dynamics/:  Part IV (Q3) - Temporal and robustness analysis
  - intelligence/: Part V (Q5) - Predictive analysis
  - universality/: Part VI (Q1) - Cross-library comparison
- data/: Input data and analysis outputs
"""

__version__ = "0.1.0"
