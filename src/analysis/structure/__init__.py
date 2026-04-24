"""
Part II: Structure - What does this graph look like?

Addresses Q2 (Scaling Laws): Does Mathlib exhibit power laws,
small-world properties, and other complex network signatures?

Modules:
- descriptive: Basic graph statistics (nodes, edges, density, connectivity)
- degree: Degree distribution analysis and power law fitting
- centrality: PageRank, HITS, betweenness centrality
- community: Community detection (Louvain algorithm)
"""

from .descriptive import basic_statistics, load_graph
from .degree import degree_distribution_analysis
from .centrality import pagerank_hits_analysis, betweenness_analysis
from .community import community_detection

__all__ = [
    "basic_statistics",
    "load_graph",
    "degree_distribution_analysis",
    "pagerank_hits_analysis",
    "betweenness_analysis",
    "community_detection",
]
