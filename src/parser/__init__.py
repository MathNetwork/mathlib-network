"""
Parser module for extracting dependency graphs from Mathlib.

This module provides parsers for two data sources:
- from_lean4export: Parse lean4export NDJSON for node attributes (name, kind, module)
- from_premises: Parse lean-training-data premises for dependency edges

Data pipeline:
    lean4export → from_lean4export.py → nodes.csv
    premises    → from_premises.py    → edges.csv
    merge.py validates and combines both files.
"""

from .from_lean4export import parse_ndjson, write_nodes_csv
from .from_premises import parse_premises, write_edges_csv
from .merge import validate_edges, compute_statistics, generate_report

__all__ = [
    "parse_ndjson",
    "write_nodes_csv",
    "parse_premises",
    "write_edges_csv",
    "validate_edges",
    "compute_statistics",
    "generate_report",
]
