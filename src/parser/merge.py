#!/usr/bin/env python3
"""
Merge and validate nodes.csv and edges.csv.

This module provides utilities to:
1. Validate that all edge sources/targets exist in nodes
2. Compute statistics (node counts by kind, edge counts by type)
3. Generate summary reports

Usage:
    python -m parser.merge --nodes nodes.csv --edges edges.csv
"""

import csv
import argparse
from typing import TextIO
from collections import Counter


def load_nodes_csv(input_file: TextIO) -> list[dict]:
    """
    Load nodes from CSV.

    Args:
        input_file: File-like object containing nodes CSV

    Returns:
        List of node dicts
    """
    reader = csv.DictReader(input_file)
    return list(reader)


def load_edges_csv(input_file: TextIO) -> list[dict]:
    """
    Load edges from CSV.

    Args:
        input_file: File-like object containing edges CSV

    Returns:
        List of edge dicts with boolean conversion for is_explicit/is_simplifier
    """
    reader = csv.DictReader(input_file)
    edges = []
    for row in reader:
        edges.append({
            "source": row["source"],
            "target": row["target"],
            "is_explicit": row["is_explicit"].lower() in ("true", "1", "yes"),
            "is_simplifier": row["is_simplifier"].lower() in ("true", "1", "yes"),
        })
    return edges


def validate_edges(nodes: list[dict], edges: list[dict]) -> dict:
    """
    Validate that all edge sources and targets exist in nodes.

    Args:
        nodes: List of node dicts
        edges: List of edge dicts

    Returns:
        Dict with keys:
            valid: bool - True if all edges reference valid nodes
            missing_sources: set of source names not in nodes
            missing_targets: set of target names not in nodes
    """
    node_names = {n["name"] for n in nodes}

    missing_sources = set()
    missing_targets = set()

    for edge in edges:
        if edge["source"] not in node_names:
            missing_sources.add(edge["source"])
        if edge["target"] not in node_names:
            missing_targets.add(edge["target"])

    return {
        "valid": len(missing_sources) == 0 and len(missing_targets) == 0,
        "missing_sources": missing_sources,
        "missing_targets": missing_targets,
    }


def compute_statistics(nodes: list[dict], edges: list[dict]) -> dict:
    """
    Compute summary statistics.

    Args:
        nodes: List of node dicts
        edges: List of edge dicts

    Returns:
        Dict with statistics
    """
    # Node statistics
    kinds = Counter(n["kind"] for n in nodes)
    modules = Counter(n["module"] for n in nodes)

    # Edge statistics
    explicit_count = sum(1 for e in edges if e.get("is_explicit"))
    simp_count = sum(1 for e in edges if e.get("is_simplifier"))
    direct_count = sum(1 for e in edges if not e.get("is_explicit") and not e.get("is_simplifier"))

    return {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "nodes_by_kind": dict(kinds),
        "nodes_by_module": dict(modules),
        "explicit_edges": explicit_count,
        "simplifier_edges": simp_count,
        "direct_edges": direct_count,
    }


def generate_report(nodes: list[dict], edges: list[dict]) -> str:
    """
    Generate a markdown report.

    Args:
        nodes: List of node dicts
        edges: List of edge dicts

    Returns:
        Markdown-formatted report string
    """
    validation = validate_edges(nodes, edges)
    stats = compute_statistics(nodes, edges)

    lines = []
    lines.append("# MathlibGraph Data Summary")
    lines.append("")

    # Validation status
    if validation["valid"]:
        lines.append("**Validation**: PASSED - All edges reference valid nodes")
    else:
        lines.append("**Validation**: WARNING - Some edges reference missing nodes")
        if validation["missing_sources"]:
            lines.append(f"  - Missing sources: {len(validation['missing_sources'])} (e.g., {list(validation['missing_sources'])[:3]})")
        if validation["missing_targets"]:
            lines.append(f"  - Missing targets: {len(validation['missing_targets'])} (e.g., {list(validation['missing_targets'])[:3]})")
    lines.append("")

    # Node statistics
    lines.append("## Nodes")
    lines.append("")
    lines.append(f"**Total**: {stats['total_nodes']:,}")
    lines.append("")
    lines.append("| Kind | Count |")
    lines.append("|------|-------|")
    for kind, count in sorted(stats["nodes_by_kind"].items(), key=lambda x: -x[1]):
        lines.append(f"| {kind} | {count:,} |")
    lines.append("")

    # Edge statistics
    lines.append("## Edges")
    lines.append("")
    lines.append(f"**Total**: {stats['total_edges']:,}")
    lines.append("")
    lines.append("| Type | Count |")
    lines.append("|------|-------|")
    lines.append(f"| Explicit (*) | {stats['explicit_edges']:,} |")
    lines.append(f"| Simplifier (s) | {stats['simplifier_edges']:,} |")
    lines.append(f"| Direct | {stats['direct_edges']:,} |")
    lines.append("")

    # Top modules
    lines.append("## Top Modules (by node count)")
    lines.append("")
    top_modules = sorted(stats["nodes_by_module"].items(), key=lambda x: -x[1])[:10]
    lines.append("| Module | Count |")
    lines.append("|--------|-------|")
    for module, count in top_modules:
        lines.append(f"| {module} | {count:,} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Merge and validate nodes.csv and edges.csv"
    )
    parser.add_argument(
        "--nodes", "-n",
        required=True,
        help="Input nodes.csv file"
    )
    parser.add_argument(
        "--edges", "-e",
        required=True,
        help="Input edges.csv file"
    )
    parser.add_argument(
        "--report", "-r",
        help="Output report file (markdown)"
    )

    args = parser.parse_args()

    print(f"Loading {args.nodes}...")
    with open(args.nodes, "r", encoding="utf-8") as f:
        nodes = load_nodes_csv(f)
    print(f"  Loaded {len(nodes)} nodes")

    print(f"Loading {args.edges}...")
    with open(args.edges, "r", encoding="utf-8") as f:
        edges = load_edges_csv(f)
    print(f"  Loaded {len(edges)} edges")

    print("\nValidating...")
    validation = validate_edges(nodes, edges)
    if validation["valid"]:
        print("  PASSED - All edges reference valid nodes")
    else:
        print("  WARNING - Some edges reference missing nodes")
        print(f"    Missing sources: {len(validation['missing_sources'])}")
        print(f"    Missing targets: {len(validation['missing_targets'])}")

    print("\nStatistics:")
    stats = compute_statistics(nodes, edges)
    print(f"  Total nodes: {stats['total_nodes']:,}")
    print(f"  Total edges: {stats['total_edges']:,}")
    print(f"  Explicit edges: {stats['explicit_edges']:,}")
    print(f"  Simplifier edges: {stats['simplifier_edges']:,}")
    print(f"  Direct edges: {stats['direct_edges']:,}")

    if args.report:
        report = generate_report(nodes, edges)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
