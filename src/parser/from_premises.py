#!/usr/bin/env python3
"""
Parse lean-training-data premises output to edges.csv.

lean-training-data (https://github.com/kim-em/lean-training-data) provides
the `premises` command which outputs declaration dependencies in text format.

Format:
    ---
    DeclarationName
    * explicit_dep    (appears in explicit arguments)
    s simp_dep        (used by simplifier)
      direct_dep      (regular dependency)
    ---

Output: edges.csv with columns (source, target, is_explicit, is_simplifier)

Usage:
    python -m parser.from_premises --input premises.txt --output edges.csv
"""

import csv
import sys
import argparse
from typing import TextIO


def parse_premises(
    input_file: TextIO,
    filter_mathlib_source: bool = False
) -> list[dict]:
    """
    Parse premises output and extract edges.

    Args:
        input_file: File-like object containing premises output
        filter_mathlib_source: If True, only include edges where source starts with Mathlib.

    Returns:
        List of dicts with keys: source, target, is_explicit, is_simplifier
    """
    edges: list[dict] = []
    current_source: str | None = None

    for line in input_file:
        line = line.rstrip("\n\r")

        # Separator between declarations
        if line == "---":
            current_source = None
            continue

        # Empty line - skip
        if not line.strip():
            continue

        # Determine if this is a dependency line or a new declaration
        # Dependency lines:
        #   - Start with whitespace (indented)
        #   - Start with "* " (explicit dependency)
        #   - Start with "s " (simplifier dependency)
        # Declaration lines:
        #   - Start at column 0 with an identifier (not whitespace, not "* ", not "s ")

        is_dependency_line = (
            line.startswith((" ", "\t")) or
            line.startswith("* ") or
            line.startswith("s ")
        )

        if not is_dependency_line:
            # This is a new declaration name
            current_source = line.strip()
            continue

        # This is a dependency line
        if current_source is None:
            # No source declaration yet, skip
            continue

        # Apply filter
        if filter_mathlib_source and not current_source.startswith("Mathlib."):
            continue

        # Parse the dependency
        is_explicit = False
        is_simplifier = False
        target = line

        if line.startswith("* "):
            is_explicit = True
            target = line[2:]
        elif line.startswith("s "):
            is_simplifier = True
            target = line[2:]
        else:
            # Regular dependency (just indented)
            target = line.lstrip()

        target = target.strip()

        if target:
            edges.append({
                "source": current_source,
                "target": target,
                "is_explicit": is_explicit,
                "is_simplifier": is_simplifier
            })

    return edges


def write_edges_csv(edges: list[dict], output_file: TextIO) -> None:
    """
    Write edges to CSV format.

    Args:
        edges: List of edge dicts
        output_file: File-like object to write to
    """
    writer = csv.DictWriter(
        output_file,
        fieldnames=["source", "target", "is_explicit", "is_simplifier"]
    )
    writer.writeheader()
    writer.writerows(edges)


def main():
    parser = argparse.ArgumentParser(
        description="Parse lean-training-data premises to edges.csv"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input premises.txt file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output CSV file"
    )
    parser.add_argument(
        "--filter-mathlib",
        action="store_true",
        help="Only include edges where source starts with Mathlib."
    )

    args = parser.parse_args()

    print(f"Parsing {args.input}...")

    with open(args.input, "r", encoding="utf-8") as f:
        edges = parse_premises(f, filter_mathlib_source=args.filter_mathlib)

    print(f"Found {len(edges)} edges")

    # Count by type
    explicit_count = sum(1 for e in edges if e["is_explicit"])
    simp_count = sum(1 for e in edges if e["is_simplifier"])
    direct_count = len(edges) - explicit_count - simp_count

    print(f"  Explicit (*): {explicit_count}")
    print(f"  Simplifier (s): {simp_count}")
    print(f"  Direct: {direct_count}")

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        write_edges_csv(edges, f)

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
