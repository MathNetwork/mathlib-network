#!/usr/bin/env python3
"""
Parse lean4export NDJSON output to nodes.csv.

lean4export (https://github.com/leanprover/lean4export) exports Lean 4
Environment to NDJSON format. This parser extracts declaration names,
kinds, and modules.

Output: nodes.csv with columns (name, kind, module)

Usage:
    python -m parser.from_lean4export --input mathlib.ndjson --output nodes.csv
"""

import json
import csv
import sys
import argparse
from typing import TextIO, Optional


def parse_ndjson(
    input_file: TextIO,
    filter_mathlib: bool = False
) -> list[dict]:
    """
    Parse lean4export NDJSON and extract nodes.

    Args:
        input_file: File-like object containing NDJSON
        filter_mathlib: If True, only include Mathlib.* declarations

    Returns:
        List of dicts with keys: name, kind, module
    """
    # Name table: integer -> string
    names: dict[int, str] = {0: ""}  # 0 is anonymous/root

    nodes: list[dict] = []

    for line_num, line in enumerate(input_file, 1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping malformed JSON at line {line_num}: {e}", file=sys.stderr)
            continue

        # Build name table from Name.str and Name.num entries
        if "str" in obj:
            # Name.str: {"str": {"pre": integer, "str": string}, "in": integer}
            pre = obj["str"]["pre"]
            s = obj["str"]["str"]
            idx = obj["in"]
            prefix = names.get(pre, "")
            names[idx] = f"{prefix}.{s}" if prefix else s

        elif "num" in obj:
            # Name.num: {"num": {"pre": integer, "i": integer}, "in": integer}
            pre = obj["num"]["pre"]
            i = obj["num"]["i"]
            idx = obj["in"]
            prefix = names.get(pre, "")
            names[idx] = f"{prefix}.{i}" if prefix else str(i)

        # Parse declarations
        elif "thm" in obj:
            # Theorem(s)
            for thm in obj["thm"]:
                name = names.get(thm["name"], f"<unknown:{thm['name']}>")
                node = _make_node(name, "theorem")
                if node and (not filter_mathlib or name.startswith("Mathlib.")):
                    nodes.append(node)

        elif "def" in obj:
            # Definition(s) or Opaque(s)
            for defn in obj["def"]:
                name = names.get(defn["name"], f"<unknown:{defn['name']}>")

                # Distinguish DefnVal vs OpaqueVal by presence of hints vs isUnsafe
                if "hints" in defn:
                    # DefnVal
                    hints = defn["hints"]
                    if hints == "abbrev":
                        kind = "abbrev"
                    elif hints == "opaque":
                        kind = "opaque"
                    elif isinstance(hints, dict) and "regular" in hints:
                        kind = "definition"
                    else:
                        kind = "definition"
                else:
                    # OpaqueVal (has isUnsafe instead of hints)
                    kind = "opaque"

                node = _make_node(name, kind)
                if node and (not filter_mathlib or name.startswith("Mathlib.")):
                    nodes.append(node)

        elif "axiomInfo" in obj:
            # Axiom
            name = names.get(obj["axiomInfo"]["name"], f"<unknown:{obj['axiomInfo']['name']}>")
            node = _make_node(name, "axiom")
            if node and (not filter_mathlib or name.startswith("Mathlib.")):
                nodes.append(node)

        elif "quotInfo" in obj:
            # Quotient
            name = names.get(obj["quotInfo"]["name"], f"<unknown:{obj['quotInfo']['name']}>")
            node = _make_node(name, "quotient")
            if node and (not filter_mathlib or name.startswith("Mathlib.")):
                nodes.append(node)

        elif "inductive" in obj:
            # Inductive type with constructors and recursors
            ind_data = obj["inductive"]

            for ind in ind_data.get("inductiveVals", []):
                name = names.get(ind["name"], f"<unknown:{ind['name']}>")
                node = _make_node(name, "inductive")
                if node and (not filter_mathlib or name.startswith("Mathlib.")):
                    nodes.append(node)

            for ctor in ind_data.get("constructorVals", []):
                name = names.get(ctor["name"], f"<unknown:{ctor['name']}>")
                node = _make_node(name, "constructor")
                if node and (not filter_mathlib or name.startswith("Mathlib.")):
                    nodes.append(node)

            for rec in ind_data.get("recursorVals", []):
                name = names.get(rec["name"], f"<unknown:{rec['name']}>")
                node = _make_node(name, "recursor")
                if node and (not filter_mathlib or name.startswith("Mathlib.")):
                    nodes.append(node)

    return nodes


def _make_node(name: str, kind: str) -> Optional[dict]:
    """
    Create a node dict with name, kind, and module.

    Module is extracted as everything before the last dot.
    """
    if not name:
        return None

    # Extract module: everything before the last dot
    if "." in name:
        module = name.rsplit(".", 1)[0]
    else:
        module = ""

    return {
        "name": name,
        "kind": kind,
        "module": module
    }


def write_nodes_csv(nodes: list[dict], output_file: TextIO) -> None:
    """
    Write nodes to CSV format.

    Args:
        nodes: List of node dicts
        output_file: File-like object to write to
    """
    writer = csv.DictWriter(output_file, fieldnames=["name", "kind", "module"])
    writer.writeheader()
    writer.writerows(nodes)


def main():
    parser = argparse.ArgumentParser(
        description="Parse lean4export NDJSON to nodes.csv"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input NDJSON file from lean4export"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output CSV file"
    )
    parser.add_argument(
        "--filter-mathlib",
        action="store_true",
        help="Only include Mathlib.* declarations"
    )

    args = parser.parse_args()

    print(f"Parsing {args.input}...")

    with open(args.input, "r", encoding="utf-8") as f:
        nodes = parse_ndjson(f, filter_mathlib=args.filter_mathlib)

    print(f"Found {len(nodes)} declarations")

    # Count by kind
    from collections import Counter
    kinds = Counter(n["kind"] for n in nodes)
    print("By kind:")
    for kind, count in kinds.most_common():
        print(f"  {kind}: {count}")

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        write_nodes_csv(nodes, f)

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
