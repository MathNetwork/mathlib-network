#!/usr/bin/env python3
"""
Parse Lean mechanism extraction NDJSON output.

The Lean script (ExtractMechanisms.lean) queries the Mathlib Environment
and outputs one JSON object per line with the following types:

    {"type":"meta",      "key":"...", "value":"..."}
    {"type":"kind",      "name":"theorem", "count":N}
    {"type":"attribute", "name":"simp", "count":N}
    {"type":"class",     "name":"Add"}
    {"type":"instance",  "name":"instAddNat", "class":"Add"}
    {"type":"structure_parent", "child":"CommMonoid", "parent":"Monoid"}
    {"type":"to_additive_pair", "source":"mul_comm", "target":"add_comm"}
    {"type":"coercion",   "name":"Int.ofNat", "coe_type":"coe"}
    {"type":"deriving_handler", "name":"BEq"}
    {"type":"stmt_proof_stats", "S_only":N, "P_only":N, "SP":N, "total_decls":N}
    {"type":"module_import", "module":"...", "imported":"...", "is_exported":true}

Usage:
    python -m parser.from_mechanisms --input mechanisms.ndjson
"""

import csv
import json
import sys
import argparse
import statistics
from pathlib import Path
from typing import TextIO, Any


def parse_mechanisms(input_file: TextIO) -> dict[str, Any]:
    """
    Parse NDJSON mechanism extraction output.

    Returns:
        Dict with keys: meta, kinds, attributes, classes, instances,
        structure_parents, to_additive_pairs
    """
    result: dict[str, Any] = {
        "meta": {},
        "kinds": {},
        "attributes": [],
        "classes": [],
        "instances": [],
        "structure_parents": [],
        "to_additive_pairs": [],
        "coercions": [],
        "deriving_handlers": [],
        "stmt_proof_stats": None,
        "module_imports": [],
        "decl_modules": {},
        "def_heights": {},
    }

    for line in input_file:
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Warning: skipping malformed JSON: {e}", file=sys.stderr)
            continue

        typ = obj.get("type")
        if typ == "meta":
            result["meta"][obj["key"]] = obj["value"]
        elif typ == "kind":
            result["kinds"][obj["name"]] = obj["count"]
        elif typ == "attribute":
            result["attributes"].append({
                "name": obj["name"],
                "count": obj["count"],
            })
        elif typ == "class":
            result["classes"].append({"name": obj["name"]})
        elif typ == "instance":
            result["instances"].append({
                "name": obj["name"],
                "class": obj.get("class", ""),
            })
        elif typ == "structure_parent":
            result["structure_parents"].append({
                "child": obj["child"],
                "parent": obj["parent"],
            })
        elif typ == "to_additive_pair":
            result["to_additive_pairs"].append({
                "source": obj["source"],
                "target": obj["target"],
            })
        elif typ == "coercion":
            result["coercions"].append({
                "name": obj["name"],
                "coe_type": obj.get("coe_type", ""),
            })
        elif typ == "deriving_handler":
            result["deriving_handlers"].append({
                "name": obj["name"],
            })
        elif typ == "stmt_proof_stats":
            result["stmt_proof_stats"] = {
                "S_only": obj["S_only"],
                "P_only": obj["P_only"],
                "SP": obj["SP"],
                "total_decls": obj["total_decls"],
            }
        elif typ == "module_import":
            result["module_imports"].append({
                "module": obj["module"],
                "imported": obj["imported"],
                "is_exported": obj.get("is_exported", True),
            })
        elif typ == "decl_module":
            result["decl_modules"][obj["name"]] = obj["module"]
        elif typ == "def_height":
            result["def_heights"][obj["name"]] = {
                "height": obj.get("height"),  # None for abbrev/opaque
                "reducibility": obj["reducibility"],
            }

    return result


def get_attributes_sorted(result: dict[str, Any]) -> list[dict]:
    """Return attributes sorted by count descending."""
    return sorted(result["attributes"], key=lambda a: a["count"], reverse=True)


def get_parent_chain(result: dict[str, Any], name: str) -> list[str]:
    """
    Reconstruct the structure inheritance chain starting from `name`.

    Returns list like ["CommMonoid", "Monoid", "MulOneClass"].
    """
    parent_map = {p["child"]: p["parent"] for p in result["structure_parents"]}
    chain = [name]
    current = name
    while current in parent_map:
        current = parent_map[current]
        chain.append(current)
    return chain


def compute_import_utilization(
    edges_path: Path,
    module_imports: list[dict],
    decl_modules: dict[str, str],
) -> dict[str, Any]:
    """
    Compute import utilization: for each import edge (m_i → m_j),
    what fraction of m_j's declarations are actually referenced by m_i.

    Args:
        edges_path: Path to mathlib_edges.csv (declaration-level edges)
        module_imports: list of {module, imported, is_exported} dicts
        decl_modules: dict mapping declaration name → file module name
            (from 'decl_module' NDJSON records)

    Returns summary statistics.
    """
    # Step 1: Count declarations per file module
    mod_decl_count: dict[str, int] = {}
    for mod in decl_modules.values():
        mod_decl_count[mod] = mod_decl_count.get(mod, 0) + 1

    # Step 2: For each (source_module, target_module), collect used target declarations
    # used[(m_i, m_j)] = set of declarations in m_j referenced by m_i
    used: dict[tuple[str, str], set[str]] = {}
    with open(edges_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src_mod = decl_modules.get(row["source"], "")
            tgt_mod = decl_modules.get(row["target"], "")
            if src_mod and tgt_mod and src_mod != tgt_mod:
                key = (src_mod, tgt_mod)
                if key not in used:
                    used[key] = set()
                used[key].add(row["target"])

    # Step 3: Compute utilization per import edge
    import_set = {(imp["module"], imp["imported"]) for imp in module_imports}
    utils = []
    zero_count = 0
    for mod_i, mod_j in import_set:
        total_j = mod_decl_count.get(mod_j, 0)
        if total_j == 0:
            continue
        used_count = len(used.get((mod_i, mod_j), set()))
        util = used_count / total_j
        utils.append(util)
        if used_count == 0:
            zero_count += 1

    utils_sorted = sorted(utils)
    return {
        "total_import_edges": len(utils),
        "mean_util": statistics.mean(utils) if utils else 0,
        "median_util": statistics.median(utils) if utils else 0,
        "p25_util": utils_sorted[len(utils_sorted) // 4] if utils else 0,
        "p75_util": utils_sorted[3 * len(utils_sorted) // 4] if utils else 0,
        "max_util": max(utils) if utils else 0,
        "zero_util_edges": zero_count,
    }


def generate_summary(result: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a summary dict suitable for the paper's decomposition table.
    """
    sorted_attrs = get_attributes_sorted(result)
    summary: dict[str, Any] = {
        "total_constants": result["meta"].get("total_constants", 0),
        "mathlib_constants": result["meta"].get("mathlib_constants", 0),
        "total_classes": len(result["classes"]),
        "total_instances": len(result["instances"]),
        "total_attributes": len(result["attributes"]),
        "total_structure_parents": len(result["structure_parents"]),
        "total_to_additive_pairs": len(result["to_additive_pairs"]),
        "total_coercions": len(result["coercions"]),
        "total_deriving_handlers": len(result["deriving_handlers"]),
        "total_module_imports": len(result["module_imports"]),
        "stmt_proof_stats": result["stmt_proof_stats"],
        "kinds": result["kinds"],
        "top_attributes": sorted_attrs[:20],
    }
    if result["module_imports"]:
        public = sum(1 for i in result["module_imports"] if i["is_exported"])
        summary["public_imports"] = public
        summary["private_imports"] = len(result["module_imports"]) - public
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Parse Lean mechanism extraction NDJSON"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input NDJSON file from ExtractMechanisms.lean"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary table"
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        result = parse_mechanisms(f)

    summary = generate_summary(result)

    print(f"Lean version:      {result['meta'].get('lean_version', '?')}")
    print(f"Total constants:   {summary['total_constants']:,}")
    print(f"Mathlib constants: {summary['mathlib_constants']:,}")
    print()
    print("Declaration kinds:")
    for kind, count in sorted(summary["kinds"].items(), key=lambda x: -x[1]):
        print(f"  {kind:20s} {count:>8,}")
    print()
    print(f"Registered attributes: {summary['total_attributes']}")
    print("Top attributes by tagged declaration count:")
    for attr in summary["top_attributes"]:
        print(f"  @[{attr['name']:20s}] {attr['count']:>8,} declarations")
    print()
    print(f"Typeclasses:       {summary['total_classes']:,}")
    print(f"Instances:         {summary['total_instances']:,}")
    print(f"Structure parents: {summary['total_structure_parents']:,}")
    print(f"to_additive pairs: {summary['total_to_additive_pairs']:,}")
    print(f"Coercions:         {summary['total_coercions']:,}")
    print(f"Deriving handlers: {summary['total_deriving_handlers']:,}")
    print(f"Module imports:    {summary['total_module_imports']:,}")
    if summary.get("public_imports") is not None:
        print(f"  Public imports:  {summary['public_imports']:,}")
        print(f"  Private imports: {summary['private_imports']:,}")
    sp = summary.get("stmt_proof_stats")
    if sp:
        total_edges = sp["S_only"] + sp["P_only"] + sp["SP"]
        print()
        print("Statement/proof edge classification:")
        print(f"  S-only (statement):  {sp['S_only']:>10,}  ({100*sp['S_only']/total_edges:.1f}%)")
        print(f"  P-only (proof):      {sp['P_only']:>10,}  ({100*sp['P_only']/total_edges:.1f}%)")
        print(f"  SP (both):           {sp['SP']:>10,}  ({100*sp['SP']/total_edges:.1f}%)")
        print(f"  Total edges:         {total_edges:>10,}")
        print(f"  Declarations analyzed: {sp['total_decls']:,}")


if __name__ == "__main__":
    main()
