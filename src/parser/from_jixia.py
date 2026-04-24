#!/usr/bin/env python3
"""
Parse jixia declaration JSON output to extract tactic usage per declaration.

Reads jixia -d output (one JSON file per Mathlib module) and produces
per-declaration tactic usage data.

Usage:
    python -m parser.from_jixia --input-dir data/jixia_decls --output data/tactic_usage.ndjson
"""

import json
import re
import sys
import argparse
from pathlib import Path
from collections import Counter
from typing import Any


# Lean tactic keywords (common ones; conservative list)
# We identify tactics by the first token of each tactic line
KNOWN_TACTICS = {
    "simp", "rw", "rewrite", "exact", "apply", "intro", "intros",
    "constructor", "cases", "induction", "rcases", "obtain",
    "have", "let", "suffices", "calc", "show",
    "ring", "linarith", "omega", "norm_num", "positivity", "polyrith",
    "field_simp", "push_neg", "contrapose", "by_contra",
    "ext", "funext", "congr",
    "trivial", "tauto", "decide", "norm_cast", "push_cast",
    "aesop", "gcongr", "refine", "convert",
    "subst", "injection", "contradiction",
    "split", "left", "right", "exfalso",
    "specialize", "generalize", "revert", "clear",
    "dsimp", "change", "unfold", "delta", "erw",
    "simpa", "simp_rw", "rfl", "rfl'",
    "assumption", "exact?", "apply?",
    "sorry", "admit",
    "first", "try", "repeat", "iterate",
    "all_goals", "any_goals", "focus",
    "skip", "done", "stop",
    "infer_instance",
    "use", "existsi", "choose",
    "set", "alias",
    "trans", "symm", "swap",
    "rintro", "refine'",
    "continuity", "measurability", "bound",
    "mono", "nontriviality",
    "abel", "group",
    "peel", "lift", "borelize",
    "filter_upwards",
    "classical",
}


def extract_tactic_names(proof_text: str) -> list[str]:
    """
    Extract tactic names from a proof text string.

    Given a proof like:
        := by
          simp [add_comm]
          ring

    Returns: ["simp", "ring"]
    """
    text = proof_text.strip()
    # Strip leading ':=' if present
    if text.startswith(":="):
        text = text[2:].strip()
    # Must start with 'by'
    if not text.startswith("by"):
        return []
    text = text[2:]

    tactics = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("--") or line.startswith("/-"):
            continue
        # Strip focus dots and case separators
        line = re.sub(r"^[·|]\s*", "", line)
        line = line.strip()
        if not line:
            continue
        # Handle 'next ... =>' and 'case ... =>' prefixes
        if re.match(r"^(next|case)\b", line):
            m = re.search(r"=>\s*(.*)", line)
            if m and m.group(1).strip():
                line = m.group(1).strip()
            else:
                continue
        # First token is the tactic name
        token = re.split(r"[\s\[\(\{<;]", line)[0].rstrip("!?'")
        # Skip 'by' (block opener), 'do' (monadic), structural keywords
        if token in ("by", "do", "where", "with", "fun", "match", "if", "then", "else", "return"):
            continue
        if token and (token in KNOWN_TACTICS or token[0].islower()):
            tactics.append(token)
    return tactics


def get_value_pp(decl: dict) -> str:
    """Safely get the pretty-printed value from a declaration."""
    v = decl.get("value")
    if v is None:
        return ""
    if isinstance(v, dict):
        return v.get("pp", "")
    return ""


def get_decl_name(decl: dict) -> str:
    """Get the fully qualified name from a declaration."""
    name = decl.get("name", [])
    if isinstance(name, list):
        return ".".join(str(n) for n in name)
    return str(name)


def process_jixia_file(filepath: Path) -> list[dict[str, Any]]:
    """
    Process a single jixia declaration JSON file.

    Returns list of {name, kind, module, tactics} dicts.
    """
    # Derive module name from filename: Mathlib_Algebra_Group_Defs.json -> Mathlib.Algebra.Group.Defs
    module = filepath.stem.replace("_", ".")

    with open(filepath, "r", encoding="utf-8") as f:
        decls = json.load(f)

    results = []
    for decl in decls:
        name = get_decl_name(decl)
        kind = decl.get("kind", "")
        value_pp = get_value_pp(decl)

        tactics = extract_tactic_names(value_pp)
        is_tactic_proof = value_pp.strip().startswith(":= by") or (
            value_pp.strip().startswith("by") and kind == "theorem"
        )

        results.append({
            "name": name,
            "kind": kind,
            "module": module,
            "is_tactic_proof": is_tactic_proof,
            "tactics": tactics,
            "tactic_count": len(tactics),
        })
    return results


def aggregate_tactic_stats(all_results: list[dict]) -> dict[str, Any]:
    """Compute summary statistics from all declaration tactic data."""
    total_decls = len(all_results)
    tactic_proofs = [r for r in all_results if r["is_tactic_proof"]]
    term_proofs = [r for r in all_results if not r["is_tactic_proof"] and r["kind"] == "theorem"]

    # Global tactic frequency
    global_freq = Counter()
    for r in tactic_proofs:
        global_freq.update(r["tactics"])

    # Per-namespace tactic distribution
    ns_freq: dict[str, Counter] = {}
    for r in tactic_proofs:
        # Top-level namespace: first two components (e.g., Mathlib.Algebra)
        parts = r["module"].split(".")
        ns = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        if ns not in ns_freq:
            ns_freq[ns] = Counter()
        ns_freq[ns].update(r["tactics"])

    return {
        "total_declarations": total_decls,
        "tactic_proofs": len(tactic_proofs),
        "term_proofs": len(term_proofs),
        "total_tactic_steps": sum(r["tactic_count"] for r in tactic_proofs),
        "top_tactics": global_freq.most_common(30),
        "namespace_distributions": {
            ns: freq.most_common(10) for ns, freq in
            sorted(ns_freq.items(), key=lambda x: -sum(x[1].values()))[:20]
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract tactic usage from jixia declaration output"
    )
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Directory containing jixia .json declaration files"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output NDJSON file (one line per declaration with tactics)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    json_files = sorted(input_dir.glob("*.json"))
    print(f"Found {len(json_files)} jixia output files", file=sys.stderr)

    all_results = []
    for jf in json_files:
        try:
            results = process_jixia_file(jf)
            all_results.extend(results)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: skipping {jf.name}: {e}", file=sys.stderr)

    print(f"Total declarations: {len(all_results)}", file=sys.stderr)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(all_results)} records to {args.output}", file=sys.stderr)

    if args.summary or not args.output:
        stats = aggregate_tactic_stats(all_results)
        print(f"\nTotal declarations: {stats['total_declarations']:,}")
        print(f"Tactic proofs:     {stats['tactic_proofs']:,}")
        print(f"Term proofs:       {stats['term_proofs']:,}")
        print(f"Total tactic steps: {stats['total_tactic_steps']:,}")
        print(f"\nTop 30 tactics:")
        for tactic, count in stats["top_tactics"]:
            print(f"  {tactic:20s} {count:>8,}")


if __name__ == "__main__":
    main()
