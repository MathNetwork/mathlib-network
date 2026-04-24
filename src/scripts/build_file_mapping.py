#!/usr/bin/env python3
"""
Build a declaration → file-module mapping by scanning Mathlib source.

For each .lean file under Mathlib/:
  1. Derive module name from path: Mathlib/Data/Nat/Basic.lean → Mathlib.Data.Nat.Basic
  2. Parse file, tracking namespace blocks
  3. For every named declaration, record (full_qualified_name, file_module)

Output: output/declaration_to_file_module.csv
"""

import re
import time
from pathlib import Path

MATHLIB_DIR = Path("/tmp/mathlib4_thm_lemma")
OUTPUT_DIR = Path("output")

# ── regex patterns ───────────────────────────────────────────

DECL_KW = (
    r"theorem|lemma|def|abbrev|class|structure|inductive|axiom|instance|opaque"
)

# Match: optional attrs + optional modifiers + keyword + name
pat_decl = re.compile(
    r"^(?:@\[[^\]]*\]\s+)*"
    r"(?:(?:protected|private|noncomputable|unsafe|partial|scoped|local)\s+)*"
    r"(?:" + DECL_KW + r")\s+"
    r"([A-Za-z_][\w'.]*)"
)

pat_ns_open = re.compile(r"^namespace\s+(\S+)")
pat_ns_close = re.compile(r"^end\s+(\S+)")
pat_end_bare = re.compile(r"^end\s*$")
pat_section_open = re.compile(r"^section\s*(.*)")

# To avoid section line matching a declaration keyword
pat_decl_start = re.compile(r"^(?:@\[|protected|private|noncomputable|unsafe|partial|scoped|local|"
                            + DECL_KW + r")")


def path_to_module(lean_file: Path, root: Path) -> str:
    """Convert file path to Lean module name.

    Mathlib/Data/Nat/Basic.lean → Mathlib.Data.Nat.Basic
    """
    rel = lean_file.relative_to(root)
    parts = list(rel.parts)
    # Strip .lean extension from last part
    parts[-1] = parts[-1].removesuffix(".lean")
    return ".".join(parts)


def scan_file(lean_file: Path, file_module: str) -> list[tuple[str, str]]:
    """Parse one .lean file, return list of (full_name, file_module)."""
    results = []
    ns_stack = []  # list of (kind, name)

    with open(lean_file, "r", errors="replace") as f:
        in_block_comment = False
        for line in f:
            stripped = line.strip()

            # Handle block comments (/- ... -/)
            if in_block_comment:
                if "-/" in stripped:
                    in_block_comment = False
                    stripped = stripped[stripped.index("-/") + 2:].strip()
                    if not stripped:
                        continue
                else:
                    continue

            if "/-" in stripped:
                # Check if block comment closes on same line
                before = stripped[:stripped.index("/-")]
                after = stripped[stripped.index("/-") + 2:]
                if "-/" in after:
                    # Single-line block comment; use text before it
                    stripped = before.strip()
                    if not stripped:
                        continue
                else:
                    in_block_comment = True
                    stripped = before.strip()
                    if not stripped:
                        continue

            # Skip line comments
            if stripped.startswith("--"):
                continue

            # Track namespace open
            m_ns = pat_ns_open.match(stripped)
            if m_ns:
                ns_stack.append(("namespace", m_ns.group(1)))
                continue

            # Track section open (but don't confuse with declarations)
            m_sec = pat_section_open.match(stripped)
            if m_sec and not pat_decl_start.match(stripped):
                ns_stack.append(("section", m_sec.group(1).strip()))
                continue

            # Track `end Foo`
            m_end = pat_ns_close.match(stripped)
            if m_end and not stripped.startswith("end_"):
                end_name = m_end.group(1)
                for i in range(len(ns_stack) - 1, -1, -1):
                    if ns_stack[i][1] == end_name:
                        ns_stack.pop(i)
                        break
                continue

            # Bare `end`
            if pat_end_bare.match(stripped):
                if ns_stack:
                    ns_stack.pop()
                continue

            # Match declarations
            m_decl = pat_decl.match(stripped)
            if m_decl:
                raw_name = m_decl.group(1)

                # Build namespace prefix (only from namespace scopes, not sections)
                ns_parts = [name for (scope, name) in ns_stack
                            if scope == "namespace" and name]
                prefix = ".".join(ns_parts)

                if prefix:
                    full_name = f"{prefix}.{raw_name}"
                else:
                    full_name = raw_name

                results.append((full_name, file_module))

    return results


def build_mapping() -> list[tuple[str, str]]:
    """Scan all Mathlib .lean files and build the mapping."""
    print("Building declaration → file-module mapping ...")
    t0 = time.time()
    mathlib_root = MATHLIB_DIR  # parent of Mathlib/
    mathlib_src = MATHLIB_DIR / "Mathlib"

    all_results = []
    file_count = 0

    for lean_file in sorted(mathlib_src.rglob("*.lean")):
        file_count += 1
        file_module = path_to_module(lean_file, mathlib_root)
        results = scan_file(lean_file, file_module)
        all_results.extend(results)

    elapsed = time.time() - t0
    print(f"  Scanned {file_count} files in {elapsed:.1f}s")
    print(f"  Found {len(all_results)} declaration → file mappings")

    # Check for duplicates (same decl name in multiple files)
    seen = {}
    dupes = 0
    for name, mod in all_results:
        if name in seen and seen[name] != mod:
            dupes += 1
        seen[name] = mod
    print(f"  Unique declarations: {len(seen)}")
    if dupes:
        print(f"  Cross-file name collisions: {dupes}")

    return all_results


def save_csv(mapping: list[tuple[str, str]]):
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "declaration_to_file_module.csv"
    with open(out_path, "w") as f:
        f.write("declaration_full_name,file_module\n")
        for name, mod in mapping:
            # Escape commas in names (shouldn't happen but be safe)
            f.write(f"{name},{mod}\n")
    print(f"  Saved to {out_path}")
    return out_path


def main():
    mapping = build_mapping()
    save_csv(mapping)


if __name__ == "__main__":
    main()
