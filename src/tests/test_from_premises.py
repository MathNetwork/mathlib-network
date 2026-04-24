#!/usr/bin/env python3
"""Tests for lean-training-data premises → edges.csv parser.

Based on lean-training-data output format:
https://github.com/kim-em/lean-training-data

Format:
- Declarations separated by `---`
- First line of block = source declaration name
- Subsequent indented lines = dependencies
- Prefixes: `*` = explicit, `s` = simplifier, none = direct
"""

import pytest
import csv
import io


class TestBasicParsing:
    """Test basic premises parsing."""

    def test_parse_single_dependency(self):
        """Single dependency without prefix."""
        from parser.from_premises import parse_premises

        text = """---
Mathlib.Test.theorem1
  Mathlib.Test.dep1
---
"""
        edges = parse_premises(io.StringIO(text))
        assert len(edges) == 1
        assert edges[0]["source"] == "Mathlib.Test.theorem1"
        assert edges[0]["target"] == "Mathlib.Test.dep1"
        assert edges[0]["is_explicit"] is False
        assert edges[0]["is_simplifier"] is False

    def test_parse_multiple_dependencies(self):
        """Multiple dependencies for one declaration."""
        from parser.from_premises import parse_premises

        text = """---
Mathlib.Test.theorem1
  dep1
  dep2
  dep3
---
"""
        edges = parse_premises(io.StringIO(text))
        assert len(edges) == 3
        targets = {e["target"] for e in edges}
        assert targets == {"dep1", "dep2", "dep3"}


class TestPrefixParsing:
    """Test prefix handling."""

    def test_explicit_prefix(self):
        """* prefix should set is_explicit=True."""
        from parser.from_premises import parse_premises

        text = """---
Mathlib.Test.theorem1
* explicit_dep
---
"""
        edges = parse_premises(io.StringIO(text))
        assert len(edges) == 1
        assert edges[0]["target"] == "explicit_dep"
        assert edges[0]["is_explicit"] is True
        assert edges[0]["is_simplifier"] is False

    def test_simplifier_prefix(self):
        """s prefix should set is_simplifier=True."""
        from parser.from_premises import parse_premises

        text = """---
Mathlib.Test.theorem1
s simp_dep
---
"""
        edges = parse_premises(io.StringIO(text))
        assert len(edges) == 1
        assert edges[0]["target"] == "simp_dep"
        assert edges[0]["is_explicit"] is False
        assert edges[0]["is_simplifier"] is True

    def test_mixed_prefixes(self):
        """Mixed prefixes in one declaration."""
        from parser.from_premises import parse_premises

        text = """---
Mathlib.Test.theorem1
  direct_dep
* explicit_dep
s simp_dep
---
"""
        edges = parse_premises(io.StringIO(text))
        assert len(edges) == 3

        by_target = {e["target"]: e for e in edges}

        assert by_target["direct_dep"]["is_explicit"] is False
        assert by_target["direct_dep"]["is_simplifier"] is False

        assert by_target["explicit_dep"]["is_explicit"] is True
        assert by_target["explicit_dep"]["is_simplifier"] is False

        assert by_target["simp_dep"]["is_explicit"] is False
        assert by_target["simp_dep"]["is_simplifier"] is True


class TestRealWorldExample:
    """Test with real-world-like example from documentation."""

    def test_list_tofinset_example(self):
        """Parse example from lean-training-data README."""
        from parser.from_premises import parse_premises

        text = """---
List.toFinset.ext_iff
* congrArg
  List.instMembershipList
  Finset
  Finset.instMembershipFinset
* Membership.mem
  List.toFinset
* iff_self
  List
* Iff
* congrFun
* congr
  True
* of_eq_true
  Eq
* Eq.trans
  DecidableEq
* forall_congr
s List.mem_toFinset
s Finset.ext_iff
s propext
---
"""
        edges = parse_premises(io.StringIO(text))

        # Count by type
        explicit_count = sum(1 for e in edges if e["is_explicit"])
        simp_count = sum(1 for e in edges if e["is_simplifier"])
        direct_count = sum(1 for e in edges if not e["is_explicit"] and not e["is_simplifier"])

        assert explicit_count == 9  # Lines starting with *
        assert simp_count == 3  # Lines starting with s
        assert direct_count == 8  # Lines with just indentation

        # Check source is consistent
        assert all(e["source"] == "List.toFinset.ext_iff" for e in edges)


class TestMultipleDeclarations:
    """Test parsing multiple declarations."""

    def test_multiple_declarations(self):
        """Multiple declarations separated by ---."""
        from parser.from_premises import parse_premises

        text = """---
Mathlib.Test.theorem1
  dep1
---
Mathlib.Test.theorem2
  dep2
  dep3
---
Mathlib.Test.theorem3
* dep4
---
"""
        edges = parse_premises(io.StringIO(text))

        sources = {e["source"] for e in edges}
        assert sources == {"Mathlib.Test.theorem1", "Mathlib.Test.theorem2", "Mathlib.Test.theorem3"}

        # Check counts per source
        by_source = {}
        for e in edges:
            by_source.setdefault(e["source"], []).append(e)

        assert len(by_source["Mathlib.Test.theorem1"]) == 1
        assert len(by_source["Mathlib.Test.theorem2"]) == 2
        assert len(by_source["Mathlib.Test.theorem3"]) == 1


class TestZeroDependencies:
    """Test declarations with no dependencies."""

    def test_declaration_with_no_deps(self):
        """Declaration with no dependencies should produce no edges."""
        from parser.from_premises import parse_premises

        text = """---
Mathlib.Test.axiom1
---
Mathlib.Test.theorem1
  dep1
---
"""
        edges = parse_premises(io.StringIO(text))
        assert len(edges) == 1
        assert edges[0]["source"] == "Mathlib.Test.theorem1"


class TestCSVOutput:
    """Test CSV output format."""

    def test_csv_has_correct_columns(self):
        """CSV should have source, target, is_explicit, is_simplifier columns."""
        from parser.from_premises import parse_premises, write_edges_csv

        text = """---
source1
  target1
---
"""
        edges = parse_premises(io.StringIO(text))
        output = io.StringIO()
        write_edges_csv(edges, output)
        output.seek(0)
        reader = csv.DictReader(output)
        assert reader.fieldnames == ["source", "target", "is_explicit", "is_simplifier"]

    def test_csv_boolean_format(self):
        """Boolean columns should be lowercase true/false."""
        from parser.from_premises import parse_premises, write_edges_csv

        text = """---
source1
* target1
s target2
---
"""
        edges = parse_premises(io.StringIO(text))
        output = io.StringIO()
        write_edges_csv(edges, output)
        output.seek(0)
        content = output.read()
        assert "true" in content.lower() or "True" in content
        assert "false" in content.lower() or "False" in content


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input(self):
        """Empty input should return empty list."""
        from parser.from_premises import parse_premises

        edges = parse_premises(io.StringIO(""))
        assert edges == []

    def test_only_separators(self):
        """Input with only --- should return empty list."""
        from parser.from_premises import parse_premises

        edges = parse_premises(io.StringIO("---\n---\n---\n"))
        assert edges == []

    def test_whitespace_handling(self):
        """Extra whitespace should be handled gracefully."""
        from parser.from_premises import parse_premises

        text = """---

Mathlib.Test.theorem1
  dep1

---
"""
        edges = parse_premises(io.StringIO(text))
        # Should still parse the one valid edge
        assert len(edges) >= 1

    def test_no_leading_separator(self):
        """Input without leading --- should still parse."""
        from parser.from_premises import parse_premises

        text = """Mathlib.Test.theorem1
  dep1
---
"""
        edges = parse_premises(io.StringIO(text))
        assert len(edges) == 1

    def test_trailing_content_after_last_separator(self):
        """Content after last --- should be parsed."""
        from parser.from_premises import parse_premises

        text = """---
Mathlib.Test.theorem1
  dep1
---
Mathlib.Test.theorem2
  dep2
"""
        edges = parse_premises(io.StringIO(text))
        sources = {e["source"] for e in edges}
        assert "Mathlib.Test.theorem2" in sources


class TestFiltering:
    """Test filtering options."""

    def test_filter_mathlib_sources(self):
        """Option to filter only Mathlib.* sources."""
        from parser.from_premises import parse_premises

        text = """---
Mathlib.Test.theorem1
  dep1
---
Std.Test.theorem2
  dep2
---
Init.Test.theorem3
  dep3
---
"""
        edges = parse_premises(io.StringIO(text), filter_mathlib_source=True)
        sources = {e["source"] for e in edges}
        assert sources == {"Mathlib.Test.theorem1"}

    def test_no_filter(self):
        """Without filter, all sources should be included."""
        from parser.from_premises import parse_premises

        text = """---
Mathlib.Test.theorem1
  dep1
---
Std.Test.theorem2
  dep2
---
"""
        edges = parse_premises(io.StringIO(text), filter_mathlib_source=False)
        sources = {e["source"] for e in edges}
        assert len(sources) == 2
