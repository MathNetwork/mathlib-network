#!/usr/bin/env python3
"""Tests for lean4export NDJSON → nodes.csv parser.

Based on lean4export format v3.0.0:
https://github.com/leanprover/lean4export/blob/master/format_ndjson.md
"""

import pytest
import csv
import io
from pathlib import Path


class TestNameResolution:
    """Test name table building and resolution."""

    def test_resolve_simple_name(self):
        """Name.str with pre=0 (anonymous) should give just the string."""
        from parser.from_lean4export import parse_ndjson

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Nat"}, "in": 1}
{"axiomInfo": {"name": 1, "levelParams": [], "type": 1, "isUnsafe": false}}
"""
        nodes = parse_ndjson(io.StringIO(ndjson))
        assert len(nodes) == 1
        assert nodes[0]["name"] == "Nat"

    def test_resolve_qualified_name(self):
        """Name.str chain should produce qualified name like Mathlib.Data.Nat."""
        from parser.from_lean4export import parse_ndjson

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Mathlib"}, "in": 1}
{"str": {"pre": 1, "str": "Data"}, "in": 2}
{"str": {"pre": 2, "str": "Nat"}, "in": 3}
{"str": {"pre": 3, "str": "add_zero"}, "in": 4}
{"thm": [{"name": 4, "levelParams": [], "type": 1, "value": 1, "all": [4]}]}
"""
        nodes = parse_ndjson(io.StringIO(ndjson))
        assert len(nodes) == 1
        assert nodes[0]["name"] == "Mathlib.Data.Nat.add_zero"
        assert nodes[0]["module"] == "Mathlib.Data.Nat"


class TestTheoremParsing:
    """Test parsing of theorem declarations."""

    def test_parse_single_theorem(self):
        """thm entry should produce kind='theorem'."""
        from parser.from_lean4export import parse_ndjson

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Mathlib"}, "in": 1}
{"str": {"pre": 1, "str": "test_thm"}, "in": 2}
{"thm": [{"name": 2, "levelParams": [], "type": 1, "value": 1, "all": [2]}]}
"""
        nodes = parse_ndjson(io.StringIO(ndjson))
        assert len(nodes) == 1
        assert nodes[0]["kind"] == "theorem"

    def test_parse_mutual_theorems(self):
        """thm array with multiple entries should produce multiple theorems."""
        from parser.from_lean4export import parse_ndjson

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Mathlib"}, "in": 1}
{"str": {"pre": 1, "str": "thm1"}, "in": 2}
{"str": {"pre": 1, "str": "thm2"}, "in": 3}
{"thm": [{"name": 2, "levelParams": [], "type": 1, "value": 1, "all": [2, 3]}, {"name": 3, "levelParams": [], "type": 1, "value": 1, "all": [2, 3]}]}
"""
        nodes = parse_ndjson(io.StringIO(ndjson))
        assert len(nodes) == 2
        assert all(n["kind"] == "theorem" for n in nodes)


class TestDefinitionParsing:
    """Test parsing of definition declarations."""

    def test_parse_regular_definition(self):
        """def with hints='regular' should produce kind='definition'."""
        from parser.from_lean4export import parse_ndjson

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Mathlib"}, "in": 1}
{"str": {"pre": 1, "str": "myDef"}, "in": 2}
{"def": [{"name": 2, "levelParams": [], "type": 1, "value": 1, "hints": {"regular": 0}, "safety": "safe", "all": [2]}]}
"""
        nodes = parse_ndjson(io.StringIO(ndjson))
        assert len(nodes) == 1
        assert nodes[0]["kind"] == "definition"

    def test_parse_abbrev(self):
        """def with hints='abbrev' should produce kind='abbrev'."""
        from parser.from_lean4export import parse_ndjson

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Mathlib"}, "in": 1}
{"str": {"pre": 1, "str": "myAbbrev"}, "in": 2}
{"def": [{"name": 2, "levelParams": [], "type": 1, "value": 1, "hints": "abbrev", "safety": "safe", "all": [2]}]}
"""
        nodes = parse_ndjson(io.StringIO(ndjson))
        assert len(nodes) == 1
        assert nodes[0]["kind"] == "abbrev"

    def test_parse_opaque_in_def_block(self):
        """OpaqueVal in def block (has isUnsafe instead of hints) should produce kind='opaque'."""
        from parser.from_lean4export import parse_ndjson

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Mathlib"}, "in": 1}
{"str": {"pre": 1, "str": "myOpaque"}, "in": 2}
{"def": [{"name": 2, "levelParams": [], "type": 1, "value": 1, "isUnsafe": false, "all": [2]}]}
"""
        nodes = parse_ndjson(io.StringIO(ndjson))
        assert len(nodes) == 1
        assert nodes[0]["kind"] == "opaque"


class TestAxiomParsing:
    """Test parsing of axiom declarations."""

    def test_parse_axiom(self):
        """axiomInfo should produce kind='axiom'."""
        from parser.from_lean4export import parse_ndjson

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Mathlib"}, "in": 1}
{"str": {"pre": 1, "str": "myAxiom"}, "in": 2}
{"axiomInfo": {"name": 2, "levelParams": [], "type": 1, "isUnsafe": false}}
"""
        nodes = parse_ndjson(io.StringIO(ndjson))
        assert len(nodes) == 1
        assert nodes[0]["kind"] == "axiom"


class TestInductiveParsing:
    """Test parsing of inductive type declarations."""

    def test_parse_inductive_with_constructors(self):
        """inductive block should produce inductive, constructor, and recursor entries."""
        from parser.from_lean4export import parse_ndjson

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Mathlib"}, "in": 1}
{"str": {"pre": 1, "str": "MyType"}, "in": 2}
{"str": {"pre": 2, "str": "mk"}, "in": 3}
{"str": {"pre": 2, "str": "rec"}, "in": 4}
{"inductive": {"inductiveVals": [{"name": 2, "levelParams": [], "type": 1, "numParams": 0, "numIndices": 0, "all": [2], "ctors": [3], "numNested": 0, "isRec": false, "isUnsafe": false, "isReflexive": false}], "constructorVals": [{"name": 3, "levelParams": [], "type": 1, "induct": 2, "cidx": 0, "numParams": 0, "numFields": 0, "isUnsafe": false}], "recursorVals": [{"name": 4, "levelParams": [], "type": 1, "all": [2], "numParams": 0, "numIndices": 0, "numMotives": 1, "numMinors": 1, "rules": [], "k": false, "isUnsafe": false}]}}
"""
        nodes = parse_ndjson(io.StringIO(ndjson))
        kinds = {n["kind"] for n in nodes}
        assert "inductive" in kinds
        assert "constructor" in kinds
        assert "recursor" in kinds


class TestQuotParsing:
    """Test parsing of quotient declarations."""

    def test_parse_quot(self):
        """quotInfo should produce kind='quotient'."""
        from parser.from_lean4export import parse_ndjson

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Mathlib"}, "in": 1}
{"str": {"pre": 1, "str": "myQuot"}, "in": 2}
{"quotInfo": {"name": 2, "levelParams": [], "type": 1, "kind": "type"}}
"""
        nodes = parse_ndjson(io.StringIO(ndjson))
        assert len(nodes) == 1
        assert nodes[0]["kind"] == "quotient"


class TestModuleExtraction:
    """Test module path extraction from qualified names."""

    def test_module_from_qualified_name(self):
        """Module should be everything before the last dot."""
        from parser.from_lean4export import parse_ndjson

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Mathlib"}, "in": 1}
{"str": {"pre": 1, "str": "Algebra"}, "in": 2}
{"str": {"pre": 2, "str": "Group"}, "in": 3}
{"str": {"pre": 3, "str": "Basic"}, "in": 4}
{"str": {"pre": 4, "str": "mul_one"}, "in": 5}
{"thm": [{"name": 5, "levelParams": [], "type": 1, "value": 1, "all": [5]}]}
"""
        nodes = parse_ndjson(io.StringIO(ndjson))
        assert nodes[0]["module"] == "Mathlib.Algebra.Group.Basic"

    def test_top_level_name_module(self):
        """Single-component name should have empty module."""
        from parser.from_lean4export import parse_ndjson

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Nat"}, "in": 1}
{"axiomInfo": {"name": 1, "levelParams": [], "type": 1, "isUnsafe": false}}
"""
        nodes = parse_ndjson(io.StringIO(ndjson))
        assert nodes[0]["module"] == ""


class TestFiltering:
    """Test namespace filtering."""

    def test_filter_mathlib_only(self):
        """Only Mathlib.* declarations should be included when filtering."""
        from parser.from_lean4export import parse_ndjson

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Mathlib"}, "in": 1}
{"str": {"pre": 1, "str": "test"}, "in": 2}
{"str": {"pre": 0, "str": "Std"}, "in": 3}
{"str": {"pre": 3, "str": "test"}, "in": 4}
{"str": {"pre": 0, "str": "Init"}, "in": 5}
{"str": {"pre": 5, "str": "test"}, "in": 6}
{"thm": [{"name": 2, "levelParams": [], "type": 1, "value": 1, "all": [2]}]}
{"thm": [{"name": 4, "levelParams": [], "type": 1, "value": 1, "all": [4]}]}
{"thm": [{"name": 6, "levelParams": [], "type": 1, "value": 1, "all": [6]}]}
"""
        nodes = parse_ndjson(io.StringIO(ndjson), filter_mathlib=True)
        assert len(nodes) == 1
        assert nodes[0]["name"] == "Mathlib.test"

    def test_no_filter(self):
        """Without filtering, all namespaces should be included."""
        from parser.from_lean4export import parse_ndjson

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Mathlib"}, "in": 1}
{"str": {"pre": 1, "str": "test"}, "in": 2}
{"str": {"pre": 0, "str": "Std"}, "in": 3}
{"str": {"pre": 3, "str": "test"}, "in": 4}
{"thm": [{"name": 2, "levelParams": [], "type": 1, "value": 1, "all": [2]}]}
{"thm": [{"name": 4, "levelParams": [], "type": 1, "value": 1, "all": [4]}]}
"""
        nodes = parse_ndjson(io.StringIO(ndjson), filter_mathlib=False)
        assert len(nodes) == 2


class TestCSVOutput:
    """Test CSV output format."""

    def test_csv_has_correct_columns(self):
        """CSV should have name, kind, module columns."""
        from parser.from_lean4export import parse_ndjson, write_nodes_csv

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Mathlib"}, "in": 1}
{"str": {"pre": 1, "str": "test"}, "in": 2}
{"thm": [{"name": 2, "levelParams": [], "type": 1, "value": 1, "all": [2]}]}
"""
        nodes = parse_ndjson(io.StringIO(ndjson))
        output = io.StringIO()
        write_nodes_csv(nodes, output)
        output.seek(0)
        reader = csv.DictReader(output)
        assert reader.fieldnames == ["name", "kind", "module"]

    def test_csv_escapes_special_chars(self):
        """CSV should properly escape commas and quotes in names."""
        from parser.from_lean4export import write_nodes_csv

        nodes = [{"name": 'test,"name', "kind": "theorem", "module": "Test"}]
        output = io.StringIO()
        write_nodes_csv(nodes, output)
        output.seek(0)
        reader = csv.DictReader(output)
        row = next(reader)
        assert row["name"] == 'test,"name'


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input(self):
        """Empty input should return empty list."""
        from parser.from_lean4export import parse_ndjson

        nodes = parse_ndjson(io.StringIO(""))
        assert nodes == []

    def test_meta_only_input(self):
        """Input with only meta should return empty list."""
        from parser.from_lean4export import parse_ndjson

        ndjson = '{"meta": {"format": {"version": "3.0.0"}}}\n'
        nodes = parse_ndjson(io.StringIO(ndjson))
        assert nodes == []

    def test_malformed_json_line_skipped(self):
        """Malformed JSON lines should be skipped with warning."""
        from parser.from_lean4export import parse_ndjson

        ndjson = """{"meta": {"format": {"version": "3.0.0"}}}
{"str": {"pre": 0, "str": "Mathlib"}, "in": 1}
{this is not valid json}
{"str": {"pre": 1, "str": "test"}, "in": 2}
{"thm": [{"name": 2, "levelParams": [], "type": 1, "value": 1, "all": [2]}]}
"""
        nodes = parse_ndjson(io.StringIO(ndjson))
        assert len(nodes) == 1  # Should still parse the valid theorem
