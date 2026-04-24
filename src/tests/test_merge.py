#!/usr/bin/env python3
"""Tests for data fusion and validation.

Tests the merge functionality that combines nodes.csv and edges.csv
and validates consistency.
"""

import pytest
import io


class TestEdgeValidation:
    """Test edge validation against nodes."""

    def test_all_edges_reference_valid_nodes(self):
        """All edge sources and targets should exist in nodes."""
        from parser.merge import validate_edges

        nodes = [
            {"name": "A", "kind": "theorem", "module": "M"},
            {"name": "B", "kind": "definition", "module": "M"},
            {"name": "C", "kind": "theorem", "module": "M"},
        ]
        edges = [
            {"source": "A", "target": "B", "is_explicit": True, "is_simplifier": False},
            {"source": "C", "target": "B", "is_explicit": False, "is_simplifier": True},
        ]

        result = validate_edges(nodes, edges)
        assert result["valid"] is True
        assert result["missing_sources"] == set()
        assert result["missing_targets"] == set()

    def test_missing_source_detected(self):
        """Edges with unknown source should be flagged."""
        from parser.merge import validate_edges

        nodes = [
            {"name": "A", "kind": "theorem", "module": "M"},
            {"name": "B", "kind": "definition", "module": "M"},
        ]
        edges = [
            {"source": "X", "target": "B", "is_explicit": True, "is_simplifier": False},
        ]

        result = validate_edges(nodes, edges)
        assert result["valid"] is False
        assert "X" in result["missing_sources"]

    def test_missing_target_detected(self):
        """Edges with unknown target should be flagged."""
        from parser.merge import validate_edges

        nodes = [
            {"name": "A", "kind": "theorem", "module": "M"},
        ]
        edges = [
            {"source": "A", "target": "Y", "is_explicit": True, "is_simplifier": False},
        ]

        result = validate_edges(nodes, edges)
        assert result["valid"] is False
        assert "Y" in result["missing_targets"]

    def test_multiple_missing_references(self):
        """Multiple missing references should all be reported."""
        from parser.merge import validate_edges

        nodes = [
            {"name": "A", "kind": "theorem", "module": "M"},
        ]
        edges = [
            {"source": "X", "target": "A", "is_explicit": True, "is_simplifier": False},
            {"source": "A", "target": "Y", "is_explicit": False, "is_simplifier": False},
            {"source": "Z", "target": "W", "is_explicit": False, "is_simplifier": True},
        ]

        result = validate_edges(nodes, edges)
        assert result["valid"] is False
        assert result["missing_sources"] == {"X", "Z"}
        assert result["missing_targets"] == {"Y", "W"}


class TestStatistics:
    """Test statistics computation."""

    def test_node_statistics(self):
        """Compute node statistics by kind."""
        from parser.merge import compute_statistics

        nodes = [
            {"name": "A", "kind": "theorem", "module": "M1"},
            {"name": "B", "kind": "theorem", "module": "M1"},
            {"name": "C", "kind": "definition", "module": "M2"},
            {"name": "D", "kind": "axiom", "module": "M2"},
            {"name": "E", "kind": "theorem", "module": "M3"},
        ]
        edges = []

        stats = compute_statistics(nodes, edges)

        assert stats["total_nodes"] == 5
        assert stats["nodes_by_kind"]["theorem"] == 3
        assert stats["nodes_by_kind"]["definition"] == 1
        assert stats["nodes_by_kind"]["axiom"] == 1

    def test_edge_statistics(self):
        """Compute edge statistics by type."""
        from parser.merge import compute_statistics

        nodes = [
            {"name": "A", "kind": "theorem", "module": "M"},
            {"name": "B", "kind": "definition", "module": "M"},
        ]
        edges = [
            {"source": "A", "target": "B", "is_explicit": True, "is_simplifier": False},
            {"source": "A", "target": "B", "is_explicit": False, "is_simplifier": True},
            {"source": "A", "target": "B", "is_explicit": False, "is_simplifier": False},
            {"source": "A", "target": "B", "is_explicit": True, "is_simplifier": False},
        ]

        stats = compute_statistics(nodes, edges)

        assert stats["total_edges"] == 4
        assert stats["explicit_edges"] == 2
        assert stats["simplifier_edges"] == 1
        assert stats["direct_edges"] == 1  # neither explicit nor simplifier

    def test_module_statistics(self):
        """Compute statistics by module."""
        from parser.merge import compute_statistics

        nodes = [
            {"name": "A", "kind": "theorem", "module": "Mathlib.Algebra"},
            {"name": "B", "kind": "theorem", "module": "Mathlib.Algebra"},
            {"name": "C", "kind": "definition", "module": "Mathlib.Data"},
        ]
        edges = []

        stats = compute_statistics(nodes, edges)

        assert stats["nodes_by_module"]["Mathlib.Algebra"] == 2
        assert stats["nodes_by_module"]["Mathlib.Data"] == 1


class TestEmptyData:
    """Test handling of empty data."""

    def test_empty_nodes(self):
        """Empty nodes should return valid result with empty sets."""
        from parser.merge import validate_edges, compute_statistics

        result = validate_edges([], [])
        assert result["valid"] is True

        stats = compute_statistics([], [])
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_nodes_only_no_edges(self):
        """Nodes without edges should be valid."""
        from parser.merge import validate_edges

        nodes = [
            {"name": "A", "kind": "theorem", "module": "M"},
        ]

        result = validate_edges(nodes, [])
        assert result["valid"] is True


class TestReportGeneration:
    """Test statistics report generation."""

    def test_generate_report_format(self):
        """Report should be well-formatted markdown."""
        from parser.merge import generate_report

        nodes = [
            {"name": "A", "kind": "theorem", "module": "M"},
            {"name": "B", "kind": "definition", "module": "M"},
        ]
        edges = [
            {"source": "A", "target": "B", "is_explicit": True, "is_simplifier": False},
        ]

        report = generate_report(nodes, edges)

        assert "# " in report  # Has heading
        assert "theorem" in report
        assert "definition" in report
        assert "1" in report  # Edge count

    def test_report_includes_validation_status(self):
        """Report should include validation status."""
        from parser.merge import generate_report

        nodes = [{"name": "A", "kind": "theorem", "module": "M"}]
        edges = [{"source": "A", "target": "X", "is_explicit": False, "is_simplifier": False}]

        report = generate_report(nodes, edges)

        # Should mention validation failure
        assert "missing" in report.lower() or "invalid" in report.lower() or "warning" in report.lower()


class TestCSVLoading:
    """Test CSV loading utilities."""

    def test_load_nodes_csv(self):
        """Load nodes from CSV."""
        from parser.merge import load_nodes_csv

        csv_content = """name,kind,module
A,theorem,M1
B,definition,M2
"""
        nodes = load_nodes_csv(io.StringIO(csv_content))
        assert len(nodes) == 2
        assert nodes[0]["name"] == "A"
        assert nodes[0]["kind"] == "theorem"

    def test_load_edges_csv(self):
        """Load edges from CSV."""
        from parser.merge import load_edges_csv

        csv_content = """source,target,is_explicit,is_simplifier
A,B,True,False
C,D,False,True
"""
        edges = load_edges_csv(io.StringIO(csv_content))
        assert len(edges) == 2
        assert edges[0]["source"] == "A"
        assert edges[0]["is_explicit"] is True
        assert edges[1]["is_simplifier"] is True
