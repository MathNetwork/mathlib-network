#!/usr/bin/env python3
"""Tests for the import graph analysis functions.

Uses small hand-crafted graphs to verify each analysis function,
plus spot-checks against the real Mathlib graph.
"""

import tempfile
from pathlib import Path

import networkx as nx
import pytest

from analysis.import_graph_utils import (
    build_import_graph,
    centrality_stats,
    connectivity_stats,
    dag_stats,
    degree_stats,
    lean_path_to_module,
    namespace_stats,
    parse_imports,
    top_level_ns,
)

MATHLIB_ROOT = Path(__file__).resolve().parent.parent.parent / "mathlib4" / "Mathlib"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def write_temp(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


def make_diamond() -> nx.DiGraph:
    """A -> B, A -> C, B -> D, C -> D (diamond DAG)."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    return G


def make_chain(n: int = 5) -> nx.DiGraph:
    """Linear chain: 0 -> 1 -> 2 -> ... -> n-1."""
    G = nx.DiGraph()
    for i in range(n - 1):
        G.add_edge(str(i), str(i + 1))
    return G


def make_namespace_graph() -> nx.DiGraph:
    """Graph with clear namespace structure for testing."""
    G = nx.DiGraph()
    G.add_edges_from([
        ("Mathlib.Algebra.Group.Defs", "Mathlib.Algebra.Notation.Defs"),
        ("Mathlib.Algebra.Group.Defs", "Mathlib.Data.Nat.Notation"),
        ("Mathlib.Algebra.Notation.Defs", "Mathlib.Init"),
        ("Mathlib.Data.Nat.Notation", "Mathlib.Init"),
        ("Mathlib.Analysis.Normed.Basic", "Mathlib.Algebra.Group.Defs"),
        ("Mathlib.Analysis.Normed.Basic", "Mathlib.Data.Nat.Notation"),
    ])
    return G


# ===================================================================
# parse_imports tests
# ===================================================================

class TestParseImports:
    def test_basic_imports(self):
        p = write_temp("module\n\npublic import Mathlib.Init\npublic import Mathlib.Data.Nat.Notation\n")
        assert parse_imports(p) == ["Mathlib.Init", "Mathlib.Data.Nat.Notation"]

    def test_block_comment_skipped(self):
        p = write_temp("/-\nCopyright\n-/\nmodule\n\nimport Mathlib.Init\n")
        assert parse_imports(p) == ["Mathlib.Init"]

    def test_imports_inside_block_comment_ignored(self):
        p = write_temp("module\n\nimport Mathlib.Real\n\n/-!\nimport Mathlib.Fake\n-/\ndef x := 1\n")
        assert parse_imports(p) == ["Mathlib.Real"]

    def test_nested_block_comments(self):
        p = write_temp("/- outer /- inner -/ still outer -/\nmodule\n\nimport Mathlib.A\n")
        assert parse_imports(p) == ["Mathlib.A"]

    def test_stop_at_declaration(self):
        p = write_temp("module\n\nimport Mathlib.A\nimport Mathlib.B\n\ntheorem x : True := trivial\n\nimport Mathlib.C\n")
        assert parse_imports(p) == ["Mathlib.A", "Mathlib.B"]

    def test_empty_file(self):
        p = write_temp("")
        assert parse_imports(p) == []

    def test_line_comment_between_imports(self):
        p = write_temp("module\n\nimport Mathlib.A\n-- comment\nimport Mathlib.B\n")
        assert parse_imports(p) == ["Mathlib.A", "Mathlib.B"]

    def test_public_and_meta_import(self):
        p = write_temp("module\n\npublic meta import Lean.Elab\npublic import Mathlib.X\n")
        assert parse_imports(p) == ["Lean.Elab", "Mathlib.X"]


# ===================================================================
# lean_path_to_module tests
# ===================================================================

class TestLeanPathToModule:
    def test_simple(self):
        p = Path("/foo/bar/Mathlib/Algebra/Group/Defs.lean")
        assert lean_path_to_module(p, Path("/foo/bar")) == "Mathlib.Algebra.Group.Defs"

    def test_single_depth(self):
        p = Path("/repo/Mathlib/Init.lean")
        assert lean_path_to_module(p, Path("/repo")) == "Mathlib.Init"


# ===================================================================
# top_level_ns tests
# ===================================================================

class TestTopLevelNs:
    def test_normal(self):
        assert top_level_ns("Mathlib.Algebra.Group.Defs") == "Algebra"

    def test_short(self):
        assert top_level_ns("Mathlib.Init") == "Init"

    def test_no_dot(self):
        assert top_level_ns("Mathlib") == "Mathlib"


# ===================================================================
# degree_stats tests
# ===================================================================

class TestDegreeStats:
    def test_diamond(self):
        G = make_diamond()
        stats = degree_stats(G)
        # A: in=0, B: in=1, C: in=1, D: in=2
        assert stats["in_degree"]["max"] == 2
        assert stats["in_degree"]["mean"] == 1.0
        # A: out=2, B: out=1, C: out=1, D: out=0
        assert stats["out_degree"]["max"] == 2
        assert stats["out_degree"]["mean"] == 1.0

    def test_chain(self):
        G = make_chain(5)
        stats = degree_stats(G)
        # 0: in=0, 1-3: in=1, 4: in=1 -> max=1
        assert stats["in_degree"]["max"] == 1
        # 0-3: out=1, 4: out=0 -> max=1
        assert stats["out_degree"]["max"] == 1

    def test_top20_order(self):
        G = make_diamond()
        stats = degree_stats(G)
        in_top = stats["in_degree"]["top_20"]
        assert in_top[0]["module"] == "D"
        assert in_top[0]["value"] == 2

    def test_isolated_node(self):
        G = nx.DiGraph()
        G.add_node("X")
        stats = degree_stats(G)
        assert stats["in_degree"]["max"] == 0
        assert stats["out_degree"]["max"] == 0


# ===================================================================
# dag_stats tests
# ===================================================================

class TestDagStats:
    def test_diamond_is_dag(self):
        G = make_diamond()
        stats = dag_stats(G)
        assert stats["is_dag"] is True

    def test_diamond_longest_path(self):
        G = make_diamond()
        stats = dag_stats(G)
        assert stats["longest_path_length"] == 2  # A->B->D or A->C->D

    def test_chain_longest_path(self):
        G = make_chain(6)
        stats = dag_stats(G)
        assert stats["longest_path_length"] == 5

    def test_sources_and_sinks(self):
        G = make_diamond()
        stats = dag_stats(G)
        assert stats["num_sources"] == 1  # A
        assert stats["num_sinks"] == 1    # D
        assert "A" in stats["sources"]
        assert "D" in stats["sinks"]

    def test_chain_sources_sinks(self):
        G = make_chain(5)
        stats = dag_stats(G)
        assert stats["num_sources"] == 1  # "0"
        assert stats["num_sinks"] == 1    # "4"

    def test_layer_count(self):
        G = make_diamond()
        stats = dag_stats(G)
        # Layers: {A}, {B, C}, {D} -> 3 layers
        assert stats["num_layers"] == 3
        assert sorted(stats["layer_sizes"]) == [1, 1, 2]

    def test_chain_layers(self):
        G = make_chain(4)
        stats = dag_stats(G)
        assert stats["num_layers"] == 4
        assert stats["layer_sizes"] == [1, 1, 1, 1]

    def test_wide_graph(self):
        G = nx.DiGraph()
        G.add_edges_from([("root", f"leaf{i}") for i in range(10)])
        stats = dag_stats(G)
        assert stats["num_layers"] == 2
        assert stats["layer_width_max"] == 10
        assert stats["num_sources"] == 1
        assert stats["num_sinks"] == 10


# ===================================================================
# namespace_stats tests
# ===================================================================

class TestNamespaceStats:
    def test_intra_vs_cross(self):
        G = make_namespace_graph()
        stats = namespace_stats(G)
        # Edges:
        #   Algebra -> Algebra (intra): Algebra.Group.Defs -> Algebra.Notation.Defs
        #   Algebra -> Data (cross):    Algebra.Group.Defs -> Data.Nat.Notation
        #   Algebra -> Init (cross):    Algebra.Notation.Defs -> Init
        #   Data -> Init (cross):       Data.Nat.Notation -> Init
        #   Analysis -> Algebra (cross): Analysis.Normed.Basic -> Algebra.Group.Defs
        #   Analysis -> Data (cross):    Analysis.Normed.Basic -> Data.Nat.Notation
        assert stats["intra_namespace_edges"] == 1
        assert stats["cross_namespace_edges"] == 5
        assert stats["intra_ratio"] == round(1 / 6, 4)

    def test_all_intra(self):
        G = nx.DiGraph()
        G.add_edges_from([
            ("Mathlib.Algebra.A", "Mathlib.Algebra.B"),
            ("Mathlib.Algebra.B", "Mathlib.Algebra.C"),
        ])
        stats = namespace_stats(G)
        assert stats["intra_namespace_edges"] == 2
        assert stats["cross_namespace_edges"] == 0
        assert stats["intra_ratio"] == 1.0

    def test_empty_graph(self):
        G = nx.DiGraph()
        G.add_node("Mathlib.X")
        stats = namespace_stats(G)
        assert stats["intra_namespace_edges"] == 0
        assert stats["cross_namespace_edges"] == 0
        assert stats["intra_ratio"] == 0

    def test_ns_matrix_structure(self):
        G = make_namespace_graph()
        stats = namespace_stats(G)
        matrix = stats["ns_matrix"]
        assert matrix["Algebra"]["Algebra"] == 1
        assert matrix["Algebra"]["Data"] == 1
        assert matrix["Algebra"]["Init"] == 1
        assert matrix["Analysis"]["Algebra"] == 1
        assert matrix["Analysis"]["Data"] == 1


# ===================================================================
# centrality_stats tests
# ===================================================================

class TestCentralityStats:
    def test_diamond_pagerank(self):
        G = make_diamond()
        stats = centrality_stats(G)
        # D should have highest PageRank (most pointed to)
        top_pr = stats["top_20_pagerank"]
        assert top_pr[0]["module"] == "D"

    def test_chain_betweenness(self):
        G = make_chain(5)
        stats = centrality_stats(G)
        # Middle nodes should have highest betweenness
        top_bc = stats["top_20_betweenness"]
        # Node "2" is in the middle of 0->1->2->3->4
        assert any(e["module"] == "2" for e in top_bc[:3])

    def test_overlap_structure(self):
        G = make_diamond()
        stats = centrality_stats(G)
        overlap = stats["overlap"]
        assert isinstance(overlap["all_three"], list)
        assert isinstance(overlap["in_degree_AND_pagerank"], list)

    def test_returns_raw_values(self):
        G = make_diamond()
        stats = centrality_stats(G)
        assert "pagerank" in stats
        assert "betweenness" in stats
        assert len(stats["pagerank"]) == 4
        assert all(0 <= v <= 1 for v in stats["pagerank"].values())


# ===================================================================
# connectivity_stats tests
# ===================================================================

class TestConnectivityStats:
    def test_connected_graph(self):
        G = make_diamond()
        stats = connectivity_stats(G)
        assert stats["num_weakly_connected_components"] == 1
        assert stats["largest_component_size"] == 4

    def test_disconnected_graph(self):
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("C", "D")])
        stats = connectivity_stats(G)
        assert stats["num_weakly_connected_components"] == 2
        assert stats["largest_component_size"] == 2

    def test_removal_increases_components(self):
        # Hub with high in-degree, connected to many chains
        # hub has in-degree 10; all others have in-degree <= 1
        G = nx.DiGraph()
        for i in range(10):
            G.add_edge(f"spoke{i}", "hub")
            G.add_edge(f"leaf{i}", f"spoke{i}")
        stats = connectivity_stats(G)
        assert stats["num_weakly_connected_components"] == 1
        # top-5 in-degree: hub (10) + 4 spokes (tied at 1)
        # Removing hub alone would split into 10 components;
        # removing hub + 4 spokes splits further
        assert stats["removed_top5_in_degree"]["num_components"] > 1

    def test_removal_nodes_listed(self):
        G = make_diamond()
        stats = connectivity_stats(G)
        assert len(stats["removed_top5_in_degree"]["nodes_removed"]) <= 5
        assert len(stats["removed_top5_betweenness"]["nodes_removed"]) <= 5


# ===================================================================
# Real Mathlib spot-checks
# ===================================================================

@pytest.mark.skipif(not MATHLIB_ROOT.exists(), reason="Mathlib repo not found")
class TestRealMathlib:
    def test_parse_data_nat_notation(self):
        p = MATHLIB_ROOT / "Data" / "Nat" / "Notation.lean"
        assert parse_imports(p) == ["Mathlib.Init"]

    def test_parse_algebra_group_defs_count(self):
        p = MATHLIB_ROOT / "Algebra" / "Group" / "Defs.lean"
        result = parse_imports(p)
        assert len(result) == 9
        assert "Mathlib.Data.Nat.Notation" in result
        assert "Mathlib.Data.Int.Notation" in result

    def test_parse_min_imports_no_leak(self):
        p = MATHLIB_ROOT / "Tactic" / "MinImports.lean"
        result = parse_imports(p)
        assert "Mathlib.Tactic.MinImports" not in result
        assert "Mathlib.Data.Sym.Sym2.Init" not in result

    def test_lean_path_to_module_real(self):
        p = MATHLIB_ROOT / "Algebra" / "Group" / "Defs.lean"
        assert lean_path_to_module(p, MATHLIB_ROOT.parent) == "Mathlib.Algebra.Group.Defs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
