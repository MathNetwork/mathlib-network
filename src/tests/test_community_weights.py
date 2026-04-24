#!/usr/bin/env python3
"""Test that directed-to-undirected weight merging for community detection is correct.

The bug: calling G.to_undirected() already creates undirected edges with weights,
then iterating G.edges() and adding weights again double-counts them.

The fix: build an empty undirected graph, then merge reciprocal edge weights manually.
"""

import networkx as nx
import pytest


def merge_to_undirected(G_directed):
    """Correctly merge a weighted directed graph into an undirected graph.

    For each pair (u, v), the undirected weight = sum of weights of u->v and v->u.
    """
    G_und = nx.Graph()
    G_und.add_nodes_from(G_directed.nodes())
    for u, v, d in G_directed.edges(data=True):
        w = d.get("weight", 1)
        if G_und.has_edge(u, v):
            G_und[u][v]["weight"] += w
        else:
            G_und.add_edge(u, v, weight=w)
    return G_und


class TestMergeToUndirected:
    """Verify correct weight merging when converting directed -> undirected."""

    def test_single_direction(self):
        """Edge A->B with weight 3 becomes undirected {A,B} with weight 3."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=3)
        U = merge_to_undirected(G)
        assert U.has_edge("A", "B")
        assert U["A"]["B"]["weight"] == 3

    def test_reciprocal_edges_sum(self):
        """A->B (w=2) and B->A (w=5) should merge to {A,B} with weight 7."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=2)
        G.add_edge("B", "A", weight=5)
        U = merge_to_undirected(G)
        assert U["A"]["B"]["weight"] == 7
        assert U.number_of_edges() == 1

    def test_default_weight_is_one(self):
        """Edges without explicit weight default to 1."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        G.add_edge("B", "A")
        U = merge_to_undirected(G)
        assert U["A"]["B"]["weight"] == 2

    def test_no_double_count_vs_to_undirected(self):
        """Demonstrate the bug: to_undirected() + manual add double-counts."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=3)
        G.add_edge("B", "A", weight=5)

        # Buggy approach (what the code was doing)
        G_buggy = G.to_undirected()
        for u, v, d in G.edges(data=True):
            if G_buggy.has_edge(u, v):
                G_buggy[u][v]["weight"] = G_buggy[u][v].get("weight", 0) + d.get("weight", 1)
            else:
                G_buggy.add_edge(u, v, weight=d.get("weight", 1))
        buggy_weight = G_buggy["A"]["B"]["weight"]

        # Correct approach
        G_correct = merge_to_undirected(G)
        correct_weight = G_correct["A"]["B"]["weight"]

        # The buggy version over-counts
        assert buggy_weight > correct_weight
        assert correct_weight == 8  # 3 + 5
        assert buggy_weight != 8   # confirms the bug exists

    def test_isolated_nodes_preserved(self):
        """Nodes with no edges should survive the conversion."""
        G = nx.DiGraph()
        G.add_node("X")
        G.add_edge("A", "B", weight=1)
        U = merge_to_undirected(G)
        assert "X" in U.nodes()
        assert U.number_of_nodes() == 3

    def test_triangle(self):
        """Three-node directed cycle: A->B->C->A, all weight 1."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=1)
        G.add_edge("B", "C", weight=1)
        G.add_edge("C", "A", weight=1)
        U = merge_to_undirected(G)
        assert U.number_of_edges() == 3
        # Each edge only has one direction, so weight stays 1
        assert U["A"]["B"]["weight"] == 1
        assert U["B"]["C"]["weight"] == 1
        assert U["C"]["A"]["weight"] == 1
