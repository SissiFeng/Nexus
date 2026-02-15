"""Comprehensive tests for causal structure recovery metrics.

Tests cover structural_hamming_distance, edge_precision_recall,
orientation_accuracy, and evaluate_structure_recovery from
optimization_copilot.causal.metrics.
"""

from __future__ import annotations

import pytest

from optimization_copilot.causal.models import CausalEdge, CausalGraph, CausalNode
from optimization_copilot.causal.metrics import (
    StructureRecoveryMetrics,
    edge_precision_recall,
    evaluate_structure_recovery,
    orientation_accuracy,
    structural_hamming_distance,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_graph(nodes: list[str], edges: list[tuple[str, str]]) -> CausalGraph:
    """Build a CausalGraph from node names and (source, target) pairs."""
    g = CausalGraph()
    for n in nodes:
        g.add_node(CausalNode(name=n))
    for src, tgt in edges:
        g.add_edge(CausalEdge(source=src, target=tgt))
    return g


# ===================================================================
# 1. Structural Hamming Distance (SHD)
# ===================================================================

class TestStructuralHammingDistance:
    """Tests for structural_hamming_distance."""

    def test_identical_graphs_shd_zero(self):
        """Identical graphs should have SHD = 0."""
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C")]
        true_g = _make_graph(nodes, edges)
        pred_g = _make_graph(nodes, edges)
        assert structural_hamming_distance(pred_g, true_g) == 0

    def test_one_missing_edge(self):
        """Predicted graph missing one edge -> SHD = 1."""
        nodes = ["A", "B", "C"]
        true_g = _make_graph(nodes, [("A", "B"), ("B", "C")])
        pred_g = _make_graph(nodes, [("A", "B")])
        assert structural_hamming_distance(pred_g, true_g) == 1

    def test_one_extra_edge(self):
        """Predicted graph with one extra edge -> SHD = 1."""
        nodes = ["A", "B", "C"]
        true_g = _make_graph(nodes, [("A", "B")])
        pred_g = _make_graph(nodes, [("A", "B"), ("B", "C")])
        assert structural_hamming_distance(pred_g, true_g) == 1

    def test_reversed_edge(self):
        """One reversed edge -> SHD = 1."""
        nodes = ["A", "B"]
        true_g = _make_graph(nodes, [("A", "B")])
        pred_g = _make_graph(nodes, [("B", "A")])
        assert structural_hamming_distance(pred_g, true_g) == 1

    def test_completely_different_graphs(self):
        """Graphs with no overlapping edges should have SHD = total unique edges."""
        nodes = ["A", "B", "C", "D"]
        true_g = _make_graph(nodes, [("A", "B"), ("C", "D")])
        pred_g = _make_graph(nodes, [("A", "C"), ("B", "D")])
        # True has A-B and C-D, predicted has A-C and B-D.
        # Missing: A-B, C-D (2). Extra: A-C, B-D (2). Total SHD = 4.
        assert structural_hamming_distance(pred_g, true_g) == 4

    def test_both_empty_graphs(self):
        """Two empty graphs (nodes only) -> SHD = 0."""
        nodes = ["A", "B", "C"]
        true_g = _make_graph(nodes, [])
        pred_g = _make_graph(nodes, [])
        assert structural_hamming_distance(pred_g, true_g) == 0

    def test_empty_vs_nonempty(self):
        """Empty predicted vs non-empty true -> SHD = number of true edges."""
        nodes = ["A", "B", "C"]
        true_g = _make_graph(nodes, [("A", "B"), ("B", "C"), ("A", "C")])
        pred_g = _make_graph(nodes, [])
        assert structural_hamming_distance(pred_g, true_g) == 3

    def test_nonempty_vs_empty(self):
        """Non-empty predicted vs empty true -> SHD = number of predicted edges."""
        nodes = ["A", "B", "C"]
        true_g = _make_graph(nodes, [])
        pred_g = _make_graph(nodes, [("A", "B"), ("B", "C")])
        assert structural_hamming_distance(pred_g, true_g) == 2

    def test_multiple_errors_combined(self):
        """Mix of missing, extra, and reversed edges."""
        nodes = ["A", "B", "C", "D"]
        true_g = _make_graph(nodes, [("A", "B"), ("B", "C"), ("C", "D")])
        # A->B correct, B->C reversed to C->B, C->D missing, A->D extra
        pred_g = _make_graph(nodes, [("A", "B"), ("C", "B"), ("A", "D")])
        # B-C: reversed -> 1. C-D: missing -> 1. A-D: extra -> 1. Total = 3.
        assert structural_hamming_distance(pred_g, true_g) == 3

    def test_single_node_graphs(self):
        """Graphs with a single node and no edges -> SHD = 0."""
        true_g = _make_graph(["X"], [])
        pred_g = _make_graph(["X"], [])
        assert structural_hamming_distance(pred_g, true_g) == 0

    def test_different_node_sets(self):
        """Graphs with partially overlapping node sets."""
        true_g = _make_graph(["A", "B", "C"], [("A", "B")])
        pred_g = _make_graph(["A", "B", "D"], [("A", "B")])
        # A-B is in both and matches. No extra skeleton edges. SHD = 0.
        assert structural_hamming_distance(pred_g, true_g) == 0

    def test_shd_symmetric_for_swap(self):
        """SHD should be the same regardless of which is predicted vs true
        for missing/extra edge cases (they map to each other)."""
        nodes = ["A", "B", "C"]
        g1 = _make_graph(nodes, [("A", "B"), ("B", "C")])
        g2 = _make_graph(nodes, [("A", "B")])
        # g1 predicted, g2 true: one extra edge -> 1
        # g2 predicted, g1 true: one missing edge -> 1
        assert structural_hamming_distance(g1, g2) == structural_hamming_distance(g2, g1)


# ===================================================================
# 2. Edge Precision / Recall / F1
# ===================================================================

class TestEdgePrecisionRecall:
    """Tests for edge_precision_recall (skeleton-level)."""

    def test_perfect_match(self):
        """Identical graphs -> (1.0, 1.0, 1.0)."""
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C")]
        true_g = _make_graph(nodes, edges)
        pred_g = _make_graph(nodes, edges)
        precision, recall, f1 = edge_precision_recall(pred_g, true_g)
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_predicted_is_subset(self):
        """Predicted edges are a subset -> precision=1.0, recall<1.0."""
        nodes = ["A", "B", "C"]
        true_g = _make_graph(nodes, [("A", "B"), ("B", "C")])
        pred_g = _make_graph(nodes, [("A", "B")])
        precision, recall, f1 = edge_precision_recall(pred_g, true_g)
        assert precision == 1.0
        assert recall == pytest.approx(0.5)
        # F1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 2/3
        assert f1 == pytest.approx(2.0 / 3.0)

    def test_predicted_is_superset(self):
        """Predicted edges are a superset -> precision<1.0, recall=1.0."""
        nodes = ["A", "B", "C"]
        true_g = _make_graph(nodes, [("A", "B")])
        pred_g = _make_graph(nodes, [("A", "B"), ("B", "C")])
        precision, recall, f1 = edge_precision_recall(pred_g, true_g)
        assert precision == pytest.approx(0.5)
        assert recall == 1.0
        # F1 = 2 * 0.5 * 1.0 / (0.5 + 1.0) = 2/3
        assert f1 == pytest.approx(2.0 / 3.0)

    def test_no_overlap(self):
        """No overlapping skeleton edges -> (0.0, 0.0, 0.0)."""
        nodes = ["A", "B", "C", "D"]
        true_g = _make_graph(nodes, [("A", "B")])
        pred_g = _make_graph(nodes, [("C", "D")])
        precision, recall, f1 = edge_precision_recall(pred_g, true_g)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_both_empty(self):
        """Both graphs have no edges -> (1.0, 1.0, 1.0)."""
        nodes = ["A", "B"]
        true_g = _make_graph(nodes, [])
        pred_g = _make_graph(nodes, [])
        precision, recall, f1 = edge_precision_recall(pred_g, true_g)
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_empty_predicted_nonempty_true(self):
        """No predicted edges, true has edges -> precision=0.0, recall=0.0."""
        nodes = ["A", "B"]
        true_g = _make_graph(nodes, [("A", "B")])
        pred_g = _make_graph(nodes, [])
        precision, recall, f1 = edge_precision_recall(pred_g, true_g)
        # tp=0, fp=0, fn=1. precision=0/(0+0)=0.0, recall=0/(0+1)=0.0
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_nonempty_predicted_empty_true(self):
        """Predicted has edges, true is empty -> precision=0.0, recall=0.0."""
        nodes = ["A", "B"]
        true_g = _make_graph(nodes, [])
        pred_g = _make_graph(nodes, [("A", "B")])
        precision, recall, f1 = edge_precision_recall(pred_g, true_g)
        # tp=0, fp=1, fn=0. precision=0/(0+1)=0.0, recall=0/(0+0)=0.0
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_reversed_edge_counts_as_match(self):
        """A reversed edge still matches at skeleton level (undirected)."""
        nodes = ["A", "B"]
        true_g = _make_graph(nodes, [("A", "B")])
        pred_g = _make_graph(nodes, [("B", "A")])
        precision, recall, f1 = edge_precision_recall(pred_g, true_g)
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_partial_overlap(self):
        """Partial overlap with some extra and some missing edges."""
        nodes = ["A", "B", "C", "D"]
        true_g = _make_graph(nodes, [("A", "B"), ("B", "C"), ("C", "D")])
        pred_g = _make_graph(nodes, [("A", "B"), ("B", "C"), ("A", "D")])
        # Skeleton true: {(A,B), (B,C), (C,D)}. Skeleton pred: {(A,B), (B,C), (A,D)}.
        # TP=2 (A-B, B-C). FP=1 (A-D). FN=1 (C-D).
        precision, recall, f1 = edge_precision_recall(pred_g, true_g)
        assert precision == pytest.approx(2.0 / 3.0)
        assert recall == pytest.approx(2.0 / 3.0)
        # F1 = 2 * (2/3) * (2/3) / (2/3 + 2/3) = 2/3
        assert f1 == pytest.approx(2.0 / 3.0)

    def test_large_graph_precision_recall(self):
        """Larger graph to verify precision/recall computation."""
        nodes = ["A", "B", "C", "D", "E"]
        true_edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]
        pred_edges = [("A", "B"), ("B", "C")]  # only 2 of 4 correct
        true_g = _make_graph(nodes, true_edges)
        pred_g = _make_graph(nodes, pred_edges)
        precision, recall, f1 = edge_precision_recall(pred_g, true_g)
        assert precision == 1.0  # 2 / (2+0)
        assert recall == pytest.approx(0.5)  # 2 / (2+2)
        assert f1 == pytest.approx(2.0 / 3.0)


# ===================================================================
# 3. Orientation Accuracy
# ===================================================================

class TestOrientationAccuracy:
    """Tests for orientation_accuracy."""

    def test_all_correct_orientations(self):
        """All shared edges have correct direction -> 1.0."""
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C")]
        true_g = _make_graph(nodes, edges)
        pred_g = _make_graph(nodes, edges)
        assert orientation_accuracy(pred_g, true_g) == 1.0

    def test_all_reversed_orientations(self):
        """All shared edges have wrong direction -> 0.0."""
        nodes = ["A", "B", "C"]
        true_g = _make_graph(nodes, [("A", "B"), ("B", "C")])
        pred_g = _make_graph(nodes, [("B", "A"), ("C", "B")])
        assert orientation_accuracy(pred_g, true_g) == 0.0

    def test_mixed_orientations(self):
        """Some correct, some reversed -> value between 0 and 1."""
        nodes = ["A", "B", "C"]
        true_g = _make_graph(nodes, [("A", "B"), ("B", "C")])
        # A->B correct, C->B reversed
        pred_g = _make_graph(nodes, [("A", "B"), ("C", "B")])
        result = orientation_accuracy(pred_g, true_g)
        assert result == pytest.approx(0.5)

    def test_no_shared_edges(self):
        """No shared skeleton edges -> returns 1.0 (vacuously correct)."""
        nodes = ["A", "B", "C", "D"]
        true_g = _make_graph(nodes, [("A", "B")])
        pred_g = _make_graph(nodes, [("C", "D")])
        assert orientation_accuracy(pred_g, true_g) == 1.0

    def test_both_empty_graphs(self):
        """Both empty -> no shared edges -> returns 1.0."""
        nodes = ["A", "B"]
        true_g = _make_graph(nodes, [])
        pred_g = _make_graph(nodes, [])
        assert orientation_accuracy(pred_g, true_g) == 1.0

    def test_one_of_three_reversed(self):
        """One out of three shared edges reversed -> 2/3."""
        nodes = ["A", "B", "C", "D"]
        true_g = _make_graph(nodes, [("A", "B"), ("B", "C"), ("C", "D")])
        # A->B correct, B->C correct, D->C reversed
        pred_g = _make_graph(nodes, [("A", "B"), ("B", "C"), ("D", "C")])
        result = orientation_accuracy(pred_g, true_g)
        assert result == pytest.approx(2.0 / 3.0)

    def test_extra_edges_ignored(self):
        """Edges in predicted but not in true are ignored."""
        nodes = ["A", "B", "C"]
        true_g = _make_graph(nodes, [("A", "B")])
        # A->B correct, plus extra B->C which is not in true
        pred_g = _make_graph(nodes, [("A", "B"), ("B", "C")])
        # Only shared edge A-B is considered and it is correct.
        assert orientation_accuracy(pred_g, true_g) == 1.0

    def test_missing_edges_ignored(self):
        """Edges in true but not in predicted are ignored."""
        nodes = ["A", "B", "C"]
        true_g = _make_graph(nodes, [("A", "B"), ("B", "C")])
        # Only A->B present and it is correct; B->C missing but ignored.
        pred_g = _make_graph(nodes, [("A", "B")])
        assert orientation_accuracy(pred_g, true_g) == 1.0

    def test_single_shared_edge_correct(self):
        """Single shared edge with correct direction -> 1.0."""
        nodes = ["X", "Y"]
        true_g = _make_graph(nodes, [("X", "Y")])
        pred_g = _make_graph(nodes, [("X", "Y")])
        assert orientation_accuracy(pred_g, true_g) == 1.0

    def test_single_shared_edge_reversed(self):
        """Single shared edge with wrong direction -> 0.0."""
        nodes = ["X", "Y"]
        true_g = _make_graph(nodes, [("X", "Y")])
        pred_g = _make_graph(nodes, [("Y", "X")])
        assert orientation_accuracy(pred_g, true_g) == 0.0


# ===================================================================
# 4. evaluate_structure_recovery (integration)
# ===================================================================

class TestEvaluateStructureRecovery:
    """Tests for the all-in-one evaluate_structure_recovery function."""

    def test_returns_structure_recovery_metrics(self):
        """Should return a StructureRecoveryMetrics instance."""
        nodes = ["A", "B"]
        g = _make_graph(nodes, [("A", "B")])
        result = evaluate_structure_recovery(g, g)
        assert isinstance(result, StructureRecoveryMetrics)

    def test_all_fields_present(self):
        """All expected fields are populated."""
        nodes = ["A", "B", "C"]
        true_g = _make_graph(nodes, [("A", "B"), ("B", "C")])
        pred_g = _make_graph(nodes, [("A", "B")])
        result = evaluate_structure_recovery(pred_g, true_g)

        assert hasattr(result, "shd")
        assert hasattr(result, "edge_precision")
        assert hasattr(result, "edge_recall")
        assert hasattr(result, "edge_f1")
        assert hasattr(result, "orientation_accuracy")
        assert hasattr(result, "n_true_edges")
        assert hasattr(result, "n_predicted_edges")

    def test_perfect_recovery(self):
        """Perfect recovery: all metrics ideal."""
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C")]
        true_g = _make_graph(nodes, edges)
        pred_g = _make_graph(nodes, edges)
        result = evaluate_structure_recovery(pred_g, true_g)

        assert result.shd == 0
        assert result.edge_precision == 1.0
        assert result.edge_recall == 1.0
        assert result.edge_f1 == 1.0
        assert result.orientation_accuracy == 1.0
        assert result.n_true_edges == 2
        assert result.n_predicted_edges == 2

    def test_matches_individual_functions(self):
        """Results should match the individual metric functions exactly."""
        nodes = ["A", "B", "C", "D"]
        true_g = _make_graph(nodes, [("A", "B"), ("B", "C"), ("C", "D")])
        pred_g = _make_graph(nodes, [("A", "B"), ("C", "B"), ("A", "D")])

        result = evaluate_structure_recovery(pred_g, true_g)

        expected_shd = structural_hamming_distance(pred_g, true_g)
        expected_prec, expected_rec, expected_f1 = edge_precision_recall(pred_g, true_g)
        expected_orient = orientation_accuracy(pred_g, true_g)

        assert result.shd == expected_shd
        assert result.edge_precision == pytest.approx(expected_prec)
        assert result.edge_recall == pytest.approx(expected_rec)
        assert result.edge_f1 == pytest.approx(expected_f1)
        assert result.orientation_accuracy == pytest.approx(expected_orient)
        assert result.n_true_edges == len(true_g.edges)
        assert result.n_predicted_edges == len(pred_g.edges)

    def test_empty_graphs(self):
        """Both empty graphs -> perfect metrics."""
        nodes = ["A", "B"]
        true_g = _make_graph(nodes, [])
        pred_g = _make_graph(nodes, [])
        result = evaluate_structure_recovery(pred_g, true_g)

        assert result.shd == 0
        assert result.edge_precision == 1.0
        assert result.edge_recall == 1.0
        assert result.edge_f1 == 1.0
        assert result.orientation_accuracy == 1.0
        assert result.n_true_edges == 0
        assert result.n_predicted_edges == 0

    def test_n_edges_counts(self):
        """n_true_edges and n_predicted_edges reflect the directed edge count."""
        nodes = ["A", "B", "C"]
        true_g = _make_graph(nodes, [("A", "B"), ("B", "C"), ("A", "C")])
        pred_g = _make_graph(nodes, [("A", "B")])
        result = evaluate_structure_recovery(pred_g, true_g)

        assert result.n_true_edges == 3
        assert result.n_predicted_edges == 1

    def test_completely_wrong_prediction(self):
        """Predicted graph shares no skeleton edges with true graph."""
        nodes = ["A", "B", "C", "D"]
        true_g = _make_graph(nodes, [("A", "B")])
        pred_g = _make_graph(nodes, [("C", "D")])
        result = evaluate_structure_recovery(pred_g, true_g)

        assert result.shd == 2  # 1 missing + 1 extra
        assert result.edge_precision == 0.0
        assert result.edge_recall == 0.0
        assert result.edge_f1 == 0.0
        # No shared edges -> orientation_accuracy defaults to 1.0
        assert result.orientation_accuracy == 1.0
        assert result.n_true_edges == 1
        assert result.n_predicted_edges == 1

    def test_reversed_graph(self):
        """All edges reversed: skeleton matches but orientation is wrong."""
        nodes = ["A", "B", "C"]
        true_g = _make_graph(nodes, [("A", "B"), ("B", "C")])
        pred_g = _make_graph(nodes, [("B", "A"), ("C", "B")])
        result = evaluate_structure_recovery(pred_g, true_g)

        # SHD: 2 reversals
        assert result.shd == 2
        # Skeleton-level: all edges match
        assert result.edge_precision == 1.0
        assert result.edge_recall == 1.0
        assert result.edge_f1 == 1.0
        # Orientation: all wrong
        assert result.orientation_accuracy == 0.0

    def test_superset_prediction_metrics(self):
        """Predicted has all true edges plus extras."""
        nodes = ["A", "B", "C", "D"]
        true_g = _make_graph(nodes, [("A", "B"), ("B", "C")])
        pred_g = _make_graph(nodes, [("A", "B"), ("B", "C"), ("A", "C"), ("A", "D")])
        result = evaluate_structure_recovery(pred_g, true_g)

        assert result.shd == 2  # 2 extra edges
        assert result.edge_precision == pytest.approx(2.0 / 4.0)
        assert result.edge_recall == 1.0
        assert result.orientation_accuracy == 1.0  # shared edges all correct
        assert result.n_true_edges == 2
        assert result.n_predicted_edges == 4
