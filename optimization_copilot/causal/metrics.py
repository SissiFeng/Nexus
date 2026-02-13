"""Reusable causal structure recovery metrics.

Provides Structural Hamming Distance (SHD), edge precision/recall/F1,
orientation accuracy, and an all-in-one evaluation function for
comparing predicted vs true causal DAGs.
"""

from __future__ import annotations

from dataclasses import dataclass
from optimization_copilot.causal.models import CausalGraph


@dataclass
class StructureRecoveryMetrics:
    """Complete structure recovery evaluation."""
    shd: int                    # Structural Hamming Distance
    edge_precision: float       # TP / (TP + FP), skeleton-level
    edge_recall: float          # TP / (TP + FN), skeleton-level
    edge_f1: float              # harmonic mean
    orientation_accuracy: float # fraction of shared edges with correct direction
    n_true_edges: int
    n_predicted_edges: int


def structural_hamming_distance(predicted: CausalGraph, true: CausalGraph) -> int:
    """Minimum edge additions + deletions + reversals to match true graph.

    For each pair (A, B) of nodes:
    - If true has A->B and predicted has A->B: 0 (match)
    - If true has A->B and predicted has nothing: +1 (missing)
    - If true has nothing and predicted has A->B: +1 (extra)
    - If true has A->B and predicted has B->A: +1 (reversal)
    """
    all_nodes = sorted(set(predicted.node_names) | set(true.node_names))
    shd = 0
    for i, a in enumerate(all_nodes):
        for b in all_nodes[i+1:]:
            true_ab = true.has_edge(a, b)
            true_ba = true.has_edge(b, a)
            pred_ab = predicted.has_edge(a, b)
            pred_ba = predicted.has_edge(b, a)

            true_has = true_ab or true_ba
            pred_has = pred_ab or pred_ba

            if true_has and not pred_has:
                shd += 1  # missing edge
            elif not true_has and pred_has:
                shd += 1  # extra edge
            elif true_has and pred_has:
                # Both have edge between a,b — check direction
                if (true_ab != pred_ab) or (true_ba != pred_ba):
                    shd += 1  # wrong direction
    return shd


def edge_precision_recall(
    predicted: CausalGraph, true: CausalGraph
) -> tuple[float, float, float]:
    """Skeleton-level precision, recall, F1.

    Skeleton = undirected edge set. An edge exists between A and B
    if either A->B or B->A exists.

    Returns (precision, recall, f1).
    """
    def _skeleton(g: CausalGraph) -> set[tuple[str, str]]:
        edges = set()
        for e in g.edges:
            pair = tuple(sorted([e.source, e.target]))
            edges.add(pair)
        return edges

    true_skel = _skeleton(true)
    pred_skel = _skeleton(predicted)

    if not pred_skel and not true_skel:
        return 1.0, 1.0, 1.0  # Both empty = perfect

    tp = len(pred_skel & true_skel)
    fp = len(pred_skel - true_skel)
    fn = len(true_skel - pred_skel)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def orientation_accuracy(predicted: CausalGraph, true: CausalGraph) -> float:
    """Fraction of shared skeleton edges with correct direction.

    Only considers edges that exist in BOTH predicted and true
    skeletons (ignoring missing/extra edges).
    Returns 1.0 if no shared edges.
    """
    def _skeleton_set(g: CausalGraph) -> set[tuple[str, str]]:
        edges = set()
        for e in g.edges:
            pair = tuple(sorted([e.source, e.target]))
            edges.add(pair)
        return edges

    true_skel = _skeleton_set(true)
    pred_skel = _skeleton_set(predicted)
    shared = true_skel & pred_skel

    if not shared:
        return 1.0  # No shared edges to evaluate

    correct = 0
    for a, b in shared:
        # Check if the direction matches
        true_dir = (true.has_edge(a, b), true.has_edge(b, a))
        pred_dir = (predicted.has_edge(a, b), predicted.has_edge(b, a))
        if true_dir == pred_dir:
            correct += 1

    return correct / len(shared)


def evaluate_structure_recovery(
    predicted: CausalGraph, true: CausalGraph
) -> StructureRecoveryMetrics:
    """All-in-one structure recovery evaluation."""
    shd = structural_hamming_distance(predicted, true)
    precision, recall, f1 = edge_precision_recall(predicted, true)
    orient_acc = orientation_accuracy(predicted, true)

    return StructureRecoveryMetrics(
        shd=shd,
        edge_precision=precision,
        edge_recall=recall,
        edge_f1=f1,
        orientation_accuracy=orient_acc,
        n_true_edges=len(true.edges),
        n_predicted_edges=len(predicted.edges),
    )
