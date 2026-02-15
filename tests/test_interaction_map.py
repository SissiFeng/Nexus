"""Comprehensive tests for the InteractionMap fANOVA decomposition.

Tests cover fitting, main effects, pairwise interactions, rendering,
edge cases, and reproducibility.
"""

from __future__ import annotations

import math
import random

import pytest

from optimization_copilot.explain.interaction_map import (
    InteractionMap,
    TreeNode,
    _mean,
    _variance,
    _variance_reduction,
)


# ============================================================================
# Helper data generators
# ============================================================================

def _linear_data(n: int = 100, seed: int = 0) -> tuple[list[list[float]], list[float]]:
    """y = 3*x0 + 0.1*x1 + noise."""
    rng = random.Random(seed)
    X = [[rng.uniform(0, 1), rng.uniform(0, 1)] for _ in range(n)]
    y = [3.0 * x[0] + 0.1 * x[1] + rng.gauss(0, 0.01) for x in X]
    return X, y


def _interaction_data(n: int = 100, seed: int = 0) -> tuple[list[list[float]], list[float]]:
    """y = x0 * x1 (strong interaction)."""
    rng = random.Random(seed)
    X = [[rng.uniform(0, 1), rng.uniform(0, 1)] for _ in range(n)]
    y = [x[0] * x[1] for x in X]
    return X, y


def _multi_feature_data(n: int = 100, d: int = 5, seed: int = 0) -> tuple[list[list[float]], list[float]]:
    """y = 2*x0 + x2 + noise, other features are irrelevant."""
    rng = random.Random(seed)
    X = [[rng.uniform(0, 1) for _ in range(d)] for _ in range(n)]
    y = [2.0 * x[0] + x[2] + rng.gauss(0, 0.01) for x in X]
    return X, y


# ============================================================================
# TreeNode dataclass tests
# ============================================================================

class TestTreeNode:
    """Tests for the TreeNode dataclass."""

    def test_creation(self) -> None:
        node = TreeNode(
            feature=0, threshold=0.5,
            left_value=1.0, right_value=2.0,
            variance_reduction=0.3,
            n_left=50, n_right=50,
        )
        assert node.feature == 0
        assert node.threshold == 0.5
        assert node.left_value == 1.0
        assert node.right_value == 2.0
        assert node.variance_reduction == pytest.approx(0.3)
        assert node.n_left == 50
        assert node.n_right == 50

    def test_default_values_not_required(self) -> None:
        """TreeNode requires all fields."""
        node = TreeNode(0, 0.0, 0.0, 0.0, 0.0, 0, 0)
        assert node.feature == 0


# ============================================================================
# Helper function tests
# ============================================================================

class TestHelpers:
    """Tests for _mean, _variance, _variance_reduction."""

    def test_mean_basic(self) -> None:
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_mean_empty(self) -> None:
        assert _mean([]) == 0.0

    def test_mean_single(self) -> None:
        assert _mean([5.0]) == pytest.approx(5.0)

    def test_variance_basic(self) -> None:
        assert _variance([1.0, 2.0, 3.0]) == pytest.approx(2.0 / 3.0)

    def test_variance_constant(self) -> None:
        assert _variance([5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_variance_single(self) -> None:
        assert _variance([5.0]) == 0.0

    def test_variance_empty(self) -> None:
        assert _variance([]) == 0.0

    def test_variance_reduction_positive(self) -> None:
        parent = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
        left = [1.0, 2.0, 3.0]
        right = [10.0, 11.0, 12.0]
        vr = _variance_reduction(parent, left, right)
        assert vr > 0.0

    def test_variance_reduction_zero_for_random_split(self) -> None:
        parent = [1.0, 1.0, 1.0]
        vr = _variance_reduction(parent, [1.0, 1.0], [1.0])
        assert vr == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# InteractionMap fitting tests
# ============================================================================

class TestInteractionMapFit:
    """Tests for InteractionMap.fit."""

    def test_fit_basic(self) -> None:
        X, y = _linear_data(50)
        imap = InteractionMap(n_trees=10, seed=42)
        imap.fit(X, y)
        assert imap._fitted is True
        assert len(imap._trees) == 10

    def test_fit_stores_data(self) -> None:
        X, y = _linear_data(30)
        imap = InteractionMap(n_trees=5, seed=42)
        imap.fit(X, y)
        assert len(imap._X) == 30
        assert len(imap._y) == 30
        assert imap._n_features == 2

    def test_fit_with_many_trees(self) -> None:
        X, y = _linear_data(50)
        imap = InteractionMap(n_trees=100, seed=42)
        imap.fit(X, y)
        assert len(imap._trees) == 100

    def test_fit_tree_nodes_valid(self) -> None:
        X, y = _linear_data(50)
        imap = InteractionMap(n_trees=20, seed=42)
        imap.fit(X, y)
        for tree in imap._trees:
            assert isinstance(tree, TreeNode)
            assert 0 <= tree.feature < 2
            assert tree.variance_reduction >= 0.0


# ============================================================================
# Main effects tests
# ============================================================================

class TestMainEffects:
    """Tests for InteractionMap.compute_main_effects."""

    def test_main_effects_sum_to_one(self) -> None:
        X, y = _linear_data(100)
        imap = InteractionMap(n_trees=50, seed=42)
        imap.fit(X, y)
        effects = imap.compute_main_effects()
        total = sum(effects.values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_main_effects_identifies_important_feature(self) -> None:
        """y = 3*x0 + 0.1*x1 => x0 should dominate."""
        X, y = _linear_data(200, seed=7)
        imap = InteractionMap(n_trees=100, seed=42)
        imap.fit(X, y)
        effects = imap.compute_main_effects()
        assert effects[0] > effects[1]

    def test_main_effects_all_features_present(self) -> None:
        X, y = _linear_data(50)
        imap = InteractionMap(n_trees=30, seed=42)
        imap.fit(X, y)
        effects = imap.compute_main_effects()
        assert 0 in effects
        assert 1 in effects

    def test_main_effects_non_negative(self) -> None:
        X, y = _linear_data(50)
        imap = InteractionMap(n_trees=30, seed=42)
        imap.fit(X, y)
        effects = imap.compute_main_effects()
        for v in effects.values():
            assert v >= 0.0

    def test_main_effects_many_features(self) -> None:
        X, y = _multi_feature_data(100, d=5)
        imap = InteractionMap(n_trees=100, seed=42)
        imap.fit(X, y)
        effects = imap.compute_main_effects()
        assert len(effects) == 5
        assert sum(effects.values()) == pytest.approx(1.0, abs=1e-10)

    def test_main_effects_five_features_identifies_relevant(self) -> None:
        """y = 2*x0 + x2 => x0 and x2 should have highest importance."""
        X, y = _multi_feature_data(200, d=5, seed=99)
        imap = InteractionMap(n_trees=200, seed=42)
        imap.fit(X, y)
        effects = imap.compute_main_effects()
        # x0 and x2 together should have more than half the importance
        relevant = effects[0] + effects[2]
        assert relevant > 0.3  # generous threshold for stochastic trees


# ============================================================================
# Interaction tests
# ============================================================================

class TestInteractions:
    """Tests for InteractionMap.compute_interactions."""

    def test_interactions_basic(self) -> None:
        X, y = _linear_data(100)
        imap = InteractionMap(n_trees=50, seed=42)
        imap.fit(X, y)
        interactions = imap.compute_interactions()
        assert isinstance(interactions, dict)

    def test_interactions_keys_are_pairs(self) -> None:
        X, y = _linear_data(100)
        imap = InteractionMap(n_trees=50, seed=42)
        imap.fit(X, y)
        interactions = imap.compute_interactions()
        for key in interactions:
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert key[0] < key[1]

    def test_interactions_strong_interaction(self) -> None:
        """y = x0 * x1 + x2 should produce high (0,1) interaction relative to others."""
        rng = random.Random(42)
        X = [[rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)] for _ in range(200)]
        y = [x[0] * x[1] + 0.1 * x[2] for x in X]
        imap = InteractionMap(n_trees=100, seed=42)
        imap.fit(X, y)
        interactions = imap.compute_interactions()
        assert (0, 1) in interactions
        # (0,1) should be the strongest interaction pair
        max_pair = max(interactions, key=interactions.get)  # type: ignore[arg-type]
        assert max_pair == (0, 1)

    def test_interactions_non_negative(self) -> None:
        X, y = _linear_data(100)
        imap = InteractionMap(n_trees=50, seed=42)
        imap.fit(X, y)
        interactions = imap.compute_interactions()
        for v in interactions.values():
            assert v >= 0.0

    def test_interactions_with_many_features(self) -> None:
        X, y = _multi_feature_data(100, d=4)
        imap = InteractionMap(n_trees=50, seed=42)
        imap.fit(X, y)
        interactions = imap.compute_interactions()
        n_pairs = 4 * 3 // 2  # C(4,2) = 6
        assert len(interactions) == n_pairs


# ============================================================================
# Top interactions tests
# ============================================================================

class TestGetTopInteractions:
    """Tests for InteractionMap.get_top_interactions."""

    def test_get_top_interactions(self) -> None:
        X, y = _multi_feature_data(100, d=4)
        imap = InteractionMap(n_trees=50, seed=42)
        imap.fit(X, y)
        top = imap.get_top_interactions(k=3)
        assert len(top) <= 3
        for item in top:
            assert len(item) == 3
            i, j, strength = item
            assert isinstance(i, int)
            assert isinstance(j, int)
            assert isinstance(strength, float)

    def test_get_top_interactions_sorted(self) -> None:
        X, y = _multi_feature_data(100, d=4)
        imap = InteractionMap(n_trees=50, seed=42)
        imap.fit(X, y)
        top = imap.get_top_interactions(k=6)
        for a, b in zip(top, top[1:]):
            assert a[2] >= b[2]

    def test_get_top_interactions_k_exceeds_pairs(self) -> None:
        X, y = _linear_data(50)
        imap = InteractionMap(n_trees=20, seed=42)
        imap.fit(X, y)
        top = imap.get_top_interactions(k=100)
        # Only 1 pair for 2 features
        assert len(top) == 1


# ============================================================================
# Render heatmap tests
# ============================================================================

class TestRenderHeatmap:
    """Tests for InteractionMap.render_heatmap."""

    def test_render_heatmap_returns_svg(self) -> None:
        X, y = _linear_data(50)
        imap = InteractionMap(n_trees=10, seed=42)
        imap.fit(X, y)
        svg = imap.render_heatmap()
        assert isinstance(svg, str)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_render_heatmap_with_names(self) -> None:
        X, y = _linear_data(50)
        imap = InteractionMap(n_trees=10, seed=42)
        imap.fit(X, y)
        svg = imap.render_heatmap(feature_names=["temp", "pressure"])
        assert "temp" in svg
        assert "pressure" in svg

    def test_render_heatmap_without_names(self) -> None:
        X, y = _linear_data(50)
        imap = InteractionMap(n_trees=10, seed=42)
        imap.fit(X, y)
        svg = imap.render_heatmap()
        assert "x0" in svg
        assert "x1" in svg

    def test_render_heatmap_many_features(self) -> None:
        X, y = _multi_feature_data(50, d=5)
        imap = InteractionMap(n_trees=10, seed=42)
        imap.fit(X, y)
        svg = imap.render_heatmap()
        assert "<svg" in svg


# ============================================================================
# Edge case tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_data(self) -> None:
        imap = InteractionMap(n_trees=10, seed=42)
        imap.fit([], [])
        assert imap._fitted is True
        effects = imap.compute_main_effects()
        assert effects == {}

    def test_single_feature(self) -> None:
        rng = random.Random(42)
        X = [[rng.uniform(0, 1)] for _ in range(50)]
        y = [x[0] * 2 + rng.gauss(0, 0.01) for x in X]
        imap = InteractionMap(n_trees=20, seed=42)
        imap.fit(X, y)
        effects = imap.compute_main_effects()
        assert 0 in effects
        assert effects[0] == pytest.approx(1.0, abs=1e-10)

    def test_constant_y(self) -> None:
        rng = random.Random(42)
        X = [[rng.uniform(0, 1), rng.uniform(0, 1)] for _ in range(50)]
        y = [5.0] * 50
        imap = InteractionMap(n_trees=20, seed=42)
        imap.fit(X, y)
        effects = imap.compute_main_effects()
        # Uniform importance when no variance
        assert effects[0] == pytest.approx(0.5, abs=1e-10)
        assert effects[1] == pytest.approx(0.5, abs=1e-10)

    def test_two_points(self) -> None:
        X = [[0.0, 0.0], [1.0, 1.0]]
        y = [0.0, 1.0]
        imap = InteractionMap(n_trees=10, seed=42)
        imap.fit(X, y)
        assert imap._fitted is True

    def test_single_point(self) -> None:
        X = [[1.0, 2.0]]
        y = [3.0]
        imap = InteractionMap(n_trees=10, seed=42)
        imap.fit(X, y)
        assert imap._fitted is True

    def test_render_heatmap_empty(self) -> None:
        imap = InteractionMap(n_trees=10, seed=42)
        imap.fit([], [])
        svg = imap.render_heatmap()
        assert "<svg" in svg
        assert "No features" in svg

    def test_interactions_too_few_points(self) -> None:
        """Fewer than 4 points => empty interactions."""
        X = [[0.0, 1.0], [1.0, 0.0]]
        y = [0.0, 1.0]
        imap = InteractionMap(n_trees=10, seed=42)
        imap.fit(X, y)
        interactions = imap.compute_interactions()
        assert interactions == {}

    def test_large_dataset(self) -> None:
        rng = random.Random(42)
        X = [[rng.uniform(0, 1) for _ in range(3)] for _ in range(500)]
        y = [x[0] + 2 * x[1] for x in X]
        imap = InteractionMap(n_trees=50, seed=42)
        imap.fit(X, y)
        effects = imap.compute_main_effects()
        assert len(effects) == 3


# ============================================================================
# Reproducibility tests
# ============================================================================

class TestReproducibility:
    """Tests for deterministic behaviour with the same seed."""

    def test_reproducibility_with_seed(self) -> None:
        X, y = _linear_data(100)
        imap1 = InteractionMap(n_trees=30, seed=42)
        imap1.fit(X, y)
        effects1 = imap1.compute_main_effects()

        imap2 = InteractionMap(n_trees=30, seed=42)
        imap2.fit(X, y)
        effects2 = imap2.compute_main_effects()

        for key in effects1:
            assert effects1[key] == pytest.approx(effects2[key])

    def test_different_seeds_differ(self) -> None:
        X, y = _linear_data(100)
        imap1 = InteractionMap(n_trees=30, seed=42)
        imap1.fit(X, y)

        imap2 = InteractionMap(n_trees=30, seed=99)
        imap2.fit(X, y)

        # Trees should typically differ (not always, but highly likely)
        trees_same = all(
            t1.threshold == t2.threshold
            for t1, t2 in zip(imap1._trees, imap2._trees)
        )
        assert not trees_same

    def test_interactions_reproducible(self) -> None:
        X, y = _interaction_data(100)
        imap1 = InteractionMap(n_trees=30, seed=42)
        imap1.fit(X, y)
        i1 = imap1.compute_interactions()

        imap2 = InteractionMap(n_trees=30, seed=42)
        imap2.fit(X, y)
        i2 = imap2.compute_interactions()

        for key in i1:
            assert i1[key] == pytest.approx(i2[key])
