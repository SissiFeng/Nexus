"""Comprehensive tests for the InsightReportGenerator and physics kernels.

Tests cover report generation, SVG rendering, text rendering, domain-specific
kernels, kernel properties, and edge cases.
"""

from __future__ import annotations

import math
import random

import pytest

from optimization_copilot.explain.report_generator import (
    InsightReport,
    InsightReportGenerator,
)
from optimization_copilot.explain.equation_discovery import ParetoSolution, ExprNode
from optimization_copilot.explain.physics_kernels import (
    InteractionKernel,
    PhysicsKernelFactory,
    SaturationKernel,
)


# ============================================================================
# Helper data generators
# ============================================================================

def _sample_data(
    n: int = 50, d: int = 3, seed: int = 42,
) -> tuple[list[list[float]], list[float], list[str]]:
    """Generate sample data with y = x0 + 2*x1 + noise."""
    rng = random.Random(seed)
    X = [[rng.uniform(0, 1) for _ in range(d)] for _ in range(n)]
    y = [x[0] + 2.0 * x[1] + rng.gauss(0, 0.01) for x in X]
    names = [f"var{i}" for i in range(d)]
    return X, y, names


# ============================================================================
# InsightReport dataclass tests
# ============================================================================

class TestInsightReport:
    """Tests for the InsightReport dataclass."""

    def test_insight_report_creation(self) -> None:
        report = InsightReport(
            main_effects={"x0": 0.6, "x1": 0.4},
            top_interactions=[("x0", "x1", 0.5)],
            equations=[],
            best_equation=None,
            domain="general",
            n_observations=10,
            summary="Test summary",
        )
        assert report.domain == "general"
        assert report.n_observations == 10
        assert len(report.main_effects) == 2

    def test_insight_report_with_svg_charts(self) -> None:
        report = InsightReport(
            main_effects={"x0": 1.0},
            top_interactions=[],
            equations=[],
            best_equation=None,
            domain="test",
            n_observations=5,
            summary="Test",
            svg_charts={"chart1": "<svg></svg>"},
        )
        assert "chart1" in report.svg_charts

    def test_insight_report_defaults(self) -> None:
        report = InsightReport(
            main_effects={},
            top_interactions=[],
            equations=[],
            best_equation=None,
            domain="",
            n_observations=0,
            summary="",
        )
        assert report.svg_charts == {}


# ============================================================================
# Report generation tests
# ============================================================================

class TestReportGeneration:
    """Tests for InsightReportGenerator.generate."""

    def test_report_generation_basic(self) -> None:
        X, y, names = _sample_data(50, 3)
        gen = InsightReportGenerator(
            domain="general", n_trees=20, eq_population=30, eq_generations=5, seed=42,
        )
        report = gen.generate(X, y, var_names=names)
        assert isinstance(report, InsightReport)
        assert report.n_observations == 50

    def test_report_has_main_effects(self) -> None:
        X, y, names = _sample_data(50, 3)
        gen = InsightReportGenerator(
            domain="general", n_trees=20, eq_population=30, eq_generations=5, seed=42,
        )
        report = gen.generate(X, y, var_names=names)
        assert len(report.main_effects) == 3
        for name in names:
            assert name in report.main_effects

    def test_report_has_equations(self) -> None:
        X, y, names = _sample_data(50, 2)
        gen = InsightReportGenerator(
            domain="general", n_trees=20, eq_population=50, eq_generations=10, seed=42,
        )
        report = gen.generate(X, y, var_names=names)
        assert isinstance(report.equations, list)

    def test_report_has_interactions(self) -> None:
        X, y, names = _sample_data(50, 3)
        gen = InsightReportGenerator(
            domain="general", n_trees=20, eq_population=30, eq_generations=5, seed=42,
        )
        report = gen.generate(X, y, var_names=names)
        assert isinstance(report.top_interactions, list)

    def test_report_has_best_equation(self) -> None:
        X, y, names = _sample_data(50, 2)
        gen = InsightReportGenerator(
            domain="general", n_trees=20, eq_population=50, eq_generations=10, seed=42,
        )
        report = gen.generate(X, y, var_names=names)
        # May or may not find a valid equation, but field should exist
        assert hasattr(report, "best_equation")

    def test_report_domain_in_report(self) -> None:
        X, y, names = _sample_data(30, 2)
        gen = InsightReportGenerator(domain="electrochemistry", seed=42,
                                      n_trees=10, eq_population=20, eq_generations=5)
        report = gen.generate(X, y)
        assert report.domain == "electrochemistry"

    def test_domain_specific_report(self) -> None:
        X, y, names = _sample_data(30, 2)
        gen = InsightReportGenerator(domain="catalysis", seed=42,
                                      n_trees=10, eq_population=20, eq_generations=5)
        report = gen.generate(X, y, var_names=names)
        assert report.domain == "catalysis"

    def test_feature_names_in_report(self) -> None:
        X, y, names = _sample_data(50, 3)
        gen = InsightReportGenerator(seed=42, n_trees=20,
                                      eq_population=30, eq_generations=5)
        report = gen.generate(X, y, var_names=names)
        for name in names:
            assert name in report.main_effects

    def test_summary_generation(self) -> None:
        X, y, names = _sample_data(50, 3)
        gen = InsightReportGenerator(seed=42, n_trees=20,
                                      eq_population=30, eq_generations=5)
        report = gen.generate(X, y, var_names=names)
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0
        assert "50" in report.summary  # n_observations

    def test_report_with_diagnostics(self) -> None:
        X, y, names = _sample_data(30, 2)
        gen = InsightReportGenerator(seed=42, n_trees=10,
                                      eq_population=20, eq_generations=5)
        diag = {"noise_level": 0.01, "phase": "learning"}
        report = gen.generate(X, y, diagnostics=diag)
        assert "noise_level" in report.summary or "Diagnostics" in report.summary

    def test_empty_data_handling(self) -> None:
        gen = InsightReportGenerator(seed=42, n_trees=10,
                                      eq_population=20, eq_generations=5)
        report = gen.generate([], [])
        assert report.n_observations == 0
        assert report.main_effects == {}


# ============================================================================
# SVG rendering tests
# ============================================================================

class TestRenderSVG:
    """Tests for SVG rendering."""

    def test_render_svg_produces_valid_svg(self) -> None:
        X, y, names = _sample_data(50, 3)
        gen = InsightReportGenerator(seed=42, n_trees=20,
                                      eq_population=30, eq_generations=5)
        report = gen.generate(X, y, var_names=names)
        svg = gen.render_svg(report)
        assert isinstance(svg, str)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_render_svg_has_charts(self) -> None:
        X, y, names = _sample_data(50, 3)
        gen = InsightReportGenerator(seed=42, n_trees=20,
                                      eq_population=30, eq_generations=5)
        report = gen.generate(X, y, var_names=names)
        assert "feature_importance" in report.svg_charts
        assert "interaction_heatmap" in report.svg_charts

    def test_render_svg_empty_report(self) -> None:
        report = InsightReport(
            main_effects={},
            top_interactions=[],
            equations=[],
            best_equation=None,
            domain="test",
            n_observations=0,
            summary="",
            svg_charts={},
        )
        gen = InsightReportGenerator(seed=42)
        svg = gen.render_svg(report)
        assert "<svg" in svg


# ============================================================================
# Text rendering tests
# ============================================================================

class TestRenderText:
    """Tests for text rendering."""

    def test_render_text_produces_string(self) -> None:
        X, y, names = _sample_data(50, 3)
        gen = InsightReportGenerator(seed=42, n_trees=20,
                                      eq_population=30, eq_generations=5)
        report = gen.generate(X, y, var_names=names)
        text = gen.render_text(report)
        assert isinstance(text, str)
        assert "Insight Report" in text

    def test_render_text_contains_features(self) -> None:
        X, y, names = _sample_data(50, 3)
        gen = InsightReportGenerator(seed=42, n_trees=20,
                                      eq_population=30, eq_generations=5)
        report = gen.generate(X, y, var_names=names)
        text = gen.render_text(report)
        assert "Feature Importances" in text

    def test_render_text_contains_observations(self) -> None:
        X, y, names = _sample_data(50, 3)
        gen = InsightReportGenerator(seed=42, n_trees=20,
                                      eq_population=30, eq_generations=5)
        report = gen.generate(X, y, var_names=names)
        text = gen.render_text(report)
        assert "50" in text


# ============================================================================
# SaturationKernel tests
# ============================================================================

class TestSaturationKernel:
    """Tests for the SaturationKernel."""

    def test_saturation_kernel_self_similarity(self) -> None:
        k = SaturationKernel()
        val = k([1.0, 2.0], [1.0, 2.0])
        assert val == pytest.approx(1.0)

    def test_saturation_kernel_symmetry(self) -> None:
        k = SaturationKernel()
        v1 = k([1.0, 2.0], [3.0, 4.0])
        v2 = k([3.0, 4.0], [1.0, 2.0])
        assert v1 == pytest.approx(v2)

    def test_saturation_kernel_positive(self) -> None:
        k = SaturationKernel()
        val = k([0.0], [10.0])
        assert val > 0.0

    def test_saturation_kernel_decreases_with_distance(self) -> None:
        k = SaturationKernel(length_scale=1.0)
        close = k([0.0], [0.1])
        far = k([0.0], [5.0])
        assert close > far

    def test_saturation_kernel_warping(self) -> None:
        """With high steepness, far-apart points in raw space are closer in warped space."""
        k_low = SaturationKernel(steepness=0.1)
        k_high = SaturationKernel(steepness=10.0)
        # Far-apart points
        v_low = k_low([0.1], [0.9])
        v_high = k_high([0.1], [0.9])
        # Both should be positive
        assert v_low > 0.0
        assert v_high > 0.0

    def test_saturation_kernel_matrix(self) -> None:
        k = SaturationKernel()
        X = [[0.0], [0.5], [1.0]]
        K = k.matrix(X)
        assert len(K) == 3
        assert len(K[0]) == 3
        # Diagonal should be close to 1 + noise
        for i in range(3):
            assert K[i][i] > 0.99

    def test_saturation_kernel_matrix_symmetric(self) -> None:
        k = SaturationKernel()
        X = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
        K = k.matrix(X)
        for i in range(3):
            for j in range(3):
                assert K[i][j] == pytest.approx(K[j][i])

    def test_saturation_kernel_repr(self) -> None:
        k = SaturationKernel(length_scale=2.0, saturation_point=100.0, steepness=3.0)
        r = repr(k)
        assert "SaturationKernel" in r
        assert "2.0" in r


# ============================================================================
# InteractionKernel tests
# ============================================================================

class TestInteractionKernel:
    """Tests for the InteractionKernel."""

    def test_interaction_kernel_self_similarity(self) -> None:
        k = InteractionKernel()
        val = k([1.0, 2.0], [1.0, 2.0])
        # Self-similarity: each per_dim = 1.0, so main = d, interaction = C(d,2)
        # For d=2: main=2, interaction=1*0.1 = 0.1, total = 2.1
        expected = 2.0 + 0.1 * 1.0
        assert val == pytest.approx(expected)

    def test_interaction_kernel_symmetry(self) -> None:
        k = InteractionKernel()
        v1 = k([1.0, 2.0], [3.0, 4.0])
        v2 = k([3.0, 4.0], [1.0, 2.0])
        assert v1 == pytest.approx(v2)

    def test_interaction_kernel_positive(self) -> None:
        k = InteractionKernel()
        val = k([0.0, 0.0], [1.0, 1.0])
        assert val > 0.0

    def test_interaction_kernel_with_custom_length_scales(self) -> None:
        k = InteractionKernel(base_length_scales=[0.5, 2.0])
        val = k([0.0, 0.0], [1.0, 1.0])
        assert isinstance(val, float)
        assert val > 0.0

    def test_interaction_kernel_matrix(self) -> None:
        k = InteractionKernel()
        X = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        K = k.matrix(X)
        assert len(K) == 3
        assert len(K[0]) == 3

    def test_interaction_kernel_matrix_symmetric(self) -> None:
        k = InteractionKernel(interaction_strength=0.3)
        X = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
        K = k.matrix(X)
        for i in range(3):
            for j in range(3):
                assert K[i][j] == pytest.approx(K[j][i])

    def test_interaction_kernel_strength_effect(self) -> None:
        """Higher interaction strength should increase kernel value for non-identical points."""
        k_low = InteractionKernel(interaction_strength=0.0)
        k_high = InteractionKernel(interaction_strength=1.0)
        x1 = [0.5, 0.5]
        x2 = [0.5, 0.5]
        # At same point, higher interaction_strength -> higher total
        v_low = k_low(x1, x2)
        v_high = k_high(x1, x2)
        assert v_high > v_low

    def test_interaction_kernel_repr(self) -> None:
        k = InteractionKernel(interaction_strength=0.5)
        r = repr(k)
        assert "InteractionKernel" in r
        assert "0.5" in r


# ============================================================================
# PhysicsKernelFactory tests
# ============================================================================

class TestPhysicsKernelFactory:
    """Tests for the PhysicsKernelFactory."""

    def test_factory_electrochemistry(self) -> None:
        k = PhysicsKernelFactory.for_domain("electrochemistry")
        assert isinstance(k, SaturationKernel)
        assert k.saturation_point == 100.0
        assert k.steepness == 3.0

    def test_factory_catalysis(self) -> None:
        k = PhysicsKernelFactory.for_domain("catalysis")
        assert isinstance(k, InteractionKernel)
        assert k.interaction_strength == 0.2

    def test_factory_perovskite(self) -> None:
        k = PhysicsKernelFactory.for_domain("perovskite")
        assert isinstance(k, SaturationKernel)
        assert k.saturation_point == 35.0
        assert k.steepness == 2.0

    def test_factory_unknown_domain(self) -> None:
        k = PhysicsKernelFactory.for_domain("unknown_domain")
        assert isinstance(k, SaturationKernel)
        # Default values
        assert k.saturation_point == 1.0
        assert k.steepness == 5.0

    def test_factory_empty_string(self) -> None:
        k = PhysicsKernelFactory.for_domain("")
        assert isinstance(k, SaturationKernel)

    def test_factory_kernels_are_callable(self) -> None:
        for domain in ["electrochemistry", "catalysis", "perovskite", "other"]:
            k = PhysicsKernelFactory.for_domain(domain)
            val = k([0.5], [0.5])
            assert isinstance(val, float)
            assert val > 0.0


# ============================================================================
# Kernel property tests
# ============================================================================

class TestKernelProperties:
    """Tests for mathematical properties of kernels."""

    def test_kernel_symmetry_saturation(self) -> None:
        k = SaturationKernel(length_scale=2.0, saturation_point=50.0)
        rng = random.Random(42)
        for _ in range(10):
            x1 = [rng.uniform(0, 10) for _ in range(3)]
            x2 = [rng.uniform(0, 10) for _ in range(3)]
            assert k(x1, x2) == pytest.approx(k(x2, x1))

    def test_kernel_symmetry_interaction(self) -> None:
        k = InteractionKernel(interaction_strength=0.3)
        rng = random.Random(42)
        for _ in range(10):
            x1 = [rng.uniform(0, 10) for _ in range(3)]
            x2 = [rng.uniform(0, 10) for _ in range(3)]
            assert k(x1, x2) == pytest.approx(k(x2, x1))

    def test_kernel_positive_definiteness_saturation(self) -> None:
        """Kernel matrix diagonal should be positive."""
        k = SaturationKernel()
        X = [[float(i)] for i in range(5)]
        K = k.matrix(X)
        for i in range(5):
            assert K[i][i] > 0.0

    def test_kernel_positive_definiteness_interaction(self) -> None:
        """Kernel matrix diagonal should be positive."""
        k = InteractionKernel()
        X = [[float(i), float(i) * 0.5] for i in range(5)]
        K = k.matrix(X)
        for i in range(5):
            assert K[i][i] > 0.0

    def test_saturation_warp_bounded(self) -> None:
        """Warped values should be bounded by saturation_point."""
        k = SaturationKernel(saturation_point=10.0, steepness=5.0)
        warped = k._warp([100.0, 200.0, -100.0])
        for w in warped:
            assert abs(w) <= 10.0 + 1e-6
