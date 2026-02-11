"""Tests for uncertainty types, agent interface contracts, and serialisation."""

from __future__ import annotations

import math

import pytest

from optimization_copilot.uncertainty.types import (
    ConfidenceLevel,
    MeasurementWithUncertainty,
    ObservationWithNoise,
    PropagationMethod,
    PropagationResult,
    UncertaintyBudget,
    UncertaintySource,
)
from optimization_copilot.uncertainty.agent_interface import (
    AgentContext,
    OptimizationFeedback,
)


# ── UncertaintySource enum ──────────────────────────────────────────────


class TestUncertaintySource:
    def test_values(self):
        assert UncertaintySource.INSTRUMENT == "instrument"
        assert UncertaintySource.MODEL_FIT == "model_fit"
        assert UncertaintySource.REPETITION == "repetition"
        assert UncertaintySource.PROPAGATED == "propagated"

    def test_is_string(self):
        assert isinstance(UncertaintySource.INSTRUMENT, str)


class TestPropagationMethod:
    def test_values(self):
        assert PropagationMethod.LINEAR == "linear"
        assert PropagationMethod.DELTA == "delta"
        assert PropagationMethod.MONTE_CARLO == "monte_carlo"


class TestConfidenceLevel:
    def test_values(self):
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.LOW == "low"


# ── MeasurementWithUncertainty ──────────────────────────────────────────


class TestMeasurementWithUncertainty:
    def test_basic_construction(self):
        m = MeasurementWithUncertainty(
            value=10.0, variance=0.04, confidence=0.9, source="test"
        )
        assert m.value == 10.0
        assert m.variance == 0.04
        assert m.confidence == 0.9
        assert m.source == "test"
        assert m.method == "direct"
        assert m.metadata == {}

    def test_std_property(self):
        m = MeasurementWithUncertainty(
            value=5.0, variance=0.25, confidence=0.8, source="s"
        )
        assert m.std == pytest.approx(0.5)

    def test_relative_uncertainty(self):
        m = MeasurementWithUncertainty(
            value=100.0, variance=1.0, confidence=0.9, source="s"
        )
        assert m.relative_uncertainty == pytest.approx(0.01)

    def test_relative_uncertainty_zero_value(self):
        m = MeasurementWithUncertainty(
            value=0.0, variance=0.01, confidence=0.5, source="s"
        )
        assert m.relative_uncertainty == float("inf")

    def test_is_reliable_true(self):
        m = MeasurementWithUncertainty(
            value=100.0, variance=1.0, confidence=0.8, source="s"
        )
        assert m.is_reliable is True

    def test_is_reliable_false_low_confidence(self):
        m = MeasurementWithUncertainty(
            value=100.0, variance=1.0, confidence=0.3, source="s"
        )
        assert m.is_reliable is False

    def test_is_reliable_false_high_uncertainty(self):
        m = MeasurementWithUncertainty(
            value=1.0, variance=1.0, confidence=0.9, source="s"
        )
        # CV = 1.0 / 1.0 = 1.0 > 0.5
        assert m.is_reliable is False

    def test_confidence_level_high(self):
        m = MeasurementWithUncertainty(
            value=1.0, variance=0.01, confidence=0.9, source="s"
        )
        assert m.confidence_level == ConfidenceLevel.HIGH

    def test_confidence_level_medium(self):
        m = MeasurementWithUncertainty(
            value=1.0, variance=0.01, confidence=0.6, source="s"
        )
        assert m.confidence_level == ConfidenceLevel.MEDIUM

    def test_confidence_level_low(self):
        m = MeasurementWithUncertainty(
            value=1.0, variance=0.01, confidence=0.2, source="s"
        )
        assert m.confidence_level == ConfidenceLevel.LOW

    def test_negative_variance_raises(self):
        with pytest.raises(ValueError, match="variance"):
            MeasurementWithUncertainty(
                value=1.0, variance=-0.1, confidence=0.5, source="s"
            )

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            MeasurementWithUncertainty(
                value=1.0, variance=0.1, confidence=1.5, source="s"
            )

    def test_confidence_negative_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            MeasurementWithUncertainty(
                value=1.0, variance=0.1, confidence=-0.1, source="s"
            )

    def test_optional_fields(self):
        m = MeasurementWithUncertainty(
            value=5.0,
            variance=0.1,
            confidence=0.7,
            source="eis",
            fit_residual=0.002,
            n_points_used=50,
            method="interpolation",
            metadata={"quality_flags": ["ok"]},
        )
        assert m.fit_residual == 0.002
        assert m.n_points_used == 50
        assert m.method == "interpolation"
        assert m.metadata == {"quality_flags": ["ok"]}

    def test_to_dict(self):
        m = MeasurementWithUncertainty(
            value=3.0, variance=0.09, confidence=0.85, source="dc"
        )
        d = m.to_dict()
        assert d["value"] == 3.0
        assert d["variance"] == 0.09
        assert d["confidence"] == 0.85
        assert d["source"] == "dc"
        assert d["fit_residual"] is None
        assert d["method"] == "direct"

    def test_from_dict(self):
        d = {
            "value": 7.0,
            "variance": 0.49,
            "confidence": 0.6,
            "source": "xrd",
            "fit_residual": 0.01,
            "n_points_used": 100,
            "method": "scherrer",
            "metadata": {"k": 0.9},
        }
        m = MeasurementWithUncertainty.from_dict(d)
        assert m.value == 7.0
        assert m.method == "scherrer"
        assert m.metadata == {"k": 0.9}

    def test_roundtrip(self):
        m = MeasurementWithUncertainty(
            value=42.0,
            variance=1.5,
            confidence=0.75,
            source="test",
            fit_residual=0.003,
            n_points_used=20,
            method="nlls",
            metadata={"flags": []},
        )
        m2 = MeasurementWithUncertainty.from_dict(m.to_dict())
        assert m2.to_dict() == m.to_dict()

    def test_from_dict_defaults(self):
        d = {"value": 1.0, "variance": 0.0, "confidence": 1.0, "source": "s"}
        m = MeasurementWithUncertainty.from_dict(d)
        assert m.method == "direct"
        assert m.metadata == {}

    def test_zero_variance(self):
        m = MeasurementWithUncertainty(
            value=5.0, variance=0.0, confidence=1.0, source="exact"
        )
        assert m.std == 0.0
        assert m.relative_uncertainty == 0.0

    def test_boundary_confidence(self):
        # Exactly 0 and 1 should be valid.
        m0 = MeasurementWithUncertainty(
            value=1.0, variance=0.1, confidence=0.0, source="s"
        )
        m1 = MeasurementWithUncertainty(
            value=1.0, variance=0.1, confidence=1.0, source="s"
        )
        assert m0.confidence == 0.0
        assert m1.confidence == 1.0


# ── UncertaintyBudget ───────────────────────────────────────────────────


class TestUncertaintyBudget:
    def test_from_contributions(self):
        budget = UncertaintyBudget.from_contributions(
            {"instrument": 0.04, "model_fit": 0.01}
        )
        assert budget.total_variance == pytest.approx(0.05)
        assert budget.dominant_source == "instrument"

    def test_fraction(self):
        budget = UncertaintyBudget.from_contributions(
            {"a": 0.3, "b": 0.7}
        )
        assert budget.fraction("a") == pytest.approx(0.3)
        assert budget.fraction("b") == pytest.approx(0.7)
        assert budget.fraction("c") == pytest.approx(0.0)

    def test_fraction_zero_total(self):
        budget = UncertaintyBudget.from_contributions({})
        assert budget.fraction("anything") == 0.0

    def test_to_dict(self):
        budget = UncertaintyBudget.from_contributions({"x": 1.0})
        d = budget.to_dict()
        assert d["contributions"] == {"x": 1.0}
        assert d["total_variance"] == 1.0
        assert d["dominant_source"] == "x"

    def test_roundtrip(self):
        budget = UncertaintyBudget.from_contributions(
            {"inst": 0.1, "fit": 0.05, "rep": 0.02}
        )
        budget2 = UncertaintyBudget.from_dict(budget.to_dict())
        assert budget2.to_dict() == budget.to_dict()

    def test_single_source(self):
        budget = UncertaintyBudget.from_contributions({"only": 0.5})
        assert budget.dominant_source == "only"
        assert budget.fraction("only") == pytest.approx(1.0)


# ── ObservationWithNoise ────────────────────────────────────────────────


class TestObservationWithNoise:
    def test_basic_construction(self):
        obs = ObservationWithNoise(
            objective_value=0.85, noise_variance=0.01
        )
        assert obs.objective_value == 0.85
        assert obs.noise_variance == 0.01
        assert obs.kpi_contributions is None
        assert obs.uncertainty_budget is None
        assert obs.metadata == {}

    def test_noise_std(self):
        obs = ObservationWithNoise(
            objective_value=1.0, noise_variance=0.04
        )
        assert obs.noise_std == pytest.approx(0.2)

    def test_negative_noise_raises(self):
        with pytest.raises(ValueError, match="noise_variance"):
            ObservationWithNoise(objective_value=1.0, noise_variance=-1.0)

    def test_with_budget(self):
        budget = UncertaintyBudget.from_contributions({"a": 0.1, "b": 0.2})
        obs = ObservationWithNoise(
            objective_value=5.0,
            noise_variance=0.3,
            uncertainty_budget=budget,
        )
        assert obs.uncertainty_budget.total_variance == pytest.approx(0.3)

    def test_to_dict(self):
        obs = ObservationWithNoise(
            objective_value=2.0,
            noise_variance=0.05,
            kpi_contributions=[{"kpi": "CE", "weight": 1.0, "variance": 0.05}],
        )
        d = obs.to_dict()
        assert d["objective_value"] == 2.0
        assert d["noise_variance"] == 0.05
        assert len(d["kpi_contributions"]) == 1
        assert d["uncertainty_budget"] is None

    def test_roundtrip(self):
        budget = UncertaintyBudget.from_contributions({"inst": 0.02})
        obs = ObservationWithNoise(
            objective_value=3.0,
            noise_variance=0.02,
            uncertainty_budget=budget,
            metadata={"propagation_method": "linear"},
        )
        obs2 = ObservationWithNoise.from_dict(obs.to_dict())
        assert obs2.to_dict() == obs.to_dict()

    def test_from_dict_minimal(self):
        d = {"objective_value": 1.0, "noise_variance": 0.0}
        obs = ObservationWithNoise.from_dict(d)
        assert obs.objective_value == 1.0
        assert obs.kpi_contributions is None
        assert obs.uncertainty_budget is None


# ── PropagationResult ───────────────────────────────────────────────────


class TestPropagationResult:
    def test_basic(self):
        budget = UncertaintyBudget.from_contributions({"a": 0.1})
        pr = PropagationResult(
            objective_value=10.0,
            objective_variance=0.1,
            method=PropagationMethod.LINEAR,
            budget=budget,
        )
        assert pr.objective_value == 10.0
        assert pr.method == PropagationMethod.LINEAR

    def test_to_observation_with_noise(self):
        budget = UncertaintyBudget.from_contributions({"inst": 0.05})
        pr = PropagationResult(
            objective_value=7.0,
            objective_variance=0.05,
            method=PropagationMethod.DELTA,
            budget=budget,
            kpi_details=[{"kpi": "Rct", "value": 50.0}],
        )
        obs = pr.to_observation_with_noise()
        assert obs.objective_value == 7.0
        assert obs.noise_variance == 0.05
        assert obs.metadata["propagation_method"] == "delta"

    def test_to_dict(self):
        budget = UncertaintyBudget.from_contributions({"mc": 0.3})
        pr = PropagationResult(
            objective_value=1.0,
            objective_variance=0.3,
            method=PropagationMethod.MONTE_CARLO,
            budget=budget,
        )
        d = pr.to_dict()
        assert d["method"] == "monte_carlo"
        assert d["budget"]["total_variance"] == pytest.approx(0.3)


# ── AgentContext ────────────────────────────────────────────────────────


class TestAgentContext:
    def _make_context(self) -> AgentContext:
        m = MeasurementWithUncertainty(
            value=5.0, variance=0.1, confidence=0.8, source="test"
        )
        obs = ObservationWithNoise(
            objective_value=5.0, noise_variance=0.1
        )
        return AgentContext(
            measurements=[m],
            observation=obs,
            gp_state={"lengthscale": 1.0},
            history=[obs],
            domain_config={"instrument": "squidstat"},
        )

    def test_construction(self):
        ctx = self._make_context()
        assert len(ctx.measurements) == 1
        assert ctx.gp_state["lengthscale"] == 1.0

    def test_to_dict(self):
        ctx = self._make_context()
        d = ctx.to_dict()
        assert len(d["measurements"]) == 1
        assert d["observation"]["objective_value"] == 5.0
        assert len(d["history"]) == 1

    def test_defaults(self):
        m = MeasurementWithUncertainty(
            value=1.0, variance=0.0, confidence=1.0, source="s"
        )
        obs = ObservationWithNoise(
            objective_value=1.0, noise_variance=0.0
        )
        ctx = AgentContext(measurements=[m], observation=obs)
        assert ctx.gp_state == {}
        assert ctx.history == []
        assert ctx.domain_config == {}


# ── OptimizationFeedback ───────────────────────────────────────────────


class TestOptimizationFeedback:
    def test_default_construction(self):
        fb = OptimizationFeedback()
        assert fb.noise_override is None
        assert fb.rerun_suggested is False
        assert fb.rerun_reason == ""

    def test_noise_override(self):
        fb = OptimizationFeedback(noise_override=0.5)
        assert fb.noise_override == 0.5

    def test_to_dict(self):
        fb = OptimizationFeedback(
            noise_override=0.1,
            rerun_suggested=True,
            rerun_reason="bad contact",
        )
        d = fb.to_dict()
        assert d["noise_override"] == 0.1
        assert d["rerun_suggested"] is True
        assert d["rerun_reason"] == "bad contact"

    def test_roundtrip(self):
        fb = OptimizationFeedback(
            noise_override=0.2,
            prior_adjustment={"lengthscale": 2.0},
            constraint_update={"CE_max": 100},
            rerun_suggested=False,
            rerun_reason="",
        )
        fb2 = OptimizationFeedback.from_dict(fb.to_dict())
        assert fb2.to_dict() == fb.to_dict()

    def test_from_dict_defaults(self):
        fb = OptimizationFeedback.from_dict({})
        assert fb.noise_override is None
        assert fb.rerun_suggested is False
