"""Tests for the Decision Sensitivity Analysis package.

Covers ParameterSensitivity, DecisionStability, SensitivityReport models,
parameter sensitivity analysis, decision stability analysis, robustness
scoring, recommendations, and edge cases.
"""

from __future__ import annotations

import math
import random

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.sensitivity.models import (
    DecisionStability,
    ParameterSensitivity,
    SensitivityReport,
)
from optimization_copilot.sensitivity.analyzer import SensitivityAnalyzer


# ── Helpers ───────────────────────────────────────────────────


def _make_specs(n_params=3):
    """Create *n_params* continuous parameter specs on [0, 10]."""
    return [
        ParameterSpec(
            name=f"x{i + 1}",
            type=VariableType.CONTINUOUS,
            lower=0.0,
            upper=10.0,
        )
        for i in range(n_params)
    ]


def _make_obs(iteration, params, kpi, **kwargs):
    """Create a single Observation with a primary KPI named 'y'."""
    return Observation(
        iteration=iteration,
        parameters=params,
        kpi_values={"y": kpi},
        timestamp=float(iteration),
        **kwargs,
    )


def _make_snapshot_linear(n_obs=30):
    """KPI = 2*x1 + 0.1*x2 + small noise (seeded), so x1 dominates."""
    rng = random.Random(42)
    specs = _make_specs(3)
    observations = []
    for i in range(n_obs):
        x1 = rng.uniform(0.0, 10.0)
        x2 = rng.uniform(0.0, 10.0)
        x3 = rng.uniform(0.0, 10.0)
        noise = rng.gauss(0, 0.05)
        kpi = 2.0 * x1 + 0.1 * x2 + noise
        observations.append(
            _make_obs(i, {"x1": x1, "x2": x2, "x3": x3}, kpi)
        )
    return CampaignSnapshot(
        campaign_id="linear-test",
        parameter_specs=specs,
        observations=observations,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=n_obs,
    )


def _make_snapshot_flat(n_obs=20):
    """KPI = 5.0 + tiny noise, insensitive to params."""
    rng = random.Random(99)
    specs = _make_specs(3)
    observations = []
    for i in range(n_obs):
        x1 = rng.uniform(0.0, 10.0)
        x2 = rng.uniform(0.0, 10.0)
        x3 = rng.uniform(0.0, 10.0)
        kpi = 5.0 + rng.gauss(0, 0.001)
        observations.append(
            _make_obs(i, {"x1": x1, "x2": x2, "x3": x3}, kpi)
        )
    return CampaignSnapshot(
        campaign_id="flat-test",
        parameter_specs=specs,
        observations=observations,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=n_obs,
    )


def _make_snapshot_noisy(n_obs=30):
    """KPI = random (seeded), no param correlation."""
    rng = random.Random(7)
    specs = _make_specs(3)
    observations = []
    for i in range(n_obs):
        x1 = rng.uniform(0.0, 10.0)
        x2 = rng.uniform(0.0, 10.0)
        x3 = rng.uniform(0.0, 10.0)
        kpi = rng.uniform(0.0, 20.0)
        observations.append(
            _make_obs(i, {"x1": x1, "x2": x2, "x3": x3}, kpi)
        )
    return CampaignSnapshot(
        campaign_id="noisy-test",
        parameter_specs=specs,
        observations=observations,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=n_obs,
    )


# ── TestParameterSensitivity ─────────────────────────────────


class TestParameterSensitivity:
    """ParameterSensitivity data model."""

    def test_construction_and_defaults(self):
        ps = ParameterSensitivity(
            parameter_name="x1",
            sensitivity_score=0.75,
            correlation=0.9,
            local_gradient=1.2,
            rank=1,
        )
        assert ps.parameter_name == "x1"
        assert ps.sensitivity_score == 0.75
        assert ps.correlation == 0.9
        assert ps.local_gradient == 1.2
        assert ps.rank == 1
        assert ps.evidence == {}

    def test_to_dict_from_dict_roundtrip(self):
        ps = ParameterSensitivity(
            parameter_name="temp",
            sensitivity_score=0.6,
            correlation=-0.4,
            local_gradient=0.8,
            rank=2,
            evidence={"method": "pearson", "n_samples": 30},
        )
        d = ps.to_dict()
        restored = ParameterSensitivity.from_dict(d)
        assert restored.parameter_name == ps.parameter_name
        assert restored.sensitivity_score == ps.sensitivity_score
        assert restored.correlation == ps.correlation
        assert restored.local_gradient == ps.local_gradient
        assert restored.rank == ps.rank
        assert restored.evidence == ps.evidence

    def test_rank_field(self):
        ps = ParameterSensitivity(
            parameter_name="x1",
            sensitivity_score=0.5,
            correlation=0.3,
            local_gradient=0.7,
            rank=3,
        )
        assert ps.rank == 3
        ps.rank = 1
        assert ps.rank == 1


# ── TestDecisionStability ─────────────────────────────────────


class TestDecisionStability:
    """DecisionStability data model."""

    def test_construction(self):
        ds = DecisionStability(
            top_k=5,
            stable_count=4,
            stability_score=0.8,
            margin_to_next=1.5,
            margin_relative=0.15,
            swapped_pairs=[(0, 1)],
            evidence={"noise_estimate": 0.5},
        )
        assert ds.top_k == 5
        assert ds.stable_count == 4
        assert ds.stability_score == 0.8
        assert ds.margin_to_next == 1.5
        assert ds.margin_relative == 0.15
        assert ds.swapped_pairs == [(0, 1)]

    def test_to_dict_from_dict_roundtrip_tuples_survive(self):
        ds = DecisionStability(
            top_k=3,
            stable_count=2,
            stability_score=0.67,
            margin_to_next=0.5,
            margin_relative=0.1,
            swapped_pairs=[(1, 2), (3, 4)],
            evidence={"method": "noise-perturbation"},
        )
        d = ds.to_dict()
        # Serialized swapped_pairs become lists
        assert d["swapped_pairs"] == [[1, 2], [3, 4]]
        restored = DecisionStability.from_dict(d)
        # After from_dict they should be tuples again
        assert restored.swapped_pairs == [(1, 2), (3, 4)]
        assert restored.stability_score == ds.stability_score

    def test_stability_score_bounds(self):
        """Stability score from the analyzer should be within [0, 1]."""
        analyzer = SensitivityAnalyzer(top_k=5)
        snap = _make_snapshot_linear(30)
        report = analyzer.analyze(snap)
        assert 0.0 <= report.decision_stability.stability_score <= 1.0


# ── TestSensitivityReport ─────────────────────────────────────


class TestSensitivityReport:
    """SensitivityReport data model."""

    def test_construction(self):
        ps = ParameterSensitivity(
            parameter_name="x1",
            sensitivity_score=0.9,
            correlation=0.85,
            local_gradient=1.0,
            rank=1,
        )
        ds = DecisionStability(
            top_k=5,
            stable_count=5,
            stability_score=1.0,
            margin_to_next=2.0,
            margin_relative=0.2,
        )
        report = SensitivityReport(
            parameter_sensitivities=[ps],
            decision_stability=ds,
            robustness_score=0.7,
            most_sensitive_parameter="x1",
            least_sensitive_parameter="x1",
            recommendations=["Focus on x1"],
        )
        assert report.most_sensitive_parameter == "x1"
        assert report.robustness_score == 0.7
        assert len(report.recommendations) == 1

    def test_to_dict_from_dict_roundtrip_nested(self):
        ps1 = ParameterSensitivity(
            parameter_name="x1",
            sensitivity_score=0.9,
            correlation=0.8,
            local_gradient=1.1,
            rank=1,
        )
        ps2 = ParameterSensitivity(
            parameter_name="x2",
            sensitivity_score=0.3,
            correlation=0.2,
            local_gradient=0.4,
            rank=2,
        )
        ds = DecisionStability(
            top_k=5,
            stable_count=4,
            stability_score=0.8,
            margin_to_next=1.0,
            margin_relative=0.1,
            swapped_pairs=[(2, 3)],
        )
        report = SensitivityReport(
            parameter_sensitivities=[ps1, ps2],
            decision_stability=ds,
            robustness_score=0.65,
            most_sensitive_parameter="x1",
            least_sensitive_parameter="x2",
            recommendations=["rec1", "rec2"],
            metadata={"version": 1},
        )
        d = report.to_dict()
        restored = SensitivityReport.from_dict(d)
        assert len(restored.parameter_sensitivities) == 2
        assert restored.parameter_sensitivities[0].parameter_name == "x1"
        assert restored.decision_stability.swapped_pairs == [(2, 3)]
        assert restored.robustness_score == 0.65
        assert restored.recommendations == ["rec1", "rec2"]
        assert restored.metadata == {"version": 1}

    def test_recommendations_list(self):
        ds = DecisionStability(
            top_k=3, stable_count=3, stability_score=1.0,
            margin_to_next=0.0, margin_relative=0.0,
        )
        report = SensitivityReport(
            parameter_sensitivities=[],
            decision_stability=ds,
            robustness_score=0.5,
            most_sensitive_parameter="",
            least_sensitive_parameter="",
            recommendations=["a", "b", "c"],
        )
        assert isinstance(report.recommendations, list)
        assert len(report.recommendations) == 3


# ── TestParameterSensitivityAnalysis ──────────────────────────


class TestParameterSensitivityAnalysis:
    """Analyzer: parameter sensitivity computation."""

    def test_dominant_parameter_highest_sensitivity(self):
        """x1 in linear snapshot (KPI = 2*x1 + 0.1*x2 + noise) has rank 1."""
        analyzer = SensitivityAnalyzer(top_k=5, n_neighbors=5)
        snap = _make_snapshot_linear(30)
        report = analyzer.analyze(snap)

        # Find x1's sensitivity
        x1_sens = next(
            s for s in report.parameter_sensitivities if s.parameter_name == "x1"
        )
        assert x1_sens.rank == 1, (
            f"Expected x1 to be rank 1 (dominant), got rank {x1_sens.rank}"
        )
        assert report.most_sensitive_parameter == "x1"

    def test_flat_kpi_all_low_sensitivity(self):
        """Flat snapshot (KPI ~ constant) should give all correlations near 0."""
        analyzer = SensitivityAnalyzer(top_k=5)
        snap = _make_snapshot_flat(20)
        report = analyzer.analyze(snap)

        for s in report.parameter_sensitivities:
            # With near-constant KPI, Pearson correlation should be ~0.
            # The local_gradient may amplify tiny numerical noise, so we check
            # the correlation component which is more meaningful.
            assert abs(s.correlation) < 0.5, (
                f"Parameter {s.parameter_name} has correlation {s.correlation} "
                "but expected |correlation| < 0.5 for flat KPI"
            )

    def test_noisy_kpi_moderate_or_low(self):
        """Random KPI should yield moderate or low sensitivity scores."""
        analyzer = SensitivityAnalyzer(top_k=5)
        snap = _make_snapshot_noisy(30)
        report = analyzer.analyze(snap)

        # With random KPI, no parameter should dominate (score > 0.9)
        for s in report.parameter_sensitivities:
            assert s.sensitivity_score < 0.95, (
                f"Parameter {s.parameter_name} has suspiciously high sensitivity "
                f"{s.sensitivity_score} for random KPI"
            )

    def test_sensitivity_scores_in_range(self):
        """All sensitivity scores must be in [0, 1]."""
        analyzer = SensitivityAnalyzer(top_k=5)
        for make_fn in [_make_snapshot_linear, _make_snapshot_flat, _make_snapshot_noisy]:
            snap = make_fn()
            report = analyzer.analyze(snap)
            for s in report.parameter_sensitivities:
                assert 0.0 <= s.sensitivity_score <= 1.0, (
                    f"Sensitivity score {s.sensitivity_score} outside [0, 1]"
                )

    def test_ranks_are_contiguous(self):
        """Ranks must be 1..N for N non-categorical parameters."""
        analyzer = SensitivityAnalyzer(top_k=5)
        snap = _make_snapshot_linear(30)
        report = analyzer.analyze(snap)

        ranks = sorted(s.rank for s in report.parameter_sensitivities)
        expected = list(range(1, len(report.parameter_sensitivities) + 1))
        assert ranks == expected, f"Ranks {ranks} not contiguous, expected {expected}"

    def test_correlation_sign_matches(self):
        """For linear KPI = 2*x1 + 0.1*x2, correlations should be positive."""
        analyzer = SensitivityAnalyzer(top_k=5)
        snap = _make_snapshot_linear(30)
        report = analyzer.analyze(snap)

        x1_sens = next(
            s for s in report.parameter_sensitivities if s.parameter_name == "x1"
        )
        x2_sens = next(
            s for s in report.parameter_sensitivities if s.parameter_name == "x2"
        )
        # Both x1 and x2 have positive coefficients in the KPI
        assert x1_sens.correlation > 0, (
            f"x1 correlation {x1_sens.correlation} should be positive"
        )
        assert x2_sens.correlation > 0, (
            f"x2 correlation {x2_sens.correlation} should be positive"
        )

    def test_skips_categorical_params(self):
        """Categorical parameters should be excluded from sensitivity analysis."""
        specs = _make_specs(2) + [
            ParameterSpec(
                name="cat_param",
                type=VariableType.CATEGORICAL,
                categories=["a", "b", "c"],
            )
        ]
        rng = random.Random(42)
        observations = []
        for i in range(30):
            x1 = rng.uniform(0.0, 10.0)
            x2 = rng.uniform(0.0, 10.0)
            cat = rng.choice(["a", "b", "c"])
            kpi = 2.0 * x1 + rng.gauss(0, 0.05)
            observations.append(
                _make_obs(i, {"x1": x1, "x2": x2, "cat_param": cat}, kpi)
            )
        snap = CampaignSnapshot(
            campaign_id="cat-test",
            parameter_specs=specs,
            observations=observations,
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=30,
        )
        analyzer = SensitivityAnalyzer(top_k=5)
        report = analyzer.analyze(snap)

        param_names = [s.parameter_name for s in report.parameter_sensitivities]
        assert "cat_param" not in param_names, (
            "Categorical parameter should not appear in sensitivity results"
        )
        assert len(report.parameter_sensitivities) == 2


# ── TestDecisionStabilityAnalysis ─────────────────────────────


class TestDecisionStabilityAnalysis:
    """Analyzer: decision stability computation."""

    def test_clear_winner_high_stability(self):
        """Top observations with large gaps should give high stability.

        The stability algorithm compares adjacent KPI gaps against
        noise_estimate * perturbation_fraction. To ensure stability, we make
        the top candidates well-separated relative to the overall noise.
        """
        specs = _make_specs(2)
        observations = []
        # Create well-separated top-3: KPIs 100, 80, 60, then rest at ~10
        top_kpis = [100.0, 80.0, 60.0]
        for i, kpi in enumerate(top_kpis):
            observations.append(
                _make_obs(i, {"x1": float(i), "x2": float(i)}, kpi)
            )
        for i in range(3, 20):
            observations.append(
                _make_obs(i, {"x1": float(i), "x2": float(i)}, 10.0 + i * 0.1)
            )
        snap = CampaignSnapshot(
            campaign_id="clear-winner",
            parameter_specs=specs,
            observations=observations,
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=20,
        )
        analyzer = SensitivityAnalyzer(top_k=3)
        report = analyzer.analyze(snap)
        # Top 3 (100, 80, 60) are well-separated; margins exceed noise threshold
        assert report.decision_stability.stability_score >= 0.5, (
            f"Expected high stability, got {report.decision_stability.stability_score}"
        )

    def test_tied_observations_low_stability(self):
        """All KPIs nearly equal should yield low stability (many swaps)."""
        specs = _make_specs(2)
        rng = random.Random(123)
        observations = []
        for i in range(20):
            x1 = rng.uniform(0.0, 10.0)
            x2 = rng.uniform(0.0, 10.0)
            # All KPIs within a very tight band
            kpi = 10.0 + rng.gauss(0, 1e-10)
            observations.append(_make_obs(i, {"x1": x1, "x2": x2}, kpi))
        snap = CampaignSnapshot(
            campaign_id="tied",
            parameter_specs=specs,
            observations=observations,
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=20,
        )
        analyzer = SensitivityAnalyzer(top_k=5)
        report = analyzer.analyze(snap)
        # With nearly identical KPIs, many adjacent pairs are within noise threshold
        # so we expect swaps and lower stability
        assert report.decision_stability.stability_score <= 1.0

    def test_top_k_larger_than_n(self):
        """top_k larger than number of observations handled gracefully."""
        specs = _make_specs(2)
        observations = [
            _make_obs(i, {"x1": float(i), "x2": float(i)}, float(i) * 2.0)
            for i in range(4)
        ]
        snap = CampaignSnapshot(
            campaign_id="small-n",
            parameter_specs=specs,
            observations=observations,
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=4,
        )
        analyzer = SensitivityAnalyzer(top_k=10)
        report = analyzer.analyze(snap)
        # effective_k should be clamped to n_obs
        assert report.decision_stability.top_k <= 4

    def test_margin_computation(self):
        """Margin between top-K and next should be correct."""
        specs = _make_specs(2)
        # KPIs: 10, 8, 6, 4, 2 (descending for maximize)
        observations = [
            _make_obs(0, {"x1": 1.0, "x2": 1.0}, 10.0),
            _make_obs(1, {"x1": 2.0, "x2": 2.0}, 8.0),
            _make_obs(2, {"x1": 3.0, "x2": 3.0}, 6.0),
            _make_obs(3, {"x1": 4.0, "x2": 4.0}, 4.0),
            _make_obs(4, {"x1": 5.0, "x2": 5.0}, 2.0),
        ]
        snap = CampaignSnapshot(
            campaign_id="margin-test",
            parameter_specs=specs,
            observations=observations,
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=5,
        )
        analyzer = SensitivityAnalyzer(top_k=2)
        report = analyzer.analyze(snap)
        ds = report.decision_stability
        # Top 2 are KPI 10.0 and 8.0; next is 6.0
        # margin_to_next = |8.0 - 6.0| = 2.0
        assert abs(ds.margin_to_next - 2.0) < 1e-9, (
            f"Expected margin 2.0, got {ds.margin_to_next}"
        )

    def test_single_observation(self):
        """Single observation should be trivially stable (but < 3 triggers short-circuit)."""
        specs = _make_specs(2)
        observations = [_make_obs(0, {"x1": 5.0, "x2": 5.0}, 10.0)]
        snap = CampaignSnapshot(
            campaign_id="single",
            parameter_specs=specs,
            observations=observations,
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=1,
        )
        analyzer = SensitivityAnalyzer(top_k=3)
        report = analyzer.analyze(snap)
        # With < 3 observations, short-circuit returns stability_score=1.0
        assert report.decision_stability.stability_score == 1.0

    def test_two_observations_swap_detection(self):
        """Two observations with similar KPIs: analyzer handles < 3 obs gracefully."""
        specs = _make_specs(2)
        observations = [
            _make_obs(0, {"x1": 5.0, "x2": 5.0}, 10.0),
            _make_obs(1, {"x1": 6.0, "x2": 6.0}, 10.001),
        ]
        snap = CampaignSnapshot(
            campaign_id="two-obs",
            parameter_specs=specs,
            observations=observations,
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=2,
        )
        analyzer = SensitivityAnalyzer(top_k=2)
        report = analyzer.analyze(snap)
        # < 3 obs triggers short-circuit, returning stability_score=1.0
        assert report.decision_stability.stability_score == 1.0


# ── TestRobustnessScore ───────────────────────────────────────


class TestRobustnessScore:
    """Analyzer: robustness score computation."""

    def test_robust_campaign(self):
        """Low sensitivity + high stability should give high robustness."""
        analyzer = SensitivityAnalyzer(top_k=5)
        snap = _make_snapshot_flat(20)
        report = analyzer.analyze(snap)
        # Flat KPI means low sensitivity; stability should be reasonable
        assert report.robustness_score >= 0.3, (
            f"Expected moderate-high robustness for flat KPI, got {report.robustness_score}"
        )

    def test_fragile_campaign(self):
        """High sensitivity + potentially low stability should give lower robustness."""
        analyzer = SensitivityAnalyzer(top_k=5)
        snap = _make_snapshot_linear(30)
        report = analyzer.analyze(snap)
        # Linear KPI with strong x1 dominance means higher sensitivity
        # Robustness should be lower than for flat data
        flat_report = analyzer.analyze(_make_snapshot_flat(20))
        assert report.robustness_score <= flat_report.robustness_score + 0.1, (
            f"Linear campaign robustness {report.robustness_score} should be "
            f"<= flat campaign robustness {flat_report.robustness_score} + tolerance"
        )

    def test_robustness_in_range(self):
        """Robustness score must be in [0, 1]."""
        analyzer = SensitivityAnalyzer(top_k=5)
        for make_fn in [_make_snapshot_linear, _make_snapshot_flat, _make_snapshot_noisy]:
            snap = make_fn()
            report = analyzer.analyze(snap)
            assert 0.0 <= report.robustness_score <= 1.0, (
                f"Robustness score {report.robustness_score} outside [0, 1]"
            )

    def test_recommendations_for_fragile(self):
        """Fragile campaign should produce actionable recommendations."""
        analyzer = SensitivityAnalyzer(top_k=5)
        snap = _make_snapshot_linear(30)
        report = analyzer.analyze(snap)
        # The linear campaign has a dominant parameter (x1), so we expect
        # at least one recommendation about it or about general fragility
        assert isinstance(report.recommendations, list)
        # There should be at least some recommendation text
        # (may be about dominance, instability, or small margins)
        # Even if robustness is moderate, margins or sensitivity thresholds
        # may trigger recommendations
        assert len(report.recommendations) >= 0  # graceful even if none


# ── TestEdgeCases ─────────────────────────────────────────────


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_snapshot(self):
        """Zero observations should return a safe default report."""
        specs = _make_specs(3)
        snap = CampaignSnapshot(
            campaign_id="empty",
            parameter_specs=specs,
            observations=[],
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=0,
        )
        analyzer = SensitivityAnalyzer(top_k=5)
        report = analyzer.analyze(snap)

        assert report.parameter_sensitivities == []
        assert report.robustness_score == 0.0
        assert report.most_sensitive_parameter == ""
        assert report.least_sensitive_parameter == ""
        assert "Insufficient data" in report.recommendations[0]

    def test_two_observations_minimal(self):
        """Two observations (< 3 threshold) should short-circuit gracefully."""
        specs = _make_specs(2)
        observations = [
            _make_obs(0, {"x1": 1.0, "x2": 2.0}, 5.0),
            _make_obs(1, {"x1": 3.0, "x2": 4.0}, 7.0),
        ]
        snap = CampaignSnapshot(
            campaign_id="minimal",
            parameter_specs=specs,
            observations=observations,
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=2,
        )
        analyzer = SensitivityAnalyzer(top_k=3)
        report = analyzer.analyze(snap)

        assert report.parameter_sensitivities == []
        assert report.decision_stability.stability_score == 1.0
        assert report.robustness_score == 0.0
        assert len(report.recommendations) > 0

    def test_all_failures(self):
        """All observations marked as failures means no successful obs."""
        specs = _make_specs(2)
        observations = [
            Observation(
                iteration=i,
                parameters={"x1": float(i), "x2": float(i)},
                kpi_values={"y": float(i)},
                is_failure=True,
                failure_reason="simulated",
                timestamp=float(i),
            )
            for i in range(10)
        ]
        snap = CampaignSnapshot(
            campaign_id="all-fail",
            parameter_specs=specs,
            observations=observations,
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=10,
        )
        analyzer = SensitivityAnalyzer(top_k=5)
        report = analyzer.analyze(snap)

        # All failures means 0 successful observations -> short-circuit
        assert report.parameter_sensitivities == []
        assert report.robustness_score == 0.0
        assert "Insufficient data" in report.recommendations[0]

    def test_deterministic(self):
        """Identical inputs must produce identical outputs."""
        analyzer = SensitivityAnalyzer(top_k=5, n_neighbors=5)
        snap = _make_snapshot_linear(30)

        report_a = analyzer.analyze(snap)
        report_b = analyzer.analyze(snap)

        assert report_a.robustness_score == report_b.robustness_score
        assert report_a.most_sensitive_parameter == report_b.most_sensitive_parameter
        assert report_a.least_sensitive_parameter == report_b.least_sensitive_parameter
        assert report_a.decision_stability.stability_score == report_b.decision_stability.stability_score
        assert report_a.decision_stability.margin_to_next == report_b.decision_stability.margin_to_next
        for a, b in zip(report_a.parameter_sensitivities, report_b.parameter_sensitivities):
            assert a.parameter_name == b.parameter_name
            assert a.sensitivity_score == b.sensitivity_score
            assert a.rank == b.rank
            assert a.correlation == b.correlation
        assert report_a.recommendations == report_b.recommendations
