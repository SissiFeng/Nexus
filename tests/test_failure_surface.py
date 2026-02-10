"""Tests for failure surface learning and avoidance (Track 3).

Verifies:
- KNN failure probability estimation
- Safe bound computation from successful observations
- Danger zone detection from failure patterns
- Risk-adjusted scoring: objective - lambda * p_fail
- Lambda modulation by RiskPosture
- Death zone avoidance: candidates in failure-heavy region get low scores
- Failure-type-specific adjustments
"""

from __future__ import annotations

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    RiskPosture,
    VariableType,
)
from optimization_copilot.feasibility.surface import (
    DangerZone,
    FailureAdjustment,
    FailureProbability,
    FailureSurface,
    FailureSurfaceLearner,
)
from optimization_copilot.feasibility.taxonomy import (
    FailureClassifier,
    FailureType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
    ]


def _make_snapshot_with_death_zone(
    n_safe: int = 15,
    n_danger: int = 15,
) -> CampaignSnapshot:
    """Create a snapshot where x1 > 7 is a 'death zone' with high failure.

    Safe region: x1 in [0, 7], all successful.
    Death zone: x1 in [7, 10], mostly failures.
    """
    specs = _make_specs()
    obs: list[Observation] = []
    idx = 0

    # Safe observations
    for i in range(n_safe):
        x1 = float(i) / n_safe * 7.0
        obs.append(Observation(
            iteration=idx,
            parameters={"x1": x1, "x2": 5.0},
            kpi_values={"y": 10.0 + x1},
            is_failure=False,
            timestamp=float(idx),
        ))
        idx += 1

    # Danger zone: mostly failures
    for i in range(n_danger):
        x1 = 7.0 + float(i) / n_danger * 3.0
        is_fail = i < int(n_danger * 0.8)  # 80% failure rate
        obs.append(Observation(
            iteration=idx,
            parameters={"x1": x1, "x2": 5.0},
            kpi_values={"y": 0.0 if is_fail else 5.0},
            is_failure=is_fail,
            failure_reason="precipitate formed" if is_fail else None,
            timestamp=float(idx),
        ))
        idx += 1

    return CampaignSnapshot(
        campaign_id="death-zone-test",
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=idx,
    )


def _make_clean_snapshot(n_obs: int = 20) -> CampaignSnapshot:
    """Snapshot with zero failures."""
    specs = _make_specs()
    obs = [
        Observation(
            iteration=i,
            parameters={"x1": float(i) / n_obs * 10, "x2": 5.0},
            kpi_values={"y": float(i) + 1},
            is_failure=False,
            timestamp=float(i),
        )
        for i in range(n_obs)
    ]
    return CampaignSnapshot(
        campaign_id="clean-test",
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=n_obs,
    )


class _MockTaxonomy:
    """Minimal mock for FailureTaxonomy."""

    def __init__(self, dominant: str = "unknown", type_rates: dict | None = None):
        self.dominant_type = type(
            "FT", (), {"value": dominant}
        )()
        self.type_rates = type_rates or {}
        self.classified_failures = []


# ---------------------------------------------------------------------------
# Tests: FailureSurface learning
# ---------------------------------------------------------------------------


class TestFailureSurfaceLearn:
    def test_basic_surface_learning(self):
        """Learn a failure surface from data with a death zone."""
        snap = _make_snapshot_with_death_zone()
        learner = FailureSurfaceLearner(k=5)
        surface = learner.learn(snap)

        assert surface.n_observations == 30
        assert surface.n_failures > 0
        assert surface.overall_failure_rate > 0.0

    def test_safe_bounds_from_successes(self):
        """Safe bounds should span successful observations."""
        snap = _make_clean_snapshot(20)
        learner = FailureSurfaceLearner()
        surface = learner.learn(snap)

        assert "x1" in surface.safe_bounds
        lo, hi = surface.safe_bounds["x1"]
        assert lo >= 0.0
        assert hi <= 10.0

    def test_danger_zones_detected(self):
        """Death zone should produce danger zone entries."""
        snap = _make_snapshot_with_death_zone(n_safe=15, n_danger=15)
        learner = FailureSurfaceLearner(danger_zone_threshold=0.3, min_samples_for_zone=3)
        surface = learner.learn(snap)

        # Should detect elevated failure rate in upper x1 region.
        x1_zones = [dz for dz in surface.danger_zones if dz.parameter == "x1"]
        assert len(x1_zones) > 0

    def test_no_danger_zones_clean_data(self):
        """Clean data should not produce danger zones."""
        snap = _make_clean_snapshot(20)
        learner = FailureSurfaceLearner()
        surface = learner.learn(snap)

        assert len(surface.danger_zones) == 0

    def test_failure_density_computed(self):
        """Failure density should be computed per parameter."""
        snap = _make_snapshot_with_death_zone()
        learner = FailureSurfaceLearner(n_bins=10)
        surface = learner.learn(snap)

        assert "x1" in surface.parameter_failure_density
        density = surface.parameter_failure_density["x1"]
        assert len(density) == 10
        # Higher bins (upper x1) should have higher failure density.
        assert density[-1] >= density[0]

    def test_empty_snapshot(self):
        """Empty snapshot should not crash."""
        snap = CampaignSnapshot(
            campaign_id="empty",
            parameter_specs=_make_specs(),
            observations=[],
            objective_names=["y"],
            objective_directions=["maximize"],
        )
        learner = FailureSurfaceLearner()
        surface = learner.learn(snap)

        assert surface.n_observations == 0
        assert surface.overall_failure_rate == 0.0


# ---------------------------------------------------------------------------
# Tests: KNN failure prediction
# ---------------------------------------------------------------------------


class TestPredictFailure:
    def test_safe_point_low_probability(self):
        """Point in safe region should have low failure probability."""
        snap = _make_snapshot_with_death_zone()
        learner = FailureSurfaceLearner(k=5)

        safe_candidate = {"x1": 3.0, "x2": 5.0}
        prob = learner.predict_failure(safe_candidate, snap)

        assert prob.p_fail < 0.5

    def test_danger_point_high_probability(self):
        """Point in death zone should have high failure probability."""
        snap = _make_snapshot_with_death_zone()
        learner = FailureSurfaceLearner(k=5)

        danger_candidate = {"x1": 9.0, "x2": 5.0}
        prob = learner.predict_failure(danger_candidate, snap)

        assert prob.p_fail > 0.3

    def test_k_neighbors_reported(self):
        """Result should report number of neighbors used."""
        snap = _make_snapshot_with_death_zone()
        learner = FailureSurfaceLearner(k=5)

        prob = learner.predict_failure({"x1": 5.0, "x2": 5.0}, snap)
        assert prob.n_neighbors == 5

    def test_few_observations_graceful(self):
        """With very few observations, should still produce a result."""
        specs = _make_specs()
        obs = [
            Observation(iteration=0, parameters={"x1": 1.0, "x2": 1.0},
                        kpi_values={"y": 5.0}, is_failure=False, timestamp=0.0),
        ]
        snap = CampaignSnapshot(
            campaign_id="tiny",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["maximize"],
        )
        learner = FailureSurfaceLearner(k=5)
        prob = learner.predict_failure({"x1": 1.0, "x2": 1.0}, snap)

        assert prob.p_fail >= 0.0
        assert prob.n_neighbors <= 1

    def test_failure_type_breakdown(self):
        """With taxonomy, should report per-type probabilities."""
        snap = _make_snapshot_with_death_zone()
        classifier = FailureClassifier()
        taxonomy = classifier.classify(snap)

        learner = FailureSurfaceLearner(k=5)
        prob = learner.predict_failure(
            {"x1": 9.0, "x2": 5.0}, snap, failure_taxonomy=taxonomy
        )

        # Should have at least some type breakdown if there are failures nearby.
        if prob.p_fail > 0:
            assert len(prob.p_by_type) > 0 or prob.p_fail == 0


# ---------------------------------------------------------------------------
# Tests: Risk-adjusted scoring
# ---------------------------------------------------------------------------


class TestAdjustScore:
    def test_conservative_penalizes_more(self):
        """Conservative risk posture should penalize failure more."""
        learner = FailureSurfaceLearner()
        prob = FailureProbability(p_fail=0.5, n_neighbors=5, avg_distance=0.1)

        score_conservative = learner.adjust_score(
            1.0, prob, RiskPosture.CONSERVATIVE
        )
        score_aggressive = learner.adjust_score(
            1.0, prob, RiskPosture.AGGRESSIVE
        )

        assert score_conservative < score_aggressive

    def test_zero_failure_no_penalty(self):
        """Zero failure probability should not change the score."""
        learner = FailureSurfaceLearner()
        prob = FailureProbability(p_fail=0.0, n_neighbors=5, avg_distance=0.1)

        score = learner.adjust_score(1.0, prob, RiskPosture.MODERATE)
        assert score == 1.0

    def test_full_failure_reduces_score(self):
        """Certain failure should significantly reduce the score."""
        learner = FailureSurfaceLearner()
        prob = FailureProbability(p_fail=1.0, n_neighbors=5, avg_distance=0.0)

        score = learner.adjust_score(1.0, prob, RiskPosture.MODERATE)
        assert score < 1.0

    def test_lambda_values(self):
        """Verify exact lambda values for each risk posture."""
        learner = FailureSurfaceLearner()
        prob = FailureProbability(p_fail=1.0, n_neighbors=5, avg_distance=0.0)

        assert learner.adjust_score(1.0, prob, RiskPosture.CONSERVATIVE) == 0.0
        assert learner.adjust_score(1.0, prob, RiskPosture.MODERATE) == 0.5
        assert abs(learner.adjust_score(1.0, prob, RiskPosture.AGGRESSIVE) - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# Tests: Death zone avoidance (integration)
# ---------------------------------------------------------------------------


class TestDeathZoneAvoidance:
    def test_safe_candidates_score_higher(self):
        """Candidates in safe region should score higher than death zone."""
        snap = _make_snapshot_with_death_zone()
        learner = FailureSurfaceLearner(k=5)

        safe = {"x1": 3.0, "x2": 5.0}
        danger = {"x1": 9.0, "x2": 5.0}

        p_safe = learner.predict_failure(safe, snap)
        p_danger = learner.predict_failure(danger, snap)

        # Same objective, different risk.
        score_safe = learner.adjust_score(10.0, p_safe, RiskPosture.MODERATE)
        score_danger = learner.adjust_score(10.0, p_danger, RiskPosture.MODERATE)

        assert score_safe > score_danger

    def test_aggressive_explorer_enters_death_zone(self):
        """Aggressive risk posture should tolerate entering death zone more."""
        snap = _make_snapshot_with_death_zone()
        learner = FailureSurfaceLearner(k=5)

        danger = {"x1": 9.0, "x2": 5.0}
        p_danger = learner.predict_failure(danger, snap)

        score_conservative = learner.adjust_score(
            10.0, p_danger, RiskPosture.CONSERVATIVE
        )
        score_aggressive = learner.adjust_score(
            10.0, p_danger, RiskPosture.AGGRESSIVE
        )

        assert score_aggressive > score_conservative


# ---------------------------------------------------------------------------
# Tests: Failure-type-specific adjustments
# ---------------------------------------------------------------------------


class TestFailureAdjustments:
    def test_hardware_reduces_exploration(self):
        taxonomy = _MockTaxonomy(dominant="hardware")
        learner = FailureSurfaceLearner()
        adjustments = learner.recommend_adjustments(failure_taxonomy=taxonomy)

        types = [a.adjustment_type for a in adjustments]
        assert "reduce_exploration" in types

    def test_chemistry_tightens_bounds(self):
        taxonomy = _MockTaxonomy(dominant="chemistry")
        learner = FailureSurfaceLearner()
        adjustments = learner.recommend_adjustments(failure_taxonomy=taxonomy)

        types = [a.adjustment_type for a in adjustments]
        assert "tighten_bounds" in types

    def test_data_increases_replicates(self):
        taxonomy = _MockTaxonomy(dominant="data")
        learner = FailureSurfaceLearner()
        adjustments = learner.recommend_adjustments(failure_taxonomy=taxonomy)

        types = [a.adjustment_type for a in adjustments]
        assert "increase_replicates" in types

    def test_protocol_enforces_checks(self):
        taxonomy = _MockTaxonomy(dominant="protocol")
        learner = FailureSurfaceLearner()
        adjustments = learner.recommend_adjustments(failure_taxonomy=taxonomy)

        types = [a.adjustment_type for a in adjustments]
        assert "enforce_protocol_checks" in types

    def test_high_failure_rate_conservative(self):
        """High overall failure rate triggers conservative exploration."""
        learner = FailureSurfaceLearner()
        surface = FailureSurface(
            safe_bounds={},
            danger_zones=[],
            parameter_failure_density={},
            n_observations=100,
            n_failures=40,
            overall_failure_rate=0.4,
        )
        adjustments = learner.recommend_adjustments(failure_surface=surface)

        types = [a.adjustment_type for a in adjustments]
        assert "conservative_exploration" in types

    def test_no_taxonomy_no_adjustments(self):
        """Without taxonomy or surface, no adjustments."""
        learner = FailureSurfaceLearner()
        adjustments = learner.recommend_adjustments()
        assert len(adjustments) == 0
