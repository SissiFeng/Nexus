"""Tests for the feasibility-first optimization module.

Verifies:
- FeasibilityClassifier: KNN-based feasibility predictions with confidence
- FeasibilityFirstScorer: Adaptive alpha blending of feasibility + objective
- SafetyBoundaryLearner: Quantile-based safe operating region learning

~30 tests covering unit, edge-case, and integration scenarios.
"""

from __future__ import annotations

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.feasibility_first.classifier import (
    FeasibilityClassifier,
    FeasibilityPrediction,
)
from optimization_copilot.feasibility_first.scorer import (
    FeasibilityFirstScorer,
    ScoredCandidate,
)
from optimization_copilot.feasibility_first.boundary import (
    SafetyBoundaryLearner,
    SafetyBoundary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_specs():
    return [
        ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
    ]


def _make_snapshot_with_death_zone(n_safe=15, n_danger=15):
    """x1 < 7 = safe region, x1 > 7 = death zone (80% failure)."""
    specs = _make_specs()
    obs = []
    idx = 0
    for i in range(n_safe):
        x1 = float(i) / n_safe * 7.0
        obs.append(Observation(
            iteration=idx, parameters={"x1": x1, "x2": 5.0},
            kpi_values={"y": 10.0 + x1}, is_failure=False, timestamp=float(idx),
        ))
        idx += 1
    for i in range(n_danger):
        x1 = 7.0 + float(i) / n_danger * 3.0
        is_fail = i < int(n_danger * 0.8)
        obs.append(Observation(
            iteration=idx, parameters={"x1": x1, "x2": 5.0},
            kpi_values={"y": 0.0 if is_fail else 5.0},
            is_failure=is_fail, timestamp=float(idx),
        ))
        idx += 1
    return CampaignSnapshot(
        campaign_id="ff-test", parameter_specs=specs, observations=obs,
        objective_names=["y"], objective_directions=["maximize"],
        current_iteration=idx,
    )


def _make_clean_snapshot(n_obs=20):
    """All successful observations."""
    specs = _make_specs()
    obs = [
        Observation(
            iteration=i, parameters={"x1": float(i) / n_obs * 10, "x2": 5.0},
            kpi_values={"y": float(i) + 1}, is_failure=False, timestamp=float(i),
        )
        for i in range(n_obs)
    ]
    return CampaignSnapshot(
        campaign_id="clean-ff", parameter_specs=specs, observations=obs,
        objective_names=["y"], objective_directions=["maximize"],
        current_iteration=n_obs,
    )


def _make_all_failure_snapshot(n_obs=20):
    """All failed observations."""
    specs = _make_specs()
    obs = [
        Observation(
            iteration=i, parameters={"x1": float(i) / n_obs * 10, "x2": 5.0},
            kpi_values={"y": 0.0}, is_failure=True, timestamp=float(i),
        )
        for i in range(n_obs)
    ]
    return CampaignSnapshot(
        campaign_id="fail-ff", parameter_specs=specs, observations=obs,
        objective_names=["y"], objective_directions=["maximize"],
        current_iteration=n_obs,
    )


# ---------------------------------------------------------------------------
# TestFeasibilityClassifier
# ---------------------------------------------------------------------------


class TestFeasibilityClassifier:
    """Tests for the FeasibilityClassifier wrapper around FailureSurfaceLearner."""

    def test_safe_point_high_feasibility(self):
        """Point at x1=3 in safe zone should have p_feasible > 0.5."""
        snap = _make_snapshot_with_death_zone()
        clf = FeasibilityClassifier(k=5)
        pred = clf.predict({"x1": 3.0, "x2": 5.0}, snap)

        assert isinstance(pred, FeasibilityPrediction)
        assert pred.p_feasible > 0.5

    def test_danger_point_low_feasibility(self):
        """Point at x1=9 in death zone should have p_feasible < 0.7."""
        snap = _make_snapshot_with_death_zone()
        clf = FeasibilityClassifier(k=5)
        pred = clf.predict({"x1": 9.0, "x2": 5.0}, snap)

        assert pred.p_feasible < 0.7

    def test_inverts_failure_probability(self):
        """p_feasible + p_failure should approximately equal 1.0."""
        snap = _make_snapshot_with_death_zone()
        clf = FeasibilityClassifier(k=5)
        pred = clf.predict({"x1": 5.0, "x2": 5.0}, snap)

        assert abs(pred.p_feasible + pred.p_failure - 1.0) < 1e-6

    def test_confidence_scales_with_neighbors(self):
        """With enough data, confidence should be > 0."""
        snap = _make_snapshot_with_death_zone()
        clf = FeasibilityClassifier(k=5)
        pred = clf.predict({"x1": 5.0, "x2": 5.0}, snap)

        assert pred.confidence > 0.0

    def test_few_observations_graceful(self):
        """Snapshot with 1 observation should still return a valid result."""
        specs = _make_specs()
        obs = [
            Observation(
                iteration=0, parameters={"x1": 5.0, "x2": 5.0},
                kpi_values={"y": 10.0}, is_failure=False, timestamp=0.0,
            ),
        ]
        snap = CampaignSnapshot(
            campaign_id="tiny-ff", parameter_specs=specs, observations=obs,
            objective_names=["y"], objective_directions=["maximize"],
            current_iteration=1,
        )
        clf = FeasibilityClassifier(k=5)
        pred = clf.predict({"x1": 5.0, "x2": 5.0}, snap)

        assert isinstance(pred, FeasibilityPrediction)
        assert 0.0 <= pred.p_feasible <= 1.0
        assert 0.0 <= pred.p_failure <= 1.0

    def test_empty_snapshot(self):
        """0 observations should return a safe default (p_feasible ~ 0.0 or 1.0)."""
        specs = _make_specs()
        snap = CampaignSnapshot(
            campaign_id="empty-ff", parameter_specs=specs, observations=[],
            objective_names=["y"], objective_directions=["maximize"],
            current_iteration=0,
        )
        clf = FeasibilityClassifier(k=5)
        pred = clf.predict({"x1": 5.0, "x2": 5.0}, snap)

        assert isinstance(pred, FeasibilityPrediction)
        # With zero observations, failure_rate = 0.0, so p_feasible = 1.0
        assert 0.0 <= pred.p_feasible <= 1.0

    def test_batch_prediction(self):
        """predict_batch returns same length as input."""
        snap = _make_snapshot_with_death_zone()
        clf = FeasibilityClassifier(k=5)
        candidates = [
            {"x1": 1.0, "x2": 5.0},
            {"x1": 5.0, "x2": 5.0},
            {"x1": 9.0, "x2": 5.0},
        ]
        preds = clf.predict_batch(candidates, snap)

        assert len(preds) == 3
        assert all(isinstance(p, FeasibilityPrediction) for p in preds)


# ---------------------------------------------------------------------------
# TestFeasibilityFirstScorer
# ---------------------------------------------------------------------------


class TestFeasibilityFirstScorer:
    """Tests for adaptive alpha blending of feasibility and objective scores."""

    def test_alpha_high_when_all_fail(self):
        """All failures should push alpha near alpha_max (0.9)."""
        snap = _make_all_failure_snapshot()
        scorer = FeasibilityFirstScorer(alpha_max=0.9, alpha_min=0.1)
        alpha = scorer.compute_alpha(snap)

        # failure_rate = 1.0, feas_rate = 0.0
        # alpha = alpha_max * 1.0 + alpha_min * 0.0 = 0.9
        assert abs(alpha - 0.9) < 0.05

    def test_alpha_low_when_all_succeed(self):
        """No failures should push alpha near alpha_min (0.1)."""
        snap = _make_clean_snapshot()
        scorer = FeasibilityFirstScorer(alpha_max=0.9, alpha_min=0.1)
        alpha = scorer.compute_alpha(snap)

        # failure_rate = 0.0, feas_rate = 1.0
        # alpha = alpha_max * 0.0 + alpha_min * 1.0 = 0.1
        assert abs(alpha - 0.1) < 0.05

    def test_alpha_intermediate(self):
        """~50% failure rate should give alpha approximately 0.5."""
        specs = _make_specs()
        obs = []
        for i in range(20):
            obs.append(Observation(
                iteration=i,
                parameters={"x1": float(i) / 20 * 10, "x2": 5.0},
                kpi_values={"y": 5.0},
                is_failure=(i % 2 == 0),  # 50% failure
                timestamp=float(i),
            ))
        snap = CampaignSnapshot(
            campaign_id="half-ff", parameter_specs=specs, observations=obs,
            objective_names=["y"], objective_directions=["maximize"],
            current_iteration=20,
        )
        scorer = FeasibilityFirstScorer(alpha_max=0.9, alpha_min=0.1)
        alpha = scorer.compute_alpha(snap)

        assert abs(alpha - 0.5) < 0.1

    def test_feasibility_dominates_early(self):
        """High failure rate: feasible candidates should rank first."""
        snap = _make_snapshot_with_death_zone()
        clf = FeasibilityClassifier(k=5)
        scorer = FeasibilityFirstScorer(alpha_max=0.9, alpha_min=0.1)

        candidates = [
            {"x1": 3.0, "x2": 5.0},  # safe
            {"x1": 9.0, "x2": 5.0},  # danger
        ]
        scored = scorer.score_candidates(
            candidates, snap, clf,
            objective_values=[10.0, 10.0],
            direction="maximize",
        )

        # Safe candidate should rank higher (first in sorted list).
        assert scored[0].parameters["x1"] < 7.0

    def test_objective_dominates_late(self):
        """Clean snapshot (alpha_min): higher obj scores should rank first."""
        snap = _make_clean_snapshot()
        clf = FeasibilityClassifier(k=5)
        scorer = FeasibilityFirstScorer(alpha_max=0.9, alpha_min=0.1)

        candidates = [
            {"x1": 3.0, "x2": 5.0},
            {"x1": 5.0, "x2": 5.0},
        ]
        # Both should have high feasibility; obj values differ.
        scored = scorer.score_candidates(
            candidates, snap, clf,
            objective_values=[1.0, 10.0],
            direction="maximize",
        )

        # Candidate with higher objective (10.0) should rank first.
        assert scored[0].parameters["x1"] == 5.0

    def test_combined_score_in_range(self):
        """All combined scores should be in [0, 1]."""
        snap = _make_snapshot_with_death_zone()
        clf = FeasibilityClassifier(k=5)
        scorer = FeasibilityFirstScorer()

        candidates = [
            {"x1": 1.0, "x2": 5.0},
            {"x1": 5.0, "x2": 5.0},
            {"x1": 9.0, "x2": 5.0},
        ]
        scored = scorer.score_candidates(
            candidates, snap, clf,
            objective_values=[1.0, 5.0, 10.0],
            direction="maximize",
        )

        for s in scored:
            assert 0.0 <= s.combined_score <= 1.0 + 1e-9

    def test_normalize_maximize(self):
        """[1, 5, 10] maximize should yield [0.0, ~0.44, 1.0]."""
        result = FeasibilityFirstScorer._normalize_objectives(
            [1.0, 5.0, 10.0], "maximize"
        )
        assert abs(result[0] - 0.0) < 1e-6
        assert abs(result[1] - 4.0 / 9.0) < 0.01
        assert abs(result[2] - 1.0) < 1e-6

    def test_normalize_minimize(self):
        """[1, 5, 10] minimize should yield [1.0, ~0.56, 0.0]."""
        result = FeasibilityFirstScorer._normalize_objectives(
            [1.0, 5.0, 10.0], "minimize"
        )
        assert abs(result[0] - 1.0) < 1e-6
        assert abs(result[1] - 5.0 / 9.0) < 0.01
        assert abs(result[2] - 0.0) < 1e-6

    def test_no_objective_values(self):
        """Without obj values, scoring should use feasibility only."""
        snap = _make_snapshot_with_death_zone()
        clf = FeasibilityClassifier(k=5)
        scorer = FeasibilityFirstScorer()

        candidates = [
            {"x1": 3.0, "x2": 5.0},
            {"x1": 9.0, "x2": 5.0},
        ]
        scored = scorer.score_candidates(
            candidates, snap, clf,
            objective_values=None,
            direction="maximize",
        )

        assert len(scored) == 2
        # All objective_score should be 0.0 when no values given.
        for s in scored:
            assert abs(s.objective_score - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# TestSafetyBoundaryLearner
# ---------------------------------------------------------------------------


class TestSafetyBoundaryLearner:
    """Tests for quantile-based safety boundary learning."""

    def test_bounds_tighter_than_specs(self):
        """Learned bounds should be inside (or equal to) spec bounds."""
        snap = _make_snapshot_with_death_zone()
        learner = SafetyBoundaryLearner(
            lower_quantile=0.05, upper_quantile=0.95,
            min_observations=5, margin_fraction=0.05,
        )
        boundary = learner.learn(snap)

        for name, (lo, hi) in boundary.parameter_bounds.items():
            spec = next(s for s in snap.parameter_specs if s.name == name)
            assert lo >= spec.lower
            assert hi <= spec.upper

    def test_bounds_from_successful_only(self):
        """With mix of safe/failed observations, bounds should reflect safe obs only."""
        snap = _make_snapshot_with_death_zone()
        learner = SafetyBoundaryLearner(
            lower_quantile=0.05, upper_quantile=0.95,
            min_observations=5, margin_fraction=0.0,
        )
        boundary = learner.learn(snap)

        # Successful observations have x1 in [0, ~7) plus the ~20% surviving danger zone.
        # The safe upper bound for x1 should be < 10.0 (spec upper).
        lo_x1, hi_x1 = boundary.parameter_bounds["x1"]
        assert hi_x1 < 10.0
        assert boundary.n_successful > 0

    def test_quantile_computation(self):
        """_quantile([1,2,3,4,5], 0.5) should be 3.0."""
        result = SafetyBoundaryLearner._quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5)
        assert abs(result - 3.0) < 1e-9

    def test_quantile_interpolation(self):
        """_quantile([1,2,3,4], 0.5) should interpolate to 2.5."""
        result = SafetyBoundaryLearner._quantile([1.0, 2.0, 3.0, 4.0], 0.5)
        assert abs(result - 2.5) < 1e-9

    def test_margin_expands_bounds(self):
        """margin_fraction > 0 should produce slightly wider bounds than margin=0."""
        snap = _make_clean_snapshot(n_obs=20)
        learner_no_margin = SafetyBoundaryLearner(
            lower_quantile=0.1, upper_quantile=0.9,
            min_observations=5, margin_fraction=0.0,
        )
        learner_with_margin = SafetyBoundaryLearner(
            lower_quantile=0.1, upper_quantile=0.9,
            min_observations=5, margin_fraction=0.1,
        )
        b_no = learner_no_margin.learn(snap)
        b_with = learner_with_margin.learn(snap)

        lo_no, hi_no = b_no.parameter_bounds["x1"]
        lo_with, hi_with = b_with.parameter_bounds["x1"]

        # Margin should widen the range (or at least not shrink it).
        range_no = hi_no - lo_no
        range_with = hi_with - lo_with
        assert range_with >= range_no - 1e-9

    def test_clipping_to_spec_bounds(self):
        """Bounds should never exceed spec.lower / spec.upper."""
        snap = _make_clean_snapshot(n_obs=20)
        learner = SafetyBoundaryLearner(
            lower_quantile=0.0, upper_quantile=1.0,
            min_observations=5, margin_fraction=0.5,  # large margin to push past bounds
        )
        boundary = learner.learn(snap)

        for name, (lo, hi) in boundary.parameter_bounds.items():
            spec = next(s for s in snap.parameter_specs if s.name == name)
            assert lo >= spec.lower - 1e-9
            assert hi <= spec.upper + 1e-9

    def test_is_within_bounds(self):
        """Point inside safe bounds should return True."""
        snap = _make_clean_snapshot(n_obs=20)
        learner = SafetyBoundaryLearner(min_observations=5)
        boundary = learner.learn(snap)

        # Middle of the range should be within bounds.
        candidate = {"x1": 5.0, "x2": 5.0}
        assert learner.is_within_bounds(candidate, boundary) is True

    def test_outside_bounds(self):
        """Point outside safe bounds should return False."""
        # Create a narrow boundary by using tight quantiles and no margin.
        snap = _make_clean_snapshot(n_obs=20)
        learner = SafetyBoundaryLearner(
            lower_quantile=0.2, upper_quantile=0.8,
            min_observations=5, margin_fraction=0.0,
        )
        boundary = learner.learn(snap)

        # x1 goes from 0..~9.5 in clean snapshot; quantile 0.2/0.8 should
        # tighten bounds well inside [0, 10]. Try a point well outside.
        lo_x1, hi_x1 = boundary.parameter_bounds["x1"]
        outside_candidate = {"x1": hi_x1 + 1.0, "x2": 5.0}
        assert learner.is_within_bounds(outside_candidate, boundary) is False

    def test_clamp_to_bounds(self):
        """Values should be clamped into the safe range."""
        snap = _make_clean_snapshot(n_obs=20)
        learner = SafetyBoundaryLearner(min_observations=5, margin_fraction=0.0)
        boundary = learner.learn(snap)

        lo_x1, hi_x1 = boundary.parameter_bounds["x1"]
        candidate = {"x1": -100.0, "x2": 100.0}
        clamped = learner.clamp_to_bounds(candidate, boundary)

        assert clamped["x1"] >= lo_x1 - 1e-9
        lo_x2, hi_x2 = boundary.parameter_bounds["x2"]
        assert clamped["x2"] <= hi_x2 + 1e-9

    def test_few_observations_no_tightening(self):
        """< min_observations should return bounds = spec bounds, ratios = 1.0."""
        specs = _make_specs()
        obs = [
            Observation(
                iteration=0, parameters={"x1": 5.0, "x2": 5.0},
                kpi_values={"y": 10.0}, is_failure=False, timestamp=0.0,
            ),
        ]
        snap = CampaignSnapshot(
            campaign_id="tiny-boundary", parameter_specs=specs, observations=obs,
            objective_names=["y"], objective_directions=["maximize"],
            current_iteration=1,
        )
        learner = SafetyBoundaryLearner(min_observations=5)
        boundary = learner.learn(snap)

        # Bounds should equal full spec range.
        assert boundary.parameter_bounds["x1"] == (0.0, 10.0)
        assert boundary.parameter_bounds["x2"] == (0.0, 10.0)
        # Tightening ratios should be 1.0 (no tightening).
        for ratio in boundary.tightening_ratios.values():
            assert abs(ratio - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining classifier, scorer, and boundary learner."""

    def test_classifier_and_scorer_pipeline(self):
        """Full pipeline: classify -> score -> rank."""
        snap = _make_snapshot_with_death_zone()
        clf = FeasibilityClassifier(k=5)
        scorer = FeasibilityFirstScorer(alpha_max=0.9, alpha_min=0.1)

        candidates = [
            {"x1": 2.0, "x2": 5.0},
            {"x1": 5.0, "x2": 5.0},
            {"x1": 9.0, "x2": 5.0},
        ]
        scored = scorer.score_candidates(
            candidates, snap, clf,
            objective_values=[8.0, 12.0, 15.0],
            direction="maximize",
        )

        assert len(scored) == 3
        # Results should be sorted by combined_score descending.
        for i in range(len(scored) - 1):
            assert scored[i].combined_score >= scored[i + 1].combined_score - 1e-9
        # Each scored candidate should have required fields.
        for s in scored:
            assert s.phase in ("feasibility_first", "objective_first")
            assert 0.0 <= s.alpha <= 1.0

    def test_safety_boundary_with_scoring(self):
        """Learn boundary, clamp candidates, then score."""
        snap = _make_snapshot_with_death_zone()
        clf = FeasibilityClassifier(k=5)
        scorer = FeasibilityFirstScorer()
        boundary_learner = SafetyBoundaryLearner(min_observations=5)

        boundary = boundary_learner.learn(snap)

        # Create a dangerous candidate and clamp it.
        dangerous = {"x1": 12.0, "x2": -5.0}
        clamped = boundary_learner.clamp_to_bounds(dangerous, boundary)

        # Clamped should be within bounds.
        assert boundary_learner.is_within_bounds(clamped, boundary)

        # Score the clamped candidate.
        scored = scorer.score_candidates(
            [clamped], snap, clf,
            objective_values=[10.0],
            direction="maximize",
        )
        assert len(scored) == 1
        assert 0.0 <= scored[0].combined_score <= 1.0 + 1e-9

    def test_deterministic(self):
        """Same input twice should produce identical output."""
        snap = _make_snapshot_with_death_zone()
        clf = FeasibilityClassifier(k=5)
        scorer = FeasibilityFirstScorer(alpha_max=0.9, alpha_min=0.1)
        boundary_learner = SafetyBoundaryLearner(min_observations=5)

        candidates = [{"x1": 3.0, "x2": 5.0}, {"x1": 8.0, "x2": 5.0}]

        # Run 1
        preds_1 = clf.predict_batch(candidates, snap)
        scored_1 = scorer.score_candidates(
            candidates, snap, clf,
            objective_values=[10.0, 5.0], direction="maximize",
        )
        boundary_1 = boundary_learner.learn(snap)

        # Run 2
        preds_2 = clf.predict_batch(candidates, snap)
        scored_2 = scorer.score_candidates(
            candidates, snap, clf,
            objective_values=[10.0, 5.0], direction="maximize",
        )
        boundary_2 = boundary_learner.learn(snap)

        # Predictions should be identical.
        for p1, p2 in zip(preds_1, preds_2):
            assert abs(p1.p_feasible - p2.p_feasible) < 1e-9
            assert abs(p1.p_failure - p2.p_failure) < 1e-9

        # Scores should be identical.
        for s1, s2 in zip(scored_1, scored_2):
            assert abs(s1.combined_score - s2.combined_score) < 1e-9

        # Boundaries should be identical.
        for name in boundary_1.parameter_bounds:
            lo1, hi1 = boundary_1.parameter_bounds[name]
            lo2, hi2 = boundary_2.parameter_bounds[name]
            assert abs(lo1 - lo2) < 1e-9
            assert abs(hi1 - hi2) < 1e-9

    def test_empty_snapshot_all_modules(self):
        """All 3 classes should handle empty snapshot gracefully."""
        specs = _make_specs()
        snap = CampaignSnapshot(
            campaign_id="empty-all", parameter_specs=specs, observations=[],
            objective_names=["y"], objective_directions=["maximize"],
            current_iteration=0,
        )

        # Classifier: should not crash.
        clf = FeasibilityClassifier(k=5)
        pred = clf.predict({"x1": 5.0, "x2": 5.0}, snap)
        assert isinstance(pred, FeasibilityPrediction)

        # Scorer: should not crash.
        scorer = FeasibilityFirstScorer()
        scored = scorer.score_candidates(
            [{"x1": 5.0, "x2": 5.0}], snap, clf,
            objective_values=[10.0], direction="maximize",
        )
        assert len(scored) == 1

        # Boundary learner: should return spec-wide bounds.
        learner = SafetyBoundaryLearner(min_observations=5)
        boundary = learner.learn(snap)
        assert boundary.n_successful == 0
        assert boundary.parameter_bounds["x1"] == (0.0, 10.0)
        assert boundary.parameter_bounds["x2"] == (0.0, 10.0)


# ---------------------------------------------------------------------------
# TestParetoProximityBoost
# ---------------------------------------------------------------------------


class TestParetoProximityBoost:
    """Tests for Pareto-proximity lambda boost in FeasibilityFirstScorer."""

    def test_no_boost_by_default(self):
        """Default pareto_proximity_boost=0 should not change alpha."""
        snap = _make_clean_snapshot()
        scorer = FeasibilityFirstScorer(alpha_max=0.9, alpha_min=0.1)
        alpha_no_prox = scorer.compute_alpha(snap)
        alpha_with_prox = scorer.compute_alpha(snap, pareto_proximity=1.0)
        assert abs(alpha_no_prox - alpha_with_prox) < 1e-9

    def test_boost_increases_alpha_near_front(self):
        """With boost enabled, proximity=1.0 should increase alpha."""
        snap = _make_clean_snapshot()
        scorer = FeasibilityFirstScorer(
            alpha_max=0.9, alpha_min=0.1, pareto_proximity_boost=0.3
        )
        alpha_base = scorer.compute_alpha(snap, pareto_proximity=0.0)
        alpha_boosted = scorer.compute_alpha(snap, pareto_proximity=1.0)
        assert alpha_boosted > alpha_base

    def test_boost_does_not_exceed_alpha_max(self):
        """Boosted alpha must not exceed alpha_max."""
        snap = _make_all_failure_snapshot()  # alpha already near max
        scorer = FeasibilityFirstScorer(
            alpha_max=0.9, alpha_min=0.1, pareto_proximity_boost=0.5
        )
        alpha = scorer.compute_alpha(snap, pareto_proximity=1.0)
        assert alpha <= 0.9 + 1e-9

    def test_boost_proportional_to_proximity(self):
        """Higher proximity should give higher alpha (monotonic)."""
        snap = _make_clean_snapshot()
        scorer = FeasibilityFirstScorer(
            alpha_max=0.9, alpha_min=0.1, pareto_proximity_boost=0.3
        )
        alpha_low = scorer.compute_alpha(snap, pareto_proximity=0.2)
        alpha_mid = scorer.compute_alpha(snap, pareto_proximity=0.5)
        alpha_high = scorer.compute_alpha(snap, pareto_proximity=0.8)
        assert alpha_low <= alpha_mid <= alpha_high
