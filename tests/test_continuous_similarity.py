"""Tests for continuous fingerprint similarity system.

Covers the continuous vector encoding in ProblemFingerprint, the RBF kernel
similarity in ExperienceStore, similarity-based transfer learning in
StrategyLearner, and similarity fallback in WeightTuner and ThresholdLearner.
"""

from __future__ import annotations

import math

import pytest

from optimization_copilot.core.models import (
    ProblemFingerprint,
    VariableType,
    ObjectiveForm,
    NoiseRegime,
    CostProfile,
    FailureInformativeness,
    DataScale,
    Dynamics,
    FeasibleRegion,
)
from optimization_copilot.meta_learning.models import (
    BackendPerformance,
    CampaignOutcome,
    MetaLearningConfig,
)
from optimization_copilot.meta_learning.experience_store import ExperienceStore
from optimization_copilot.meta_learning.strategy_learner import StrategyLearner
from optimization_copilot.meta_learning.weight_tuner import WeightTuner
from optimization_copilot.meta_learning.threshold_learner import ThresholdLearner
from optimization_copilot.portfolio.scorer import ScoringWeights


# ── Helpers ────────────────────────────────────────────────────


def _make_outcome(
    campaign_id: str = "camp1",
    fingerprint: ProblemFingerprint | None = None,
    backends: list[dict] | None = None,
    phase_transitions: list[tuple[str, str, int]] | None = None,
    failure_type_counts: dict[str, int] | None = None,
    stabilization_used: dict[str, str] | None = None,
    total_iterations: int = 50,
    best_kpi: float = 0.85,
    timestamp: float = 1.0,
) -> CampaignOutcome:
    """Build a CampaignOutcome with reasonable defaults."""
    if fingerprint is None:
        fingerprint = ProblemFingerprint()
    if backends is None:
        backends = [
            {
                "backend_name": "tpe",
                "convergence_iteration": 30,
                "final_best_kpi": 0.85,
                "regret": 0.15,
                "sample_efficiency": 0.017,
                "failure_rate": 0.1,
            },
            {
                "backend_name": "random",
                "convergence_iteration": None,
                "final_best_kpi": 0.6,
                "regret": 0.4,
                "sample_efficiency": 0.008,
                "failure_rate": 0.2,
            },
        ]
    if phase_transitions is None:
        phase_transitions = [
            ("cold_start", "learning", 10),
            ("learning", "exploitation", 25),
        ]
    if failure_type_counts is None:
        failure_type_counts = {"hardware": 2, "chemistry": 1}
    if stabilization_used is None:
        stabilization_used = {"tpe": "noise_smoothing_window=5"}

    backend_perfs = [BackendPerformance(**b) for b in backends]

    return CampaignOutcome(
        campaign_id=campaign_id,
        fingerprint=fingerprint,
        phase_transitions=phase_transitions,
        backend_performances=backend_perfs,
        failure_type_counts=failure_type_counts,
        stabilization_used=stabilization_used,
        total_iterations=total_iterations,
        best_kpi=best_kpi,
        timestamp=timestamp,
    )


# ====================================================================
# 1. TestContinuousVector
# ====================================================================


class TestContinuousVector:
    """Tests for ProblemFingerprint.to_continuous_vector()."""

    def test_default_fingerprint_vector(self):
        """Default ProblemFingerprint gives all-zero vector.

        Defaults: continuous=0, single=0, low=0, uniform=0, weak=0,
        tiny=0, static=0, wide=0, effective_dimensionality=-1 -> 0.0.
        """
        fp = ProblemFingerprint()
        vec = fp.to_continuous_vector()
        assert vec == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_max_fingerprint_vector(self):
        """Fingerprint with all maximum ordinal values gives all-1.0 enum dims.

        categorical=1.0, constrained=1.0, high=1.0, heterogeneous=1.0,
        strong=1.0, moderate=1.0, time_series=1.0, fragmented=1.0.
        """
        fp = ProblemFingerprint(
            variable_types=VariableType.CATEGORICAL,
            objective_form=ObjectiveForm.CONSTRAINED,
            noise_regime=NoiseRegime.HIGH,
            cost_profile=CostProfile.HETEROGENEOUS,
            failure_informativeness=FailureInformativeness.STRONG,
            data_scale=DataScale.MODERATE,
            dynamics=Dynamics.TIME_SERIES,
            feasible_region=FeasibleRegion.FRAGMENTED,
            effective_dimensionality=100,
        )
        vec = fp.to_continuous_vector()
        # First 8 dimensions should all be 1.0
        for i in range(8):
            assert vec[i] == 1.0, f"Dimension {i} should be 1.0, got {vec[i]}"
        # Dimensionality: 100 / (100 + 20) = 0.8333...
        assert abs(vec[8] - 100.0 / 120.0) < 1e-9

    def test_intermediate_values(self):
        """Mixed/multi_objective/medium/small/narrow give ~0.5 range values."""
        fp = ProblemFingerprint(
            variable_types=VariableType.MIXED,           # 0.67
            objective_form=ObjectiveForm.MULTI_OBJECTIVE, # 0.5
            noise_regime=NoiseRegime.MEDIUM,              # 0.5
            cost_profile=CostProfile.UNIFORM,             # 0.0 (binary)
            failure_informativeness=FailureInformativeness.WEAK,  # 0.0 (binary)
            data_scale=DataScale.SMALL,                   # 0.5
            dynamics=Dynamics.STATIC,                     # 0.0 (binary)
            feasible_region=FeasibleRegion.NARROW,        # 0.5
        )
        vec = fp.to_continuous_vector()
        assert abs(vec[0] - 0.67) < 1e-9   # mixed
        assert abs(vec[1] - 0.5) < 1e-9    # multi_objective
        assert abs(vec[2] - 0.5) < 1e-9    # medium noise
        assert abs(vec[5] - 0.5) < 1e-9    # small data_scale
        assert abs(vec[7] - 0.5) < 1e-9    # narrow feasible_region

    def test_dimensionality_encoding(self):
        """effective_dimensionality normalization: dim/(dim+20) when dim>0, else 0."""
        test_cases = [
            (-1, 0.0),     # negative -> 0.0
            (0, 0.0),      # zero -> 0.0
            (20, 0.5),     # midpoint: 20/(20+20) = 0.5
            (100, 100.0 / 120.0),  # 100/(100+20) ~ 0.833
        ]
        for dim, expected in test_cases:
            fp = ProblemFingerprint(effective_dimensionality=dim)
            vec = fp.to_continuous_vector()
            assert abs(vec[8] - expected) < 1e-9, (
                f"dim={dim}: expected {expected}, got {vec[8]}"
            )

    def test_vector_length(self):
        """to_continuous_vector() always returns exactly 9 elements."""
        fingerprints = [
            ProblemFingerprint(),
            ProblemFingerprint(
                variable_types=VariableType.CATEGORICAL,
                effective_dimensionality=50,
            ),
            ProblemFingerprint(
                noise_regime=NoiseRegime.HIGH,
                data_scale=DataScale.MODERATE,
            ),
        ]
        for fp in fingerprints:
            vec = fp.to_continuous_vector()
            assert len(vec) == 9

    def test_ordinal_ordering(self):
        """DataScale ordinal ordering: TINY < SMALL < MODERATE."""
        fp_tiny = ProblemFingerprint(data_scale=DataScale.TINY)
        fp_small = ProblemFingerprint(data_scale=DataScale.SMALL)
        fp_moderate = ProblemFingerprint(data_scale=DataScale.MODERATE)

        # data_scale is dimension index 5
        assert fp_tiny.to_continuous_vector()[5] < fp_small.to_continuous_vector()[5]
        assert fp_small.to_continuous_vector()[5] < fp_moderate.to_continuous_vector()[5]


# ====================================================================
# 2. TestRBFSimilarity
# ====================================================================


class TestRBFSimilarity:
    """Tests for the RBF kernel similarity in ExperienceStore."""

    def _sim(
        self, fp1: ProblemFingerprint, fp2: ProblemFingerprint
    ) -> float:
        """Compute similarity using a temporary ExperienceStore."""
        store = ExperienceStore()
        return store._fingerprint_similarity(fp1, fp2)

    def test_identical_fingerprints(self):
        """Identical fingerprints have similarity = 1.0."""
        fp = ProblemFingerprint(
            variable_types=VariableType.MIXED,
            noise_regime=NoiseRegime.MEDIUM,
        )
        assert self._sim(fp, fp) == 1.0

    def test_completely_different(self):
        """Maximally different fingerprints have low similarity (<0.2)."""
        fp_min = ProblemFingerprint(
            variable_types=VariableType.CONTINUOUS,
            objective_form=ObjectiveForm.SINGLE,
            noise_regime=NoiseRegime.LOW,
            cost_profile=CostProfile.UNIFORM,
            failure_informativeness=FailureInformativeness.WEAK,
            data_scale=DataScale.TINY,
            dynamics=Dynamics.STATIC,
            feasible_region=FeasibleRegion.WIDE,
            effective_dimensionality=-1,
        )
        fp_max = ProblemFingerprint(
            variable_types=VariableType.CATEGORICAL,
            objective_form=ObjectiveForm.CONSTRAINED,
            noise_regime=NoiseRegime.HIGH,
            cost_profile=CostProfile.HETEROGENEOUS,
            failure_informativeness=FailureInformativeness.STRONG,
            data_scale=DataScale.MODERATE,
            dynamics=Dynamics.TIME_SERIES,
            feasible_region=FeasibleRegion.FRAGMENTED,
            effective_dimensionality=100,
        )
        sim = self._sim(fp_min, fp_max)
        # sq_dist = 1^2 + 1^2 + 1^2 + 1^2 + 1^2 + 1^2 + 1^2 + 1^2 + 0.833^2
        # = 8 + 0.694 = 8.694
        # sim = exp(-8.694/2) = exp(-4.347) ~ 0.013
        assert sim < 0.2, f"Expected < 0.2, got {sim}"

    def test_one_dimension_different(self):
        """Changing only noise_regime LOW->HIGH (diff=1.0): sim = exp(-0.5)."""
        fp_base = ProblemFingerprint()
        fp_noisy = ProblemFingerprint(noise_regime=NoiseRegime.HIGH)
        sim = self._sim(fp_base, fp_noisy)
        # sq_dist = 1.0^2 = 1.0; sim = exp(-1.0/2.0) = exp(-0.5)
        expected = math.exp(-0.5)
        assert abs(sim - expected) < 1e-9

    def test_similar_fingerprints_high_similarity(self):
        """Changing only noise LOW->MEDIUM (diff=0.5): sim = exp(-0.125)."""
        fp_base = ProblemFingerprint()
        fp_medium = ProblemFingerprint(noise_regime=NoiseRegime.MEDIUM)
        sim = self._sim(fp_base, fp_medium)
        # sq_dist = 0.5^2 = 0.25; sim = exp(-0.25/2.0) = exp(-0.125)
        expected = math.exp(-0.125)
        assert abs(sim - expected) < 1e-9

    def test_monotonic_distance(self):
        """More dimensions changed leads to strictly lower similarity."""
        fp_base = ProblemFingerprint()

        # 1 dim changed
        fp_1 = ProblemFingerprint(noise_regime=NoiseRegime.HIGH)
        # 2 dims changed
        fp_2 = ProblemFingerprint(
            noise_regime=NoiseRegime.HIGH,
            cost_profile=CostProfile.HETEROGENEOUS,
        )
        # 3 dims changed
        fp_3 = ProblemFingerprint(
            noise_regime=NoiseRegime.HIGH,
            cost_profile=CostProfile.HETEROGENEOUS,
            dynamics=Dynamics.TIME_SERIES,
        )

        sim_1 = self._sim(fp_base, fp_1)
        sim_2 = self._sim(fp_base, fp_2)
        sim_3 = self._sim(fp_base, fp_3)

        assert sim_1 > sim_2 > sim_3
        # Also check all are in valid range
        assert 0.0 < sim_3 < sim_2 < sim_1 < 1.0

    def test_smooth_gradient(self):
        """LOW->MEDIUM->HIGH gives monotonically decreasing similarity (no jumps)."""
        fp_base = ProblemFingerprint()
        fp_medium = ProblemFingerprint(noise_regime=NoiseRegime.MEDIUM)
        fp_high = ProblemFingerprint(noise_regime=NoiseRegime.HIGH)

        sim_med = self._sim(fp_base, fp_medium)
        sim_high = self._sim(fp_base, fp_high)

        assert sim_med > sim_high, (
            f"MEDIUM sim ({sim_med}) should be > HIGH sim ({sim_high})"
        )
        # The gap between 1.0->sim_med should be smaller than 1.0->sim_high
        # (smooth, not a cliff from MEDIUM to HIGH)
        drop_to_med = 1.0 - sim_med
        drop_to_high = 1.0 - sim_high
        assert drop_to_high > drop_to_med

    def test_symmetry(self):
        """sim(a, b) == sim(b, a) for any pair of fingerprints."""
        fp_a = ProblemFingerprint(
            variable_types=VariableType.MIXED,
            noise_regime=NoiseRegime.MEDIUM,
            data_scale=DataScale.SMALL,
        )
        fp_b = ProblemFingerprint(
            variable_types=VariableType.CATEGORICAL,
            noise_regime=NoiseRegime.HIGH,
            data_scale=DataScale.MODERATE,
            effective_dimensionality=50,
        )
        assert self._sim(fp_a, fp_b) == self._sim(fp_b, fp_a)


# ====================================================================
# 3. TestSimilarityTransferLearning
# ====================================================================


class TestSimilarityTransferLearning:
    """Tests that continuous similarity improves ExperienceStore.get_similar()
    and downstream StrategyLearner.
    """

    def test_similar_problems_ranked_by_similarity(self):
        """Store 3 fingerprints with decreasing similarity to query;
        get_similar returns them in correct order."""
        store = ExperienceStore()

        # Query: default fingerprint (all zeros vector)
        query = ProblemFingerprint()

        # fp_close: 1 dim differs by 0.5
        fp_close = ProblemFingerprint(noise_regime=NoiseRegime.MEDIUM)
        # fp_mid: 1 dim differs by 1.0
        fp_mid = ProblemFingerprint(noise_regime=NoiseRegime.HIGH)
        # fp_far: 3 dims differ by 1.0 each
        fp_far = ProblemFingerprint(
            noise_regime=NoiseRegime.HIGH,
            cost_profile=CostProfile.HETEROGENEOUS,
            dynamics=Dynamics.TIME_SERIES,
        )

        store.record_outcome(_make_outcome(campaign_id="close", fingerprint=fp_close))
        store.record_outcome(_make_outcome(campaign_id="mid", fingerprint=fp_mid))
        store.record_outcome(_make_outcome(campaign_id="far", fingerprint=fp_far))

        results = store.get_similar(query, max_results=10)
        assert len(results) == 3

        # Should be sorted by similarity descending
        ids = [r.outcome.campaign_id for r, _ in results]
        assert ids == ["close", "mid", "far"]

        # Verify similarities are monotonically decreasing
        sims = [s for _, s in results]
        assert sims[0] > sims[1] > sims[2]

    def test_partial_match_still_retrieved(self):
        """Fingerprint differing in 2 dims still has reasonable similarity (>0.3)."""
        store = ExperienceStore()

        fp_stored = ProblemFingerprint(
            noise_regime=NoiseRegime.HIGH,          # differs from default
            feasible_region=FeasibleRegion.NARROW,   # differs from default
        )
        store.record_outcome(_make_outcome(fingerprint=fp_stored))

        query = ProblemFingerprint()  # all defaults
        results = store.get_similar(query, max_results=5)
        assert len(results) == 1
        _, sim = results[0]
        # sq_dist = 1.0^2 + 0.5^2 = 1.25; sim = exp(-0.625) ~ 0.535
        assert sim > 0.3, f"Expected > 0.3, got {sim}"

    def test_strategy_learner_uses_continuous(self):
        """StrategyLearner with only 'similar' (not exact) fingerprint data
        should still produce rankings via similarity-based matching."""
        cfg = MetaLearningConfig(min_experiences_for_learning=2)
        store = ExperienceStore(config=cfg)

        # Store outcomes with a SIMILAR fingerprint (not exact match to query)
        fp_stored = ProblemFingerprint(noise_regime=NoiseRegime.MEDIUM)
        for i in range(3):
            store.record_outcome(
                _make_outcome(
                    campaign_id=f"c{i}",
                    fingerprint=fp_stored,
                    timestamp=float(i),
                )
            )

        learner = StrategyLearner(store, config=cfg)
        # Query with default fingerprint (different from stored)
        ranked = learner.rank_backends(ProblemFingerprint())
        # Should still produce results because get_similar finds them
        assert len(ranked) > 0
        names = [name for name, _ in ranked]
        assert "tpe" in names

    def test_old_discrete_would_fail(self):
        """Scenario where old discrete approach gives same score for two
        different fingerprints, but continuous correctly discriminates.

        Old discrete: count matching fields / 8.
        FP_A differs from query only in noise (LOW->MEDIUM): discrete = 7/8
        FP_B differs from query only in noise (LOW->HIGH): discrete = 7/8
        Both get 0.875 in discrete -- indistinguishable.

        Continuous:
        FP_A: sq_dist = 0.5^2 = 0.25 -> sim = exp(-0.125) ~ 0.882
        FP_B: sq_dist = 1.0^2 = 1.0  -> sim = exp(-0.5)   ~ 0.607
        Clearly discriminated.
        """
        store = ExperienceStore()

        query = ProblemFingerprint()
        fp_a = ProblemFingerprint(noise_regime=NoiseRegime.MEDIUM)
        fp_b = ProblemFingerprint(noise_regime=NoiseRegime.HIGH)

        store.record_outcome(_make_outcome(campaign_id="a", fingerprint=fp_a))
        store.record_outcome(_make_outcome(campaign_id="b", fingerprint=fp_b))

        results = store.get_similar(query, max_results=10)
        sims = {r.outcome.campaign_id: s for r, s in results}

        # Continuous approach distinguishes them
        assert sims["a"] > sims["b"], (
            f"Expected fp_a (MEDIUM noise, sim={sims['a']:.4f}) > "
            f"fp_b (HIGH noise, sim={sims['b']:.4f})"
        )
        # The gap should be significant (not just floating point noise)
        gap = sims["a"] - sims["b"]
        assert gap > 0.2, f"Gap should be significant, got {gap:.4f}"

    def test_old_discrete_multi_field_discrimination(self):
        """Additional discrimination test: two fingerprints differ in 1 field
        each from the query, but the ordinal distance differs.

        FP_C: data_scale TINY->SMALL (ordinal diff = 0.5)
        FP_D: data_scale TINY->MODERATE (ordinal diff = 1.0)

        Old discrete: both 7/8 = 0.875
        Continuous: FP_C sim > FP_D sim
        """
        store = ExperienceStore()

        query = ProblemFingerprint()
        fp_c = ProblemFingerprint(data_scale=DataScale.SMALL)
        fp_d = ProblemFingerprint(data_scale=DataScale.MODERATE)

        store.record_outcome(_make_outcome(campaign_id="c", fingerprint=fp_c))
        store.record_outcome(_make_outcome(campaign_id="d", fingerprint=fp_d))

        results = store.get_similar(query, max_results=10)
        sims = {r.outcome.campaign_id: s for r, s in results}

        assert sims["c"] > sims["d"]


# ====================================================================
# 4. TestWeightTunerSimilarityFallback
# ====================================================================


class TestWeightTunerSimilarityFallback:
    """Tests for similarity-based fallback in WeightTuner.suggest_weights()."""

    def _make_tuner_with_data(
        self,
        stored_fp: ProblemFingerprint,
        n_campaigns: int = 5,
        min_exp: int = 3,
    ) -> tuple[WeightTuner, ExperienceStore]:
        """Create a WeightTuner with learned data for the given fingerprint."""
        cfg = MetaLearningConfig(min_experiences_for_learning=min_exp)
        store = ExperienceStore(config=cfg)
        tuner = WeightTuner(store, config=cfg)

        for i in range(n_campaigns):
            outcome = _make_outcome(
                campaign_id=f"c{i}",
                fingerprint=stored_fp,
                timestamp=float(i),
            )
            store.record_outcome(outcome)
            tuner.update_from_outcome(outcome, ScoringWeights())

        return tuner, store

    def test_exact_match_still_preferred(self):
        """With exact match available, returns exact match weights."""
        fp = ProblemFingerprint(noise_regime=NoiseRegime.MEDIUM)
        tuner, store = self._make_tuner_with_data(fp, n_campaigns=5)

        result = tuner.suggest_weights(fp)
        assert result is not None
        assert result.n_campaigns == 5

    def test_fallback_to_similar(self):
        """No exact match but similar fingerprint exists -> returns learned weights."""
        # Store data for MEDIUM noise fingerprint
        stored_fp = ProblemFingerprint(noise_regime=NoiseRegime.MEDIUM)
        tuner, store = self._make_tuner_with_data(stored_fp, n_campaigns=5)

        # Query with LOW noise (default) - similar but not exact
        query_fp = ProblemFingerprint()  # noise_regime=LOW
        # Similarity: sq_dist = 0.5^2 = 0.25 -> sim = exp(-0.125) ~ 0.882 > 0.5
        result = tuner.suggest_weights(query_fp)
        assert result is not None, "Should fall back to similar fingerprint"
        assert result.n_campaigns == 5

    def test_no_fallback_if_too_different(self):
        """Very different fingerprint (sim < 0.5) returns None."""
        # Store data for a maximally different fingerprint
        stored_fp = ProblemFingerprint(
            variable_types=VariableType.CATEGORICAL,
            objective_form=ObjectiveForm.CONSTRAINED,
            noise_regime=NoiseRegime.HIGH,
            cost_profile=CostProfile.HETEROGENEOUS,
            failure_informativeness=FailureInformativeness.STRONG,
            data_scale=DataScale.MODERATE,
            dynamics=Dynamics.TIME_SERIES,
            feasible_region=FeasibleRegion.FRAGMENTED,
        )
        tuner, store = self._make_tuner_with_data(stored_fp, n_campaigns=5)

        # Query with default (all zeros) -- very far away
        query_fp = ProblemFingerprint()
        # sq_dist = 1+1+1+1+1+1+1+1 = 8 -> sim = exp(-4) ~ 0.018 < 0.5
        result = tuner.suggest_weights(query_fp)
        assert result is None, "Should not fall back to very different fingerprint"

    def test_cold_start_guard(self):
        """Similar fingerprint exists but n_campaigns < min_experiences -> None."""
        stored_fp = ProblemFingerprint(noise_regime=NoiseRegime.MEDIUM)
        # Only 2 campaigns, but min_exp=3
        tuner, store = self._make_tuner_with_data(
            stored_fp, n_campaigns=2, min_exp=3
        )

        query_fp = ProblemFingerprint()
        result = tuner.suggest_weights(query_fp)
        assert result is None, (
            "Should not return weights when n_campaigns < min_experiences"
        )


# ====================================================================
# 5. TestThresholdLearnerSimilarityFallback
# ====================================================================


class TestThresholdLearnerSimilarityFallback:
    """Tests for similarity-based fallback in ThresholdLearner.suggest_thresholds()."""

    def _make_learner_with_data(
        self,
        stored_fp: ProblemFingerprint,
        n_campaigns: int = 5,
        min_exp: int = 3,
    ) -> tuple[ThresholdLearner, ExperienceStore]:
        """Create a ThresholdLearner with learned data for the given fingerprint."""
        cfg = MetaLearningConfig(min_experiences_for_learning=min_exp)
        store = ExperienceStore(config=cfg)
        learner = ThresholdLearner(store, config=cfg)

        for i in range(n_campaigns):
            outcome = _make_outcome(
                campaign_id=f"c{i}",
                fingerprint=stored_fp,
                phase_transitions=[
                    ("cold_start", "learning", 10),
                    ("learning", "exploitation", 25),
                ],
                timestamp=float(i),
            )
            store.record_outcome(outcome)
            learner.update_from_outcome(outcome)

        return learner, store

    def test_exact_match_still_preferred(self):
        """With exact match available, returns exact match thresholds."""
        fp = ProblemFingerprint(noise_regime=NoiseRegime.MEDIUM)
        learner, store = self._make_learner_with_data(fp, n_campaigns=5)

        result = learner.suggest_thresholds(fp)
        assert result is not None
        assert result.n_campaigns == 5

    def test_fallback_to_similar(self):
        """No exact match but similar fingerprint exists -> returns learned thresholds."""
        stored_fp = ProblemFingerprint(noise_regime=NoiseRegime.MEDIUM)
        learner, store = self._make_learner_with_data(stored_fp, n_campaigns=5)

        # Query with LOW noise (default) - similar but not exact
        query_fp = ProblemFingerprint()
        # Similarity ~ 0.882 > 0.5
        result = learner.suggest_thresholds(query_fp)
        assert result is not None, "Should fall back to similar fingerprint"
        assert result.n_campaigns == 5

    def test_no_fallback_if_too_different(self):
        """Very different fingerprint (sim < 0.5) returns None."""
        stored_fp = ProblemFingerprint(
            variable_types=VariableType.CATEGORICAL,
            objective_form=ObjectiveForm.CONSTRAINED,
            noise_regime=NoiseRegime.HIGH,
            cost_profile=CostProfile.HETEROGENEOUS,
            failure_informativeness=FailureInformativeness.STRONG,
            data_scale=DataScale.MODERATE,
            dynamics=Dynamics.TIME_SERIES,
            feasible_region=FeasibleRegion.FRAGMENTED,
        )
        learner, store = self._make_learner_with_data(stored_fp, n_campaigns=5)

        query_fp = ProblemFingerprint()
        result = learner.suggest_thresholds(query_fp)
        assert result is None, "Should not fall back to very different fingerprint"

    def test_cold_start_guard(self):
        """Similar fingerprint exists but n_campaigns < min_experiences -> None."""
        stored_fp = ProblemFingerprint(noise_regime=NoiseRegime.MEDIUM)
        learner, store = self._make_learner_with_data(
            stored_fp, n_campaigns=2, min_exp=3
        )

        query_fp = ProblemFingerprint()
        result = learner.suggest_thresholds(query_fp)
        assert result is None, (
            "Should not return thresholds when n_campaigns < min_experiences"
        )
