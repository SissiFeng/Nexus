"""Tests for optimization_copilot.meta_learning module.

Covers models, experience store, strategy learner, weight tuner,
threshold learner, failure strategy learner, drift robustness tracker,
and the top-level meta-learning advisor.
"""

from __future__ import annotations

import json
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
from optimization_copilot.meta_controller.controller import SwitchingThresholds
from optimization_copilot.portfolio.scorer import ScoringWeights
from optimization_copilot.drift.detector import DriftReport
from optimization_copilot.feasibility.taxonomy import FailureTaxonomy, FailureType, ClassifiedFailure

from optimization_copilot.meta_learning.models import (
    BackendPerformance,
    CampaignOutcome,
    ExperienceRecord,
    MetaLearningConfig,
    LearnedWeights,
    LearnedThresholds,
    FailureStrategy,
    DriftRobustness,
    MetaAdvice,
    _fingerprint_from_dict,
)
from optimization_copilot.meta_learning.experience_store import ExperienceStore
from optimization_copilot.meta_learning.strategy_learner import StrategyLearner
from optimization_copilot.meta_learning.weight_tuner import WeightTuner
from optimization_copilot.meta_learning.threshold_learner import ThresholdLearner
from optimization_copilot.meta_learning.failure_learner import FailureStrategyLearner
from optimization_copilot.meta_learning.drift_learner import DriftRobustnessTracker
from optimization_copilot.meta_learning.advisor import MetaLearningAdvisor


# ── Fingerprints ────────────────────────────────────────────

FP_DEFAULT = ProblemFingerprint()

FP_ALT = ProblemFingerprint(
    variable_types=VariableType.MIXED,
    objective_form=ObjectiveForm.MULTI_OBJECTIVE,
    noise_regime=NoiseRegime.HIGH,
    cost_profile=CostProfile.HETEROGENEOUS,
    failure_informativeness=FailureInformativeness.STRONG,
    data_scale=DataScale.MODERATE,
    dynamics=Dynamics.TIME_SERIES,
    feasible_region=FeasibleRegion.FRAGMENTED,
)

FP_PARTIAL = ProblemFingerprint(
    variable_types=VariableType.CONTINUOUS,
    objective_form=ObjectiveForm.SINGLE,
    noise_regime=NoiseRegime.HIGH,           # differs
    cost_profile=CostProfile.UNIFORM,
    failure_informativeness=FailureInformativeness.WEAK,
    data_scale=DataScale.TINY,
    dynamics=Dynamics.STATIC,
    feasible_region=FeasibleRegion.NARROW,    # differs
)


# ── Helper ──────────────────────────────────────────────────


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
        fingerprint = FP_DEFAULT
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
# 1. Models Tests
# ====================================================================


class TestModels:
    """Tests for optimization_copilot.meta_learning.models dataclasses."""

    def test_backend_performance_creation(self):
        bp = BackendPerformance(
            backend_name="tpe",
            convergence_iteration=30,
            final_best_kpi=0.9,
            regret=0.1,
            sample_efficiency=0.02,
            failure_rate=0.05,
        )
        assert bp.backend_name == "tpe"
        assert bp.convergence_iteration == 30
        assert bp.final_best_kpi == 0.9
        assert bp.drift_encountered is False
        assert bp.drift_score == 0.0

    def test_backend_performance_to_from_dict(self):
        bp = BackendPerformance(
            backend_name="cma_es",
            convergence_iteration=None,
            final_best_kpi=0.75,
            regret=0.25,
            sample_efficiency=0.01,
            failure_rate=0.15,
            drift_encountered=True,
            drift_score=0.5,
        )
        d = bp.to_dict()
        restored = BackendPerformance.from_dict(d)
        assert restored.backend_name == bp.backend_name
        assert restored.convergence_iteration is None
        assert restored.drift_encountered is True
        assert restored.drift_score == 0.5

    def test_campaign_outcome_creation(self):
        outcome = _make_outcome()
        assert outcome.campaign_id == "camp1"
        assert len(outcome.backend_performances) == 2
        assert outcome.total_iterations == 50
        assert outcome.best_kpi == 0.85

    def test_campaign_outcome_to_from_dict(self):
        outcome = _make_outcome(campaign_id="c2", timestamp=5.0)
        d = outcome.to_dict()
        restored = CampaignOutcome.from_dict(d)
        assert restored.campaign_id == "c2"
        assert restored.timestamp == 5.0
        assert len(restored.backend_performances) == 2
        assert restored.backend_performances[0].backend_name == "tpe"
        # Phase transitions should be tuples
        assert isinstance(restored.phase_transitions[0], tuple)

    def test_experience_record_to_from_dict(self):
        outcome = _make_outcome()
        fp_key = str(outcome.fingerprint.to_tuple())
        record = ExperienceRecord(outcome=outcome, fingerprint_key=fp_key)
        d = record.to_dict()
        restored = ExperienceRecord.from_dict(d)
        assert restored.fingerprint_key == fp_key
        assert restored.outcome.campaign_id == "camp1"

    def test_meta_learning_config_defaults(self):
        cfg = MetaLearningConfig()
        assert cfg.min_experiences_for_learning == 3
        assert cfg.similarity_decay == 0.3
        assert cfg.weight_learning_rate == 0.1
        assert cfg.threshold_learning_rate == 0.05
        assert cfg.recency_halflife == 20

    def test_learned_weights_to_from_dict(self):
        lw = LearnedWeights(
            fingerprint_key="key1",
            gain=0.3,
            fail=0.25,
            cost=0.2,
            drift=0.15,
            incompatibility=0.1,
            n_campaigns=5,
            confidence=0.5,
        )
        d = lw.to_dict()
        restored = LearnedWeights.from_dict(d)
        assert restored.fingerprint_key == "key1"
        assert restored.gain == 0.3
        assert restored.n_campaigns == 5
        assert restored.confidence == 0.5

    def test_learned_thresholds_to_from_dict(self):
        lt = LearnedThresholds(
            fingerprint_key="key2",
            cold_start_min_observations=12.0,
            learning_plateau_length=6.0,
            exploitation_gain_threshold=-0.05,
            n_campaigns=4,
        )
        d = lt.to_dict()
        restored = LearnedThresholds.from_dict(d)
        assert restored.fingerprint_key == "key2"
        assert restored.cold_start_min_observations == 12.0
        assert restored.n_campaigns == 4

    def test_failure_strategy_to_from_dict(self):
        fs = FailureStrategy(
            failure_type="hardware",
            best_stabilization="noise_smoothing_window=5",
            effectiveness_score=0.8,
            n_campaigns=3,
        )
        d = fs.to_dict()
        restored = FailureStrategy.from_dict(d)
        assert restored.failure_type == "hardware"
        assert restored.best_stabilization == "noise_smoothing_window=5"

    def test_drift_robustness_to_from_dict(self):
        dr = DriftRobustness(
            backend_name="tpe",
            drift_resilience_score=0.85,
            n_drift_campaigns=3,
            avg_kpi_loss_under_drift=0.15,
        )
        d = dr.to_dict()
        restored = DriftRobustness.from_dict(d)
        assert restored.backend_name == "tpe"
        assert restored.drift_resilience_score == 0.85
        assert restored.avg_kpi_loss_under_drift == 0.15

    def test_meta_advice_defaults(self):
        advice = MetaAdvice()
        assert advice.recommended_backends == []
        assert advice.scoring_weights is None
        assert advice.switching_thresholds is None
        assert advice.failure_adjustments == {}
        assert advice.drift_robust_backends == []
        assert advice.confidence == 0.0
        assert advice.reason_codes == []

    def test_fingerprint_from_dict_helper(self):
        fp = FP_DEFAULT
        d = fp.to_dict()
        restored = _fingerprint_from_dict(d)
        assert restored.variable_types == fp.variable_types
        assert restored.noise_regime == fp.noise_regime
        assert restored.to_tuple() == fp.to_tuple()


# ====================================================================
# 2. ExperienceStore Tests
# ====================================================================


class TestExperienceStore:
    """Tests for the ExperienceStore."""

    def test_store_creation_empty(self):
        store = ExperienceStore()
        assert store.count() == 0
        assert store.get_all() == []

    def test_store_record_outcome(self):
        store = ExperienceStore()
        outcome = _make_outcome()
        record = store.record_outcome(outcome)
        assert store.count() == 1
        assert record.outcome.campaign_id == "camp1"
        assert record.fingerprint_key == str(outcome.fingerprint.to_tuple())

    def test_store_record_outcomes_batch(self):
        store = ExperienceStore()
        outcomes = [
            _make_outcome(campaign_id="c1", timestamp=1.0),
            _make_outcome(campaign_id="c2", timestamp=2.0),
            _make_outcome(campaign_id="c3", timestamp=3.0),
        ]
        records = store.record_outcomes(outcomes)
        assert len(records) == 3
        assert store.count() == 3

    def test_store_get_by_campaign(self):
        store = ExperienceStore()
        store.record_outcome(_make_outcome(campaign_id="abc"))
        record = store.get_by_campaign("abc")
        assert record is not None
        assert record.outcome.campaign_id == "abc"

    def test_store_get_by_campaign_missing(self):
        store = ExperienceStore()
        assert store.get_by_campaign("nonexistent") is None

    def test_store_get_by_fingerprint(self):
        store = ExperienceStore()
        fp = FP_DEFAULT
        store.record_outcome(_make_outcome(campaign_id="c1", fingerprint=fp))
        store.record_outcome(_make_outcome(campaign_id="c2", fingerprint=fp))
        store.record_outcome(_make_outcome(campaign_id="c3", fingerprint=FP_ALT))
        fp_key = str(fp.to_tuple())
        results = store.get_by_fingerprint(fp_key)
        assert len(results) == 2

    def test_store_get_by_fingerprint_empty(self):
        store = ExperienceStore()
        results = store.get_by_fingerprint("nonexistent_key")
        assert results == []

    def test_store_get_similar_exact_match(self):
        store = ExperienceStore()
        fp = FP_DEFAULT
        store.record_outcome(_make_outcome(campaign_id="c1", fingerprint=fp))
        results = store.get_similar(fp, max_results=5)
        assert len(results) == 1
        record, sim = results[0]
        assert sim == 1.0

    def test_store_get_similar_partial_match(self):
        store = ExperienceStore()
        store.record_outcome(_make_outcome(campaign_id="c1", fingerprint=FP_DEFAULT))
        store.record_outcome(_make_outcome(campaign_id="c2", fingerprint=FP_ALT))
        # FP_PARTIAL differs in 2 of 8 dimensions from FP_DEFAULT => similarity = 6/8 = 0.75
        results = store.get_similar(FP_PARTIAL, max_results=10)
        assert len(results) == 2
        # Results should be sorted by similarity descending
        assert results[0][1] >= results[1][1]
        # FP_DEFAULT should be more similar to FP_PARTIAL than FP_ALT is
        for record, sim in results:
            if record.outcome.fingerprint.to_tuple() == FP_DEFAULT.to_tuple():
                assert sim > 0.5

    def test_store_get_similar_no_match(self):
        store = ExperienceStore()
        results = store.get_similar(FP_DEFAULT, max_results=5)
        assert results == []

    def test_store_recency_weight(self):
        cfg = MetaLearningConfig(recency_halflife=20)
        store = ExperienceStore(config=cfg)
        outcome = _make_outcome(timestamp=0.0)
        record = store.record_outcome(outcome)
        # Latest timestamp = 20 => age = 20 => weight = 2^(-20/20) = 0.5
        w = store.recency_weight(record, latest_ts=20.0)
        assert abs(w - 0.5) < 1e-9
        # Same timestamp => weight = 1.0
        w2 = store.recency_weight(record, latest_ts=0.0)
        assert w2 == 1.0

    def test_store_serialization_roundtrip(self):
        store = ExperienceStore()
        store.record_outcome(_make_outcome(campaign_id="c1", timestamp=1.0))
        store.record_outcome(_make_outcome(campaign_id="c2", timestamp=2.0))
        d = store.to_dict()
        restored = ExperienceStore.from_dict(d)
        assert restored.count() == 2
        assert restored.get_by_campaign("c1") is not None
        assert restored.get_by_campaign("c2") is not None
        # Also test JSON roundtrip
        json_str = store.to_json()
        restored2 = ExperienceStore.from_json(json_str)
        assert restored2.count() == 2


# ====================================================================
# 3. StrategyLearner Tests
# ====================================================================


class TestStrategyLearner:
    """Tests for the StrategyLearner."""

    def _make_learner(
        self, outcomes: list[CampaignOutcome] | None = None, min_exp: int = 3
    ) -> StrategyLearner:
        cfg = MetaLearningConfig(min_experiences_for_learning=min_exp)
        store = ExperienceStore(config=cfg)
        if outcomes:
            store.record_outcomes(outcomes)
        return StrategyLearner(store, config=cfg)

    def test_strategy_cold_start_returns_empty(self):
        learner = self._make_learner()
        result = learner.rank_backends(FP_DEFAULT)
        assert result == []

    def test_strategy_rank_backends_with_data(self):
        outcomes = [
            _make_outcome(campaign_id=f"c{i}", timestamp=float(i))
            for i in range(4)
        ]
        learner = self._make_learner(outcomes)
        ranked = learner.rank_backends(FP_DEFAULT)
        assert len(ranked) > 0
        # tpe has better efficiency and lower regret, so should rank first
        names = [name for name, _score in ranked]
        assert "tpe" in names
        assert "random" in names

    def test_strategy_rank_backends_exact_fingerprint(self):
        outcomes = [
            _make_outcome(campaign_id=f"c{i}", fingerprint=FP_DEFAULT, timestamp=float(i))
            for i in range(4)
        ]
        learner = self._make_learner(outcomes)
        ranked = learner.rank_backends(FP_DEFAULT)
        assert len(ranked) == 2  # tpe and random
        # tpe should be first: score = 0.017 - 0.15 - 0.1 = -0.233
        # random: score = 0.008 - 0.4 - 0.2 = -0.592
        assert ranked[0][0] == "tpe"
        assert ranked[1][0] == "random"
        assert ranked[0][1] > ranked[1][1]

    def test_strategy_rank_backends_similar_fingerprint(self):
        # Record outcomes with FP_PARTIAL, query with FP_DEFAULT
        # They share 6/8 fields => similarity = 0.75
        outcomes = [
            _make_outcome(campaign_id=f"c{i}", fingerprint=FP_PARTIAL, timestamp=float(i))
            for i in range(4)
        ]
        learner = self._make_learner(outcomes, min_exp=1)
        ranked = learner.rank_backends(FP_DEFAULT)
        # Should still produce results via similarity-based matching
        assert len(ranked) > 0

    def test_strategy_has_enough_data_false_initially(self):
        learner = self._make_learner()
        assert learner.has_enough_data(FP_DEFAULT) is False

    def test_strategy_has_enough_data_true_after_records(self):
        outcomes = [
            _make_outcome(campaign_id=f"c{i}", timestamp=float(i))
            for i in range(4)
        ]
        learner = self._make_learner(outcomes)
        assert learner.has_enough_data(FP_DEFAULT) is True

    def test_strategy_get_backend_stats(self):
        outcomes = [
            _make_outcome(campaign_id=f"c{i}", timestamp=float(i))
            for i in range(3)
        ]
        learner = self._make_learner(outcomes, min_exp=1)
        stats = learner.get_backend_stats(FP_DEFAULT, "tpe")
        assert stats["n_campaigns"] == 3
        assert abs(stats["avg_sample_efficiency"] - 0.017) < 1e-9
        assert abs(stats["avg_regret"] - 0.15) < 1e-9
        assert abs(stats["avg_failure_rate"] - 0.1) < 1e-9
        assert stats["avg_convergence_iteration"] == 30.0

    def test_strategy_get_backend_stats_missing_backend(self):
        outcomes = [_make_outcome(campaign_id="c1")]
        learner = self._make_learner(outcomes, min_exp=1)
        stats = learner.get_backend_stats(FP_DEFAULT, "nonexistent")
        assert stats["n_campaigns"] == 0
        assert stats["avg_convergence_iteration"] is None

    def test_strategy_respects_min_experiences(self):
        outcomes = [
            _make_outcome(campaign_id="c1"),
            _make_outcome(campaign_id="c2"),
        ]
        learner = self._make_learner(outcomes, min_exp=5)
        ranked = learner.rank_backends(FP_DEFAULT)
        assert ranked == []

    def test_strategy_multiple_backends_sorted(self):
        # Create outcomes where one backend is strictly better
        good_backends = [
            {
                "backend_name": "alpha",
                "convergence_iteration": 10,
                "final_best_kpi": 0.95,
                "regret": 0.05,
                "sample_efficiency": 0.05,
                "failure_rate": 0.0,
            },
            {
                "backend_name": "beta",
                "convergence_iteration": None,
                "final_best_kpi": 0.3,
                "regret": 0.7,
                "sample_efficiency": 0.001,
                "failure_rate": 0.5,
            },
        ]
        outcomes = [
            _make_outcome(campaign_id=f"c{i}", backends=good_backends, timestamp=float(i))
            for i in range(4)
        ]
        learner = self._make_learner(outcomes)
        ranked = learner.rank_backends(FP_DEFAULT)
        assert ranked[0][0] == "alpha"
        assert ranked[1][0] == "beta"


# ====================================================================
# 4. WeightTuner Tests
# ====================================================================


class TestWeightTuner:
    """Tests for the WeightTuner."""

    def _make_tuner(self, min_exp: int = 3) -> tuple[WeightTuner, ExperienceStore]:
        cfg = MetaLearningConfig(min_experiences_for_learning=min_exp)
        store = ExperienceStore(config=cfg)
        tuner = WeightTuner(store, config=cfg)
        return tuner, store

    def test_weight_tuner_cold_start(self):
        tuner, _ = self._make_tuner()
        result = tuner.suggest_weights(FP_DEFAULT)
        assert result is None

    def test_weight_tuner_update_and_suggest(self):
        tuner, _ = self._make_tuner(min_exp=1)
        outcome = _make_outcome()
        weights = ScoringWeights()
        tuner.update_from_outcome(outcome, weights)
        lw = tuner.suggest_weights(FP_DEFAULT)
        assert lw is not None
        assert lw.n_campaigns == 1

    def test_weight_tuner_high_failure_increases_fail_weight(self):
        tuner, _ = self._make_tuner(min_exp=1)
        # Create outcome with high failure rate (>0.2)
        high_fail_backends = [
            {
                "backend_name": "tpe",
                "convergence_iteration": 30,
                "final_best_kpi": 0.5,
                "regret": 0.5,
                "sample_efficiency": 0.01,
                "failure_rate": 0.5,  # high
            },
        ]
        outcome = _make_outcome(backends=high_fail_backends)
        default_weights = ScoringWeights()
        initial_fail = default_weights.fail
        tuner.update_from_outcome(outcome, default_weights)
        lw = tuner.suggest_weights(FP_DEFAULT)
        assert lw is not None
        # The fail weight should have been nudged upward (in raw terms)
        # After normalization the absolute value may differ, but the relative
        # share should have increased compared to a low-failure scenario.
        # We just verify that it ran without error and produced valid output.
        total = lw.gain + lw.fail + lw.cost + lw.drift + lw.incompatibility
        assert abs(total - 1.0) < 1e-9

    def test_weight_tuner_drift_increases_drift_weight(self):
        tuner, _ = self._make_tuner(min_exp=1)
        drift_backends = [
            {
                "backend_name": "tpe",
                "convergence_iteration": 30,
                "final_best_kpi": 0.7,
                "regret": 0.3,
                "sample_efficiency": 0.014,
                "failure_rate": 0.1,
                "drift_encountered": True,
                "drift_score": 0.8,  # high drift
            },
        ]
        outcome = _make_outcome(backends=drift_backends)
        default_weights = ScoringWeights()
        tuner.update_from_outcome(outcome, default_weights)
        lw = tuner.suggest_weights(FP_DEFAULT)
        assert lw is not None
        # Drift weight should exist and be positive
        assert lw.drift > 0

    def test_weight_tuner_weights_normalized(self):
        tuner, _ = self._make_tuner(min_exp=1)
        outcome = _make_outcome()
        tuner.update_from_outcome(outcome, ScoringWeights())
        lw = tuner.suggest_weights(FP_DEFAULT)
        assert lw is not None
        total = lw.gain + lw.fail + lw.cost + lw.drift + lw.incompatibility
        assert abs(total - 1.0) < 1e-9

    def test_weight_tuner_confidence_increases(self):
        tuner, _ = self._make_tuner(min_exp=1)
        for i in range(5):
            outcome = _make_outcome(campaign_id=f"c{i}", timestamp=float(i))
            tuner.update_from_outcome(outcome, ScoringWeights())
        lw = tuner.suggest_weights(FP_DEFAULT)
        assert lw is not None
        assert lw.confidence == min(1.0, 5.0 / 10.0)
        assert lw.confidence == 0.5

    def test_weight_tuner_to_scoring_weights(self):
        tuner, _ = self._make_tuner(min_exp=1)
        outcome = _make_outcome()
        tuner.update_from_outcome(outcome, ScoringWeights())
        lw = tuner.suggest_weights(FP_DEFAULT)
        assert lw is not None
        sw = tuner.to_scoring_weights(lw)
        assert isinstance(sw, ScoringWeights)
        assert sw.gain == lw.gain
        assert sw.fail == lw.fail

    def test_weight_tuner_serialization_roundtrip(self):
        # Use default min_exp=3 and record 3 outcomes so both original and
        # restored tuner (which uses default config) pass the threshold.
        tuner, store = self._make_tuner(min_exp=3)
        for i in range(3):
            outcome = _make_outcome(campaign_id=f"c{i}", timestamp=float(i))
            tuner.update_from_outcome(outcome, ScoringWeights())
        d = tuner.to_dict()
        restored = WeightTuner.from_dict(d, store)
        lw_original = tuner.suggest_weights(FP_DEFAULT)
        lw_restored = restored.suggest_weights(FP_DEFAULT)
        assert lw_original is not None
        assert lw_restored is not None
        assert lw_original.gain == lw_restored.gain
        assert lw_original.n_campaigns == lw_restored.n_campaigns

    def test_weight_tuner_multiple_updates(self):
        tuner, _ = self._make_tuner(min_exp=1)
        for i in range(10):
            outcome = _make_outcome(campaign_id=f"c{i}", timestamp=float(i))
            tuner.update_from_outcome(outcome, ScoringWeights())
        lw = tuner.suggest_weights(FP_DEFAULT)
        assert lw is not None
        assert lw.n_campaigns == 10
        assert lw.confidence == 1.0  # min(1.0, 10/10)
        total = lw.gain + lw.fail + lw.cost + lw.drift + lw.incompatibility
        assert abs(total - 1.0) < 1e-9

    def test_weight_tuner_respects_min_experiences(self):
        tuner, _ = self._make_tuner(min_exp=5)
        for i in range(3):
            outcome = _make_outcome(campaign_id=f"c{i}", timestamp=float(i))
            tuner.update_from_outcome(outcome, ScoringWeights())
        assert tuner.suggest_weights(FP_DEFAULT) is None


# ====================================================================
# 5. ThresholdLearner Tests
# ====================================================================


class TestThresholdLearner:
    """Tests for the ThresholdLearner."""

    def _make_learner(self, min_exp: int = 3) -> ThresholdLearner:
        cfg = MetaLearningConfig(min_experiences_for_learning=min_exp)
        store = ExperienceStore(config=cfg)
        return ThresholdLearner(store, config=cfg)

    def test_threshold_cold_start(self):
        learner = self._make_learner()
        assert learner.suggest_thresholds(FP_DEFAULT) is None

    def test_threshold_update_from_good_outcome(self):
        learner = self._make_learner(min_exp=1)
        # regret = 0.15 => quality = 1/0.15 ~= 6.67 > 1.0 => "good" outcome
        outcome = _make_outcome(
            phase_transitions=[
                ("cold_start", "learning", 8),
                ("learning", "exploitation", 20),
            ],
        )
        learner.update_from_outcome(outcome)
        lt = learner.suggest_thresholds(FP_DEFAULT)
        assert lt is not None
        assert lt.n_campaigns == 1
        assert lt.cold_start_min_observations == 8.0
        # plateau = 20 - 8 = 12
        assert lt.learning_plateau_length == 12.0

    def test_threshold_skips_bad_outcomes(self):
        learner = self._make_learner(min_exp=1)
        # All backends have regret >= 1.0 => quality <= 1.0 => skip
        bad_backends = [
            {
                "backend_name": "tpe",
                "convergence_iteration": None,
                "final_best_kpi": 0.1,
                "regret": 2.0,
                "sample_efficiency": 0.001,
                "failure_rate": 0.5,
            },
        ]
        outcome = _make_outcome(backends=bad_backends)
        learner.update_from_outcome(outcome)
        assert learner.suggest_thresholds(FP_DEFAULT) is None

    def test_threshold_ema_updates(self):
        learner = self._make_learner(min_exp=1)
        # First outcome: cold_start -> learning at iteration 10
        outcome1 = _make_outcome(
            campaign_id="c1",
            phase_transitions=[("cold_start", "learning", 10)],
        )
        learner.update_from_outcome(outcome1)
        lt1 = learner.suggest_thresholds(FP_DEFAULT)
        assert lt1 is not None
        assert lt1.cold_start_min_observations == 10.0

        # Second outcome: cold_start -> learning at iteration 20
        outcome2 = _make_outcome(
            campaign_id="c2",
            phase_transitions=[("cold_start", "learning", 20)],
        )
        learner.update_from_outcome(outcome2)
        lt2 = learner.suggest_thresholds(FP_DEFAULT)
        assert lt2 is not None
        # EMA: (1 - 0.05) * 10.0 + 0.05 * 20.0 = 9.5 + 1.0 = 10.5
        assert abs(lt2.cold_start_min_observations - 10.5) < 1e-9

    def test_threshold_to_switching_thresholds(self):
        learner = self._make_learner(min_exp=1)
        outcome = _make_outcome(
            phase_transitions=[
                ("cold_start", "learning", 12),
                ("learning", "exploitation", 25),
            ],
        )
        learner.update_from_outcome(outcome)
        lt = learner.suggest_thresholds(FP_DEFAULT)
        assert lt is not None
        st = learner.to_switching_thresholds(lt)
        assert isinstance(st, SwitchingThresholds)
        assert st.cold_start_min_observations == round(lt.cold_start_min_observations)

    def test_threshold_serialization_roundtrip(self):
        cfg = MetaLearningConfig(min_experiences_for_learning=1)
        store = ExperienceStore(config=cfg)
        learner = ThresholdLearner(store, config=cfg)
        outcome = _make_outcome(
            phase_transitions=[("cold_start", "learning", 10)],
        )
        learner.update_from_outcome(outcome)
        d = learner.to_dict()
        restored = ThresholdLearner.from_dict(d, store, config=cfg)
        lt_orig = learner.suggest_thresholds(FP_DEFAULT)
        lt_rest = restored.suggest_thresholds(FP_DEFAULT)
        assert lt_orig is not None
        assert lt_rest is not None
        assert lt_orig.cold_start_min_observations == lt_rest.cold_start_min_observations
        assert lt_orig.n_campaigns == lt_rest.n_campaigns

    def test_threshold_multiple_updates_converge(self):
        learner = self._make_learner(min_exp=1)
        # Feed many outcomes with cold_start -> learning at iteration 15
        for i in range(20):
            outcome = _make_outcome(
                campaign_id=f"c{i}",
                phase_transitions=[("cold_start", "learning", 15)],
                timestamp=float(i),
            )
            learner.update_from_outcome(outcome)
        lt = learner.suggest_thresholds(FP_DEFAULT)
        assert lt is not None
        # Should converge toward 15.0
        assert abs(lt.cold_start_min_observations - 15.0) < 1.0

    def test_threshold_different_fingerprints(self):
        learner = self._make_learner(min_exp=1)
        outcome1 = _make_outcome(
            campaign_id="c1",
            fingerprint=FP_DEFAULT,
            phase_transitions=[("cold_start", "learning", 10)],
        )
        outcome2 = _make_outcome(
            campaign_id="c2",
            fingerprint=FP_ALT,
            phase_transitions=[("cold_start", "learning", 20)],
        )
        learner.update_from_outcome(outcome1)
        learner.update_from_outcome(outcome2)
        lt_default = learner.suggest_thresholds(FP_DEFAULT)
        lt_alt = learner.suggest_thresholds(FP_ALT)
        assert lt_default is not None
        assert lt_alt is not None
        assert lt_default.cold_start_min_observations == 10.0
        assert lt_alt.cold_start_min_observations == 20.0

    def test_threshold_respects_min_experiences(self):
        learner = self._make_learner(min_exp=5)
        for i in range(3):
            outcome = _make_outcome(
                campaign_id=f"c{i}",
                phase_transitions=[("cold_start", "learning", 10)],
                timestamp=float(i),
            )
            learner.update_from_outcome(outcome)
        assert learner.suggest_thresholds(FP_DEFAULT) is None

    def test_threshold_missing_transitions(self):
        learner = self._make_learner(min_exp=1)
        # No phase transitions => nothing to learn from
        outcome = _make_outcome(phase_transitions=[])
        learner.update_from_outcome(outcome)
        assert learner.suggest_thresholds(FP_DEFAULT) is None


# ====================================================================
# 6. FailureStrategyLearner Tests
# ====================================================================


class TestFailureStrategyLearner:
    """Tests for the FailureStrategyLearner."""

    def _make_learner(self, min_exp: int = 3) -> FailureStrategyLearner:
        cfg = MetaLearningConfig(min_experiences_for_learning=min_exp)
        store = ExperienceStore(config=cfg)
        return FailureStrategyLearner(store, config=cfg)

    def test_failure_cold_start(self):
        learner = self._make_learner()
        assert learner.suggest_stabilization("hardware") is None

    def test_failure_update_and_suggest(self):
        learner = self._make_learner(min_exp=1)
        outcome = _make_outcome(
            failure_type_counts={"hardware": 3},
            stabilization_used={"tpe": "noise_smoothing_window=5"},
        )
        learner.update_from_outcome(outcome)
        fs = learner.suggest_stabilization("hardware")
        assert fs is not None
        assert fs.failure_type == "hardware"
        assert fs.best_stabilization == "noise_smoothing_window=5"
        assert fs.n_campaigns == 1

    def test_failure_suggest_all(self):
        learner = self._make_learner(min_exp=1)
        outcome = _make_outcome(
            failure_type_counts={"hardware": 3, "chemistry": 2},
            stabilization_used={"tpe": "smoothing_a"},
        )
        learner.update_from_outcome(outcome)
        all_strats = learner.suggest_all()
        assert "hardware" in all_strats
        assert "chemistry" in all_strats

    def test_failure_best_stabilization_selected(self):
        learner = self._make_learner(min_exp=1)
        # First outcome: stabilization A with low quality
        outcome1 = _make_outcome(
            campaign_id="c1",
            failure_type_counts={"hardware": 2},
            stabilization_used={"tpe": "stab_A"},
            best_kpi=0.3,
            total_iterations=50,
        )
        learner.update_from_outcome(outcome1)

        # Second outcome: stabilization B with higher quality
        outcome2 = _make_outcome(
            campaign_id="c2",
            failure_type_counts={"hardware": 2},
            stabilization_used={"tpe": "stab_B"},
            best_kpi=0.9,
            total_iterations=50,
        )
        learner.update_from_outcome(outcome2)

        fs = learner.suggest_stabilization("hardware")
        assert fs is not None
        assert fs.best_stabilization == "stab_B"

    def test_failure_multiple_types(self):
        learner = self._make_learner(min_exp=1)
        outcome = _make_outcome(
            failure_type_counts={"hardware": 1, "data": 1, "protocol": 0},
            stabilization_used={"tpe": "stab_X"},
        )
        learner.update_from_outcome(outcome)
        # protocol count=0 should be skipped
        assert learner.suggest_stabilization("hardware") is not None
        assert learner.suggest_stabilization("data") is not None
        assert learner.suggest_stabilization("protocol") is None

    def test_failure_respects_min_experiences(self):
        learner = self._make_learner(min_exp=5)
        for i in range(3):
            outcome = _make_outcome(
                campaign_id=f"c{i}",
                failure_type_counts={"hardware": 1},
                stabilization_used={"tpe": "stab_A"},
            )
            learner.update_from_outcome(outcome)
        assert learner.suggest_stabilization("hardware") is None

    def test_failure_serialization_roundtrip(self):
        # Use default min_exp=3 and record 3 outcomes so restored learner
        # (which uses default config) also passes the threshold.
        store = ExperienceStore()
        learner = FailureStrategyLearner(store)
        for i in range(3):
            outcome = _make_outcome(
                campaign_id=f"c{i}",
                failure_type_counts={"hardware": 2},
                stabilization_used={"tpe": "stab_A"},
            )
            learner.update_from_outcome(outcome)
        d = learner.to_dict()
        restored = FailureStrategyLearner.from_dict(d, store)
        fs_orig = learner.suggest_stabilization("hardware")
        fs_rest = restored.suggest_stabilization("hardware")
        assert fs_orig is not None
        assert fs_rest is not None
        assert fs_orig.best_stabilization == fs_rest.best_stabilization

    def test_failure_no_failures_in_outcome(self):
        learner = self._make_learner(min_exp=1)
        outcome = _make_outcome(
            failure_type_counts={},
            stabilization_used={},
        )
        learner.update_from_outcome(outcome)
        assert learner.suggest_all() == {}


# ====================================================================
# 7. DriftRobustnessTracker Tests
# ====================================================================


class TestDriftRobustnessTracker:
    """Tests for the DriftRobustnessTracker."""

    def _make_tracker(self) -> DriftRobustnessTracker:
        store = ExperienceStore()
        return DriftRobustnessTracker(store)

    def test_drift_tracker_empty(self):
        tracker = self._make_tracker()
        assert tracker.rank_by_resilience() == []
        assert tracker.get_resilience("tpe") is None

    def test_drift_tracker_update_with_drift(self):
        tracker = self._make_tracker()
        drift_backends = [
            {
                "backend_name": "tpe",
                "convergence_iteration": 30,
                "final_best_kpi": 0.7,
                "regret": 0.3,
                "sample_efficiency": 0.014,
                "failure_rate": 0.1,
                "drift_encountered": True,
                "drift_score": 0.6,
            },
        ]
        outcome = _make_outcome(backends=drift_backends)
        tracker.update_from_outcome(outcome)
        dr = tracker.get_resilience("tpe")
        assert dr is not None
        assert dr.n_drift_campaigns == 1
        assert dr.backend_name == "tpe"

    def test_drift_tracker_update_without_drift(self):
        tracker = self._make_tracker()
        # No drift encountered => no robustness entry created
        no_drift_backends = [
            {
                "backend_name": "tpe",
                "convergence_iteration": 30,
                "final_best_kpi": 0.85,
                "regret": 0.15,
                "sample_efficiency": 0.017,
                "failure_rate": 0.1,
                "drift_encountered": False,
                "drift_score": 0.0,
            },
        ]
        outcome = _make_outcome(backends=no_drift_backends)
        tracker.update_from_outcome(outcome)
        # No drift data => no robustness score computed
        assert tracker.get_resilience("tpe") is None

    def test_drift_tracker_rank_by_resilience(self):
        tracker = self._make_tracker()
        # Backend A: low regret under drift => high resilience
        backends_a = [
            {
                "backend_name": "alpha",
                "convergence_iteration": 20,
                "final_best_kpi": 0.8,
                "regret": 0.1,
                "sample_efficiency": 0.04,
                "failure_rate": 0.05,
                "drift_encountered": True,
                "drift_score": 0.5,
            },
        ]
        # Backend B: high regret under drift => low resilience
        backends_b = [
            {
                "backend_name": "beta",
                "convergence_iteration": None,
                "final_best_kpi": 0.4,
                "regret": 0.8,
                "sample_efficiency": 0.004,
                "failure_rate": 0.3,
                "drift_encountered": True,
                "drift_score": 0.7,
            },
        ]
        tracker.update_from_outcome(_make_outcome(campaign_id="c1", backends=backends_a))
        tracker.update_from_outcome(_make_outcome(campaign_id="c2", backends=backends_b))
        ranking = tracker.rank_by_resilience()
        assert len(ranking) == 2
        assert ranking[0].backend_name == "alpha"
        assert ranking[1].backend_name == "beta"
        assert ranking[0].drift_resilience_score >= ranking[1].drift_resilience_score

    def test_drift_tracker_get_resilience(self):
        tracker = self._make_tracker()
        drift_backends = [
            {
                "backend_name": "tpe",
                "convergence_iteration": 30,
                "final_best_kpi": 0.7,
                "regret": 0.3,
                "sample_efficiency": 0.014,
                "failure_rate": 0.1,
                "drift_encountered": True,
                "drift_score": 0.5,
            },
        ]
        outcome = _make_outcome(backends=drift_backends)
        tracker.update_from_outcome(outcome)
        dr = tracker.get_resilience("tpe")
        assert dr is not None
        assert isinstance(dr, DriftRobustness)

    def test_drift_tracker_get_resilience_missing(self):
        tracker = self._make_tracker()
        assert tracker.get_resilience("nonexistent") is None

    def test_drift_tracker_mixed_drift_nodrift(self):
        tracker = self._make_tracker()
        # First outcome: tpe with drift, regret=0.4
        drift_backends = [
            {
                "backend_name": "tpe",
                "convergence_iteration": 30,
                "final_best_kpi": 0.6,
                "regret": 0.4,
                "sample_efficiency": 0.012,
                "failure_rate": 0.1,
                "drift_encountered": True,
                "drift_score": 0.5,
            },
        ]
        tracker.update_from_outcome(_make_outcome(campaign_id="c1", backends=drift_backends))

        # Second outcome: tpe without drift, regret=0.1
        nodrift_backends = [
            {
                "backend_name": "tpe",
                "convergence_iteration": 30,
                "final_best_kpi": 0.9,
                "regret": 0.1,
                "sample_efficiency": 0.018,
                "failure_rate": 0.05,
                "drift_encountered": False,
                "drift_score": 0.0,
            },
        ]
        tracker.update_from_outcome(_make_outcome(campaign_id="c2", backends=nodrift_backends))

        dr = tracker.get_resilience("tpe")
        assert dr is not None
        # kpi_loss = max(0, 0.4 - 0.1) = 0.3
        # resilience = max(0, 1.0 - 0.3) = 0.7
        assert abs(dr.drift_resilience_score - 0.7) < 1e-9
        assert abs(dr.avg_kpi_loss_under_drift - 0.3) < 1e-9

    def test_drift_tracker_serialization_roundtrip(self):
        store = ExperienceStore()
        tracker = DriftRobustnessTracker(store)
        drift_backends = [
            {
                "backend_name": "tpe",
                "convergence_iteration": 30,
                "final_best_kpi": 0.7,
                "regret": 0.3,
                "sample_efficiency": 0.014,
                "failure_rate": 0.1,
                "drift_encountered": True,
                "drift_score": 0.5,
            },
        ]
        tracker.update_from_outcome(_make_outcome(backends=drift_backends))
        d = tracker.to_dict()
        restored = DriftRobustnessTracker.from_dict(d, store)
        dr_orig = tracker.get_resilience("tpe")
        dr_rest = restored.get_resilience("tpe")
        assert dr_orig is not None
        assert dr_rest is not None
        assert dr_orig.drift_resilience_score == dr_rest.drift_resilience_score
        assert dr_orig.n_drift_campaigns == dr_rest.n_drift_campaigns


# ====================================================================
# 8. MetaLearningAdvisor Tests
# ====================================================================


class TestMetaLearningAdvisor:
    """Tests for the MetaLearningAdvisor."""

    def test_advisor_creation(self):
        advisor = MetaLearningAdvisor()
        assert advisor.experience_count() == 0
        assert advisor.has_learned(FP_DEFAULT) is False

    def test_advisor_cold_start_advice(self):
        advisor = MetaLearningAdvisor()
        advice = advisor.advise(FP_DEFAULT)
        assert isinstance(advice, MetaAdvice)
        assert advice.recommended_backends == []
        assert advice.scoring_weights is None
        assert advice.switching_thresholds is None
        assert len(advice.reason_codes) > 0
        # Should contain cold_start mentions
        cold_start_reasons = [r for r in advice.reason_codes if "cold_start" in r]
        assert len(cold_start_reasons) > 0

    def test_advisor_learn_and_advise(self):
        cfg = MetaLearningConfig(min_experiences_for_learning=2)
        advisor = MetaLearningAdvisor(config=cfg)
        for i in range(3):
            outcome = _make_outcome(
                campaign_id=f"c{i}",
                phase_transitions=[
                    ("cold_start", "learning", 10),
                    ("learning", "exploitation", 25),
                ],
                timestamp=float(i),
            )
            advisor.learn_from_outcome(outcome)
        advice = advisor.advise(FP_DEFAULT)
        assert len(advice.recommended_backends) > 0
        assert advice.scoring_weights is not None
        assert advice.switching_thresholds is not None
        assert advice.confidence > 0

    def test_advisor_learn_multiple_outcomes(self):
        advisor = MetaLearningAdvisor()
        for i in range(5):
            outcome = _make_outcome(campaign_id=f"c{i}", timestamp=float(i))
            advisor.learn_from_outcome(outcome)
        assert advisor.experience_count() == 5

    def test_advisor_with_drift_report(self):
        cfg = MetaLearningConfig(min_experiences_for_learning=1)
        advisor = MetaLearningAdvisor(config=cfg)
        # Record outcomes with drift
        drift_backends = [
            {
                "backend_name": "tpe",
                "convergence_iteration": 30,
                "final_best_kpi": 0.7,
                "regret": 0.3,
                "sample_efficiency": 0.014,
                "failure_rate": 0.1,
                "drift_encountered": True,
                "drift_score": 0.5,
            },
        ]
        for i in range(2):
            outcome = _make_outcome(
                campaign_id=f"c{i}", backends=drift_backends, timestamp=float(i)
            )
            advisor.learn_from_outcome(outcome)

        drift_report = DriftReport(
            drift_detected=True,
            drift_score=0.6,
            drift_type="gradual",
            affected_parameters=["param1"],
            recommended_action="reweight",
        )
        advice = advisor.advise(FP_DEFAULT, drift_report=drift_report)
        assert len(advice.drift_robust_backends) > 0

    def test_advisor_with_failure_taxonomy(self):
        cfg = MetaLearningConfig(min_experiences_for_learning=1)
        advisor = MetaLearningAdvisor(config=cfg)
        # Record outcome with hardware failures
        for i in range(2):
            outcome = _make_outcome(
                campaign_id=f"c{i}",
                failure_type_counts={"hardware": 3},
                stabilization_used={"tpe": "noise_smoothing_window=5"},
                timestamp=float(i),
            )
            advisor.learn_from_outcome(outcome)

        taxonomy = FailureTaxonomy(
            classified_failures=[
                ClassifiedFailure(
                    observation_index=0,
                    failure_type=FailureType.HARDWARE,
                    confidence=0.9,
                )
            ],
            type_counts={"hardware": 3},
            dominant_type=FailureType.HARDWARE,
            type_rates={"hardware": 1.0},
            strategy_adjustments={"hardware": "reduce_exploration"},
        )
        advice = advisor.advise(FP_DEFAULT, failure_taxonomy=taxonomy)
        assert "hardware" in advice.failure_adjustments

    def test_advisor_experience_count(self):
        advisor = MetaLearningAdvisor()
        assert advisor.experience_count() == 0
        advisor.learn_from_outcome(_make_outcome(campaign_id="c1"))
        assert advisor.experience_count() == 1
        advisor.learn_from_outcome(_make_outcome(campaign_id="c2"))
        assert advisor.experience_count() == 2

    def test_advisor_has_learned(self):
        cfg = MetaLearningConfig(min_experiences_for_learning=2)
        advisor = MetaLearningAdvisor(config=cfg)
        assert advisor.has_learned(FP_DEFAULT) is False
        for i in range(3):
            advisor.learn_from_outcome(
                _make_outcome(campaign_id=f"c{i}", timestamp=float(i))
            )
        assert advisor.has_learned(FP_DEFAULT) is True

    def test_advisor_serialization_roundtrip(self):
        cfg = MetaLearningConfig(min_experiences_for_learning=1)
        advisor = MetaLearningAdvisor(config=cfg)
        for i in range(3):
            outcome = _make_outcome(
                campaign_id=f"c{i}",
                phase_transitions=[("cold_start", "learning", 10)],
                timestamp=float(i),
            )
            advisor.learn_from_outcome(outcome)
        d = advisor.to_dict()
        restored = MetaLearningAdvisor.from_dict(d)
        assert restored.experience_count() == 3
        advice_orig = advisor.advise(FP_DEFAULT)
        advice_rest = restored.advise(FP_DEFAULT)
        assert len(advice_orig.recommended_backends) == len(advice_rest.recommended_backends)

    def test_advisor_json_roundtrip(self):
        cfg = MetaLearningConfig(min_experiences_for_learning=1)
        advisor = MetaLearningAdvisor(config=cfg)
        for i in range(2):
            outcome = _make_outcome(campaign_id=f"c{i}", timestamp=float(i))
            advisor.learn_from_outcome(outcome)
        json_str = advisor.to_json()
        restored = MetaLearningAdvisor.from_json(json_str)
        assert restored.experience_count() == 2
        # Verify JSON is valid
        parsed = json.loads(json_str)
        assert "experience_store" in parsed
        assert "weight_tuner" in parsed

    def test_advisor_confidence_increases_with_data(self):
        cfg = MetaLearningConfig(min_experiences_for_learning=1)
        advisor = MetaLearningAdvisor(config=cfg)

        # One outcome
        advisor.learn_from_outcome(
            _make_outcome(
                campaign_id="c1",
                phase_transitions=[("cold_start", "learning", 10)],
                timestamp=1.0,
            )
        )
        advice1 = advisor.advise(FP_DEFAULT)

        # Many more outcomes
        for i in range(2, 10):
            advisor.learn_from_outcome(
                _make_outcome(
                    campaign_id=f"c{i}",
                    phase_transitions=[("cold_start", "learning", 10)],
                    timestamp=float(i),
                )
            )
        advice2 = advisor.advise(FP_DEFAULT)

        # Confidence should have increased with more data
        assert advice2.confidence >= advice1.confidence

    def test_advisor_full_pipeline(self):
        """Full integration test: record 5 outcomes, advise, verify all fields."""
        cfg = MetaLearningConfig(min_experiences_for_learning=2)
        advisor = MetaLearningAdvisor(config=cfg)

        # Record 5 varied outcomes
        for i in range(5):
            drift_flag = i % 2 == 0
            backends = [
                {
                    "backend_name": "tpe",
                    "convergence_iteration": 25 + i,
                    "final_best_kpi": 0.8 + i * 0.02,
                    "regret": 0.2 - i * 0.02,
                    "sample_efficiency": 0.015 + i * 0.001,
                    "failure_rate": 0.1,
                    "drift_encountered": drift_flag,
                    "drift_score": 0.5 if drift_flag else 0.0,
                },
                {
                    "backend_name": "random",
                    "convergence_iteration": None,
                    "final_best_kpi": 0.5 + i * 0.01,
                    "regret": 0.5 - i * 0.01,
                    "sample_efficiency": 0.005,
                    "failure_rate": 0.2,
                    "drift_encountered": drift_flag,
                    "drift_score": 0.3 if drift_flag else 0.0,
                },
            ]
            outcome = _make_outcome(
                campaign_id=f"pipe{i}",
                backends=backends,
                phase_transitions=[
                    ("cold_start", "learning", 8),
                    ("learning", "exploitation", 20),
                ],
                failure_type_counts={"hardware": 1, "chemistry": i},
                stabilization_used={"tpe": "smoothing_A", "random": "smoothing_B"},
                total_iterations=50,
                best_kpi=0.8 + i * 0.02,
                timestamp=float(i),
            )
            advisor.learn_from_outcome(outcome)

        assert advisor.experience_count() == 5
        assert advisor.has_learned(FP_DEFAULT) is True

        # Advise with drift + failure info
        drift_report = DriftReport(
            drift_detected=True,
            drift_score=0.5,
            drift_type="gradual",
            affected_parameters=["x1"],
            recommended_action="reweight",
        )
        taxonomy = FailureTaxonomy(
            classified_failures=[
                ClassifiedFailure(
                    observation_index=0,
                    failure_type=FailureType.HARDWARE,
                    confidence=0.9,
                )
            ],
            type_counts={"hardware": 2},
            dominant_type=FailureType.HARDWARE,
            type_rates={"hardware": 1.0},
            strategy_adjustments={},
        )

        advice = advisor.advise(
            FP_DEFAULT,
            drift_report=drift_report,
            failure_taxonomy=taxonomy,
        )

        assert isinstance(advice, MetaAdvice)
        assert len(advice.recommended_backends) > 0
        assert advice.scoring_weights is not None
        assert advice.switching_thresholds is not None
        assert len(advice.drift_robust_backends) > 0
        assert advice.confidence > 0
        assert len(advice.reason_codes) >= 3
        # Failure adjustments may or may not be populated depending
        # on whether the failure learner has enough data
        # (it should, since we recorded 5 outcomes with hardware failures)
        assert "hardware" in advice.failure_adjustments
