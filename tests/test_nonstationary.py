"""Tests for the Non-Stationary Optimization module.

Covers:
  - TimeWeighter (exponential, sliding_window, linear_decay strategies)
  - SeasonalDetector (autocorrelation-based periodicity detection)
  - NonStationaryAdapter (assessment and decision adaptation)
"""

from __future__ import annotations

import math

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    Phase,
    RiskPosture,
    StabilizeSpec,
    StrategyDecision,
    VariableType,
)
from optimization_copilot.nonstationary.weighter import TimeWeighter, TimeWeights
from optimization_copilot.nonstationary.seasonal import SeasonalDetector, SeasonalPattern
from optimization_copilot.nonstationary.adapter import NonStationaryAdapter, NonStationaryAssessment


# ── Helpers ───────────────────────────────────────────────


def _make_specs():
    return [
        ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
    ]


def _make_snapshot(kpi_values, timestamps=None):
    n = len(kpi_values)
    if timestamps is None:
        timestamps = [float(i) for i in range(n)]
    obs = [
        Observation(
            iteration=i,
            parameters={"x1": float(i) / max(n, 1) * 10},
            kpi_values={"y": kpi_values[i]},
            timestamp=timestamps[i],
        )
        for i in range(n)
    ]
    return CampaignSnapshot(
        campaign_id="ns-test",
        parameter_specs=_make_specs(),
        observations=obs,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=n,
    )


def _base_decision(phase=Phase.LEARNING):
    return StrategyDecision(
        backend_name="tpe",
        stabilize_spec=StabilizeSpec(),
        exploration_strength=0.5,
        batch_size=3,
        risk_posture=RiskPosture.MODERATE,
        phase=phase,
        reason_codes=["test"],
    )


# ── TestTimeWeighterExponential ──────────────────────────


class TestTimeWeighterExponential:
    """Tests for the exponential decay strategy of TimeWeighter."""

    def test_recent_higher_weight(self):
        """Most recent observation should have the highest weight."""
        snap = _make_snapshot([1.0] * 10)
        tw = TimeWeighter(strategy="exponential", decay_rate=0.1)
        result = tw.compute_weights(snap)
        assert result.weights[9] > result.weights[0]

    def test_all_same_timestamp(self):
        """When all timestamps are 0, the weighter falls back to iteration indices.

        With decay_rate=0, all weights should be 1.0 regardless.
        """
        snap = _make_snapshot([2.0] * 5, timestamps=[0.0] * 5)
        tw = TimeWeighter(strategy="exponential", decay_rate=0.0)
        result = tw.compute_weights(snap)
        for i in range(5):
            assert abs(result.weights[i] - 1.0) < 1e-9

    def test_decay_rate_zero(self):
        """With zero decay rate all weights should be approximately 1.0."""
        snap = _make_snapshot([1.0] * 8)
        tw = TimeWeighter(strategy="exponential", decay_rate=0.0)
        result = tw.compute_weights(snap)
        for w in result.weights.values():
            assert abs(w - 1.0) < 1e-9

    def test_high_decay_rate(self):
        """With a very high decay rate, the oldest observation weight should be near zero."""
        snap = _make_snapshot([1.0] * 10)
        tw = TimeWeighter(strategy="exponential", decay_rate=1.0)
        result = tw.compute_weights(snap)
        # The most recent should be 1.0 (normalized).
        assert abs(result.weights[9] - 1.0) < 1e-9
        # The oldest should be very small: exp(-1.0 * 9) ~ 0.000123
        assert result.weights[0] < 0.01

    def test_empty_snapshot(self):
        """An empty snapshot should produce empty weights and effective_window=0."""
        snap = _make_snapshot([])
        tw = TimeWeighter(strategy="exponential", decay_rate=0.1)
        result = tw.compute_weights(snap)
        assert result.weights == {}
        assert result.effective_window == 0

    def test_single_observation(self):
        """A single observation should have weight 1.0."""
        snap = _make_snapshot([42.0])
        tw = TimeWeighter(strategy="exponential", decay_rate=0.1)
        result = tw.compute_weights(snap)
        assert abs(result.weights[0] - 1.0) < 1e-9

    def test_weights_normalized(self):
        """The maximum weight across all observations should be exactly 1.0."""
        snap = _make_snapshot([1.0] * 15)
        tw = TimeWeighter(strategy="exponential", decay_rate=0.2)
        result = tw.compute_weights(snap)
        max_w = max(result.weights.values())
        assert abs(max_w - 1.0) < 1e-9


# ── TestTimeWeighterSlidingWindow ────────────────────────


class TestTimeWeighterSlidingWindow:
    """Tests for the sliding_window strategy of TimeWeighter."""

    def test_only_recent_window(self):
        """Only the last window_size observations should receive weight 1.0; rest get 0.0."""
        snap = _make_snapshot([1.0] * 20)
        tw = TimeWeighter(strategy="sliding_window", window_size=5)
        result = tw.compute_weights(snap)
        # Last 5 observations (iterations 15-19) should have weight 1.0.
        for i in range(15, 20):
            assert abs(result.weights[i] - 1.0) < 1e-9
        # Older observations should have weight 0.0.
        for i in range(15):
            assert abs(result.weights[i] - 0.0) < 1e-9

    def test_window_larger_than_data(self):
        """When window_size exceeds number of observations, all weights should be 1.0."""
        snap = _make_snapshot([1.0] * 10)
        tw = TimeWeighter(strategy="sliding_window", window_size=100)
        result = tw.compute_weights(snap)
        for w in result.weights.values():
            assert abs(w - 1.0) < 1e-9

    def test_effective_window_matches(self):
        """effective_window should equal min(window_size, n_obs)."""
        snap = _make_snapshot([1.0] * 10)
        tw5 = TimeWeighter(strategy="sliding_window", window_size=5)
        result5 = tw5.compute_weights(snap)
        assert result5.effective_window == 5

        tw100 = TimeWeighter(strategy="sliding_window", window_size=100)
        result100 = tw100.compute_weights(snap)
        assert result100.effective_window == 10


# ── TestTimeWeighterLinearDecay ──────────────────────────


class TestTimeWeighterLinearDecay:
    """Tests for the linear_decay strategy of TimeWeighter."""

    def test_linear_decrease(self):
        """Weights should decrease from 1.0 for the most recent observation."""
        snap = _make_snapshot([1.0] * 10)
        tw = TimeWeighter(strategy="linear_decay", decay_rate=0.05)
        result = tw.compute_weights(snap)
        # The most recent (iteration=9) should be the highest.
        assert abs(result.weights[9] - 1.0) < 1e-9
        # Each earlier observation should have a smaller or equal weight.
        for i in range(8, 0, -1):
            assert result.weights[i] >= result.weights[i - 1] - 1e-9

    def test_oldest_clipped_to_zero(self):
        """With a sufficiently high decay rate, very old observations should be clipped to 0.0."""
        snap = _make_snapshot([1.0] * 20)
        tw = TimeWeighter(strategy="linear_decay", decay_rate=0.1)
        result = tw.compute_weights(snap)
        # For obs at iteration 0, raw = max(0, 1 - 0.1*19) = max(0, -0.9) = 0.0
        assert abs(result.weights[0] - 0.0) < 1e-9

    def test_weight_observations_metadata(self):
        """weight_observations should set metadata['time_weight'] on each observation."""
        snap = _make_snapshot([1.0] * 5)
        tw = TimeWeighter(strategy="linear_decay", decay_rate=0.1)
        weighted = tw.weight_observations(snap)
        assert len(weighted) == 5
        for obs in weighted:
            assert "time_weight" in obs.metadata
            assert isinstance(obs.metadata["time_weight"], float)
        # The most recent observation should have the highest time_weight.
        assert weighted[-1].metadata["time_weight"] >= weighted[0].metadata["time_weight"]


# ── TestSeasonalDetector ─────────────────────────────────


class TestSeasonalDetector:
    """Tests for SeasonalDetector autocorrelation-based periodicity detection."""

    def test_no_seasonality(self):
        """A flat series should not be detected as seasonal."""
        snap = _make_snapshot([5.0] * 30)
        det = SeasonalDetector()
        result = det.detect(snap)
        assert result.detected is False

    def test_sinusoidal_pattern(self):
        """A clear sinusoidal signal should be detected as seasonal."""
        n = 40
        kpi = [10.0 + 5.0 * math.sin(2 * math.pi * i / 8.0) for i in range(n)]
        snap = _make_snapshot(kpi)
        det = SeasonalDetector(min_period=3, correlation_threshold=0.5, min_observations=12)
        result = det.detect(snap)
        assert result.detected is True
        assert result.period is not None

    def test_period_detection_accuracy(self):
        """Detected period should be within 1 of the true period."""
        n = 40
        true_period = 8
        kpi = [10.0 + 5.0 * math.sin(2 * math.pi * i / float(true_period)) for i in range(n)]
        snap = _make_snapshot(kpi)
        det = SeasonalDetector(min_period=3, correlation_threshold=0.5, min_observations=12)
        result = det.detect(snap)
        assert result.detected is True
        assert result.period is not None
        assert abs(result.period - true_period) <= 1

    def test_too_few_observations(self):
        """With fewer observations than min_observations, detection should be False."""
        snap = _make_snapshot([1.0, 2.0, 3.0, 4.0, 5.0])
        det = SeasonalDetector(min_observations=12)
        result = det.detect(snap)
        assert result.detected is False

    def test_noisy_seasonal(self):
        """A seasonal signal with deterministic noise should still be detected."""
        n = 40
        kpi = [
            10.0 + 5.0 * math.sin(2 * math.pi * i / 10.0) + 0.5 * ((i * 7) % 11 - 5)
            for i in range(n)
        ]
        snap = _make_snapshot(kpi)
        det = SeasonalDetector(min_period=3, correlation_threshold=0.4, min_observations=12)
        result = det.detect(snap)
        assert result.detected is True

    def test_autocorrelation_at_zero(self):
        """Autocorrelation at lag=0 should be approximately 1.0 for any non-constant series."""
        n = 30
        kpi = [float(i) for i in range(n)]
        snap = _make_snapshot(kpi)
        det = SeasonalDetector()
        # Access the private method to test autocorrelation at lag 0.
        values = [obs.kpi_values["y"] for obs in snap.observations]
        ac0 = det._autocorrelation(values, 0)
        assert abs(ac0 - 1.0) < 1e-9

    def test_empty_series(self):
        """An empty snapshot should produce detected=False."""
        snap = _make_snapshot([])
        det = SeasonalDetector()
        result = det.detect(snap)
        assert result.detected is False


# ── TestNonStationaryAdapter ─────────────────────────────


class TestNonStationaryAdapter:
    """Tests for NonStationaryAdapter.assess()."""

    def test_static_world_no_adaptation(self):
        """Stable KPIs should result in strategy='static' and is_nonstationary=False."""
        snap = _make_snapshot([5.0] * 30)
        adapter = NonStationaryAdapter()
        assessment = adapter.assess(snap)
        assert assessment.recommended_strategy == "static"
        assert assessment.is_nonstationary is False

    def test_drift_triggers_reweight(self):
        """A step change in KPI should trigger 'reweight' or 'sliding_window' strategy."""
        kpi = [5.0] * 20 + [50.0] * 10
        snap = _make_snapshot(kpi)
        adapter = NonStationaryAdapter()
        assessment = adapter.assess(snap)
        assert assessment.recommended_strategy in ("reweight", "sliding_window")

    def test_seasonal_triggers_adjust(self):
        """A sinusoidal KPI series should be detected as seasonal.

        When drift is also detected the strategy may be 'reweight' or
        'sliding_window' rather than 'seasonal_adjust', but the seasonal
        pattern must still be flagged.  When drift is suppressed (via a
        custom drift detector) the strategy should be 'seasonal_adjust'.
        """
        n = 40
        kpi = [10.0 + 5.0 * math.sin(2 * math.pi * i / 8.0) for i in range(n)]
        snap = _make_snapshot(kpi)

        # Use a no-drift stub so the seasonal signal is the only trigger.
        class _NoDrift:
            def detect(self, snapshot):
                from optimization_copilot.drift.detector import DriftReport
                return DriftReport(
                    drift_detected=False, drift_score=0.0, drift_type="none",
                    affected_parameters=[], recommended_action="continue",
                )

        adapter = NonStationaryAdapter(drift_detector=_NoDrift())
        assessment = adapter.assess(snap)
        assert assessment.seasonal_pattern.detected is True
        assert "seasonal" in assessment.recommended_strategy

    def test_recommended_window_no_drift(self):
        """With stable data the recommended window should be at least n//2."""
        n = 30
        snap = _make_snapshot([5.0] * n)
        adapter = NonStationaryAdapter()
        assessment = adapter.assess(snap)
        assert assessment.recommended_window >= n // 2

    def test_recommended_window_with_season(self):
        """When a period of 8 is detected, the recommended window should be >= 16."""
        n = 40
        kpi = [10.0 + 5.0 * math.sin(2 * math.pi * i / 8.0) for i in range(n)]
        snap = _make_snapshot(kpi)
        adapter = NonStationaryAdapter()
        assessment = adapter.assess(snap)
        if assessment.seasonal_pattern.detected and assessment.seasonal_pattern.period:
            assert assessment.recommended_window >= 2 * assessment.seasonal_pattern.period

    def test_is_nonstationary_flag(self):
        """Stable data should yield is_nonstationary=False; drift data should yield True."""
        stable_snap = _make_snapshot([5.0] * 30)
        adapter = NonStationaryAdapter()
        stable_assessment = adapter.assess(stable_snap)
        assert stable_assessment.is_nonstationary is False

        drift_kpi = [5.0] * 20 + [50.0] * 10
        drift_snap = _make_snapshot(drift_kpi)
        drift_assessment = adapter.assess(drift_snap)
        assert drift_assessment.is_nonstationary is True

    def test_assess_empty_snapshot(self):
        """An empty snapshot should produce is_nonstationary=False."""
        snap = _make_snapshot([])
        adapter = NonStationaryAdapter()
        assessment = adapter.assess(snap)
        assert assessment.is_nonstationary is False


# ── TestAdaptDecision ────────────────────────────────────


class TestAdaptDecision:
    """Tests for NonStationaryAdapter.adapt_decision()."""

    def _static_assessment(self):
        """Helper: return a non-stationary assessment with is_nonstationary=False."""
        return NonStationaryAssessment(
            time_weights=TimeWeights(weights={}, effective_window=0, decay_rate=0.1, strategy="exponential"),
            seasonal_pattern=SeasonalPattern(
                detected=False, period=None, autocorrelation=0.0,
                candidate_periods=[], candidate_correlations=[], confidence=0.0,
            ),
            drift_report=None,
            is_nonstationary=False,
            recommended_window=15,
            recommended_strategy="static",
        )

    def _drift_assessment(self, strategy="reweight", drift_score=0.5):
        """Helper: return a non-stationary assessment with drift detected."""

        class _FakeDrift:
            drift_detected = True

            def __init__(self, score):
                self.drift_score = score

        return NonStationaryAssessment(
            time_weights=TimeWeights(weights={}, effective_window=5, decay_rate=0.1, strategy="exponential"),
            seasonal_pattern=SeasonalPattern(
                detected=False, period=None, autocorrelation=0.0,
                candidate_periods=[], candidate_correlations=[], confidence=0.0,
            ),
            drift_report=_FakeDrift(drift_score),
            is_nonstationary=True,
            recommended_window=10,
            recommended_strategy=strategy,
        )

    def test_no_change_when_static(self):
        """When the assessment indicates stationarity, the decision should be returned unchanged."""
        decision = _base_decision()
        adapter = NonStationaryAdapter()
        assessment = self._static_assessment()
        result = adapter.adapt_decision(decision, assessment)
        assert result is decision  # Same object returned.

    def test_exploration_boost_on_drift(self):
        """When drift is detected (drift_score > 0.3), exploration_strength should increase."""
        decision = _base_decision()
        adapter = NonStationaryAdapter()
        assessment = self._drift_assessment(strategy="reweight", drift_score=0.5)
        result = adapter.adapt_decision(decision, assessment)
        assert result.exploration_strength > decision.exploration_strength

    def test_reweighting_strategy_set(self):
        """When strategy is 'reweight', stabilize_spec.reweighting_strategy should be 'recency'."""
        decision = _base_decision()
        adapter = NonStationaryAdapter()
        assessment = self._drift_assessment(strategy="reweight", drift_score=0.4)
        result = adapter.adapt_decision(decision, assessment)
        assert result.stabilize_spec.reweighting_strategy == "recency"

    def test_reason_codes_added(self):
        """When nonstationary, a reason code starting with 'nonstationary:' should be appended."""
        decision = _base_decision()
        adapter = NonStationaryAdapter()
        assessment = self._drift_assessment(strategy="sliding_window", drift_score=0.6)
        result = adapter.adapt_decision(decision, assessment)
        ns_codes = [c for c in result.reason_codes if c.startswith("nonstationary:")]
        assert len(ns_codes) >= 1

    def test_returns_new_decision(self):
        """The original decision should not be mutated; a new object should be returned."""
        decision = _base_decision()
        original_exploration = decision.exploration_strength
        original_codes = list(decision.reason_codes)
        original_reweighting = decision.stabilize_spec.reweighting_strategy

        adapter = NonStationaryAdapter()
        assessment = self._drift_assessment(strategy="reweight", drift_score=0.5)
        result = adapter.adapt_decision(decision, assessment)

        # The original decision should be unchanged.
        assert decision.exploration_strength == original_exploration
        assert decision.reason_codes == original_codes
        assert decision.stabilize_spec.reweighting_strategy == original_reweighting
        # The result should be a different object.
        assert result is not decision


# ── TestIntegration ──────────────────────────────────────


class TestIntegration:
    """End-to-end integration tests running the full pipeline."""

    def test_full_pipeline_stationary(self):
        """Stable data through full assess + adapt pipeline should produce no adaptation."""
        snap = _make_snapshot([5.0] * 30)
        adapter = NonStationaryAdapter()
        assessment = adapter.assess(snap)
        decision = _base_decision()
        result = adapter.adapt_decision(decision, assessment)
        assert assessment.recommended_strategy == "static"
        assert result is decision  # Unchanged.

    def test_full_pipeline_drifting(self):
        """Step-change data through full pipeline should produce adaptation actions."""
        kpi = [5.0] * 20 + [50.0] * 10
        snap = _make_snapshot(kpi)
        adapter = NonStationaryAdapter()
        assessment = adapter.assess(snap)
        decision = _base_decision()
        result = adapter.adapt_decision(decision, assessment)
        assert assessment.is_nonstationary is True
        # Some adaptation should have been applied (new decision object).
        assert result is not decision
        # At least one nonstationary reason code should be present.
        ns_codes = [c for c in result.reason_codes if c.startswith("nonstationary:")]
        assert len(ns_codes) >= 1

    def test_deterministic(self):
        """Running the same input twice should produce identical output."""
        kpi = [5.0] * 20 + [50.0] * 10
        snap = _make_snapshot(kpi)
        adapter = NonStationaryAdapter()

        assessment1 = adapter.assess(snap)
        decision1 = adapter.adapt_decision(_base_decision(), assessment1)

        assessment2 = adapter.assess(snap)
        decision2 = adapter.adapt_decision(_base_decision(), assessment2)

        assert assessment1.is_nonstationary == assessment2.is_nonstationary
        assert assessment1.recommended_strategy == assessment2.recommended_strategy
        assert assessment1.recommended_window == assessment2.recommended_window
        assert abs(decision1.exploration_strength - decision2.exploration_strength) < 1e-9
        assert decision1.reason_codes == decision2.reason_codes
        assert decision1.stabilize_spec.reweighting_strategy == decision2.stabilize_spec.reweighting_strategy
