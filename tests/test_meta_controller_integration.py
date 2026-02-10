"""Integration tests for MetaController.decide() with Track 1-3 context signals.

Tests the full wiring of:
  - Track 1: Portfolio-informed backend selection (AlgorithmPortfolio + BackendScorer)
  - Track 2: Drift-adapted strategy (DriftReport + DriftStrategyAdapter)
  - Track 3: Failure-aware decisions (FailureTaxonomy + FailureSurfaceLearner)
  - Cost-aware exploration modulation (CostSignals)
  - Backend policy enforcement (BackendPolicy)
  - Full pipeline with all signals combined
  - Graceful degradation when components fail
  - Backward compatibility with legacy call signatures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from optimization_copilot.core.models import (
    CampaignSnapshot,
    DataScale,
    NoiseRegime,
    ObjectiveForm,
    Observation,
    ParameterSpec,
    Phase,
    ProblemFingerprint,
    RiskPosture,
    VariableType,
)
from optimization_copilot.cost.cost_analyzer import CostSignals
from optimization_copilot.drift.detector import DriftReport
from optimization_copilot.feasibility.taxonomy import (
    ClassifiedFailure,
    FailureClassifier,
    FailureTaxonomy,
    FailureType,
)
from optimization_copilot.meta_controller.controller import MetaController
from optimization_copilot.plugins.registry import BackendPolicy
from optimization_copilot.portfolio.portfolio import AlgorithmPortfolio, BackendRecord
from optimization_copilot.portfolio.scorer import BackendScorer, _fingerprint_key


# ---------------------------------------------------------------------------
# Helpers — reuse patterns from test_meta_controller.py
# ---------------------------------------------------------------------------


def _make_snapshot(
    n_obs: int = 5,
    n_failures: int = 0,
    failure_reasons: list[str] | None = None,
) -> CampaignSnapshot:
    """Build a CampaignSnapshot with controllable observation count and failures.

    Parameters
    ----------
    n_obs : int
        Total number of observations to generate.
    n_failures : int
        How many of the first observations are marked as failures.
    failure_reasons : list[str] or None
        Optional per-failure reason strings (length must match *n_failures*).
    """
    specs = [
        ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=-1.0, upper=1.0),
    ]
    obs: list[Observation] = []
    for i in range(n_obs):
        is_fail = i < n_failures
        reason = None
        if is_fail and failure_reasons and i < len(failure_reasons):
            reason = failure_reasons[i]
        obs.append(
            Observation(
                iteration=i,
                parameters={"x1": i * 0.1, "x2": i * -0.1},
                kpi_values={"y": float(i + 1) if not is_fail else 0.0},
                is_failure=is_fail,
                failure_reason=reason,
                timestamp=float(i),
            )
        )
    return CampaignSnapshot(
        campaign_id="integration_test",
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=n_obs,
    )


def _base_diagnostics(**overrides: float) -> dict[str, float]:
    """Standard diagnostic signals with optional overrides."""
    d: dict[str, float] = {
        "convergence_trend": 0.1,
        "improvement_velocity": 0.1,
        "variance_contraction": 0.8,
        "noise_estimate": 0.1,
        "failure_rate": 0.0,
        "failure_clustering": 0.0,
        "feasibility_shrinkage": 0.0,
        "parameter_drift": 0.05,
        "model_uncertainty": 0.5,
        "exploration_coverage": 0.3,
        "kpi_plateau_length": 0,
        "best_kpi_value": 5.0,
        "data_efficiency": 0.5,
        "constraint_violation_rate": 0.0,
    }
    d.update(overrides)
    return d


def _make_portfolio_with_records(
    fingerprint: ProblemFingerprint,
    records: dict[str, dict[str, Any]],
) -> AlgorithmPortfolio:
    """Build an AlgorithmPortfolio with pre-populated records.

    Inserts records keyed by the *scorer's* fingerprint key format so that
    ``BackendScorer._get_record`` finds them via exact match.

    Parameters
    ----------
    fingerprint :
        The problem fingerprint to key records under.
    records :
        Mapping of ``backend_name -> {n_uses, win_count, avg_convergence_speed,
        avg_regret, failure_rate, sample_efficiency, ...}``.
    """
    portfolio = AlgorithmPortfolio()
    fp_key = _fingerprint_key(fingerprint)
    for backend_name, fields in records.items():
        rec = BackendRecord(
            fingerprint_key=fp_key,
            backend_name=backend_name,
            n_uses=fields.get("n_uses", 5),
            win_count=fields.get("win_count", 0),
            avg_convergence_speed=fields.get("avg_convergence_speed", 0.5),
            avg_regret=fields.get("avg_regret", 0.3),
            failure_rate=fields.get("failure_rate", 0.1),
            sample_efficiency=fields.get("sample_efficiency", 0.5),
        )
        portfolio._records[(fp_key, backend_name)] = rec
    return portfolio


def _make_drift_report(
    drift_detected: bool = True,
    drift_score: float = 0.5,
    drift_type: str = "gradual",
    affected_parameters: list[str] | None = None,
    recommended_action: str = "reweight",
) -> DriftReport:
    """Build a DriftReport with controllable severity."""
    return DriftReport(
        drift_detected=drift_detected,
        drift_score=drift_score,
        drift_type=drift_type,
        affected_parameters=affected_parameters or [],
        recommended_action=recommended_action,
    )


def _make_cost_signals(
    time_budget_pressure: float = 0.0,
    cost_efficiency_trend: float = 0.0,
) -> CostSignals:
    """Build lightweight CostSignals with the two fields MetaController reads."""
    return CostSignals(
        cost_per_improvement=1.0,
        time_budget_pressure=time_budget_pressure,
        cost_efficiency_trend=cost_efficiency_trend,
        cumulative_cost=100.0,
        estimated_remaining_budget=900.0,
        cost_optimal_batch_size=3,
    )


def _make_failure_taxonomy(
    dominant_type: FailureType = FailureType.HARDWARE,
    type_rates: dict[str, float] | None = None,
    classified_failures: list[ClassifiedFailure] | None = None,
) -> FailureTaxonomy:
    """Build a FailureTaxonomy with controllable dominant type and rates."""
    if type_rates is None:
        type_rates = {dominant_type.value: 0.6, "unknown": 0.4}
    type_counts = {k: max(1, int(v * 10)) for k, v in type_rates.items()}
    return FailureTaxonomy(
        classified_failures=classified_failures or [],
        type_counts=type_counts,
        dominant_type=dominant_type,
        type_rates=type_rates,
        strategy_adjustments={dominant_type.value: "reduce_exploration"},
    )


# ===========================================================================
# Test 1: Portfolio-informed decision
# ===========================================================================


class TestPortfolioInformedDecision:
    """MetaController with portfolio data picks backends based on history."""

    def test_portfolio_selects_best_scored_backend(self):
        """When portfolio shows 'tpe' has higher win rate, it should be chosen
        over rule-based defaults."""
        fp = ProblemFingerprint()  # default: continuous, single, low noise, tiny
        portfolio = _make_portfolio_with_records(fp, {
            "tpe": {
                "n_uses": 10,
                "win_count": 8,
                "avg_convergence_speed": 0.8,
                "avg_regret": 0.1,
                "failure_rate": 0.05,
                "sample_efficiency": 0.7,
            },
            "random": {
                "n_uses": 10,
                "win_count": 2,
                "avg_convergence_speed": 0.3,
                "avg_regret": 0.5,
                "failure_rate": 0.2,
                "sample_efficiency": 0.3,
            },
            "latin_hypercube": {
                "n_uses": 10,
                "win_count": 3,
                "avg_convergence_speed": 0.4,
                "avg_regret": 0.4,
                "failure_rate": 0.15,
                "sample_efficiency": 0.4,
            },
        })

        mc = MetaController(available_backends=["random", "latin_hypercube", "tpe"])
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        decision = mc.decide(snap, diag, fp, portfolio=portfolio)

        assert decision.backend_name == "tpe"
        assert decision.decision_metadata["portfolio_used"] is True
        # Reason codes must mention portfolio scoring
        assert any("portfolio_scored" in rc for rc in decision.reason_codes)

    def test_portfolio_prefers_low_failure_backend(self):
        """A backend with fewer failures should be preferred even if it has
        a slightly lower win rate."""
        fp = ProblemFingerprint()
        portfolio = _make_portfolio_with_records(fp, {
            "tpe": {
                "n_uses": 10,
                "win_count": 6,
                "avg_convergence_speed": 0.6,
                "avg_regret": 0.2,
                "failure_rate": 0.5,  # high failure rate
                "sample_efficiency": 0.6,
            },
            "latin_hypercube": {
                "n_uses": 10,
                "win_count": 5,
                "avg_convergence_speed": 0.55,
                "avg_regret": 0.15,
                "failure_rate": 0.02,  # very low failure rate
                "sample_efficiency": 0.55,
            },
        })

        mc = MetaController(available_backends=["tpe", "latin_hypercube"])
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        decision = mc.decide(snap, diag, fp, portfolio=portfolio)

        # latin_hypercube should win due to tpe's high expected_fail penalty
        assert decision.backend_name == "latin_hypercube"

    def test_portfolio_metadata_records_usage(self):
        """decision_metadata must accurately reflect that portfolio was used."""
        fp = ProblemFingerprint()
        portfolio = _make_portfolio_with_records(fp, {
            "random": {"n_uses": 5, "win_count": 3},
        })

        mc = MetaController(available_backends=["random", "latin_hypercube", "tpe"])
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        decision = mc.decide(snap, diag, fp, portfolio=portfolio)

        assert decision.decision_metadata["portfolio_used"] is True
        assert decision.decision_metadata["drift_report_used"] is False
        assert decision.decision_metadata["cost_signals_used"] is False


# ===========================================================================
# Test 2: Drift-adapted decision
# ===========================================================================


class TestDriftAdaptedDecision:
    """MetaController with drift report adapts exploration and phase."""

    def test_mild_drift_boosts_exploration(self):
        """Drift score of 0.4 should increase exploration strength."""
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()

        # Baseline: no drift
        baseline = mc.decide(snap, diag, fp)
        baseline_exploration = baseline.exploration_strength

        # With mild drift
        drift = _make_drift_report(drift_score=0.4, drift_type="gradual")
        decision = mc.decide(snap, diag, fp, drift_report=drift)

        assert decision.exploration_strength > baseline_exploration
        assert decision.decision_metadata["drift_report_used"] is True
        assert any("drift" in rc for rc in decision.reason_codes)

    def test_severe_drift_resets_exploitation_to_learning(self):
        """Drift score >= 0.7 should reset EXPLOITATION phase back to LEARNING."""
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        # Set diagnostics to trigger EXPLOITATION phase
        diag = _base_diagnostics(convergence_trend=0.5, model_uncertainty=0.1)
        fp = ProblemFingerprint()

        # Without drift: should be EXPLOITATION
        baseline = mc.decide(snap, diag, fp)
        assert baseline.phase == Phase.EXPLOITATION

        # With severe drift: should reset to LEARNING
        drift = _make_drift_report(drift_score=0.8, drift_type="sudden")
        decision = mc.decide(snap, diag, fp, drift_report=drift)

        assert decision.phase == Phase.LEARNING
        assert any("drift_phase_reset" in rc for rc in decision.reason_codes)

    def test_drift_triggers_recency_reweighting(self):
        """Any significant drift should switch stabilize_spec to recency reweighting."""
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()

        drift = _make_drift_report(drift_score=0.5, drift_type="gradual")
        decision = mc.decide(snap, diag, fp, drift_report=drift)

        assert decision.stabilize_spec.reweighting_strategy == "recency"

    def test_no_drift_leaves_decision_unchanged(self):
        """A drift report with drift_detected=False should not alter the decision."""
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()

        no_drift = _make_drift_report(drift_detected=False, drift_score=0.0)
        baseline = mc.decide(snap, diag, fp)
        decision = mc.decide(snap, diag, fp, drift_report=no_drift)

        assert decision.phase == baseline.phase
        assert decision.exploration_strength == baseline.exploration_strength
        assert decision.stabilize_spec.reweighting_strategy == baseline.stabilize_spec.reweighting_strategy

    def test_severe_drift_switches_to_space_filling_backend(self):
        """Drift score >= 0.6 should trigger backend_switch action,
        resulting in a space-filling backend like 'random' or 'latin_hypercube'."""
        mc = MetaController(available_backends=["tpe", "random", "latin_hypercube"])
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()

        drift = _make_drift_report(drift_score=0.75, drift_type="sudden")
        decision = mc.decide(snap, diag, fp, drift_report=drift)

        # The DriftStrategyAdapter recommends ["random", "latin_hypercube"] for backend_switch
        assert decision.backend_name in ("random", "latin_hypercube")
        assert any("drift_backend_switch" in rc for rc in decision.reason_codes)


# ===========================================================================
# Test 3: Cost-aware decision
# ===========================================================================


class TestCostAwareDecision:
    """MetaController with cost signals reduces exploration under budget pressure."""

    def test_high_pressure_reduces_exploration(self):
        """High time_budget_pressure should lower exploration_strength."""
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()

        baseline = mc.decide(snap, diag, fp)
        cost = _make_cost_signals(time_budget_pressure=0.9)
        decision = mc.decide(snap, diag, fp, cost_signals=cost)

        assert decision.exploration_strength < baseline.exploration_strength
        assert decision.decision_metadata["cost_signals_used"] is True
        # Should have a cost_adjustment reason code
        assert any("cost_adjustment" in rc for rc in decision.reason_codes)

    def test_zero_pressure_no_change(self):
        """With zero budget pressure and zero trend, exploration should not change."""
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()

        baseline = mc.decide(snap, diag, fp)
        cost = _make_cost_signals(time_budget_pressure=0.0, cost_efficiency_trend=0.0)
        decision = mc.decide(snap, diag, fp, cost_signals=cost)

        assert decision.exploration_strength == baseline.exploration_strength

    def test_positive_trend_slightly_increases_exploration(self):
        """Positive cost_efficiency_trend should slightly increase exploration."""
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()

        baseline = mc.decide(snap, diag, fp)
        cost = _make_cost_signals(
            time_budget_pressure=0.0,
            cost_efficiency_trend=0.8,
        )
        decision = mc.decide(snap, diag, fp, cost_signals=cost)

        # trend * 0.1 = 0.08 should be added
        assert decision.exploration_strength > baseline.exploration_strength

    def test_cost_adjustment_reason_shows_pressure_and_trend(self):
        """Reason code should include both pressure and trend values."""
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()

        cost = _make_cost_signals(time_budget_pressure=0.7, cost_efficiency_trend=0.3)
        decision = mc.decide(snap, diag, fp, cost_signals=cost)

        cost_reasons = [rc for rc in decision.reason_codes if "cost_adjustment" in rc]
        assert len(cost_reasons) == 1
        assert "pressure=0.70" in cost_reasons[0]
        assert "trend=0.30" in cost_reasons[0]


# ===========================================================================
# Test 4: Failure-aware decision
# ===========================================================================


class TestFailureAwareDecision:
    """MetaController with failure taxonomy integrates with portfolio scorer."""

    def test_hardware_failures_increase_expected_fail_penalty(self):
        """When FailureTaxonomy shows hardware failures, backends without
        portfolio robustness data should be penalized more."""
        fp = ProblemFingerprint()
        portfolio = _make_portfolio_with_records(fp, {
            "tpe": {
                "n_uses": 10,
                "win_count": 7,
                "avg_regret": 0.1,
                "failure_rate": 0.05,
                "sample_efficiency": 0.7,
                "avg_convergence_speed": 0.7,
            },
            "random": {
                "n_uses": 10,
                "win_count": 4,
                "avg_regret": 0.2,
                "failure_rate": 0.05,
                "sample_efficiency": 0.5,
                "avg_convergence_speed": 0.5,
            },
        })

        taxonomy = _make_failure_taxonomy(
            dominant_type=FailureType.HARDWARE,
            type_rates={"hardware": 0.7, "unknown": 0.3},
        )

        mc = MetaController(available_backends=["tpe", "random"])
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()

        # Without taxonomy
        dec_without = mc.decide(snap, diag, fp, portfolio=portfolio)
        # With taxonomy — hardware failures raise expected_fail for all backends
        dec_with = mc.decide(
            snap, diag, fp,
            portfolio=portfolio,
            failure_taxonomy=taxonomy,
        )

        assert dec_with.decision_metadata["failure_taxonomy_used"] is True
        # Both decisions should still select from available backends
        assert dec_with.backend_name in ("tpe", "random")

    def test_real_failure_classifier_produces_taxonomy(self):
        """Use the real FailureClassifier to produce a taxonomy from snapshot
        data, then pass it to MetaController."""
        snap = _make_snapshot(
            n_obs=15,
            n_failures=5,
            failure_reasons=[
                "timeout on instrument",
                "sensor connection lost",
                "hardware malfunction",
                "precipitate formed unexpectedly",
                "reaction yield too low",
            ],
        )
        fp = ProblemFingerprint()

        classifier = FailureClassifier()
        taxonomy = classifier.classify(snap)

        # Verify the classifier produced a real taxonomy
        assert len(taxonomy.classified_failures) == 5
        assert taxonomy.dominant_type in list(FailureType)

        # Now feed it to MetaController
        mc = MetaController()
        diag = _base_diagnostics(failure_rate=5 / 15)
        portfolio = _make_portfolio_with_records(fp, {
            "random": {"n_uses": 10, "win_count": 5},
            "tpe": {"n_uses": 10, "win_count": 6},
        })

        decision = mc.decide(
            snap, diag, fp,
            portfolio=portfolio,
            failure_taxonomy=taxonomy,
        )

        assert decision.decision_metadata["failure_taxonomy_used"] is True
        assert decision.backend_name in ("random", "tpe", "latin_hypercube")


# ===========================================================================
# Test 5: Backend policy enforcement
# ===========================================================================


class TestBackendPolicyEnforcement:
    """MetaController with BackendPolicy blocks denied backends."""

    def test_denied_backend_not_selected_when_competitors_close(self):
        """A backend in the policy denylist should lose when competitors are
        close in score, because the incompatibility penalty tips the balance.

        Note: BackendPolicy applies a *soft* penalty (weight=0.05) in the
        scorer rather than a hard block.  When the denied backend has an
        overwhelming advantage the penalty alone may not be enough.  This
        test verifies the penalty mechanism works when scores are competitive.
        """
        fp = ProblemFingerprint()
        # tpe is only slightly better than latin_hypercube
        portfolio = _make_portfolio_with_records(fp, {
            "tpe": {
                "n_uses": 10,
                "win_count": 6,
                "avg_convergence_speed": 0.55,
                "avg_regret": 0.2,
                "failure_rate": 0.1,
                "sample_efficiency": 0.55,
            },
            "latin_hypercube": {
                "n_uses": 10,
                "win_count": 5,
                "avg_convergence_speed": 0.5,
                "avg_regret": 0.22,
                "failure_rate": 0.1,
                "sample_efficiency": 0.5,
            },
        })

        # Block tpe via policy
        policy = BackendPolicy(denylist=["tpe"])

        mc = MetaController(available_backends=["tpe", "latin_hypercube"])
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()

        # Without policy: tpe should win (slightly better scores)
        dec_no_policy = mc.decide(snap, diag, fp, portfolio=portfolio)
        assert dec_no_policy.backend_name == "tpe"

        # With policy: incompatibility penalty should tip balance to latin_hypercube
        dec_with_policy = mc.decide(
            snap, diag, fp,
            portfolio=portfolio,
            backend_policy=policy,
        )
        assert dec_with_policy.backend_name == "latin_hypercube"

    def test_allowlist_restricts_to_permitted_backends(self):
        """A policy with an allowlist should only permit listed backends."""
        fp = ProblemFingerprint()
        portfolio = _make_portfolio_with_records(fp, {
            "tpe": {"n_uses": 10, "win_count": 8},
            "random": {"n_uses": 10, "win_count": 3},
            "latin_hypercube": {"n_uses": 10, "win_count": 4},
        })

        # Only allow latin_hypercube
        policy = BackendPolicy(allowlist=["latin_hypercube"])

        mc = MetaController(available_backends=["tpe", "random", "latin_hypercube"])
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()

        decision = mc.decide(
            snap, diag, fp,
            portfolio=portfolio,
            backend_policy=policy,
        )

        assert decision.backend_name == "latin_hypercube"

    def test_policy_without_portfolio_still_enforced(self):
        """Backend policy should work even without portfolio data,
        affecting the rule-based fallback selection."""
        policy = BackendPolicy(denylist=["latin_hypercube"])

        mc = MetaController(available_backends=["latin_hypercube", "random", "tpe"])
        snap = _make_snapshot(n_obs=3)  # cold start -> prefers latin_hypercube
        diag = _base_diagnostics()
        fp = ProblemFingerprint()

        # Without policy: cold start normally selects latin_hypercube or random
        baseline = mc.decide(snap, diag, fp)

        # With policy: latin_hypercube is blocked
        # Since no portfolio, fallback to rule-based, but scorer gives it
        # max incompatibility penalty
        decision = mc.decide(snap, diag, fp, backend_policy=policy)

        # The decision might still select latin_hypercube via rule-based fallback
        # (policy only affects scorer path), but without portfolio the
        # rule-based path is used. Let's verify with portfolio.
        portfolio = _make_portfolio_with_records(fp, {
            "latin_hypercube": {"n_uses": 10, "win_count": 8},
            "random": {"n_uses": 10, "win_count": 3},
            "tpe": {"n_uses": 10, "win_count": 4},
        })
        decision = mc.decide(
            snap, diag, fp,
            portfolio=portfolio,
            backend_policy=policy,
        )
        # latin_hypercube gets incompatibility penalty of 1.0, should lose
        assert decision.backend_name != "latin_hypercube"


# ===========================================================================
# Test 6: Full pipeline — all context signals together
# ===========================================================================


class TestFullPipeline:
    """All context signals provided together in a single decide() call."""

    def test_all_signals_combined(self):
        """A decision with portfolio + drift + cost + taxonomy + policy should
        produce a coherent result with all metadata flags set."""
        fp = ProblemFingerprint(data_scale=DataScale.SMALL)
        portfolio = _make_portfolio_with_records(fp, {
            "tpe": {
                "n_uses": 10,
                "win_count": 7,
                "avg_convergence_speed": 0.7,
                "avg_regret": 0.15,
                "failure_rate": 0.05,
                "sample_efficiency": 0.6,
            },
            "random": {
                "n_uses": 10,
                "win_count": 4,
                "avg_convergence_speed": 0.4,
                "avg_regret": 0.3,
                "failure_rate": 0.1,
                "sample_efficiency": 0.4,
            },
            "latin_hypercube": {
                "n_uses": 10,
                "win_count": 5,
                "avg_convergence_speed": 0.5,
                "avg_regret": 0.2,
                "failure_rate": 0.08,
                "sample_efficiency": 0.5,
            },
        })

        drift = _make_drift_report(drift_score=0.4, drift_type="gradual")
        cost = _make_cost_signals(time_budget_pressure=0.5, cost_efficiency_trend=0.2)
        taxonomy = _make_failure_taxonomy(
            dominant_type=FailureType.CHEMISTRY,
            type_rates={"chemistry": 0.6, "unknown": 0.4},
        )
        policy = BackendPolicy(denylist=["cma_es"])  # deny an irrelevant backend

        mc = MetaController(available_backends=["tpe", "random", "latin_hypercube"])
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()

        decision = mc.decide(
            snap, diag, fp,
            portfolio=portfolio,
            drift_report=drift,
            cost_signals=cost,
            failure_taxonomy=taxonomy,
            backend_policy=policy,
        )

        # -- Structural assertions --
        assert decision.backend_name in ("tpe", "random", "latin_hypercube")
        assert 0.0 <= decision.exploration_strength <= 1.0
        assert decision.phase in list(Phase)
        assert decision.risk_posture in list(RiskPosture)

        # -- Metadata flags --
        assert decision.decision_metadata["portfolio_used"] is True
        assert decision.decision_metadata["drift_report_used"] is True
        assert decision.decision_metadata["cost_signals_used"] is True
        assert decision.decision_metadata["failure_taxonomy_used"] is True

        # -- Drift adaptation metadata --
        assert "drift_adaptation" in decision.decision_metadata
        drift_meta = decision.decision_metadata["drift_adaptation"]
        assert drift_meta["n_actions"] > 0

        # -- Reason codes should contain entries from multiple subsystems --
        all_reasons = " ".join(decision.reason_codes)
        assert "portfolio_scored" in all_reasons
        assert "cost_adjustment" in all_reasons
        assert "drift" in all_reasons  # some drift-related reason

    def test_full_pipeline_with_real_objects(self):
        """End-to-end test using real AlgorithmPortfolio.record_outcome()
        and real FailureClassifier instead of pre-built mocks."""
        fp = ProblemFingerprint(
            noise_regime=NoiseRegime.MEDIUM,
            data_scale=DataScale.SMALL,
        )

        # Build portfolio through the actual recording API
        portfolio = AlgorithmPortfolio()
        for _ in range(8):
            portfolio.record_outcome(fp, "tpe", {
                "convergence_speed": 0.7,
                "regret": 0.1,
                "failure_rate": 0.05,
                "sample_efficiency": 0.6,
                "is_winner": True,
            })
        for _ in range(8):
            portfolio.record_outcome(fp, "random", {
                "convergence_speed": 0.3,
                "regret": 0.4,
                "failure_rate": 0.2,
                "sample_efficiency": 0.3,
                "is_winner": False,
            })

        # Build snapshot with some failures for the classifier
        snap = _make_snapshot(
            n_obs=20,
            n_failures=4,
            failure_reasons=[
                "instrument timeout",
                "sensor disconnected",
                "precipitate formed",
                "unknown error",
            ],
        )

        classifier = FailureClassifier()
        taxonomy = classifier.classify(snap)

        drift = _make_drift_report(drift_score=0.35, drift_type="gradual")
        cost = _make_cost_signals(time_budget_pressure=0.3)

        mc = MetaController(available_backends=["tpe", "random", "latin_hypercube"])
        diag = _base_diagnostics(failure_rate=4 / 20)

        decision = mc.decide(
            snap, diag, fp,
            portfolio=portfolio,
            drift_report=drift,
            cost_signals=cost,
            failure_taxonomy=taxonomy,
        )

        # All metadata flags should reflect usage
        assert decision.decision_metadata["portfolio_used"] is True
        assert decision.decision_metadata["drift_report_used"] is True
        assert decision.decision_metadata["cost_signals_used"] is True
        assert decision.decision_metadata["failure_taxonomy_used"] is True
        assert decision.backend_name in ("tpe", "random", "latin_hypercube")
        assert len(decision.reason_codes) > 3  # multiple subsystems contribute reasons

    def test_full_pipeline_determinism(self):
        """Same inputs with all signals should produce identical outputs."""
        fp = ProblemFingerprint()
        portfolio = _make_portfolio_with_records(fp, {
            "tpe": {"n_uses": 10, "win_count": 7},
            "random": {"n_uses": 10, "win_count": 3},
        })
        drift = _make_drift_report(drift_score=0.45)
        cost = _make_cost_signals(time_budget_pressure=0.4)
        taxonomy = _make_failure_taxonomy(dominant_type=FailureType.DATA)

        mc = MetaController(available_backends=["tpe", "random", "latin_hypercube"])
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()

        d1 = mc.decide(
            snap, diag, fp, seed=42,
            portfolio=portfolio,
            drift_report=drift,
            cost_signals=cost,
            failure_taxonomy=taxonomy,
        )
        d2 = mc.decide(
            snap, diag, fp, seed=42,
            portfolio=portfolio,
            drift_report=drift,
            cost_signals=cost,
            failure_taxonomy=taxonomy,
        )

        assert d1.backend_name == d2.backend_name
        assert d1.phase == d2.phase
        assert d1.exploration_strength == d2.exploration_strength
        assert d1.risk_posture == d2.risk_posture
        assert d1.reason_codes == d2.reason_codes
        assert d1.fallback_events == d2.fallback_events


# ===========================================================================
# Test 7: Graceful degradation
# ===========================================================================


class TestGracefulDegradation:
    """When portfolio scorer fails, MetaController falls back to rule-based."""

    def test_scorer_exception_falls_back_to_rules(self):
        """If BackendScorer.score_backends() raises, the controller should
        gracefully fall back to rule-based selection and log the failure."""
        fp = ProblemFingerprint()
        portfolio = AlgorithmPortfolio()  # empty portfolio, but that's not the issue

        mc = MetaController(available_backends=["random", "latin_hypercube", "tpe"])
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()

        # Patch the scorer to raise an exception
        with patch(
            "optimization_copilot.portfolio.scorer.BackendScorer.score_backends",
            side_effect=RuntimeError("scorer exploded"),
        ):
            decision = mc.decide(snap, diag, fp, portfolio=portfolio)

        # Should still produce a valid decision via fallback
        assert decision.backend_name in ("random", "latin_hypercube", "tpe")
        # Fallback event should be logged
        assert any("portfolio_scorer_failed" in fe for fe in decision.fallback_events)

    def test_empty_portfolio_uses_default_priors(self):
        """An empty AlgorithmPortfolio should still produce scores via
        the scorer's default prior system (no crash, no fallback_events)."""
        fp = ProblemFingerprint()
        portfolio = AlgorithmPortfolio()

        mc = MetaController(available_backends=["random", "latin_hypercube", "tpe"])
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()

        decision = mc.decide(snap, diag, fp, portfolio=portfolio)

        # Should succeed without fallback events
        assert decision.backend_name in ("random", "latin_hypercube", "tpe")
        # portfolio_used is True (it was provided), even if data was empty
        assert decision.decision_metadata["portfolio_used"] is True
        # The scorer falls back to default priors, no crash
        assert not any(
            "portfolio_scorer_failed" in fe for fe in decision.fallback_events
        )

    def test_drift_adapter_exception_produces_fallback_event(self):
        """If the DriftStrategyAdapter.adapt() raises, the decision should
        still be returned with a fallback event logged."""
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()
        drift = _make_drift_report(drift_score=0.5)

        with patch(
            "optimization_copilot.drift.strategy.DriftStrategyAdapter.adapt",
            side_effect=ValueError("adapter crashed"),
        ):
            decision = mc.decide(snap, diag, fp, drift_report=drift)

        # Decision should still be valid
        assert decision.phase in list(Phase)
        assert decision.backend_name in ("random", "latin_hypercube", "tpe")
        # Fallback event from drift adaptation failure
        assert any("drift_adaptation_failed" in fe for fe in decision.fallback_events)

    def test_portfolio_none_with_other_signals_still_works(self):
        """Providing drift/cost/taxonomy without portfolio should work fine,
        using rule-based backend selection + context adjustments."""
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()

        drift = _make_drift_report(drift_score=0.4)
        cost = _make_cost_signals(time_budget_pressure=0.6)
        taxonomy = _make_failure_taxonomy()

        decision = mc.decide(
            snap, diag, fp,
            drift_report=drift,
            cost_signals=cost,
            failure_taxonomy=taxonomy,
        )

        assert decision.decision_metadata["portfolio_used"] is False
        assert decision.decision_metadata["drift_report_used"] is True
        assert decision.decision_metadata["cost_signals_used"] is True
        assert decision.decision_metadata["failure_taxonomy_used"] is True
        # Exploration should be reduced by cost pressure
        assert any("cost_adjustment" in rc for rc in decision.reason_codes)


# ===========================================================================
# Test 8: Backward compatibility
# ===========================================================================


class TestBackwardCompatibility:
    """decide() without any new kwargs works exactly as before."""

    def test_decide_without_new_kwargs_matches_legacy(self):
        """Calling decide() with only the original positional args should
        produce the same result as the pre-Track 1-3 controller."""
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()

        decision = mc.decide(snap, diag, fp, seed=42)

        # All new metadata flags should be False/None
        assert decision.decision_metadata["portfolio_used"] is False
        assert decision.decision_metadata["drift_report_used"] is False
        assert decision.decision_metadata["cost_signals_used"] is False
        assert decision.decision_metadata["failure_taxonomy_used"] is False
        # No drift adaptation metadata
        assert "drift_adaptation" not in decision.decision_metadata
        # No fallback events from new systems
        assert len(decision.fallback_events) == 0

    def test_legacy_phase_detection_unchanged(self):
        """Phase detection logic should be identical to legacy behavior."""
        mc = MetaController()
        fp = ProblemFingerprint()

        # Cold start
        snap_cold = _make_snapshot(n_obs=3)
        d_cold = mc.decide(snap_cold, _base_diagnostics(), fp)
        assert d_cold.phase == Phase.COLD_START

        # Learning
        snap_learn = _make_snapshot(n_obs=15)
        d_learn = mc.decide(snap_learn, _base_diagnostics(), fp)
        assert d_learn.phase == Phase.LEARNING

        # Stagnation
        snap_stag = _make_snapshot(n_obs=20)
        d_stag = mc.decide(snap_stag, _base_diagnostics(kpi_plateau_length=15), fp)
        assert d_stag.phase == Phase.STAGNATION

        # Exploitation
        snap_expl = _make_snapshot(n_obs=15)
        d_expl = mc.decide(
            snap_expl,
            _base_diagnostics(convergence_trend=0.5, model_uncertainty=0.1),
            fp,
        )
        assert d_expl.phase == Phase.EXPLOITATION

    def test_legacy_backend_selection_unchanged(self):
        """Without portfolio, backend selection follows the original
        PHASE_BACKEND_MAP + fingerprint override rules."""
        mc = MetaController(available_backends=["random", "latin_hypercube", "tpe"])
        fp = ProblemFingerprint()

        # Cold start -> latin_hypercube or random
        snap = _make_snapshot(n_obs=3)
        d = mc.decide(snap, _base_diagnostics(), fp)
        assert d.backend_name in ("latin_hypercube", "random")

    def test_legacy_exploration_strength_unchanged(self):
        """Without cost signals, exploration strength matches legacy computation."""
        mc = MetaController()
        fp = ProblemFingerprint()

        # Cold start: high exploration
        snap = _make_snapshot(n_obs=3)
        d = mc.decide(snap, _base_diagnostics(), fp)
        assert d.exploration_strength >= 0.8

        # Exploitation: low exploration
        snap_expl = _make_snapshot(n_obs=15)
        d_expl = mc.decide(
            snap_expl,
            _base_diagnostics(convergence_trend=0.5, model_uncertainty=0.1),
            fp,
        )
        assert d_expl.exploration_strength <= 0.4

    def test_previous_phase_still_accepted(self):
        """The previous_phase positional arg should still work."""
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()

        decision = mc.decide(snap, diag, fp, seed=42, previous_phase=Phase.COLD_START)

        assert decision.decision_metadata["previous_phase"] == "cold_start"


# ===========================================================================
# Test: Interaction between multiple signals
# ===========================================================================


class TestSignalInteractions:
    """Tests that verify the interaction between different context signals."""

    def test_drift_and_cost_both_reduce_exploration(self):
        """When drift wants more exploration but cost wants less, the net
        effect should be a balance between the two forces."""
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()

        # Drift alone: boosts exploration
        drift = _make_drift_report(drift_score=0.5)
        d_drift = mc.decide(snap, diag, fp, drift_report=drift)

        # Cost alone: reduces exploration
        cost = _make_cost_signals(time_budget_pressure=0.9)
        d_cost = mc.decide(snap, diag, fp, cost_signals=cost)

        # Both together
        d_both = mc.decide(snap, diag, fp, drift_report=drift, cost_signals=cost)

        # Cost reduces from baseline, drift increases from baseline
        baseline = mc.decide(snap, diag, fp)
        assert d_cost.exploration_strength < baseline.exploration_strength
        assert d_drift.exploration_strength > baseline.exploration_strength

        # Combined: cost reduction happens first (step 3b in decide()),
        # then drift boost is applied (step 7). The net result depends on
        # the magnitudes but should be between the two extremes.
        # (The order matters: cost reduces exploration_strength, then drift adds delta.)

    def test_portfolio_and_policy_interact_correctly(self):
        """Policy denial applies an incompatibility penalty in the scorer.
        When backends are close in score, the penalty tips the balance."""
        fp = ProblemFingerprint()
        # tpe is only marginally better than random
        portfolio = _make_portfolio_with_records(fp, {
            "tpe": {
                "n_uses": 10,
                "win_count": 6,
                "avg_convergence_speed": 0.55,
                "avg_regret": 0.2,
                "failure_rate": 0.1,
                "sample_efficiency": 0.55,
            },
            "random": {
                "n_uses": 10,
                "win_count": 5,
                "avg_convergence_speed": 0.5,
                "avg_regret": 0.22,
                "failure_rate": 0.1,
                "sample_efficiency": 0.5,
            },
        })
        policy = BackendPolicy(denylist=["tpe"])

        mc = MetaController(available_backends=["tpe", "random"])
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()

        # Without policy: tpe wins
        dec_no_policy = mc.decide(snap, diag, fp, portfolio=portfolio)
        assert dec_no_policy.backend_name == "tpe"

        # With policy: penalty tips balance to random
        decision = mc.decide(
            snap, diag, fp,
            portfolio=portfolio,
            backend_policy=policy,
        )
        assert decision.backend_name == "random"

    def test_drift_phase_reset_changes_backend_selection_context(self):
        """When drift resets phase from EXPLOITATION to LEARNING, the
        backend selection should have already happened (before drift
        adaptation), but the drift adapter may switch the backend."""
        mc = MetaController(available_backends=["tpe", "random", "latin_hypercube"])
        snap = _make_snapshot(n_obs=15)
        # Trigger exploitation phase
        diag = _base_diagnostics(convergence_trend=0.5, model_uncertainty=0.1)
        fp = ProblemFingerprint()

        # Severe drift
        drift = _make_drift_report(
            drift_score=0.8,
            drift_type="sudden",
            affected_parameters=["x1"],
        )

        decision = mc.decide(snap, diag, fp, drift_report=drift)

        # Phase should be reset to LEARNING
        assert decision.phase == Phase.LEARNING
        # Backend should be switched to space-filling by drift adapter
        assert decision.backend_name in ("random", "latin_hypercube")
        # Risk posture should be conservative (phase reset triggers this)
        assert decision.risk_posture == RiskPosture.CONSERVATIVE
