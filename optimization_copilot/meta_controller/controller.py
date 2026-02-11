"""Meta-Controller: core intelligence for strategy selection and phase orchestration.

Optionally integrates with:
- ``BackendScorer`` for portfolio-based backend ranking
- ``DriftStrategyAdapter`` for drift-triggered strategy adaptation
- ``FailureSurfaceLearner`` for failure-aware risk modulation
- ``CostSignals`` for budget-aware exploration tuning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Phase,
    ProblemFingerprint,
    RiskPosture,
    StabilizeSpec,
    StrategyDecision,
    DataScale,
    NoiseRegime,
    VariableType,
    ObjectiveForm,
)


@dataclass
class SwitchingThresholds:
    """Configurable thresholds for phase transitions."""
    cold_start_min_observations: int = 10
    learning_plateau_length: int = 5
    exploitation_improvement_slope: float = -0.1
    stagnation_plateau_length: int = 10
    stagnation_failure_spike: float = 0.5
    coverage_plateau: float = 0.8
    uncertainty_collapse: float = 0.1


# Default backend preferences per phase and problem type
PHASE_BACKEND_MAP: dict[Phase, list[str]] = {
    Phase.COLD_START: ["latin_hypercube", "random"],
    Phase.LEARNING: ["tpe", "random_forest_surrogate", "latin_hypercube"],
    Phase.EXPLOITATION: ["tpe", "cma_es"],
    Phase.STAGNATION: ["random", "latin_hypercube", "cma_es"],
    Phase.TERMINATION: ["tpe"],
}

FINGERPRINT_BACKEND_OVERRIDES: dict[str, list[str]] = {
    "high_noise": ["random", "latin_hypercube"],
    "mixed_variables": ["tpe", "random"],
    "multi_objective": ["random", "latin_hypercube"],  # Safe defaults
    "tiny_data": ["random", "latin_hypercube"],
}


class MetaController:
    """Orchestrates optimization strategy selection across campaign phases.

    Deterministic: same (snapshot, diagnostics, fingerprint, seed) → same decision.
    """

    def __init__(
        self,
        thresholds: SwitchingThresholds | None = None,
        available_backends: list[str] | None = None,
    ) -> None:
        self.thresholds = thresholds or SwitchingThresholds()
        self.available_backends = available_backends or [
            "random", "latin_hypercube", "tpe",
        ]

    def decide(
        self,
        snapshot: CampaignSnapshot,
        diagnostics: dict[str, float],
        fingerprint: ProblemFingerprint,
        seed: int = 42,
        previous_phase: Phase | None = None,
        *,
        portfolio: Any | None = None,
        drift_report: Any | None = None,
        cost_signals: Any | None = None,
        failure_taxonomy: Any | None = None,
        backend_policy: Any | None = None,
    ) -> StrategyDecision:
        """Make a deterministic strategy decision.

        Parameters
        ----------
        snapshot : CampaignSnapshot
        diagnostics : dict with the 17 diagnostic signal values
        fingerprint : ProblemFingerprint
        seed : int for deterministic tie-breaking
        previous_phase : Phase or None
        portfolio : AlgorithmPortfolio or None
            Historical backend performance data for informed selection.
        drift_report : DriftReport or None
            Drift detection results for strategy adaptation.
        cost_signals : CostSignals or None
            Budget/cost context for exploration modulation.
        failure_taxonomy : FailureTaxonomy or None
            Failure classification for risk-aware scoring.
        backend_policy : BackendPolicy or None
            Hard constraints on which backends are allowed.

        Returns
        -------
        StrategyDecision with full audit trail
        """
        reason_codes: list[str] = []
        fallback_events: list[str] = []

        # 1. Determine phase
        phase = self._determine_phase(
            snapshot, diagnostics, fingerprint, previous_phase, reason_codes
        )

        # 2. Select backend — use portfolio scorer if available
        backend = self._select_backend_with_context(
            phase, fingerprint, reason_codes, fallback_events, seed,
            portfolio=portfolio,
            drift_report=drift_report,
            cost_signals=cost_signals,
            failure_taxonomy=failure_taxonomy,
            backend_policy=backend_policy,
        )

        # 3. Compute exploration strength
        exploration = self._compute_exploration_strength(
            phase, diagnostics, fingerprint
        )

        # 3b. Adjust exploration for cost pressure
        if cost_signals is not None:
            exploration = self._adjust_exploration_for_cost(
                exploration, cost_signals, reason_codes
            )

        # 4. Determine risk posture
        risk = self._determine_risk_posture(phase, diagnostics, snapshot)

        # 5. Build stabilize spec
        stabilize = self._build_stabilize_spec(diagnostics, fingerprint)

        # 6. Batch size hints
        batch_size = self._compute_batch_size(phase, snapshot, fingerprint)

        decision = StrategyDecision(
            backend_name=backend,
            stabilize_spec=stabilize,
            exploration_strength=exploration,
            batch_size=batch_size,
            risk_posture=risk,
            phase=phase,
            reason_codes=reason_codes,
            fallback_events=fallback_events,
            decision_metadata={
                "seed": seed,
                "previous_phase": previous_phase.value if previous_phase else None,
                "n_observations": snapshot.n_observations,
                "portfolio_used": portfolio is not None,
                "drift_report_used": drift_report is not None,
                "cost_signals_used": cost_signals is not None,
                "failure_taxonomy_used": failure_taxonomy is not None,
            },
        )

        # 7. Apply drift adaptation (post-decision adjustment)
        if drift_report is not None:
            decision = self._apply_drift_adaptation(
                decision, drift_report, reason_codes
            )

        return decision

    def _determine_phase(
        self,
        snapshot: CampaignSnapshot,
        diagnostics: dict[str, float],
        fingerprint: ProblemFingerprint,
        previous_phase: Phase | None,
        reason_codes: list[str],
    ) -> Phase:
        n_obs = snapshot.n_observations
        th = self.thresholds

        # Cold start: not enough data
        if n_obs < th.cold_start_min_observations:
            reason_codes.append(f"cold_start:n_obs={n_obs}<{th.cold_start_min_observations}")
            return Phase.COLD_START

        plateau = diagnostics.get("kpi_plateau_length", 0)
        failure_cluster = diagnostics.get("failure_clustering", 0)
        convergence = diagnostics.get("convergence_trend", 0)
        coverage = diagnostics.get("exploration_coverage", 0)
        uncertainty = diagnostics.get("model_uncertainty", 1.0)

        # Stagnation detection
        if plateau > th.stagnation_plateau_length:
            reason_codes.append(f"stagnation:plateau={plateau}>{th.stagnation_plateau_length}")
            return Phase.STAGNATION

        if failure_cluster > th.stagnation_failure_spike:
            reason_codes.append(
                f"stagnation:failure_spike={failure_cluster:.2f}>{th.stagnation_failure_spike}"
            )
            return Phase.STAGNATION

        # Exploitation: convergence is good, uncertainty is low
        if convergence > 0.3 and uncertainty < 0.3:
            reason_codes.append(
                f"exploitation:convergence={convergence:.2f},uncertainty={uncertainty:.2f}"
            )
            return Phase.EXPLOITATION

        if coverage > th.coverage_plateau and uncertainty < th.uncertainty_collapse:
            reason_codes.append(
                f"exploitation:coverage={coverage:.2f},uncertainty_collapse={uncertainty:.2f}"
            )
            return Phase.EXPLOITATION

        # Learning phase: default for sufficient data
        reason_codes.append(f"learning:n_obs={n_obs},coverage={coverage:.2f}")
        return Phase.LEARNING

    def _select_backend(
        self,
        phase: Phase,
        fingerprint: ProblemFingerprint,
        reason_codes: list[str],
        fallback_events: list[str],
        seed: int,
    ) -> str:
        # Get preferred backends for this phase
        preferred = list(PHASE_BACKEND_MAP.get(phase, ["random"]))

        # Apply fingerprint overrides
        if fingerprint.noise_regime == NoiseRegime.HIGH:
            overrides = FINGERPRINT_BACKEND_OVERRIDES["high_noise"]
            preferred = self._prioritize(preferred, overrides)
            reason_codes.append("backend_override:high_noise")

        if fingerprint.variable_types == VariableType.MIXED:
            overrides = FINGERPRINT_BACKEND_OVERRIDES["mixed_variables"]
            preferred = self._prioritize(preferred, overrides)
            reason_codes.append("backend_override:mixed_variables")

        if fingerprint.objective_form == ObjectiveForm.MULTI_OBJECTIVE:
            overrides = FINGERPRINT_BACKEND_OVERRIDES["multi_objective"]
            preferred = self._prioritize(preferred, overrides)
            reason_codes.append("backend_override:multi_objective")

        if fingerprint.data_scale == DataScale.TINY:
            overrides = FINGERPRINT_BACKEND_OVERRIDES["tiny_data"]
            preferred = self._prioritize(preferred, overrides)
            reason_codes.append("backend_override:tiny_data")

        # Select first available backend
        for backend in preferred:
            if backend in self.available_backends:
                reason_codes.append(f"backend_selected:{backend}")
                return backend

        # Fallback: use first available
        fallback = self.available_backends[0] if self.available_backends else "random"
        fallback_events.append(f"no_preferred_available:using_{fallback}")
        return fallback

    def _compute_exploration_strength(
        self,
        phase: Phase,
        diagnostics: dict[str, float],
        fingerprint: ProblemFingerprint,
    ) -> float:
        """0.0 = pure exploitation, 1.0 = pure exploration."""
        base = {
            Phase.COLD_START: 0.9,
            Phase.LEARNING: 0.6,
            Phase.EXPLOITATION: 0.2,
            Phase.STAGNATION: 0.8,
            Phase.TERMINATION: 0.1,
        }.get(phase, 0.5)

        # Adjust based on diagnostics
        coverage = diagnostics.get("exploration_coverage", 0)
        if coverage < 0.3:
            base = min(1.0, base + 0.15)

        noise = diagnostics.get("noise_estimate", 0)
        if noise > 0.5:
            base = min(1.0, base + 0.1)

        # Tiny data → more exploration
        if fingerprint.data_scale == DataScale.TINY:
            base = min(1.0, base + 0.1)

        # UQ calibration: increase exploration when uncertainty is miscalibrated
        miscal = diagnostics.get("miscalibration_score", 0)
        overconf = diagnostics.get("overconfidence_rate", 0)
        if miscal > 0.3 or overconf > 0.3:
            uq_boost = max(miscal, overconf) * 0.2
            base = min(1.0, base + uq_boost)

        return round(base, 2)

    @staticmethod
    def _determine_risk_posture(
        phase: Phase,
        diagnostics: dict[str, float],
        snapshot: CampaignSnapshot,
    ) -> RiskPosture:
        if phase in (Phase.COLD_START, Phase.STAGNATION):
            return RiskPosture.CONSERVATIVE

        failure_rate = diagnostics.get("failure_rate", 0)
        if failure_rate > 0.3:
            return RiskPosture.CONSERVATIVE

        if phase == Phase.EXPLOITATION:
            convergence = diagnostics.get("convergence_trend", 0)
            if convergence > 0.5:
                return RiskPosture.AGGRESSIVE

        return RiskPosture.MODERATE

    @staticmethod
    def _build_stabilize_spec(
        diagnostics: dict[str, float],
        fingerprint: ProblemFingerprint,
    ) -> StabilizeSpec:
        noise = diagnostics.get("noise_estimate", 0)
        failure_rate = diagnostics.get("failure_rate", 0)

        # Noise handling
        window = 3
        if noise > 0.5:
            window = 5
        elif noise > 0.3:
            window = 4

        # Outlier sigma
        sigma = 3.0
        if fingerprint.noise_regime == NoiseRegime.HIGH:
            sigma = 2.5

        # Failure handling
        failure_handling = "penalize"
        if failure_rate > 0.4:
            failure_handling = "exclude"
        elif failure_rate > 0.2:
            failure_handling = "impute"

        # Reweighting
        reweighting = "none"
        if noise > 0.3:
            reweighting = "recency"

        return StabilizeSpec(
            noise_smoothing_window=window,
            outlier_rejection_sigma=sigma,
            failure_handling=failure_handling,
            reweighting_strategy=reweighting,
        )

    @staticmethod
    def _compute_batch_size(
        phase: Phase,
        snapshot: CampaignSnapshot,
        fingerprint: ProblemFingerprint,
    ) -> int:
        if phase == Phase.COLD_START:
            n_params = len(snapshot.parameter_specs)
            return max(2, min(n_params * 2, 10))

        if phase == Phase.EXPLOITATION:
            return 1

        if phase == Phase.STAGNATION:
            return max(3, len(snapshot.parameter_specs))

        # Learning
        return max(1, min(5, len(snapshot.parameter_specs)))

    @staticmethod
    def _prioritize(preferred: list[str], overrides: list[str]) -> list[str]:
        """Move override backends to front of preferred list."""
        result = [b for b in overrides if b in preferred]
        result.extend(b for b in preferred if b not in result)
        return result

    # -- Context-aware backend selection ----------------------------------

    def _select_backend_with_context(
        self,
        phase: Phase,
        fingerprint: ProblemFingerprint,
        reason_codes: list[str],
        fallback_events: list[str],
        seed: int,
        *,
        portfolio: Any | None = None,
        drift_report: Any | None = None,
        cost_signals: Any | None = None,
        failure_taxonomy: Any | None = None,
        backend_policy: Any | None = None,
    ) -> str:
        """Select backend using portfolio scorer when available, else rules."""
        if portfolio is not None:
            try:
                from optimization_copilot.portfolio.scorer import BackendScorer
                scorer = BackendScorer()
                scores = scorer.score_backends(
                    fingerprint=fingerprint,
                    portfolio=portfolio,
                    available_backends=self.available_backends,
                    drift_report=drift_report,
                    cost_signals=cost_signals,
                    failure_taxonomy=failure_taxonomy,
                    backend_policy=backend_policy,
                )
                if scores:
                    best = scores[0]
                    reason_codes.append(
                        f"portfolio_scored:{best.backend_name}"
                        f"(score={best.total_score:.3f},conf={best.confidence:.2f})"
                    )
                    return best.backend_name
            except Exception as exc:
                fallback_events.append(f"portfolio_scorer_failed:{exc}")

        # Use AutoSampler hint if backend_policy is a backend name string.
        if isinstance(backend_policy, str) and backend_policy in self.available_backends:
            reason_codes.append(f"auto_sampler_hint:{backend_policy}")
            return backend_policy

        # Fallback to rule-based selection.
        return self._select_backend(
            phase, fingerprint, reason_codes, fallback_events, seed
        )

    @staticmethod
    def _adjust_exploration_for_cost(
        exploration: float,
        cost_signals: Any,
        reason_codes: list[str],
    ) -> float:
        """Modulate exploration based on budget pressure."""
        pressure = getattr(cost_signals, "time_budget_pressure", 0.0)
        trend = getattr(cost_signals, "cost_efficiency_trend", 0.0)

        adjustment = -pressure * 0.3 + trend * 0.1
        new_exploration = max(0.0, min(1.0, exploration + adjustment))

        if abs(adjustment) > 0.01:
            reason_codes.append(
                f"cost_adjustment:{adjustment:+.3f}"
                f"(pressure={pressure:.2f},trend={trend:.2f})"
            )

        return round(new_exploration, 4)

    def _apply_drift_adaptation(
        self,
        decision: StrategyDecision,
        drift_report: Any,
        reason_codes: list[str],
    ) -> StrategyDecision:
        """Apply drift-based strategy adaptation to the decision."""
        try:
            from optimization_copilot.drift.strategy import DriftStrategyAdapter
            adapter = DriftStrategyAdapter()
            adaptation = adapter.adapt(
                drift_report=drift_report,
                current_phase=decision.phase,
                current_exploration=decision.exploration_strength,
            )
            if adaptation.actions:
                decision = adapter.apply(decision, adaptation)
        except Exception as exc:
            decision.fallback_events.append(f"drift_adaptation_failed:{exc}")

        return decision
