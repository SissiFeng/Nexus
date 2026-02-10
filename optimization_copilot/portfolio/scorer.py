"""Backend scoring engine that converts portfolio history into strategy priors.

Computes a weighted, explainable score for each candidate backend given
the current ProblemFingerprint and optional context signals (drift, cost,
failure taxonomy).  The score is deterministic and falls back to a
rule-based baseline when portfolio data is insufficient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import ProblemFingerprint, RiskPosture


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BackendScore:
    """Explainable score for a single backend candidate."""

    backend_name: str
    total_score: float
    expected_gain: float
    expected_fail: float
    expected_cost: float
    drift_penalty: float
    incompatibility_penalty: float
    confidence: float  # 0-1, reliability of the score
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "total_score": self.total_score,
            "expected_gain": self.expected_gain,
            "expected_fail": self.expected_fail,
            "expected_cost": self.expected_cost,
            "drift_penalty": self.drift_penalty,
            "incompatibility_penalty": self.incompatibility_penalty,
            "confidence": self.confidence,
            "breakdown": dict(self.breakdown),
        }


@dataclass
class ScoringWeights:
    """Configurable weights for the scoring formula."""

    gain: float = 0.35
    fail: float = 0.25
    cost: float = 0.20
    drift: float = 0.15
    incompatibility: float = 0.05


# ---------------------------------------------------------------------------
# Default rule-based priors (cold-start fallback)
# ---------------------------------------------------------------------------

# When portfolio has no data for a fingerprint, use these static priors.
# Higher = better default preference.
_DEFAULT_BACKEND_PRIOR: dict[str, float] = {
    "tpe": 0.7,
    "random": 0.5,
    "latin_hypercube": 0.6,
    "rf_surrogate": 0.65,
    "cma_es": 0.55,
    "genetic": 0.50,
    "local_search": 0.45,
}

_DEFAULT_PRIOR_SCORE = 0.5  # for unknown backends


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class BackendScorer:
    """Converts portfolio records + context into ranked backend scores.

    Parameters
    ----------
    weights : ScoringWeights | None
        Weight configuration for the multi-criteria score.
    default_backends : dict[str, float] | None
        Rule-based prior scores used when portfolio data is missing.
    min_observations_for_trust : int
        Minimum portfolio runs before trusting the data over the prior.
    """

    def __init__(
        self,
        weights: ScoringWeights | None = None,
        default_backends: dict[str, float] | None = None,
        min_observations_for_trust: int = 3,
    ) -> None:
        self.weights = weights or ScoringWeights()
        self.default_backends = default_backends or dict(_DEFAULT_BACKEND_PRIOR)
        self.min_observations_for_trust = min_observations_for_trust

    def score_backends(
        self,
        fingerprint: ProblemFingerprint,
        portfolio: Any | None = None,
        available_backends: list[str] | None = None,
        *,
        drift_report: Any | None = None,
        cost_signals: Any | None = None,
        failure_taxonomy: Any | None = None,
        backend_policy: Any | None = None,
    ) -> list[BackendScore]:
        """Score all available backends and return them ranked (best first).

        Parameters
        ----------
        fingerprint :
            Current problem characterization.
        portfolio :
            An ``AlgorithmPortfolio`` instance (or None for cold-start).
        available_backends :
            Backend names to consider.  Defaults to all known defaults.
        drift_report :
            Optional ``DriftReport`` for drift-aware scoring.
        cost_signals :
            Optional ``CostSignals`` for cost-aware scoring.
        failure_taxonomy :
            Optional ``FailureTaxonomy`` for failure-aware scoring.
        backend_policy :
            Optional policy with ``is_allowed(name) -> bool``.

        Returns
        -------
        list[BackendScore]
            Sorted descending by ``total_score``.
        """
        if available_backends is None:
            available_backends = list(self.default_backends.keys())

        scores: list[BackendScore] = []
        for name in available_backends:
            score = self._score_single(
                name,
                fingerprint,
                portfolio,
                drift_report=drift_report,
                cost_signals=cost_signals,
                failure_taxonomy=failure_taxonomy,
                backend_policy=backend_policy,
            )
            scores.append(score)

        # Deterministic sort: by total_score desc, then name asc for ties.
        scores.sort(key=lambda s: (-s.total_score, s.backend_name))
        return scores

    def _score_single(
        self,
        backend_name: str,
        fingerprint: ProblemFingerprint,
        portfolio: Any | None,
        *,
        drift_report: Any | None = None,
        cost_signals: Any | None = None,
        failure_taxonomy: Any | None = None,
        backend_policy: Any | None = None,
    ) -> BackendScore:
        """Compute score for a single backend."""
        w = self.weights

        # -- Expected gain ------------------------------------------------
        expected_gain = self._compute_expected_gain(
            backend_name, fingerprint, portfolio
        )

        # -- Expected failure ---------------------------------------------
        expected_fail = self._compute_expected_fail(
            backend_name, fingerprint, portfolio, failure_taxonomy
        )

        # -- Expected cost ------------------------------------------------
        expected_cost = self._compute_expected_cost(
            backend_name, fingerprint, portfolio, cost_signals
        )

        # -- Drift penalty ------------------------------------------------
        drift_penalty = self._compute_drift_penalty(
            backend_name, fingerprint, portfolio, drift_report
        )

        # -- Incompatibility penalty --------------------------------------
        incompatibility_penalty = self._compute_incompatibility(
            backend_name, backend_policy
        )

        # -- Confidence ---------------------------------------------------
        confidence = self._compute_confidence(
            backend_name, fingerprint, portfolio
        )

        # -- Total score --------------------------------------------------
        total = (
            w.gain * expected_gain
            - w.fail * expected_fail
            - w.cost * expected_cost
            - w.drift * drift_penalty
            - w.incompatibility * incompatibility_penalty
        )

        breakdown = {
            "w_gain": w.gain,
            "w_fail": w.fail,
            "w_cost": w.cost,
            "w_drift": w.drift,
            "w_incompatibility": w.incompatibility,
            "gain_component": w.gain * expected_gain,
            "fail_component": -w.fail * expected_fail,
            "cost_component": -w.cost * expected_cost,
            "drift_component": -w.drift * drift_penalty,
            "incompat_component": -w.incompatibility * incompatibility_penalty,
        }

        return BackendScore(
            backend_name=backend_name,
            total_score=round(total, 6),
            expected_gain=round(expected_gain, 6),
            expected_fail=round(expected_fail, 6),
            expected_cost=round(expected_cost, 6),
            drift_penalty=round(drift_penalty, 6),
            incompatibility_penalty=round(incompatibility_penalty, 6),
            confidence=round(confidence, 6),
            breakdown=breakdown,
        )

    # -- Component computations -------------------------------------------

    def _compute_expected_gain(
        self,
        backend_name: str,
        fingerprint: ProblemFingerprint,
        portfolio: Any | None,
    ) -> float:
        """Expected performance gain (0-1 scale, higher is better)."""
        record = self._get_record(backend_name, fingerprint, portfolio)
        if record is None:
            return self.default_backends.get(backend_name, _DEFAULT_PRIOR_SCORE)

        # Blend portfolio evidence with prior based on confidence.
        n = record.get("n_uses", 0)
        trust = min(1.0, n / max(self.min_observations_for_trust, 1))
        prior = self.default_backends.get(backend_name, _DEFAULT_PRIOR_SCORE)

        # Portfolio gain: combine win rate, convergence speed, and inverse regret.
        win_rate = record.get("win_count", 0) / max(n, 1)
        speed = record.get("avg_convergence_speed", 0.5)
        inv_regret = 1.0 - min(record.get("avg_regret", 0.5), 1.0)
        efficiency = record.get("sample_efficiency", 0.5)

        portfolio_gain = (
            0.3 * win_rate + 0.3 * speed + 0.2 * inv_regret + 0.2 * efficiency
        )

        return trust * portfolio_gain + (1.0 - trust) * prior

    def _compute_expected_fail(
        self,
        backend_name: str,
        fingerprint: ProblemFingerprint,
        portfolio: Any | None,
        failure_taxonomy: Any | None,
    ) -> float:
        """Expected failure rate (0-1 scale, lower is better)."""
        record = self._get_record(backend_name, fingerprint, portfolio)
        base_fail = 0.1  # optimistic default

        if record is not None and record.get("n_uses", 0) >= self.min_observations_for_trust:
            base_fail = record.get("failure_rate", 0.1)

        # Adjust if taxonomy shows systemic failures.
        if failure_taxonomy is not None:
            dominant = getattr(failure_taxonomy, "dominant_type", None)
            type_rates = getattr(failure_taxonomy, "type_rates", {})
            total_fail_rate = sum(type_rates.values()) if type_rates else 0.0

            # Backends that aren't drift-robust are penalized more under
            # hardware-type failures.
            if dominant is not None and hasattr(dominant, "value"):
                dtype = dominant.value
                if dtype == "hardware":
                    base_fail = max(base_fail, total_fail_rate * 0.8)
                elif dtype == "chemistry":
                    base_fail = max(base_fail, total_fail_rate * 0.6)

        return min(base_fail, 1.0)

    def _compute_expected_cost(
        self,
        backend_name: str,
        fingerprint: ProblemFingerprint,
        portfolio: Any | None,
        cost_signals: Any | None,
    ) -> float:
        """Expected cost penalty (0-1 scale, lower is better)."""
        record = self._get_record(backend_name, fingerprint, portfolio)
        base_cost = 0.0

        if record is not None:
            # cost_efficiency: higher is better → lower penalty.
            ce = record.get("cost_efficiency", 0.0)
            base_cost = max(0.0, 1.0 - ce) if ce > 0 else 0.3

        if cost_signals is not None:
            pressure = getattr(cost_signals, "time_budget_pressure", 0.0)
            # Under high budget pressure, penalize expensive backends more.
            base_cost = base_cost * (1.0 + pressure)

        return min(base_cost, 1.0)

    def _compute_drift_penalty(
        self,
        backend_name: str,
        fingerprint: ProblemFingerprint,
        portfolio: Any | None,
        drift_report: Any | None,
    ) -> float:
        """Drift-aware penalty (0-1 scale, lower is better)."""
        if drift_report is None:
            return 0.0

        severity = getattr(drift_report, "drift_score", 0.0)
        if severity < 0.3:
            return 0.0

        record = self._get_record(backend_name, fingerprint, portfolio)
        if record is not None:
            robustness = record.get("drift_robustness", 0.5)
            # penalty = severity * (1 - robustness): robust backends penalized less
            return severity * (1.0 - robustness)

        # No portfolio data → moderate penalty under drift.
        return severity * 0.5

    def _compute_incompatibility(
        self,
        backend_name: str,
        backend_policy: Any | None,
    ) -> float:
        """Hard constraint penalty: 1.0 if disallowed, 0.0 otherwise."""
        if backend_policy is None:
            return 0.0

        # Check for is_allowed method (BackendPolicy interface).
        is_allowed = getattr(backend_policy, "is_allowed", None)
        if is_allowed is not None and not is_allowed(backend_name):
            return 1.0

        # Check for denylist attribute.
        denylist = getattr(backend_policy, "denylist", None)
        if denylist is not None and backend_name in denylist:
            return 1.0

        return 0.0

    def _compute_confidence(
        self,
        backend_name: str,
        fingerprint: ProblemFingerprint,
        portfolio: Any | None,
    ) -> float:
        """Confidence in the score (0-1, based on data availability)."""
        record = self._get_record(backend_name, fingerprint, portfolio)
        if record is None:
            return 0.0  # pure prior, no confidence

        n = record.get("n_uses", 0)
        # Saturates at ~10 runs.
        return min(1.0, n / 10.0)

    # -- Portfolio data access -------------------------------------------

    def _get_record(
        self,
        backend_name: str,
        fingerprint: ProblemFingerprint,
        portfolio: Any | None,
    ) -> dict[str, Any] | None:
        """Extract portfolio record as a dict, or None if unavailable."""
        if portfolio is None:
            return None

        # Try exact fingerprint match first.
        fp_key = _fingerprint_key(fingerprint)
        records = getattr(portfolio, "_records", {})

        key = (fp_key, backend_name)
        if key in records:
            rec = records[key]
            return _record_to_dict(rec)

        # Try cross-fingerprint aggregation.
        aggregate = getattr(portfolio, "_aggregate_for_backend", None)
        if aggregate is not None:
            agg = aggregate(backend_name)
            if agg is not None:
                return agg

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fingerprint_key(fp: ProblemFingerprint) -> str:
    """Convert fingerprint to a stable string key."""
    return "|".join(str(v) for v in fp.to_tuple())


def _record_to_dict(record: Any) -> dict[str, Any]:
    """Convert a BackendRecord (or similar) to a plain dict."""
    if hasattr(record, "to_dict"):
        return record.to_dict()
    if hasattr(record, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(record)
    return dict(record) if isinstance(record, dict) else {}
