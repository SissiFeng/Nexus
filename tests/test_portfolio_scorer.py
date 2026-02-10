"""Tests for portfolio-based backend scoring (Track 1).

Verifies:
- Deterministic scoring given fixed inputs
- Fallback to rule-based baseline when portfolio is empty
- Monotonicity: improving a single metric should not worsen the score
- Incompatibility penalty blocks disallowed backends
- Drift and cost awareness in scoring
"""

from __future__ import annotations

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
from optimization_copilot.portfolio.scorer import (
    BackendScore,
    BackendScorer,
    ScoringWeights,
)
from optimization_copilot.portfolio.portfolio import (
    AlgorithmPortfolio,
    BackendRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fp(**kwargs) -> ProblemFingerprint:
    return ProblemFingerprint(**kwargs)


def _make_portfolio_with_records(
    records: list[tuple[str, str, dict]],
) -> AlgorithmPortfolio:
    """Build a portfolio with pre-set records.

    records: [(fingerprint_key, backend_name, stats_dict), ...]
    """
    portfolio = AlgorithmPortfolio()
    fp = _make_fp()

    for fp_key, backend_name, stats in records:
        # Directly inject records for testing.
        key = (fp_key, backend_name)
        portfolio._records[key] = BackendRecord(
            fingerprint_key=fp_key,
            backend_name=backend_name,
            n_uses=stats.get("n_uses", 10),
            win_count=stats.get("win_count", 5),
            avg_convergence_speed=stats.get("avg_convergence_speed", 0.5),
            avg_regret=stats.get("avg_regret", 0.2),
            failure_rate=stats.get("failure_rate", 0.1),
            sample_efficiency=stats.get("sample_efficiency", 0.5),
        )
    return portfolio


def _fp_key(fp: ProblemFingerprint) -> str:
    return "|".join(str(v) for v in fp.to_tuple())


class _MockDriftReport:
    def __init__(self, drift_score=0.0, drift_detected=False):
        self.drift_score = drift_score
        self.drift_detected = drift_detected


class _MockCostSignals:
    def __init__(self, time_budget_pressure=0.0, cost_efficiency_trend=0.0):
        self.time_budget_pressure = time_budget_pressure
        self.cost_efficiency_trend = cost_efficiency_trend


class _MockPolicy:
    def __init__(self, denied: set[str] | None = None):
        self._denied = denied or set()

    def is_allowed(self, name: str) -> bool:
        return name not in self._denied


# ---------------------------------------------------------------------------
# Tests: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self):
        """Identical inputs must produce identical scores."""
        fp = _make_fp()
        scorer = BackendScorer()
        backends = ["tpe", "random", "latin_hypercube"]

        scores1 = scorer.score_backends(fp, available_backends=backends)
        scores2 = scorer.score_backends(fp, available_backends=backends)

        assert len(scores1) == len(scores2)
        for s1, s2 in zip(scores1, scores2):
            assert s1.backend_name == s2.backend_name
            assert s1.total_score == s2.total_score

    def test_ordering_stable_across_calls(self):
        """Ranking order must be identical across calls."""
        fp = _make_fp()
        key = _fp_key(fp)
        portfolio = _make_portfolio_with_records([
            (key, "tpe", {"win_count": 8, "n_uses": 10}),
            (key, "random", {"win_count": 2, "n_uses": 10}),
        ])
        scorer = BackendScorer()
        backends = ["tpe", "random"]

        for _ in range(5):
            scores = scorer.score_backends(fp, portfolio, backends)
            assert scores[0].backend_name == "tpe"


# ---------------------------------------------------------------------------
# Tests: Fallback (no portfolio data)
# ---------------------------------------------------------------------------


class TestFallback:
    def test_no_portfolio_uses_defaults(self):
        """Without portfolio data, uses rule-based priors."""
        fp = _make_fp()
        scorer = BackendScorer()
        scores = scorer.score_backends(fp, portfolio=None, available_backends=["tpe", "random"])

        assert len(scores) == 2
        # All scores should use default priors.
        for s in scores:
            assert s.confidence == 0.0  # no portfolio data

    def test_empty_portfolio_uses_defaults(self):
        """Empty portfolio behaves like no portfolio."""
        fp = _make_fp()
        portfolio = AlgorithmPortfolio()
        scorer = BackendScorer()
        scores = scorer.score_backends(fp, portfolio, ["tpe", "random"])

        for s in scores:
            assert s.confidence == 0.0

    def test_default_prior_ranking(self):
        """TPE should rank higher than random in default priors."""
        fp = _make_fp()
        scorer = BackendScorer()
        scores = scorer.score_backends(fp, available_backends=["tpe", "random"])

        names = [s.backend_name for s in scores]
        assert names.index("tpe") < names.index("random")


# ---------------------------------------------------------------------------
# Tests: Monotonicity
# ---------------------------------------------------------------------------


class TestMonotonicity:
    def test_higher_win_rate_higher_score(self):
        """Increasing win count should not decrease the score."""
        fp = _make_fp()
        key = _fp_key(fp)
        scorer = BackendScorer()

        # Low win rate
        p_low = _make_portfolio_with_records([
            (key, "tpe", {"win_count": 2, "n_uses": 10}),
        ])
        score_low = scorer.score_backends(fp, p_low, ["tpe"])[0]

        # High win rate
        p_high = _make_portfolio_with_records([
            (key, "tpe", {"win_count": 9, "n_uses": 10}),
        ])
        score_high = scorer.score_backends(fp, p_high, ["tpe"])[0]

        assert score_high.total_score >= score_low.total_score

    def test_lower_regret_higher_score(self):
        """Decreasing regret should not decrease the score."""
        fp = _make_fp()
        key = _fp_key(fp)
        scorer = BackendScorer()

        p_high_regret = _make_portfolio_with_records([
            (key, "tpe", {"avg_regret": 0.8, "n_uses": 10}),
        ])
        p_low_regret = _make_portfolio_with_records([
            (key, "tpe", {"avg_regret": 0.1, "n_uses": 10}),
        ])

        score_bad = scorer.score_backends(fp, p_high_regret, ["tpe"])[0]
        score_good = scorer.score_backends(fp, p_low_regret, ["tpe"])[0]

        assert score_good.total_score >= score_bad.total_score

    def test_lower_failure_rate_higher_score(self):
        """Decreasing failure rate should not decrease the score."""
        fp = _make_fp()
        key = _fp_key(fp)
        scorer = BackendScorer()

        p_high_fail = _make_portfolio_with_records([
            (key, "tpe", {"failure_rate": 0.5, "n_uses": 10}),
        ])
        p_low_fail = _make_portfolio_with_records([
            (key, "tpe", {"failure_rate": 0.05, "n_uses": 10}),
        ])

        score_bad = scorer.score_backends(fp, p_high_fail, ["tpe"])[0]
        score_good = scorer.score_backends(fp, p_low_fail, ["tpe"])[0]

        assert score_good.total_score >= score_bad.total_score

    def test_higher_speed_higher_score(self):
        """Faster convergence should not decrease the score."""
        fp = _make_fp()
        key = _fp_key(fp)
        scorer = BackendScorer()

        p_slow = _make_portfolio_with_records([
            (key, "tpe", {"avg_convergence_speed": 0.2, "n_uses": 10}),
        ])
        p_fast = _make_portfolio_with_records([
            (key, "tpe", {"avg_convergence_speed": 0.9, "n_uses": 10}),
        ])

        score_slow = scorer.score_backends(fp, p_slow, ["tpe"])[0]
        score_fast = scorer.score_backends(fp, p_fast, ["tpe"])[0]

        assert score_fast.total_score >= score_slow.total_score


# ---------------------------------------------------------------------------
# Tests: Incompatibility
# ---------------------------------------------------------------------------


class TestIncompatibility:
    def test_denied_backend_penalized(self):
        """Denied backends should score lower."""
        fp = _make_fp()
        scorer = BackendScorer()
        policy = _MockPolicy(denied={"random"})

        scores = scorer.score_backends(
            fp, available_backends=["tpe", "random"],
            backend_policy=policy,
        )

        tpe_score = next(s for s in scores if s.backend_name == "tpe")
        random_score = next(s for s in scores if s.backend_name == "random")

        assert tpe_score.total_score > random_score.total_score
        assert random_score.incompatibility_penalty == 1.0

    def test_no_policy_no_penalty(self):
        """Without policy, no incompatibility penalty."""
        fp = _make_fp()
        scorer = BackendScorer()
        scores = scorer.score_backends(fp, available_backends=["tpe"])

        assert scores[0].incompatibility_penalty == 0.0


# ---------------------------------------------------------------------------
# Tests: Drift awareness
# ---------------------------------------------------------------------------


class TestDriftAwareness:
    def test_no_drift_no_penalty(self):
        """Without drift, drift penalty should be zero."""
        fp = _make_fp()
        scorer = BackendScorer()
        scores = scorer.score_backends(fp, available_backends=["tpe"])
        assert scores[0].drift_penalty == 0.0

    def test_drift_increases_penalty(self):
        """High drift severity should increase drift penalty."""
        fp = _make_fp()
        scorer = BackendScorer()

        no_drift = _MockDriftReport(drift_score=0.0)
        high_drift = _MockDriftReport(drift_score=0.8, drift_detected=True)

        scores_calm = scorer.score_backends(
            fp, available_backends=["tpe"], drift_report=no_drift
        )
        scores_drift = scorer.score_backends(
            fp, available_backends=["tpe"], drift_report=high_drift
        )

        assert scores_drift[0].drift_penalty > scores_calm[0].drift_penalty


# ---------------------------------------------------------------------------
# Tests: Cost awareness
# ---------------------------------------------------------------------------


class TestCostAwareness:
    def test_high_pressure_reduces_score(self):
        """High budget pressure should reduce scores via cost penalty."""
        fp = _make_fp()
        key = _fp_key(fp)
        portfolio = _make_portfolio_with_records([
            (key, "tpe", {"n_uses": 10}),
        ])
        scorer = BackendScorer()

        low_pressure = _MockCostSignals(time_budget_pressure=0.1)
        high_pressure = _MockCostSignals(time_budget_pressure=0.9)

        scores_low = scorer.score_backends(
            fp, portfolio, ["tpe"], cost_signals=low_pressure
        )
        scores_high = scorer.score_backends(
            fp, portfolio, ["tpe"], cost_signals=high_pressure
        )

        assert scores_high[0].expected_cost >= scores_low[0].expected_cost


# ---------------------------------------------------------------------------
# Tests: Score structure
# ---------------------------------------------------------------------------


class TestScoreStructure:
    def test_score_has_all_fields(self):
        """BackendScore should contain all expected fields."""
        fp = _make_fp()
        scorer = BackendScorer()
        scores = scorer.score_backends(fp, available_backends=["tpe"])

        s = scores[0]
        assert isinstance(s.backend_name, str)
        assert isinstance(s.total_score, float)
        assert isinstance(s.expected_gain, float)
        assert isinstance(s.expected_fail, float)
        assert isinstance(s.expected_cost, float)
        assert isinstance(s.drift_penalty, float)
        assert isinstance(s.incompatibility_penalty, float)
        assert isinstance(s.confidence, float)
        assert isinstance(s.breakdown, dict)

    def test_to_dict_roundtrip(self):
        """to_dict should produce a serializable dict."""
        fp = _make_fp()
        scorer = BackendScorer()
        scores = scorer.score_backends(fp, available_backends=["tpe"])

        d = scores[0].to_dict()
        assert d["backend_name"] == "tpe"
        assert "total_score" in d

    def test_confidence_grows_with_data(self):
        """More portfolio data → higher confidence."""
        fp = _make_fp()
        key = _fp_key(fp)
        scorer = BackendScorer()

        p_few = _make_portfolio_with_records([
            (key, "tpe", {"n_uses": 2}),
        ])
        p_many = _make_portfolio_with_records([
            (key, "tpe", {"n_uses": 10}),
        ])

        score_few = scorer.score_backends(fp, p_few, ["tpe"])[0]
        score_many = scorer.score_backends(fp, p_many, ["tpe"])[0]

        assert score_many.confidence >= score_few.confidence


# ---------------------------------------------------------------------------
# Tests: Portfolio-driven ranking
# ---------------------------------------------------------------------------


class TestPortfolioRanking:
    def test_better_backend_ranks_first(self):
        """Backend with better portfolio stats should rank first."""
        fp = _make_fp()
        key = _fp_key(fp)
        portfolio = _make_portfolio_with_records([
            (key, "tpe", {"win_count": 9, "avg_regret": 0.05, "n_uses": 10}),
            (key, "random", {"win_count": 1, "avg_regret": 0.6, "n_uses": 10}),
        ])
        scorer = BackendScorer()
        scores = scorer.score_backends(fp, portfolio, ["tpe", "random"])

        assert scores[0].backend_name == "tpe"
        assert scores[0].total_score > scores[1].total_score

    def test_unknown_fingerprint_uses_aggregation(self):
        """If exact fingerprint is missing, aggregation should still work."""
        fp_known = _make_fp(noise_regime=NoiseRegime.LOW)
        fp_unknown = _make_fp(noise_regime=NoiseRegime.HIGH)
        key_known = _fp_key(fp_known)

        portfolio = _make_portfolio_with_records([
            (key_known, "tpe", {"win_count": 8, "n_uses": 10}),
        ])
        scorer = BackendScorer()

        # Score for unknown fingerprint should still work (via aggregation).
        scores = scorer.score_backends(fp_unknown, portfolio, ["tpe"])
        assert len(scores) == 1
        assert scores[0].backend_name == "tpe"
