"""Tests for the Algorithm Portfolio Learning module."""

from optimization_copilot.core.models import (
    ProblemFingerprint,
    VariableType,
    NoiseRegime,
)
from optimization_copilot.portfolio.portfolio import (
    AlgorithmPortfolio,
    BackendRecord,
    PortfolioStats,
    _fingerprint_key,
)


def _make_fp(**kwargs) -> ProblemFingerprint:
    """Create a ProblemFingerprint with optional overrides."""
    return ProblemFingerprint(**kwargs)


def _winning_outcome(**overrides) -> dict:
    base = {
        "convergence_speed": 0.8,
        "regret": 0.05,
        "failure_rate": 0.0,
        "sample_efficiency": 0.9,
        "is_winner": True,
    }
    base.update(overrides)
    return base


def _losing_outcome(**overrides) -> dict:
    base = {
        "convergence_speed": 0.3,
        "regret": 0.4,
        "failure_rate": 0.2,
        "sample_efficiency": 0.2,
        "is_winner": False,
    }
    base.update(overrides)
    return base


class TestRecordOutcome:
    """Test recording outcomes and basic stats accumulation."""

    def test_single_outcome(self):
        portfolio = AlgorithmPortfolio()
        fp = _make_fp()
        portfolio.record_outcome(fp, "tpe", _winning_outcome())

        key = (_fingerprint_key(fp), "tpe")
        rec = portfolio._records[key]
        assert rec.n_uses == 1
        assert rec.win_count == 1
        assert abs(rec.avg_convergence_speed - 0.8) < 1e-9
        assert abs(rec.avg_regret - 0.05) < 1e-9

    def test_multiple_outcomes_incremental_mean(self):
        portfolio = AlgorithmPortfolio()
        fp = _make_fp()

        portfolio.record_outcome(fp, "bo", {"convergence_speed": 1.0, "regret": 0.0,
                                             "failure_rate": 0.0, "sample_efficiency": 1.0,
                                             "is_winner": True})
        portfolio.record_outcome(fp, "bo", {"convergence_speed": 0.0, "regret": 1.0,
                                             "failure_rate": 1.0, "sample_efficiency": 0.0,
                                             "is_winner": False})

        key = (_fingerprint_key(fp), "bo")
        rec = portfolio._records[key]
        assert rec.n_uses == 2
        assert rec.win_count == 1
        assert abs(rec.avg_convergence_speed - 0.5) < 1e-9
        assert abs(rec.avg_regret - 0.5) < 1e-9
        assert abs(rec.failure_rate - 0.5) < 1e-9
        assert abs(rec.sample_efficiency - 0.5) < 1e-9

    def test_separate_fingerprints_separate_records(self):
        portfolio = AlgorithmPortfolio()
        fp1 = _make_fp(noise_regime=NoiseRegime.LOW)
        fp2 = _make_fp(noise_regime=NoiseRegime.HIGH)

        portfolio.record_outcome(fp1, "tpe", _winning_outcome())
        portfolio.record_outcome(fp2, "tpe", _losing_outcome())

        key1 = (_fingerprint_key(fp1), "tpe")
        key2 = (_fingerprint_key(fp2), "tpe")
        assert portfolio._records[key1].win_count == 1
        assert portfolio._records[key2].win_count == 0


class TestRankBackends:
    """Test backend ranking logic."""

    def test_higher_win_rate_ranks_higher(self):
        portfolio = AlgorithmPortfolio()
        fp = _make_fp()

        # Give "good_backend" 10 wins
        for _ in range(10):
            portfolio.record_outcome(fp, "good_backend", _winning_outcome())

        # Give "bad_backend" 10 losses
        for _ in range(10):
            portfolio.record_outcome(fp, "bad_backend", _losing_outcome())

        stats = portfolio.rank_backends(fp, ["good_backend", "bad_backend"])
        assert stats.portfolio_rank[0] == "good_backend"
        assert stats.portfolio_rank[1] == "bad_backend"

    def test_confidence_grows_with_usage(self):
        portfolio = AlgorithmPortfolio()
        fp = _make_fp()

        # Record 1 outcome
        portfolio.record_outcome(fp, "tpe", _winning_outcome())
        stats_1 = portfolio.rank_backends(fp, ["tpe"])
        conf_1 = stats_1.confidence["tpe"]

        # Record 4 more
        for _ in range(4):
            portfolio.record_outcome(fp, "tpe", _winning_outcome())
        stats_5 = portfolio.rank_backends(fp, ["tpe"])
        conf_5 = stats_5.confidence["tpe"]

        # Record 5 more (total 10)
        for _ in range(5):
            portfolio.record_outcome(fp, "tpe", _winning_outcome())
        stats_10 = portfolio.rank_backends(fp, ["tpe"])
        conf_10 = stats_10.confidence["tpe"]

        assert conf_1 < conf_5 < conf_10
        assert abs(conf_1 - 0.1) < 1e-9   # 1/10
        assert abs(conf_5 - 0.5) < 1e-9   # 5/10
        assert abs(conf_10 - 1.0) < 1e-9  # 10/10 capped at 1.0

    def test_confidence_caps_at_one(self):
        portfolio = AlgorithmPortfolio()
        fp = _make_fp()

        for _ in range(20):
            portfolio.record_outcome(fp, "bo", _winning_outcome())

        stats = portfolio.rank_backends(fp, ["bo"])
        assert stats.confidence["bo"] == 1.0

    def test_unknown_backend_gets_zero_score(self):
        portfolio = AlgorithmPortfolio()
        fp = _make_fp()

        stats = portfolio.rank_backends(fp, ["unknown_backend"])
        assert stats.confidence["unknown_backend"] == 0.0
        assert stats.expected_gain["unknown_backend"] == 0.0
        assert stats.risk_penalty["unknown_backend"] == 0.0

    def test_unknown_fingerprint_uses_fallback(self):
        portfolio = AlgorithmPortfolio()
        fp_known = _make_fp(noise_regime=NoiseRegime.LOW)
        fp_unknown = _make_fp(noise_regime=NoiseRegime.HIGH)

        # Record data only for fp_known
        for _ in range(10):
            portfolio.record_outcome(fp_known, "tpe", _winning_outcome())

        # Rank for fp_unknown -- should use cross-fingerprint aggregation
        stats = portfolio.rank_backends(fp_unknown, ["tpe"])
        assert "tpe" in stats.portfolio_rank
        # Confidence should be reduced by the 0.3 cross-fp factor
        assert stats.confidence["tpe"] < 1.0
        # Expected gain should be > 0 (fallback found data)
        assert stats.expected_gain["tpe"] > 0.0

    def test_known_fingerprint_preferred_over_fallback(self):
        """When exact-match data exists, it should be used instead of aggregation."""
        portfolio = AlgorithmPortfolio()
        fp = _make_fp()

        for _ in range(10):
            portfolio.record_outcome(fp, "tpe", _winning_outcome())

        stats = portfolio.rank_backends(fp, ["tpe"])
        # Full confidence from exact match, not reduced by 0.3
        assert abs(stats.confidence["tpe"] - 1.0) < 1e-9

    def test_all_backends_returned_in_rank(self):
        portfolio = AlgorithmPortfolio()
        fp = _make_fp()
        backends = ["a", "b", "c"]

        stats = portfolio.rank_backends(fp, backends)
        assert set(stats.portfolio_rank) == set(backends)
        assert len(stats.portfolio_rank) == 3


class TestSerialization:
    """Test to_dict / from_dict round-trip."""

    def test_empty_portfolio_roundtrip(self):
        portfolio = AlgorithmPortfolio()
        data = portfolio.to_dict()
        restored = AlgorithmPortfolio.from_dict(data)
        assert len(restored._records) == 0

    def test_populated_portfolio_roundtrip(self):
        portfolio = AlgorithmPortfolio()
        fp = _make_fp()

        for _ in range(5):
            portfolio.record_outcome(fp, "tpe", _winning_outcome())
        for _ in range(3):
            portfolio.record_outcome(fp, "bo", _losing_outcome())

        data = portfolio.to_dict()
        restored = AlgorithmPortfolio.from_dict(data)

        assert len(restored._records) == 2

        key_tpe = (_fingerprint_key(fp), "tpe")
        key_bo = (_fingerprint_key(fp), "bo")

        assert restored._records[key_tpe].n_uses == 5
        assert restored._records[key_tpe].win_count == 5
        assert restored._records[key_bo].n_uses == 3
        assert restored._records[key_bo].win_count == 0

    def test_roundtrip_preserves_float_values(self):
        portfolio = AlgorithmPortfolio()
        fp = _make_fp()

        portfolio.record_outcome(fp, "cma_es", {
            "convergence_speed": 0.75,
            "regret": 0.12,
            "failure_rate": 0.05,
            "sample_efficiency": 0.88,
            "is_winner": True,
        })

        data = portfolio.to_dict()
        restored = AlgorithmPortfolio.from_dict(data)

        key = (_fingerprint_key(fp), "cma_es")
        rec = restored._records[key]
        assert abs(rec.avg_convergence_speed - 0.75) < 1e-9
        assert abs(rec.avg_regret - 0.12) < 1e-9
        assert abs(rec.failure_rate - 0.05) < 1e-9
        assert abs(rec.sample_efficiency - 0.88) < 1e-9

    def test_ranking_after_roundtrip(self):
        """Rankings should be identical before and after serialization."""
        portfolio = AlgorithmPortfolio()
        fp = _make_fp()

        for _ in range(10):
            portfolio.record_outcome(fp, "winner", _winning_outcome())
        for _ in range(10):
            portfolio.record_outcome(fp, "loser", _losing_outcome())

        stats_before = portfolio.rank_backends(fp, ["winner", "loser"])

        data = portfolio.to_dict()
        restored = AlgorithmPortfolio.from_dict(data)
        stats_after = restored.rank_backends(fp, ["winner", "loser"])

        assert stats_before.portfolio_rank == stats_after.portfolio_rank
        for name in ["winner", "loser"]:
            assert abs(stats_before.confidence[name] - stats_after.confidence[name]) < 1e-9
            assert abs(stats_before.expected_gain[name] - stats_after.expected_gain[name]) < 1e-9
            assert abs(stats_before.risk_penalty[name] - stats_after.risk_penalty[name]) < 1e-9


class TestMerge:
    """Test merging portfolios from different campaigns."""

    def test_merge_non_overlapping(self):
        p1 = AlgorithmPortfolio()
        p2 = AlgorithmPortfolio()
        fp = _make_fp()

        p1.record_outcome(fp, "tpe", _winning_outcome())
        p2.record_outcome(fp, "bo", _losing_outcome())

        p1.merge(p2)

        assert len(p1._records) == 2
        key_tpe = (_fingerprint_key(fp), "tpe")
        key_bo = (_fingerprint_key(fp), "bo")
        assert p1._records[key_tpe].n_uses == 1
        assert p1._records[key_bo].n_uses == 1

    def test_merge_overlapping_combines_stats(self):
        p1 = AlgorithmPortfolio()
        p2 = AlgorithmPortfolio()
        fp = _make_fp()

        # p1: 4 outcomes for tpe, all with convergence_speed=1.0
        for _ in range(4):
            p1.record_outcome(fp, "tpe", {
                "convergence_speed": 1.0, "regret": 0.0,
                "failure_rate": 0.0, "sample_efficiency": 1.0,
                "is_winner": True,
            })

        # p2: 6 outcomes for tpe, all with convergence_speed=0.0
        for _ in range(6):
            p2.record_outcome(fp, "tpe", {
                "convergence_speed": 0.0, "regret": 1.0,
                "failure_rate": 1.0, "sample_efficiency": 0.0,
                "is_winner": False,
            })

        p1.merge(p2)

        key = (_fingerprint_key(fp), "tpe")
        rec = p1._records[key]
        assert rec.n_uses == 10
        assert rec.win_count == 4
        # Weighted mean: (1.0 * 4 + 0.0 * 6) / 10 = 0.4
        assert abs(rec.avg_convergence_speed - 0.4) < 1e-9
        # Weighted mean: (0.0 * 4 + 1.0 * 6) / 10 = 0.6
        assert abs(rec.avg_regret - 0.6) < 1e-9

    def test_merge_does_not_modify_source(self):
        p1 = AlgorithmPortfolio()
        p2 = AlgorithmPortfolio()
        fp = _make_fp()

        p2.record_outcome(fp, "bo", _winning_outcome())
        original_data = p2.to_dict()

        p1.merge(p2)

        assert p2.to_dict() == original_data

    def test_merge_preserves_rankings(self):
        """After merging, the combined data should produce correct rankings."""
        p1 = AlgorithmPortfolio()
        p2 = AlgorithmPortfolio()
        fp = _make_fp()

        # p1 has data showing "fast" wins
        for _ in range(5):
            p1.record_outcome(fp, "fast", _winning_outcome())

        # p2 has data showing "slow" loses
        for _ in range(5):
            p2.record_outcome(fp, "slow", _losing_outcome())

        p1.merge(p2)

        stats = p1.rank_backends(fp, ["fast", "slow"])
        assert stats.portfolio_rank[0] == "fast"


class TestBackendRecord:
    """Test BackendRecord dataclass."""

    def test_default_values(self):
        rec = BackendRecord(fingerprint_key="test", backend_name="bo")
        assert rec.n_uses == 0
        assert rec.win_count == 0
        assert rec.avg_convergence_speed == 0.0

    def test_to_dict_from_dict_roundtrip(self):
        rec = BackendRecord(
            fingerprint_key="fp1",
            backend_name="tpe",
            n_uses=5,
            win_count=3,
            avg_convergence_speed=0.75,
            avg_regret=0.1,
            failure_rate=0.05,
            sample_efficiency=0.9,
        )
        data = rec.to_dict()
        restored = BackendRecord.from_dict(data)
        assert restored.fingerprint_key == rec.fingerprint_key
        assert restored.backend_name == rec.backend_name
        assert restored.n_uses == rec.n_uses
        assert restored.win_count == rec.win_count
        assert abs(restored.avg_convergence_speed - rec.avg_convergence_speed) < 1e-9


class TestPortfolioStats:
    """Test PortfolioStats dataclass."""

    def test_default_values(self):
        stats = PortfolioStats()
        assert stats.portfolio_rank == []
        assert stats.expected_gain == {}
        assert stats.risk_penalty == {}
        assert stats.confidence == {}

    def test_populated(self):
        stats = PortfolioStats(
            portfolio_rank=["a", "b"],
            expected_gain={"a": 0.9, "b": 0.3},
            risk_penalty={"a": 0.1, "b": 0.5},
            confidence={"a": 1.0, "b": 0.5},
        )
        assert stats.portfolio_rank[0] == "a"
        assert stats.expected_gain["a"] == 0.9


class TestFingerprintKey:
    """Test that fingerprint serialization is stable and distinct."""

    def test_same_fingerprint_same_key(self):
        fp1 = _make_fp()
        fp2 = _make_fp()
        assert _fingerprint_key(fp1) == _fingerprint_key(fp2)

    def test_different_fingerprint_different_key(self):
        fp1 = _make_fp(variable_types=VariableType.CONTINUOUS)
        fp2 = _make_fp(variable_types=VariableType.DISCRETE)
        assert _fingerprint_key(fp1) != _fingerprint_key(fp2)
