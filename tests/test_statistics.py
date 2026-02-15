"""Tests for the pure-Python statistical testing module."""

from __future__ import annotations

import math
import warnings

import pytest

from optimization_copilot.case_studies.statistics import (
    compute_effect_size,
    paired_comparison_table,
    rank_strategies,
    wilcoxon_signed_rank_test,
)


# ---------------------------------------------------------------------------
# wilcoxon_signed_rank_test
# ---------------------------------------------------------------------------


class TestWilcoxonIdentical:
    """Identical paired samples should yield a high (non-significant) p-value."""

    def test_wilcoxon_identical(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        y = list(x)
        result = wilcoxon_signed_rank_test(x, y)
        # All differences are zero -> n_effective == 0 -> p_value == 1.0
        assert result["p_value"] == 1.0
        assert result["n_effective"] == 0


class TestWilcoxonClearlyDifferent:
    """Very different paired samples should yield p < 0.05."""

    def test_wilcoxon_clearly_different(self):
        x = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = wilcoxon_signed_rank_test(x, y)
        assert result["p_value"] < 0.05


class TestWilcoxonSymmetric:
    """test(x, y) statistic should equal test(y, x) statistic."""

    def test_wilcoxon_symmetric(self):
        x = [3.0, 5.0, 7.0, 9.0, 11.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        r1 = wilcoxon_signed_rank_test(x, y)
        r2 = wilcoxon_signed_rank_test(y, x)
        assert r1["statistic"] == r2["statistic"]
        assert abs(r1["p_value"] - r2["p_value"]) < 1e-10


class TestWilcoxonEffectSizeRange:
    """Effect size r should be in [0, 1] (or near that range)."""

    def test_wilcoxon_effect_size_range(self):
        x = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = wilcoxon_signed_rank_test(x, y)
        assert 0.0 <= result["effect_size"]
        # r = |z| / sqrt(n) can theoretically exceed 1, but for moderate
        # samples it should be bounded.
        assert result["effect_size"] <= 2.0


class TestWilcoxonNEffective:
    """n_effective should be the count of non-zero differences."""

    def test_wilcoxon_n_effective(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = wilcoxon_signed_rank_test(x, y)
        # First 5 are zero-difference, last 5 are nonzero
        assert result["n_effective"] == 5


class TestWilcoxonWithTies:
    """Should handle tied absolute differences gracefully."""

    def test_wilcoxon_with_ties(self):
        x = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        y = [4.0, 4.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0]
        # All |d| = 1, half positive half negative
        result = wilcoxon_signed_rank_test(x, y)
        # Should not error, p_value should be high (not significant)
        assert result["p_value"] > 0.05
        assert result["n_effective"] == 10


class TestWilcoxonTooSmallSample:
    """Should still compute for n < 10 but may warn."""

    def test_wilcoxon_too_small_sample(self):
        x = [3.0, 5.0, 7.0]
        y = [1.0, 2.0, 3.0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wilcoxon_signed_rank_test(x, y)
            assert len(w) == 1
            assert "normal approximation" in str(w[0].message).lower()
        assert "p_value" in result
        assert result["n_effective"] == 3


class TestWilcoxonSinglePair:
    """Edge case: n=1 should still return a result."""

    def test_wilcoxon_single_pair(self):
        x = [5.0]
        y = [3.0]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = wilcoxon_signed_rank_test(x, y)
        assert result["n_effective"] == 1
        assert "statistic" in result


class TestWilcoxonAllZeros:
    """All differences zero -> n_effective=0, p_value=1."""

    def test_wilcoxon_all_zeros(self):
        x = [5.0, 5.0, 5.0, 5.0, 5.0]
        y = [5.0, 5.0, 5.0, 5.0, 5.0]
        result = wilcoxon_signed_rank_test(x, y)
        assert result["n_effective"] == 0
        assert result["p_value"] == 1.0
        assert result["effect_size"] == 0.0


class TestWilcoxonMismatchedLengths:
    """Mismatched lengths should raise ValueError."""

    def test_mismatched(self):
        with pytest.raises(ValueError, match="same length"):
            wilcoxon_signed_rank_test([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# compute_effect_size (Cohen's d)
# ---------------------------------------------------------------------------


class TestEffectSizeIdentical:
    """Identical samples should produce effect size ~0."""

    def test_compute_effect_size_identical(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = list(x)
        d = compute_effect_size(x, y)
        assert d == 0.0


class TestEffectSizeLargeDiff:
    """Very different samples should produce effect size > 0.5."""

    def test_compute_effect_size_large_diff(self):
        x = [10.0, 20.0, 30.0, 40.0, 50.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = compute_effect_size(x, y)
        assert d > 0.5


class TestEffectSizeNonNegative:
    """Cohen's d should always be >= 0."""

    def test_non_negative(self):
        x = [1.0, 2.0, 3.0]
        y = [10.0, 20.0, 30.0]
        d = compute_effect_size(x, y)
        assert d >= 0.0


# ---------------------------------------------------------------------------
# paired_comparison_table
# ---------------------------------------------------------------------------


class TestPairedComparisonTable3Strategies:
    """All pairs should be present in the output."""

    def test_paired_comparison_table_3_strategies(self):
        results = {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "B": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            "C": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        }
        table = paired_comparison_table(results)
        assert "A" in table
        assert "B" in table["A"]
        assert "C" in table["A"]
        assert "A" in table["B"]
        assert "p_value" in table["A"]["B"]


class TestPairedComparisonTableWinner:
    """Winner should be the strategy with lower median."""

    def test_paired_comparison_table_winner(self):
        results = {
            "good": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "bad": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
        }
        table = paired_comparison_table(results)
        assert table["good"]["bad"]["winner"] == "good"
        assert table["bad"]["good"]["winner"] == "good"


class TestPairedComparisonTableSelf:
    """Diagonal (self-comparison) should be tie with p=1."""

    def test_paired_comparison_table_self(self):
        results = {
            "X": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "Y": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        }
        table = paired_comparison_table(results)
        assert table["X"]["X"]["winner"] == "tie"
        assert table["X"]["X"]["p_value"] == 1.0


# ---------------------------------------------------------------------------
# rank_strategies
# ---------------------------------------------------------------------------


class TestRankStrategiesMinimize:
    """Lower median should rank first for minimize."""

    def test_rank_strategies_minimize(self):
        results = {
            "A": [5.0, 6.0, 7.0],
            "B": [1.0, 2.0, 3.0],
            "C": [10.0, 11.0, 12.0],
        }
        ranked = rank_strategies(results, direction="minimize")
        assert ranked[0][0] == "B"
        assert ranked[-1][0] == "C"


class TestRankStrategiesMaximize:
    """Higher median should rank first for maximize."""

    def test_rank_strategies_maximize(self):
        results = {
            "A": [5.0, 6.0, 7.0],
            "B": [1.0, 2.0, 3.0],
            "C": [10.0, 11.0, 12.0],
        }
        ranked = rank_strategies(results, direction="maximize")
        assert ranked[0][0] == "C"
        assert ranked[-1][0] == "B"


class TestRankStrategiesInvalidDirection:
    """Invalid direction should raise ValueError."""

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="direction"):
            rank_strategies({"A": [1.0]}, direction="up")


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


class TestWilcoxonLargerSample:
    """Larger sample with clear difference should produce low p-value."""

    def test_larger_sample(self):
        n = 30
        x = [float(i * 2) for i in range(n)]
        y = [float(i) for i in range(n)]
        result = wilcoxon_signed_rank_test(x, y)
        assert result["p_value"] < 0.01
        assert result["n_effective"] == n - 1  # diff at i=0 is zero


class TestEffectSizeMismatchedLengths:
    """Mismatched lengths should raise ValueError."""

    def test_mismatched(self):
        with pytest.raises(ValueError):
            compute_effect_size([1.0], [1.0, 2.0])


class TestPairedComparisonTableEmpty:
    """Empty results should return empty table."""

    def test_empty(self):
        table = paired_comparison_table({})
        assert table == {}


class TestRankStrategiesSingle:
    """Single strategy should return a list of one."""

    def test_single(self):
        ranked = rank_strategies({"A": [3.0, 4.0, 5.0]})
        assert len(ranked) == 1
        assert ranked[0][0] == "A"
        assert ranked[0][1] == 4.0
