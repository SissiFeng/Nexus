"""Tests for Layer 4: Decision Robustness Analysis."""

from __future__ import annotations

import math
import random

from optimization_copilot.robustness.models import (
    BootstrapResult,
    ConclusionRobustness,
    RobustnessReport,
)
from optimization_copilot.robustness.bootstrap import BootstrapAnalyzer
from optimization_copilot.robustness.conclusion import ConclusionRobustnessChecker
from optimization_copilot.robustness.sensitivity import DecisionSensitivityAnalyzer
from optimization_copilot.robustness.consistency import CrossModelConsistency


# ── Helpers ────────────────────────────────────────────────────────────


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


# ── TestBootstrapAnalyzer ──────────────────────────────────────────────


class TestBootstrapAnalyzer:
    """Tests for BootstrapAnalyzer."""

    def test_ci_contains_known_mean(self):
        """Bootstrap CI of mean([1,2,3,4,5]) should contain 3.0."""
        random.seed(42)
        analyzer = BootstrapAnalyzer(n_bootstrap=2000, seed=42)
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = analyzer.bootstrap_ci(data, _mean, confidence=0.95)

        assert result.ci_lower <= 3.0 <= result.ci_upper
        assert abs(result.observed_value - 3.0) < 1e-10
        assert result.confidence_level == 0.95
        assert result.n_bootstrap == 2000
        assert len(result.bootstrap_distribution) == 2000
        assert result.std_error > 0

    def test_ci_narrow_for_constant_data(self):
        """Constant data should give a very narrow CI."""
        analyzer = BootstrapAnalyzer(n_bootstrap=500, seed=42)
        data = [5.0] * 20
        result = analyzer.bootstrap_ci(data, _mean, confidence=0.95)

        assert abs(result.observed_value - 5.0) < 1e-10
        assert abs(result.ci_lower - 5.0) < 1e-10
        assert abs(result.ci_upper - 5.0) < 1e-10
        assert result.std_error < 1e-10

    def test_bootstrap_top_k_stable_ranking(self):
        """Values far apart should give stable top-K."""
        analyzer = BootstrapAnalyzer(n_bootstrap=500, seed=42)
        values = [100.0, 50.0, 10.0, 5.0, 1.0]
        names = ["A", "B", "C", "D", "E"]
        result = analyzer.bootstrap_top_k(values, names, k=3)

        assert result["original_top_k"] == ["A", "B", "C"]
        # Top-1 should almost always be "A" since it's much higher
        assert result["stability_score"] > 0.9
        assert result["top_k_frequency"]["A"] > 0.9

    def test_bootstrap_top_k_unstable_ranking(self):
        """Nearly identical values should give unstable ranking."""
        analyzer = BootstrapAnalyzer(n_bootstrap=500, seed=42)
        values = [5.001, 5.000, 4.999]
        names = ["A", "B", "C"]
        result = analyzer.bootstrap_top_k(values, names, k=1)

        # With nearly identical values, top-1 should vary
        assert result["stability_score"] < 1.0

    def test_bootstrap_correlation_perfect(self):
        """Perfect correlation should have tight CI near 1.0."""
        analyzer = BootstrapAnalyzer(n_bootstrap=1000, seed=42)
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        result = analyzer.bootstrap_correlation(xs, ys)

        assert abs(result.observed_value - 1.0) < 1e-10
        assert result.ci_lower > 0.95
        assert result.ci_upper <= 1.0 + 1e-10
        assert result.statistic_name == "pearson_r"

    def test_bootstrap_correlation_zero(self):
        """Uncorrelated data should have CI containing 0."""
        analyzer = BootstrapAnalyzer(n_bootstrap=1000, seed=42)
        # Construct data with no linear relationship
        rng = random.Random(42)
        xs = [rng.gauss(0, 1) for _ in range(50)]
        ys = [rng.gauss(0, 1) for _ in range(50)]
        result = analyzer.bootstrap_correlation(xs, ys)

        # CI should be wide and close to 0
        assert result.ci_lower < 0.5
        assert result.ci_upper > -0.5

    def test_bootstrap_feature_importance(self):
        """Feature importance CIs should be non-negative and cover observed."""
        analyzer = BootstrapAnalyzer(n_bootstrap=200, seed=42)
        # x0 strongly correlated with y, x1 not
        rng = random.Random(42)
        n = 50
        X = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(n)]
        y = [X[i][0] * 2.0 + rng.gauss(0, 0.1) for i in range(n)]

        def importance_fn(X_: list[list[float]], y_: list[float]) -> dict[str, float]:
            n_ = len(y_)
            if n_ < 2:
                return {"x0": 0.0, "x1": 0.0}
            result = {}
            mean_y = sum(y_) / n_
            var_y = sum((yi - mean_y) ** 2 for yi in y_)
            for j, name in enumerate(["x0", "x1"]):
                col = [X_[i][j] for i in range(n_)]
                mean_x = sum(col) / n_
                var_x = sum((xi - mean_x) ** 2 for xi in col)
                if var_x < 1e-15 or var_y < 1e-15:
                    result[name] = 0.0
                    continue
                cov = sum((col[i] - mean_x) * (y_[i] - mean_y) for i in range(n_))
                result[name] = abs(cov / math.sqrt(var_x * var_y))
            return result

        results = analyzer.bootstrap_feature_importance(X, y, importance_fn)

        assert "x0" in results
        assert "x1" in results
        # x0 should have much higher importance
        assert results["x0"].observed_value > results["x1"].observed_value
        # CI should contain observed value
        assert results["x0"].ci_lower <= results["x0"].observed_value <= results["x0"].ci_upper

    def test_percentile(self):
        """Percentile function should return correct boundary values."""
        sorted_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert abs(BootstrapAnalyzer._percentile(sorted_vals, 0.0) - 1.0) < 1e-10
        assert abs(BootstrapAnalyzer._percentile(sorted_vals, 1.0) - 5.0) < 1e-10
        assert abs(BootstrapAnalyzer._percentile(sorted_vals, 0.5) - 3.0) < 1e-10

    def test_pearson_r_known_values(self):
        """Pearson r for known perfect positive correlation is 1.0."""
        r = BootstrapAnalyzer._pearson_r([1, 2, 3], [2, 4, 6])
        assert abs(r - 1.0) < 1e-10

    def test_pearson_r_negative(self):
        """Pearson r for perfect negative correlation is -1.0."""
        r = BootstrapAnalyzer._pearson_r([1, 2, 3], [6, 4, 2])
        assert abs(r - (-1.0)) < 1e-10


# ── TestConclusionRobustness ──────────────────────────────────────────


class TestConclusionRobustness:
    """Tests for ConclusionRobustnessChecker."""

    def test_stable_ranking(self):
        """Clear winner should have stability close to 1.0."""
        checker = ConclusionRobustnessChecker(n_bootstrap=500, seed=42)
        values = [10.0, 5.0, 1.0]
        names = ["A", "B", "C"]
        result = checker.check_ranking_stability(values, names, k=1)

        assert result.conclusion_type == "ranking"
        assert result.stability_score > 0.9
        assert result.n_bootstrap == 500
        assert "original_top_k" in result.details
        assert result.details["original_top_k"][0] == "A"

    def test_unstable_ranking(self):
        """Nearly identical values should yield lower stability than well-separated ones."""
        checker = ConclusionRobustnessChecker(n_bootstrap=500, seed=42)
        values = [5.001, 5.000, 4.999, 4.998, 4.997]
        names = ["A", "B", "C", "D", "E"]
        result = checker.check_ranking_stability(values, names, k=1)

        assert result.conclusion_type == "ranking"
        # With nearly identical values, stability should be noticeably below 1.0
        assert result.stability_score < 0.95
        # But compare to a well-separated case to confirm relative instability
        checker2 = ConclusionRobustnessChecker(n_bootstrap=500, seed=42)
        stable_result = checker2.check_ranking_stability(
            [100.0, 50.0, 10.0, 5.0, 1.0],
            ["A", "B", "C", "D", "E"],
            k=1,
        )
        assert result.stability_score < stable_result.stability_score

    def test_importance_stability_clear_signal(self):
        """When one feature dominates, importance ordering should be stable."""
        checker = ConclusionRobustnessChecker(n_bootstrap=200, seed=42)
        rng = random.Random(42)
        n = 100
        X = [[rng.gauss(0, 1), rng.gauss(0, 0.01)] for _ in range(n)]
        y = [X[i][0] * 5.0 + rng.gauss(0, 0.1) for i in range(n)]

        result = checker.check_importance_stability(X, y, var_names=["x0", "x1"])

        assert result.conclusion_type == "importance"
        assert result.stability_score > 0.7
        assert "original_ranking" in result.details
        assert result.details["original_ranking"][0] == "x0"

    def test_pareto_stability_clear_front(self):
        """With a clearly dominated set, Pareto front should be stable."""
        checker = ConclusionRobustnessChecker(n_bootstrap=300, seed=42)
        # A is clearly Pareto optimal (low in both), D is clearly dominated
        objectives = [
            [1.0, 1.0],   # A: Pareto optimal
            [5.0, 0.5],   # B: Pareto optimal
            [0.5, 5.0],   # C: Pareto optimal
            [10.0, 10.0], # D: dominated
        ]
        names = ["A", "B", "C", "D"]

        result = checker.check_pareto_stability(objectives, names)

        assert result.conclusion_type == "pareto"
        assert result.n_bootstrap == 300
        assert "original_pareto" in result.details
        assert "D" not in result.details["original_pareto"]
        # Stability should be reasonable for a clear front
        assert result.stability_score > 0.0

    def test_pareto_optimal_check(self):
        """_is_pareto_optimal correctly identifies dominated points."""
        points = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 4.0]]
        # [4, 4] is dominated by all others
        assert not ConclusionRobustnessChecker._is_pareto_optimal(
            points[3], points, minimize=True
        )
        # [1, 3] is not dominated (best in first objective)
        assert ConclusionRobustnessChecker._is_pareto_optimal(
            points[0], points, minimize=True
        )

    def test_comprehensive_robustness(self):
        """Comprehensive report should contain ranking and importance analyses."""
        checker = ConclusionRobustnessChecker(n_bootstrap=200, seed=42)
        rng = random.Random(42)
        n = 50
        values = [10.0, 5.0, 1.0]
        names = ["A", "B", "C"]
        X = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(n)]
        y = [X[i][0] * 3.0 + rng.gauss(0, 0.5) for i in range(n)]

        report = checker.comprehensive_robustness(values, names, X=X, y=y)

        assert isinstance(report, RobustnessReport)
        assert len(report.analyses) == 2  # ranking + importance
        assert report.overall_robustness > 0.0
        assert report.analyses[0].conclusion_type == "ranking"
        assert report.analyses[1].conclusion_type == "importance"

    def test_comprehensive_robustness_minimal(self):
        """Comprehensive report with minimal data still works."""
        checker = ConclusionRobustnessChecker(n_bootstrap=100, seed=42)
        report = checker.comprehensive_robustness([5.0, 3.0], ["A", "B"])

        assert isinstance(report, RobustnessReport)
        assert len(report.analyses) == 1  # ranking only
        assert report.analyses[0].conclusion_type == "ranking"


# ── TestDecisionSensitivity ───────────────────────────────────────────


class TestDecisionSensitivity:
    """Tests for DecisionSensitivityAnalyzer."""

    def test_clear_winner_high_stability(self):
        """One value much higher than others should give high stability."""
        analyzer = DecisionSensitivityAnalyzer(seed=42)
        values = [100.0, 10.0, 5.0, 1.0]
        names = ["A", "B", "C", "D"]
        result = analyzer.decision_sensitivity(
            values, names, noise_std=1.0, n_perturbations=500
        )

        assert result["stability"] > 0.9
        assert result["top1_frequency"]["A"] > 0.9
        assert result["mean_rank_change"] < 0.5

    def test_close_race_low_stability(self):
        """Similar values should give low stability."""
        analyzer = DecisionSensitivityAnalyzer(seed=42)
        values = [10.01, 10.00, 9.99, 9.98]
        names = ["A", "B", "C", "D"]
        result = analyzer.decision_sensitivity(
            values, names, noise_std=1.0, n_perturbations=500
        )

        assert result["stability"] < 0.5
        # Mean rank change should be non-trivial
        assert result["mean_rank_change"] > 0.0

    def test_recommendation_confidence_clear(self):
        """Clear top-K items should have high confidence."""
        analyzer = DecisionSensitivityAnalyzer(seed=42)
        values = [100.0, 80.0, 60.0, 1.0, 0.5]
        names = ["A", "B", "C", "D", "E"]
        conf = analyzer.recommendation_confidence(
            values, names, k=3, noise_std=1.0, n_perturbations=500
        )

        assert conf["A"] > 0.95
        assert conf["B"] > 0.95
        assert conf["C"] > 0.90
        assert conf["D"] < 0.3
        assert conf["E"] < 0.3

    def test_value_at_risk_returns_reasonable_quantile(self):
        """VaR should be lower than expected best for risk-averse quantile."""
        analyzer = DecisionSensitivityAnalyzer(seed=42)
        values = [10.0, 8.0, 5.0]
        uncertainties = [2.0, 1.0, 0.5]
        result = analyzer.value_at_risk(
            values, uncertainties, quantile=0.05, n_samples=2000
        )

        # VaR (5th percentile of best) should be less than expected best
        assert result["var"] < result["expected_best"]
        assert result["var"] > 0.0  # should still be positive
        assert len(result["best_values"]) == 2000

    def test_value_at_risk_zero_uncertainty(self):
        """Zero uncertainty should give VaR equal to max value."""
        analyzer = DecisionSensitivityAnalyzer(seed=42)
        values = [10.0, 5.0, 1.0]
        uncertainties = [0.0, 0.0, 0.0]
        result = analyzer.value_at_risk(values, uncertainties, quantile=0.05)

        assert abs(result["var"] - 10.0) < 1e-10
        assert abs(result["expected_best"] - 10.0) < 1e-10

    def test_empty_input(self):
        """Empty values should return graceful defaults."""
        analyzer = DecisionSensitivityAnalyzer(seed=42)
        result = analyzer.decision_sensitivity([], [], noise_std=0.1)
        assert result["stability"] == 0.0


# ── TestCrossModelConsistency ─────────────────────────────────────────


class TestCrossModelConsistency:
    """Tests for CrossModelConsistency."""

    def test_kendall_tau_perfect_agreement(self):
        """Same ranking should give tau = 1.0."""
        checker = CrossModelConsistency()
        ranking = ["A", "B", "C", "D", "E"]
        tau = checker.kendall_tau(ranking, ranking)
        assert abs(tau - 1.0) < 1e-10

    def test_kendall_tau_reverse_ranking(self):
        """Reversed ranking should give tau = -1.0."""
        checker = CrossModelConsistency()
        ranking1 = ["A", "B", "C", "D"]
        ranking2 = ["D", "C", "B", "A"]
        tau = checker.kendall_tau(ranking1, ranking2)
        assert abs(tau - (-1.0)) < 1e-10

    def test_kendall_tau_partial_swap(self):
        """One adjacent swap should give tau close to but less than 1.0."""
        checker = CrossModelConsistency()
        ranking1 = ["A", "B", "C", "D"]
        ranking2 = ["A", "C", "B", "D"]  # swap B and C
        tau = checker.kendall_tau(ranking1, ranking2)
        # 5 concordant, 1 discordant: (5-1)/6 = 0.667
        assert abs(tau - (2.0 / 3.0)) < 1e-10

    def test_model_agreement_three_models(self):
        """Three models with varying agreement."""
        checker = CrossModelConsistency()
        model_rankings = {
            "GP": ["A", "B", "C", "D"],
            "RF": ["A", "B", "C", "D"],  # same as GP
            "TPE": ["D", "C", "B", "A"],  # reversed
        }
        result = checker.model_agreement(model_rankings)

        # GP-RF: tau = 1.0, GP-TPE: tau = -1.0, RF-TPE: tau = -1.0
        assert abs(result["pairwise"][("GP", "RF")] - 1.0) < 1e-10
        assert abs(result["pairwise"][("GP", "TPE")] - (-1.0)) < 1e-10
        assert abs(result["pairwise"][("RF", "TPE")] - (-1.0)) < 1e-10
        # Mean: (1 + -1 + -1) / 3 = -1/3
        assert abs(result["mean_agreement"] - (-1.0 / 3.0)) < 1e-10

    def test_model_agreement_all_same(self):
        """All models identical should give mean agreement = 1.0."""
        checker = CrossModelConsistency()
        model_rankings = {
            "M1": ["A", "B", "C"],
            "M2": ["A", "B", "C"],
            "M3": ["A", "B", "C"],
        }
        result = checker.model_agreement(model_rankings)
        assert abs(result["mean_agreement"] - 1.0) < 1e-10

    def test_ensemble_confidence(self):
        """Models that agree should have high agreement score."""
        checker = CrossModelConsistency()
        model_predictions = {
            "GP": [10.0, 5.0, 1.0],
            "RF": [10.1, 4.9, 1.1],
            "TPE": [9.9, 5.1, 0.9],
        }
        names = ["A", "B", "C"]
        result = checker.ensemble_confidence(model_predictions, names)

        assert len(result["per_item"]) == 3
        assert result["overall_agreement"] > 0.5
        # All predictions are close, so agreement should be high
        for item in result["per_item"]:
            assert item["agreement_score"] > 0.5
            assert item["std"] < 1.0

    def test_ensemble_confidence_disagreement(self):
        """Wildly different predictions should give low agreement."""
        checker = CrossModelConsistency()
        model_predictions = {
            "GP": [100.0, 1.0],
            "RF": [1.0, 100.0],
        }
        names = ["A", "B"]
        result = checker.ensemble_confidence(model_predictions, names)

        # Both items have large disagreement
        for item in result["per_item"]:
            assert item["std"] > 10.0
            assert item["agreement_score"] < 0.1

    def test_disagreement_regions(self):
        """Regions with high model disagreement should be identified."""
        checker = CrossModelConsistency()
        model_predictions = {
            "GP": [10.0, 5.0, 1.0, 50.0],
            "RF": [10.0, 5.0, 1.0, 1.0],   # disagrees on item 3
        }
        X = [[1.0], [2.0], [3.0], [4.0]]
        names = ["A", "B", "C", "D"]
        regions = checker.disagreement_regions(model_predictions, X, names)

        assert len(regions) == 4
        # D should be first (highest disagreement)
        assert regions[0]["name"] == "D"
        assert regions[0]["disagreement_score"] > 10.0
        # A, B, C should have zero disagreement
        assert regions[-1]["disagreement_score"] < 1e-10

    def test_disagreement_regions_no_names(self):
        """Disagreement regions should auto-generate names."""
        checker = CrossModelConsistency()
        model_predictions = {
            "GP": [10.0, 5.0],
            "RF": [10.0, 5.0],
        }
        X = [[1.0], [2.0]]
        regions = checker.disagreement_regions(model_predictions, X)

        assert len(regions) == 2
        assert regions[0]["name"] in ("item_0", "item_1")

    def test_kendall_tau_empty(self):
        """Empty rankings should return 0.0."""
        checker = CrossModelConsistency()
        assert checker.kendall_tau([], []) == 0.0

    def test_kendall_tau_single_item(self):
        """Single-item ranking should return 0.0 (no pairs)."""
        checker = CrossModelConsistency()
        assert checker.kendall_tau(["A"], ["A"]) == 0.0


# ── TestDataclasses ───────────────────────────────────────────────────


class TestDataclasses:
    """Tests for data models."""

    def test_bootstrap_result_defaults(self):
        """BootstrapResult should have sensible defaults."""
        result = BootstrapResult(
            statistic_name="mean",
            observed_value=3.0,
            ci_lower=2.0,
            ci_upper=4.0,
            confidence_level=0.95,
            n_bootstrap=1000,
        )
        assert result.bootstrap_distribution == []
        assert result.std_error == 0.0

    def test_conclusion_robustness_defaults(self):
        """ConclusionRobustness should have default empty details."""
        cr = ConclusionRobustness(
            conclusion_type="ranking",
            stability_score=0.9,
            n_bootstrap=1000,
        )
        assert cr.details == {}

    def test_robustness_report_defaults(self):
        """RobustnessReport should have sensible defaults."""
        report = RobustnessReport()
        assert report.analyses == []
        assert report.overall_robustness == 0.0
        assert report.warnings == []
