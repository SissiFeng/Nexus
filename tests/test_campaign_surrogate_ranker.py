"""Comprehensive tests for campaign surrogate and ranker modules.

Tests cover:
- FingerprintSurrogate: init, fit, predict, loo_errors, edge cases
- CandidateRanker: UCB/EI/PI strategies, minimize/maximize, edge cases
- Data models: SurrogateFitResult, PredictionResult, RankedCandidate, RankedTable
"""

from __future__ import annotations

import math

import pytest

from optimization_copilot.campaign.surrogate import (
    FingerprintSurrogate,
    PredictionResult,
    SurrogateFitResult,
)
from optimization_copilot.campaign.ranker import (
    CandidateRanker,
    RankedCandidate,
    RankedTable,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SMILES_SMALL = ["CC", "CCC", "CCCC", "C=C", "C#C"]
Y_SMALL = [1.0, 2.0, 3.0, 1.5, 2.5]

SMILES_LARGE = ["CC", "CCC", "CCCC", "C=C", "C#C", "CCO", "CC=O", "c1ccccc1"]
Y_LARGE = [1.0, 2.0, 3.0, 1.5, 2.5, 4.0, 3.5, 5.0]

CANDIDATE_SMILES = ["CCCCC", "CC=CC", "CCCO"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def surrogate() -> FingerprintSurrogate:
    """Default surrogate with standard parameters."""
    return FingerprintSurrogate(n_gram=3, fp_size=128, length_scale=1.0, noise=1e-4, seed=42)


@pytest.fixture
def fitted_surrogate(surrogate: FingerprintSurrogate) -> FingerprintSurrogate:
    """Surrogate already fitted on SMILES_SMALL / Y_SMALL."""
    surrogate.fit(SMILES_SMALL, Y_SMALL, objective_name="yield")
    return surrogate


@pytest.fixture
def ranker() -> CandidateRanker:
    """Default ranker instance."""
    return CandidateRanker()


# ===========================================================================
# TestFingerprintSurrogate
# ===========================================================================


class TestFingerprintSurrogate:
    """Tests for FingerprintSurrogate."""

    # --- Initialization ---

    def test_init_defaults(self) -> None:
        """Default constructor creates an unfitted model."""
        s = FingerprintSurrogate()
        assert s.fitted is False
        assert s.fit_result is None

    def test_init_custom_params(self) -> None:
        """Custom hyperparameters are stored correctly."""
        s = FingerprintSurrogate(n_gram=2, fp_size=64, length_scale=0.5, noise=1e-3, seed=99)
        assert s.fitted is False
        assert s.fit_result is None

    # --- Fit validation errors ---

    def test_fit_fewer_than_two_observations(self, surrogate: FingerprintSurrogate) -> None:
        """fit() rejects fewer than 2 observations."""
        with pytest.raises(ValueError, match="at least 2"):
            surrogate.fit(["CC"], [1.0])

    def test_fit_empty_input(self, surrogate: FingerprintSurrogate) -> None:
        """fit() rejects empty input lists."""
        with pytest.raises(ValueError, match="at least 2"):
            surrogate.fit([], [])

    def test_fit_mismatched_lengths(self, surrogate: FingerprintSurrogate) -> None:
        """fit() rejects mismatched SMILES and y-value lengths."""
        with pytest.raises(ValueError, match="Length mismatch"):
            surrogate.fit(["CC", "CCC"], [1.0, 2.0, 3.0])

    def test_fit_mismatched_lengths_reverse(self, surrogate: FingerprintSurrogate) -> None:
        """fit() rejects more SMILES than y-values."""
        with pytest.raises(ValueError, match="Length mismatch"):
            surrogate.fit(["CC", "CCC", "CCCC"], [1.0, 2.0])

    # --- Successful fit ---

    def test_fit_returns_surrogate_fit_result(self, surrogate: FingerprintSurrogate) -> None:
        """fit() returns a SurrogateFitResult with correct fields."""
        result = surrogate.fit(SMILES_SMALL, Y_SMALL, objective_name="yield")
        assert isinstance(result, SurrogateFitResult)
        assert result.n_training == len(SMILES_SMALL)
        assert result.n_features == 128
        assert result.objective_name == "yield"
        assert result.duration_ms >= 0.0

    def test_fit_sets_fitted_flag(self, surrogate: FingerprintSurrogate) -> None:
        """fit() sets the fitted property to True."""
        assert surrogate.fitted is False
        surrogate.fit(SMILES_SMALL, Y_SMALL)
        assert surrogate.fitted is True

    def test_fit_result_property(self, surrogate: FingerprintSurrogate) -> None:
        """fit_result property returns the most recent result."""
        result = surrogate.fit(SMILES_SMALL, Y_SMALL, objective_name="test_obj")
        assert surrogate.fit_result is result

    def test_fit_computes_y_statistics(self, surrogate: FingerprintSurrogate) -> None:
        """fit() computes correct mean and std of y-values."""
        result = surrogate.fit(SMILES_SMALL, Y_SMALL)
        expected_mean = sum(Y_SMALL) / len(Y_SMALL)
        assert math.isclose(result.y_mean, expected_mean, rel_tol=1e-9)
        # std is computed with n-1 denominator
        var = sum((y - expected_mean) ** 2 for y in Y_SMALL) / (len(Y_SMALL) - 1)
        expected_std = math.sqrt(var)
        assert math.isclose(result.y_std, expected_std, rel_tol=1e-9)

    def test_fit_minimum_two_observations(self, surrogate: FingerprintSurrogate) -> None:
        """fit() succeeds with exactly 2 observations (minimum)."""
        result = surrogate.fit(["CC", "CCC"], [1.0, 2.0])
        assert result.n_training == 2
        assert surrogate.fitted is True

    def test_fit_constant_y_values(self, surrogate: FingerprintSurrogate) -> None:
        """fit() handles constant y-values (zero variance) gracefully."""
        result = surrogate.fit(["CC", "CCC", "CCCC"], [5.0, 5.0, 5.0])
        assert result.n_training == 3
        # When variance is near zero, y_std should default to 1.0
        assert math.isclose(result.y_std, 1.0, rel_tol=1e-9)

    def test_fit_with_large_dataset(self, surrogate: FingerprintSurrogate) -> None:
        """fit() works with the larger SMILES set."""
        result = surrogate.fit(SMILES_LARGE, Y_LARGE, objective_name="activity")
        assert result.n_training == len(SMILES_LARGE)
        assert result.objective_name == "activity"

    # --- Predict ---

    def test_predict_before_fit_raises(self, surrogate: FingerprintSurrogate) -> None:
        """predict() raises RuntimeError if model not fitted."""
        with pytest.raises(RuntimeError, match="not fitted"):
            surrogate.predict(CANDIDATE_SMILES)

    def test_predict_returns_correct_count(
        self, fitted_surrogate: FingerprintSurrogate
    ) -> None:
        """predict() returns one PredictionResult per candidate."""
        preds = fitted_surrogate.predict(CANDIDATE_SMILES)
        assert len(preds) == len(CANDIDATE_SMILES)

    def test_predict_returns_prediction_result_type(
        self, fitted_surrogate: FingerprintSurrogate
    ) -> None:
        """predict() returns PredictionResult instances."""
        preds = fitted_surrogate.predict(CANDIDATE_SMILES)
        for p in preds:
            assert isinstance(p, PredictionResult)

    def test_predict_finite_mean(self, fitted_surrogate: FingerprintSurrogate) -> None:
        """Predicted means are finite numbers."""
        preds = fitted_surrogate.predict(CANDIDATE_SMILES)
        for p in preds:
            assert math.isfinite(p.mean), f"Non-finite mean for {p.smiles}: {p.mean}"

    def test_predict_positive_std(self, fitted_surrogate: FingerprintSurrogate) -> None:
        """Predicted stds are positive (finite) numbers."""
        preds = fitted_surrogate.predict(CANDIDATE_SMILES)
        for p in preds:
            assert p.std > 0, f"Non-positive std for {p.smiles}: {p.std}"
            assert math.isfinite(p.std), f"Non-finite std for {p.smiles}: {p.std}"

    def test_predict_preserves_smiles(self, fitted_surrogate: FingerprintSurrogate) -> None:
        """Predicted results preserve the original SMILES strings."""
        preds = fitted_surrogate.predict(CANDIDATE_SMILES)
        for pred, smi in zip(preds, CANDIDATE_SMILES):
            assert pred.smiles == smi

    def test_predict_training_points_near_observed(
        self, fitted_surrogate: FingerprintSurrogate
    ) -> None:
        """Predictions on training points should be close to observed values."""
        preds = fitted_surrogate.predict(SMILES_SMALL)
        for pred, y_obs in zip(preds, Y_SMALL):
            # GP should interpolate near training points
            assert abs(pred.mean - y_obs) < 1.0, (
                f"Prediction {pred.mean} too far from observed {y_obs} for {pred.smiles}"
            )

    def test_predict_training_points_low_std(
        self, fitted_surrogate: FingerprintSurrogate
    ) -> None:
        """Standard deviation at training points should be small."""
        preds = fitted_surrogate.predict(SMILES_SMALL)
        for pred in preds:
            # At training points, GP uncertainty should be low
            assert pred.std < 1.0, (
                f"Unexpectedly high std {pred.std} at training point {pred.smiles}"
            )

    def test_predict_single_candidate(self, fitted_surrogate: FingerprintSurrogate) -> None:
        """predict() works with a single-element candidate list."""
        preds = fitted_surrogate.predict(["CCCCCC"])
        assert len(preds) == 1
        assert math.isfinite(preds[0].mean)
        assert preds[0].std > 0

    def test_predict_empty_list(self, fitted_surrogate: FingerprintSurrogate) -> None:
        """predict() returns empty list for empty input."""
        preds = fitted_surrogate.predict([])
        assert preds == []

    # --- Leave-one-out errors ---

    def test_loo_errors_returns_correct_count(
        self, surrogate: FingerprintSurrogate
    ) -> None:
        """loo_errors returns one entry per observation."""
        errors = surrogate.loo_errors(SMILES_SMALL, Y_SMALL)
        assert len(errors) == len(SMILES_SMALL)

    def test_loo_errors_dict_structure(self, surrogate: FingerprintSurrogate) -> None:
        """Each LOO error dict has the expected keys."""
        errors = surrogate.loo_errors(SMILES_SMALL, Y_SMALL)
        for err in errors:
            assert "smiles" in err
            assert "actual" in err
            assert "predicted" in err
            assert "error" in err

    def test_loo_errors_too_few_returns_empty(
        self, surrogate: FingerprintSurrogate
    ) -> None:
        """loo_errors returns empty list for fewer than 3 observations."""
        errors = surrogate.loo_errors(["CC", "CCC"], [1.0, 2.0])
        assert errors == []

    def test_loo_errors_error_is_actual_minus_predicted(
        self, surrogate: FingerprintSurrogate
    ) -> None:
        """error field equals actual - predicted."""
        errors = surrogate.loo_errors(SMILES_SMALL, Y_SMALL)
        for err in errors:
            if err["predicted"] is not None and err["error"] is not None:
                assert math.isclose(
                    err["error"],
                    err["actual"] - err["predicted"],
                    rel_tol=1e-9,
                )

    # --- Refit behaviour ---

    def test_refit_overwrites_previous(self, surrogate: FingerprintSurrogate) -> None:
        """Fitting a second time overwrites the previous model."""
        r1 = surrogate.fit(SMILES_SMALL[:3], Y_SMALL[:3], objective_name="first")
        assert r1.n_training == 3
        r2 = surrogate.fit(SMILES_LARGE, Y_LARGE, objective_name="second")
        assert r2.n_training == len(SMILES_LARGE)
        assert surrogate.fit_result is r2
        assert surrogate.fit_result.objective_name == "second"


# ===========================================================================
# TestCandidateRanker
# ===========================================================================


class TestCandidateRanker:
    """Tests for CandidateRanker."""

    # --- Shared helpers ---

    @staticmethod
    def _make_predictions(
        means: list[float], stds: list[float]
    ) -> list[tuple[float, float]]:
        return list(zip(means, stds))

    @staticmethod
    def _make_names(n: int) -> list[str]:
        return [f"mol_{i}" for i in range(n)]

    @staticmethod
    def _make_params(n: int) -> list[dict]:
        return [{"smiles": f"C{'C' * i}"} for i in range(n)]

    # --- Validation errors ---

    def test_rank_mismatched_names_params(self, ranker: CandidateRanker) -> None:
        """rank() rejects mismatched name and param lengths."""
        with pytest.raises(ValueError, match="Mismatched lengths"):
            ranker.rank(
                candidate_names=["a", "b"],
                candidate_params=[{"x": 1}],
                predictions=[(1.0, 0.1), (2.0, 0.2)],
                objective_name="obj",
            )

    def test_rank_mismatched_predictions(self, ranker: CandidateRanker) -> None:
        """rank() rejects mismatched prediction count."""
        with pytest.raises(ValueError, match="Mismatched lengths"):
            ranker.rank(
                candidate_names=["a", "b"],
                candidate_params=[{"x": 1}, {"x": 2}],
                predictions=[(1.0, 0.1)],
                objective_name="obj",
            )

    def test_rank_unknown_strategy(self, ranker: CandidateRanker) -> None:
        """rank() rejects unknown acquisition strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            ranker.rank(
                candidate_names=["a"],
                candidate_params=[{"x": 1}],
                predictions=[(1.0, 0.1)],
                objective_name="obj",
                strategy="unknown",
            )

    # --- UCB strategy ---

    def test_ucb_minimize_ranks_low_mean_first(self, ranker: CandidateRanker) -> None:
        """UCB minimize: candidates with lower predicted mean rank higher."""
        names = ["high", "low", "mid"]
        params = [{"v": 3}, {"v": 1}, {"v": 2}]
        # Equal stds so ranking is driven by mean
        preds = [(3.0, 0.1), (1.0, 0.1), (2.0, 0.1)]
        table = ranker.rank(
            candidate_names=names,
            candidate_params=params,
            predictions=preds,
            objective_name="cost",
            direction="minimize",
            strategy="ucb",
        )
        assert table.candidates[0].name == "low"
        assert table.candidates[1].name == "mid"
        assert table.candidates[2].name == "high"

    def test_ucb_maximize_ranks_high_mean_first(self, ranker: CandidateRanker) -> None:
        """UCB maximize: candidates with higher predicted mean rank higher."""
        names = ["high", "low", "mid"]
        params = [{"v": 3}, {"v": 1}, {"v": 2}]
        # Equal stds, so ranking driven by negated mean
        preds = [(3.0, 0.1), (1.0, 0.1), (2.0, 0.1)]
        table = ranker.rank(
            candidate_names=names,
            candidate_params=params,
            predictions=preds,
            objective_name="yield",
            direction="maximize",
            strategy="ucb",
        )
        # Maximize: negate means then UCB = -mu - kappa*sigma; lower is better
        # For high: -3.0 - 2*0.1 = -3.2; mid: -2.0 - 0.2 = -2.2; low: -1.0 - 0.2 = -1.2
        # Ascending sort: high(-3.2) < mid(-2.2) < low(-1.2)
        assert table.candidates[0].name == "high"
        assert table.candidates[-1].name == "low"

    def test_ucb_exploration_prefers_uncertain(self, ranker: CandidateRanker) -> None:
        """UCB with high kappa prefers candidates with higher uncertainty."""
        names = ["certain", "uncertain"]
        params = [{"v": 1}, {"v": 2}]
        # Same mean but very different stds
        preds = [(2.0, 0.01), (2.0, 2.0)]
        table = ranker.rank(
            candidate_names=names,
            candidate_params=params,
            predictions=preds,
            objective_name="cost",
            direction="minimize",
            strategy="ucb",
            kappa=10.0,  # high exploration
        )
        # UCB = mu - kappa*sigma: certain=2.0-0.1=1.9, uncertain=2.0-20.0=-18.0
        # Lower is better -> uncertain ranks first
        assert table.candidates[0].name == "uncertain"

    # --- EI strategy ---

    def test_ei_minimize_prefers_improvement(self, ranker: CandidateRanker) -> None:
        """EI minimize: candidate with lower mean has higher EI."""
        names = ["worse", "better"]
        params = [{"v": 1}, {"v": 2}]
        preds = [(5.0, 1.0), (0.5, 1.0)]
        table = ranker.rank(
            candidate_names=names,
            candidate_params=params,
            predictions=preds,
            objective_name="cost",
            direction="minimize",
            strategy="ei",
            best_observed=2.0,
        )
        # EI is higher for the candidate more likely to improve on best
        assert table.candidates[0].name == "better"

    def test_ei_maximize_prefers_improvement(self, ranker: CandidateRanker) -> None:
        """EI maximize: candidate with higher mean has higher EI."""
        names = ["low", "high"]
        params = [{"v": 1}, {"v": 2}]
        preds = [(1.0, 1.0), (5.0, 1.0)]
        table = ranker.rank(
            candidate_names=names,
            candidate_params=params,
            predictions=preds,
            objective_name="yield",
            direction="maximize",
            strategy="ei",
            best_observed=2.0,
        )
        assert table.candidates[0].name == "high"

    def test_ei_without_best_observed_returns_zero_scores(
        self, ranker: CandidateRanker
    ) -> None:
        """EI with no best_observed yields all-zero acquisition scores."""
        names = ["a", "b"]
        params = [{"v": 1}, {"v": 2}]
        preds = [(1.0, 1.0), (2.0, 1.0)]
        table = ranker.rank(
            candidate_names=names,
            candidate_params=params,
            predictions=preds,
            objective_name="obj",
            strategy="ei",
            best_observed=None,
        )
        for c in table.candidates:
            assert c.acquisition_score == 0.0

    # --- PI strategy ---

    def test_pi_minimize_prefers_improvement(self, ranker: CandidateRanker) -> None:
        """PI minimize: candidate with lower mean has higher PI."""
        names = ["worse", "better"]
        params = [{"v": 1}, {"v": 2}]
        preds = [(5.0, 1.0), (0.5, 1.0)]
        table = ranker.rank(
            candidate_names=names,
            candidate_params=params,
            predictions=preds,
            objective_name="cost",
            direction="minimize",
            strategy="pi",
            best_observed=2.0,
        )
        assert table.candidates[0].name == "better"

    def test_pi_without_best_observed_returns_zero_scores(
        self, ranker: CandidateRanker
    ) -> None:
        """PI with no best_observed yields all-zero acquisition scores."""
        names = ["a", "b"]
        params = [{"v": 1}, {"v": 2}]
        preds = [(1.0, 1.0), (2.0, 1.0)]
        table = ranker.rank(
            candidate_names=names,
            candidate_params=params,
            predictions=preds,
            objective_name="obj",
            strategy="pi",
            best_observed=None,
        )
        for c in table.candidates:
            assert c.acquisition_score == 0.0

    # --- RankedTable structure ---

    def test_ranked_table_metadata(self, ranker: CandidateRanker) -> None:
        """RankedTable carries correct metadata."""
        names = ["a", "b", "c"]
        params = [{"x": i} for i in range(3)]
        preds = [(1.0, 0.1), (2.0, 0.2), (3.0, 0.3)]
        table = ranker.rank(
            candidate_names=names,
            candidate_params=params,
            predictions=preds,
            objective_name="yield",
            direction="maximize",
            strategy="ucb",
            best_observed=1.5,
        )
        assert table.objective_name == "yield"
        assert table.direction == "maximize"
        assert table.acquisition_strategy == "ucb"
        assert table.best_observed == 1.5
        assert table.n_candidates == 3

    def test_ranked_table_ranks_are_sequential(self, ranker: CandidateRanker) -> None:
        """Ranks in the table are 1, 2, 3, ..., n."""
        names = self._make_names(5)
        params = self._make_params(5)
        preds = self._make_predictions(
            [1.0, 2.0, 3.0, 4.0, 5.0], [0.1] * 5
        )
        table = ranker.rank(
            candidate_names=names,
            candidate_params=params,
            predictions=preds,
            objective_name="obj",
            direction="minimize",
            strategy="ucb",
        )
        ranks = [c.rank for c in table.candidates]
        assert ranks == [1, 2, 3, 4, 5]

    def test_ranked_table_top_n(self, ranker: CandidateRanker) -> None:
        """top_n() returns the first n candidates."""
        names = self._make_names(5)
        params = self._make_params(5)
        preds = self._make_predictions([5.0, 4.0, 3.0, 2.0, 1.0], [0.1] * 5)
        table = ranker.rank(
            candidate_names=names,
            candidate_params=params,
            predictions=preds,
            objective_name="obj",
            direction="minimize",
            strategy="ucb",
        )
        top3 = table.top_n(3)
        assert len(top3) == 3
        assert all(c.rank <= 3 for c in top3)

    def test_ranked_table_top_n_exceeds_total(self, ranker: CandidateRanker) -> None:
        """top_n(k) with k > n_candidates returns all candidates."""
        names = self._make_names(3)
        params = self._make_params(3)
        preds = self._make_predictions([1.0, 2.0, 3.0], [0.1] * 3)
        table = ranker.rank(
            candidate_names=names,
            candidate_params=params,
            predictions=preds,
            objective_name="obj",
            direction="minimize",
            strategy="ucb",
        )
        top10 = table.top_n(10)
        assert len(top10) == 3

    def test_ranked_candidate_preserves_original_predictions(
        self, ranker: CandidateRanker
    ) -> None:
        """RankedCandidate stores the original (un-negated) mean and std."""
        names = ["a"]
        params = [{"x": 1}]
        preds = [(3.5, 0.7)]
        table = ranker.rank(
            candidate_names=names,
            candidate_params=params,
            predictions=preds,
            objective_name="yield",
            direction="maximize",
            strategy="ucb",
        )
        c = table.candidates[0]
        assert c.predicted_mean == 3.5
        assert c.predicted_std == 0.7

    def test_rank_single_candidate(self, ranker: CandidateRanker) -> None:
        """Ranking a single candidate works."""
        table = ranker.rank(
            candidate_names=["only"],
            candidate_params=[{"x": 1}],
            predictions=[(2.0, 0.5)],
            objective_name="obj",
            direction="minimize",
            strategy="ucb",
        )
        assert table.n_candidates == 1
        assert table.candidates[0].rank == 1
        assert table.candidates[0].name == "only"

    def test_rank_all_strategies_produce_valid_tables(
        self, ranker: CandidateRanker
    ) -> None:
        """All three strategies produce valid ranked tables."""
        names = self._make_names(4)
        params = self._make_params(4)
        preds = self._make_predictions([1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5])
        for strat in ("ucb", "ei", "pi"):
            table = ranker.rank(
                candidate_names=names,
                candidate_params=params,
                predictions=preds,
                objective_name="obj",
                direction="minimize",
                strategy=strat,
                best_observed=2.5,
            )
            assert table.n_candidates == 4
            assert table.acquisition_strategy == strat
            ranks = [c.rank for c in table.candidates]
            assert sorted(ranks) == [1, 2, 3, 4]


# ===========================================================================
# TestDataModels
# ===========================================================================


class TestDataModels:
    """Tests for data model serialization and structure."""

    # --- SurrogateFitResult ---

    def test_surrogate_fit_result_to_dict(self) -> None:
        """SurrogateFitResult.to_dict() contains all expected fields."""
        r = SurrogateFitResult(
            n_training=10,
            y_mean=2.5,
            y_std=1.1,
            n_features=128,
            objective_name="yield",
            duration_ms=12.3,
        )
        d = r.to_dict()
        assert d["n_training"] == 10
        assert d["y_mean"] == 2.5
        assert d["y_std"] == 1.1
        assert d["n_features"] == 128
        assert d["objective_name"] == "yield"
        assert d["duration_ms"] == 12.3

    def test_surrogate_fit_result_to_dict_keys(self) -> None:
        """SurrogateFitResult.to_dict() has exactly the expected keys."""
        r = SurrogateFitResult(
            n_training=5, y_mean=1.0, y_std=0.5, n_features=64
        )
        expected_keys = {
            "n_training", "y_mean", "y_std", "n_features",
            "objective_name", "duration_ms",
        }
        assert set(r.to_dict().keys()) == expected_keys

    def test_surrogate_fit_result_default_values(self) -> None:
        """SurrogateFitResult defaults are correct."""
        r = SurrogateFitResult(n_training=3, y_mean=0.0, y_std=1.0, n_features=32)
        assert r.objective_name == ""
        assert r.duration_ms == 0.0

    # --- PredictionResult ---

    def test_prediction_result_to_dict(self) -> None:
        """PredictionResult.to_dict() returns correct fields."""
        p = PredictionResult(smiles="CCO", mean=2.5, std=0.3)
        d = p.to_dict()
        assert d == {"smiles": "CCO", "mean": 2.5, "std": 0.3}

    def test_prediction_result_to_dict_keys(self) -> None:
        """PredictionResult.to_dict() has exactly the expected keys."""
        p = PredictionResult(smiles="CC", mean=1.0, std=0.1)
        assert set(p.to_dict().keys()) == {"smiles", "mean", "std"}

    # --- RankedCandidate ---

    def test_ranked_candidate_to_dict(self) -> None:
        """RankedCandidate.to_dict() returns all fields."""
        c = RankedCandidate(
            rank=1,
            name="mol_A",
            parameters={"smiles": "CC", "conc": 0.5},
            predicted_mean=3.2,
            predicted_std=0.4,
            acquisition_score=-1.5,
        )
        d = c.to_dict()
        assert d["rank"] == 1
        assert d["name"] == "mol_A"
        assert d["parameters"] == {"smiles": "CC", "conc": 0.5}
        assert d["predicted_mean"] == 3.2
        assert d["predicted_std"] == 0.4
        assert d["acquisition_score"] == -1.5

    def test_ranked_candidate_to_dict_keys(self) -> None:
        """RankedCandidate.to_dict() has exactly the expected keys."""
        c = RankedCandidate(
            rank=1, name="x", parameters={},
            predicted_mean=0.0, predicted_std=0.0, acquisition_score=0.0,
        )
        expected_keys = {
            "rank", "name", "parameters",
            "predicted_mean", "predicted_std", "acquisition_score",
        }
        assert set(c.to_dict().keys()) == expected_keys

    # --- RankedTable ---

    def test_ranked_table_to_dict(self) -> None:
        """RankedTable.to_dict() includes candidates and metadata."""
        c1 = RankedCandidate(
            rank=1, name="a", parameters={"x": 1},
            predicted_mean=1.0, predicted_std=0.1, acquisition_score=0.9,
        )
        c2 = RankedCandidate(
            rank=2, name="b", parameters={"x": 2},
            predicted_mean=2.0, predicted_std=0.2, acquisition_score=0.5,
        )
        table = RankedTable(
            candidates=[c1, c2],
            objective_name="yield",
            direction="maximize",
            acquisition_strategy="ei",
            best_observed=1.5,
        )
        d = table.to_dict()
        assert d["objective_name"] == "yield"
        assert d["direction"] == "maximize"
        assert d["acquisition_strategy"] == "ei"
        assert d["best_observed"] == 1.5
        assert d["n_candidates"] == 2
        assert len(d["candidates"]) == 2
        assert d["candidates"][0]["rank"] == 1

    def test_ranked_table_to_dict_keys(self) -> None:
        """RankedTable.to_dict() has exactly the expected keys."""
        table = RankedTable(
            candidates=[],
            objective_name="obj",
            direction="minimize",
            acquisition_strategy="ucb",
        )
        expected_keys = {
            "candidates", "objective_name", "direction",
            "acquisition_strategy", "best_observed", "n_candidates",
        }
        assert set(table.to_dict().keys()) == expected_keys

    def test_ranked_table_n_candidates_property(self) -> None:
        """n_candidates property returns the correct count."""
        table = RankedTable(
            candidates=[
                RankedCandidate(
                    rank=i, name=f"c{i}", parameters={},
                    predicted_mean=0.0, predicted_std=0.0, acquisition_score=0.0,
                )
                for i in range(1, 6)
            ],
            objective_name="obj",
            direction="minimize",
            acquisition_strategy="ucb",
        )
        assert table.n_candidates == 5

    # --- Serialization round-trips ---

    def test_surrogate_fit_result_roundtrip(self) -> None:
        """SurrogateFitResult can be round-tripped through to_dict."""
        original = SurrogateFitResult(
            n_training=7,
            y_mean=3.14,
            y_std=1.59,
            n_features=256,
            objective_name="binding_affinity",
            duration_ms=42.0,
        )
        d = original.to_dict()
        restored = SurrogateFitResult(**d)
        assert restored == original

    def test_prediction_result_roundtrip(self) -> None:
        """PredictionResult can be round-tripped through to_dict."""
        original = PredictionResult(smiles="c1ccccc1", mean=4.2, std=0.8)
        d = original.to_dict()
        restored = PredictionResult(**d)
        assert restored == original

    def test_ranked_candidate_roundtrip(self) -> None:
        """RankedCandidate can be round-tripped through to_dict."""
        original = RankedCandidate(
            rank=3,
            name="benzene",
            parameters={"smiles": "c1ccccc1", "mw": 78.11},
            predicted_mean=5.5,
            predicted_std=1.2,
            acquisition_score=0.73,
        )
        d = original.to_dict()
        restored = RankedCandidate(**d)
        assert restored == original
