"""Tests for the counterfactual evaluation module."""

from __future__ import annotations

from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.plugins.base import AlgorithmPlugin
from optimization_copilot.counterfactual.evaluator import (
    CounterfactualEvaluator,
    CounterfactualResult,
    _nearest_observation,
    _normalised_distance,
)


# ---------------------------------------------------------------------------
# Stub plugin for testing
# ---------------------------------------------------------------------------

class _MonotoneSuggestPlugin(AlgorithmPlugin):
    """Toy plugin that always suggests the midpoint of the search space.

    Useful for deterministic, reproducible tests.
    """

    def __init__(self, name_str: str = "midpoint") -> None:
        self._name = name_str
        self._specs: list[ParameterSpec] = []

    def name(self) -> str:
        return self._name

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = parameter_specs

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        suggestion: dict[str, Any] = {}
        for spec in self._specs:
            if spec.type == VariableType.CATEGORICAL:
                suggestion[spec.name] = (
                    spec.categories[0] if spec.categories else "a"
                )
            else:
                lo = spec.lower if spec.lower is not None else 0.0
                hi = spec.upper if spec.upper is not None else 1.0
                suggestion[spec.name] = (lo + hi) / 2.0
        return [suggestion] * n_suggestions

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": True,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": False,
            "requires_observations": False,
            "max_dimensions": None,
        }


class _GreedyBestPlugin(AlgorithmPlugin):
    """Plugin that always suggests the parameters of the best observation so far."""

    def __init__(self, name_str: str = "greedy") -> None:
        self._name = name_str
        self._best_params: dict[str, Any] = {}
        self._specs: list[ParameterSpec] = []

    def name(self) -> str:
        return self._name

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = parameter_specs
        if observations:
            best = max(
                (o for o in observations if not o.is_failure),
                key=lambda o: list(o.kpi_values.values())[0],
                default=observations[0],
            )
            self._best_params = dict(best.parameters)

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        if not self._best_params:
            return [{}]
        return [dict(self._best_params)] * n_suggestions

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": False,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": False,
            "requires_observations": True,
            "max_dimensions": None,
        }


# ---------------------------------------------------------------------------
# Helpers to build test data
# ---------------------------------------------------------------------------

def _make_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
    ]


def _make_snapshot(
    n_obs: int = 10,
    direction: str = "maximize",
) -> CampaignSnapshot:
    """Create a snapshot with monotonically improving KPI values.

    KPI values go from 1.0 up to n_obs (maximise), making the last
    observation the best.
    """
    specs = _make_specs()
    obs: list[Observation] = []
    for i in range(n_obs):
        obs.append(
            Observation(
                iteration=i,
                parameters={"x1": i / max(n_obs - 1, 1), "x2": 0.5},
                kpi_values={"y": float(i + 1)},
                qc_passed=True,
                is_failure=False,
                timestamp=float(i),
            )
        )
    return CampaignSnapshot(
        campaign_id="cf-test",
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=[direction],
        current_iteration=n_obs,
    )


# ---------------------------------------------------------------------------
# Tests: _nearest_observation
# ---------------------------------------------------------------------------

class TestNearestObservation:
    def test_exact_match(self):
        specs = _make_specs()
        obs = [
            Observation(iteration=0, parameters={"x1": 0.0, "x2": 0.0},
                        kpi_values={"y": 1.0}),
            Observation(iteration=1, parameters={"x1": 1.0, "x2": 1.0},
                        kpi_values={"y": 2.0}),
        ]
        nearest = _nearest_observation({"x1": 1.0, "x2": 1.0}, obs, specs)
        assert nearest.iteration == 1

    def test_closer_match(self):
        specs = _make_specs()
        obs = [
            Observation(iteration=0, parameters={"x1": 0.0, "x2": 0.0},
                        kpi_values={"y": 1.0}),
            Observation(iteration=1, parameters={"x1": 0.8, "x2": 0.8},
                        kpi_values={"y": 2.0}),
            Observation(iteration=2, parameters={"x1": 0.3, "x2": 0.3},
                        kpi_values={"y": 3.0}),
        ]
        nearest = _nearest_observation({"x1": 0.75, "x2": 0.75}, obs, specs)
        assert nearest.iteration == 1

    def test_categorical_match(self):
        specs = [
            ParameterSpec(name="cat", type=VariableType.CATEGORICAL,
                          categories=["a", "b", "c"]),
        ]
        obs = [
            Observation(iteration=0, parameters={"cat": "a"},
                        kpi_values={"y": 1.0}),
            Observation(iteration=1, parameters={"cat": "b"},
                        kpi_values={"y": 2.0}),
        ]
        nearest = _nearest_observation({"cat": "b"}, obs, specs)
        assert nearest.iteration == 1

    def test_empty_raises(self):
        specs = _make_specs()
        try:
            _nearest_observation({"x1": 0.5}, [], specs)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_normalised_distance_identical(self):
        specs = _make_specs()
        obs = Observation(iteration=0, parameters={"x1": 0.5, "x2": 0.5},
                          kpi_values={"y": 1.0})
        d = _normalised_distance({"x1": 0.5, "x2": 0.5}, obs, specs)
        assert d == 0.0

    def test_normalised_distance_max(self):
        specs = _make_specs()
        obs = Observation(iteration=0, parameters={"x1": 0.0, "x2": 0.0},
                          kpi_values={"y": 1.0})
        d = _normalised_distance({"x1": 1.0, "x2": 1.0}, obs, specs)
        assert d == 1.0  # sqrt((1^2 + 1^2) / 2) = 1.0


# ---------------------------------------------------------------------------
# Tests: evaluate_replay with simple monotone data
# ---------------------------------------------------------------------------

class TestEvaluateReplay:
    def test_basic_replay_runs(self):
        snapshot = _make_snapshot(n_obs=10, direction="maximize")
        evaluator = CounterfactualEvaluator()
        plugin = _MonotoneSuggestPlugin("alt-midpoint")

        result = evaluator.evaluate_replay(
            snapshot=snapshot,
            actual_backend="original",
            alternative_backend="alt-midpoint",
            alternative_plugin=plugin,
            seed=42,
        )
        assert isinstance(result, CounterfactualResult)
        assert result.baseline_backend == "original"
        assert result.alternative_backend == "alt-midpoint"
        assert result.method == "replay"
        assert 0.0 <= result.confidence <= 1.0

    def test_baseline_best_kpi_is_correct(self):
        snapshot = _make_snapshot(n_obs=10, direction="maximize")
        evaluator = CounterfactualEvaluator()
        plugin = _MonotoneSuggestPlugin()

        result = evaluator.evaluate_replay(
            snapshot=snapshot,
            actual_backend="orig",
            alternative_backend="mid",
            alternative_plugin=plugin,
        )
        # With monotone data 1..10, the best maximize KPI is 10.0.
        assert result.baseline_best_kpi == 10.0

    def test_greedy_plugin_matches_or_beats_baseline(self):
        """A greedy plugin replaying on monotone data should find the
        best point relatively fast."""
        snapshot = _make_snapshot(n_obs=15, direction="maximize")
        evaluator = CounterfactualEvaluator()
        plugin = _GreedyBestPlugin("greedy")

        result = evaluator.evaluate_replay(
            snapshot=snapshot,
            actual_backend="orig",
            alternative_backend="greedy",
            alternative_plugin=plugin,
        )
        # The greedy plugin is inherently one step behind (it can only suggest
        # params from the history it has seen, never the final observation),
        # so it achieves close to but not necessarily equal to the baseline best.
        assert result.estimated_alternative_kpi >= result.baseline_best_kpi * 0.9

    def test_speedup_is_finite(self):
        snapshot = _make_snapshot(n_obs=10, direction="maximize")
        evaluator = CounterfactualEvaluator()
        plugin = _MonotoneSuggestPlugin()

        result = evaluator.evaluate_replay(
            snapshot=snapshot,
            actual_backend="orig",
            alternative_backend="mid",
            alternative_plugin=plugin,
        )
        assert result.estimated_speedup is not None
        assert not (result.estimated_speedup != result.estimated_speedup)  # not NaN

    def test_minimize_direction(self):
        """Test with minimize objective: KPI decreases from 10 to 1."""
        specs = _make_specs()
        obs = [
            Observation(
                iteration=i,
                parameters={"x1": i / 9.0, "x2": 0.5},
                kpi_values={"y": 10.0 - i},
                timestamp=float(i),
            )
            for i in range(10)
        ]
        snapshot = CampaignSnapshot(
            campaign_id="min-test",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["minimize"],
            current_iteration=10,
        )
        evaluator = CounterfactualEvaluator()
        plugin = _MonotoneSuggestPlugin()

        result = evaluator.evaluate_replay(
            snapshot=snapshot,
            actual_backend="orig",
            alternative_backend="mid",
            alternative_plugin=plugin,
        )
        assert result.baseline_best_kpi == 1.0
        assert result.method == "replay"


# ---------------------------------------------------------------------------
# Tests: insufficient data
# ---------------------------------------------------------------------------

class TestInsufficientData:
    def test_low_observation_count_gives_low_confidence(self):
        snapshot = _make_snapshot(n_obs=3, direction="maximize")
        evaluator = CounterfactualEvaluator()
        plugin = _MonotoneSuggestPlugin()

        result = evaluator.evaluate_replay(
            snapshot=snapshot,
            actual_backend="orig",
            alternative_backend="mid",
            alternative_plugin=plugin,
        )
        # 3 observations is below MIN_OBSERVATIONS (5), so we get very low confidence.
        assert result.confidence < 0.3
        assert result.details.get("reason") == "insufficient_data"

    def test_zero_observations(self):
        specs = _make_specs()
        snapshot = CampaignSnapshot(
            campaign_id="empty",
            parameter_specs=specs,
            observations=[],
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=0,
        )
        evaluator = CounterfactualEvaluator()
        plugin = _MonotoneSuggestPlugin()

        result = evaluator.evaluate_replay(
            snapshot=snapshot,
            actual_backend="orig",
            alternative_backend="mid",
            alternative_plugin=plugin,
        )
        assert result.confidence == 0.0
        assert result.details.get("reason") == "insufficient_data"


# ---------------------------------------------------------------------------
# Tests: evaluate_all_alternatives
# ---------------------------------------------------------------------------

class TestEvaluateAllAlternatives:
    def test_returns_one_result_per_alternative(self):
        snapshot = _make_snapshot(n_obs=10, direction="maximize")
        evaluator = CounterfactualEvaluator()
        alternatives = {
            "midpoint": _MonotoneSuggestPlugin("midpoint"),
            "greedy": _GreedyBestPlugin("greedy"),
        }

        results = evaluator.evaluate_all_alternatives(
            snapshot=snapshot,
            actual_backend="orig",
            alternatives=alternatives,
        )
        assert len(results) == 2
        names = {r.alternative_backend for r in results}
        assert names == {"midpoint", "greedy"}

    def test_empty_alternatives(self):
        snapshot = _make_snapshot(n_obs=10, direction="maximize")
        evaluator = CounterfactualEvaluator()

        results = evaluator.evaluate_all_alternatives(
            snapshot=snapshot,
            actual_backend="orig",
            alternatives={},
        )
        assert results == []


# ---------------------------------------------------------------------------
# Tests: summary_sentence
# ---------------------------------------------------------------------------

class TestSummarySentence:
    def test_low_confidence_sentence(self):
        result = CounterfactualResult(
            baseline_backend="A",
            alternative_backend="B",
            baseline_best_kpi=10.0,
            estimated_alternative_kpi=10.0,
            estimated_speedup=0.0,
            confidence=0.1,
            method="replay",
        )
        sentence = CounterfactualEvaluator.summary_sentence(result)
        assert "Insufficient data" in sentence
        assert "B" in sentence
        assert "A" in sentence

    def test_faster_sentence(self):
        result = CounterfactualResult(
            baseline_backend="A",
            alternative_backend="B",
            baseline_best_kpi=10.0,
            estimated_alternative_kpi=10.0,
            estimated_speedup=0.18,
            confidence=0.7,
            method="replay",
        )
        sentence = CounterfactualEvaluator.summary_sentence(result)
        assert "faster" in sentence
        assert "18%" in sentence
        assert "B" in sentence

    def test_slower_sentence(self):
        result = CounterfactualResult(
            baseline_backend="A",
            alternative_backend="B",
            baseline_best_kpi=10.0,
            estimated_alternative_kpi=8.0,
            estimated_speedup=-0.25,
            confidence=0.8,
            method="replay",
        )
        sentence = CounterfactualEvaluator.summary_sentence(result)
        assert "slower" in sentence
        assert "25%" in sentence

    def test_similar_sentence(self):
        result = CounterfactualResult(
            baseline_backend="A",
            alternative_backend="B",
            baseline_best_kpi=10.0,
            estimated_alternative_kpi=10.0,
            estimated_speedup=0.005,
            confidence=0.6,
            method="replay",
        )
        sentence = CounterfactualEvaluator.summary_sentence(result)
        assert "similarly" in sentence

    def test_confidence_in_sentence(self):
        result = CounterfactualResult(
            baseline_backend="A",
            alternative_backend="B",
            baseline_best_kpi=10.0,
            estimated_alternative_kpi=10.0,
            estimated_speedup=0.18,
            confidence=0.7,
            method="replay",
        )
        sentence = CounterfactualEvaluator.summary_sentence(result)
        assert "confidence" in sentence
        assert "70%" in sentence
