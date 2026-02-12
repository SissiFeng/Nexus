"""Comprehensive tests for the deterministic imputation module."""

from __future__ import annotations

import random

from optimization_copilot.core.models import Observation, ParameterSpec, VariableType
from optimization_copilot.imputation import (
    DeterministicImputer,
    ImputationConfig,
    ImputationRecord,
    ImputationResult,
    ImputationStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_observations(
    n_good: int = 15,
    n_failures: int = 5,
) -> tuple[list[Observation], list[str], list[ParameterSpec]]:
    """Create a reproducible set of observations for testing."""
    rng = random.Random(42)
    specs = [
        ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 10.0),
        ParameterSpec("y", VariableType.CONTINUOUS, -1.0, 1.0),
    ]
    obj_names = ["kpi_a", "kpi_b"]
    obs: list[Observation] = []
    for i in range(n_good):
        obs.append(Observation(
            iteration=i,
            parameters={"x": rng.uniform(0, 10), "y": rng.uniform(-1, 1)},
            kpi_values={"kpi_a": rng.uniform(0, 100), "kpi_b": rng.uniform(0, 50)},
        ))
    for i in range(n_failures):
        obs.append(Observation(
            iteration=n_good + i,
            parameters={"x": rng.uniform(0, 10), "y": rng.uniform(-1, 1)},
            kpi_values={},
            is_failure=True,
            failure_reason="experiment_failed",
        ))
    return obs, obj_names, specs


def _make_simple_obs(
    kpi_values: dict[str, float] | None = None,
    is_failure: bool = False,
    iteration: int = 0,
    params: dict | None = None,
    metadata: dict | None = None,
) -> Observation:
    """Quick helper for building single observations."""
    return Observation(
        iteration=iteration,
        parameters=params or {"x": 1.0, "y": 0.0},
        kpi_values=kpi_values or {},
        is_failure=is_failure,
        failure_reason="failed" if is_failure else None,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# TestImputationModels
# ---------------------------------------------------------------------------

class TestImputationModels:
    """Tests for the data models in imputation.models."""

    def test_strategy_enum_values(self) -> None:
        assert ImputationStrategy.WORST_VALUE.value == "worst_value"
        assert ImputationStrategy.COLUMN_MEDIAN.value == "column_median"
        assert ImputationStrategy.COLUMN_MEAN.value == "column_mean"
        assert ImputationStrategy.KNN_PROXY.value == "knn_proxy"

    def test_strategy_enum_membership(self) -> None:
        assert len(ImputationStrategy) == 4

    def test_config_defaults(self) -> None:
        cfg = ImputationConfig()
        assert cfg.strategy == ImputationStrategy.WORST_VALUE
        assert cfg.seed == 42
        assert cfg.knn_k == 3
        assert cfg.per_kpi_strategy is None

    def test_record_creation(self) -> None:
        rec = ImputationRecord(
            observation_index=5,
            kpi_name="yield",
            original_value=None,
            imputed_value=3.14,
            strategy=ImputationStrategy.COLUMN_MEAN,
            source_columns=["yield"],
        )
        assert rec.observation_index == 5
        assert rec.imputed_value == 3.14
        assert rec.k_neighbors is None
        assert rec.neighbor_indices is None

    def test_result_creation(self) -> None:
        result = ImputationResult(
            observations=[],
            records=[],
            decision_hash="abc123",
            config_used=ImputationConfig(),
        )
        assert result.decision_hash == "abc123"
        assert result.config_used.strategy == ImputationStrategy.WORST_VALUE

    def test_per_kpi_strategy_override(self) -> None:
        cfg = ImputationConfig(
            strategy=ImputationStrategy.WORST_VALUE,
            per_kpi_strategy={"yield": ImputationStrategy.COLUMN_MEDIAN},
        )
        assert cfg.per_kpi_strategy["yield"] == ImputationStrategy.COLUMN_MEDIAN
        assert cfg.strategy == ImputationStrategy.WORST_VALUE


# ---------------------------------------------------------------------------
# TestWorstValueImputation
# ---------------------------------------------------------------------------

class TestWorstValueImputation:
    """Tests for WORST_VALUE strategy."""

    def test_replaces_with_min(self) -> None:
        obs, obj_names, specs = _make_observations(n_good=5, n_failures=1)
        cfg = ImputationConfig(strategy=ImputationStrategy.WORST_VALUE)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, obj_names, specs)

        # The failure observation should have KPIs imputed
        failure_obs = [o for o in result.observations if o.is_failure]
        assert len(failure_obs) == 1

        # Compute expected worst for kpi_a
        valid_a = [o.kpi_values["kpi_a"] for o in obs if not o.is_failure]
        expected_worst_a = min(valid_a)
        assert failure_obs[0].kpi_values["kpi_a"] == expected_worst_a

    def test_no_valid_values_returns_zero(self) -> None:
        # All observations are failures
        obs = [_make_simple_obs(is_failure=True, iteration=i) for i in range(3)]
        cfg = ImputationConfig(strategy=ImputationStrategy.WORST_VALUE)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["kpi_a"], [])

        for o in result.observations:
            assert o.kpi_values.get("kpi_a", 0.0) == 0.0

    def test_single_valid_value(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"kpi_a": 7.5}, iteration=0),
            _make_simple_obs(is_failure=True, iteration=1),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.WORST_VALUE)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["kpi_a"], [])

        imputed_obs = result.observations[1]
        assert imputed_obs.kpi_values["kpi_a"] == 7.5

    def test_multiple_kpis_imputed_independently(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 10.0, "b": 100.0}, iteration=0),
            _make_simple_obs(kpi_values={"a": 20.0, "b": 50.0}, iteration=1),
            _make_simple_obs(is_failure=True, iteration=2),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.WORST_VALUE)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["a", "b"], [])

        imputed = result.observations[2]
        assert imputed.kpi_values["a"] == 10.0  # min of [10, 20]
        assert imputed.kpi_values["b"] == 50.0  # min of [100, 50]

    def test_worst_value_record_detail(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 3.0}, iteration=0),
            _make_simple_obs(kpi_values={"a": 5.0}, iteration=1),
            _make_simple_obs(is_failure=True, iteration=2),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.WORST_VALUE)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["a"], [])

        assert len(result.records) == 1
        assert result.records[0].strategy == ImputationStrategy.WORST_VALUE
        assert result.records[0].imputed_value == 3.0
        assert result.records[0].original_value is None


# ---------------------------------------------------------------------------
# TestColumnMedianImputation
# ---------------------------------------------------------------------------

class TestColumnMedianImputation:
    """Tests for COLUMN_MEDIAN strategy."""

    def test_odd_count_middle(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 1.0}, iteration=0),
            _make_simple_obs(kpi_values={"a": 3.0}, iteration=1),
            _make_simple_obs(kpi_values={"a": 5.0}, iteration=2),
            _make_simple_obs(is_failure=True, iteration=3),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.COLUMN_MEDIAN)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["a"], [])

        assert result.observations[3].kpi_values["a"] == 3.0

    def test_even_count_average_of_middles(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 1.0}, iteration=0),
            _make_simple_obs(kpi_values={"a": 3.0}, iteration=1),
            _make_simple_obs(kpi_values={"a": 5.0}, iteration=2),
            _make_simple_obs(kpi_values={"a": 7.0}, iteration=3),
            _make_simple_obs(is_failure=True, iteration=4),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.COLUMN_MEDIAN)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["a"], [])

        # median of [1, 3, 5, 7] = (3 + 5) / 2 = 4.0
        assert result.observations[4].kpi_values["a"] == 4.0

    def test_single_value(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 42.0}, iteration=0),
            _make_simple_obs(is_failure=True, iteration=1),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.COLUMN_MEDIAN)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["a"], [])

        assert result.observations[1].kpi_values["a"] == 42.0

    def test_already_sorted_values(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 10.0}, iteration=0),
            _make_simple_obs(kpi_values={"a": 20.0}, iteration=1),
            _make_simple_obs(kpi_values={"a": 30.0}, iteration=2),
            _make_simple_obs(kpi_values={"a": 40.0}, iteration=3),
            _make_simple_obs(kpi_values={"a": 50.0}, iteration=4),
            _make_simple_obs(is_failure=True, iteration=5),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.COLUMN_MEDIAN)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["a"], [])

        assert result.observations[5].kpi_values["a"] == 30.0


# ---------------------------------------------------------------------------
# TestColumnMeanImputation
# ---------------------------------------------------------------------------

class TestColumnMeanImputation:
    """Tests for COLUMN_MEAN strategy."""

    def test_correct_mean(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 2.0}, iteration=0),
            _make_simple_obs(kpi_values={"a": 4.0}, iteration=1),
            _make_simple_obs(kpi_values={"a": 6.0}, iteration=2),
            _make_simple_obs(is_failure=True, iteration=3),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.COLUMN_MEAN)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["a"], [])

        assert result.observations[3].kpi_values["a"] == 4.0

    def test_single_value(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 99.0}, iteration=0),
            _make_simple_obs(is_failure=True, iteration=1),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.COLUMN_MEAN)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["a"], [])

        assert result.observations[1].kpi_values["a"] == 99.0

    def test_empty_valid_values(self) -> None:
        obs = [_make_simple_obs(is_failure=True, iteration=0)]
        cfg = ImputationConfig(strategy=ImputationStrategy.COLUMN_MEAN)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["a"], [])

        assert result.observations[0].kpi_values.get("a", 0.0) == 0.0


# ---------------------------------------------------------------------------
# TestKnnProxyImputation
# ---------------------------------------------------------------------------

class TestKnnProxyImputation:
    """Tests for KNN_PROXY strategy."""

    def _make_knn_obs(self) -> tuple[list[Observation], list[str], list[ParameterSpec]]:
        """Create observations with known positions for KNN testing."""
        specs = [
            ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 10.0),
            ParameterSpec("y", VariableType.CONTINUOUS, 0.0, 10.0),
        ]
        obj_names = ["kpi"]
        obs = [
            # Good observations at known positions
            Observation(iteration=0, parameters={"x": 1.0, "y": 1.0}, kpi_values={"kpi": 10.0}),
            Observation(iteration=1, parameters={"x": 2.0, "y": 2.0}, kpi_values={"kpi": 20.0}),
            Observation(iteration=2, parameters={"x": 5.0, "y": 5.0}, kpi_values={"kpi": 50.0}),
            Observation(iteration=3, parameters={"x": 9.0, "y": 9.0}, kpi_values={"kpi": 90.0}),
            # Failure at (1.5, 1.5) -- closest to obs 0 and 1
            Observation(
                iteration=4, parameters={"x": 1.5, "y": 1.5},
                kpi_values={}, is_failure=True, failure_reason="failed",
            ),
        ]
        return obs, obj_names, specs

    def test_k1_nearest_neighbor(self) -> None:
        obs, obj_names, specs = self._make_knn_obs()
        cfg = ImputationConfig(strategy=ImputationStrategy.KNN_PROXY, knn_k=1)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, obj_names, specs)

        imputed = result.observations[4]
        # Nearest to (1.5, 1.5) is (1.0, 1.0) with kpi=10 or (2.0, 2.0) with kpi=20
        # Both are equidistant: dist = sqrt((0.05)^2 + (0.05)^2) = 0.0707
        # Tie-broken by original index, so obs 0 wins -> kpi = 10.0
        assert imputed.kpi_values["kpi"] == 10.0

    def test_k3_average_of_neighbors(self) -> None:
        obs, obj_names, specs = self._make_knn_obs()
        cfg = ImputationConfig(strategy=ImputationStrategy.KNN_PROXY, knn_k=3)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, obj_names, specs)

        imputed = result.observations[4]
        # 3 nearest: obs 0 (10), obs 1 (20), obs 2 (50) -> mean = 26.666...
        assert abs(imputed.kpi_values["kpi"] - (10.0 + 20.0 + 50.0) / 3) < 1e-9

    def test_k_greater_than_available(self) -> None:
        obs, obj_names, specs = self._make_knn_obs()
        cfg = ImputationConfig(strategy=ImputationStrategy.KNN_PROXY, knn_k=100)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, obj_names, specs)

        imputed = result.observations[4]
        # All 4 valid observations used
        expected = (10.0 + 20.0 + 50.0 + 90.0) / 4
        assert abs(imputed.kpi_values["kpi"] - expected) < 1e-9

    def test_categorical_parameter_distance(self) -> None:
        specs = [
            ParameterSpec("cat", VariableType.CATEGORICAL, categories=["a", "b", "c"]),
            ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 10.0),
        ]
        obs = [
            Observation(iteration=0, parameters={"cat": "a", "x": 5.0}, kpi_values={"kpi": 10.0}),
            Observation(iteration=1, parameters={"cat": "b", "x": 5.0}, kpi_values={"kpi": 30.0}),
            Observation(iteration=2, parameters={"cat": "a", "x": 5.0}, kpi_values={"kpi": 20.0}),
            Observation(
                iteration=3, parameters={"cat": "a", "x": 5.0},
                kpi_values={}, is_failure=True, failure_reason="failed",
            ),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.KNN_PROXY, knn_k=2)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["kpi"], specs)

        imputed = result.observations[3]
        # Nearest with matching "a": obs 0 (10) and obs 2 (20)
        # obs 1 has cat mismatch (distance += 1.0), so it's farther
        assert imputed.kpi_values["kpi"] == 15.0  # (10 + 20) / 2

    def test_continuous_parameter_normalization(self) -> None:
        specs = [
            ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 100.0),
        ]
        obs = [
            Observation(iteration=0, parameters={"x": 10.0}, kpi_values={"kpi": 1.0}),
            Observation(iteration=1, parameters={"x": 90.0}, kpi_values={"kpi": 9.0}),
            Observation(
                iteration=2, parameters={"x": 12.0},
                kpi_values={}, is_failure=True, failure_reason="failed",
            ),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.KNN_PROXY, knn_k=1)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["kpi"], specs)

        # x=12 normalized: 0.12; obs0 at 0.10, obs1 at 0.90
        # Nearest is obs0 -> kpi = 1.0
        assert result.observations[2].kpi_values["kpi"] == 1.0

    def test_mixed_parameter_types(self) -> None:
        specs = [
            ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 10.0),
            ParameterSpec("cat", VariableType.CATEGORICAL, categories=["a", "b"]),
        ]
        obs = [
            Observation(iteration=0, parameters={"x": 1.0, "cat": "a"}, kpi_values={"kpi": 10.0}),
            Observation(iteration=1, parameters={"x": 1.0, "cat": "b"}, kpi_values={"kpi": 90.0}),
            Observation(
                iteration=2, parameters={"x": 1.0, "cat": "a"},
                kpi_values={}, is_failure=True, failure_reason="failed",
            ),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.KNN_PROXY, knn_k=1)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["kpi"], specs)

        # obs0 same cat "a", x same -> distance 0. obs1 different cat -> distance 1.0
        assert result.observations[2].kpi_values["kpi"] == 10.0

    def test_single_non_missing_observation(self) -> None:
        specs = [ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 10.0)]
        obs = [
            Observation(iteration=0, parameters={"x": 5.0}, kpi_values={"kpi": 42.0}),
            Observation(
                iteration=1, parameters={"x": 3.0},
                kpi_values={}, is_failure=True, failure_reason="failed",
            ),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.KNN_PROXY, knn_k=3)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["kpi"], specs)

        assert result.observations[1].kpi_values["kpi"] == 42.0

    def test_neighbor_indices_reported(self) -> None:
        obs, obj_names, specs = self._make_knn_obs()
        cfg = ImputationConfig(strategy=ImputationStrategy.KNN_PROXY, knn_k=2)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, obj_names, specs)

        assert len(result.records) >= 1
        rec = result.records[0]
        assert rec.neighbor_indices is not None
        assert len(rec.neighbor_indices) == 2
        # The two nearest neighbors should be indices 0 and 1
        assert set(rec.neighbor_indices) == {0, 1}


# ---------------------------------------------------------------------------
# TestDeterministicHash (ACCEPTANCE CRITERIA)
# ---------------------------------------------------------------------------

class TestDeterministicHash:
    """Acceptance tests for deterministic reproducibility."""

    def test_same_snapshot_same_hash(self) -> None:
        obs, obj_names, specs = _make_observations()
        cfg = ImputationConfig()
        imputer = DeterministicImputer(cfg)

        result1 = imputer.impute(obs, obj_names, specs)
        result2 = imputer.impute(obs, obj_names, specs)

        assert result1.decision_hash == result2.decision_hash

    def test_different_seed_different_hash(self) -> None:
        obs, obj_names, specs = _make_observations()
        cfg1 = ImputationConfig(seed=42)
        cfg2 = ImputationConfig(seed=99)

        result1 = DeterministicImputer(cfg1).impute(obs, obj_names, specs)
        result2 = DeterministicImputer(cfg2).impute(obs, obj_names, specs)

        assert result1.decision_hash != result2.decision_hash

    def test_different_observations_different_hash(self) -> None:
        obs1, obj_names, specs = _make_observations(n_good=10, n_failures=2)
        obs2, _, _ = _make_observations(n_good=8, n_failures=4)
        cfg = ImputationConfig()

        result1 = DeterministicImputer(cfg).impute(obs1, obj_names, specs)
        result2 = DeterministicImputer(cfg).impute(obs2, obj_names, specs)

        assert result1.decision_hash != result2.decision_hash

    def test_hash_format(self) -> None:
        obs, obj_names, specs = _make_observations()
        cfg = ImputationConfig()
        result = DeterministicImputer(cfg).impute(obs, obj_names, specs)

        assert len(result.decision_hash) == 16
        # All hex characters
        assert all(c in "0123456789abcdef" for c in result.decision_hash)

    def test_multiple_runs_stable(self) -> None:
        obs, obj_names, specs = _make_observations()
        cfg = ImputationConfig()
        imputer = DeterministicImputer(cfg)

        hashes = [imputer.impute(obs, obj_names, specs).decision_hash for _ in range(5)]
        assert len(set(hashes)) == 1, f"Got different hashes: {hashes}"

    def test_config_change_changes_hash(self) -> None:
        obs, obj_names, specs = _make_observations()
        cfg_worst = ImputationConfig(strategy=ImputationStrategy.WORST_VALUE)
        cfg_mean = ImputationConfig(strategy=ImputationStrategy.COLUMN_MEAN)

        result1 = DeterministicImputer(cfg_worst).impute(obs, obj_names, specs)
        result2 = DeterministicImputer(cfg_mean).impute(obs, obj_names, specs)

        assert result1.decision_hash != result2.decision_hash


# ---------------------------------------------------------------------------
# TestMetadataTraceability
# ---------------------------------------------------------------------------

class TestMetadataTraceability:
    """Tests for metadata enrichment and traceability."""

    def test_imputed_flag_set(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 5.0}, iteration=0),
            _make_simple_obs(is_failure=True, iteration=1),
        ]
        cfg = ImputationConfig()
        result = DeterministicImputer(cfg).impute(obs, ["a"], [])

        assert result.observations[1].metadata["imputed"] is True

    def test_imputation_method_set(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 5.0}, iteration=0),
            _make_simple_obs(is_failure=True, iteration=1),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.COLUMN_MEDIAN)
        result = DeterministicImputer(cfg).impute(obs, ["a"], [])

        assert result.observations[1].metadata["imputation_method"] == "column_median"

    def test_source_columns_listed(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 1.0, "b": 2.0}, iteration=0),
            _make_simple_obs(is_failure=True, iteration=1),
        ]
        cfg = ImputationConfig()
        result = DeterministicImputer(cfg).impute(obs, ["a", "b"], [])

        meta = result.observations[1].metadata
        assert "source_columns" in meta
        assert isinstance(meta["source_columns"], list)
        assert len(meta["source_columns"]) > 0

    def test_original_value_preserved(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 5.0}, iteration=0),
            _make_simple_obs(is_failure=True, iteration=1),
        ]
        cfg = ImputationConfig()
        result = DeterministicImputer(cfg).impute(obs, ["a"], [])

        assert len(result.records) == 1
        assert result.records[0].original_value is None  # was missing

    def test_non_imputed_observations_unchanged(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 5.0, "b": 10.0}, iteration=0),
            _make_simple_obs(kpi_values={"a": 7.0, "b": 14.0}, iteration=1),
            _make_simple_obs(is_failure=True, iteration=2),
        ]
        cfg = ImputationConfig()
        result = DeterministicImputer(cfg).impute(obs, ["a", "b"], [])

        # First two observations should be unchanged
        assert result.observations[0].kpi_values == {"a": 5.0, "b": 10.0}
        assert result.observations[1].kpi_values == {"a": 7.0, "b": 14.0}
        assert "imputed" not in result.observations[0].metadata
        assert "imputed" not in result.observations[1].metadata

    def test_metadata_preserved_from_original(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 5.0}, iteration=0),
            _make_simple_obs(
                is_failure=True, iteration=1,
                metadata={"lab": "lab-1", "operator": "alice"},
            ),
        ]
        cfg = ImputationConfig()
        result = DeterministicImputer(cfg).impute(obs, ["a"], [])

        meta = result.observations[1].metadata
        assert meta["lab"] == "lab-1"
        assert meta["operator"] == "alice"
        assert meta["imputed"] is True

    def test_per_kpi_strategy_reflected_in_method(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 2.0, "b": 8.0}, iteration=0),
            _make_simple_obs(kpi_values={"a": 4.0, "b": 12.0}, iteration=1),
            _make_simple_obs(is_failure=True, iteration=2),
        ]
        cfg = ImputationConfig(
            strategy=ImputationStrategy.WORST_VALUE,
            per_kpi_strategy={"b": ImputationStrategy.COLUMN_MEAN},
        )
        result = DeterministicImputer(cfg).impute(obs, ["a", "b"], [])

        # Check records: "a" should be worst_value, "b" should be column_mean
        a_records = [r for r in result.records if r.kpi_name == "a"]
        b_records = [r for r in result.records if r.kpi_name == "b"]
        assert a_records[0].strategy == ImputationStrategy.WORST_VALUE
        assert b_records[0].strategy == ImputationStrategy.COLUMN_MEAN

        # Verify imputed values
        imputed_obs = result.observations[2]
        assert imputed_obs.kpi_values["a"] == 2.0  # min(2, 4)
        assert imputed_obs.kpi_values["b"] == 10.0  # mean(8, 12)

    def test_imputation_details_in_metadata(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 5.0}, iteration=0),
            _make_simple_obs(is_failure=True, iteration=1),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.COLUMN_MEAN)
        result = DeterministicImputer(cfg).impute(obs, ["a"], [])

        meta = result.observations[1].metadata
        assert "imputation_details" in meta
        assert "a" in meta["imputation_details"]
        detail = meta["imputation_details"]["a"]
        assert detail["method"] == "column_mean"
        assert detail["imputed_value"] == 5.0
        assert detail["original_value"] is None


# ---------------------------------------------------------------------------
# TestImputerIntegration
# ---------------------------------------------------------------------------

class TestImputerIntegration:
    """Integration tests with realistic data scenarios."""

    def test_realistic_multi_kpi_snapshot(self) -> None:
        obs, obj_names, specs = _make_observations(n_good=15, n_failures=5)
        cfg = ImputationConfig(strategy=ImputationStrategy.WORST_VALUE)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, obj_names, specs)

        assert len(result.observations) == 20
        assert len(result.records) == 10  # 5 failures * 2 KPIs each
        assert len(result.decision_hash) == 16

        # All failures should now have KPI values
        for o in result.observations:
            if o.is_failure:
                assert "kpi_a" in o.kpi_values
                assert "kpi_b" in o.kpi_values
                assert o.metadata["imputed"] is True

    def test_mix_of_failures_and_missing(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 10.0, "b": 20.0}, iteration=0),
            _make_simple_obs(kpi_values={"a": 30.0, "b": 40.0}, iteration=1),
            # Missing "b" only (not a failure)
            _make_simple_obs(kpi_values={"a": 50.0}, iteration=2),
            # Full failure
            _make_simple_obs(is_failure=True, iteration=3),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.COLUMN_MEAN)
        imputer = DeterministicImputer(cfg)
        result = imputer.impute(obs, ["a", "b"], [])

        # obs[2] should have "b" imputed: mean of [20, 40] = 30
        assert result.observations[2].kpi_values["b"] == 30.0
        # obs[2]'s "a" should remain 50.0
        assert result.observations[2].kpi_values["a"] == 50.0

        # obs[3] should have both imputed
        assert "a" in result.observations[3].kpi_values
        assert "b" in result.observations[3].kpi_values

    def test_all_observations_are_failures(self) -> None:
        obs = [_make_simple_obs(is_failure=True, iteration=i) for i in range(5)]
        cfg = ImputationConfig(strategy=ImputationStrategy.WORST_VALUE)
        result = DeterministicImputer(cfg).impute(obs, ["a"], [])

        # Should still produce output (all imputed to 0.0)
        assert len(result.observations) == 5
        for o in result.observations:
            assert o.kpi_values.get("a", 0.0) == 0.0

    def test_no_observations_need_imputation(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 1.0, "b": 2.0}, iteration=0),
            _make_simple_obs(kpi_values={"a": 3.0, "b": 4.0}, iteration=1),
        ]
        cfg = ImputationConfig()
        result = DeterministicImputer(cfg).impute(obs, ["a", "b"], [])

        assert len(result.records) == 0
        assert result.observations[0].kpi_values == {"a": 1.0, "b": 2.0}
        assert result.observations[1].kpi_values == {"a": 3.0, "b": 4.0}

    def test_with_categorical_parameters(self) -> None:
        specs = [
            ParameterSpec("cat", VariableType.CATEGORICAL, categories=["a", "b", "c"]),
            ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 10.0),
        ]
        obs = [
            Observation(iteration=0, parameters={"cat": "a", "x": 2.0},
                        kpi_values={"yield": 80.0}),
            Observation(iteration=1, parameters={"cat": "b", "x": 5.0},
                        kpi_values={"yield": 60.0}),
            Observation(iteration=2, parameters={"cat": "a", "x": 3.0},
                        kpi_values={"yield": 85.0}),
            Observation(iteration=3, parameters={"cat": "a", "x": 2.5},
                        kpi_values={}, is_failure=True, failure_reason="failed"),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.KNN_PROXY, knn_k=2)
        result = DeterministicImputer(cfg).impute(obs, ["yield"], specs)

        imputed = result.observations[3]
        assert "yield" in imputed.kpi_values
        assert imputed.metadata["imputed"] is True
        # Nearest neighbors by cat match + x distance should be obs 0 and 2
        # (both have cat="a" and close x values)
        expected = (80.0 + 85.0) / 2
        assert abs(imputed.kpi_values["yield"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_observations_list(self) -> None:
        cfg = ImputationConfig()
        result = DeterministicImputer(cfg).impute([], ["a"], [])

        assert result.observations == []
        assert result.records == []
        assert len(result.decision_hash) == 16

    def test_single_observation_failure(self) -> None:
        obs = [_make_simple_obs(is_failure=True, iteration=0)]
        cfg = ImputationConfig()
        result = DeterministicImputer(cfg).impute(obs, ["a"], [])

        assert len(result.observations) == 1
        assert result.observations[0].kpi_values.get("a", 0.0) == 0.0

    def test_no_kpis_requested(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={}, iteration=0),
            _make_simple_obs(is_failure=True, iteration=1),
        ]
        cfg = ImputationConfig()
        result = DeterministicImputer(cfg).impute(obs, [], [])

        # No KPIs to impute -- observations pass through
        assert len(result.observations) == 2
        assert len(result.records) == 0

    def test_partial_kpis(self) -> None:
        obs = [
            _make_simple_obs(kpi_values={"a": 10.0, "b": 20.0, "c": 30.0}, iteration=0),
            # Has "a" but missing "b" and "c"
            _make_simple_obs(kpi_values={"a": 15.0}, iteration=1),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.COLUMN_MEAN)
        result = DeterministicImputer(cfg).impute(obs, ["a", "b", "c"], [])

        imputed = result.observations[1]
        assert imputed.kpi_values["a"] == 15.0  # unchanged
        assert imputed.kpi_values["b"] == 20.0  # only valid value
        assert imputed.kpi_values["c"] == 30.0  # only valid value

    def test_very_large_values(self) -> None:
        large = 1e300
        obs = [
            _make_simple_obs(kpi_values={"a": large}, iteration=0),
            _make_simple_obs(kpi_values={"a": large * 0.5}, iteration=1),
            _make_simple_obs(is_failure=True, iteration=2),
        ]
        cfg = ImputationConfig(strategy=ImputationStrategy.COLUMN_MEAN)
        result = DeterministicImputer(cfg).impute(obs, ["a"], [])

        expected_mean = (large + large * 0.5) / 2
        assert abs(result.observations[2].kpi_values["a"] - expected_mean) < 1e290
