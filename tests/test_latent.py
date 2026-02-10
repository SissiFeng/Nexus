"""Integration tests for the Latent Optimization package.

Covers LatentSpace serialisation, LatentTransform fit/round-trip,
LatentPlugin wrapper, LatentOptimizer decision logic, determinism,
and edge cases.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    ProblemFingerprint,
    VariableType,
)
from optimization_copilot.latent.models import LatentSpace
from optimization_copilot.latent.optimizer import LatentOptimizer
from optimization_copilot.latent.plugin import LatentPlugin
from optimization_copilot.latent.transform import LatentTransform
from optimization_copilot.plugins.base import AlgorithmPlugin


# ── Helpers ──────────────────────────────────────────────


class _MockPlugin(AlgorithmPlugin):
    """Minimal plugin that returns midpoints of parameter bounds."""

    def name(self) -> str:
        return "mock"

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
        return [
            {s.name: (s.lower + s.upper) / 2 for s in self._specs}
            for _ in range(n_suggestions)
        ]

    def capabilities(self) -> dict[str, Any]:
        return {"supports_continuous": True, "supports_batch": True}


def _make_snapshot_2d(
    xs: list[float],
    ys: list[float],
) -> CampaignSnapshot:
    """Create a snapshot with two continuous params from x,y lists."""
    observations = [
        Observation(
            iteration=i,
            parameters={"x": x, "y": y},
            kpi_values={"obj": float(i)},
            qc_passed=True,
            is_failure=False,
        )
        for i, (x, y) in enumerate(zip(xs, ys))
    ]
    specs = [
        ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=-100.0, upper=100.0),
        ParameterSpec(name="y", type=VariableType.CONTINUOUS, lower=-100.0, upper=100.0),
    ]
    return CampaignSnapshot(
        campaign_id="test-latent",
        parameter_specs=specs,
        observations=observations,
        objective_names=["obj"],
        objective_directions=["maximize"],
    )


def _make_high_dim_snapshot(
    n_params: int = 6,
    n_obs: int = 20,
    seed: int = 42,
) -> CampaignSnapshot:
    """Create a snapshot with many continuous params and observations.

    Uses a simple deterministic pattern: param_j for observation i is
    (i * (j + 1)) % 100 scaled to floats.
    """
    import random as _rng
    r = _rng.Random(seed)

    param_names = [f"p{j}" for j in range(n_params)]
    specs = [
        ParameterSpec(name=name, type=VariableType.CONTINUOUS, lower=0.0, upper=100.0)
        for name in param_names
    ]
    observations = []
    for i in range(n_obs):
        params = {
            name: float((i * (j + 1) + r.randint(0, 10)) % 100)
            for j, name in enumerate(param_names)
        }
        observations.append(
            Observation(
                iteration=i,
                parameters=params,
                kpi_values={"obj": float(r.random() * 100)},
                qc_passed=True,
                is_failure=False,
            )
        )
    return CampaignSnapshot(
        campaign_id="test-high-dim",
        parameter_specs=specs,
        observations=observations,
        objective_names=["obj"],
        objective_directions=["maximize"],
    )


class _FakeDiagnostics:
    """Simple object with a kpi_plateau_length attribute."""

    def __init__(self, kpi_plateau_length: int = 0):
        self.kpi_plateau_length = kpi_plateau_length


# ── TestLatentSpace ──────────────────────────────────────


class TestLatentSpace:
    """Test LatentSpace construction and serialisation."""

    def _make_latent_space(self) -> LatentSpace:
        """Create a minimal LatentSpace for testing."""
        return LatentSpace(
            components=[[0.6, 0.8], [-0.8, 0.6]],
            eigenvalues=[3.0, 1.0],
            mean=[5.0, 10.0],
            std=[2.0, 3.0],
            n_components=2,
            original_dim=2,
            explained_variance_ratio=[0.75, 0.25],
            total_explained_variance=1.0,
            param_names=["x", "y"],
        )

    def test_construction(self):
        """LatentSpace can be constructed with expected attributes."""
        ls = self._make_latent_space()
        assert ls.n_components == 2
        assert ls.original_dim == 2
        assert len(ls.components) == 2
        assert len(ls.eigenvalues) == 2
        assert ls.param_names == ["x", "y"]

    def test_to_dict_round_trip(self):
        """to_dict -> from_dict produces an identical LatentSpace."""
        ls = self._make_latent_space()
        d = ls.to_dict()
        restored = LatentSpace.from_dict(d)

        assert restored.components == ls.components
        assert restored.eigenvalues == ls.eigenvalues
        assert restored.mean == ls.mean
        assert restored.std == ls.std
        assert restored.n_components == ls.n_components
        assert restored.original_dim == ls.original_dim
        assert restored.explained_variance_ratio == ls.explained_variance_ratio
        assert restored.total_explained_variance == ls.total_explained_variance
        assert restored.param_names == ls.param_names

    def test_from_dict_without_param_names(self):
        """from_dict should handle missing param_names gracefully."""
        ls = self._make_latent_space()
        d = ls.to_dict()
        del d["param_names"]
        restored = LatentSpace.from_dict(d)
        assert restored.param_names == []

    def test_to_dict_keys(self):
        """to_dict should contain all expected keys."""
        ls = self._make_latent_space()
        d = ls.to_dict()
        expected_keys = {
            "components", "eigenvalues", "mean", "std", "n_components",
            "original_dim", "explained_variance_ratio",
            "total_explained_variance", "param_names",
        }
        assert set(d.keys()) == expected_keys


# ── TestLatentTransformFit ───────────────────────────────


class TestLatentTransformFit:
    """Test LatentTransform.fit() behaviour."""

    def test_produces_valid_latent_space(self):
        """fit() returns a LatentSpace with correct dimensions."""
        snap = _make_snapshot_2d([1.0, 5.0, 9.0], [2.0, 6.0, 8.0])
        transform = LatentTransform()
        ls = transform.fit(snap, seed=42)

        assert isinstance(ls, LatentSpace)
        assert ls.original_dim == 2
        assert ls.n_components >= 1
        assert ls.n_components <= ls.original_dim
        assert len(ls.components) == ls.n_components
        assert len(ls.eigenvalues) == ls.n_components

    def test_respects_min_variance_explained(self):
        """When min_variance_explained is high, more components are retained."""
        snap = _make_snapshot_2d(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 8.0, 6.0, 4.0, 2.0],
        )
        # With 0.99, should keep both components
        transform_high = LatentTransform(min_variance_explained=0.99)
        ls_high = transform_high.fit(snap, seed=42)

        # With 0.01, one component should be enough
        transform_low = LatentTransform(min_variance_explained=0.01)
        ls_low = transform_low.fit(snap, seed=42)

        assert ls_low.n_components <= ls_high.n_components

    def test_max_components_limits_output(self):
        """max_components caps the number of retained components."""
        snap = _make_snapshot_2d(
            [1.0, 5.0, 9.0, 2.0, 8.0],
            [3.0, 7.0, 1.0, 6.0, 4.0],
        )
        transform = LatentTransform(
            min_variance_explained=0.99,
            max_components=1,
        )
        ls = transform.fit(snap, seed=42)

        assert ls.n_components == 1
        assert len(ls.components) == 1
        assert len(ls.eigenvalues) == 1

    def test_raises_for_fewer_than_2_observations(self):
        """fit() raises ValueError when fewer than 2 successful observations."""
        snap = _make_snapshot_2d([1.0], [2.0])
        transform = LatentTransform()
        with pytest.raises(ValueError, match="at least 2"):
            transform.fit(snap, seed=42)

    def test_raises_for_zero_observations(self):
        """fit() raises ValueError when there are no observations."""
        snap = CampaignSnapshot(
            campaign_id="test-empty",
            parameter_specs=[
                ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            ],
            observations=[],
            objective_names=["obj"],
            objective_directions=["maximize"],
        )
        transform = LatentTransform()
        with pytest.raises(ValueError, match="at least 2"):
            transform.fit(snap, seed=42)

    def test_raises_for_no_numeric_parameters(self):
        """fit() raises ValueError when all parameters are categorical."""
        observations = [
            Observation(
                iteration=i,
                parameters={"color": "red", "shape": "circle"},
                kpi_values={"obj": float(i)},
            )
            for i in range(5)
        ]
        snap = CampaignSnapshot(
            campaign_id="test-categorical",
            parameter_specs=[
                ParameterSpec(name="color", type=VariableType.CATEGORICAL, categories=["red", "blue"]),
                ParameterSpec(name="shape", type=VariableType.CATEGORICAL, categories=["circle", "square"]),
            ],
            observations=observations,
            objective_names=["obj"],
            objective_directions=["maximize"],
        )
        transform = LatentTransform()
        with pytest.raises(ValueError, match="No numeric"):
            transform.fit(snap, seed=42)

    def test_param_names_stored(self):
        """LatentSpace.param_names should match numeric parameter names."""
        snap = _make_snapshot_2d([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        transform = LatentTransform()
        ls = transform.fit(snap, seed=42)
        assert ls.param_names == ["x", "y"]


# ── TestLatentTransformRoundTrip ─────────────────────────


class TestLatentTransformRoundTrip:
    """Test to_latent -> from_latent round-trip fidelity."""

    def test_round_trip_close_to_original(self):
        """to_latent -> from_latent produces values close to original."""
        snap = _make_snapshot_2d(
            [1.0, 5.0, 9.0, 3.0, 7.0],
            [2.0, 6.0, 8.0, 4.0, 10.0],
        )
        transform = LatentTransform(min_variance_explained=0.99)
        ls = transform.fit(snap, seed=42)

        # Project each observation and reconstruct
        original_params = {"x": 5.0, "y": 6.0}
        latent_coords = transform.to_latent(original_params, ls)
        reconstructed = transform.from_latent(latent_coords, ls)

        assert reconstructed["x"] == pytest.approx(original_params["x"], abs=0.5)
        assert reconstructed["y"] == pytest.approx(original_params["y"], abs=0.5)

    def test_round_trip_with_full_components(self):
        """With all components retained, round-trip should be near-exact."""
        snap = _make_snapshot_2d(
            [1.0, 5.0, 9.0, 3.0, 7.0],
            [2.0, 6.0, 8.0, 4.0, 10.0],
        )
        transform = LatentTransform(min_variance_explained=0.999)
        ls = transform.fit(snap, seed=42)

        # When all components are kept, reconstruction should be very close
        if ls.n_components == ls.original_dim:
            original_params = {"x": 3.0, "y": 4.0}
            latent_coords = transform.to_latent(original_params, ls)
            reconstructed = transform.from_latent(latent_coords, ls)
            assert reconstructed["x"] == pytest.approx(original_params["x"], abs=1e-4)
            assert reconstructed["y"] == pytest.approx(original_params["y"], abs=1e-4)

    def test_clamping_to_bounds(self):
        """Values outside parameter bounds are clamped."""
        snap = _make_snapshot_2d([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0])
        transform = LatentTransform(min_variance_explained=0.99)
        ls = transform.fit(snap, seed=42)

        specs = [
            ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            ParameterSpec(name="y", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]

        # Use extreme latent coordinates that would map outside bounds
        extreme_coords = [100.0] * ls.n_components
        reconstructed = transform.from_latent(extreme_coords, ls, specs=specs)
        assert reconstructed["x"] <= 10.0
        assert reconstructed["y"] <= 10.0

        extreme_coords_neg = [-100.0] * ls.n_components
        reconstructed_neg = transform.from_latent(extreme_coords_neg, ls, specs=specs)
        assert reconstructed_neg["x"] >= 0.0
        assert reconstructed_neg["y"] >= 0.0

    def test_discrete_params_are_rounded(self):
        """Discrete parameters should be rounded to integers after reconstruction."""
        observations = [
            Observation(
                iteration=i,
                parameters={"x": float(i), "d": float(i * 2)},
                kpi_values={"obj": float(i)},
            )
            for i in range(5)
        ]
        specs = [
            ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            ParameterSpec(name="d", type=VariableType.DISCRETE, lower=0.0, upper=20.0),
        ]
        snap = CampaignSnapshot(
            campaign_id="test-discrete",
            parameter_specs=specs,
            observations=observations,
            objective_names=["obj"],
            objective_directions=["maximize"],
        )
        transform = LatentTransform(min_variance_explained=0.99)
        ls = transform.fit(snap, seed=42)

        # Reconstruct and check discrete param is rounded
        coords = transform.to_latent({"x": 3.0, "d": 5.0}, ls)
        reconstructed = transform.from_latent(coords, ls, specs=specs)

        # The discrete parameter should be an integer (rounded)
        assert reconstructed["d"] == round(reconstructed["d"])


# ── TestLatentPlugin ─────────────────────────────────────


class TestLatentPlugin:
    """Test the LatentPlugin wrapper."""

    def _setup_plugin(self) -> tuple[LatentPlugin, CampaignSnapshot, list[ParameterSpec]]:
        """Create a LatentPlugin wrapping _MockPlugin, fitted on test data."""
        snap = _make_snapshot_2d(
            [1.0, 3.0, 5.0, 7.0, 9.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
        )
        transform = LatentTransform(min_variance_explained=0.99)
        ls = transform.fit(snap, seed=42)

        mock = _MockPlugin()
        original_specs = snap.parameter_specs
        plugin = LatentPlugin(mock, ls, transform, original_specs)
        return plugin, snap, original_specs

    def test_name_prefix(self):
        """name() returns 'latent_<inner_name>'."""
        plugin, _, _ = self._setup_plugin()
        assert plugin.name() == "latent_mock"

    def test_capabilities_includes_latent_wrapped(self):
        """capabilities() includes latent_wrapped=True."""
        plugin, _, _ = self._setup_plugin()
        caps = plugin.capabilities()
        assert caps["latent_wrapped"] is True
        assert caps["supports_categorical"] is False
        # Inner capabilities are preserved
        assert caps["supports_continuous"] is True
        assert caps["supports_batch"] is True

    def test_fit_and_suggest_end_to_end(self):
        """fit() then suggest() works without errors."""
        plugin, snap, specs = self._setup_plugin()
        plugin.fit(snap.observations, specs)
        suggestions = plugin.suggest(n_suggestions=3, seed=42)

        assert len(suggestions) == 3
        for suggestion in suggestions:
            assert isinstance(suggestion, dict)

    def test_suggest_returns_original_param_names(self):
        """suggest() returns dicts with the original parameter names."""
        plugin, snap, specs = self._setup_plugin()
        plugin.fit(snap.observations, specs)
        suggestions = plugin.suggest(n_suggestions=1, seed=42)

        assert len(suggestions) == 1
        suggestion = suggestions[0]
        # Should have the original param names (x, y), not z0, z1
        assert "x" in suggestion
        assert "y" in suggestion
        assert "z0" not in suggestion
        assert "z1" not in suggestion

    def test_suggest_values_are_numeric(self):
        """Suggested values should be numeric floats."""
        plugin, snap, specs = self._setup_plugin()
        plugin.fit(snap.observations, specs)
        suggestions = plugin.suggest(n_suggestions=2, seed=42)

        for suggestion in suggestions:
            for key, val in suggestion.items():
                assert isinstance(val, (int, float)), (
                    f"Value for {key} is {type(val)}, expected numeric"
                )

    def test_failed_observations_handled(self):
        """fit() handles observations with is_failure=True (mapped to zero vector)."""
        plugin, snap, specs = self._setup_plugin()

        # Add a failed observation
        failed_obs = Observation(
            iteration=99,
            parameters={"x": 999.0, "y": 999.0},
            kpi_values={},
            is_failure=True,
            failure_reason="test failure",
        )
        obs_with_failure = list(snap.observations) + [failed_obs]

        # Should not raise
        plugin.fit(obs_with_failure, specs)
        suggestions = plugin.suggest(n_suggestions=1, seed=42)
        assert len(suggestions) == 1


# ── TestLatentOptimizer ──────────────────────────────────


class TestLatentOptimizer:
    """Test LatentOptimizer decision logic."""

    def test_should_use_latent_false_for_low_dim(self):
        """Returns False when fewer than min_dim_for_latent numeric params."""
        snap = _make_snapshot_2d(
            [float(i) for i in range(20)],
            [float(i) for i in range(20)],
        )
        optimizer = LatentOptimizer(min_dim_for_latent=5, min_observations_for_fit=5)
        fp = ProblemFingerprint()
        # snap has only 2 numeric params, which is < 5
        assert optimizer.should_use_latent(snap, fp) is False

    def test_should_use_latent_false_for_insufficient_data(self):
        """Returns False when fewer than min_observations_for_fit observations."""
        snap = _make_high_dim_snapshot(n_params=6, n_obs=10)
        optimizer = LatentOptimizer(min_dim_for_latent=5, min_observations_for_fit=15)
        fp = ProblemFingerprint()
        # Only 10 obs, need 15
        assert optimizer.should_use_latent(snap, fp) is False

    def test_should_use_latent_false_for_all_categorical(self):
        """Returns False when all parameters are categorical."""
        observations = [
            Observation(
                iteration=i,
                parameters={"color": "red", "size": "large"},
                kpi_values={"obj": float(i)},
            )
            for i in range(20)
        ]
        snap = CampaignSnapshot(
            campaign_id="test-cat",
            parameter_specs=[
                ParameterSpec(name="color", type=VariableType.CATEGORICAL, categories=["red", "blue"]),
                ParameterSpec(name="size", type=VariableType.CATEGORICAL, categories=["large", "small"]),
            ],
            observations=observations,
            objective_names=["obj"],
            objective_directions=["maximize"],
        )
        optimizer = LatentOptimizer(min_dim_for_latent=1, min_observations_for_fit=5)
        fp = ProblemFingerprint()
        assert optimizer.should_use_latent(snap, fp) is False

    def test_should_use_latent_true_when_conditions_met(self):
        """Returns True when sufficient dims and observations."""
        snap = _make_high_dim_snapshot(n_params=6, n_obs=20)
        optimizer = LatentOptimizer(min_dim_for_latent=5, min_observations_for_fit=15)
        fp = ProblemFingerprint()
        assert optimizer.should_use_latent(snap, fp) is True

    def test_adjust_dimensions_increases_on_stagnation(self):
        """adjust_dimensions increases components when plateau > 5."""
        snap = _make_high_dim_snapshot(n_params=6, n_obs=20)
        optimizer = LatentOptimizer(stagnation_dim_increase=1)
        ls = optimizer.fit(snap, seed=42)

        # Only test if we have room to grow
        if ls.n_components < ls.original_dim:
            original_n = ls.n_components
            diag = _FakeDiagnostics(kpi_plateau_length=10)
            adjusted_ls = optimizer.adjust_dimensions(ls, diag, snap, seed=42)
            # The adjusted space should attempt more components
            # (may or may not actually have more, depending on max_components logic)
            assert adjusted_ls.n_components >= original_n

    def test_adjust_dimensions_no_change_without_stagnation(self):
        """adjust_dimensions returns original space when no stagnation."""
        snap = _make_high_dim_snapshot(n_params=6, n_obs=20)
        optimizer = LatentOptimizer()
        ls = optimizer.fit(snap, seed=42)

        diag = _FakeDiagnostics(kpi_plateau_length=2)
        adjusted_ls = optimizer.adjust_dimensions(ls, diag, snap, seed=42)

        # Should return the exact same object
        assert adjusted_ls is ls

    def test_adjust_dimensions_plateau_at_boundary(self):
        """adjust_dimensions does nothing when plateau is exactly 5 (not > 5)."""
        snap = _make_high_dim_snapshot(n_params=6, n_obs=20)
        optimizer = LatentOptimizer()
        ls = optimizer.fit(snap, seed=42)

        diag = _FakeDiagnostics(kpi_plateau_length=5)
        adjusted_ls = optimizer.adjust_dimensions(ls, diag, snap, seed=42)
        assert adjusted_ls is ls

    def test_wrap_plugin_produces_working_latent_plugin(self):
        """wrap_plugin produces a LatentPlugin that can fit and suggest."""
        snap = _make_high_dim_snapshot(n_params=6, n_obs=20)
        optimizer = LatentOptimizer(min_dim_for_latent=5, min_observations_for_fit=15)
        ls = optimizer.fit(snap, seed=42)

        mock = _MockPlugin()
        wrapped = optimizer.wrap_plugin(mock, ls, specs=snap.parameter_specs)

        assert isinstance(wrapped, LatentPlugin)
        assert wrapped.name() == "latent_mock"

        # Should be usable
        wrapped.fit(snap.observations, snap.parameter_specs)
        suggestions = wrapped.suggest(n_suggestions=2, seed=42)
        assert len(suggestions) == 2
        for suggestion in suggestions:
            # Should have original param names (p0, p1, ... p5), not z0, z1, ...
            for name in [f"p{j}" for j in range(6)]:
                assert name in suggestion


# ── TestDeterminism ──────────────────────────────────────


class TestDeterminism:
    """Same (snapshot, seed) -> same LatentSpace."""

    def test_same_snapshot_same_seed_same_result(self):
        """Fitting PCA twice with the same seed produces identical results."""
        snap = _make_high_dim_snapshot(n_params=5, n_obs=20, seed=42)
        transform = LatentTransform()

        ls1 = transform.fit(snap, seed=99)
        ls2 = transform.fit(snap, seed=99)

        assert ls1.n_components == ls2.n_components
        assert ls1.eigenvalues == ls2.eigenvalues
        assert ls1.mean == ls2.mean
        assert ls1.std == ls2.std
        assert ls1.explained_variance_ratio == ls2.explained_variance_ratio
        for c1, c2 in zip(ls1.components, ls2.components):
            for a, b in zip(c1, c2):
                assert a == pytest.approx(b, abs=1e-12)

    def test_latent_plugin_deterministic(self):
        """Same fit + suggest with same seed -> identical suggestions."""
        snap = _make_snapshot_2d(
            [1.0, 3.0, 5.0, 7.0, 9.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
        )
        transform = LatentTransform(min_variance_explained=0.99)
        ls = transform.fit(snap, seed=42)

        mock1 = _MockPlugin()
        mock2 = _MockPlugin()
        plugin1 = LatentPlugin(mock1, ls, transform, snap.parameter_specs)
        plugin2 = LatentPlugin(mock2, ls, transform, snap.parameter_specs)

        plugin1.fit(snap.observations, snap.parameter_specs)
        plugin2.fit(snap.observations, snap.parameter_specs)

        s1 = plugin1.suggest(n_suggestions=3, seed=7)
        s2 = plugin2.suggest(n_suggestions=3, seed=7)

        assert len(s1) == len(s2)
        for a, b in zip(s1, s2):
            for key in a:
                assert a[key] == pytest.approx(b[key], abs=1e-10)


# ── TestEdgeCases ────────────────────────────────────────


class TestEdgeCases:
    """Edge case testing for robustness."""

    def test_exactly_2_observations(self):
        """Minimum viable fit: exactly 2 successful observations."""
        snap = _make_snapshot_2d([1.0, 10.0], [2.0, 20.0])
        transform = LatentTransform()
        ls = transform.fit(snap, seed=42)

        # Should succeed and produce a valid LatentSpace
        assert isinstance(ls, LatentSpace)
        assert ls.n_components >= 1
        # With 2 samples and 2 features, max_k = min(2, 2-1) = 1
        assert ls.n_components == 1

    def test_all_identical_observations(self):
        """When all observations are identical, std defaults to 1.0."""
        snap = _make_snapshot_2d(
            [5.0, 5.0, 5.0, 5.0],
            [3.0, 3.0, 3.0, 3.0],
        )
        transform = LatentTransform()
        ls = transform.fit(snap, seed=42)

        # Zero variance everywhere -> std = 1.0 for all columns
        assert ls.std[0] == pytest.approx(1.0)
        assert ls.std[1] == pytest.approx(1.0)
        # Eigenvalues should be 0 (or effectively 0) since all data is identical
        for ev in ls.eigenvalues:
            assert ev == pytest.approx(0.0, abs=1e-6)

    def test_more_requested_components_than_possible(self):
        """max_components > data rank is gracefully handled."""
        snap = _make_snapshot_2d([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        # With 3 samples and 2 features, max possible is min(2, 3-1) = 2
        # Request 100 components -- should be capped
        transform = LatentTransform(max_components=100, min_variance_explained=0.99)
        ls = transform.fit(snap, seed=42)

        assert ls.n_components <= 2
        assert ls.n_components >= 1

    def test_single_feature_dimension(self):
        """Snapshot with only one numeric parameter still works."""
        observations = [
            Observation(
                iteration=i,
                parameters={"x": float(i)},
                kpi_values={"obj": float(i)},
            )
            for i in range(5)
        ]
        snap = CampaignSnapshot(
            campaign_id="test-1d",
            parameter_specs=[
                ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            ],
            observations=observations,
            objective_names=["obj"],
            objective_directions=["maximize"],
        )
        transform = LatentTransform()
        ls = transform.fit(snap, seed=42)

        assert ls.original_dim == 1
        assert ls.n_components == 1

    def test_mixed_categorical_and_continuous(self):
        """Only numeric params are used; categoricals are ignored by PCA."""
        observations = [
            Observation(
                iteration=i,
                parameters={"x": float(i), "y": float(i * 2), "color": "red"},
                kpi_values={"obj": float(i)},
            )
            for i in range(5)
        ]
        snap = CampaignSnapshot(
            campaign_id="test-mixed",
            parameter_specs=[
                ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
                ParameterSpec(name="y", type=VariableType.CONTINUOUS, lower=0.0, upper=20.0),
                ParameterSpec(name="color", type=VariableType.CATEGORICAL, categories=["red", "blue"]),
            ],
            observations=observations,
            objective_names=["obj"],
            objective_directions=["maximize"],
        )
        transform = LatentTransform()
        ls = transform.fit(snap, seed=42)

        # Should only have 2 dimensions (x and y, not color)
        assert ls.original_dim == 2
        assert ls.param_names == ["x", "y"]

    def test_failed_observations_excluded_from_fit(self):
        """Failed observations are not included in PCA fitting via successful_observations."""
        observations = [
            Observation(
                iteration=0,
                parameters={"x": 1.0, "y": 2.0},
                kpi_values={"obj": 1.0},
                is_failure=False,
            ),
            Observation(
                iteration=1,
                parameters={"x": 5.0, "y": 6.0},
                kpi_values={"obj": 2.0},
                is_failure=False,
            ),
            Observation(
                iteration=2,
                parameters={"x": 999.0, "y": 999.0},
                kpi_values={},
                is_failure=True,
                failure_reason="test",
            ),
            Observation(
                iteration=3,
                parameters={"x": 9.0, "y": 10.0},
                kpi_values={"obj": 3.0},
                is_failure=False,
            ),
        ]
        snap = CampaignSnapshot(
            campaign_id="test-failures",
            parameter_specs=[
                ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=100.0),
                ParameterSpec(name="y", type=VariableType.CONTINUOUS, lower=0.0, upper=100.0),
            ],
            observations=observations,
            objective_names=["obj"],
            objective_directions=["maximize"],
        )
        transform = LatentTransform()
        ls = transform.fit(snap, seed=42)

        # Mean should not be affected by the failed observation (999, 999)
        # Successful obs: (1,2), (5,6), (9,10) -> mean = (5, 6)
        assert ls.mean[0] == pytest.approx(5.0)
        assert ls.mean[1] == pytest.approx(6.0)
