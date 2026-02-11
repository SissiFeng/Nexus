"""Tests for Pareto navigation: target bands, aspiration levels, and auto-aspiration."""

from __future__ import annotations

import pytest

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.multi_objective.pareto import (
    AspirationLevel,
    MultiObjectiveAnalyzer,
    ParetoNavigator,
    TargetBand,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(
    iteration: int,
    kpis: dict[str, float],
    params: dict | None = None,
) -> Observation:
    return Observation(
        iteration=iteration,
        parameters=params or {"x": 0.5},
        kpi_values=kpis,
        qc_passed=True,
        is_failure=False,
        timestamp=float(iteration),
    )


def _make_mo_snapshot(
    observations: list[Observation],
    obj_names: list[str] | None = None,
    obj_dirs: list[str] | None = None,
    metadata: dict | None = None,
) -> CampaignSnapshot:
    if obj_names is None:
        obj_names = ["yield", "purity"]
    if obj_dirs is None:
        obj_dirs = ["maximize", "maximize"]
    return CampaignSnapshot(
        campaign_id="test_mo",
        parameter_specs=[
            ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ],
        observations=observations,
        objective_names=obj_names,
        objective_directions=obj_dirs,
        constraints=[],
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# TargetBand filtering
# ---------------------------------------------------------------------------

class TestFilterToBand:

    def setup_method(self):
        self.nav = ParetoNavigator()

    def test_all_in_band(self):
        obs = [
            _make_obs(0, {"yield": 5.0, "purity": 0.9}),
            _make_obs(1, {"yield": 7.0, "purity": 0.8}),
        ]
        bands = [TargetBand("yield", lower=4.0, upper=8.0)]
        result = self.nav.filter_to_band(obs, bands)
        assert len(result.selected) == 2

    def test_some_filtered(self):
        obs = [
            _make_obs(0, {"yield": 3.0, "purity": 0.9}),
            _make_obs(1, {"yield": 7.0, "purity": 0.8}),
            _make_obs(2, {"yield": 12.0, "purity": 0.7}),
        ]
        bands = [TargetBand("yield", lower=5.0, upper=10.0)]
        result = self.nav.filter_to_band(obs, bands)
        assert len(result.selected) == 1
        assert result.selected[0].kpi_values["yield"] == 7.0

    def test_multi_objective_bands(self):
        obs = [
            _make_obs(0, {"yield": 5.0, "purity": 0.95}),
            _make_obs(1, {"yield": 8.0, "purity": 0.6}),
            _make_obs(2, {"yield": 7.0, "purity": 0.85}),
        ]
        bands = [
            TargetBand("yield", lower=6.0),
            TargetBand("purity", lower=0.7),
        ]
        result = self.nav.filter_to_band(obs, bands)
        assert len(result.selected) == 1
        assert result.selected_indices == [2]

    def test_empty_band_passes_all(self):
        obs = [_make_obs(i, {"yield": float(i)}) for i in range(5)]
        result = self.nav.filter_to_band(obs, [])
        assert len(result.selected) == 5


# ---------------------------------------------------------------------------
# Focus region (aspiration-based ranking)
# ---------------------------------------------------------------------------

class TestFocusRegion:

    def setup_method(self):
        self.nav = ParetoNavigator()

    def test_closest_to_aspiration_first(self):
        obs = [
            _make_obs(0, {"yield": 5.0, "purity": 0.9}),
            _make_obs(1, {"yield": 10.0, "purity": 0.5}),
            _make_obs(2, {"yield": 7.0, "purity": 0.7}),
        ]
        aspirations = [
            AspirationLevel("yield", target=7.0),
            AspirationLevel("purity", target=0.7),
        ]
        result = self.nav.focus_region(obs, aspirations)
        # obs[2] is exactly at (7, 0.7), should be first
        assert result.selected[0].kpi_values["yield"] == 7.0
        assert result.distances[0] == pytest.approx(0.0)

    def test_distances_monotonically_sorted(self):
        obs = [
            _make_obs(i, {"yield": float(i), "purity": float(10 - i)})
            for i in range(10)
        ]
        aspirations = [AspirationLevel("yield", target=5.0)]
        result = self.nav.focus_region(obs, aspirations)
        for i in range(len(result.distances) - 1):
            assert result.distances[i] <= result.distances[i + 1]

    def test_empty_observations(self):
        result = self.nav.focus_region([], [AspirationLevel("yield", target=5)])
        assert result.selected == []

    def test_empty_aspirations(self):
        obs = [_make_obs(0, {"yield": 5.0})]
        result = self.nav.focus_region(obs, [])
        assert len(result.selected) == 1


# ---------------------------------------------------------------------------
# Auto aspiration
# ---------------------------------------------------------------------------

class TestAutoAspiration:

    def test_early_campaign_median_target(self):
        """Early in campaign (low progress), aspiration → median."""
        obs = [_make_obs(i, {"yield": float(i)}) for i in range(10)]
        snap = _make_mo_snapshot(
            obs, obj_names=["yield"], obj_dirs=["maximize"],
            metadata={"budget": 100},
        )
        aspirations = ParetoNavigator.auto_aspiration(snap, progress_fraction=0.1)
        assert len(aspirations) == 1
        asp = aspirations[0]
        # At 10% progress: target ≈ median + 0.1*(best - median)
        median = 5.0  # sorted: 0..9, median[5] = 5
        best = 9.0
        expected = median + 0.1 * (best - median)
        assert asp.target == pytest.approx(expected, abs=0.5)

    def test_late_campaign_best_target(self):
        """Late in campaign (high progress), aspiration → best."""
        obs = [_make_obs(i, {"yield": float(i)}) for i in range(10)]
        snap = _make_mo_snapshot(
            obs, obj_names=["yield"], obj_dirs=["maximize"],
            metadata={"budget": 10},
        )
        aspirations = ParetoNavigator.auto_aspiration(snap, progress_fraction=1.0)
        asp = aspirations[0]
        assert asp.target == pytest.approx(9.0)
        assert asp.tolerance == pytest.approx(0.0, abs=0.01)

    def test_auto_aspiration_multi_objective(self):
        obs = [
            _make_obs(i, {"yield": float(i), "purity": 1.0 - i * 0.1})
            for i in range(10)
        ]
        snap = _make_mo_snapshot(obs, obj_names=["yield", "purity"],
                                  obj_dirs=["maximize", "maximize"])
        aspirations = ParetoNavigator.auto_aspiration(snap, progress_fraction=0.5)
        assert len(aspirations) == 2
        assert aspirations[0].objective == "yield"
        assert aspirations[1].objective == "purity"

    def test_empty_snapshot(self):
        snap = _make_mo_snapshot([], obj_names=["yield"], obj_dirs=["maximize"])
        aspirations = ParetoNavigator.auto_aspiration(snap)
        assert aspirations == []

    def test_tolerance_decreases_with_progress(self):
        obs = [_make_obs(i, {"yield": float(i)}) for i in range(10)]
        snap = _make_mo_snapshot(obs, obj_names=["yield"], obj_dirs=["maximize"])
        early = ParetoNavigator.auto_aspiration(snap, progress_fraction=0.2)
        late = ParetoNavigator.auto_aspiration(snap, progress_fraction=0.8)
        assert early[0].tolerance > late[0].tolerance


# ---------------------------------------------------------------------------
# Integration: filter Pareto front then navigate
# ---------------------------------------------------------------------------

class TestParetoNavigationIntegration:

    def test_filter_pareto_then_focus(self):
        """Filter Pareto front to band, then rank by aspiration."""
        obs = [
            _make_obs(0, {"yield": 2.0, "purity": 0.99}),
            _make_obs(1, {"yield": 5.0, "purity": 0.85}),
            _make_obs(2, {"yield": 8.0, "purity": 0.7}),
            _make_obs(3, {"yield": 10.0, "purity": 0.4}),
        ]
        snap = _make_mo_snapshot(obs)

        analyzer = MultiObjectiveAnalyzer()
        result = analyzer.analyze(snap)

        nav = ParetoNavigator()
        # Filter to acceptable yield and purity
        bands = [
            TargetBand("yield", lower=4.0, upper=9.0),
            TargetBand("purity", lower=0.6),
        ]
        filtered = nav.filter_to_band(result.pareto_front, bands)
        assert len(filtered.selected) >= 1

        # Focus on aspiration
        aspirations = [
            AspirationLevel("yield", target=6.0),
            AspirationLevel("purity", target=0.8),
        ]
        focused = nav.focus_region(filtered.selected, aspirations)
        assert len(focused.selected) >= 1
        # Closest to (6.0, 0.8) should be ranked first
        assert focused.distances[0] <= focused.distances[-1]
