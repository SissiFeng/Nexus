"""Tests for cross-lab comparison and reproducibility scoring (Pain Point 7)."""

from __future__ import annotations

import random

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.validation.scenarios import (
    CampaignComparison,
    CrossLabComparator,
    ReproducibilityReport,
    ReproducibilityScorer,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_snapshot(
    campaign_id: str,
    n_obs: int = 20,
    param_names: list[str] | None = None,
    obj_names: list[str] | None = None,
    seed: int = 42,
    kpi_base: float = 5.0,
    kpi_slope: float = 0.3,
) -> CampaignSnapshot:
    if param_names is None:
        param_names = ["x0", "x1", "x2"]
    if obj_names is None:
        obj_names = ["y"]

    specs = [
        ParameterSpec(name=n, type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
        for n in param_names
    ]
    rng = random.Random(seed)
    obs = []
    for i in range(n_obs):
        params = {n: rng.random() for n in param_names}
        kpis = {obj: kpi_base + i * kpi_slope + rng.gauss(0, 0.1) for obj in obj_names}
        obs.append(Observation(
            iteration=i,
            parameters=params,
            kpi_values=kpis,
            timestamp=float(i),
        ))

    return CampaignSnapshot(
        campaign_id=campaign_id,
        parameter_specs=specs,
        observations=obs,
        objective_names=obj_names,
        objective_directions=["maximize"] * len(obj_names),
        current_iteration=n_obs,
    )


# ── CrossLabComparator ────────────────────────────────────────────────────────

class TestCrossLabComparator:

    def test_identical_campaigns_perfect_alignment(self):
        """Same snapshot compared to itself should have high alignment."""
        snap = _make_snapshot("lab_a", n_obs=20)
        comp = CrossLabComparator()
        result = comp.compare(snap, snap)
        assert result.parameter_overlap == 1.0
        assert result.objective_alignment == 1.0
        assert result.phase_agreement is True

    def test_disjoint_parameters_zero_overlap(self):
        snap_a = _make_snapshot("lab_a", param_names=["x0", "x1"])
        snap_b = _make_snapshot("lab_b", param_names=["y0", "y1"])
        comp = CrossLabComparator()
        result = comp.compare(snap_a, snap_b)
        assert result.parameter_overlap == 0.0

    def test_partial_parameter_overlap(self):
        snap_a = _make_snapshot("lab_a", param_names=["x0", "x1", "x2"])
        snap_b = _make_snapshot("lab_b", param_names=["x1", "x2", "x3"])
        comp = CrossLabComparator()
        result = comp.compare(snap_a, snap_b)
        # Jaccard: {x1,x2} / {x0,x1,x2,x3} = 2/4 = 0.5
        assert result.parameter_overlap == 0.5

    def test_disjoint_objectives_zero_alignment(self):
        snap_a = _make_snapshot("lab_a", obj_names=["yield"])
        snap_b = _make_snapshot("lab_b", obj_names=["purity"])
        comp = CrossLabComparator()
        result = comp.compare(snap_a, snap_b)
        assert result.objective_alignment == 0.0

    def test_kpi_correlation_same_trend(self):
        """Two campaigns with the same upward KPI trend → positive correlation."""
        snap_a = _make_snapshot("lab_a", n_obs=20, kpi_slope=0.5, seed=1)
        snap_b = _make_snapshot("lab_b", n_obs=20, kpi_slope=0.5, seed=2)
        comp = CrossLabComparator()
        result = comp.compare(snap_a, snap_b)
        # Both have increasing KPIs with the same slope, best KPIs should be similar
        assert result.kpi_correlation >= 0.0  # ratio-based for single objective

    def test_summary_populated(self):
        snap_a = _make_snapshot("lab_a")
        snap_b = _make_snapshot("lab_b")
        comp = CrossLabComparator()
        result = comp.compare(snap_a, snap_b)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0
        assert "Parameter overlap" in result.summary

    def test_comparison_returns_campaign_comparison(self):
        snap = _make_snapshot("lab")
        comp = CrossLabComparator()
        result = comp.compare(snap, snap)
        assert isinstance(result, CampaignComparison)


# ── ReproducibilityScorer ─────────────────────────────────────────────────────

class TestReproducibilityScorer:

    def test_score_returns_report(self):
        snap = _make_snapshot("repro", n_obs=20)
        scorer = ReproducibilityScorer()
        report = scorer.score(snap)
        assert isinstance(report, ReproducibilityReport)

    def test_score_range_zero_to_one(self):
        snap = _make_snapshot("repro", n_obs=20)
        scorer = ReproducibilityScorer()
        report = scorer.score(snap)
        assert 0.0 <= report.seed_stability <= 1.0
        assert 0.0 <= report.backend_consistency <= 1.0
        assert 0.0 <= report.overall_score <= 1.0

    def test_overall_is_weighted_combination(self):
        snap = _make_snapshot("repro", n_obs=20)
        scorer = ReproducibilityScorer()
        report = scorer.score(snap)
        expected = report.seed_stability * 0.7 + report.backend_consistency * 0.3
        assert abs(report.overall_score - expected) < 1e-9

    def test_details_contain_expected_keys(self):
        snap = _make_snapshot("repro", n_obs=20)
        scorer = ReproducibilityScorer()
        report = scorer.score(snap)
        expected_keys = {
            "n_seeds", "phase_agreement", "backend_agreement",
            "most_common_phase", "most_common_backend",
        }
        assert expected_keys <= set(report.details.keys())

    def test_custom_seeds(self):
        snap = _make_snapshot("repro", n_obs=20)
        scorer = ReproducibilityScorer()
        report = scorer.score(snap, seeds=[10, 20, 30])
        assert report.details["n_seeds"] == 3

    def test_single_backend_set_full_consistency(self):
        """With only one backend config, consistency should be 1.0."""
        snap = _make_snapshot("repro", n_obs=20)
        scorer = ReproducibilityScorer()
        report = scorer.score(snap, backends=[["random", "latin_hypercube", "tpe"]])
        assert report.backend_consistency == 1.0

    def test_deterministic_snapshot_high_stability(self):
        """A well-behaved convergent campaign should have high seed stability."""
        snap = _make_snapshot("repro", n_obs=30, kpi_slope=0.5, seed=42)
        scorer = ReproducibilityScorer()
        report = scorer.score(snap, seeds=[42, 43, 44, 45, 46])
        # With clear convergence, phases should agree across most seeds
        assert report.seed_stability >= 0.5

    def test_cold_start_reproducible(self):
        """A cold-start snapshot (very few obs) should be highly reproducible."""
        snap = _make_snapshot("repro", n_obs=4, seed=42)
        scorer = ReproducibilityScorer()
        report = scorer.score(snap)
        # Cold start is deterministic — always cold_start phase
        assert report.details["most_common_phase"] == "cold_start"
        assert report.seed_stability >= 0.8
