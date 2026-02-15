"""Tests for the DimensionAnalyzer and ProblemFingerprint dimension fields.

Covers:
- Fixed dimension detection (nunique == 1)
- Effective dimension detection (nunique > 1)
- Simplification hints: degenerate, ranking, bandit, line_search, reduced_bo, full_bo
- Edge cases: empty observations, all fixed, all varying
- ProblemFingerprint new fields and serialization
- Integration: fingerprint changes when dimension analysis is applied
"""

from __future__ import annotations

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    ProblemFingerprint,
    VariableType,
)
from optimization_copilot.profiler.dimension_analyzer import (
    DimensionAnalyzer,
    DimensionAnalysis,
)


# ── Helpers ───────────────────────────────────────────────


def _make_snapshot(
    *,
    specs: list[ParameterSpec] | None = None,
    observations: list[Observation] | None = None,
    n_obs: int = 10,
    param_fn=None,
    objective_names: list[str] | None = None,
    objective_directions: list[str] | None = None,
) -> CampaignSnapshot:
    """Build a CampaignSnapshot for dimension analyzer tests.

    If *observations* is given it is used directly; otherwise *n_obs*
    observations are generated using *param_fn*.
    """
    if specs is None:
        specs = [
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
            ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
            ParameterSpec(name="x3", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
            ParameterSpec(name="x4", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ]
    if objective_names is None:
        objective_names = ["y"]
    if objective_directions is None:
        objective_directions = ["maximize"]

    if observations is not None:
        obs = observations
    else:
        if param_fn is None:
            param_fn = lambda i: {s.name: float(i) * 0.1 for s in specs}
        obs = [
            Observation(
                iteration=i,
                parameters=param_fn(i),
                kpi_values={"y": float(i)},
                timestamp=float(i),
            )
            for i in range(n_obs)
        ]

    return CampaignSnapshot(
        campaign_id="dim-test",
        parameter_specs=specs,
        observations=obs,
        objective_names=objective_names,
        objective_directions=objective_directions,
        current_iteration=len(obs),
    )


analyzer = DimensionAnalyzer()


# ── Tests: all dimensions vary ────────────────────────────


class TestAllVary:
    def test_no_fixed_dimensions(self):
        """When all parameters vary, fixed_dimensions should be empty."""
        snap = _make_snapshot()
        result = analyzer.analyze(snap)
        assert result.fixed_dimensions == []

    def test_all_effective(self):
        """All parameters should be in effective_dimensions."""
        snap = _make_snapshot()
        result = analyzer.analyze(snap)
        assert set(result.effective_dimensions) == {"x1", "x2", "x3", "x4"}

    def test_counts(self):
        snap = _make_snapshot()
        result = analyzer.analyze(snap)
        assert result.n_original == 4
        assert result.n_effective == 4

    def test_full_bo_hint(self):
        """4 effective dimensions -> full_bo."""
        snap = _make_snapshot()
        result = analyzer.analyze(snap)
        assert result.simplification_hint == "full_bo"


# ── Tests: fixed dimension detection ──────────────────────


class TestFixedDetection:
    def test_one_fixed(self):
        """One parameter has nunique==1 -> detected as fixed."""
        snap = _make_snapshot(
            param_fn=lambda i: {
                "x1": 0.5,  # fixed
                "x2": float(i) * 0.1,
                "x3": float(i) * 0.2,
                "x4": float(i) * 0.3,
            }
        )
        result = analyzer.analyze(snap)
        assert result.fixed_dimensions == ["x1"]
        assert "x1" not in result.effective_dimensions
        assert result.n_effective == 3

    def test_two_fixed(self):
        """Two parameters fixed -> both detected."""
        snap = _make_snapshot(
            param_fn=lambda i: {
                "x1": 0.5,  # fixed
                "x2": float(i) * 0.1,
                "x3": 1.0,  # fixed
                "x4": float(i) * 0.3,
            }
        )
        result = analyzer.analyze(snap)
        assert set(result.fixed_dimensions) == {"x1", "x3"}
        assert set(result.effective_dimensions) == {"x2", "x4"}
        assert result.n_effective == 2

    def test_mixed_fixed_varying(self):
        """Three fixed, one varying."""
        snap = _make_snapshot(
            param_fn=lambda i: {
                "x1": 0.5,
                "x2": 0.5,
                "x3": 0.5,
                "x4": float(i) * 0.1,
            }
        )
        result = analyzer.analyze(snap)
        assert set(result.fixed_dimensions) == {"x1", "x2", "x3"}
        assert result.effective_dimensions == ["x4"]
        assert result.n_effective == 1


# ── Tests: all dimensions fixed (degenerate) ─────────────


class TestDegenerate:
    def test_all_fixed(self):
        """All parameters constant -> hint = 'degenerate'."""
        snap = _make_snapshot(
            param_fn=lambda i: {"x1": 1.0, "x2": 2.0, "x3": 3.0, "x4": 4.0}
        )
        result = analyzer.analyze(snap)
        assert result.n_effective == 0
        assert result.simplification_hint == "degenerate"

    def test_all_fixed_lists(self):
        """All dimensions should be in fixed_dimensions."""
        snap = _make_snapshot(
            param_fn=lambda i: {"x1": 1.0, "x2": 2.0, "x3": 3.0, "x4": 4.0}
        )
        result = analyzer.analyze(snap)
        assert set(result.fixed_dimensions) == {"x1", "x2", "x3", "x4"}
        assert result.effective_dimensions == []


# ── Tests: single effective categorical → ranking/bandit ──


class TestCategoricalSingleDim:
    def test_ranking_many_categories(self):
        """Single effective categorical with >5 unique values -> 'ranking'."""
        specs = [
            ParameterSpec(
                name="cat", type=VariableType.CATEGORICAL,
                categories=["a", "b", "c", "d", "e", "f", "g"],
            ),
            ParameterSpec(name="fixed1", type=VariableType.CONTINUOUS, lower=0, upper=1),
        ]
        obs = [
            Observation(
                iteration=i,
                parameters={"cat": specs[0].categories[i % 7], "fixed1": 0.5},
                kpi_values={"y": float(i)},
                timestamp=float(i),
            )
            for i in range(14)
        ]
        snap = _make_snapshot(specs=specs, observations=obs)
        result = analyzer.analyze(snap)
        assert result.fixed_dimensions == ["fixed1"]
        assert result.effective_dimensions == ["cat"]
        assert result.simplification_hint == "ranking"

    def test_bandit_few_categories(self):
        """Single effective categorical with <=5 unique values -> 'bandit'."""
        specs = [
            ParameterSpec(
                name="cat", type=VariableType.CATEGORICAL,
                categories=["a", "b", "c"],
            ),
            ParameterSpec(name="fixed1", type=VariableType.CONTINUOUS, lower=0, upper=1),
        ]
        obs = [
            Observation(
                iteration=i,
                parameters={"cat": specs[0].categories[i % 3], "fixed1": 0.5},
                kpi_values={"y": float(i)},
                timestamp=float(i),
            )
            for i in range(9)
        ]
        snap = _make_snapshot(specs=specs, observations=obs)
        result = analyzer.analyze(snap)
        assert result.simplification_hint == "bandit"

    def test_bandit_exactly_five(self):
        """Exactly 5 unique categorical values -> 'bandit' (<=5)."""
        specs = [
            ParameterSpec(
                name="cat", type=VariableType.CATEGORICAL,
                categories=["a", "b", "c", "d", "e"],
            ),
            ParameterSpec(name="fixed1", type=VariableType.CONTINUOUS, lower=0, upper=1),
        ]
        obs = [
            Observation(
                iteration=i,
                parameters={"cat": specs[0].categories[i % 5], "fixed1": 0.5},
                kpi_values={"y": float(i)},
                timestamp=float(i),
            )
            for i in range(10)
        ]
        snap = _make_snapshot(specs=specs, observations=obs)
        result = analyzer.analyze(snap)
        assert result.simplification_hint == "bandit"

    def test_ranking_six_categories(self):
        """6 unique categorical values -> 'ranking' (>5)."""
        specs = [
            ParameterSpec(
                name="cat", type=VariableType.CATEGORICAL,
                categories=["a", "b", "c", "d", "e", "f"],
            ),
            ParameterSpec(name="fixed1", type=VariableType.CONTINUOUS, lower=0, upper=1),
        ]
        obs = [
            Observation(
                iteration=i,
                parameters={"cat": specs[0].categories[i % 6], "fixed1": 0.5},
                kpi_values={"y": float(i)},
                timestamp=float(i),
            )
            for i in range(12)
        ]
        snap = _make_snapshot(specs=specs, observations=obs)
        result = analyzer.analyze(snap)
        assert result.simplification_hint == "ranking"


# ── Tests: single effective continuous → line_search ──────


class TestLinSearch:
    def test_single_continuous_effective(self):
        """Single effective continuous dimension -> 'line_search'."""
        specs = [
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0, upper=1),
            ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0, upper=1),
        ]
        snap = _make_snapshot(
            specs=specs,
            param_fn=lambda i: {"x1": float(i) * 0.1, "x2": 0.5},
        )
        result = analyzer.analyze(snap)
        assert result.simplification_hint == "line_search"

    def test_single_discrete_effective(self):
        """Single effective discrete dimension -> 'line_search'."""
        specs = [
            ParameterSpec(name="d1", type=VariableType.DISCRETE, lower=1, upper=10),
            ParameterSpec(name="d2", type=VariableType.DISCRETE, lower=1, upper=10),
        ]
        snap = _make_snapshot(
            specs=specs,
            param_fn=lambda i: {"d1": i + 1, "d2": 5},
        )
        result = analyzer.analyze(snap)
        assert result.simplification_hint == "line_search"


# ── Tests: reduced_bo (2-3 effective dims) ────────────────


class TestReducedBo:
    def test_two_effective(self):
        """Two effective dimensions -> 'reduced_bo'."""
        snap = _make_snapshot(
            param_fn=lambda i: {
                "x1": float(i) * 0.1,
                "x2": float(i) * 0.2,
                "x3": 0.5,  # fixed
                "x4": 0.5,  # fixed
            }
        )
        result = analyzer.analyze(snap)
        assert result.n_effective == 2
        assert result.simplification_hint == "reduced_bo"

    def test_three_effective(self):
        """Three effective dimensions -> 'reduced_bo'."""
        snap = _make_snapshot(
            param_fn=lambda i: {
                "x1": float(i) * 0.1,
                "x2": float(i) * 0.2,
                "x3": float(i) * 0.3,
                "x4": 0.5,  # fixed
            }
        )
        result = analyzer.analyze(snap)
        assert result.n_effective == 3
        assert result.simplification_hint == "reduced_bo"


# ── Tests: empty observations ─────────────────────────────


class TestEmptyObservations:
    def test_no_observations_all_fixed(self):
        """With no observations, all parameters are treated as fixed (nunique=0 <= 1)."""
        snap = _make_snapshot(n_obs=0, observations=[])
        result = analyzer.analyze(snap)
        assert result.n_effective == 0
        assert set(result.fixed_dimensions) == {"x1", "x2", "x3", "x4"}
        assert result.simplification_hint == "degenerate"

    def test_single_observation(self):
        """Single observation: all params have nunique=1 -> all fixed."""
        snap = _make_snapshot(n_obs=1)
        result = analyzer.analyze(snap)
        assert result.n_effective == 0
        assert result.simplification_hint == "degenerate"


# ── Tests: no parameter specs ─────────────────────────────


class TestNoSpecs:
    def test_empty_specs(self):
        """No parameter specs -> n_original=0, n_effective=0, degenerate."""
        snap = _make_snapshot(
            specs=[],
            observations=[
                Observation(
                    iteration=0,
                    parameters={},
                    kpi_values={"y": 1.0},
                    timestamp=0.0,
                ),
            ],
        )
        result = analyzer.analyze(snap)
        assert result.n_original == 0
        assert result.n_effective == 0
        assert result.simplification_hint == "degenerate"


# ── Tests: parameter with None values ─────────────────────


class TestNoneValues:
    def test_none_values_ignored(self):
        """Parameters with None values should not count toward unique values."""
        specs = [
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0, upper=1),
        ]
        obs = [
            Observation(iteration=0, parameters={"x1": None}, kpi_values={"y": 1.0}, timestamp=0.0),
            Observation(iteration=1, parameters={"x1": 0.5}, kpi_values={"y": 2.0}, timestamp=1.0),
            Observation(iteration=2, parameters={"x1": None}, kpi_values={"y": 3.0}, timestamp=2.0),
        ]
        snap = _make_snapshot(specs=specs, observations=obs)
        result = analyzer.analyze(snap)
        # Only one non-None unique value -> fixed
        assert result.fixed_dimensions == ["x1"]

    def test_missing_parameter_key(self):
        """Parameter key missing from observation -> treated as None, not counted."""
        specs = [
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0, upper=1),
            ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0, upper=1),
        ]
        obs = [
            Observation(iteration=0, parameters={"x1": 0.1}, kpi_values={"y": 1.0}, timestamp=0.0),
            Observation(iteration=1, parameters={"x1": 0.2, "x2": 0.5}, kpi_values={"y": 2.0}, timestamp=1.0),
            Observation(iteration=2, parameters={"x1": 0.3}, kpi_values={"y": 3.0}, timestamp=2.0),
        ]
        snap = _make_snapshot(specs=specs, observations=obs)
        result = analyzer.analyze(snap)
        # x1 has 3 unique -> effective; x2 has 1 unique -> fixed
        assert "x1" in result.effective_dimensions
        assert "x2" in result.fixed_dimensions


# ── Tests: DimensionAnalysis dataclass defaults ───────────


class TestDimensionAnalysisDefaults:
    def test_defaults(self):
        da = DimensionAnalysis()
        assert da.fixed_dimensions == []
        assert da.effective_dimensions == []
        assert da.n_original == 0
        assert da.n_effective == 0
        assert da.simplification_hint == "full_bo"


# ── Tests: ProblemFingerprint new fields ──────────────────


class TestFingerprintNewFields:
    def test_default_values(self):
        """New fields should have sensible defaults."""
        fp = ProblemFingerprint()
        assert fp.fixed_dimensions == []
        assert fp.effective_dimensionality == -1
        assert fp.simplification_hint == "full_bo"
        assert fp.encoding_metadata == {}

    def test_to_dict_handles_mixed_types(self):
        """to_dict must handle both enum fields and non-enum fields."""
        fp = ProblemFingerprint(
            fixed_dimensions=["x1", "x2"],
            effective_dimensionality=3,
            simplification_hint="reduced_bo",
            encoding_metadata={"key": "val"},
        )
        d = fp.to_dict()
        assert d["variable_types"] == "continuous"  # enum -> .value
        assert d["fixed_dimensions"] == ["x1", "x2"]  # list -> as-is
        assert d["effective_dimensionality"] == 3  # int -> as-is
        assert d["simplification_hint"] == "reduced_bo"  # str -> as-is
        assert d["encoding_metadata"] == {"key": "val"}  # dict -> as-is

    def test_to_dict_all_enum_fields_still_work(self):
        """Existing enum fields should still serialize to string values."""
        fp = ProblemFingerprint()
        d = fp.to_dict()
        assert d["noise_regime"] == "low"
        assert d["data_scale"] == "tiny"

    def test_to_tuple_includes_new_fields(self):
        """to_tuple should include the new fields at the end."""
        fp = ProblemFingerprint(
            fixed_dimensions=["a", "b"],
            effective_dimensionality=5,
            simplification_hint="bandit",
        )
        t = fp.to_tuple()
        assert len(t) == 11
        # Last 3 elements
        assert t[8] == ("a", "b")  # tuple of fixed_dimensions
        assert t[9] == 5
        assert t[10] == "bandit"

    def test_to_tuple_empty_fixed(self):
        """Empty fixed_dimensions -> empty tuple."""
        fp = ProblemFingerprint()
        t = fp.to_tuple()
        assert t[8] == ()
        assert t[9] == -1  # default effective_dimensionality
        assert t[10] == "full_bo"

    def test_to_dict_length(self):
        """to_dict should now have 12 fields (8 original + 4 new)."""
        fp = ProblemFingerprint()
        d = fp.to_dict()
        assert len(d) == 12


# ── KEY ACCEPTANCE TEST: fingerprint changes with fixed dims ──


class TestFingerprintChangesWithFixedDims:
    """Integration test showing that dimension analysis results
    properly update a ProblemFingerprint and that serialization
    includes the new fields.
    """

    def test_fingerprint_changes_with_fixed_dims(self):
        """Create a snapshot where 2 of 4 params are fixed,
        run DimensionAnalyzer, apply results to ProblemFingerprint,
        verify the fingerprint reflects the analysis.
        """
        # Snapshot: x1 and x3 are fixed, x2 and x4 vary
        snap = _make_snapshot(
            param_fn=lambda i: {
                "x1": 0.5,  # fixed
                "x2": float(i) * 0.1,  # varies
                "x3": 1.0,  # fixed
                "x4": float(i) * 0.2,  # varies
            }
        )

        # Run dimension analysis
        dim_result = analyzer.analyze(snap)
        assert set(dim_result.fixed_dimensions) == {"x1", "x3"}
        assert set(dim_result.effective_dimensions) == {"x2", "x4"}
        assert dim_result.n_effective == 2
        assert dim_result.simplification_hint == "reduced_bo"

        # Apply to a ProblemFingerprint
        fp = ProblemFingerprint(
            fixed_dimensions=dim_result.fixed_dimensions,
            effective_dimensionality=dim_result.n_effective,
            simplification_hint=dim_result.simplification_hint,
        )

        # Verify the fingerprint reflects the dimension analysis
        assert fp.fixed_dimensions == ["x1", "x3"]
        assert fp.effective_dimensionality == 2
        assert fp.simplification_hint == "reduced_bo"

        # Verify to_dict includes new fields
        d = fp.to_dict()
        assert d["fixed_dimensions"] == ["x1", "x3"]
        assert d["effective_dimensionality"] == 2
        assert d["simplification_hint"] == "reduced_bo"

        # Verify to_tuple includes new fields
        t = fp.to_tuple()
        assert t[8] == ("x1", "x3")
        assert t[9] == 2
        assert t[10] == "reduced_bo"

    def test_default_fingerprint_differs_from_analysed(self):
        """A default fingerprint should differ from one with analysis applied."""
        default_fp = ProblemFingerprint()
        analysed_fp = ProblemFingerprint(
            fixed_dimensions=["x1"],
            effective_dimensionality=3,
            simplification_hint="reduced_bo",
        )

        assert default_fp.fixed_dimensions != analysed_fp.fixed_dimensions
        assert default_fp.effective_dimensionality != analysed_fp.effective_dimensionality
        assert default_fp.simplification_hint != analysed_fp.simplification_hint
        assert default_fp.to_tuple() != analysed_fp.to_tuple()

    def test_fingerprint_to_dict_all_types_correct(self):
        """Verify every field in to_dict has the correct type."""
        fp = ProblemFingerprint(
            fixed_dimensions=["a"],
            effective_dimensionality=2,
            simplification_hint="line_search",
            encoding_metadata={"enc": True},
        )
        d = fp.to_dict()

        # Enum fields should be strings
        for key in [
            "variable_types", "objective_form", "noise_regime",
            "cost_profile", "failure_informativeness", "data_scale",
            "dynamics", "feasible_region",
        ]:
            assert isinstance(d[key], str), f"{key} should be str"

        # New fields
        assert isinstance(d["fixed_dimensions"], list)
        assert isinstance(d["effective_dimensionality"], int)
        assert isinstance(d["simplification_hint"], str)
        assert isinstance(d["encoding_metadata"], dict)
