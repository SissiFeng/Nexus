"""Tests for constraint discovery module."""

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.constraints.discovery import (
    ConstraintDiscoverer,
    ConstraintMigrator,
    ConstraintReport,
    ConstraintTracker,
    DiscoveredConstraint,
)


# ── helpers ───────────────────────────────────────────


def _make_snapshot(
    observations: list[Observation],
    specs: list[ParameterSpec] | None = None,
) -> CampaignSnapshot:
    if specs is None:
        specs = [
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
            ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ]
    return CampaignSnapshot(
        campaign_id="test",
        parameter_specs=specs,
        observations=observations,
        objective_names=["y"],
        objective_directions=["minimize"],
        current_iteration=len(observations),
    )


def _obs(x1: float, x2: float, is_failure: bool = False, iteration: int = 0) -> Observation:
    return Observation(
        iteration=iteration,
        parameters={"x1": x1, "x2": x2},
        kpi_values={"y": 0.0 if is_failure else 1.0},
        qc_passed=not is_failure,
        is_failure=is_failure,
        failure_reason="fail" if is_failure else None,
    )


# ── univariate threshold discovery ────────────────────


class TestUnivariateThresholdDiscovery:
    """Failures above a clear cutoff should be detected."""

    def test_clear_threshold(self):
        """When x1 > 0.8 always fails and x1 <= 0.8 always succeeds,
        the discoverer should find a threshold constraint on x1."""
        obs = []
        # successes at low x1
        for i in range(10):
            obs.append(_obs(x1=i * 0.08, x2=0.5, is_failure=False, iteration=i))
        # failures at high x1
        for i in range(10):
            obs.append(_obs(x1=0.82 + i * 0.02, x2=0.5, is_failure=True, iteration=10 + i))

        snap = _make_snapshot(obs)
        discoverer = ConstraintDiscoverer(min_support=3, min_confidence=0.3, n_splits=10)
        report = discoverer.discover(snap)

        # must find at least one threshold constraint on x1
        threshold_constraints = [
            c for c in report.constraints
            if c.constraint_type == "threshold" and "x1" in c.parameters
        ]
        assert len(threshold_constraints) >= 1
        c = threshold_constraints[0]
        assert c.failure_rate_above > c.failure_rate_below
        assert c.threshold_value is not None
        assert 0.5 < c.threshold_value < 0.95
        assert c.n_supporting >= 3

    def test_threshold_on_second_param(self):
        """Threshold can be discovered on x2 as well."""
        obs = []
        for i in range(10):
            obs.append(_obs(x1=0.5, x2=i * 0.07, is_failure=False, iteration=i))
        for i in range(10):
            obs.append(_obs(x1=0.5, x2=0.85 + i * 0.015, is_failure=True, iteration=10 + i))

        snap = _make_snapshot(obs)
        discoverer = ConstraintDiscoverer(min_support=3, min_confidence=0.3, n_splits=10)
        report = discoverer.discover(snap)

        x2_constraints = [
            c for c in report.constraints
            if c.constraint_type == "threshold" and "x2" in c.parameters
        ]
        assert len(x2_constraints) >= 1


# ── interaction threshold discovery ───────────────────


class TestInteractionThresholdDiscovery:
    def test_interaction_quadrant(self):
        """Failures concentrated in the (high x1, high x2) quadrant."""
        obs = []
        # good observations in three quadrants
        for i in range(8):
            obs.append(_obs(x1=0.1 + i * 0.04, x2=0.1 + i * 0.04, is_failure=False, iteration=i))
        for i in range(8):
            obs.append(_obs(x1=0.7 + i * 0.02, x2=0.1 + i * 0.04, is_failure=False, iteration=8 + i))
        for i in range(8):
            obs.append(_obs(x1=0.1 + i * 0.04, x2=0.7 + i * 0.02, is_failure=False, iteration=16 + i))
        # failures in the HH quadrant
        for i in range(8):
            obs.append(_obs(x1=0.7 + i * 0.03, x2=0.7 + i * 0.03, is_failure=True, iteration=24 + i))

        snap = _make_snapshot(obs)
        discoverer = ConstraintDiscoverer(min_support=3, min_confidence=0.3, n_splits=10)
        report = discoverer.discover(snap)

        interaction_constraints = [
            c for c in report.constraints if c.constraint_type == "interaction"
        ]
        assert len(interaction_constraints) >= 1
        ic = interaction_constraints[0]
        assert set(ic.parameters) == {"x1", "x2"}
        assert ic.failure_rate_above > 0.5

    def test_no_interaction_when_uniform(self):
        """No interaction constraint when failures are spread uniformly."""
        obs = []
        # uniform mix across all quadrants
        for i in range(20):
            x1 = (i % 10) * 0.1
            x2 = (i // 2) * 0.1
            obs.append(_obs(x1=x1, x2=x2, is_failure=(i % 5 == 0), iteration=i))

        snap = _make_snapshot(obs)
        discoverer = ConstraintDiscoverer(min_support=3, min_confidence=0.6, n_splits=10)
        report = discoverer.discover(snap)

        interaction_constraints = [
            c for c in report.constraints if c.constraint_type == "interaction"
        ]
        assert len(interaction_constraints) == 0


# ── no constraints when data is clean ─────────────────


class TestNoConstraintsCleanData:
    def test_all_successes(self):
        """No constraints should be discovered when there are zero failures."""
        obs = [_obs(x1=i * 0.1, x2=i * 0.1, is_failure=False, iteration=i) for i in range(10)]
        snap = _make_snapshot(obs)
        discoverer = ConstraintDiscoverer()
        report = discoverer.discover(snap)

        assert len(report.constraints) == 0
        assert report.n_observations_analyzed == 10
        # coverage is 1.0 when there are no failures
        assert report.coverage == 1.0

    def test_empty_observations(self):
        """Empty snapshot produces empty report."""
        snap = _make_snapshot([])
        discoverer = ConstraintDiscoverer()
        report = discoverer.discover(snap)
        assert len(report.constraints) == 0
        assert report.n_observations_analyzed == 0


# ── confidence calculation ────────────────────────────


class TestConfidenceCalculation:
    def test_high_confidence_large_gap(self):
        """Large failure-rate gap + many observations => high confidence."""
        d = ConstraintDiscoverer()
        # diff=0.8, n_supporting=20, n_total=40
        conf = d._compute_confidence(0.8, 20, 40)
        assert conf > 0.6

    def test_low_confidence_small_gap(self):
        """Small gap => lower confidence."""
        d = ConstraintDiscoverer()
        conf = d._compute_confidence(0.1, 2, 20)
        assert conf < 0.5

    def test_confidence_increases_with_support(self):
        """More supporting observations raise confidence."""
        d = ConstraintDiscoverer()
        c_few = d._compute_confidence(0.5, 3, 30)
        c_many = d._compute_confidence(0.5, 15, 30)
        assert c_many > c_few

    def test_confidence_bounded_zero_one(self):
        """Confidence must stay in [0, 1]."""
        d = ConstraintDiscoverer()
        for diff in [0.0, 0.5, 1.0]:
            for n in [1, 10, 100]:
                conf = d._compute_confidence(diff, n, max(n, 1))
                assert 0.0 <= conf <= 1.0


# ── coverage computation ─────────────────────────────


class TestCoverageComputation:
    def test_full_coverage(self):
        """All failures explained when all fall above the threshold."""
        obs = []
        for i in range(10):
            obs.append(_obs(x1=i * 0.05, x2=0.5, is_failure=False, iteration=i))
        for i in range(5):
            obs.append(_obs(x1=0.85 + i * 0.03, x2=0.5, is_failure=True, iteration=10 + i))

        snap = _make_snapshot(obs)
        discoverer = ConstraintDiscoverer(min_support=3, min_confidence=0.3, n_splits=10)
        report = discoverer.discover(snap)

        # all 5 failures are above the threshold, so coverage should be high
        assert report.coverage >= 0.8

    def test_partial_coverage(self):
        """Some failures outside any discovered constraint region."""
        obs = []
        # successes everywhere
        for i in range(10):
            obs.append(_obs(x1=i * 0.08, x2=0.5, is_failure=False, iteration=i))
        # cluster of failures at high x1 (will be explained)
        for i in range(5):
            obs.append(_obs(x1=0.85 + i * 0.03, x2=0.5, is_failure=True, iteration=10 + i))
        # stray failure at low x1 (will NOT be explained by threshold at high x1)
        obs.append(_obs(x1=0.05, x2=0.5, is_failure=True, iteration=15))

        snap = _make_snapshot(obs)
        discoverer = ConstraintDiscoverer(min_support=3, min_confidence=0.3, n_splits=10)
        report = discoverer.discover(snap)

        # at most 5 out of 6 failures explained
        assert report.coverage < 1.0

    def test_zero_coverage_no_constraints(self):
        """If no constraints are found but failures exist, coverage is 0."""
        # scattered failures with no pattern (too few to meet min_support)
        obs = []
        for i in range(10):
            obs.append(_obs(x1=i * 0.1, x2=0.5, is_failure=False, iteration=i))
        obs.append(_obs(x1=0.5, x2=0.5, is_failure=True, iteration=10))

        snap = _make_snapshot(obs)
        discoverer = ConstraintDiscoverer(min_support=5, min_confidence=0.9, n_splits=10)
        report = discoverer.discover(snap)

        if not report.constraints:
            assert report.coverage == 0.0


# ── min_support filtering ────────────────────────────


class TestMinSupportFiltering:
    def test_below_min_support_filtered(self):
        """Constraints with too few supporting observations are excluded."""
        obs = []
        for i in range(15):
            obs.append(_obs(x1=i * 0.06, x2=0.5, is_failure=False, iteration=i))
        # only 2 failures above the threshold
        obs.append(_obs(x1=0.95, x2=0.5, is_failure=True, iteration=15))
        obs.append(_obs(x1=0.98, x2=0.5, is_failure=True, iteration=16))

        snap = _make_snapshot(obs)
        discoverer = ConstraintDiscoverer(min_support=5, min_confidence=0.3, n_splits=10)
        report = discoverer.discover(snap)

        # with min_support=5, the 2-failure cluster should be filtered out
        x1_constraints = [
            c for c in report.constraints
            if c.constraint_type == "threshold" and "x1" in c.parameters
        ]
        assert len(x1_constraints) == 0

    def test_above_min_support_kept(self):
        """Constraints meeting min_support are kept."""
        obs = []
        for i in range(10):
            obs.append(_obs(x1=i * 0.08, x2=0.5, is_failure=False, iteration=i))
        for i in range(5):
            obs.append(_obs(x1=0.85 + i * 0.03, x2=0.5, is_failure=True, iteration=10 + i))

        snap = _make_snapshot(obs)
        discoverer = ConstraintDiscoverer(min_support=3, min_confidence=0.3, n_splits=10)
        report = discoverer.discover(snap)

        x1_constraints = [
            c for c in report.constraints
            if c.constraint_type == "threshold" and "x1" in c.parameters
        ]
        assert len(x1_constraints) >= 1


# ── categorical parameters ───────────────────────────


class TestCategoricalParameters:
    def test_categorical_skipped(self):
        """Categorical parameters should be skipped by univariate discovery."""
        specs = [
            ParameterSpec(
                name="cat1",
                type=VariableType.CATEGORICAL,
                categories=["a", "b", "c"],
            ),
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ]
        obs = []
        for i in range(10):
            obs.append(
                Observation(
                    iteration=i,
                    parameters={"cat1": "a", "x1": i * 0.08},
                    kpi_values={"y": 1.0},
                    is_failure=False,
                )
            )
        for i in range(5):
            obs.append(
                Observation(
                    iteration=10 + i,
                    parameters={"cat1": "b", "x1": 0.85 + i * 0.03},
                    kpi_values={"y": 0.0},
                    is_failure=True,
                )
            )

        snap = _make_snapshot(obs, specs=specs)
        discoverer = ConstraintDiscoverer(min_support=3, min_confidence=0.3, n_splits=10)
        report = discoverer.discover(snap)

        # no threshold constraint should reference the categorical parameter
        for c in report.constraints:
            if c.constraint_type == "threshold":
                assert "cat1" not in c.parameters

    def test_mixed_type_skipped(self):
        """MIXED type parameters should also be skipped."""
        specs = [
            ParameterSpec(name="m1", type=VariableType.MIXED, lower=0.0, upper=1.0),
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ]
        obs = []
        for i in range(10):
            obs.append(
                Observation(
                    iteration=i,
                    parameters={"m1": 0.5, "x1": i * 0.08},
                    kpi_values={"y": 1.0},
                    is_failure=False,
                )
            )
        for i in range(5):
            obs.append(
                Observation(
                    iteration=10 + i,
                    parameters={"m1": 0.9, "x1": 0.85 + i * 0.03},
                    kpi_values={"y": 0.0},
                    is_failure=True,
                )
            )

        snap = _make_snapshot(obs, specs=specs)
        discoverer = ConstraintDiscoverer(min_support=3, min_confidence=0.3, n_splits=10)
        report = discoverer.discover(snap)

        for c in report.constraints:
            if c.constraint_type == "threshold":
                assert "m1" not in c.parameters


# ── edge case: few observations ──────────────────────


class TestFewObservations:
    def test_single_observation(self):
        """A single observation cannot produce constraints."""
        obs = [_obs(x1=0.5, x2=0.5, is_failure=True, iteration=0)]
        snap = _make_snapshot(obs)
        discoverer = ConstraintDiscoverer(min_support=1, min_confidence=0.0, n_splits=10)
        report = discoverer.discover(snap)
        assert report.n_observations_analyzed == 1
        # cannot find meaningful threshold with 1 obs
        assert len(report.constraints) == 0

    def test_two_observations(self):
        """Two observations -- minimal data, no interaction constraints."""
        obs = [
            _obs(x1=0.1, x2=0.1, is_failure=False, iteration=0),
            _obs(x1=0.9, x2=0.9, is_failure=True, iteration=1),
        ]
        snap = _make_snapshot(obs)
        discoverer = ConstraintDiscoverer(min_support=1, min_confidence=0.0, n_splits=10)
        report = discoverer.discover(snap)
        # interaction requires >=4 observations
        interaction_constraints = [
            c for c in report.constraints if c.constraint_type == "interaction"
        ]
        assert len(interaction_constraints) == 0

    def test_three_observations_min_support_three(self):
        """Three observations where two are failures -- barely meets
        support for a threshold constraint."""
        obs = [
            _obs(x1=0.1, x2=0.5, is_failure=False, iteration=0),
            _obs(x1=0.8, x2=0.5, is_failure=True, iteration=1),
            _obs(x1=0.9, x2=0.5, is_failure=True, iteration=2),
        ]
        snap = _make_snapshot(obs)
        # min_support=3 should filter out the 2-failure cluster
        discoverer = ConstraintDiscoverer(min_support=3, min_confidence=0.3, n_splits=10)
        report = discoverer.discover(snap)
        threshold_constraints = [
            c for c in report.constraints if c.constraint_type == "threshold"
        ]
        assert len(threshold_constraints) == 0

        # min_support=2 should keep it
        discoverer2 = ConstraintDiscoverer(min_support=2, min_confidence=0.0, n_splits=10)
        report2 = discoverer2.discover(snap)
        threshold_constraints2 = [
            c for c in report2.constraints if c.constraint_type == "threshold"
        ]
        assert len(threshold_constraints2) >= 1


# ── report structure ─────────────────────────────────


class TestReportStructure:
    def test_report_fields(self):
        """ConstraintReport has expected fields."""
        obs = [_obs(x1=i * 0.1, x2=0.5, is_failure=False, iteration=i) for i in range(5)]
        snap = _make_snapshot(obs)
        report = ConstraintDiscoverer().discover(snap)
        assert isinstance(report, ConstraintReport)
        assert isinstance(report.constraints, list)
        assert isinstance(report.n_observations_analyzed, int)
        assert isinstance(report.coverage, float)

    def test_discovered_constraint_fields(self):
        """DiscoveredConstraint has expected fields with correct types."""
        c = DiscoveredConstraint(
            constraint_type="threshold",
            parameters=["x1"],
            condition="x1 > 0.8 -> 80% failure rate",
            threshold_value=0.8,
            failure_rate_above=0.8,
            failure_rate_below=0.1,
            confidence=0.9,
            n_supporting=10,
        )
        assert c.constraint_type == "threshold"
        assert c.parameters == ["x1"]
        assert c.threshold_value == 0.8
        assert c.confidence == 0.9


# ── constraint migration ──────────────────────────────


class TestConstraintMigrator:

    def _make_source_report(self) -> ConstraintReport:
        return ConstraintReport(
            constraints=[
                DiscoveredConstraint(
                    constraint_type="threshold",
                    parameters=["x1"],
                    condition="x1 > 0.8 -> 80% failure rate",
                    threshold_value=0.8,
                    failure_rate_above=0.8,
                    failure_rate_below=0.1,
                    confidence=0.9,
                    n_supporting=10,
                ),
                DiscoveredConstraint(
                    constraint_type="threshold",
                    parameters=["x2"],
                    condition="x2 > 0.7 -> 70% failure rate",
                    threshold_value=0.7,
                    failure_rate_above=0.7,
                    failure_rate_below=0.1,
                    confidence=0.4,  # below default min_confidence
                    n_supporting=5,
                ),
                DiscoveredConstraint(
                    constraint_type="threshold",
                    parameters=["x3"],
                    condition="x3 > 0.5 -> 60% failure rate",
                    threshold_value=0.5,
                    failure_rate_above=0.6,
                    failure_rate_below=0.1,
                    confidence=0.8,
                    n_supporting=8,
                ),
            ],
            n_observations_analyzed=50,
            coverage=0.9,
        )

    def test_compatible_parameters_migrated(self):
        """Constraints whose parameters exist in target are kept."""
        source = self._make_source_report()
        # target has x1 and x2 but not x3
        target = _make_snapshot([
            _obs(x1=0.5, x2=0.5, iteration=0),
        ])
        migrator = ConstraintMigrator()
        result = migrator.migrate(source, target)
        # x1 constraint (confidence=0.9 >= 0.5) is kept
        # x2 constraint (confidence=0.4 < 0.5) is filtered
        # x3 does not exist in target
        param_names = [c.parameters[0] for c in result.constraints]
        assert "x1" in param_names
        assert "x2" not in param_names
        assert "x3" not in param_names

    def test_threshold_outside_range_filtered(self):
        """Constraint with threshold outside target range is excluded."""
        source = ConstraintReport(
            constraints=[
                DiscoveredConstraint(
                    constraint_type="threshold",
                    parameters=["x1"],
                    condition="x1 > 2.0 -> failure",
                    threshold_value=2.0,  # outside [0, 1]
                    failure_rate_above=0.9,
                    failure_rate_below=0.1,
                    confidence=0.9,
                    n_supporting=10,
                ),
            ],
            n_observations_analyzed=20,
            coverage=0.8,
        )
        target = _make_snapshot([_obs(x1=0.5, x2=0.5)])
        result = ConstraintMigrator().migrate(source, target)
        assert len(result.constraints) == 0

    def test_empty_source_returns_empty(self):
        source = ConstraintReport(constraints=[], n_observations_analyzed=0, coverage=1.0)
        target = _make_snapshot([_obs(x1=0.5, x2=0.5)])
        result = ConstraintMigrator().migrate(source, target)
        assert len(result.constraints) == 0


# ── constraint tightening tracking ────────────────────


class TestConstraintTracker:

    def test_tightened_constraint_detected(self):
        """Higher failure rate in new report = tightened."""
        old = ConstraintReport(
            constraints=[
                DiscoveredConstraint(
                    constraint_type="threshold",
                    parameters=["x1"],
                    condition="x1 > 0.8",
                    threshold_value=0.8,
                    failure_rate_above=0.5,
                    failure_rate_below=0.1,
                    confidence=0.8,
                    n_supporting=10,
                ),
            ],
            n_observations_analyzed=30,
            coverage=0.8,
        )
        new = ConstraintReport(
            constraints=[
                DiscoveredConstraint(
                    constraint_type="threshold",
                    parameters=["x1"],
                    condition="x1 > 0.8",
                    threshold_value=0.75,
                    failure_rate_above=0.8,  # increased from 0.5
                    failure_rate_below=0.1,
                    confidence=0.9,
                    n_supporting=15,
                ),
            ],
            n_observations_analyzed=50,
            coverage=0.9,
        )
        tracker = ConstraintTracker()
        evolution = tracker.compare(old, new)
        assert evolution.n_tightened == 1
        assert evolution.n_relaxed == 0
        assert evolution.deltas[0].direction == "tightened"

    def test_new_constraint_detected(self):
        """Constraint appearing only in new report is 'new'."""
        old = ConstraintReport(constraints=[], n_observations_analyzed=20, coverage=1.0)
        new = ConstraintReport(
            constraints=[
                DiscoveredConstraint(
                    constraint_type="threshold",
                    parameters=["x1"],
                    condition="x1 > 0.7",
                    threshold_value=0.7,
                    failure_rate_above=0.6,
                    failure_rate_below=0.1,
                    confidence=0.7,
                    n_supporting=5,
                ),
            ],
            n_observations_analyzed=30,
            coverage=0.8,
        )
        evolution = ConstraintTracker().compare(old, new)
        assert evolution.n_new == 1
        assert evolution.deltas[0].direction == "new"

    def test_removed_constraint_detected(self):
        """Constraint only in old report is 'removed'."""
        old = ConstraintReport(
            constraints=[
                DiscoveredConstraint(
                    constraint_type="threshold",
                    parameters=["x2"],
                    condition="x2 > 0.8",
                    threshold_value=0.8,
                    failure_rate_above=0.7,
                    failure_rate_below=0.1,
                    confidence=0.8,
                    n_supporting=8,
                ),
            ],
            n_observations_analyzed=30,
            coverage=0.7,
        )
        new = ConstraintReport(constraints=[], n_observations_analyzed=40, coverage=1.0)
        evolution = ConstraintTracker().compare(old, new)
        assert evolution.n_removed == 1
        assert evolution.deltas[0].direction == "removed"

    def test_unchanged_constraint(self):
        """Same failure rate in both reports = unchanged."""
        c = DiscoveredConstraint(
            constraint_type="threshold",
            parameters=["x1"],
            condition="x1 > 0.8",
            threshold_value=0.8,
            failure_rate_above=0.7,
            failure_rate_below=0.1,
            confidence=0.8,
            n_supporting=10,
        )
        old = ConstraintReport(constraints=[c], n_observations_analyzed=30, coverage=0.8)
        new = ConstraintReport(constraints=[c], n_observations_analyzed=40, coverage=0.8)
        evolution = ConstraintTracker().compare(old, new)
        assert evolution.n_tightened == 0
        assert evolution.n_relaxed == 0
        assert evolution.deltas[0].direction == "unchanged"
