"""Tests for failure taxonomy and conditional failure modelling."""

from __future__ import annotations

import unittest

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.feasibility.taxonomy import (
    ClassifiedFailure,
    FailureClassifier,
    FailureTaxonomy,
    FailureType,
)


# ── Helpers ───────────────────────────────────────────────


def _make_spec(name: str, lo: float = 0.0, hi: float = 10.0) -> ParameterSpec:
    return ParameterSpec(name=name, type=VariableType.CONTINUOUS, lower=lo, upper=hi)


def _make_obs(
    iteration: int,
    params: dict,
    *,
    kpi: dict | None = None,
    is_failure: bool = False,
    failure_reason: str | None = None,
    qc_passed: bool = True,
) -> Observation:
    return Observation(
        iteration=iteration,
        parameters=params,
        kpi_values=kpi or {},
        qc_passed=qc_passed,
        is_failure=is_failure,
        failure_reason=failure_reason,
    )


def _make_snapshot(
    specs: list[ParameterSpec],
    observations: list[Observation],
) -> CampaignSnapshot:
    return CampaignSnapshot(
        campaign_id="test",
        parameter_specs=specs,
        observations=observations,
        objective_names=["yield"],
        objective_directions=["maximize"],
    )


# ── Tests ─────────────────────────────────────────────────


class TestFailureType(unittest.TestCase):
    """Smoke test the enum values."""

    def test_values(self) -> None:
        self.assertEqual(FailureType.HARDWARE.value, "hardware")
        self.assertEqual(FailureType.CHEMISTRY.value, "chemistry")
        self.assertEqual(FailureType.DATA.value, "data")
        self.assertEqual(FailureType.PROTOCOL.value, "protocol")
        self.assertEqual(FailureType.UNKNOWN.value, "unknown")


class TestHardwareClassification(unittest.TestCase):
    """Hardware failures should be detected via keyword matching."""

    def test_hardware_keyword_timeout(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(0, {"x": 5.0}, kpi={"yield": 0.5}),
            _make_obs(
                1, {"x": 3.0},
                is_failure=True,
                failure_reason="instrument timeout during measurement",
            ),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        hw_failures = [
            f for f in taxonomy.classified_failures
            if f.failure_type == FailureType.HARDWARE
        ]
        self.assertGreaterEqual(len(hw_failures), 1)
        self.assertIn("hardware", [kw for e in hw_failures[0].evidence for kw in e.split()])

    def test_hardware_keyword_sensor(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(0, {"x": 5.0}, kpi={"yield": 0.8}),
            _make_obs(
                1, {"x": 7.0},
                is_failure=True,
                failure_reason="sensor malfunction",
            ),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        hw = [f for f in taxonomy.classified_failures if f.failure_type == FailureType.HARDWARE]
        self.assertEqual(len(hw), 1)
        self.assertGreater(hw[0].confidence, 0.0)

    def test_hardware_keyword_power(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(
                0, {"x": 2.0},
                is_failure=True,
                failure_reason="power failure in reactor",
            ),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        self.assertEqual(taxonomy.classified_failures[0].failure_type, FailureType.HARDWARE)


class TestChemistryClassification(unittest.TestCase):
    """Chemistry failures should be detected via keyword and clustering."""

    def test_chemistry_keyword_precipitate(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(0, {"x": 5.0}, kpi={"yield": 0.5}),
            _make_obs(
                1, {"x": 8.0},
                is_failure=True,
                failure_reason="precipitate formed in reactor",
            ),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        ch = [f for f in taxonomy.classified_failures if f.failure_type == FailureType.CHEMISTRY]
        self.assertEqual(len(ch), 1)

    def test_chemistry_keyword_pH(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(
                0, {"x": 1.0},
                is_failure=True,
                failure_reason="pH out of range",
            ),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        self.assertEqual(taxonomy.classified_failures[0].failure_type, FailureType.CHEMISTRY)


class TestDataClassification(unittest.TestCase):
    """Data failures: qc_passed=False without is_failure flag."""

    def test_qc_failed_not_is_failure(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(0, {"x": 5.0}, kpi={"yield": 0.5}, qc_passed=False),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        data_failures = [
            f for f in taxonomy.classified_failures
            if f.failure_type == FailureType.DATA
        ]
        self.assertEqual(len(data_failures), 1)
        self.assertGreater(data_failures[0].confidence, 0.0)

    def test_is_failure_with_kpi_and_qc_false(self) -> None:
        """is_failure=True + qc_passed=False + kpi_values => DATA signal present."""
        specs = [_make_spec("x")]
        obs = [
            _make_obs(
                0, {"x": 5.0},
                kpi={"yield": 0.1},
                is_failure=True,
                failure_reason="",
                qc_passed=False,
            ),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        # The observation is is_failure, so it goes through _classify_single.
        # It has kpi_values and qc_passed=False which gives DATA signal.
        cf = taxonomy.classified_failures[0]
        self.assertEqual(cf.failure_type, FailureType.DATA)


class TestSystematicVsSporadic(unittest.TestCase):
    """Spatially clustered failures -> CHEMISTRY; scattered -> HARDWARE/UNKNOWN."""

    def test_clustered_failures_classified_as_chemistry(self) -> None:
        """Failures all near x=8.0 should be systematic -> chemistry."""
        specs = [_make_spec("x", 0.0, 10.0)]
        obs = [
            _make_obs(0, {"x": 1.0}, kpi={"yield": 0.9}),
            _make_obs(1, {"x": 7.8}, is_failure=True, failure_reason=""),
            _make_obs(2, {"x": 8.0}, is_failure=True, failure_reason=""),
            _make_obs(3, {"x": 8.2}, is_failure=True, failure_reason=""),
            _make_obs(4, {"x": 2.0}, kpi={"yield": 0.7}),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        chem = [f for f in taxonomy.classified_failures if f.failure_type == FailureType.CHEMISTRY]
        self.assertGreaterEqual(len(chem), 1)
        # Check evidence mentions clustering
        all_evidence = " ".join(e for cf in chem for e in cf.evidence)
        self.assertIn("cluster", all_evidence)

    def test_scattered_failures_not_chemistry(self) -> None:
        """Failures spread across the space should not be chemistry-systematic."""
        specs = [_make_spec("x", 0.0, 10.0)]
        obs = [
            _make_obs(0, {"x": 5.0}, kpi={"yield": 0.5}),
            _make_obs(1, {"x": 0.5}, is_failure=True, failure_reason=""),
            _make_obs(2, {"x": 5.0}, is_failure=True, failure_reason=""),
            _make_obs(3, {"x": 9.5}, is_failure=True, failure_reason=""),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        # With no keywords and scattered positions, these should not be CHEMISTRY
        chem = [f for f in taxonomy.classified_failures if f.failure_type == FailureType.CHEMISTRY]
        # All failures are scattered -> sporadic evidence, none should be CHEMISTRY
        self.assertEqual(len(chem), 0)


class TestStrategyAdjustments(unittest.TestCase):
    """Test that strategy adjustments match the dominant failure type."""

    def _dominant_snapshot(self, failure_reason: str, n: int = 3) -> CampaignSnapshot:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(i, {"x": float(i)}, is_failure=True, failure_reason=failure_reason)
            for i in range(n)
        ]
        return _make_snapshot(specs, obs)

    def test_hardware_dominant_reduce_exploration(self) -> None:
        snap = self._dominant_snapshot("instrument timeout")
        taxonomy = FailureClassifier().classify(snap)
        self.assertEqual(taxonomy.dominant_type, FailureType.HARDWARE)
        self.assertEqual(taxonomy.strategy_adjustments["dominant"], "reduce_exploration")

    def test_chemistry_dominant_adjust_bounds(self) -> None:
        snap = self._dominant_snapshot("precipitate formed")
        taxonomy = FailureClassifier().classify(snap)
        self.assertEqual(taxonomy.dominant_type, FailureType.CHEMISTRY)
        self.assertEqual(taxonomy.strategy_adjustments["dominant"], "adjust_bounds")

    def test_data_dominant_increase_replicates(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(
                i, {"x": float(i)},
                kpi={"yield": 0.1},
                is_failure=True,
                failure_reason="",
                qc_passed=False,
            )
            for i in range(3)
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        self.assertEqual(taxonomy.dominant_type, FailureType.DATA)
        self.assertEqual(taxonomy.strategy_adjustments["dominant"], "increase_replicates")

    def test_protocol_dominant_enforce_checks(self) -> None:
        snap = self._dominant_snapshot("protocol violation step 3")
        taxonomy = FailureClassifier().classify(snap)
        self.assertEqual(taxonomy.dominant_type, FailureType.PROTOCOL)
        self.assertEqual(taxonomy.strategy_adjustments["dominant"], "enforce_protocol_checks")

    def test_unknown_dominant_conservative(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(0, {"x": 1.0}, is_failure=True, failure_reason=""),
            _make_obs(1, {"x": 5.0}, is_failure=True, failure_reason=""),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        # With no keywords and only 2 failures (not enough for clustering),
        # the dominant type should be UNKNOWN
        self.assertEqual(taxonomy.dominant_type, FailureType.UNKNOWN)
        self.assertEqual(taxonomy.strategy_adjustments["dominant"], "conservative_exploration")


class TestNoFailures(unittest.TestCase):
    """Snapshot with no failures should return empty taxonomy."""

    def test_empty_taxonomy(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(0, {"x": 5.0}, kpi={"yield": 0.9}),
            _make_obs(1, {"x": 3.0}, kpi={"yield": 0.7}),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        self.assertEqual(len(taxonomy.classified_failures), 0)
        self.assertEqual(sum(taxonomy.type_counts.values()), 0)
        # Dominant defaults to UNKNOWN when there are no failures
        self.assertEqual(taxonomy.dominant_type, FailureType.UNKNOWN)


class TestMixedFailureTypes(unittest.TestCase):
    """Snapshots with multiple failure types should produce mixed taxonomy."""

    def test_mixed_hw_and_chem(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(0, {"x": 5.0}, kpi={"yield": 0.9}),
            _make_obs(
                1, {"x": 2.0},
                is_failure=True, failure_reason="instrument timeout",
            ),
            _make_obs(
                2, {"x": 8.0},
                is_failure=True, failure_reason="precipitate formed",
            ),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        types_seen = {f.failure_type for f in taxonomy.classified_failures}
        self.assertIn(FailureType.HARDWARE, types_seen)
        self.assertIn(FailureType.CHEMISTRY, types_seen)
        self.assertEqual(len(taxonomy.classified_failures), 2)
        # Both types present -> mixed strategy should appear
        self.assertIn("mixed", taxonomy.strategy_adjustments)
        self.assertEqual(taxonomy.strategy_adjustments["mixed"], "conservative_exploration")

    def test_mixed_three_types(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(
                0, {"x": 1.0},
                is_failure=True, failure_reason="sensor malfunction",
            ),
            _make_obs(
                1, {"x": 5.0},
                is_failure=True, failure_reason="yield too low, reaction incomplete",
            ),
            _make_obs(
                2, {"x": 9.0},
                kpi={"yield": 0.01},
                qc_passed=False,
            ),  # DATA type (not is_failure)
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        types_seen = {f.failure_type for f in taxonomy.classified_failures}
        self.assertGreaterEqual(len(types_seen), 2)
        self.assertEqual(len(taxonomy.classified_failures), 3)

    def test_type_rates_sum_to_one(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(
                0, {"x": 1.0},
                is_failure=True, failure_reason="instrument timeout",
            ),
            _make_obs(
                1, {"x": 5.0},
                is_failure=True, failure_reason="precipitate formed",
            ),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        total_rate = sum(taxonomy.type_rates.values())
        self.assertAlmostEqual(total_rate, 1.0, places=5)


class TestCustomKeywords(unittest.TestCase):
    """Custom keywords should override defaults."""

    def test_custom_hardware_keyword(self) -> None:
        classifier = FailureClassifier(hardware_keywords=["overheated"])
        specs = [_make_spec("x")]
        obs = [
            _make_obs(
                0, {"x": 5.0},
                is_failure=True, failure_reason="reactor overheated",
            ),
        ]
        taxonomy = classifier.classify(_make_snapshot(specs, obs))
        self.assertEqual(taxonomy.classified_failures[0].failure_type, FailureType.HARDWARE)

    def test_custom_chemistry_keyword(self) -> None:
        classifier = FailureClassifier(chemistry_keywords=["emulsion"])
        specs = [_make_spec("x")]
        obs = [
            _make_obs(
                0, {"x": 5.0},
                is_failure=True, failure_reason="emulsion broke",
            ),
        ]
        taxonomy = classifier.classify(_make_snapshot(specs, obs))
        self.assertEqual(taxonomy.classified_failures[0].failure_type, FailureType.CHEMISTRY)

    def test_default_keyword_not_matched_with_custom(self) -> None:
        """When custom keywords are provided, defaults should not apply."""
        classifier = FailureClassifier(hardware_keywords=["overheated"])
        specs = [_make_spec("x")]
        obs = [
            _make_obs(
                0, {"x": 5.0},
                is_failure=True,
                failure_reason="instrument timeout",  # default keyword, not in custom list
            ),
        ]
        taxonomy = classifier.classify(_make_snapshot(specs, obs))
        # "instrument" and "timeout" are NOT in the custom hardware list,
        # so it should not be classified as HARDWARE via keyword match
        cf = taxonomy.classified_failures[0]
        self.assertNotEqual(cf.failure_type, FailureType.HARDWARE)


class TestClassifiedFailureDataclass(unittest.TestCase):
    """Test ClassifiedFailure data integrity."""

    def test_confidence_in_range(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(
                0, {"x": 5.0},
                is_failure=True, failure_reason="instrument timeout",
            ),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        for cf in taxonomy.classified_failures:
            self.assertGreaterEqual(cf.confidence, 0.0)
            self.assertLessEqual(cf.confidence, 1.0)

    def test_evidence_is_nonempty(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(
                0, {"x": 5.0},
                is_failure=True, failure_reason="instrument timeout",
            ),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        for cf in taxonomy.classified_failures:
            self.assertGreater(len(cf.evidence), 0)


class TestTaxonomyStructure(unittest.TestCase):
    """Verify the FailureTaxonomy dataclass fields."""

    def test_all_failure_types_in_counts(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(
                0, {"x": 5.0},
                is_failure=True, failure_reason="timeout",
            ),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        for ft in FailureType:
            self.assertIn(ft.value, taxonomy.type_counts)
            self.assertIn(ft.value, taxonomy.type_rates)

    def test_dominant_type_has_highest_count(self) -> None:
        specs = [_make_spec("x")]
        obs = [
            _make_obs(0, {"x": 1.0}, is_failure=True, failure_reason="timeout"),
            _make_obs(1, {"x": 2.0}, is_failure=True, failure_reason="connection lost"),
            _make_obs(2, {"x": 3.0}, is_failure=True, failure_reason="precipitate"),
        ]
        taxonomy = FailureClassifier().classify(_make_snapshot(specs, obs))
        dom_count = taxonomy.type_counts[taxonomy.dominant_type.value]
        for count in taxonomy.type_counts.values():
            self.assertGreaterEqual(dom_count, count)


if __name__ == "__main__":
    unittest.main()
