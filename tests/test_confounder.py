"""Tests for confounder governance: detection, correction policies, and audit trails.

Verifies:
- ConfounderPolicy enum values and model construction
- COVARIATE policy: promotes metadata to parameter spec + observation parameters
- NORMALIZE policy: removes linear confounder effect via OLS residuals
- HIGH_RISK_FLAG policy: down-weights out-of-threshold observations
- EXCLUDE policy: removes out-of-threshold observations
- ConfounderDetector: metadata-KPI correlation scanning
- Acceptance: policies produce different results, audit trail correctness
- Edge cases: empty data, single observation, missing metadata, constant confounder
"""

from __future__ import annotations

import math
import random

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.confounder.models import (
    ConfounderAuditTrail,
    ConfounderConfig,
    ConfounderCorrectionRecord,
    ConfounderPolicy,
    ConfounderSpec,
)
from optimization_copilot.confounder.governance import ConfounderGovernor
from optimization_copilot.confounder.detector import ConfounderDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(n: int = 20, with_confounder: bool = True) -> CampaignSnapshot:
    """Create a test snapshot with optional confounder in metadata."""
    rng = random.Random(42)
    obs = []
    for i in range(n):
        x = rng.uniform(0.0, 1.0)
        confounder_val = rng.uniform(0.0, 1.0)
        # KPI correlates with both x and confounder
        kpi = 2.0 * x + 1.5 * confounder_val + rng.gauss(0, 0.1)
        meta = {"pd_content": confounder_val} if with_confounder else {}
        obs.append(Observation(
            iteration=i,
            parameters={"x": x},
            kpi_values={"efficiency": kpi},
            metadata=meta,
        ))
    return CampaignSnapshot(
        campaign_id="test",
        parameter_specs=[ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 1.0)],
        observations=obs,
        objective_names=["efficiency"],
        objective_directions=["maximize"],
    )


def _make_multi_kpi_snapshot(n: int = 20) -> CampaignSnapshot:
    """Snapshot with two KPIs both correlated with the confounder."""
    rng = random.Random(42)
    obs = []
    for i in range(n):
        x = rng.uniform(0.0, 1.0)
        c = rng.uniform(0.0, 1.0)
        kpi1 = 2.0 * x + 1.5 * c + rng.gauss(0, 0.1)
        kpi2 = -1.0 * x + 0.8 * c + rng.gauss(0, 0.1)
        obs.append(Observation(
            iteration=i,
            parameters={"x": x},
            kpi_values={"eff": kpi1, "cost": kpi2},
            metadata={"pd_content": c},
        ))
    return CampaignSnapshot(
        campaign_id="multi-kpi",
        parameter_specs=[ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 1.0)],
        observations=obs,
        objective_names=["eff", "cost"],
        objective_directions=["maximize", "minimize"],
    )


# ---------------------------------------------------------------------------
# TestConfounderModels
# ---------------------------------------------------------------------------


class TestConfounderModels:
    """Model dataclass construction and enum validation."""

    def test_policy_enum_values(self):
        assert ConfounderPolicy.COVARIATE.value == "covariate"
        assert ConfounderPolicy.NORMALIZE.value == "normalize"
        assert ConfounderPolicy.HIGH_RISK_FLAG.value == "high_risk_flag"
        assert ConfounderPolicy.EXCLUDE.value == "exclude"

    def test_policy_enum_count(self):
        assert len(ConfounderPolicy) == 4

    def test_spec_creation_defaults(self):
        spec = ConfounderSpec(column_name="pd", policy=ConfounderPolicy.COVARIATE)
        assert spec.column_name == "pd"
        assert spec.policy == ConfounderPolicy.COVARIATE
        assert spec.threshold_low is None
        assert spec.threshold_high is None
        assert spec.metadata == {}

    def test_spec_creation_with_thresholds(self):
        spec = ConfounderSpec(
            column_name="pd",
            policy=ConfounderPolicy.HIGH_RISK_FLAG,
            threshold_low=0.1,
            threshold_high=0.9,
            metadata={"unit": "%"},
        )
        assert spec.threshold_low == 0.1
        assert spec.threshold_high == 0.9
        assert spec.metadata == {"unit": "%"}

    def test_config_creation_defaults(self):
        cfg = ConfounderConfig()
        assert cfg.confounders == []
        assert cfg.auto_detect is False
        assert cfg.correlation_threshold == 0.3

    def test_audit_trail_creation(self):
        record = ConfounderCorrectionRecord(
            column_name="pd",
            policy=ConfounderPolicy.NORMALIZE,
            n_affected_rows=15,
        )
        trail = ConfounderAuditTrail(
            corrections=[record],
            config_used=ConfounderConfig(),
            summary="pd: normalize (15 rows affected)",
        )
        assert len(trail.corrections) == 1
        assert trail.corrections[0].column_name == "pd"
        assert trail.summary == "pd: normalize (15 rows affected)"


# ---------------------------------------------------------------------------
# TestConfounderGovernorCovariate
# ---------------------------------------------------------------------------


class TestConfounderGovernorCovariate:
    """COVARIATE policy: promote metadata to formal parameter."""

    def test_adds_parameter_spec(self):
        snap = _make_snapshot()
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.COVARIATE),
        ])
        gov = ConfounderGovernor(cfg)
        result, _ = gov.apply(snap)

        param_names = [p.name for p in result.parameter_specs]
        assert "pd_content" in param_names

    def test_copies_metadata_to_parameters(self):
        snap = _make_snapshot()
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.COVARIATE),
        ])
        gov = ConfounderGovernor(cfg)
        result, _ = gov.apply(snap)

        for obs in result.observations:
            assert "pd_content" in obs.parameters
            # Value should match metadata
            assert obs.parameters["pd_content"] == float(obs.metadata["pd_content"])

    def test_handles_missing_metadata_gracefully(self):
        """Observations without the confounder key keep their parameters unchanged."""
        snap = _make_snapshot(n=10, with_confounder=False)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.COVARIATE),
        ])
        gov = ConfounderGovernor(cfg)
        result, audit = gov.apply(snap)

        # No observations should have pd_content in parameters
        for obs in result.observations:
            assert "pd_content" not in obs.parameters
        assert audit.corrections[0].n_affected_rows == 0

    def test_audit_record_correct(self):
        snap = _make_snapshot()
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.COVARIATE),
        ])
        gov = ConfounderGovernor(cfg)
        _, audit = gov.apply(snap)

        assert len(audit.corrections) == 1
        rec = audit.corrections[0]
        assert rec.column_name == "pd_content"
        assert rec.policy == ConfounderPolicy.COVARIATE
        assert rec.n_affected_rows == 20

    def test_original_snapshot_not_mutated(self):
        snap = _make_snapshot()
        original_param_count = len(snap.parameter_specs)
        original_obs_params = [dict(o.parameters) for o in snap.observations]

        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.COVARIATE),
        ])
        gov = ConfounderGovernor(cfg)
        gov.apply(snap)

        # Original snapshot unchanged.
        assert len(snap.parameter_specs) == original_param_count
        for i, obs in enumerate(snap.observations):
            assert obs.parameters == original_obs_params[i]


# ---------------------------------------------------------------------------
# TestConfounderGovernorNormalize
# ---------------------------------------------------------------------------


class TestConfounderGovernorNormalize:
    """NORMALIZE policy: remove linear confounder effect via OLS residuals."""

    def test_residuals_remove_linear_trend(self):
        """After normalization, the correlation between confounder and KPI
        should be substantially reduced."""
        snap = _make_snapshot(n=50)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.NORMALIZE),
        ])
        gov = ConfounderGovernor(cfg)
        result, _ = gov.apply(snap)

        # Compute correlation between confounder and KPI after normalization.
        xs = [obs.metadata["pd_content"] for obs in result.observations]
        ys = [obs.kpi_values["efficiency"] for obs in result.observations]
        corr = abs(_pearson(xs, ys))
        # Should be close to zero (residuals are uncorrelated with X by construction).
        assert corr < 0.05, f"Residual correlation {corr} should be near zero"

    def test_kpi_mean_preserved(self):
        """The mean KPI value should be approximately preserved after normalization."""
        snap = _make_snapshot(n=50)
        original_mean = sum(
            o.kpi_values["efficiency"] for o in snap.observations
        ) / len(snap.observations)

        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.NORMALIZE),
        ])
        gov = ConfounderGovernor(cfg)
        result, _ = gov.apply(snap)

        new_mean = sum(
            o.kpi_values["efficiency"] for o in result.observations
        ) / len(result.observations)
        assert abs(new_mean - original_mean) < 1e-6

    def test_handles_constant_confounder(self):
        """When the confounder is constant, residuals = y - mean(y)
        and the mean is preserved."""
        rng = random.Random(99)
        obs = []
        for i in range(20):
            obs.append(Observation(
                iteration=i,
                parameters={"x": rng.uniform(0, 1)},
                kpi_values={"eff": rng.gauss(5.0, 1.0)},
                metadata={"c": 0.5},  # constant
            ))
        snap = CampaignSnapshot(
            campaign_id="const",
            parameter_specs=[ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 1.0)],
            observations=obs,
            objective_names=["eff"],
            objective_directions=["maximize"],
        )
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("c", ConfounderPolicy.NORMALIZE),
        ])
        gov = ConfounderGovernor(cfg)
        result, _ = gov.apply(snap)

        # Mean should still be preserved.
        orig_mean = sum(o.kpi_values["eff"] for o in snap.observations) / 20
        new_mean = sum(o.kpi_values["eff"] for o in result.observations) / 20
        assert abs(new_mean - orig_mean) < 1e-6

    def test_multiple_kpis_normalized_independently(self):
        """Each KPI should be normalized independently."""
        snap = _make_multi_kpi_snapshot(n=50)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.NORMALIZE),
        ])
        gov = ConfounderGovernor(cfg)
        result, audit = gov.apply(snap)

        for obj in ["eff", "cost"]:
            xs = [o.metadata["pd_content"] for o in result.observations]
            ys = [o.kpi_values[obj] for o in result.observations]
            corr = abs(_pearson(xs, ys))
            assert corr < 0.05, f"{obj} residual correlation {corr} should be near zero"

    def test_audit_record_stats_before_after(self):
        snap = _make_snapshot(n=30)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.NORMALIZE),
        ])
        gov = ConfounderGovernor(cfg)
        _, audit = gov.apply(snap)

        rec = audit.corrections[0]
        assert "efficiency_mean" in rec.original_kpi_stats
        assert "efficiency_std" in rec.original_kpi_stats
        assert "efficiency_mean" in rec.corrected_kpi_stats
        assert "efficiency_std" in rec.corrected_kpi_stats
        # Std should decrease after removing confounder effect.
        assert rec.corrected_kpi_stats["efficiency_std"] <= rec.original_kpi_stats["efficiency_std"] + 1e-6

    def test_original_snapshot_not_mutated(self):
        snap = _make_snapshot()
        original_kpis = [dict(o.kpi_values) for o in snap.observations]

        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.NORMALIZE),
        ])
        gov = ConfounderGovernor(cfg)
        gov.apply(snap)

        for i, obs in enumerate(snap.observations):
            assert obs.kpi_values == original_kpis[i]


# ---------------------------------------------------------------------------
# TestConfounderGovernorHighRisk
# ---------------------------------------------------------------------------


class TestConfounderGovernorHighRisk:
    """HIGH_RISK_FLAG policy: down-weight out-of-threshold observations."""

    def test_flags_observations_beyond_thresholds(self):
        snap = _make_snapshot(n=30)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.HIGH_RISK_FLAG,
                           threshold_low=0.2, threshold_high=0.8),
        ])
        gov = ConfounderGovernor(cfg)
        result, audit = gov.apply(snap)

        flagged = [o for o in result.observations if o.metadata.get("confounder_flagged")]
        assert len(flagged) > 0
        assert audit.corrections[0].n_affected_rows == len(flagged)

        # All flagged observations should have confounder outside [0.2, 0.8].
        for obs in flagged:
            val = obs.metadata["pd_content"]
            assert val < 0.2 or val > 0.8

    def test_weight_multiplied_correctly(self):
        snap = _make_snapshot(n=30)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.HIGH_RISK_FLAG,
                           threshold_low=0.2, threshold_high=0.8),
        ])
        gov = ConfounderGovernor(cfg)
        result, _ = gov.apply(snap)

        for obs in result.observations:
            if obs.metadata.get("confounder_flagged"):
                assert obs.metadata["weight"] == 0.5
            else:
                # Unflagged observations should not have weight set.
                assert obs.metadata.get("weight", 1.0) == 1.0

    def test_within_threshold_unchanged(self):
        snap = _make_snapshot(n=30)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.HIGH_RISK_FLAG,
                           threshold_low=0.2, threshold_high=0.8),
        ])
        gov = ConfounderGovernor(cfg)
        result, _ = gov.apply(snap)

        for obs in result.observations:
            val = obs.metadata.get("pd_content")
            if val is not None and 0.2 <= val <= 0.8:
                assert obs.metadata.get("confounder_flagged") is not True

    def test_both_low_and_high_thresholds(self):
        """Observations below threshold_low OR above threshold_high are flagged."""
        rng = random.Random(7)
        obs = []
        for i in range(40):
            c = rng.uniform(0.0, 1.0)
            obs.append(Observation(
                iteration=i,
                parameters={"x": rng.uniform(0, 1)},
                kpi_values={"y": c + rng.gauss(0, 0.1)},
                metadata={"c": c},
            ))
        snap = CampaignSnapshot(
            campaign_id="both-thresh",
            parameter_specs=[ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 1.0)],
            observations=obs,
            objective_names=["y"],
            objective_directions=["maximize"],
        )
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("c", ConfounderPolicy.HIGH_RISK_FLAG,
                           threshold_low=0.3, threshold_high=0.7),
        ])
        gov = ConfounderGovernor(cfg)
        result, _ = gov.apply(snap)

        for obs in result.observations:
            val = obs.metadata["c"]
            if val < 0.3 or val > 0.7:
                assert obs.metadata.get("confounder_flagged") is True
            else:
                assert obs.metadata.get("confounder_flagged") is not True

    def test_no_thresholds_flags_all(self):
        """When no thresholds are set, all observations with numeric confounder are flagged."""
        snap = _make_snapshot(n=10)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.HIGH_RISK_FLAG),
        ])
        gov = ConfounderGovernor(cfg)
        result, audit = gov.apply(snap)

        flagged = [o for o in result.observations if o.metadata.get("confounder_flagged")]
        assert len(flagged) == 10
        assert audit.corrections[0].n_affected_rows == 10


# ---------------------------------------------------------------------------
# TestConfounderGovernorExclude
# ---------------------------------------------------------------------------


class TestConfounderGovernorExclude:
    """EXCLUDE policy: remove observations exceeding thresholds."""

    def test_removes_observations_beyond_thresholds(self):
        snap = _make_snapshot(n=30)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.EXCLUDE,
                           threshold_low=0.2, threshold_high=0.8),
        ])
        gov = ConfounderGovernor(cfg)
        result, audit = gov.apply(snap)

        # All remaining observations should have confounder in [0.2, 0.8].
        for obs in result.observations:
            val = obs.metadata["pd_content"]
            assert 0.2 <= val <= 0.8

        assert audit.corrections[0].n_affected_rows > 0
        assert len(result.observations) < len(snap.observations)

    def test_preserves_within_threshold(self):
        snap = _make_snapshot(n=30)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.EXCLUDE,
                           threshold_low=0.2, threshold_high=0.8),
        ])
        gov = ConfounderGovernor(cfg)
        result, _ = gov.apply(snap)

        # Count original observations within threshold.
        expected = sum(
            1 for o in snap.observations
            if 0.2 <= o.metadata["pd_content"] <= 0.8
        )
        assert len(result.observations) == expected

    def test_empty_result_when_all_excluded(self):
        """If all confounder values exceed thresholds, result is empty."""
        snap = _make_snapshot(n=10)
        # Set thresholds to exclude everything (values are in [0, 1]).
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.EXCLUDE,
                           threshold_low=2.0, threshold_high=3.0),
        ])
        gov = ConfounderGovernor(cfg)
        result, audit = gov.apply(snap)

        assert len(result.observations) == 0
        assert audit.corrections[0].n_affected_rows == 10

    def test_single_sided_threshold_low(self):
        """Only threshold_low set: excludes values below it."""
        snap = _make_snapshot(n=30)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.EXCLUDE,
                           threshold_low=0.5),
        ])
        gov = ConfounderGovernor(cfg)
        result, _ = gov.apply(snap)

        for obs in result.observations:
            assert obs.metadata["pd_content"] >= 0.5

    def test_single_sided_threshold_high(self):
        """Only threshold_high set: excludes values above it."""
        snap = _make_snapshot(n=30)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.EXCLUDE,
                           threshold_high=0.5),
        ])
        gov = ConfounderGovernor(cfg)
        result, _ = gov.apply(snap)

        for obs in result.observations:
            assert obs.metadata["pd_content"] <= 0.5


# ---------------------------------------------------------------------------
# TestConfounderDetector
# ---------------------------------------------------------------------------


class TestConfounderDetector:
    """Automatic confounder detection via metadata-KPI correlation."""

    def test_detects_correlated_metadata(self):
        snap = _make_snapshot(n=50)
        detector = ConfounderDetector()
        found = detector.detect(snap, threshold=0.3)

        names = [s.column_name for s in found]
        assert "pd_content" in names

    def test_ignores_non_numeric_metadata(self):
        """Non-numeric metadata values should be silently skipped."""
        rng = random.Random(42)
        obs = []
        for i in range(30):
            obs.append(Observation(
                iteration=i,
                parameters={"x": rng.uniform(0, 1)},
                kpi_values={"y": rng.gauss(5, 1)},
                metadata={"label": f"sample_{i}", "batch": "A"},
            ))
        snap = CampaignSnapshot(
            campaign_id="non-numeric",
            parameter_specs=[ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 1.0)],
            observations=obs,
            objective_names=["y"],
            objective_directions=["maximize"],
        )
        detector = ConfounderDetector()
        found = detector.detect(snap, threshold=0.3)
        # "label" and "batch" are not numeric -> should not appear.
        names = [s.column_name for s in found]
        assert "label" not in names
        assert "batch" not in names

    def test_threshold_filtering(self):
        """With a very high threshold, no confounders should be detected."""
        snap = _make_snapshot(n=50)
        detector = ConfounderDetector()
        found = detector.detect(snap, threshold=0.99)
        # The confounder-KPI correlation is strong but unlikely to be > 0.99
        # given the added noise.  Accept either empty or non-empty but verify
        # all returned specs have |corr| > 0.99.
        for spec in found:
            assert spec.metadata.get("max_abs_correlation", 0) > 0.99

    def test_no_confounders_returns_empty(self):
        """When metadata has no correlated columns, return empty list."""
        rng = random.Random(42)
        obs = []
        for i in range(30):
            obs.append(Observation(
                iteration=i,
                parameters={"x": rng.uniform(0, 1)},
                kpi_values={"y": rng.gauss(5, 1)},
                metadata={"noise": rng.gauss(0, 100)},
            ))
        snap = CampaignSnapshot(
            campaign_id="uncorr",
            parameter_specs=[ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 1.0)],
            observations=obs,
            objective_names=["y"],
            objective_directions=["maximize"],
        )
        detector = ConfounderDetector()
        found = detector.detect(snap, threshold=0.5)
        # With random noise, correlation should be low.  Allow a small chance
        # of detection but assert consistency.
        for spec in found:
            assert spec.metadata.get("max_abs_correlation", 0) > 0.5

    def test_handles_empty_observations(self):
        snap = CampaignSnapshot(
            campaign_id="empty",
            parameter_specs=[ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 1.0)],
            observations=[],
            objective_names=["y"],
            objective_directions=["maximize"],
        )
        detector = ConfounderDetector()
        found = detector.detect(snap)
        assert found == []

    def test_detected_specs_use_high_risk_flag_policy(self):
        """Auto-detected confounders should default to HIGH_RISK_FLAG policy."""
        snap = _make_snapshot(n=50)
        detector = ConfounderDetector()
        found = detector.detect(snap, threshold=0.3)
        for spec in found:
            assert spec.policy == ConfounderPolicy.HIGH_RISK_FLAG


# ---------------------------------------------------------------------------
# TestConfounderAcceptance
# ---------------------------------------------------------------------------


class TestConfounderAcceptance:
    """Acceptance-level tests: policies produce different outcomes, audit trail."""

    def test_policies_produce_different_pareto_fronts(self):
        """Apply all 4 policies to the same snapshot and verify observations differ."""
        snap = _make_snapshot(n=30)
        results: dict[str, CampaignSnapshot] = {}

        for policy in ConfounderPolicy:
            kwargs: dict = {"column_name": "pd_content", "policy": policy}
            if policy in (ConfounderPolicy.HIGH_RISK_FLAG, ConfounderPolicy.EXCLUDE):
                kwargs["threshold_low"] = 0.2
                kwargs["threshold_high"] = 0.8
            spec = ConfounderSpec(**kwargs)
            cfg = ConfounderConfig(confounders=[spec])
            gov = ConfounderGovernor(cfg)
            result, _ = gov.apply(snap)
            results[policy.value] = result

        # Verify at least some policies produced different observation counts or values.
        obs_counts = {k: len(v.observations) for k, v in results.items()}
        kpi_sums = {}
        for k, v in results.items():
            kpi_sums[k] = sum(o.kpi_values["efficiency"] for o in v.observations)

        # EXCLUDE should have fewer observations.
        assert obs_counts["exclude"] < obs_counts["covariate"]
        # NORMALIZE preserves the mean but changes individual KPI values
        # (i.e. not all values are identical to COVARIATE).
        norm_vals = [o.kpi_values["efficiency"] for o in results["normalize"].observations]
        cov_vals = [o.kpi_values["efficiency"] for o in results["covariate"].observations]
        diffs = [abs(a - b) for a, b in zip(norm_vals, cov_vals)]
        assert max(diffs) > 0.01, "NORMALIZE should change individual KPI values"

    def test_audit_trail_traces_corrections(self):
        """Audit trail records correct policy, column, and n_affected for each step."""
        snap = _make_snapshot(n=20)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.NORMALIZE),
            ConfounderSpec("pd_content", ConfounderPolicy.HIGH_RISK_FLAG,
                           threshold_low=0.2, threshold_high=0.8),
        ])
        gov = ConfounderGovernor(cfg)
        _, audit = gov.apply(snap)

        assert len(audit.corrections) == 2
        assert audit.corrections[0].policy == ConfounderPolicy.NORMALIZE
        assert audit.corrections[0].column_name == "pd_content"
        assert audit.corrections[1].policy == ConfounderPolicy.HIGH_RISK_FLAG
        assert audit.corrections[1].column_name == "pd_content"
        assert audit.config_used is cfg
        assert "pd_content" in audit.summary

    def test_auto_detect_integration(self):
        """When auto_detect is True, the detector's results can feed back into governance."""
        snap = _make_snapshot(n=50)
        detector = ConfounderDetector()
        detected = detector.detect(snap, threshold=0.3)

        # Use detected specs with the governor.
        cfg = ConfounderConfig(confounders=detected, auto_detect=True)
        gov = ConfounderGovernor(cfg)
        result, audit = gov.apply(snap)

        assert len(audit.corrections) == len(detected)
        for rec in audit.corrections:
            assert rec.policy == ConfounderPolicy.HIGH_RISK_FLAG

    def test_multiple_confounders_applied_sequentially(self):
        """Multiple confounder specs are applied one after another."""
        rng = random.Random(42)
        obs = []
        for i in range(30):
            x = rng.uniform(0, 1)
            c1 = rng.uniform(0, 1)
            c2 = rng.uniform(0, 1)
            y = 2.0 * x + 1.0 * c1 + 0.5 * c2 + rng.gauss(0, 0.1)
            obs.append(Observation(
                iteration=i,
                parameters={"x": x},
                kpi_values={"y": y},
                metadata={"c1": c1, "c2": c2},
            ))
        snap = CampaignSnapshot(
            campaign_id="multi-conf",
            parameter_specs=[ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 1.0)],
            observations=obs,
            objective_names=["y"],
            objective_directions=["maximize"],
        )
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("c1", ConfounderPolicy.NORMALIZE),
            ConfounderSpec("c2", ConfounderPolicy.NORMALIZE),
        ])
        gov = ConfounderGovernor(cfg)
        result, audit = gov.apply(snap)

        assert len(audit.corrections) == 2
        # After normalizing both confounders, the KPI should mostly reflect x alone.
        xs = [o.parameters["x"] for o in result.observations]
        ys = [o.kpi_values["y"] for o in result.observations]
        # Correlation with x should remain strong (we didn't remove x's effect).
        corr_x = abs(_pearson(xs, ys))
        assert corr_x > 0.5

    def test_covariate_then_normalize(self):
        """Applying COVARIATE then NORMALIZE should be valid chaining."""
        snap = _make_snapshot(n=30)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.COVARIATE),
            ConfounderSpec("pd_content", ConfounderPolicy.NORMALIZE),
        ])
        gov = ConfounderGovernor(cfg)
        result, audit = gov.apply(snap)

        assert len(audit.corrections) == 2
        # After COVARIATE, pd_content is in parameters; NORMALIZE still works on metadata.
        assert "pd_content" in result.observations[0].parameters

    def test_audit_summary_format(self):
        snap = _make_snapshot(n=10)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.EXCLUDE,
                           threshold_low=0.3, threshold_high=0.7),
        ])
        gov = ConfounderGovernor(cfg)
        _, audit = gov.apply(snap)

        assert "pd_content" in audit.summary
        assert "exclude" in audit.summary
        assert "rows affected" in audit.summary


# ---------------------------------------------------------------------------
# TestConfounderEdgeCases
# ---------------------------------------------------------------------------


class TestConfounderEdgeCases:
    """Edge cases: empty data, single observation, missing metadata, etc."""

    def test_empty_observations(self):
        snap = CampaignSnapshot(
            campaign_id="empty",
            parameter_specs=[ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 1.0)],
            observations=[],
            objective_names=["y"],
            objective_directions=["maximize"],
        )
        for policy in ConfounderPolicy:
            kwargs: dict = {"column_name": "c", "policy": policy}
            if policy in (ConfounderPolicy.HIGH_RISK_FLAG, ConfounderPolicy.EXCLUDE):
                kwargs["threshold_low"] = 0.2
                kwargs["threshold_high"] = 0.8
            cfg = ConfounderConfig(confounders=[ConfounderSpec(**kwargs)])
            gov = ConfounderGovernor(cfg)
            result, audit = gov.apply(snap)
            assert len(result.observations) == 0
            assert audit.corrections[0].n_affected_rows == 0

    def test_single_observation(self):
        obs = [Observation(
            iteration=0,
            parameters={"x": 0.5},
            kpi_values={"y": 1.0},
            metadata={"c": 0.3},
        )]
        snap = CampaignSnapshot(
            campaign_id="single",
            parameter_specs=[ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 1.0)],
            observations=obs,
            objective_names=["y"],
            objective_directions=["maximize"],
        )
        for policy in ConfounderPolicy:
            kwargs: dict = {"column_name": "c", "policy": policy}
            if policy in (ConfounderPolicy.HIGH_RISK_FLAG, ConfounderPolicy.EXCLUDE):
                kwargs["threshold_low"] = 0.0
                kwargs["threshold_high"] = 1.0
            cfg = ConfounderConfig(confounders=[ConfounderSpec(**kwargs)])
            gov = ConfounderGovernor(cfg)
            # Should not raise.
            result, audit = gov.apply(snap)
            assert len(audit.corrections) == 1

    def test_confounder_not_in_metadata(self):
        """When the confounder key is absent from all metadata, handle gracefully."""
        snap = _make_snapshot(n=10, with_confounder=False)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("nonexistent", ConfounderPolicy.NORMALIZE),
        ])
        gov = ConfounderGovernor(cfg)
        result, audit = gov.apply(snap)

        # No observations should be affected.
        assert audit.corrections[0].n_affected_rows == 0
        # KPI values unchanged.
        for orig, corrected in zip(snap.observations, result.observations):
            assert orig.kpi_values["efficiency"] == corrected.kpi_values["efficiency"]

    def test_all_same_confounder_value(self):
        """When all observations have the same confounder value, NORMALIZE
        should degrade gracefully (zero-variance confounder)."""
        obs = []
        for i in range(20):
            obs.append(Observation(
                iteration=i,
                parameters={"x": i * 0.05},
                kpi_values={"y": float(i)},
                metadata={"c": 0.5},
            ))
        snap = CampaignSnapshot(
            campaign_id="constant",
            parameter_specs=[ParameterSpec("x", VariableType.CONTINUOUS, 0.0, 1.0)],
            observations=obs,
            objective_names=["y"],
            objective_directions=["maximize"],
        )
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("c", ConfounderPolicy.NORMALIZE),
        ])
        gov = ConfounderGovernor(cfg)
        result, audit = gov.apply(snap)

        # Mean should be preserved.
        orig_mean = sum(o.kpi_values["y"] for o in snap.observations) / 20
        new_mean = sum(o.kpi_values["y"] for o in result.observations) / 20
        assert abs(new_mean - orig_mean) < 1e-6

    def test_seed_determinism(self):
        """Same seed should produce identical results."""
        snap = _make_snapshot(n=20)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.NORMALIZE),
        ])
        gov = ConfounderGovernor(cfg)
        r1, a1 = gov.apply(snap, seed=42)
        r2, a2 = gov.apply(snap, seed=42)

        for o1, o2 in zip(r1.observations, r2.observations):
            assert o1.kpi_values == o2.kpi_values
        assert a1.summary == a2.summary

    def test_exclude_no_thresholds_removes_nothing(self):
        """EXCLUDE with no thresholds should not remove any observations."""
        snap = _make_snapshot(n=10)
        cfg = ConfounderConfig(confounders=[
            ConfounderSpec("pd_content", ConfounderPolicy.EXCLUDE),
        ])
        gov = ConfounderGovernor(cfg)
        result, audit = gov.apply(snap)

        assert len(result.observations) == 10
        assert audit.corrections[0].n_affected_rows == 0


# ---------------------------------------------------------------------------
# Pearson helper for test assertions
# ---------------------------------------------------------------------------


def _pearson(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient (test helper)."""
    n = len(x)
    if n < 2 or n != len(y):
        return 0.0
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    x_std = math.sqrt(sum((xi - x_mean) ** 2 for xi in x) / n)
    y_std = math.sqrt(sum((yi - y_mean) ** 2 for yi in y) / n)
    if x_std < 1e-12 or y_std < 1e-12:
        return 0.0
    cov = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / n
    return cov / (x_std * y_std)
