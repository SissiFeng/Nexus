"""Acceptance tests: deterministic replay and backend validity.

Category 1 — Deterministic Replay Tests:
  1.1 Golden replay: fixed snapshot + seed → N runs → identical decisions
  1.2 Mutation invariance: irrelevant perturbations → decision unchanged
  1.3 Version drift: config change → decision change documented in reason_codes

Category 2 — Backend Validity Tests:
  Capability profiles, 20 ProblemFingerprints, capability matching, fallback.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, replace
from typing import Any

import pytest

from optimization_copilot.backends.builtin import (
    LatinHypercubeSampler,
    RandomSampler,
    TPESampler,
)
from optimization_copilot.core.hashing import decision_hash, snapshot_hash
from optimization_copilot.core.models import (
    CampaignSnapshot,
    CostProfile,
    DataScale,
    Dynamics,
    FailureInformativeness,
    FeasibleRegion,
    NoiseRegime,
    ObjectiveForm,
    Observation,
    ParameterSpec,
    Phase,
    ProblemFingerprint,
    RiskPosture,
    StabilizeSpec,
    StrategyDecision,
    VariableType,
)
from optimization_copilot.diagnostics.engine import DiagnosticEngine
from optimization_copilot.dsl.spec import (
    BudgetDef,
    Direction,
    ObjectiveDef,
    OptimizationSpec,
    ParamType,
    ParameterDef,
)
from optimization_copilot.meta_controller.controller import MetaController, SwitchingThresholds
from optimization_copilot.plugins.registry import PluginRegistry
from optimization_copilot.profiler.profiler import ProblemProfiler
from optimization_copilot.replay.engine import ReplayEngine
from optimization_copilot.replay.log import DecisionLog
from optimization_copilot.validation.scenarios import GOLDEN_SCENARIOS, ValidationRunner


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run_pipeline(
    snapshot: CampaignSnapshot,
    seed: int = 42,
    available_backends: list[str] | None = None,
    thresholds: SwitchingThresholds | None = None,
) -> StrategyDecision:
    """Run the full decision pipeline and return the StrategyDecision."""
    diag = DiagnosticEngine().compute(snapshot)
    fp = ProblemProfiler().profile(snapshot)
    ctrl = MetaController(
        thresholds=thresholds,
        available_backends=available_backends,
    )
    return ctrl.decide(snapshot, diag.to_dict(), fp, seed=seed)


def _make_continuous_specs(n: int = 3) -> list[ParameterSpec]:
    return [
        ParameterSpec(name=f"x{i}", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
        for i in range(n)
    ]


def _make_observations(
    n: int,
    n_params: int = 3,
    seed: int = 42,
    failure_fn=None,
    kpi_fn=None,
) -> list[Observation]:
    """Generate deterministic observations."""
    rng = random.Random(seed)
    obs = []
    for i in range(n):
        params = {f"x{j}": rng.random() for j in range(n_params)}
        is_fail = failure_fn(i) if failure_fn else False
        if kpi_fn:
            kpi = kpi_fn(i, params) if not is_fail else 0.0
        else:
            kpi = sum(params.values()) + i * 0.1 if not is_fail else 0.0
        obs.append(Observation(
            iteration=i,
            parameters=params,
            kpi_values={"y": kpi},
            is_failure=is_fail,
            timestamp=float(i),
        ))
    return obs


def _make_snapshot(
    n_obs: int = 20,
    n_params: int = 3,
    seed: int = 42,
    campaign_id: str = "acceptance_test",
    **kwargs,
) -> CampaignSnapshot:
    """Create a standard CampaignSnapshot with deterministic data."""
    return CampaignSnapshot(
        campaign_id=campaign_id,
        parameter_specs=_make_continuous_specs(n_params),
        observations=_make_observations(n_obs, n_params=n_params, seed=seed, **kwargs),
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=n_obs,
    )


def _make_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register(RandomSampler)
    registry.register(LatinHypercubeSampler)
    registry.register(TPESampler)
    return registry


# ---------------------------------------------------------------------------
# 1.1 Golden Replay Determinism
# ---------------------------------------------------------------------------


class TestGoldenReplayDeterminism:
    """Fixed snapshot + fixed seed → N runs → identical decision hash & reason_codes."""

    N_RUNS = 5

    @pytest.mark.parametrize(
        "scenario",
        GOLDEN_SCENARIOS,
        ids=[s.name for s in GOLDEN_SCENARIOS],
    )
    def test_golden_scenario_replay(self, scenario):
        """Each golden scenario produces identical decisions across N runs."""
        runner = ValidationRunner()
        results = [runner.run_scenario(scenario) for _ in range(self.N_RUNS)]

        # All decision hashes must be identical.
        hashes = {r.decision_hash for r in results}
        assert len(hashes) == 1, (
            f"Decision hash diverged across {self.N_RUNS} runs: {hashes}"
        )

        # All reason_codes must be identical.
        reason_sets = {tuple(r.decision.reason_codes) for r in results}
        assert len(reason_sets) == 1, (
            f"Reason codes diverged: {reason_sets}"
        )

        # All phases must be identical.
        phases = {r.decision.phase for r in results}
        assert len(phases) == 1, f"Phase diverged: {phases}"

    def test_custom_snapshot_replay(self):
        """A custom 20-observation snapshot produces identical decisions across N runs."""
        snap = _make_snapshot(n_obs=20, seed=99)
        decisions = [_run_pipeline(snap, seed=42) for _ in range(self.N_RUNS)]

        hashes = {decision_hash(d) for d in decisions}
        assert len(hashes) == 1, f"Decision hash diverged: {hashes}"

        reasons = {tuple(d.reason_codes) for d in decisions}
        assert len(reasons) == 1, f"Reason codes diverged: {reasons}"

    def test_cold_start_replay(self):
        """Cold start (3 observations) replays identically."""
        snap = _make_snapshot(n_obs=3, seed=77)
        decisions = [_run_pipeline(snap, seed=42) for _ in range(self.N_RUNS)]

        hashes = {decision_hash(d) for d in decisions}
        assert len(hashes) == 1

        for d in decisions:
            assert d.phase == Phase.COLD_START

    def test_campaign_replay_determinism(self):
        """Multi-iteration campaign produces identical decision logs across 2 runs."""
        spec = OptimizationSpec(
            campaign_id="acceptance-campaign-replay",
            parameters=[
                ParameterDef(name="x1", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
                ParameterDef(name="x2", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
            ],
            objectives=[ObjectiveDef(name="y", direction=Direction.MAXIMIZE)],
            budget=BudgetDef(max_samples=10),
            seed=42,
        )
        registry = _make_registry()
        evaluator = lambda p: {"y": p["x1"] + p["x2"]}

        log1 = _run_campaign_log(spec, registry, evaluator)
        log2 = _run_campaign_log(spec, registry, evaluator)

        assert log1.n_entries == log2.n_entries
        assert log1.n_entries > 0

        for i in range(log1.n_entries):
            e1, e2 = log1.entries[i], log2.entries[i]
            assert e1.decision_hash == e2.decision_hash, (
                f"Decision hash mismatch at iteration {i}: "
                f"{e1.decision_hash} != {e2.decision_hash}"
            )
            assert e1.reason_codes == e2.reason_codes
            assert e1.phase == e2.phase

    def test_replay_engine_self_consistency(self):
        """ReplayEngine.verify() passes for a self-recorded log."""
        spec = OptimizationSpec(
            campaign_id="acceptance-replay-verify",
            parameters=[
                ParameterDef(name="x1", type=ParamType.CONTINUOUS, lower=0.0, upper=5.0),
                ParameterDef(name="x2", type=ParamType.CONTINUOUS, lower=0.0, upper=5.0),
            ],
            objectives=[ObjectiveDef(name="y", direction=Direction.MAXIMIZE)],
            budget=BudgetDef(max_samples=8),
            seed=42,
        )
        registry = _make_registry()
        evaluator = lambda p: {"y": p["x1"] * p["x2"]}

        log = _run_campaign_log(spec, registry, evaluator)
        assert log.n_entries > 0

        replay_engine = ReplayEngine(registry=registry)
        verification = replay_engine.verify(log)
        assert verification.verified is True, (
            f"Self-consistency failed:\n{verification.summary()}"
        )
        assert verification.n_mismatches == 0


# ---------------------------------------------------------------------------
# 1.2 Mutation Invariance
# ---------------------------------------------------------------------------


class TestMutationInvariance:
    """Irrelevant perturbations to snapshot must NOT change the decision."""

    def _base_snapshot(self) -> CampaignSnapshot:
        """Fixture: a 20-obs snapshot in LEARNING phase."""
        return _make_snapshot(n_obs=20, seed=42, campaign_id="mutation_base")

    def _assert_decisions_identical(
        self,
        base: CampaignSnapshot,
        mutated: CampaignSnapshot,
        seed: int = 42,
    ):
        """Assert both snapshots produce identical decisions."""
        d_base = _run_pipeline(base, seed=seed)
        d_mutated = _run_pipeline(mutated, seed=seed)

        assert decision_hash(d_base) == decision_hash(d_mutated), (
            f"Decision hash changed.\n"
            f"  Base reason_codes: {d_base.reason_codes}\n"
            f"  Mutated reason_codes: {d_mutated.reason_codes}"
        )
        assert d_base.reason_codes == d_mutated.reason_codes
        assert d_base.phase == d_mutated.phase
        assert d_base.backend_name == d_mutated.backend_name
        assert d_base.exploration_strength == d_mutated.exploration_strength
        assert d_base.risk_posture == d_mutated.risk_posture

    def test_extra_metadata_keys_no_effect(self):
        """Adding extra keys to snapshot.metadata doesn't change the decision."""
        base = self._base_snapshot()
        mutated = replace(base, metadata={"extra_key": "extra_value", "debug": True})
        self._assert_decisions_identical(base, mutated)

    def test_large_metadata_payload_no_effect(self):
        """Even large metadata payloads don't change the decision."""
        base = self._base_snapshot()
        big_meta = {f"key_{i}": f"value_{i}" for i in range(100)}
        mutated = replace(base, metadata=big_meta)
        self._assert_decisions_identical(base, mutated)

    def test_campaign_id_change_no_effect(self):
        """Changing campaign_id doesn't change the decision."""
        base = self._base_snapshot()
        mutated = replace(base, campaign_id="completely_different_id_12345")
        self._assert_decisions_identical(base, mutated)

    def test_observation_metadata_no_effect(self):
        """Adding metadata to individual observations doesn't change the decision."""
        base = self._base_snapshot()
        new_obs = []
        for obs in base.observations:
            new_obs.append(replace(obs, metadata={"injected": True, "source": "test"}))
        mutated = replace(base, observations=new_obs)
        self._assert_decisions_identical(base, mutated)

    def test_timestamp_change_no_effect(self):
        """Changing observation timestamps doesn't change the decision.

        Timestamps are only used by ProblemProfiler._classify_cost_profile to
        determine CostProfile, which is not used by MetaController.
        """
        base = self._base_snapshot()
        new_obs = []
        for i, obs in enumerate(base.observations):
            # Scramble timestamps completely
            new_obs.append(replace(obs, timestamp=float(1000 + i * 7.3)))
        mutated = replace(base, observations=new_obs)
        self._assert_decisions_identical(base, mutated)

    def test_current_iteration_change_no_effect(self):
        """Changing current_iteration doesn't change the decision.

        current_iteration is metadata — the pipeline uses observations directly.
        """
        base = self._base_snapshot()
        mutated = replace(base, current_iteration=999)
        self._assert_decisions_identical(base, mutated)

    def test_observation_metadata_order_no_effect(self):
        """Reordering observation metadata dict keys doesn't change the decision."""
        base = self._base_snapshot()
        new_obs = []
        for obs in base.observations:
            # Create metadata with reversed key insertion order.
            meta = {"z_last": 1, "a_first": 2, "m_middle": 3}
            new_obs.append(replace(obs, metadata=meta))
        mutated = replace(base, observations=new_obs)

        # Also create version with opposite insertion order.
        new_obs2 = []
        for obs in base.observations:
            meta = {"a_first": 2, "m_middle": 3, "z_last": 1}
            new_obs2.append(replace(obs, metadata=meta))
        mutated2 = replace(base, observations=new_obs2)

        self._assert_decisions_identical(mutated, mutated2)

    def test_multiple_mutations_combined_no_effect(self):
        """Multiple irrelevant mutations combined still don't change the decision."""
        base = self._base_snapshot()
        new_obs = []
        for i, obs in enumerate(base.observations):
            new_obs.append(replace(
                obs,
                metadata={"batch": i, "source": "combined_test"},
                timestamp=float(5000 + i * 13),
            ))
        mutated = replace(
            base,
            campaign_id="combined_mutation_test",
            metadata={"combined": True, "irrelevant": [1, 2, 3]},
            current_iteration=12345,
            observations=new_obs,
        )
        self._assert_decisions_identical(base, mutated)

    def test_snapshot_hash_changes_but_decision_unchanged(self):
        """Verify that metadata mutations DO change snapshot_hash but NOT decision_hash."""
        base = self._base_snapshot()
        mutated = replace(base, metadata={"new_key": "new_value"})

        # Snapshot hashes should differ (metadata is part of the snapshot).
        assert snapshot_hash(base) != snapshot_hash(mutated)

        # But decisions must be identical.
        d_base = _run_pipeline(base)
        d_mutated = _run_pipeline(mutated)
        assert decision_hash(d_base) == decision_hash(d_mutated)


# ---------------------------------------------------------------------------
# 1.3 Version Drift
# ---------------------------------------------------------------------------


class TestVersionDrift:
    """Config changes → decision changes must be documented in reason_codes."""

    def _learning_snapshot(self) -> CampaignSnapshot:
        """A snapshot that lands in LEARNING phase with default thresholds."""
        return _make_snapshot(n_obs=15, seed=42)

    def _stagnation_snapshot(self) -> CampaignSnapshot:
        """A snapshot with flat KPIs that lands in STAGNATION."""
        rng = random.Random(88)
        obs = []
        for i in range(25):
            params = {f"x{j}": rng.random() for j in range(3)}
            obs.append(Observation(
                iteration=i,
                parameters=params,
                kpi_values={"y": 10.0 + rng.gauss(0, 0.01)},  # flat
                timestamp=float(i),
            ))
        return CampaignSnapshot(
            campaign_id="stagnation_drift",
            parameter_specs=_make_continuous_specs(3),
            observations=obs,
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=25,
        )

    def test_backend_removal_documented(self):
        """Removing a backend → fallback is documented in reason_codes."""
        snap = self._learning_snapshot()
        diag = DiagnosticEngine().compute(snap).to_dict()
        fp = ProblemProfiler().profile(snap)

        ctrl_v1 = MetaController(available_backends=["random", "latin_hypercube", "tpe"])
        ctrl_v2 = MetaController(available_backends=["random", "latin_hypercube"])

        d1 = ctrl_v1.decide(snap, diag, fp, seed=42)
        d2 = ctrl_v2.decide(snap, diag, fp, seed=42)

        # v2 must never select removed backend.
        assert d2.backend_name != "tpe"

        # Selection is documented in reason_codes or fallback_events.
        all_audit = d2.reason_codes + d2.fallback_events
        assert any(
            "backend_selected" in entry or "no_preferred_available" in entry
            for entry in all_audit
        ), f"Backend selection not documented: {all_audit}"

    def test_backend_addition_documented(self):
        """Adding a backend (even if not selected) → decision is still documented."""
        snap = self._learning_snapshot()
        diag = DiagnosticEngine().compute(snap).to_dict()
        fp = ProblemProfiler().profile(snap)

        ctrl_v1 = MetaController(available_backends=["random", "latin_hypercube", "tpe"])
        ctrl_v2 = MetaController(
            available_backends=["random", "latin_hypercube", "tpe", "cma_es"]
        )

        d1 = ctrl_v1.decide(snap, diag, fp, seed=42)
        d2 = ctrl_v2.decide(snap, diag, fp, seed=42)

        # Both must document which backend was selected.
        assert any("backend_selected" in rc for rc in d1.reason_codes)
        assert any("backend_selected" in rc for rc in d2.reason_codes)

    def test_threshold_change_phase_documented(self):
        """Tightening cold_start threshold → phase change is documented."""
        snap = self._learning_snapshot()  # 15 observations
        diag = DiagnosticEngine().compute(snap).to_dict()
        fp = ProblemProfiler().profile(snap)

        # Default: cold_start_min_observations=10 → LEARNING phase.
        ctrl_v1 = MetaController()
        d1 = ctrl_v1.decide(snap, diag, fp, seed=42)
        assert d1.phase != Phase.COLD_START  # should be learning or beyond

        # Tightened: cold_start_min_observations=20 → COLD_START phase.
        ctrl_v2 = MetaController(thresholds=SwitchingThresholds(
            cold_start_min_observations=20,
        ))
        d2 = ctrl_v2.decide(snap, diag, fp, seed=42)
        assert d2.phase == Phase.COLD_START

        # Phase change must be documented in reason_codes.
        assert any("cold_start" in rc for rc in d2.reason_codes)

    def test_stagnation_threshold_documented(self):
        """Changing stagnation_plateau_length → earlier stagnation is documented."""
        snap = self._stagnation_snapshot()
        diag = DiagnosticEngine().compute(snap).to_dict()
        fp = ProblemProfiler().profile(snap)

        # With default thresholds.
        d1 = MetaController().decide(snap, diag, fp, seed=42)

        # With very tight stagnation threshold.
        d2 = MetaController(thresholds=SwitchingThresholds(
            stagnation_plateau_length=1,
        )).decide(snap, diag, fp, seed=42)

        # At least one should detect stagnation.
        stagnation_detected = d1.phase == Phase.STAGNATION or d2.phase == Phase.STAGNATION
        assert stagnation_detected

        # If stagnation, reason_codes must document it.
        if d2.phase == Phase.STAGNATION:
            assert any("stagnation" in rc for rc in d2.reason_codes)

    def test_every_decision_has_phase_reason(self):
        """Every decision must have at least one reason_code documenting the phase."""
        for scenario in GOLDEN_SCENARIOS:
            decision = _run_pipeline(scenario.snapshot, seed=scenario.seed)
            phase_documented = any(
                decision.phase.value in rc for rc in decision.reason_codes
            )
            assert phase_documented, (
                f"Scenario {scenario.name}: phase={decision.phase.value} "
                f"not found in reason_codes={decision.reason_codes}"
            )

    def test_every_decision_has_backend_reason(self):
        """Every decision must have a reason_code documenting backend selection."""
        for scenario in GOLDEN_SCENARIOS:
            decision = _run_pipeline(scenario.snapshot, seed=scenario.seed)
            backend_documented = any(
                "backend_selected" in rc or "no_preferred_available" in rc
                for rc in decision.reason_codes + decision.fallback_events
            )
            assert backend_documented, (
                f"Scenario {scenario.name}: backend={decision.backend_name} "
                f"not documented in reason_codes={decision.reason_codes} "
                f"or fallback_events={decision.fallback_events}"
            )

    def test_backend_removal_fallback_deterministic(self):
        """Backend removal fallback is deterministic across N runs."""
        snap = self._learning_snapshot()
        diag = DiagnosticEngine().compute(snap).to_dict()
        fp = ProblemProfiler().profile(snap)

        # Only keep one backend.
        ctrl = MetaController(available_backends=["latin_hypercube"])

        decisions = [ctrl.decide(snap, diag, fp, seed=42) for _ in range(5)]
        backends = {d.backend_name for d in decisions}
        assert len(backends) == 1, f"Fallback is non-deterministic: {backends}"

    def test_reason_codes_explain_v1_v2_differences(self):
        """When v1 and v2 decisions differ, reason_codes must be different too."""
        snap = self._learning_snapshot()
        diag = DiagnosticEngine().compute(snap).to_dict()
        fp = ProblemProfiler().profile(snap)

        # v1: default
        ctrl_v1 = MetaController()
        # v2: force cold start by raising threshold
        ctrl_v2 = MetaController(thresholds=SwitchingThresholds(
            cold_start_min_observations=100,
        ))

        d1 = ctrl_v1.decide(snap, diag, fp, seed=42)
        d2 = ctrl_v2.decide(snap, diag, fp, seed=42)

        if d1.phase != d2.phase:
            # reason_codes must differ to explain the phase change.
            assert d1.reason_codes != d2.reason_codes, (
                "Phase changed but reason_codes are identical"
            )


# ---------------------------------------------------------------------------
# 2. Backend Validity Tests
# ---------------------------------------------------------------------------

# Capability profiles for each built-in backend.
BACKEND_CAPABILITY_PROFILES: dict[str, dict[str, Any]] = {
    "random_sampler": {
        "supports_categorical": True,
        "supports_continuous": True,
        "supports_discrete": True,
        "supports_constraints": False,
        "supports_batch": True,
        "requires_observations": False,
        "max_dimensions": None,
        "noise_tolerance": "high",
    },
    "latin_hypercube_sampler": {
        "supports_categorical": True,
        "supports_continuous": True,
        "supports_discrete": True,
        "supports_constraints": False,
        "supports_batch": True,
        "requires_observations": False,
        "max_dimensions": None,
        "noise_tolerance": "high",
    },
    "tpe_sampler": {
        "supports_categorical": True,
        "supports_continuous": True,
        "supports_discrete": True,
        "supports_constraints": False,
        "supports_batch": True,
        "requires_observations": True,
        "max_dimensions": None,
        "noise_tolerance": "medium",
    },
}

# MetaController uses short names (without _sampler suffix) for backend selection.
# Map from controller name to plugin name.
CONTROLLER_TO_PLUGIN: dict[str, str] = {
    "random": "random_sampler",
    "latin_hypercube": "latin_hypercube_sampler",
    "tpe": "tpe_sampler",
}


def _fingerprint_requirements(fp: ProblemFingerprint, n_observations: int) -> dict[str, Any]:
    """Derive minimum backend requirements from a ProblemFingerprint."""
    reqs: dict[str, Any] = {}
    if n_observations < 4:
        reqs["requires_observations"] = False
    if fp.variable_types == VariableType.CATEGORICAL:
        reqs["supports_categorical"] = True
    elif fp.variable_types == VariableType.MIXED:
        reqs["supports_categorical"] = True
        reqs["supports_continuous"] = True
    elif fp.variable_types == VariableType.CONTINUOUS:
        reqs["supports_continuous"] = True
    elif fp.variable_types == VariableType.DISCRETE:
        reqs["supports_discrete"] = True
    return reqs


def _backend_satisfies(backend_controller_name: str, requirements: dict[str, Any]) -> bool:
    """Check if a backend's capabilities satisfy the given requirements."""
    plugin_name = CONTROLLER_TO_PLUGIN.get(backend_controller_name)
    if plugin_name is None:
        return False  # Unknown backend
    caps = BACKEND_CAPABILITY_PROFILES.get(plugin_name)
    if caps is None:
        return False
    for key, required in requirements.items():
        actual = caps.get(key)
        if actual is None:
            return False
        if isinstance(required, bool):
            if required and not actual:
                return False
            if not required and actual:
                # requires_observations=False means we CANNOT use a backend
                # that requires observations.
                return False
        elif actual != required:
            return False
    return True


# 20 ProblemFingerprint test scenarios.

@dataclass
class FingerprintScenario:
    """A test scenario with a fingerprint and matching snapshot parameters."""
    name: str
    n_obs: int
    n_params: int
    specs_fn: Any = None  # Optional custom spec builder
    constraints: list[dict] | None = None
    objective_names: list[str] | None = None
    objective_directions: list[str] | None = None
    seed: int = 42


def _categorical_specs(n: int) -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name=f"c{i}", type=VariableType.CATEGORICAL,
            categories=["A", "B", "C"],
        )
        for i in range(n)
    ]


def _discrete_specs(n: int) -> list[ParameterSpec]:
    return [
        ParameterSpec(name=f"d{i}", type=VariableType.DISCRETE, lower=0.0, upper=100.0)
        for i in range(n)
    ]


def _mixed_specs(n: int) -> list[ParameterSpec]:
    specs = []
    for i in range(n):
        if i % 3 == 0:
            specs.append(ParameterSpec(
                name=f"m{i}", type=VariableType.CATEGORICAL,
                categories=["X", "Y", "Z"],
            ))
        elif i % 3 == 1:
            specs.append(ParameterSpec(
                name=f"m{i}", type=VariableType.DISCRETE, lower=0.0, upper=50.0,
            ))
        else:
            specs.append(ParameterSpec(
                name=f"m{i}", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0,
            ))
    return specs


FINGERPRINT_SCENARIOS: list[FingerprintScenario] = [
    # 1. Cold start, simplest
    FingerprintScenario("fp_01_cold_continuous", n_obs=0, n_params=3),
    # 2. Learning phase basic
    FingerprintScenario("fp_02_learning_basic", n_obs=15, n_params=3),
    # 3. Convergence (moderate data)
    FingerprintScenario("fp_03_convergence", n_obs=50, n_params=3),
    # 4. Cold start discrete
    FingerprintScenario("fp_04_cold_discrete", n_obs=3, n_params=3,
                        specs_fn=lambda: _discrete_specs(3)),
    # 5. Categorical only
    FingerprintScenario("fp_05_categorical", n_obs=12, n_params=3,
                        specs_fn=lambda: _categorical_specs(3)),
    # 6. Mixed variables
    FingerprintScenario("fp_06_mixed", n_obs=15, n_params=6,
                        specs_fn=lambda: _mixed_specs(6)),
    # 7. Multi-objective
    FingerprintScenario("fp_07_multi_obj", n_obs=20, n_params=3,
                        objective_names=["y1", "y2"],
                        objective_directions=["maximize", "minimize"]),
    # 8. Constrained
    FingerprintScenario("fp_08_constrained", n_obs=15, n_params=3,
                        constraints=[{"target": "y", "lower": 0.5}]),
    # 9. Noisy cold start
    FingerprintScenario("fp_09_noisy_cold", n_obs=5, n_params=3, seed=11),
    # 10. Noisy learning
    FingerprintScenario("fp_10_noisy_learning", n_obs=20, n_params=3, seed=22),
    # 11. Noisy mature
    FingerprintScenario("fp_11_noisy_mature", n_obs=50, n_params=3, seed=33),
    # 12. Hardest cold start (mixed + multi-obj + 0 obs)
    FingerprintScenario("fp_12_hardest_cold", n_obs=0, n_params=6,
                        specs_fn=lambda: _mixed_specs(6),
                        objective_names=["y1", "y2"],
                        objective_directions=["maximize", "minimize"]),
    # 13. Mixed multi-obj medium
    FingerprintScenario("fp_13_mixed_multi", n_obs=15, n_params=6,
                        specs_fn=lambda: _mixed_specs(6),
                        objective_names=["y1", "y2"],
                        objective_directions=["maximize", "minimize"]),
    # 14. Categorical + noise
    FingerprintScenario("fp_14_cat_noisy", n_obs=10, n_params=3,
                        specs_fn=lambda: _categorical_specs(3), seed=44),
    # 15. Just under cold start threshold
    FingerprintScenario("fp_15_near_cold", n_obs=8, n_params=3),
    # 16. Discrete constrained
    FingerprintScenario("fp_16_disc_constrained", n_obs=40, n_params=4,
                        specs_fn=lambda: _discrete_specs(4),
                        constraints=[{"target": "y", "upper": 100.0}]),
    # 17. Large mixed
    FingerprintScenario("fp_17_large_mixed", n_obs=60, n_params=6,
                        specs_fn=lambda: _mixed_specs(6)),
    # 18. Worst case cold start
    FingerprintScenario("fp_18_worst_cold", n_obs=2, n_params=3,
                        objective_names=["y1", "y2"],
                        objective_directions=["maximize", "minimize"]),
    # 19. Categorical + constrained + noise
    FingerprintScenario("fp_19_cat_constr", n_obs=15, n_params=3,
                        specs_fn=lambda: _categorical_specs(3),
                        constraints=[{"target": "y", "lower": 0.0, "upper": 5.0}]),
    # 20. Full complexity
    FingerprintScenario("fp_20_full_complex", n_obs=50, n_params=6,
                        specs_fn=lambda: _mixed_specs(6),
                        objective_names=["y1", "y2"],
                        objective_directions=["maximize", "minimize"]),
]


def _build_snapshot_for_scenario(sc: FingerprintScenario) -> CampaignSnapshot:
    """Build a CampaignSnapshot matching the fingerprint scenario."""
    if sc.specs_fn:
        specs = sc.specs_fn()
    else:
        specs = _make_continuous_specs(sc.n_params)

    n_params = len(specs)
    obj_names = sc.objective_names or ["y"]
    obj_dirs = sc.objective_directions or ["maximize"]

    rng = random.Random(sc.seed)
    obs = []
    for i in range(sc.n_obs):
        params = {}
        for spec in specs:
            if spec.type == VariableType.CATEGORICAL:
                params[spec.name] = rng.choice(spec.categories)
            elif spec.type == VariableType.DISCRETE:
                params[spec.name] = rng.randint(int(spec.lower), int(spec.upper))
            else:
                params[spec.name] = rng.uniform(spec.lower, spec.upper)

        kpi_values = {}
        for name in obj_names:
            kpi_values[name] = rng.uniform(0.0, 10.0) + i * 0.1

        obs.append(Observation(
            iteration=i,
            parameters=params,
            kpi_values=kpi_values,
            timestamp=float(i),
        ))

    return CampaignSnapshot(
        campaign_id=f"acceptance_{sc.name}",
        parameter_specs=specs,
        observations=obs,
        objective_names=obj_names,
        objective_directions=obj_dirs,
        constraints=sc.constraints or [],
        current_iteration=sc.n_obs,
    )


class TestBackendValidity:
    """Verify backend selection never violates capability constraints."""

    @pytest.mark.parametrize(
        "scenario",
        FINGERPRINT_SCENARIOS,
        ids=[s.name for s in FINGERPRINT_SCENARIOS],
    )
    def test_backend_capability_match(self, scenario: FingerprintScenario):
        """Selected backend satisfies the requirements derived from fingerprint."""
        snap = _build_snapshot_for_scenario(scenario)
        decision = _run_pipeline(snap, seed=42)

        fp = ProblemProfiler().profile(snap)
        requirements = _fingerprint_requirements(fp, snap.n_observations)

        backend_name = decision.backend_name
        assert _backend_satisfies(backend_name, requirements), (
            f"Backend '{backend_name}' does not satisfy requirements {requirements} "
            f"for scenario {scenario.name} (fp={fp.to_dict()}, n_obs={snap.n_observations})"
        )

    def test_cold_start_never_selects_obs_dependent(self):
        """With 0 observations, backend must not require observations."""
        for n_obs in [0, 1, 2, 3]:
            snap = _make_snapshot(n_obs=n_obs, seed=42)
            decision = _run_pipeline(snap, seed=42)

            plugin_name = CONTROLLER_TO_PLUGIN.get(decision.backend_name)
            if plugin_name and plugin_name in BACKEND_CAPABILITY_PROFILES:
                caps = BACKEND_CAPABILITY_PROFILES[plugin_name]
                assert not caps["requires_observations"], (
                    f"Backend '{decision.backend_name}' requires observations "
                    f"but was selected with only {n_obs} observations"
                )

    def test_tpe_not_selected_without_observations(self):
        """TPE specifically must not be selected when n_obs < cold_start threshold."""
        for n_obs in [0, 3, 5, 8]:
            snap = _make_snapshot(n_obs=n_obs, seed=42)
            decision = _run_pipeline(snap, seed=42)
            assert decision.backend_name != "tpe", (
                f"TPE selected with only {n_obs} observations"
            )

    def test_fallback_when_preferred_unavailable(self):
        """When preferred backends aren't available, deterministic fallback occurs."""
        snap = _make_snapshot(n_obs=20, seed=42)
        diag = DiagnosticEngine().compute(snap).to_dict()
        fp = ProblemProfiler().profile(snap)

        # Only a made-up backend name — no phase map match.
        ctrl = MetaController(available_backends=["exotic_sampler"])
        decision = ctrl.decide(snap, diag, fp, seed=42)

        # Should fall back to the first available backend.
        assert decision.backend_name == "exotic_sampler"
        assert len(decision.fallback_events) > 0, (
            "Fallback must be documented in fallback_events"
        )

    def test_deterministic_fallback(self):
        """Fallback backend is always available_backends[0], not random."""
        snap = _make_snapshot(n_obs=20, seed=42)
        diag = DiagnosticEngine().compute(snap).to_dict()
        fp = ProblemProfiler().profile(snap)

        ctrl = MetaController(available_backends=["fallback_a", "fallback_b"])
        decisions = [ctrl.decide(snap, diag, fp, seed=42) for _ in range(5)]

        backends = {d.backend_name for d in decisions}
        assert len(backends) == 1, f"Non-deterministic fallback: {backends}"
        assert decisions[0].backend_name == "fallback_a"

    def test_all_backends_have_capability_profiles(self):
        """Every registered backend plugin has a defined capability profile."""
        registry = _make_registry()
        for plugin_name in registry.list_plugins():
            assert plugin_name in BACKEND_CAPABILITY_PROFILES, (
                f"Backend '{plugin_name}' missing from BACKEND_CAPABILITY_PROFILES"
            )

    def test_capability_profile_matches_plugin(self):
        """BACKEND_CAPABILITY_PROFILES is consistent with each plugin's capabilities()."""
        registry = _make_registry()
        for plugin_name in registry.list_plugins():
            plugin = registry.get(plugin_name)
            actual_caps = plugin.capabilities()
            profile = BACKEND_CAPABILITY_PROFILES[plugin_name]

            # Every key in actual capabilities must be in the profile.
            for key, value in actual_caps.items():
                assert key in profile, (
                    f"Plugin '{plugin_name}' has capability '{key}' "
                    f"not reflected in BACKEND_CAPABILITY_PROFILES"
                )
                assert profile[key] == value, (
                    f"Plugin '{plugin_name}' capability '{key}': "
                    f"profile says {profile[key]}, plugin says {value}"
                )

    def test_fingerprint_override_respects_availability(self):
        """When fingerprint triggers override and preferred isn't available, fallback works."""
        # Create a snapshot that triggers high_noise override.
        rng = random.Random(55)
        obs = []
        for i in range(20):
            params = {f"x{j}": rng.random() for j in range(3)}
            # Very noisy KPIs to trigger HIGH noise regime.
            kpi = rng.gauss(10.0, 20.0)
            obs.append(Observation(
                iteration=i,
                parameters=params,
                kpi_values={"y": kpi},
                timestamp=float(i),
            ))

        snap = CampaignSnapshot(
            campaign_id="noise_override_test",
            parameter_specs=_make_continuous_specs(3),
            observations=obs,
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=20,
        )

        diag = DiagnosticEngine().compute(snap).to_dict()
        fp = ProblemProfiler().profile(snap)

        # Remove "random" from available — high_noise override prefers it.
        ctrl = MetaController(available_backends=["tpe", "latin_hypercube"])
        decision = ctrl.decide(snap, diag, fp, seed=42)

        # Must select one of the available backends.
        assert decision.backend_name in ["tpe", "latin_hypercube"]

        # Must be documented.
        all_codes = decision.reason_codes + decision.fallback_events
        assert len(all_codes) > 0

    def test_backend_validity_across_all_phases(self):
        """Selected backend is always in available_backends, regardless of phase."""
        available = ["random", "latin_hypercube", "tpe"]
        snapshots = [
            _make_snapshot(n_obs=3, seed=1),    # COLD_START
            _make_snapshot(n_obs=15, seed=2),   # LEARNING
            _make_snapshot(n_obs=50, seed=3),   # likely LEARNING or EXPLOITATION
        ]
        for snap in snapshots:
            decision = _run_pipeline(snap, seed=42, available_backends=available)
            assert decision.backend_name in available, (
                f"Backend '{decision.backend_name}' not in available list {available}"
            )


# ---------------------------------------------------------------------------
# Campaign log helper (used by golden replay tests)
# ---------------------------------------------------------------------------


def _run_campaign_log(
    spec: OptimizationSpec,
    registry: PluginRegistry,
    evaluator,
) -> DecisionLog:
    """Run a campaign and build a DecisionLog, identical to test_replay pattern."""
    from optimization_copilot.meta_controller.controller import MetaController

    diag_engine = DiagnosticEngine()
    profiler = ProblemProfiler()
    controller = MetaController(available_backends=registry.list_plugins())

    log = DecisionLog(
        campaign_id=spec.campaign_id,
        spec=spec.to_dict(),
        base_seed=spec.seed,
        metadata={"source": "acceptance_test"},
    )

    param_specs = [
        ParameterSpec(
            name=p.name,
            type=VariableType(p.type.value),
            lower=p.lower,
            upper=p.upper,
            categories=p.categories,
        )
        for p in spec.parameters
        if not p.frozen
    ]
    obj_names = [o.name for o in spec.objectives]
    obj_dirs = [o.direction.value for o in spec.objectives]

    accumulated_obs: list[Observation] = []
    previous_phase: Phase | None = None
    max_samples = spec.budget.max_samples or 100
    iteration = 0

    while len(accumulated_obs) < max_samples and iteration < 100:
        seed = spec.seed + iteration

        snapshot = CampaignSnapshot(
            campaign_id=spec.campaign_id,
            parameter_specs=list(param_specs),
            observations=list(accumulated_obs),
            objective_names=list(obj_names),
            objective_directions=list(obj_dirs),
            current_iteration=iteration,
            metadata={},
        )

        diag_vector = diag_engine.compute(snapshot)
        diag_dict = diag_vector.to_dict()
        fingerprint = profiler.profile(snapshot)

        decision = controller.decide(
            snapshot=snapshot,
            diagnostics=diag_dict,
            fingerprint=fingerprint,
            seed=seed,
            previous_phase=previous_phase,
        )

        plugin = registry.get(decision.backend_name)
        plugin.fit(snapshot.observations, snapshot.parameter_specs)
        candidates = plugin.suggest(n_suggestions=decision.batch_size, seed=seed)

        results: list[dict[str, Any]] = []
        for cand in candidates:
            try:
                kpi_values = evaluator(cand)
                results.append({
                    "iteration": iteration,
                    "parameters": dict(cand),
                    "kpi_values": dict(kpi_values),
                    "qc_passed": True,
                    "is_failure": False,
                    "failure_reason": None,
                    "timestamp": 0.0,
                    "metadata": {},
                })
            except Exception as exc:
                results.append({
                    "iteration": iteration,
                    "parameters": dict(cand),
                    "kpi_values": {},
                    "qc_passed": False,
                    "is_failure": True,
                    "failure_reason": str(exc),
                    "timestamp": 0.0,
                    "metadata": {},
                })

        entry = ReplayEngine.record_iteration(
            iteration=iteration,
            snapshot=snapshot,
            diagnostics_vector=diag_dict,
            fingerprint_dict=fingerprint.to_dict(),
            decision=decision,
            candidates=candidates,
            results=results,
            seed=seed,
        )
        log.append(entry)
        previous_phase = decision.phase

        for r in results:
            accumulated_obs.append(
                Observation(
                    iteration=r["iteration"],
                    parameters=r["parameters"],
                    kpi_values=r["kpi_values"],
                    qc_passed=r["qc_passed"],
                    is_failure=r["is_failure"],
                    failure_reason=r.get("failure_reason"),
                    timestamp=r.get("timestamp", 0.0),
                    metadata=r.get("metadata", {}),
                )
            )

        iteration += 1

    return log
