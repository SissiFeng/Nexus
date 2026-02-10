"""Golden scenarios for validation and regression testing.

Each scenario defines a synthetic CampaignSnapshot with known characteristics
and the expected pipeline behaviour (phase, backend family, risk posture, etc.).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    Phase,
    ProblemFingerprint,
    RiskPosture,
    StrategyDecision,
    VariableType,
)
from optimization_copilot.core.hashing import snapshot_hash, decision_hash
from optimization_copilot.diagnostics.engine import DiagnosticEngine
from optimization_copilot.profiler.profiler import ProblemProfiler
from optimization_copilot.meta_controller.controller import MetaController
from optimization_copilot.explainability.explainer import DecisionExplainer


# ── Data structures ──────────────────────────────────────


@dataclass
class ScenarioExpectation:
    """What we expect from the pipeline for a given scenario."""
    expected_phase: Phase
    expected_risk: RiskPosture | None = None
    expected_backend_family: list[str] | None = None
    min_exploration: float | None = None
    max_exploration: float | None = None
    expect_fallback: bool = False
    custom_checks: list[Callable[[StrategyDecision, dict], bool]] = field(
        default_factory=list
    )


@dataclass
class GoldenScenario:
    """A reproducible test scenario with known expected behaviour."""
    name: str
    description: str
    snapshot: CampaignSnapshot
    expectation: ScenarioExpectation
    seed: int = 42


@dataclass
class ScenarioResult:
    """Result of running a single golden scenario."""
    scenario_name: str
    passed: bool
    decision: StrategyDecision
    diagnostics: dict[str, Any]
    fingerprint: ProblemFingerprint
    failures: list[str] = field(default_factory=list)
    snapshot_hash: str = ""
    decision_hash: str = ""


# ── Scenario builders ────────────────────────────────────


def _make_specs(n_params: int = 3) -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name=f"x{i}",
            type=VariableType.CONTINUOUS,
            lower=0.0,
            upper=1.0,
        )
        for i in range(n_params)
    ]


def _make_obs(
    n: int,
    kpi_fn: Callable[[int, dict[str, float]], float],
    n_params: int = 3,
    failure_fn: Callable[[int], bool] | None = None,
    seed: int = 42,
) -> list[Observation]:
    """Generate observations with deterministic pseudo-random parameters."""
    rng = random.Random(seed)
    obs = []
    for i in range(n):
        params = {f"x{j}": rng.random() for j in range(n_params)}
        is_fail = failure_fn(i) if failure_fn else False
        kpi = kpi_fn(i, params) if not is_fail else 0.0
        obs.append(Observation(
            iteration=i,
            parameters=params,
            kpi_values={"y": kpi},
            is_failure=is_fail,
            timestamp=float(i),
        ))
    return obs


# ── Scenario 1: Clean convergence ────────────────────────

def _clean_convergence() -> GoldenScenario:
    """Steady improvement, no noise, no failures — should reach EXPLOITATION."""
    def kpi_fn(i: int, params: dict) -> float:
        return 10.0 + i * 0.5  # monotone increasing

    snap = CampaignSnapshot(
        campaign_id="golden_clean_convergence",
        parameter_specs=_make_specs(3),
        observations=_make_obs(30, kpi_fn, n_params=3),
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=30,
    )
    return GoldenScenario(
        name="clean_convergence",
        description="Steady monotone improvement → exploitation phase",
        snapshot=snap,
        expectation=ScenarioExpectation(
            expected_phase=Phase.EXPLOITATION,
            expected_risk=RiskPosture.AGGRESSIVE,
            max_exploration=0.4,
        ),
    )


# ── Scenario 2: Cold start ───────────────────────────────

def _cold_start() -> GoldenScenario:
    """Very few observations — should stay in COLD_START."""
    def kpi_fn(i: int, params: dict) -> float:
        return sum(params.values())

    snap = CampaignSnapshot(
        campaign_id="golden_cold_start",
        parameter_specs=_make_specs(5),
        observations=_make_obs(4, kpi_fn, n_params=5),
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=4,
    )
    return GoldenScenario(
        name="cold_start",
        description="Only 4 observations → cold start phase",
        snapshot=snap,
        expectation=ScenarioExpectation(
            expected_phase=Phase.COLD_START,
            expected_risk=RiskPosture.CONSERVATIVE,
            min_exploration=0.8,
        ),
    )


# ── Scenario 3: Failure-heavy regime ─────────────────────

def _failure_heavy() -> GoldenScenario:
    """High failure rate with clustering — should detect STAGNATION."""
    def kpi_fn(i: int, params: dict) -> float:
        return 5.0 + i * 0.1

    def failure_fn(i: int) -> bool:
        # First 10 OK, then heavy failures
        return i >= 10

    snap = CampaignSnapshot(
        campaign_id="golden_failure_heavy",
        parameter_specs=_make_specs(3),
        observations=_make_obs(20, kpi_fn, n_params=3, failure_fn=failure_fn),
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=20,
    )
    return GoldenScenario(
        name="failure_heavy",
        description="50% failures clustered at end → stagnation + conservative",
        snapshot=snap,
        expectation=ScenarioExpectation(
            expected_phase=Phase.STAGNATION,
            expected_risk=RiskPosture.CONSERVATIVE,
            min_exploration=0.5,
        ),
    )


# ── Scenario 4: Noisy plateau ────────────────────────────

def _noisy_plateau() -> GoldenScenario:
    """Noisy KPIs with no real improvement — should detect plateau/stagnation.

    Uses a constant base KPI (10.0) with small noise so the best-so-far
    plateaus quickly, triggering stagnation detection.
    """
    rng = random.Random(123)
    specs = _make_specs(2)
    obs = []
    for i in range(25):
        params = {f"x{j}": rng.random() for j in range(2)}
        # Constant base + tiny noise → best-so-far flatlines early
        kpi = 10.0 + rng.gauss(0, 0.05)
        obs.append(Observation(
            iteration=i,
            parameters=params,
            kpi_values={"y": kpi},
            timestamp=float(i),
        ))

    snap = CampaignSnapshot(
        campaign_id="golden_noisy_plateau",
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=25,
    )
    return GoldenScenario(
        name="noisy_plateau",
        description="Constant KPIs with tiny noise → stagnation (long plateau)",
        snapshot=snap,
        expectation=ScenarioExpectation(
            expected_phase=Phase.STAGNATION,
            min_exploration=0.5,
        ),
    )


# ── Scenario 5: Mixed-variable problem ───────────────────

def _mixed_variables() -> GoldenScenario:
    """Continuous + categorical parameters."""
    specs = [
        ParameterSpec(name="temp", type=VariableType.CONTINUOUS, lower=20.0, upper=100.0),
        ParameterSpec(name="pressure", type=VariableType.CONTINUOUS, lower=1.0, upper=10.0),
        ParameterSpec(name="catalyst", type=VariableType.CATEGORICAL, categories=["A", "B", "C"]),
    ]
    rng = random.Random(77)
    obs = []
    for i in range(15):
        params = {
            "temp": 20 + rng.random() * 80,
            "pressure": 1 + rng.random() * 9,
            "catalyst": rng.choice(["A", "B", "C"]),
        }
        kpi = params["temp"] * 0.1 + params["pressure"] * 0.5
        if params["catalyst"] == "B":
            kpi += 3.0
        obs.append(Observation(
            iteration=i,
            parameters=params,
            kpi_values={"y": kpi},
            timestamp=float(i),
        ))

    snap = CampaignSnapshot(
        campaign_id="golden_mixed_vars",
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=15,
    )
    return GoldenScenario(
        name="mixed_variables",
        description="Mixed continuous+categorical → learning phase",
        snapshot=snap,
        expectation=ScenarioExpectation(
            expected_phase=Phase.LEARNING,
        ),
    )


# ── Scenario registry ────────────────────────────────────

GOLDEN_SCENARIOS: list[GoldenScenario] = [
    _clean_convergence(),
    _cold_start(),
    _failure_heavy(),
    _noisy_plateau(),
    _mixed_variables(),
]


# ── Validation runner ────────────────────────────────────


class ValidationRunner:
    """Run golden scenarios through the full pipeline and check expectations."""

    def __init__(self, available_backends: list[str] | None = None) -> None:
        self.engine = DiagnosticEngine()
        self.profiler = ProblemProfiler()
        self.controller = MetaController(available_backends=available_backends)
        self.explainer = DecisionExplainer()

    def run_scenario(self, scenario: GoldenScenario) -> ScenarioResult:
        """Execute a single scenario through the full pipeline."""
        snap = scenario.snapshot
        expect = scenario.expectation

        # 1. Diagnostics
        diag_vector = self.engine.compute(snap)
        diag_dict = diag_vector.to_dict()

        # 2. Fingerprint
        fingerprint = self.profiler.profile(snap)

        # 3. Meta-controller decision
        decision = self.controller.decide(
            snap, diag_dict, fingerprint, seed=scenario.seed
        )

        # 4. Explainability (ensure no crash)
        report = self.explainer.explain(decision, fingerprint, diag_dict)

        # 5. Hashing (ensure determinism)
        snap_h = snapshot_hash(snap)
        dec_h = decision_hash(decision)

        # 6. Check expectations
        failures: list[str] = []

        if decision.phase != expect.expected_phase:
            failures.append(
                f"phase: expected {expect.expected_phase.value}, "
                f"got {decision.phase.value}"
            )

        if expect.expected_risk and decision.risk_posture != expect.expected_risk:
            failures.append(
                f"risk: expected {expect.expected_risk.value}, "
                f"got {decision.risk_posture.value}"
            )

        if expect.expected_backend_family:
            if decision.backend_name not in expect.expected_backend_family:
                failures.append(
                    f"backend: expected one of {expect.expected_backend_family}, "
                    f"got {decision.backend_name}"
                )

        if expect.min_exploration is not None:
            if decision.exploration_strength < expect.min_exploration:
                failures.append(
                    f"exploration too low: {decision.exploration_strength} "
                    f"< {expect.min_exploration}"
                )

        if expect.max_exploration is not None:
            if decision.exploration_strength > expect.max_exploration:
                failures.append(
                    f"exploration too high: {decision.exploration_strength} "
                    f"> {expect.max_exploration}"
                )

        if expect.expect_fallback and not decision.fallback_events:
            failures.append("expected fallback events but got none")

        for check_fn in expect.custom_checks:
            if not check_fn(decision, diag_dict):
                failures.append(f"custom check {check_fn.__name__} failed")

        # Verify explainability report is populated
        if not report.summary:
            failures.append("explainability report has empty summary")

        return ScenarioResult(
            scenario_name=scenario.name,
            passed=len(failures) == 0,
            decision=decision,
            diagnostics=diag_dict,
            fingerprint=fingerprint,
            failures=failures,
            snapshot_hash=snap_h,
            decision_hash=dec_h,
        )

    def run_all(self) -> list[ScenarioResult]:
        """Run all golden scenarios."""
        return [self.run_scenario(s) for s in GOLDEN_SCENARIOS]

    def verify_determinism(self, scenario: GoldenScenario, n_runs: int = 3) -> bool:
        """Run the same scenario multiple times and verify identical output."""
        hashes = set()
        for _ in range(n_runs):
            result = self.run_scenario(scenario)
            hashes.add(result.decision_hash)
        return len(hashes) == 1
