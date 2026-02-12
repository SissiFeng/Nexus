"""Fidelity graph with stage gates for multi-fidelity candidate screening.

Provides a directed acyclic graph of fidelity stages connected by
:class:`StageGate` objects.  Candidates flow from low-fidelity to
high-fidelity stages; at each gate the accumulated KPI values are
checked against :class:`GateCondition` predicates and only qualifying
candidates proceed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ── Gate condition primitives ─────────────────────────────

_GATE_OPERATORS: dict[str, Callable[[float, float], bool]] = {
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
    ">": lambda v, t: v > t,
    "<": lambda v, t: v < t,
}


@dataclass
class GateCondition:
    """A single condition for a stage gate.

    Parameters
    ----------
    kpi_name : str
        Name of the KPI to evaluate.
    operator : str
        Comparison operator: ``">="`` | ``"<="`` | ``">"`` | ``"<"``.
    threshold : float
        Threshold value to compare against.
    """

    kpi_name: str
    operator: str  # ">=", "<=", ">", "<"
    threshold: float

    def evaluate(self, kpi_values: dict[str, float]) -> bool:
        """Check if the condition is met by the given KPI values.

        Returns ``False`` when the required KPI is missing from
        *kpi_values* or the operator is unrecognised.
        """
        val = kpi_values.get(self.kpi_name)
        if val is None:
            return False
        op_fn = _GATE_OPERATORS.get(self.operator)
        if op_fn is None:
            return False
        return op_fn(val, self.threshold)


# ── Gate decision result ──────────────────────────────────


@dataclass
class GateDecision:
    """Result of evaluating a stage gate.

    Parameters
    ----------
    gate_name : str
        Human-readable identifier for the gate.
    passed : bool
        Whether the gate was passed overall.
    condition_results : list[tuple[str, bool]]
        Per-condition results as ``(description, passed)`` pairs.
    """

    gate_name: str
    passed: bool
    condition_results: list[tuple[str, bool]]

    @property
    def summary(self) -> str:
        """One-line human-readable summary of the gate decision."""
        passed_str = "PASSED" if self.passed else "FAILED"
        details = "; ".join(
            f"{desc}={'pass' if ok else 'fail'}"
            for desc, ok in self.condition_results
        )
        return f"Gate '{self.gate_name}': {passed_str} ({details})"


# ── Stage gate ────────────────────────────────────────────


@dataclass
class StageGate:
    """A gate between two fidelity stages.

    Parameters
    ----------
    from_stage : str
        Name of the upstream stage.
    to_stage : str
        Name of the downstream stage.
    conditions : list[GateCondition]
        Conditions that must be satisfied.
    gate_mode : str
        ``"all"`` requires every condition to pass;
        ``"any"`` requires at least one condition to pass.
    """

    from_stage: str
    to_stage: str
    conditions: list[GateCondition]
    gate_mode: str = "all"  # "all" | "any"

    @property
    def name(self) -> str:
        """Canonical gate identifier ``'from->to'``."""
        return f"{self.from_stage}->{self.to_stage}"

    def evaluate(self, kpi_values: dict[str, float]) -> GateDecision:
        """Evaluate this gate against *kpi_values*.

        Returns
        -------
        GateDecision
            Contains the overall result and per-condition details.
        """
        results: list[tuple[str, bool]] = []
        for cond in self.conditions:
            desc = f"{cond.kpi_name}{cond.operator}{cond.threshold}"
            passed = cond.evaluate(kpi_values)
            results.append((desc, passed))

        if self.gate_mode == "all":
            overall = all(ok for _, ok in results)
        else:  # "any"
            overall = any(ok for _, ok in results) if results else False

        return GateDecision(
            gate_name=self.name,
            passed=overall,
            condition_results=results,
        )


# ── Fidelity stage ────────────────────────────────────────


@dataclass
class FidelityStage:
    """A stage in the fidelity graph.

    Parameters
    ----------
    name : str
        Unique identifier for this stage.
    fidelity_level : int
        Ordinal fidelity level (1 = lowest fidelity, higher = more expensive).
    cost : float
        Cost per evaluation at this stage.
    kpis : list[str]
        KPI names produced at this stage.
    gates_out : list[StageGate]
        Outgoing gates (populated by :class:`FidelityGraph.add_gate`).
    """

    name: str
    fidelity_level: int
    cost: float
    kpis: list[str]
    gates_out: list[StageGate] = field(default_factory=list)


# ── Candidate trajectory ──────────────────────────────────


@dataclass
class CandidateTrajectory:
    """Record of a candidate's journey through the fidelity graph.

    Parameters
    ----------
    candidate_id : str
        Unique identifier for the candidate.
    parameters : dict[str, Any]
        Parameter dictionary for the candidate.
    stages_completed : list[str]
        Stage names the candidate was evaluated at.
    stages_skipped : list[str]
        Stage names skipped because a gate failed.
    gate_decisions : list[GateDecision]
        All gate evaluation results encountered.
    total_cost : float
        Accumulated evaluation cost.
    reached_final_stage : bool
        Whether the candidate completed the last stage.
    kpi_values_by_stage : dict[str, dict[str, float]]
        KPI values keyed by stage name.
    """

    candidate_id: str
    parameters: dict[str, Any]
    stages_completed: list[str]
    stages_skipped: list[str]
    gate_decisions: list[GateDecision]
    total_cost: float
    reached_final_stage: bool
    kpi_values_by_stage: dict[str, dict[str, float]] = field(
        default_factory=dict,
    )


# ── Fidelity graph ────────────────────────────────────────


class FidelityGraph:
    """Directed acyclic graph of fidelity stages with gates.

    Candidates flow through stages from low to high fidelity.
    At each stage gate, candidates are evaluated and only those
    passing the gate proceed to the next stage.
    """

    def __init__(self) -> None:
        self._stages: dict[str, FidelityStage] = {}
        self._gates: list[StageGate] = []

    # ── Mutation ──────────────────────────────────────────

    def add_stage(self, stage: FidelityStage) -> None:
        """Add a fidelity stage to the graph.

        Raises
        ------
        ValueError
            If a stage with the same name already exists.
        """
        if stage.name in self._stages:
            raise ValueError(f"Stage '{stage.name}' already exists in graph")
        self._stages[stage.name] = stage

    def add_gate(self, gate: StageGate) -> None:
        """Add a gate between two stages.

        Raises
        ------
        ValueError
            If either the source or target stage is not in the graph.
        """
        if gate.from_stage not in self._stages:
            raise ValueError(
                f"Source stage '{gate.from_stage}' not in graph"
            )
        if gate.to_stage not in self._stages:
            raise ValueError(
                f"Target stage '{gate.to_stage}' not in graph"
            )
        self._gates.append(gate)
        self._stages[gate.from_stage].gates_out.append(gate)

    # ── Queries ───────────────────────────────────────────

    @property
    def stages(self) -> list[FidelityStage]:
        """All stages in insertion order."""
        return list(self._stages.values())

    @property
    def gates(self) -> list[StageGate]:
        """All gates in insertion order."""
        return list(self._gates)

    def get_stage(self, name: str) -> FidelityStage:
        """Retrieve a stage by name.

        Raises
        ------
        KeyError
            If the stage does not exist.
        """
        if name not in self._stages:
            raise KeyError(f"Stage '{name}' not found in graph")
        return self._stages[name]

    def topological_order(self) -> list[str]:
        """Return stages in topological order (sorted by fidelity_level)."""
        return [
            s.name
            for s in sorted(
                self._stages.values(), key=lambda s: s.fidelity_level
            )
        ]

    # ── Evaluation helpers ────────────────────────────────

    def evaluate_gate(
        self,
        gate: StageGate,
        kpi_values: dict[str, float],
    ) -> GateDecision:
        """Evaluate a single gate."""
        return gate.evaluate(kpi_values)

    def _incoming_gates(self, stage_name: str) -> list[StageGate]:
        """Return all gates whose target is *stage_name*."""
        return [g for g in self._gates if g.to_stage == stage_name]

    # ── Candidate execution ───────────────────────────────

    def run_candidate(
        self,
        candidate_id: str,
        parameters: dict[str, Any],
        stage_evaluator: Callable[[str, dict[str, Any]], dict[str, float]],
    ) -> CandidateTrajectory:
        """Run a single candidate through the fidelity graph.

        Parameters
        ----------
        candidate_id : str
            Unique identifier for the candidate.
        parameters : dict[str, Any]
            Parameter dictionary for the candidate.
        stage_evaluator : Callable[[str, dict[str, Any]], dict[str, float]]
            ``evaluator(stage_name, parameters)`` returns a dict of
            KPI name to value for the given stage.

        Returns
        -------
        CandidateTrajectory
            Full history including completed/skipped stages, gate
            decisions, costs, and KPI values.
        """
        order = self.topological_order()
        completed: list[str] = []
        skipped: list[str] = []
        gate_decisions: list[GateDecision] = []
        total_cost: float = 0.0
        kpi_by_stage: dict[str, dict[str, float]] = {}
        all_kpis: dict[str, float] = {}  # accumulated across stages

        for stage_name in order:
            stage = self._stages[stage_name]

            # Check incoming gates
            incoming = self._incoming_gates(stage_name)
            can_enter = True
            for gate in incoming:
                decision = gate.evaluate(all_kpis)
                gate_decisions.append(decision)
                if not decision.passed:
                    can_enter = False

            # First stage in topological order always enters
            if stage_name == order[0]:
                can_enter = True

            if can_enter:
                kpis = stage_evaluator(stage_name, parameters)
                kpi_by_stage[stage_name] = kpis
                all_kpis.update(kpis)
                total_cost += stage.cost
                completed.append(stage_name)
            else:
                skipped.append(stage_name)

        final_stage = order[-1] if order else ""
        reached_final = final_stage in completed

        return CandidateTrajectory(
            candidate_id=candidate_id,
            parameters=parameters,
            stages_completed=completed,
            stages_skipped=skipped,
            gate_decisions=gate_decisions,
            total_cost=total_cost,
            reached_final_stage=reached_final,
            kpi_values_by_stage=kpi_by_stage,
        )

    # ── Dunder helpers ────────────────────────────────────

    def __len__(self) -> int:
        return len(self._stages)

    def __contains__(self, name: str) -> bool:
        return name in self._stages
