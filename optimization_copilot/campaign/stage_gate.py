"""Human-readable stage gate screening protocol.

Wraps :class:`FidelityGraph` to produce structured, actionable screening
protocols like: *"Do UV-Vis first; if band_gap not in [2.2, 2.6] eV,
skip HER test"*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


@dataclass
class ProtocolStep:
    """One step in a screening protocol.

    Parameters
    ----------
    step_number : int
        Sequential step number (1-based).
    stage_name : str
        Name of the fidelity stage.
    description : str
        Human-readable description of what to do.
    cost : float
        Cost of evaluating this stage.
    kpis_measured : list[str]
        KPIs produced at this stage.
    gate_conditions : list[str]
        Human-readable gate conditions (e.g. ``"band_gap >= 2.2"``).
    action_if_pass : str
        What to do if the gate passes.
    action_if_fail : str
        What to do if the gate fails.
    """

    step_number: int
    stage_name: str
    description: str
    cost: float
    kpis_measured: list[str]
    gate_conditions: list[str]
    action_if_pass: str
    action_if_fail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "stage_name": self.stage_name,
            "description": self.description,
            "cost": self.cost,
            "kpis_measured": self.kpis_measured,
            "gate_conditions": self.gate_conditions,
            "action_if_pass": self.action_if_pass,
            "action_if_fail": self.action_if_fail,
        }


@dataclass
class ScreeningProtocol:
    """Complete multi-stage screening protocol.

    Parameters
    ----------
    steps : list[ProtocolStep]
        Ordered screening steps.
    total_cost_all_pass : float
        Cost if a candidate passes every stage.
    cost_first_stage : float
        Cost of just the first (cheapest) screening step.
    """

    steps: list[ProtocolStep]
    total_cost_all_pass: float
    cost_first_stage: float

    def summary(self) -> str:
        """Generate a human-readable protocol summary."""
        if not self.steps:
            return "No screening protocol defined."

        lines: list[str] = []
        for step in self.steps:
            lines.append(
                f"Step {step.step_number}: {step.description}"
            )
            for cond in step.gate_conditions:
                lines.append(f"  Gate: {cond}")
            lines.append(f"  Pass -> {step.action_if_pass}")
            lines.append(f"  Fail -> {step.action_if_fail}")

        lines.append("")
        lines.append(f"Full evaluation cost: {self.total_cost_all_pass:.1f}")
        lines.append(f"Early-exit cost (fail first stage): {self.cost_first_stage:.1f}")
        return "\n".join(lines)

    @property
    def n_stages(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "total_cost_all_pass": self.total_cost_all_pass,
            "cost_first_stage": self.cost_first_stage,
            "n_stages": self.n_stages,
            "summary": self.summary(),
        }


# ------------------------------------------------------------------
# Protocol builder
# ------------------------------------------------------------------


class StageGateProtocol:
    """Build human-readable screening protocols from a FidelityGraph."""

    def build_protocol(self, fidelity_graph: Any) -> ScreeningProtocol:
        """Generate a screening protocol from a fidelity graph.

        Parameters
        ----------
        fidelity_graph : FidelityGraph
            Graph with stages and gates defined.

        Returns
        -------
        ScreeningProtocol
            Ordered steps with gate conditions and cost analysis.
        """
        stage_order = fidelity_graph.topological_order()
        all_gates = fidelity_graph.gates

        # Build a map: from_stage → list of gates
        gates_from: dict[str, list] = {}
        for gate in all_gates:
            gates_from.setdefault(gate.from_stage, []).append(gate)

        steps: list[ProtocolStep] = []
        running_cost = 0.0

        for step_num, stage_name in enumerate(stage_order, 1):
            stage = fidelity_graph.get_stage(stage_name)
            running_cost += stage.cost

            # Gather gate conditions for gates leaving this stage
            gate_descriptions: list[str] = []
            next_stages: list[str] = []
            for gate in gates_from.get(stage_name, []):
                next_stages.append(gate.to_stage)
                for cond in gate.conditions:
                    gate_descriptions.append(
                        f"{cond.kpi_name} {cond.operator} {cond.threshold}"
                    )

            # Determine actions
            if next_stages:
                action_pass = f"Proceed to {next_stages[0]}"
            elif step_num < len(stage_order):
                action_pass = f"Proceed to {stage_order[step_num]}"
            else:
                action_pass = "Accept candidate for full evaluation"

            action_fail = (
                "Skip candidate (save remaining cost)"
                if gate_descriptions
                else "Continue"
            )

            steps.append(ProtocolStep(
                step_number=step_num,
                stage_name=stage_name,
                description=f"Run {stage_name} (cost: {stage.cost:.1f})",
                cost=stage.cost,
                kpis_measured=list(stage.kpis),
                gate_conditions=gate_descriptions,
                action_if_pass=action_pass,
                action_if_fail=action_fail,
            ))

        return ScreeningProtocol(
            steps=steps,
            total_cost_all_pass=running_cost,
            cost_first_stage=steps[0].cost if steps else 0.0,
        )

    def build_simple_protocol(
        self,
        stages: list[dict[str, Any]],
    ) -> ScreeningProtocol:
        """Build a protocol from a simple list of stage dicts.

        Useful when a full FidelityGraph is not available.  Each dict
        should have: ``name``, ``cost``, ``kpis``, and optional
        ``gate_conditions`` (list of strings).

        Parameters
        ----------
        stages : list[dict]
            Stage definitions ordered from cheapest to most expensive.

        Returns
        -------
        ScreeningProtocol
        """
        steps: list[ProtocolStep] = []
        running_cost = 0.0

        for step_num, stage_def in enumerate(stages, 1):
            cost = stage_def.get("cost", 0.0)
            running_cost += cost
            gate_conds = stage_def.get("gate_conditions", [])

            if step_num < len(stages):
                next_name = stages[step_num].get("name", f"Stage-{step_num + 1}")
                action_pass = f"Proceed to {next_name}"
            else:
                action_pass = "Accept candidate for full evaluation"

            action_fail = (
                "Skip candidate (save remaining cost)"
                if gate_conds
                else "Continue"
            )

            steps.append(ProtocolStep(
                step_number=step_num,
                stage_name=stage_def.get("name", f"Stage-{step_num}"),
                description=stage_def.get("description", f"Run {stage_def.get('name', f'Stage-{step_num}')} (cost: {cost:.1f})"),
                cost=cost,
                kpis_measured=stage_def.get("kpis", []),
                gate_conditions=gate_conds,
                action_if_pass=action_pass,
                action_if_fail=action_fail,
            ))

        return ScreeningProtocol(
            steps=steps,
            total_cost_all_pass=running_cost,
            cost_first_stage=steps[0].cost if steps else 0.0,
        )
