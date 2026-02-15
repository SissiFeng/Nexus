"""Offline simulator for multi-fidelity screening strategies.

Compares gated vs. ungated (full-evaluation) approaches to quantify
cost savings and false-reject rates for a given :class:`FidelityGraph`
and set of candidate parameter dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from optimization_copilot.workflow.fidelity_graph import (
    CandidateTrajectory,
    FidelityGraph,
)


# ── Per-gate statistics ───────────────────────────────────


@dataclass
class PerGateStats:
    """Aggregate statistics for a single gate across a simulation.

    Parameters
    ----------
    gate_name : str
        Canonical gate identifier (``"from->to"``).
    n_evaluated : int
        Number of candidates evaluated at this gate.
    n_passed : int
        Number of candidates that passed.
    n_failed : int
        Number of candidates that failed.
    pass_rate : float
        Fraction of evaluated candidates that passed.
    """

    gate_name: str
    n_evaluated: int
    n_passed: int
    n_failed: int
    pass_rate: float


# ── Simulation result ─────────────────────────────────────


@dataclass
class SimulationResult:
    """Aggregate result of a fidelity simulation.

    Parameters
    ----------
    n_candidates : int
        Total candidates in the simulation.
    n_passed_all : int
        Candidates that reached the final stage with gates active.
    n_truly_good : int
        Candidates that meet the final-stage KPI threshold when all
        stages are evaluated without gates.
    false_reject_rate : float
        Fraction of truly-good candidates rejected by the gates.
    cost_with_gates : float
        Total evaluation cost using gated screening.
    cost_without_gates : float
        Total evaluation cost if every candidate traverses all stages.
    cost_savings_fraction : float
        ``1 - cost_with_gates / cost_without_gates``.
    per_gate_stats : list[PerGateStats]
        Per-gate aggregate statistics.
    trajectories : list[CandidateTrajectory]
        Individual candidate trajectories from the gated run.
    """

    n_candidates: int
    n_passed_all: int
    n_truly_good: int
    false_reject_rate: float
    cost_with_gates: float
    cost_without_gates: float
    cost_savings_fraction: float
    per_gate_stats: list[PerGateStats]
    trajectories: list[CandidateTrajectory]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a plain dictionary (no trajectories)."""
        return {
            "n_candidates": self.n_candidates,
            "n_passed_all": self.n_passed_all,
            "n_truly_good": self.n_truly_good,
            "false_reject_rate": self.false_reject_rate,
            "cost_with_gates": self.cost_with_gates,
            "cost_without_gates": self.cost_without_gates,
            "cost_savings_fraction": self.cost_savings_fraction,
            "per_gate_stats": [
                {
                    "gate_name": g.gate_name,
                    "n_evaluated": g.n_evaluated,
                    "n_passed": g.n_passed,
                    "n_failed": g.n_failed,
                    "pass_rate": g.pass_rate,
                }
                for g in self.per_gate_stats
            ],
        }


# ── Simulator ─────────────────────────────────────────────


class FidelitySimulator:
    """Offline simulator for multi-fidelity screening strategies.

    Given a :class:`FidelityGraph` and a set of candidates, the
    simulator performs two passes:

    1. **Ungated pass** -- every candidate is evaluated at every stage
       (ignoring gates) to identify *truly good* candidates.
    2. **Gated pass** -- candidates flow through the graph with gates
       active; only those passing each gate proceed.

    The comparison yields:

    * **false_reject_rate** -- fraction of truly-good candidates that
      are rejected by the gates.
    * **cost_savings_fraction** -- how much cheaper the gated approach
      is compared to full evaluation.
    """

    def simulate(
        self,
        graph: FidelityGraph,
        candidates: list[dict[str, Any]],
        evaluator: Callable[[str, dict[str, Any]], dict[str, float]],
        final_kpi_name: str,
        final_kpi_threshold: float,
        final_kpi_direction: str = "maximize",
        seed: int = 42,
    ) -> SimulationResult:
        """Run the simulation.

        Parameters
        ----------
        graph : FidelityGraph
            Fidelity graph with stages and gates.
        candidates : list[dict[str, Any]]
            Each element must have ``"id"`` (str) and ``"parameters"``
            (dict) keys.
        evaluator : Callable[[str, dict[str, Any]], dict[str, float]]
            ``evaluator(stage_name, parameters)`` returns a dict of
            KPI name to value.
        final_kpi_name : str
            KPI to check at the final stage for "truly good" status.
        final_kpi_threshold : float
            Threshold for *truly good*.
        final_kpi_direction : str
            ``"maximize"`` means good = ``kpi >= threshold``;
            ``"minimize"`` means good = ``kpi <= threshold``.
        seed : int
            Random seed (reserved for deterministic tie-breaking in
            future extensions).

        Returns
        -------
        SimulationResult
            Contains all comparison metrics, per-gate stats, and
            individual trajectories.
        """
        order = graph.topological_order()
        total_cost_per_candidate = sum(
            graph._stages[s].cost for s in order
        )

        # ── Phase 1: ungated (full evaluation) ────────────
        truly_good_ids: set[str] = set()
        cost_without_gates: float = 0.0

        for cand in candidates:
            cand_id = cand["id"]
            params = cand["parameters"]
            all_kpis: dict[str, float] = {}
            for stage_name in order:
                kpis = evaluator(stage_name, params)
                all_kpis.update(kpis)
            cost_without_gates += total_cost_per_candidate

            final_val = all_kpis.get(final_kpi_name)
            if final_val is not None:
                if (
                    final_kpi_direction == "maximize"
                    and final_val >= final_kpi_threshold
                ):
                    truly_good_ids.add(cand_id)
                elif (
                    final_kpi_direction == "minimize"
                    and final_val <= final_kpi_threshold
                ):
                    truly_good_ids.add(cand_id)

        # ── Phase 2: gated evaluation ─────────────────────
        trajectories: list[CandidateTrajectory] = []
        gate_eval_counts: dict[str, dict[str, int]] = {}

        for cand in candidates:
            cand_id = cand["id"]
            params = cand["parameters"]
            trajectory = graph.run_candidate(cand_id, params, evaluator)
            trajectories.append(trajectory)

            for gd in trajectory.gate_decisions:
                if gd.gate_name not in gate_eval_counts:
                    gate_eval_counts[gd.gate_name] = {
                        "evaluated": 0,
                        "passed": 0,
                        "failed": 0,
                    }
                gate_eval_counts[gd.gate_name]["evaluated"] += 1
                if gd.passed:
                    gate_eval_counts[gd.gate_name]["passed"] += 1
                else:
                    gate_eval_counts[gd.gate_name]["failed"] += 1

        # ── Compute metrics ───────────────────────────────
        passed_ids = {
            t.candidate_id for t in trajectories if t.reached_final_stage
        }
        cost_with_gates = sum(t.total_cost for t in trajectories)

        # False reject rate
        if truly_good_ids:
            rejected_good = truly_good_ids - passed_ids
            false_reject_rate = len(rejected_good) / len(truly_good_ids)
        else:
            false_reject_rate = 0.0

        # Cost savings
        if cost_without_gates > 0:
            cost_savings = 1.0 - cost_with_gates / cost_without_gates
        else:
            cost_savings = 0.0

        # Per-gate stats
        per_gate: list[PerGateStats] = []
        for gate_name, counts in gate_eval_counts.items():
            n_eval = counts["evaluated"]
            n_pass = counts["passed"]
            n_fail = counts["failed"]
            per_gate.append(
                PerGateStats(
                    gate_name=gate_name,
                    n_evaluated=n_eval,
                    n_passed=n_pass,
                    n_failed=n_fail,
                    pass_rate=n_pass / n_eval if n_eval > 0 else 0.0,
                )
            )

        return SimulationResult(
            n_candidates=len(candidates),
            n_passed_all=len(passed_ids),
            n_truly_good=len(truly_good_ids),
            false_reject_rate=false_reject_rate,
            cost_with_gates=cost_with_gates,
            cost_without_gates=cost_without_gates,
            cost_savings_fraction=cost_savings,
            per_gate_stats=per_gate,
            trajectories=trajectories,
        )
