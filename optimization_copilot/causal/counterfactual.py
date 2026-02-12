"""Counterfactual reasoning via structural causal models.

Implements the three-step counterfactual procedure (Abduction, Action,
Prediction) and probabilities of causation (necessity, sufficiency).
"""

from __future__ import annotations

from typing import Callable

from optimization_copilot.causal.models import CausalGraph


class CounterfactualReasoner:
    """Counterfactual reasoning engine using structural causal models.

    Parameters
    ----------
    graph : CausalGraph
        The causal DAG defining variable relationships.
    structural_equations : dict[str, Callable]
        Maps variable name to a callable ``f(parent_values: dict) -> value``.
        Each equation represents the deterministic component of the SCM.
        The callable receives a dict of parent variable values and returns
        the computed value for the target variable.
    """

    def __init__(
        self,
        graph: CausalGraph,
        structural_equations: dict[str, Callable],
    ) -> None:
        self.graph = graph
        self.structural_equations = structural_equations

    def counterfactual(
        self,
        factual: dict,
        intervention: dict[str, float],
        query_var: str,
    ) -> dict:
        """Compute a counterfactual query using the three-step procedure.

        Three steps:
        1. **Abduction**: Infer noise terms U from factual evidence.
        2. **Action**: Modify structural equations for intervention variables.
        3. **Prediction**: Compute counterfactual value using inferred noise.

        Parameters
        ----------
        factual : dict
            Observed (factual) values for all variables.
        intervention : dict[str, float]
            Variables to intervene on and their counterfactual values.
        query_var : str
            The variable whose counterfactual value we want to compute.

        Returns
        -------
        dict
            Dictionary with ``"factual_value"``, ``"counterfactual_value"``,
            and ``"noise_terms"``.
        """
        # Step 1: Abduction - infer noise from factual observations
        noise = self._abduction(factual)

        # Step 2 & 3: Action + Prediction
        # Compute all variables in topological order under the intervention
        topo_order = self.graph.topological_sort()
        cf_values: dict[str, float] = {}

        for var in topo_order:
            if var in intervention:
                # Action: set to intervention value
                cf_values[var] = intervention[var]
            elif var in self.structural_equations:
                # Prediction: use structural equation + inferred noise
                parent_names = self.graph.parents(var)
                parent_values = {p: cf_values[p] for p in parent_names if p in cf_values}

                # Compute deterministic part
                deterministic = self.structural_equations[var](parent_values)

                # Add inferred noise
                u_var = noise.get(var, 0.0)
                cf_values[var] = deterministic + u_var
            else:
                # Exogenous variable: use factual value
                cf_values[var] = factual.get(var, 0.0)

        factual_value = factual.get(query_var, 0.0)
        counterfactual_value = cf_values.get(query_var, 0.0)

        return {
            "factual_value": factual_value,
            "counterfactual_value": counterfactual_value,
            "noise_terms": noise,
        }

    def probability_of_necessity(
        self,
        data: list[dict],
        treatment: str,
        outcome: str,
    ) -> float:
        """Estimate the Probability of Necessity (PN).

        PN = P(Y_{x=0} = 0 | X=1, Y=1)
        "Given that X=1 and Y=1 occurred, what is the probability that
        Y would have been 0 had X been 0?"

        Uses the observational data and the structural model for estimation.

        Parameters
        ----------
        data : list[dict]
            Observational data as list of records.
        treatment : str
            Treatment variable name.
        outcome : str
            Outcome variable name.

        Returns
        -------
        float
            Estimated probability of necessity in [0, 1].
        """
        if not data:
            return 0.0

        # Get treatment/outcome medians for binarization
        t_vals = sorted(d[treatment] for d in data if treatment in d)
        o_vals = sorted(d[outcome] for d in data if outcome in d)
        if not t_vals or not o_vals:
            return 0.0

        t_median = t_vals[len(t_vals) // 2]
        o_median = o_vals[len(o_vals) // 2]

        # Cases where X=1 (high) and Y=1 (high)
        xy_cases = [
            d for d in data
            if treatment in d and outcome in d
            and d[treatment] >= t_median
            and d[outcome] >= o_median
        ]

        if not xy_cases:
            return 0.0

        # For each factual (X=1, Y=1) case, compute counterfactual Y(X=0)
        necessary_count = 0
        valid_count = 0

        for factual in xy_cases:
            try:
                result = self.counterfactual(
                    factual=factual,
                    intervention={treatment: t_median - abs(t_median) - 1.0},
                    query_var=outcome,
                )
                cf_y = result["counterfactual_value"]
                valid_count += 1
                # Would Y have been low (0) under no treatment?
                if cf_y < o_median:
                    necessary_count += 1
            except Exception:
                continue

        if valid_count == 0:
            return 0.0
        return necessary_count / valid_count

    def probability_of_sufficiency(
        self,
        data: list[dict],
        treatment: str,
        outcome: str,
    ) -> float:
        """Estimate the Probability of Sufficiency (PS).

        PS = P(Y_{x=1} = 1 | X=0, Y=0)
        "Given that X=0 and Y=0 occurred, what is the probability that
        Y would have been 1 had X been 1?"

        Parameters
        ----------
        data : list[dict]
            Observational data.
        treatment : str
            Treatment variable name.
        outcome : str
            Outcome variable name.

        Returns
        -------
        float
            Estimated probability of sufficiency in [0, 1].
        """
        if not data:
            return 0.0

        # Get treatment/outcome medians for binarization
        t_vals = sorted(d[treatment] for d in data if treatment in d)
        o_vals = sorted(d[outcome] for d in data if outcome in d)
        if not t_vals or not o_vals:
            return 0.0

        t_median = t_vals[len(t_vals) // 2]
        o_median = o_vals[len(o_vals) // 2]

        # Cases where X=0 (low) and Y=0 (low)
        no_xy_cases = [
            d for d in data
            if treatment in d and outcome in d
            and d[treatment] < t_median
            and d[outcome] < o_median
        ]

        if not no_xy_cases:
            return 0.0

        # For each factual (X=0, Y=0) case, compute counterfactual Y(X=1)
        sufficient_count = 0
        valid_count = 0

        for factual in no_xy_cases:
            try:
                result = self.counterfactual(
                    factual=factual,
                    intervention={treatment: t_median + abs(t_median) + 1.0},
                    query_var=outcome,
                )
                cf_y = result["counterfactual_value"]
                valid_count += 1
                # Would Y have been high (1) under treatment?
                if cf_y >= o_median:
                    sufficient_count += 1
            except Exception:
                continue

        if valid_count == 0:
            return 0.0
        return sufficient_count / valid_count

    # -- Internal helpers -------------------------------------------------------

    def _abduction(self, factual: dict) -> dict[str, float]:
        """Infer noise terms from factual observations.

        For each endogenous variable V with structural equation
        V = f(parents) + U_V, infer U_V = V_observed - f(parents_observed).

        Parameters
        ----------
        factual : dict
            Observed values for variables.

        Returns
        -------
        dict[str, float]
            Inferred noise terms for each variable.
        """
        noise: dict[str, float] = {}
        topo_order = self.graph.topological_sort()

        for var in topo_order:
            if var not in self.structural_equations:
                # Exogenous variable: noise is the value itself
                noise[var] = 0.0
                continue

            if var not in factual:
                noise[var] = 0.0
                continue

            # Get parent values from factual data
            parent_names = self.graph.parents(var)
            parent_values = {
                p: factual[p] for p in parent_names if p in factual
            }

            # Compute deterministic prediction
            try:
                deterministic = self.structural_equations[var](parent_values)
            except Exception:
                noise[var] = 0.0
                continue

            # Noise = observed - predicted
            noise[var] = factual[var] - deterministic

        return noise
