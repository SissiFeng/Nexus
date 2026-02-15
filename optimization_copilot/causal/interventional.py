"""Interventional engine implementing the do-operator via graph mutilation.

Provides causal effect estimation through backdoor and frontdoor adjustment
formulas, as well as valid adjustment set identification.
"""

from __future__ import annotations

import math
from collections import deque

from optimization_copilot.causal.models import CausalGraph


class InterventionalEngine:
    """do-operator simulation via graph mutilation.

    Implements Pearl's do-calculus for estimating causal effects from
    observational data combined with the causal graph structure.
    """

    def do(
        self,
        graph: CausalGraph,
        intervention: dict[str, float],
        data: list[dict],
        target: str,
    ) -> dict:
        """Estimate the distribution of *target* under do(*intervention*).

        Mutilates the graph by removing incoming edges to intervention
        variables, finds a valid adjustment set, and computes the
        interventional distribution.

        Parameters
        ----------
        graph : CausalGraph
            The causal graph.
        intervention : dict[str, float]
            Mapping from variable names to their intervened values.
        data : list[dict]
            Observational data as a list of records.
        target : str
            The outcome variable.

        Returns
        -------
        dict
            Dictionary with keys ``"mean"``, ``"std"``, ``"n_adjusted"``,
            and ``"adjustment_set"``.
        """
        # Mutilate: remove incoming edges to intervention variables
        mutilated = graph.copy()
        for var in intervention:
            parents = list(mutilated.parents(var))
            for parent in parents:
                mutilated.remove_edge(parent, var)

        # Find treatment variable (first intervention variable for simplicity)
        treatments = list(intervention.keys())
        treatment = treatments[0]

        # Find valid adjustment set
        adj_set = self.find_valid_adjustment_set(graph, treatment, target)

        if adj_set is None:
            adj_set = set()

        # Use backdoor adjustment
        mean_effect = self._compute_adjusted_mean(
            data, treatment, target, intervention[treatment], adj_set,
        )

        # Also compute variance for confidence
        values = self._get_adjusted_values(
            data, treatment, target, intervention[treatment], adj_set,
        )
        n = len(values)
        std = 0.0
        if n > 1:
            var = sum((v - mean_effect) ** 2 for v in values) / (n - 1)
            std = math.sqrt(max(var, 0.0))

        return {
            "mean": mean_effect,
            "std": std,
            "n_adjusted": n,
            "adjustment_set": sorted(adj_set),
        }

    def backdoor_adjustment(
        self,
        graph: CausalGraph,
        treatment: str,
        outcome: str,
        data: list[dict],
    ) -> float:
        """Compute causal effect using the backdoor adjustment formula.

        ``E[Y | do(X=1)] - E[Y | do(X=0)]`` via
        ``sum_z P(Y|X,Z) * P(Z)`` for binary/continuous variables.

        Parameters
        ----------
        graph : CausalGraph
            The causal graph.
        treatment : str
            Treatment variable name.
        outcome : str
            Outcome variable name.
        data : list[dict]
            Observational data as list of records.

        Returns
        -------
        float
            Estimated causal effect (difference in adjusted means).
        """
        adj_set = self.find_valid_adjustment_set(graph, treatment, outcome)
        if adj_set is None:
            adj_set = set()

        # Stratified estimation
        return self._stratified_effect(data, treatment, outcome, adj_set)

    def frontdoor_adjustment(
        self,
        graph: CausalGraph,
        treatment: str,
        mediator: str,
        outcome: str,
        data: list[dict],
    ) -> float:
        """Compute causal effect via the frontdoor adjustment formula.

        Applicable when treatment -> mediator -> outcome and all
        backdoor paths from treatment to mediator are blocked by the
        empty set, and all backdoor paths from mediator to outcome are
        blocked by treatment.

        Parameters
        ----------
        graph : CausalGraph
            The causal graph.
        treatment : str
            Treatment variable name.
        mediator : str
            Mediator variable name.
        outcome : str
            Outcome variable name.
        data : list[dict]
            Observational data as list of records.

        Returns
        -------
        float
            Estimated causal effect via frontdoor formula.
        """
        # Frontdoor: E[Y|do(X=x)] = sum_m P(M=m|X=x) * sum_x' P(Y|X=x',M=m) * P(X=x')
        n = len(data)
        if n == 0:
            return 0.0

        # Split treatment into high/low by median
        treatment_vals = sorted(d[treatment] for d in data if treatment in d)
        if not treatment_vals:
            return 0.0
        median_t = treatment_vals[len(treatment_vals) // 2]

        # E[Y|do(X=high)] - E[Y|do(X=low)]
        def frontdoor_expectation(x_val_high: bool) -> float:
            """Estimate E[Y|do(X=x)] using frontdoor formula."""
            total = 0.0
            count = 0

            # Group by mediator values (discretize into bins)
            med_vals = [d[mediator] for d in data if mediator in d]
            if not med_vals:
                return 0.0
            med_median = sorted(med_vals)[len(med_vals) // 2]

            for m_high in [True, False]:
                # P(M=m|X=x)
                x_group = [
                    d for d in data
                    if treatment in d
                    and ((d[treatment] >= median_t) == x_val_high)
                ]
                if not x_group:
                    continue

                m_given_x = sum(
                    1 for d in x_group
                    if mediator in d
                    and ((d[mediator] >= med_median) == m_high)
                ) / len(x_group)

                # sum_x' P(Y|X=x',M=m) * P(X=x')
                inner_sum = 0.0
                for x_prime_high in [True, False]:
                    # P(X=x')
                    p_x_prime = sum(
                        1 for d in data
                        if treatment in d
                        and ((d[treatment] >= median_t) == x_prime_high)
                    ) / n

                    # E[Y|X=x', M=m]
                    matching = [
                        d[outcome] for d in data
                        if (treatment in d and mediator in d and outcome in d)
                        and ((d[treatment] >= median_t) == x_prime_high)
                        and ((d[mediator] >= med_median) == m_high)
                    ]
                    if matching:
                        e_y = sum(matching) / len(matching)
                        inner_sum += e_y * p_x_prime

                total += m_given_x * inner_sum
                count += 1

            return total

        e_high = frontdoor_expectation(True)
        e_low = frontdoor_expectation(False)
        return e_high - e_low

    def find_valid_adjustment_set(
        self,
        graph: CausalGraph,
        treatment: str,
        outcome: str,
    ) -> set[str] | None:
        """Find a minimal valid adjustment set using the backdoor criterion.

        A set Z satisfies the backdoor criterion relative to (X, Y) if:
        1. No node in Z is a descendant of X
        2. Z blocks every path between X and Y that contains an arrow into X

        Parameters
        ----------
        graph : CausalGraph
            The causal graph.
        treatment : str
            Treatment variable name.
        outcome : str
            Outcome variable name.

        Returns
        -------
        set[str] | None
            A valid adjustment set, or ``None`` if no valid set can be found.
        """
        # Get descendants of treatment (cannot be in adjustment set)
        descendants_of_treatment = graph.descendants(treatment)

        # Candidate adjustment variables: parents of treatment that are not
        # descendants of treatment
        all_nodes = set(graph.node_names) - {treatment, outcome}
        candidates = all_nodes - descendants_of_treatment

        # Try the parents of treatment first (most common minimal set)
        parents_of_treatment = graph.parents(treatment)
        valid_parents = parents_of_treatment - descendants_of_treatment - {outcome}

        if valid_parents:
            # Verify d-separation in the mutilated graph
            mutilated = graph.copy()
            for parent in list(mutilated.parents(treatment)):
                mutilated.remove_edge(parent, treatment)
            if mutilated.d_separated(treatment, outcome, valid_parents):
                return valid_parents

        # Try all ancestors of treatment that are not descendants
        ancestors_of_treatment = graph.ancestors(treatment)
        candidate_set = (ancestors_of_treatment & candidates) - {outcome}

        if candidate_set:
            mutilated = graph.copy()
            for parent in list(mutilated.parents(treatment)):
                mutilated.remove_edge(parent, treatment)
            if mutilated.d_separated(treatment, outcome, candidate_set):
                return candidate_set

        # Try all valid candidates
        if candidates:
            valid_candidates = candidates - {outcome}
            return valid_candidates if valid_candidates else set()

        return set()

    # -- Internal helpers -------------------------------------------------------

    def _compute_adjusted_mean(
        self,
        data: list[dict],
        treatment: str,
        target: str,
        treatment_value: float,
        adj_set: set[str],
    ) -> float:
        """Compute the adjusted mean of target given do(treatment=value)."""
        if not data:
            return 0.0

        if not adj_set:
            # No adjustment needed: simple conditional mean
            # Use closest observations to treatment_value
            sorted_data = sorted(
                [d for d in data if treatment in d and target in d],
                key=lambda d: abs(d[treatment] - treatment_value),
            )
            if not sorted_data:
                return 0.0
            # Use nearest 30% of data
            k = max(1, len(sorted_data) // 3)
            return sum(d[target] for d in sorted_data[:k]) / k

        # Stratified adjustment
        return self._stratified_mean(
            data, treatment, target, treatment_value, adj_set,
        )

    def _stratified_mean(
        self,
        data: list[dict],
        treatment: str,
        target: str,
        treatment_value: float,
        adj_set: set[str],
    ) -> float:
        """Stratified mean via adjustment formula."""
        # Bin adjustment variables at their medians
        medians: dict[str, float] = {}
        for var in adj_set:
            vals = sorted(d[var] for d in data if var in d)
            if vals:
                medians[var] = vals[len(vals) // 2]
            else:
                medians[var] = 0.0

        total = 0.0
        weight = 0.0
        n = len(data)

        # Iterate over strata (2^|adj_set| strata using median splits)
        n_adj = len(adj_set)
        adj_list = sorted(adj_set)

        for stratum in range(1 << n_adj):
            # Determine which side of the median for each adj variable
            stratum_filter: dict[str, bool] = {}
            for bit, var in enumerate(adj_list):
                stratum_filter[var] = bool(stratum & (1 << bit))

            # Filter data to this stratum
            stratum_data = data
            for var, high in stratum_filter.items():
                stratum_data = [
                    d for d in stratum_data
                    if var in d and ((d[var] >= medians[var]) == high)
                ]

            if not stratum_data:
                continue

            # P(Z=z) = proportion in this stratum
            p_z = len(stratum_data) / n

            # E[Y | X close to treatment_value, Z=z]
            sorted_stratum = sorted(
                [d for d in stratum_data if treatment in d and target in d],
                key=lambda d: abs(d[treatment] - treatment_value),
            )
            if not sorted_stratum:
                continue

            k = max(1, len(sorted_stratum) // 3)
            e_y_given_x_z = sum(d[target] for d in sorted_stratum[:k]) / k

            total += e_y_given_x_z * p_z
            weight += p_z

        if weight < 1e-12:
            return 0.0
        return total / weight

    def _get_adjusted_values(
        self,
        data: list[dict],
        treatment: str,
        target: str,
        treatment_value: float,
        adj_set: set[str],
    ) -> list[float]:
        """Get the outcome values for observations near the treatment value."""
        filtered = [
            d for d in data
            if treatment in d and target in d
        ]
        sorted_data = sorted(
            filtered, key=lambda d: abs(d[treatment] - treatment_value),
        )
        k = max(1, len(sorted_data) // 3)
        return [d[target] for d in sorted_data[:k]]

    def _stratified_effect(
        self,
        data: list[dict],
        treatment: str,
        outcome: str,
        adj_set: set[str],
    ) -> float:
        """Estimate ATE via stratified comparison at median split."""
        if not data:
            return 0.0

        # Split treatment at median
        t_vals = sorted(d[treatment] for d in data if treatment in d)
        if not t_vals:
            return 0.0
        median_t = t_vals[len(t_vals) // 2]

        high_group = [
            d for d in data
            if treatment in d and outcome in d and d[treatment] >= median_t
        ]
        low_group = [
            d for d in data
            if treatment in d and outcome in d and d[treatment] < median_t
        ]

        if not high_group or not low_group:
            return 0.0

        if not adj_set:
            mean_high = sum(d[outcome] for d in high_group) / len(high_group)
            mean_low = sum(d[outcome] for d in low_group) / len(low_group)
            return mean_high - mean_low

        # Stratified: weight by P(Z)
        medians: dict[str, float] = {}
        for var in adj_set:
            vals = sorted(d[var] for d in data if var in d)
            if vals:
                medians[var] = vals[len(vals) // 2]

        adj_list = sorted(adj_set)
        n_adj = len(adj_list)
        total_effect = 0.0
        total_weight = 0.0
        n = len(data)

        for stratum in range(1 << n_adj):
            stratum_filter: dict[str, bool] = {}
            for bit, var in enumerate(adj_list):
                stratum_filter[var] = bool(stratum & (1 << bit))

            s_high = high_group
            s_low = low_group
            s_all = data
            for var, high in stratum_filter.items():
                if var not in medians:
                    continue
                s_high = [
                    d for d in s_high
                    if var in d and ((d[var] >= medians[var]) == high)
                ]
                s_low = [
                    d for d in s_low
                    if var in d and ((d[var] >= medians[var]) == high)
                ]
                s_all = [
                    d for d in s_all
                    if var in d and ((d[var] >= medians[var]) == high)
                ]

            if not s_high or not s_low or not s_all:
                continue

            p_z = len(s_all) / n
            mean_h = sum(d[outcome] for d in s_high) / len(s_high)
            mean_l = sum(d[outcome] for d in s_low) / len(s_low)

            total_effect += (mean_h - mean_l) * p_z
            total_weight += p_z

        if total_weight < 1e-12:
            return 0.0
        return total_effect / total_weight
