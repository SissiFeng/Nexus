"""Causal effect estimation from observational data.

Provides Average Treatment Effect (ATE), Conditional ATE (CATE), and
Natural Direct Effect (NDE) estimation using adjustment-based approaches.
"""

from __future__ import annotations

import math

from optimization_copilot.causal.models import CausalGraph


class CausalEffectEstimator:
    """Estimate causal effects from observational data and a causal graph.

    Supports ATE, CATE (by subgroup), and natural direct effect estimation
    via mediation analysis.
    """

    def ate(
        self,
        data: list[dict],
        treatment: str,
        outcome: str,
        adjustment_set: set[str],
        graph: CausalGraph,
    ) -> dict:
        """Estimate the Average Treatment Effect via adjustment.

        Splits data by median of treatment variable and computes the
        adjusted difference in expected outcomes.

        Parameters
        ----------
        data : list[dict]
            Observational data as list of records.
        treatment : str
            Treatment variable name.
        outcome : str
            Outcome variable name.
        adjustment_set : set[str]
            Variables to adjust for (must satisfy backdoor criterion).
        graph : CausalGraph
            The causal graph (used for validation context).

        Returns
        -------
        dict
            Dictionary with ``"ate"``, ``"se"`` (standard error),
            ``"ci_lower"``, ``"ci_upper"`` (95% confidence interval),
            and ``"n_obs"``.
        """
        valid_data = [
            d for d in data
            if treatment in d and outcome in d
            and all(v in d for v in adjustment_set)
        ]
        n = len(valid_data)
        if n < 4:
            return {
                "ate": 0.0,
                "se": float("inf"),
                "ci_lower": float("-inf"),
                "ci_upper": float("inf"),
                "n_obs": n,
            }

        # Split by treatment median
        t_vals = sorted(d[treatment] for d in valid_data)
        median_t = t_vals[n // 2]

        treated = [d for d in valid_data if d[treatment] >= median_t]
        control = [d for d in valid_data if d[treatment] < median_t]

        if not treated or not control:
            return {
                "ate": 0.0,
                "se": float("inf"),
                "ci_lower": float("-inf"),
                "ci_upper": float("inf"),
                "n_obs": n,
            }

        if not adjustment_set:
            # Simple difference in means
            mean_t = sum(d[outcome] for d in treated) / len(treated)
            mean_c = sum(d[outcome] for d in control) / len(control)
            ate_val = mean_t - mean_c

            # Standard error
            var_t = sum((d[outcome] - mean_t) ** 2 for d in treated) / max(len(treated) - 1, 1)
            var_c = sum((d[outcome] - mean_c) ** 2 for d in control) / max(len(control) - 1, 1)
            se = math.sqrt(var_t / len(treated) + var_c / len(control))
        else:
            # Stratified estimation
            ate_val, se = self._stratified_ate(
                valid_data, treated, control, outcome, adjustment_set, n,
            )

        ci_lower = ate_val - 1.96 * se
        ci_upper = ate_val + 1.96 * se

        return {
            "ate": ate_val,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": n,
        }

    def cate(
        self,
        data: list[dict],
        treatment: str,
        outcome: str,
        adjustment_set: set[str],
        subgroup_var: str,
    ) -> dict:
        """Estimate the Conditional ATE by subgroup.

        Splits the data by the median of *subgroup_var* and estimates
        ATE separately for each subgroup.

        Parameters
        ----------
        data : list[dict]
            Observational data as list of records.
        treatment : str
            Treatment variable name.
        outcome : str
            Outcome variable name.
        adjustment_set : set[str]
            Adjustment set variables.
        subgroup_var : str
            Variable to stratify by for conditional effects.

        Returns
        -------
        dict
            Dictionary with ``"subgroups"`` mapping subgroup labels to
            their estimated ATEs, and ``"overall_ate"``.
        """
        valid_data = [
            d for d in data
            if treatment in d and outcome in d and subgroup_var in d
            and all(v in d for v in adjustment_set)
        ]

        if not valid_data:
            return {"subgroups": {}, "overall_ate": 0.0}

        # Split by subgroup median
        sg_vals = sorted(d[subgroup_var] for d in valid_data)
        median_sg = sg_vals[len(sg_vals) // 2]

        high_group = [d for d in valid_data if d[subgroup_var] >= median_sg]
        low_group = [d for d in valid_data if d[subgroup_var] < median_sg]

        subgroups: dict[str, dict] = {}
        graph = CausalGraph()  # placeholder for ate call

        if high_group:
            high_result = self.ate(
                high_group, treatment, outcome, adjustment_set, graph,
            )
            subgroups[f"{subgroup_var}_high"] = high_result

        if low_group:
            low_result = self.ate(
                low_group, treatment, outcome, adjustment_set, graph,
            )
            subgroups[f"{subgroup_var}_low"] = low_result

        # Overall ATE
        overall = self.ate(valid_data, treatment, outcome, adjustment_set, graph)

        return {
            "subgroups": subgroups,
            "overall_ate": overall["ate"],
        }

    def natural_direct_effect(
        self,
        graph: CausalGraph,
        data: list[dict],
        treatment: str,
        mediator: str,
        outcome: str,
    ) -> float:
        """Estimate the Natural Direct Effect via mediation analysis.

        NDE = E[Y(x=1, M(x=0))] - E[Y(x=0, M(x=0))]

        Uses the difference-in-coefficients approach:
        NDE = Total Effect - Natural Indirect Effect

        Parameters
        ----------
        graph : CausalGraph
            The causal graph.
        data : list[dict]
            Observational data.
        treatment : str
            Treatment variable name.
        mediator : str
            Mediator variable name.
        outcome : str
            Outcome variable name.

        Returns
        -------
        float
            Estimated natural direct effect.
        """
        valid_data = [
            d for d in data
            if treatment in d and mediator in d and outcome in d
        ]
        n = len(valid_data)
        if n < 4:
            return 0.0

        # Split treatment at median
        t_vals = sorted(d[treatment] for d in valid_data)
        median_t = t_vals[n // 2]

        treated = [d for d in valid_data if d[treatment] >= median_t]
        control = [d for d in valid_data if d[treatment] < median_t]

        if not treated or not control:
            return 0.0

        # Total effect: E[Y|X=high] - E[Y|X=low]
        te = (
            sum(d[outcome] for d in treated) / len(treated)
            - sum(d[outcome] for d in control) / len(control)
        )

        # Natural Indirect Effect (NIE) via product of coefficients:
        # NIE = (E[M|X=high] - E[M|X=low]) * beta_{M->Y|X}

        # Effect of X on M
        alpha = (
            sum(d[mediator] for d in treated) / len(treated)
            - sum(d[mediator] for d in control) / len(control)
        )

        # Effect of M on Y controlling for X (using regression-like approach)
        # beta = Cov(M, Y | X) / Var(M | X)
        # We compute this within treatment groups and pool

        beta = self._conditional_coefficient(
            valid_data, mediator, outcome, treatment, median_t,
        )

        # NIE = alpha * beta
        nie = alpha * beta

        # NDE = TE - NIE
        nde = te - nie
        return nde

    # -- Internal helpers -------------------------------------------------------

    def _stratified_ate(
        self,
        all_data: list[dict],
        treated: list[dict],
        control: list[dict],
        outcome: str,
        adjustment_set: set[str],
        n: int,
    ) -> tuple[float, float]:
        """Compute stratified ATE with standard error."""
        medians: dict[str, float] = {}
        for var in adjustment_set:
            vals = sorted(d[var] for d in all_data if var in d)
            if vals:
                medians[var] = vals[len(vals) // 2]

        adj_list = sorted(adjustment_set)
        n_adj = len(adj_list)
        total_effect = 0.0
        total_weight = 0.0
        weighted_var = 0.0

        for stratum in range(1 << n_adj):
            stratum_filter: dict[str, bool] = {}
            for bit, var in enumerate(adj_list):
                stratum_filter[var] = bool(stratum & (1 << bit))

            s_treated = treated
            s_control = control
            s_all = all_data
            for var, high in stratum_filter.items():
                if var not in medians:
                    continue
                s_treated = [
                    d for d in s_treated
                    if var in d and ((d[var] >= medians[var]) == high)
                ]
                s_control = [
                    d for d in s_control
                    if var in d and ((d[var] >= medians[var]) == high)
                ]
                s_all = [
                    d for d in s_all
                    if var in d and ((d[var] >= medians[var]) == high)
                ]

            if not s_treated or not s_control or not s_all:
                continue

            p_z = len(s_all) / n
            mean_t = sum(d[outcome] for d in s_treated) / len(s_treated)
            mean_c = sum(d[outcome] for d in s_control) / len(s_control)
            stratum_effect = mean_t - mean_c

            # Within-stratum variance
            var_t = sum((d[outcome] - mean_t) ** 2 for d in s_treated) / max(len(s_treated) - 1, 1)
            var_c = sum((d[outcome] - mean_c) ** 2 for d in s_control) / max(len(s_control) - 1, 1)
            stratum_var = var_t / len(s_treated) + var_c / len(s_control)

            total_effect += stratum_effect * p_z
            total_weight += p_z
            weighted_var += (p_z ** 2) * stratum_var

        if total_weight < 1e-12:
            return 0.0, float("inf")

        ate = total_effect / total_weight
        se = math.sqrt(max(weighted_var, 0.0)) / total_weight
        return ate, se

    @staticmethod
    def _conditional_coefficient(
        data: list[dict],
        predictor: str,
        response: str,
        conditioning: str,
        cond_median: float,
    ) -> float:
        """Simple regression coefficient of predictor on response within strata."""
        betas: list[float] = []
        weights: list[int] = []

        for high in [True, False]:
            group = [
                d for d in data
                if conditioning in d
                and ((d[conditioning] >= cond_median) == high)
                and predictor in d and response in d
            ]
            if len(group) < 3:
                continue

            m_mean = sum(d[predictor] for d in group) / len(group)
            y_mean = sum(d[response] for d in group) / len(group)

            cov = sum(
                (d[predictor] - m_mean) * (d[response] - y_mean)
                for d in group
            ) / len(group)
            var_m = sum(
                (d[predictor] - m_mean) ** 2 for d in group
            ) / len(group)

            if var_m > 1e-12:
                betas.append(cov / var_m)
                weights.append(len(group))

        if not betas:
            return 0.0

        total_w = sum(weights)
        return sum(b * w for b, w in zip(betas, weights)) / total_w
