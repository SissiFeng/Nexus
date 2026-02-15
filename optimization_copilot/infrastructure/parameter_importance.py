"""Parameter importance analysis for optimization campaigns.

Provides three methods to assess which parameters have the most
influence on the objective function:

1. Variance-based: Compare parameter distributions in top vs. all observations
2. Correlation-based: Pearson correlation between each parameter and objective
3. PedAnova: ANOVA F-statistic between top-k% and bottom-k% groups

References:
- fANOVA (Hutter et al., 2014): Functional ANOVA for parameter importance
- PED-ANOVA (Watanabe, 2023): Practical parameter importance in Optuna
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImportanceResult:
    """Result of parameter importance analysis.

    Attributes:
        scores: Mapping of parameter name to importance score in [0, 1].
        method: Which analysis method produced these scores.
        details: Additional method-specific details (e.g., raw statistics).
    """

    scores: dict[str, float]
    method: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "scores": dict(self.scores),
            "method": self.method,
            "details": dict(self.details),
        }


class ParameterImportanceAnalyzer:
    """Parameter importance analysis with multiple methods.

    Three methods are available:

    - ``variance``: Compare parameter variance in top-20% vs all observations.
      Higher variance reduction in top performers indicates importance.
    - ``correlation``: Absolute Pearson correlation between each parameter
      and the objective. Works for continuous parameters only.
    - ``pedanova``: ANOVA F-statistic between top-20% and bottom-20%
      observations for each parameter. More robust to non-linear effects.
    - ``auto``: Uses pedanova if enough data (>= 10 observations),
      otherwise falls back to correlation.

    Usage::

        analyzer = ParameterImportanceAnalyzer(method="auto")
        result = analyzer.analyze(
            observations=[{"x": 1.0, "y": 2.0, "objective": 0.8}, ...],
            parameter_specs=[
                {"name": "x", "type": "continuous"},
                {"name": "y", "type": "continuous"},
            ],
            objective_key="objective",
        )
        print(result.scores)  # {"x": 0.85, "y": 0.32}
    """

    # Minimum observations for pedanova method
    _MIN_PEDANOVA_OBS = 10

    def __init__(self, method: str = "auto"):
        """Initialize the analyzer.

        Args:
            method: Analysis method to use. One of "variance",
                "correlation", "pedanova", or "auto".

        Raises:
            ValueError: If method is not recognized.
        """
        valid_methods = {"variance", "correlation", "pedanova", "auto"}
        if method not in valid_methods:
            raise ValueError(
                f"Unknown method '{method}'. Valid methods: {valid_methods}"
            )
        self._method = method

    @property
    def method(self) -> str:
        """Return the configured method name."""
        return self._method

    def analyze(
        self,
        observations: list[dict[str, Any]],
        parameter_specs: list[dict[str, Any]],
        objective_key: str = "objective",
    ) -> ImportanceResult:
        """Compute importance scores for all parameters.

        Args:
            observations: List of observation dicts, each containing
                parameter values and an objective value.
            parameter_specs: List of parameter specification dicts,
                each with at least 'name' and 'type' keys.
                Type should be 'continuous', 'integer', or 'categorical'.
            objective_key: Key in observation dicts for the objective value.

        Returns:
            ImportanceResult with normalized scores in [0, 1].

        Raises:
            ValueError: If no observations or parameter specs are provided.
        """
        if not observations:
            raise ValueError("No observations provided for importance analysis")
        if not parameter_specs:
            raise ValueError("No parameter specs provided")

        # Filter observations that have the objective key
        valid_obs = [
            obs for obs in observations
            if obs.get(objective_key) is not None
        ]
        if not valid_obs:
            # All scores 0 if no valid observations
            scores = {spec["name"]: 0.0 for spec in parameter_specs}
            return ImportanceResult(
                scores=scores,
                method=self._method,
                details={"reason": "no valid observations with objective"},
            )

        # Determine effective method
        effective_method = self._method
        if effective_method == "auto":
            if len(valid_obs) >= self._MIN_PEDANOVA_OBS:
                effective_method = "pedanova"
            else:
                effective_method = "correlation"

        # Dispatch to method implementation
        if effective_method == "variance":
            raw_scores = self._variance_importance(
                valid_obs, parameter_specs, objective_key
            )
        elif effective_method == "correlation":
            raw_scores = self._correlation_importance(
                valid_obs, parameter_specs, objective_key
            )
        elif effective_method == "pedanova":
            raw_scores = self._pedanova_importance(
                valid_obs, parameter_specs, objective_key
            )
        else:
            # Should not reach here due to __init__ validation
            raw_scores = {spec["name"]: 0.0 for spec in parameter_specs}

        # Normalize to [0, 1]
        normalized = self._normalize_scores(raw_scores)

        return ImportanceResult(
            scores=normalized,
            method=effective_method,
            details={"raw_scores": raw_scores, "n_observations": len(valid_obs)},
        )

    def _variance_importance(
        self,
        observations: list[dict[str, Any]],
        specs: list[dict[str, Any]],
        obj_key: str,
    ) -> dict[str, float]:
        """Variance-based importance analysis.

        Compares the variance of each parameter in the top-20%
        observations (by objective) against variance in all observations.
        A large variance reduction in the top group indicates the
        parameter is important (good values cluster tightly).

        Args:
            observations: Valid observations with objective values.
            specs: Parameter specifications.
            obj_key: Objective key in observation dicts.

        Returns:
            Raw importance scores (not yet normalized).
        """
        # Sort by objective descending (higher = better)
        sorted_obs = sorted(
            observations,
            key=lambda o: o.get(obj_key, float("-inf")),
            reverse=True,
        )

        n_top = max(1, int(math.ceil(len(sorted_obs) * 0.2)))
        top_obs = sorted_obs[:n_top]

        scores: dict[str, float] = {}
        for spec in specs:
            pname = spec["name"]
            ptype = spec.get("type", "continuous")

            if ptype == "categorical":
                # For categorical: measure entropy reduction
                scores[pname] = self._categorical_variance_importance(
                    observations, top_obs, pname
                )
                continue

            # Continuous / integer: compare variance
            all_values = self._extract_numeric(observations, pname)
            top_values = self._extract_numeric(top_obs, pname)

            if len(all_values) < 2 or len(top_values) < 1:
                scores[pname] = 0.0
                continue

            var_all = self._variance(all_values)
            var_top = self._variance(top_values) if len(top_values) >= 2 else 0.0

            if var_all <= 0:
                scores[pname] = 0.0
            else:
                # Variance reduction ratio: 1 - (var_top / var_all)
                # High reduction = important parameter
                scores[pname] = max(0.0, 1.0 - var_top / var_all)

        return scores

    def _correlation_importance(
        self,
        observations: list[dict[str, Any]],
        specs: list[dict[str, Any]],
        obj_key: str,
    ) -> dict[str, float]:
        """Correlation-based importance analysis.

        Computes the absolute Pearson correlation between each continuous
        parameter and the objective. Categorical parameters get score 0.

        Args:
            observations: Valid observations with objective values.
            specs: Parameter specifications.
            obj_key: Objective key in observation dicts.

        Returns:
            Raw importance scores (absolute correlations).
        """
        obj_values = [obs[obj_key] for obs in observations]

        scores: dict[str, float] = {}
        for spec in specs:
            pname = spec["name"]
            ptype = spec.get("type", "continuous")

            if ptype == "categorical":
                scores[pname] = 0.0
                continue

            param_values = self._extract_numeric(observations, pname)
            if len(param_values) != len(obj_values) or len(param_values) < 2:
                scores[pname] = 0.0
                continue

            corr = self._pearson_correlation(param_values, obj_values)
            scores[pname] = abs(corr)

        return scores

    def _pedanova_importance(
        self,
        observations: list[dict[str, Any]],
        specs: list[dict[str, Any]],
        obj_key: str,
    ) -> dict[str, float]:
        """PedAnova importance analysis.

        Computes one-way ANOVA F-statistic between top-20% and bottom-20%
        observations for each parameter. A high F-statistic means the
        parameter distributions differ significantly between the two groups.

        Steps:
        1. Sort observations by objective
        2. Take top-20% and bottom-20%
        3. For each continuous parameter, compute F = between_var / within_var
        4. Return raw F-statistics (normalized later)

        Args:
            observations: Valid observations with objective values.
            specs: Parameter specifications.
            obj_key: Objective key in observation dicts.

        Returns:
            Raw F-statistic scores for each parameter.
        """
        # Sort by objective descending
        sorted_obs = sorted(
            observations,
            key=lambda o: o.get(obj_key, float("-inf")),
            reverse=True,
        )

        n = len(sorted_obs)
        n_group = max(1, int(math.ceil(n * 0.2)))
        top_obs = sorted_obs[:n_group]
        bottom_obs = sorted_obs[-n_group:] if n_group < n else sorted_obs

        scores: dict[str, float] = {}
        for spec in specs:
            pname = spec["name"]
            ptype = spec.get("type", "continuous")

            if ptype == "categorical":
                # For categorical: use chi-squared-like statistic
                scores[pname] = self._categorical_f_statistic(
                    top_obs, bottom_obs, pname
                )
                continue

            top_values = self._extract_numeric(top_obs, pname)
            bottom_values = self._extract_numeric(bottom_obs, pname)

            if len(top_values) < 1 or len(bottom_values) < 1:
                scores[pname] = 0.0
                continue

            f_stat = self._f_statistic(top_values, bottom_values)
            scores[pname] = f_stat

        return scores

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        """Normalize scores to [0, 1] range.

        Uses max-normalization: each score is divided by the maximum
        score. If all scores are zero, returns all zeros.

        Args:
            scores: Raw importance scores.

        Returns:
            Normalized scores in [0, 1].
        """
        if not scores:
            return {}

        max_score = max(scores.values())
        if max_score <= 0:
            return {k: 0.0 for k in scores}

        return {k: v / max_score for k, v in scores.items()}

    @staticmethod
    def _pearson_correlation(x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient (pure Python).

        Args:
            x: First variable values.
            y: Second variable values (same length as x).

        Returns:
            Pearson correlation in [-1, 1]. Returns 0 if either
            variable has zero variance.
        """
        n = len(x)
        if n < 2:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov_xy = 0.0
        var_x = 0.0
        var_y = 0.0

        for xi, yi in zip(x, y):
            dx = xi - mean_x
            dy = yi - mean_y
            cov_xy += dx * dy
            var_x += dx * dx
            var_y += dy * dy

        denom = math.sqrt(var_x * var_y)
        if denom < 1e-15:
            return 0.0

        return cov_xy / denom

    @staticmethod
    def _f_statistic(group_a: list[float], group_b: list[float]) -> float:
        """Compute one-way ANOVA F-statistic between two groups.

        F = between-group variance / within-group variance

        Args:
            group_a: Values in group A.
            group_b: Values in group B.

        Returns:
            F-statistic value. Returns 0 if within-group variance is zero.
        """
        n_a = len(group_a)
        n_b = len(group_b)
        n_total = n_a + n_b

        if n_total < 2 or n_a == 0 or n_b == 0:
            return 0.0

        mean_a = sum(group_a) / n_a
        mean_b = sum(group_b) / n_b
        grand_mean = (sum(group_a) + sum(group_b)) / n_total

        # Between-group sum of squares (SSB)
        ssb = n_a * (mean_a - grand_mean) ** 2 + n_b * (mean_b - grand_mean) ** 2

        # Within-group sum of squares (SSW)
        ssw = sum((v - mean_a) ** 2 for v in group_a) + sum(
            (v - mean_b) ** 2 for v in group_b
        )

        # Degrees of freedom
        df_between = 1  # 2 groups - 1
        df_within = n_total - 2  # n_total - k groups

        if df_within <= 0 or ssw <= 0:
            # If no within-group variance, groups are perfectly separated
            return float("inf") if ssb > 0 else 0.0

        msb = ssb / df_between
        msw = ssw / df_within

        return msb / msw

    @staticmethod
    def _extract_numeric(
        observations: list[dict[str, Any]], key: str
    ) -> list[float]:
        """Extract numeric values for a parameter from observations.

        Args:
            observations: List of observation dicts.
            key: Parameter name to extract.

        Returns:
            List of float values (observations missing the key or
            with non-numeric values are skipped).
        """
        values: list[float] = []
        for obs in observations:
            val = obs.get(key)
            if isinstance(val, (int, float)) and not (
                isinstance(val, float) and (math.isnan(val) or math.isinf(val))
            ):
                values.append(float(val))
        return values

    @staticmethod
    def _variance(values: list[float]) -> float:
        """Compute sample variance (pure Python).

        Args:
            values: List of numeric values (at least 2 elements).

        Returns:
            Sample variance. Returns 0 for fewer than 2 values.
        """
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        return sum((v - mean) ** 2 for v in values) / (n - 1)

    @staticmethod
    def _categorical_variance_importance(
        all_obs: list[dict[str, Any]],
        top_obs: list[dict[str, Any]],
        param_name: str,
    ) -> float:
        """Compute importance for a categorical parameter.

        Measures how concentrated the top observations are in
        specific categories compared to the overall distribution.
        Uses Kullback-Leibler divergence approximation.

        Args:
            all_obs: All observations.
            top_obs: Top-performing observations.
            param_name: Parameter name.

        Returns:
            Importance score (higher = more concentrated in top).
        """
        # Count category frequencies
        all_counts: dict[str, int] = {}
        for obs in all_obs:
            val = obs.get(param_name)
            if val is not None:
                key = str(val)
                all_counts[key] = all_counts.get(key, 0) + 1

        top_counts: dict[str, int] = {}
        for obs in top_obs:
            val = obs.get(param_name)
            if val is not None:
                key = str(val)
                top_counts[key] = top_counts.get(key, 0) + 1

        n_all = sum(all_counts.values())
        n_top = sum(top_counts.values())

        if n_all == 0 or n_top == 0:
            return 0.0

        # KL divergence: sum p_top * log(p_top / p_all)
        kl_div = 0.0
        for cat, count_top in top_counts.items():
            p_top = count_top / n_top
            count_all = all_counts.get(cat, 0)
            p_all = count_all / n_all if count_all > 0 else 1e-10
            if p_top > 0:
                kl_div += p_top * math.log(p_top / p_all)

        return max(0.0, kl_div)

    @staticmethod
    def _categorical_f_statistic(
        top_obs: list[dict[str, Any]],
        bottom_obs: list[dict[str, Any]],
        param_name: str,
    ) -> float:
        """Compute a chi-squared-like statistic for categorical parameters.

        Measures how different the category distributions are between
        top and bottom groups.

        Args:
            top_obs: Top-performing observations.
            bottom_obs: Bottom-performing observations.
            param_name: Parameter name.

        Returns:
            Chi-squared-like statistic.
        """
        top_counts: dict[str, int] = {}
        for obs in top_obs:
            val = obs.get(param_name)
            if val is not None:
                key = str(val)
                top_counts[key] = top_counts.get(key, 0) + 1

        bottom_counts: dict[str, int] = {}
        for obs in bottom_obs:
            val = obs.get(param_name)
            if val is not None:
                key = str(val)
                bottom_counts[key] = bottom_counts.get(key, 0) + 1

        n_top = sum(top_counts.values())
        n_bottom = sum(bottom_counts.values())

        if n_top == 0 or n_bottom == 0:
            return 0.0

        # All categories
        all_cats = set(top_counts.keys()) | set(bottom_counts.keys())
        n_total = n_top + n_bottom

        # Chi-squared-like statistic
        chi_sq = 0.0
        for cat in all_cats:
            o_top = top_counts.get(cat, 0)
            o_bottom = bottom_counts.get(cat, 0)
            total_cat = o_top + o_bottom

            # Expected counts under independence
            e_top = n_top * total_cat / n_total
            e_bottom = n_bottom * total_cat / n_total

            if e_top > 0:
                chi_sq += (o_top - e_top) ** 2 / e_top
            if e_bottom > 0:
                chi_sq += (o_bottom - e_bottom) ** 2 / e_bottom

        return chi_sq

    def to_dict(self) -> dict[str, Any]:
        """Serialize the analyzer configuration.

        Returns:
            Dictionary representation of the analyzer settings.
        """
        return {
            "method": self._method,
        }
