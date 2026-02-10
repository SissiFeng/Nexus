"""Sensitivity analyzer for optimization decisions."""

from __future__ import annotations

import math

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)

from .models import DecisionStability, ParameterSensitivity, SensitivityReport


# ── Helper functions ──────────────────────────────────


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-15 or dy < 1e-15:
        return 0.0
    return num / (dx * dy)


def _primary_kpi(obs: Observation, objective_names: list[str]) -> float | None:
    if not objective_names:
        return None
    return obs.kpi_values.get(objective_names[0])


def _is_maximizing(snapshot: CampaignSnapshot) -> bool:
    if not snapshot.objective_directions:
        return True
    return snapshot.objective_directions[0] == "maximize"


def _normalized_distance(
    obs_a: Observation,
    obs_b: Observation,
    specs: list[ParameterSpec],
) -> float:
    total = 0.0
    count = 0
    for spec in specs:
        if spec.type == VariableType.CATEGORICAL:
            continue
        if spec.upper is None or spec.lower is None or spec.upper == spec.lower:
            continue
        va = obs_a.parameters.get(spec.name, 0.0)
        vb = obs_b.parameters.get(spec.name, 0.0)
        total += ((float(va) - float(vb)) / (spec.upper - spec.lower)) ** 2
        count += 1
    if count == 0:
        return 0.0
    return math.sqrt(total / count)


# ── Analyzer ──────────────────────────────────────────


class SensitivityAnalyzer:
    """Analyse how sensitive the current best decision is to parameter changes."""

    def __init__(
        self,
        top_k: int = 5,
        n_neighbors: int = 5,
        perturbation_fraction: float = 0.05,
    ) -> None:
        self._top_k = top_k
        self._n_neighbors = n_neighbors
        self._perturbation_fraction = perturbation_fraction

    # ── public API ────────────────────────────────────

    def analyze(self, snapshot: CampaignSnapshot) -> SensitivityReport:
        """Run a full sensitivity analysis on *snapshot*."""
        successful = snapshot.successful_observations

        if len(successful) < 3:
            return SensitivityReport(
                parameter_sensitivities=[],
                decision_stability=DecisionStability(
                    top_k=self._top_k,
                    stable_count=0,
                    stability_score=1.0,
                    margin_to_next=0.0,
                    margin_relative=0.0,
                    swapped_pairs=[],
                    evidence={},
                ),
                robustness_score=0.0,
                most_sensitive_parameter="",
                least_sensitive_parameter="",
                recommendations=["Insufficient data for sensitivity analysis."],
                metadata={},
            )

        parameter_sensitivities = self._compute_parameter_sensitivities(
            successful, snapshot
        )
        decision_stability = self._compute_decision_stability(successful, snapshot)
        robustness_score = self._compute_robustness_score(
            parameter_sensitivities, decision_stability
        )

        if parameter_sensitivities:
            most_sensitive = next(
                (s.parameter_name for s in parameter_sensitivities if s.rank == 1), ""
            )
            least_sensitive = parameter_sensitivities[-1].parameter_name
        else:
            most_sensitive = ""
            least_sensitive = ""

        recommendations = self._generate_recommendations(
            parameter_sensitivities, decision_stability, robustness_score
        )

        return SensitivityReport(
            parameter_sensitivities=parameter_sensitivities,
            decision_stability=decision_stability,
            robustness_score=robustness_score,
            most_sensitive_parameter=most_sensitive,
            least_sensitive_parameter=least_sensitive,
            recommendations=recommendations,
            metadata={},
        )

    # ── parameter sensitivities ───────────────────────

    def _compute_parameter_sensitivities(
        self,
        successful: list[Observation],
        snapshot: CampaignSnapshot,
    ) -> list[ParameterSensitivity]:
        results: list[ParameterSensitivity] = []
        maximizing = _is_maximizing(snapshot)

        for param in snapshot.parameter_specs:
            if param.type == VariableType.CATEGORICAL:
                continue

            # Extract paired (param_value, kpi_value) where both exist.
            pairs: list[tuple[float, float]] = []
            for obs in successful:
                if param.name not in obs.parameters:
                    continue
                kval = _primary_kpi(obs, snapshot.objective_names)
                if kval is None:
                    continue
                pairs.append((float(obs.parameters[param.name]), kval))

            if len(pairs) < 2:
                results.append(
                    ParameterSensitivity(
                        parameter_name=param.name,
                        sensitivity_score=0.0,
                        correlation=0.0,
                        local_gradient=0.0,
                        rank=0,
                    )
                )
                continue

            pvals = [p for p, _ in pairs]
            kvals = [k for _, k in pairs]

            param_range = (
                (param.upper - param.lower)
                if param.upper is not None and param.lower is not None
                else 0.0
            )
            if param_range == 0.0:
                results.append(
                    ParameterSensitivity(
                        parameter_name=param.name,
                        sensitivity_score=0.0,
                        correlation=0.0,
                        local_gradient=0.0,
                        rank=0,
                    )
                )
                continue

            correlation = _pearson_correlation(pvals, kvals)

            # Local gradient around the best observation.
            best_idx = (
                max(range(len(kvals)), key=lambda i: kvals[i])
                if maximizing
                else min(range(len(kvals)), key=lambda i: kvals[i])
            )
            best_obs = successful[
                next(
                    j
                    for j, obs in enumerate(successful)
                    if obs.parameters.get(param.name) is not None
                    and _primary_kpi(obs, snapshot.objective_names) is not None
                    and float(obs.parameters[param.name]) == pvals[best_idx]
                    and _primary_kpi(obs, snapshot.objective_names) == kvals[best_idx]
                )
            ]

            # Find nearest neighbours by normalised distance.
            distances: list[tuple[float, Observation]] = []
            for obs in successful:
                if obs is best_obs:
                    continue
                d = _normalized_distance(obs, best_obs, snapshot.parameter_specs)
                distances.append((d, obs))
            distances.sort(key=lambda t: t[0])
            neighbours = [obs for _, obs in distances[: self._n_neighbors]]

            gradients: list[float] = []
            for nb in neighbours:
                nb_pval = nb.parameters.get(param.name)
                nb_kval = _primary_kpi(nb, snapshot.objective_names)
                if nb_pval is None or nb_kval is None:
                    continue
                delta_param = float(nb_pval) - pvals[best_idx]
                if abs(delta_param) < 1e-15:
                    continue
                delta_kpi = nb_kval - kvals[best_idx]
                gradients.append(abs(delta_kpi / delta_param))

            kpi_range = max(kvals) - min(kvals)
            if gradients and kpi_range > 0:
                avg_gradient = _mean(gradients)
                local_gradient = avg_gradient * param_range / kpi_range
            else:
                local_gradient = 0.0

            sensitivity_score = 0.5 * abs(correlation) + 0.5 * min(
                1.0, max(0.0, local_gradient)
            )
            sensitivity_score = max(0.0, min(1.0, sensitivity_score))

            results.append(
                ParameterSensitivity(
                    parameter_name=param.name,
                    sensitivity_score=sensitivity_score,
                    correlation=correlation,
                    local_gradient=local_gradient,
                    rank=0,
                )
            )

        # Sort descending by sensitivity_score and assign ranks.
        results.sort(key=lambda s: s.sensitivity_score, reverse=True)
        for i, ps in enumerate(results):
            ps.rank = i + 1

        return results

    # ── decision stability ────────────────────────────

    def _compute_decision_stability(
        self,
        successful: list[Observation],
        snapshot: CampaignSnapshot,
    ) -> DecisionStability:
        kpi_name = snapshot.objective_names[0] if snapshot.objective_names else None
        if kpi_name is None:
            return DecisionStability(
                top_k=self._top_k,
                stable_count=0,
                stability_score=0.0,
                margin_to_next=0.0,
                margin_relative=0.0,
                swapped_pairs=[],
                evidence={},
            )

        maximizing = _is_maximizing(snapshot)

        # Build (kpi, index) pairs, sort best-first.
        kpi_pairs: list[tuple[float, int]] = []
        for idx, obs in enumerate(successful):
            kval = obs.kpi_values.get(kpi_name)
            if kval is not None:
                kpi_pairs.append((kval, idx))

        kpi_pairs.sort(key=lambda t: t[0], reverse=maximizing)
        kpi_sorted = [k for k, _ in kpi_pairs]

        effective_k = min(self._top_k, len(kpi_sorted))
        if effective_k < 1:
            return DecisionStability(
                top_k=self._top_k,
                stable_count=0,
                stability_score=1.0,
                margin_to_next=0.0,
                margin_relative=0.0,
                swapped_pairs=[],
                evidence={},
            )

        # Margin between the last top-K element and the next one.
        if len(kpi_sorted) > effective_k:
            margin = abs(kpi_sorted[effective_k - 1] - kpi_sorted[effective_k])
        else:
            margin = abs(kpi_sorted[0] - kpi_sorted[-1]) if len(kpi_sorted) > 1 else 0.0

        kpi_range = max(kpi_sorted) - min(kpi_sorted) if kpi_sorted else 0.0
        margin_relative = margin / kpi_range if kpi_range > 0 else 1.0

        noise_estimate = _std(kpi_sorted)

        # Swap detection: adjacent pairs within the top (effective_k + 1).
        check_range = min(effective_k + 1, len(kpi_sorted))
        swapped_pairs: list[tuple[int, int]] = []
        for i in range(check_range - 1):
            if abs(kpi_sorted[i] - kpi_sorted[i + 1]) < noise_estimate * self._perturbation_fraction:
                swapped_pairs.append((i, i + 1))

        # Count unique indices from swapped pairs that fall within top-K.
        swap_indices: set[int] = set()
        for a, b in swapped_pairs:
            if a < effective_k:
                swap_indices.add(a)
            if b < effective_k:
                swap_indices.add(b)

        stable_count = effective_k - len(swap_indices)
        stability_score = stable_count / effective_k

        return DecisionStability(
            top_k=effective_k,
            stable_count=stable_count,
            stability_score=stability_score,
            margin_to_next=margin,
            margin_relative=margin_relative,
            swapped_pairs=swapped_pairs,
            evidence={
                "noise_estimate": noise_estimate,
                "n_observations": len(successful),
            },
        )

    # ── robustness ────────────────────────────────────

    @staticmethod
    def _compute_robustness_score(
        sensitivities: list[ParameterSensitivity],
        stability: DecisionStability,
    ) -> float:
        if not sensitivities:
            avg_sens = 0.0
        else:
            avg_sens = _mean([s.sensitivity_score for s in sensitivities])
        robustness = 0.4 * (1.0 - avg_sens) + 0.6 * stability.stability_score
        return max(0.0, min(1.0, robustness))

    # ── recommendations ───────────────────────────────

    @staticmethod
    def _generate_recommendations(
        sensitivities: list[ParameterSensitivity],
        stability: DecisionStability,
        robustness: float,
    ) -> list[str]:
        recs: list[str] = []
        if robustness < 0.3:
            recs.append(
                "Decision is fragile. Consider gathering more data before committing."
            )
        if sensitivities and sensitivities[0].sensitivity_score > 0.8:
            recs.append(
                f"Parameter '{sensitivities[0].parameter_name}' dominates KPI. "
                "Focus experimental effort there."
            )
        if stability.stability_score < 0.5:
            recs.append(
                f"Top-{stability.top_k} ranking is unstable. "
                "The current best may not be reliably best."
            )
        if stability.margin_relative < 0.05:
            recs.append(
                "Very small margin between top candidates. "
                "Differences may be within noise."
            )
        if sensitivities and all(
            s.sensitivity_score < 0.2 for s in sensitivities
        ):
            recs.append(
                "KPI is insensitive to all parameters. "
                "Check if the right parameters are being varied."
            )
        return recs
