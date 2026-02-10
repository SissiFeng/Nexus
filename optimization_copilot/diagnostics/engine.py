"""Diagnostic Signal Engine: computes 14 health signals from campaign history."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from optimization_copilot.core.models import CampaignSnapshot, Observation


# ---------------------------------------------------------------------------
# DiagnosticsVector
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticsVector:
    """14-signal diagnostic summary of an optimization campaign."""

    convergence_trend: float = 0.0
    improvement_velocity: float = 0.0
    variance_contraction: float = 1.0
    noise_estimate: float = 0.0
    failure_rate: float = 0.0
    failure_clustering: float = 0.0
    feasibility_shrinkage: float = 0.0
    parameter_drift: float = 0.0
    model_uncertainty: float = 0.0
    exploration_coverage: float = 0.0
    kpi_plateau_length: int = 0
    best_kpi_value: float = 0.0
    data_efficiency: float = 0.0
    constraint_violation_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticsVector:
        return cls(**data)


# ---------------------------------------------------------------------------
# Helpers (pure functions, no external deps)
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    """Arithmetic mean; returns 0.0 for empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    """Population variance; returns 0.0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def _std(values: list[float]) -> float:
    return math.sqrt(_variance(values))


def _linreg_slope(xs: list[float], ys: list[float]) -> float:
    """Ordinary least-squares slope.  Returns 0.0 when undefined."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0.0:
        return 0.0
    return num / den


def _primary_kpi(obs: Observation, objective_names: list[str]) -> float | None:
    """Extract the primary (first) KPI value from an observation."""
    if not objective_names:
        return None
    name = objective_names[0]
    return obs.kpi_values.get(name)


def _is_maximizing(snapshot: CampaignSnapshot) -> bool:
    """Whether the primary objective is being maximized."""
    if not snapshot.objective_directions:
        return False
    return snapshot.objective_directions[0].lower().startswith("max")


def _best_so_far(values: list[float], maximize: bool) -> list[float]:
    """Cumulative best-so-far series."""
    if not values:
        return []
    result: list[float] = []
    best = values[0]
    for v in values:
        if maximize:
            best = max(best, v)
        else:
            best = min(best, v)
        result.append(best)
    return result


def _window_size(n: int, fraction: float = 0.25, minimum: int = 1) -> int:
    """Compute a rolling-window size as a fraction of n, with a minimum."""
    return max(int(n * fraction), minimum)


# ---------------------------------------------------------------------------
# DiagnosticEngine
# ---------------------------------------------------------------------------

class DiagnosticEngine:
    """Computes 14 diagnostic signals from a CampaignSnapshot.

    All computations use only the Python standard library and basic math.
    Edge cases (empty history, single point, all failures) are handled
    gracefully by returning safe defaults.

    Parameters
    ----------
    window_fraction : float
        Fraction of observations used as the "recent" window (default 0.25).
    improvement_threshold : float
        Minimum relative change to count as "meaningful improvement"
        for plateau detection (default 0.01 = 1 %).
    n_bins : int
        Number of bins per parameter dimension for exploration coverage
        (default 10).
    """

    def __init__(
        self,
        window_fraction: float = 0.25,
        improvement_threshold: float = 0.01,
        n_bins: int = 10,
    ) -> None:
        self.window_fraction = window_fraction
        self.improvement_threshold = improvement_threshold
        self.n_bins = n_bins

    # -- public API ---------------------------------------------------------

    def compute(self, snapshot: CampaignSnapshot) -> DiagnosticsVector:
        """Compute all 14 diagnostic signals from *snapshot*."""
        successful = snapshot.successful_observations
        all_obs = snapshot.observations
        maximize = _is_maximizing(snapshot)
        obj_names = snapshot.objective_names

        # Extract primary KPI series (successful only)
        kpi_values: list[float] = []
        kpi_iterations: list[float] = []
        for obs in successful:
            v = _primary_kpi(obs, obj_names)
            if v is not None:
                kpi_values.append(v)
                kpi_iterations.append(float(obs.iteration))

        n_kpi = len(kpi_values)
        n_all = len(all_obs)
        kpi_window = _window_size(n_kpi, self.window_fraction)
        obs_window = _window_size(n_all, self.window_fraction)

        return DiagnosticsVector(
            convergence_trend=self._convergence_trend(
                kpi_iterations, kpi_values, maximize
            ),
            improvement_velocity=self._improvement_velocity(
                kpi_values, kpi_window, maximize
            ),
            variance_contraction=self._variance_contraction(kpi_values, kpi_window),
            noise_estimate=self._noise_estimate(kpi_values, kpi_window),
            failure_rate=self._failure_rate(snapshot),
            failure_clustering=self._failure_clustering(all_obs, obs_window),
            feasibility_shrinkage=self._feasibility_shrinkage(all_obs, obs_window),
            parameter_drift=self._parameter_drift(
                successful, kpi_values, kpi_window, maximize, obj_names
            ),
            model_uncertainty=self._model_uncertainty(kpi_values, kpi_window),
            exploration_coverage=self._exploration_coverage(successful, snapshot),
            kpi_plateau_length=self._kpi_plateau_length(kpi_values, maximize),
            best_kpi_value=self._best_kpi_value(kpi_values, maximize),
            data_efficiency=self._data_efficiency(kpi_values, n_all, maximize),
            constraint_violation_rate=self._constraint_violation_rate(snapshot),
        )

    # -- individual signals -------------------------------------------------

    def _convergence_trend(
        self,
        iterations: list[float],
        kpi_values: list[float],
        maximize: bool,
    ) -> float:
        """Slope of best-so-far KPI over iterations (normalized to [-1, 1])."""
        if len(kpi_values) < 2:
            return 0.0
        bsf = _best_so_far(kpi_values, maximize)
        slope = _linreg_slope(iterations, bsf)
        # Normalize by KPI range so result is scale-independent
        kpi_range = max(kpi_values) - min(kpi_values)
        if kpi_range == 0.0:
            return 0.0
        iter_range = iterations[-1] - iterations[0]
        if iter_range == 0.0:
            return 0.0
        # normalized slope: how much of the kpi_range is gained per unit
        # of iteration range
        normalized = slope * iter_range / kpi_range
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, normalized))

    def _improvement_velocity(
        self,
        kpi_values: list[float],
        window: int,
        maximize: bool,
    ) -> float:
        """Rate of improvement in the recent window vs the previous window.

        Returns a value in [-1, 1] where positive means the recent window
        is improving faster.
        """
        n = len(kpi_values)
        if n < 2:
            return 0.0
        # Split into two halves relative to the window
        recent = kpi_values[-window:]
        previous = kpi_values[:-window] if n > window else kpi_values[:1]

        if maximize:
            recent_improvement = max(recent) - min(recent)
            prev_improvement = max(previous) - min(previous) if len(previous) > 1 else 0.0
        else:
            recent_improvement = min(recent) - max(recent)  # negative if improving
            prev_improvement = (
                (min(previous) - max(previous)) if len(previous) > 1 else 0.0
            )

        # For minimization, improvement is negative (lower is better),
        # so we negate to make "more improvement" positive.
        if not maximize:
            recent_improvement = -recent_improvement
            prev_improvement = -prev_improvement

        # Normalize
        total = recent_improvement + prev_improvement
        if total == 0.0:
            return 0.0
        velocity = (recent_improvement - prev_improvement) / total
        return max(-1.0, min(1.0, velocity))

    def _variance_contraction(
        self, kpi_values: list[float], window: int
    ) -> float:
        """Ratio of KPI variance in recent window vs full history.

        Values < 1 indicate the variance is contracting (converging).
        """
        if len(kpi_values) < 2:
            return 1.0
        full_var = _variance(kpi_values)
        if full_var == 0.0:
            return 1.0
        recent_var = _variance(kpi_values[-window:])
        return recent_var / full_var

    def _noise_estimate(self, kpi_values: list[float], window: int) -> float:
        """Coefficient of variation (std / |mean|) in the recent window."""
        if len(kpi_values) < 2:
            return 0.0
        recent = kpi_values[-window:]
        m = _mean(recent)
        if m == 0.0:
            return 0.0
        return _std(recent) / abs(m)

    @staticmethod
    def _failure_rate(snapshot: CampaignSnapshot) -> float:
        return snapshot.failure_rate

    @staticmethod
    def _failure_clustering(
        all_obs: list[Observation], window: int
    ) -> float:
        """Ratio of recent failure rate to overall failure rate.

        >1 means failures are clustered in recent observations.
        Returns 0.0 when there are no failures.
        """
        n = len(all_obs)
        if n == 0:
            return 0.0
        total_failures = sum(1 for o in all_obs if o.is_failure)
        if total_failures == 0:
            return 0.0
        overall_rate = total_failures / n

        recent = all_obs[-window:]
        recent_failures = sum(1 for o in recent if o.is_failure)
        recent_rate = recent_failures / len(recent)

        if overall_rate == 0.0:
            return 0.0
        return recent_rate / overall_rate

    @staticmethod
    def _feasibility_shrinkage(
        all_obs: list[Observation], window: int
    ) -> float:
        """Change in feasibility rate between early and recent windows.

        Negative means the feasible region is shrinking.
        """
        n = len(all_obs)
        if n < 2:
            return 0.0
        early = all_obs[:window]
        recent = all_obs[-window:]
        early_feas = sum(1 for o in early if not o.is_failure) / len(early)
        recent_feas = sum(1 for o in recent if not o.is_failure) / len(recent)
        return recent_feas - early_feas

    @staticmethod
    def _parameter_drift(
        successful: list[Observation],
        kpi_values: list[float],
        window: int,
        maximize: bool,
        obj_names: list[str],
    ) -> float:
        """Average absolute change in the best parameters over recent iterations.

        We track the cumulative-best observation and measure how much its
        parameters shift in the recent window.
        """
        if len(successful) < 2 or len(kpi_values) < 2:
            return 0.0

        # Build cumulative best observation sequence
        best_obs_sequence: list[Observation] = []
        best_val: float | None = None
        best_obs: Observation | None = None
        for obs in successful:
            v = _primary_kpi(obs, obj_names)
            if v is None:
                continue
            if best_val is None:
                best_val = v
                best_obs = obs
            elif (maximize and v > best_val) or (not maximize and v < best_val):
                best_val = v
                best_obs = obs
            best_obs_sequence.append(best_obs)  # type: ignore[arg-type]

        if len(best_obs_sequence) < 2:
            return 0.0

        # Compute parameter drift in the recent window of best-obs changes
        recent_bests = best_obs_sequence[-window:]
        if len(recent_bests) < 2:
            return 0.0

        drifts: list[float] = []
        for i in range(1, len(recent_bests)):
            prev_params = recent_bests[i - 1].parameters
            curr_params = recent_bests[i].parameters
            for key in prev_params:
                pv = prev_params.get(key)
                cv = curr_params.get(key)
                if isinstance(pv, (int, float)) and isinstance(cv, (int, float)):
                    drifts.append(abs(cv - pv))

        return _mean(drifts)

    @staticmethod
    def _model_uncertainty(kpi_values: list[float], window: int) -> float:
        """Spread of KPI values in the recent window (std / |mean|).

        This is a proxy for model uncertainty when we don't have an actual
        surrogate model.
        """
        if len(kpi_values) < 2:
            return 0.0
        recent = kpi_values[-window:]
        m = _mean(recent)
        if m == 0.0:
            return 0.0
        return _std(recent) / abs(m)

    def _exploration_coverage(
        self,
        successful: list[Observation],
        snapshot: CampaignSnapshot,
    ) -> float:
        """Fraction of parameter space explored (unique bin combinations)."""
        specs = snapshot.parameter_specs
        if not specs or not successful:
            return 0.0

        n_bins = self.n_bins

        # For each numeric parameter, determine bins
        def _bin_value(value: Any, lower: float | None, upper: float | None) -> int:
            if not isinstance(value, (int, float)):
                return hash(value) % n_bins
            lo = lower if lower is not None else 0.0
            hi = upper if upper is not None else 1.0
            if hi == lo:
                return 0
            frac = (float(value) - lo) / (hi - lo)
            frac = max(0.0, min(1.0, frac))
            b = int(frac * n_bins)
            return min(b, n_bins - 1)

        visited: set[tuple[int, ...]] = set()
        for obs in successful:
            cell: list[int] = []
            for spec in specs:
                val = obs.parameters.get(spec.name)
                if val is None:
                    cell.append(-1)
                else:
                    cell.append(_bin_value(val, spec.lower, spec.upper))
            visited.add(tuple(cell))

        total_cells = n_bins ** len(specs)
        if total_cells == 0:
            return 0.0
        return len(visited) / total_cells

    def _kpi_plateau_length(
        self, kpi_values: list[float], maximize: bool
    ) -> int:
        """Number of observations since the last meaningful improvement."""
        if len(kpi_values) < 2:
            return 0

        bsf = _best_so_far(kpi_values, maximize)
        kpi_range = max(kpi_values) - min(kpi_values)
        if kpi_range == 0.0:
            # All values identical => plateau for entire history
            return len(kpi_values) - 1

        threshold = self.improvement_threshold * kpi_range
        last_improvement_idx = 0
        for i in range(1, len(bsf)):
            if abs(bsf[i] - bsf[i - 1]) > threshold:
                last_improvement_idx = i

        return len(kpi_values) - 1 - last_improvement_idx

    @staticmethod
    def _best_kpi_value(kpi_values: list[float], maximize: bool) -> float:
        """Current best observed KPI value."""
        if not kpi_values:
            return 0.0
        return max(kpi_values) if maximize else min(kpi_values)

    @staticmethod
    def _data_efficiency(
        kpi_values: list[float], n_observations: int, maximize: bool
    ) -> float:
        """Improvement per observation: total improvement / n_observations."""
        if n_observations == 0 or len(kpi_values) < 2:
            return 0.0
        if maximize:
            total_improvement = max(kpi_values) - kpi_values[0]
        else:
            total_improvement = kpi_values[0] - min(kpi_values)
        return total_improvement / n_observations

    @staticmethod
    def _constraint_violation_rate(snapshot: CampaignSnapshot) -> float:
        """Fraction of observations that violated constraints.

        Uses ``qc_passed=False`` as a proxy for constraint violation.
        If the snapshot has explicit constraints, observations whose KPI
        values fall outside constraint bounds are also counted.
        """
        if not snapshot.observations:
            return 0.0

        violations = 0
        for obs in snapshot.observations:
            if not obs.qc_passed:
                violations += 1
                continue
            # Check explicit constraints
            for constraint in snapshot.constraints:
                target = constraint.get("target")
                lo = constraint.get("lower")
                hi = constraint.get("upper")
                if target and target in obs.kpi_values:
                    val = obs.kpi_values[target]
                    if lo is not None and val < lo:
                        violations += 1
                        break
                    if hi is not None and val > hi:
                        violations += 1
                        break

        return violations / len(snapshot.observations)
