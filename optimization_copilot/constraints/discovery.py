"""Constraint discovery from historical optimization data.

Automatically discovers implicit constraints by detecting parameter
thresholds where failure rate spikes.  Uses only stdlib + math.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)


# ── Data classes ──────────────────────────────────────


@dataclass
class DiscoveredConstraint:
    """A single constraint discovered from historical data."""

    constraint_type: str  # "threshold", "interaction", "range"
    parameters: list[str]  # which parameters are involved
    condition: str  # human-readable, e.g. "x1 > 0.8 -> 75% failure rate"
    threshold_value: float | None  # for single-param thresholds
    failure_rate_above: float  # failure rate when condition is met
    failure_rate_below: float  # failure rate when condition is NOT met
    confidence: float  # 0-1
    n_supporting: int  # observations supporting this constraint


@dataclass
class ConstraintReport:
    """Summary of all discovered constraints for a campaign snapshot."""

    constraints: list[DiscoveredConstraint]
    n_observations_analyzed: int
    coverage: float  # fraction of failures explained by discovered constraints


# ── Discovery engine ──────────────────────────────────


class ConstraintDiscoverer:
    """Discover implicit constraints from optimization history."""

    def __init__(
        self,
        min_support: int = 3,
        min_confidence: float = 0.6,
        n_splits: int = 10,
    ) -> None:
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.n_splits = n_splits

    # ── public API ────────────────────────────────────

    def discover(self, snapshot: CampaignSnapshot) -> ConstraintReport:
        """Run all discovery strategies and return a consolidated report."""
        constraints: list[DiscoveredConstraint] = []
        constraints.extend(self._discover_univariate_thresholds(snapshot))
        constraints.extend(self._discover_interaction_thresholds(snapshot))
        coverage = self._compute_coverage(constraints, snapshot)
        return ConstraintReport(
            constraints=constraints,
            n_observations_analyzed=snapshot.n_observations,
            coverage=coverage,
        )

    # ── univariate thresholds ─────────────────────────

    def _discover_univariate_thresholds(
        self, snapshot: CampaignSnapshot
    ) -> list[DiscoveredConstraint]:
        """For each continuous parameter, find the split that maximises
        the failure-rate gap between the two sides."""
        results: list[DiscoveredConstraint] = []
        obs = snapshot.observations
        if not obs:
            return results

        for spec in snapshot.parameter_specs:
            if spec.type in (VariableType.CATEGORICAL, VariableType.MIXED):
                continue
            if spec.lower is None or spec.upper is None:
                continue

            best = self._best_split_for_param(spec, obs)
            if best is not None:
                results.append(best)

        return results

    def _best_split_for_param(
        self, spec: ParameterSpec, obs: list[Observation]
    ) -> DiscoveredConstraint | None:
        lo, hi = spec.lower, spec.upper
        if lo is None or hi is None or hi <= lo:
            return None

        step = (hi - lo) / self.n_splits
        best_diff = 0.0
        best_constraint: DiscoveredConstraint | None = None

        for i in range(1, self.n_splits):
            split = lo + i * step
            above: list[Observation] = []
            below: list[Observation] = []
            for o in obs:
                val = o.parameters.get(spec.name)
                if val is None:
                    continue
                if val > split:
                    above.append(o)
                else:
                    below.append(o)

            if not above or not below:
                continue

            fr_above = sum(1 for o in above if o.is_failure) / len(above)
            fr_below = sum(1 for o in below if o.is_failure) / len(below)
            diff = abs(fr_above - fr_below)

            if diff <= 0.3:
                continue

            # pick the "high-failure" side
            if fr_above >= fr_below:
                n_supp = sum(1 for o in above if o.is_failure)
                condition = (
                    f"{spec.name} > {split:.4g} -> "
                    f"{fr_above * 100:.0f}% failure rate"
                )
                threshold = split
                rate_above = fr_above
                rate_below = fr_below
            else:
                n_supp = sum(1 for o in below if o.is_failure)
                condition = (
                    f"{spec.name} <= {split:.4g} -> "
                    f"{fr_below * 100:.0f}% failure rate"
                )
                threshold = split
                rate_above = fr_above
                rate_below = fr_below

            if n_supp < self.min_support:
                continue

            confidence = self._compute_confidence(diff, n_supp, len(obs))
            if confidence < self.min_confidence:
                continue

            if diff > best_diff:
                best_diff = diff
                best_constraint = DiscoveredConstraint(
                    constraint_type="threshold",
                    parameters=[spec.name],
                    condition=condition,
                    threshold_value=threshold,
                    failure_rate_above=rate_above,
                    failure_rate_below=rate_below,
                    confidence=confidence,
                    n_supporting=n_supp,
                )

        return best_constraint

    # ── interaction thresholds ────────────────────────

    def _discover_interaction_thresholds(
        self, snapshot: CampaignSnapshot
    ) -> list[DiscoveredConstraint]:
        """For each pair of continuous parameters, check whether any
        quadrant (median split) has a significantly higher failure rate."""
        results: list[DiscoveredConstraint] = []
        obs = snapshot.observations
        if len(obs) < 4:
            return results

        continuous = [
            s
            for s in snapshot.parameter_specs
            if s.type not in (VariableType.CATEGORICAL, VariableType.MIXED)
            and s.lower is not None
            and s.upper is not None
        ]

        for s1, s2 in combinations(continuous, 2):
            found = self._check_interaction(s1, s2, obs)
            if found is not None:
                results.append(found)

        return results

    def _check_interaction(
        self,
        s1: ParameterSpec,
        s2: ParameterSpec,
        obs: list[Observation],
    ) -> DiscoveredConstraint | None:
        vals1 = [o.parameters.get(s1.name) for o in obs]
        vals2 = [o.parameters.get(s2.name) for o in obs]
        # skip if any None
        if None in vals1 or None in vals2:
            return None

        med1 = _median(vals1)  # type: ignore[arg-type]
        med2 = _median(vals2)  # type: ignore[arg-type]

        # partition observations into four quadrants
        quadrants: dict[str, list[Observation]] = {
            "HH": [],
            "HL": [],
            "LH": [],
            "LL": [],
        }
        for o in obs:
            v1 = o.parameters[s1.name]
            v2 = o.parameters[s2.name]
            q1 = "H" if v1 > med1 else "L"
            q2 = "H" if v2 > med2 else "L"
            quadrants[q1 + q2].append(o)

        overall_fr = sum(1 for o in obs if o.is_failure) / len(obs)

        best_diff = 0.0
        best_constraint: DiscoveredConstraint | None = None

        for label, q_obs in quadrants.items():
            if not q_obs:
                continue
            q_fr = sum(1 for o in q_obs if o.is_failure) / len(q_obs)
            rest = [o for o in obs if o not in q_obs]
            rest_fr = (sum(1 for o in rest if o.is_failure) / len(rest)) if rest else 0.0
            diff = q_fr - rest_fr  # positive means this quadrant has more failures

            if diff <= 0.3:
                continue

            n_supp = sum(1 for o in q_obs if o.is_failure)
            if n_supp < self.min_support:
                continue

            confidence = self._compute_confidence(diff, n_supp, len(obs))
            if confidence < self.min_confidence:
                continue

            cond_parts = []
            if label[0] == "H":
                cond_parts.append(f"{s1.name} > {med1:.4g}")
            else:
                cond_parts.append(f"{s1.name} <= {med1:.4g}")
            if label[1] == "H":
                cond_parts.append(f"{s2.name} > {med2:.4g}")
            else:
                cond_parts.append(f"{s2.name} <= {med2:.4g}")

            condition = (
                " AND ".join(cond_parts)
                + f" -> {q_fr * 100:.0f}% failure rate"
            )

            if diff > best_diff:
                best_diff = diff
                best_constraint = DiscoveredConstraint(
                    constraint_type="interaction",
                    parameters=[s1.name, s2.name],
                    condition=condition,
                    threshold_value=None,
                    failure_rate_above=q_fr,
                    failure_rate_below=rest_fr,
                    confidence=confidence,
                    n_supporting=n_supp,
                )

        return best_constraint

    # ── coverage ──────────────────────────────────────

    def _compute_coverage(
        self,
        constraints: list[DiscoveredConstraint],
        snapshot: CampaignSnapshot,
    ) -> float:
        """Fraction of failures explained by at least one constraint."""
        failures = [o for o in snapshot.observations if o.is_failure]
        if not failures:
            return 1.0  # no failures to explain
        if not constraints:
            return 0.0

        explained = 0
        for o in failures:
            if self._is_explained(o, constraints, snapshot):
                explained += 1

        return explained / len(failures)

    @staticmethod
    def _is_explained(
        obs: Observation,
        constraints: list[DiscoveredConstraint],
        snapshot: CampaignSnapshot,
    ) -> bool:
        """Check whether an observation falls in a region flagged
        by at least one constraint."""
        for c in constraints:
            if c.constraint_type == "threshold":
                pname = c.parameters[0]
                val = obs.parameters.get(pname)
                if val is None or c.threshold_value is None:
                    continue
                # the high-failure side is "above" when
                # failure_rate_above > failure_rate_below, else "below"
                if c.failure_rate_above >= c.failure_rate_below:
                    if val > c.threshold_value:
                        return True
                else:
                    if val <= c.threshold_value:
                        return True

            elif c.constraint_type == "interaction":
                # parse the condition to figure out the quadrant
                # We stored parameters=[s1, s2] and the condition string
                # contains the direction indicators.  Instead of parsing
                # the string, re-derive from stored rates: the flagged
                # quadrant is encoded in the condition.
                if _obs_matches_interaction(obs, c, snapshot):
                    return True

        return False

    # ── helpers ───────────────────────────────────────

    @staticmethod
    def _compute_confidence(
        diff: float, n_supporting: int, n_total: int
    ) -> float:
        """Confidence score combining effect size and sample support.

        - diff:          absolute failure-rate gap (0-1)
        - n_supporting:  how many failure observations in the flagged region
        - n_total:       total observations
        """
        # effect component: how big the gap is (max 1.0)
        effect = min(diff / 1.0, 1.0)
        # support component: asymptotic ramp  n/(n+k)
        support = n_supporting / (n_supporting + 5)
        # fraction component: how much of the data we're looking at
        fraction = min(n_supporting / max(n_total, 1), 1.0)
        # weighted combination
        return 0.5 * effect + 0.35 * support + 0.15 * fraction


# ── module-level helpers ──────────────────────────────


def _median(values: list[float]) -> float:
    """Compute median without numpy."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


# ── Cross-campaign constraint transfer ────────────────


class ConstraintMigrator:
    """Transfer discovered constraints from a source campaign to a new one.

    Validates parameter compatibility: only constraints whose parameter names
    exist in the target campaign (and have overlapping ranges) are kept.
    """

    def migrate(
        self,
        source_report: ConstraintReport,
        target_snapshot: CampaignSnapshot,
        *,
        min_confidence: float = 0.5,
    ) -> ConstraintReport:
        """Migrate compatible constraints from *source_report* to *target_snapshot*.

        Parameters
        ----------
        source_report : ConstraintReport
            Constraints discovered in a previous campaign.
        target_snapshot : CampaignSnapshot
            The new campaign to receive constraints.
        min_confidence : float
            Only migrate constraints above this confidence.

        Returns
        -------
        ConstraintReport with applicable subset of source constraints.
        """
        target_params = {s.name: s for s in target_snapshot.parameter_specs}
        migrated: list[DiscoveredConstraint] = []

        for c in source_report.constraints:
            if c.confidence < min_confidence:
                continue
            if not all(p in target_params for p in c.parameters):
                continue
            # For threshold constraints, verify range overlap
            if c.constraint_type == "threshold" and c.threshold_value is not None:
                spec = target_params[c.parameters[0]]
                if spec.lower is not None and c.threshold_value < spec.lower:
                    continue
                if spec.upper is not None and c.threshold_value > spec.upper:
                    continue
            migrated.append(c)

        return ConstraintReport(
            constraints=migrated,
            n_observations_analyzed=source_report.n_observations_analyzed,
            coverage=source_report.coverage,
        )


# ── Constraint tightening/relaxation tracking ────────


@dataclass
class ConstraintDelta:
    """Change in a single constraint across two snapshots."""

    parameter: str
    old_threshold: float | None
    new_threshold: float | None
    old_failure_rate: float
    new_failure_rate: float
    direction: str  # "tightened", "relaxed", "unchanged", "new", "removed"


@dataclass
class ConstraintEvolution:
    """Summary of how constraints evolved between two time points."""

    deltas: list[ConstraintDelta]
    n_tightened: int
    n_relaxed: int
    n_new: int
    n_removed: int


class ConstraintTracker:
    """Track how constraints evolve across successive snapshots."""

    def compare(
        self,
        old_report: ConstraintReport,
        new_report: ConstraintReport,
    ) -> ConstraintEvolution:
        """Compare two constraint reports and produce evolution summary."""
        old_by_params = {tuple(c.parameters): c for c in old_report.constraints}
        new_by_params = {tuple(c.parameters): c for c in new_report.constraints}

        deltas: list[ConstraintDelta] = []
        n_tightened = 0
        n_relaxed = 0
        n_new = 0
        n_removed = 0

        all_keys = set(old_by_params) | set(new_by_params)
        for key in sorted(all_keys):
            old_c = old_by_params.get(key)
            new_c = new_by_params.get(key)

            if old_c is None and new_c is not None:
                deltas.append(ConstraintDelta(
                    parameter=",".join(key),
                    old_threshold=None,
                    new_threshold=new_c.threshold_value,
                    old_failure_rate=0.0,
                    new_failure_rate=new_c.failure_rate_above,
                    direction="new",
                ))
                n_new += 1
            elif old_c is not None and new_c is None:
                deltas.append(ConstraintDelta(
                    parameter=",".join(key),
                    old_threshold=old_c.threshold_value,
                    new_threshold=None,
                    old_failure_rate=old_c.failure_rate_above,
                    new_failure_rate=0.0,
                    direction="removed",
                ))
                n_removed += 1
            elif old_c is not None and new_c is not None:
                direction = "unchanged"
                if (
                    old_c.threshold_value is not None
                    and new_c.threshold_value is not None
                ):
                    # Tightened = threshold moved toward the safe region
                    # (higher failure_rate_above with same or lower threshold)
                    if new_c.failure_rate_above > old_c.failure_rate_above + 0.05:
                        direction = "tightened"
                        n_tightened += 1
                    elif new_c.failure_rate_above < old_c.failure_rate_above - 0.05:
                        direction = "relaxed"
                        n_relaxed += 1
                deltas.append(ConstraintDelta(
                    parameter=",".join(key),
                    old_threshold=old_c.threshold_value,
                    new_threshold=new_c.threshold_value,
                    old_failure_rate=old_c.failure_rate_above,
                    new_failure_rate=new_c.failure_rate_above,
                    direction=direction,
                ))

        return ConstraintEvolution(
            deltas=deltas,
            n_tightened=n_tightened,
            n_relaxed=n_relaxed,
            n_new=n_new,
            n_removed=n_removed,
        )


def _obs_matches_interaction(
    obs: Observation,
    constraint: DiscoveredConstraint,
    snapshot: CampaignSnapshot,
) -> bool:
    """Check if an observation falls in the flagged quadrant of an
    interaction constraint by re-deriving medians."""
    if len(constraint.parameters) != 2:
        return False
    p1, p2 = constraint.parameters
    v1 = obs.parameters.get(p1)
    v2 = obs.parameters.get(p2)
    if v1 is None or v2 is None:
        return False

    # re-derive medians from snapshot
    all_v1 = [o.parameters.get(p1) for o in snapshot.observations]
    all_v2 = [o.parameters.get(p2) for o in snapshot.observations]
    if None in all_v1 or None in all_v2:
        return False
    med1 = _median(all_v1)  # type: ignore[arg-type]
    med2 = _median(all_v2)  # type: ignore[arg-type]

    # Determine which quadrant the constraint flags by inspecting the condition
    cond = constraint.condition
    h1 = f"{p1} >" in cond and f"{p1} <=" not in cond
    h2 = f"{p2} >" in cond and f"{p2} <=" not in cond

    matches1 = (v1 > med1) if h1 else (v1 <= med1)
    matches2 = (v2 > med2) if h2 else (v2 <= med2)
    return matches1 and matches2
