"""Decision Schema Hardening — versionable, diffable, reviewable decision rules.

Makes the MetaController's decision logic explicit as versioned rule objects
so that decision paths become auditable artefacts: diffable across runs,
reviewable in code review, and reproducible under version control.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from optimization_copilot.core.models import Phase, RiskPosture
from optimization_copilot.core.hashing import _sha256


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_signature(
    rule_id: str,
    version: str,
    triggers: frozenset[str],
    action: str,
) -> str:
    """SHA-256 (truncated to 16 hex chars) of the sorted concatenation."""
    parts = sorted([rule_id, version, str(sorted(triggers)), action])
    return _sha256("|".join(parts))


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DecisionRule:
    """A single, versioned decision rule.

    Attributes
    ----------
    rule_id : str
        Stable identifier, e.g. ``"phase_cold_start_v1"``.
    version : str
        Semantic version string, e.g. ``"1.0.0"``.
    description : str
        Human-readable explanation of what the rule does.
    trigger_conditions : list[str]
        Diagnostic conditions that cause this rule to fire.
    action : str
        What the rule does when it fires, e.g. ``"set_phase=cold_start"``.
    priority : int
        Evaluation order — higher values are checked first.
    """

    rule_id: str
    version: str
    description: str
    trigger_conditions: tuple[str, ...]  # frozen-friendly
    action: str
    priority: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["trigger_conditions"] = list(self.trigger_conditions)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionRule:
        data = data.copy()
        data["trigger_conditions"] = tuple(data["trigger_conditions"])
        return cls(**data)


@dataclass(frozen=True)
class RuleSignature:
    """Cryptographic receipt proving *which* rule fired and why.

    Attributes
    ----------
    rule_id : str
    version : str
    trigger_set : frozenset[str]
        Which diagnostic conditions actually matched.
    action_taken : str
    signature_hash : str
        SHA-256 of ``rule_id + version + trigger_set + action``.
    """

    rule_id: str
    version: str
    trigger_set: frozenset[str]
    action_taken: str
    signature_hash: str


@dataclass
class RuleDiff:
    """Structured diff between two rule sets.

    Attributes
    ----------
    added_rules : list[str]
        Rule IDs present in the "after" set but not in the "before" set.
    removed_rules : list[str]
        Rule IDs present in the "before" set but not in the "after" set.
    modified_rules : dict[str, dict[str, tuple]]
        ``{rule_id: {field: (old, new)}}`` for rules that changed.
    n_unchanged : int
        Count of rules present in both sets with no changes.
    """
    added_rules: list[str] = field(default_factory=list)
    removed_rules: list[str] = field(default_factory=list)
    modified_rules: dict[str, dict[str, tuple]] = field(default_factory=dict)
    n_unchanged: int = 0


@dataclass
class RuleTrace:
    """Complete audit trail of every rule that fired for one decision cycle.

    Attributes
    ----------
    signatures : list[RuleSignature]
        All rules that fired, in evaluation order.
    decision_path : list[str]
        Ordered sequence of ``rule_id`` values that contributed.
    rule_versions : dict[str, str]
        ``{rule_id: version}`` for every rule that fired.
    """

    signatures: list[RuleSignature] = field(default_factory=list)
    decision_path: list[str] = field(default_factory=list)
    rule_versions: dict[str, str] = field(default_factory=dict)


# ── Default rule set ──────────────────────────────────────────────────────────

def _default_rules() -> list[DecisionRule]:
    """Pre-loaded rule set mirroring MetaController logic."""
    return [
        # --- Phase rules (priority 100–50) ---
        DecisionRule(
            rule_id="phase_cold_start_v1",
            version="1.0.0",
            description="Too few observations — enter cold-start phase.",
            trigger_conditions=("n_observations < cold_start_min_observations",),
            action="set_phase=cold_start",
            priority=100,
        ),
        DecisionRule(
            rule_id="phase_stagnation_plateau_v1",
            version="1.0.0",
            description="KPI plateau detected — enter stagnation phase.",
            trigger_conditions=("kpi_plateau_length > stagnation_plateau_length",),
            action="set_phase=stagnation",
            priority=90,
        ),
        DecisionRule(
            rule_id="phase_stagnation_failure_v1",
            version="1.0.0",
            description="Failure clustering exceeds threshold — enter stagnation.",
            trigger_conditions=("failure_clustering > stagnation_failure_spike",),
            action="set_phase=stagnation",
            priority=85,
        ),
        DecisionRule(
            rule_id="phase_exploitation_convergence_v1",
            version="1.0.0",
            description="Converging with low uncertainty — enter exploitation.",
            trigger_conditions=(
                "convergence_trend > 0.3",
                "model_uncertainty < 0.3",
            ),
            action="set_phase=exploitation",
            priority=70,
        ),
        DecisionRule(
            rule_id="phase_exploitation_coverage_v1",
            version="1.0.0",
            description="High coverage + low uncertainty — enter exploitation.",
            trigger_conditions=(
                "exploration_coverage > coverage_plateau",
                "model_uncertainty < uncertainty_collapse",
            ),
            action="set_phase=exploitation",
            priority=65,
        ),
        DecisionRule(
            rule_id="phase_learning_v1",
            version="1.0.0",
            description="Default phase when nothing else matches.",
            trigger_conditions=("default",),
            action="set_phase=learning",
            priority=50,
        ),
        # --- Risk rules (priority 40–20) ---
        DecisionRule(
            rule_id="risk_conservative_phase_v1",
            version="1.0.0",
            description="Conservative risk in cold-start or stagnation phases.",
            trigger_conditions=("phase in (cold_start, stagnation)",),
            action="set_risk=conservative",
            priority=40,
        ),
        DecisionRule(
            rule_id="risk_conservative_failure_v1",
            version="1.0.0",
            description="High failure rate triggers conservative posture.",
            trigger_conditions=("failure_rate > 0.3",),
            action="set_risk=conservative",
            priority=35,
        ),
        DecisionRule(
            rule_id="risk_aggressive_v1",
            version="1.0.0",
            description="Exploitation with strong convergence allows aggression.",
            trigger_conditions=(
                "phase == exploitation",
                "convergence_trend > 0.5",
            ),
            action="set_risk=aggressive",
            priority=25,
        ),
        DecisionRule(
            rule_id="risk_moderate_v1",
            version="1.0.0",
            description="Default risk posture.",
            trigger_conditions=("default",),
            action="set_risk=moderate",
            priority=20,
        ),
    ]


# ── Decision Rule Engine ──────────────────────────────────────────────────────

# Thresholds mirroring MetaController defaults
_DEFAULT_THRESHOLDS = {
    "cold_start_min_observations": 10,
    "stagnation_plateau_length": 10,
    "stagnation_failure_spike": 0.5,
    "coverage_plateau": 0.8,
    "uncertainty_collapse": 0.1,
}


class DecisionRuleEngine:
    """Evaluate versioned decision rules against diagnostics.

    The engine pre-loads the default rule set that mirrors the
    ``MetaController`` logic.  Rules are evaluated in descending priority
    order.  Phase rules are *exclusive* (first match wins); risk rules are
    also exclusive (first match wins).

    Parameters
    ----------
    thresholds : dict | None
        Override default threshold values used during condition evaluation.
    """

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        self._rules: dict[str, DecisionRule] = {}
        self._thresholds = dict(_DEFAULT_THRESHOLDS)
        if thresholds:
            self._thresholds.update(thresholds)
        for rule in _default_rules():
            self._rules[rule.rule_id] = rule

    # ── Rule CRUD ─────────────────────────────────────────────────────────

    def add_rule(self, rule: DecisionRule) -> None:
        """Register (or replace) a rule."""
        self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> None:
        """Remove a rule by id.  Raises ``KeyError`` if not found."""
        del self._rules[rule_id]

    def get_rule(self, rule_id: str) -> DecisionRule:
        """Return a rule by id.  Raises ``KeyError`` if not found."""
        return self._rules[rule_id]

    @property
    def rules(self) -> list[DecisionRule]:
        """All registered rules sorted by descending priority."""
        return sorted(self._rules.values(), key=lambda r: r.priority, reverse=True)

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(
        self,
        diagnostics: dict[str, float],
        snapshot_size: int,
        phase: str | None = None,
    ) -> RuleTrace:
        """Evaluate all rules in priority order and return a ``RuleTrace``.

        Parameters
        ----------
        diagnostics : dict
            Diagnostic signal vector (17 signals from the diagnostic engine).
        snapshot_size : int
            Number of observations in the campaign snapshot.
        phase : str | None
            If provided, used for risk-rule evaluation.  When ``None`` the
            phase is determined from the phase rules that fire.
        """
        trace = RuleTrace()
        resolved_phase: str | None = phase
        resolved_risk: str | None = None

        for rule in self.rules:
            matched, triggers = self._check_conditions(
                rule, diagnostics, snapshot_size, resolved_phase,
            )
            if not matched:
                continue

            sig = RuleSignature(
                rule_id=rule.rule_id,
                version=rule.version,
                trigger_set=frozenset(triggers),
                action_taken=rule.action,
                signature_hash=_compute_signature(
                    rule.rule_id, rule.version, frozenset(triggers), rule.action,
                ),
            )
            trace.signatures.append(sig)
            trace.decision_path.append(rule.rule_id)
            trace.rule_versions[rule.rule_id] = rule.version

            # Phase rules are exclusive — first match wins
            if rule.action.startswith("set_phase=") and resolved_phase is None:
                resolved_phase = rule.action.split("=", 1)[1]

            # Risk rules are exclusive — first match wins
            if rule.action.startswith("set_risk=") and resolved_risk is None:
                resolved_risk = rule.action.split("=", 1)[1]

        return trace

    # ── Trace comparison ──────────────────────────────────────────────────

    @staticmethod
    def diff_traces(trace_a: RuleTrace, trace_b: RuleTrace) -> dict[str, Any]:
        """Compare two traces and return a structured diff.

        Returns a dict with keys:
        - ``added_rules``:   rule_ids in *b* but not in *a*
        - ``removed_rules``: rule_ids in *a* but not in *b*
        - ``changed_versions``: ``{rule_id: (old_ver, new_ver)}``
        - ``changed_actions``: ``{rule_id: (old_action, new_action)}``
        - ``path_changed``:  bool — whether the decision path differs
        """
        ids_a = set(trace_a.rule_versions.keys())
        ids_b = set(trace_b.rule_versions.keys())

        sigs_a = {s.rule_id: s for s in trace_a.signatures}
        sigs_b = {s.rule_id: s for s in trace_b.signatures}

        changed_versions: dict[str, tuple[str, str]] = {}
        changed_actions: dict[str, tuple[str, str]] = {}

        for rid in ids_a & ids_b:
            if trace_a.rule_versions[rid] != trace_b.rule_versions[rid]:
                changed_versions[rid] = (
                    trace_a.rule_versions[rid],
                    trace_b.rule_versions[rid],
                )
            if sigs_a[rid].action_taken != sigs_b[rid].action_taken:
                changed_actions[rid] = (
                    sigs_a[rid].action_taken,
                    sigs_b[rid].action_taken,
                )

        return {
            "added_rules": sorted(ids_b - ids_a),
            "removed_rules": sorted(ids_a - ids_b),
            "changed_versions": changed_versions,
            "changed_actions": changed_actions,
            "path_changed": trace_a.decision_path != trace_b.decision_path,
        }

    # ── Rule set diff ────────────────────────────────────────────────────

    @staticmethod
    def diff_rule_sets(
        rules_a: list[DecisionRule],
        rules_b: list[DecisionRule],
    ) -> RuleDiff:
        """Compare two lists of DecisionRule and produce a structured diff.

        Parameters
        ----------
        rules_a :
            "Before" rule set.
        rules_b :
            "After" rule set.

        Returns
        -------
        RuleDiff describing added, removed, and modified rules.
        """
        map_a = {r.rule_id: r for r in rules_a}
        map_b = {r.rule_id: r for r in rules_b}

        ids_a = set(map_a.keys())
        ids_b = set(map_b.keys())

        added = sorted(ids_b - ids_a)
        removed = sorted(ids_a - ids_b)

        modified: dict[str, dict[str, tuple[Any, Any]]] = {}
        for rid in sorted(ids_a & ids_b):
            ra, rb = map_a[rid], map_b[rid]
            changes: dict[str, tuple[Any, Any]] = {}
            if ra.version != rb.version:
                changes["version"] = (ra.version, rb.version)
            if ra.trigger_conditions != rb.trigger_conditions:
                changes["trigger_conditions"] = (
                    list(ra.trigger_conditions),
                    list(rb.trigger_conditions),
                )
            if ra.action != rb.action:
                changes["action"] = (ra.action, rb.action)
            if ra.priority != rb.priority:
                changes["priority"] = (ra.priority, rb.priority)
            if ra.description != rb.description:
                changes["description"] = (ra.description, rb.description)
            if changes:
                modified[rid] = changes

        return RuleDiff(
            added_rules=added,
            removed_rules=removed,
            modified_rules=modified,
            n_unchanged=len(ids_a & ids_b) - len(modified),
        )

    # ── Serialisation ─────────────────────────────────────────────────────

    def export_rules(self) -> list[dict[str, Any]]:
        """Export all rules as a JSON-serialisable list of dicts."""
        return [r.to_dict() for r in self.rules]

    def import_rules(self, rules: list[dict[str, Any]]) -> None:
        """Replace the current rule set with *rules* (list of dicts)."""
        self._rules.clear()
        for d in rules:
            rule = DecisionRule.from_dict(d)
            self._rules[rule.rule_id] = rule

    # ── Internal condition evaluator ──────────────────────────────────────

    def _check_conditions(
        self,
        rule: DecisionRule,
        diagnostics: dict[str, float],
        snapshot_size: int,
        resolved_phase: str | None,
    ) -> tuple[bool, list[str]]:
        """Check whether *rule*'s trigger conditions hold.

        Returns ``(matched: bool, triggers: list[str])`` where *triggers*
        is the subset of conditions that actually matched.
        """
        triggers: list[str] = []
        th = self._thresholds

        for cond in rule.trigger_conditions:
            # "default" always matches
            if cond == "default":
                triggers.append(cond)
                continue

            matched = self._evaluate_single_condition(
                cond, diagnostics, snapshot_size, resolved_phase, th,
            )
            if matched:
                triggers.append(cond)
            else:
                # All conditions must match (AND logic)
                return False, []

        return len(triggers) > 0, triggers

    @staticmethod
    def _evaluate_single_condition(
        cond: str,
        diagnostics: dict[str, float],
        snapshot_size: int,
        resolved_phase: str | None,
        thresholds: dict[str, float],
    ) -> bool:
        """Evaluate one human-readable condition string."""

        # n_observations < cold_start_min_observations
        if cond == "n_observations < cold_start_min_observations":
            return snapshot_size < thresholds["cold_start_min_observations"]

        # kpi_plateau_length > stagnation_plateau_length
        if cond == "kpi_plateau_length > stagnation_plateau_length":
            return diagnostics.get("kpi_plateau_length", 0) > thresholds["stagnation_plateau_length"]

        # failure_clustering > stagnation_failure_spike
        if cond == "failure_clustering > stagnation_failure_spike":
            return diagnostics.get("failure_clustering", 0) > thresholds["stagnation_failure_spike"]

        # convergence_trend > 0.3
        if cond == "convergence_trend > 0.3":
            return diagnostics.get("convergence_trend", 0) > 0.3

        # model_uncertainty < 0.3
        if cond == "model_uncertainty < 0.3":
            return diagnostics.get("model_uncertainty", 1.0) < 0.3

        # exploration_coverage > coverage_plateau
        if cond == "exploration_coverage > coverage_plateau":
            return diagnostics.get("exploration_coverage", 0) > thresholds["coverage_plateau"]

        # model_uncertainty < uncertainty_collapse
        if cond == "model_uncertainty < uncertainty_collapse":
            return diagnostics.get("model_uncertainty", 1.0) < thresholds["uncertainty_collapse"]

        # phase in (cold_start, stagnation)
        if cond == "phase in (cold_start, stagnation)":
            return resolved_phase in ("cold_start", "stagnation")

        # failure_rate > 0.3
        if cond == "failure_rate > 0.3":
            return diagnostics.get("failure_rate", 0) > 0.3

        # phase == exploitation
        if cond == "phase == exploitation":
            return resolved_phase == "exploitation"

        # convergence_trend > 0.5
        if cond == "convergence_trend > 0.5":
            return diagnostics.get("convergence_trend", 0) > 0.5

        # Unknown condition — never matches
        return False
