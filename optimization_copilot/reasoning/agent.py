"""Optimization Reasoning Agent: generates human-readable explanations.

Provides template-based (and optionally LLM-enhanced) narrative generation
for surgery actions, failure clusters, and campaign status.  The LLM is
**never** in the critical path -- every public method works fully with
pure template text when no API key is available.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    Phase,
    ProblemFingerprint,
    StrategyDecision,
)
from optimization_copilot.diagnostics.engine import DiagnosticsVector
from optimization_copilot.feasibility.feasibility import FeasibilityMap
from optimization_copilot.feasibility.taxonomy import (
    ClassifiedFailure,
    FailureTaxonomy,
    FailureType,
)
from optimization_copilot.surgery.models import (
    ActionType,
    DerivedType,
    SurgeryAction,
    SurgeryReport,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RewriteSuggestion:
    """A human-readable explanation for a single surgery action."""

    action_type: str
    target_params: list[str]
    explanation: str
    confidence_narrative: str
    evidence_summary: str
    generated_by: str = "template"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target_params": list(self.target_params),
            "explanation": self.explanation,
            "confidence_narrative": self.confidence_narrative,
            "evidence_summary": self.evidence_summary,
            "generated_by": self.generated_by,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewriteSuggestion:
        data = data.copy()
        data["target_params"] = list(data.get("target_params", []))
        data["metadata"] = dict(data.get("metadata", {}))
        return cls(**data)


@dataclass
class FailureCluster:
    """A group of related failures in parameter space."""

    cluster_id: int
    failure_indices: list[int]
    failure_type: str
    parameter_ranges: dict[str, tuple[float, float]]
    explanation: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "failure_indices": list(self.failure_indices),
            "failure_type": self.failure_type,
            "parameter_ranges": {
                k: list(v) for k, v in self.parameter_ranges.items()
            },
            "explanation": self.explanation,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureCluster:
        data = data.copy()
        data["failure_indices"] = list(data.get("failure_indices", []))
        # Convert lists back to tuples for parameter_ranges
        raw_ranges = data.get("parameter_ranges", {})
        data["parameter_ranges"] = {
            k: tuple(v) for k, v in raw_ranges.items()
        }
        return cls(**data)


@dataclass
class FailureClusterReport:
    """Aggregated failure clustering analysis."""

    clusters: list[FailureCluster]
    overall_pattern: str
    dominant_failure_mode: str
    recommendations: list[str]
    generated_by: str = "template"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clusters": [c.to_dict() for c in self.clusters],
            "overall_pattern": self.overall_pattern,
            "dominant_failure_mode": self.dominant_failure_mode,
            "recommendations": list(self.recommendations),
            "generated_by": self.generated_by,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureClusterReport:
        data = data.copy()
        data["clusters"] = [
            FailureCluster.from_dict(c) for c in data.get("clusters", [])
        ]
        data["recommendations"] = list(data.get("recommendations", []))
        data["metadata"] = dict(data.get("metadata", {}))
        return cls(**data)


@dataclass
class CampaignNarrative:
    """Full narrative summary of a campaign's current state."""

    campaign_id: str
    executive_summary: str
    phase_description: str
    diagnostic_summary: str
    strategy_rationale: str
    failure_analysis: str
    recommendations: list[str]
    generated_by: str = "template"
    metadata: dict[str, Any] = field(default_factory=dict)

    def format_text(self) -> str:
        """Return a multi-section plain-text narrative."""
        sections: list[str] = []
        sections.append(f"=== CAMPAIGN NARRATIVE: {self.campaign_id} ===")
        sections.append("")
        sections.append("-- EXECUTIVE SUMMARY --")
        sections.append(self.executive_summary)
        sections.append("")
        sections.append("-- CURRENT PHASE --")
        sections.append(self.phase_description)
        sections.append("")
        sections.append("-- DIAGNOSTIC SIGNALS --")
        sections.append(self.diagnostic_summary)
        sections.append("")
        sections.append("-- STRATEGY RATIONALE --")
        sections.append(self.strategy_rationale)
        if self.failure_analysis:
            sections.append("")
            sections.append("-- FAILURE ANALYSIS --")
            sections.append(self.failure_analysis)
        sections.append("")
        sections.append("-- RECOMMENDATIONS --")
        for i, rec in enumerate(self.recommendations, 1):
            sections.append(f"  {i}. {rec}")
        sections.append("")
        sections.append(f"[Generated by: {self.generated_by}]")
        return "\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "executive_summary": self.executive_summary,
            "phase_description": self.phase_description,
            "diagnostic_summary": self.diagnostic_summary,
            "strategy_rationale": self.strategy_rationale,
            "failure_analysis": self.failure_analysis,
            "recommendations": list(self.recommendations),
            "generated_by": self.generated_by,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignNarrative:
        data = data.copy()
        data["recommendations"] = list(data.get("recommendations", []))
        data["metadata"] = dict(data.get("metadata", {}))
        return cls(**data)


# ---------------------------------------------------------------------------
# LLM caller (stdlib only -- no external dependencies)
# ---------------------------------------------------------------------------


def _default_llm_caller(
    api_key: str, model: str
) -> Callable[[str], str | None]:
    """Return a callable that sends a prompt to the Anthropic messages API.

    Uses only ``urllib.request`` so that the package has zero runtime
    dependencies beyond the standard library.
    """

    def call(prompt: str) -> str | None:
        import urllib.error
        import urllib.request

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        body = json.dumps(
            {
                "model": model,
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        req = urllib.request.Request(
            url, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                return data["content"][0]["text"]
        except Exception:
            return None

    return call


# ---------------------------------------------------------------------------
# ReasoningAgent
# ---------------------------------------------------------------------------


class ReasoningAgent:
    """Generate human-readable reasoning for optimization decisions.

    The LLM is **optional**.  Every public method produces complete output
    via deterministic templates when no API key is available.  The
    ``generated_by`` field on each result indicates whether the template
    or the LLM produced the text.

    Parameters
    ----------
    api_key:
        Anthropic API key.  Falls back to ``MODEL_API_KEY`` then
        ``ANTHROPIC_API_KEY`` environment variables.
    model:
        Model identifier used when calling the LLM.
    llm_caller:
        Optional pre-built callable ``(prompt) -> text | None``.  When
        supplied, *api_key* and *model* are ignored.
    use_llm:
        Master switch.  Set to ``False`` to force template-only mode even
        when an API key is available.
    """

    # Phase descriptions used by _template_narrative
    _PHASE_DESCRIPTIONS: dict[str, str] = {
        "cold_start": (
            "The campaign is in the cold-start phase, gathering initial "
            "observations to build a baseline understanding of the search space."
        ),
        "learning": (
            "The campaign is actively learning, balancing exploration of "
            "unknown regions with exploitation of promising areas."
        ),
        "exploitation": (
            "The campaign has entered the exploitation phase, focusing on "
            "refining the best-known regions to squeeze out further gains."
        ),
        "stagnation": (
            "The campaign appears stagnant -- recent observations have not "
            "yielded meaningful improvements over the current best."
        ),
        "termination": (
            "The campaign is approaching termination. Budget or convergence "
            "criteria are close to being met."
        ),
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-6",
        llm_caller: Callable[[str], str | None] | None = None,
        use_llm: bool = True,
    ) -> None:
        if llm_caller is not None:
            self._llm_caller = llm_caller
        else:
            key = (
                api_key
                or os.environ.get("MODEL_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
            )
            if key and use_llm:
                self._llm_caller: Callable[[str], str | None] | None = (
                    _default_llm_caller(key, model)
                )
            else:
                self._llm_caller = None

    # ------------------------------------------------------------------
    # Public API: explain_surgery
    # ------------------------------------------------------------------

    def explain_surgery(
        self,
        report: SurgeryReport,
        snapshot: CampaignSnapshot,
    ) -> list[RewriteSuggestion]:
        """Explain each surgery action in *report* as a human-readable suggestion."""
        if not report.has_actions:
            return []

        suggestions: list[RewriteSuggestion] = []
        for action in report.actions:
            template_text = self._surgery_template(action)
            generated_by = "template"

            if self._llm_caller is not None:
                prompt = (
                    "You are an optimization expert. Explain the following "
                    "search-space surgery action in 2-3 concise sentences "
                    "that a scientist can understand.\n\n"
                    f"Action type: {action.action_type.value}\n"
                    f"Target parameters: {action.target_params}\n"
                    f"Reason: {action.reason}\n"
                    f"Confidence: {action.confidence}\n"
                    f"Evidence: {json.dumps(action.evidence)}\n"
                    f"Campaign: {snapshot.campaign_id}, "
                    f"{snapshot.n_observations} observations, "
                    f"{snapshot.n_failures} failures\n"
                )
                llm_text = self._llm_caller(prompt)
                if llm_text:
                    template_text = llm_text
                    generated_by = "llm"

            suggestions.append(
                RewriteSuggestion(
                    action_type=action.action_type.value,
                    target_params=list(action.target_params),
                    explanation=template_text,
                    confidence_narrative=self._confidence_narrative(
                        action.confidence
                    ),
                    evidence_summary=self._evidence_summary(action.evidence),
                    generated_by=generated_by,
                )
            )
        return suggestions

    # ------------------------------------------------------------------
    # Public API: explain_failures
    # ------------------------------------------------------------------

    def explain_failures(
        self,
        feasibility_map: FeasibilityMap,
        snapshot: CampaignSnapshot,
        taxonomy: FailureTaxonomy | None = None,
        seed: int = 42,
    ) -> FailureClusterReport:
        """Cluster and explain failures in *snapshot*."""
        failures = [o for o in snapshot.observations if o.is_failure]
        if not failures:
            return FailureClusterReport(
                clusters=[],
                overall_pattern="No failures observed in this campaign.",
                dominant_failure_mode="none",
                recommendations=["No failure-related actions needed."],
                generated_by="template",
            )

        # Build clusters
        if taxonomy is not None and taxonomy.classified_failures:
            clusters = self._cluster_by_taxonomy(failures, taxonomy, snapshot)
        else:
            clusters = self._cluster_by_proximity(failures, snapshot, seed)

        # Dominant failure mode
        if clusters:
            dominant = max(clusters, key=lambda c: c.count)
            dominant_failure_mode = dominant.failure_type
        else:
            dominant_failure_mode = "unknown"

        # Overall pattern (template)
        n_fail = len(failures)
        n_obs = snapshot.n_observations
        rate = snapshot.failure_rate
        template_pattern = (
            f"Observed {n_fail} failures out of {n_obs} observations "
            f"({rate:.1%} failure rate) across {len(clusters)} cluster(s). "
            f"Dominant failure mode: {dominant_failure_mode}."
        )

        overall_pattern = template_pattern
        generated_by = "template"

        if self._llm_caller is not None:
            prompt = (
                "Summarize the following failure clustering analysis in "
                "2-3 sentences for a scientist.\n\n"
                f"Total failures: {n_fail}/{n_obs}\n"
                f"Clusters: {len(clusters)}\n"
                f"Dominant mode: {dominant_failure_mode}\n"
                f"Cluster details: {[c.to_dict() for c in clusters]}\n"
            )
            llm_text = self._llm_caller(prompt)
            if llm_text:
                overall_pattern = llm_text
                generated_by = "llm"

        recommendations = self._failure_recommendations(
            clusters, feasibility_map, snapshot
        )

        return FailureClusterReport(
            clusters=clusters,
            overall_pattern=overall_pattern,
            dominant_failure_mode=dominant_failure_mode,
            recommendations=recommendations,
            generated_by=generated_by,
        )

    # ------------------------------------------------------------------
    # Public API: generate_narrative
    # ------------------------------------------------------------------

    def generate_narrative(
        self,
        snapshot: CampaignSnapshot,
        diagnostics: DiagnosticsVector,
        decision: StrategyDecision,
        fingerprint: ProblemFingerprint | None = None,
        surgery_report: SurgeryReport | None = None,
    ) -> CampaignNarrative:
        """Produce a full campaign narrative."""
        narrative = self._template_narrative(
            snapshot, diagnostics, decision, fingerprint, surgery_report
        )

        # Try to enhance executive_summary via LLM
        if self._llm_caller is not None:
            prompt = (
                "Write a concise 2-3 sentence executive summary for this "
                "optimization campaign status.\n\n"
                f"Campaign: {snapshot.campaign_id}\n"
                f"Phase: {decision.phase.value}\n"
                f"Observations: {snapshot.n_observations}\n"
                f"Failures: {snapshot.n_failures}\n"
                f"Backend: {decision.backend_name}\n"
                f"Exploration strength: {decision.exploration_strength}\n"
                f"Convergence trend: {diagnostics.convergence_trend}\n"
                f"Plateau length: {diagnostics.kpi_plateau_length}\n"
                f"Best KPI: {diagnostics.best_kpi_value}\n"
            )
            llm_text = self._llm_caller(prompt)
            if llm_text:
                narrative = CampaignNarrative(
                    campaign_id=narrative.campaign_id,
                    executive_summary=llm_text,
                    phase_description=narrative.phase_description,
                    diagnostic_summary=narrative.diagnostic_summary,
                    strategy_rationale=narrative.strategy_rationale,
                    failure_analysis=narrative.failure_analysis,
                    recommendations=narrative.recommendations,
                    generated_by="llm",
                    metadata=narrative.metadata,
                )

        return narrative

    # ------------------------------------------------------------------
    # Surgery template helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _surgery_template(action: SurgeryAction) -> str:
        """Produce a template explanation for a single surgery action."""
        at = action.action_type

        if at == ActionType.TIGHTEN_RANGE:
            param = action.target_params[0] if action.target_params else "?"
            orig_lo = action.evidence.get("original_lower", "?")
            orig_hi = action.evidence.get("original_upper", "?")
            pct = action.evidence.get("range_reduction_pct", "?")
            return (
                f"Tighten the range of '{param}' from "
                f"[{orig_lo}, {orig_hi}] to "
                f"[{action.new_lower}, {action.new_upper}] "
                f"(a {pct}% reduction). "
                f"This focuses the search on the most promising sub-region."
            )

        if at == ActionType.FREEZE_PARAMETER:
            param = action.target_params[0] if action.target_params else "?"
            imp = action.evidence.get("importance_score", "?")
            return (
                f"Freeze parameter '{param}' at value {action.freeze_value}. "
                f"Its importance score is {imp}, indicating it has minimal "
                f"impact on the objective and can be held constant to reduce "
                f"dimensionality."
            )

        if at == ActionType.CONDITIONAL_FREEZE:
            param = action.target_params[0] if action.target_params else "?"
            corr = action.evidence.get("correlations", {})
            return (
                f"Conditionally freeze '{param}' when "
                f"'{action.condition_param}' is "
                f"{action.condition_direction} {action.condition_threshold}. "
                f"Correlation evidence: {corr}. "
                f"This removes a redundant degree of freedom in that region."
            )

        if at == ActionType.MERGE_PARAMETERS:
            corr = action.evidence.get("correlation", "?")
            primary = action.evidence.get("primary_param", "?")
            secondary = action.evidence.get("secondary_param", "?")
            return (
                f"Merge parameters '{primary}' and '{secondary}' "
                f"(correlation: {corr}). Their strong co-movement suggests "
                f"a single combined parameter captures the relevant variation."
            )

        if at == ActionType.DERIVE_PARAMETER:
            if action.derived_type == DerivedType.LOG:
                param = action.target_params[0] if action.target_params else "?"
                ratio = action.evidence.get("range_ratio", "?")
                return (
                    f"Replace '{param}' with its log-transform "
                    f"'{action.derived_name}'. The parameter spans a range "
                    f"ratio of {ratio}, suggesting log-scale is more "
                    f"appropriate for uniform exploration."
                )
            if action.derived_type == DerivedType.RATIO:
                p0 = action.target_params[0] if len(action.target_params) > 0 else "?"
                p1 = action.target_params[1] if len(action.target_params) > 1 else "?"
                corr = action.evidence.get("correlation", "?")
                return (
                    f"Derive a ratio parameter '{action.derived_name}' from "
                    f"'{p0}' and '{p1}' (correlation: {corr}). Modelling "
                    f"their ratio directly may simplify the landscape."
                )
            # Other derived types: generic
            return (
                f"Derive new parameter '{action.derived_name}' "
                f"({action.derived_type.value if action.derived_type else '?'} "
                f"transform) from {action.target_params}. "
                f"Reason: {action.reason or 'structural simplification'}."
            )

        if at == ActionType.REMOVE_PARAMETER:
            return action.reason or "Remove parameter (no further explanation)."

        # Fallback
        return action.reason or "No explanation available."

    @staticmethod
    def _confidence_narrative(confidence: float) -> str:
        """Convert a numeric confidence score to a human-readable band."""
        if confidence >= 0.8:
            return (
                f"High confidence ({confidence:.2f}): strong evidence "
                f"supports this action."
            )
        if confidence >= 0.5:
            return (
                f"Moderate confidence ({confidence:.2f}): reasonable evidence "
                f"supports this action, but additional data could strengthen "
                f"the case."
            )
        return (
            f"Low confidence ({confidence:.2f}): limited evidence; "
            f"treat as exploratory suggestion."
        )

    @staticmethod
    def _evidence_summary(evidence: dict[str, Any]) -> str:
        """Summarize an evidence dict as a human-readable string."""
        if not evidence:
            return "No supporting evidence."
        parts = [f"{k}={v}" for k, v in evidence.items()]
        return "Evidence: " + ", ".join(parts)

    # ------------------------------------------------------------------
    # Failure clustering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cluster_by_taxonomy(
        failures: list[Observation],
        taxonomy: FailureTaxonomy,
        snapshot: CampaignSnapshot,
    ) -> list[FailureCluster]:
        """Group failures using taxonomy classifications."""
        # Map observation index -> Observation for failures
        obs_list = snapshot.observations
        failure_set = {id(o) for o in failures}

        # Group classified failures by type
        groups: dict[str, list[ClassifiedFailure]] = {}
        for cf in taxonomy.classified_failures:
            ft_val = cf.failure_type.value
            groups.setdefault(ft_val, []).append(cf)

        clusters: list[FailureCluster] = []
        for cid, (ft_val, classified_list) in enumerate(groups.items()):
            indices = [cf.observation_index for cf in classified_list]

            # Compute parameter ranges across these failure observations
            param_ranges: dict[str, tuple[float, float]] = {}
            for spec in snapshot.parameter_specs:
                vals: list[float] = []
                for idx in indices:
                    if 0 <= idx < len(obs_list):
                        v = obs_list[idx].parameters.get(spec.name)
                        if isinstance(v, (int, float)):
                            vals.append(float(v))
                if vals:
                    param_ranges[spec.name] = (min(vals), max(vals))

            clusters.append(
                FailureCluster(
                    cluster_id=cid,
                    failure_indices=indices,
                    failure_type=ft_val,
                    parameter_ranges=param_ranges,
                    explanation=(
                        f"Cluster of {len(indices)} failures classified as "
                        f"'{ft_val}' by the failure taxonomy."
                    ),
                    count=len(indices),
                )
            )
        return clusters

    @staticmethod
    def _cluster_by_proximity(
        failures: list[Observation],
        snapshot: CampaignSnapshot,
        seed: int,
    ) -> list[FailureCluster]:
        """Simple single-linkage agglomerative clustering in normalised space."""
        obs_list = snapshot.observations
        specs = snapshot.parameter_specs

        # Find failure indices in the full observation list
        failure_indices: list[int] = []
        for i, obs in enumerate(obs_list):
            if obs.is_failure:
                failure_indices.append(i)

        if not failure_indices:
            return []

        rng = random.Random(seed)
        shuffled = list(failure_indices)
        rng.shuffle(shuffled)

        # Normalise each failure's parameters to [0, 1]
        def _normalise(obs: Observation) -> list[float]:
            result: list[float] = []
            for spec in specs:
                v = obs.parameters.get(spec.name)
                if not isinstance(v, (int, float)):
                    result.append(0.5)
                    continue
                lo = spec.lower if spec.lower is not None else 0.0
                hi = spec.upper if spec.upper is not None else 1.0
                rng_size = hi - lo
                if rng_size == 0:
                    result.append(0.5)
                else:
                    result.append((float(v) - lo) / rng_size)
            return result

        normalised = {idx: _normalise(obs_list[idx]) for idx in failure_indices}

        def _dist(a: int, b: int) -> float:
            va, vb = normalised[a], normalised[b]
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb)))

        # Each point starts in its own cluster (use shuffled order for
        # deterministic tie-breaking)
        cluster_map: dict[int, int] = {}  # failure_index -> cluster_label
        for label, idx in enumerate(shuffled):
            cluster_map[idx] = label

        # Single-linkage: merge clusters if any pair of members < 0.3
        threshold = 0.3
        changed = True
        while changed:
            changed = False
            labels = sorted(set(cluster_map.values()))
            for i_pos in range(len(labels)):
                for j_pos in range(i_pos + 1, len(labels)):
                    la, lb = labels[i_pos], labels[j_pos]
                    members_a = [
                        k for k, v in cluster_map.items() if v == la
                    ]
                    members_b = [
                        k for k, v in cluster_map.items() if v == lb
                    ]
                    # Check if any pair is close enough
                    should_merge = False
                    for ma in members_a:
                        for mb in members_b:
                            if _dist(ma, mb) < threshold:
                                should_merge = True
                                break
                        if should_merge:
                            break
                    if should_merge:
                        for k in members_b:
                            cluster_map[k] = la
                        changed = True
                        break
                if changed:
                    break

        # Group by final label
        label_groups: dict[int, list[int]] = {}
        for idx, label in cluster_map.items():
            label_groups.setdefault(label, []).append(idx)

        clusters: list[FailureCluster] = []
        for cid, (_, indices) in enumerate(sorted(label_groups.items())):
            indices_sorted = sorted(indices)
            param_ranges: dict[str, tuple[float, float]] = {}
            for spec in specs:
                vals: list[float] = []
                for idx in indices_sorted:
                    v = obs_list[idx].parameters.get(spec.name)
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                if vals:
                    param_ranges[spec.name] = (min(vals), max(vals))

            clusters.append(
                FailureCluster(
                    cluster_id=cid,
                    failure_indices=indices_sorted,
                    failure_type="proximity",
                    parameter_ranges=param_ranges,
                    explanation=(
                        f"Proximity cluster of {len(indices_sorted)} failures "
                        f"within normalised distance < {threshold}."
                    ),
                    count=len(indices_sorted),
                )
            )
        return clusters

    @staticmethod
    def _failure_recommendations(
        clusters: list[FailureCluster],
        feasibility_map: FeasibilityMap,
        snapshot: CampaignSnapshot,
    ) -> list[str]:
        """Generate actionable recommendations from failure analysis."""
        recs: list[str] = []
        failure_rate = snapshot.failure_rate

        if failure_rate > 0.3:
            recs.append(
                f"Failure rate is {failure_rate:.1%} -- consider tightening "
                f"parameter bounds to avoid infeasible regions."
            )

        n_zones = len(feasibility_map.infeasible_zones)
        if n_zones > 0:
            recs.append(
                f"Identified {n_zones} infeasible zone(s) in parameter space; "
                f"future sampling should avoid these regions."
            )

        for cluster in clusters:
            if cluster.count >= 5:
                recs.append(
                    f"Cluster {cluster.cluster_id} ({cluster.failure_type}) "
                    f"contains {cluster.count} failures -- investigate root "
                    f"cause in parameter ranges: {cluster.parameter_ranges}."
                )

        if feasibility_map.feasibility_score < 0.5:
            recs.append(
                f"Overall feasibility score is low "
                f"({feasibility_map.feasibility_score:.2f}); consider "
                f"relaxing constraints or expanding the search space."
            )

        if not recs:
            recs.append("Failure patterns are manageable.")

        return recs

    # ------------------------------------------------------------------
    # Narrative helpers
    # ------------------------------------------------------------------

    def _template_narrative(
        self,
        snapshot: CampaignSnapshot,
        diagnostics: DiagnosticsVector,
        decision: StrategyDecision,
        fingerprint: ProblemFingerprint | None,
        surgery_report: SurgeryReport | None,
    ) -> CampaignNarrative:
        """Build a complete narrative from templates (no LLM)."""
        phase_val = decision.phase.value
        phase_desc_text = self._PHASE_DESCRIPTIONS.get(
            phase_val, f"The campaign is in the '{phase_val}' phase."
        )

        # Executive summary
        executive_summary = (
            f"Campaign '{snapshot.campaign_id}' has completed "
            f"{snapshot.n_observations} observations with "
            f"{snapshot.n_failures} failure(s). "
            f"Currently in the {phase_val} phase using the "
            f"'{decision.backend_name}' backend with exploration "
            f"strength {decision.exploration_strength:.2f}."
        )

        # Phase description
        phase_description = (
            f"Phase: {phase_val}. {phase_desc_text} "
            f"Risk posture: {decision.risk_posture.value}. "
            f"Batch size: {decision.batch_size}."
        )

        # Diagnostic summary
        diagnostic_summary = (
            f"Convergence trend: {diagnostics.convergence_trend:+.3f}. "
            f"KPI plateau length: {diagnostics.kpi_plateau_length} observations. "
            f"Failure rate: {diagnostics.failure_rate:.1%}. "
            f"Exploration coverage: {diagnostics.exploration_coverage:.1%}. "
            f"Noise estimate: {diagnostics.noise_estimate:.3f}."
        )

        # Strategy rationale
        reason_parts: list[str] = []
        if decision.reason_codes:
            reason_parts.append(
                "Reason codes: " + ", ".join(decision.reason_codes) + "."
            )
        if decision.fallback_events:
            reason_parts.append(
                "Fallback events: " + ", ".join(decision.fallback_events) + "."
            )
        if not reason_parts:
            reason_parts.append("No specific reason codes recorded.")
        strategy_rationale = " ".join(reason_parts)

        # Failure analysis
        n_fail = snapshot.n_failures
        fail_rate = snapshot.failure_rate
        if n_fail > 0:
            failure_analysis = (
                f"{n_fail} failure(s) observed ({fail_rate:.1%} failure rate)."
            )
        else:
            failure_analysis = ""

        # Recommendations
        recommendations: list[str] = []
        if diagnostics.kpi_plateau_length > 10:
            recommendations.append(
                f"Plateau detected ({diagnostics.kpi_plateau_length} "
                f"observations without improvement). Consider increasing "
                f"exploration or switching strategy."
            )
        if fail_rate > 0.2:
            recommendations.append(
                f"Failure rate ({fail_rate:.1%}) is elevated. Review "
                f"parameter bounds and constraints."
            )
        if diagnostics.exploration_coverage < 0.1:
            recommendations.append(
                "Exploration coverage is low. Increase exploration strength "
                "or batch size to sample more of the space."
            )
        if surgery_report is not None and surgery_report.has_actions:
            recommendations.append(
                f"Search-space surgery recommended {surgery_report.n_actions} "
                f"action(s) (space reduction: "
                f"{surgery_report.space_reduction_ratio:.1%}). "
                f"Review and apply suggested modifications."
            )
        if not recommendations:
            recommendations.append(
                "Campaign is progressing normally. No immediate action needed."
            )

        return CampaignNarrative(
            campaign_id=snapshot.campaign_id,
            executive_summary=executive_summary,
            phase_description=phase_description,
            diagnostic_summary=diagnostic_summary,
            strategy_rationale=strategy_rationale,
            failure_analysis=failure_analysis,
            recommendations=recommendations,
            generated_by="template",
        )

    def _llm_narrative(
        self,
        snapshot: CampaignSnapshot,
        diagnostics: DiagnosticsVector,
        decision: StrategyDecision,
    ) -> str | None:
        """Attempt to generate an LLM-enhanced executive summary."""
        if self._llm_caller is None:
            return None
        prompt = (
            "Write a concise 2-3 sentence executive summary for this "
            "optimization campaign status.\n\n"
            f"Campaign: {snapshot.campaign_id}\n"
            f"Phase: {decision.phase.value}\n"
            f"Observations: {snapshot.n_observations}\n"
            f"Failures: {snapshot.n_failures}\n"
            f"Backend: {decision.backend_name}\n"
            f"Exploration strength: {decision.exploration_strength}\n"
            f"Convergence trend: {diagnostics.convergence_trend}\n"
            f"Plateau length: {diagnostics.kpi_plateau_length}\n"
            f"Best KPI: {diagnostics.best_kpi_value}\n"
        )
        return self._llm_caller(prompt)
