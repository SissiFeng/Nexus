"""LiteraturePriorAgent -- provides parameter priors from domain knowledge.

Uses DomainConfig (mod #2) as the primary source, falling back to
hardcoded prior tables when no domain configuration is available.
"""

from __future__ import annotations

from typing import Any

from optimization_copilot.agents.base import (
    AgentContext,
    AgentMode,
    OptimizationFeedback,
    ScientificAgent,
    TriggerCondition,
)
from optimization_copilot.agents.literature.prior_tables import (
    PriorEntry,
    PriorTable,
)


# Activate in early campaign iterations
_EARLY_CAMPAIGN_THRESHOLD = 5

# Minimum coverage to generate actionable feedback
_MIN_COVERAGE_FOR_FEEDBACK = 0.3


class LiteraturePriorAgent(ScientificAgent):
    """Agent that provides starting-point priors from literature/domain knowledge.

    Reads DomainConfig first (mod #3) to get domain-specific constraints
    and parameter ranges. Falls back to hardcoded prior tables.

    Parameters
    ----------
    prior_table : PriorTable | None
        Custom prior table. Defaults to the built-in tables.
    mode : AgentMode
        Operational mode.
    """

    def __init__(
        self,
        prior_table: PriorTable | None = None,
        mode: AgentMode = AgentMode.PRAGMATIC,
    ) -> None:
        super().__init__(mode=mode)
        self._prior_table = prior_table if prior_table is not None else PriorTable()

        self._trigger_conditions = [
            TriggerCondition(
                name="early_campaign",
                check_fn_name="check_early_campaign",
                priority=7,
                description=(
                    f"Activates in early campaign (iteration <= {_EARLY_CAMPAIGN_THRESHOLD})"
                ),
            ),
            TriggerCondition(
                name="domain_config_available",
                check_fn_name="check_domain_config",
                priority=5,
                description="Activates when domain configuration is available",
            ),
        ]

    def name(self) -> str:
        return "literature_prior"

    def should_activate(self, context: AgentContext) -> bool:
        """Activate in early campaign or when domain_config is available."""
        if context.iteration <= _EARLY_CAMPAIGN_THRESHOLD:
            return True
        if context.domain_config is not None:
            return True
        return False

    def validate_context(self, context: AgentContext) -> bool:
        """Validate that we have either domain_config or known domain."""
        if context.domain_config is not None:
            return True
        domain = context.metadata.get("domain")
        if domain and self._prior_table.get_priors(domain):
            return True
        # Even without domain, we can still check if history has parameters
        if context.optimization_history:
            first = context.optimization_history[0]
            if "parameters" in first:
                return True
        return False

    def analyze(self, context: AgentContext) -> dict[str, Any]:
        """Provide parameter priors from domain knowledge.

        Priority:
        1. DomainConfig constraints (mod #3)
        2. Hardcoded prior tables
        3. No priors available

        Returns
        -------
        dict[str, Any]
            Keys: ``priors``, ``domain``, ``n_parameters_matched``,
            ``coverage``, ``source``.
        """
        domain = self._resolve_domain(context)
        param_names = context.get_parameter_names()

        # Try DomainConfig first (mod #3)
        dc_priors = self._priors_from_domain_config(context, param_names)
        if dc_priors:
            coverage = len(dc_priors) / len(param_names) if param_names else 0.0
            return {
                "priors": dc_priors,
                "domain": domain,
                "n_parameters_matched": len(dc_priors),
                "coverage": round(coverage, 3),
                "source": "domain_config",
            }

        # Fallback to prior tables
        if domain is None:
            return {
                "priors": [],
                "domain": None,
                "n_parameters_matched": 0,
                "coverage": 0.0,
                "source": "none",
            }

        table_priors = self._priors_from_table(domain, param_names)
        coverage = len(table_priors) / len(param_names) if param_names else 0.0

        return {
            "priors": table_priors,
            "domain": domain,
            "n_parameters_matched": len(table_priors),
            "coverage": round(coverage, 3),
            "source": "prior_table",
        }

    def get_optimization_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        """Convert prior analysis to optimization feedback.

        Returns ``prior_update`` feedback if coverage is sufficient.
        """
        priors = analysis_result.get("priors", [])
        coverage = analysis_result.get("coverage", 0.0)

        if coverage < _MIN_COVERAGE_FOR_FEEDBACK or not priors:
            return None

        domain = analysis_result.get("domain", "unknown")
        source = analysis_result.get("source", "unknown")

        # Build prior_update payload
        prior_payload: dict[str, Any] = {}
        for p in priors:
            prior_payload[p["parameter"]] = {
                "mean": p["mean"],
                "std": p["std"],
            }

        confidence = min(0.8, 0.4 + coverage * 0.5)

        return OptimizationFeedback(
            agent_name=self.name(),
            feedback_type="prior_update",
            confidence=confidence,
            payload={
                "parameter_priors": prior_payload,
                "domain": domain,
                "source": source,
                "coverage": coverage,
            },
            reasoning=(
                f"Literature priors for {len(priors)} parameters in domain "
                f"'{domain}' (source: {source}, coverage: {coverage:.0%}). "
                f"Suggested starting regions based on domain knowledge."
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_domain(context: AgentContext) -> str | None:
        """Resolve domain name from context."""
        if context.domain_config is not None:
            return context.domain_config.domain_name
        domain = context.metadata.get("domain")
        if isinstance(domain, str) and domain:
            return domain
        return None

    @staticmethod
    def _priors_from_domain_config(
        context: AgentContext,
        param_names: list[str],
    ) -> list[dict[str, Any]]:
        """Extract prior information from DomainConfig constraints.

        Uses ``get_constraints()`` to find min/max bounds and derives
        mean/std from those bounds.
        """
        if context.domain_config is None:
            return []

        constraints = context.domain_config.get_constraints()
        if not constraints:
            return []

        priors: list[dict[str, Any]] = []
        for pname in param_names:
            if pname in constraints:
                c = constraints[pname]
                pmin = c.get("min")
                pmax = c.get("max")
                if pmin is not None and pmax is not None:
                    try:
                        fmin = float(pmin)
                        fmax = float(pmax)
                        mean = (fmin + fmax) / 2.0
                        std = (fmax - fmin) / 4.0  # ~95% within bounds
                        priors.append({
                            "parameter": pname,
                            "mean": mean,
                            "std": std,
                            "source": "domain_config",
                            "notes": f"Derived from constraints [{fmin}, {fmax}]",
                        })
                    except (TypeError, ValueError):
                        continue

        return priors

    def _priors_from_table(
        self, domain: str, param_names: list[str]
    ) -> list[dict[str, Any]]:
        """Extract prior information from hardcoded tables."""
        priors: list[dict[str, Any]] = []

        for pname in param_names:
            entry = self._prior_table.get_prior_for_parameter(domain, pname)
            if entry is not None:
                priors.append({
                    "parameter": entry.parameter,
                    "mean": entry.prior_mean,
                    "std": entry.prior_std,
                    "source": entry.source,
                    "notes": entry.notes,
                })

        return priors
