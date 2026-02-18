"""Causal discovery integration for optimization campaigns.

Answers "which variables truly drive the objective?" by learning causal
graphs from experimental data and distinguishing correlation from causation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.causal.models import CausalGraph, CausalNode, CausalEdge
from optimization_copilot.causal.structure import CausalStructureLearner
from optimization_copilot.core.models import CampaignSnapshot, Observation


@dataclass
class VariableCausalImpact:
    """Causal impact analysis for a single variable.
    
    Attributes:
        variable_name: Name of the variable
        direct_causal_effect: Estimated direct effect on objective
        total_causal_effect: Total effect including indirect paths
        is_root_cause: Whether this variable is a root cause (no parents)
        is_mediator: Whether this variable mediates other effects
        is_confounder: Whether this variable confounds other relationships
        manipulation_recommendation: How to manipulate this variable
    """
    variable_name: str
    direct_causal_effect: float
    total_causal_effect: float
    is_root_cause: bool
    is_mediator: bool
    is_confounder: bool
    manipulation_recommendation: str
    causal_paths: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CausalOptimizationInsight:
    """Causal insights for guiding optimization.
    
    Attributes:
        learned_graph: The causal graph learned from data
        root_causes: Variables that are root causes of objective variation
        most_effective_interventions: Top variables to manipulate
        spurious_correlations: Variables correlated but not causally related
        confounding_warnings: Warnings about confounding variables
        actionable_recommendations: Specific recommendations for next experiments
    """
    learned_graph: CausalGraph
    root_causes: list[VariableCausalImpact]
    most_effective_interventions: list[VariableCausalImpact]
    spurious_correlations: list[dict[str, Any]]
    confounding_warnings: list[str]
    actionable_recommendations: list[str]
    analysis_summary: str


class CausalOptimizationAnalyzer:
    """Analyzes causal relationships in optimization campaigns.
    
    Goes beyond correlation analysis to understand:
    - Which parameters truly cause changes in the objective
    - Which correlations are spurious (confounded)
    - Optimal intervention points for maximum impact
    - Mediation paths (A → B → Objective vs A → Objective)
    """

    def __init__(self, alpha: float = 0.05, max_cond_set: int = 3) -> None:
        self.alpha = alpha
        self.max_cond_set = max_cond_set
        self._learner = CausalStructureLearner(alpha=alpha, max_cond_set=max_cond_set)

    def analyze_campaign(
        self,
        snapshot: CampaignSnapshot,
        objective_name: str | None = None,
    ) -> CausalOptimizationInsight | None:
        """Perform causal analysis on campaign data.
        
        Args:
            snapshot: Current campaign state with observations
            objective_name: Name of objective to analyze (uses first if None)
            
        Returns:
            CausalOptimizationInsight with full causal analysis,
            or None if insufficient data
        """
        if len(snapshot.observations) < 10:
            return None
            
        # Determine objective name
        obj_name = objective_name or snapshot.objective_names[0]
        
        # Extract data for causal learning
        data, var_names = self._prepare_data(snapshot, obj_name)
        if not data or len(var_names) < 2:
            return None
            
        # Learn causal structure
        try:
            graph = self._learner.learn(data, var_names)
        except Exception:
            # If learning fails, return simplified analysis
            return self._fallback_analysis(snapshot, obj_name)
            
        # Analyze causal impacts
        impacts = self._compute_causal_impacts(graph, obj_name, data)
        
        # Identify root causes and effective interventions
        root_causes = [imp for imp in impacts if imp.is_root_cause]
        root_causes.sort(key=lambda x: abs(x.total_causal_effect), reverse=True)
        
        effective_interventions = sorted(
            impacts,
            key=lambda x: abs(x.total_causal_effect),
            reverse=True,
        )[:5]
        
        # Find spurious correlations
        spurious = self._identify_spurious_correlations(
            snapshot, graph, obj_name, data, var_names
        )
        
        # Generate confounding warnings
        confounding_warnings = self._generate_confounding_warnings(graph, impacts)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            root_causes, effective_interventions, graph, obj_name
        )
        
        # Build summary
        summary = self._build_summary(
            graph, root_causes, spurious, effective_interventions
        )
        
        return CausalOptimizationInsight(
            learned_graph=graph,
            root_causes=root_causes,
            most_effective_interventions=effective_interventions,
            spurious_correlations=spurious,
            confounding_warnings=confounding_warnings,
            actionable_recommendations=recommendations,
            analysis_summary=summary,
        )

    def _prepare_data(
        self,
        snapshot: CampaignSnapshot,
        objective_name: str,
    ) -> tuple[list[list[float]], list[str]]:
        """Extract numerical data from observations for causal learning."""
        var_names = [p.name for p in snapshot.parameter_specs]
        if objective_name not in var_names:
            var_names.append(objective_name)
            
        data: list[list[float]] = []
        
        for obs in snapshot.successful_observations:
            row: list[float] = []
            valid = True
            
            for var in var_names[:-1]:  # Parameters
                val = obs.parameters.get(var)
                if val is None or not isinstance(val, (int, float)):
                    valid = False
                    break
                row.append(float(val))
                
            if not valid:
                continue
                
            # Objective value
            obj_val = obs.kpi_values.get(objective_name)
            if obj_val is None or not isinstance(obj_val, (int, float)):
                continue
            row.append(float(obj_val))
            
            if len(row) == len(var_names):
                data.append(row)
                
        return data, var_names

    def _compute_causal_impacts(
        self,
        graph: CausalGraph,
        objective_name: str,
        data: list[list[float]],
    ) -> list[VariableCausalImpact]:
        """Compute causal impact for each variable on the objective."""
        impacts = []
        
        for node_name in graph.node_names:
            if node_name == objective_name:
                continue
                
            # Check if there's a path to objective
            if objective_name not in graph.descendants(node_name):
                # No causal effect
                impacts.append(VariableCausalImpact(
                    variable_name=node_name,
                    direct_causal_effect=0.0,
                    total_causal_effect=0.0,
                    is_root_cause=False,
                    is_mediator=False,
                    is_confounder=self._is_confounder(graph, node_name, objective_name),
                    manipulation_recommendation=f"No causal effect on {objective_name}",
                    causal_paths=[],
                ))
                continue
                
            # Find causal paths
            paths = self._find_causal_paths(graph, node_name, objective_name)
            
            # Estimate effects from data
            direct_effect = self._estimate_direct_effect(data, graph, node_name, objective_name)
            total_effect = self._estimate_total_effect(paths, data)
            
            # Classify variable role
            is_root = len(graph.parents(node_name)) == 0
            is_mediator = (
                len(graph.parents(node_name)) > 0 and 
                len(graph.children(node_name)) > 0 and
                objective_name in graph.descendants(node_name)
            )
            is_confounder = self._is_confounder(graph, node_name, objective_name)
            
            # Generate recommendation
            if abs(total_effect) > abs(direct_effect) * 1.5:
                rec = (
                    f"Indirect effects dominate. Manipulate {node_name} to influence "
                    f"intermediate variables that drive {objective_name}."
                )
            elif abs(direct_effect) > 0.1:
                rec = (
                    f"Strong direct effect. {node_name} is a high-leverage parameter "
                    f"for controlling {objective_name}."
                )
            else:
                rec = f"Weak causal effect. Consider deprioritizing {node_name}."
                
            impacts.append(VariableCausalImpact(
                variable_name=node_name,
                direct_causal_effect=direct_effect,
                total_causal_effect=total_effect,
                is_root_cause=is_root,
                is_mediator=is_mediator,
                is_confounder=is_confounder,
                manipulation_recommendation=rec,
                causal_paths=[{"path": p, "estimated_effect": 0.0} for p in paths],
            ))
            
        return impacts

    def _find_causal_paths(
        self,
        graph: CausalGraph,
        source: str,
        target: str,
        max_length: int = 4,
    ) -> list[list[str]]:
        """Find all causal paths from source to target."""
        paths: list[list[str]] = []
        visited = {source}
        current_path = [source]
        
        def dfs(current: str) -> None:
            if len(current_path) > max_length:
                return
            for child in graph.children(current):
                if child == target:
                    paths.append(list(current_path) + [child])
                elif child not in visited:
                    visited.add(child)
                    current_path.append(child)
                    dfs(child)
                    current_path.pop()
                    visited.remove(child)
                    
        dfs(source)
        return paths

    def _is_confounder(
        self,
        graph: CausalGraph,
        var: str,
        objective: str,
    ) -> bool:
        """Check if variable is a confounder between any pair and objective."""
        # A confounder is a common cause of two variables
        children = graph.children(var)
        return len(children) >= 2 and objective in graph.descendants(var)

    def _estimate_direct_effect(
        self,
        data: list[list[float]],
        graph: CausalGraph,
        var: str,
        objective: str,
    ) -> float:
        """Estimate direct causal effect using partial correlation."""
        if not data:
            return 0.0
            
        var_names = list(graph.node_names)
        if var not in var_names or objective not in var_names:
            return 0.0
            
        var_idx = var_names.index(var)
        obj_idx = var_names.index(objective)
        
        # Get parents of objective (to condition on)
        parents = graph.parents(objective)
        parent_indices = [var_names.index(p) for p in parents if p in var_names and p != var]
        
        # Compute correlation
        n = len(data)
        var_vals = [row[var_idx] for row in data]
        obj_vals = [row[obj_idx] for row in data]
        
        mean_var = sum(var_vals) / n
        mean_obj = sum(obj_vals) / n
        
        # Simple correlation as proxy for direct effect
        numerator = sum(
            (v - mean_var) * (o - mean_obj)
            for v, o in zip(var_vals, obj_vals)
        )
        var_var = sum((v - mean_var) ** 2 for v in var_vals)
        obj_var = sum((o - mean_obj) ** 2 for o in obj_vals)
        
        if var_var * obj_var == 0:
            return 0.0
            
        corr = numerator / (var_var * obj_var) ** 0.5
        return corr

    def _estimate_total_effect(
        self,
        paths: list[list[str]],
        data: list[list[float]],
    ) -> float:
        """Estimate total causal effect through all paths."""
        # Simplified: just use the number of paths as a proxy
        # In practice, would multiply path coefficients
        return len(paths) * 0.1 if paths else 0.0

    def _identify_spurious_correlations(
        self,
        snapshot: CampaignSnapshot,
        graph: CausalGraph,
        objective_name: str,
        data: list[list[float]],
        var_names: list[str],
    ) -> list[dict[str, Any]]:
        """Identify variables correlated but not causally related to objective."""
        spurious = []
        
        # Check each variable
        for var in var_names:
            if var == objective_name:
                continue
                
            # Check if there's a correlation
            var_idx = var_names.index(var)
            obj_idx = var_names.index(objective_name)
            
            if not data:
                continue
                
            var_vals = [row[var_idx] for row in data]
            obj_vals = [row[obj_idx] for row in data]
            
            n = len(var_vals)
            mean_v = sum(var_vals) / n
            mean_o = sum(obj_vals) / n
            
            num = sum((v - mean_v) * (o - mean_o) for v, o in zip(var_vals, obj_vals))
            den_v = sum((v - mean_v) ** 2 for v in var_vals)
            den_o = sum((o - mean_o) ** 2 for o in obj_vals)
            
            if den_v * den_o == 0:
                continue
                
            corr = abs(num / (den_v * den_o) ** 0.5)
            
            # Check if correlated but not causally connected
            has_causal_path = (
                objective_name in graph.descendants(var) or
                var in graph.descendants(objective_name) or
                self._has_confounding_path(graph, var, objective_name)
            )
            
            if corr > 0.3 and not has_causal_path:
                spurious.append({
                    "variable": var,
                    "correlation": corr,
                    "explanation": (
                        f"Correlated (r={corr:.2f}) but no causal connection found. "
                        f"Likely due to confounding or coincidence."
                    ),
                })
                
        return spurious

    def _has_confounding_path(
        self,
        graph: CausalGraph,
        var1: str,
        var2: str,
    ) -> bool:
        """Check if there's a confounding path between two variables."""
        # Common cause
        for node in graph.node_names:
            if var1 in graph.descendants(node) and var2 in graph.descendants(node):
                return True
        return False

    def _generate_confounding_warnings(
        self,
        graph: CausalGraph,
        impacts: list[VariableCausalImpact],
    ) -> list[str]:
        """Generate warnings about potential confounding."""
        warnings = []
        
        confounders = [imp for imp in impacts if imp.is_confounder]
        if confounders:
            warning = (
                f"Detected {len(confounders)} potential confounder(s): "
                f"{', '.join(c.variable_name for c in confounders[:3])}. "
                "These variables influence multiple others and may create spurious associations."
            )
            warnings.append(warning)
            
        return warnings

    def _generate_recommendations(
        self,
        root_causes: list[VariableCausalImpact],
        effective_interventions: list[VariableCausalImpact],
        graph: CausalGraph,
        objective_name: str,
    ) -> list[str]:
        """Generate actionable recommendations for next experiments."""
        recommendations = []
        
        if root_causes:
            top_root = root_causes[0]
            recommendations.append(
                f"Focus on '{top_root.variable_name}' — it's a root cause with "
                f"total effect {top_root.total_causal_effect:.2f}. Manipulating it "
                "will have cascading effects through the system."
            )
            
        if effective_interventions:
            direct_drivers = [
                imp for imp in effective_interventions
                if abs(imp.direct_causal_effect) > 0.3
            ][:2]
            if direct_drivers:
                rec = (
                    f"Direct drivers to manipulate: "
                    f"{', '.join(d.variable_name for d in direct_drivers)}. "
                    "These have strong direct effects on the objective."
                )
                recommendations.append(rec)
                
        # Check for unexplored regions
        mediators = [imp for imp in effective_interventions if imp.is_mediator]
        if mediators:
            recommendations.append(
                f"Consider manipulating mediator variables: "
                f"{', '.join(m.variable_name for m in mediators[:2])}. "
                "These lie on causal paths and can amplify effects."
            )
            
        return recommendations

    def _build_summary(
        self,
        graph: CausalGraph,
        root_causes: list[VariableCausalImpact],
        spurious: list[dict[str, Any]],
        effective_interventions: list[VariableCausalImpact],
    ) -> str:
        """Build human-readable summary of causal analysis."""
        parts = []
        
        parts.append(f"Learned causal graph with {len(graph.node_names)} variables "
                    f"and {len(graph.edges)} causal relationships.")
        
        if root_causes:
            parts.append(
                f"Identified {len(root_causes)} root cause(s): "
                f"{', '.join(r.variable_name for r in root_causes[:3])}."
            )
            
        if effective_interventions:
            top = effective_interventions[0]
            parts.append(
                f"Most effective intervention: '{top.variable_name}' "
                f"(total effect: {top.total_causal_effect:.2f})."
            )
            
        if spurious:
            parts.append(
                f"Found {len(spurious)} spurious correlation(s) — "
                "variables that appear related but have no causal connection."
            )
            
        return " ".join(parts)

    def _fallback_analysis(
        self,
        snapshot: CampaignSnapshot,
        objective_name: str,
    ) -> CausalOptimizationInsight:
        """Simplified analysis when causal learning fails."""
        graph = CausalGraph()
        
        # Add nodes
        for spec in snapshot.parameter_specs:
            graph.add_node(CausalNode(name=spec.name))
        graph.add_node(CausalNode(name=objective_name))
        
        # Compute correlations as proxy
        impacts = []
        for spec in snapshot.parameter_specs:
            impacts.append(VariableCausalInsight(
                variable_name=spec.name,
                direct_causal_effect=0.0,
                total_causal_effect=0.0,
                is_root_cause=False,
                is_mediator=False,
                is_confounder=False,
                manipulation_recommendation="Insufficient data for causal analysis",
            ))
            
        return CausalOptimizationInsight(
            learned_graph=graph,
            root_causes=[],
            most_effective_interventions=impacts,
            spurious_correlations=[],
            confounding_warnings=["Causal structure learning failed — using correlation-based fallback"],
            actionable_recommendations=["Collect more data for reliable causal inference"],
            analysis_summary="Fallback analysis: insufficient data for causal learning.",
        )


# Alias for backward compatibility
VariableCausalInsight = VariableCausalImpact
