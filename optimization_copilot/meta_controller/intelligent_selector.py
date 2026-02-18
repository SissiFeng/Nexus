"""Intelligent algorithm selector with explainable decision-making.

Analyzes campaign data characteristics (noise level, dimensionality, constraints)
to automatically select and switch algorithms with human-readable explanations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    NoiseRegime,
    Phase,
    ProblemFingerprint,
    VariableType,
)


@dataclass
class AlgorithmRecommendation:
    """A recommendation for which algorithm to use, with full explanation.
    
    Attributes:
        algorithm: The recommended algorithm name (e.g., "tpe", "cma_es", "gp_bo")
        confidence: Confidence score 0-1 based on historical performance match
        reason: Human-readable explanation of why this algorithm was chosen
        trade_offs: Description of trade-offs for this choice
        expected_performance: Expected performance characteristics
        when_to_switch: Conditions under which to consider switching algorithms
    """
    algorithm: str
    confidence: float
    reason: str
    trade_offs: str
    expected_performance: str
    when_to_switch: str
    data_characteristics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmSwitchExplanation:
    """Explanation of why an algorithm switch occurred or should occur.
    
    Attributes:
        from_algorithm: Previous algorithm name
        to_algorithm: New algorithm name
        trigger: What triggered the switch (e.g., "noise_regime_change", "stagnation")
        explanation: Detailed human-readable explanation
        confidence_gain: Expected performance improvement from switching
    """
    from_algorithm: str
    to_algorithm: str
    trigger: str
    explanation: str
    confidence_gain: float
    supporting_evidence: dict[str, Any] = field(default_factory=dict)


class IntelligentAlgorithmSelector:
    """Selects optimization algorithms based on data characteristics.
    
    Analyzes:
    - Noise level (signal-to-noise ratio)
    - Problem dimensionality and variable types
    - Number and type of constraints
    - Current optimization phase
    - Historical performance data
    
    Provides explainable recommendations for algorithm selection and switching.
    """

    # Algorithm characteristics and when they work best
    ALGORITHM_PROFILES: dict[str, dict[str, Any]] = {
        "tpe": {
            "name": "Tree-structured Parzen Estimator",
            "best_for": ["mixed_variables", "conditional_spaces", "medium_noise"],
            "avoids": ["high_dimensional_continuous", "very_high_noise"],
            "strengths": ["sample_efficient", "handles_discrete", "robust"],
            "weaknesses": ["struggles_high_dim", "slower_convergence_continuous"],
            "ideal_noise": ("low", "medium"),
            "ideal_dim": (1, 50),
        },
        "cma_es": {
            "name": "Covariance Matrix Adaptation",
            "best_for": ["continuous", "non_separable", "unimodal", "low_noise"],
            "avoids": ["mixed_variables", "high_noise", "many_local_optima"],
            "strengths": ["fast_convergence", "rotation_invariant", "gradient_free"],
            "weaknesses": ["needs_many_samples", "struggles_discrete", "noise_sensitive"],
            "ideal_noise": ("low",),
            "ideal_dim": (2, 100),
        },
        "gp_bo": {
            "name": "Gaussian Process Bayesian Optimization",
            "best_for": ["low_dim", "expensive_evals", "low_noise", "continuous"],
            "avoids": ["high_dim", "high_noise", "large_scale"],
            "strengths": ["sample_efficient", "good_uncertainty", "smooth_functions"],
            "weaknesses": ["cubic_scaling", "struggles_noise", "struggles_discrete"],
            "ideal_noise": ("low", "medium"),
            "ideal_dim": (1, 20),
        },
        "random": {
            "name": "Random Search",
            "best_for": ["high_noise", "high_dim", "baseline", "exploration"],
            "avoids": [],
            "strengths": ["simple", "robust", "parallelizable", "no_hyperparams"],
            "weaknesses": ["no_exploitation", "slow_convergence"],
            "ideal_noise": ("low", "medium", "high"),
            "ideal_dim": (1, 1000),
        },
        "latin_hypercube": {
            "name": "Latin Hypercube Sampling",
            "best_for": ["initial_sampling", "space_filling", "uniform_coverage"],
            "avoids": ["exploitation_phase"],
            "strengths": ["space_filling", "deterministic", "good_coverage"],
            "weaknesses": ["not_adaptive", "no_exploitation"],
            "ideal_noise": ("low", "medium", "high"),
            "ideal_dim": (1, 100),
        },
        "random_forest_surrogate": {
            "name": "Random Forest Surrogate",
            "best_for": ["mixed_variables", "high_noise", "non_smooth", "categorical"],
            "avoids": ["very_high_dim_continuous"],
            "strengths": ["handles_mixed", "noise_robust", "parallelizable"],
            "weaknesses": ["less_sample_efficient", "poor_uncertainty"],
            "ideal_noise": ("medium", "high"),
            "ideal_dim": (1, 100),
        },
        "nsga2": {
            "name": "NSGA-II (Multi-objective)",
            "best_for": ["multi_objective", "pareto_front", "diverse_solutions"],
            "avoids": ["single_objective"],
            "strengths": ["pareto_diverse", "robust", "well_tested"],
            "weaknesses": ["population_required", "slow_convergence"],
            "ideal_noise": ("low", "medium", "high"),
            "ideal_dim": (1, 100),
        },
    }

    def __init__(self) -> None:
        self._selection_history: list[dict[str, Any]] = []

    def analyze_data_characteristics(
        self,
        snapshot: CampaignSnapshot,
        diagnostics: dict[str, float],
        fingerprint: ProblemFingerprint,
    ) -> dict[str, Any]:
        """Extract key data characteristics for algorithm selection.
        
        Returns a dictionary with:
        - noise_regime: estimated noise level
        - n_dimensions: number of parameters
        - variable_type: continuous/discrete/mixed
        - n_constraints: number of constraints
        - n_observations: sample size
        - phase: current optimization phase
        """
        n_dim = len(snapshot.parameter_specs)
        
        # Estimate noise regime from diagnostics
        noise_estimate = diagnostics.get("noise_estimate", 0.1)
        snr = diagnostics.get("snr", 10.0)
        
        # Use fingerprint noise regime if available
        if hasattr(fingerprint, 'noise_regime') and fingerprint.noise_regime:
            from optimization_copilot.core.models import NoiseRegime
            if isinstance(fingerprint.noise_regime, NoiseRegime):
                noise_regime = fingerprint.noise_regime.value
            else:
                noise_regime = str(fingerprint.noise_regime)
        elif noise_estimate > 0.5 or snr < 2:
            noise_regime = "high"
        elif noise_estimate > 0.2 or snr < 5:
            noise_regime = "medium"
        else:
            noise_regime = "low"
            
        # Variable type analysis
        var_types = set()
        for spec in snapshot.parameter_specs:
            var_types.add(spec.type.value if hasattr(spec.type, 'value') else str(spec.type))
        
        if len(var_types) == 1 and "continuous" in var_types:
            var_type = "continuous"
        elif "categorical" in var_types or len(var_types) > 1:
            var_type = "mixed"
        else:
            var_type = "discrete"
            
        # Constraint analysis
        n_constraints = len(snapshot.constraints)
        has_inequality = any(c.get("type") == "inequality" for c in snapshot.constraints)
        has_equality = any(c.get("type") == "equality" for c in snapshot.constraints)
        
        return {
            "noise_regime": noise_regime,
            "noise_estimate": noise_estimate,
            "snr": snr,
            "n_dimensions": n_dim,
            "variable_type": var_type,
            "variable_types_detail": list(var_types),
            "n_constraints": n_constraints,
            "has_inequality_constraints": has_inequality,
            "has_equality_constraints": has_equality,
            "n_observations": snapshot.n_observations,
            "n_failures": snapshot.n_failures,
            "failure_rate": snapshot.failure_rate,
        }

    def score_algorithm_fit(
        self,
        algorithm: str,
        characteristics: dict[str, Any],
        phase: Phase,
    ) -> float:
        """Score how well an algorithm fits the problem characteristics.
        
        Returns a score between 0 and 1, where 1 is perfect fit.
        """
        profile = self.ALGORITHM_PROFILES.get(algorithm, {})
        if not profile:
            return 0.0
            
        scores = []
        
        # Noise fit
        noise = characteristics["noise_regime"]
        ideal_noise = profile.get("ideal_noise", ("low", "medium", "high"))
        if noise in ideal_noise:
            scores.append(1.0)
        elif len(ideal_noise) == 2 and noise in ["low", "medium", "high"]:
            scores.append(0.5)  # Partial match
        else:
            scores.append(0.2)  # Poor match
            
        # Dimension fit
        dim = characteristics["n_dimensions"]
        ideal_dim = profile.get("ideal_dim", (1, 100))
        if ideal_dim[0] <= dim <= ideal_dim[1]:
            scores.append(1.0)
        elif dim < ideal_dim[0]:
            scores.append(0.8)  # Underspecified is ok
        else:
            # Penalize for exceeding ideal dimensions
            excess = dim - ideal_dim[1]
            penalty = max(0, 1 - excess / 50)  # Graceful degradation
            scores.append(penalty)
            
        # Variable type fit
        var_type = characteristics["variable_type"]
        best_for = profile.get("best_for", [])
        avoids = profile.get("avoids", [])
        
        if var_type in best_for or f"{var_type}_variables" in best_for:
            scores.append(1.0)
        elif any(var_type in b for b in best_for):
            scores.append(0.8)
        elif var_type in avoids or f"{var_type}_variables" in avoids:
            scores.append(0.1)
        else:
            scores.append(0.6)  # Neutral
            
        # Phase alignment
        if phase == Phase.COLD_START and algorithm in ["latin_hypercube", "random"]:
            scores.append(1.0)
        elif phase == Phase.EXPLOITATION and algorithm in ["tpe", "cma_es", "gp_bo"]:
            scores.append(1.0)
        elif phase == Phase.STAGNATION and algorithm in ["random", "latin_hypercube"]:
            scores.append(1.0)
        else:
            scores.append(0.7)
            
        return sum(scores) / len(scores)

    def recommend_algorithm(
        self,
        snapshot: CampaignSnapshot,
        diagnostics: dict[str, float],
        fingerprint: ProblemFingerprint,
        phase: Phase,
        available_algorithms: list[str] | None = None,
    ) -> AlgorithmRecommendation:
        """Recommend the best algorithm with a full explanation.
        
        This is the main entry point for intelligent algorithm selection.
        """
        available = available_algorithms or list(self.ALGORITHM_PROFILES.keys())
        
        # Analyze data characteristics
        chars = self.analyze_data_characteristics(snapshot, diagnostics, fingerprint)
        
        # Score all algorithms
        scored_algorithms = [
            (alg, self.score_algorithm_fit(alg, chars, phase))
            for alg in available
        ]
        scored_algorithms.sort(key=lambda x: x[1], reverse=True)
        
        best_alg, best_score = scored_algorithms[0]
        profile = self.ALGORITHM_PROFILES[best_alg]
        
        # Build explanation
        reason_parts = [
            f"Selected '{profile['name']}' based on:",
            f"  • Noise level: {chars['noise_regime']} (SNR ≈ {chars['snr']:.1f})",
            f"  • Dimensions: {chars['n_dimensions']} ({chars['variable_type']} variables)",
        ]
        
        if chars['n_constraints'] > 0:
            reason_parts.append(f"  • Constraints: {chars['n_constraints']}")
            
        reason_parts.append(f"  • Optimization phase: {phase.value}")
        
        # Build trade-offs description
        trade_offs = f"Strengths: {', '.join(profile.get('strengths', []))}. "
        trade_offs += f"Limitations: {', '.join(profile.get('weaknesses', []))}."
        
        # Build expected performance
        if best_score > 0.8:
            expected = "Excellent fit. Expected fast convergence and good final performance."
        elif best_score > 0.6:
            expected = "Good fit. Expected reasonable convergence with some compromises."
        else:
            expected = "Moderate fit. May need monitoring for potential issues."
            
        # Build switch recommendations
        if chars["noise_regime"] == "high":
            switch_trigger = "If noise decreases significantly, consider switching to TPE or GP-BO"
        elif chars["n_dimensions"] > 50:
            switch_trigger = "If dimensionality reduction becomes possible, consider CMA-ES or GP-BO"
        else:
            switch_trigger = "Monitor for stagnation; if detected, switch to random or Latin hypercube"
            
        # Record selection history
        self._selection_history.append({
            "algorithm": best_alg,
            "score": best_score,
            "characteristics": chars,
            "phase": phase.value,
        })
        
        return AlgorithmRecommendation(
            algorithm=best_alg,
            confidence=best_score,
            reason="\n".join(reason_parts),
            trade_offs=trade_offs,
            expected_performance=expected,
            when_to_switch=switch_trigger,
            data_characteristics=chars,
        )

    def explain_switch(
        self,
        current_algorithm: str,
        snapshot: CampaignSnapshot,
        diagnostics: dict[str, float],
        fingerprint: ProblemFingerprint,
        phase: Phase,
    ) -> AlgorithmSwitchExplanation | None:
        """Explain whether and why to switch algorithms.
        
        Returns None if no switch is recommended, or an explanation if a switch
        would be beneficial.
        """
        # Get current recommendation
        recommendation = self.recommend_algorithm(
            snapshot, diagnostics, fingerprint, phase
        )
        
        # If already using the best algorithm, no switch needed
        if recommendation.algorithm == current_algorithm:
            return None
            
        # Determine what triggered the switch recommendation
        chars = recommendation.data_characteristics
        current_profile = self.ALGORITHM_PROFILES.get(current_algorithm, {})
        
        triggers = []
        
        # Noise-based trigger
        noise = chars["noise_regime"]
        ideal_noise = current_profile.get("ideal_noise", ("low",))
        if noise not in ideal_noise and len(ideal_noise) < 3:
            if noise == "high" and "high" not in ideal_noise:
                triggers.append("high_noise_detected")
            elif noise == "low" and "low" in ["low"] and "low" not in ideal_noise:
                triggers.append("low_noise_opportunity")
                
        # Phase-based trigger
        if phase == Phase.STAGNATION and current_algorithm not in ["random", "latin_hypercube"]:
            triggers.append("stagnation_detected")
            
        # Dimensionality trigger
        dim = chars["n_dimensions"]
        ideal_dim = current_profile.get("ideal_dim", (1, 100))
        if dim > ideal_dim[1]:
            triggers.append("dimensionality_mismatch")
            
        if not triggers:
            triggers.append("better_fit_available")
            
        trigger = triggers[0] if triggers else "performance_optimization"
        
        # Build explanation
        new_profile = self.ALGORITHM_PROFILES.get(recommendation.algorithm, {})
        
        explanation = (
            f"Switching from '{current_profile.get('name', current_algorithm)}' to "
            f"'{new_profile.get('name', recommendation.algorithm)}' because:\n"
        )
        
        if "high_noise_detected" in triggers:
            explanation += (
                f"• Current algorithm struggles with high noise (SNR ≈ {chars['snr']:.1f}). "
                f"'{new_profile.get('name')}' is more robust to noisy evaluations."
            )
        elif "stagnation_detected" in triggers:
            explanation += (
                "• Optimization has stagnated. Switching to exploration-focused "
                "algorithm to escape local optima."
            )
        elif "dimensionality_mismatch" in triggers:
            explanation += (
                f"• Problem dimensionality ({dim}D) exceeds ideal range for current algorithm. "
                f"'{new_profile.get('name')}' handles high dimensions better."
            )
        else:
            explanation += (
                f"• '{new_profile.get('name')}' is a {int((recommendation.confidence - 0.5) * 200)}% "
                f"better fit for current problem characteristics."
            )
            
        # Estimate confidence gain
        current_score = self.score_algorithm_fit(current_algorithm, chars, phase)
        confidence_gain = max(0, recommendation.confidence - current_score)
        
        return AlgorithmSwitchExplanation(
            from_algorithm=current_algorithm,
            to_algorithm=recommendation.algorithm,
            trigger=trigger,
            explanation=explanation,
            confidence_gain=confidence_gain,
            supporting_evidence={
                "current_score": current_score,
                "recommended_score": recommendation.confidence,
                "data_characteristics": chars,
                "available_alternatives": [
                    alg for alg, _ in [
                        (recommendation.algorithm, recommendation.confidence),
                    ]
                ],
            },
        )

    def get_selection_history(self) -> list[dict[str, Any]]:
        """Return the history of algorithm selections."""
        return list(self._selection_history)
