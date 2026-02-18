"""Enhanced multi-fidelity optimization with adaptive fidelity selection.

Manages the full pipeline from cheap simulations → expensive experiments,
with intelligent promotion strategies and cost-aware planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from optimization_copilot.core.models import CampaignSnapshot, Observation
from optimization_copilot.multi_fidelity.planner import (
    FidelityLevel,
    FidelityPlan,
    MultiFidelityPlanner,
    MultiFidelityPlan,
)


@dataclass
class FidelityGateResult:
    """Result of passing a candidate through a fidelity gate.
    
    Attributes:
        candidate: The parameter configuration
        passed: Whether it passed the gate
        score: Score at this fidelity level
        estimated_high_fidelity_score: Predicted score at highest fidelity
        uncertainty: Uncertainty in the prediction
        cost_incurred: Cost spent to evaluate
        next_fidelity: Which fidelity to evaluate at next (if any)
    """
    candidate: dict[str, float]
    passed: bool
    score: float
    estimated_high_fidelity_score: float
    uncertainty: float
    cost_incurred: float
    next_fidelity: str | None


@dataclass
class AdaptiveFidelityState:
    """State tracking for adaptive multi-fidelity optimization.
    
    Attributes:
        current_stage: Current fidelity stage (0 = lowest)
        budget_spent: Total budget consumed so far
        budget_remaining: Remaining budget
        candidates_at_stage: Map of stage -> candidates being evaluated
        promoted_candidates: Candidates promoted to higher fidelity
        discarded_candidates: Candidates eliminated at lower fidelity
        fidelity_correlation: Estimated correlation between fidelity levels
    """
    current_stage: int = 0
    budget_spent: float = 0.0
    budget_remaining: float = 0.0
    candidates_at_stage: dict[int, list[dict[str, float]]] = field(default_factory=dict)
    promoted_candidates: list[dict[str, float]] = field(default_factory=list)
    discarded_candidates: list[tuple[dict[str, float], str]] = field(default_factory=list)
    fidelity_correlation: dict[tuple[int, int], float] = field(default_factory=dict)
    
    @property
    def n_candidates_evaluated(self) -> int:
        """Total number of candidates evaluated across all stages."""
        return sum(len(c) for c in self.candidates_at_stage.values())
    
    @property
    def promotion_rate(self) -> float:
        """Fraction of candidates promoted to higher fidelity."""
        total = len(self.promoted_candidates) + len(self.discarded_candidates)
        if total == 0:
            return 0.0
        return len(self.promoted_candidates) / total


class EnhancedMultiFidelityOptimizer:
    """Enhanced multi-fidelity optimizer with adaptive strategies.
    
    Features:
    - Adaptive fidelity selection based on correlation estimates
    - Cost-aware candidate promotion with uncertainty quantification
    - Dynamic budget reallocation between stages
    - Transfer learning between fidelity levels
    - Early stopping for unpromising candidates
    """

    def __init__(
        self,
        fidelity_levels: list[FidelityLevel] | None = None,
        cost_budget: float | None = None,
    ) -> None:
        self.planner = MultiFidelityPlanner(fidelity_levels)
        self.cost_budget = cost_budget
        self._state = AdaptiveFidelityState(budget_remaining=cost_budget or float('inf'))
        self._promotion_thresholds: dict[int, float] = {}
        
    def initialize_campaign(
        self,
        n_candidates: int,
        budget: float | None = None,
    ) -> MultiFidelityPlan:
        """Initialize a new multi-fidelity campaign.
        
        Args:
            n_candidates: Initial pool of candidates
            budget: Total cost budget (overrides constructor value)
            
        Returns:
            MultiFidelityPlan with staged evaluation strategy
        """
        self.cost_budget = budget or self.cost_budget
        
        # Create initial plan
        dummy_snapshot = CampaignSnapshot(
            campaign_id="mf_init",
            parameter_specs=[],
            observations=[],
            objective_names=["objective"],
            objective_directions=["maximize"],
        )
        
        plan = self.planner.plan(
            snapshot=dummy_snapshot,
            budget=self.cost_budget,
            n_total=n_candidates,
        )
        
        # Initialize state
        self._state = AdaptiveFidelityState(
            budget_remaining=self.cost_budget or float('inf'),
            candidates_at_stage={0: []},
        )
        
        return plan

    def evaluate_at_fidelity(
        self,
        candidate: dict[str, float],
        fidelity_idx: int,
        evaluator: Callable[[dict[str, float], int], float],
    ) -> FidelityGateResult:
        """Evaluate a candidate at a specific fidelity level.
        
        Args:
            candidate: Parameter configuration
            fidelity_idx: Index of fidelity level to evaluate at
            evaluator: Function (params, fidelity_idx) -> score
            
        Returns:
            FidelityGateResult with evaluation outcome
        """
        fidelity = self.planner.fidelity_levels[fidelity_idx]
        
        # Evaluate
        score = evaluator(candidate, fidelity_idx)
        cost = fidelity.cost_multiplier
        
        # Update state
        self._state.budget_spent += cost
        if self.cost_budget:
            self._state.budget_remaining = self.cost_budget - self._state.budget_spent
        
        # Track candidates at this stage
        if fidelity_idx not in self._state.candidates_at_stage:
            self._state.candidates_at_stage[fidelity_idx] = []
        self._state.candidates_at_stage[fidelity_idx].append(candidate)
            
        # Estimate high-fidelity score
        estimated_high, uncertainty = self._estimate_high_fidelity_score(
            score, fidelity_idx
        )
        
        # Determine if passes gate
        passed = self._passes_gate(score, fidelity_idx)
        
        # Determine next fidelity
        next_fidelity = None
        if passed and fidelity_idx < len(self.planner.fidelity_levels) - 1:
            next_fidelity = self.planner.fidelity_levels[fidelity_idx + 1].name
            
        return FidelityGateResult(
            candidate=candidate,
            passed=passed,
            score=score,
            estimated_high_fidelity_score=estimated_high,
            uncertainty=uncertainty,
            cost_incurred=cost,
            next_fidelity=next_fidelity,
        )

    def adaptive_promotion_strategy(
        self,
        candidates: list[dict[str, float]],
        observations_by_fidelity: dict[int, list[tuple[dict[str, float], float]]],
        current_stage: int,
    ) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        """Adaptively decide which candidates to promote.
        
        Uses Thompson sampling-style approach with uncertainty to balance
        exploration (promote uncertain but potentially good candidates) vs
        exploitation (promote known good candidates).
        
        Args:
            candidates: Candidates at current stage
            observations_by_fidelity: Map of fidelity -> (params, score) list
            current_stage: Current fidelity stage index
            
        Returns:
            Tuple of (promoted candidates, discarded candidates)
        """
        if not candidates:
            return [], []
            
        # Get observations at current fidelity
        current_obs = observations_by_fidelity.get(current_stage, [])
        
        if len(current_obs) < 5:
            # Not enough data, use fixed threshold
            return self._fixed_threshold_promotion(candidates, current_obs, current_stage)
            
        # Estimate correlation with higher fidelity
        if current_stage + 1 in observations_by_fidelity:
            correlation = self._estimate_fidelity_correlation(
                observations_by_fidelity[current_stage],
                observations_by_fidelity[current_stage + 1],
            )
            self._state.fidelity_correlation[(current_stage, current_stage + 1)] = correlation
        
        # Score each candidate with uncertainty
        scored_candidates = []
        for candidate in candidates:
            score, uncertainty = self._score_with_uncertainty(
                candidate, current_obs
            )
            # Thompson sampling: sample from posterior
            import random
            sampled_score = random.gauss(score, uncertainty)
            scored_candidates.append((sampled_score, candidate))
            
        # Sort by sampled score
        scored_candidates.sort(reverse=True)
        
        # Determine promotion fraction based on budget and stage
        n_levels = len(self.planner.fidelity_levels)
        base_fraction = 0.5
        
        # Adjust based on remaining budget
        if self.cost_budget:
            budget_frac = self._state.budget_remaining / self.cost_budget
            if budget_frac < 0.3:
                base_fraction = 0.3  # Be more conservative with low budget
            elif budget_frac > 0.7:
                base_fraction = 0.6  # Can explore more with high budget
                
        # Adjust based on stage (promote more aggressively in early stages)
        stage_adjustment = 1 - (current_stage / max(1, n_levels - 1)) * 0.3
        promotion_fraction = base_fraction * stage_adjustment
        
        n_promote = max(1, int(len(scored_candidates) * promotion_fraction))
        
        promoted = [c for _, c in scored_candidates[:n_promote]]
        discarded = [c for _, c in scored_candidates[n_promote:]]
        
        return promoted, discarded

    def _estimate_high_fidelity_score(
        self,
        low_fidelity_score: float,
        fidelity_idx: int,
    ) -> tuple[float, float]:
        """Estimate what the score would be at highest fidelity.
        
        Returns:
            Tuple of (estimated_score, uncertainty)
        """
        n_levels = len(self.planner.fidelity_levels)
        if fidelity_idx >= n_levels - 1:
            return low_fidelity_score, 0.0
            
        # Check if we have correlation estimate
        corr_key = (fidelity_idx, n_levels - 1)
        correlation = self._state.fidelity_correlation.get(corr_key, 0.7)
        
        # Linear correction based on correlation
        # If correlation is 1, scores are identical
        # If correlation is 0, low-fidelity is uninformative
        target_fidelity = self.planner.fidelity_levels[-1]
        source_fidelity = self.planner.fidelity_levels[fidelity_idx]
        
        # Adjust for noise difference
        noise_ratio = source_fidelity.noise_multiplier / target_fidelity.noise_multiplier
        
        # Estimate with regression toward mean
        estimated = low_fidelity_score * correlation
        
        # Uncertainty increases with fidelity gap
        uncertainty = (1 - correlation) * noise_ratio * abs(low_fidelity_score)
        
        return estimated, uncertainty

    def _passes_gate(self, score: float, fidelity_idx: int) -> bool:
        """Check if a score passes the promotion threshold."""
        threshold = self._promotion_thresholds.get(fidelity_idx)
        if threshold is None:
            return True  # No threshold set, pass all
        return score >= threshold

    def _fixed_threshold_promotion(
        self,
        candidates: list[dict[str, float]],
        observations: list[tuple[dict[str, float], float]],
        stage: int,
    ) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        """Promote top fraction when insufficient data for adaptive strategy."""
        if not observations:
            # No observations yet, promote all to get initial data
            return candidates, []
            
        # Get scores for candidates that have been evaluated
        scores = [score for _, score in observations]
        if not scores:
            return candidates, []
            
        scores.sort(reverse=True)
        
        # Use 50th percentile as threshold
        threshold_idx = len(scores) // 2
        threshold = scores[threshold_idx] if threshold_idx < len(scores) else scores[-1]
        
        # Set threshold for future evaluations
        self._promotion_thresholds[stage] = threshold
        
        # Find candidates that meet threshold
        promoted = []
        discarded = []
        
        for candidate in candidates:
            # Find observation for this candidate
            cand_score = None
            for obs_params, score in observations:
                if obs_params == candidate:
                    cand_score = score
                    break
                    
            if cand_score is not None and cand_score >= threshold:
                promoted.append(candidate)
            else:
                discarded.append(candidate)
                
        return promoted, discarded

    def _estimate_fidelity_correlation(
        self,
        low_fidelity_obs: list[tuple[dict[str, float], float]],
        high_fidelity_obs: list[tuple[dict[str, float], float]],
    ) -> float:
        """Estimate correlation between two fidelity levels."""
        # Match observations by parameters
        pairs = []
        for low_params, low_score in low_fidelity_obs:
            for high_params, high_score in high_fidelity_obs:
                if low_params == high_params:
                    pairs.append((low_score, high_score))
                    break
                    
        if len(pairs) < 3:
            return 0.7  # Default correlation
            
        # Compute correlation
        n = len(pairs)
        low_scores = [p[0] for p in pairs]
        high_scores = [p[1] for p in pairs]
        
        mean_low = sum(low_scores) / n
        mean_high = sum(high_scores) / n
        
        num = sum(
            (l - mean_low) * (h - mean_high)
            for l, h in pairs
        )
        den_low = sum((l - mean_low) ** 2 for l in low_scores)
        den_high = sum((h - mean_high) ** 2 for h in high_scores)
        
        if den_low * den_high == 0:
            return 0.0
            
        correlation = num / (den_low * den_high) ** 0.5
        return max(0.0, min(1.0, correlation))  # Clamp to [0, 1]

    def _score_with_uncertainty(
        self,
        candidate: dict[str, float],
        observations: list[tuple[dict[str, float], float]],
    ) -> tuple[float, float]:
        """Score a candidate with uncertainty estimate.
        
        Uses k-NN style estimation from nearby observations.
        """
        # Find exact match
        for obs_params, score in observations:
            if obs_params == candidate:
                return score, 0.0
                
        # Find nearest neighbors (simplified: by parameter distance)
        distances = []
        for obs_params, score in observations:
            dist = self._parameter_distance(candidate, obs_params)
            distances.append((dist, score))
            
        if not distances:
            return 0.0, 1.0  # No data, high uncertainty
            
        distances.sort()
        
        # Use k nearest
        k = min(3, len(distances))
        nearest = distances[:k]
        
        # Weighted average by inverse distance
        weights = [1 / (d + 0.01) for d, _ in nearest]
        scores = [s for _, s in nearest]
        
        total_weight = sum(weights)
        weighted_score = sum(w * s for w, s in zip(weights, scores)) / total_weight
        
        # Uncertainty based on distance and variance
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        avg_distance = sum(d for d, _ in nearest) / len(nearest)
        
        uncertainty = (variance ** 0.5 + avg_distance) / 2
        
        return weighted_score, uncertainty

    def _parameter_distance(
        self,
        params1: dict[str, float],
        params2: dict[str, float],
    ) -> float:
        """Compute normalized Euclidean distance between parameter sets."""
        common_keys = set(params1.keys()) & set(params2.keys())
        if not common_keys:
            return float('inf')
            
        squared_diffs = []
        for key in common_keys:
            v1 = params1[key]
            v2 = params2[key]
            
            # Normalize by range (assume [-10, 10] if unknown)
            range_val = max(abs(v1), abs(v2), 1.0)
            normalized_diff = (v1 - v2) / range_val
            squared_diffs.append(normalized_diff ** 2)
            
        return (sum(squared_diffs) / len(squared_diffs)) ** 0.5

    def get_cost_efficiency_report(self) -> dict[str, Any]:
        """Generate a report on cost efficiency of multi-fidelity approach."""
        if not self.cost_budget:
            return {"error": "No budget was set"}
            
        budget_used_frac = self._state.budget_spent / self.cost_budget
        
        return {
            "budget_total": self.cost_budget,
            "budget_spent": self._state.budget_spent,
            "budget_remaining": self._state.budget_remaining,
            "budget_utilization": budget_used_frac,
            "candidates_evaluated": self._state.n_candidates_evaluated,
            "candidates_promoted": len(self._state.promoted_candidates),
            "candidates_discarded": len(self._state.discarded_candidates),
            "promotion_rate": self._state.promotion_rate,
            "cost_per_candidate": (
                self._state.budget_spent / max(1, self._state.n_candidates_evaluated)
            ),
            "fidelity_correlations": {
                f"{k[0]}->{k[1]}": v
                for k, v in self._state.fidelity_correlation.items()
            },
        }
