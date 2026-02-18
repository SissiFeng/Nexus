"""API routes for intelligent optimization features.

Exposes endpoints for:
- Algorithm selection with explanations
- Causal discovery analysis
- Multi-fidelity planning
- Asynchronous SDL experiment management
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from optimization_copilot.meta_controller.intelligent_selector import (
    IntelligentAlgorithmSelector,
    AlgorithmRecommendation,
    AlgorithmSwitchExplanation,
)
from optimization_copilot.causal.optimizer_integration import (
    CausalOptimizationAnalyzer,
    CausalOptimizationInsight,
)
from optimization_copilot.multi_fidelity.enhanced_planner import (
    EnhancedMultiFidelityOptimizer,
    AdaptiveFidelityState,
)
from optimization_copilot.batch.sdl_async_manager import SDLAsyncManager
from optimization_copilot.core.models import CampaignSnapshot


router = APIRouter(prefix="/intelligent", tags=["intelligent"])


# -- Request/Response Models ----------------------------------------------

class AlgorithmSelectionRequest(BaseModel):
    campaign_id: str
    snapshot: dict[str, Any]
    diagnostics: dict[str, float]
    fingerprint: dict[str, Any]
    phase: str
    current_algorithm: str | None = None


class AlgorithmSelectionResponse(BaseModel):
    recommendation: dict[str, Any]
    switch_explanation: dict[str, Any] | None


class CausalAnalysisRequest(BaseModel):
    campaign_id: str
    snapshot: dict[str, Any]
    objective_name: str | None = None


class CausalAnalysisResponse(BaseModel):
    insight: dict[str, Any] | None
    summary: str


class MultiFidelityPlanRequest(BaseModel):
    campaign_id: str
    n_candidates: int = Field(default=20, ge=2)
    budget: float | None = None
    fidelity_levels: list[dict[str, Any]] | None = None


class MultiFidelityPlanResponse(BaseModel):
    plan: dict[str, Any]
    estimated_efficiency_gain: float


class SDLSubmitRequest(BaseModel):
    campaign_id: str
    parameters: dict[str, float]
    priority: str = "NORMAL"
    estimated_duration: float | None = None
    resource_requirements: list[str] | None = None
    dependencies: list[str] | None = None


class SDLSubmitResponse(BaseModel):
    experiment_id: str
    queue_position: int
    estimated_start_time: float | None


# -- Algorithm Selection Endpoints ----------------------------------------

@router.post("/algorithm/select", response_model=AlgorithmSelectionResponse)
async def select_algorithm(request: AlgorithmSelectionRequest) -> AlgorithmSelectionResponse:
    """Get intelligent algorithm recommendation with full explanation.
    
    Analyzes data characteristics (noise, dimensionality, constraints)
    to recommend the best algorithm and explain why.
    """
    try:
        from optimization_copilot.core.models import Phase, ProblemFingerprint
        
        selector = IntelligentAlgorithmSelector()
        
        # Parse inputs
        snapshot = CampaignSnapshot.from_dict(request.snapshot)
        phase = Phase(request.phase)
        fingerprint = ProblemFingerprint(**request.fingerprint)
        
        # Get recommendation
        rec = selector.recommend_algorithm(
            snapshot=snapshot,
            diagnostics=request.diagnostics,
            fingerprint=fingerprint,
            phase=phase,
        )
        
        # Get switch explanation if current algorithm is different
        switch_exp = None
        if request.current_algorithm and request.current_algorithm != rec.algorithm:
            switch = selector.explain_switch(
                current_algorithm=request.current_algorithm,
                snapshot=snapshot,
                diagnostics=request.diagnostics,
                fingerprint=fingerprint,
                phase=phase,
            )
            if switch:
                switch_exp = {
                    "from_algorithm": switch.from_algorithm,
                    "to_algorithm": switch.to_algorithm,
                    "trigger": switch.trigger,
                    "explanation": switch.explanation,
                    "confidence_gain": switch.confidence_gain,
                }
        
        return AlgorithmSelectionResponse(
            recommendation={
                "algorithm": rec.algorithm,
                "confidence": rec.confidence,
                "reason": rec.reason,
                "trade_offs": rec.trade_offs,
                "expected_performance": rec.expected_performance,
                "when_to_switch": rec.when_to_switch,
                "data_characteristics": rec.data_characteristics,
            },
            switch_explanation=switch_exp,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -- Causal Discovery Endpoints -------------------------------------------

@router.post("/causal/analyze", response_model=CausalAnalysisResponse)
async def analyze_causal_relationships(request: CausalAnalysisRequest) -> CausalAnalysisResponse:
    """Analyze causal relationships in campaign data.
    
    Answers "which variables truly drive the objective?" by learning
    causal graphs and distinguishing correlation from causation.
    """
    try:
        snapshot = CampaignSnapshot.from_dict(request.snapshot)
        
        analyzer = CausalOptimizationAnalyzer()
        insight = analyzer.analyze_campaign(snapshot, request.objective_name)
        
        if insight is None:
            return CausalAnalysisResponse(
                insight=None,
                summary="Insufficient data for causal analysis (need at least 10 observations).",
            )
        
        # Serialize insight
        return CausalAnalysisResponse(
            insight={
                "root_causes": [
                    {
                        "variable": rc.variable_name,
                        "direct_effect": rc.direct_causal_effect,
                        "total_effect": rc.total_causal_effect,
                        "is_root_cause": rc.is_root_cause,
                        "recommendation": rc.manipulation_recommendation,
                    }
                    for rc in insight.root_causes
                ],
                "most_effective_interventions": [
                    {
                        "variable": ei.variable_name,
                        "total_effect": ei.total_causal_effect,
                        "recommendation": ei.manipulation_recommendation,
                    }
                    for ei in insight.most_effective_interventions[:5]
                ],
                "spurious_correlations": insight.spurious_correlations,
                "confounding_warnings": insight.confounding_warnings,
                "actionable_recommendations": insight.actionable_recommendations,
            },
            summary=insight.analysis_summary,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -- Multi-Fidelity Endpoints ---------------------------------------------

@router.post("/multifidelity/plan", response_model=MultiFidelityPlanResponse)
async def plan_multifidelity(request: MultiFidelityPlanRequest) -> MultiFidelityPlanResponse:
    """Create multi-fidelity evaluation plan.
    
    Plans cheap simulation → expensive experiment pipeline with
    cost-aware candidate promotion.
    """
    try:
        from optimization_copilot.multi_fidelity.planner import FidelityLevel
        
        # Parse fidelity levels if provided
        fidelity_levels = None
        if request.fidelity_levels:
            fidelity_levels = [
                FidelityLevel(
                    name=fl["name"],
                    cost_multiplier=fl["cost_multiplier"],
                    noise_multiplier=fl["noise_multiplier"],
                    correlation_with_truth=fl["correlation_with_truth"],
                )
                for fl in request.fidelity_levels
            ]
        
        optimizer = EnhancedMultiFidelityOptimizer(
            fidelity_levels=fidelity_levels,
            cost_budget=request.budget,
        )
        
        plan = optimizer.initialize_campaign(
            n_candidates=request.n_candidates,
            budget=request.budget,
        )
        
        return MultiFidelityPlanResponse(
            plan={
                "stages": [
                    {
                        "stage": s.stage,
                        "fidelity": s.fidelity_level.name,
                        "n_candidates": s.n_candidates,
                        "promotion_threshold": s.promotion_threshold,
                        "backend_hint": s.backend_hint,
                        "reason": s.reason,
                    }
                    for s in plan.stages
                ],
                "total_estimated_cost": plan.total_estimated_cost,
                "efficiency_gain": plan.efficiency_gain,
            },
            estimated_efficiency_gain=plan.efficiency_gain,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/multifidelity/{campaign_id}/status")
async def get_multifidelity_status(campaign_id: str) -> dict[str, Any]:
    """Get status of multi-fidelity optimization."""
    # This would typically retrieve from a database
    # For now, return placeholder
    return {
        "campaign_id": campaign_id,
        "status": "active",
        "message": "Multi-fidelity status tracking not yet implemented in API",
    }


# -- SDL Async Experiment Endpoints ---------------------------------------

# Global SDL manager instance (would be per-campaign in production)
_sdl_manager: SDLAsyncManager | None = None

def _get_sdl_manager() -> SDLAsyncManager:
    global _sdl_manager
    if _sdl_manager is None:
        _sdl_manager = SDLAsyncManager()
    return _sdl_manager


@router.post("/sdl/submit", response_model=SDLSubmitResponse)
async def submit_sdl_experiment(request: SDLSubmitRequest) -> SDLSubmitResponse:
    """Submit an experiment to the SDL queue.
    
    Manages asynchronous execution with priority scheduling
    and resource-aware dispatch.
    """
    try:
        from optimization_copilot.batch.sdl_async_manager import ExperimentPriority
        
        manager = _get_sdl_manager()
        
        # Parse priority
        priority_map = {
            "CRITICAL": ExperimentPriority.CRITICAL,
            "HIGH": ExperimentPriority.HIGH,
            "NORMAL": ExperimentPriority.NORMAL,
            "LOW": ExperimentPriority.LOW,
            "BACKGROUND": ExperimentPriority.BACKGROUND,
        }
        priority = priority_map.get(request.priority, ExperimentPriority.NORMAL)
        
        exp_id = manager.submit_experiment(
            parameters=request.parameters,
            priority=priority,
            estimated_duration=request.estimated_duration,
            resource_requirements=request.resource_requirements,
            dependencies=request.dependencies,
            metadata={"campaign_id": request.campaign_id},
        )
        
        # Get queue position
        queue_status = manager.get_queue_status()
        
        return SDLSubmitResponse(
            experiment_id=exp_id,
            queue_position=queue_status["queued_experiments"],
            estimated_start_time=manager._estimate_wait_time() if hasattr(manager, '_estimate_wait_time') else None,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sdl/{experiment_id}/status")
async def get_experiment_status(experiment_id: str) -> dict[str, Any]:
    """Get status of a submitted experiment."""
    manager = _get_sdl_manager()
    exp = manager.get_experiment(experiment_id)
    
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return {
        "experiment_id": exp.experiment_id,
        "status": exp.status.value,
        "priority": exp.priority.name,
        "wait_time": exp.wait_time,
        "execution_time": exp.execution_time if exp.started_at else None,
        "result": exp.result,
        "error": exp.error_message if exp.status.value == "failed" else None,
    }


@router.get("/sdl/queue/status")
async def get_sdl_queue_status() -> dict[str, Any]:
    """Get overall SDL queue status."""
    manager = _get_sdl_manager()
    return manager.get_queue_status()


@router.get("/sdl/statistics")
async def get_sdl_statistics() -> dict[str, Any]:
    """Get SDL execution statistics."""
    manager = _get_sdl_manager()
    return manager.get_statistics()


@router.post("/sdl/{experiment_id}/complete")
async def complete_sdl_experiment(
    experiment_id: str,
    result: dict[str, Any],
) -> dict[str, str]:
    """Mark an experiment as completed (called by lab equipment)."""
    manager = _get_sdl_manager()
    
    try:
        manager.complete_experiment(experiment_id, result)
        return {"status": "completed", "experiment_id": experiment_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/sdl/{experiment_id}/fail")
async def fail_sdl_experiment(
    experiment_id: str,
    error_message: str = "",
) -> dict[str, str]:
    """Mark an experiment as failed."""
    manager = _get_sdl_manager()
    
    try:
        manager.fail_experiment(experiment_id, error_message)
        return {"status": "failed", "experiment_id": experiment_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
