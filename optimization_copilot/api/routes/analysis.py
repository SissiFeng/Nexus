"""DataAnalysisPipeline API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline
from optimization_copilot.agents.execution_trace import TracedResult
from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)

router = APIRouter(prefix="/analysis", tags=["analysis"])


# ── Request schemas ───────────────────────────────────────


class TopKRequest(BaseModel):
    values: list[float]
    names: list[str]
    k: int = 5
    descending: bool = True


class RankingRequest(BaseModel):
    values: list[float]
    names: list[str]
    descending: bool = True


class OutlierRequest(BaseModel):
    values: list[float]
    names: list[str]
    n_sigma: float = 2.0


class CorrelationRequest(BaseModel):
    xs: list[float]
    ys: list[float]


class FanovaRequest(BaseModel):
    X: list[list[float]]
    y: list[float]
    var_names: list[str] | None = None
    n_trees: int = 50
    seed: int = 42


class SymregRequest(BaseModel):
    X: list[list[float]]
    y: list[float]
    var_names: list[str] | None = None
    pop_size: int = 200
    n_gen: int = 50
    seed: int = 42


class ParetoRequest(BaseModel):
    observations: list[dict[str, Any]]
    parameter_specs: list[dict[str, Any]]
    objectives: list[str]
    objective_directions: list[str]
    campaign_id: str = "analysis"
    weights: dict[str, float] | None = None


class DiagnosticsRequest(BaseModel):
    observations: list[dict[str, Any]]
    parameter_specs: list[dict[str, Any]]
    objectives: list[str]
    objective_directions: list[str]
    campaign_id: str = "analysis"


class MolecularRequest(BaseModel):
    smiles_list: list[str]
    observations: list[dict[str, Any]]
    parameter_specs: list[dict[str, Any]]
    objective_name: str
    n_suggestions: int = 5
    seed: int = 42


class CausalDiscoverRequest(BaseModel):
    data: list[list[float]]
    var_names: list[str]
    alpha: float = 0.05


class InterventionRequest(BaseModel):
    graph_dict: dict[str, Any]
    intervention: dict[str, float]
    data: list[list[float]]
    target: str


class CounterfactualRequest(BaseModel):
    graph_dict: dict[str, Any]
    equations: dict[str, str]
    factual: dict[str, float]
    intervention: dict[str, float]
    query: str


class PhysicsConstrainedGPRequest(BaseModel):
    X: list[list[float]]
    y: list[float]
    constraints: list[dict[str, Any]]
    kernel_type: str = "rbf"


class ODESolveRequest(BaseModel):
    rhs_code: str
    y0: list[float]
    t_span: list[float]
    n_steps: int = 100


class HypothesisGenerateRequest(BaseModel):
    data: list[list[float]]
    var_names: list[str]
    target_index: int = -1


class HypothesisTestRequest(BaseModel):
    hypotheses: list[dict[str, Any]]
    X: list[list[float]]
    y: list[float]
    var_names: list[str] | None = None


class HypothesisStatusRequest(BaseModel):
    tracker_state: dict[str, Any]


class BootstrapCIRequest(BaseModel):
    data: list[float]
    statistic_fn_name: str = "mean"
    n_bootstrap: int = 1000
    confidence: float = 0.95
    seed: int = 42


class ConclusionRobustnessRequest(BaseModel):
    values: list[float]
    names: list[str]
    analysis_type: str = "ranking"
    k: int = 1
    n_bootstrap: int = 500
    seed: int = 42


class DecisionSensitivityRequest(BaseModel):
    values: list[float]
    names: list[str]
    uncertainties: list[float] | None = None
    n_perturbations: int = 100
    seed: int = 42


class CrossModelConsistencyRequest(BaseModel):
    X: list[list[float]]
    y: list[float]
    model_types: list[str] | None = None


class HybridFitRequest(BaseModel):
    X: list[list[float]]
    y: list[float]
    theory_type: str
    theory_params: dict[str, float]


# ── Helpers ───────────────────────────────────────────────


def _serialize_result(result: TracedResult) -> dict[str, Any]:
    """Convert a TracedResult into a JSON-serializable dict."""
    return {
        "value": result.value,
        "tag": result.tag.value,
        "traces": [t.to_dict() for t in result.traces],
        "is_computed": result.is_computed,
    }


def _build_snapshot(
    observations: list[dict[str, Any]],
    parameter_specs: list[dict[str, Any]],
    objectives: list[str],
    objective_directions: list[str],
    campaign_id: str,
) -> CampaignSnapshot:
    """Build a CampaignSnapshot from raw request dicts.

    Raises
    ------
    ValueError
        If a parameter spec contains an unrecognised variable type.
    """
    params = [
        ParameterSpec(
            name=p["name"],
            type=VariableType(p["type"]),
            lower=p.get("lower"),
            upper=p.get("upper"),
            categories=p.get("categories"),
        )
        for p in parameter_specs
    ]

    obs_list = [
        Observation(
            iteration=o.get("iteration", i),
            parameters=o.get("parameters", {}),
            kpi_values=o.get("kpi_values", {}),
        )
        for i, o in enumerate(observations)
    ]

    return CampaignSnapshot(
        campaign_id=campaign_id,
        parameter_specs=params,
        observations=obs_list,
        objective_names=objectives,
        objective_directions=objective_directions,
    )


# ── Endpoints ─────────────────────────────────────────────


@router.post("/top-k")
def run_top_k(req: TopKRequest) -> dict[str, Any]:
    """Return the top-K entries by value."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_top_k(req.values, req.names, req.k, req.descending)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/ranking")
def run_ranking(req: RankingRequest) -> dict[str, Any]:
    """Return a full ranking of all entries."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_ranking(req.values, req.names, req.descending)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/outliers")
def run_outliers(req: OutlierRequest) -> dict[str, Any]:
    """Detect statistical outliers via z-score."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_outlier_detection(req.values, req.names, req.n_sigma)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/correlation")
def run_correlation(req: CorrelationRequest) -> dict[str, Any]:
    """Compute Pearson correlation between two variables."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_correlation(req.xs, req.ys)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/fanova")
def run_fanova(req: FanovaRequest) -> dict[str, Any]:
    """Run fANOVA decomposition to identify feature importance."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_fanova(
            req.X, req.y, req.var_names, req.n_trees, req.seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/symreg")
def run_symreg(req: SymregRequest) -> dict[str, Any]:
    """Run symbolic regression to discover interpretable equations."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_symreg(
            req.X,
            req.y,
            req.var_names,
            population_size=req.pop_size,
            n_generations=req.n_gen,
            seed=req.seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/pareto")
def run_pareto(req: ParetoRequest) -> dict[str, Any]:
    """Perform multi-objective Pareto analysis."""
    pipeline = DataAnalysisPipeline()
    try:
        snapshot = _build_snapshot(
            req.observations,
            req.parameter_specs,
            req.objectives,
            req.objective_directions,
            req.campaign_id,
        )
        result = pipeline.run_pareto_analysis(snapshot, req.weights)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/diagnostics")
def run_diagnostics(req: DiagnosticsRequest) -> dict[str, Any]:
    """Compute diagnostic signals for a campaign snapshot."""
    pipeline = DataAnalysisPipeline()
    try:
        snapshot = _build_snapshot(
            req.observations,
            req.parameter_specs,
            req.objectives,
            req.objective_directions,
            req.campaign_id,
        )
        result = pipeline.run_diagnostics(snapshot)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/molecular")
def run_molecular(req: MolecularRequest) -> dict[str, Any]:
    """Run the end-to-end molecular optimization pipeline."""
    pipeline = DataAnalysisPipeline()
    try:
        param_specs = [
            ParameterSpec(
                name=p["name"],
                type=VariableType(p["type"]),
                lower=p.get("lower"),
                upper=p.get("upper"),
                categories=p.get("categories"),
            )
            for p in req.parameter_specs
        ]
        obs_list = [
            Observation(
                iteration=o.get("iteration", i),
                parameters=o.get("parameters", {}),
                kpi_values=o.get("kpi_values", {}),
            )
            for i, o in enumerate(req.observations)
        ]
        result = pipeline.run_molecular_pipeline(
            req.smiles_list,
            obs_list,
            param_specs,
            req.objective_name,
            req.n_suggestions,
            req.seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/causal/discover")
def run_causal_discover(req: CausalDiscoverRequest) -> dict[str, Any]:
    """Discover causal structure from data via PC algorithm."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_causal_discovery(req.data, req.var_names, req.alpha)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/causal/intervene")
def run_causal_intervene(req: InterventionRequest) -> dict[str, Any]:
    """Do-operator interventional reasoning."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_intervention(req.graph_dict, req.intervention, req.data, req.target)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/causal/counterfactual")
def run_causal_counterfactual(req: CounterfactualRequest) -> dict[str, Any]:
    """SCM counterfactual reasoning."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_counterfactual(
            req.graph_dict, req.equations, req.factual, req.intervention, req.query,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/physics/constrained-gp")
def run_physics_constrained_gp(req: PhysicsConstrainedGPRequest) -> dict[str, Any]:
    """Physics-constrained GP with conservation laws."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_physics_constrained_gp(req.X, req.y, req.constraints, req.kernel_type)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/physics/ode-solve")
def run_physics_ode_solve(req: ODESolveRequest) -> dict[str, Any]:
    """Solve ODE system with RK4 integrator."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_ode_solve(req.rhs_code, req.y0, tuple(req.t_span), req.n_steps)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/hypothesis/generate")
def run_hypothesis_generate(req: HypothesisGenerateRequest) -> dict[str, Any]:
    """Generate competing hypotheses from data."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_hypothesis_generate(req.data, req.var_names, req.target_index)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/hypothesis/test")
def run_hypothesis_test(req: HypothesisTestRequest) -> dict[str, Any]:
    """Test hypotheses against data using BIC scoring."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_hypothesis_test(req.hypotheses, req.X, req.y, req.var_names)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/hypothesis/status")
def run_hypothesis_status(req: HypothesisStatusRequest) -> dict[str, Any]:
    """Get hypothesis tracker status report."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_hypothesis_status(req.tracker_state)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/robustness/bootstrap")
def run_robustness_bootstrap(req: BootstrapCIRequest) -> dict[str, Any]:
    """Bootstrap confidence interval for a statistic."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_bootstrap_ci(
            req.data, req.statistic_fn_name, req.n_bootstrap, req.confidence, req.seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/robustness/conclusion")
def run_robustness_conclusion(req: ConclusionRobustnessRequest) -> dict[str, Any]:
    """Check robustness of analytical conclusions."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_conclusion_robustness(
            req.values, req.names, req.analysis_type, req.k, req.n_bootstrap, req.seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/robustness/sensitivity")
def run_robustness_sensitivity(req: DecisionSensitivityRequest) -> dict[str, Any]:
    """Analyze sensitivity of decisions to data perturbations."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_decision_sensitivity(
            req.values, req.names, req.uncertainties, req.n_perturbations, req.seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/robustness/consistency")
def run_robustness_consistency(req: CrossModelConsistencyRequest) -> dict[str, Any]:
    """Check prediction consistency across model types."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_cross_model_consistency(req.X, req.y, req.model_types)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)


@router.post("/hybrid/fit")
def run_hybrid_fit(req: HybridFitRequest) -> dict[str, Any]:
    """Fit a hybrid theory+GP model."""
    pipeline = DataAnalysisPipeline()
    try:
        result = pipeline.run_hybrid_fit(req.X, req.y, req.theory_type, req.theory_params)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serialize_result(result)
