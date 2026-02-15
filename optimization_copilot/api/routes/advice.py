"""Meta-learning advice routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from optimization_copilot.api.deps import get_workspace
from optimization_copilot.api.schemas import AdviceRequest, AdviceResponse

router = APIRouter(prefix="/advice", tags=["advice"])


@router.post("", response_model=AdviceResponse)
def get_advice(req: AdviceRequest) -> AdviceResponse:
    """Get meta-learning advice for a problem fingerprint."""
    workspace = get_workspace()
    advisor_data = workspace.load_advisor()

    if advisor_data is None:
        return AdviceResponse(
            recommended_backends=[],
            confidence=0.0,
            reason_codes=["no_experience_data"],
        )

    from optimization_copilot.core.models import ProblemFingerprint
    from optimization_copilot.meta_learning.advisor import MetaLearningAdvisor

    advisor = MetaLearningAdvisor.from_dict(advisor_data)
    fingerprint = ProblemFingerprint(**{
        k: v for k, v in req.fingerprint.items()
    })

    advice = advisor.advise(fingerprint)

    weights_dict = None
    if advice.scoring_weights is not None:
        weights_dict = {
            "gain": advice.scoring_weights.gain,
            "fail": advice.scoring_weights.fail,
            "cost": advice.scoring_weights.cost,
            "drift": advice.scoring_weights.drift,
            "incompatibility": advice.scoring_weights.incompatibility,
        }

    thresholds_dict = None
    if advice.switching_thresholds is not None:
        thresholds_dict = {
            "cold_start_min_observations": advice.switching_thresholds.cold_start_min_observations,
            "learning_plateau_length": advice.switching_thresholds.learning_plateau_length,
        }

    return AdviceResponse(
        recommended_backends=advice.recommended_backends,
        scoring_weights=weights_dict,
        switching_thresholds=thresholds_dict,
        failure_adjustments=advice.failure_adjustments,
        drift_robust_backends=advice.drift_robust_backends,
        confidence=advice.confidence,
        reason_codes=advice.reason_codes,
    )


@router.get("/experience-count")
def experience_count() -> dict:
    """Get total experience count from meta-learning advisor."""
    workspace = get_workspace()
    advisor_data = workspace.load_advisor()

    if advisor_data is None:
        return {"count": 0}

    from optimization_copilot.meta_learning.advisor import MetaLearningAdvisor

    advisor = MetaLearningAdvisor.from_dict(advisor_data)
    return {"count": advisor.experience_count()}
