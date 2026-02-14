"""API route aggregation."""

from fastapi import APIRouter

from optimization_copilot.api.routes.campaigns import router as campaigns_router
from optimization_copilot.api.routes.campaigns import search_router
from optimization_copilot.api.routes.store import router as store_router
from optimization_copilot.api.routes.advice import router as advice_router
from optimization_copilot.api.routes.reports import router as reports_router
from optimization_copilot.api.routes.ws import router as ws_router
from optimization_copilot.api.routes.loop import router as loop_router
from optimization_copilot.api.routes.analysis import router as analysis_router
from optimization_copilot.api.routes.frontend_v2 import router as frontend_v2_router
from optimization_copilot.api.routes.insights import router as insights_router


def create_api_router() -> APIRouter:
    """Create the aggregated API router."""
    api = APIRouter(prefix="/api")
    api.include_router(campaigns_router)
    api.include_router(search_router)
    api.include_router(store_router)
    api.include_router(advice_router)
    api.include_router(reports_router)
    api.include_router(ws_router)
    api.include_router(loop_router)
    api.include_router(analysis_router)
    api.include_router(frontend_v2_router)
    api.include_router(insights_router)
    return api
