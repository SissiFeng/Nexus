"""FastAPI application factory."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from optimization_copilot.api.deps import AppState, set_app_state
from optimization_copilot.api.routes import create_api_router
from optimization_copilot.platform.auth import AuthManager
from optimization_copilot.platform.campaign_manager import CampaignManager
from optimization_copilot.platform.campaign_runner import CampaignRunner
from optimization_copilot.platform.events import AsyncEventBus
from optimization_copilot.platform.rag import RAGIndex
from optimization_copilot.platform.workspace import Workspace


def create_app(
    workspace_dir: str | Path = "./workspace",
    title: str = "Optimization Copilot",
    version: str = "0.3.0",
) -> FastAPI:
    """Create and configure the FastAPI application."""
    # Initialize platform services
    workspace = Workspace(workspace_dir)
    workspace.init()

    event_bus = AsyncEventBus()
    auth = AuthManager(workspace)
    manager = CampaignManager(workspace)
    runner = CampaignRunner(
        workspace=workspace,
        manager=manager,
        event_bus=event_bus,
    )

    # Load or create RAG index
    rag = RAGIndex()
    rag_data = workspace.load_rag_index()
    if rag_data is not None:
        rag = RAGIndex.from_dict(rag_data)

    # Set global app state for dependency injection
    state = AppState(
        workspace=workspace,
        manager=manager,
        runner=runner,
        auth=auth,
        event_bus=event_bus,
        rag=rag,
    )
    set_app_state(state)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        event_bus.loop = asyncio.get_running_loop()
        try:
            yield
        finally:
            # Shutdown — ensure both cleanup steps run even if one fails
            try:
                workspace.save_rag_index(rag.to_dict())
            except Exception:
                import logging
                logging.getLogger(__name__).exception(
                    "Failed to save RAG index during shutdown"
                )
            event_bus.clear()

    app = FastAPI(
        title=title,
        version=version,
        description="One-stop intelligent experiment decision platform",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store state on app for testing access
    app.state.platform = state

    # Include API routes
    app.include_router(create_api_router())

    # Health check
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": version}

    return app
