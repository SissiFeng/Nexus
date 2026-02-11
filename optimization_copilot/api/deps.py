"""FastAPI dependency injection."""

from __future__ import annotations

from typing import Any

from optimization_copilot.platform.auth import AuthManager
from optimization_copilot.platform.campaign_manager import CampaignManager
from optimization_copilot.platform.campaign_runner import CampaignRunner
from optimization_copilot.platform.events import AsyncEventBus
from optimization_copilot.platform.models import ApiKey, Role
from optimization_copilot.platform.rag import RAGIndex
from optimization_copilot.platform.workspace import Workspace


class AppState:
    """Shared application state initialized by the app factory."""

    def __init__(
        self,
        workspace: Workspace,
        manager: CampaignManager,
        runner: CampaignRunner,
        auth: AuthManager,
        event_bus: AsyncEventBus,
        rag: RAGIndex,
    ) -> None:
        self.workspace = workspace
        self.manager = manager
        self.runner = runner
        self.auth = auth
        self.event_bus = event_bus
        self.rag = rag


# Global app state — set by create_app()
_state: AppState | None = None


def set_app_state(state: AppState) -> None:
    global _state
    _state = state


def get_app_state() -> AppState:
    if _state is None:
        raise RuntimeError("App state not initialized. Call create_app() first.")
    return _state


def get_workspace() -> Workspace:
    return get_app_state().workspace


def get_manager() -> CampaignManager:
    return get_app_state().manager


def get_runner() -> CampaignRunner:
    return get_app_state().runner


def get_auth() -> AuthManager:
    return get_app_state().auth


def get_event_bus() -> AsyncEventBus:
    return get_app_state().event_bus


def get_rag() -> RAGIndex:
    return get_app_state().rag


def authenticate(api_key_header: str | None) -> ApiKey | None:
    """Authenticate from X-API-Key header. Returns None if auth is disabled or key invalid."""
    if api_key_header is None:
        return None
    auth = get_auth()
    return auth.authenticate(api_key_header)
