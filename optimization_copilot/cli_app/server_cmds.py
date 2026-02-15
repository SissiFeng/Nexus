"""Server management CLI commands."""

from __future__ import annotations

import click

from optimization_copilot.platform.auth import AuthManager
from optimization_copilot.platform.models import Role
from optimization_copilot.platform.workspace import Workspace


@click.group()
def server() -> None:
    """Manage the API server."""
    pass


@server.command("init")
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize workspace and create first admin API key."""
    workspace_dir = ctx.obj["workspace_dir"]
    ws = Workspace(workspace_dir)
    manifest = ws.init()

    click.echo(f"Workspace initialized: {workspace_dir}")
    click.echo(f"  ID: {manifest.workspace_id}")

    # Create initial admin key if none exist
    auth = AuthManager(ws)
    keys = auth.list_keys()
    if not keys:
        raw_key = auth.create_key("admin", Role.ADMIN)
        click.echo(f"\nAdmin API key created:")
        click.echo(f"  {raw_key}")
        click.echo(f"  Save this key — it won't be shown again.")
    else:
        click.echo(f"\n{len(keys)} API key(s) already exist.")


@server.command("start")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8000, type=int, help="Port to listen on")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.pass_context
def start(ctx: click.Context, host: str, port: int, reload: bool) -> None:
    """Start the API server."""
    workspace_dir = ctx.obj["workspace_dir"]

    # Ensure workspace is initialized
    ws = Workspace(workspace_dir)
    ws.init()

    click.echo(f"Starting server on {host}:{port}")
    click.echo(f"Workspace: {workspace_dir}")
    click.echo(f"API docs: http://{host}:{port}/docs")

    import uvicorn

    uvicorn.run(
        "optimization_copilot.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )
