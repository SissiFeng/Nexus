"""Click CLI entry point."""

from __future__ import annotations

import click

from optimization_copilot.cli_app.campaign_cmds import campaign
from optimization_copilot.cli_app.meta_cmds import meta
from optimization_copilot.cli_app.store_cmds import store
from optimization_copilot.cli_app.server_cmds import server
from optimization_copilot.platform.workspace import Workspace


@click.group()
@click.option(
    "--workspace", "-w",
    default="./workspace",
    envvar="OCP_WORKSPACE",
    help="Workspace directory path",
)
@click.pass_context
def cli(ctx: click.Context, workspace: str) -> None:
    """Optimization Copilot — Intelligent Experiment Decision Platform."""
    ctx.ensure_object(dict)
    ctx.obj["workspace_dir"] = workspace


cli.add_command(campaign)
cli.add_command(meta)
cli.add_command(store)
cli.add_command(server)


if __name__ == "__main__":
    cli()
