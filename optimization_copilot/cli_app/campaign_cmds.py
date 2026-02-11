"""Campaign management CLI commands."""

from __future__ import annotations

import json
import sys

import click

from optimization_copilot.platform.campaign_manager import (
    CampaignManager,
    InvalidTransitionError,
)
from optimization_copilot.platform.models import CampaignStatus
from optimization_copilot.platform.workspace import (
    CampaignNotFoundError,
    Workspace,
)


def _get_manager(ctx: click.Context) -> CampaignManager:
    ws = Workspace(ctx.obj["workspace_dir"])
    ws.init()
    return CampaignManager(ws)


@click.group()
def campaign() -> None:
    """Manage optimization campaigns."""
    pass


@campaign.command("create")
@click.option("--spec", "-s", required=True, type=click.Path(exists=True), help="Path to spec JSON file")
@click.option("--name", "-n", default="", help="Campaign display name")
@click.option("--tag", "-t", multiple=True, help="Tags (can specify multiple)")
@click.pass_context
def create(ctx: click.Context, spec: str, name: str, tag: tuple[str, ...]) -> None:
    """Create a new campaign from a spec file."""
    with open(spec) as f:
        spec_dict = json.load(f)

    manager = _get_manager(ctx)
    record = manager.create(spec_dict=spec_dict, name=name, tags=list(tag))
    click.echo(f"Created campaign: {record.campaign_id}")
    click.echo(f"  Name: {record.name}")
    click.echo(f"  Status: {record.status.value}")


@campaign.command("list")
@click.option("--status", type=click.Choice([s.value for s in CampaignStatus]), default=None)
@click.pass_context
def list_campaigns(ctx: click.Context, status: str | None) -> None:
    """List all campaigns."""
    manager = _get_manager(ctx)
    filter_status = CampaignStatus(status) if status else None
    records = manager.list_all(status=filter_status)

    if not records:
        click.echo("No campaigns found.")
        return

    for r in records:
        kpi_str = f"  KPI={r.best_kpi:.4f}" if r.best_kpi is not None else ""
        click.echo(f"  {r.campaign_id[:8]}  {r.status.value:10s}  iter={r.iteration:4d}{kpi_str}  {r.name}")


@campaign.command("status")
@click.argument("campaign_id")
@click.pass_context
def status(ctx: click.Context, campaign_id: str) -> None:
    """Show campaign status."""
    try:
        manager = _get_manager(ctx)
        record = manager.get(campaign_id)
    except CampaignNotFoundError:
        click.echo(f"Campaign not found: {campaign_id}", err=True)
        sys.exit(1)

    click.echo(f"Campaign: {record.campaign_id}")
    click.echo(f"  Name:       {record.name}")
    click.echo(f"  Status:     {record.status.value}")
    click.echo(f"  Iteration:  {record.iteration}")
    click.echo(f"  Trials:     {record.total_trials}")
    if record.best_kpi is not None:
        click.echo(f"  Best KPI:   {record.best_kpi:.6f}")
    if record.error_message:
        click.echo(f"  Error:      {record.error_message}")
    if record.tags:
        click.echo(f"  Tags:       {', '.join(record.tags)}")


@campaign.command("start")
@click.argument("campaign_id")
@click.pass_context
def start(ctx: click.Context, campaign_id: str) -> None:
    """Start a campaign (requires running server for async execution)."""
    click.echo("Note: Use 'copilot server start' for async campaign execution.")
    click.echo(f"To start via API: POST /api/campaigns/{campaign_id}/start")


@campaign.command("stop")
@click.argument("campaign_id")
@click.pass_context
def stop(ctx: click.Context, campaign_id: str) -> None:
    """Stop a running campaign."""
    click.echo(f"To stop via API: POST /api/campaigns/{campaign_id}/stop")


@campaign.command("pause")
@click.argument("campaign_id")
@click.pass_context
def pause(ctx: click.Context, campaign_id: str) -> None:
    """Pause a running campaign."""
    click.echo(f"To pause via API: POST /api/campaigns/{campaign_id}/pause")


@campaign.command("resume")
@click.argument("campaign_id")
@click.pass_context
def resume(ctx: click.Context, campaign_id: str) -> None:
    """Resume a paused campaign."""
    click.echo(f"To resume via API: POST /api/campaigns/{campaign_id}/resume")


@campaign.command("delete")
@click.argument("campaign_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete(ctx: click.Context, campaign_id: str, yes: bool) -> None:
    """Archive (soft-delete) a campaign."""
    if not yes:
        click.confirm(f"Archive campaign {campaign_id}?", abort=True)

    try:
        manager = _get_manager(ctx)
        manager.delete(campaign_id)
        click.echo(f"Campaign {campaign_id} archived.")
    except CampaignNotFoundError:
        click.echo(f"Campaign not found: {campaign_id}", err=True)
        sys.exit(1)
    except InvalidTransitionError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@campaign.command("compare")
@click.argument("campaign_ids", nargs=-1, required=True)
@click.pass_context
def compare(ctx: click.Context, campaign_ids: tuple[str, ...]) -> None:
    """Compare multiple campaigns side by side."""
    if len(campaign_ids) < 2:
        click.echo("Need at least 2 campaign IDs to compare.", err=True)
        sys.exit(1)

    try:
        manager = _get_manager(ctx)
        report = manager.compare(list(campaign_ids))
    except CampaignNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo("Campaign Comparison:")
    click.echo("-" * 60)
    for r in report.records:
        kpi_str = f"{r.best_kpi:.6f}" if r.best_kpi is not None else "N/A"
        click.echo(f"  {r.campaign_id[:8]}  {r.status.value:10s}  iter={r.iteration}  kpi={kpi_str}  {r.name}")

    if report.kpi_comparison:
        click.echo("\nKPI Comparison:")
        for kpi_name, values in report.kpi_comparison.items():
            vals = [f"{v:.4f}" if v is not None else "N/A" for v in values]
            click.echo(f"  {kpi_name}: {' | '.join(vals)}")

    if report.winner:
        click.echo(f"\nWinner: {report.winner}")
