"""Experiment store CLI commands."""

from __future__ import annotations

import json
import sys

import click

from optimization_copilot.platform.workspace import Workspace


def _get_workspace(ctx: click.Context) -> Workspace:
    ws = Workspace(ctx.obj["workspace_dir"])
    ws.init()
    return ws


@click.group()
def store() -> None:
    """Query experiment store data."""
    pass


@store.command("summary")
@click.argument("campaign_id")
@click.pass_context
def summary(ctx: click.Context, campaign_id: str) -> None:
    """Show store summary for a campaign."""
    ws = _get_workspace(ctx)
    store_data = ws.load_store(campaign_id)
    if store_data is None:
        click.echo(f"No store data found for campaign: {campaign_id}", err=True)
        sys.exit(1)

    from optimization_copilot.store.store import ExperimentStore

    exp_store = ExperimentStore.from_dict(store_data)
    s = exp_store.summary(campaign_id=campaign_id)

    click.echo(f"Store Summary for {campaign_id}:")
    click.echo(f"  Experiments:  {s.n_experiments}")
    click.echo(f"  Campaigns:    {s.n_campaigns}")
    click.echo(f"  Artifacts:    {s.n_artifacts}")
    click.echo(f"  Parameters:   {', '.join(s.parameter_names)}")
    click.echo(f"  KPIs:         {', '.join(s.kpi_names)}")


@store.command("query")
@click.argument("campaign_id")
@click.option("--iteration", "-i", type=int, default=None, help="Filter by iteration")
@click.option("--only-successful", is_flag=True, help="Only show successful experiments")
@click.pass_context
def query(
    ctx: click.Context,
    campaign_id: str,
    iteration: int | None,
    only_successful: bool,
) -> None:
    """Query experiments from store."""
    ws = _get_workspace(ctx)
    store_data = ws.load_store(campaign_id)
    if store_data is None:
        click.echo(f"No store data found for campaign: {campaign_id}", err=True)
        sys.exit(1)

    from optimization_copilot.store.store import ExperimentStore, StoreQuery

    exp_store = ExperimentStore.from_dict(store_data)
    q = StoreQuery(
        campaign_id=campaign_id,
        iteration=iteration,
        only_successful=only_successful,
    )
    experiments = exp_store.query(q)

    click.echo(f"Found {len(experiments)} experiments:")
    for exp in experiments[:20]:
        d = exp.to_dict()
        click.echo(f"  {d.get('experiment_id', 'N/A')[:8]}  iter={d.get('iteration', 'N/A')}")

    if len(experiments) > 20:
        click.echo(f"  ... and {len(experiments) - 20} more")


@store.command("export")
@click.argument("campaign_id")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output JSON file path")
@click.pass_context
def export(ctx: click.Context, campaign_id: str, output: str) -> None:
    """Export store data as JSON."""
    ws = _get_workspace(ctx)
    store_data = ws.load_store(campaign_id)
    if store_data is None:
        click.echo(f"No store data found for campaign: {campaign_id}", err=True)
        sys.exit(1)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(store_data, f, indent=2, ensure_ascii=False)

    click.echo(f"Exported store data to {output}")
