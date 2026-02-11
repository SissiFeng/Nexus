"""Meta-learning advisor CLI commands."""

from __future__ import annotations

import json
import sys

import click

from optimization_copilot.platform.workspace import Workspace


def _get_workspace(ctx: click.Context) -> Workspace:
    ws = Workspace(ctx.obj["workspace_dir"])
    ws.init()
    return ws


@click.group("meta-learning")
def meta() -> None:
    """Meta-learning advisor commands."""
    pass


@meta.command("show")
@click.pass_context
def show(ctx: click.Context) -> None:
    """Show meta-learning advisor state."""
    ws = _get_workspace(ctx)
    advisor_data = ws.load_advisor()

    if advisor_data is None:
        click.echo("No meta-learning advisor data found.")
        click.echo("The advisor learns from completed campaigns.")
        return

    # Display experience store stats
    store_data = advisor_data.get("experience_store", {})
    experiences = store_data.get("outcomes", [])
    experience_count = len(experiences)

    click.echo("Meta-Learning Advisor State:")
    click.echo(f"  Experiences: {experience_count}")

    # Config info
    config = advisor_data.get("config", {})
    if config:
        click.echo(f"  Min experiences for learning: {config.get('min_experiences_for_learning', 'N/A')}")
        click.echo(f"  Similarity decay: {config.get('similarity_decay', 'N/A')}")

    # Weight tuner state
    weight_tuner = advisor_data.get("weight_tuner", {})
    learned_weights = weight_tuner.get("learned_weights", {})
    if learned_weights:
        click.echo(f"  Learned weight profiles: {len(learned_weights)}")

    # Threshold learner state
    threshold_learner = advisor_data.get("threshold_learner", {})
    learned_thresholds = threshold_learner.get("learned_thresholds", {})
    if learned_thresholds:
        click.echo(f"  Learned threshold profiles: {len(learned_thresholds)}")

    # Failure learner state
    failure_learner = advisor_data.get("failure_learner", {})
    strategies = failure_learner.get("strategies", {})
    if strategies:
        click.echo(f"  Failure strategies: {len(strategies)}")

    # Drift tracker state
    drift_tracker = advisor_data.get("drift_tracker", {})
    robustness = drift_tracker.get("robustness", {})
    if robustness:
        click.echo(f"  Drift robustness profiles: {len(robustness)}")


@meta.command("advice")
@click.argument("spec_file", type=click.Path(exists=True))
@click.pass_context
def advice(ctx: click.Context, spec_file: str) -> None:
    """Get advice for a campaign spec."""
    ws = _get_workspace(ctx)
    advisor_data = ws.load_advisor()

    if advisor_data is None:
        click.echo("No meta-learning advisor data found.")
        click.echo("The advisor needs completed campaigns to provide advice.")
        return

    # Load the spec file
    with open(spec_file) as f:
        try:
            spec_dict = json.load(f)
        except json.JSONDecodeError as e:
            click.echo(f"Invalid JSON in spec file: {e}", err=True)
            sys.exit(1)

    # Try to create a fingerprint and get advice
    try:
        from optimization_copilot.meta_learning.advisor import MetaLearningAdvisor

        advisor = MetaLearningAdvisor.from_dict(advisor_data)

        # Build a minimal fingerprint from the spec
        from optimization_copilot.core.models import (
            ProblemFingerprint,
            VariableType,
            ObjectiveForm,
            NoiseRegime,
            CostProfile,
            FailureInformativeness,
            DataScale,
            Dynamics,
            FeasibleRegion,
        )

        # Infer basic fingerprint from spec
        parameters = spec_dict.get("parameters", [])
        objectives = spec_dict.get("objectives", [])

        # Variable types
        param_types = {p.get("type", "continuous") for p in parameters}
        if len(param_types) == 1:
            vtype_str = param_types.pop()
            try:
                vtype = VariableType(vtype_str)
            except ValueError:
                vtype = VariableType.CONTINUOUS
        elif len(param_types) > 1:
            vtype = VariableType.MIXED
        else:
            vtype = VariableType.CONTINUOUS

        # Objective form
        if len(objectives) > 1:
            obj_form = ObjectiveForm.MULTI_OBJECTIVE
        else:
            obj_form = ObjectiveForm.SINGLE

        fingerprint = ProblemFingerprint(
            variable_types=vtype,
            objective_form=obj_form,
            noise_regime=NoiseRegime.MEDIUM,
            cost_profile=CostProfile.UNIFORM,
            failure_informativeness=FailureInformativeness.WEAK,
            data_scale=DataScale.TINY,
            dynamics=Dynamics.STATIC,
            feasible_region=FeasibleRegion.WIDE,
        )

        meta_advice = advisor.advise(fingerprint)

        click.echo("Meta-Learning Advice:")
        click.echo(f"  Confidence: {meta_advice.confidence:.2f}")
        if meta_advice.recommended_backends:
            click.echo(f"  Recommended backends: {', '.join(meta_advice.recommended_backends)}")
        if meta_advice.reason_codes:
            click.echo("  Reasons:")
            for reason in meta_advice.reason_codes:
                click.echo(f"    - {reason}")

    except ImportError as e:
        click.echo(f"Cannot load meta-learning module: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error generating advice: {e}", err=True)
        sys.exit(1)
