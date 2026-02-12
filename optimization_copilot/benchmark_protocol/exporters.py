"""Export benchmark definitions and results to external formats."""
from __future__ import annotations

from typing import Any

from .schema import BenchmarkSchema
from .protocol import BenchmarkResult


class AtlasExporter:
    """Export to Olympus/Atlas format.

    Atlas uses a flat list of parameter dicts with bounds and a list of
    objectives with direction.
    """

    @staticmethod
    def export_schema(schema: BenchmarkSchema) -> dict[str, Any]:
        """Export benchmark schema to Atlas/Olympus format.

        Args:
            schema: The benchmark schema to export.

        Returns:
            Dictionary in Atlas-compatible format.
        """
        parameters = []
        for p in schema.parameters:
            param: dict[str, Any] = {
                "name": p.name,
                "type": p.type,
            }
            if p.type in ("continuous", "discrete"):
                param["low"] = p.lower
                param["high"] = p.upper
            elif p.type == "categorical":
                param["options"] = list(p.categories) if p.categories else []
            parameters.append(param)

        objectives = []
        for o in schema.objectives:
            obj: dict[str, Any] = {
                "name": o.name,
                "goal": o.direction,
            }
            if o.target is not None:
                obj["target"] = o.target
            objectives.append(obj)

        return {
            "name": schema.name,
            "version": schema.version,
            "description": schema.description,
            "domain": schema.domain,
            "parameters": parameters,
            "objectives": objectives,
            "constraints": list(schema.constraints),
            "budget": schema.evaluation_budget,
            "noise_level": schema.noise_level,
            "metadata": dict(schema.metadata),
        }

    @staticmethod
    def export_result(result: BenchmarkResult) -> dict[str, Any]:
        """Export benchmark result to Atlas/Olympus format.

        Args:
            result: The benchmark result to export.

        Returns:
            Dictionary in Atlas-compatible result format.
        """
        campaign = []
        for obs in result.observations:
            entry: dict[str, Any] = {}
            entry.update(obs.get("parameters", {}))
            entry.update(obs.get("kpi_values", {}))
            campaign.append(entry)

        return {
            "benchmark": result.benchmark_name,
            "algorithm": result.algorithm_name,
            "campaign": campaign,
            "best": {
                "value": result.best_value,
                "parameters": dict(result.best_parameters),
            },
            "summary": {
                "total_cost": result.total_cost,
                "wall_time": result.wall_time_seconds,
                "n_evaluations": result.n_evaluations,
            },
            "metadata": dict(result.metadata),
        }


class BayBEExporter:
    """Export to BayBE format.

    BayBE uses a searchspace with typed parameter entries and an
    objective configuration block.
    """

    @staticmethod
    def export_schema(schema: BenchmarkSchema) -> dict[str, Any]:
        """Export benchmark schema to BayBE format.

        Args:
            schema: The benchmark schema to export.

        Returns:
            Dictionary in BayBE-compatible format.
        """
        searchspace_params = []
        for p in schema.parameters:
            if p.type == "continuous":
                searchspace_params.append({
                    "type": "NumericalContinuousParameter",
                    "name": p.name,
                    "bounds": [p.lower, p.upper],
                })
            elif p.type == "discrete":
                searchspace_params.append({
                    "type": "NumericalDiscreteParameter",
                    "name": p.name,
                    "values": [p.lower, p.upper],
                })
            elif p.type == "categorical":
                searchspace_params.append({
                    "type": "CategoricalParameter",
                    "name": p.name,
                    "values": list(p.categories) if p.categories else [],
                })

        objective_targets = []
        for o in schema.objectives:
            target: dict[str, Any] = {
                "name": o.name,
                "mode": o.direction.upper(),
            }
            if o.target is not None:
                target["target_value"] = o.target
            objective_targets.append(target)

        objective_mode = "SINGLE" if len(schema.objectives) == 1 else "DESIRABILITY"

        return {
            "searchspace": {
                "parameters": searchspace_params,
                "constraints": list(schema.constraints),
            },
            "objective": {
                "mode": objective_mode,
                "targets": objective_targets,
            },
            "settings": {
                "evaluation_budget": schema.evaluation_budget,
                "noise_level": schema.noise_level,
            },
            "metadata": {
                "name": schema.name,
                "version": schema.version,
                "description": schema.description,
                "domain": schema.domain,
            },
        }

    @staticmethod
    def export_result(result: BenchmarkResult) -> dict[str, Any]:
        """Export benchmark result to BayBE format.

        Args:
            result: The benchmark result to export.

        Returns:
            Dictionary in BayBE-compatible result format.
        """
        measurements = []
        for obs in result.observations:
            measurement: dict[str, Any] = {
                "parameters": obs.get("parameters", {}),
                "targets": obs.get("kpi_values", {}),
            }
            measurements.append(measurement)

        return {
            "benchmark": result.benchmark_name,
            "recommender": result.algorithm_name,
            "measurements": measurements,
            "best_measurement": {
                "value": result.best_value,
                "parameters": dict(result.best_parameters),
            },
            "statistics": {
                "total_cost": result.total_cost,
                "wall_time_seconds": result.wall_time_seconds,
                "n_evaluations": result.n_evaluations,
            },
            "metadata": dict(result.metadata),
        }


class AxExporter:
    """Export to Ax/BoTorch format.

    Ax uses an experiment configuration with typed parameters and
    named metrics with optimization direction.
    """

    @staticmethod
    def export_schema(schema: BenchmarkSchema) -> dict[str, Any]:
        """Export benchmark schema to Ax/BoTorch format.

        Args:
            schema: The benchmark schema to export.

        Returns:
            Dictionary in Ax-compatible format.
        """
        ax_parameters = []
        for p in schema.parameters:
            if p.type == "continuous":
                ax_parameters.append({
                    "name": p.name,
                    "type": "range",
                    "value_type": "float",
                    "bounds": [p.lower, p.upper],
                })
            elif p.type == "discrete":
                ax_parameters.append({
                    "name": p.name,
                    "type": "range",
                    "value_type": "int",
                    "bounds": [p.lower, p.upper],
                })
            elif p.type == "categorical":
                ax_parameters.append({
                    "name": p.name,
                    "type": "choice",
                    "values": list(p.categories) if p.categories else [],
                })

        metrics = []
        for o in schema.objectives:
            metric: dict[str, Any] = {
                "name": o.name,
                "lower_is_better": o.direction == "minimize",
            }
            if o.target is not None:
                metric["target"] = o.target
            metrics.append(metric)

        return {
            "experiment": {
                "name": schema.name,
                "description": schema.description,
                "parameters": ax_parameters,
                "metrics": metrics,
            },
            "optimization_config": {
                "objective_name": schema.objectives[0].name if schema.objectives else None,
                "minimize": (
                    schema.objectives[0].direction == "minimize"
                    if schema.objectives
                    else True
                ),
            },
            "generation_strategy": {
                "max_trials": schema.evaluation_budget,
            },
            "metadata": {
                "version": schema.version,
                "domain": schema.domain,
                "noise_level": schema.noise_level,
            },
        }

    @staticmethod
    def export_result(result: BenchmarkResult) -> dict[str, Any]:
        """Export benchmark result to Ax/BoTorch format.

        Args:
            result: The benchmark result to export.

        Returns:
            Dictionary in Ax-compatible result format.
        """
        trials = []
        for i, obs in enumerate(result.observations):
            trial: dict[str, Any] = {
                "trial_index": i,
                "arm_parameters": obs.get("parameters", {}),
                "metric_values": obs.get("kpi_values", {}),
                "trial_status": "COMPLETED",
            }
            trials.append(trial)

        return {
            "experiment_name": result.benchmark_name,
            "algorithm": result.algorithm_name,
            "trials": trials,
            "best_trial": {
                "value": result.best_value,
                "parameters": dict(result.best_parameters),
            },
            "experiment_summary": {
                "total_cost": result.total_cost,
                "wall_time_seconds": result.wall_time_seconds,
                "n_trials": result.n_evaluations,
            },
            "metadata": dict(result.metadata),
        }
