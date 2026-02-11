"""Infrastructure integration layer -- bundles all opt-in infrastructure modules.

Provides ``InfrastructureStack``, a unified container for the 10 infrastructure
modules.  All modules are opt-in: if not configured via ``InfrastructureConfig``,
the corresponding helper methods are safe no-ops.

Designed to be consumed by ``OptimizationEngine`` at four integration points:

1. **pre_decide** -- before ``MetaController.decide()``
2. **post_suggest** -- after ``plugin.suggest()``
3. **post_evaluate** -- after trial evaluation feedback
4. **check_stopping** -- at the top of each iteration

Full serialization (``to_dict`` / ``from_dict``) enables checkpointing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from optimization_copilot.infrastructure.auto_sampler import AutoSampler
from optimization_copilot.infrastructure.batch_scheduler import BatchScheduler
from optimization_copilot.infrastructure.constraint_engine import (
    Constraint,
    ConstraintEngine,
    ConstraintType,
)
from optimization_copilot.infrastructure.cost_tracker import CostTracker, TrialCost
from optimization_copilot.infrastructure.domain_encoding import (
    CustomDescriptorEncoding,
    EncodingPipeline,
    OneHotEncoding,
    OrdinalEncoding,
    SpatialEncoding,
)
from optimization_copilot.infrastructure.multi_fidelity import (
    FidelityLevel,
    MultiFidelityManager,
)
from optimization_copilot.infrastructure.parameter_importance import (
    ImportanceResult,
    ParameterImportanceAnalyzer,
)
from optimization_copilot.infrastructure.robust_optimizer import RobustOptimizer
from optimization_copilot.infrastructure.stopping_rule import StoppingDecision, StoppingRule
from optimization_copilot.infrastructure.transfer_learning import (
    CampaignData,
    TransferLearningEngine,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class InfrastructureConfig:
    """Configuration for the infrastructure stack.

    All fields are optional.  Unconfigured modules are disabled and
    their corresponding methods on ``InfrastructureStack`` become no-ops.
    """

    # -- Cost tracking --
    budget: float | None = None
    cost_field: str = "total_cost"

    # -- Stopping rules --
    max_trials: int | None = None
    max_cost: float | None = None
    improvement_patience: int = 15
    improvement_threshold: float = 0.01
    min_uncertainty: float | None = None

    # -- Constraints (list of dicts: name, constraint_type, evaluate, ...) --
    constraints: list[dict[str, Any]] = field(default_factory=list)

    # -- AutoSampler --
    available_backends: list[str] = field(default_factory=list)
    sampler_weights: dict[str, float] | None = None

    # -- Transfer learning --
    historical_campaigns: list[dict[str, Any]] = field(default_factory=list)

    # -- Batch scheduling --
    n_workers: int = 1
    batch_strategy: str = "simple"

    # -- Multi-fidelity --
    fidelity_levels: list[dict[str, Any]] = field(default_factory=list)

    # -- Parameter importance (always created; method configurable) --
    importance_method: str = "auto"

    # -- Domain encoding --
    encodings: dict[str, dict[str, Any]] = field(default_factory=dict)

    # -- Robust optimization --
    input_noise: dict[str, float] = field(default_factory=dict)
    n_perturbations: int = 20

    # -- Serialization helpers --

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain dict."""
        return {
            "budget": self.budget,
            "cost_field": self.cost_field,
            "max_trials": self.max_trials,
            "max_cost": self.max_cost,
            "improvement_patience": self.improvement_patience,
            "improvement_threshold": self.improvement_threshold,
            "min_uncertainty": self.min_uncertainty,
            "constraints": [dict(c) for c in self.constraints],
            "available_backends": list(self.available_backends),
            "sampler_weights": dict(self.sampler_weights) if self.sampler_weights else None,
            "historical_campaigns": [dict(c) for c in self.historical_campaigns],
            "n_workers": self.n_workers,
            "batch_strategy": self.batch_strategy,
            "fidelity_levels": [dict(f) for f in self.fidelity_levels],
            "importance_method": self.importance_method,
            "encodings": {k: dict(v) for k, v in self.encodings.items()},
            "input_noise": dict(self.input_noise),
            "n_perturbations": self.n_perturbations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InfrastructureConfig:
        """Deserialize from a plain dict."""
        return cls(
            budget=data.get("budget"),
            cost_field=data.get("cost_field", "total_cost"),
            max_trials=data.get("max_trials"),
            max_cost=data.get("max_cost"),
            improvement_patience=data.get("improvement_patience", 15),
            improvement_threshold=data.get("improvement_threshold", 0.01),
            min_uncertainty=data.get("min_uncertainty"),
            constraints=data.get("constraints", []),
            available_backends=data.get("available_backends", []),
            sampler_weights=data.get("sampler_weights"),
            historical_campaigns=data.get("historical_campaigns", []),
            n_workers=data.get("n_workers", 1),
            batch_strategy=data.get("batch_strategy", "simple"),
            fidelity_levels=data.get("fidelity_levels", []),
            importance_method=data.get("importance_method", "auto"),
            encodings=data.get("encodings", {}),
            input_noise=data.get("input_noise", {}),
            n_perturbations=data.get("n_perturbations", 20),
        )


# ---------------------------------------------------------------------------
# Integration Stack
# ---------------------------------------------------------------------------


class InfrastructureStack:
    """Unified integration layer for all infrastructure modules.

    All modules are opt-in.  Unconfigured modules are ``None`` and their
    corresponding helper methods are safe no-ops that return sensible
    defaults (empty lists, ``None``, etc.).

    Typical lifecycle in the engine loop::

        stack = InfrastructureStack(config)

        # 0. Warm start (before first iteration)
        warm_points = stack.warm_start_points(parameter_specs)

        for iteration in campaign:
            # 1. Check stopping
            decision = stack.check_stopping(n_trials, best_values)
            if decision and decision.should_stop:
                break

            # 2. Pre-decide: gather signals for MetaController
            signals = stack.pre_decide_signals(snapshot, diagnostics, fingerprint)
            strategy = meta_controller.decide(..., **signals)

            # 3. Post-suggest: filter + robustify
            candidates = stack.filter_candidates(raw_candidates, param_specs)
            adj_acq = stack.robustify_acquisition(candidates, acq_vals, param_specs)

            # 4. Post-evaluate: record results
            stack.record_trial(params, kpi_values, wall_time)
    """

    def __init__(self, config: InfrastructureConfig | None = None) -> None:
        cfg = config or InfrastructureConfig()
        self._config = cfg

        # -- Cost tracker (if budget is set) --
        self._cost_tracker: CostTracker | None = None
        if cfg.budget is not None:
            self._cost_tracker = CostTracker(
                budget=cfg.budget,
                cost_field=cfg.cost_field,
            )

        # -- Stopping rule (if any stopping criterion is configured) --
        self._stopping_rule: StoppingRule | None = None
        if any([
            cfg.max_trials is not None,
            cfg.max_cost is not None,
            cfg.min_uncertainty is not None,
        ]):
            self._stopping_rule = StoppingRule(
                max_trials=cfg.max_trials,
                max_cost=cfg.max_cost,
                improvement_patience=cfg.improvement_patience,
                improvement_threshold=cfg.improvement_threshold,
                min_uncertainty=cfg.min_uncertainty,
            )

        # -- Constraint engine (if constraints are defined) --
        self._constraint_engine: ConstraintEngine | None = None
        if cfg.constraints:
            constraints = self._build_constraints(cfg.constraints)
            self._constraint_engine = ConstraintEngine(constraints=constraints)

        # -- AutoSampler (if backends are specified) --
        self._auto_sampler: AutoSampler | None = None
        if cfg.available_backends:
            self._auto_sampler = AutoSampler(
                available_backends=cfg.available_backends,
                weights=cfg.sampler_weights,
            )

        # -- Transfer learning engine (if historical campaigns exist) --
        self._transfer_engine: TransferLearningEngine | None = None
        if cfg.historical_campaigns:
            self._transfer_engine = TransferLearningEngine()
            for camp in cfg.historical_campaigns:
                self._transfer_engine.register_campaign(
                    campaign_id=camp["campaign_id"],
                    parameter_specs=camp.get("parameter_specs", []),
                    observations=camp.get("observations", []),
                    metadata=camp.get("metadata"),
                )

        # -- Batch scheduler (if n_workers > 1) --
        self._batch_scheduler: BatchScheduler | None = None
        if cfg.n_workers > 1:
            self._batch_scheduler = BatchScheduler(
                n_workers=cfg.n_workers,
                batch_strategy=cfg.batch_strategy,
            )

        # -- Multi-fidelity manager (if fidelity levels are defined) --
        self._multi_fidelity: MultiFidelityManager | None = None
        if cfg.fidelity_levels:
            levels = [
                FidelityLevel(
                    level=f["level"],
                    name=f["name"],
                    cost_multiplier=f["cost_multiplier"],
                    correlation=f.get("correlation", 0.8),
                )
                for f in cfg.fidelity_levels
            ]
            self._multi_fidelity = MultiFidelityManager(levels)

        # -- Parameter importance analyzer (always created) --
        self._importance_analyzer = ParameterImportanceAnalyzer(
            method=cfg.importance_method,
        )

        # -- Encoding pipeline (if encodings are configured) --
        self._encoding_pipeline: EncodingPipeline | None = None
        if cfg.encodings:
            self._encoding_pipeline = self._build_encoding_pipeline(cfg.encodings)

        # -- Robust optimizer (if input noise is specified) --
        self._robust_optimizer: RobustOptimizer | None = None
        if cfg.input_noise:
            self._robust_optimizer = RobustOptimizer(
                input_noise=cfg.input_noise,
                n_perturbations=cfg.n_perturbations,
            )

    # ── Module accessors (read-only) ──────────────────────────

    @property
    def cost_tracker(self) -> CostTracker | None:
        return self._cost_tracker

    @property
    def stopping_rule(self) -> StoppingRule | None:
        return self._stopping_rule

    @property
    def constraint_engine(self) -> ConstraintEngine | None:
        return self._constraint_engine

    @property
    def auto_sampler(self) -> AutoSampler | None:
        return self._auto_sampler

    @property
    def transfer_engine(self) -> TransferLearningEngine | None:
        return self._transfer_engine

    @property
    def batch_scheduler(self) -> BatchScheduler | None:
        return self._batch_scheduler

    @property
    def multi_fidelity(self) -> MultiFidelityManager | None:
        return self._multi_fidelity

    @property
    def importance_analyzer(self) -> ParameterImportanceAnalyzer:
        return self._importance_analyzer

    @property
    def encoding_pipeline(self) -> EncodingPipeline | None:
        return self._encoding_pipeline

    @property
    def robust_optimizer(self) -> RobustOptimizer | None:
        return self._robust_optimizer

    # ── Pre-decide helpers ────────────────────────────────────

    def pre_decide_signals(
        self,
        snapshot: Any,
        diagnostics: dict[str, float],
        fingerprint: Any,
    ) -> dict[str, Any]:
        """Prepare extra keyword arguments for ``MetaController.decide()``.

        Returns a dict suitable for ``**``-unpacking into ``decide()``.
        Keys with ``None`` values are omitted so they fall back to
        the controller's defaults.

        Returned keys (when active):
        - ``cost_signals``: dict with budget info from CostTracker
        - ``backend_policy``: str backend hint from AutoSampler
        """
        signals: dict[str, Any] = {}

        # Cost signals
        if self._cost_tracker is not None:
            signals["cost_signals"] = {
                "total_spent": self._cost_tracker.total_spent,
                "remaining_budget": self._cost_tracker.remaining_budget,
                "average_cost_per_trial": self._cost_tracker.average_cost_per_trial,
                "estimated_remaining_trials": self._cost_tracker.estimated_remaining_trials(),
                "n_trials_recorded": self._cost_tracker.n_trials,
            }

        # AutoSampler hint
        if self._auto_sampler is not None:
            n_obs = 0
            has_constraints = False
            is_multi_obj = False
            n_dims = 1
            noise_level = "low"
            phase = "learning"
            budget_remaining: float | None = None

            # Extract what we can from snapshot
            if hasattr(snapshot, "n_observations"):
                n_obs = snapshot.n_observations
            if hasattr(snapshot, "constraints"):
                has_constraints = bool(snapshot.constraints)
            if hasattr(snapshot, "objective_names"):
                is_multi_obj = len(snapshot.objective_names) > 1
            if hasattr(snapshot, "parameter_specs"):
                n_dims = len(snapshot.parameter_specs)

            # Noise from fingerprint
            if hasattr(fingerprint, "noise_regime"):
                noise_level = fingerprint.noise_regime.value if hasattr(
                    fingerprint.noise_regime, "value"
                ) else str(fingerprint.noise_regime)

            # Phase from diagnostics
            if "phase" in diagnostics:
                phase = str(diagnostics["phase"])

            # Budget from cost tracker
            if self._cost_tracker is not None:
                budget_remaining_val = self._cost_tracker.remaining_budget
                if budget_remaining_val is not None:
                    est = self._cost_tracker.estimated_remaining_trials()
                    budget_remaining = float(est) if est is not None else None

            result = self._auto_sampler.select(
                phase=phase,
                n_observations=n_obs,
                has_constraints=has_constraints,
                is_multi_objective=is_multi_obj,
                n_dimensions=n_dims,
                noise_level=noise_level,
                budget_remaining=budget_remaining,
            )
            signals["backend_policy"] = result.backend_name

        return signals

    def check_stopping(
        self,
        n_trials: int,
        best_values: list[float] | None = None,
        current_uncertainty: float | None = None,
    ) -> StoppingDecision | None:
        """Check whether stopping criteria are met.

        Returns a ``StoppingDecision`` if any criterion triggers,
        or ``None`` if no stopping rule is configured.
        """
        if self._stopping_rule is None:
            return None

        total_cost = 0.0
        if self._cost_tracker is not None:
            total_cost = self._cost_tracker.total_spent

        return self._stopping_rule.should_stop(
            n_trials=n_trials,
            total_cost=total_cost,
            best_values=best_values,
            current_uncertainty=current_uncertainty,
        )

    # ── Post-suggest helpers ──────────────────────────────────

    def filter_candidates(
        self,
        candidates: list[dict[str, Any]],
        parameter_specs: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Apply constraint filtering to candidates.

        If no constraint engine is configured, returns all candidates
        unchanged.
        """
        if self._constraint_engine is None or not candidates:
            return list(candidates)
        return self._constraint_engine.filter_candidates(candidates, parameter_specs)

    def robustify_acquisition(
        self,
        candidates: list[dict[str, Any]],
        acquisition_values: list[float],
        parameter_specs: list[dict[str, Any]],
    ) -> list[float]:
        """Penalize candidates in noise-sensitive regions.

        If no robust optimizer is configured, returns the original
        acquisition values unchanged.

        Uses ``robustify_candidates`` (penalty-based) rather than the
        full MC ``robustify_acquisition`` to avoid requiring an
        acquisition function callable.
        """
        if self._robust_optimizer is None:
            return list(acquisition_values)
        return self._robust_optimizer.robustify_candidates(
            candidates=candidates,
            acquisition_values=acquisition_values,
            parameter_specs=parameter_specs,
        )

    def weight_by_constraints(
        self,
        acquisition_values: list[float],
        candidates: list[dict[str, Any]],
    ) -> list[float]:
        """Weight acquisition values by constraint feasibility.

        Applies soft-constraint penalties and unknown-constraint
        feasibility probabilities.  No-op if no constraint engine.
        """
        if self._constraint_engine is None:
            return list(acquisition_values)
        return self._constraint_engine.constraint_weighted_acquisition(
            acquisition_values=acquisition_values,
            candidates=candidates,
        )

    # ── Post-evaluate helpers ─────────────────────────────────

    def record_trial(
        self,
        trial_params: dict[str, Any],
        kpi_values: dict[str, float],
        wall_time: float = 0.0,
        resource_cost: float = 0.0,
        compute_cost: float = 0.0,
        trial_id: str = "",
        constraint_results: dict[str, bool] | None = None,
        fidelity_level: int | None = None,
    ) -> None:
        """Update all relevant modules with a completed trial.

        Modules updated (when active):
        - **CostTracker**: records cost breakdown.
        - **ConstraintEngine**: updates unknown constraint models.
        - **MultiFidelityManager**: adds observation at specified fidelity.
        """
        # Cost tracking
        if self._cost_tracker is not None:
            cost = TrialCost(
                trial_id=trial_id or f"trial_{self._cost_tracker.n_trials}",
                wall_time_seconds=wall_time,
                resource_cost=resource_cost,
                compute_cost=compute_cost,
                fidelity_level=fidelity_level or 0,
            )
            self._cost_tracker.record_trial(cost)

        # Constraint model updates
        if self._constraint_engine is not None and constraint_results:
            self._constraint_engine.update_unknown_constraints(
                x=trial_params,
                constraint_results=constraint_results,
            )

        # Multi-fidelity observation
        if self._multi_fidelity is not None and fidelity_level is not None:
            obs: dict[str, Any] = dict(trial_params)
            if kpi_values:
                first_kpi = next(iter(kpi_values.values()))
                obs["objective"] = first_kpi
            self._multi_fidelity.add_observation(obs, fidelity_level)

    # ── Transfer learning ─────────────────────────────────────

    def warm_start_points(
        self,
        parameter_specs: list[dict[str, Any]],
        n_points: int = 5,
        min_similarity: float = 0.3,
        current_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Get warm-start points from similar historical campaigns.

        Returns an empty list if no transfer engine is configured or
        no sufficiently similar campaigns are found.
        """
        if self._transfer_engine is None:
            return []
        return self._transfer_engine.warm_start_points(
            current_specs=parameter_specs,
            n_points=n_points,
            min_similarity=min_similarity,
            current_metadata=current_metadata,
        )

    # ── Encoding ──────────────────────────────────────────────

    def encode_params(self, params: dict[str, Any]) -> list[float]:
        """Encode parameters using domain encodings.

        Returns a flat feature vector.  If no encoding pipeline is
        configured, extracts numeric values in dict iteration order.
        """
        if self._encoding_pipeline is not None:
            return self._encoding_pipeline.encode_params(params)
        # Fallback: collect numeric values
        return [
            float(v) for v in params.values()
            if isinstance(v, (int, float))
        ]

    def decode_features(
        self,
        features: list[float],
        param_names: list[str],
    ) -> dict[str, Any]:
        """Decode features back to a parameter dict.

        Requires the encoding pipeline; raises ``RuntimeError`` if
        no pipeline is configured.
        """
        if self._encoding_pipeline is None:
            raise RuntimeError(
                "Cannot decode features without a configured encoding pipeline"
            )
        return self._encoding_pipeline.decode_features(features, param_names)

    # ── Analysis ──────────────────────────────────────────────

    def analyze_importance(
        self,
        observations: list[dict[str, Any]],
        parameter_specs: list[dict[str, Any]],
        objective_key: str = "objective",
    ) -> ImportanceResult | None:
        """Run parameter importance analysis.

        Returns ``None`` if there are no valid observations.
        """
        if not observations or not parameter_specs:
            return None
        try:
            return self._importance_analyzer.analyze(
                observations=observations,
                parameter_specs=parameter_specs,
                objective_key=objective_key,
            )
        except ValueError:
            return None

    # ── Batch scheduling ──────────────────────────────────────

    def schedule_batch(
        self,
        suggestions: list[dict[str, Any]],
    ) -> list[Any]:
        """Create trial objects from suggestions via the batch scheduler.

        Returns the raw suggestions as a list if no batch scheduler
        is configured.
        """
        if self._batch_scheduler is None:
            return list(suggestions)
        return self._batch_scheduler.request_batch(suggestions)

    def needs_backfill(self) -> bool:
        """Check whether idle workers need new suggestions."""
        if self._batch_scheduler is None:
            return False
        return self._batch_scheduler.needs_backfill()

    def backfill_count(self) -> int:
        """Number of new suggestions needed to fill idle workers."""
        if self._batch_scheduler is None:
            return 0
        return self._batch_scheduler.backfill_count()

    # ── Multi-fidelity ────────────────────────────────────────

    def suggest_fidelity(
        self,
        candidate: dict[str, Any],
        budget_remaining: float | None = None,
    ) -> dict[str, Any] | None:
        """Suggest fidelity level for a candidate.

        Returns a dict with ``level``, ``name``, ``cost_multiplier``
        or ``None`` if multi-fidelity is not configured.
        """
        if self._multi_fidelity is None:
            return None
        fl = self._multi_fidelity.suggest_fidelity(candidate, budget_remaining)
        return {
            "level": fl.level,
            "name": fl.name,
            "cost_multiplier": fl.cost_multiplier,
        }

    # ── Summary ───────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a dict summarizing all active modules and their state."""
        result: dict[str, Any] = {
            "active_modules": [],
        }

        if self._cost_tracker is not None:
            result["active_modules"].append("cost_tracker")
            result["cost_tracker"] = {
                "budget": self._cost_tracker.budget,
                "total_spent": self._cost_tracker.total_spent,
                "remaining_budget": self._cost_tracker.remaining_budget,
                "n_trials": self._cost_tracker.n_trials,
            }

        if self._stopping_rule is not None:
            result["active_modules"].append("stopping_rule")
            result["stopping_rule"] = {
                "active_criteria": self._stopping_rule.active_criteria(),
            }

        if self._constraint_engine is not None:
            result["active_modules"].append("constraint_engine")
            result["constraint_engine"] = self._constraint_engine.feasibility_summary()

        if self._auto_sampler is not None:
            result["active_modules"].append("auto_sampler")
            result["auto_sampler"] = {
                "available_backends": self._auto_sampler.available_backends,
                "n_selections": len(self._auto_sampler.selection_history),
            }

        if self._transfer_engine is not None:
            result["active_modules"].append("transfer_engine")
            result["transfer_engine"] = {
                "n_campaigns": self._transfer_engine.n_campaigns,
                "campaign_ids": self._transfer_engine.campaign_ids,
            }

        if self._batch_scheduler is not None:
            result["active_modules"].append("batch_scheduler")
            result["batch_scheduler"] = self._batch_scheduler.summary()

        if self._multi_fidelity is not None:
            result["active_modules"].append("multi_fidelity")
            result["multi_fidelity"] = self._multi_fidelity.fidelity_summary()

        result["active_modules"].append("importance_analyzer")
        result["importance_analyzer"] = {
            "method": self._importance_analyzer.method,
        }

        if self._encoding_pipeline is not None:
            result["active_modules"].append("encoding_pipeline")
            result["encoding_pipeline"] = {
                "param_names": self._encoding_pipeline.param_names,
                "total_features": self._encoding_pipeline.total_features(),
            }

        if self._robust_optimizer is not None:
            result["active_modules"].append("robust_optimizer")
            result["robust_optimizer"] = {
                "noise_config": self._robust_optimizer.noise_config,
                "n_perturbations": self._robust_optimizer.n_perturbations,
            }

        return result

    # ── Serialization ─────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full stack state for checkpointing.

        Each module serializes its own state via its ``to_dict()``
        method.  The config is also stored so the stack can be
        restored without the original ``InfrastructureConfig``.
        """
        data: dict[str, Any] = {
            "config": self._config.to_dict(),
        }

        if self._cost_tracker is not None:
            data["cost_tracker"] = self._cost_tracker.to_dict()

        if self._stopping_rule is not None:
            data["stopping_rule"] = self._stopping_rule.to_dict()

        if self._constraint_engine is not None:
            data["constraint_engine"] = self._constraint_engine.to_dict()

        if self._auto_sampler is not None:
            data["auto_sampler"] = self._auto_sampler.to_dict()

        if self._transfer_engine is not None:
            data["transfer_engine"] = self._transfer_engine.to_dict()

        if self._batch_scheduler is not None:
            data["batch_scheduler"] = self._batch_scheduler.to_dict()

        if self._multi_fidelity is not None:
            data["multi_fidelity"] = self._multi_fidelity.to_dict()

        data["importance_analyzer"] = self._importance_analyzer.to_dict()

        if self._encoding_pipeline is not None:
            data["encoding_pipeline"] = self._encoding_pipeline.to_dict()

        if self._robust_optimizer is not None:
            data["robust_optimizer"] = self._robust_optimizer.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InfrastructureStack:
        """Restore a stack from serialized state.

        Reconstructs the config, creates the stack (which initializes
        modules from config), then overlays persisted module state
        where available.
        """
        config = InfrastructureConfig.from_dict(data["config"])
        stack = cls(config)

        # Overlay persisted state onto initialized modules
        if "cost_tracker" in data and stack._cost_tracker is not None:
            stack._cost_tracker = CostTracker.from_dict(data["cost_tracker"])

        if "stopping_rule" in data and stack._stopping_rule is not None:
            stack._stopping_rule = StoppingRule.from_dict(data["stopping_rule"])

        if "auto_sampler" in data and stack._auto_sampler is not None:
            stack._auto_sampler = AutoSampler.from_dict(data["auto_sampler"])

        if "transfer_engine" in data and stack._transfer_engine is not None:
            stack._transfer_engine = TransferLearningEngine.from_dict(
                data["transfer_engine"]
            )

        if "batch_scheduler" in data and stack._batch_scheduler is not None:
            stack._batch_scheduler = BatchScheduler.from_dict(
                data["batch_scheduler"]
            )

        if "multi_fidelity" in data and stack._multi_fidelity is not None:
            stack._multi_fidelity = MultiFidelityManager.from_dict(
                data["multi_fidelity"]
            )

        if "robust_optimizer" in data and stack._robust_optimizer is not None:
            stack._robust_optimizer = RobustOptimizer.from_dict(
                data["robust_optimizer"]
            )

        # Note: ConstraintEngine and EncodingPipeline are not fully
        # round-trippable because Constraint.evaluate is a callable
        # and Encoding subclasses need type-specific reconstruction.
        # They are re-created from config above, which is sufficient
        # for most use cases.

        return stack

    # ── Private helpers ───────────────────────────────────────

    @staticmethod
    def _build_constraints(
        constraint_defs: list[dict[str, Any]],
    ) -> list[Constraint]:
        """Build ``Constraint`` objects from config dicts.

        Each dict must have ``name`` and ``constraint_type`` keys.
        Optional keys: ``evaluate`` (callable), ``tolerance``,
        ``safety_probability``.
        """
        constraints: list[Constraint] = []
        for d in constraint_defs:
            ct = ConstraintType(d.get("constraint_type", "known_hard"))
            constraints.append(
                Constraint(
                    name=d["name"],
                    constraint_type=ct,
                    evaluate=d.get("evaluate"),
                    tolerance=d.get("tolerance", 0.0),
                    safety_probability=d.get("safety_probability", 0.95),
                )
            )
        return constraints

    @staticmethod
    def _build_encoding_pipeline(
        encoding_config: dict[str, dict[str, Any]],
    ) -> EncodingPipeline:
        """Build an ``EncodingPipeline`` from config dicts.

        Each entry maps a parameter name to an encoding spec dict
        with a ``type`` key and type-specific configuration.

        Supported types:
        - ``one_hot``: requires ``categories`` list
        - ``ordinal``: requires ``ordered_categories`` list
        - ``custom_descriptor``: requires ``descriptor_table`` dict
        - ``spatial``: optional ``coord_type`` (default ``"latlon"``)
        """
        _BUILDERS: dict[str, Callable[..., Any]] = {
            "one_hot": lambda d: OneHotEncoding(d["categories"]),
            "ordinal": lambda d: OrdinalEncoding(d["ordered_categories"]),
            "custom_descriptor": lambda d: CustomDescriptorEncoding(d["descriptor_table"]),
            "spatial": lambda d: SpatialEncoding(d.get("coord_type", "latlon")),
        }

        pipeline = EncodingPipeline()
        for param_name, spec in encoding_config.items():
            enc_type = spec.get("type", "")
            builder = _BUILDERS.get(enc_type)
            if builder is not None:
                pipeline.add_encoding(param_name, builder(spec))
        return pipeline

    def __repr__(self) -> str:
        modules = self.summary().get("active_modules", [])
        return f"InfrastructureStack(active={modules})"
