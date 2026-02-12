"""Closed-loop campaign manager.

The ``CampaignLoop`` is the single entry point for running an iterative
optimization campaign.  Each call to :meth:`run_iteration` produces a
:class:`CampaignDeliverable` with three layers of output.  When new
experimental results arrive, :meth:`ingest_results` updates the model,
compares predictions to actuals, and produces the next deliverable.

Typical workflow::

    loop = CampaignLoop(snapshot, candidates, "smiles", ["HER"], {"HER": "minimize"})
    deliverable = loop.run_iteration()
    # experimentalist runs top-5 from deliverable.next_batch ...
    deliverable = loop.ingest_results(new_observations)
    # repeat
"""

from __future__ import annotations

import time
from typing import Any

from optimization_copilot.core.models import CampaignSnapshot, Observation
from optimization_copilot.campaign.surrogate import (
    FingerprintSurrogate,
    PredictionResult,
)
from optimization_copilot.campaign.ranker import (
    CandidateRanker,
    RankedCandidate,
    RankedTable,
)
from optimization_copilot.campaign.stage_gate import (
    ScreeningProtocol,
    StageGateProtocol,
)
from optimization_copilot.campaign.output import (
    CampaignDeliverable,
    Layer1Dashboard,
    Layer2Intelligence,
    Layer3Reasoning,
    LearningReport,
    ModelMetrics,
)


class CampaignLoop:
    """Closed-loop optimization campaign manager.

    Manages the full cycle: fit surrogate → rank candidates → produce
    deliverable → ingest results → update model → next iteration.

    Parameters
    ----------
    snapshot : CampaignSnapshot
        Historical campaign data (observations + parameter specs).
    candidates : list[dict[str, Any]]
        Untested candidate parameter dicts (must include the SMILES param).
    smiles_param : str
        Name of the parameter containing SMILES strings.
    objectives : list[str]
        Objective names to fit and rank on.
    objective_directions : dict[str, str]
        ``{objective_name: "minimize" | "maximize"}``.
    fidelity_graph : object | None
        Optional :class:`FidelityGraph` for stage gate protocol.
    batch_size : int
        Number of candidates to recommend per iteration (default 5).
    acquisition_strategy : str
        Acquisition function: ``"ucb"``, ``"ei"``, or ``"pi"`` (default ``"ucb"``).
    kappa : float
        UCB exploration parameter (default 2.0).
    n_gram : int
        N-gram size for SMILES fingerprinting (default 3).
    fp_size : int
        Fingerprint bit-vector length (default 128).
    length_scale : float
        GP kernel length-scale (default 1.0).
    noise : float
        GP observation noise (default 1e-4).
    seed : int
        Random seed (default 42).
    """

    def __init__(
        self,
        snapshot: CampaignSnapshot,
        candidates: list[dict[str, Any]],
        smiles_param: str,
        objectives: list[str],
        objective_directions: dict[str, str],
        fidelity_graph: Any = None,
        batch_size: int = 5,
        acquisition_strategy: str = "ucb",
        kappa: float = 2.0,
        n_gram: int = 3,
        fp_size: int = 128,
        length_scale: float = 1.0,
        noise: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self._snapshot = snapshot
        self._candidates = list(candidates)
        self._smiles_param = smiles_param
        self._objectives = list(objectives)
        self._directions = dict(objective_directions)
        self._fidelity_graph = fidelity_graph
        self._batch_size = batch_size
        self._strategy = acquisition_strategy
        self._kappa = kappa
        self._seed = seed

        # Surrogate config
        self._n_gram = n_gram
        self._fp_size = fp_size
        self._length_scale = length_scale
        self._noise = noise

        # State
        self._surrogates: dict[str, FingerprintSurrogate] = {}
        self._ranker = CandidateRanker()
        self._iteration = 0
        self._history: list[CampaignDeliverable] = []
        self._last_predictions: dict[str, list[PredictionResult]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def iteration(self) -> int:
        """Current iteration number."""
        return self._iteration

    @property
    def n_candidates_remaining(self) -> int:
        """Number of untested candidates."""
        return len(self._candidates)

    @property
    def history(self) -> list[CampaignDeliverable]:
        """All deliverables produced so far."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Core: run_iteration
    # ------------------------------------------------------------------

    def run_iteration(self) -> CampaignDeliverable:
        """Run one iteration of the campaign loop.

        1. Extract observed SMILES and objective values.
        2. For each objective: fit surrogate → predict candidates → rank.
        3. Build stage gate protocol (if fidelity graph provided).
        4. Compute Pareto front (if multi-objective).
        5. Package into a 3-layer :class:`CampaignDeliverable`.

        Returns
        -------
        CampaignDeliverable
            Complete output with dashboard, intelligence, and reasoning.
        """
        self._iteration += 1
        self._last_predictions.clear()

        # -- Extract observed data --
        successful = self._snapshot.successful_observations
        observed_smiles = [
            obs.parameters[self._smiles_param]
            for obs in successful
            if self._smiles_param in obs.parameters
        ]

        # -- Candidate info --
        candidate_smiles = [c[self._smiles_param] for c in self._candidates]
        candidate_names = [
            c.get("name", c.get(self._smiles_param, f"Candidate-{i}"))
            for i, c in enumerate(self._candidates)
        ]

        # -- Fit surrogates and rank per objective --
        model_metrics: list[ModelMetrics] = []
        primary_table: RankedTable | None = None

        for obj_name in self._objectives:
            direction = self._directions.get(obj_name, "maximize")

            # Gather y-values for this objective (only where present)
            y_pairs: list[tuple[str, float]] = []
            for obs in successful:
                if (
                    obj_name in obs.kpi_values
                    and self._smiles_param in obs.parameters
                ):
                    y_pairs.append((
                        obs.parameters[self._smiles_param],
                        obs.kpi_values[obj_name],
                    ))

            if len(y_pairs) < 2:
                # Not enough data for GP — skip this objective
                continue

            train_smiles = [s for s, _ in y_pairs]
            train_y = [y for _, y in y_pairs]

            # Fit surrogate
            surrogate = FingerprintSurrogate(
                n_gram=self._n_gram,
                fp_size=self._fp_size,
                length_scale=self._length_scale,
                noise=self._noise,
                seed=self._seed,
            )
            fit_info = surrogate.fit(train_smiles, train_y, objective_name=obj_name)
            self._surrogates[obj_name] = surrogate

            # Predict on untested candidates
            if not candidate_smiles:
                continue
            predictions = surrogate.predict(candidate_smiles)
            self._last_predictions[obj_name] = predictions

            # Best observed value
            best_obs = (
                min(train_y) if direction == "minimize" else max(train_y)
            )

            # Rank
            pred_tuples = [(p.mean, p.std) for p in predictions]
            ranked_table = self._ranker.rank(
                candidate_names=candidate_names,
                candidate_params=self._candidates,
                predictions=pred_tuples,
                objective_name=obj_name,
                direction=direction,
                strategy=self._strategy,
                kappa=self._kappa,
                best_observed=best_obs,
            )
            if primary_table is None:
                primary_table = ranked_table

            model_metrics.append(ModelMetrics(
                objective_name=obj_name,
                n_training_points=fit_info.n_training,
                y_mean=fit_info.y_mean,
                y_std=fit_info.y_std,
                fit_duration_ms=fit_info.duration_ms,
            ))

        # -- Fallback: if no objective was fittable, produce empty table --
        if primary_table is None:
            primary_table = RankedTable(
                candidates=[],
                objective_name=self._objectives[0] if self._objectives else "",
                direction=self._directions.get(
                    self._objectives[0] if self._objectives else "", "maximize"
                ),
                acquisition_strategy=self._strategy,
            )

        # -- Stage gate protocol --
        protocol: ScreeningProtocol | None = None
        if self._fidelity_graph is not None:
            builder = StageGateProtocol()
            protocol = builder.build_protocol(self._fidelity_graph)

        # -- Pareto analysis (multi-objective) --
        pareto_summary: dict[str, Any] | None = None
        if len(self._objectives) > 1 and len(successful) >= 2:
            pareto_summary = self._compute_pareto()

        # -- Assemble deliverable --
        dashboard = Layer1Dashboard(
            ranked_table=primary_table,
            batch_size=self._batch_size,
            screening_protocol=protocol,
            iteration=self._iteration,
        )

        intelligence = Layer2Intelligence(
            pareto_summary=pareto_summary,
            model_metrics=model_metrics,
            learning_report=None,
            iteration_count=self._iteration,
        )

        reasoning = Layer3Reasoning()

        deliverable = CampaignDeliverable(
            iteration=self._iteration,
            dashboard=dashboard,
            intelligence=intelligence,
            reasoning=reasoning,
        )

        self._history.append(deliverable)
        return deliverable

    # ------------------------------------------------------------------
    # Data return: ingest_results
    # ------------------------------------------------------------------

    def ingest_results(
        self,
        new_observations: list[Observation],
    ) -> CampaignDeliverable:
        """Ingest new experimental results and produce next-round deliverable.

        1. Compare model predictions against actual results (learning report).
        2. Add new observations to the snapshot.
        3. Remove tested candidates from the pool.
        4. Re-run the campaign loop to produce updated recommendations.
        5. Attach the learning report to the deliverable.

        Parameters
        ----------
        new_observations : list[Observation]
            Freshly completed experiments.

        Returns
        -------
        CampaignDeliverable
            Updated deliverable with learning report attached.
        """
        # 1. Build learning report (compare predictions vs actuals)
        learning_report = self._build_learning_report(new_observations)

        # 2. Add new observations to snapshot
        for obs in new_observations:
            self._snapshot.observations.append(obs)
        self._snapshot.current_iteration += 1

        # 3. Remove tested candidates
        tested_smiles = {
            obs.parameters.get(self._smiles_param)
            for obs in new_observations
            if self._smiles_param in obs.parameters
        }
        self._candidates = [
            c for c in self._candidates
            if c.get(self._smiles_param) not in tested_smiles
        ]

        # 4. Re-run iteration
        deliverable = self.run_iteration()

        # 5. Attach learning report
        deliverable.intelligence.learning_report = learning_report

        return deliverable

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_learning_report(
        self,
        new_observations: list[Observation],
    ) -> LearningReport:
        """Compare prior predictions against actual experimental results."""
        errors: list[dict[str, Any]] = []
        obs_summaries: list[dict[str, Any]] = []

        for obs in new_observations:
            smiles = obs.parameters.get(self._smiles_param, "")
            name = obs.metadata.get("name", smiles[:20] if smiles else "unknown")

            obs_summaries.append({
                "name": name,
                "smiles": smiles,
                "kpi_values": dict(obs.kpi_values),
                "is_failure": obs.is_failure,
            })

            if obs.is_failure:
                continue

            for obj_name in self._objectives:
                if obj_name not in obs.kpi_values:
                    continue
                actual = obs.kpi_values[obj_name]

                # Find this candidate's prediction
                predicted_mu = self._find_prediction(smiles, obj_name)
                if predicted_mu is None:
                    continue

                error = actual - predicted_mu
                pct = abs(error / actual) * 100.0 if actual != 0 else 0.0

                errors.append({
                    "name": name,
                    "objective": obj_name,
                    "predicted": predicted_mu,
                    "actual": actual,
                    "error": error,
                    "pct_error": round(pct, 1),
                })

        # MAE
        mae = (
            sum(abs(e["error"]) for e in errors) / len(errors)
            if errors
            else 0.0
        )

        # Human-readable summary
        summaries: list[str] = []
        for e in errors:
            direction = "higher" if e["error"] > 0 else "lower"
            summaries.append(
                f"{e['name']} {e['objective']}: "
                f"{e['pct_error']:.0f}% {direction} than predicted"
            )

        return LearningReport(
            new_observations=obs_summaries,
            prediction_errors=errors,
            mean_absolute_error=mae,
            model_updated=True,
            summary="; ".join(summaries) if summaries else "No comparable predictions",
        )

    def _find_prediction(
        self,
        smiles: str,
        objective_name: str,
    ) -> float | None:
        """Look up the last prediction for a specific candidate/objective."""
        predictions = self._last_predictions.get(objective_name, [])
        for pred in predictions:
            if pred.smiles == smiles:
                return pred.mean
        return None

    def _compute_pareto(self) -> dict[str, Any] | None:
        """Run Pareto analysis on the current snapshot."""
        try:
            from optimization_copilot.multi_objective.pareto import (
                MultiObjectiveAnalyzer,
            )

            analyzer = MultiObjectiveAnalyzer()
            result = analyzer.analyze(self._snapshot)
            return {
                "n_pareto_optimal": len(result.pareto_front),
                "pareto_indices": result.pareto_indices,
                "dominance_ranks": result.dominance_ranks,
                "tradeoff_report": result.tradeoff_report,
            }
        except Exception:
            return None
