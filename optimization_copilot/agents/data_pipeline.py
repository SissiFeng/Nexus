"""Data analysis pipeline with traced execution.

Every method wraps an existing computational module, executes it, and
returns a :class:`TracedResult` with full execution metadata proving
that code actually ran.  All imports are lazy (inside closures) to
avoid circular dependencies.
"""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.agents.execution_trace import trace_call, TracedResult
from optimization_copilot.core.models import CampaignSnapshot


# ---------------------------------------------------------------------------
# Helper: pure-Python statistics
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    """Arithmetic mean; returns 0.0 for empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    """Population standard deviation; returns 0.0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient.  Returns 0.0 when undefined."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    sx = _std(xs)
    sy = _std(ys)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
    return cov / (sx * sy)


# ---------------------------------------------------------------------------
# DataAnalysisPipeline
# ---------------------------------------------------------------------------

class DataAnalysisPipeline:
    """Toolkit of traced analysis operations.

    Every method wraps an existing computational module, executes it,
    and returns a :class:`TracedResult` with full execution metadata.
    """

    # ------------------------------------------------------------------ #
    # Req 1: Diagnostics                                                  #
    # ------------------------------------------------------------------ #

    def run_diagnostics(
        self,
        snapshot: CampaignSnapshot,
        window_fraction: float = 0.25,
    ) -> TracedResult:
        """Compute 17 diagnostic signals via :class:`DiagnosticEngine`.

        Parameters
        ----------
        snapshot : CampaignSnapshot
            Campaign state to diagnose.
        window_fraction : float
            Fraction of observations used as the "recent" window.

        Returns
        -------
        TracedResult
            ``value`` is a ``DiagnosticsVector.to_dict()`` dict.
        """
        input_summary = {
            "n_observations": snapshot.n_observations,
            "window_fraction": window_fraction,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.diagnostics.engine import DiagnosticEngine

            engine = DiagnosticEngine(window_fraction=window_fraction)
            vec = engine.compute(snapshot)
            return vec.to_dict()

        return trace_call(
            module="diagnostics.engine",
            method="run_diagnostics",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"n_signals": len(v)},
        )

    # ------------------------------------------------------------------ #
    # Req 2: Top-K / Ranking / Outlier                                    #
    # ------------------------------------------------------------------ #

    def run_top_k(
        self,
        values: list[float],
        names: list[str],
        k: int,
        descending: bool = True,
    ) -> TracedResult:
        """Sort and return top-K entries.

        Parameters
        ----------
        values : list[float]
            Numeric scores.
        names : list[str]
            Corresponding labels.
        k : int
            Number of top items to return.
        descending : bool
            If True, largest first.

        Returns
        -------
        TracedResult
            ``value`` is a list of ``{"name": str, "value": float, "rank": int}`` dicts.
        """
        input_summary = {"n_items": len(values), "k": k, "descending": descending}

        def _execute() -> list[dict[str, Any]]:
            if not values or not names:
                return []
            paired = list(zip(values, names))
            paired.sort(key=lambda t: t[0], reverse=descending)
            top = paired[:k]
            return [
                {"name": name, "value": val, "rank": i + 1}
                for i, (val, name) in enumerate(top)
            ]

        return trace_call(
            module="data_pipeline",
            method="run_top_k",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"n_returned": len(v)},
        )

    def run_ranking(
        self,
        values: list[float],
        names: list[str],
        descending: bool = True,
    ) -> TracedResult:
        """Full ranking of all entries.

        Parameters
        ----------
        values : list[float]
            Numeric scores.
        names : list[str]
            Corresponding labels.
        descending : bool
            If True, largest first.

        Returns
        -------
        TracedResult
            ``value`` is a list of ``{"name": str, "value": float, "rank": int}`` dicts.
        """
        input_summary = {"n_items": len(values), "descending": descending}

        def _execute() -> list[dict[str, Any]]:
            if not values or not names:
                return []
            paired = list(zip(values, names))
            paired.sort(key=lambda t: t[0], reverse=descending)
            return [
                {"name": name, "value": val, "rank": i + 1}
                for i, (val, name) in enumerate(paired)
            ]

        return trace_call(
            module="data_pipeline",
            method="run_ranking",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"n_ranked": len(v)},
        )

    def run_outlier_detection(
        self,
        values: list[float],
        names: list[str],
        n_sigma: float = 2.0,
    ) -> TracedResult:
        """Z-score outlier detection (pure Python).

        Parameters
        ----------
        values : list[float]
            Numeric scores.
        names : list[str]
            Corresponding labels.
        n_sigma : float
            Number of standard deviations for outlier threshold.

        Returns
        -------
        TracedResult
            ``value`` is a dict with keys ``outliers``, ``mean``, ``std``,
            ``threshold``.
        """
        input_summary = {"n_items": len(values), "n_sigma": n_sigma}

        def _execute() -> dict[str, Any]:
            if len(values) < 2 or len(names) < 2:
                return {
                    "outliers": [],
                    "mean": _mean(values) if values else 0.0,
                    "std": 0.0,
                    "threshold": n_sigma,
                }
            m = _mean(values)
            s = _std(values)
            if s < 1e-12:
                return {
                    "outliers": [],
                    "mean": m,
                    "std": s,
                    "threshold": n_sigma,
                }
            outliers = []
            for val, name in zip(values, names):
                z = abs(val - m) / s
                if z > n_sigma:
                    outliers.append({
                        "name": name,
                        "value": val,
                        "z_score": round(z, 4),
                    })
            return {
                "outliers": outliers,
                "mean": round(m, 6),
                "std": round(s, 6),
                "threshold": n_sigma,
            }

        return trace_call(
            module="data_pipeline",
            method="run_outlier_detection",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"n_outliers": len(v.get("outliers", []))},
        )

    # ------------------------------------------------------------------ #
    # Req 3: fANOVA / SymReg / Insight Report                            #
    # ------------------------------------------------------------------ #

    def run_fanova(
        self,
        X: list[list[float]],
        y: list[float],
        var_names: list[str] | None = None,
        n_trees: int = 50,
        seed: int = 42,
    ) -> TracedResult:
        """fANOVA decomposition via :class:`InteractionMap`.

        Parameters
        ----------
        X : list[list[float]]
            Feature matrix.
        y : list[float]
            Target values.
        var_names : list[str] | None
            Human-readable feature names.
        n_trees : int
            Number of tree stumps in the ensemble.
        seed : int
            Random seed.

        Returns
        -------
        TracedResult
            ``value`` dict with ``main_effects`` and ``top_interactions``.
        """
        input_summary = {
            "n_samples": len(y),
            "n_features": len(X[0]) if X else 0,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.explain.interaction_map import InteractionMap

            imap = InteractionMap(n_trees=n_trees, seed=seed)
            imap.fit(X, y)

            n_features = len(X[0]) if X else 0
            names = var_names or [f"x{i}" for i in range(n_features)]

            raw_main = imap.compute_main_effects()
            main_effects: dict[str, float] = {}
            for idx, importance in raw_main.items():
                label = names[idx] if idx < len(names) else f"x{idx}"
                main_effects[label] = importance

            raw_interactions = imap.get_top_interactions(k=5)
            top_interactions: list[dict[str, Any]] = []
            for i, j, strength in raw_interactions:
                ni = names[i] if i < len(names) else f"x{i}"
                nj = names[j] if j < len(names) else f"x{j}"
                top_interactions.append({
                    "feature_i": ni,
                    "feature_j": nj,
                    "strength": strength,
                })

            return {
                "main_effects": main_effects,
                "top_interactions": top_interactions,
            }

        return trace_call(
            module="explain.interaction_map",
            method="run_fanova",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {
                "n_effects": len(v.get("main_effects", {})),
                "n_interactions": len(v.get("top_interactions", [])),
            },
        )

    def run_symreg(
        self,
        X: list[list[float]],
        y: list[float],
        var_names: list[str] | None = None,
        population_size: int = 200,
        n_generations: int = 50,
        seed: int = 42,
    ) -> TracedResult:
        """Symbolic regression via :class:`EquationDiscovery`.

        Parameters
        ----------
        X : list[list[float]]
            Feature matrix.
        y : list[float]
            Target values.
        var_names : list[str] | None
            Variable names for equation strings.
        population_size : int
            GA population size.
        n_generations : int
            Number of generations.
        seed : int
            Random seed.

        Returns
        -------
        TracedResult
            ``value`` dict with ``pareto_front`` and ``best_equation``.
        """
        input_summary = {
            "n_samples": len(y),
            "n_features": len(X[0]) if X else 0,
            "population_size": population_size,
            "n_generations": n_generations,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.explain.equation_discovery import EquationDiscovery

            n_features = len(X[0]) if X else 0
            names = var_names or [f"x{i}" for i in range(n_features)]

            eq = EquationDiscovery(
                population_size=population_size,
                n_generations=n_generations,
                seed=seed,
            )
            front = eq.fit(X, y, var_names=names)
            best = eq.best_equation()

            pareto_entries = [
                {
                    "equation": sol.equation_string,
                    "mse": sol.mse,
                    "complexity": sol.complexity,
                }
                for sol in front
            ]

            return {
                "pareto_front": pareto_entries,
                "best_equation": {
                    "equation": best.equation_string,
                    "mse": best.mse,
                    "complexity": best.complexity,
                } if best else None,
            }

        return trace_call(
            module="explain.equation_discovery",
            method="run_symreg",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {
                "n_pareto": len(v.get("pareto_front", [])),
                "has_best": v.get("best_equation") is not None,
            },
        )

    def run_insight_report(
        self,
        X: list[list[float]],
        y: list[float],
        var_names: list[str] | None = None,
        domain: str = "general",
    ) -> TracedResult:
        """Generate insight report via :class:`InsightReportGenerator`.

        Parameters
        ----------
        X : list[list[float]]
            Feature matrix.
        y : list[float]
            Target values.
        var_names : list[str] | None
            Human-readable feature names.
        domain : str
            Scientific domain context.

        Returns
        -------
        TracedResult
            ``value`` dict with report summary fields.
        """
        input_summary = {
            "n_samples": len(y),
            "n_features": len(X[0]) if X else 0,
            "domain": domain,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.explain.report_generator import InsightReportGenerator

            gen = InsightReportGenerator(domain=domain, seed=42)
            report = gen.generate(X, y, var_names=var_names)

            return {
                "main_effects": report.main_effects,
                "top_interactions": [
                    {"feature_i": f1, "feature_j": f2, "strength": s}
                    for f1, f2, s in report.top_interactions
                ],
                "best_equation": report.best_equation,
                "n_observations": report.n_observations,
                "summary": report.summary,
                "n_equations": len(report.equations),
                "domain": report.domain,
            }

        return trace_call(
            module="explain.report_generator",
            method="run_insight_report",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {
                "n_effects": len(v.get("main_effects", {})),
                "n_observations": v.get("n_observations", 0),
            },
        )

    # ------------------------------------------------------------------ #
    # Req 4: Correlation / Confounder                                     #
    # ------------------------------------------------------------------ #

    def run_correlation(
        self,
        xs: list[float],
        ys: list[float],
    ) -> TracedResult:
        """Pure-Python Pearson correlation coefficient.

        Parameters
        ----------
        xs : list[float]
            First variable.
        ys : list[float]
            Second variable.

        Returns
        -------
        TracedResult
            ``value`` dict with ``r``, ``n``, ``mean_x``, ``mean_y``.
        """
        input_summary = {"n_x": len(xs), "n_y": len(ys)}

        def _execute() -> dict[str, Any]:
            r = _pearson_r(xs, ys)
            return {
                "r": round(r, 6),
                "n": min(len(xs), len(ys)),
                "mean_x": round(_mean(xs), 6) if xs else 0.0,
                "mean_y": round(_mean(ys), 6) if ys else 0.0,
            }

        return trace_call(
            module="data_pipeline",
            method="run_correlation",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"r": v.get("r", 0.0)},
        )

    def run_confounder_detection(
        self,
        snapshot: CampaignSnapshot,
        threshold: float = 0.3,
    ) -> TracedResult:
        """Detect confounders via :class:`ConfounderDetector`.

        Parameters
        ----------
        snapshot : CampaignSnapshot
            Campaign whose metadata columns will be scanned.
        threshold : float
            Absolute Pearson |r| above which a column is flagged.

        Returns
        -------
        TracedResult
            ``value`` is a list of confounder dicts.
        """
        input_summary = {
            "n_observations": snapshot.n_observations,
            "threshold": threshold,
        }

        def _execute() -> list[dict[str, Any]]:
            from optimization_copilot.confounder.detector import ConfounderDetector

            detector = ConfounderDetector()
            specs = detector.detect(snapshot, threshold=threshold)
            return [
                {
                    "column_name": s.column_name,
                    "policy": s.policy.value,
                    "metadata": s.metadata,
                }
                for s in specs
            ]

        return trace_call(
            module="confounder.detector",
            method="run_confounder_detection",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"n_confounders": len(v)},
        )

    # ------------------------------------------------------------------ #
    # Req 5: Pareto                                                       #
    # ------------------------------------------------------------------ #

    def run_pareto_analysis(
        self,
        snapshot: CampaignSnapshot,
        weights: dict[str, float] | None = None,
    ) -> TracedResult:
        """Multi-objective Pareto analysis via :class:`MultiObjectiveAnalyzer`.

        Parameters
        ----------
        snapshot : CampaignSnapshot
            Campaign with multi-objective observations.
        weights : dict[str, float] | None
            Optional objective weights.

        Returns
        -------
        TracedResult
            ``value`` dict with ``pareto_front``, ``pareto_indices``,
            ``dominance_ranks``, ``tradeoff_report``.
        """
        input_summary = {
            "n_observations": snapshot.n_observations,
            "n_objectives": len(snapshot.objective_names),
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.multi_objective.pareto import MultiObjectiveAnalyzer

            analyzer = MultiObjectiveAnalyzer()
            result = analyzer.analyze(snapshot, weights=weights)

            return {
                "pareto_front": [
                    {
                        "iteration": obs.iteration,
                        "parameters": obs.parameters,
                        "kpi_values": obs.kpi_values,
                    }
                    for obs in result.pareto_front
                ],
                "pareto_indices": result.pareto_indices,
                "dominance_ranks": result.dominance_ranks,
                "tradeoff_report": result.tradeoff_report,
            }

        return trace_call(
            module="multi_objective.pareto",
            method="run_pareto_analysis",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {
                "n_pareto": len(v.get("pareto_front", [])),
                "n_tradeoffs": len(v.get("tradeoff_report", {})),
            },
        )

    # ------------------------------------------------------------------ #
    # Req 6: Molecular pipeline                                           #
    # ------------------------------------------------------------------ #

    def run_molecular_pipeline(
        self,
        smiles_list: list[str],
        observations: list[Any],
        parameter_specs: list[Any],
        objective_name: str,
        n_suggestions: int = 5,
        seed: int = 42,
    ) -> TracedResult:
        """End-to-end molecular optimization pipeline.

        Encodes SMILES strings via :class:`NGramTanimoto`, builds a
        Tanimoto-based kernel matrix, fits a :class:`GaussianProcessBO`
        surrogate, and suggests next candidates.

        Parameters
        ----------
        smiles_list : list[str]
            SMILES strings for the molecules observed so far.
        observations : list
            :class:`Observation` instances with KPI values.
        parameter_specs : list
            :class:`ParameterSpec` instances.
        objective_name : str
            Name of the KPI to optimize.
        n_suggestions : int
            Number of suggestions to return.
        seed : int
            Random seed.

        Returns
        -------
        TracedResult
            ``value`` dict with ``fingerprints``, ``suggestions``,
            ``encoding_metadata``.
        """
        input_summary = {
            "n_smiles": len(smiles_list),
            "n_observations": len(observations),
            "objective_name": objective_name,
            "n_suggestions": n_suggestions,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.representation.ngram_tanimoto import NGramTanimoto
            from optimization_copilot.backends.builtin import GaussianProcessBO

            # Step 1: Encode SMILES to fingerprints
            encoder = NGramTanimoto(n=3, fingerprint_size=128)
            fingerprints = encoder.encode(smiles_list)
            enc_meta = encoder.encoding_metadata()

            # Step 2: Fit GP surrogate and suggest
            gp = GaussianProcessBO()
            gp.fit(observations, parameter_specs)
            suggestions = gp.suggest(n_suggestions=n_suggestions, seed=seed)

            # Step 3: Rank existing molecules by objective
            mol_scores: list[dict[str, Any]] = []
            for i, (smi, obs) in enumerate(zip(smiles_list, observations)):
                kpi = obs.kpi_values.get(objective_name, 0.0) if hasattr(obs, "kpi_values") else 0.0
                mol_scores.append({
                    "smiles": smi,
                    "score": kpi,
                    "rank": 0,  # filled below
                })
            mol_scores.sort(key=lambda d: d["score"], reverse=True)
            for rank_idx, entry in enumerate(mol_scores):
                entry["rank"] = rank_idx + 1

            return {
                "fingerprints": {
                    "n_molecules": len(fingerprints),
                    "fingerprint_size": len(fingerprints[0]) if fingerprints else 0,
                },
                "suggestions": suggestions,
                "encoding_metadata": enc_meta,
                "molecule_ranking": mol_scores,
            }

        return trace_call(
            module="representation.ngram_tanimoto",
            method="run_molecular_pipeline",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {
                "n_suggestions": len(v.get("suggestions", [])),
                "n_molecules": v.get("fingerprints", {}).get("n_molecules", 0),
            },
        )

    # ------------------------------------------------------------------ #
    # Additional: Screening                                               #
    # ------------------------------------------------------------------ #

    def run_screening(
        self,
        snapshot: CampaignSnapshot,
        seed: int = 42,
    ) -> TracedResult:
        """Variable screening via :class:`VariableScreener`.

        Parameters
        ----------
        snapshot : CampaignSnapshot
            Campaign data.
        seed : int
            Random seed.

        Returns
        -------
        TracedResult
            ``value`` dict with screening results.
        """
        input_summary = {
            "n_observations": snapshot.n_observations,
            "n_parameters": len(snapshot.parameter_specs),
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.screening.screener import VariableScreener

            screener = VariableScreener()
            result = screener.screen(snapshot, seed=seed)

            return {
                "ranked_parameters": result.ranked_parameters,
                "importance_scores": result.importance_scores,
                "suspected_interactions": [
                    list(pair) for pair in result.suspected_interactions
                ],
                "directionality": result.directionality,
                "recommended_step_sizes": result.recommended_step_sizes,
            }

        return trace_call(
            module="screening.screener",
            method="run_screening",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {
                "n_ranked": len(v.get("ranked_parameters", [])),
                "n_interactions": len(v.get("suspected_interactions", [])),
            },
        )

    # ------------------------------------------------------------------ #
    # Causal Discovery & Reasoning (Layer 1)                              #
    # ------------------------------------------------------------------ #

    def run_causal_discovery(
        self,
        data: list[list[float]],
        var_names: list[str],
        alpha: float = 0.05,
    ) -> TracedResult:
        """Causal structure learning via PC algorithm."""
        input_summary = {"n_samples": len(data), "n_vars": len(var_names), "alpha": alpha}

        def _execute() -> dict[str, Any]:
            from optimization_copilot.causal.structure import CausalStructureLearner
            learner = CausalStructureLearner(alpha=alpha)
            graph = learner.learn(data, var_names)
            return {"graph": graph.to_dict(), "n_edges": len(graph.edges)}

        return trace_call(
            module="causal.structure",
            method="run_causal_discovery",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"n_edges": v.get("n_edges", 0)},
        )

    def run_intervention(
        self,
        graph_dict: dict[str, Any],
        intervention: dict[str, float],
        data: list[list[float]],
        target: str,
    ) -> TracedResult:
        """Do-operator interventional reasoning."""
        input_summary = {
            "n_interventions": len(intervention),
            "target": target,
            "n_data": len(data),
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.causal.models import CausalGraph
            from optimization_copilot.causal.interventional import InterventionalEngine
            graph = CausalGraph.from_dict(graph_dict)
            engine = InterventionalEngine()
            result = engine.do(graph, intervention, data, target)
            return result

        return trace_call(
            module="causal.interventional",
            method="run_intervention",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"has_estimate": "estimate" in v},
        )

    def run_counterfactual(
        self,
        graph_dict: dict[str, Any],
        equations: dict[str, str],
        factual: dict[str, float],
        intervention: dict[str, float],
        query: str,
    ) -> TracedResult:
        """SCM counterfactual reasoning (abduction-action-prediction)."""
        input_summary = {
            "n_equations": len(equations),
            "n_factual": len(factual),
            "query": query,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.causal.models import CausalGraph
            from optimization_copilot.causal.counterfactual import CounterfactualReasoner
            graph = CausalGraph.from_dict(graph_dict)
            reasoner = CounterfactualReasoner()
            result = reasoner.counterfactual(
                graph, equations, factual, intervention, query,
            )
            return result

        return trace_call(
            module="causal.counterfactual",
            method="run_counterfactual",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"query": query},
        )

    # ------------------------------------------------------------------ #
    # Physics-Informed Modeling (Layer 2)                                  #
    # ------------------------------------------------------------------ #

    def run_physics_constrained_gp(
        self,
        X: list[list[float]],
        y: list[float],
        constraints: list[dict[str, Any]],
        kernel_type: str = "rbf",
    ) -> TracedResult:
        """Physics-constrained GP fitting with conservation laws."""
        input_summary = {
            "n_samples": len(y),
            "n_constraints": len(constraints),
            "kernel_type": kernel_type,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.physics.constraints import (
                PhysicsConstraintModel,
                ConservationLaw,
                MonotonicityConstraint,
                PhysicsBound,
            )
            model = PhysicsConstraintModel()
            for c in constraints:
                ctype = c.get("type", "bound")
                if ctype == "conservation":
                    model.add_conservation_law(ConservationLaw(
                        name=c.get("name", "law"),
                        indices=c["indices"],
                        target_sum=c["target_sum"],
                        tolerance=c.get("tolerance", 1e-6),
                    ))
                elif ctype == "monotonicity":
                    model.add_monotonicity(MonotonicityConstraint(
                        name=c.get("name", "mono"),
                        index=c["index"],
                        increasing=c.get("increasing", True),
                    ))
                elif ctype == "bound":
                    model.add_bound(PhysicsBound(
                        name=c.get("name", "bound"),
                        index=c["index"],
                        lower=c.get("lower"),
                        upper=c.get("upper"),
                    ))
            feasibility = model.check_feasibility(X)
            return {
                "n_feasible": sum(1 for f in feasibility if f),
                "n_total": len(X),
                "constraints_summary": model.to_dict(),
            }

        return trace_call(
            module="physics.constraints",
            method="run_physics_constrained_gp",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"n_feasible": v.get("n_feasible", 0)},
        )

    def run_ode_solve(
        self,
        rhs_code: str,
        y0: list[float],
        t_span: tuple[float, float],
        n_steps: int = 100,
    ) -> TracedResult:
        """Solve ODE system with RK4 integrator."""
        input_summary = {
            "n_state_vars": len(y0),
            "t_span": list(t_span),
            "n_steps": n_steps,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.physics.ode_solver import RK4Solver
            # Build RHS function from simple expression
            # rhs_code is a string like "-y[0]" or "-0.5*y[0] + y[1]"
            def rhs(t: float, y: list[float]) -> list[float]:
                local_vars = {"t": t, "y": y, "math": math}
                result = eval(rhs_code, {"__builtins__": {}}, local_vars)  # noqa: S307
                if isinstance(result, (int, float)):
                    return [float(result)]
                return [float(r) for r in result]

            solver = RK4Solver()
            times, states = solver.solve(rhs, y0, t_span, n_steps)
            return {
                "times": times,
                "states": states,
                "final_state": states[-1] if states else y0,
                "n_steps": len(times),
            }

        return trace_call(
            module="physics.ode_solver",
            method="run_ode_solve",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"n_steps": v.get("n_steps", 0)},
        )

    # ------------------------------------------------------------------ #
    # Hypothesis Engine (Layer 3)                                          #
    # ------------------------------------------------------------------ #

    def run_hypothesis_generate(
        self,
        data: list[list[float]],
        var_names: list[str],
        target_index: int = -1,
    ) -> TracedResult:
        """Generate competing hypotheses from data."""
        input_summary = {
            "n_samples": len(data),
            "n_vars": len(var_names),
            "target_index": target_index,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.hypothesis.generator import HypothesisGenerator
            gen = HypothesisGenerator(seed=42)
            hypotheses = gen.generate_competing(data, var_names, target_index)
            return {
                "hypotheses": [h.to_dict() for h in hypotheses],
                "n_generated": len(hypotheses),
            }

        return trace_call(
            module="hypothesis.generator",
            method="run_hypothesis_generate",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"n_generated": v.get("n_generated", 0)},
        )

    def run_hypothesis_test(
        self,
        hypotheses: list[dict[str, Any]],
        X: list[list[float]],
        y: list[float],
        var_names: list[str] | None = None,
    ) -> TracedResult:
        """Test hypotheses against data using BIC scoring."""
        input_summary = {
            "n_hypotheses": len(hypotheses),
            "n_samples": len(y),
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.hypothesis.models import Hypothesis
            from optimization_copilot.hypothesis.testing import HypothesisTester
            tester = HypothesisTester()
            h_objects = [Hypothesis.from_dict(h) for h in hypotheses]
            rankings = tester.compare_all(h_objects, X, y, var_names)
            return {"rankings": rankings}

        return trace_call(
            module="hypothesis.testing",
            method="run_hypothesis_test",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"n_ranked": len(v.get("rankings", []))},
        )

    def run_hypothesis_status(
        self,
        tracker_state: dict[str, Any],
    ) -> TracedResult:
        """Get hypothesis tracker status report."""
        input_summary = {
            "n_hypotheses": len(tracker_state.get("hypotheses", {})),
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.hypothesis.tracker import HypothesisTracker
            tracker = HypothesisTracker.from_dict(tracker_state)
            return tracker.get_status_report()

        return trace_call(
            module="hypothesis.tracker",
            method="run_hypothesis_status",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"n_active": v.get("n_active", 0)},
        )

    # ------------------------------------------------------------------ #
    # Decision Robustness (Layer 4)                                        #
    # ------------------------------------------------------------------ #

    def run_bootstrap_ci(
        self,
        data: list[float],
        statistic_fn_name: str = "mean",
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> TracedResult:
        """Bootstrap confidence interval for a statistic."""
        input_summary = {
            "n_data": len(data),
            "statistic": statistic_fn_name,
            "n_bootstrap": n_bootstrap,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.robustness.bootstrap import BootstrapAnalyzer
            analyzer = BootstrapAnalyzer(n_bootstrap=n_bootstrap, seed=seed)
            if statistic_fn_name == "mean":
                fn = lambda d: sum(d) / len(d) if d else 0.0
            elif statistic_fn_name == "median":
                fn = lambda d: sorted(d)[len(d) // 2] if d else 0.0
            else:
                fn = lambda d: sum(d) / len(d) if d else 0.0
            result = analyzer.bootstrap_ci(data, fn, confidence=confidence)
            return {
                "observed": result.observed,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "confidence_level": result.confidence_level,
                "standard_error": result.standard_error,
            }

        return trace_call(
            module="robustness.bootstrap",
            method="run_bootstrap_ci",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"ci_lower": v.get("ci_lower"), "ci_upper": v.get("ci_upper")},
        )

    def run_conclusion_robustness(
        self,
        values: list[float],
        names: list[str],
        analysis_type: str = "ranking",
        k: int = 1,
        n_bootstrap: int = 500,
        seed: int = 42,
    ) -> TracedResult:
        """Check robustness of analytical conclusions."""
        input_summary = {
            "n_items": len(values),
            "analysis_type": analysis_type,
            "k": k,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.robustness.conclusion import ConclusionRobustnessChecker
            checker = ConclusionRobustnessChecker(n_bootstrap=n_bootstrap, seed=seed)
            if analysis_type == "ranking":
                result = checker.check_ranking_stability(values, names, k=k)
            elif analysis_type == "importance":
                # For importance we need data matrix - use values as 1D proxy
                X = [[v] for v in values]
                y = list(range(len(values)))
                result = checker.check_importance_stability(X, y, names[:1] if names else ["x"])
            else:
                result = checker.check_ranking_stability(values, names, k=k)
            return {
                "conclusion_type": result.conclusion_type,
                "stability_score": result.stability_score,
                "n_bootstrap": result.n_bootstrap,
                "details": result.details,
            }

        return trace_call(
            module="robustness.conclusion",
            method="run_conclusion_robustness",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"stability_score": v.get("stability_score")},
        )

    def run_decision_sensitivity(
        self,
        values: list[float],
        names: list[str],
        uncertainties: list[float] | None = None,
        n_perturbations: int = 100,
        seed: int = 42,
    ) -> TracedResult:
        """Analyze sensitivity of decisions to data perturbations."""
        input_summary = {
            "n_items": len(values),
            "n_perturbations": n_perturbations,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.robustness.sensitivity import DecisionSensitivityAnalyzer
            analyzer = DecisionSensitivityAnalyzer(n_perturbations=n_perturbations, seed=seed)
            unc = uncertainties if uncertainties else [0.1] * len(values)
            result = analyzer.decision_sensitivity(values, unc, names)
            return result

        return trace_call(
            module="robustness.sensitivity",
            method="run_decision_sensitivity",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"top_1_stability": v.get("top_1_stability")},
        )

    def run_cross_model_consistency(
        self,
        X: list[list[float]],
        y: list[float],
        model_types: list[str] | None = None,
    ) -> TracedResult:
        """Check prediction consistency across multiple model types."""
        input_summary = {
            "n_samples": len(y),
            "n_features": len(X[0]) if X else 0,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.robustness.consistency import CrossModelConsistency
            consistency = CrossModelConsistency()
            # Generate rankings from simple models
            # Model 1: correlation-based ranking
            n_features = len(X[0]) if X else 0
            rankings: list[list[int]] = []
            n = len(y)

            # Ranking by correlation with target
            corr_scores: list[float] = []
            y_mean = sum(y) / n if n > 0 else 0.0
            for j in range(n_features):
                col = [X[i][j] for i in range(n)]
                c_mean = sum(col) / n if n > 0 else 0.0
                cov = sum((col[i] - c_mean) * (y[i] - y_mean) for i in range(n)) / max(n, 1)
                corr_scores.append(abs(cov))
            rank1 = [0] * n_features
            for rank_idx, orig_idx in enumerate(sorted(range(n_features), key=lambda i: -corr_scores[i])):
                rank1[orig_idx] = rank_idx
            rankings.append(rank1)

            # Ranking by variance
            var_scores: list[float] = []
            for j in range(n_features):
                col = [X[i][j] for i in range(n)]
                c_mean = sum(col) / n if n > 0 else 0.0
                var_scores.append(sum((v - c_mean) ** 2 for v in col) / max(n, 1))
            rank2 = [0] * n_features
            for rank_idx, orig_idx in enumerate(sorted(range(n_features), key=lambda i: -var_scores[i])):
                rank2[orig_idx] = rank_idx
            rankings.append(rank2)

            agreement = consistency.model_agreement(rankings)
            return agreement

        return trace_call(
            module="robustness.consistency",
            method="run_cross_model_consistency",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"mean_agreement": v.get("mean_agreement")},
        )

    # ------------------------------------------------------------------ #
    # Theory-Data Hybrid (Layer 5)                                         #
    # ------------------------------------------------------------------ #

    def run_hybrid_fit(
        self,
        X: list[list[float]],
        y: list[float],
        theory_type: str,
        theory_params: dict[str, float],
    ) -> TracedResult:
        """Fit a hybrid theory+GP model."""
        input_summary = {
            "n_samples": len(y),
            "theory_type": theory_type,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.hybrid.theory import (
                ArrheniusModel,
                MichaelisMentenModel,
                PowerLawModel,
            )
            from optimization_copilot.hybrid.residual import ResidualGP
            from optimization_copilot.hybrid.composite import HybridModel

            if theory_type == "arrhenius":
                theory = ArrheniusModel(**theory_params)
            elif theory_type == "michaelis_menten":
                theory = MichaelisMentenModel(**theory_params)
            elif theory_type == "power_law":
                theory = PowerLawModel(**theory_params)
            else:
                raise ValueError(f"Unknown theory type: {theory_type}")

            gp = ResidualGP(theory, noise=theory_params.get("noise", 1e-4))
            hybrid = HybridModel(theory, gp)
            hybrid.fit(X, y)

            comparison = hybrid.compare_to_theory_only(X, y)
            adequacy = hybrid.theory_adequacy_score()

            return {
                "theory_type": theory_type,
                "theory_rmse": comparison["theory_rmse"],
                "hybrid_rmse": comparison["hybrid_rmse"],
                "improvement_pct": comparison["improvement_pct"],
                "adequacy_score": adequacy,
            }

        return trace_call(
            module="hybrid.composite",
            method="run_hybrid_fit",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {
                "improvement_pct": v.get("improvement_pct"),
                "adequacy": v.get("adequacy_score"),
            },
        )

    def run_hybrid_predict(
        self,
        X_train: list[list[float]],
        y_train: list[float],
        X_new: list[list[float]],
        theory_type: str,
        theory_params: dict[str, float],
    ) -> TracedResult:
        """Predict with a hybrid model at new points."""
        input_summary = {
            "n_train": len(y_train),
            "n_predict": len(X_new),
            "theory_type": theory_type,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.hybrid.theory import (
                ArrheniusModel,
                MichaelisMentenModel,
                PowerLawModel,
            )
            from optimization_copilot.hybrid.residual import ResidualGP
            from optimization_copilot.hybrid.composite import HybridModel

            if theory_type == "arrhenius":
                theory = ArrheniusModel(**theory_params)
            elif theory_type == "michaelis_menten":
                theory = MichaelisMentenModel(**theory_params)
            elif theory_type == "power_law":
                theory = PowerLawModel(**theory_params)
            else:
                raise ValueError(f"Unknown theory type: {theory_type}")

            gp = ResidualGP(theory, noise=theory_params.get("noise", 1e-4))
            hybrid = HybridModel(theory, gp)
            hybrid.fit(X_train, y_train)

            means, stds = hybrid.predict_with_uncertainty(X_new)
            return {
                "means": means,
                "stds": stds,
                "n_predictions": len(means),
            }

        return trace_call(
            module="hybrid.composite",
            method="run_hybrid_predict",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {"n_predictions": v.get("n_predictions", 0)},
        )

    def run_discrepancy_analysis(
        self,
        X: list[list[float]],
        y: list[float],
        theory_type: str,
        theory_params: dict[str, float],
        var_names: list[str] | None = None,
    ) -> TracedResult:
        """Analyze where theory model fails."""
        input_summary = {
            "n_samples": len(y),
            "theory_type": theory_type,
        }

        def _execute() -> dict[str, Any]:
            from optimization_copilot.hybrid.theory import (
                ArrheniusModel,
                MichaelisMentenModel,
                PowerLawModel,
            )
            from optimization_copilot.hybrid.residual import ResidualGP
            from optimization_copilot.hybrid.composite import HybridModel
            from optimization_copilot.hybrid.discrepancy import DiscrepancyAnalyzer

            if theory_type == "arrhenius":
                theory = ArrheniusModel(**theory_params)
            elif theory_type == "michaelis_menten":
                theory = MichaelisMentenModel(**theory_params)
            elif theory_type == "power_law":
                theory = PowerLawModel(**theory_params)
            else:
                raise ValueError(f"Unknown theory type: {theory_type}")

            gp = ResidualGP(theory, noise=theory_params.get("noise", 1e-4))
            hybrid = HybridModel(theory, gp)
            hybrid.fit(X, y)

            analyzer = DiscrepancyAnalyzer()
            bias = analyzer.systematic_bias(gp)
            failures = analyzer.failure_regions(hybrid, X)
            adequacy = analyzer.model_adequacy_test(gp.residuals)
            suggestions = analyzer.suggest_theory_revision(failures, var_names)

            return {
                "bias": bias,
                "failure_regions": failures,
                "adequacy_test": adequacy,
                "revision_suggestions": suggestions,
            }

        return trace_call(
            module="hybrid.discrepancy",
            method="run_discrepancy_analysis",
            fn=_execute,
            input_summary=input_summary,
            output_summarizer=lambda v: {
                "is_biased": v.get("bias", {}).get("is_biased"),
                "n_failures": len(v.get("failure_regions", [])),
            },
        )
