"""Auto-generated insight reports combining interaction maps, equations, and diagnostics.

The :class:`InsightReportGenerator` orchestrates the full explain pipeline:
fANOVA decomposition, symbolic regression, and SVG visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from optimization_copilot.explain.interaction_map import InteractionMap
from optimization_copilot.explain.equation_discovery import EquationDiscovery, ParetoSolution
from optimization_copilot.visualization.svg_renderer import SVGCanvas


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class InsightReport:
    """Structured output from the insight report generator."""

    main_effects: dict[str, float]
    top_interactions: list[tuple[str, str, float]]
    equations: list[ParetoSolution]
    best_equation: str | None
    domain: str
    n_observations: int
    summary: str
    svg_charts: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# InsightReportGenerator
# ---------------------------------------------------------------------------

class InsightReportGenerator:
    """Generate insight reports from experimental data.

    Parameters
    ----------
    domain : str
        Scientific domain (used for contextual labels).
    n_trees : int
        Number of trees for the :class:`InteractionMap`.
    eq_population : int
        Population size for :class:`EquationDiscovery`.
    eq_generations : int
        Number of evolutionary generations.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        domain: str = "general",
        n_trees: int = 50,
        eq_population: int = 200,
        eq_generations: int = 50,
        seed: int = 42,
    ) -> None:
        self.domain = domain
        self.n_trees = n_trees
        self.eq_population = eq_population
        self.eq_generations = eq_generations
        self.seed = seed

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        X: list[list[float]],
        y: list[float],
        var_names: list[str] | None = None,
        diagnostics: dict | None = None,
    ) -> InsightReport:
        """Generate a full insight report.

        Parameters
        ----------
        X : list[list[float]]
            Feature matrix.
        y : list[float]
            Target values.
        var_names : list[str] | None
            Human-readable feature names.
        diagnostics : dict | None
            Optional diagnostic metadata to include.

        Returns
        -------
        InsightReport
            Structured report with visualisations.
        """
        n = len(y)
        n_features = len(X[0]) if X else 0
        names = var_names if var_names and len(var_names) == n_features else [
            f"x{i}" for i in range(n_features)
        ]

        # --- fANOVA decomposition ---
        imap = InteractionMap(n_trees=self.n_trees, seed=self.seed)
        imap.fit(X, y)

        raw_main = imap.compute_main_effects()
        main_effects: dict[str, float] = {}
        for i, importance in raw_main.items():
            name = names[i] if i < len(names) else f"x{i}"
            main_effects[name] = importance

        raw_interactions = imap.get_top_interactions(k=5)
        top_interactions: list[tuple[str, str, float]] = []
        for i, j, strength in raw_interactions:
            ni = names[i] if i < len(names) else f"x{i}"
            nj = names[j] if j < len(names) else f"x{j}"
            top_interactions.append((ni, nj, strength))

        # --- Symbolic regression ---
        eq = EquationDiscovery(
            population_size=self.eq_population,
            n_generations=self.eq_generations,
            seed=self.seed,
        )
        equations = eq.fit(X, y, var_names=names) if n >= 2 else []
        best = eq.best_equation()
        best_str = best.equation_string if best else None

        # --- Summary ---
        summary = self._build_summary(
            main_effects, top_interactions, best_str, n, self.domain, diagnostics
        )

        # --- SVG charts ---
        svg_charts: dict[str, str] = {}
        svg_charts["interaction_heatmap"] = imap.render_heatmap(feature_names=names)
        svg_charts["feature_importance"] = self._render_importance_bar_chart(
            main_effects
        )
        if equations:
            svg_charts["pareto_equations"] = self._render_pareto_chart(equations)

        return InsightReport(
            main_effects=main_effects,
            top_interactions=top_interactions,
            equations=equations,
            best_equation=best_str,
            domain=self.domain,
            n_observations=n,
            summary=summary,
            svg_charts=svg_charts,
        )

    # ------------------------------------------------------------------
    # SVG renderers
    # ------------------------------------------------------------------

    def render_svg(self, report: InsightReport) -> str:
        """Render key visualisations as a single combined SVG.

        Parameters
        ----------
        report : InsightReport
            A previously generated report.

        Returns
        -------
        str
            Combined SVG XML string.
        """
        # Combine the sub-charts into one canvas
        charts = list(report.svg_charts.values())
        if not charts:
            c = SVGCanvas(width=400, height=100, background="#ffffff")
            c.text(200, 50, "No data to visualise", font_size=14,
                   fill="#666", text_anchor="middle")
            return c.to_string()

        # Just return the feature importance chart as the primary SVG
        return report.svg_charts.get(
            "feature_importance",
            report.svg_charts.get("interaction_heatmap", charts[0]),
        )

    def render_text(self, report: InsightReport) -> str:
        """Render a plain-text summary of the report.

        Parameters
        ----------
        report : InsightReport
            A previously generated report.

        Returns
        -------
        str
            Multi-line text summary.
        """
        lines: list[str] = []
        lines.append(f"=== Insight Report ({report.domain}) ===")
        lines.append(f"Observations: {report.n_observations}")
        lines.append("")

        lines.append("Feature Importances:")
        for name, score in sorted(
            report.main_effects.items(), key=lambda t: t[1], reverse=True
        ):
            lines.append(f"  {name}: {score:.4f}")
        lines.append("")

        if report.top_interactions:
            lines.append("Top Interactions:")
            for f1, f2, strength in report.top_interactions:
                lines.append(f"  {f1} x {f2}: {strength:.4f}")
            lines.append("")

        if report.best_equation:
            lines.append(f"Best Equation: {report.best_equation}")
            lines.append("")

        if report.equations:
            lines.append("Pareto Front:")
            for sol in report.equations[:5]:
                lines.append(
                    f"  MSE={sol.mse:.4f}  complexity={sol.complexity}  "
                    f"{sol.equation_string}"
                )
            lines.append("")

        lines.append(report.summary)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        main_effects: dict[str, float],
        top_interactions: list[tuple[str, str, float]],
        best_equation: str | None,
        n_obs: int,
        domain: str,
        diagnostics: dict | None,
    ) -> str:
        """Build a human-readable text summary."""
        parts: list[str] = []
        parts.append(
            f"Analysis of {n_obs} observations in the {domain} domain."
        )

        if main_effects:
            top_feat = max(main_effects, key=main_effects.get)  # type: ignore[arg-type]
            parts.append(
                f"Most important feature: {top_feat} "
                f"(importance={main_effects[top_feat]:.3f})."
            )

        if top_interactions:
            f1, f2, s = top_interactions[0]
            parts.append(
                f"Strongest interaction: {f1} x {f2} (strength={s:.3f})."
            )

        if best_equation:
            parts.append(f"Best discovered equation: {best_equation}.")

        if diagnostics:
            parts.append(f"Diagnostics: {diagnostics}")

        return " ".join(parts)

    @staticmethod
    def _render_importance_bar_chart(main_effects: dict[str, float]) -> str:
        """Render a horizontal bar chart of feature importances."""
        if not main_effects:
            c = SVGCanvas(width=400, height=100, background="#ffffff")
            c.text(200, 50, "No features", font_size=14, fill="#666",
                   text_anchor="middle")
            return c.to_string()

        # Sort by importance descending
        sorted_feats = sorted(
            main_effects.items(), key=lambda t: t[1], reverse=True
        )

        n = len(sorted_feats)
        bar_height = 25
        gap = 8
        label_width = 100
        chart_width = 300
        margin_top = 40
        width = label_width + chart_width + 80
        height = margin_top + n * (bar_height + gap) + 20

        canvas = SVGCanvas(width=int(width), height=int(height), background="#ffffff")
        canvas.text(width / 2, 20, "Feature Importances",
                    font_size=14, fill="#333", text_anchor="middle")

        max_val = max(v for _, v in sorted_feats) if sorted_feats else 1.0
        if max_val < 1e-15:
            max_val = 1.0

        for i, (name, score) in enumerate(sorted_feats):
            y = margin_top + i * (bar_height + gap)
            bar_w = (score / max_val) * chart_width

            # Label
            canvas.text(label_width - 5, y + bar_height / 2 + 4,
                        name, font_size=11, fill="#333", text_anchor="end")
            # Bar
            canvas.rect(label_width, y, max(bar_w, 1), bar_height,
                        fill="#4a90d9", stroke="none")
            # Value
            canvas.text(label_width + bar_w + 5, y + bar_height / 2 + 4,
                        f"{score:.3f}", font_size=10, fill="#666")

        return canvas.to_string()

    @staticmethod
    def _render_pareto_chart(equations: list[ParetoSolution]) -> str:
        """Render a scatter plot of the Pareto front (MSE vs complexity)."""
        if not equations:
            c = SVGCanvas(width=400, height=100, background="#ffffff")
            c.text(200, 50, "No equations", font_size=14, fill="#666",
                   text_anchor="middle")
            return c.to_string()

        margin = 60
        plot_w = 320
        plot_h = 250
        width = plot_w + 2 * margin
        height = plot_h + 2 * margin

        canvas = SVGCanvas(width=int(width), height=int(height), background="#ffffff")
        canvas.text(width / 2, 20, "Pareto Front: MSE vs Complexity",
                    font_size=14, fill="#333", text_anchor="middle")

        # Data ranges
        mses = [s.mse for s in equations]
        cplxs = [float(s.complexity) for s in equations]
        min_mse = min(mses)
        max_mse = max(mses) if max(mses) > min_mse else min_mse + 1.0
        min_cplx = min(cplxs)
        max_cplx = max(cplxs) if max(cplxs) > min_cplx else min_cplx + 1.0

        def _scale_x(cplx: float) -> float:
            return margin + (cplx - min_cplx) / (max_cplx - min_cplx) * plot_w

        def _scale_y(mse: float) -> float:
            return margin + plot_h - (mse - min_mse) / (max_mse - min_mse) * plot_h

        # Axes
        canvas.line(margin, margin + plot_h, margin + plot_w, margin + plot_h,
                    stroke="#333")
        canvas.line(margin, margin, margin, margin + plot_h, stroke="#333")
        canvas.text(width / 2, height - 10, "Complexity",
                    font_size=11, fill="#333", text_anchor="middle")
        canvas.text(15, height / 2, "MSE", font_size=11, fill="#333",
                    text_anchor="middle")

        # Points
        for sol in equations:
            cx = _scale_x(float(sol.complexity))
            cy = _scale_y(sol.mse)
            canvas.circle(cx, cy, 4, fill="#e74c3c", stroke="#c0392b")

        # Connect with line (Pareto front)
        if len(equations) > 1:
            pts = [(_scale_x(float(s.complexity)), _scale_y(s.mse)) for s in equations]
            canvas.polyline(pts, stroke="#e74c3c", fill="none", stroke_width=1.5)

        return canvas.to_string()
