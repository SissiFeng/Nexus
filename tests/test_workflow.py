"""Comprehensive tests for the multi-stage workflow package.

Covers ExperimentStage, StageDAG, ProxyModel, ContinueValue,
MultiStageBayesianOptimizer, and visualization utilities.
"""

from __future__ import annotations

import math
import random

import pytest

from optimization_copilot.workflow.stage import ExperimentStage, StageDAG
from optimization_copilot.workflow.proxy_model import ProxyModel
from optimization_copilot.workflow.continue_value import ContinueValue
from optimization_copilot.workflow.multi_stage_bo import (
    MultiStageBayesianOptimizer,
    StageResult,
)
from optimization_copilot.workflow.visualization import (
    render_stage_flow,
    render_savings_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_linear_dag() -> StageDAG:
    """Create a simple 3-stage linear DAG: prep -> run -> analyze."""
    dag = StageDAG()
    dag.add_stage(ExperimentStage(
        name="prep", parameters=["temp", "pressure"],
        kpis=["purity"], cost=1.0, duration_hours=0.5,
    ))
    dag.add_stage(ExperimentStage(
        name="run", parameters=["temp", "pressure", "catalyst"],
        kpis=["yield"], cost=5.0, duration_hours=2.0,
        dependencies=["prep"],
    ))
    dag.add_stage(ExperimentStage(
        name="analyze", parameters=["temp"],
        kpis=["stability"], cost=2.0, duration_hours=1.0,
        dependencies=["run"],
    ))
    return dag


def _make_diamond_dag() -> StageDAG:
    """Create a diamond DAG: A -> B, A -> C, B -> D, C -> D."""
    dag = StageDAG()
    dag.add_stage(ExperimentStage(
        name="A", parameters=["x"], kpis=["kpi_a"], cost=1.0,
    ))
    dag.add_stage(ExperimentStage(
        name="B", parameters=["x", "y"], kpis=["kpi_b"], cost=2.0,
        dependencies=["A"],
    ))
    dag.add_stage(ExperimentStage(
        name="C", parameters=["x", "z"], kpis=["kpi_c"], cost=3.0,
        dependencies=["A"],
    ))
    dag.add_stage(ExperimentStage(
        name="D", parameters=["x", "y", "z"], kpis=["kpi_d"], cost=4.0,
        dependencies=["B", "C"],
    ))
    return dag


# ===========================================================================
# ExperimentStage tests
# ===========================================================================

class TestExperimentStage:
    """Tests for ExperimentStage dataclass."""

    def test_basic_creation(self) -> None:
        stage = ExperimentStage(
            name="mixing", parameters=["speed", "time"],
            kpis=["homogeneity"],
        )
        assert stage.name == "mixing"
        assert stage.parameters == ["speed", "time"]
        assert stage.kpis == ["homogeneity"]

    def test_default_values(self) -> None:
        stage = ExperimentStage(name="s1", parameters=[], kpis=[])
        assert stage.cost == 1.0
        assert stage.duration_hours == 1.0
        assert stage.dependencies == []
        assert stage.metadata == {}

    def test_custom_cost_and_duration(self) -> None:
        stage = ExperimentStage(
            name="expensive", parameters=["a"], kpis=["b"],
            cost=100.0, duration_hours=48.0,
        )
        assert stage.cost == 100.0
        assert stage.duration_hours == 48.0

    def test_dependencies(self) -> None:
        stage = ExperimentStage(
            name="post", parameters=["x"], kpis=["y"],
            dependencies=["pre1", "pre2"],
        )
        assert stage.dependencies == ["pre1", "pre2"]

    def test_metadata(self) -> None:
        stage = ExperimentStage(
            name="m", parameters=[], kpis=[],
            metadata={"lab": "chem", "priority": 1},
        )
        assert stage.metadata["lab"] == "chem"
        assert stage.metadata["priority"] == 1


# ===========================================================================
# StageDAG tests
# ===========================================================================

class TestStageDAG:
    """Tests for StageDAG."""

    def test_add_and_get_stage(self) -> None:
        dag = StageDAG()
        stage = ExperimentStage(name="s1", parameters=["x"], kpis=["y"])
        dag.add_stage(stage)
        assert dag.get_stage("s1") is stage

    def test_add_duplicate_raises(self) -> None:
        dag = StageDAG()
        dag.add_stage(ExperimentStage(name="s1", parameters=[], kpis=[]))
        with pytest.raises(ValueError, match="already exists"):
            dag.add_stage(ExperimentStage(name="s1", parameters=[], kpis=[]))

    def test_get_nonexistent_raises(self) -> None:
        dag = StageDAG()
        with pytest.raises(KeyError, match="not found"):
            dag.get_stage("nonexistent")

    def test_topological_sort_linear(self) -> None:
        dag = _make_linear_dag()
        order = dag.topological_order()
        assert order == ["prep", "run", "analyze"]

    def test_topological_sort_diamond(self) -> None:
        dag = _make_diamond_dag()
        order = dag.topological_order()
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_cycle_detection(self) -> None:
        dag = StageDAG()
        dag.add_stage(ExperimentStage(
            name="a", parameters=[], kpis=[], dependencies=["b"],
        ))
        dag.add_stage(ExperimentStage(
            name="b", parameters=[], kpis=[], dependencies=["a"],
        ))
        with pytest.raises(ValueError, match="cycle"):
            dag.topological_order()

    def test_validate_valid(self) -> None:
        dag = _make_linear_dag()
        assert dag.validate() is True

    def test_validate_missing_dependency(self) -> None:
        dag = StageDAG()
        dag.add_stage(ExperimentStage(
            name="orphan", parameters=[], kpis=[],
            dependencies=["nonexistent"],
        ))
        with pytest.raises(ValueError, match="not in the DAG"):
            dag.validate()

    def test_get_ready_stages_initial(self) -> None:
        dag = _make_linear_dag()
        ready = dag.get_ready_stages(set())
        assert ready == ["prep"]

    def test_get_ready_stages_after_completion(self) -> None:
        dag = _make_linear_dag()
        ready = dag.get_ready_stages({"prep"})
        assert "run" in ready
        assert "prep" not in ready

    def test_get_ready_diamond(self) -> None:
        dag = _make_diamond_dag()
        ready = dag.get_ready_stages({"A"})
        assert sorted(ready) == ["B", "C"]

    def test_get_ready_diamond_partial(self) -> None:
        dag = _make_diamond_dag()
        ready = dag.get_ready_stages({"A", "B"})
        assert "C" in ready
        assert "D" not in ready  # C not yet completed

    def test_get_ready_diamond_both(self) -> None:
        dag = _make_diamond_dag()
        ready = dag.get_ready_stages({"A", "B", "C"})
        assert "D" in ready

    def test_total_cost(self) -> None:
        dag = _make_linear_dag()
        assert dag.total_cost() == 8.0  # 1 + 5 + 2

    def test_stages_list(self) -> None:
        dag = _make_linear_dag()
        assert len(dag.stages()) == 3

    def test_len(self) -> None:
        dag = _make_linear_dag()
        assert len(dag) == 3

    def test_contains(self) -> None:
        dag = _make_linear_dag()
        assert "prep" in dag
        assert "nonexistent" not in dag

    def test_empty_dag(self) -> None:
        dag = StageDAG()
        assert dag.topological_order() == []
        assert dag.total_cost() == 0.0
        assert dag.stages() == []

    def test_single_stage(self) -> None:
        dag = StageDAG()
        dag.add_stage(ExperimentStage(name="only", parameters=["x"], kpis=["y"]))
        assert dag.topological_order() == ["only"]
        assert dag.get_ready_stages(set()) == ["only"]


# ===========================================================================
# ProxyModel tests
# ===========================================================================

class TestProxyModel:
    """Tests for ProxyModel GP proxy."""

    def test_fit_and_predict(self) -> None:
        model = ProxyModel()
        X = [[0.0], [0.5], [1.0]]
        y = [0.0, 0.25, 1.0]
        model.fit(X, y)
        means, variances = model.predict([[0.25], [0.75]])
        assert len(means) == 2
        assert len(variances) == 2
        # Predictions should be in reasonable range
        for m in means:
            assert -1.0 < m < 2.0

    def test_predict_single(self) -> None:
        model = ProxyModel()
        X = [[0.0], [1.0]]
        y = [0.0, 1.0]
        model.fit(X, y)
        mu, var = model.predict_single([0.5])
        assert isinstance(mu, float)
        assert isinstance(var, float)
        assert var >= 0.0

    def test_fit_recovers_training_points(self) -> None:
        """GP should interpolate (near-zero variance at training points)."""
        model = ProxyModel()
        X = [[0.0], [0.5], [1.0]]
        y = [1.0, 2.0, 3.0]
        model.fit(X, y, noise=1e-8)
        for xi, yi in zip(X, y):
            mu, var = model.predict_single(xi)
            assert abs(mu - yi) < 0.1, f"Expected ~{yi}, got {mu}"
            assert var < 0.1

    def test_unfitted_predict_raises(self) -> None:
        model = ProxyModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict([[0.0]])

    def test_unfitted_predict_single_raises(self) -> None:
        model = ProxyModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_single([0.0])

    def test_empty_fit(self) -> None:
        model = ProxyModel()
        model.fit([], [])
        assert not model.is_fitted

    def test_single_point_fit(self) -> None:
        model = ProxyModel()
        model.fit([[0.5]], [1.0])
        assert model.is_fitted
        mu, var = model.predict_single([0.5])
        assert abs(mu - 1.0) < 0.5

    def test_length_scale_auto(self) -> None:
        model = ProxyModel()
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [0.0, 1.0, 4.0, 9.0]
        model.fit(X, y)
        assert model.length_scale > 0

    def test_length_scale_fixed(self) -> None:
        model = ProxyModel(length_scale=0.5)
        X = [[0.0], [1.0]]
        y = [0.0, 1.0]
        model.fit(X, y)
        assert model.length_scale == 0.5

    def test_n_train(self) -> None:
        model = ProxyModel()
        assert model.n_train == 0
        model.fit([[0.0]], [1.0])
        assert model.n_train == 1

    def test_mismatched_X_y_raises(self) -> None:
        model = ProxyModel()
        with pytest.raises(ValueError, match="X has 2 rows but y has 1"):
            model.fit([[0.0], [1.0]], [0.0])

    def test_multidimensional_input(self) -> None:
        model = ProxyModel()
        X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        y = [0.0, 1.0, 1.0, 2.0]
        model.fit(X, y)
        mu, var = model.predict_single([0.5, 0.5])
        assert isinstance(mu, float)
        assert var >= 0.0


# ===========================================================================
# ContinueValue tests
# ===========================================================================

class TestContinueValue:
    """Tests for ContinueValue knowledge gradient."""

    def _make_fitted_models(self, dag: StageDAG) -> dict[str, ProxyModel]:
        """Create and fit proxy models with synthetic data."""
        models: dict[str, ProxyModel] = {}
        rng = random.Random(42)
        for stage in dag.stages():
            for kpi in stage.kpis:
                model = ProxyModel()
                n = 10
                X = [[rng.uniform(0, 1) for _ in stage.parameters] for _ in range(n)]
                y = [rng.gauss(0.5, 0.2) for _ in range(n)]
                model.fit(X, y)
                models[kpi] = model
        return models

    def test_compute_returns_float(self) -> None:
        dag = _make_linear_dag()
        models = self._make_fitted_models(dag)
        cv = ContinueValue(models, dag)
        value = cv.compute("prep", [0.5, 0.5], {"purity": 0.8})
        assert isinstance(value, float)

    def test_compute_last_stage_zero(self) -> None:
        dag = _make_linear_dag()
        models = self._make_fitted_models(dag)
        cv = ContinueValue(models, dag)
        value = cv.compute("analyze", [0.5], {"stability": 0.9})
        assert value == 0.0

    def test_should_continue_last_stage_false(self) -> None:
        dag = _make_linear_dag()
        models = self._make_fitted_models(dag)
        cv = ContinueValue(models, dag)
        assert cv.should_continue("analyze", [0.5], {"stability": 0.9}) is False

    def test_should_continue_returns_bool(self) -> None:
        dag = _make_linear_dag()
        models = self._make_fitted_models(dag)
        cv = ContinueValue(models, dag)
        result = cv.should_continue("prep", [0.5, 0.5], {"purity": 0.8})
        assert isinstance(result, bool)

    def test_expected_final_value(self) -> None:
        dag = _make_linear_dag()
        models = self._make_fitted_models(dag)
        cv = ContinueValue(models, dag)
        value = cv.expected_final_value([0.5, 0.5], {"purity": 0.8})
        assert isinstance(value, float)

    def test_expected_final_value_all_observed(self) -> None:
        dag = _make_linear_dag()
        models = self._make_fitted_models(dag)
        cv = ContinueValue(models, dag)
        value = cv.expected_final_value(
            [0.5, 0.5],
            {"purity": 0.8, "yield": 0.9, "stability": 0.7},
        )
        assert isinstance(value, float)
        # Average of all three
        expected = (0.8 + 0.9 + 0.7) / 3
        assert abs(value - expected) < 0.01

    def test_no_proxy_models(self) -> None:
        dag = _make_linear_dag()
        cv = ContinueValue({}, dag)
        value = cv.compute("prep", [0.5, 0.5], {})
        assert isinstance(value, float)

    def test_threshold_affects_continue(self) -> None:
        dag = _make_linear_dag()
        models = self._make_fitted_models(dag)
        cv = ContinueValue(models, dag)
        # With very high threshold, should not continue
        result = cv.should_continue(
            "prep", [0.5, 0.5], {"purity": 0.8}, threshold=1000.0
        )
        assert result is False

    def test_stage_information_value(self) -> None:
        dag = _make_linear_dag()
        models = self._make_fitted_models(dag)
        cv = ContinueValue(models, dag)
        iv = cv.stage_information_value("run", [0.5, 0.5, 0.5], {"purity": 0.8})
        assert isinstance(iv, float)

    def test_empty_observed_kpis(self) -> None:
        dag = _make_linear_dag()
        models = self._make_fitted_models(dag)
        cv = ContinueValue(models, dag)
        value = cv.compute("prep", [0.5, 0.5], {})
        assert isinstance(value, float)


# ===========================================================================
# MultiStageBayesianOptimizer tests
# ===========================================================================

class TestMultiStageBayesianOptimizer:
    """Tests for MultiStageBayesianOptimizer."""

    def test_creation(self) -> None:
        dag = _make_linear_dag()
        opt = MultiStageBayesianOptimizer(dag, seed=42)
        assert opt is not None

    def test_add_observation(self) -> None:
        dag = _make_linear_dag()
        opt = MultiStageBayesianOptimizer(dag)
        opt.add_observation("prep", {"temp": 0.5, "pressure": 0.3}, {"purity": 0.8})
        obs = opt.get_all_observations("prep")
        assert len(obs) == 1

    def test_add_multiple_observations(self) -> None:
        dag = _make_linear_dag()
        opt = MultiStageBayesianOptimizer(dag)
        for i in range(5):
            opt.add_observation(
                "prep",
                {"temp": i * 0.2, "pressure": 0.5},
                {"purity": 0.5 + i * 0.1},
            )
        obs = opt.get_all_observations("prep")
        assert len(obs) == 5

    def test_suggest_next_no_data(self) -> None:
        dag = _make_linear_dag()
        opt = MultiStageBayesianOptimizer(dag, seed=42)
        suggestions = opt.suggest_next("prep", n_suggestions=3)
        assert len(suggestions) == 3
        for s in suggestions:
            assert "temp" in s
            assert "pressure" in s

    def test_suggest_next_with_data(self) -> None:
        dag = _make_linear_dag()
        opt = MultiStageBayesianOptimizer(dag, seed=42)
        # Add some observations
        for i in range(5):
            opt.add_observation(
                "prep",
                {"temp": i * 0.2, "pressure": 0.5},
                {"purity": 0.5 + i * 0.1},
            )
        suggestions = opt.suggest_next("prep", n_suggestions=2)
        assert len(suggestions) == 2

    def test_suggest_values_in_range(self) -> None:
        dag = _make_linear_dag()
        opt = MultiStageBayesianOptimizer(dag, seed=42)
        suggestions = opt.suggest_next("prep", n_suggestions=5)
        for s in suggestions:
            for v in s.values():
                assert 0.0 <= v <= 1.0

    def test_should_continue_stage(self) -> None:
        dag = _make_linear_dag()
        opt = MultiStageBayesianOptimizer(dag, seed=42)
        # Add observations to build models
        rng = random.Random(42)
        for _ in range(10):
            opt.add_observation(
                "prep",
                {"temp": rng.uniform(0, 1), "pressure": rng.uniform(0, 1)},
                {"purity": rng.uniform(0, 1)},
            )
        result = opt.should_continue_stage(
            "prep", {"temp": 0.5, "pressure": 0.5}, {"purity": 0.8}
        )
        assert isinstance(result, bool)

    def test_should_continue_last_stage(self) -> None:
        dag = _make_linear_dag()
        opt = MultiStageBayesianOptimizer(dag)
        result = opt.should_continue_stage(
            "analyze", {"temp": 0.5}, {"stability": 0.8}
        )
        assert result is False

    def test_get_savings_report_initial(self) -> None:
        dag = _make_linear_dag()
        opt = MultiStageBayesianOptimizer(dag)
        report = opt.get_savings_report()
        assert report["total_cost_spent"] == 0.0
        assert report["total_cost_saved"] == 0.0
        assert report["total_evaluations"] == 0
        assert report["early_terminations"] == 0
        assert report["savings_ratio"] == 0.0

    def test_get_proxy_model(self) -> None:
        dag = _make_linear_dag()
        opt = MultiStageBayesianOptimizer(dag)
        model = opt.get_proxy_model("purity")
        assert isinstance(model, ProxyModel)
        assert not model.is_fitted

    def test_get_proxy_model_after_observation(self) -> None:
        dag = _make_linear_dag()
        opt = MultiStageBayesianOptimizer(dag)
        opt.add_observation("prep", {"temp": 0.5, "pressure": 0.3}, {"purity": 0.8})
        model = opt.get_proxy_model("purity")
        assert model.is_fitted

    def test_run_campaign(self) -> None:
        dag = _make_linear_dag()
        opt = MultiStageBayesianOptimizer(dag, seed=42)

        def evaluate_fn(stage_name: str, params: dict[str, float]) -> dict[str, float]:
            stage = dag.get_stage(stage_name)
            return {kpi: random.Random(42).uniform(0, 1) for kpi in stage.kpis}

        results = opt.run_campaign(evaluate_fn, n_iterations=3)
        assert len(results) > 0
        assert all(isinstance(r, StageResult) for r in results)

    def test_run_campaign_tracks_cost(self) -> None:
        dag = _make_linear_dag()
        opt = MultiStageBayesianOptimizer(dag, seed=42)

        call_count = [0]
        def evaluate_fn(stage_name: str, params: dict[str, float]) -> dict[str, float]:
            call_count[0] += 1
            stage = dag.get_stage(stage_name)
            return {kpi: 0.5 for kpi in stage.kpis}

        results = opt.run_campaign(evaluate_fn, n_iterations=2)
        report = opt.get_savings_report()
        assert report["total_cost_spent"] > 0
        assert report["total_evaluations"] == 2

    def test_run_campaign_result_fields(self) -> None:
        dag = StageDAG()
        dag.add_stage(ExperimentStage(
            name="only", parameters=["x"], kpis=["y"], cost=1.0,
        ))
        opt = MultiStageBayesianOptimizer(dag, seed=42)

        def evaluate_fn(stage_name: str, params: dict[str, float]) -> dict[str, float]:
            return {"y": params.get("x", 0.5)}

        results = opt.run_campaign(evaluate_fn, n_iterations=1)
        assert len(results) == 1
        r = results[0]
        assert r.stage_name == "only"
        assert "x" in r.parameters
        assert "y" in r.kpi_values
        assert r.cost == 1.0

    def test_observation_nonexistent_stage_raises(self) -> None:
        dag = _make_linear_dag()
        opt = MultiStageBayesianOptimizer(dag)
        with pytest.raises(KeyError):
            opt.add_observation("nonexistent", {"x": 0.5}, {"y": 0.5})

    def test_get_all_observations_empty(self) -> None:
        dag = _make_linear_dag()
        opt = MultiStageBayesianOptimizer(dag)
        obs = opt.get_all_observations("prep")
        assert obs == []


# ===========================================================================
# Visualization tests
# ===========================================================================

class TestVisualization:
    """Tests for SVG visualization functions."""

    def test_render_stage_flow_basic(self) -> None:
        dag = _make_linear_dag()
        svg = render_stage_flow(dag)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "prep" in svg
        assert "run" in svg
        assert "analyze" in svg

    def test_render_stage_flow_with_completed(self) -> None:
        dag = _make_linear_dag()
        svg = render_stage_flow(dag, completed={"prep", "run"})
        assert "<svg" in svg
        assert "prep" in svg

    def test_render_stage_flow_empty_dag(self) -> None:
        dag = StageDAG()
        svg = render_stage_flow(dag)
        assert "<svg" in svg
        assert "No stages" in svg

    def test_render_savings_report(self) -> None:
        report = {
            "total_cost_spent": 15.0,
            "total_cost_saved": 5.0,
            "total_evaluations": 10,
            "early_terminations": 3,
            "savings_ratio": 0.25,
            "per_stage_counts": {"prep": 10, "run": 7, "analyze": 4},
        }
        svg = render_savings_report(report)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "Cost Savings Report" in svg

    def test_render_savings_report_zeros(self) -> None:
        report = {
            "total_cost_spent": 0.0,
            "total_cost_saved": 0.0,
            "total_evaluations": 0,
            "early_terminations": 0,
            "savings_ratio": 0.0,
            "per_stage_counts": {},
        }
        svg = render_savings_report(report)
        assert "<svg" in svg


# ===========================================================================
# Reproducibility tests
# ===========================================================================

class TestReproducibility:
    """Tests for deterministic behavior with fixed seeds."""

    def test_suggest_reproducible(self) -> None:
        dag = _make_linear_dag()
        opt1 = MultiStageBayesianOptimizer(dag, seed=123)
        opt2 = MultiStageBayesianOptimizer(dag, seed=123)
        s1 = opt1.suggest_next("prep", n_suggestions=3)
        s2 = opt2.suggest_next("prep", n_suggestions=3)
        assert s1 == s2

    def test_different_seeds_differ(self) -> None:
        dag = _make_linear_dag()
        opt1 = MultiStageBayesianOptimizer(dag, seed=1)
        opt2 = MultiStageBayesianOptimizer(dag, seed=999)
        s1 = opt1.suggest_next("prep", n_suggestions=3)
        s2 = opt2.suggest_next("prep", n_suggestions=3)
        assert s1 != s2

    def test_campaign_reproducible(self) -> None:
        dag = _make_linear_dag()

        def run_campaign(seed: int) -> list[StageResult]:
            opt = MultiStageBayesianOptimizer(dag, seed=seed)
            rng = random.Random(seed)
            def evaluate_fn(stage_name: str, params: dict[str, float]) -> dict[str, float]:
                stage = dag.get_stage(stage_name)
                return {kpi: rng.uniform(0, 1) for kpi in stage.kpis}
            return opt.run_campaign(evaluate_fn, n_iterations=3)

        r1 = run_campaign(42)
        r2 = run_campaign(42)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.stage_name == b.stage_name
            assert a.parameters == b.parameters

    def test_proxy_model_reproducible(self) -> None:
        rng = random.Random(42)
        X = [[rng.uniform(0, 1)] for _ in range(10)]
        y = [rng.gauss(0, 1) for _ in range(10)]

        m1 = ProxyModel()
        m1.fit(X, y)
        mu1, var1 = m1.predict_single([0.5])

        m2 = ProxyModel()
        m2.fit(X, y)
        mu2, var2 = m2.predict_single([0.5])

        assert mu1 == mu2
        assert var1 == var2

    def test_proxy_model_y_mean(self) -> None:
        model = ProxyModel()
        model.fit([[0.0], [1.0]], [2.0, 4.0])
        assert model.y_mean() == 3.0

    def test_proxy_model_y_best(self) -> None:
        model = ProxyModel()
        model.fit([[0.0], [1.0]], [2.0, 4.0])
        assert model.y_best() == 2.0

    def test_proxy_model_y_mean_empty(self) -> None:
        model = ProxyModel()
        assert model.y_mean() == 0.0

    def test_proxy_model_y_best_empty_raises(self) -> None:
        model = ProxyModel()
        with pytest.raises(RuntimeError, match="No training data"):
            model.y_best()
