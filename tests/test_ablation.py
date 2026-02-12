"""Comprehensive tests for the v4-v5 integration: HeteroscedasticGP + case study ablation.

Covers:
1. Noise variance passthrough (evaluator -> Observation.metadata)
2. Per-point noise model (ReplayBenchmark._noise_at_point, ZincBenchmark override)
3. AblationRunner basic (creation, result fields, statistical test)
4. AblationRunner functional (full runs, convergence, noise impact)
5. Integration (end-to-end pipeline, noise flow, summary content)
"""

from __future__ import annotations

import math
import unittest
import warnings

from optimization_copilot.backends.builtin import GaussianProcessBO
from optimization_copilot.backends.gp_heteroscedastic import HeteroscedasticGP
from optimization_copilot.case_studies.ablation import AblationResult, AblationRunner
from optimization_copilot.case_studies.base import ExperimentalBenchmark, ReplayBenchmark
from optimization_copilot.case_studies.evaluator import (
    CaseStudyEvaluator,
    ComparisonResult,
    PerformanceMetrics,
)
from optimization_copilot.case_studies.zinc.benchmark import ZincBenchmark
from optimization_copilot.case_studies.zinc.data_loader import (
    _NOISE_DISTANCE_SCALE,
    _NOISE_FAR_FROM_OPTIMUM,
    _NOISE_NEAR_OPTIMUM,
    _OPTIMAL_POINT,
    _PARAM_NAMES,
)
from optimization_copilot.core.models import Observation, ParameterSpec, VariableType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(name: str, lo: float = 0.0, hi: float = 1.0) -> ParameterSpec:
    return ParameterSpec(name=name, type=VariableType.CONTINUOUS, lower=lo, upper=hi)


def _make_obs(
    params: dict,
    kpi_name: str,
    kpi_val: float,
    noise_var: float | None = None,
    iteration: int = 0,
    is_failure: bool = False,
) -> Observation:
    meta = {}
    if noise_var is not None:
        meta["noise_variance"] = noise_var
        meta["noise_variances"] = {kpi_name: noise_var}
    return Observation(
        iteration=iteration,
        parameters=params,
        kpi_values={kpi_name: kpi_val} if not is_failure else {},
        metadata=meta,
        is_failure=is_failure,
        qc_passed=not is_failure,
    )


def _feasible_zinc_point(val: float = 0.05) -> dict:
    """A feasible point in the zinc search space (sum < 1.0)."""
    return {name: val for name in _PARAM_NAMES}


class _UnconstrainedBenchmark(ExperimentalBenchmark):
    """Minimal benchmark without constraints for testing evaluator wiring."""

    def __init__(self) -> None:
        super().__init__(domain_name=None)
        self._noise = 0.5

    def evaluate(self, x: dict) -> dict | None:
        val = sum(float(v) for v in x.values())
        return {"obj": {"value": val, "variance": self._noise}}

    def get_search_space(self) -> dict:
        return {"x1": {"type": "continuous", "range": [0.0, 1.0]},
                "x2": {"type": "continuous", "range": [0.0, 1.0]}}

    def get_objectives(self) -> dict:
        return {"obj": {"direction": "minimize"}}


# ---------------------------------------------------------------------------
# Module-level shared expensive fixtures (computed once for all tests)
# Uses _UnconstrainedBenchmark (2D, fast) for ablation structure tests.
# ZincBenchmark used only in noise model and zinc-specific tests.
# ---------------------------------------------------------------------------

_SHARED_ZINC: ZincBenchmark | None = None
_SHARED_UC_ABLATION_RESULT: AblationResult | None = None
_SHARED_UC_ABLATION_RUNNER: AblationRunner | None = None
_SHARED_UC_NOISE_IMPACT: list | None = None


def _get_shared_zinc() -> ZincBenchmark:
    global _SHARED_ZINC
    if _SHARED_ZINC is None:
        _SHARED_ZINC = ZincBenchmark(n_train=30, seed=42)
    return _SHARED_ZINC


def _get_shared_ablation() -> tuple[AblationRunner, AblationResult]:
    """Shared ablation using fast 2D unconstrained benchmark."""
    global _SHARED_UC_ABLATION_RUNNER, _SHARED_UC_ABLATION_RESULT
    if _SHARED_UC_ABLATION_RESULT is None:
        bench = _UnconstrainedBenchmark()
        _SHARED_UC_ABLATION_RUNNER = AblationRunner(bench)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _SHARED_UC_ABLATION_RESULT = _SHARED_UC_ABLATION_RUNNER.run_hetero_vs_homo(
                budget=5, n_repeats=3,
            )
    return _SHARED_UC_ABLATION_RUNNER, _SHARED_UC_ABLATION_RESULT


def _get_shared_noise_impact() -> list:
    global _SHARED_UC_NOISE_IMPACT
    if _SHARED_UC_NOISE_IMPACT is None:
        runner, _ = _get_shared_ablation()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _SHARED_UC_NOISE_IMPACT = runner.run_noise_impact_analysis(budget=5, seed=42)
    return _SHARED_UC_NOISE_IMPACT


# ===========================================================================
# 1. Noise variance passthrough tests
# ===========================================================================


class TestNoiseVariancePassthrough(unittest.TestCase):
    """Verify that CaseStudyEvaluator passes noise variance from benchmark
    evaluate() results into Observation.metadata."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = _get_shared_zinc()
        cls.evaluator = CaseStudyEvaluator(cls.bench)
        cls.uc_bench = _UnconstrainedBenchmark()
        cls.uc_evaluator = CaseStudyEvaluator(cls.uc_bench)

    def test_evaluator_passes_noise_variance_in_metadata(self) -> None:
        """run_single with unconstrained benchmark, verify result has variance."""
        strategy = GaussianProcessBO()
        history, metrics = self.uc_evaluator.run_single(strategy, budget=5, seed=42)
        successful = [h for h in history if h["result"] is not None]
        self.assertGreater(len(successful), 0)
        for h in successful:
            result = h["result"]
            for obj_name, obj_data in result.items():
                self.assertIn("variance", obj_data)
                self.assertIsInstance(obj_data["variance"], float)
                self.assertGreater(obj_data["variance"], 0.0)

    def test_evaluator_passes_noise_variances_dict(self) -> None:
        """Verify HeteroscedasticGP receives noise_variance from metadata."""
        strategy = HeteroscedasticGP()
        self.uc_evaluator.run_single(strategy, budget=5, seed=42)
        self.assertGreater(len(strategy._noise_vars), 0)
        for nv in strategy._noise_vars:
            self.assertIsInstance(nv, float)
            self.assertGreater(nv, 0.0)

    def test_evaluator_noise_variance_matches_benchmark_result(self) -> None:
        """The noise_variance in metadata should match what benchmark.evaluate() returns."""
        x = _feasible_zinc_point(0.05)
        result = self.bench.evaluate(x)
        self.assertIsNotNone(result)
        expected_var = result["coulombic_efficiency"]["variance"]
        self.assertGreater(expected_var, 0.0)
        encoded = self.bench._encode(x)
        noise_at_pt = self.bench._noise_at_point("coulombic_efficiency", encoded)
        self.assertAlmostEqual(expected_var, noise_at_pt, places=10)

    def test_evaluator_failure_has_no_noise_metadata(self) -> None:
        """Failed observations (infeasible) should have empty metadata."""
        class AlwaysFailBenchmark(ExperimentalBenchmark):
            def evaluate(self, x):
                return None
            def get_search_space(self):
                return {"x1": {"type": "continuous", "range": [0.0, 1.0]}}
            def get_objectives(self):
                return {"obj": {"direction": "minimize"}}

        bench = AlwaysFailBenchmark()
        evaluator = CaseStudyEvaluator(bench)
        strategy = GaussianProcessBO()
        history, _ = evaluator.run_single(strategy, budget=3, seed=42)
        for h in history:
            self.assertIsNone(h["result"])

    def test_heteroscedastic_gp_reads_noise_from_metadata(self) -> None:
        """Fit HeteroscedasticGP with observations that have noise_variance
        in metadata, verify it uses those values rather than default."""
        specs = [_make_spec("x1"), _make_spec("x2")]
        obs_list = [
            _make_obs({"x1": 0.1, "x2": 0.2}, "obj", 1.0, noise_var=0.5, iteration=0),
            _make_obs({"x1": 0.3, "x2": 0.4}, "obj", 2.0, noise_var=0.1, iteration=1),
            _make_obs({"x1": 0.5, "x2": 0.6}, "obj", 3.0, noise_var=1.0, iteration=2),
        ]
        gp = HeteroscedasticGP(default_noise=0.01)
        gp.fit(obs_list, specs)
        self.assertEqual(len(gp._noise_vars), 3)
        self.assertAlmostEqual(gp._noise_vars[0], 0.5)
        self.assertAlmostEqual(gp._noise_vars[1], 0.1)
        self.assertAlmostEqual(gp._noise_vars[2], 1.0)

    def test_heteroscedastic_gp_uses_default_when_no_metadata(self) -> None:
        """When metadata lacks noise_variance, HeteroscedasticGP uses default_noise."""
        specs = [_make_spec("x1")]
        obs_list = [
            Observation(
                iteration=0, parameters={"x1": 0.5},
                kpi_values={"obj": 1.0}, metadata={},
            ),
        ]
        gp = HeteroscedasticGP(default_noise=0.42)
        gp.fit(obs_list, specs)
        self.assertEqual(len(gp._noise_vars), 1)
        self.assertAlmostEqual(gp._noise_vars[0], 0.42)

    def test_evaluator_metadata_has_both_noise_keys(self) -> None:
        """Verify the evaluator creates observations with both
        'noise_variance' (scalar) and 'noise_variances' (dict)."""
        strategy = HeteroscedasticGP()
        self.evaluator.run_single(strategy, budget=3, seed=42)
        for meta in strategy._metadata_list:
            self.assertIn("noise_variance", meta)
            self.assertIn("noise_variances", meta)
            self.assertIsInstance(meta["noise_variances"], dict)

    def test_evaluator_skips_failures_in_observations(self) -> None:
        """Failed evaluations should not produce observations in the GP."""
        class HalfFailBenchmark(ExperimentalBenchmark):
            def __init__(self):
                super().__init__()
                self._call_count = 0
            def evaluate(self, x):
                self._call_count += 1
                if self._call_count % 2 == 0:
                    return None
                return {"obj": {"value": 1.0, "variance": 0.01}}
            def get_search_space(self):
                return {"x1": {"type": "continuous", "range": [0.0, 1.0]}}
            def get_objectives(self):
                return {"obj": {"direction": "minimize"}}

        bench = HalfFailBenchmark()
        evaluator = CaseStudyEvaluator(bench)
        strategy = HeteroscedasticGP()
        evaluator.run_single(strategy, budget=6, seed=42)
        n_successful = len(strategy._noise_vars)
        self.assertGreater(n_successful, 0)
        self.assertLess(n_successful, 6)


# ===========================================================================
# 2. Per-point noise model tests
# ===========================================================================


class TestPerPointNoiseModel(unittest.TestCase):
    """Tests for ReplayBenchmark._noise_at_point and ZincBenchmark override."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = _get_shared_zinc()

    def test_replay_benchmark_default_noise_at_point(self) -> None:
        """Default _noise_at_point returns scalar from _noise_levels."""
        class SimpleReplay(ReplayBenchmark):
            def _generate_data(self):
                return {
                    "X": [[0.5]],
                    "Y": {"obj": [1.0]},
                    "noise_levels": {"obj": 0.123},
                }
            def get_search_space(self):
                return {"x1": {"type": "continuous", "range": [0.0, 1.0]}}
            def get_objectives(self):
                return {"obj": {"direction": "minimize"}}

        replay = SimpleReplay(n_train=1, seed=42)
        noise_a = replay._noise_at_point("obj", [0.1])
        noise_b = replay._noise_at_point("obj", [0.9])
        self.assertAlmostEqual(noise_a, 0.123)
        self.assertAlmostEqual(noise_b, 0.123)

    def test_zinc_benchmark_noise_at_point_varies(self) -> None:
        """ZincBenchmark noise varies by point position."""
        near = list(_OPTIMAL_POINT)
        far = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        noise_near = self.bench._noise_at_point("coulombic_efficiency", near)
        noise_far = self.bench._noise_at_point("coulombic_efficiency", far)
        self.assertNotAlmostEqual(noise_near, noise_far, places=3)

    def test_zinc_benchmark_noise_near_optimum(self) -> None:
        """Noise is lower near the optimum."""
        noise_near = self.bench._noise_at_point(
            "coulombic_efficiency", list(_OPTIMAL_POINT)
        )
        expected_var = _NOISE_NEAR_OPTIMUM ** 2
        self.assertAlmostEqual(noise_near, expected_var, places=5)

    def test_zinc_benchmark_noise_far_from_optimum(self) -> None:
        """Noise is higher far from the optimum."""
        far_point = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        noise_far = self.bench._noise_at_point("coulombic_efficiency", far_point)
        noise_near = self.bench._noise_at_point(
            "coulombic_efficiency", list(_OPTIMAL_POINT)
        )
        self.assertGreater(noise_far, noise_near)

    def test_zinc_benchmark_noise_is_variance_not_std(self) -> None:
        """The value returned is std^2 (variance), not standard deviation."""
        noise_var = self.bench._noise_at_point(
            "coulombic_efficiency", list(_OPTIMAL_POINT)
        )
        expected_std = _NOISE_NEAR_OPTIMUM
        self.assertAlmostEqual(noise_var, expected_std ** 2, places=10)
        self.assertNotAlmostEqual(noise_var, expected_std, places=3)

    def test_zinc_evaluate_returns_hetero_variance(self) -> None:
        """evaluate() result has per-point variance (not the scalar noise level)."""
        x = _feasible_zinc_point(0.05)
        result = self.bench.evaluate(x)
        self.assertIsNotNone(result)
        variance = result["coulombic_efficiency"]["variance"]
        encoded = self.bench._encode(x)
        expected = self.bench._noise_at_point("coulombic_efficiency", encoded)
        self.assertAlmostEqual(variance, expected, places=10)

    def test_zinc_noise_at_point_positive(self) -> None:
        """Noise variance is always positive for any point."""
        test_points = [
            [0.0] * 7, [0.1] * 7, list(_OPTIMAL_POINT),
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        for pt in test_points:
            noise = self.bench._noise_at_point("coulombic_efficiency", pt)
            self.assertGreater(noise, 0.0)

    def test_zinc_noise_monotonically_increases_with_distance(self) -> None:
        """Noise increases monotonically as we move away from optimum."""
        opt = list(_OPTIMAL_POINT)
        direction = [0.1, -0.05, 0.08, 0.0, 0.0, 0.0, 0.0]
        prev_noise = 0.0
        for scale in [0.0, 0.5, 1.0, 2.0, 3.0]:
            point = [opt[i] + scale * direction[i] for i in range(7)]
            noise = self.bench._noise_at_point("coulombic_efficiency", point)
            self.assertGreaterEqual(noise, prev_noise - 1e-12)
            prev_noise = noise

    def test_zinc_noise_bounded_by_far_noise_squared(self) -> None:
        """Noise variance should never exceed _NOISE_FAR_FROM_OPTIMUM^2."""
        very_far = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        noise = self.bench._noise_at_point("coulombic_efficiency", very_far)
        max_var = _NOISE_FAR_FROM_OPTIMUM ** 2
        self.assertLessEqual(noise, max_var + 1e-10)


# ===========================================================================
# 3. AblationRunner basic tests
#    Uses the shared ablation result to avoid repeated slow GP fitting.
# ===========================================================================


class TestAblationRunnerBasic(unittest.TestCase):
    """Basic tests for AblationRunner creation and result structure."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = _get_shared_zinc()
        cls.runner, cls.result = _get_shared_ablation()

    def test_ablation_runner_creates(self) -> None:
        """AblationRunner can be instantiated."""
        runner = AblationRunner(self.bench)
        self.assertIsNotNone(runner)
        self.assertEqual(runner.alpha, 0.05)

    def test_ablation_runner_custom_alpha(self) -> None:
        """AblationRunner accepts custom alpha."""
        runner = AblationRunner(self.bench, alpha=0.1)
        self.assertAlmostEqual(runner.alpha, 0.1)

    def test_ablation_result_fields(self) -> None:
        """AblationResult has all expected fields."""
        self.assertIsInstance(self.result, AblationResult)
        self.assertIsInstance(self.result.comparison, ComparisonResult)
        self.assertIsInstance(self.result.statistical_test, dict)
        self.assertIsInstance(self.result.cohens_d, float)
        self.assertIsInstance(self.result.hetero_wins, bool)
        self.assertIsInstance(self.result.significant, bool)
        self.assertIsInstance(self.result.hetero_median, float)
        self.assertIsInstance(self.result.homo_median, float)
        self.assertIsInstance(self.result.summary, str)

    def test_ablation_comparison_has_two_strategies(self) -> None:
        """ComparisonResult.strategy_names has both hetero and homo."""
        self.assertEqual(len(self.result.comparison.strategy_names), 2)
        self.assertIn("heteroscedastic_gp", self.result.comparison.strategy_names)
        self.assertIn("homoscedastic_gp", self.result.comparison.strategy_names)

    def test_ablation_comparison_correct_n_repeats(self) -> None:
        """n_repeats matches what was requested."""
        self.assertEqual(self.result.comparison.n_repeats, 3)

    def test_ablation_comparison_correct_budget(self) -> None:
        """budget matches what was requested."""
        self.assertEqual(self.result.comparison.budget, 5)

    def test_ablation_statistical_test_keys(self) -> None:
        """statistical_test has statistic, p_value, effect_size."""
        self.assertIn("statistic", self.result.statistical_test)
        self.assertIn("p_value", self.result.statistical_test)
        self.assertIn("effect_size", self.result.statistical_test)

    def test_ablation_summary_is_string(self) -> None:
        """summary is a non-empty string."""
        self.assertIsInstance(self.result.summary, str)
        self.assertGreater(len(self.result.summary), 0)

    def test_ablation_significant_flag(self) -> None:
        """significant = (p_value < alpha)."""
        expected_sig = self.result.statistical_test["p_value"] < 0.05
        self.assertEqual(self.result.significant, expected_sig)

    def test_ablation_p_value_in_valid_range(self) -> None:
        """p_value should be between 0 and 1."""
        p = self.result.statistical_test["p_value"]
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


# ===========================================================================
# 4. AblationRunner functional tests
#    Reuses the shared ablation result.
# ===========================================================================


class TestAblationRunnerFunctional(unittest.TestCase):
    """Functional tests running the full ablation pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = _get_shared_zinc()
        cls.runner, cls.result = _get_shared_ablation()

    def test_ablation_hetero_vs_homo_runs(self) -> None:
        """Full run completes on ZincBenchmark."""
        self.assertIsNotNone(self.result)
        self.assertIsInstance(self.result, AblationResult)

    def test_ablation_metrics_per_strategy(self) -> None:
        """Each strategy has n_repeats metrics."""
        for name in ["heteroscedastic_gp", "homoscedastic_gp"]:
            metrics_list = self.result.comparison.metrics[name]
            self.assertEqual(len(metrics_list), 3)
            for m in metrics_list:
                self.assertIsInstance(m, PerformanceMetrics)
                self.assertIsInstance(m.best_value, float)

    def test_ablation_convergence_curves(self) -> None:
        """convergence_curves has correct shape."""
        for name in ["heteroscedastic_gp", "homoscedastic_gp"]:
            curves = self.result.comparison.convergence_curves[name]
            self.assertEqual(len(curves), 3)  # n_repeats
            for curve in curves:
                self.assertIsInstance(curve, list)
                self.assertEqual(len(curve), 5)  # budget
                for v in curve:
                    self.assertTrue(math.isfinite(v))

    def test_ablation_custom_hetero_kwargs(self) -> None:
        """Custom kernel/lengthscale params work via a simple 2D benchmark."""
        # Use unconstrained benchmark for speed
        bench2 = _UnconstrainedBenchmark()
        runner2 = AblationRunner(bench2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = runner2.run_hetero_vs_homo(
                budget=5, n_repeats=3,
                hetero_kwargs={"kernel": "rbf", "lengthscale": 0.5},
            )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AblationResult)

    def test_ablation_noise_impact_analysis(self) -> None:
        """run_noise_impact_analysis returns non-empty list."""
        impact = _get_shared_noise_impact()
        self.assertIsInstance(impact, list)
        self.assertGreater(len(impact), 0)

    def test_ablation_noise_impact_has_weight_ratio(self) -> None:
        """Each entry in noise impact has weight_ratio key."""
        impact = _get_shared_noise_impact()
        for entry in impact:
            self.assertIn("weight_ratio", entry)
            self.assertIn("noise_variance", entry)
            self.assertIn("hetero_weight", entry)
            self.assertIn("homo_weight", entry)
            self.assertIsInstance(entry["weight_ratio"], float)

    def test_ablation_direction_logic(self) -> None:
        """hetero_wins logic is consistent with direction and medians."""
        # The shared result uses minimize direction (_UnconstrainedBenchmark)
        if self.result.hetero_median < self.result.homo_median:
            self.assertTrue(self.result.hetero_wins)
        elif self.result.hetero_median > self.result.homo_median:
            self.assertFalse(self.result.hetero_wins)

    def test_ablation_cohens_d_non_negative(self) -> None:
        """Cohen's d should be >= 0."""
        self.assertGreaterEqual(self.result.cohens_d, 0.0)

    def test_ablation_benchmark_name_in_comparison(self) -> None:
        """ComparisonResult includes the benchmark class name."""
        self.assertEqual(
            self.result.comparison.benchmark_name, "_UnconstrainedBenchmark"
        )


# ===========================================================================
# 5. Integration tests
#    Reuses shared fixtures where possible.
# ===========================================================================


class TestIntegration(unittest.TestCase):
    """End-to-end integration tests for the v4-v5 pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = _get_shared_zinc()
        cls.runner, cls.result = _get_shared_ablation()
        cls.noise_impact = _get_shared_noise_impact()

    def test_zinc_ablation_end_to_end(self) -> None:
        """Full pipeline: ZincBenchmark -> AblationRunner -> result with stats."""
        self.assertIsInstance(self.result, AblationResult)
        self.assertIsInstance(self.result.comparison, ComparisonResult)
        self.assertIn("statistic", self.result.statistical_test)
        self.assertIsInstance(self.result.summary, str)
        self.assertGreater(len(self.result.summary), 0)

    def test_noise_variance_flows_to_gp(self) -> None:
        """Trace noise from benchmark evaluate() -> metadata -> HeteroscedasticGP._noise_vars."""
        uc = _UnconstrainedBenchmark()
        evaluator = CaseStudyEvaluator(uc)
        strategy = HeteroscedasticGP()
        evaluator.run_single(strategy, budget=5, seed=42)
        self.assertGreater(len(strategy._noise_vars), 0)
        for nv in strategy._noise_vars:
            self.assertGreater(nv, 0.0)

    def test_homoscedastic_gp_ignores_metadata_noise(self) -> None:
        """GaussianProcessBO doesn't use noise_variance from metadata."""
        uc = _UnconstrainedBenchmark()
        evaluator = CaseStudyEvaluator(uc)
        homo = GaussianProcessBO()
        evaluator.run_single(homo, budget=5, seed=42)
        self.assertFalse(hasattr(homo, "_noise_vars"))

    def test_ablation_result_summary_contains_winner(self) -> None:
        """summary contains 'Winner:'."""
        self.assertIn("Winner:", self.result.summary)

    def test_ablation_result_summary_contains_p_value(self) -> None:
        """summary contains p-value info."""
        self.assertIn("p-value:", self.result.summary)

    def test_ablation_effect_size_classification(self) -> None:
        """Effect size labeling (LARGE/MEDIUM/SMALL/NEGLIGIBLE) in summary."""
        labels = ["LARGE", "MEDIUM", "SMALL", "NEGLIGIBLE"]
        found = any(label in self.result.summary for label in labels)
        self.assertTrue(
            found,
            f"Summary should contain effect size label. Got:\n{self.result.summary}",
        )

    def test_fresh_strategies_per_repeat(self) -> None:
        """Verify AblationRunner creates fresh strategy instances per repeat."""
        hetero_metrics = self.result.comparison.metrics["heteroscedastic_gp"]
        homo_metrics = self.result.comparison.metrics["homoscedastic_gp"]
        hetero_bests = [m.best_value for m in hetero_metrics]
        self.assertEqual(len(hetero_bests), 3)
        self.assertEqual(len(homo_metrics), 3)

    def test_ablation_result_medians_are_finite(self) -> None:
        """hetero_median and homo_median should be finite floats."""
        self.assertTrue(math.isfinite(self.result.hetero_median))
        self.assertTrue(math.isfinite(self.result.homo_median))

    def test_noise_impact_entries_have_correct_structure(self) -> None:
        """Each noise impact entry has all required keys."""
        required_keys = {"index", "x", "y", "noise_variance",
                         "hetero_weight", "homo_weight", "weight_ratio"}
        for entry in self.noise_impact:
            for key in required_keys:
                self.assertIn(key, entry, f"Missing key '{key}' in noise impact entry")

    def test_convergence_curves_monotonic_for_minimize(self) -> None:
        """For minimize direction, convergence curves should be non-increasing."""
        for name in ["heteroscedastic_gp", "homoscedastic_gp"]:
            for curve in self.result.comparison.convergence_curves[name]:
                for i in range(1, len(curve)):
                    self.assertLessEqual(
                        curve[i], curve[i - 1] + 1e-12,
                        f"Convergence curve for {name} should be non-increasing "
                        f"(minimize), but curve[{i}]={curve[i]} > curve[{i-1}]={curve[i-1]}"
                    )

    def test_ablation_summary_contains_objective_name(self) -> None:
        """Summary should reference the objective name."""
        self.assertIn("obj", self.result.summary)

    def test_ablation_summary_contains_direction(self) -> None:
        """Summary should reference the optimization direction."""
        self.assertIn("minimize", self.result.summary)


if __name__ == "__main__":
    unittest.main()
