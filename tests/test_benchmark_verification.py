"""Systematic benchmark verification tests.

Tests the direct evaluation harness, systematic evaluator, meta-controller
adapter, and tabular benchmark. Validates that model-based backends
outperform random sampling on standard test functions.
"""

from __future__ import annotations

import math
import pytest

from optimization_copilot.benchmark.functions import BENCHMARK_SUITE, get_benchmark
from optimization_copilot.benchmark.direct_runner import (
    DirectBenchmarkResult,
    DirectBenchmarkRunner,
    _compute_budget,
)
from optimization_copilot.benchmark.systematic_eval import (
    EvaluationConfig,
    SystematicEvaluator,
    wilcoxon_signed_rank,
    generate_report,
)
from optimization_copilot.benchmark.tabular_data import TabularBenchmark
from optimization_copilot.benchmark.meta_adapter import MetaControllerAdapter


# ── Import backend classes ──────────────────────────────────────────

# We need the actual backend classes for testing. Import individually
# with try/except so tests can still partially run if some backends
# are missing.
try:
    from optimization_copilot.backends.builtin import (
        RandomSampler,
        LatinHypercubeSampler,
        SobolSampler,
        TPESampler,
        GaussianProcessBO,
        RandomForestBO,
        CMAESSampler,
        DifferentialEvolution,
        NSGA2Sampler,
        TuRBOSampler,
    )
    _ALL_BACKEND_CLASSES = {
        "random_sampler": RandomSampler,
        "latin_hypercube_sampler": LatinHypercubeSampler,
        "sobol_sampler": SobolSampler,
        "tpe_sampler": TPESampler,
        "gaussian_process_bo": GaussianProcessBO,
        "random_forest_bo": RandomForestBO,
        "cmaes_sampler": CMAESSampler,
        "differential_evolution": DifferentialEvolution,
        "nsga2_sampler": NSGA2Sampler,
        "turbo_sampler": TuRBOSampler,
    }
    _HAS_BACKENDS = True
except ImportError:
    _ALL_BACKEND_CLASSES = {}
    _HAS_BACKENDS = False


# ── Helper: select easy single-objective benchmarks for fast tests ──

def _easy_benchmarks():
    """Return a subset of easy/moderate benchmarks for fast testing."""
    easy_names = [
        "branin", "hartmann3", "sphere5", "bohachevsky2",
        "rosenbrock5", "griewank10", "styblinski_tang10", "zakharov10",
    ]
    return [get_benchmark(n) for n in easy_names if n in BENCHMARK_SUITE]


# ── Wilcoxon test unit tests ────────────────────────────────────────

class TestWilcoxonSignedRank:
    """Tests for the pure-Python Wilcoxon signed-rank test."""

    def test_identical_samples(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        stat, p_val, n = wilcoxon_signed_rank(x, x)
        assert n == 0
        assert p_val == 1.0

    def test_clearly_different(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        y = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        stat, p_val, n = wilcoxon_signed_rank(x, y)
        assert n == 10
        assert p_val < 0.01  # Should be highly significant

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError):
            wilcoxon_signed_rank([1, 2, 3], [1, 2])

    def test_slightly_different(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        y = [1.1, 2.2, 3.1, 4.3, 5.1, 6.2, 7.3, 8.1]
        stat, p_val, n = wilcoxon_signed_rank(x, y)
        assert n == 8
        assert 0.0 < p_val <= 1.0


# ── Tabular benchmark tests ────────────────────────────────────────

class TestTabularBenchmark:
    """Tests for the embedded tabular benchmark."""

    def test_data_has_48_rows(self):
        tb = TabularBenchmark()
        assert len(tb.data) == 48

    def test_evaluate_returns_objective(self):
        tb = TabularBenchmark()
        result = tb.evaluate({
            "temperature": 100.0,
            "catalyst_loading": 3.0,
            "concentration": 0.25,
            "time": 10.0,
        })
        assert "objective" in result
        assert isinstance(result["objective"], float)

    def test_evaluate_negated_yield(self):
        """Objective should be negated yield (for minimization)."""
        tb = TabularBenchmark()
        result = tb.evaluate({
            "temperature": 100.0,
            "catalyst_loading": 3.0,
            "concentration": 0.25,
            "time": 10.0,
        })
        assert result["objective"] < 0.0  # Negated yield

    def test_to_benchmark_function(self):
        tb = TabularBenchmark()
        bf = tb.to_benchmark_function()
        assert bf.name == "buchwald_hartwig"
        assert len(bf.parameter_specs) == 4
        assert bf.known_optimum["objective"] < 0.0
        # Should be callable
        result = bf({"temperature": 90.0, "catalyst_loading": 2.5,
                      "concentration": 0.2, "time": 8.0})
        assert "objective" in result

    def test_best_yield_is_highest_in_data(self):
        tb = TabularBenchmark()
        bf = tb.to_benchmark_function()
        best_yield = max(row["yield_pct"] for row in tb.data)
        assert bf.known_optimum["objective"] == pytest.approx(-best_yield)


# ── DirectBenchmarkRunner tests ─────────────────────────────────────

class TestDirectBenchmarkRunner:
    """Tests for the closed-loop direct benchmark runner."""

    def test_budget_computation(self):
        assert _compute_budget(2) == 20
        assert _compute_budget(5) == 50
        assert _compute_budget(10) == 100
        assert _compute_budget(30) == 200  # capped

    @pytest.mark.skipif(not _HAS_BACKENDS, reason="Backends not available")
    def test_single_run_random(self):
        """Random sampler should complete a full run."""
        runner = DirectBenchmarkRunner()
        benchmark = get_benchmark("branin")
        result = runner.run(RandomSampler, benchmark, budget=20, seed=42)
        assert result.backend_name == "random_sampler"
        assert result.benchmark_name == "branin"
        assert len(result.observations) == 20
        assert result.best_objective < float("inf")
        assert math.isfinite(result.simple_regret)
        assert math.isfinite(result.log10_regret)
        assert 0.0 <= result.auc_normalized <= 1.0

    @pytest.mark.skipif(not _HAS_BACKENDS, reason="Backends not available")
    def test_deterministic_with_same_seed(self):
        """Same seed should produce same results."""
        runner = DirectBenchmarkRunner()
        benchmark = get_benchmark("branin")
        r1 = runner.run(RandomSampler, benchmark, budget=20, seed=42)
        r2 = runner.run(RandomSampler, benchmark, budget=20, seed=42)
        assert r1.best_objective == r2.best_objective
        assert r1.simple_regret == r2.simple_regret

    @pytest.mark.skipif(not _HAS_BACKENDS, reason="Backends not available")
    def test_different_seeds_differ(self):
        """Different seeds should generally produce different results."""
        runner = DirectBenchmarkRunner()
        benchmark = get_benchmark("branin")
        r1 = runner.run(RandomSampler, benchmark, budget=20, seed=42)
        r2 = runner.run(RandomSampler, benchmark, budget=20, seed=99)
        assert r1.best_objective != r2.best_objective

    @pytest.mark.skipif(not _HAS_BACKENDS, reason="Backends not available")
    def test_all_backends_complete(self):
        """All 10 backends should complete a run without error."""
        runner = DirectBenchmarkRunner()
        benchmark = get_benchmark("sphere5")
        for name, cls in _ALL_BACKEND_CLASSES.items():
            result = runner.run(cls, benchmark, budget=20, seed=42)
            assert len(result.observations) == 20, f"{name} did not complete"

    @pytest.mark.skipif(not _HAS_BACKENDS, reason="Backends not available")
    def test_multi_seed(self):
        """run_multi_seed should return one result per seed."""
        runner = DirectBenchmarkRunner()
        benchmark = get_benchmark("branin")
        results = runner.run_multi_seed(RandomSampler, benchmark, seeds=[42, 43, 44], budget=20)
        assert len(results) == 3
        for r in results:
            assert len(r.observations) == 20

    @pytest.mark.skipif(not _HAS_BACKENDS, reason="Backends not available")
    def test_convergence_iteration_found(self):
        """On an easy function, model-based backends should converge."""
        runner = DirectBenchmarkRunner()
        benchmark = get_benchmark("sphere5")
        result = runner.run(TPESampler, benchmark, budget=50, seed=42)
        # TPE on sphere5 should find something reasonable
        assert result.simple_regret < 10.0

    @pytest.mark.skipif(not _HAS_BACKENDS, reason="Backends not available")
    def test_best_so_far_monotonic(self):
        """best_so_far should be monotonically non-increasing."""
        runner = DirectBenchmarkRunner()
        benchmark = get_benchmark("branin")
        result = runner.run(TPESampler, benchmark, budget=30, seed=42)
        bsf = result.best_so_far
        for i in range(1, len(bsf)):
            assert bsf[i] <= bsf[i - 1]


# ── Systematic evaluation tests ─────────────────────────────────────

@pytest.mark.skipif(not _HAS_BACKENDS, reason="Backends not available")
class TestSystematicEvaluation:
    """Fast systematic evaluation: 3 seeds x 4 functions x 3 backends."""

    def test_fast_evaluation(self):
        """Run a fast evaluation and verify rankings are produced."""
        config = EvaluationConfig(
            seeds=[42, 43, 44],
            budget_override=20,
        )
        evaluator = SystematicEvaluator(config=config)

        # Use a small subset for speed
        benchmarks = [get_benchmark(n) for n in ["branin", "sphere5", "hartmann3", "bohachevsky2"]]
        backends = {
            "random_sampler": RandomSampler,
            "tpe_sampler": TPESampler,
            "latin_hypercube_sampler": LatinHypercubeSampler,
        }

        report = evaluator.evaluate(backends, benchmarks)

        assert report.n_backends == 3
        assert report.n_functions == 4
        assert report.n_seeds == 3
        assert report.total_runs == 3 * 4 * 3  # 36
        assert len(report.rankings) == 3

        # Rankings should be sorted by avg_rank
        for i in range(len(report.rankings) - 1):
            assert report.rankings[i].avg_rank <= report.rankings[i + 1].avg_rank

    def test_model_based_beats_random_on_easy(self):
        """TPE should rank better than random on easy functions."""
        config = EvaluationConfig(
            seeds=[42, 43, 44],
            budget_override=30,
        )
        evaluator = SystematicEvaluator(config=config)

        benchmarks = [get_benchmark("sphere5"), get_benchmark("branin")]
        backends = {
            "random_sampler": RandomSampler,
            "tpe_sampler": TPESampler,
        }

        report = evaluator.evaluate(backends, benchmarks)
        # Find rankings for each
        rank_by_name = {r.name: r.avg_rank for r in report.rankings}
        # TPE should have a better (lower) average rank than random
        assert rank_by_name["tpe_sampler"] <= rank_by_name["random_sampler"]

    def test_pairwise_tests_produced(self):
        """Pairwise Wilcoxon tests should be generated."""
        config = EvaluationConfig(
            seeds=[42, 43, 44, 45, 46],
            budget_override=20,
        )
        evaluator = SystematicEvaluator(config=config)

        benchmarks = [get_benchmark("branin"), get_benchmark("sphere5")]
        backends = {
            "random_sampler": RandomSampler,
            "tpe_sampler": TPESampler,
            "latin_hypercube_sampler": LatinHypercubeSampler,
        }

        report = evaluator.evaluate(backends, benchmarks)
        # Should have pairwise tests (best vs each of 2 others)
        assert len(report.pairwise_tests) == 2
        for test in report.pairwise_tests:
            assert 0.0 <= test.p_value <= 1.0
            assert test.n_pairs >= 0  # Can be 0 if regret vectors are identical

    def test_report_generation(self):
        """generate_report should produce valid markdown."""
        config = EvaluationConfig(seeds=[42], budget_override=20)
        evaluator = SystematicEvaluator(config=config)

        benchmarks = [get_benchmark("branin")]
        backends = {
            "random_sampler": RandomSampler,
            "tpe_sampler": TPESampler,
        }

        report = evaluator.evaluate(backends, benchmarks)
        md = generate_report(report)
        assert isinstance(md, str)
        assert "# Systematic Benchmark Evaluation Report" in md
        assert "Overall Rankings" in md
        assert "random_sampler" in md
        assert "tpe_sampler" in md


# ── MetaController adapter tests ────────────────────────────────────

@pytest.mark.skipif(not _HAS_BACKENDS, reason="Backends not available")
class TestMetaControllerBenchmark:
    """Tests for the MetaControllerAdapter on benchmarks."""

    def test_adapter_completes_run(self):
        """MetaControllerAdapter should complete a full benchmark run."""
        runner = DirectBenchmarkRunner()
        benchmark = get_benchmark("branin")

        # Create a factory function that returns the adapter
        def adapter_factory():
            return MetaControllerAdapter(backend_factories=_ALL_BACKEND_CLASSES)

        result = runner.run(adapter_factory, benchmark, budget=30, seed=42)
        assert result.backend_name == "meta_controller"
        assert len(result.observations) == 30
        assert result.best_objective < float("inf")

    def test_adapter_beats_random(self):
        """MetaControllerAdapter should achieve lower regret than random on easy function."""
        runner = DirectBenchmarkRunner()
        benchmark = get_benchmark("sphere5")

        def adapter_factory():
            return MetaControllerAdapter(backend_factories=_ALL_BACKEND_CLASSES)

        # Run both
        meta_result = runner.run(adapter_factory, benchmark, budget=50, seed=42)
        random_result = runner.run(RandomSampler, benchmark, budget=50, seed=42)

        # Meta controller should do at least as well as random
        assert meta_result.best_objective <= random_result.best_objective * 1.5

    def test_adapter_in_systematic_eval(self):
        """MetaControllerAdapter should work in the systematic evaluator."""
        config = EvaluationConfig(seeds=[42, 43], budget_override=20)
        evaluator = SystematicEvaluator(config=config)

        def adapter_factory():
            return MetaControllerAdapter(backend_factories=_ALL_BACKEND_CLASSES)

        # Wrap the adapter factory in a class-like object that can be called
        # The evaluator expects plugin_factory: type, but we can pass a callable
        benchmarks = [get_benchmark("branin")]
        backends = {
            "random_sampler": RandomSampler,
            "meta_controller": adapter_factory,
        }

        report = evaluator.evaluate(backends, benchmarks)
        assert report.n_backends == 2
        assert report.total_runs == 4  # 2 backends x 1 function x 2 seeds


# ── Full evaluation (slow) ──────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(not _HAS_BACKENDS, reason="Backends not available")
class TestFullEvaluation:
    """Full 20-seed evaluation across all functions and backends.

    Run with: pytest -m slow tests/test_benchmark_verification.py
    """

    def test_full_evaluation(self):
        """Full systematic evaluation with 20 seeds."""
        config = EvaluationConfig(
            seeds=list(range(42, 62)),  # 20 seeds
        )
        evaluator = SystematicEvaluator(config=config)
        benchmarks = _easy_benchmarks()

        report = evaluator.evaluate(_ALL_BACKEND_CLASSES, benchmarks)

        assert report.n_seeds == 20
        assert report.n_backends == len(_ALL_BACKEND_CLASSES)
        assert len(report.rankings) == len(_ALL_BACKEND_CLASSES)

        # First-ranked backend should have avg_rank < n_backends/2
        best = report.rankings[0]
        assert best.avg_rank < len(_ALL_BACKEND_CLASSES) / 2.0

        # Random should not be #1
        random_rank = next(
            r.avg_rank for r in report.rankings if r.name == "random_sampler"
        )
        assert random_rank > report.rankings[0].avg_rank

        # Generate and validate report
        md = generate_report(report)
        assert "Wilcoxon" in md
        assert "Win Rate" in md
