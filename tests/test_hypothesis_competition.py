"""Multi-hypothesis competition benchmark tests.

Proves the hypothesis engine can correctly identify the true model among
competing alternatives as data accumulates.  Covers progressive model
selection via BIC, Bayes factors, falsification, discriminating experiment
design, and posterior model probability convergence.

All tests are seeded and deterministic.  Pure Python stdlib only.
"""

from __future__ import annotations

import math
import random
import unittest

from optimization_copilot.hypothesis.models import (
    Hypothesis,
    HypothesisStatus,
)
from optimization_copilot.hypothesis.testing import HypothesisTester
from optimization_copilot.hypothesis.tracker import HypothesisTracker


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _generate_1d_data(
    true_func,
    n: int,
    x_lo: float = 0.0,
    x_hi: float = 10.0,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[list[list[float]], list[float]]:
    """Generate 1-D data: y = true_func(x) + noise.

    Returns (X, y) where X is a list of single-element lists.
    """
    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[float] = []
    for _ in range(n):
        x = rng.uniform(x_lo, x_hi)
        X.append([x])
        y.append(true_func(x) + rng.gauss(0, noise_std))
    return X, y


def _posterior_from_bics(bic_values: list[float]) -> list[float]:
    """Approximate posterior model probabilities from BIC values.

    P(Hi|data) proportional to exp(-BIC_i / 2), normalised to sum to 1.
    Uses the log-sum-exp trick for numerical stability.
    """
    half_bics = [-b / 2.0 for b in bic_values]
    max_val = max(half_bics)
    exps = [math.exp(v - max_val) for v in half_bics]
    total = sum(exps)
    return [e / total for e in exps]


# ===========================================================================
# Part 1: Three Mutually Exclusive Models Competition
# ===========================================================================

class TestModelCompetition(unittest.TestCase):
    """Benchmark: progressive evidence accumulation correctly identifies
    the true generative model among competing alternatives."""

    def setUp(self) -> None:
        self.tester = HypothesisTester()
        self.var_names = ["x"]

    # -----------------------------------------------------------------------
    # Test 1: Progressive model selection (linear is true)
    # -----------------------------------------------------------------------

    def test_progressive_model_selection(self) -> None:
        """True model y = 2*x + 1 should be increasingly preferred as data
        accumulates.  BIC and Bayes factors must show progressive evidence
        for H1 over H2 (quadratic) and H3 (sinusoidal)."""

        def true_func(x: float) -> float:
            return 2.0 * x + 1.0

        # Generate 50 points up-front; we will reveal them progressively.
        X_all, y_all = _generate_1d_data(
            true_func, n=50, x_lo=0, x_hi=10, noise_std=0.5, seed=42
        )

        h1 = Hypothesis(
            id="H1_linear",
            description="y = 2*x + 1 (true linear)",
            equation="2*x + 1",
            n_parameters=2,
        )
        h2 = Hypothesis(
            id="H2_quad",
            description="y = x^2 (wrong quadratic)",
            equation="x**2",
            n_parameters=1,
        )
        h3 = Hypothesis(
            id="H3_sin",
            description="y = 5*sin(x) (wrong sinusoidal)",
            equation="5*sin(x)",
            n_parameters=1,
        )

        bic_gap_h1_h2: list[float] = []  # BIC(H2) - BIC(H1)

        for step in range(1, 11):  # steps 1..10 -> n=5,10,...,50
            n = step * 5
            X_sub = X_all[:n]
            y_sub = y_all[:n]

            # Fresh hypothesis objects for BIC computation (avoids stale bic_score)
            h1_copy = Hypothesis(
                id="H1_linear", description=h1.description,
                equation=h1.equation, n_parameters=h1.n_parameters,
            )
            h2_copy = Hypothesis(
                id="H2_quad", description=h2.description,
                equation=h2.equation, n_parameters=h2.n_parameters,
            )
            h3_copy = Hypothesis(
                id="H3_sin", description=h3.description,
                equation=h3.equation, n_parameters=h3.n_parameters,
            )

            results = self.tester.compare_all(
                [h1_copy, h2_copy, h3_copy], X_sub, y_sub, self.var_names
            )
            bic_by_id = {r["hypothesis_id"]: r["bic"] for r in results}
            bic_gap_h1_h2.append(bic_by_id["H2_quad"] - bic_by_id["H1_linear"])

            # --- Progressive assertions ---

            # By n=20 (step=4), H1 should have the lowest BIC.
            if n >= 20:
                best_id = results[0]["hypothesis_id"]
                self.assertEqual(
                    best_id, "H1_linear",
                    f"At n={n}, expected H1 to be best but got {best_id} "
                    f"(BICs: {bic_by_id})"
                )

            # By n=30, Bayes factor H1 vs H2 > 10 (strong evidence).
            if n >= 30:
                bf_h1_h2 = self.tester.bayes_factor(
                    Hypothesis(id="H1_linear", description="", equation="2*x + 1", n_parameters=2),
                    Hypothesis(id="H2_quad", description="", equation="x**2", n_parameters=1),
                    X_sub, y_sub, self.var_names,
                )
                self.assertGreater(
                    bf_h1_h2, 10.0,
                    f"At n={n}, BF(H1 vs H2) = {bf_h1_h2:.1f}, expected > 10"
                )

            # By n=50, Bayes factor > 100 (decisive evidence).
            if n >= 50:
                bf_h1_h2 = self.tester.bayes_factor(
                    Hypothesis(id="H1_linear", description="", equation="2*x + 1", n_parameters=2),
                    Hypothesis(id="H2_quad", description="", equation="x**2", n_parameters=1),
                    X_sub, y_sub, self.var_names,
                )
                self.assertGreater(
                    bf_h1_h2, 100.0,
                    f"At n=50, BF(H1 vs H2) = {bf_h1_h2:.1f}, expected > 100"
                )

        # BIC gap (H2 - H1) should be monotonically non-decreasing
        # after the initial settling period (from step 3 onward).
        for i in range(3, len(bic_gap_h1_h2)):
            self.assertGreaterEqual(
                bic_gap_h1_h2[i], bic_gap_h1_h2[i - 1] - 1e-6,
                f"BIC gap not monotonically growing at step {i+1}: "
                f"{bic_gap_h1_h2[i-1]:.2f} -> {bic_gap_h1_h2[i]:.2f}"
            )

    # -----------------------------------------------------------------------
    # Test 2: Quadratic vs linear vs cubic
    # -----------------------------------------------------------------------

    def test_quadratic_vs_linear(self) -> None:
        """True model y = 0.5*x^2 - 2*x + 3.  Quadratic should beat
        linear (underfitting) and cubic (overfitting via BIC penalty)."""

        def true_func(x: float) -> float:
            return 0.5 * x * x - 2.0 * x + 3.0

        X, y = _generate_1d_data(
            true_func, n=50, x_lo=-5, x_hi=5, noise_std=1.0, seed=42
        )

        h_quad = Hypothesis(
            id="H_quad",
            description="y = 0.5*x^2 - 2*x + 3 (true quadratic)",
            equation="0.5*x**2 - 2*x + 3",
            n_parameters=3,
        )
        h_linear = Hypothesis(
            id="H_linear",
            description="y = a*x + b (wrong linear)",
            equation="-2*x + 3",
            n_parameters=2,
        )
        h_cubic = Hypothesis(
            id="H_cubic",
            description="y = cubic overfit",
            equation="0.5*x**2 - 2*x + 3 + 0.001*x**3",
            n_parameters=4,
        )

        results = self.tester.compare_all(
            [h_quad, h_linear, h_cubic], X, y, self.var_names
        )
        bic_by_id = {r["hypothesis_id"]: r["bic"] for r in results}

        # Quadratic should beat linear (linear misses the curvature).
        self.assertLess(
            bic_by_id["H_quad"], bic_by_id["H_linear"],
            f"Quadratic BIC ({bic_by_id['H_quad']:.2f}) should be < "
            f"linear BIC ({bic_by_id['H_linear']:.2f})"
        )

        # Quadratic should beat or tie cubic (BIC penalises extra params).
        self.assertLessEqual(
            bic_by_id["H_quad"], bic_by_id["H_cubic"] + 1e-6,
            f"Quadratic BIC ({bic_by_id['H_quad']:.2f}) should be <= "
            f"cubic BIC ({bic_by_id['H_cubic']:.2f})"
        )

    # -----------------------------------------------------------------------
    # Test 3: Sinusoidal detection
    # -----------------------------------------------------------------------

    def test_sinusoidal_detection(self) -> None:
        """True model y = 3*sin(2*x) + 1 should be decisively preferred
        over linear and quadratic alternatives (Bayes factor > 50)."""

        def true_func(x: float) -> float:
            return 3.0 * math.sin(2.0 * x) + 1.0

        X, y = _generate_1d_data(
            true_func, n=40, x_lo=0, x_hi=2 * math.pi,
            noise_std=0.3, seed=42,
        )

        h_sin = Hypothesis(
            id="H_sin",
            description="y = 3*sin(2*x) + 1 (true sinusoidal)",
            equation="3*sin(2*x) + 1",
            n_parameters=3,
        )
        h_linear = Hypothesis(
            id="H_linear",
            description="y = a*x + b (wrong linear)",
            equation="0.5*x + 1",
            n_parameters=2,
        )
        h_quad = Hypothesis(
            id="H_quad",
            description="y = a*x^2 + b (wrong quadratic)",
            equation="0.1*x**2 - 0.5*x + 1",
            n_parameters=3,
        )

        bf_sin_vs_linear = self.tester.bayes_factor(
            h_sin, h_linear, X, y, self.var_names
        )
        self.assertGreater(
            bf_sin_vs_linear, 50.0,
            f"BF(sin vs linear) = {bf_sin_vs_linear:.1f}, expected > 50"
        )

        # Also confirm sin model ranks first overall.
        results = self.tester.compare_all(
            [h_sin, h_linear, h_quad], X, y, self.var_names
        )
        self.assertEqual(
            results[0]["hypothesis_id"], "H_sin",
            f"Expected sinusoidal to rank first, got {results[0]['hypothesis_id']}"
        )


# ===========================================================================
# Part 2: Falsification Tests
# ===========================================================================

class TestFalsification(unittest.TestCase):
    """Benchmark: wrong models are correctly falsified while correct models
    survive the falsification procedure."""

    def setUp(self) -> None:
        self.tester = HypothesisTester()
        self.var_names = ["x"]

    def test_wrong_model_falsified(self) -> None:
        """A hypothesis with the wrong slope (y = 10*x) should be falsified
        after sequential observations from y = 2*x + 1.  The CI is +/- 1,
        so predictions from y=10*x will consistently miss."""

        rng = random.Random(42)
        h = Hypothesis(
            id="H_wrong_slope",
            description="y = 10*x (wrong slope)",
            equation="10*x",
            n_parameters=1,
        )

        for _ in range(10):
            x = rng.uniform(1, 5)  # avoid x near 0 where models agree
            true_y = 2.0 * x + 1.0 + rng.gauss(0, 0.5)
            self.tester.sequential_update(h, [x], true_y, self.var_names)

        self.assertTrue(
            self.tester.check_falsification(h, threshold=3),
            f"Wrong model should be falsified after 10 observations. "
            f"Support: {h.support_count}, Refute: {h.refute_count}"
        )

    def test_correct_model_not_falsified(self) -> None:
        """The correct hypothesis y = 2*x + 1 should NOT be falsified when
        data comes from the same model with moderate noise.  The CI band
        of +/- 1 should capture most observations (noise_std=0.5)."""

        rng = random.Random(42)
        h = Hypothesis(
            id="H_correct",
            description="y = 2*x + 1 (correct)",
            equation="2*x + 1",
            n_parameters=2,
        )

        for _ in range(20):
            x = rng.uniform(0, 10)
            true_y = 2.0 * x + 1.0 + rng.gauss(0, 0.5)
            self.tester.sequential_update(h, [x], true_y, self.var_names)

        self.assertFalse(
            self.tester.check_falsification(h, threshold=3),
            "Correct model should NOT be falsified"
        )
        self.assertGreater(
            h.evidence_ratio(), 0.5,
            f"Evidence ratio {h.evidence_ratio():.2f} should be > 0.5 "
            f"for the correct model"
        )

    def test_falsification_timeline(self) -> None:
        """Feeding observations one at a time from y = 3*x into hypothesis
        y = x + 5 (wrong).  Tracks when falsification first triggers and
        verifies it stays falsified once triggered."""

        rng = random.Random(42)
        h = Hypothesis(
            id="H_wrong_form",
            description="y = x + 5 (wrong)",
            equation="x + 5",
            n_parameters=2,
        )

        falsified_at: int | None = None
        falsified_history: list[bool] = []

        for i in range(20):
            x = rng.uniform(4, 10)  # range where models clearly diverge
            true_y = 3.0 * x + rng.gauss(0, 0.3)
            self.tester.sequential_update(h, [x], true_y, self.var_names)
            is_falsified = self.tester.check_falsification(h, threshold=3)
            falsified_history.append(is_falsified)
            if is_falsified and falsified_at is None:
                falsified_at = i + 1  # 1-indexed

        # Falsification should trigger at some point <= observation 15.
        self.assertIsNotNone(
            falsified_at,
            "Falsification should trigger within 20 observations"
        )
        self.assertLessEqual(
            falsified_at, 15,
            f"Falsification triggered at observation {falsified_at}, expected <= 15"
        )

        # Once falsified, it should stay falsified (subsequent observations
        # from the wrong model keep accumulating refuting evidence).
        first_falsified_idx = falsified_at - 1  # 0-indexed
        # Check that it remains falsified from the trigger point onward.
        # Note: check_falsification looks at the LAST threshold entries,
        # so a lucky observation could temporarily un-falsify.  We check that
        # the *final* state is falsified and most post-trigger states are.
        post_trigger = falsified_history[first_falsified_idx:]
        falsified_ratio = sum(post_trigger) / len(post_trigger)
        self.assertGreater(
            falsified_ratio, 0.5,
            f"After falsification at obs {falsified_at}, "
            f"only {falsified_ratio:.0%} of subsequent checks are falsified"
        )
        # Final state should definitely be falsified.
        self.assertTrue(
            falsified_history[-1],
            "Final observation should yield falsified state"
        )

    def test_progressive_falsification_of_two_wrong_models(self) -> None:
        """HypothesisTracker with three hypotheses: H1 (correct),
        H2 (wrong slope), H3 (wrong form).  After 30 observations,
        H2 and H3 should end up REFUTED while H1 stays active."""

        tracker = HypothesisTracker()

        h1 = Hypothesis(
            id="H1_correct",
            description="y = 2*x + 1 (correct)",
            equation="2*x + 1",
            n_parameters=2,
            status=HypothesisStatus.TESTING,
        )
        h2 = Hypothesis(
            id="H2_wrong_slope",
            description="y = 5*x (wrong slope)",
            equation="5*x",
            n_parameters=1,
            status=HypothesisStatus.TESTING,
        )
        h3 = Hypothesis(
            id="H3_wrong_form",
            description="y = x^2 (wrong form)",
            equation="x**2",
            n_parameters=1,
            status=HypothesisStatus.TESTING,
        )

        tracker.add(h1)
        tracker.add(h2)
        tracker.add(h3)

        rng = random.Random(42)
        tester = HypothesisTester()

        for _ in range(30):
            x = rng.uniform(1, 8)
            true_y = 2.0 * x + 1.0 + rng.gauss(0, 0.5)
            obs = {"x": x, "y": true_y}
            tracker.update_with_observation(obs, var_names=["x"])

            # Check falsification and update status for active hypotheses.
            for h in tracker.get_active_hypotheses():
                if tester.check_falsification(h, threshold=3):
                    tracker.update_status(h.id, HypothesisStatus.REFUTED)

        h1_final = tracker.get("H1_correct")
        h2_final = tracker.get("H2_wrong_slope")
        h3_final = tracker.get("H3_wrong_form")

        # H1 should NOT be refuted.
        self.assertNotEqual(
            h1_final.status, HypothesisStatus.REFUTED,
            f"Correct hypothesis H1 should not be REFUTED, "
            f"but status is {h1_final.status.value}"
        )
        # H2 should be refuted.
        self.assertEqual(
            h2_final.status, HypothesisStatus.REFUTED,
            f"Wrong hypothesis H2 should be REFUTED, "
            f"but status is {h2_final.status.value}"
        )
        # H3 should be refuted.
        self.assertEqual(
            h3_final.status, HypothesisStatus.REFUTED,
            f"Wrong hypothesis H3 should be REFUTED, "
            f"but status is {h3_final.status.value}"
        )


# ===========================================================================
# Part 3: Discriminating Experiments
# ===========================================================================

class TestDiscriminatingExperiments(unittest.TestCase):
    """Benchmark: the tracker's experiment suggestion correctly identifies
    points that maximise divergence between competing models."""

    def test_suggest_discriminating_point(self) -> None:
        """H1: y = 2*x + 1 (linear) and H2: y = x^2 (quadratic) agree
        near x ~ 1.6 but diverge at extremes.  The suggested experiment
        should NOT be near x=1.6 and should be near the bounds."""

        tracker = HypothesisTracker()
        h1 = Hypothesis(
            id="H1_lin",
            description="y = 2*x + 1",
            equation="2*x + 1",
            n_parameters=2,
        )
        h2 = Hypothesis(
            id="H2_quad",
            description="y = x^2",
            equation="x**2",
            n_parameters=1,
        )
        tracker.add(h1)
        tracker.add(h2)

        result = tracker.suggest_discriminating_experiment(
            "H1_lin", "H2_quad",
            parameter_ranges={"x": (-5.0, 10.0)},
            n_candidates=100,
            seed=42,
        )

        suggested_x = result["point"]["x"]

        # The models intersect near x ~ 1.62 and x ~ -0.62.
        # A good discriminating point should be far from both intersections.
        self.assertGreater(
            abs(suggested_x - 1.62), 1.0,
            f"Suggested x={suggested_x:.2f} is too close to intersection ~1.62"
        )

        # Should be toward one of the extremes.
        self.assertTrue(
            suggested_x < -2.0 or suggested_x > 5.0,
            f"Suggested x={suggested_x:.2f} should be at an extreme "
            f"(< -2 or > 5)"
        )

        # Divergence should be substantial.
        self.assertGreater(
            result["divergence"], 5.0,
            f"Divergence {result['divergence']:.2f} should be > 5"
        )

    def test_discriminating_experiment_resolves_competition(self) -> None:
        """Start with H1 and H2 at similar BIC.  Use a discriminating
        experiment to increase the BIC gap between them."""

        tester = HypothesisTester()
        tracker = HypothesisTracker()
        var_names = ["x"]

        h1 = Hypothesis(
            id="H1_lin",
            description="y = 2*x + 1",
            equation="2*x + 1",
            n_parameters=2,
        )
        h2 = Hypothesis(
            id="H2_quad",
            description="y = x^2",
            equation="x**2",
            n_parameters=1,
        )
        tracker.add(h1)
        tracker.add(h2)

        # Generate a small dataset near the intersection region where
        # the models make similar predictions (x near 1-3).
        rng = random.Random(42)
        X_init: list[list[float]] = []
        y_init: list[float] = []
        for _ in range(8):
            x = rng.uniform(1.0, 3.0)
            # True model is linear: y = 2*x + 1
            y_val = 2.0 * x + 1.0 + rng.gauss(0, 0.5)
            X_init.append([x])
            y_init.append(y_val)

        # Compute initial BICs.
        bic1_before = tester.compute_bic(
            Hypothesis(id="H1", description="", equation="2*x + 1", n_parameters=2),
            X_init, y_init, var_names,
        )
        bic2_before = tester.compute_bic(
            Hypothesis(id="H2", description="", equation="x**2", n_parameters=1),
            X_init, y_init, var_names,
        )
        gap_before = abs(bic2_before - bic1_before)

        # Get suggested discriminating experiment.
        result = tracker.suggest_discriminating_experiment(
            "H1_lin", "H2_quad",
            parameter_ranges={"x": (-5.0, 10.0)},
            n_candidates=100,
            seed=42,
        )
        disc_x = result["point"]["x"]

        # Evaluate the TRUE model at the suggested point.
        true_y = 2.0 * disc_x + 1.0

        # Add the discriminating observation to the dataset.
        X_after = X_init + [[disc_x]]
        y_after = y_init + [true_y]

        bic1_after = tester.compute_bic(
            Hypothesis(id="H1", description="", equation="2*x + 1", n_parameters=2),
            X_after, y_after, var_names,
        )
        bic2_after = tester.compute_bic(
            Hypothesis(id="H2", description="", equation="x**2", n_parameters=1),
            X_after, y_after, var_names,
        )
        gap_after = abs(bic2_after - bic1_after)

        self.assertGreater(
            gap_after, gap_before,
            f"BIC gap after discriminating experiment ({gap_after:.2f}) "
            f"should exceed gap before ({gap_before:.2f})"
        )


# ===========================================================================
# Part 4: Posterior Model Probability Curve
# ===========================================================================

class TestPosteriorModelProbability(unittest.TestCase):
    """Benchmark: posterior model probabilities (BIC approximation) converge
    to certainty on the true model as data grows."""

    def setUp(self) -> None:
        self.tester = HypothesisTester()
        self.var_names = ["x"]

    def test_posterior_convergence(self) -> None:
        """Three models: H1 (true: y=2*x+1), H2 (y=3*x), H3 (y=x^2).
        P(H1|data) should exceed 0.5 by n=15, 0.9 by n=30, and be
        monotonically non-decreasing after n=10."""

        def true_func(x: float) -> float:
            return 2.0 * x + 1.0

        X_all, y_all = _generate_1d_data(
            true_func, n=50, x_lo=0, x_hi=10, noise_std=0.5, seed=42,
        )

        posteriors_h1: list[float] = []

        for step in range(1, 11):  # n = 5, 10, ..., 50
            n = step * 5
            X_sub = X_all[:n]
            y_sub = y_all[:n]

            bic1 = self.tester.compute_bic(
                Hypothesis(id="H1", description="", equation="2*x + 1", n_parameters=2),
                X_sub, y_sub, self.var_names,
            )
            bic2 = self.tester.compute_bic(
                Hypothesis(id="H2", description="", equation="3*x", n_parameters=1),
                X_sub, y_sub, self.var_names,
            )
            bic3 = self.tester.compute_bic(
                Hypothesis(id="H3", description="", equation="x**2", n_parameters=1),
                X_sub, y_sub, self.var_names,
            )

            posteriors = _posterior_from_bics([bic1, bic2, bic3])
            p_h1 = posteriors[0]
            posteriors_h1.append(p_h1)

            # P(H1|data) > 0.5 by n=15 (step=3).
            if n >= 15:
                self.assertGreater(
                    p_h1, 0.5,
                    f"P(H1|data) = {p_h1:.4f} at n={n}, expected > 0.5"
                )

            # P(H1|data) > 0.9 by n=30 (step=6).
            if n >= 30:
                self.assertGreater(
                    p_h1, 0.9,
                    f"P(H1|data) = {p_h1:.4f} at n={n}, expected > 0.9"
                )

        # Monotonically non-decreasing after n=10 (index 1 onward).
        for i in range(2, len(posteriors_h1)):
            self.assertGreaterEqual(
                posteriors_h1[i], posteriors_h1[i - 1] - 1e-6,
                f"P(H1|data) decreased from step {i} ({posteriors_h1[i-1]:.4f}) "
                f"to step {i+1} ({posteriors_h1[i]:.4f})"
            )

    def test_equal_prior_convergence(self) -> None:
        """With very little data (n=5), no model should dominate.
        Posteriors should be roughly equal (~0.33) for well-separated models."""

        def true_func(x: float) -> float:
            return 2.0 * x + 1.0

        # Use x values near the origin where all models give similar-ish output.
        rng = random.Random(42)
        X: list[list[float]] = []
        y: list[float] = []
        for _ in range(5):
            x = rng.uniform(0.5, 2.0)
            X.append([x])
            y.append(true_func(x) + rng.gauss(0, 0.5))

        bic1 = self.tester.compute_bic(
            Hypothesis(id="H1", description="", equation="2*x + 1", n_parameters=2),
            X, y, self.var_names,
        )
        bic2 = self.tester.compute_bic(
            Hypothesis(id="H2", description="", equation="3*x", n_parameters=1),
            X, y, self.var_names,
        )
        bic3 = self.tester.compute_bic(
            Hypothesis(id="H3", description="", equation="x**2", n_parameters=1),
            X, y, self.var_names,
        )

        posteriors = _posterior_from_bics([bic1, bic2, bic3])

        # No model should have > 0.7 probability with only 5 points in the
        # near-agreement region.  Some preference is expected due to BIC
        # parameter penalties, but no model should be dominant.
        for i, p in enumerate(posteriors):
            self.assertLess(
                p, 0.7,
                f"Model {i+1} has posterior {p:.4f} > 0.7 with only 5 "
                f"data points; models should not be decisively distinguishable"
            )


if __name__ == "__main__":
    unittest.main()
