"""Tests for the Hypothesis Engine (Layer 3).

Covers hypothesis models, generation from multiple sources, BIC-based
testing, falsification, sequential updates, tracking, and serialization.
"""

from __future__ import annotations

import math
import random
import unittest

from optimization_copilot.hypothesis.generator import HypothesisGenerator
from optimization_copilot.hypothesis.models import (
    Evidence,
    Hypothesis,
    HypothesisStatus,
    Prediction,
)
from optimization_copilot.hypothesis.testing import HypothesisTester
from optimization_copilot.hypothesis.tracker import HypothesisTracker


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_linear_data(
    n: int = 100, seed: int = 42
) -> tuple[list[list[float]], list[float], list[str]]:
    """y = 2*x0 + 3*x1 + noise."""
    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[float] = []
    for _ in range(n):
        x0 = rng.gauss(0, 1)
        x1 = rng.gauss(0, 1)
        X.append([x0, x1])
        y.append(2.0 * x0 + 3.0 * x1 + rng.gauss(0, 0.1))
    return X, y, ["x0", "x1"]


# ---------------------------------------------------------------------------
# TestHypothesisModels
# ---------------------------------------------------------------------------

class TestHypothesisModels(unittest.TestCase):
    """Test Hypothesis, Prediction, and Evidence data models."""

    def test_hypothesis_creation(self) -> None:
        h = Hypothesis(id="H0001", description="test hypothesis")
        self.assertEqual(h.id, "H0001")
        self.assertEqual(h.status, HypothesisStatus.PROPOSED)
        self.assertEqual(h.evidence_ratio(), 0.5)

    def test_evidence_ratio(self) -> None:
        h = Hypothesis(id="H0001", description="test")
        # Add supporting evidence
        pred = Prediction(
            hypothesis_id="H0001",
            variable="y",
            predicted_value=5.0,
            confidence_interval=(4.0, 6.0),
        )
        ev_support = Evidence(prediction=pred, observed_value=5.2)
        h.add_evidence(ev_support)
        self.assertEqual(h.support_count, 1)
        self.assertEqual(h.refute_count, 0)
        self.assertEqual(h.evidence_ratio(), 1.0)

        # Add refuting evidence
        ev_refute = Evidence(prediction=pred, observed_value=10.0)
        h.add_evidence(ev_refute)
        self.assertEqual(h.support_count, 1)
        self.assertEqual(h.refute_count, 1)
        self.assertAlmostEqual(h.evidence_ratio(), 0.5)

    def test_evidence_within_ci(self) -> None:
        pred = Prediction(
            hypothesis_id="H0001",
            variable="y",
            predicted_value=5.0,
            confidence_interval=(4.0, 6.0),
        )
        ev_in = Evidence(prediction=pred, observed_value=5.5)
        self.assertTrue(ev_in.within_ci)
        self.assertAlmostEqual(ev_in.residual, 0.5)

        ev_out = Evidence(prediction=pred, observed_value=7.0)
        self.assertFalse(ev_out.within_ci)
        self.assertAlmostEqual(ev_out.residual, 2.0)

    def test_prediction_creation(self) -> None:
        pred = Prediction(
            hypothesis_id="H0001",
            variable="y",
            predicted_value=3.14,
            confidence_interval=(2.0, 4.0),
            condition={"x0": 1.0},
        )
        self.assertEqual(pred.variable, "y")
        self.assertEqual(pred.confidence_interval, (2.0, 4.0))

    def test_serialization_roundtrip(self) -> None:
        pred = Prediction(
            hypothesis_id="H0001",
            variable="y",
            predicted_value=5.0,
            confidence_interval=(4.0, 6.0),
        )
        ev = Evidence(prediction=pred, observed_value=5.2)

        h = Hypothesis(
            id="H0001",
            description="roundtrip test",
            equation="2 * x0 + 3 * x1",
            source="symreg",
            n_parameters=3,
        )
        h.add_evidence(ev)

        d = h.to_dict()
        h2 = Hypothesis.from_dict(d)

        self.assertEqual(h2.id, h.id)
        self.assertEqual(h2.description, h.description)
        self.assertEqual(h2.equation, h.equation)
        self.assertEqual(h2.source, h.source)
        self.assertEqual(h2.support_count, h.support_count)
        self.assertEqual(h2.n_parameters, h.n_parameters)
        self.assertEqual(len(h2.evidence), 1)


# ---------------------------------------------------------------------------
# TestHypothesisGenerator
# ---------------------------------------------------------------------------

class TestHypothesisGenerator(unittest.TestCase):
    """Test hypothesis generation from various analysis sources."""

    def setUp(self) -> None:
        self.gen = HypothesisGenerator(seed=42)

    def test_from_symreg(self) -> None:
        pareto = [
            {"equation": "x0", "complexity": 1, "r_squared": 0.6},
            {"equation": "2*x0 + 3*x1", "complexity": 3, "r_squared": 0.98},
        ]
        hyps = self.gen.from_symreg(pareto)
        self.assertEqual(len(hyps), 2)
        self.assertEqual(hyps[0].source, "symreg")
        self.assertEqual(hyps[0].equation, "x0")
        self.assertEqual(hyps[0].n_parameters, 1)
        self.assertEqual(hyps[1].equation, "2*x0 + 3*x1")
        self.assertEqual(hyps[1].n_parameters, 3)
        self.assertIn("R²=0.9800", hyps[1].description)

    def test_from_fanova(self) -> None:
        main = {"x0": 0.5, "x1": 0.3, "x2": 0.05}
        interactions = [
            {"vars": ["x0", "x1"], "importance": 0.15},
        ]
        hyps = self.gen.from_fanova(main, interactions, threshold=0.1)
        # x0 (0.5) and x1 (0.3) pass threshold; x2 (0.05) does not
        # Plus 1 interaction
        self.assertEqual(len(hyps), 3)
        sources = {h.source for h in hyps}
        self.assertEqual(sources, {"fanova"})
        # Check that x2 is not included
        descs = " ".join(h.description for h in hyps)
        self.assertNotIn("x2", descs)

    def test_from_correlation(self) -> None:
        corrs = {"x0": 0.85, "x1": -0.6, "x2": 0.1}
        hyps = self.gen.from_correlation(corrs, threshold=0.3)
        self.assertEqual(len(hyps), 2)
        # x0 should be first (highest abs corr)
        self.assertIn("positively", hyps[0].description)
        self.assertIn("negatively", hyps[1].description)
        self.assertEqual(hyps[0].source, "correlation")

    def test_from_causal_graph(self) -> None:
        graph_dict = {
            "nodes": {"X": {}, "M": {}, "Y": {}},
            "edges": [
                {"source": "X", "target": "M"},
                {"source": "M", "target": "Y"},
                {"source": "X", "target": "Y"},
            ],
        }
        hyps = self.gen.from_causal_graph(graph_dict, target="Y")
        # Expected paths: M -> Y, X -> M -> Y, X -> Y
        self.assertGreaterEqual(len(hyps), 2)
        mechanisms = [h.causal_mechanism for h in hyps]
        self.assertIn("M -> Y", mechanisms)
        self.assertIn("X -> Y", mechanisms)
        self.assertIn("X -> M -> Y", mechanisms)

    def test_generate_competing(self) -> None:
        random.seed(42)
        rng = random.Random(42)
        data: list[list[float]] = []
        for _ in range(100):
            x0 = rng.gauss(0, 1)
            x1 = rng.gauss(0, 1)
            y = 2 * x0 + 3 * x1 + rng.gauss(0, 0.1)
            data.append([x0, x1, y])

        hyps = self.gen.generate_competing(
            data, var_names=["x0", "x1", "y"], target_index=-1
        )
        self.assertGreater(len(hyps), 0)
        sources = {h.source for h in hyps}
        # Should generate both correlation and fanova-based hypotheses
        self.assertTrue(sources.intersection({"correlation", "fanova"}))


# ---------------------------------------------------------------------------
# TestHypothesisTester
# ---------------------------------------------------------------------------

class TestHypothesisTester(unittest.TestCase):
    """Test hypothesis scoring, comparison, and falsification."""

    def setUp(self) -> None:
        self.tester = HypothesisTester()
        self.X, self.y, self.var_names = make_linear_data(n=100, seed=42)

    def test_compute_bic_true_vs_wrong(self) -> None:
        """True model y=2*x0+3*x1 should have lower BIC than y=x0."""
        h_true = Hypothesis(
            id="H_true",
            description="true model",
            equation="2*x0 + 3*x1",
            n_parameters=2,
        )
        h_wrong = Hypothesis(
            id="H_wrong",
            description="wrong model",
            equation="x0",
            n_parameters=1,
        )
        bic_true = self.tester.compute_bic(
            h_true, self.X, self.y, self.var_names
        )
        bic_wrong = self.tester.compute_bic(
            h_wrong, self.X, self.y, self.var_names
        )
        self.assertLess(bic_true, bic_wrong)

    def test_bayes_factor(self) -> None:
        """Bayes factor should favour the true model."""
        h_true = Hypothesis(
            id="H_true",
            description="true",
            equation="2*x0 + 3*x1",
            n_parameters=2,
        )
        h_wrong = Hypothesis(
            id="H_wrong",
            description="wrong",
            equation="x0",
            n_parameters=1,
        )
        bf = self.tester.bayes_factor(
            h_true, h_wrong, self.X, self.y, self.var_names
        )
        self.assertGreater(bf, 1.0)

    def test_sequential_update(self) -> None:
        h = Hypothesis(
            id="H_seq",
            description="sequential test",
            equation="2*x0 + 3*x1",
            n_parameters=2,
        )
        # Observation close to prediction
        new_x = [1.0, 1.0]
        new_y = 5.0  # predicted: 2*1 + 3*1 = 5.0
        h = self.tester.sequential_update(h, new_x, new_y, self.var_names)
        self.assertEqual(len(h.evidence), 1)
        self.assertTrue(h.evidence[0].within_ci)
        self.assertEqual(h.support_count, 1)

    def test_check_falsification(self) -> None:
        h = Hypothesis(
            id="H_false",
            description="falsification test",
            equation="x0",
            n_parameters=1,
        )
        # Add 3 consecutive refuting observations (far from prediction)
        for _ in range(3):
            pred = Prediction(
                hypothesis_id=h.id,
                variable="y",
                predicted_value=0.0,
                confidence_interval=(-0.5, 0.5),
            )
            ev = Evidence(prediction=pred, observed_value=100.0)
            h.add_evidence(ev)

        self.assertTrue(self.tester.check_falsification(h, threshold=3))

        # Add a supporting observation at the end
        pred_ok = Prediction(
            hypothesis_id=h.id,
            variable="y",
            predicted_value=0.0,
            confidence_interval=(-0.5, 0.5),
        )
        ev_ok = Evidence(prediction=pred_ok, observed_value=0.1)
        h.add_evidence(ev_ok)
        # Now the last 3 are [refute, refute, support] -> not falsified
        self.assertFalse(self.tester.check_falsification(h, threshold=3))

    def test_compare_all(self) -> None:
        h1 = Hypothesis(
            id="H1",
            description="true",
            equation="2*x0 + 3*x1",
            n_parameters=2,
        )
        h2 = Hypothesis(
            id="H2",
            description="wrong",
            equation="x0",
            n_parameters=1,
        )
        h3 = Hypothesis(
            id="H3",
            description="also wrong",
            equation="x1",
            n_parameters=1,
        )
        results = self.tester.compare_all(
            [h1, h2, h3], self.X, self.y, self.var_names
        )
        self.assertEqual(len(results), 3)
        # Results sorted by BIC ascending
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["hypothesis_id"], "H1")
        # Best has BF_vs_best == 1.0
        self.assertAlmostEqual(results[0]["bayes_factor_vs_best"], 1.0)
        # Worse hypotheses have BF > 1
        self.assertGreater(results[1]["bayes_factor_vs_best"], 1.0)


# ---------------------------------------------------------------------------
# TestHypothesisTracker
# ---------------------------------------------------------------------------

class TestHypothesisTracker(unittest.TestCase):
    """Test hypothesis lifecycle management and experiment suggestion."""

    def setUp(self) -> None:
        self.tracker = HypothesisTracker()

    def test_add_and_get(self) -> None:
        h = Hypothesis(id="H0001", description="test add")
        self.tracker.add(h)
        retrieved = self.tracker.get("H0001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, "H0001")  # type: ignore[union-attr]
        self.assertIsNone(self.tracker.get("HXXXX"))

    def test_update_status(self) -> None:
        h = Hypothesis(id="H0001", description="status test")
        self.tracker.add(h)
        self.tracker.update_status("H0001", HypothesisStatus.TESTING)
        self.assertEqual(
            self.tracker.get("H0001").status,  # type: ignore[union-attr]
            HypothesisStatus.TESTING,
        )

    def test_update_with_observation(self) -> None:
        h = Hypothesis(
            id="H0001",
            description="obs test",
            equation="2*x0 + 3*x1",
            n_parameters=2,
        )
        self.tracker.add(h)
        obs = {"x0": 1.0, "x1": 1.0, "y": 5.0}
        self.tracker.update_with_observation(obs, var_names=["x0", "x1"])
        updated = self.tracker.get("H0001")
        self.assertEqual(len(updated.evidence), 1)  # type: ignore[union-attr]

    def test_suggest_discriminating_experiment(self) -> None:
        h1 = Hypothesis(
            id="H1",
            description="linear",
            equation="2*x0 + 3*x1",
            n_parameters=2,
        )
        h2 = Hypothesis(
            id="H2",
            description="quadratic",
            equation="x0**2 + x1",
            n_parameters=2,
        )
        self.tracker.add(h1)
        self.tracker.add(h2)

        result = self.tracker.suggest_discriminating_experiment(
            "H1",
            "H2",
            parameter_ranges={"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)},
            n_candidates=50,
            seed=42,
        )
        self.assertIn("point", result)
        self.assertIn("divergence", result)
        self.assertGreater(result["divergence"], 0.0)

    def test_get_status_report(self) -> None:
        h1 = Hypothesis(id="H1", description="hyp 1")
        h2 = Hypothesis(
            id="H2",
            description="hyp 2",
            status=HypothesisStatus.SUPPORTED,
        )
        self.tracker.add(h1)
        self.tracker.add(h2)

        report = self.tracker.get_status_report()
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["counts_by_status"]["proposed"], 1)
        self.assertEqual(report["counts_by_status"]["supported"], 1)
        self.assertGreater(len(report["top_hypotheses"]), 0)

    def test_get_active_hypotheses(self) -> None:
        h1 = Hypothesis(id="H1", description="proposed")
        h2 = Hypothesis(
            id="H2",
            description="testing",
            status=HypothesisStatus.TESTING,
        )
        h3 = Hypothesis(
            id="H3",
            description="refuted",
            status=HypothesisStatus.REFUTED,
        )
        self.tracker.add(h1)
        self.tracker.add(h2)
        self.tracker.add(h3)

        active = self.tracker.get_active_hypotheses()
        active_ids = {h.id for h in active}
        self.assertEqual(active_ids, {"H1", "H2"})

    def test_serialization_roundtrip(self) -> None:
        h1 = Hypothesis(
            id="H1",
            description="roundtrip 1",
            equation="x0 + x1",
            source="symreg",
        )
        h2 = Hypothesis(
            id="H2",
            description="roundtrip 2",
            status=HypothesisStatus.TESTING,
        )
        self.tracker.add(h1)
        self.tracker.add(h2)
        self.tracker.update_status("H2", HypothesisStatus.SUPPORTED)

        d = self.tracker.to_dict()
        tracker2 = HypothesisTracker.from_dict(d)

        self.assertEqual(len(tracker2.hypotheses), 2)
        self.assertEqual(
            tracker2.get("H2").status,  # type: ignore[union-attr]
            HypothesisStatus.SUPPORTED,
        )
        self.assertEqual(
            tracker2.get("H1").equation,  # type: ignore[union-attr]
            "x0 + x1",
        )
        self.assertGreater(len(tracker2.history), 0)


if __name__ == "__main__":
    unittest.main()
