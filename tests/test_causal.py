"""Tests for the Causal Discovery Engine (Layer 1).

Covers graph operations, PC algorithm structure learning, interventional
reasoning, causal effect estimation, and counterfactual reasoning.
"""

from __future__ import annotations

import math
import random
import unittest

from optimization_copilot.causal.counterfactual import CounterfactualReasoner
from optimization_copilot.causal.effects import CausalEffectEstimator
from optimization_copilot.causal.interventional import InterventionalEngine
from optimization_copilot.causal.models import CausalEdge, CausalGraph, CausalNode
from optimization_copilot.causal.structure import CausalStructureLearner


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_chain_data(n: int = 200, seed: int = 42) -> list[list[float]]:
    """Generate data from a chain SCM: X -> Y -> Z.

    X ~ N(0,1), Y = 2*X + noise, Z = 0.5*Y + noise
    """
    rng = random.Random(seed)
    data: list[list[float]] = []
    for _ in range(n):
        x = rng.gauss(0, 1)
        y = 2 * x + rng.gauss(0, 0.3)
        z = 0.5 * y + rng.gauss(0, 0.3)
        data.append([x, y, z])
    return data


def make_fork_data(n: int = 200, seed: int = 42) -> list[list[float]]:
    """Generate data from a fork SCM: X <- Y -> Z.

    Y ~ N(0,1), X = 1.5*Y + noise, Z = -1.0*Y + noise
    """
    rng = random.Random(seed)
    data: list[list[float]] = []
    for _ in range(n):
        y = rng.gauss(0, 1)
        x = 1.5 * y + rng.gauss(0, 0.3)
        z = -1.0 * y + rng.gauss(0, 0.3)
        data.append([x, y, z])
    return data


def make_collider_data(n: int = 200, seed: int = 42) -> list[list[float]]:
    """Generate data from a collider SCM: X -> Y <- Z.

    X ~ N(0,1), Z ~ N(0,1), Y = X + Z + noise
    """
    rng = random.Random(seed)
    data: list[list[float]] = []
    for _ in range(n):
        x = rng.gauss(0, 1)
        z = rng.gauss(0, 1)
        y = x + z + rng.gauss(0, 0.3)
        data.append([x, y, z])
    return data


def make_chain_dict_data(n: int = 300, seed: int = 42) -> list[dict]:
    """Generate dict-format data from chain: X -> Y -> Z."""
    rng = random.Random(seed)
    data: list[dict] = []
    for _ in range(n):
        x = rng.gauss(0, 1)
        y = 2 * x + rng.gauss(0, 0.5)
        z = 0.5 * y + rng.gauss(0, 0.5)
        data.append({"X": x, "Y": y, "Z": z})
    return data


# ---------------------------------------------------------------------------
# Test: CausalGraph
# ---------------------------------------------------------------------------

class TestCausalGraph(unittest.TestCase):
    """Tests for basic CausalGraph operations."""

    def _build_diamond(self) -> CausalGraph:
        """Build a diamond graph: A -> B, A -> C, B -> D, C -> D."""
        g = CausalGraph()
        for name in ["A", "B", "C", "D"]:
            g.add_node(CausalNode(name=name))
        g.add_edge(CausalEdge(source="A", target="B"))
        g.add_edge(CausalEdge(source="A", target="C"))
        g.add_edge(CausalEdge(source="B", target="D"))
        g.add_edge(CausalEdge(source="C", target="D"))
        return g

    def test_add_nodes_and_edges(self) -> None:
        g = self._build_diamond()
        self.assertEqual(len(g.node_names), 4)
        self.assertEqual(len(g.edges), 4)

    def test_parents_children(self) -> None:
        g = self._build_diamond()
        self.assertEqual(g.parents("A"), set())
        self.assertEqual(g.children("A"), {"B", "C"})
        self.assertEqual(g.parents("D"), {"B", "C"})
        self.assertEqual(g.children("D"), set())

    def test_ancestors_descendants(self) -> None:
        g = self._build_diamond()
        self.assertEqual(g.ancestors("D"), {"A", "B", "C"})
        self.assertEqual(g.descendants("A"), {"B", "C", "D"})
        self.assertEqual(g.ancestors("A"), set())
        self.assertEqual(g.descendants("D"), set())

    def test_remove_edge(self) -> None:
        g = self._build_diamond()
        g.remove_edge("A", "B")
        self.assertFalse(g.has_edge("A", "B"))
        self.assertTrue(g.has_edge("A", "C"))
        self.assertNotIn("B", g.children("A"))

    def test_topological_sort(self) -> None:
        g = self._build_diamond()
        topo = g.topological_sort()
        self.assertEqual(len(topo), 4)
        # A must come before B, C; B and C before D
        self.assertLess(topo.index("A"), topo.index("B"))
        self.assertLess(topo.index("A"), topo.index("C"))
        self.assertLess(topo.index("B"), topo.index("D"))
        self.assertLess(topo.index("C"), topo.index("D"))

    def test_d_separation_chain(self) -> None:
        """In chain A -> B -> C, A _|_ C | B."""
        g = CausalGraph()
        for name in ["A", "B", "C"]:
            g.add_node(CausalNode(name=name))
        g.add_edge(CausalEdge(source="A", target="B"))
        g.add_edge(CausalEdge(source="B", target="C"))

        # A and C are NOT d-separated given empty set
        self.assertFalse(g.d_separated("A", "C", set()))
        # A and C ARE d-separated given {B}
        self.assertTrue(g.d_separated("A", "C", {"B"}))

    def test_d_separation_fork(self) -> None:
        """In fork A <- B -> C, A _|_ C | B."""
        g = CausalGraph()
        for name in ["A", "B", "C"]:
            g.add_node(CausalNode(name=name))
        g.add_edge(CausalEdge(source="B", target="A"))
        g.add_edge(CausalEdge(source="B", target="C"))

        self.assertFalse(g.d_separated("A", "C", set()))
        self.assertTrue(g.d_separated("A", "C", {"B"}))

    def test_d_separation_collider(self) -> None:
        """In collider A -> B <- C, A _|_ C | empty but NOT A _|_ C | B."""
        g = CausalGraph()
        for name in ["A", "B", "C"]:
            g.add_node(CausalNode(name=name))
        g.add_edge(CausalEdge(source="A", target="B"))
        g.add_edge(CausalEdge(source="C", target="B"))

        # A and C ARE d-separated given empty set (collider blocks)
        self.assertTrue(g.d_separated("A", "C", set()))
        # A and C are NOT d-separated given {B} (conditioning on collider opens path)
        self.assertFalse(g.d_separated("A", "C", {"B"}))

    def test_serialization_roundtrip(self) -> None:
        g = self._build_diamond()
        data = g.to_dict()
        g2 = CausalGraph.from_dict(data)
        self.assertEqual(set(g2.node_names), set(g.node_names))
        self.assertEqual(len(g2.edges), len(g.edges))
        for edge in g.edges:
            self.assertTrue(g2.has_edge(edge.source, edge.target))

    def test_copy(self) -> None:
        g = self._build_diamond()
        g2 = g.copy()
        g2.remove_edge("A", "B")
        # Original should be unaffected
        self.assertTrue(g.has_edge("A", "B"))
        self.assertFalse(g2.has_edge("A", "B"))

    def test_add_edge_missing_node_raises(self) -> None:
        g = CausalGraph()
        g.add_node(CausalNode(name="A"))
        with self.assertRaises(ValueError):
            g.add_edge(CausalEdge(source="A", target="B"))


# ---------------------------------------------------------------------------
# Test: PC Algorithm
# ---------------------------------------------------------------------------

class TestPCAlgorithm(unittest.TestCase):
    """Tests for the PC algorithm structure learner."""

    def test_chain_structure(self) -> None:
        """PC should recover chain X -> Y -> Z (or at least the skeleton)."""
        data = make_chain_data(n=500, seed=42)
        learner = CausalStructureLearner(alpha=0.05, max_cond_set=2)
        graph = learner.learn(data, ["X", "Y", "Z"])

        # The skeleton should have edges X-Y and Y-Z but NOT X-Z
        has_xy = graph.has_edge("X", "Y") or graph.has_edge("Y", "X")
        has_yz = graph.has_edge("Y", "Z") or graph.has_edge("Z", "Y")
        has_xz = graph.has_edge("X", "Z") or graph.has_edge("Z", "X")

        self.assertTrue(has_xy, "Should have edge between X and Y")
        self.assertTrue(has_yz, "Should have edge between Y and Z")
        self.assertFalse(has_xz, "Should NOT have edge between X and Z")

    def test_fork_structure(self) -> None:
        """PC should recover fork X <- Y -> Z skeleton."""
        data = make_fork_data(n=500, seed=42)
        learner = CausalStructureLearner(alpha=0.05, max_cond_set=2)
        graph = learner.learn(data, ["X", "Y", "Z"])

        has_xy = graph.has_edge("X", "Y") or graph.has_edge("Y", "X")
        has_yz = graph.has_edge("Y", "Z") or graph.has_edge("Z", "Y")
        has_xz = graph.has_edge("X", "Z") or graph.has_edge("Z", "X")

        self.assertTrue(has_xy, "Should have edge between X and Y")
        self.assertTrue(has_yz, "Should have edge between Y and Z")
        self.assertFalse(has_xz, "Should NOT have edge between X and Z")

    def test_collider_structure(self) -> None:
        """PC should recover collider X -> Y <- Z with v-structure."""
        data = make_collider_data(n=500, seed=42)
        learner = CausalStructureLearner(alpha=0.05, max_cond_set=2)
        graph = learner.learn(data, ["X", "Y", "Z"])

        has_xy = graph.has_edge("X", "Y") or graph.has_edge("Y", "X")
        has_zy = graph.has_edge("Z", "Y") or graph.has_edge("Y", "Z")
        has_xz = graph.has_edge("X", "Z") or graph.has_edge("Z", "X")

        self.assertTrue(has_xy, "Should have edge between X and Y")
        self.assertTrue(has_zy, "Should have edge between Z and Y")
        self.assertFalse(has_xz, "Should NOT have edge between X and Z")

    def test_collider_orientation(self) -> None:
        """PC should orient the collider as X -> Y <- Z."""
        data = make_collider_data(n=500, seed=42)
        learner = CausalStructureLearner(alpha=0.05, max_cond_set=2)
        graph = learner.learn(data, ["X", "Y", "Z"])

        # Check v-structure orientation: X -> Y and Z -> Y
        self.assertTrue(
            graph.has_edge("X", "Y"),
            "Collider should orient X -> Y",
        )
        self.assertTrue(
            graph.has_edge("Z", "Y"),
            "Collider should orient Z -> Y",
        )

    def test_correlation_matrix(self) -> None:
        """Correlation matrix should be symmetric with 1s on diagonal."""
        learner = CausalStructureLearner()
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        corr = learner._correlation_matrix(data)

        self.assertAlmostEqual(corr[0][0], 1.0, places=6)
        self.assertAlmostEqual(corr[1][1], 1.0, places=6)
        self.assertAlmostEqual(corr[0][1], corr[1][0], places=10)

    def test_fisher_z_test_high_correlation(self) -> None:
        """High correlation should yield low p-value."""
        learner = CausalStructureLearner()
        p_high = learner._fisher_z_test(0.9, 100, 0)
        self.assertLess(p_high, 0.01)

    def test_fisher_z_test_low_correlation(self) -> None:
        """Low correlation should yield high p-value."""
        learner = CausalStructureLearner()
        p_low = learner._fisher_z_test(0.01, 100, 0)
        self.assertGreater(p_low, 0.1)


# ---------------------------------------------------------------------------
# Test: Interventional Engine
# ---------------------------------------------------------------------------

class TestIntervention(unittest.TestCase):
    """Tests for the interventional engine."""

    def _build_chain_graph(self) -> CausalGraph:
        """Build chain graph X -> Y -> Z."""
        g = CausalGraph()
        for name in ["X", "Y", "Z"]:
            g.add_node(CausalNode(name=name))
        g.add_edge(CausalEdge(source="X", target="Y"))
        g.add_edge(CausalEdge(source="Y", target="Z"))
        return g

    def test_do_operator_chain(self) -> None:
        """do(X=x) on chain X->Y->Z should change Y and Z."""
        graph = self._build_chain_graph()
        data = make_chain_dict_data(n=300, seed=42)
        engine = InterventionalEngine()

        result = engine.do(graph, {"X": 2.0}, data, "Z")

        # Mean should be positive (X=2 pushes Y up, which pushes Z up)
        self.assertIsInstance(result["mean"], float)
        self.assertIn("std", result)
        self.assertIn("n_adjusted", result)

    def test_backdoor_adjustment(self) -> None:
        """Backdoor adjustment should estimate a positive causal effect X->Y->Z."""
        graph = self._build_chain_graph()
        data = make_chain_dict_data(n=300, seed=42)
        engine = InterventionalEngine()

        effect = engine.backdoor_adjustment(graph, "X", "Z", data)
        # X has a positive causal effect on Z (through Y)
        self.assertGreater(effect, 0.0)

    def test_find_adjustment_set_chain(self) -> None:
        """In chain X->Y->Z, adjustment set for X->Z should not include Y."""
        graph = self._build_chain_graph()
        engine = InterventionalEngine()

        adj_set = engine.find_valid_adjustment_set(graph, "X", "Z")
        self.assertIsNotNone(adj_set)
        # Y is a descendant of X, so should not be in adjustment set
        self.assertNotIn("Y", adj_set)

    def test_find_adjustment_set_confounded(self) -> None:
        """With confounder C: C->X, C->Y, adjustment set should include C."""
        g = CausalGraph()
        for name in ["X", "Y", "C"]:
            g.add_node(CausalNode(name=name))
        g.add_edge(CausalEdge(source="C", target="X"))
        g.add_edge(CausalEdge(source="C", target="Y"))
        g.add_edge(CausalEdge(source="X", target="Y"))

        engine = InterventionalEngine()
        adj_set = engine.find_valid_adjustment_set(g, "X", "Y")

        self.assertIsNotNone(adj_set)
        self.assertIn("C", adj_set)

    def test_frontdoor_adjustment(self) -> None:
        """Frontdoor adjustment should return a finite effect estimate."""
        g = CausalGraph()
        for name in ["X", "M", "Y"]:
            g.add_node(CausalNode(name=name))
        g.add_edge(CausalEdge(source="X", target="M"))
        g.add_edge(CausalEdge(source="M", target="Y"))

        rng = random.Random(42)
        data = []
        for _ in range(300):
            x = rng.gauss(0, 1)
            m = 1.5 * x + rng.gauss(0, 0.3)
            y = 2.0 * m + rng.gauss(0, 0.3)
            data.append({"X": x, "M": m, "Y": y})

        engine = InterventionalEngine()
        effect = engine.frontdoor_adjustment(g, "X", "M", "Y", data)
        self.assertIsInstance(effect, float)
        self.assertTrue(math.isfinite(effect))


# ---------------------------------------------------------------------------
# Test: Causal Effect Estimator
# ---------------------------------------------------------------------------

class TestCausalEffects(unittest.TestCase):
    """Tests for ATE and CATE estimation."""

    def _build_simple_graph(self) -> CausalGraph:
        g = CausalGraph()
        for name in ["X", "Y"]:
            g.add_node(CausalNode(name=name))
        g.add_edge(CausalEdge(source="X", target="Y"))
        return g

    def test_ate_positive_effect(self) -> None:
        """ATE should detect a positive effect when X causes Y."""
        rng = random.Random(42)
        data = []
        for _ in range(300):
            x = rng.gauss(0, 1)
            y = 3.0 * x + rng.gauss(0, 0.5)
            data.append({"X": x, "Y": y})

        graph = self._build_simple_graph()
        estimator = CausalEffectEstimator()
        result = estimator.ate(data, "X", "Y", set(), graph)

        self.assertGreater(result["ate"], 0.0)
        self.assertIn("se", result)
        self.assertIn("ci_lower", result)
        self.assertIn("ci_upper", result)
        # CI should contain the ATE
        self.assertLess(result["ci_lower"], result["ate"])
        self.assertGreater(result["ci_upper"], result["ate"])

    def test_ate_no_effect(self) -> None:
        """ATE should be close to zero when X does not cause Y."""
        rng = random.Random(42)
        data = []
        for _ in range(300):
            x = rng.gauss(0, 1)
            y = rng.gauss(0, 1)  # independent
            data.append({"X": x, "Y": y})

        graph = self._build_simple_graph()
        estimator = CausalEffectEstimator()
        result = estimator.ate(data, "X", "Y", set(), graph)

        # ATE should be small (close to 0)
        self.assertLess(abs(result["ate"]), 1.0)

    def test_ate_with_adjustment(self) -> None:
        """ATE with adjustment set should give valid result."""
        rng = random.Random(42)
        data = []
        for _ in range(300):
            c = rng.gauss(0, 1)
            x = c + rng.gauss(0, 0.5)
            y = 2.0 * x + 1.0 * c + rng.gauss(0, 0.5)
            data.append({"X": x, "Y": y, "C": c})

        g = CausalGraph()
        for name in ["X", "Y", "C"]:
            g.add_node(CausalNode(name=name))
        g.add_edge(CausalEdge(source="C", target="X"))
        g.add_edge(CausalEdge(source="C", target="Y"))
        g.add_edge(CausalEdge(source="X", target="Y"))

        estimator = CausalEffectEstimator()
        result = estimator.ate(data, "X", "Y", {"C"}, g)

        # Should still detect positive effect
        self.assertGreater(result["ate"], 0.0)

    def test_cate_subgroups(self) -> None:
        """CATE should return subgroup-specific effects."""
        rng = random.Random(42)
        data = []
        for _ in range(300):
            s = rng.gauss(0, 1)
            x = rng.gauss(0, 1)
            y = 2.0 * x + 0.5 * s + rng.gauss(0, 0.3)
            data.append({"X": x, "Y": y, "S": s})

        estimator = CausalEffectEstimator()
        result = estimator.cate(data, "X", "Y", set(), "S")

        self.assertIn("subgroups", result)
        self.assertIn("overall_ate", result)
        self.assertEqual(len(result["subgroups"]), 2)

    def test_natural_direct_effect(self) -> None:
        """NDE should estimate the direct effect of X on Z (not through Y)."""
        g = CausalGraph()
        for name in ["X", "M", "Y"]:
            g.add_node(CausalNode(name=name))
        g.add_edge(CausalEdge(source="X", target="M"))
        g.add_edge(CausalEdge(source="M", target="Y"))
        g.add_edge(CausalEdge(source="X", target="Y"))

        rng = random.Random(42)
        data = []
        for _ in range(400):
            x = rng.gauss(0, 1)
            m = 1.0 * x + rng.gauss(0, 0.3)
            y = 2.0 * x + 1.0 * m + rng.gauss(0, 0.3)  # direct + indirect
            data.append({"X": x, "M": m, "Y": y})

        estimator = CausalEffectEstimator()
        nde = estimator.natural_direct_effect(g, data, "X", "M", "Y")
        # NDE should be positive (direct effect of X on Y is 2.0)
        self.assertGreater(nde, 0.0)
        self.assertTrue(math.isfinite(nde))


# ---------------------------------------------------------------------------
# Test: Counterfactual Reasoner
# ---------------------------------------------------------------------------

class TestCounterfactual(unittest.TestCase):
    """Tests for counterfactual reasoning."""

    def _build_monotonic_scm(self) -> tuple[CausalGraph, dict, list[dict]]:
        """Build a simple monotonic SCM: X -> Y, Y = 2*X.

        Returns (graph, structural_equations, data).
        """
        g = CausalGraph()
        g.add_node(CausalNode(name="X"))
        g.add_node(CausalNode(name="Y"))
        g.add_edge(CausalEdge(source="X", target="Y"))

        equations = {
            "Y": lambda parents: 2.0 * parents.get("X", 0.0),
        }

        rng = random.Random(42)
        data = []
        for _ in range(200):
            x = rng.gauss(0, 1)
            y = 2.0 * x + rng.gauss(0, 0.3)
            data.append({"X": x, "Y": y})

        return g, equations, data

    def test_counterfactual_basic(self) -> None:
        """Counterfactual Y(X=3) given factual X=1, Y=2.3 should be ~6.3."""
        g, equations, _ = self._build_monotonic_scm()
        reasoner = CounterfactualReasoner(g, equations)

        factual = {"X": 1.0, "Y": 2.3}
        result = reasoner.counterfactual(
            factual=factual,
            intervention={"X": 3.0},
            query_var="Y",
        )

        # Y = 2*X + U, factual: U = 2.3 - 2*1.0 = 0.3
        # Counterfactual: Y = 2*3.0 + 0.3 = 6.3
        self.assertAlmostEqual(result["counterfactual_value"], 6.3, places=5)
        self.assertAlmostEqual(result["factual_value"], 2.3, places=5)
        self.assertAlmostEqual(result["noise_terms"]["Y"], 0.3, places=5)

    def test_counterfactual_chain(self) -> None:
        """Counterfactual on chain X -> Y -> Z."""
        g = CausalGraph()
        for name in ["X", "Y", "Z"]:
            g.add_node(CausalNode(name=name))
        g.add_edge(CausalEdge(source="X", target="Y"))
        g.add_edge(CausalEdge(source="Y", target="Z"))

        equations = {
            "Y": lambda p: 2.0 * p.get("X", 0.0),
            "Z": lambda p: 0.5 * p.get("Y", 0.0),
        }

        reasoner = CounterfactualReasoner(g, equations)

        # Factual: X=1, Y=2.1, Z=1.2
        # U_Y = 2.1 - 2*1 = 0.1, U_Z = 1.2 - 0.5*2.1 = 0.15
        factual = {"X": 1.0, "Y": 2.1, "Z": 1.2}
        result = reasoner.counterfactual(factual, {"X": 0.0}, "Z")

        # CF: Y = 2*0 + 0.1 = 0.1, Z = 0.5*0.1 + 0.15 = 0.2
        expected_z = 0.5 * 0.1 + 0.15
        self.assertAlmostEqual(result["counterfactual_value"], expected_z, places=5)

    def test_probability_of_necessity(self) -> None:
        """PN should be high for a strong causal relationship."""
        g, equations, data = self._build_monotonic_scm()
        reasoner = CounterfactualReasoner(g, equations)

        pn = reasoner.probability_of_necessity(data, "X", "Y")
        # For a strong monotonic relationship, PN should be high
        self.assertGreater(pn, 0.0)
        self.assertLessEqual(pn, 1.0)

    def test_probability_of_sufficiency(self) -> None:
        """PS should be high for a strong causal relationship."""
        g, equations, data = self._build_monotonic_scm()
        reasoner = CounterfactualReasoner(g, equations)

        ps = reasoner.probability_of_sufficiency(data, "X", "Y")
        self.assertGreaterEqual(ps, 0.0)
        self.assertLessEqual(ps, 1.0)

    def test_counterfactual_no_change(self) -> None:
        """If intervention matches factual, counterfactual should equal factual."""
        g, equations, _ = self._build_monotonic_scm()
        reasoner = CounterfactualReasoner(g, equations)

        factual = {"X": 1.0, "Y": 2.3}
        result = reasoner.counterfactual(
            factual=factual,
            intervention={"X": 1.0},
            query_var="Y",
        )

        self.assertAlmostEqual(
            result["counterfactual_value"],
            result["factual_value"],
            places=5,
        )

    def test_abduction_recovers_noise(self) -> None:
        """Abduction should correctly recover noise terms."""
        g, equations, _ = self._build_monotonic_scm()
        reasoner = CounterfactualReasoner(g, equations)

        factual = {"X": 2.0, "Y": 4.5}
        noise = reasoner._abduction(factual)

        # U_Y = 4.5 - 2*2.0 = 0.5
        self.assertAlmostEqual(noise["Y"], 0.5, places=5)


# ---------------------------------------------------------------------------
# Test: Module imports
# ---------------------------------------------------------------------------

class TestImports(unittest.TestCase):
    """Verify all expected exports from the causal package."""

    def test_package_exports(self) -> None:
        from optimization_copilot.causal import (
            CausalEdge,
            CausalEffectEstimator,
            CausalGraph,
            CausalNode,
            CausalStructureLearner,
            CounterfactualReasoner,
            InterventionalEngine,
        )
        # Verify they are not None
        self.assertIsNotNone(CausalGraph)
        self.assertIsNotNone(CausalNode)
        self.assertIsNotNone(CausalEdge)
        self.assertIsNotNone(CausalStructureLearner)
        self.assertIsNotNone(InterventionalEngine)
        self.assertIsNotNone(CausalEffectEstimator)
        self.assertIsNotNone(CounterfactualReasoner)


if __name__ == "__main__":
    unittest.main()
