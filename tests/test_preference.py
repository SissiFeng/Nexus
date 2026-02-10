"""Tests for the preference learning package.

Verifies:
- PairwisePreference: construction, serialization, confidence, metadata
- PreferenceModel: construction, serialization, scores, log-likelihood
- PreferenceRanking: construction, serialization, nested dict keys
- PreferenceLearner.fit (Bradley-Terry): empty, single, transitive, cyclic,
  convergence, determinism, confidence weighting, self-comparison guard
- PreferenceLearner.log_likelihood: correctness of fitted model LL
- PreferenceLearner.rank_with_preferences: Pareto + preference integration
- PreferenceLearner.add_preference: immutability, index validation
- Edge cases: single observation, disconnected comparison graph

~32 tests covering unit, edge-case, and integration scenarios.
"""

from __future__ import annotations

import math

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.multi_objective.pareto import (
    MultiObjectiveAnalyzer,
    ParetoResult,
)
from optimization_copilot.preference.models import (
    PairwisePreference,
    PreferenceModel,
    PreferenceRanking,
)
from optimization_copilot.preference.learner import PreferenceLearner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_specs(n_params: int = 2) -> list[ParameterSpec]:
    """Continuous params on [0, 10]."""
    return [
        ParameterSpec(
            name=f"x{i + 1}",
            type=VariableType.CONTINUOUS,
            lower=0.0,
            upper=10.0,
        )
        for i in range(n_params)
    ]


def _make_obs(
    iteration: int,
    params: dict,
    kpis: dict,
    **kwargs,
) -> Observation:
    """Single Observation helper."""
    return Observation(
        iteration=iteration,
        parameters=params,
        kpi_values=kpis,
        **kwargs,
    )


def _make_multi_obj_snapshot(n_obs: int = 10) -> CampaignSnapshot:
    """Multi-objective snapshot with 2 KPIs where obs have varied trade-offs.

    KPI 'accuracy' increases with index, 'latency' also increases (minimize).
    This creates a natural trade-off between the two objectives.
    """
    specs = _make_specs()
    obs = []
    for i in range(n_obs):
        t = i / max(n_obs - 1, 1)
        obs.append(
            _make_obs(
                iteration=i,
                params={"x1": t * 10.0, "x2": (1 - t) * 10.0},
                kpis={
                    "accuracy": 0.5 + 0.5 * t,   # higher is better
                    "latency": 1.0 + 9.0 * t,     # lower is better
                },
            )
        )
    return CampaignSnapshot(
        campaign_id="multi-obj-pref",
        parameter_specs=specs,
        observations=obs,
        objective_names=["accuracy", "latency"],
        objective_directions=["maximize", "minimize"],
        current_iteration=n_obs,
    )


def _make_preferences(
    pairs: list[tuple[int, int]],
) -> list[PairwisePreference]:
    """Build list[PairwisePreference] from (winner, loser) tuples."""
    return [
        PairwisePreference(winner_idx=w, loser_idx=l)
        for w, l in pairs
    ]


# ---------------------------------------------------------------------------
# TestPairwisePreference
# ---------------------------------------------------------------------------


class TestPairwisePreference:
    """Tests for PairwisePreference dataclass."""

    def test_construction_defaults(self):
        """Default confidence=1.0 and empty metadata."""
        p = PairwisePreference(winner_idx=0, loser_idx=1)
        assert p.winner_idx == 0
        assert p.loser_idx == 1
        assert p.confidence == 1.0
        assert p.metadata == {}

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization roundtrip preserves all fields."""
        p = PairwisePreference(winner_idx=2, loser_idx=5, confidence=0.8)
        d = p.to_dict()
        restored = PairwisePreference.from_dict(d)
        assert restored.winner_idx == p.winner_idx
        assert restored.loser_idx == p.loser_idx
        assert abs(restored.confidence - p.confidence) < 1e-9
        assert restored.metadata == p.metadata

    def test_custom_confidence(self):
        """Non-default confidence is stored correctly."""
        p = PairwisePreference(winner_idx=0, loser_idx=1, confidence=0.3)
        assert abs(p.confidence - 0.3) < 1e-9

    def test_with_metadata(self):
        """Metadata dict is stored and survives roundtrip."""
        meta = {"source": "human", "session": 42}
        p = PairwisePreference(winner_idx=0, loser_idx=1, metadata=meta)
        assert p.metadata["source"] == "human"
        assert p.metadata["session"] == 42

        restored = PairwisePreference.from_dict(p.to_dict())
        assert restored.metadata == meta


# ---------------------------------------------------------------------------
# TestPreferenceModel
# ---------------------------------------------------------------------------


class TestPreferenceModel:
    """Tests for PreferenceModel dataclass."""

    def test_construction(self):
        """All fields are stored correctly."""
        m = PreferenceModel(
            scores={0: 1.0, 1: 0.5},
            n_preferences=3,
            n_items=2,
            converged=True,
            n_iterations=10,
            log_likelihood=-1.5,
        )
        assert m.scores[0] == 1.0
        assert m.scores[1] == 0.5
        assert m.n_preferences == 3
        assert m.n_items == 2
        assert m.converged is True
        assert m.n_iterations == 10
        assert abs(m.log_likelihood - (-1.5)) < 1e-9

    def test_to_dict_from_dict_roundtrip(self):
        """Int keys survive serialization (JSON converts to str, from_dict converts back)."""
        m = PreferenceModel(
            scores={0: 1.0, 1: 0.5, 2: 0.3},
            n_preferences=5,
            n_items=3,
            converged=True,
            n_iterations=20,
            log_likelihood=-2.0,
        )
        d = m.to_dict()
        # Simulate JSON roundtrip: keys become strings
        d["scores"] = {str(k): v for k, v in d["scores"].items()}
        restored = PreferenceModel.from_dict(d)
        assert isinstance(list(restored.scores.keys())[0], int)
        assert restored.scores[0] == 1.0
        assert restored.scores[2] == 0.3

    def test_empty_scores(self):
        """Model with empty scores dict is valid."""
        m = PreferenceModel(
            scores={},
            n_preferences=0,
            n_items=0,
            converged=True,
            n_iterations=0,
        )
        assert len(m.scores) == 0

    def test_log_likelihood_stored(self):
        """Log-likelihood default is 0.0 and can be set."""
        m_default = PreferenceModel(
            scores={0: 1.0}, n_preferences=0, n_items=1,
            converged=True, n_iterations=0,
        )
        assert m_default.log_likelihood == 0.0

        m_set = PreferenceModel(
            scores={0: 1.0}, n_preferences=1, n_items=1,
            converged=True, n_iterations=5, log_likelihood=-0.693,
        )
        assert abs(m_set.log_likelihood - (-0.693)) < 1e-6


# ---------------------------------------------------------------------------
# TestPreferenceRanking
# ---------------------------------------------------------------------------


class TestPreferenceRanking:
    """Tests for PreferenceRanking dataclass."""

    def test_construction(self):
        """All fields stored correctly."""
        r = PreferenceRanking(
            ranked_indices=[0, 2, 1],
            utility_scores={0: 1.0, 1: 0.3, 2: 0.7},
            dominance_ranks=[1, 2, 1],
            preference_within_rank={1: [0, 2], 2: [1]},
        )
        assert r.ranked_indices == [0, 2, 1]
        assert r.dominance_ranks == [1, 2, 1]
        assert r.preference_within_rank[1] == [0, 2]

    def test_to_dict_from_dict_roundtrip(self):
        """Nested dict keys (int) survive serialization roundtrip."""
        r = PreferenceRanking(
            ranked_indices=[0, 1],
            utility_scores={0: 1.0, 1: 0.5},
            dominance_ranks=[1, 1],
            preference_within_rank={1: [0, 1]},
        )
        d = r.to_dict()
        # Simulate JSON roundtrip: int keys become strings
        d["utility_scores"] = {str(k): v for k, v in d["utility_scores"].items()}
        d["preference_within_rank"] = {
            str(k): v for k, v in d["preference_within_rank"].items()
        }
        restored = PreferenceRanking.from_dict(d)
        assert isinstance(list(restored.utility_scores.keys())[0], int)
        assert isinstance(list(restored.preference_within_rank.keys())[0], int)
        assert restored.ranked_indices == [0, 1]

    def test_preference_within_rank_structure(self):
        """preference_within_rank groups indices by dominance rank."""
        r = PreferenceRanking(
            ranked_indices=[2, 0, 1],
            utility_scores={0: 0.5, 1: 0.3, 2: 1.0},
            dominance_ranks=[2, 2, 1],
            preference_within_rank={1: [2], 2: [0, 1]},
        )
        # Rank 1 contains index 2; rank 2 contains indices 0, 1
        assert 2 in r.preference_within_rank[1]
        assert 0 in r.preference_within_rank[2]
        assert 1 in r.preference_within_rank[2]


# ---------------------------------------------------------------------------
# TestBradleyTerryFit
# ---------------------------------------------------------------------------


class TestBradleyTerryFit:
    """Tests for PreferenceLearner.fit (Bradley-Terry MM algorithm)."""

    def test_empty_preferences_returns_uniform(self):
        """No preferences: all scores should be equal (uniform)."""
        learner = PreferenceLearner()
        model = learner.fit([], n_items=4)
        scores = list(model.scores.values())
        assert len(scores) == 4
        # All should be the same
        for s in scores:
            assert abs(s - scores[0]) < 1e-9
        assert model.converged is True
        assert model.n_iterations == 0
        assert model.n_preferences == 0

    def test_single_preference_winner_higher(self):
        """A>B should result in score_A > score_B."""
        learner = PreferenceLearner()
        prefs = _make_preferences([(0, 1)])
        model = learner.fit(prefs, n_items=2)
        assert model.scores[0] > model.scores[1]

    def test_three_item_transitive(self):
        """A>B, B>C should yield scores A > B > C."""
        learner = PreferenceLearner()
        prefs = _make_preferences([(0, 1), (1, 2)])
        model = learner.fit(prefs, n_items=3)
        assert model.scores[0] > model.scores[1]
        assert model.scores[1] > model.scores[2]

    def test_three_item_cycle(self):
        """A>B, B>C, C>A (cycle): scores should be approximately equal."""
        learner = PreferenceLearner()
        prefs = _make_preferences([(0, 1), (1, 2), (2, 0)])
        model = learner.fit(prefs, n_items=3)
        scores = list(model.scores.values())
        # In a perfect cycle, all items win and lose equally, so scores converge
        max_s = max(scores)
        min_s = min(scores)
        assert max_s - min_s < 0.15  # approximately equal

    def test_convergence_flag(self):
        """With enough iterations and cyclic preferences, model should converge."""
        # Cyclic preferences converge reliably because scores equalize
        learner = PreferenceLearner(max_iterations=500, epsilon=1e-4)
        prefs = _make_preferences([(0, 1), (1, 2), (2, 0)])
        model = learner.fit(prefs, n_items=3)
        assert model.converged is True

    def test_deterministic(self):
        """Same inputs produce identical outputs."""
        learner = PreferenceLearner()
        prefs = _make_preferences([(0, 1), (1, 2), (0, 2)])
        model1 = learner.fit(prefs, n_items=3)
        model2 = learner.fit(prefs, n_items=3)
        for i in range(3):
            assert abs(model1.scores[i] - model2.scores[i]) < 1e-12

    def test_confidence_weighting(self):
        """High confidence preference has more effect than low confidence."""
        learner = PreferenceLearner()

        # Case 1: A>B with high confidence, B>A with low confidence
        prefs_high_A = [
            PairwisePreference(winner_idx=0, loser_idx=1, confidence=5.0),
            PairwisePreference(winner_idx=1, loser_idx=0, confidence=1.0),
        ]
        model_high_A = learner.fit(prefs_high_A, n_items=2)

        # Case 2: A>B with low confidence, B>A with high confidence
        prefs_high_B = [
            PairwisePreference(winner_idx=0, loser_idx=1, confidence=1.0),
            PairwisePreference(winner_idx=1, loser_idx=0, confidence=5.0),
        ]
        model_high_B = learner.fit(prefs_high_B, n_items=2)

        # In case 1, A should score higher; in case 2, B should score higher
        assert model_high_A.scores[0] > model_high_A.scores[1]
        assert model_high_B.scores[1] > model_high_B.scores[0]

    def test_self_comparison_raises(self):
        """Comparing an item to itself should raise ValueError."""
        learner = PreferenceLearner()
        prefs = [PairwisePreference(winner_idx=2, loser_idx=2)]
        try:
            learner.fit(prefs, n_items=3)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "2" in str(e)


# ---------------------------------------------------------------------------
# TestLogLikelihood
# ---------------------------------------------------------------------------


class TestLogLikelihood:
    """Tests for log-likelihood of fitted Bradley-Terry model."""

    def test_positive_model_higher_ll(self):
        """A fitted model should have a negative (or zero) log-likelihood,
        and adding more consistent preferences should keep LL reasonable."""
        learner = PreferenceLearner()
        # Fit with consistent transitive preferences
        prefs = _make_preferences([(0, 1), (0, 2), (1, 2)])
        model = learner.fit(prefs, n_items=3)

        # LL should be negative (log of probabilities < 1)
        assert model.log_likelihood <= 0.0

        # Fit with conflicting preferences (A>B and B>A)
        prefs_conflict = [
            PairwisePreference(winner_idx=0, loser_idx=1, confidence=1.0),
            PairwisePreference(winner_idx=1, loser_idx=0, confidence=1.0),
        ]
        model_conflict = learner.fit(prefs_conflict, n_items=2)

        # Conflicting model: each preference gets p ~ 0.5, so LL ~ 2*log(0.5) ~ -1.39
        # The transitive model pushes probabilities toward 1 so its LL is closer to 0
        assert model.log_likelihood > model_conflict.log_likelihood

    def test_perfect_model_ll_near_zero(self):
        """A strong/dominant model should have LL closer to 0 (less negative)."""
        learner = PreferenceLearner()
        # Give many repeated preferences to strengthen the model
        prefs = _make_preferences([(0, 1)] * 10)
        model = learner.fit(prefs, n_items=2)
        # LL is sum of log(p_winner), each p_winner should be close to 1
        # so LL should be close to 0 (each term close to log(1)=0)
        assert model.log_likelihood > -1.0  # not too negative
        assert model.log_likelihood <= 0.0  # LL is always <= 0

    def test_empty_preferences_ll(self):
        """No preferences: LL should be 0.0."""
        learner = PreferenceLearner()
        model = learner.fit([], n_items=3)
        assert model.log_likelihood == 0.0


# ---------------------------------------------------------------------------
# TestRankWithPreferences
# ---------------------------------------------------------------------------


class TestRankWithPreferences:
    """Tests for PreferenceLearner.rank_with_preferences."""

    def test_reranks_within_pareto_layer(self):
        """Preferences should reorder items within the same dominance rank."""
        snap = _make_multi_obj_snapshot(n_obs=5)
        learner = PreferenceLearner()

        # Compute Pareto result to see which obs share a rank
        pareto_result = MultiObjectiveAnalyzer().analyze(snap)

        # Find a rank with multiple items
        from collections import Counter
        rank_counts = Counter(pareto_result.dominance_ranks)
        multi_rank = None
        for rank, count in rank_counts.items():
            if count >= 2:
                multi_rank = rank
                break

        if multi_rank is not None:
            # Get indices in that rank
            indices_in_rank = [
                i for i, r in enumerate(pareto_result.dominance_ranks)
                if r == multi_rank
            ]
            # Prefer the last one over the first one
            winner = indices_in_rank[-1]
            loser = indices_in_rank[0]
            prefs = _make_preferences([(winner, loser)])

            ranking = learner.rank_with_preferences(
                snap, prefs, pareto_result=pareto_result,
            )
            # Within this rank, winner should come before loser
            within = ranking.preference_within_rank[multi_rank]
            assert within.index(winner) < within.index(loser)

    def test_dominance_rank_preserved(self):
        """Rank-1 items should always appear before rank-2 items."""
        snap = _make_multi_obj_snapshot(n_obs=10)
        learner = PreferenceLearner()

        pareto_result = MultiObjectiveAnalyzer().analyze(snap)
        # Prefer a rank-2 item over a rank-1 item (should not override dominance)
        rank1_indices = [
            i for i, r in enumerate(pareto_result.dominance_ranks) if r == 1
        ]
        rank2_indices = [
            i for i, r in enumerate(pareto_result.dominance_ranks) if r == 2
        ]

        if rank1_indices and rank2_indices:
            # Even if we prefer rank2 over rank1, dominance takes priority
            prefs = _make_preferences([(rank2_indices[0], rank1_indices[0])])
            ranking = learner.rank_with_preferences(
                snap, prefs, pareto_result=pareto_result,
            )
            # All rank-1 items should appear before any rank-2 item
            ranked = ranking.ranked_indices
            last_rank1_pos = max(
                ranked.index(i) for i in rank1_indices
            )
            first_rank2_pos = min(
                ranked.index(i) for i in rank2_indices
            )
            assert last_rank1_pos < first_rank2_pos

    def test_no_preferences_returns_original_order(self):
        """With no preferences, ranking should follow dominance + index order."""
        snap = _make_multi_obj_snapshot(n_obs=5)
        learner = PreferenceLearner()
        ranking = learner.rank_with_preferences(snap, [])
        # All utility scores should be equal (uniform)
        scores = list(ranking.utility_scores.values())
        for s in scores:
            assert abs(s - scores[0]) < 1e-9

    def test_with_explicit_pareto_result(self):
        """Passing a pre-computed ParetoResult should work correctly."""
        snap = _make_multi_obj_snapshot(n_obs=6)
        learner = PreferenceLearner()
        pareto_result = MultiObjectiveAnalyzer().analyze(snap)
        prefs = _make_preferences([(0, 1)])
        ranking = learner.rank_with_preferences(
            snap, prefs, pareto_result=pareto_result,
        )
        assert len(ranking.ranked_indices) == len(snap.successful_observations)
        assert set(ranking.ranked_indices) == set(range(len(snap.successful_observations)))

    def test_without_pareto_result_computes_internally(self):
        """When pareto_result is None, it should be computed internally."""
        snap = _make_multi_obj_snapshot(n_obs=6)
        learner = PreferenceLearner()
        prefs = _make_preferences([(0, 1)])

        # Should not raise; internally computes Pareto result
        ranking = learner.rank_with_preferences(snap, prefs, pareto_result=None)
        assert len(ranking.ranked_indices) == len(snap.successful_observations)
        assert ranking.metadata.get("converged") is not None


# ---------------------------------------------------------------------------
# TestAddPreference
# ---------------------------------------------------------------------------


class TestAddPreference:
    """Tests for PreferenceLearner.add_preference."""

    def test_add_returns_new_list(self):
        """add_preference returns a new list, not mutating the original."""
        learner = PreferenceLearner()
        original = _make_preferences([(0, 1)])
        result = learner.add_preference(original, winner_idx=2, loser_idx=3)
        assert len(result) == 2
        assert len(original) == 1  # original unchanged
        assert result[-1].winner_idx == 2
        assert result[-1].loser_idx == 3

    def test_add_negative_index_raises(self):
        """Negative indices should raise ValueError."""
        learner = PreferenceLearner()
        try:
            learner.add_preference([], winner_idx=-1, loser_idx=0)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "non-negative" in str(e).lower() or "-1" in str(e)

        try:
            learner.add_preference([], winner_idx=0, loser_idx=-2)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "non-negative" in str(e).lower() or "-2" in str(e)

    def test_add_self_comparison_raises(self):
        """Adding a preference where winner == loser should raise ValueError."""
        learner = PreferenceLearner()
        try:
            learner.add_preference([], winner_idx=3, loser_idx=3)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "3" in str(e)


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for the preference learning system."""

    def test_single_observation_campaign(self):
        """Campaign with one observation should work without error."""
        specs = _make_specs()
        obs = [
            _make_obs(
                iteration=0,
                params={"x1": 5.0, "x2": 5.0},
                kpis={"accuracy": 0.9, "latency": 2.0},
            ),
        ]
        snap = CampaignSnapshot(
            campaign_id="single-obs-pref",
            parameter_specs=specs,
            observations=obs,
            objective_names=["accuracy", "latency"],
            objective_directions=["maximize", "minimize"],
            current_iteration=1,
        )
        learner = PreferenceLearner()
        # No preferences possible with just one item
        ranking = learner.rank_with_preferences(snap, [])
        assert len(ranking.ranked_indices) == 1
        assert ranking.ranked_indices[0] == 0

    def test_disconnected_comparison_graph(self):
        """Items not compared to anything should keep their prior score."""
        learner = PreferenceLearner()
        # 5 items, but only compare 0 vs 1; items 2, 3, 4 are disconnected
        prefs = _make_preferences([(0, 1)])
        model = learner.fit(prefs, n_items=5)

        # Winner (0) should be highest
        assert model.scores[0] > model.scores[1]

        # Disconnected items (2, 3, 4) should all have the same score
        # (they keep their prior, which is the same for all)
        assert abs(model.scores[2] - model.scores[3]) < 1e-9
        assert abs(model.scores[3] - model.scores[4]) < 1e-9

        # Disconnected items should have a score between winner and loser
        # (or at least a reasonable prior value)
        assert model.scores[2] >= 0.0
        assert model.scores[2] <= 1.0
