"""Tests for the Multi-Objective Preference Protocol (P2-5).

Covers:
- ObjectivePreferenceConfig serialization (to_dict / from_dict roundtrip)
- EpsilonConstraint (lower, upper, both)
- PreferenceProtocol.compute_scalarized_score across methods & directions
- filter_by_epsilon_constraints edge cases
- rank_observations ordering, failure skipping, epsilon filtering, explanations
- apply_to_snapshot metadata storage
- KEY ACCEPTANCE TEST: different preference configs produce different rankings
- objective_subset changes rankings
- scalarization method changes rankings
"""

from __future__ import annotations

import math

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.preference.protocol import (
    EpsilonConstraint,
    ObjectivePreferenceConfig,
    PreferenceProtocol,
)


# ── Helpers ───────────────────────────────────────────────────


def _specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
    ]


def _obs(kpis: dict[str, float], *, fail: bool = False, iteration: int = 0) -> Observation:
    return Observation(
        iteration=iteration,
        parameters={"x1": 1.0},
        kpi_values=kpis,
        is_failure=fail,
    )


def _snap(
    observations: list[Observation],
    obj_names: list[str] | None = None,
    obj_dirs: list[str] | None = None,
) -> CampaignSnapshot:
    obj_names = obj_names or ["y"]
    obj_dirs = obj_dirs or ["maximize"]
    return CampaignSnapshot(
        campaign_id="pref-test",
        parameter_specs=_specs(),
        observations=observations,
        objective_names=obj_names,
        objective_directions=obj_dirs,
        current_iteration=len(observations),
    )


def _multi_snap(
    observations: list[Observation],
) -> CampaignSnapshot:
    """3-objective snapshot: maximize y1, minimize y2, maximize y3."""
    return CampaignSnapshot(
        campaign_id="multi-test",
        parameter_specs=_specs(),
        observations=observations,
        objective_names=["y1", "y2", "y3"],
        objective_directions=["maximize", "minimize", "maximize"],
        current_iteration=len(observations),
    )


# ══════════════════════════════════════════════════════════════
# ObjectivePreferenceConfig serialization
# ══════════════════════════════════════════════════════════════


class TestConfigSerialization:

    def test_to_dict_default(self):
        cfg = ObjectivePreferenceConfig()
        d = cfg.to_dict()
        assert d["weights"] == {}
        assert d["epsilon_constraints"] == []
        assert d["objective_subset"] == []
        assert d["scalarization_method"] == "weighted_sum"

    def test_to_dict_with_values(self):
        cfg = ObjectivePreferenceConfig(
            weights={"a": 0.7, "b": 0.3},
            epsilon_constraints=[
                EpsilonConstraint(objective="a", lower_bound=1.0),
            ],
            objective_subset=["a", "b"],
            scalarization_method="tchebycheff",
        )
        d = cfg.to_dict()
        assert d["weights"] == {"a": 0.7, "b": 0.3}
        assert len(d["epsilon_constraints"]) == 1
        assert d["epsilon_constraints"][0]["objective"] == "a"
        assert d["epsilon_constraints"][0]["lower_bound"] == 1.0
        assert d["epsilon_constraints"][0]["upper_bound"] is None
        assert d["objective_subset"] == ["a", "b"]
        assert d["scalarization_method"] == "tchebycheff"

    def test_roundtrip(self):
        cfg = ObjectivePreferenceConfig(
            weights={"x": 2.5, "y": 0.5},
            epsilon_constraints=[
                EpsilonConstraint(objective="x", lower_bound=0.0, upper_bound=10.0),
                EpsilonConstraint(objective="y", upper_bound=5.0),
            ],
            objective_subset=["x"],
            scalarization_method="achievement",
        )
        d = cfg.to_dict()
        restored = ObjectivePreferenceConfig.from_dict(d)
        assert restored.weights == cfg.weights
        assert len(restored.epsilon_constraints) == 2
        assert restored.epsilon_constraints[0].objective == "x"
        assert restored.epsilon_constraints[0].lower_bound == 0.0
        assert restored.epsilon_constraints[0].upper_bound == 10.0
        assert restored.epsilon_constraints[1].objective == "y"
        assert restored.epsilon_constraints[1].lower_bound is None
        assert restored.epsilon_constraints[1].upper_bound == 5.0
        assert restored.objective_subset == ["x"]
        assert restored.scalarization_method == "achievement"

    def test_from_dict_defaults(self):
        cfg = ObjectivePreferenceConfig.from_dict({})
        assert cfg.weights == {}
        assert cfg.epsilon_constraints == []
        assert cfg.objective_subset == []
        assert cfg.scalarization_method == "weighted_sum"

    def test_roundtrip_empty(self):
        cfg = ObjectivePreferenceConfig()
        restored = ObjectivePreferenceConfig.from_dict(cfg.to_dict())
        assert restored.to_dict() == cfg.to_dict()


# ══════════════════════════════════════════════════════════════
# EpsilonConstraint
# ══════════════════════════════════════════════════════════════


class TestEpsilonConstraint:

    def test_lower_bound_only(self):
        ec = EpsilonConstraint(objective="y", lower_bound=2.0)
        assert ec.lower_bound == 2.0
        assert ec.upper_bound is None

    def test_upper_bound_only(self):
        ec = EpsilonConstraint(objective="y", upper_bound=8.0)
        assert ec.lower_bound is None
        assert ec.upper_bound == 8.0

    def test_both_bounds(self):
        ec = EpsilonConstraint(objective="y", lower_bound=1.0, upper_bound=9.0)
        assert ec.lower_bound == 1.0
        assert ec.upper_bound == 9.0


# ══════════════════════════════════════════════════════════════
# compute_scalarized_score
# ══════════════════════════════════════════════════════════════


class TestComputeScalarizedScore:

    def test_weighted_sum_equal_weights(self):
        """Equal weights (default) with a single maximize objective."""
        cfg = ObjectivePreferenceConfig()
        proto = PreferenceProtocol(cfg)
        obs = _obs({"y": 5.0})
        score = proto.compute_scalarized_score(obs, ["y"], ["maximize"])
        assert math.isclose(score, 5.0)

    def test_weighted_sum_custom_weights_single(self):
        cfg = ObjectivePreferenceConfig(weights={"y": 2.0})
        proto = PreferenceProtocol(cfg)
        obs = _obs({"y": 5.0})
        # weight normalised to 1.0, so score = 1.0 * 5.0 = 5.0
        score = proto.compute_scalarized_score(obs, ["y"], ["maximize"])
        assert math.isclose(score, 5.0)

    def test_weighted_sum_two_objectives(self):
        cfg = ObjectivePreferenceConfig(weights={"a": 3.0, "b": 1.0})
        proto = PreferenceProtocol(cfg)
        obs = _obs({"a": 10.0, "b": 2.0})
        # Normalised weights: a=0.75, b=0.25
        # Score = 0.75*10 + 0.25*2 = 7.5 + 0.5 = 8.0
        score = proto.compute_scalarized_score(obs, ["a", "b"], ["maximize", "maximize"])
        assert math.isclose(score, 8.0)

    def test_weighted_sum_minimize_direction(self):
        """Minimize objective is negated so higher == better."""
        cfg = ObjectivePreferenceConfig()
        proto = PreferenceProtocol(cfg)
        obs = _obs({"cost": 3.0})
        score = proto.compute_scalarized_score(obs, ["cost"], ["minimize"])
        assert math.isclose(score, -3.0)

    def test_weighted_sum_mixed_directions(self):
        cfg = ObjectivePreferenceConfig(weights={"profit": 1.0, "cost": 1.0})
        proto = PreferenceProtocol(cfg)
        obs = _obs({"profit": 10.0, "cost": 4.0})
        # profit (max): 10, cost (min): -4
        # weights: 0.5, 0.5 => 0.5*10 + 0.5*(-4) = 5 - 2 = 3
        score = proto.compute_scalarized_score(
            obs, ["profit", "cost"], ["maximize", "minimize"]
        )
        assert math.isclose(score, 3.0)

    def test_tchebycheff(self):
        cfg = ObjectivePreferenceConfig(
            weights={"a": 1.0, "b": 1.0},
            scalarization_method="tchebycheff",
        )
        proto = PreferenceProtocol(cfg)
        obs = _obs({"a": 10.0, "b": 2.0})
        # weights: 0.5, 0.5 => min(0.5*10, 0.5*2) = min(5, 1) = 1.0
        score = proto.compute_scalarized_score(obs, ["a", "b"], ["maximize", "maximize"])
        assert math.isclose(score, 1.0)

    def test_tchebycheff_minimize(self):
        cfg = ObjectivePreferenceConfig(
            weights={"a": 1.0, "b": 1.0},
            scalarization_method="tchebycheff",
        )
        proto = PreferenceProtocol(cfg)
        obs = _obs({"a": 10.0, "b": 3.0})
        # a(max):10, b(min):-3  => min(0.5*10, 0.5*(-3)) = -1.5
        score = proto.compute_scalarized_score(obs, ["a", "b"], ["maximize", "minimize"])
        assert math.isclose(score, -1.5)

    def test_achievement(self):
        cfg = ObjectivePreferenceConfig(
            weights={"a": 1.0, "b": 1.0},
            scalarization_method="achievement",
        )
        proto = PreferenceProtocol(cfg)
        obs = _obs({"a": 10.0, "b": 2.0})
        # weights: 0.5, 0.5 => -max(0.5*10, 0.5*2) = -max(5, 1) = -5
        score = proto.compute_scalarized_score(obs, ["a", "b"], ["maximize", "maximize"])
        assert math.isclose(score, -5.0)

    def test_achievement_negative_values(self):
        cfg = ObjectivePreferenceConfig(
            weights={"a": 1.0},
            scalarization_method="achievement",
        )
        proto = PreferenceProtocol(cfg)
        obs = _obs({"a": -3.0})
        # normalised (max): -3, weight 1 => -max(1*|-3|) = -3
        score = proto.compute_scalarized_score(obs, ["a"], ["maximize"])
        assert math.isclose(score, -3.0)

    def test_missing_kpi_uses_zero(self):
        cfg = ObjectivePreferenceConfig()
        proto = PreferenceProtocol(cfg)
        obs = _obs({})
        score = proto.compute_scalarized_score(obs, ["y"], ["maximize"])
        assert math.isclose(score, 0.0)

    def test_unknown_method_falls_back_to_weighted_sum(self):
        cfg = ObjectivePreferenceConfig(scalarization_method="unknown_method")
        proto = PreferenceProtocol(cfg)
        obs = _obs({"y": 7.0})
        score = proto.compute_scalarized_score(obs, ["y"], ["maximize"])
        assert math.isclose(score, 7.0)

    def test_three_objectives_weighted(self):
        cfg = ObjectivePreferenceConfig(weights={"y1": 2.0, "y2": 1.0, "y3": 1.0})
        proto = PreferenceProtocol(cfg)
        obs = _obs({"y1": 8.0, "y2": 2.0, "y3": 4.0})
        # y1(max):8, y2(min):-2, y3(max):4
        # normalised weights: 0.5, 0.25, 0.25
        # score = 0.5*8 + 0.25*(-2) + 0.25*4 = 4 - 0.5 + 1 = 4.5
        score = proto.compute_scalarized_score(
            obs,
            ["y1", "y2", "y3"],
            ["maximize", "minimize", "maximize"],
        )
        assert math.isclose(score, 4.5)


# ══════════════════════════════════════════════════════════════
# filter_by_epsilon_constraints
# ══════════════════════════════════════════════════════════════


class TestFilterByEpsilonConstraints:

    def test_no_constraints_returns_all(self):
        cfg = ObjectivePreferenceConfig()
        proto = PreferenceProtocol(cfg)
        obs_list = [_obs({"y": v}) for v in [1, 2, 3]]
        result = proto.filter_by_epsilon_constraints(obs_list)
        assert len(result) == 3

    def test_lower_bound_only(self):
        cfg = ObjectivePreferenceConfig(
            epsilon_constraints=[EpsilonConstraint(objective="y", lower_bound=2.0)]
        )
        proto = PreferenceProtocol(cfg)
        obs_list = [_obs({"y": v}) for v in [1.0, 2.0, 3.0]]
        result = proto.filter_by_epsilon_constraints(obs_list)
        assert len(result) == 2
        assert all(o.kpi_values["y"] >= 2.0 for o in result)

    def test_upper_bound_only(self):
        cfg = ObjectivePreferenceConfig(
            epsilon_constraints=[EpsilonConstraint(objective="y", upper_bound=2.0)]
        )
        proto = PreferenceProtocol(cfg)
        obs_list = [_obs({"y": v}) for v in [1.0, 2.0, 3.0]]
        result = proto.filter_by_epsilon_constraints(obs_list)
        assert len(result) == 2
        assert all(o.kpi_values["y"] <= 2.0 for o in result)

    def test_both_bounds(self):
        cfg = ObjectivePreferenceConfig(
            epsilon_constraints=[
                EpsilonConstraint(objective="y", lower_bound=2.0, upper_bound=4.0)
            ]
        )
        proto = PreferenceProtocol(cfg)
        obs_list = [_obs({"y": v}) for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
        result = proto.filter_by_epsilon_constraints(obs_list)
        assert len(result) == 3
        vals = sorted(o.kpi_values["y"] for o in result)
        assert vals == [2.0, 3.0, 4.0]

    def test_multiple_constraints_all_must_pass(self):
        cfg = ObjectivePreferenceConfig(
            epsilon_constraints=[
                EpsilonConstraint(objective="a", lower_bound=2.0),
                EpsilonConstraint(objective="b", upper_bound=5.0),
            ]
        )
        proto = PreferenceProtocol(cfg)
        obs_list = [
            _obs({"a": 3.0, "b": 4.0}),  # passes both
            _obs({"a": 1.0, "b": 4.0}),  # fails a >= 2
            _obs({"a": 3.0, "b": 6.0}),  # fails b <= 5
            _obs({"a": 1.0, "b": 6.0}),  # fails both
        ]
        result = proto.filter_by_epsilon_constraints(obs_list)
        assert len(result) == 1
        assert result[0].kpi_values["a"] == 3.0
        assert result[0].kpi_values["b"] == 4.0

    def test_missing_objective_excluded(self):
        cfg = ObjectivePreferenceConfig(
            epsilon_constraints=[EpsilonConstraint(objective="y", lower_bound=0.0)]
        )
        proto = PreferenceProtocol(cfg)
        obs_list = [
            _obs({"y": 1.0}),
            _obs({"z": 1.0}),  # missing y
        ]
        result = proto.filter_by_epsilon_constraints(obs_list)
        assert len(result) == 1
        assert "y" in result[0].kpi_values

    def test_empty_observations(self):
        cfg = ObjectivePreferenceConfig(
            epsilon_constraints=[EpsilonConstraint(objective="y", lower_bound=0.0)]
        )
        proto = PreferenceProtocol(cfg)
        result = proto.filter_by_epsilon_constraints([])
        assert result == []

    def test_all_filtered_out(self):
        cfg = ObjectivePreferenceConfig(
            epsilon_constraints=[EpsilonConstraint(objective="y", lower_bound=100.0)]
        )
        proto = PreferenceProtocol(cfg)
        obs_list = [_obs({"y": v}) for v in [1, 2, 3]]
        result = proto.filter_by_epsilon_constraints(obs_list)
        assert result == []

    def test_exact_boundary_values_included(self):
        cfg = ObjectivePreferenceConfig(
            epsilon_constraints=[
                EpsilonConstraint(objective="y", lower_bound=2.0, upper_bound=2.0)
            ]
        )
        proto = PreferenceProtocol(cfg)
        obs_list = [_obs({"y": 2.0}), _obs({"y": 2.0001}), _obs({"y": 1.9999})]
        result = proto.filter_by_epsilon_constraints(obs_list)
        assert len(result) == 1
        assert result[0].kpi_values["y"] == 2.0


# ══════════════════════════════════════════════════════════════
# rank_observations
# ══════════════════════════════════════════════════════════════


class TestRankObservations:

    def test_ranks_by_score_descending(self):
        cfg = ObjectivePreferenceConfig()
        proto = PreferenceProtocol(cfg)
        snap = _snap(
            [_obs({"y": 1.0}), _obs({"y": 3.0}), _obs({"y": 2.0})],
        )
        ranked = proto.rank_observations(snap)
        assert len(ranked) == 3
        indices = [r[0] for r in ranked]
        assert indices == [1, 2, 0]

    def test_skips_failures(self):
        cfg = ObjectivePreferenceConfig()
        proto = PreferenceProtocol(cfg)
        snap = _snap(
            [_obs({"y": 10.0}, fail=True), _obs({"y": 1.0}), _obs({"y": 5.0})],
        )
        ranked = proto.rank_observations(snap)
        assert len(ranked) == 2
        indices = [r[0] for r in ranked]
        assert 0 not in indices
        assert indices == [2, 1]

    def test_applies_epsilon_constraints(self):
        cfg = ObjectivePreferenceConfig(
            epsilon_constraints=[EpsilonConstraint(objective="y", lower_bound=3.0)]
        )
        proto = PreferenceProtocol(cfg)
        snap = _snap(
            [_obs({"y": 1.0}), _obs({"y": 5.0}), _obs({"y": 3.0}), _obs({"y": 4.0})],
        )
        ranked = proto.rank_observations(snap)
        assert len(ranked) == 3
        indices = [r[0] for r in ranked]
        assert indices == [1, 3, 2]

    def test_generates_explanations(self):
        cfg = ObjectivePreferenceConfig(weights={"y": 2.0})
        proto = PreferenceProtocol(cfg)
        snap = _snap([_obs({"y": 4.0})])
        ranked = proto.rank_observations(snap)
        assert len(ranked) == 1
        _, score, explanation = ranked[0]
        assert "Score=" in explanation
        assert "weighted_sum" in explanation
        assert "y=4" in explanation
        assert "weight=2.00" in explanation

    def test_empty_snapshot(self):
        cfg = ObjectivePreferenceConfig()
        proto = PreferenceProtocol(cfg)
        snap = _snap([])
        ranked = proto.rank_observations(snap)
        assert ranked == []

    def test_all_failures(self):
        cfg = ObjectivePreferenceConfig()
        proto = PreferenceProtocol(cfg)
        snap = _snap([_obs({"y": 1.0}, fail=True), _obs({"y": 2.0}, fail=True)])
        ranked = proto.rank_observations(snap)
        assert ranked == []

    def test_minimize_ranking_order(self):
        """Lower cost should rank higher when direction=minimize."""
        cfg = ObjectivePreferenceConfig()
        proto = PreferenceProtocol(cfg)
        snap = _snap(
            [_obs({"cost": 10.0}), _obs({"cost": 1.0}), _obs({"cost": 5.0})],
            obj_names=["cost"],
            obj_dirs=["minimize"],
        )
        ranked = proto.rank_observations(snap)
        indices = [r[0] for r in ranked]
        # Lower cost -> higher score (negated) -> ranked first
        assert indices == [1, 2, 0]


# ══════════════════════════════════════════════════════════════
# apply_to_snapshot
# ══════════════════════════════════════════════════════════════


class TestApplyToSnapshot:

    def test_stores_config_in_metadata(self):
        cfg = ObjectivePreferenceConfig(
            weights={"y": 0.5},
            scalarization_method="tchebycheff",
        )
        proto = PreferenceProtocol(cfg)
        snap = _snap([_obs({"y": 1.0})])
        proto.apply_to_snapshot(snap)
        assert "preference_config" in snap.metadata
        stored = snap.metadata["preference_config"]
        assert stored["weights"] == {"y": 0.5}
        assert stored["scalarization_method"] == "tchebycheff"

    def test_returns_same_snapshot(self):
        cfg = ObjectivePreferenceConfig()
        proto = PreferenceProtocol(cfg)
        snap = _snap([_obs({"y": 1.0})])
        result = proto.apply_to_snapshot(snap)
        assert result is snap


# ══════════════════════════════════════════════════════════════
# KEY ACCEPTANCE TEST: different preferences -> different rankings
# ══════════════════════════════════════════════════════════════


class TestPreferenceChangesDifferentRankings:

    def _make_diverse_snapshot(self) -> CampaignSnapshot:
        """12 observations with 3 objectives."""
        observations = [
            _obs({"y1": 10.0, "y2": 1.0, "y3": 2.0}, iteration=0),
            _obs({"y1": 9.0, "y2": 2.0, "y3": 3.0}, iteration=1),
            _obs({"y1": 8.0, "y2": 3.0, "y3": 5.0}, iteration=2),
            _obs({"y1": 7.0, "y2": 4.0, "y3": 7.0}, iteration=3),
            _obs({"y1": 6.0, "y2": 5.0, "y3": 8.0}, iteration=4),
            _obs({"y1": 5.0, "y2": 6.0, "y3": 9.0}, iteration=5),
            _obs({"y1": 4.0, "y2": 7.0, "y3": 10.0}, iteration=6),
            _obs({"y1": 3.0, "y2": 3.0, "y3": 6.0}, iteration=7),
            _obs({"y1": 2.0, "y2": 2.0, "y3": 4.0}, iteration=8),
            _obs({"y1": 1.0, "y2": 1.0, "y3": 1.0}, iteration=9),
            _obs({"y1": 6.0, "y2": 6.0, "y3": 6.0}, iteration=10),
            _obs({"y1": 5.0, "y2": 5.0, "y3": 11.0}, iteration=11),
        ]
        return _multi_snap(observations)

    def test_preference_changes_produce_different_rankings(self):
        """Config A (y1-heavy) and Config B (y3-heavy) give different top-3."""
        snap = self._make_diverse_snapshot()

        config_a = ObjectivePreferenceConfig(
            weights={"y1": 0.8, "y2": 0.1, "y3": 0.1},
        )
        config_b = ObjectivePreferenceConfig(
            weights={"y1": 0.1, "y2": 0.1, "y3": 0.8},
        )

        ranked_a = PreferenceProtocol(config_a).rank_observations(snap)
        ranked_b = PreferenceProtocol(config_b).rank_observations(snap)

        top3_a = [r[0] for r in ranked_a[:3]]
        top3_b = [r[0] for r in ranked_b[:3]]

        # The two configs must produce different top-3 orderings
        assert top3_a != top3_b, (
            f"Expected different top-3 rankings but got "
            f"A={top3_a}, B={top3_b}"
        )

    def test_explanations_reference_weights(self):
        """Explanation strings include the configured weight values."""
        snap = self._make_diverse_snapshot()

        config_a = ObjectivePreferenceConfig(
            weights={"y1": 0.8, "y2": 0.1, "y3": 0.1},
        )
        ranked = PreferenceProtocol(config_a).rank_observations(snap)
        _, _, explanation = ranked[0]
        assert "weight=0.80" in explanation
        assert "weight=0.10" in explanation

    def test_config_b_favours_y3(self):
        """Config B should rank observations with high y3 near the top."""
        snap = self._make_diverse_snapshot()

        config_b = ObjectivePreferenceConfig(
            weights={"y1": 0.1, "y2": 0.1, "y3": 0.8},
        )
        ranked_b = PreferenceProtocol(config_b).rank_observations(snap)
        top_idx = ranked_b[0][0]
        top_obs = snap.observations[top_idx]
        # The top observation should have the highest y3 value
        all_y3 = [o.kpi_values["y3"] for o in snap.observations]
        assert top_obs.kpi_values["y3"] == max(all_y3)

    def test_config_a_favours_y1(self):
        """Config A should rank observations with high y1 near the top."""
        snap = self._make_diverse_snapshot()

        config_a = ObjectivePreferenceConfig(
            weights={"y1": 0.8, "y2": 0.1, "y3": 0.1},
        )
        ranked_a = PreferenceProtocol(config_a).rank_observations(snap)
        top_idx = ranked_a[0][0]
        top_obs = snap.observations[top_idx]
        all_y1 = [o.kpi_values["y1"] for o in snap.observations]
        assert top_obs.kpi_values["y1"] == max(all_y1)


# ══════════════════════════════════════════════════════════════
# objective_subset changes rankings
# ══════════════════════════════════════════════════════════════


class TestObjectiveSubset:

    def test_subset_changes_ranking(self):
        """Using a subset of objectives should change the ranking."""
        observations = [
            _obs({"y1": 10.0, "y2": 1.0, "y3": 1.0}),
            _obs({"y1": 1.0, "y2": 1.0, "y3": 10.0}),
            _obs({"y1": 5.0, "y2": 1.0, "y3": 5.0}),
        ]
        snap = _multi_snap(observations)

        cfg_all = ObjectivePreferenceConfig()
        cfg_y3 = ObjectivePreferenceConfig(objective_subset=["y3"])

        ranked_all = PreferenceProtocol(cfg_all).rank_observations(snap)
        ranked_y3 = PreferenceProtocol(cfg_y3).rank_observations(snap)

        top_all = ranked_all[0][0]
        top_y3 = ranked_y3[0][0]
        # When only y3 is considered, obs[1] should be top (y3=10)
        assert top_y3 == 1
        # They should differ
        assert top_all != top_y3 or (
            [r[0] for r in ranked_all] != [r[0] for r in ranked_y3]
        )

    def test_subset_single_objective(self):
        """Subset with a single objective reduces to single-objective ranking."""
        observations = [
            _obs({"y1": 3.0, "y2": 5.0, "y3": 1.0}),
            _obs({"y1": 1.0, "y2": 5.0, "y3": 9.0}),
            _obs({"y1": 2.0, "y2": 5.0, "y3": 5.0}),
        ]
        snap = _multi_snap(observations)

        cfg = ObjectivePreferenceConfig(objective_subset=["y2"])
        ranked = PreferenceProtocol(cfg).rank_observations(snap)
        # All have y2=5 (minimize => -5 for all) so scores are equal
        scores = [r[1] for r in ranked]
        assert all(math.isclose(s, scores[0]) for s in scores)

    def test_subset_two_objectives(self):
        """Subset with two of three objectives."""
        observations = [
            _obs({"y1": 10.0, "y2": 10.0, "y3": 1.0}),
            _obs({"y1": 1.0, "y2": 1.0, "y3": 10.0}),
        ]
        snap = _multi_snap(observations)

        # Only y1 and y3 (both maximize): obs[0] y1=10,y3=1 => sum=11/2=5.5
        # obs[1] y1=1,y3=10 => sum=11/2=5.5
        cfg_y1_y3 = ObjectivePreferenceConfig(objective_subset=["y1", "y3"])
        ranked = PreferenceProtocol(cfg_y1_y3).rank_observations(snap)
        scores = [r[1] for r in ranked]
        assert math.isclose(scores[0], scores[1])


# ══════════════════════════════════════════════════════════════
# scalarization method changes rankings
# ══════════════════════════════════════════════════════════════


class TestScalarizationMethodChangesRankings:

    def test_weighted_sum_vs_tchebycheff(self):
        """Different methods should produce different rankings for imbalanced obs."""
        observations = [
            _obs({"a": 10.0, "b": 0.5}),   # high a, low b
            _obs({"a": 5.0, "b": 5.0}),     # balanced
            _obs({"a": 0.5, "b": 10.0}),    # low a, high b
        ]
        snap = _snap(observations, obj_names=["a", "b"], obj_dirs=["maximize", "maximize"])

        cfg_ws = ObjectivePreferenceConfig(scalarization_method="weighted_sum")
        cfg_tc = ObjectivePreferenceConfig(scalarization_method="tchebycheff")

        ranked_ws = PreferenceProtocol(cfg_ws).rank_observations(snap)
        ranked_tc = PreferenceProtocol(cfg_tc).rank_observations(snap)

        # Weighted sum: (10+0.5)/2=5.25, (5+5)/2=5, (0.5+10)/2=5.25
        # Tchebycheff: min(5.0, 0.25)=0.25, min(2.5,2.5)=2.5, min(0.25,5.0)=0.25
        # Tchebycheff should rank the balanced obs[1] first
        assert ranked_tc[0][0] == 1

    def test_weighted_sum_vs_achievement(self):
        observations = [
            _obs({"a": 10.0, "b": 1.0}),
            _obs({"a": 3.0, "b": 3.0}),
        ]
        snap = _snap(observations, obj_names=["a", "b"], obj_dirs=["maximize", "maximize"])

        cfg_ws = ObjectivePreferenceConfig(scalarization_method="weighted_sum")
        cfg_ac = ObjectivePreferenceConfig(scalarization_method="achievement")

        ranked_ws = PreferenceProtocol(cfg_ws).rank_observations(snap)
        ranked_ac = PreferenceProtocol(cfg_ac).rank_observations(snap)

        # Weighted sum: obs0=(10+1)/2=5.5, obs1=(3+3)/2=3 => obs0 wins
        assert ranked_ws[0][0] == 0
        # Achievement: obs0=-max(5.0,0.5)=-5, obs1=-max(1.5,1.5)=-1.5 => obs1 wins
        assert ranked_ac[0][0] == 1

    def test_all_three_methods_different_scores(self):
        """All three methods give different numeric scores for the same obs."""
        obs = _obs({"a": 8.0, "b": 2.0})
        names = ["a", "b"]
        dirs = ["maximize", "maximize"]

        score_ws = PreferenceProtocol(
            ObjectivePreferenceConfig(scalarization_method="weighted_sum")
        ).compute_scalarized_score(obs, names, dirs)

        score_tc = PreferenceProtocol(
            ObjectivePreferenceConfig(scalarization_method="tchebycheff")
        ).compute_scalarized_score(obs, names, dirs)

        score_ac = PreferenceProtocol(
            ObjectivePreferenceConfig(scalarization_method="achievement")
        ).compute_scalarized_score(obs, names, dirs)

        # ws=5.0, tc=1.0, ac=-4.0 -> all different
        assert not math.isclose(score_ws, score_tc)
        assert not math.isclose(score_ws, score_ac)
        assert not math.isclose(score_tc, score_ac)


# ══════════════════════════════════════════════════════════════
# Edge cases and integration
# ══════════════════════════════════════════════════════════════


class TestEdgeCases:

    def test_single_observation(self):
        cfg = ObjectivePreferenceConfig()
        proto = PreferenceProtocol(cfg)
        snap = _snap([_obs({"y": 42.0})])
        ranked = proto.rank_observations(snap)
        assert len(ranked) == 1
        assert ranked[0][0] == 0
        assert math.isclose(ranked[0][1], 42.0)

    def test_identical_scores_stable_order(self):
        """Observations with identical scores should all appear."""
        cfg = ObjectivePreferenceConfig()
        proto = PreferenceProtocol(cfg)
        snap = _snap([_obs({"y": 5.0}), _obs({"y": 5.0}), _obs({"y": 5.0})])
        ranked = proto.rank_observations(snap)
        assert len(ranked) == 3
        indices = sorted(r[0] for r in ranked)
        assert indices == [0, 1, 2]

    def test_failure_and_epsilon_combined(self):
        """Both failure and epsilon constraint filter independently."""
        cfg = ObjectivePreferenceConfig(
            epsilon_constraints=[EpsilonConstraint(objective="y", lower_bound=3.0)]
        )
        proto = PreferenceProtocol(cfg)
        snap = _snap([
            _obs({"y": 5.0}, fail=True),   # excluded by failure
            _obs({"y": 1.0}),               # excluded by epsilon
            _obs({"y": 4.0}),               # passes both
            _obs({"y": 3.0}, fail=True),    # excluded by failure
        ])
        ranked = proto.rank_observations(snap)
        assert len(ranked) == 1
        assert ranked[0][0] == 2

    def test_large_weight_disparity(self):
        """Extremely skewed weights still produce valid results."""
        cfg = ObjectivePreferenceConfig(
            weights={"a": 1000.0, "b": 0.001},
        )
        proto = PreferenceProtocol(cfg)
        obs = _obs({"a": 5.0, "b": 5.0})
        score = proto.compute_scalarized_score(obs, ["a", "b"], ["maximize", "maximize"])
        # weight_a ~ 1.0, weight_b ~ 0.000001
        # Score should be dominated by 'a'
        assert score > 4.9

    def test_zero_weights(self):
        """If all weights are zero, total_weight is 0 and no division occurs."""
        cfg = ObjectivePreferenceConfig(weights={"y": 0.0})
        proto = PreferenceProtocol(cfg)
        obs = _obs({"y": 5.0})
        # total_weight=0 => weights stay as [0.0] => score=0
        score = proto.compute_scalarized_score(obs, ["y"], ["maximize"])
        assert math.isclose(score, 0.0)
