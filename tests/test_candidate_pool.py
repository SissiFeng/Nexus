"""Tests for the candidate_pool module."""

from __future__ import annotations

import pytest

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.candidate_pool import CandidatePool, PoolCandidate, PoolVersion


# ── Fixtures / helpers ────────────────────────────────────


def _make_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=-1.0, upper=1.0),
    ]


def _make_snapshot(
    n_obs: int = 5,
    n_failures: int = 1,
    campaign_id: str = "test-001",
) -> CampaignSnapshot:
    specs = _make_specs()
    obs: list[Observation] = []
    for i in range(n_obs):
        is_fail = i < n_failures
        obs.append(
            Observation(
                iteration=i,
                parameters={"x1": i * 0.1, "x2": i * -0.1},
                kpi_values={"y": float(i) if not is_fail else 0.0},
                qc_passed=not is_fail,
                is_failure=is_fail,
                failure_reason="test_fail" if is_fail else None,
                timestamp=float(i),
            )
        )
    return CampaignSnapshot(
        campaign_id=campaign_id,
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=n_obs,
    )


def _make_empty_snapshot() -> CampaignSnapshot:
    specs = _make_specs()
    return CampaignSnapshot(
        campaign_id="empty-001",
        parameter_specs=specs,
        observations=[],
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=0,
    )


def _make_candidate_dicts(n: int = 10) -> list[dict]:
    return [{"x1": i * 0.05, "x2": i * -0.05} for i in range(n)]


# ── PoolCandidate tests ──────────────────────────────────


class TestPoolCandidate:
    def test_creation_defaults(self):
        c = PoolCandidate(parameters={"x1": 0.5})
        assert c.parameters == {"x1": 0.5}
        assert c.score == 0.0
        assert c.rank == 0
        assert c.metadata == {}
        assert c.source == "external"

    def test_creation_with_all_fields(self):
        c = PoolCandidate(
            parameters={"x1": 0.5, "x2": -0.3},
            score=1.5,
            rank=3,
            metadata={"origin": "library"},
            source="tabular",
        )
        assert c.score == 1.5
        assert c.rank == 3
        assert c.source == "tabular"
        assert c.metadata["origin"] == "library"

    def test_parameters_are_dict(self):
        c = PoolCandidate(parameters={"a": 1, "b": 2})
        assert isinstance(c.parameters, dict)
        assert len(c.parameters) == 2

    def test_metadata_independence(self):
        """Each candidate should get its own metadata dict."""
        c1 = PoolCandidate(parameters={"x": 1})
        c2 = PoolCandidate(parameters={"x": 2})
        c1.metadata["key"] = "val"
        assert "key" not in c2.metadata


# ── PoolVersion tests ─────────────────────────────────────


class TestPoolVersion:
    def _make_version(self, n: int = 5) -> PoolVersion:
        candidates = [
            PoolCandidate(parameters={"x": i}, score=float(n - i), rank=i + 1)
            for i in range(n)
        ]
        return PoolVersion(
            version=1,
            candidates=candidates,
            scoring_seed=42,
            scoring_snapshot_hash="abc123",
            timestamp=100.0,
        )

    def test_n_candidates(self):
        v = self._make_version(5)
        assert v.n_candidates == 5

    def test_n_candidates_empty(self):
        v = PoolVersion(
            version=1, candidates=[], scoring_seed=42, scoring_snapshot_hash="x"
        )
        assert v.n_candidates == 0

    def test_top_n_returns_best(self):
        v = self._make_version(5)
        top = v.top_n(2)
        assert len(top) == 2
        assert top[0].rank == 1
        assert top[1].rank == 2

    def test_top_n_larger_than_pool(self):
        v = self._make_version(3)
        top = v.top_n(10)
        assert len(top) == 3

    def test_top_n_zero(self):
        v = self._make_version(5)
        top = v.top_n(0)
        assert len(top) == 0

    def test_top_n_sorts_by_rank(self):
        """Candidates given in non-sorted order should still return sorted."""
        candidates = [
            PoolCandidate(parameters={"x": 3}, score=1.0, rank=3),
            PoolCandidate(parameters={"x": 1}, score=3.0, rank=1),
            PoolCandidate(parameters={"x": 2}, score=2.0, rank=2),
        ]
        v = PoolVersion(
            version=1, candidates=candidates, scoring_seed=42,
            scoring_snapshot_hash="abc",
        )
        top = v.top_n(2)
        assert top[0].rank == 1
        assert top[1].rank == 2

    def test_to_dict_structure(self):
        v = self._make_version(2)
        d = v.to_dict()
        assert d["version"] == 1
        assert d["n_candidates"] == 2
        assert d["scoring_seed"] == 42
        assert d["scoring_snapshot_hash"] == "abc123"
        assert d["timestamp"] == 100.0
        assert len(d["candidates"]) == 2

    def test_to_dict_candidate_keys(self):
        v = self._make_version(1)
        d = v.to_dict()
        cand = d["candidates"][0]
        assert set(cand.keys()) == {"parameters", "score", "rank", "source", "metadata"}

    def test_to_dict_roundtrip_values(self):
        v = self._make_version(3)
        d = v.to_dict()
        for i, cand_dict in enumerate(d["candidates"]):
            assert cand_dict["rank"] == v.candidates[i].rank
            assert cand_dict["score"] == v.candidates[i].score


# ── CandidatePool.load tests ─────────────────────────────


class TestCandidatePoolLoad:
    def test_load_basic(self):
        pool = CandidatePool()
        pool.load([{"x1": 0.1}, {"x1": 0.2}, {"x1": 0.3}])
        assert pool.n_candidates == 3

    def test_load_empty(self):
        pool = CandidatePool()
        pool.load([])
        assert pool.n_candidates == 0

    def test_load_replaces_previous(self):
        pool = CandidatePool()
        pool.load([{"x1": 0.1}])
        assert pool.n_candidates == 1
        pool.load([{"x1": 0.2}, {"x1": 0.3}])
        assert pool.n_candidates == 2

    def test_load_preserves_parameters(self):
        pool = CandidatePool()
        pool.load([{"x1": 0.5, "x2": -0.3}])
        assert pool._candidates[0].parameters == {"x1": 0.5, "x2": -0.3}

    def test_load_candidates_are_copies(self):
        """Modifying the original dict should not affect loaded candidates."""
        original = {"x1": 0.5}
        pool = CandidatePool()
        pool.load([original])
        original["x1"] = 999.0
        assert pool._candidates[0].parameters["x1"] == 0.5


# ── CandidatePool.load_from_rows tests ───────────────────


class TestCandidatePoolLoadFromRows:
    def test_separates_params_and_metadata(self):
        rows = [
            {"x1": 0.5, "x2": -0.3, "name": "mol_A", "weight": 100},
            {"x1": 0.7, "x2": 0.1, "name": "mol_B", "weight": 200},
        ]
        pool = CandidatePool()
        pool.load_from_rows(rows, param_names=["x1", "x2"])
        assert pool.n_candidates == 2
        assert pool._candidates[0].parameters == {"x1": 0.5, "x2": -0.3}
        assert pool._candidates[0].metadata == {"name": "mol_A", "weight": 100}

    def test_source_is_tabular(self):
        pool = CandidatePool()
        pool.load_from_rows([{"x1": 0.5, "extra": "val"}], param_names=["x1"])
        assert pool._candidates[0].source == "tabular"

    def test_missing_param_skipped(self):
        """If a param_name is not in the row, it should be skipped."""
        pool = CandidatePool()
        pool.load_from_rows([{"x1": 0.5}], param_names=["x1", "x2"])
        assert pool._candidates[0].parameters == {"x1": 0.5}

    def test_empty_rows(self):
        pool = CandidatePool()
        pool.load_from_rows([], param_names=["x1"])
        assert pool.n_candidates == 0


# ── CandidatePool.score tests ────────────────────────────


class TestCandidatePoolScore:
    def test_returns_pool_version(self):
        pool = CandidatePool()
        pool.load(_make_candidate_dicts(5))
        snapshot = _make_snapshot()
        specs = _make_specs()
        result = pool.score(snapshot, specs)
        assert isinstance(result, PoolVersion)

    def test_ranks_candidates(self):
        pool = CandidatePool()
        pool.load(_make_candidate_dicts(5))
        snapshot = _make_snapshot()
        specs = _make_specs()
        result = pool.score(snapshot, specs)
        ranks = [c.rank for c in result.candidates]
        assert sorted(ranks) == list(range(1, 6))

    def test_version_counter_increments(self):
        pool = CandidatePool()
        pool.load(_make_candidate_dicts(3))
        snapshot = _make_snapshot()
        specs = _make_specs()
        v1 = pool.score(snapshot, specs)
        v2 = pool.score(snapshot, specs)
        assert v1.version == 1
        assert v2.version == 2

    def test_snapshot_hash_is_deterministic(self):
        pool = CandidatePool()
        pool.load(_make_candidate_dicts(3))
        snapshot = _make_snapshot()
        specs = _make_specs()
        v1 = pool.score(snapshot, specs, seed=42)
        v2 = pool.score(snapshot, specs, seed=42)
        assert v1.scoring_snapshot_hash == v2.scoring_snapshot_hash

    def test_raises_on_empty_pool(self):
        pool = CandidatePool()
        snapshot = _make_snapshot()
        specs = _make_specs()
        with pytest.raises(ValueError, match="No candidates loaded"):
            pool.score(snapshot, specs)

    def test_scores_are_float(self):
        pool = CandidatePool()
        pool.load(_make_candidate_dicts(3))
        snapshot = _make_snapshot()
        specs = _make_specs()
        result = pool.score(snapshot, specs)
        for c in result.candidates:
            assert isinstance(c.score, float)

    def test_candidates_have_parameters(self):
        pool = CandidatePool()
        pool.load(_make_candidate_dicts(3))
        snapshot = _make_snapshot()
        specs = _make_specs()
        result = pool.score(snapshot, specs)
        for c in result.candidates:
            assert "x1" in c.parameters
            assert "x2" in c.parameters

    def test_n_candidates_matches_pool(self):
        pool = CandidatePool()
        pool.load(_make_candidate_dicts(7))
        snapshot = _make_snapshot()
        specs = _make_specs()
        result = pool.score(snapshot, specs)
        assert result.n_candidates == 7


# ── CandidatePool.suggest tests ──────────────────────────


class TestCandidatePoolSuggest:
    def test_returns_top_n_dicts(self):
        pool = CandidatePool()
        pool.load(_make_candidate_dicts(10))
        snapshot = _make_snapshot()
        specs = _make_specs()
        pool.score(snapshot, specs)
        suggestions = pool.suggest(3)
        assert len(suggestions) == 3
        assert all(isinstance(s, dict) for s in suggestions)

    def test_raises_without_scoring(self):
        pool = CandidatePool()
        pool.load(_make_candidate_dicts(5))
        with pytest.raises(ValueError, match="No scored versions"):
            pool.suggest(3)

    def test_suggest_returns_parameters_only(self):
        pool = CandidatePool()
        pool.load(_make_candidate_dicts(5))
        snapshot = _make_snapshot()
        specs = _make_specs()
        pool.score(snapshot, specs)
        suggestions = pool.suggest(2)
        for s in suggestions:
            assert "x1" in s
            assert "x2" in s
            # Should be plain dicts, not PoolCandidate fields
            assert "score" not in s
            assert "rank" not in s


# ── CandidatePool properties tests ───────────────────────


class TestCandidatePoolProperties:
    def test_versions_initially_empty(self):
        pool = CandidatePool()
        assert pool.versions == []

    def test_latest_version_none_initially(self):
        pool = CandidatePool()
        assert pool.latest_version is None

    def test_latest_version_after_scoring(self):
        pool = CandidatePool()
        pool.load(_make_candidate_dicts(3))
        snapshot = _make_snapshot()
        specs = _make_specs()
        v = pool.score(snapshot, specs)
        assert pool.latest_version is v

    def test_versions_list_is_copy(self):
        """Modifying returned versions list should not affect pool."""
        pool = CandidatePool()
        pool.load(_make_candidate_dicts(3))
        snapshot = _make_snapshot()
        specs = _make_specs()
        pool.score(snapshot, specs)
        versions = pool.versions
        versions.clear()
        assert len(pool.versions) == 1


# ── KEY ACCEPTANCE TESTS (determinism) ────────────────────


class TestDeterministicScoring:
    def test_deterministic_scoring_same_snapshot(self):
        """Same pool + same snapshot + same seed -> identical ranks and scores."""
        candidates = _make_candidate_dicts(10)
        snapshot = _make_snapshot()
        specs = _make_specs()

        pool1 = CandidatePool()
        pool1.load(candidates)
        v1 = pool1.score(snapshot, specs, seed=42)

        pool2 = CandidatePool()
        pool2.load(candidates)
        v2 = pool2.score(snapshot, specs, seed=42)

        assert v1.n_candidates == v2.n_candidates
        for c1, c2 in zip(v1.candidates, v2.candidates):
            assert c1.parameters == c2.parameters
            assert c1.score == c2.score
            assert c1.rank == c2.rank

    def test_deterministic_scoring_different_seed(self):
        """Different seed does not change scores (seed only affects hash).

        The scoring algorithm uses snapshot data directly; the seed
        is only part of the version hash, not the scoring formula.
        """
        candidates = _make_candidate_dicts(10)
        snapshot = _make_snapshot()
        specs = _make_specs()

        pool1 = CandidatePool()
        pool1.load(candidates)
        v1 = pool1.score(snapshot, specs, seed=42)

        pool2 = CandidatePool()
        pool2.load(candidates)
        v2 = pool2.score(snapshot, specs, seed=99)

        for c1, c2 in zip(v1.candidates, v2.candidates):
            assert c1.score == c2.score
            assert c1.rank == c2.rank

        # But hashes should differ
        assert v1.scoring_snapshot_hash != v2.scoring_snapshot_hash

    def test_deterministic_scoring_different_snapshot(self):
        """Different snapshot -> different scores."""
        candidates = _make_candidate_dicts(10)
        specs = _make_specs()

        snapshot_a = _make_snapshot(n_obs=3, n_failures=0)
        snapshot_b = _make_snapshot(n_obs=10, n_failures=5)

        pool1 = CandidatePool()
        pool1.load(candidates)
        v1 = pool1.score(snapshot_a, specs, seed=42)

        pool2 = CandidatePool()
        pool2.load(candidates)
        v2 = pool2.score(snapshot_b, specs, seed=42)

        # At least some scores should differ
        scores_a = [c.score for c in v1.candidates]
        scores_b = [c.score for c in v2.candidates]
        assert scores_a != scores_b

    def test_version_tracking(self):
        """Multiple score() calls create separate versions."""
        pool = CandidatePool()
        pool.load(_make_candidate_dicts(5))
        snapshot = _make_snapshot()
        specs = _make_specs()

        v1 = pool.score(snapshot, specs)
        v2 = pool.score(snapshot, specs)
        v3 = pool.score(snapshot, specs)

        assert len(pool.versions) == 3
        assert v1.version == 1
        assert v2.version == 2
        assert v3.version == 3
        assert pool.latest_version is v3

    def test_stable_recommendations(self):
        """suggest() returns same results after repeated scoring with same inputs."""
        candidates = _make_candidate_dicts(10)
        snapshot = _make_snapshot()
        specs = _make_specs()

        pool = CandidatePool()
        pool.load(candidates)
        pool.score(snapshot, specs, seed=42)
        suggestions_1 = pool.suggest(3)

        # Score again with same inputs (creates new version)
        pool.score(snapshot, specs, seed=42)
        suggestions_2 = pool.suggest(3)

        assert suggestions_1 == suggestions_2

    def test_scoring_without_observations(self):
        """Empty snapshot -> hash-based scoring works deterministically."""
        candidates = _make_candidate_dicts(10)
        empty_snapshot = _make_empty_snapshot()
        specs = _make_specs()

        pool1 = CandidatePool()
        pool1.load(candidates)
        v1 = pool1.score(empty_snapshot, specs, seed=42)

        pool2 = CandidatePool()
        pool2.load(candidates)
        v2 = pool2.score(empty_snapshot, specs, seed=42)

        # Should be identical
        for c1, c2 in zip(v1.candidates, v2.candidates):
            assert c1.score == c2.score
            assert c1.rank == c2.rank
            assert c1.parameters == c2.parameters

        # All scores should be in [0, 1]
        for c in v1.candidates:
            assert 0.0 <= c.score <= 1.0


# ── Categorical parameter tests ──────────────────────────


class TestCategoricalParameters:
    def test_categorical_scoring(self):
        """Pool with categorical parameters should score without errors."""
        specs = [
            ParameterSpec(name="color", type=VariableType.CATEGORICAL, categories=["red", "blue", "green"]),
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ]
        candidates = [
            {"color": "red", "x1": 0.1},
            {"color": "blue", "x1": 0.5},
            {"color": "green", "x1": 0.9},
        ]
        snapshot = CampaignSnapshot(
            campaign_id="cat-test",
            parameter_specs=specs,
            observations=[
                Observation(
                    iteration=0,
                    parameters={"color": "red", "x1": 0.3},
                    kpi_values={"y": 1.0},
                )
            ],
            objective_names=["y"],
            objective_directions=["maximize"],
        )
        pool = CandidatePool()
        pool.load(candidates)
        result = pool.score(snapshot, specs)
        assert result.n_candidates == 3
        ranks = sorted(c.rank for c in result.candidates)
        assert ranks == [1, 2, 3]

    def test_missing_parameter_defaults_to_zero(self):
        """Candidate missing a parameter should use 0.0 in the vector."""
        specs = [
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
            ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ]
        pool = CandidatePool()
        pool.load([{"x1": 0.5}])  # missing x2
        empty_snapshot = _make_empty_snapshot()
        result = pool.score(empty_snapshot, specs)
        assert result.n_candidates == 1


# ── Internal helper tests ─────────────────────────────────


class TestInternalHelpers:
    def test_euclidean_distance_same(self):
        assert CandidatePool._euclidean_distance([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_euclidean_distance_known(self):
        dist = CandidatePool._euclidean_distance([0.0, 0.0], [3.0, 4.0])
        assert abs(dist - 5.0) < 1e-10

    def test_euclidean_distance_single_dim(self):
        dist = CandidatePool._euclidean_distance([0.0], [1.0])
        assert abs(dist - 1.0) < 1e-10

    def test_paramdict_to_vector_continuous(self):
        specs = [ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0)]
        vec = CandidatePool._paramdict_to_vector({"x": 5.0}, specs)
        assert abs(vec[0] - 0.5) < 1e-10

    def test_paramdict_to_vector_categorical(self):
        specs = [ParameterSpec(name="c", type=VariableType.CATEGORICAL)]
        vec = CandidatePool._paramdict_to_vector({"c": "blue"}, specs)
        assert 0.0 <= vec[0] <= 1.0

    def test_paramdict_to_vector_missing(self):
        specs = [ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)]
        vec = CandidatePool._paramdict_to_vector({}, specs)
        assert vec[0] == 0.0

    def test_compute_snapshot_hash_deterministic(self):
        snap = _make_snapshot()
        h1 = CandidatePool._compute_snapshot_hash(snap, seed=42)
        h2 = CandidatePool._compute_snapshot_hash(snap, seed=42)
        assert h1 == h2

    def test_compute_snapshot_hash_changes_with_seed(self):
        snap = _make_snapshot()
        h1 = CandidatePool._compute_snapshot_hash(snap, seed=42)
        h2 = CandidatePool._compute_snapshot_hash(snap, seed=99)
        assert h1 != h2
