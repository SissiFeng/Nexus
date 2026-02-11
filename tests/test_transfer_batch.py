"""Comprehensive tests for TransferLearningEngine and BatchScheduler.

Tests cover initialization, core operations, serialization round-trips,
similarity computation, warm-start logic, data pooling, trial lifecycle,
worker management, and edge cases for both modules.
"""

from __future__ import annotations

import pytest

from optimization_copilot.infrastructure.transfer_learning import (
    CampaignData,
    TransferLearningEngine,
)
from optimization_copilot.infrastructure.batch_scheduler import (
    AsyncTrial,
    BatchScheduler,
    TrialStatus,
)


# =========================================================================
# Helpers / Fixtures
# =========================================================================

def _make_specs(*names: str, lower: float = 0.0, upper: float = 1.0):
    """Create continuous parameter specs from names."""
    return [
        {"name": n, "type": "continuous", "lower": lower, "upper": upper}
        for n in names
    ]


def _make_observations(param_names: list[str], n: int = 10):
    """Create observations with linearly increasing objective values."""
    obs = []
    for i in range(n):
        o = {name: i * 0.1 for name in param_names}
        o["objective"] = float(i)
        obs.append(o)
    return obs


# =========================================================================
# TransferLearningEngine -- CampaignData
# =========================================================================


class TestCampaignData:
    """Tests for the CampaignData dataclass."""

    def test_creation_basic(self):
        specs = _make_specs("x", "y")
        obs = [{"x": 0.5, "y": 0.3, "objective": 1.0}]
        cd = CampaignData(
            campaign_id="c1",
            parameter_specs=specs,
            observations=obs,
        )
        assert cd.campaign_id == "c1"
        assert cd.n_observations == 1
        assert cd.param_names == {"x", "y"}

    def test_creation_with_metadata(self):
        cd = CampaignData(
            campaign_id="c2",
            parameter_specs=_make_specs("a"),
            observations=[],
            metadata={"domain": "chemistry"},
        )
        assert cd.metadata == {"domain": "chemistry"}

    def test_default_metadata_is_empty(self):
        cd = CampaignData(
            campaign_id="c3",
            parameter_specs=[],
            observations=[],
        )
        assert cd.metadata == {}

    def test_param_names_property(self):
        specs = _make_specs("alpha", "beta", "gamma")
        cd = CampaignData("c", specs, [])
        assert cd.param_names == {"alpha", "beta", "gamma"}

    def test_to_dict(self):
        specs = _make_specs("x")
        obs = [{"x": 0.1, "objective": 2.0}]
        cd = CampaignData("c1", specs, obs, metadata={"k": "v"})
        d = cd.to_dict()
        assert d["campaign_id"] == "c1"
        assert len(d["parameter_specs"]) == 1
        assert d["observations"] == obs
        assert d["metadata"] == {"k": "v"}

    def test_from_dict(self):
        raw = {
            "campaign_id": "c_from",
            "parameter_specs": [{"name": "z", "type": "continuous", "lower": 0, "upper": 10}],
            "observations": [{"z": 5, "objective": 3.0}],
            "metadata": {"origin": "test"},
        }
        cd = CampaignData.from_dict(raw)
        assert cd.campaign_id == "c_from"
        assert cd.param_names == {"z"}
        assert cd.n_observations == 1
        assert cd.metadata["origin"] == "test"

    def test_to_dict_from_dict_roundtrip(self):
        specs = _make_specs("a", "b")
        obs = _make_observations(["a", "b"], n=5)
        cd = CampaignData("rt", specs, obs, metadata={"round": "trip"})
        restored = CampaignData.from_dict(cd.to_dict())
        assert restored.campaign_id == cd.campaign_id
        assert restored.param_names == cd.param_names
        assert restored.n_observations == cd.n_observations
        assert restored.metadata == cd.metadata

    def test_from_dict_with_missing_optional_fields(self):
        raw = {"campaign_id": "minimal"}
        cd = CampaignData.from_dict(raw)
        assert cd.campaign_id == "minimal"
        assert cd.parameter_specs == []
        assert cd.observations == []
        assert cd.metadata == {}


# =========================================================================
# TransferLearningEngine -- Initialization and Registration
# =========================================================================


class TestTransferLearningEngineInit:
    """Tests for engine initialization and campaign registration."""

    def test_empty_engine(self):
        engine = TransferLearningEngine()
        assert engine.n_campaigns == 0
        assert engine.campaign_ids == []
        assert engine.transfer_log == []

    def test_register_single_campaign(self):
        engine = TransferLearningEngine()
        specs = _make_specs("x")
        obs = [{"x": 0.5, "objective": 1.0}]
        engine.register_campaign("camp1", specs, obs)
        assert engine.n_campaigns == 1
        assert engine.campaign_ids == ["camp1"]

    def test_register_multiple_campaigns(self):
        engine = TransferLearningEngine()
        for i in range(3):
            engine.register_campaign(
                f"camp_{i}",
                _make_specs("x"),
                [{"x": 0.1 * i, "objective": float(i)}],
            )
        assert engine.n_campaigns == 3
        assert engine.campaign_ids == ["camp_0", "camp_1", "camp_2"]

    def test_register_duplicate_replaces(self):
        engine = TransferLearningEngine()
        specs = _make_specs("x")
        engine.register_campaign("dup", specs, [{"x": 0.1, "objective": 1.0}])
        engine.register_campaign("dup", specs, [{"x": 0.9, "objective": 9.0}])
        assert engine.n_campaigns == 1
        # Verify the replacement campaign's data is the latest
        data = engine.to_dict()
        assert data["history"][0]["observations"][0]["x"] == 0.9

    def test_unregister_campaign(self):
        engine = TransferLearningEngine()
        engine.register_campaign("a", _make_specs("x"), [])
        engine.register_campaign("b", _make_specs("x"), [])
        assert engine.unregister_campaign("a") is True
        assert engine.n_campaigns == 1
        assert engine.campaign_ids == ["b"]

    def test_unregister_nonexistent_returns_false(self):
        engine = TransferLearningEngine()
        assert engine.unregister_campaign("nope") is False

    def test_register_with_metadata(self):
        engine = TransferLearningEngine()
        engine.register_campaign(
            "meta",
            _make_specs("x"),
            [],
            metadata={"domain": "biology"},
        )
        assert engine.n_campaigns == 1


# =========================================================================
# TransferLearningEngine -- Similarity
# =========================================================================


class TestTransferLearningEngineSimilarity:
    """Tests for compute_similarity and its components."""

    def test_identical_campaigns_high_similarity(self):
        engine = TransferLearningEngine()
        specs = _make_specs("x", "y")
        obs = _make_observations(["x", "y"], n=5)
        target = CampaignData("target", specs, obs)
        sim = engine.compute_similarity(specs, target)
        # Identical parameter names and ranges -> param_overlap=1.0, range_overlap=1.0
        # metadata similarity = 0.0 (both empty via compute_similarity)
        # Expected: 0.4*1.0 + 0.4*1.0 + 0.2*0.0 = 0.8
        assert sim == pytest.approx(0.8, abs=0.01)

    def test_completely_different_campaigns_low_similarity(self):
        engine = TransferLearningEngine()
        specs_a = _make_specs("x", "y")
        specs_b = _make_specs("a", "b")
        target = CampaignData("t", specs_b, [])
        sim = engine.compute_similarity(specs_a, target)
        # No parameter overlap -> param_overlap=0, range_overlap=0, metadata=0
        assert sim == pytest.approx(0.0, abs=0.01)

    def test_partial_overlap(self):
        engine = TransferLearningEngine()
        specs_a = _make_specs("x", "y", "z")
        specs_b = _make_specs("y", "z", "w")
        target = CampaignData("t", specs_b, [])
        sim = engine.compute_similarity(specs_a, target)
        # Jaccard: |{y,z}| / |{x,y,z,w}| = 2/4 = 0.5
        # Range overlap on shared (y,z): identical ranges -> 1.0
        # metadata: 0.0
        # Expected: 0.4*0.5 + 0.4*1.0 + 0.2*0 = 0.6
        assert sim == pytest.approx(0.6, abs=0.01)

    def test_similarity_with_different_ranges(self):
        engine = TransferLearningEngine()
        specs_a = [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 10.0}]
        specs_b = [{"name": "x", "type": "continuous", "lower": 5.0, "upper": 15.0}]
        target = CampaignData("t", specs_b, [])
        sim = engine.compute_similarity(specs_a, target)
        # param_overlap = 1.0 (Jaccard: 1/1)
        # range_overlap: intersection [5,10]=5, union [0,15]=15, ratio=5/15=1/3
        # metadata = 0.0
        # Expected: 0.4*1.0 + 0.4*(1/3) + 0.2*0 ~ 0.533
        assert sim == pytest.approx(0.533, abs=0.01)

    def test_similarity_with_metadata(self):
        engine = TransferLearningEngine()
        specs = _make_specs("x")
        target = CampaignData("t", specs, [], metadata={"domain": "chem", "lab": "A"})
        sim = engine.compute_similarity_with_meta(
            specs, target, current_metadata={"domain": "chem", "lab": "B"}
        )
        # param_overlap = 1.0, range_overlap = 1.0
        # metadata: shared keys = {domain, lab}, union = {domain, lab}
        # matching values: domain matches, lab doesn't -> 1/2 = 0.5
        # Expected: 0.4*1.0 + 0.4*1.0 + 0.2*0.5 = 0.9
        assert sim == pytest.approx(0.9, abs=0.01)

    def test_similarity_with_categorical_params(self):
        engine = TransferLearningEngine()
        specs_a = [{"name": "color", "type": "categorical", "categories": ["red", "green", "blue"]}]
        specs_b = [{"name": "color", "type": "categorical", "categories": ["red", "green", "yellow"]}]
        target = CampaignData("t", specs_b, [])
        sim = engine.compute_similarity(specs_a, target)
        # param_overlap = 1.0
        # range_overlap: Jaccard on categories {red,green} / {red,green,blue,yellow} = 2/4 = 0.5
        # metadata = 0.0
        # Expected: 0.4*1.0 + 0.4*0.5 + 0.2*0 = 0.6
        assert sim == pytest.approx(0.6, abs=0.01)

    def test_empty_specs_both_sides(self):
        engine = TransferLearningEngine()
        target = CampaignData("t", [], [])
        sim = engine.compute_similarity([], target)
        # param_overlap = 0.0 (both empty), range_overlap = 0.0, metadata = 0.0
        assert sim == pytest.approx(0.0, abs=0.01)


# =========================================================================
# TransferLearningEngine -- Warm Start
# =========================================================================


class TestTransferLearningEngineWarmStart:
    """Tests for warm_start_points."""

    def test_warm_start_returns_top_observations(self):
        engine = TransferLearningEngine()
        specs = _make_specs("x")
        obs = [{"x": float(i) * 0.1, "objective": float(i)} for i in range(10)]
        engine.register_campaign("source", specs, obs)

        points = engine.warm_start_points(specs, n_points=5, min_similarity=0.5)
        # Top 20% of 10 = 2 observations (obj 9.0 and 8.0)
        assert len(points) <= 5
        assert len(points) >= 1
        # Should include the best objective
        objectives = [p["objective"] for p in points]
        assert max(objectives) == 9.0

    def test_warm_start_empty_when_no_similar(self):
        engine = TransferLearningEngine()
        specs_a = _make_specs("x", "y")
        specs_b = _make_specs("a", "b")
        engine.register_campaign("source", specs_b, _make_observations(["a", "b"]))

        points = engine.warm_start_points(specs_a, n_points=5, min_similarity=0.5)
        assert points == []

    def test_warm_start_respects_n_points(self):
        engine = TransferLearningEngine()
        specs = _make_specs("x")
        obs = [{"x": float(i) * 0.1, "objective": float(i)} for i in range(100)]
        engine.register_campaign("big", specs, obs)

        points = engine.warm_start_points(specs, n_points=3, min_similarity=0.5)
        assert len(points) <= 3

    def test_warm_start_logs_transfer_event(self):
        engine = TransferLearningEngine()
        specs = _make_specs("x")
        engine.register_campaign("s", specs, [{"x": 0.5, "objective": 1.0}])
        engine.warm_start_points(specs, n_points=5, min_similarity=0.5)
        log = engine.transfer_log
        assert len(log) == 1
        assert log[0]["event"] == "warm_start"

    def test_warm_start_with_no_campaigns(self):
        engine = TransferLearningEngine()
        points = engine.warm_start_points(_make_specs("x"), n_points=5)
        assert points == []

    def test_warm_start_clips_to_target_bounds(self):
        engine = TransferLearningEngine()
        source_specs = [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 20.0}]
        target_specs = [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 5.0}]
        obs = [{"x": 15.0, "objective": 100.0}]
        engine.register_campaign("wide", source_specs, obs)

        points = engine.warm_start_points(target_specs, n_points=5, min_similarity=0.0)
        if points:
            # The value 15.0 should be clipped to 5.0
            assert points[0]["x"] <= 5.0


# =========================================================================
# TransferLearningEngine -- Data Pooling
# =========================================================================


class TestTransferLearningEngineDataPooling:
    """Tests for transfer_data."""

    def test_transfer_data_adds_weight(self):
        engine = TransferLearningEngine()
        specs = _make_specs("x")
        obs = [{"x": 0.5, "objective": 1.0}]
        engine.register_campaign("src", specs, obs)

        pooled = engine.transfer_data(specs, min_similarity=0.5)
        assert len(pooled) >= 1
        for item in pooled:
            assert "_transfer_weight" in item
            assert 0.0 < item["_transfer_weight"] <= 1.0

    def test_transfer_data_empty_when_below_threshold(self):
        engine = TransferLearningEngine()
        specs_a = _make_specs("x", "y")
        specs_b = _make_specs("a", "b")
        engine.register_campaign("src", specs_b, _make_observations(["a", "b"]))

        pooled = engine.transfer_data(specs_a, min_similarity=0.5)
        assert pooled == []

    def test_transfer_data_requires_compatible_params(self):
        engine = TransferLearningEngine()
        # Source has x,y; current needs x,y,z -> source not compatible (missing z)
        source_specs = _make_specs("x", "y")
        current_specs = _make_specs("x", "y", "z")
        engine.register_campaign("partial", source_specs, _make_observations(["x", "y"]))

        pooled = engine.transfer_data(current_specs, min_similarity=0.0)
        assert pooled == []

    def test_transfer_data_logs_event(self):
        engine = TransferLearningEngine()
        specs = _make_specs("x")
        engine.register_campaign("s", specs, [{"x": 0.5, "objective": 1.0}])
        engine.transfer_data(specs, min_similarity=0.5)
        log = engine.transfer_log
        assert any(e["event"] == "data_pool" for e in log)


# =========================================================================
# TransferLearningEngine -- find_similar_campaigns
# =========================================================================


class TestTransferLearningEngineFindSimilar:
    """Tests for find_similar_campaigns."""

    def test_find_similar_returns_sorted(self):
        engine = TransferLearningEngine()
        specs_current = _make_specs("x", "y", "z")
        # Campaign with high overlap
        engine.register_campaign("high", _make_specs("x", "y", "z"), [])
        # Campaign with low overlap
        engine.register_campaign("low", _make_specs("x", "a", "b"), [])

        results = engine.find_similar_campaigns(specs_current, min_similarity=0.0)
        assert len(results) >= 2
        # First result should have higher similarity
        assert results[0][1] >= results[1][1]
        assert results[0][0] == "high"

    def test_find_similar_filters_by_threshold(self):
        engine = TransferLearningEngine()
        engine.register_campaign("only", _make_specs("a", "b"), [])
        # No overlap with "x","y"
        results = engine.find_similar_campaigns(_make_specs("x", "y"), min_similarity=0.5)
        assert len(results) == 0

    def test_find_similar_empty_history(self):
        engine = TransferLearningEngine()
        results = engine.find_similar_campaigns(_make_specs("x"), min_similarity=0.0)
        assert results == []


# =========================================================================
# TransferLearningEngine -- Serialization
# =========================================================================


class TestTransferLearningEngineSerialization:
    """Tests for engine to_dict / from_dict round-trip."""

    def test_empty_engine_roundtrip(self):
        engine = TransferLearningEngine()
        restored = TransferLearningEngine.from_dict(engine.to_dict())
        assert restored.n_campaigns == 0
        assert restored.transfer_log == []

    def test_engine_roundtrip_with_campaigns(self):
        engine = TransferLearningEngine()
        specs = _make_specs("x", "y")
        obs = _make_observations(["x", "y"], n=3)
        engine.register_campaign("c1", specs, obs, metadata={"k": "v"})
        engine.register_campaign("c2", specs, obs)

        restored = TransferLearningEngine.from_dict(engine.to_dict())
        assert restored.n_campaigns == 2
        assert restored.campaign_ids == ["c1", "c2"]

    def test_engine_roundtrip_preserves_transfer_log(self):
        engine = TransferLearningEngine()
        specs = _make_specs("x")
        engine.register_campaign("s", specs, [{"x": 0.5, "objective": 1.0}])
        engine.warm_start_points(specs, n_points=5, min_similarity=0.5)

        restored = TransferLearningEngine.from_dict(engine.to_dict())
        assert len(restored.transfer_log) == 1
        assert restored.transfer_log[0]["event"] == "warm_start"

    def test_repr(self):
        engine = TransferLearningEngine()
        r = repr(engine)
        assert "TransferLearningEngine" in r
        assert "n_campaigns=0" in r


# =========================================================================
# TransferLearningEngine -- Edge Cases
# =========================================================================


class TestTransferLearningEngineEdgeCases:
    """Edge case tests for the transfer learning engine."""

    def test_campaign_with_no_observations(self):
        engine = TransferLearningEngine()
        engine.register_campaign("empty", _make_specs("x"), [])
        points = engine.warm_start_points(_make_specs("x"), min_similarity=0.0)
        assert points == []

    def test_campaign_with_single_observation(self):
        engine = TransferLearningEngine()
        specs = _make_specs("x")
        engine.register_campaign("one", specs, [{"x": 0.5, "objective": 42.0}])
        points = engine.warm_start_points(specs, n_points=10, min_similarity=0.5)
        # Top 20% of 1 observation = at least 1
        assert len(points) >= 1
        assert points[0]["objective"] == 42.0


# =========================================================================
# BatchScheduler -- TrialStatus
# =========================================================================


class TestTrialStatus:
    """Tests for TrialStatus enum."""

    def test_enum_values(self):
        assert TrialStatus.PENDING.value == "pending"
        assert TrialStatus.RUNNING.value == "running"
        assert TrialStatus.COMPLETED.value == "completed"
        assert TrialStatus.FAILED.value == "failed"

    def test_enum_from_value(self):
        assert TrialStatus("pending") is TrialStatus.PENDING
        assert TrialStatus("failed") is TrialStatus.FAILED


# =========================================================================
# BatchScheduler -- AsyncTrial
# =========================================================================


class TestAsyncTrial:
    """Tests for the AsyncTrial dataclass."""

    def test_creation_defaults(self):
        trial = AsyncTrial(trial_id="t1", params={"x": 0.5})
        assert trial.trial_id == "t1"
        assert trial.params == {"x": 0.5}
        assert trial.status == TrialStatus.PENDING
        assert trial.worker_id is None
        assert trial.completed_at is None
        assert trial.result is None
        assert trial.error == ""

    def test_is_active_when_pending(self):
        trial = AsyncTrial(trial_id="t", params={})
        assert trial.is_active is True
        assert trial.is_terminal is False

    def test_is_active_when_running(self):
        trial = AsyncTrial(trial_id="t", params={}, status=TrialStatus.RUNNING)
        assert trial.is_active is True

    def test_is_terminal_when_completed(self):
        trial = AsyncTrial(trial_id="t", params={}, status=TrialStatus.COMPLETED)
        assert trial.is_terminal is True
        assert trial.is_active is False

    def test_is_terminal_when_failed(self):
        trial = AsyncTrial(trial_id="t", params={}, status=TrialStatus.FAILED)
        assert trial.is_terminal is True

    def test_duration_none_when_not_completed(self):
        trial = AsyncTrial(trial_id="t", params={})
        assert trial.duration is None

    def test_duration_computed_when_completed(self):
        trial = AsyncTrial(
            trial_id="t", params={}, submitted_at=100.0, completed_at=105.5
        )
        assert trial.duration == pytest.approx(5.5)

    def test_to_dict(self):
        trial = AsyncTrial(
            trial_id="t1",
            params={"x": 1},
            status=TrialStatus.RUNNING,
            worker_id="w1",
        )
        d = trial.to_dict()
        assert d["trial_id"] == "t1"
        assert d["status"] == "running"
        assert d["worker_id"] == "w1"

    def test_from_dict(self):
        raw = {
            "trial_id": "t2",
            "params": {"y": 2},
            "status": "completed",
            "worker_id": "w2",
            "submitted_at": 10.0,
            "completed_at": 15.0,
            "result": {"objective": 3.0},
            "error": "",
        }
        trial = AsyncTrial.from_dict(raw)
        assert trial.trial_id == "t2"
        assert trial.status == TrialStatus.COMPLETED
        assert trial.result == {"objective": 3.0}

    def test_to_dict_from_dict_roundtrip(self):
        trial = AsyncTrial(
            trial_id="rt",
            params={"a": 1, "b": 2},
            status=TrialStatus.FAILED,
            worker_id="w3",
            submitted_at=50.0,
            completed_at=55.0,
            error="timeout",
        )
        restored = AsyncTrial.from_dict(trial.to_dict())
        assert restored.trial_id == trial.trial_id
        assert restored.params == trial.params
        assert restored.status == trial.status
        assert restored.worker_id == trial.worker_id
        assert restored.error == trial.error


# =========================================================================
# BatchScheduler -- Initialization
# =========================================================================


class TestBatchSchedulerInit:
    """Tests for BatchScheduler initialization."""

    def test_default_init(self):
        bs = BatchScheduler()
        assert bs.n_workers == 1
        assert bs.batch_strategy == "simple"
        assert bs.total_trials == 0

    def test_custom_init(self):
        bs = BatchScheduler(n_workers=4, batch_strategy="greedy")
        assert bs.n_workers == 4
        assert bs.batch_strategy == "greedy"

    def test_round_robin_strategy(self):
        bs = BatchScheduler(n_workers=2, batch_strategy="round_robin")
        assert bs.batch_strategy == "round_robin"

    def test_invalid_n_workers_raises(self):
        with pytest.raises(ValueError, match="n_workers must be >= 1"):
            BatchScheduler(n_workers=0)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="batch_strategy must be one of"):
            BatchScheduler(batch_strategy="unknown")


# =========================================================================
# BatchScheduler -- request_batch
# =========================================================================


class TestBatchSchedulerRequestBatch:
    """Tests for request_batch."""

    def test_request_batch_creates_trials(self):
        bs = BatchScheduler(n_workers=4)
        suggestions = [{"x": 0.1}, {"x": 0.2}, {"x": 0.3}]
        trials = bs.request_batch(suggestions)
        assert len(trials) == 3
        assert bs.total_trials == 3

    def test_request_batch_trials_are_pending(self):
        bs = BatchScheduler()
        trials = bs.request_batch([{"x": 1}])
        assert all(t.status == TrialStatus.PENDING for t in trials)

    def test_request_batch_trial_ids_unique(self):
        bs = BatchScheduler(n_workers=4)
        trials = bs.request_batch([{"x": i} for i in range(10)])
        ids = [t.trial_id for t in trials]
        assert len(ids) == len(set(ids))

    def test_request_batch_empty_suggestions(self):
        bs = BatchScheduler()
        trials = bs.request_batch([])
        assert trials == []
        assert bs.total_trials == 0


# =========================================================================
# BatchScheduler -- Trial Lifecycle
# =========================================================================


class TestBatchSchedulerLifecycle:
    """Tests for submit_trial, complete_trial, fail_trial."""

    def test_submit_trial(self):
        bs = BatchScheduler(n_workers=2)
        trials = bs.request_batch([{"x": 0.5}])
        tid = trials[0].trial_id
        bs.submit_trial(tid, "worker_1")
        trial = bs.get_trial(tid)
        assert trial.status == TrialStatus.RUNNING
        assert trial.worker_id == "worker_1"

    def test_complete_trial(self):
        bs = BatchScheduler(n_workers=2)
        trials = bs.request_batch([{"x": 0.5}])
        tid = trials[0].trial_id
        bs.submit_trial(tid, "w1")
        bs.complete_trial(tid, {"objective": 42.0})
        trial = bs.get_trial(tid)
        assert trial.status == TrialStatus.COMPLETED
        assert trial.result == {"objective": 42.0}
        assert trial.completed_at is not None

    def test_complete_pending_trial_directly(self):
        """complete_trial should work on PENDING trials too (per source code)."""
        bs = BatchScheduler()
        trials = bs.request_batch([{"x": 1}])
        tid = trials[0].trial_id
        bs.complete_trial(tid, {"objective": 5.0})
        trial = bs.get_trial(tid)
        assert trial.status == TrialStatus.COMPLETED

    def test_fail_trial(self):
        bs = BatchScheduler(n_workers=2)
        trials = bs.request_batch([{"x": 0.5}])
        tid = trials[0].trial_id
        bs.submit_trial(tid, "w1")
        bs.fail_trial(tid, "crashed")
        trial = bs.get_trial(tid)
        assert trial.status == TrialStatus.FAILED
        assert trial.error == "crashed"
        assert trial.completed_at is not None

    def test_fail_pending_trial(self):
        """fail_trial should work on PENDING trials too."""
        bs = BatchScheduler()
        trials = bs.request_batch([{"x": 1}])
        tid = trials[0].trial_id
        bs.fail_trial(tid, "cancelled")
        trial = bs.get_trial(tid)
        assert trial.status == TrialStatus.FAILED

    def test_submit_nonexistent_trial_raises(self):
        bs = BatchScheduler()
        with pytest.raises(KeyError):
            bs.submit_trial("nonexistent", "w1")

    def test_complete_nonexistent_trial_raises(self):
        bs = BatchScheduler()
        with pytest.raises(KeyError):
            bs.complete_trial("nonexistent", {"objective": 1.0})

    def test_fail_nonexistent_trial_raises(self):
        bs = BatchScheduler()
        with pytest.raises(KeyError):
            bs.fail_trial("nonexistent", "err")

    def test_submit_running_trial_raises(self):
        """Cannot submit a trial that is already RUNNING."""
        bs = BatchScheduler()
        trials = bs.request_batch([{"x": 1}])
        tid = trials[0].trial_id
        bs.submit_trial(tid, "w1")
        with pytest.raises(ValueError, match="expected pending"):
            bs.submit_trial(tid, "w2")

    def test_complete_already_completed_raises(self):
        """After completion the trial moves to completed list; completing again raises KeyError."""
        bs = BatchScheduler()
        trials = bs.request_batch([{"x": 1}])
        tid = trials[0].trial_id
        bs.complete_trial(tid, {"objective": 1.0})
        # Trial is now in completed list, not active -> KeyError
        with pytest.raises(KeyError):
            bs.complete_trial(tid, {"objective": 2.0})

    def test_fail_already_failed_raises(self):
        bs = BatchScheduler()
        trials = bs.request_batch([{"x": 1}])
        tid = trials[0].trial_id
        bs.fail_trial(tid, "err")
        with pytest.raises(KeyError):
            bs.fail_trial(tid, "err2")


# =========================================================================
# BatchScheduler -- Query Methods
# =========================================================================


class TestBatchSchedulerQueries:
    """Tests for get_pending_results and related query methods."""

    def test_get_pending_results_only_completed(self):
        bs = BatchScheduler(n_workers=4)
        trials = bs.request_batch([{"x": 1}, {"x": 2}, {"x": 3}])
        bs.submit_trial(trials[0].trial_id, "w1")
        bs.complete_trial(trials[0].trial_id, {"objective": 10.0})
        bs.submit_trial(trials[1].trial_id, "w2")
        bs.fail_trial(trials[1].trial_id, "err")
        # trials[2] still pending

        results = bs.get_pending_results()
        assert len(results) == 1
        assert results[0]["objective"] == 10.0
        assert results[0]["x"] == 1
        assert "_trial_id" in results[0]

    def test_get_trial_finds_active(self):
        bs = BatchScheduler()
        trials = bs.request_batch([{"x": 1}])
        found = bs.get_trial(trials[0].trial_id)
        assert found is not None
        assert found.trial_id == trials[0].trial_id

    def test_get_trial_finds_completed(self):
        bs = BatchScheduler()
        trials = bs.request_batch([{"x": 1}])
        tid = trials[0].trial_id
        bs.complete_trial(tid, {"objective": 1})
        found = bs.get_trial(tid)
        assert found is not None
        assert found.status == TrialStatus.COMPLETED

    def test_get_trial_returns_none_for_unknown(self):
        bs = BatchScheduler()
        assert bs.get_trial("unknown") is None


# =========================================================================
# BatchScheduler -- Worker Management
# =========================================================================


class TestBatchSchedulerWorkers:
    """Tests for worker management methods."""

    def test_count_idle_workers_all_idle(self):
        bs = BatchScheduler(n_workers=3)
        assert bs.count_idle_workers() == 3

    def test_count_idle_workers_some_running(self):
        bs = BatchScheduler(n_workers=3)
        trials = bs.request_batch([{"x": 1}, {"x": 2}])
        bs.submit_trial(trials[0].trial_id, "w1")
        assert bs.count_idle_workers() == 2

    def test_needs_backfill_true(self):
        bs = BatchScheduler(n_workers=2)
        # No active trials, 2 idle workers
        assert bs.needs_backfill() is True

    def test_needs_backfill_false_when_pending_exist(self):
        bs = BatchScheduler(n_workers=2)
        bs.request_batch([{"x": 1}])  # Creates a PENDING trial
        assert bs.needs_backfill() is False

    def test_needs_backfill_false_when_fully_utilized(self):
        bs = BatchScheduler(n_workers=1)
        trials = bs.request_batch([{"x": 1}])
        bs.submit_trial(trials[0].trial_id, "w1")
        assert bs.needs_backfill() is False

    def test_backfill_count(self):
        bs = BatchScheduler(n_workers=4)
        trials = bs.request_batch([{"x": 1}])
        bs.submit_trial(trials[0].trial_id, "w1")
        # 1 running, 3 idle, 0 pending -> need 3
        assert bs.backfill_count() == 3

    def test_active_worker_ids(self):
        bs = BatchScheduler(n_workers=3)
        trials = bs.request_batch([{"x": 1}, {"x": 2}])
        bs.submit_trial(trials[0].trial_id, "alpha")
        bs.submit_trial(trials[1].trial_id, "beta")
        ids = bs.active_worker_ids()
        assert set(ids) == {"alpha", "beta"}


# =========================================================================
# BatchScheduler -- Summary
# =========================================================================


class TestBatchSchedulerSummary:
    """Tests for the summary method."""

    def test_summary_empty(self):
        bs = BatchScheduler(n_workers=2, batch_strategy="greedy")
        s = bs.summary()
        assert s["n_active"] == 0
        assert s["n_pending"] == 0
        assert s["n_running"] == 0
        assert s["n_completed"] == 0
        assert s["n_failed"] == 0
        assert s["n_idle_workers"] == 2
        assert s["n_workers"] == 2
        assert s["total_trials"] == 0
        assert s["batch_strategy"] == "greedy"

    def test_summary_mixed_state(self):
        bs = BatchScheduler(n_workers=4)
        trials = bs.request_batch([{"x": i} for i in range(4)])
        bs.submit_trial(trials[0].trial_id, "w1")
        bs.submit_trial(trials[1].trial_id, "w2")
        bs.complete_trial(trials[0].trial_id, {"objective": 1.0})
        bs.fail_trial(trials[1].trial_id, "err")

        s = bs.summary()
        assert s["n_pending"] == 2  # trials[2] and trials[3]
        assert s["n_running"] == 0  # both submitted are now terminal
        assert s["n_completed"] == 1
        assert s["n_failed"] == 1
        assert s["total_trials"] == 4
        assert s["n_idle_workers"] == 4  # none running


# =========================================================================
# BatchScheduler -- Serialization
# =========================================================================


class TestBatchSchedulerSerialization:
    """Tests for BatchScheduler to_dict / from_dict round-trip."""

    def test_empty_scheduler_roundtrip(self):
        bs = BatchScheduler(n_workers=3, batch_strategy="round_robin")
        restored = BatchScheduler.from_dict(bs.to_dict())
        assert restored.n_workers == 3
        assert restored.batch_strategy == "round_robin"
        assert restored.total_trials == 0

    def test_scheduler_roundtrip_with_trials(self):
        bs = BatchScheduler(n_workers=2)
        trials = bs.request_batch([{"x": 1}, {"x": 2}])
        bs.submit_trial(trials[0].trial_id, "w1")
        bs.complete_trial(trials[0].trial_id, {"objective": 5.0})

        data = bs.to_dict()
        restored = BatchScheduler.from_dict(data)
        assert restored.total_trials == 2
        # 1 active (pending), 1 completed
        assert len(restored.get_active_trials()) == 1
        assert len(restored.get_completed_trials()) == 1

    def test_scheduler_roundtrip_preserves_counter(self):
        bs = BatchScheduler()
        bs.request_batch([{"x": 1}, {"x": 2}])
        data = bs.to_dict()
        assert data["trial_counter"] == 2
        restored = BatchScheduler.from_dict(data)
        assert restored._trial_counter == 2

    def test_repr(self):
        bs = BatchScheduler(n_workers=2, batch_strategy="greedy")
        r = repr(bs)
        assert "BatchScheduler" in r
        assert "n_workers=2" in r
        assert "greedy" in r
