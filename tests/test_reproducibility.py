"""Tests for the optimization_copilot.reproducibility package.

Covers:
- EventType enum values
- CampaignEvent creation, to_dict/from_dict round-trip
- CampaignLogger.log creates events with correct type
- CampaignLogger.get_events returns all events
- CampaignLogger.get_events_by_type filters correctly
- CampaignLogger.export_jsonl / from_jsonl round-trip via tmp file
- CampaignReplayer.replay with identical events -> 100% score
- CampaignReplayer.replay with different events -> <100% score
- ReplayResult fields
- FAIRGenerator.from_snapshot creates metadata
- FAIRMetadata.to_json produces valid JSON
- FAIRMetadata.to_jsonld includes @context
- FAIRMetadata fields
"""

from __future__ import annotations

import json
import os
import tempfile

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.reproducibility import (
    CampaignEvent,
    CampaignLogger,
    CampaignReplayer,
    EventType,
    FAIRGenerator,
    FAIRMetadata,
    ReplayResult,
)


# ---------------------------------------------------------------------------
# EventType tests
# ---------------------------------------------------------------------------


class TestEventType:
    def test_enum_values(self):
        assert EventType.INIT.value == "init"
        assert EventType.SUGGEST.value == "suggest"
        assert EventType.OBSERVE.value == "observe"
        assert EventType.DECISION.value == "decision"
        assert EventType.SWITCH.value == "switch"
        assert EventType.DIAGNOSTIC.value == "diagnostic"
        assert EventType.ERROR.value == "error"
        assert EventType.COMPLETE.value == "complete"

    def test_enum_count(self):
        assert len(EventType) == 8


# ---------------------------------------------------------------------------
# CampaignEvent tests
# ---------------------------------------------------------------------------


class TestCampaignEvent:
    def test_to_dict_from_dict_round_trip(self):
        event = CampaignEvent(
            event_id="abc-123",
            event_type=EventType.SUGGEST,
            timestamp=1000.0,
            iteration=5,
            data={"x": 1.0, "y": 2.0},
            random_seed=42,
        )
        d = event.to_dict()
        restored = CampaignEvent.from_dict(d)

        assert restored.event_id == event.event_id
        assert restored.event_type == event.event_type
        assert restored.timestamp == event.timestamp
        assert restored.iteration == event.iteration
        assert restored.data == event.data
        assert restored.random_seed == event.random_seed

    def test_to_dict_event_type_is_string(self):
        event = CampaignEvent(
            event_id="x",
            event_type=EventType.ERROR,
            timestamp=0.0,
            iteration=0,
        )
        d = event.to_dict()
        assert isinstance(d["event_type"], str)
        assert d["event_type"] == "error"


# ---------------------------------------------------------------------------
# CampaignLogger tests
# ---------------------------------------------------------------------------


class TestCampaignLogger:
    def test_log_creates_event_with_correct_type(self):
        logger = CampaignLogger(campaign_id="test-campaign")
        event = logger.log(EventType.INIT, data={"msg": "start"}, seed=42)

        assert isinstance(event, CampaignEvent)
        assert event.event_type == EventType.INIT
        assert event.data == {"msg": "start"}
        assert event.random_seed == 42

    def test_log_generates_unique_event_ids(self):
        logger = CampaignLogger()
        e1 = logger.log(EventType.SUGGEST)
        e2 = logger.log(EventType.SUGGEST)
        assert e1.event_id != e2.event_id

    def test_get_events_returns_all_events(self):
        logger = CampaignLogger()
        logger.log(EventType.INIT)
        logger.log(EventType.SUGGEST)
        logger.log(EventType.OBSERVE)

        events = logger.get_events()
        assert len(events) == 3

    def test_get_events_by_type_filters_correctly(self):
        logger = CampaignLogger()
        logger.log(EventType.INIT)
        logger.log(EventType.SUGGEST, iteration=1)
        logger.log(EventType.SUGGEST, iteration=2)
        logger.log(EventType.OBSERVE, iteration=1)

        suggest_events = logger.get_events_by_type(EventType.SUGGEST)
        assert len(suggest_events) == 2
        assert all(e.event_type == EventType.SUGGEST for e in suggest_events)

        init_events = logger.get_events_by_type(EventType.INIT)
        assert len(init_events) == 1

        error_events = logger.get_events_by_type(EventType.ERROR)
        assert len(error_events) == 0

    def test_export_jsonl_from_jsonl_round_trip(self):
        logger = CampaignLogger(campaign_id="round-trip")
        logger.log(EventType.INIT, data={"step": 0}, seed=1)
        logger.log(EventType.SUGGEST, data={"x": 0.5}, seed=2, iteration=1)
        logger.log(EventType.OBSERVE, data={"y": 1.2}, iteration=1)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            tmp_path = f.name

        try:
            logger.export_jsonl(tmp_path)

            # Verify file is valid JSONL
            with open(tmp_path, "r") as fh:
                lines = [line.strip() for line in fh if line.strip()]
            assert len(lines) == 3
            for line in lines:
                parsed = json.loads(line)
                assert "event_id" in parsed
                assert "event_type" in parsed

            # Round-trip: reconstruct from JSONL
            restored = CampaignLogger.from_jsonl(tmp_path)
            original_events = logger.get_events()
            restored_events = restored.get_events()

            assert len(restored_events) == len(original_events)
            for orig, rest in zip(original_events, restored_events):
                assert orig.event_id == rest.event_id
                assert orig.event_type == rest.event_type
                assert orig.iteration == rest.iteration
                assert orig.data == rest.data
                assert orig.random_seed == rest.random_seed
        finally:
            os.unlink(tmp_path)

    def test_len_and_iter(self):
        logger = CampaignLogger()
        logger.log(EventType.INIT)
        logger.log(EventType.COMPLETE)

        assert len(logger) == 2
        types = [e.event_type for e in logger]
        assert types == [EventType.INIT, EventType.COMPLETE]


# ---------------------------------------------------------------------------
# CampaignReplayer tests
# ---------------------------------------------------------------------------


class TestCampaignReplayer:
    def _make_events(self, n: int = 3) -> list[CampaignEvent]:
        logger = CampaignLogger()
        logger.log(EventType.INIT, data={"step": 0}, seed=1)
        logger.log(EventType.SUGGEST, data={"x": 0.5}, seed=2, iteration=1)
        logger.log(EventType.OBSERVE, data={"y": 1.2}, iteration=1)
        return logger.get_events()[:n]

    def test_replay_identical_events_returns_100_percent(self):
        events = self._make_events()
        replayer = CampaignReplayer()
        result = replayer.replay(events)

        assert result.reproducibility_score == 1.0
        assert result.matches == len(events)
        assert result.mismatches == 0
        assert len(result.mismatch_details) == 0

    def test_replay_different_events_returns_less_than_100(self):
        events = self._make_events()
        replayer = CampaignReplayer()

        def mutating_replay(event: CampaignEvent) -> CampaignEvent:
            return CampaignEvent(
                event_id=event.event_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                iteration=event.iteration,
                data={"mutated": True},  # different data
                random_seed=event.random_seed,
            )

        result = replayer.replay(events, replay_fn=mutating_replay)
        assert result.reproducibility_score < 1.0
        assert result.mismatches > 0
        assert len(result.mismatch_details) > 0

    def test_replay_result_fields(self):
        events = self._make_events(2)
        replayer = CampaignReplayer()
        result = replayer.replay(events)

        assert isinstance(result, ReplayResult)
        assert len(result.original_events) == 2
        assert len(result.replayed_events) == 2
        assert isinstance(result.matches, int)
        assert isinstance(result.mismatches, int)
        assert isinstance(result.reproducibility_score, float)
        assert isinstance(result.mismatch_details, list)

    def test_replay_empty_events_returns_score_1(self):
        replayer = CampaignReplayer()
        result = replayer.replay([])
        assert result.reproducibility_score == 1.0
        assert result.matches == 0
        assert result.mismatches == 0

    def test_compare_events_static_method(self):
        e1 = CampaignEvent(
            event_id="a", event_type=EventType.SUGGEST,
            timestamp=1.0, iteration=1, data={"x": 1}
        )
        e2 = CampaignEvent(
            event_id="b", event_type=EventType.SUGGEST,
            timestamp=2.0, iteration=1, data={"x": 1}
        )
        e3 = CampaignEvent(
            event_id="c", event_type=EventType.OBSERVE,
            timestamp=1.0, iteration=1, data={"x": 1}
        )

        # Same type, iteration, data -> True (event_id and timestamp ignored)
        assert CampaignReplayer.compare_events(e1, e2) is True
        # Different type -> False
        assert CampaignReplayer.compare_events(e1, e3) is False


# ---------------------------------------------------------------------------
# FAIRMetadata tests
# ---------------------------------------------------------------------------


class TestFAIRMetadata:
    def test_fields(self):
        meta = FAIRMetadata(
            identifier="uuid-123",
            title="Test Campaign",
            creators=["Alice", "Bob"],
            description="A test dataset.",
            keywords=["optimization"],
        )
        assert meta.identifier == "uuid-123"
        assert meta.title == "Test Campaign"
        assert meta.creators == ["Alice", "Bob"]
        assert meta.description == "A test dataset."
        assert meta.keywords == ["optimization"]
        assert meta.license == "CC-BY-4.0"
        assert meta.version == "1.0"
        assert meta.format == "application/json"
        assert meta.access_rights == "open"

    def test_to_json_produces_valid_json(self):
        meta = FAIRMetadata(
            identifier="id-1",
            title="Title",
            creators=["Creator"],
            description="Desc",
        )
        json_str = meta.to_json()
        parsed = json.loads(json_str)
        assert parsed["identifier"] == "id-1"
        assert parsed["title"] == "Title"
        assert parsed["creators"] == ["Creator"]

    def test_to_jsonld_includes_context(self):
        meta = FAIRMetadata(
            identifier="id-2",
            title="LD Title",
            creators=["Person"],
            description="LD Desc",
        )
        jsonld_str = meta.to_jsonld()
        parsed = json.loads(jsonld_str)
        assert "@context" in parsed
        assert parsed["@context"] == "https://schema.org"
        assert parsed["@type"] == "Dataset"
        assert parsed["name"] == "LD Title"
        assert len(parsed["creator"]) == 1
        assert parsed["creator"][0]["@type"] == "Person"

    def test_to_dict(self):
        meta = FAIRMetadata(
            identifier="id-3",
            title="Dict Title",
            creators=["C"],
            description="D",
        )
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert d["identifier"] == "id-3"
        assert "keywords" in d


# ---------------------------------------------------------------------------
# FAIRGenerator tests
# ---------------------------------------------------------------------------


class TestFAIRGenerator:
    def _make_snapshot(self) -> CampaignSnapshot:
        return CampaignSnapshot(
            campaign_id="camp-1",
            parameter_specs=[
                ParameterSpec(name="temp", type=VariableType.CONTINUOUS, lower=100, upper=500),
                ParameterSpec(name="pressure", type=VariableType.CONTINUOUS, lower=1, upper=10),
            ],
            observations=[
                Observation(iteration=1, parameters={"temp": 200, "pressure": 5}, kpi_values={"yield": 0.8}),
                Observation(iteration=2, parameters={"temp": 300, "pressure": 7}, kpi_values={"yield": 0.9}),
            ],
            objective_names=["yield"],
            objective_directions=["maximize"],
            metadata={"description": "Catalysis optimization campaign"},
        )

    def test_from_snapshot_creates_metadata(self):
        snapshot = self._make_snapshot()
        meta = FAIRGenerator.from_snapshot(
            snapshot=snapshot,
            title="Catalysis Campaign",
            creators=["Lab Team"],
        )
        assert isinstance(meta, FAIRMetadata)
        assert meta.title == "Catalysis Campaign"
        assert meta.creators == ["Lab Team"]
        assert "Catalysis optimization campaign" in meta.description
        assert meta.identifier  # non-empty UUID
        assert meta.created  # non-empty ISO timestamp
        assert meta.modified  # non-empty ISO timestamp

    def test_from_snapshot_keywords_include_params_and_objectives(self):
        snapshot = self._make_snapshot()
        meta = FAIRGenerator.from_snapshot(
            snapshot=snapshot,
            title="Test",
            creators=["A"],
        )
        assert "temp" in meta.keywords
        assert "pressure" in meta.keywords
        assert "yield" in meta.keywords
        assert "optimization" in meta.keywords
        assert "camp-1" in meta.keywords

    def test_generate_explicit_params(self):
        meta = FAIRGenerator.generate(
            campaign_id="explicit-1",
            title="Explicit Campaign",
            creators=["Engineer"],
            description="Manual description",
            n_observations=50,
            keywords=["custom-kw"],
        )
        assert isinstance(meta, FAIRMetadata)
        assert meta.title == "Explicit Campaign"
        assert meta.description == "Manual description"
        assert "custom-kw" in meta.keywords
        assert "explicit-1" in meta.keywords
