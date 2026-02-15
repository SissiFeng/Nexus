"""Comprehensive tests for the optimization_copilot.store module.

Covers Artifact, Experiment, ExperimentStore, StoreBridge, and StoreQuery.
"""

from __future__ import annotations

import json

import pytest

from optimization_copilot.store.models import Artifact, ArtifactType, Experiment
from optimization_copilot.store.store import ExperimentStore, StoreQuery, StoreSummary
from optimization_copilot.store.bridge import StoreBridge
from optimization_copilot.dsl.spec import (
    OptimizationSpec,
    ParameterDef,
    ObjectiveDef,
    BudgetDef,
    ParamType,
    Direction,
)
from optimization_copilot.feature_extraction.extractors import CurveData


# ── Helpers ───────────────────────────────────────────────


def _make_experiment(
    campaign_id: str = "camp1",
    iteration: int = 0,
    experiment_id: str | None = None,
    **kwargs,
) -> Experiment:
    """Create an Experiment with sensible defaults for testing."""
    eid = experiment_id or f"{campaign_id}-{iteration:04d}"
    return Experiment(
        experiment_id=eid,
        campaign_id=campaign_id,
        iteration=iteration,
        parameters=kwargs.get("parameters", {"x": 0.5}),
        kpi_values=kwargs.get("kpi_values", {"y": 1.0}),
        **{k: v for k, v in kwargs.items() if k not in ("parameters", "kpi_values")},
    )


def _make_curve_artifact(
    artifact_id: str = "curve-1",
    name: str = "eis_curve",
    x_values: list[float] | None = None,
    y_values: list[float] | None = None,
    metadata: dict | None = None,
) -> Artifact:
    """Create a CURVE artifact with sensible defaults."""
    return Artifact(
        artifact_id=artifact_id,
        artifact_type=ArtifactType.CURVE,
        name=name,
        data={
            "x_values": x_values or [1.0, 2.0, 3.0],
            "y_values": y_values or [10.0, 20.0, 30.0],
            "metadata": metadata or {"type": "eis"},
        },
    )


def _make_spec(campaign_id: str = "test") -> OptimizationSpec:
    """Create a minimal OptimizationSpec for testing."""
    return OptimizationSpec(
        campaign_id=campaign_id,
        parameters=[
            ParameterDef(name="x", type=ParamType.CONTINUOUS, lower=0.0, upper=1.0),
        ],
        objectives=[
            ObjectiveDef(name="y", direction=Direction.MAXIMIZE),
        ],
        budget=BudgetDef(max_iterations=10),
    )


# ── TestArtifact ──────────────────────────────────────────


class TestArtifact:
    """Tests for the Artifact dataclass."""

    def test_create_curve_artifact(self):
        art = Artifact(
            artifact_id="a1",
            artifact_type=ArtifactType.CURVE,
            name="eis_spectrum",
            data={"x_values": [1.0], "y_values": [2.0], "metadata": {}},
        )
        assert art.artifact_type == ArtifactType.CURVE
        assert art.name == "eis_spectrum"

    def test_create_binary_artifact(self):
        art = Artifact(
            artifact_id="a2",
            artifact_type=ArtifactType.BINARY,
            name="image_data",
            data="base64encodedstring==",
        )
        assert art.artifact_type == ArtifactType.BINARY
        assert art.data == "base64encodedstring=="

    def test_create_spectral_artifact(self):
        art = Artifact(
            artifact_id="a3",
            artifact_type=ArtifactType.SPECTRAL,
            name="uv_vis",
            data={"wavelengths": [400, 500], "intensities": [0.5, 0.8]},
        )
        assert art.artifact_type == ArtifactType.SPECTRAL

    def test_create_metadata_artifact(self):
        art = Artifact(
            artifact_id="a4",
            artifact_type=ArtifactType.METADATA,
            name="run_info",
            data={"operator": "alice", "lab": "B3"},
        )
        assert art.artifact_type == ArtifactType.METADATA

    def test_create_raw_artifact(self):
        art = Artifact(
            artifact_id="a5",
            artifact_type=ArtifactType.RAW,
            name="raw_dump",
            data=[1, 2, 3, 4],
        )
        assert art.artifact_type == ArtifactType.RAW
        assert art.data == [1, 2, 3, 4]

    def test_to_dict(self):
        art = _make_curve_artifact()
        d = art.to_dict()
        assert d["artifact_id"] == "curve-1"
        assert d["artifact_type"] == "curve"
        assert d["name"] == "eis_curve"
        assert "x_values" in d["data"]
        assert isinstance(d["metadata"], dict)

    def test_from_dict(self):
        raw = {
            "artifact_id": "a10",
            "artifact_type": "binary",
            "name": "img",
            "data": "abc123",
            "metadata": {"note": "test"},
        }
        art = Artifact.from_dict(raw)
        assert art.artifact_id == "a10"
        assert art.artifact_type == ArtifactType.BINARY
        assert art.metadata == {"note": "test"}

    def test_round_trip(self):
        original = _make_curve_artifact(metadata={"unit": "ohm"})
        restored = Artifact.from_dict(original.to_dict())
        assert restored.artifact_id == original.artifact_id
        assert restored.artifact_type == original.artifact_type
        assert restored.name == original.name
        assert restored.data == original.data
        assert restored.metadata == original.metadata


# ── TestExperiment ────────────────────────────────────────


class TestExperiment:
    """Tests for the Experiment dataclass."""

    def test_creation_minimal(self):
        exp = _make_experiment()
        assert exp.experiment_id == "camp1-0000"
        assert exp.campaign_id == "camp1"
        assert exp.iteration == 0
        assert exp.parameters == {"x": 0.5}
        assert exp.kpi_values == {"y": 1.0}

    def test_defaults(self):
        exp = _make_experiment()
        assert exp.metadata == {}
        assert exp.artifacts == []
        assert exp.timestamp == 0.0
        assert exp.qc_passed is True
        assert exp.is_failure is False
        assert exp.failure_reason is None

    def test_with_artifacts(self):
        art = _make_curve_artifact()
        exp = _make_experiment(artifacts=[art])
        assert len(exp.artifacts) == 1
        assert exp.artifacts[0].artifact_type == ArtifactType.CURVE

    def test_to_dict(self):
        exp = _make_experiment(metadata={"source": "robot"})
        d = exp.to_dict()
        assert d["experiment_id"] == "camp1-0000"
        assert d["campaign_id"] == "camp1"
        assert d["iteration"] == 0
        assert d["parameters"] == {"x": 0.5}
        assert d["kpi_values"] == {"y": 1.0}
        assert d["metadata"] == {"source": "robot"}
        assert d["artifacts"] == []
        assert d["qc_passed"] is True
        assert d["is_failure"] is False
        assert d["failure_reason"] is None

    def test_from_dict(self):
        raw = {
            "experiment_id": "e1",
            "campaign_id": "c1",
            "iteration": 5,
            "parameters": {"a": 1},
            "kpi_values": {"b": 2.0},
            "metadata": {},
            "artifacts": [],
            "timestamp": 100.0,
            "qc_passed": False,
            "is_failure": True,
            "failure_reason": "exploded",
        }
        exp = Experiment.from_dict(raw)
        assert exp.experiment_id == "e1"
        assert exp.iteration == 5
        assert exp.qc_passed is False
        assert exp.is_failure is True
        assert exp.failure_reason == "exploded"

    def test_round_trip_with_artifacts(self):
        art = _make_curve_artifact()
        original = _make_experiment(artifacts=[art], metadata={"key": "val"})
        restored = Experiment.from_dict(original.to_dict())
        assert restored.experiment_id == original.experiment_id
        assert len(restored.artifacts) == 1
        assert restored.artifacts[0].artifact_id == art.artifact_id
        assert restored.artifacts[0].data == art.data
        assert restored.metadata == original.metadata

    def test_failure_experiment(self):
        exp = _make_experiment(
            is_failure=True,
            failure_reason="reactor overflow",
            qc_passed=False,
        )
        assert exp.is_failure is True
        assert exp.failure_reason == "reactor overflow"
        assert exp.qc_passed is False

    def test_custom_metadata(self):
        exp = _make_experiment(metadata={"lab": "B5", "operator": "bob", "temp_c": 25})
        assert exp.metadata["lab"] == "B5"
        assert exp.metadata["operator"] == "bob"
        assert exp.metadata["temp_c"] == 25

    def test_qc_fields_independent(self):
        """qc_passed and is_failure are independent flags."""
        exp = _make_experiment(qc_passed=False, is_failure=False)
        assert exp.qc_passed is False
        assert exp.is_failure is False

    def test_from_dict_missing_artifacts_key(self):
        """from_dict should handle missing 'artifacts' key gracefully."""
        raw = {
            "experiment_id": "e2",
            "campaign_id": "c2",
            "iteration": 0,
            "parameters": {"x": 0.1},
            "kpi_values": {"y": 0.9},
            "metadata": {},
            "timestamp": 0.0,
            "qc_passed": True,
            "is_failure": False,
            "failure_reason": None,
        }
        exp = Experiment.from_dict(raw)
        assert exp.artifacts == []


# ── TestExperimentStore ───────────────────────────────────


class TestExperimentStore:
    """Tests for the ExperimentStore class."""

    def test_add_and_get_experiment(self):
        store = ExperimentStore()
        exp = _make_experiment()
        store.add_experiment(exp)
        retrieved = store.get_experiment("camp1-0000")
        assert retrieved.experiment_id == exp.experiment_id
        assert retrieved.parameters == {"x": 0.5}

    def test_add_experiments_batch(self):
        store = ExperimentStore()
        exps = [_make_experiment(iteration=i) for i in range(5)]
        store.add_experiments(exps)
        assert len(store) == 5

    def test_duplicate_rejection(self):
        store = ExperimentStore()
        exp = _make_experiment()
        store.add_experiment(exp)
        with pytest.raises(ValueError, match="Duplicate experiment_id"):
            store.add_experiment(exp)

    def test_get_nonexistent_raises_key_error(self):
        store = ExperimentStore()
        with pytest.raises(KeyError, match="Experiment not found"):
            store.get_experiment("nonexistent")

    def test_attach_artifact(self):
        store = ExperimentStore()
        exp = _make_experiment()
        store.add_experiment(exp)
        art = _make_curve_artifact()
        store.attach_artifact("camp1-0000", art)
        retrieved = store.get_experiment("camp1-0000")
        assert len(retrieved.artifacts) == 1
        assert retrieved.artifacts[0].artifact_id == "curve-1"

    def test_attach_artifact_nonexistent_raises(self):
        store = ExperimentStore()
        art = _make_curve_artifact()
        with pytest.raises(KeyError, match="Experiment not found"):
            store.attach_artifact("no-such-id", art)

    def test_list_campaigns(self):
        store = ExperimentStore()
        store.add_experiment(_make_experiment(campaign_id="beta", iteration=0))
        store.add_experiment(_make_experiment(campaign_id="alpha", iteration=0))
        store.add_experiment(_make_experiment(campaign_id="beta", iteration=1))
        campaigns = store.list_campaigns()
        assert campaigns == ["alpha", "beta"]

    def test_get_by_campaign(self):
        store = ExperimentStore()
        store.add_experiment(_make_experiment(campaign_id="c1", iteration=0))
        store.add_experiment(_make_experiment(campaign_id="c1", iteration=1))
        store.add_experiment(_make_experiment(campaign_id="c2", iteration=0))
        results = store.get_by_campaign("c1")
        assert len(results) == 2
        assert all(r.campaign_id == "c1" for r in results)

    def test_get_by_campaign_sorted_by_iteration(self):
        store = ExperimentStore()
        store.add_experiment(_make_experiment(campaign_id="c1", iteration=3))
        store.add_experiment(_make_experiment(campaign_id="c1", iteration=1))
        store.add_experiment(_make_experiment(campaign_id="c1", iteration=2))
        results = store.get_by_campaign("c1")
        assert [r.iteration for r in results] == [1, 2, 3]

    def test_get_by_iteration(self):
        store = ExperimentStore()
        store.add_experiment(_make_experiment(campaign_id="c1", iteration=0))
        store.add_experiment(_make_experiment(campaign_id="c1", iteration=1))
        store.add_experiment(_make_experiment(campaign_id="c1", iteration=2))
        results = store.get_by_iteration("c1", 1)
        assert len(results) == 1
        assert results[0].iteration == 1

    def test_query_only_successful(self):
        store = ExperimentStore()
        store.add_experiment(_make_experiment(iteration=0, is_failure=False))
        store.add_experiment(_make_experiment(iteration=1, is_failure=True))
        store.add_experiment(_make_experiment(iteration=2, is_failure=False))
        results = store.query(StoreQuery(only_successful=True))
        assert len(results) == 2
        assert all(not r.is_failure for r in results)

    def test_query_has_artifact_type(self):
        store = ExperimentStore()
        exp_with = _make_experiment(iteration=0, artifacts=[_make_curve_artifact()])
        exp_without = _make_experiment(iteration=1)
        store.add_experiments([exp_with, exp_without])
        results = store.query(StoreQuery(has_artifact_type=ArtifactType.CURVE))
        assert len(results) == 1
        assert results[0].iteration == 0

    def test_query_iteration_range(self):
        store = ExperimentStore()
        for i in range(10):
            store.add_experiment(_make_experiment(iteration=i))
        results = store.query(StoreQuery(iteration_range=(3, 6)))
        iterations = [r.iteration for r in results]
        assert iterations == [3, 4, 5, 6]

    def test_query_combined_filters(self):
        store = ExperimentStore()
        store.add_experiment(
            _make_experiment(campaign_id="c1", iteration=0, is_failure=False)
        )
        store.add_experiment(
            _make_experiment(campaign_id="c1", iteration=1, is_failure=True)
        )
        store.add_experiment(
            _make_experiment(campaign_id="c2", iteration=0, is_failure=False)
        )
        results = store.query(
            StoreQuery(campaign_id="c1", only_successful=True)
        )
        assert len(results) == 1
        assert results[0].campaign_id == "c1"
        assert results[0].is_failure is False

    def test_summary_whole_store(self):
        store = ExperimentStore()
        exp1 = _make_experiment(
            campaign_id="c1",
            iteration=0,
            parameters={"x": 0.1, "z": 0.2},
            kpi_values={"y": 1.0, "w": 2.0},
        )
        exp2 = _make_experiment(campaign_id="c2", iteration=3)
        store.add_experiments([exp1, exp2])

        s = store.summary()
        assert s.n_experiments == 2
        assert s.n_campaigns == 2
        assert sorted(s.campaign_ids) == ["c1", "c2"]
        assert s.n_artifacts == 0
        assert "x" in s.parameter_names
        assert "y" in s.kpi_names
        assert s.iteration_range == (0, 3)

    def test_summary_by_campaign(self):
        store = ExperimentStore()
        store.add_experiment(_make_experiment(campaign_id="c1", iteration=0))
        store.add_experiment(_make_experiment(campaign_id="c1", iteration=5))
        store.add_experiment(_make_experiment(campaign_id="c2", iteration=0))

        s = store.summary(campaign_id="c1")
        assert s.n_experiments == 2
        assert s.n_campaigns == 1
        assert s.campaign_ids == ["c1"]
        assert s.iteration_range == (0, 5)

    def test_column_names(self):
        store = ExperimentStore()
        store.add_experiment(
            _make_experiment(
                campaign_id="c1",
                iteration=0,
                parameters={"a": 1, "b": 2},
                kpi_values={"k1": 0.5},
                metadata={"source": "robot"},
            )
        )
        cols = store.column_names("c1")
        assert cols["parameters"] == ["a", "b"]
        assert cols["kpis"] == ["k1"]
        assert cols["metadata"] == ["source"]

    def test_column_values(self):
        store = ExperimentStore()
        store.add_experiment(
            _make_experiment(campaign_id="c1", iteration=0, parameters={"x": 0.1})
        )
        store.add_experiment(
            _make_experiment(campaign_id="c1", iteration=1, parameters={"x": 0.9})
        )
        vals = store.column_values("c1", "x")
        assert vals == [0.1, 0.9]

    def test_column_values_from_kpi(self):
        store = ExperimentStore()
        store.add_experiment(
            _make_experiment(campaign_id="c1", iteration=0, kpi_values={"y": 10.0})
        )
        store.add_experiment(
            _make_experiment(campaign_id="c1", iteration=1, kpi_values={"y": 20.0})
        )
        vals = store.column_values("c1", "y")
        assert vals == [10.0, 20.0]

    def test_serialization_to_dict_from_dict(self):
        store = ExperimentStore()
        exp = _make_experiment(artifacts=[_make_curve_artifact()])
        store.add_experiment(exp)

        d = store.to_dict()
        restored = ExperimentStore.from_dict(d)
        assert len(restored) == 1
        r_exp = restored.get_experiment("camp1-0000")
        assert r_exp.parameters == {"x": 0.5}
        assert len(r_exp.artifacts) == 1

    def test_serialization_to_json_from_json(self):
        store = ExperimentStore()
        store.add_experiment(_make_experiment(iteration=0))
        store.add_experiment(_make_experiment(iteration=1))

        json_str = store.to_json()
        parsed = json.loads(json_str)
        assert "experiments" in parsed
        assert len(parsed["experiments"]) == 2

        restored = ExperimentStore.from_json(json_str)
        assert len(restored) == 2

    def test_len_empty(self):
        store = ExperimentStore()
        assert len(store) == 0

    def test_len_after_additions(self):
        store = ExperimentStore()
        for i in range(7):
            store.add_experiment(_make_experiment(iteration=i))
        assert len(store) == 7


# ── TestStoreBridge ───────────────────────────────────────


class TestStoreBridge:
    """Tests for the StoreBridge static methods."""

    def test_to_observations_basic(self):
        store = ExperimentStore()
        store.add_experiment(_make_experiment(campaign_id="c1", iteration=0))
        store.add_experiment(_make_experiment(campaign_id="c1", iteration=1))

        obs = StoreBridge.to_observations(store, "c1")
        assert len(obs) == 2
        assert obs[0].iteration == 0
        assert obs[1].iteration == 1

    def test_to_observations_strips_artifacts(self):
        """Observations should NOT contain artifact data."""
        store = ExperimentStore()
        exp = _make_experiment(
            campaign_id="c1", iteration=0, artifacts=[_make_curve_artifact()]
        )
        store.add_experiment(exp)

        obs = StoreBridge.to_observations(store, "c1")
        assert len(obs) == 1
        # Observation has no 'artifacts' attribute
        assert not hasattr(obs[0], "artifacts")

    def test_to_observations_preserves_kpi_and_params(self):
        store = ExperimentStore()
        store.add_experiment(
            _make_experiment(
                campaign_id="c1",
                iteration=0,
                parameters={"x": 0.3},
                kpi_values={"y": 7.7},
            )
        )
        obs = StoreBridge.to_observations(store, "c1")
        assert obs[0].parameters == {"x": 0.3}
        assert obs[0].kpi_values == {"y": 7.7}

    def test_to_observations_empty_store(self):
        store = ExperimentStore()
        obs = StoreBridge.to_observations(store, "nonexistent")
        assert obs == []

    def test_to_observations_preserves_failure_info(self):
        store = ExperimentStore()
        store.add_experiment(
            _make_experiment(
                campaign_id="c1",
                iteration=0,
                is_failure=True,
                failure_reason="boom",
                qc_passed=False,
            )
        )
        obs = StoreBridge.to_observations(store, "c1")
        assert obs[0].is_failure is True
        assert obs[0].failure_reason == "boom"
        assert obs[0].qc_passed is False

    def test_to_campaign_snapshot(self):
        store = ExperimentStore()
        store.add_experiment(
            _make_experiment(
                campaign_id="test", iteration=0, parameters={"x": 0.5}
            )
        )
        spec = _make_spec(campaign_id="test")
        snapshot = StoreBridge.to_campaign_snapshot(store, spec)
        assert snapshot.campaign_id == "test"
        assert len(snapshot.observations) == 1
        assert snapshot.objective_names == ["y"]
        assert snapshot.objective_directions == ["maximize"]

    def test_to_campaign_snapshot_uses_spec_campaign_id(self):
        store = ExperimentStore()
        store.add_experiment(
            _make_experiment(campaign_id="test", iteration=0)
        )
        spec = _make_spec(campaign_id="test")
        snapshot = StoreBridge.to_campaign_snapshot(store, spec)
        assert snapshot.campaign_id == "test"

    def test_to_campaign_snapshot_override_campaign_id(self):
        store = ExperimentStore()
        store.add_experiment(
            _make_experiment(campaign_id="other", iteration=0)
        )
        spec = _make_spec(campaign_id="test")
        snapshot = StoreBridge.to_campaign_snapshot(
            store, spec, campaign_id="other"
        )
        assert len(snapshot.observations) == 1

    def test_to_campaign_snapshot_empty(self):
        store = ExperimentStore()
        spec = _make_spec()
        snapshot = StoreBridge.to_campaign_snapshot(store, spec)
        assert snapshot.observations == []

    def test_extract_curves_basic(self):
        store = ExperimentStore()
        art = _make_curve_artifact(
            x_values=[1.0, 2.0],
            y_values=[10.0, 20.0],
            metadata={"type": "eis"},
        )
        exp = _make_experiment(campaign_id="c1", iteration=0, artifacts=[art])
        store.add_experiment(exp)

        curves = StoreBridge.extract_curves(store, "c1")
        assert len(curves) == 1
        assert isinstance(curves[0], CurveData)
        assert curves[0].x_values == [1.0, 2.0]
        assert curves[0].y_values == [10.0, 20.0]
        assert curves[0].metadata == {"type": "eis"}

    def test_extract_curves_filter_by_name(self):
        store = ExperimentStore()
        art1 = _make_curve_artifact(artifact_id="a1", name="eis_curve")
        art2 = _make_curve_artifact(artifact_id="a2", name="uv_curve")
        exp = _make_experiment(
            campaign_id="c1", iteration=0, artifacts=[art1, art2]
        )
        store.add_experiment(exp)

        curves = StoreBridge.extract_curves(store, "c1", artifact_name="eis_curve")
        assert len(curves) == 1

        curves_all = StoreBridge.extract_curves(store, "c1")
        assert len(curves_all) == 2

    def test_extract_curves_skips_non_curve_artifacts(self):
        store = ExperimentStore()
        binary_art = Artifact(
            artifact_id="b1",
            artifact_type=ArtifactType.BINARY,
            name="image",
            data="base64data",
        )
        curve_art = _make_curve_artifact()
        exp = _make_experiment(
            campaign_id="c1", iteration=0, artifacts=[binary_art, curve_art]
        )
        store.add_experiment(exp)

        curves = StoreBridge.extract_curves(store, "c1")
        assert len(curves) == 1

    def test_extract_curves_empty_campaign(self):
        store = ExperimentStore()
        curves = StoreBridge.extract_curves(store, "nonexistent")
        assert curves == []

    def test_to_observations_iteration_mapping(self):
        """Iteration values should be preserved from Experiment to Observation."""
        store = ExperimentStore()
        for i in [0, 5, 10]:
            store.add_experiment(_make_experiment(campaign_id="c1", iteration=i))
        obs = StoreBridge.to_observations(store, "c1")
        assert [o.iteration for o in obs] == [0, 5, 10]


# ── TestStoreQuery ────────────────────────────────────────


class TestStoreQuery:
    """Tests for the StoreQuery dataclass and its interaction with store.query()."""

    def test_default_query_matches_all(self):
        store = ExperimentStore()
        store.add_experiment(_make_experiment(iteration=0))
        store.add_experiment(_make_experiment(iteration=1))
        results = store.query(StoreQuery())
        assert len(results) == 2

    def test_campaign_filter(self):
        store = ExperimentStore()
        store.add_experiment(_make_experiment(campaign_id="c1", iteration=0))
        store.add_experiment(_make_experiment(campaign_id="c2", iteration=0))
        results = store.query(StoreQuery(campaign_id="c1"))
        assert len(results) == 1
        assert results[0].campaign_id == "c1"

    def test_iteration_filter(self):
        store = ExperimentStore()
        store.add_experiment(_make_experiment(iteration=0))
        store.add_experiment(_make_experiment(iteration=1))
        store.add_experiment(_make_experiment(iteration=2))
        results = store.query(StoreQuery(iteration=1))
        assert len(results) == 1
        assert results[0].iteration == 1

    def test_range_filter_inclusive(self):
        store = ExperimentStore()
        for i in range(5):
            store.add_experiment(_make_experiment(iteration=i))
        results = store.query(StoreQuery(iteration_range=(1, 3)))
        assert [r.iteration for r in results] == [1, 2, 3]

    def test_artifact_filter(self):
        store = ExperimentStore()
        store.add_experiment(
            _make_experiment(iteration=0, artifacts=[_make_curve_artifact()])
        )
        store.add_experiment(_make_experiment(iteration=1))
        # Should find only the experiment with a CURVE artifact
        results = store.query(StoreQuery(has_artifact_type=ArtifactType.CURVE))
        assert len(results) == 1
        assert results[0].iteration == 0
        # Should find nothing when looking for SPECTRAL
        results = store.query(StoreQuery(has_artifact_type=ArtifactType.SPECTRAL))
        assert len(results) == 0

    def test_only_successful_filter(self):
        store = ExperimentStore()
        store.add_experiment(_make_experiment(iteration=0, is_failure=False))
        store.add_experiment(_make_experiment(iteration=1, is_failure=True))
        results = store.query(StoreQuery(only_successful=True))
        assert len(results) == 1
        assert results[0].is_failure is False

    def test_combined_query(self):
        store = ExperimentStore()
        # c1, iteration=0, success, no artifacts
        store.add_experiment(
            _make_experiment(campaign_id="c1", iteration=0, is_failure=False)
        )
        # c1, iteration=1, failure, with curve
        store.add_experiment(
            _make_experiment(
                campaign_id="c1",
                iteration=1,
                is_failure=True,
                artifacts=[_make_curve_artifact(artifact_id="x1")],
            )
        )
        # c1, iteration=2, success, with curve
        store.add_experiment(
            _make_experiment(
                campaign_id="c1",
                iteration=2,
                is_failure=False,
                artifacts=[_make_curve_artifact(artifact_id="x2")],
            )
        )
        # c2, iteration=0, success, with curve
        store.add_experiment(
            _make_experiment(
                campaign_id="c2",
                iteration=0,
                is_failure=False,
                artifacts=[_make_curve_artifact(artifact_id="x3")],
            )
        )

        # campaign c1 + successful + has curve + iteration range [0, 2]
        results = store.query(
            StoreQuery(
                campaign_id="c1",
                only_successful=True,
                has_artifact_type=ArtifactType.CURVE,
                iteration_range=(0, 2),
            )
        )
        assert len(results) == 1
        assert results[0].iteration == 2
        assert results[0].campaign_id == "c1"
