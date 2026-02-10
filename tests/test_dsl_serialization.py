"""Tests for DSL serialization (JSON and YAML round-trip, file I/O)."""

import json
import os
import tempfile

import pytest

from optimization_copilot.dsl.spec import (
    BudgetDef,
    ConditionDef,
    Direction,
    DiversityStrategy,
    ObjectiveDef,
    OptimizationSpec,
    ParallelDef,
    ParameterDef,
    ParamType,
    RiskPreference,
)
from optimization_copilot.dsl.serialization import (
    from_file,
    from_json,
    from_yaml_string,
    to_file,
    to_json,
    to_yaml_string,
)


# ── Helpers ──────────────────────────────────────────────


def _make_continuous_param(
    name: str = "x1",
    lower: float = 0.0,
    upper: float = 1.0,
    **kwargs,
) -> ParameterDef:
    return ParameterDef(
        name=name,
        type=ParamType.CONTINUOUS,
        lower=lower,
        upper=upper,
        **kwargs,
    )


def _make_categorical_param(
    name: str = "method",
    categories: list[str] | None = None,
    **kwargs,
) -> ParameterDef:
    if categories is None:
        categories = ["adam", "sgd", "rmsprop"]
    return ParameterDef(
        name=name,
        type=ParamType.CATEGORICAL,
        categories=categories,
        **kwargs,
    )


def _make_objective(
    name: str = "loss",
    direction: Direction = Direction.MINIMIZE,
    **kwargs,
) -> ObjectiveDef:
    return ObjectiveDef(name=name, direction=direction, **kwargs)


def _make_full_spec(**kwargs) -> OptimizationSpec:
    """Create a rich spec with various features for thorough testing."""
    defaults = {
        "campaign_id": "serial-test",
        "parameters": [
            _make_continuous_param(name="temperature", lower=100.0, upper=500.0),
            ParameterDef(
                name="num_layers",
                type=ParamType.DISCRETE,
                lower=1,
                upper=10,
                step_size=1,
                description="Number of hidden layers",
            ),
            _make_categorical_param(
                name="optimizer",
                categories=["adam", "sgd", "rmsprop"],
            ),
            _make_continuous_param(
                name="learning_rate",
                lower=1e-5,
                upper=1e-1,
                condition=ConditionDef(parent_name="optimizer", parent_value="adam"),
                description="LR for Adam only",
            ),
            _make_continuous_param(
                name="fixed_param",
                lower=0.0,
                upper=1.0,
                frozen=True,
                frozen_value=0.42,
            ),
        ],
        "objectives": [
            _make_objective(name="val_loss", direction=Direction.MINIMIZE),
            ObjectiveDef(
                name="throughput",
                direction=Direction.MAXIMIZE,
                constraint_lower=10.0,
                constraint_upper=None,
                is_primary=False,
                weight=0.3,
            ),
        ],
        "budget": BudgetDef(
            max_samples=200,
            max_time_seconds=7200.0,
            max_cost=1000.0,
            max_iterations=50,
        ),
        "risk_preference": RiskPreference.CONSERVATIVE,
        "parallel": ParallelDef(
            batch_size=4,
            diversity_strategy=DiversityStrategy.COVERAGE,
        ),
        "name": "Full Test Campaign",
        "description": "A comprehensive test spec",
        "metadata": {"team": "research", "version": 3},
        "seed": 123,
    }
    defaults.update(kwargs)
    return OptimizationSpec(**defaults)


def _make_minimal_spec() -> OptimizationSpec:
    """Minimal valid spec for edge case testing."""
    return OptimizationSpec(
        campaign_id="minimal",
        parameters=[_make_continuous_param()],
        objectives=[_make_objective()],
        budget=BudgetDef(),
    )


# ── TestJSONRoundTrip ────────────────────────────────────


class TestJSONRoundTrip:
    def test_full_spec_roundtrip(self):
        original = _make_full_spec()
        json_str = to_json(original)
        restored = from_json(json_str)

        assert restored.campaign_id == original.campaign_id
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.seed == original.seed
        assert restored.risk_preference == original.risk_preference

    def test_parameters_preserved(self):
        original = _make_full_spec()
        restored = from_json(to_json(original))

        assert len(restored.parameters) == len(original.parameters)
        for orig_p, rest_p in zip(original.parameters, restored.parameters):
            assert rest_p.name == orig_p.name
            assert rest_p.type == orig_p.type
            assert rest_p.lower == orig_p.lower
            assert rest_p.upper == orig_p.upper
            assert rest_p.categories == orig_p.categories
            assert rest_p.frozen == orig_p.frozen
            assert rest_p.frozen_value == orig_p.frozen_value
            assert rest_p.description == orig_p.description

    def test_condition_preserved(self):
        original = _make_full_spec()
        restored = from_json(to_json(original))

        # The 4th param (learning_rate) has a condition.
        lr_param = restored.parameters[3]
        assert lr_param.name == "learning_rate"
        assert lr_param.condition is not None
        assert lr_param.condition.parent_name == "optimizer"
        assert lr_param.condition.parent_value == "adam"

    def test_objectives_preserved(self):
        original = _make_full_spec()
        restored = from_json(to_json(original))

        assert len(restored.objectives) == 2
        assert restored.objectives[0].name == "val_loss"
        assert restored.objectives[0].direction == Direction.MINIMIZE
        assert restored.objectives[1].name == "throughput"
        assert restored.objectives[1].direction == Direction.MAXIMIZE
        assert restored.objectives[1].constraint_lower == 10.0
        assert restored.objectives[1].constraint_upper is None
        assert restored.objectives[1].weight == 0.3

    def test_budget_preserved(self):
        original = _make_full_spec()
        restored = from_json(to_json(original))

        assert restored.budget.max_samples == original.budget.max_samples
        assert restored.budget.max_time_seconds == original.budget.max_time_seconds
        assert restored.budget.max_cost == original.budget.max_cost
        assert restored.budget.max_iterations == original.budget.max_iterations

    def test_parallel_preserved(self):
        original = _make_full_spec()
        restored = from_json(to_json(original))

        assert restored.parallel.batch_size == 4
        assert restored.parallel.diversity_strategy == DiversityStrategy.COVERAGE

    def test_metadata_preserved(self):
        original = _make_full_spec()
        restored = from_json(to_json(original))

        assert restored.metadata == {"team": "research", "version": 3}

    def test_json_is_valid_json(self):
        spec = _make_full_spec()
        json_str = to_json(spec)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "campaign_id" in parsed

    def test_json_indent(self):
        spec = _make_minimal_spec()
        json_str = to_json(spec, indent=4)
        # 4-space indent should produce lines with 4-space prefix.
        lines = json_str.split("\n")
        indented_lines = [l for l in lines if l.startswith("    ")]
        assert len(indented_lines) > 0


# ── TestJSONEdgeCases ────────────────────────────────────


class TestJSONEdgeCases:
    def test_empty_metadata(self):
        spec = _make_minimal_spec()
        assert spec.metadata == {}
        restored = from_json(to_json(spec))
        assert restored.metadata == {}

    def test_none_values_in_budget(self):
        spec = _make_minimal_spec()
        assert spec.budget.max_samples is None
        restored = from_json(to_json(spec))
        assert restored.budget.max_samples is None
        assert restored.budget.max_time_seconds is None

    def test_none_condition(self):
        spec = _make_minimal_spec()
        restored = from_json(to_json(spec))
        assert restored.parameters[0].condition is None

    def test_empty_description(self):
        spec = _make_minimal_spec()
        assert spec.description == ""
        restored = from_json(to_json(spec))
        assert restored.description == ""

    def test_special_characters_in_strings(self):
        spec = _make_full_spec(
            name='Campaign with "quotes" & <special> chars',
            description="Line1\nLine2\ttab",
            metadata={"key:with:colons": "value/with/slashes"},
        )
        restored = from_json(to_json(spec))
        assert restored.name == 'Campaign with "quotes" & <special> chars'
        assert restored.description == "Line1\nLine2\ttab"
        assert restored.metadata["key:with:colons"] == "value/with/slashes"

    def test_unicode_strings(self):
        spec = _make_full_spec(
            name="Optimierung Kampagne",
            description="Beschreibung mit Umlauten: aou",
        )
        restored = from_json(to_json(spec))
        assert restored.name == "Optimierung Kampagne"

    def test_empty_categories_list_not_possible_but_none(self):
        """A param with categories=None should round-trip as None."""
        p = _make_continuous_param()
        assert p.categories is None
        spec = OptimizationSpec(
            campaign_id="test",
            parameters=[p],
            objectives=[_make_objective()],
            budget=BudgetDef(),
        )
        restored = from_json(to_json(spec))
        assert restored.parameters[0].categories is None


# ── TestYAMLRoundTrip ────────────────────────────────────


class TestYAMLRoundTrip:
    def test_full_spec_roundtrip(self):
        original = _make_full_spec()
        yaml_str = to_yaml_string(original)
        restored = from_yaml_string(yaml_str)

        assert restored.campaign_id == original.campaign_id
        assert restored.name == original.name
        assert restored.seed == original.seed
        assert restored.risk_preference == original.risk_preference

    def test_parameters_preserved(self):
        original = _make_full_spec()
        restored = from_yaml_string(to_yaml_string(original))

        assert len(restored.parameters) == len(original.parameters)
        for orig_p, rest_p in zip(original.parameters, restored.parameters):
            assert rest_p.name == orig_p.name
            assert rest_p.type == orig_p.type

    def test_condition_preserved_yaml(self):
        original = _make_full_spec()
        restored = from_yaml_string(to_yaml_string(original))

        lr_param = restored.parameters[3]
        assert lr_param.name == "learning_rate"
        assert lr_param.condition is not None
        assert lr_param.condition.parent_name == "optimizer"
        assert lr_param.condition.parent_value == "adam"

    def test_objectives_preserved_yaml(self):
        original = _make_full_spec()
        restored = from_yaml_string(to_yaml_string(original))

        assert len(restored.objectives) == 2
        assert restored.objectives[0].direction == Direction.MINIMIZE
        assert restored.objectives[1].direction == Direction.MAXIMIZE

    def test_budget_preserved_yaml(self):
        original = _make_full_spec()
        restored = from_yaml_string(to_yaml_string(original))

        assert restored.budget.max_samples == original.budget.max_samples
        assert restored.budget.max_time_seconds == original.budget.max_time_seconds

    def test_parallel_preserved_yaml(self):
        original = _make_full_spec()
        restored = from_yaml_string(to_yaml_string(original))

        assert restored.parallel.batch_size == 4
        assert restored.parallel.diversity_strategy == DiversityStrategy.COVERAGE

    def test_frozen_values_preserved_yaml(self):
        original = _make_full_spec()
        restored = from_yaml_string(to_yaml_string(original))

        frozen_p = restored.parameters[4]
        assert frozen_p.name == "fixed_param"
        assert frozen_p.frozen is True
        assert frozen_p.frozen_value == pytest.approx(0.42)

    def test_minimal_spec_yaml_roundtrip(self):
        original = _make_minimal_spec()
        yaml_str = to_yaml_string(original)
        restored = from_yaml_string(yaml_str)

        assert restored.campaign_id == original.campaign_id
        assert len(restored.parameters) == 1
        assert len(restored.objectives) == 1


# ── TestYAMLFormat ───────────────────────────────────────


class TestYAMLFormat:
    def test_yaml_output_is_string(self):
        spec = _make_minimal_spec()
        yaml_str = to_yaml_string(spec)
        assert isinstance(yaml_str, str)
        assert len(yaml_str) > 0

    def test_yaml_contains_top_level_keys(self):
        spec = _make_full_spec()
        yaml_str = to_yaml_string(spec)
        assert "campaign_id:" in yaml_str
        assert "parameters:" in yaml_str
        assert "objectives:" in yaml_str
        assert "budget:" in yaml_str
        assert "risk_preference:" in yaml_str
        assert "parallel:" in yaml_str
        assert "seed:" in yaml_str

    def test_yaml_uses_indentation(self):
        spec = _make_full_spec()
        yaml_str = to_yaml_string(spec)
        lines = yaml_str.splitlines()
        indented_lines = [l for l in lines if l.startswith("  ")]
        assert len(indented_lines) > 0

    def test_yaml_list_items_use_dash(self):
        spec = _make_full_spec()
        yaml_str = to_yaml_string(spec)
        assert "- name:" in yaml_str or "- " in yaml_str

    def test_yaml_null_rendered(self):
        spec = _make_minimal_spec()
        yaml_str = to_yaml_string(spec)
        assert "null" in yaml_str

    def test_yaml_boolean_rendered(self):
        spec = _make_full_spec()
        yaml_str = to_yaml_string(spec)
        assert "true" in yaml_str or "false" in yaml_str

    def test_yaml_quoted_special_chars(self):
        spec = _make_full_spec(
            name='Name with "quotes"',
        )
        yaml_str = to_yaml_string(spec)
        # The name should be quoted in the YAML output.
        assert '"' in yaml_str


# ── TestFileIO ───────────────────────────────────────────


class TestFileIO:
    def test_write_read_json(self):
        spec = _make_full_spec()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, dir="/tmp"
        ) as f:
            path = f.name

        try:
            to_file(spec, path, format="json")
            restored = from_file(path)

            assert restored.campaign_id == spec.campaign_id
            assert restored.name == spec.name
            assert restored.seed == spec.seed
            assert len(restored.parameters) == len(spec.parameters)
            assert len(restored.objectives) == len(spec.objectives)
        finally:
            os.unlink(path)

    def test_write_read_yaml(self):
        spec = _make_full_spec()
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, dir="/tmp"
        ) as f:
            path = f.name

        try:
            to_file(spec, path, format="yaml")
            restored = from_file(path)

            assert restored.campaign_id == spec.campaign_id
            assert restored.name == spec.name
            assert restored.seed == spec.seed
            assert len(restored.parameters) == len(spec.parameters)
        finally:
            os.unlink(path)

    def test_write_read_yml_extension(self):
        spec = _make_minimal_spec()
        with tempfile.NamedTemporaryFile(
            suffix=".yml", delete=False, dir="/tmp"
        ) as f:
            path = f.name

        try:
            to_file(spec, path, format="yml")
            restored = from_file(path)

            assert restored.campaign_id == spec.campaign_id
        finally:
            os.unlink(path)

    def test_json_file_content_is_valid_json(self):
        spec = _make_minimal_spec()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, dir="/tmp"
        ) as f:
            path = f.name

        try:
            to_file(spec, path)
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            parsed = json.loads(content)
            assert isinstance(parsed, dict)
        finally:
            os.unlink(path)

    def test_unsupported_format_raises(self):
        spec = _make_minimal_spec()
        with tempfile.NamedTemporaryFile(
            suffix=".xml", delete=False, dir="/tmp"
        ) as f:
            path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                to_file(spec, path, format="xml")
        finally:
            os.unlink(path)

    def test_unsupported_extension_on_read_raises(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, dir="/tmp"
        ) as f:
            path = f.name
            f.write(b"some content")

        try:
            with pytest.raises(ValueError, match="Cannot auto-detect"):
                from_file(path)
        finally:
            os.unlink(path)

    def test_file_roundtrip_preserves_condition(self):
        spec = _make_full_spec()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, dir="/tmp"
        ) as f:
            path = f.name

        try:
            to_file(spec, path)
            restored = from_file(path)

            lr_param = restored.parameters[3]
            assert lr_param.condition is not None
            assert lr_param.condition.parent_name == "optimizer"
        finally:
            os.unlink(path)

    def test_yaml_file_roundtrip_preserves_all(self):
        spec = _make_full_spec()
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, dir="/tmp"
        ) as f:
            path = f.name

        try:
            to_file(spec, path, format="yaml")
            restored = from_file(path)

            assert restored.risk_preference == RiskPreference.CONSERVATIVE
            assert restored.parallel.batch_size == 4
            assert restored.budget.max_samples == 200
            assert len(restored.objectives) == 2
        finally:
            os.unlink(path)
