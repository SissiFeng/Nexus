"""Tests for the Click CLI application.

Uses Click's CliRunner for invocation testing.
"""

from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

from optimization_copilot.cli_app.main import cli


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def runner():
    """Create a Click CliRunner."""
    return CliRunner()


@pytest.fixture
def workspace(tmp_path):
    """Return a temporary workspace directory path."""
    return str(tmp_path / "workspace")


@pytest.fixture
def spec_path(tmp_path):
    """Create a temporary spec JSON file and return its path."""
    spec = {
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [
            {"name": "y", "direction": "minimize"},
        ],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return str(path)


@pytest.fixture
def multi_param_spec_path(tmp_path):
    """Spec with multiple parameters."""
    spec = {
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            {"name": "y", "type": "continuous", "bounds": [-1.0, 1.0]},
        ],
        "objectives": [
            {"name": "loss", "direction": "minimize"},
            {"name": "accuracy", "direction": "maximize"},
        ],
    }
    path = tmp_path / "multi_spec.json"
    path.write_text(json.dumps(spec))
    return str(path)


def _invoke(runner, args, workspace=None):
    """Helper to invoke CLI with optional workspace."""
    cmd = []
    if workspace:
        cmd.extend(["--workspace", workspace])
    cmd.extend(args)
    return runner.invoke(cli, cmd)


# ── Top-Level Help ───────────────────────────────────────────────


class TestCLIHelp:
    """Help text tests."""

    def test_cli_help(self, runner):
        result = _invoke(runner, ["--help"])
        assert result.exit_code == 0
        assert "Optimization Copilot" in result.output

    def test_campaign_help(self, runner):
        result = _invoke(runner, ["campaign", "--help"])
        assert result.exit_code == 0
        assert "Manage optimization campaigns" in result.output

    def test_store_help(self, runner):
        result = _invoke(runner, ["store", "--help"])
        assert result.exit_code == 0
        assert "Query experiment store" in result.output

    def test_server_help(self, runner):
        result = _invoke(runner, ["server", "--help"])
        assert result.exit_code == 0
        assert "Manage the API server" in result.output

    def test_campaign_create_help(self, runner):
        result = _invoke(runner, ["campaign", "create", "--help"])
        assert result.exit_code == 0
        assert "--spec" in result.output

    def test_campaign_list_help(self, runner):
        result = _invoke(runner, ["campaign", "list", "--help"])
        assert result.exit_code == 0
        assert "--status" in result.output

    def test_server_start_help(self, runner):
        result = _invoke(runner, ["server", "start", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output


# ── Server Commands ──────────────────────────────────────────────


class TestServerCommands:
    """Server init and management."""

    def test_server_init_creates_workspace(self, runner, workspace):
        result = _invoke(runner, ["server", "init"], workspace=workspace)
        assert result.exit_code == 0
        assert "Workspace initialized" in result.output
        assert "Admin API key created" in result.output

    def test_server_init_idempotent(self, runner, workspace):
        """Second init says keys exist instead of creating new ones."""
        result1 = _invoke(runner, ["server", "init"], workspace=workspace)
        assert result1.exit_code == 0
        assert "Admin API key created" in result1.output

        result2 = _invoke(runner, ["server", "init"], workspace=workspace)
        assert result2.exit_code == 0
        assert "already exist" in result2.output

    def test_server_init_shows_workspace_id(self, runner, workspace):
        result = _invoke(runner, ["server", "init"], workspace=workspace)
        assert result.exit_code == 0
        assert "ID:" in result.output


# ── Campaign Create ──────────────────────────────────────────────


class TestCampaignCreate:
    """Campaign creation commands."""

    def test_create_campaign(self, runner, workspace, spec_path):
        result = _invoke(
            runner, ["campaign", "create", "--spec", spec_path], workspace=workspace
        )
        assert result.exit_code == 0
        assert "Created campaign:" in result.output
        assert "Status: draft" in result.output

    def test_create_campaign_with_name(self, runner, workspace, spec_path):
        result = _invoke(
            runner,
            ["campaign", "create", "--spec", spec_path, "--name", "My Experiment"],
            workspace=workspace,
        )
        assert result.exit_code == 0
        assert "My Experiment" in result.output

    def test_create_campaign_with_tags(self, runner, workspace, spec_path):
        result = _invoke(
            runner,
            [
                "campaign", "create",
                "--spec", spec_path,
                "--tag", "batch1",
                "--tag", "priority",
            ],
            workspace=workspace,
        )
        assert result.exit_code == 0
        assert "Created campaign:" in result.output

    def test_create_campaign_invalid_spec_path(self, runner, workspace):
        result = _invoke(
            runner,
            ["campaign", "create", "--spec", "/nonexistent/spec.json"],
            workspace=workspace,
        )
        assert result.exit_code != 0

    def test_create_campaign_no_spec_flag(self, runner, workspace):
        """Missing required --spec flag."""
        result = _invoke(
            runner, ["campaign", "create"], workspace=workspace
        )
        assert result.exit_code != 0


# ── Campaign List ────────────────────────────────────────────────


class TestCampaignList:
    """Campaign listing commands."""

    def test_list_empty(self, runner, workspace):
        # Initialize workspace first
        _invoke(runner, ["server", "init"], workspace=workspace)

        result = _invoke(runner, ["campaign", "list"], workspace=workspace)
        assert result.exit_code == 0
        assert "No campaigns found" in result.output

    def test_list_after_create(self, runner, workspace, spec_path):
        _invoke(
            runner,
            ["campaign", "create", "--spec", spec_path, "--name", "Listed Campaign"],
            workspace=workspace,
        )

        result = _invoke(runner, ["campaign", "list"], workspace=workspace)
        assert result.exit_code == 0
        assert "draft" in result.output

    def test_list_multiple_campaigns(self, runner, workspace, spec_path):
        _invoke(
            runner, ["campaign", "create", "--spec", spec_path, "--name", "A"],
            workspace=workspace,
        )
        _invoke(
            runner, ["campaign", "create", "--spec", spec_path, "--name", "B"],
            workspace=workspace,
        )

        result = _invoke(runner, ["campaign", "list"], workspace=workspace)
        assert result.exit_code == 0
        # Should show both campaigns (two draft entries)
        assert result.output.count("draft") >= 2


# ── Campaign Status ──────────────────────────────────────────────


class TestCampaignStatus:
    """Campaign status command."""

    def test_status_shows_details(self, runner, workspace, spec_path):
        create_result = _invoke(
            runner,
            ["campaign", "create", "--spec", spec_path, "--name", "Status Test"],
            workspace=workspace,
        )
        # Extract campaign ID from output
        cid = self._extract_campaign_id(create_result.output)

        result = _invoke(
            runner, ["campaign", "status", cid], workspace=workspace
        )
        assert result.exit_code == 0
        assert "Campaign:" in result.output
        assert "Status:" in result.output
        assert "draft" in result.output

    def test_status_invalid_id(self, runner, workspace):
        _invoke(runner, ["server", "init"], workspace=workspace)

        result = _invoke(
            runner, ["campaign", "status", "nonexistent-id"], workspace=workspace
        )
        assert result.exit_code != 0

    @staticmethod
    def _extract_campaign_id(output: str) -> str:
        """Extract campaign ID from 'Created campaign: <id>' output."""
        for line in output.strip().split("\n"):
            if "Created campaign:" in line:
                return line.split("Created campaign:")[1].strip()
        raise ValueError(f"Could not extract campaign ID from: {output}")


# ── Campaign Delete ──────────────────────────────────────────────


class TestCampaignDelete:
    """Campaign deletion commands."""

    def test_delete_with_yes_flag(self, runner, workspace, spec_path):
        create_result = _invoke(
            runner, ["campaign", "create", "--spec", spec_path], workspace=workspace
        )
        cid = self._extract_campaign_id(create_result.output)

        result = _invoke(
            runner, ["campaign", "delete", cid, "--yes"], workspace=workspace
        )
        assert result.exit_code == 0
        assert "archived" in result.output

    def test_delete_without_yes_prompts(self, runner, workspace, spec_path):
        """Without --yes, should prompt for confirmation."""
        create_result = _invoke(
            runner, ["campaign", "create", "--spec", spec_path], workspace=workspace
        )
        cid = self._extract_campaign_id(create_result.output)

        # Provide 'y' via input
        result = runner.invoke(
            cli,
            ["--workspace", workspace, "campaign", "delete", cid],
            input="y\n",
        )
        assert result.exit_code == 0
        assert "archived" in result.output

    def test_delete_without_yes_abort(self, runner, workspace, spec_path):
        """Without --yes, abort on 'n'."""
        create_result = _invoke(
            runner, ["campaign", "create", "--spec", spec_path], workspace=workspace
        )
        cid = self._extract_campaign_id(create_result.output)

        result = runner.invoke(
            cli,
            ["--workspace", workspace, "campaign", "delete", cid],
            input="n\n",
        )
        assert result.exit_code != 0  # Aborted

    def test_delete_nonexistent_campaign(self, runner, workspace):
        _invoke(runner, ["server", "init"], workspace=workspace)

        result = _invoke(
            runner, ["campaign", "delete", "nonexistent-id", "--yes"], workspace=workspace
        )
        assert result.exit_code != 0

    @staticmethod
    def _extract_campaign_id(output: str) -> str:
        for line in output.strip().split("\n"):
            if "Created campaign:" in line:
                return line.split("Created campaign:")[1].strip()
        raise ValueError(f"Could not extract campaign ID from: {output}")


# ── Campaign Compare ─────────────────────────────────────────────


class TestCampaignCompare:
    """Campaign comparison commands."""

    def test_compare_with_fewer_than_two_ids(self, runner, workspace):
        _invoke(runner, ["server", "init"], workspace=workspace)

        result = _invoke(
            runner, ["campaign", "compare", "single-id"], workspace=workspace
        )
        assert result.exit_code != 0

    def test_compare_two_campaigns(self, runner, workspace, spec_path):
        c1 = _invoke(
            runner, ["campaign", "create", "--spec", spec_path, "--name", "A"],
            workspace=workspace,
        )
        c2 = _invoke(
            runner, ["campaign", "create", "--spec", spec_path, "--name", "B"],
            workspace=workspace,
        )

        cid1 = self._extract_campaign_id(c1.output)
        cid2 = self._extract_campaign_id(c2.output)

        result = _invoke(
            runner, ["campaign", "compare", cid1, cid2], workspace=workspace
        )
        assert result.exit_code == 0
        assert "Campaign Comparison" in result.output

    @staticmethod
    def _extract_campaign_id(output: str) -> str:
        for line in output.strip().split("\n"):
            if "Created campaign:" in line:
                return line.split("Created campaign:")[1].strip()
        raise ValueError(f"Could not extract campaign ID from: {output}")


# ── Store Commands ───────────────────────────────────────────────


class TestStoreCommands:
    """Store query commands."""

    def test_store_summary_no_data(self, runner, workspace):
        _invoke(runner, ["server", "init"], workspace=workspace)

        result = _invoke(
            runner, ["store", "summary", "nonexistent-id"], workspace=workspace
        )
        assert result.exit_code != 0
        assert "No store data found" in result.output

    def test_store_query_no_data(self, runner, workspace):
        _invoke(runner, ["server", "init"], workspace=workspace)

        result = _invoke(
            runner, ["store", "query", "nonexistent-id"], workspace=workspace
        )
        assert result.exit_code != 0
        assert "No store data found" in result.output

    def test_store_export_no_data(self, runner, workspace, tmp_path):
        _invoke(runner, ["server", "init"], workspace=workspace)

        output_path = str(tmp_path / "export.json")
        result = _invoke(
            runner,
            ["store", "export", "nonexistent-id", "--output", output_path],
            workspace=workspace,
        )
        assert result.exit_code != 0
        assert "No store data found" in result.output


# ── Workspace Option ─────────────────────────────────────────────


class TestWorkspaceOption:
    """Workspace option handling."""

    def test_workspace_option_passed(self, runner, tmp_path):
        ws = str(tmp_path / "custom_workspace")
        result = _invoke(runner, ["server", "init"], workspace=ws)
        assert result.exit_code == 0
        assert os.path.isdir(ws)

    def test_default_workspace(self, runner):
        """CLI uses default ./workspace when not specified."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        # Help should mention the default
        assert "./workspace" in result.output or "workspace" in result.output.lower()


# ── Campaign Export ─────────────────────────────────────────────


class TestCampaignExport:
    """Campaign export command."""

    def test_export_campaign(self, runner, workspace, spec_path, tmp_path):
        create_result = _invoke(
            runner, ["campaign", "create", "--spec", spec_path], workspace=workspace
        )
        cid = self._extract_campaign_id(create_result.output)

        output_path = str(tmp_path / "export.json")
        result = _invoke(
            runner,
            ["campaign", "export", cid, "--output", output_path],
            workspace=workspace,
        )
        assert result.exit_code == 0
        assert "Exported campaign" in result.output

        # Verify the exported file is valid JSON with expected keys
        with open(output_path) as f:
            archive = json.load(f)
        assert "record" in archive
        assert "spec" in archive
        assert "result" in archive
        assert "audit" in archive
        assert archive["record"]["campaign_id"] == cid
        assert archive["spec"]["parameters"] is not None

    def test_export_campaign_result_and_audit_null(self, runner, workspace, spec_path, tmp_path):
        """For a fresh campaign, result and audit should be null in the archive."""
        create_result = _invoke(
            runner, ["campaign", "create", "--spec", spec_path], workspace=workspace
        )
        cid = self._extract_campaign_id(create_result.output)

        output_path = str(tmp_path / "export.json")
        _invoke(
            runner,
            ["campaign", "export", cid, "--output", output_path],
            workspace=workspace,
        )
        with open(output_path) as f:
            archive = json.load(f)
        assert archive["result"] is None
        assert archive["audit"] is None

    def test_export_nonexistent_campaign(self, runner, workspace, tmp_path):
        _invoke(runner, ["server", "init"], workspace=workspace)

        output_path = str(tmp_path / "export.json")
        result = _invoke(
            runner,
            ["campaign", "export", "nonexistent-id", "--output", output_path],
            workspace=workspace,
        )
        assert result.exit_code != 0
        assert "Campaign not found" in result.output

    def test_export_help(self, runner):
        result = _invoke(runner, ["campaign", "export", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output

    @staticmethod
    def _extract_campaign_id(output: str) -> str:
        for line in output.strip().split("\n"):
            if "Created campaign:" in line:
                return line.split("Created campaign:")[1].strip()
        raise ValueError(f"Could not extract campaign ID from: {output}")


# ── Campaign Validate ───────────────────────────────────────────


class TestCampaignValidate:
    """Campaign validate command."""

    def test_validate_valid_spec(self, runner, spec_path):
        result = _invoke(runner, ["campaign", "validate", "--spec", spec_path])
        assert result.exit_code == 0
        assert "Validation OK" in result.output
        assert "Parameters: 1" in result.output
        assert "Objectives: 1" in result.output

    def test_validate_multi_param_spec(self, runner, multi_param_spec_path):
        result = _invoke(runner, ["campaign", "validate", "--spec", multi_param_spec_path])
        assert result.exit_code == 0
        assert "Validation OK" in result.output
        assert "Parameters: 2" in result.output
        assert "Objectives: 2" in result.output

    def test_validate_missing_parameters(self, runner, tmp_path):
        spec = {"objectives": [{"name": "y", "direction": "minimize"}]}
        path = tmp_path / "bad_spec.json"
        path.write_text(json.dumps(spec))

        result = _invoke(runner, ["campaign", "validate", "--spec", str(path)])
        assert result.exit_code != 0
        assert "Missing required key: 'parameters'" in result.output

    def test_validate_missing_objectives(self, runner, tmp_path):
        spec = {"parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}]}
        path = tmp_path / "bad_spec.json"
        path.write_text(json.dumps(spec))

        result = _invoke(runner, ["campaign", "validate", "--spec", str(path)])
        assert result.exit_code != 0
        assert "Missing required key: 'objectives'" in result.output

    def test_validate_parameter_missing_name(self, runner, tmp_path):
        spec = {
            "parameters": [{"type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        path = tmp_path / "bad_spec.json"
        path.write_text(json.dumps(spec))

        result = _invoke(runner, ["campaign", "validate", "--spec", str(path)])
        assert result.exit_code != 0
        assert "missing 'name'" in result.output

    def test_validate_parameter_missing_type(self, runner, tmp_path):
        spec = {
            "parameters": [{"name": "x", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        path = tmp_path / "bad_spec.json"
        path.write_text(json.dumps(spec))

        result = _invoke(runner, ["campaign", "validate", "--spec", str(path)])
        assert result.exit_code != 0
        assert "missing 'type'" in result.output

    def test_validate_continuous_param_missing_bounds(self, runner, tmp_path):
        spec = {
            "parameters": [{"name": "x", "type": "continuous"}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        path = tmp_path / "bad_spec.json"
        path.write_text(json.dumps(spec))

        result = _invoke(runner, ["campaign", "validate", "--spec", str(path)])
        assert result.exit_code != 0
        assert "continuous type requires 'bounds'" in result.output

    def test_validate_objective_missing_name(self, runner, tmp_path):
        spec = {
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"direction": "minimize"}],
        }
        path = tmp_path / "bad_spec.json"
        path.write_text(json.dumps(spec))

        result = _invoke(runner, ["campaign", "validate", "--spec", str(path)])
        assert result.exit_code != 0
        assert "missing 'name'" in result.output

    def test_validate_objective_missing_direction(self, runner, tmp_path):
        spec = {
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y"}],
        }
        path = tmp_path / "bad_spec.json"
        path.write_text(json.dumps(spec))

        result = _invoke(runner, ["campaign", "validate", "--spec", str(path)])
        assert result.exit_code != 0
        assert "missing 'direction'" in result.output

    def test_validate_invalid_json(self, runner, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{")

        result = _invoke(runner, ["campaign", "validate", "--spec", str(path)])
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_validate_nonexistent_file(self, runner):
        result = _invoke(runner, ["campaign", "validate", "--spec", "/nonexistent/spec.json"])
        assert result.exit_code != 0

    def test_validate_help(self, runner):
        result = _invoke(runner, ["campaign", "validate", "--help"])
        assert result.exit_code == 0
        assert "--spec" in result.output


# ── Meta-Learning Commands ──────────────────────────────────────


class TestMetaLearningHelp:
    """Meta-learning command help tests."""

    def test_meta_learning_help(self, runner):
        result = _invoke(runner, ["meta-learning", "--help"])
        assert result.exit_code == 0
        assert "Meta-learning advisor commands" in result.output

    def test_meta_learning_show_help(self, runner):
        result = _invoke(runner, ["meta-learning", "show", "--help"])
        assert result.exit_code == 0
        assert "Show meta-learning advisor state" in result.output

    def test_meta_learning_advice_help(self, runner):
        result = _invoke(runner, ["meta-learning", "advice", "--help"])
        assert result.exit_code == 0
        assert "Get advice for a campaign spec" in result.output


class TestMetaLearningShow:
    """Meta-learning show command."""

    def test_show_no_advisor_data(self, runner, workspace):
        _invoke(runner, ["server", "init"], workspace=workspace)

        result = _invoke(
            runner, ["meta-learning", "show"], workspace=workspace
        )
        assert result.exit_code == 0
        assert "No meta-learning advisor data found" in result.output

    def test_show_with_advisor_data(self, runner, workspace):
        _invoke(runner, ["server", "init"], workspace=workspace)

        # Manually write advisor data to workspace
        from optimization_copilot.platform.workspace import Workspace

        ws = Workspace(workspace)
        ws.init()
        advisor_data = {
            "config": {
                "min_experiences_for_learning": 3,
                "similarity_decay": 0.3,
            },
            "experience_store": {
                "outcomes": [
                    {"campaign_id": "test-1"},
                    {"campaign_id": "test-2"},
                ],
            },
            "weight_tuner": {
                "learned_weights": {"key1": {}},
            },
            "threshold_learner": {
                "learned_thresholds": {},
            },
            "failure_learner": {
                "strategies": {"type1": {}},
            },
            "drift_tracker": {
                "robustness": {},
            },
        }
        ws.save_advisor(advisor_data)

        result = _invoke(
            runner, ["meta-learning", "show"], workspace=workspace
        )
        assert result.exit_code == 0
        assert "Meta-Learning Advisor State:" in result.output
        assert "Experiences: 2" in result.output
        assert "Learned weight profiles: 1" in result.output
        assert "Failure strategies: 1" in result.output


class TestMetaLearningAdvice:
    """Meta-learning advice command."""

    def test_advice_no_advisor_data(self, runner, workspace, spec_path):
        _invoke(runner, ["server", "init"], workspace=workspace)

        result = _invoke(
            runner, ["meta-learning", "advice", spec_path], workspace=workspace
        )
        assert result.exit_code == 0
        assert "No meta-learning advisor data found" in result.output

    def test_advice_invalid_json(self, runner, workspace, tmp_path):
        _invoke(runner, ["server", "init"], workspace=workspace)

        # Write advisor data so we pass the first check
        from optimization_copilot.platform.workspace import Workspace

        ws = Workspace(workspace)
        ws.init()
        ws.save_advisor({"config": {}, "experience_store": {"outcomes": []}})

        bad_path = tmp_path / "bad.json"
        bad_path.write_text("not valid json")

        result = _invoke(
            runner, ["meta-learning", "advice", str(bad_path)], workspace=workspace
        )
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_advice_nonexistent_spec(self, runner, workspace):
        _invoke(runner, ["server", "init"], workspace=workspace)

        result = _invoke(
            runner, ["meta-learning", "advice", "/nonexistent/spec.json"], workspace=workspace
        )
        assert result.exit_code != 0
