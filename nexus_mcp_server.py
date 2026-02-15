"""
Nexus MCP Server — Wraps Nexus optimization platform as MCP tools.

Connects to a running Nexus backend (default: http://localhost:8000)
and exposes 7 tools for LLM-driven experiment optimization.

Usage:
    pip install "mcp[cli]"
    python nexus_mcp_server.py

Or add to your Claude Desktop / MCP client config:
    {
      "mcpServers": {
        "nexus": {
          "command": "python",
          "args": ["path/to/nexus_mcp_server.py"]
        }
      }
    }
"""

from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from mcp.server.fastmcp import FastMCP

NEXUS_URL = os.getenv("NEXUS_URL", "http://localhost:8000")

mcp = FastMCP("Nexus")


# ── helpers ──────────────────────────────────────────────────────

def _api(method: str, path: str, body: dict | None = None) -> dict:
    """Call the Nexus REST API."""
    url = f"{NEXUS_URL}/api{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        error_body = e.read().decode() if e.fp else str(e)
        return {"error": f"HTTP {e.code}", "detail": error_body}
    except URLError as e:
        return {"error": "connection_failed", "detail": str(e.reason)}


def _get(path: str) -> dict:
    return _api("GET", path)


def _post(path: str, body: dict | None = None) -> dict:
    return _api("POST", path, body)


# ── tools ────────────────────────────────────────────────────────

@mcp.tool()
def nexus_create_campaign(
    name: str,
    data: list[dict[str, str]],
    parameters: list[dict],
    objectives: list[dict],
    description: str = "",
    batch_size: int = 5,
    exploration_weight: float = 0.5,
) -> dict:
    """Create a new optimization campaign from experimental data.

    Args:
        name: Campaign name (e.g. "Suzuki Coupling Yield Optimization")
        data: List of row dicts from CSV. Each dict maps column name to string value.
              Example: [{"temperature": "80", "pressure": "1.5", "yield": "0.72"}, ...]
        parameters: List of parameter specs. Each has:
            - name: column name
            - type: "continuous" or "categorical"
            - lower: min bound (continuous only)
            - upper: max bound (continuous only)
        objectives: List of objective specs. Each has:
            - name: column name
            - direction: "minimize" or "maximize"
        description: Optional campaign description
        batch_size: Number of experiments per suggestion batch (default 5)
        exploration_weight: 0.0 (exploit) to 1.0 (explore), default 0.5

    Returns:
        Campaign details including campaign_id, status, total_trials, best_kpi
    """
    return _post("/campaigns/from-upload", {
        "name": name,
        "description": description,
        "data": data,
        "mapping": {
            "parameters": parameters,
            "objectives": objectives,
            "metadata": [],
            "ignored": [],
        },
        "batch_size": batch_size,
        "exploration_weight": exploration_weight,
    })


@mcp.tool()
def nexus_get_diagnostics(campaign_id: str) -> dict:
    """Get real-time diagnostic signals for a campaign.

    Returns 8+ health metrics with traffic-light interpretation:
    - convergence_trend: How strongly the objective is improving (0-1)
    - improvement_velocity: Rate of KPI improvement
    - exploration_coverage: Fraction of parameter space explored
    - failure_rate: Proportion of failed experiments
    - noise_estimate: Estimated noise level in observations
    - plateau_length: Iterations since last improvement
    - signal_to_noise_ratio: Signal strength vs noise
    - best_kpi_value: Current best objective value

    Args:
        campaign_id: Campaign UUID

    Returns:
        Dict of diagnostic signal names to numeric values
    """
    return _get(f"/campaigns/{campaign_id}/diagnostics")


@mcp.tool()
def nexus_suggest_next(campaign_id: str, n: int = 5) -> dict:
    """Suggest the next batch of experiments to run.

    Uses the campaign's optimization backend (GP-BO, TPE, CMA-ES, etc.)
    to generate candidates that balance exploration and exploitation.

    Args:
        campaign_id: Campaign UUID
        n: Number of suggestions (1-50, default 5)

    Returns:
        suggestions: List of parameter dicts to try
        predicted_values: Expected objective values
        predicted_uncertainties: Confidence intervals
        backend_used: Which algorithm generated these
        phase: Current optimization phase (learning/exploitation)
    """
    return _get(f"/campaigns/{campaign_id}/suggestions?n={n}")


@mcp.tool()
def nexus_ingest_results(
    campaign_id: str,
    data: list[dict[str, str]],
) -> dict:
    """Feed back experimental results into the campaign.

    After running suggested experiments, submit the results so Nexus
    can update its model and generate better suggestions.

    Args:
        campaign_id: Campaign UUID
        data: List of row dicts with parameter values and measured objectives.
              Must include all parameter and objective columns.
              Example: [{"temperature": "85", "pressure": "1.2", "yield": "0.81"}]

    Returns:
        campaign_id, appended count, new total, updated best_kpi
    """
    return _post(f"/campaigns/{campaign_id}/append", {"data": data})


@mcp.tool()
def nexus_explain_decision(campaign_id: str, question: str) -> dict:
    """Ask Nexus to explain its optimization decisions in natural language.

    Supports questions like:
    - "Why did you switch to exploitation?"
    - "Which parameters matter most?"
    - "What patterns have you found?"
    - "Should I focus on temperature or pressure?"

    The response includes computed diagnostic signals, not just LLM text.

    Args:
        campaign_id: Campaign UUID
        question: Natural language question about the optimization

    Returns:
        reply: Natural language explanation
        role: "agent" or "suggestion"
        metadata: Supporting data (diagnostics, recommendations, suggestions)
    """
    return _post(f"/chat/{campaign_id}", {"message": question})


@mcp.tool()
def nexus_causal_discovery(
    data: list[list[float]],
    var_names: list[str],
    alpha: float = 0.05,
) -> dict:
    """Run causal discovery (PC algorithm) on experimental data.

    Discovers causal relationships between variables — which parameters
    actually cause changes in objectives vs. mere correlations.

    Args:
        data: 2D array of observations. Each inner list is one row.
              Columns correspond to var_names.
        var_names: Names for each column (e.g. ["temperature", "pressure", "yield"])
        alpha: Significance level for conditional independence tests (default 0.05)

    Returns:
        Causal graph structure with edges and their strengths
    """
    return _post("/analysis/causal/discover", {
        "data": data,
        "var_names": var_names,
        "alpha": alpha,
    })


@mcp.tool()
def nexus_hypothesis_status(tracker_state: dict) -> dict:
    """Check the status of tracked scientific hypotheses.

    Nexus tracks hypotheses through their lifecycle:
    PROPOSED → TESTING → SUPPORTED / REFUTED / INCONCLUSIVE

    Pass the current tracker state to get a status report.

    Args:
        tracker_state: Dict containing hypothesis tracker state.
            Expected keys: hypotheses (list of hypothesis dicts),
            each with: id, statement, status, evidence, tests_run

    Returns:
        Status report with hypothesis lifecycle summary
    """
    return _post("/analysis/hypothesis/status", {
        "tracker_state": tracker_state,
    })


# ── entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
