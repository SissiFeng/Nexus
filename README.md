# Nexus

**Intelligent Optimization Platform for Scientific Experiments**

🌐 **[View Landing Page](https://sissifeng.github.io/Nexus)** — Interactive product showcase

---

**Intelligent Optimization Platform for Scientific Experiments**

Nexus transforms how scientists run Bayesian optimization campaigns — from black-box tool to transparent research partner. Upload your data, get intelligent suggestions, and understand every decision the optimizer makes.

---

## The Problem

Scientists running experimental optimization (materials discovery, drug formulation, process engineering) face two barriers:

- **Opacity** — Existing tools don't explain why a suggestion was made
- **Programming requirements** — Most tools demand ML expertise to configure

The result: researchers don't trust suggestions, can't diagnose stalled optimizations, or abandon the tool entirely.

## The Solution

Nexus wraps Bayesian optimization with a diagnostic intelligence layer:

1. **Zero-code setup** — Upload CSV, map columns visually, start optimizing
2. **148+ inline visualizations** — Real-time diagnostics that surface problems before they waste trials
3. **Transparent suggestions** — Every recommendation includes novelty scores, risk profiles, and provenance
4. **AI chat** — Ask "Why did you switch strategies?" and get answers backed by computed signals
5. **Campaign memory** — Decision journals, learning curves, and hypothesis tracking

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for the web frontend)

### Installation

```bash
# Clone the repository
git clone https://github.com/sissifeng/optimization-copilot.git
cd optimization-copilot

# Install Python package
pip install -e ".[dev]"

# (Optional) Set up API key for AI chat features
cp .env.example .env
# Edit .env and add your Anthropic API key

# Install frontend dependencies
cd optimization_copilot/web
npm install
cd ../..
```

### Start the Platform

```bash
# Terminal 1: Start the backend API server
nexus server start
# or: python -m optimization_copilot.cli_app.main server start
# API runs at http://localhost:8000

# Terminal 2: Start the frontend dev server
cd optimization_copilot/web
npm run dev
# UI runs at http://localhost:5173
```

Open **http://localhost:5173** in your browser.

---

## User Guide

### 1. Create a Campaign

From the Dashboard, click **+ New Campaign**. You have two options:

- **Quick Start Templates** — Choose a domain (Chemical Synthesis, Materials Discovery, Bioprocess, Formulation Design) for guided setup
- **Upload Your Own Data** — Drop a CSV file with your experimental results

After uploading, the **Column Mapper** lets you:
- Assign columns as **Parameters** (inputs you control), **Objectives** (outputs you measure), or **Metadata**
- Set min/max bounds for continuous parameters
- Choose optimization direction (minimize/maximize) per objective

Click **Create Campaign** to enter the Workspace.

### 2. The Workspace

The Workspace has 5 tabs:

| Tab | What It Shows |
|-----|---------------|
| **Overview** | Campaign health: iteration count, best result, phase (Learning/Exploitation), diagnostic signals with traffic-light indicators (green/yellow/red), convergence trend, budget efficiency |
| **Explore** | Search space analysis: parameter importance, parallel coordinates, boundary utilization, local optima map, concept drift detection |
| **Suggestions** | Next experiments to run: predicted outcomes with uncertainty, novelty scores, redundancy checks, risk-return profiles |
| **Insights** | AI-generated analysis: correlations, optimal regions, failure patterns, parameter interactions |
| **History** | Campaign timeline: outcome distribution, explore/exploit phases, learning curve projection, decision journal |

### 3. AI Chat

The chat panel (bottom-right) understands your campaign context. Try:

- `"What is the best result so far?"` — Summarizes top findings
- `"Which parameters matter most?"` — Runs feature importance analysis
- `"Why did you switch to exploitation?"` — Explains strategy decisions with diagnostic signals
- `"Should I focus on temperature or pressure?"` — Compares parameter influence

**Quick Actions** (one-click buttons):
- **Discover** — Parameter importance ranking
- **Suggest** — Generate next batch of experiments
- **Show** — Display current diagnostics
- **Why** — Explain the current optimization strategy
- **Focus** — Narrow search to a specific region

### 4. Closed-Loop Optimization

1. Go to **Suggestions** tab → click **Generate Next Experiments**
2. Run the suggested experiments in your lab
3. Upload results (CSV or manual entry)
4. The platform re-diagnoses automatically — diagnostics update, phase may switch, new suggestions adapt

### 5. Demo Gallery

Click **Demos** in the nav bar to explore pre-built datasets:
- OER Catalyst Optimization
- Suzuki Coupling Yield
- Battery Electrolyte Formulation
- And more

---

## Architecture

```
optimization-copilot/
├── optimization_copilot/          # Python backend package
│   ├── core/                      # CampaignSnapshot, Observation, ParameterSpec
│   ├── diagnostics/               # 14-signal diagnostic engine
│   ├── backends/                  # 10+ optimization backends (GP-BO, TPE, CMA-ES, ...)
│   ├── meta_controller/           # Phase orchestration, strategy switching
│   ├── agents/                    # Scientific agent framework
│   ├── multi_objective/           # Pareto analysis, many-objective support
│   ├── explainability/            # Decision explanations
│   ├── safety/                    # Hazard classification, emergency protocols
│   ├── reproducibility/           # Campaign logging, replay, FAIR metadata
│   ├── api/                       # FastAPI routes (campaigns, chat, analysis)
│   ├── cli_app/                   # Click CLI commands
│   ├── web/                       # React frontend (see below)
│   └── ... (80+ subpackages)
│
├── optimization_copilot/web/      # React 18 + TypeScript + Vite
│   └── src/
│       ├── App.tsx                # Router, nav, theme toggle
│       ├── api.ts                 # Typed API client
│       ├── pages/
│       │   ├── Dashboard.tsx      # Campaign list, search, tags
│       │   ├── NewCampaign.tsx    # Upload wizard, column mapper
│       │   ├── Workspace.tsx      # 5-tab workspace (18,000+ lines)
│       │   ├── DemoGallery.tsx    # Pre-built datasets
│       │   └── ...
│       └── components/
│           ├── ChatPanel.tsx      # AI chat interface
│           ├── FileUpload.tsx     # CSV drag-and-drop
│           ├── ColumnMapper.tsx   # Visual column mapping
│           └── ...
│
├── tests/                         # 155+ test files, 6,300+ tests
├── pyproject.toml                 # Python package config
└── .github/workflows/             # CI/CD
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, FastAPI, Pydantic v2 |
| Frontend | React 18, TypeScript strict, Vite 5 |
| Styling | CSS custom properties, JetBrains Mono, light/dark themes |
| Optimization | 10+ backends: GP-BO, TPE, CMA-ES, NSGA-II, MOBO, and more |
| Intelligence | 14-signal diagnostic engine, parameter importance, insight generation |
| Visualizations | 148+ inline SVG micro-visualizations (no charting library dependencies) |
| AI Chat | Claude API integration for natural-language experiment guidance |
| Testing | pytest (backend), 155 test files |

### Key Design Decisions

- **Pure inline SVG** — All 148+ visualizations are hand-crafted SVG, computed client-side from raw trial data. No dependency on D3, Chart.js, or other charting libraries.
- **Client-side computation** — k-NN, k-means clustering, cosine similarity, Pearson correlation, PCA, variance decomposition — all run in the browser.
- **5-second auto-refresh** — The workspace polls the API every 5 seconds for live updates.
- **Zero ML framework dependency** — The Python backend implements all optimization algorithms from scratch. No scikit-learn, PyTorch, or TensorFlow required.

---

## CLI Reference

```bash
# Start the API server
nexus server start [--host 0.0.0.0] [--port 8000]

# Campaign management
nexus campaign list
nexus campaign create --name "My Campaign" --data path/to/data.csv
nexus campaign status <campaign-id>

# Meta-learning
nexus meta fingerprint <campaign-id>
nexus meta similar <campaign-id>

# Store operations
nexus store list
nexus store export <campaign-id> --format csv
```

## API Endpoints

The backend exposes a REST API at `http://localhost:8000/api`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/campaigns` | List all campaigns |
| `POST` | `/campaigns` | Create a campaign |
| `POST` | `/campaigns/from-upload` | Create from CSV upload |
| `GET` | `/campaigns/:id` | Get campaign details |
| `GET` | `/campaigns/:id/diagnostics` | Get diagnostic signals |
| `GET` | `/campaigns/:id/importance` | Get parameter importance |
| `GET` | `/campaigns/:id/suggestions` | Get next experiment suggestions |
| `GET` | `/campaigns/:id/insights` | Get AI-generated insights |
| `POST` | `/chat/:id` | Send a chat message |
| `POST` | `/loop` | Create optimization loop |
| `POST` | `/loop/:id/iterate` | Run one iteration |
| `POST` | `/loop/:id/ingest` | Feed back results |
| `GET` | `/demo-datasets` | List demo datasets |
| `GET` | `/reports/:id/audit` | Get audit trail |
| `POST` | `/analysis/fanova` | Run fANOVA analysis |

Full API docs available at `http://localhost:8000/docs` (Swagger UI) when the server is running.

---

## MCP Server (Model Context Protocol)

Nexus ships as an MCP server so any LLM client (Claude Desktop, Claude Code, etc.) can drive optimization campaigns through tool calls.

### Setup

```bash
# Install MCP dependency
pip install -e ".[mcp]"

# Make sure the Nexus backend is running
nexus server start
```

### Add to Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nexus": {
      "command": "python",
      "args": ["/absolute/path/to/nexus_mcp_server.py"],
      "env": {
        "NEXUS_URL": "http://localhost:8000"
      }
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `nexus_create_campaign` | Create a campaign from CSV data with parameter/objective mapping |
| `nexus_get_diagnostics` | Get 8+ real-time health signals (convergence, noise, coverage, etc.) |
| `nexus_suggest_next` | Generate next batch of experiments with predicted outcomes |
| `nexus_ingest_results` | Feed back experimental results to update the model |
| `nexus_explain_decision` | Ask natural language questions about optimization decisions |
| `nexus_causal_discovery` | Run PC algorithm to find causal relationships in data |
| `nexus_hypothesis_status` | Track scientific hypotheses through their lifecycle |

### Example Conversation

With the MCP server connected, you can ask Claude:

> "Create a campaign from my synthesis data, then suggest the next 5 experiments"

Claude will call `nexus_create_campaign` with your CSV data, then `nexus_suggest_next` to get optimized experiment parameters.

> "Why did Nexus switch to exploitation mode?"

Claude calls `nexus_explain_decision` and gets back diagnostic-backed reasoning.

> "Are temperature and pressure causally linked to yield, or just correlated?"

Claude calls `nexus_causal_discovery` to run the PC algorithm and reports back the causal graph.

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_api.py

# Run tests matching a pattern
pytest -k "diagnostic"
```

### Code Quality

```bash
# Lint with ruff
ruff check optimization_copilot/

# Type checking
cd optimization_copilot/web && npx tsc --noEmit
```

### Building the Frontend

```bash
cd optimization_copilot/web
npm run build    # Production build → dist/
npm run preview  # Preview production build
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MODEL_API_KEY` | Optional | Anthropic API key for AI chat features |
| `ANTHROPIC_API_KEY` | Optional | Alternative API key variable |
| `OCP_WORKSPACE` | Optional | Workspace directory path (default: `./workspace`) |

The platform works without an API key — AI chat features will return basic analytical responses instead of LLM-powered ones.

---

Key stats:
- **364 Python source files** across 80+ subpackages
- **155 test files** with 6,300+ tests
- **148+ inline SVG visualizations** across 5 workspace tabs
- **10+ optimization backends** implemented from scratch
- **14-signal diagnostic engine** for real-time campaign health monitoring
- **Zero external ML dependencies** — pure Python implementations

---

## License

MIT License. See [LICENSE](LICENSE) for details.
