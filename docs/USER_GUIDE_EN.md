# Optimization Copilot — User Guide for Scientists

> An intelligent optimization agent that helps scientists design better experiments, discover hidden patterns, and accelerate research through AI-driven Bayesian optimization.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Creating a Campaign](#2-creating-a-campaign)
3. [Workspace Overview](#3-workspace-overview)
4. [AI Chat Agent](#4-ai-chat-agent)
5. [Insight Discovery](#5-insight-discovery)
6. [Experiment Suggestions](#6-experiment-suggestions)
7. [Diagnostics & Monitoring](#7-diagnostics--monitoring)
8. [Steering & Human-in-the-Loop](#8-steering--human-in-the-loop)
9. [Multi-Objective Optimization](#9-multi-objective-optimization)
10. [Data Export & Reproducibility](#10-data-export--reproducibility)
11. [Safety & Constraints](#11-safety--constraints)
12. [Advanced Analysis](#12-advanced-analysis)
13. [FAQ](#13-faq)

---

## 1. Quick Start

### What is Optimization Copilot?

Optimization Copilot is an AI-powered experiment optimization platform. You upload your experimental data, and the system:

- **Learns** the relationship between your parameters and outcomes
- **Suggests** the next best experiments to run
- **Discovers** hidden patterns, correlations, and optimal regions
- **Explains** why certain conditions work better than others
- **Steers** the search based on your expert knowledge

### 30-Second Workflow

```
Upload CSV  →  Map Columns  →  Create Campaign  →  Chat with Agent  →  Run Suggested Experiments  →  Repeat
```

1. Open the app at `http://localhost:5173`
2. Click **"+ New Campaign"** on the Dashboard
3. Upload your CSV file with experimental data
4. Map columns to parameters and objectives
5. Click **"Create Campaign"**
6. In the Workspace, chat with the AI: **"Discover insights from data"**
7. Ask: **"What should I try next?"** to get experiment suggestions

---

## 2. Creating a Campaign

### Uploading Data

Navigate to **New Campaign** (`/new-campaign`). You can upload data in CSV format.

**Required data structure:**

| Column Type | Description | Example |
|-------------|-------------|---------|
| **Parameters** | Input variables you control | temperature, pressure, catalyst_loading |
| **Objectives** | Output metrics you want to optimize | yield, selectivity, purity |
| **Metadata** (optional) | Labels or identifiers | sample_id, batch_number, operator |

### Column Mapping

After uploading, the system auto-detects column types:

- **Numeric columns** → detected as continuous parameters or objectives
- **Text columns** → detected as categorical parameters or metadata

You can manually adjust each column's role:

| Role | What it means |
|------|---------------|
| **Continuous Parameter** | Numeric input with min/max bounds (e.g., temperature 20–200 C) |
| **Categorical Parameter** | Discrete choices (e.g., solvent: ethanol, methanol, water) |
| **Objective (Minimize)** | Lower is better (e.g., cost, error, toxicity) |
| **Objective (Maximize)** | Higher is better (e.g., yield, efficiency, selectivity) |
| **Metadata** | Not used in optimization, kept for reference |
| **Ignored** | Excluded from the campaign entirely |

### Campaign Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Batch Size** | 5 | Number of experiments suggested per round |
| **Exploration Weight** | 0.5 | 0 = exploit known good regions, 1 = explore unknown regions |

---

## 3. Workspace Overview

After creating a campaign, you enter the **Workspace** — a 6-tab interface:

### Tab: Overview

- **Summary cards**: iteration count, total trials, best KPI, current phase
- **Convergence plot**: how your objective improves over iterations (blue = best so far, gray = individual trials)
- **Diagnostics cards**: 8 key health metrics at a glance
- **Phase timeline**: visual history of campaign phases (cold_start → learning → exploitation)

### Tab: Explore

- **Scatter matrix**: parameter-vs-parameter and parameter-vs-objective scatter plots
- Visual identification of clusters, trends, and outliers

### Tab: Suggestions

- **Next experiments**: AI-generated suggestions with predicted values and uncertainty
- **Suggestion cards**: ranked by acquisition score (balancing expected improvement vs exploration)

### Tab: Insights

- **Full insight panel**: all discovered patterns organized into sections (see [Section 5](#5-insight-discovery))

### Tab: History

- **Trial table**: complete history of all experiments with parameters, objectives, and status
- Sortable, filterable, searchable

### Tab: Export

- Download your data as **CSV**, **JSON**, or **XLSX**
- Includes all parameters, objectives, metadata, QC status, and timestamps

### Chat Panel

A persistent chat panel on the right side of the Workspace. Click the chat bubble icon or use it directly. Quick action buttons at the bottom provide one-click access to common queries.

---

## 4. AI Chat Agent

The chat agent understands natural language questions about your campaign. Here's what you can ask:

### Discover Insights
> **"Discover insights from data"**
> **"What patterns do you see?"**
> **"Analyze my data"**

Returns rich structured insights with color-coded cards:
- **Discovery** (blue): key findings about best conditions
- **Recommendation** (green): actionable suggestions for optimal ranges
- **Warning** (orange): risk zones and failure patterns
- **Trend** (purple): convergence and exploration trends

Plus: correlation bars, top conditions table, optimal parameter ranges, and parameter interactions.

### Get Suggestions
> **"Suggest next parameter values"**
> **"What should I try next?"**
> **"Recommend 10 experiments"**

Returns a ranked list of suggested experiments with parameters and predicted outcomes.

### View Diagnostics
> **"Show diagnostics"**
> **"How is my campaign doing?"**
> **"Is the optimization converging?"**

Returns 8 health metrics: convergence trend, improvement velocity, best KPI, exploration coverage, failure rate, noise estimate, plateau length, signal-to-noise ratio.

### Understand Importance
> **"Which parameter matters most?"**
> **"Show parameter importance"**
> **"What drives the objective?"**

Returns fANOVA-based importance scores for each parameter.

### Get Explanations
> **"Why did you recommend this?"**
> **"Why is it stuck?"**
> **"Explain the current phase"**

Provides human-readable explanations of the optimization state and rationale.

### Steering Instructions
> **"Focus on specific region"**
> **"How do I steer the search?"**

Guides you on how to use steering directives to focus or avoid certain parameter regions.

### Export Help
> **"Export results"**
> **"How do I download my data?"**

Provides information on available export formats and how to access them.

---

## 5. Insight Discovery

The insight engine performs 7 types of analysis on your data:

### 5.1 Top Performing Conditions

Identifies the best experiments and what they have in common.

**Example output:**
```
Best objective: -0.3402
Top results share: ni_load ≈ 0.633
```

### 5.2 Parameter-Objective Correlations

Computes Pearson correlation between each parameter and each objective.

| Parameter | Correlation | Strength | Direction |
|-----------|-------------|----------|-----------|
| ni_load   | +0.408     | moderate | positive  |
| mn_load   | -0.235     | weak     | negative  |

**Interpretation:** Positive correlation means increasing the parameter tends to improve the objective. Negative means the opposite.

### 5.3 Parameter Interactions

Detects synergistic and antagonistic effects between parameter pairs.

**Example:**
```
ni_load × fe_load: synergistic interaction (r=0.445)
```
This means adjusting ni_load and fe_load together has a stronger effect than adjusting either alone.

### 5.4 Optimal Parameter Ranges

Identifies the parameter ranges where your objective performs best.

**Example:**
```
mn_load: best in [0, 0.6] (overall [0, 1]) → +6.8% improvement
```
This means experiments with mn_load between 0 and 0.6 achieve 6.8% better results on average.

### 5.5 Failure Patterns

Identifies parameter ranges with elevated failure rates.

**Example:**
```
temperature > 180°C: failure rate 35% (vs 8% overall)
```

### 5.6 Trend Detection

Monitors optimization dynamics over time:

- **Convergence**: Is the optimization converging?
- **Variance reduction**: Are results becoming more consistent?
- **Exploration narrowing**: Is the search focusing on promising regions?
- **Acceleration**: Are recent iterations improving faster?

### 5.7 Natural Language Summaries

Synthesizes all insights into prioritized, human-readable statements with importance scores and categories (discovery / warning / recommendation / trend).

---

## 6. Experiment Suggestions

### How Suggestions Work

The system uses **Bayesian Optimization** to suggest experiments:

1. **Surrogate model**: A statistical model (Gaussian Process or Random Forest) learns the relationship between parameters and objectives from your data
2. **Acquisition function**: Balances exploitation (try near known good regions) vs exploration (try under-explored regions)
3. **Ranking**: Candidates are ranked by acquisition score
4. **Diversity**: Batch suggestions are diversified to avoid redundancy

### Available Optimization Algorithms

| Algorithm | Best for | Description |
|-----------|----------|-------------|
| **Gaussian Process BO** | Small-medium datasets, continuous parameters | Gold-standard Bayesian optimization |
| **Random Forest BO** | Mixed parameter types, noisy data | Robust to noise and categorical variables |
| **TPE** | High-dimensional problems | Tree-structured Parzen Estimator |
| **TuRBO** | Local optimization, exploitation | Trust-region Bayesian optimization |
| **CMA-ES** | Continuous parameters, no gradient | Evolutionary strategy |
| **NSGA-II** | Multi-objective optimization | Non-dominated sorting genetic algorithm |
| **Latin Hypercube** | Initial space-filling | Stratified random sampling |
| **Sobol** | Initial space-filling | Quasi-random low-discrepancy sequence |

The system **automatically selects** the best algorithm based on your data characteristics (number of parameters, parameter types, noise level, dataset size).

### Adjusting Exploration vs Exploitation

- **Exploration weight = 0**: Focus entirely on known good regions (exploitation)
- **Exploration weight = 0.5**: Balanced (default)
- **Exploration weight = 1**: Focus on under-explored regions (exploration)

---

## 7. Diagnostics & Monitoring

### Campaign Phases

Your campaign automatically progresses through phases:

| Phase | Description | What happens |
|-------|-------------|-------------|
| **Cold Start** | Not enough data yet | Space-filling sampling (LHS/Sobol) |
| **Learning** | Building surrogate model | Active learning with balanced exploration |
| **Exploitation** | Converging on best region | Focused search near optima |
| **Stagnation** | No recent improvement | May trigger strategy switch or exploration boost |
| **Termination** | Campaign complete | Final results available |

### Diagnostic Metrics

| Metric | What it tells you | Good value |
|--------|-------------------|------------|
| **Convergence Trend** | Rate of improvement per iteration | > 0 (improving) |
| **Improvement Velocity** | Recent improvement speed | > 0 |
| **Best KPI** | Best objective value found | Depends on your problem |
| **Exploration Coverage** | Fraction of parameter space explored | 30–80% |
| **Failure Rate** | Proportion of failed experiments | < 20% |
| **Noise Estimate** | Measurement noise level | Lower is better |
| **Plateau Length** | Iterations without improvement | < 20 is normal |
| **Signal-to-Noise** | Signal magnitude vs noise | > 3 is good |

---

## 8. Steering & Human-in-the-Loop

### Steering the Optimization

You can guide the search based on your domain expertise:

**Focus on a region** — Tell the optimizer to concentrate on specific parameter ranges:
```
POST /api/campaigns/{id}/steer
{
  "action": "focus_region",
  "region_bounds": {
    "temperature": [150, 200],
    "pressure": [1.0, 3.0]
  },
  "reason": "Literature suggests this range is most promising"
}
```

**Avoid a region** — Exclude unsafe or unproductive parameter ranges:
```
{
  "action": "avoid_region",
  "region_bounds": {
    "temperature": [250, 300]
  },
  "reason": "Equipment cannot safely operate above 250C"
}
```

### Expert Prior Injection

You can inject domain knowledge as prior beliefs:

- **Soft bounds**: Prefer certain parameter ranges without hard exclusion
- **Region focus**: Weight exploration toward known promising regions
- **Parameter relationships**: Encode known dependencies

### Progressive Autonomy

The system supports graduated autonomy levels:

| Level | Human role | AI role |
|-------|-----------|---------|
| **Manual** | Human decides all experiments | AI provides analysis only |
| **Supervised** | AI suggests, human approves each batch | AI generates suggestions |
| **Collaborative** | AI decides, human can veto | AI runs with oversight |
| **Autonomous** | Human monitors only | AI runs fully autonomously |

Trust increases automatically when AI suggestions lead to good outcomes.

---

## 9. Multi-Objective Optimization

For problems with multiple objectives (e.g., maximize yield AND minimize cost):

### Pareto Front

The system tracks the **Pareto front** — the set of solutions where no objective can be improved without worsening another.

### Interactive Exploration

- **Weighted query**: Specify importance weights for each objective
- **Aspiration levels**: Set target values for each objective
- **Ideal point**: Find the solution nearest to your ideal
- **Tradeoff analysis**: Understand the slope of tradeoffs between objectives

### Many-Objective Problems (>3 objectives)

For problems with more than 3 objectives, the system uses:
- **Hypervolume indicator**: Quality measure for Pareto fronts
- **IGD metric**: Inverted Generational Distance
- **Hypervolume contribution ranking**: Discriminating ranking when most points are non-dominated

---

## 10. Data Export & Reproducibility

### Export Formats

| Format | Contents | Use case |
|--------|----------|----------|
| **CSV** | Parameters, objectives, metadata, timestamps | Spreadsheet analysis |
| **JSON** | Full campaign structure with schema | Programmatic access |
| **XLSX** | Excel-compatible tabular data | Sharing with collaborators |

### Audit Trail

Every decision is logged with a cryptographic hash chain:
- Campaign creation, parameter changes
- Each suggestion batch with algorithm used
- Each observation submitted
- Strategy switches and phase transitions
- Steering directives applied

### Reproducibility

- **Deterministic execution**: Given the same seed and data, the system produces identical suggestions
- **Campaign replay**: Re-execute a logged campaign to verify reproducibility
- **FAIR metadata**: Generate Findable, Accessible, Interoperable, Reusable metadata for your datasets

---

## 11. Safety & Constraints

### Defining Safe Operating Ranges

For each parameter, you can define safety limits:

| Hazard Level | Meaning |
|-------------|---------|
| **Green** | Safe — normal operating range |
| **Yellow** | Caution — approaching limits |
| **Red** | Danger — exceeds safe range |

### Safety Categories

The system supports hazard categories relevant to laboratory work:
- Thermal, Pressure, Toxicity, Flammability
- Corrosion, Reactivity, Electrical, Mechanical

### Emergency Protocol

If proposed experiments approach unsafe regions:

1. **Warning**: Alert the user, continue with monitoring
2. **Pause**: Halt suggestions until human review
3. **Fallback**: Revert to known safe parameter region
4. **Stop**: Immediately halt campaign

---

## 12. Advanced Analysis

### Available through the Analysis API:

| Analysis | Description | When to use |
|----------|-------------|-------------|
| **fANOVA** | Feature importance via functional ANOVA | Identify which parameters matter most |
| **Correlation** | Pearson correlation coefficient | Quick relationship check |
| **Outlier Detection** | N-sigma statistical outlier identification | Quality control |
| **Symbolic Regression** | Auto-discover governing equations | Find mathematical relationships |
| **Pareto Analysis** | Multi-objective front computation | Multi-objective problems |
| **Causal Discovery** | Learn causal structure from data | Understand cause-effect |
| **Intervention Analysis** | Simulate parameter interventions | "What if I change X?" |
| **Counterfactual Reasoning** | "What would have happened if...?" | Retrospective analysis |
| **Physics-Constrained GP** | Surrogate with physical constraints | Domain-informed modeling |
| **Hypothesis Generation** | Auto-generate scientific hypotheses | Discovery |
| **Hypothesis Testing** | Statistical validation of hypotheses | Verification |
| **Bootstrap CI** | Confidence intervals via bootstrapping | Uncertainty quantification |
| **Robustness Testing** | How robust are your conclusions? | Sensitivity analysis |

---

## 13. FAQ

### How much data do I need to start?

- **Minimum**: 3 observations (but suggestions will be exploratory)
- **Good**: 10–20 observations (surrogate model starts learning)
- **Great**: 50+ observations (reliable importance and interaction detection)

### What if I have categorical parameters?

Fully supported. The system uses one-hot encoding internally and selects algorithms that handle mixed parameter types (e.g., Random Forest BO, TPE).

### Can I add new data to an existing campaign?

Yes. Submit new trial results via the chat ("I have new results") or the API. The model updates incrementally.

### What if some experiments fail?

Failed experiments are informative! The system learns from failures to map unsafe or unproductive regions and avoid them in future suggestions.

### Can I compare multiple campaigns?

Yes. Use the **Compare** page (`/compare`) to compare campaigns side by side. Select 2+ campaigns to see which achieved better results.

### How do I know when to stop?

Watch these signals:
- **Plateau length** > 30 iterations → optimization may have converged
- **Convergence trend** ≈ 0 → no more improvement expected
- **Improvement velocity** ≈ 0 → diminishing returns

The chat agent will also warn you: "Optimization is converging — improvement rate has slowed significantly."

### Is my data safe?

- All data is stored locally on your server
- Cryptographic audit trail prevents tampering
- No data is sent to external services (except the AI chat, which uses the configured LLM API)

---

*Built with Optimization Copilot — accelerating scientific discovery through intelligent experiment design.*
