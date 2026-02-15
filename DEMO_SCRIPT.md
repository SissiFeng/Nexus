# Optimization Copilot — 3-Minute Demo Script

## Setup
- Open browser to `http://localhost:5173`
- Light theme enabled (default)

---

## Act 1: Landing & Campaign Creation (0:00–0:40)

**Dashboard** — Show the clean landing page with campaign cards, search, and tag filters.

> "This is Optimization Copilot — an intelligent platform that helps scientists run Bayesian optimization campaigns without writing code."

Click **"New Campaign"** in the nav bar.

> "Creating a new campaign is simple. Upload a CSV of your experimental data, map columns to parameters and objectives, set bounds, and hit Create."

Show the **column mapper** — drag columns to parameter/objective roles, set min/max bounds.

Click **"Create Campaign"** → redirects to the Workspace.

---

## Act 2: The Workspace — Overview Tab (0:40–1:20)

> "The workspace is the command center. The Overview tab gives you an instant health check of your optimization."

Scroll through the **diagnostic cards** — convergence trend, exploration coverage, failure rate, signal-to-noise. Each has a colored status dot (green/yellow/red) and sparkline.

> "Every metric has a traffic-light indicator. Green means healthy, yellow means watch, red means act."

Point out a few **inline micro-visualizations**:
- **Budget Efficiency Curve** — are we spending trials wisely?
- **Optimization Momentum** — is progress accelerating or stalling?
- **Acquisition Collapse Gauge** — is the search narrowing too fast?
- **Phase ROI Breakdown** — which phase of the campaign delivered the most improvement?

> "These aren't static numbers. They're live computations on your data, updating every 5 seconds."

---

## Act 3: Explore Tab — Understanding the Search Space (1:20–1:50)

Click the **Explore** tab.

> "The Explore tab helps you understand where you've been searching and what the landscape looks like."

Highlight:
- **Parallel Coordinates** — multi-dimensional parameter visualization
- **Parameter Boundary Utilization** — are you exploring the full range or stuck in one corner?
- **Local Optima Map** — cluster analysis showing how many distinct good regions exist
- **Concept Drift Detector** — has the optimal region shifted over time?

> "A researcher told me: 'I never knew my optimization was stuck in one corner of the space.' These visualizations make that immediately obvious."

---

## Act 4: Suggestions Tab — What to Try Next (1:50–2:20)

Click the **Suggestions** tab. Click **"Generate Next Experiments"**.

> "The system suggests the next batch of experiments to run. But we don't just give you numbers — we give you context."

Show:
- **Suggestion cards** with parameter values and predicted outcomes
- **Suggestion Redundancy Check** — are the suggestions diverse or overlapping?
- **Suggestion Novelty Score** — how different is each suggestion from what you've already tried?
- **Risk-Return Profile** — expected improvement vs. uncertainty

> "You can bookmark suggestions, compare them side-by-side, and export them for your lab notebook."

---

## Act 5: History Tab — Learning from the Past (2:20–2:50)

Click the **History** tab.

> "The History tab is your campaign's memory. Every trial, every decision, every pattern."

Show:
- **Decision Journal** — auto-generated narrative of key campaign events
- **Improvement Source Decomposition** — were improvements from genuine discoveries or noise exploitation?
- **Learning Curve Projection** — how many more trials until convergence?
- **Hypothesis Rejection Timeline** — which structural assumptions were tested and rejected?

> "This is what separates a tool from a copilot. It doesn't just run optimization — it helps you understand what happened and why."

---

## Act 6: Wrap-Up (2:50–3:00)

Navigate back to **Dashboard**.

> "Optimization Copilot: 148+ inline visualizations, real-time diagnostics, zero-code setup. Built for scientists who want to understand their optimization, not just run it."

---

## Key Stats for Q&A
- **148+ micro-visualizations** across 4 tabs (37 batches of 4)
- **37 batch commits** on the feature branch
- **Pure inline SVG** — no external charting libraries
- **Client-side computation** — k-NN, clustering, cosine similarity, variance decomposition, Pearson correlation
- **5-second auto-refresh** with real-time data
- **Light/dark theme** with CSS custom properties
- **JetBrains Mono** typography for a scientific aesthetic
