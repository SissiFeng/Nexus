# OptimizationAgent (Optimization Copilot / Wizard)

> **Role:** Intelligent optimization decision layer for otbot  
> **Layer:** L2 (parallel to PlannerAgent / DesignAgent)  
> **Replacement Target:** `strategy_selector.select_strategy()`  
> **Core Goal:** Automatically select, adapt, and justify optimization strategies based on experimental history — reproducibly, safely, and cross-domain.

---

## 1. System Positioning

### Responsibilities
- Choose optimization backend per campaign state
- Adapt strategies over time (exploration → exploitation → restart)
- Stabilize noisy/failed experimental data
- Maximize convergence efficiency under real-world constraints

### Explicit Non-Responsibilities
- ❌ Workflow scheduling (PlannerAgent)
- ❌ Candidate generation mechanics (DesignAgent)
- ❌ Stopping logic (StopAgent)
- ❌ Hardware control
- ❌ Database querying (handled by QueryAgent)

---

## 2. Interfaces (No Schema Changes)

### Input
`CampaignSnapshot`
- Complete parameter history
- KPI history
- QC flags
- failure metadata
- timestamps & iteration context

### Output
`StrategyDecision`
- backend_name
- stabilize_spec
- exploration strength
- batch control hints
- optional risk posture
- decision metadata

---

## 3. Determinism & Reproducibility Layer

### Core Principle

StrategyDecision = f(
CampaignSnapshot,
diagnostics_vector,
seed,
agent_version,
backend_versions
)

### Guarantees
- Identical input → identical decision
- Version-pinned behavior
- Full audit trail

### Logged Artifacts
- snapshot_hash
- diagnostics_hash
- decision_hash
- reason_codes
- fallback events

---

## 4. Diagnostic Signal Engine (Existing Inputs)

OptimizationAgent consumes:

- convergence trends
- improvement velocity
- variance contraction
- noise estimates
- failure rate & clustering
- feasibility shrinkage
- parameter drift
- model uncertainty proxies
- exploration coverage metrics

(14 total signals)

---

## 5. Problem Profiler

Automatically classifies optimization context:

| Dimension | Detected Values |
|----------|----------------|
| Variable types | continuous / discrete / categorical / mixed |
| Objective form | single / multi-objective / constrained |
| Noise regime | low / medium / high |
| Cost profile | uniform / heterogeneous |
| Failure informativeness | weak / strong |
| Data scale | tiny / small / moderate |
| Dynamics | static / time-series |
| Feasible region | wide / narrow / fragmented |

Outputs `ProblemFingerprint`.

---

## 6. Optimization Backend Pool (Plugin-Based)

### Cold Start / Design of Experiments
- Random sampling
- Latin Hypercube Sampling (LHS)
- Sobol / Halton sequences
- Factorial / fractional designs

### Surrogate-Based Optimization
- KNN surrogate + EI/UCB
- TPE (Tree Parzen Estimator)
- Random Forest surrogate + EI
- Simplified GP (optional)

### Online Learning / Bandits
- UCB
- Thompson Sampling
- Successive Halving / Hyperband

### Evolutionary & Heuristic
- CMA-ES
- Genetic Algorithms / NSGA-II
- Local search
- Simulated annealing

### Multi-fidelity
- proxy screening → high-precision refinement
- staged scheduling

---

## 7. Meta-Controller (Core Intelligence)

### Phase Orchestration

1. **Cold Start**
   - space coverage
   - noise estimation
   - feasibility probing

2. **Learning Phase**
   - surrogate fitting
   - structure discovery
   - interaction detection

3. **Exploitation Phase**
   - aggressive improvement

4. **Stagnation Handling**
   - restarts
   - strategy switching
   - search space reshaping

5. **Termination Coordination**
   - cooperates with StopAgent

---

### Switching Triggers

- improvement slope thresholds
- uncertainty collapse
- feasibility failure spikes
- model residual anomalies
- coverage plateaus

---

## 8. Stabilization & Data Conditioning (`stabilize_spec`)

Structured policies:

- noise smoothing windows
- outlier rejection rules
- failure handling semantics
- censored data policies
- constraint tightening schedules
- reweighting strategies
- retry normalization

All validated before execution.

---

## 9. High-Dimensional Variable Screening

When user doesn’t know what matters:

### Techniques

- Morris one-at-a-time screening
- sparse regression (RSM-lite)
- tree-based importance
- feasibility boundary probing

### Outputs

- ranked active parameters
- suspected interactions
- directionality hints
- recommended step sizes

Dynamic re-screening when stagnation occurs.

---

## 10. Feasibility & Failure Learning

- failure as signal modeling
- safe-region mapping
- infeasible zone exclusion
- adaptive constraint enforcement

Supports:
- chemical safety
- hardware limits
- protocol stability

---

## 11. Multi-Objective Optimization

- Pareto front tracking
- dominance ranking
- preference weighting
- target band optimization
- tradeoff reporting

---

## 12. Backend Governance & Safety

### BackendPolicy

- allowlist / denylist
- capability matching
- fallback backends
- constraint compatibility

### Failure Handling

- deterministic rollback
- forced safe backend
- error code logging

---

## 13. Plugin Architecture

### Plugin Types

- AlgorithmPlugin
- SurrogatePlugin
- ScreeningPlugin
- FeasibilityModelPlugin
- StabilizationPolicyPlugin

### Contract

```python
fit(data)
suggest(state)
update(result)
capabilities()


⸻

14. Explainability Layer (Human Amplification)

Each decision reports:
	•	selected strategy
	•	triggering diagnostics
	•	phase transitions
	•	risk posture
	•	coverage status
	•	remaining uncertainty

Without hallucinated reasoning.

⸻

15. Cross-Domain Compatibility

Works across:
	•	electrochemistry
	•	synthesis optimization
	•	materials formulation
	•	polymer processing
	•	biological assays
	•	multi-step workflows

Via fingerprint + plugin abstraction.

⸻

16. Validation & Regression System

Golden Scenarios
	•	known convergence patterns
	•	failure-heavy regimes
	•	noise stress tests
	•	mixed-variable benchmarks

Metrics
	•	convergence speed
	•	regret proxy
	•	stability
	•	failure avoidance
	•	reproducibility

⸻

17. Deployment Characteristics
	•	pure function core
	•	async wrapper
	•	no hardware dependency
	•	deterministic runtime
	•	rollback safe

⸻

18. Long-Term Extensions (Optional)
	•	cross-campaign meta-learning
	•	optimizer performance leaderboard
	•	automatic backend pruning
	•	adaptive hyperparameter learning
	•	simulation-informed priors

⸻

Summary

OptimizationAgent is:

✔ deterministic
✔ explainable
✔ reproducible
✔ cross-domain
✔ safety-aware
✔ strategy-adaptive

It upgrades otbot from:

“runs optimization algorithms”

to:

“intelligently chooses and orchestrates optimization itself”

⸻

This is not a better BO.
This is an optimization brain.

