# Optimization Copilot — Capability Overview

> **One-line positioning:** An intelligent optimization decision layer that automatically selects, switches, and adjusts optimization strategies based on experimental history, with fully traceable decision explanations. Supports cross-domain generalization, compliance auditing, deterministic replay, automated data import, cross-project meta-learning, and rich visualization.

---

## Architecture Overview

```
  Raw Data (CSV/JSON)                 OptimizationSpec (DSL)
        |                                      |
        v                                      |
+-----------------------+                      |
| DataIngestionAgent    |                      |
| (auto-parse/infer)    |                      |
+----------+------------+                      |
           v                                   |
+-----------------------+   +----------------+ |
| ExperimentStore       |-->| ProblemBuilder |-+
| (unified storage)     |   | (guided setup) |
+----------+------------+   +----------------+
           |                       |
           v                       v
                    +-------------------------+
                    |   OptimizationEngine    |
                    | (full lifecycle engine) |
                    +-----------+-------------+
                                |
         CampaignSnapshot (experiment data)
                |
        +-------+--------+
        v                v
+--------------+  +---------------+  +----------------+
| Diagnostic   |  | Problem       |  | Data Quality   |
| Engine       |  | Profiler      |  | Engine         |
| (17 signals) |  | (8-dim print) |  | (noise/batch)  |
+------+-------+  +------+--------+  +-------+--------+
       |                 |                    |
       v                 v                    v
  +----------------------------------------------------+
  |              Meta-Controller                        |
  |  Phase detection -> Strategy -> Risk -> Dispatch    |
  |                                                     |
  |  Portfolio <-- Drift <-- NonStationary <-- Cost     |
  |       ^                                             |
  |       |  MetaLearningAdvisor (cross-project)        |
  |       |  Strategy + Weights + Thresholds + Drift    |
  +------------+------------+------------+--------------+
               |            |            |
      +--------+    +-------+    +-------+
      v             v            v
+----------+ +----------+ +--------------+
|Stabilizer| |Screener/ | |Feasibility   |
| (clean)  | |Surgery   | |First + Safety|
+----------+ |(screen)  | |(safe region) |
             +----------+ +--------------+
               |
               v
      +-----------------+     +------------------+
      |DecisionExplainer|     | Compliance Engine |
      |+ ExplanGraph    |     | Audit + Reports  |
      +-----------------+     +------------------+
               |                       |
               v                       v
         DecisionLog  ----> ReplayEngine (deterministic)
                |
                v
        ExperienceStore (cross-campaign memory)
                |
                v
  +---------------------------------------------------+
  |            Platform Layer                          |
  |  FastAPI + WebSocket + CLI + React SPA             |
  +---------------------------------------------------+
                |
                v
  +---------------------------------------------------+
  |         Visualization & Analysis Layer             |
  |  VSUP colormaps | SHAP charts | SDL monitoring    |
  |  Design space (PCA/t-SNE/iSOM) | Hexbin coverage  |
  +---------------------------------------------------+
                |
                v
  +---------------------------------------------------+
  |         Campaign Engine Layer                      |
  |  Surrogate → Ranker → StageGate → Deliverable     |
  |  CampaignLoop (closed-loop iteration)              |
  +---------------------------------------------------+
                |
                v
  +---------------------------------------------------+
  |         Agent Layer (Code Execution Enforcement)   |
  |  ExecutionTrace | DataAnalysisPipeline | Guard     |
  |  TracedScientificAgent | Orchestrator              |
  +---------------------------------------------------+
```

---

## Module Inventory (450+ modules across 87 packages)

### I. Core Intelligence

#### 1. Core Data Models (`core/`)

| Data Structure | Description |
|----------------|-------------|
| `CampaignSnapshot` | Complete snapshot of an optimization campaign: parameter specs, observation history, objective definitions |
| `StrategyDecision` | Output decision: backend selection, exploration strength, batch size, risk posture, audit trail |
| `ProblemFingerprint` | 8-dimensional problem fingerprint for automatic problem classification |
| `Observation` | Single experimental observation (parameters, KPIs, failure flag, timestamp) |
| `StabilizeSpec` | Data preprocessing strategy (denoising, outlier handling, failure treatment) |

**Deterministic hashing:** `snapshot_hash()` / `decision_hash()` / `diagnostics_hash()` — same input + same seed = identical output; every decision is traceable.

---

#### 2. Diagnostic Signal Engine (`diagnostics/`)

Computes **17 health signals** in real-time from experimental history:

| # | Signal | Meaning |
|---|--------|---------|
| 1 | `convergence_trend` | Best-value convergence trend (linear regression slope) |
| 2 | `improvement_velocity` | Recent improvement rate vs. historical rate |
| 3 | `variance_contraction` | Whether KPI variance is shrinking (convergence indicator) |
| 4 | `noise_estimate` | Recent KPI noise level (coefficient of variation) |
| 5 | `failure_rate` | Proportion of failed experiments |
| 6 | `failure_clustering` | Whether failures are concentrated in recent trials |
| 7 | `feasibility_shrinkage` | Whether the feasible region is shrinking |
| 8 | `parameter_drift` | Whether optimal parameters are still drifting |
| 9 | `model_uncertainty` | Proxy metric for model uncertainty |
| 10 | `exploration_coverage` | Parameter space exploration coverage |
| 11 | `kpi_plateau_length` | How many rounds KPI has been stagnant |
| 12 | `best_kpi_value` | Current best KPI value |
| 13 | `data_efficiency` | Average improvement per experiment |
| 14 | `constraint_violation_rate` | Constraint violation rate |
| 15 | `miscalibration_score` | UQ calibration error (ECE) |
| 16 | `overconfidence_rate` | Proportion of overly narrow prediction intervals |
| 17 | `signal_to_noise_ratio` | KPI signal-to-noise ratio (|mean|/std) |

---

#### 3. Problem Fingerprint Profiler (`profiler/`)

Automatically infers **8 dimensions** from data:

| Dimension | Categories | Basis |
|-----------|------------|-------|
| Variable type | continuous / discrete / categorical / mixed | Parameter specs |
| Objective form | single / multi_objective / constrained | Objective count & constraints |
| Noise level | low / medium / high | KPI coefficient of variation |
| Cost distribution | uniform / heterogeneous | Timestamp interval analysis |
| Failure informativeness | weak / strong | Parameter diversity of failure points |
| Data scale | tiny(<10) / small(<50) / moderate(50+) | Observation count |
| Temporal characteristics | static / time_series | Lag-1 autocorrelation |
| Feasible region | wide / narrow / fragmented | Failure rate |

**Continuous vector encoding**: `to_continuous_vector()` maps the 8 enum dimensions to [0,1] via domain-aware ordinal encoding (e.g., noise: low=0.0, medium=0.5, high=1.0) plus normalized effective dimensionality — enabling RBF kernel similarity for cross-campaign transfer learning.

---

#### 4. Meta-Controller (`meta_controller/`)

**Core intelligence module** — five-phase automatic orchestration:

```
Cold Start --> Learning --> Exploitation
                  |               |
                  v               v
             Stagnation <---------+
                  |
                  v
            Termination
```

| Phase | Trigger | Exploration | Risk Posture | Recommended Backends |
|-------|---------|-------------|--------------|---------------------|
| **Cold Start** | Observations < 10 | 0.9 (high) | Conservative | LHS, Random |
| **Learning** | Sufficient data, still learning | 0.6 (balanced) | Moderate | TPE, RF Surrogate |
| **Exploitation** | Strong convergence + low uncertainty | 0.2 (exploit) | Aggressive | TPE, CMA-ES |
| **Stagnation** | Long KPI plateau / failure surge | 0.8 (restart) | Conservative | Random, LHS |
| **Termination** | Collaborative termination judgment | 0.1 | — | TPE |

**Adaptive features:**
- Auto-adjusts backend priority based on problem fingerprint (e.g., high noise -> prioritize Random/LHS)
- Automatic fallback when preferred backend is unavailable, with event logging
- Exploration strength dynamically tuned by coverage, noise, and data scale
- Accepts `backend_policy` (allowlist/denylist) for governance constraints
- Accepts `drift_report` and `cost_signals` for adaptive adjustments

---

### II. Algorithm Management

#### 5. Optimization Backend Pool (`backends/` + `plugins/`)

**Plugin architecture** — extensible algorithm registry with **10 built-in backends**:

| Backend | Purpose | Key Features |
|---------|---------|--------------|
| `RandomSampler` | Baseline / Cold Start | Uniform random sampling |
| `LatinHypercubeSampler` | Space-filling design | Stratified sampling with good coverage |
| `TPESampler` | History-informed optimization | Good/bad split Bayesian method |
| `SobolSampler` | Quasi-random design | Low-discrepancy sequences (up to 21 dims) |
| `GaussianProcessBO` | Bayesian optimization | GP surrogate + RBF kernel + EI acquisition |
| `RandomForestBO` | SMAC-style optimization | Ensemble of decision trees; full variable support |
| `CMAESSampler` | Evolution strategy | Covariance matrix adaptation; rank-1/rank-mu updates |
| `DifferentialEvolution` | Population-based optimization | Classic DE/rand/1/bin; persistent population |
| `NSGA2Sampler` | Multi-objective optimization | Non-dominated sort + crowding distance |
| `TuRBOSampler` | Trust-region Bayesian | Dynamic trust region shrink/grow per success/failure |

```python
class AlgorithmPlugin(ABC):
    def name(self) -> str: ...
    def fit(self, observations, parameter_specs) -> None: ...
    def suggest(self, n_suggestions, seed) -> list[dict]: ...
    def capabilities(self) -> dict: ...
```

Supports `BackendPolicy` (allowlist / denylist) governance.

---

#### 6. Algorithm Portfolio Learner (`portfolio/`)

| Class | Description |
|-------|-------------|
| `AlgorithmPortfolio` | Records per-backend historical performance by fingerprint, provides ranking |
| `BackendScorer` | Multi-dimensional weighted scoring: history + fingerprint match + incompatibility penalty + cost signals |
| `BackendScore` | Score breakdown (per-dimension contribution), supports explainable backend selection |
| `ScoringWeights` | Scoring weight configuration (history, fingerprint_match, incompatibility, cost) |

---

#### 7. Plugin Marketplace (`marketplace/`)

| Class | Description |
|-------|-------------|
| `Marketplace` | Plugin registry + health tracking + automatic culling |
| `CullPolicy` | Culling policy (auto-delist unhealthy plugins by failure rate / performance) |
| `MarketplaceStatus` | Plugin status (active / probation / culled) |

---

#### 8. Pipeline Composer (`composer/`)

| Class | Description |
|-------|-------------|
| `AlgorithmComposer` | Select and orchestrate multi-stage optimization pipelines |
| `PipelineStage` | Pipeline stage definition (backend, exit condition, transition rules) |
| `PIPELINE_TEMPLATES` | Built-in templates: `exploration_first`, `screening_then_optimize`, `restart_on_stagnation` |

---

### III. Data Processing

#### 9. Data Stabilizer (`stabilization/`)

| Strategy | Description |
|----------|-------------|
| Failure handling | `exclude` / `penalize` (retain with flag) / `impute` (fill with worst value) |
| Outlier removal | N-sigma rule-based automatic removal |
| Reweighting | `recency` (recent data weighted higher) / `quality` (high-quality data weighted higher) |
| Noise smoothing | Moving average window |

---

#### 10. Data Quality Engine (`data_quality/`)

| Class | Description |
|-------|-------------|
| `DataQualityReport` | Comprehensive data quality report |
| `NoiseDecomposition` | Noise decomposition: experimental noise vs. model noise vs. systematic drift |
| `BatchEffect` | Batch effect detection: systematic bias between experimental batches |
| `InstrumentDrift` | Instrument drift detection |

---

#### 11. Feature Extraction (`feature_extraction/`)

| Class | Description |
|-------|-------------|
| `FeatureExtractor` | ABC base class, extracts named scalar features from measurement curves |
| `CurveData` | Curve data container |
| `ExtractedFeatures` | Feature extraction results |
| `EISNyquistExtractor` | EIS impedance spectrum (solution resistance, polarization resistance, semicircle diameter, Warburg slope) |
| `UVVisExtractor` | UV-Vis spectrum (peak position, peak height, FWHM, total absorbance) |
| `XRDPatternExtractor` | XRD pattern (main peak angle, peak count, crystallinity index, background level) |
| `VersionLockedRegistry` | Registry with allowlist/denylist + version locking |
| `CurveEmbedder` | PCA dimensionality reduction: curve -> low-dimensional latent vector |
| `check_extractor_consistency()` | Consistency verification: same data + same version -> same KPI |

---

#### 12. Variable Screener (`screening/`)

- **Importance ranking**: Parameter-KPI correlation-based importance scoring
- **Direction hints**: Positive/negative influence per parameter
- **Interaction detection**: Product-term correlation for parameter interactions
- **Step-size recommendation**: Auto-recommended search step based on importance

---

#### 13. Parameter Surgeon (`surgery/`)

| Class | Description |
|-------|-------------|
| `Surgeon` | Diagnose and execute dimensionality reduction based on screening results |

---

#### 14. Latent Space Reduction (`latent/`)

| Class | Description |
|-------|-------------|
| `LatentTransform` | Pure stdlib PCA via power iteration (no numpy) |
| `LatentOptimizer` | Control when/how to apply reduction (auto-triggered for high-dimensional problems) |

---

### IV. Adaptive Intelligence

#### 15. Drift Detector (`drift/`)

| Class | Description |
|-------|-------------|
| `DriftDetector` | Multi-strategy concept drift detection (KPI step change, parameter-KPI correlation shift, residual analysis) |
| `DriftReport` | Drift report (`drift_detected` flag + per-dimension detection results) |
| `DriftStrategyAdapter` | Map drift signals to strategy adjustments (phase reset, exploration boost, etc.) |
| `DriftAction` | Specific drift response actions |

**Two-layer false positive control:**
- **Detector FP < 15%**: Infrequent false alarms on stationary data
- **Action FP < 5%**: Drift false positives do not cause actual strategy changes

---

#### 16. Non-Stationary Adaptation (`nonstationary/`)

| Class | Description |
|-------|-------------|
| `NonStationaryAdapter` | Integrates time weighting + seasonal detection + drift signals |
| `SeasonalDetector` | Autocorrelation-based periodic pattern detection |
| `TimeWeighter` | Time-decay observation weighting |

---

#### 17. Curriculum Learning Engine (`curriculum/`)

| Class | Description |
|-------|-------------|
| `CurriculumEngine` | Progressive difficulty management, parameter ranking + gradual search range expansion |

---

### V. Safety & Feasibility

#### 18. Feasibility Learner (`feasibility/`)

| Class | Description |
|-------|-------------|
| `FeasibilityLearner` | Learn safe regions, safety boundaries, and danger zones from failure data |
| `FailureSurface` | Failure probability surface: estimate safety boundaries and danger zones |
| `FailureClassifier` | Structured failure taxonomy (hardware / chemistry / data / protocol / unknown) |
| `FailureTaxonomy` | Failure classification results + statistical distribution |

---

#### 19. Safety-First Scoring (`feasibility_first/`)

| Class | Description |
|-------|-------------|
| `SafetyBoundaryLearner` | Learn conservative safety boundaries via quantile estimation |
| `FeasibilityClassifier` | KNN classifier predicting candidate feasibility + confidence score |
| `FeasibilityFirstScorer` | Adaptive blend of feasibility and objective scores (dynamically weighted by failure rate) |

---

#### 20. Constraint Discovery (`constraints/`)

| Class | Description |
|-------|-------------|
| `ConstraintDiscoverer` | Discover implicit constraints from optimization history (threshold detection + interaction detection) |
| `DiscoveredConstraint` | Discovered constraint description |
| `ConstraintReport` | Constraint discovery report |

---

### VI. Multi-Objective & Preference

#### 21. Multi-Objective Optimization (`multi_objective/`)

- **Pareto front detection**: Non-dominated sorting to find current optimal set
- **Dominance ranking**: Layer-by-layer ranking (Rank 1 = Pareto front)
- **Trade-off analysis**: Correlation analysis between objective pairs (conflict / synergy / independent)
- **Weighted scoring**: User-defined weight scalarization
- **Many-objective optimization** (>3 objectives):
  - `HypervolumeIndicator`: 2D exact sweep, 3D+ Monte Carlo approximation
  - `IGDMetric`: Inverted Generational Distance for Pareto front quality
  - `ManyObjectiveRanker`: Rank by hypervolume contribution
- **Interactive Pareto exploration**:
  - `InteractiveParetoExplorer`: Weighted queries, aspiration levels, bounds filtering
  - `nearest_to_ideal()`: Normalized Euclidean search
  - `tradeoff_analysis()`: Slope, Pearson correlation, elasticity between objective pairs

---

#### 22. Preference Learning (`preference/`)

| Class | Description |
|-------|-------------|
| `PreferenceLearner` | Learn utility scores from pairwise preferences (Bradley-Terry MM algorithm) |

---

### VII. Efficiency & Cost

#### 23. Cost-Aware Analysis (`cost/`)

| Class | Description |
|-------|-------------|
| `CostAnalyzer` | Cost-aware optimization analysis: budget pressure, efficiency metrics, exploration adjustment |
| `CostSignals` | Cost signals (spending, efficiency, budget remaining), fed into MetaController |

---

#### 24. Batch Diversification (`batch/`)

| Class | Description |
|-------|-------------|
| `BatchDiversifier` | Batch diversification strategies (maximin / coverage / hybrid) |
| `BatchPolicy` | Ensure within-batch parameter diversity, avoid redundant sampling |
| `AsyncBatchExecutor` | PipeBO-style async execution: submit/complete/poll lifecycle, `incorporate_partial()` re-ranks candidates as partial results arrive |
| `TimeAwareScheduler` | Time-aware scheduling: longest-job-first bin-packing to minimize wall-clock time, deadline-aware early stopping |

---

#### 25. Multi-Fidelity Planning (`multi_fidelity/`)

| Class | Description |
|-------|-------------|
| `MultiFidelityPlanner` | Two-stage optimization: cheap screening + expensive fine-tuning |
| `FidelityLevel` | Fidelity level definition |
| `MultiFidelityPlan` | Execution plan with successive halving |

---

### VIII. Explainability

#### 26. Decision Explainer (`explainability/`)

Generates **human-readable reports** for each strategy decision:

- **Summary**: Current phase + selected strategy + phase changes
- **Triggering diagnostics**: Which signals drove this decision
- **Phase transition explanation**: If phase changed, why
- **Risk assessment**: Current risk posture and reasoning
- **Coverage status**: How much of parameter space has been explored
- **Uncertainty assessment**: Confidence in the recommendation

**Principle:** Only report what the algorithm actually computed — no speculative explanations.

---

#### 27. Explanation Graph (`explanation_graph/`)

| Class | Description |
|-------|-------------|
| `GraphBuilder` | Build DAG-form explanation graph from diagnostics, decisions, and failure surfaces |

---

#### 28. Reasoning Explanation (`reasoning/`)

| Class | Description |
|-------|-------------|
| `RewriteSuggestion` | Templated human-readable explanations (surgery actions, campaign status) |
| `FailureCluster` | Failure cluster description |

---

#### 29. Sensitivity Analysis (`sensitivity/`)

| Class | Description |
|-------|-------------|
| `SensitivityAnalyzer` | Parameter sensitivity and decision stability analysis (correlation + distance metrics) |

---

### IX. Compliance & Governance

#### 30. Compliance Audit System (`compliance/`)

| Class | Description |
|-------|-------------|
| `AuditEntry` | Single audit record (converted from DecisionLogEntry) |
| `AuditLog` | Hash-chained audit log (each record links to the previous record's hash) |
| `verify_chain()` | Verify audit chain integrity (detect tampering) |
| `ChainVerification` | Chain verification result (valid / first_broken_index) |
| `ComplianceReport` | Structured compliance report: campaign summary, iteration logs, final recommendation, rule version |
| `ComplianceEngine` | High-level compliance orchestration: audit log + chain verification + report generation |

**Tamper-proof guarantee:** If any record is modified, `verify_chain()` pinpoints the first break location.

---

#### 31. Decision Rule Engine (`schema/`)

| Class | Description |
|-------|-------------|
| `DecisionRule` | Versionable, auditable decision rules |
| `RuleSignature` | Rule signature (for codifying MetaController judgment logic) |

---

#### 32. Decision Replay (`replay/`)

| Class | Description |
|-------|-------------|
| `DecisionLog` | Per-round audit trail: snapshot, diagnostics, decision, experiment results |
| `DecisionLogEntry` | Single log entry (14 fields), supports JSON serialization and file I/O |
| `ReplayEngine` | Deterministic replay engine, three modes: |
| | `VERIFY` — Verify historical decisions are reproducible |
| | `COMPARE` — Compare two strategies' historical choices |
| | `WHAT_IF` — Hypothetical analysis (what if a different strategy was used?) |

---

### X. Experiment Orchestration

#### 33. Declarative DSL (`dsl/`)

| Class | Description |
|-------|-------------|
| `OptimizationSpec` | Declarative optimization specification (parameters, objectives, budget, constraints) |
| `ParameterDef` | Parameter definition (name, type, bounds, categories) |
| `ObjectiveDef` | Objective definition (name, direction: minimize/maximize) |
| `BudgetDef` | Budget definition (max iterations, max time) |
| `SpecBridge` | DSL -> core model conversion (ParameterSpec, CampaignSnapshot, ProblemFingerprint) |
| `SpecValidator` | Spec validation + human-readable error messages |

Supports JSON serialization / deserialization with full `to_dict()` / `from_dict()` round-trip.

---

#### 34. Optimization Engine (`engine/`)

| Class | Description |
|-------|-------------|
| `OptimizationEngine` | Full lifecycle orchestration: diagnostics -> fingerprint -> meta-control -> plugin dispatch -> result logging |
| `EngineConfig` | Engine configuration (max iterations, batch size, seed) |
| `EngineResult` | Engine run result |
| `CampaignState` | Mutable campaign state management + checkpoint / resume serialization |
| `Trial` | Single trial lifecycle management (pending -> running -> completed / failed) |
| `TrialBatch` | Batch trial operations |

**Input validation:** Engine rejects empty parameter lists or empty objective lists with clear error messages.

---

### XI. Benchmarking

#### 35. Benchmark Runner (`benchmark/`)

| Class | Description |
|-------|-------------|
| `BenchmarkRunner` | Standardized evaluation: multi-landscape x multi-seed backend comparison |
| `BenchmarkResult` | Single evaluation result (AUC, best-so-far series, step count) |
| `Leaderboard` | Backend ranking table with stability verification |

---

#### 36. Synthetic Benchmark Generator (`benchmark_generator/`)

| Class | Description |
|-------|-------------|
| `SyntheticObjective` | Configurable synthetic objective function |
| `LandscapeType` | 4 landscapes: `SPHERE`, `ROSENBROCK`, `ACKLEY`, `RASTRIGIN` |
| `BenchmarkGenerator` | Generate complete campaign scenarios (with noise, failures, drift, constraints) |

Parameters: `n_dimensions`, `noise_sigma`, `failure_rate`, `failure_zones`, `constraints`, `n_objectives`, `drift_rate`, `has_categorical`, `categorical_effect`.

---

#### 37. Counterfactual Evaluation (`counterfactual/`)

| Class | Description |
|-------|-------------|
| `CounterfactualEvaluator` | Offline hypothetical analysis: what if a different backend was used? |
| `CounterfactualResult` | Counterfactual result (speedup ratio, KPI range) |

---

### XII. Validation

#### 38. Golden Scenarios (`validation/`)

5 golden scenarios for regression testing:

| Scenario | Simulated Data | Expected Behavior |
|----------|---------------|-------------------|
| `clean_convergence` | 30 observations, monotonically increasing KPI | -> Exploitation phase |
| `cold_start` | Only 4 observations | -> Cold Start phase |
| `failure_heavy` | 50% failure rate, concentrated in second half | -> Stagnation phase |
| `noisy_plateau` | Constant KPI + minor noise | -> Stagnation phase |
| `mixed_variables` | Continuous + categorical parameters | -> Learning phase |

---

### XIII. Data Entry Layer

#### 39. Unified Experiment Store (`store/`)

| Class | Description |
|-------|-------------|
| `ExperimentStore` | Persistent experiment data warehouse: row-level append, campaign-level query, snapshot export |
| `ExperimentRecord` | Single experiment record (parameters + KPIs + metadata + timestamp) |
| `StoreQuery` | Flexible query: by campaign / parameter range / time window / KPI range |
| `StoreStats` | Storage summary statistics: campaign list, parameter columns, total row count |

**Storage features:**
- Pure Python dict backend, JSON round-trip serialization
- Row-level append + bulk import (`add_records()` / `bulk_import()`)
- Automatic campaign ID indexing, O(1) campaign query
- Snapshot bridge: `to_campaign_snapshot()` directly outputs `CampaignSnapshot`

---

#### 40. Data Ingestion Agent (`ingestion/`)

| Class | Description |
|-------|-------------|
| `DataIngestionAgent` | Auto-parse CSV/JSON + column role inference + interactive confirmation |
| `ColumnProfile` | Column profile: type inference, unique value count, missing rate, numeric range |
| `IngestionResult` | Import result: success count + skip count + error list |
| `RoleMapping` | Column role mapping: parameter / kpi / metadata / ignore |

**Ingestion capabilities:**
- CSV / JSON automatic format detection
- Column type inference (numeric / categorical / datetime / text)
- Heuristic-based role guessing (parameter vs. KPI vs. metadata columns)
- User confirmation -> auto-write to `ExperimentStore`
- Missing value / type mismatch error reporting

---

#### 41. Problem Builder (`problem_builder/`)

| Class | Description |
|-------|-------------|
| `ProblemBuilder` | Fluent API for `OptimizationSpec`: chained `.add_parameter().set_objective().build()` |
| `ProblemGuide` | Guided interactive modeling: step-by-step prompts for parameters, objectives, constraints |
| `BuilderValidation` | Build-time validation: parameter name uniqueness, bound legality, objective direction check |

**Modeling modes:**
- **Fluent mode**: Programmatic chained calls, ideal for script integration
- **Guide mode**: Interactive guidance, ideal for new users
- Both modes output standard `OptimizationSpec`, directly usable by `OptimizationEngine`

---

### XIV. Cross-Project Meta-Learning (7 sub-modules)

#### 42. Meta-Learning Data Models (`meta_learning/models.py`)

| Data Structure | Description |
|----------------|-------------|
| `CampaignOutcome` | Completed campaign summary: fingerprint, phase transitions, backend performance, failure types, best KPI |
| `BackendPerformance` | Per-backend performance record: convergence iteration, regret, sample efficiency, failure rate, drift impact |
| `ExperienceRecord` | Experience record: `CampaignOutcome` + fingerprint key |
| `MetaLearningConfig` | Meta-learning config: cold start threshold, similarity decay, EMA learning rate, recency half-life |
| `LearnedWeights` | Learned scoring weights (gain / fail / cost / drift / incompatibility) |
| `LearnedThresholds` | Learned phase switching thresholds (cold start obs / learning plateau / exploitation gain threshold) |
| `FailureStrategy` | Failure type -> best stabilization strategy mapping |
| `DriftRobustness` | Backend drift robustness score (resilience score + KPI loss) |
| `MetaAdvice` | Meta-learning advice output: recommended backends + scoring weights + switching thresholds + failure strategies + drift-robust backends |

---

#### 43. Cross-Campaign Experience Store (`meta_learning/experience_store.py`)

| Class | Description |
|-------|-------------|
| `ExperienceStore` | Cross-campaign persistent experience store: record outcomes, query by fingerprint, retrieve similar fingerprints |

**Core capabilities:**
- Exact query by campaign ID / aggregate query by fingerprint key
- **Continuous fingerprint similarity**: RBF kernel `k(x,y) = exp(-||x-y||² / 2)` on domain-aware ordinal-encoded continuous vectors (9 dimensions: 8 enum ordinals + normalized dimensionality). Produces smooth similarity gradients instead of binary match/no-match
- **Recency weighting**: `recency_halflife` controls old experience decay
- JSON serialization / deserialization

**Similarity-based fallback**: `WeightTuner` and `ThresholdLearner` both support similarity-based lookup — when no exact fingerprint match exists, they find the most similar learned fingerprint (threshold > 0.5) and transfer its learned parameters.

---

#### 44. Strategy Learner (`meta_learning/strategy_learner.py`)

| Class | Description |
|-------|-------------|
| `StrategyLearner` | Fingerprint -> backend affinity learning: exact match + similar match fallback, KNN-style ranking |

---

#### 45. Weight Tuner (`meta_learning/weight_tuner.py`)

| Class | Description |
|-------|-------------|
| `WeightTuner` | EMA-based optimal `ScoringWeights` learning: high failure -> raise fail weight, high drift -> raise drift weight |

---

#### 46. Threshold Learner (`meta_learning/threshold_learner.py`)

| Class | Description |
|-------|-------------|
| `ThresholdLearner` | Learn `SwitchingThresholds` from phase transition timing: EMA fusion of high-quality transition rounds |

---

#### 47. Failure Strategy Learner (`meta_learning/failure_learner.py`)

| Class | Description |
|-------|-------------|
| `FailureStrategyLearner` | Failure type -> stabilization strategy mapping: ranked by average outcome quality |

---

#### 48. Drift Robustness Tracker (`meta_learning/drift_learner.py`)

| Class | Description |
|-------|-------------|
| `DriftRobustnessTracker` | Track per-backend performance under drift: resilience = 1.0 - KPI loss |

---

#### 49. Meta-Learning Advisor (`meta_learning/advisor.py`)

| Class | Description |
|-------|-------------|
| `MetaLearningAdvisor` | Top-level orchestrator: combines 5 sub-learners, outputs unified `MetaAdvice` |

**Integration (pure injection, no existing files modified):**

| Existing Component | Injection Point | MetaAdvice Provides |
|-------------------|-----------------|---------------------|
| `MetaController.decide(portfolio=...)` | portfolio parameter | `recommended_backends` -> `AlgorithmPortfolio` |
| `BackendScorer(weights=...)` | constructor parameter | `scoring_weights` -> `ScoringWeights` |
| `MetaController(thresholds=...)` | constructor parameter | `switching_thresholds` -> `SwitchingThresholds` |
| `FailureTaxonomy` | MetaController consumer | `failure_adjustments` |
| `DriftReport` | MetaController consumer | `drift_robust_backends` |

---

### XV. Pure-Python Math Library (`backends/_math/`)

Zero-dependency mathematical primitives powering all backends:

| Module | Functions | Purpose |
|--------|-----------|---------|
| `linalg.py` | `vec_dot`, `mat_mul`, `mat_vec`, `transpose`, `identity`, `mat_add`, `mat_scale`, `outer_product`, `cholesky`, `solve_lower`, `solve_upper`, `solve_cholesky`, `mat_inv`, `determinant`, `eigen_symmetric` | Linear algebra (15 functions) |
| `stats.py` | `norm_pdf`, `norm_cdf`, `norm_ppf`, `norm_logpdf`, `binary_entropy` | Statistical distributions (5 functions) |
| `sobol.py` | `sobol_sequence`, `SOBOL_DIRECTION_NUMBERS` | Quasi-random sequences (21 dims) |
| `kernels.py` | `rbf_kernel`, `matern52_kernel`, `distance_matrix`, `kernel_matrix` | GP kernel functions (4 functions) |
| `acquisition.py` | `expected_improvement`, `upper_confidence_bound`, `probability_of_improvement`, `log_expected_improvement_per_cost` | Acquisition functions (4 functions) |

**Total: 28+ pure-Python math functions** — Cholesky decomposition, eigendecomposition via power iteration, Sobol sequences, GP kernels, acquisition functions, all with zero external dependencies.

---

### XVI. Infrastructure Layer (`infrastructure/`)

Enterprise-scale optimization orchestration:

| Module | Key Classes | Purpose |
|--------|-------------|---------|
| `auto_sampler.py` | `AutoSampler`, `SelectionResult` | Auto-select best backend per problem fingerprint |
| `batch_scheduler.py` | `BatchScheduler`, `AsyncTrial`, `TrialStatus` | Queue & schedule trial batches; async execution |
| `constraint_engine.py` | `ConstraintEngine`, `Constraint`, `ConstraintEvaluation` | Handle inequality/equality constraints; feasibility GP |
| `cost_tracker.py` | `CostTracker`, `TrialCost` | Track evaluation costs; cost-aware optimization |
| `domain_encoding.py` | `EncodingPipeline`, `OneHotEncoding`, `OrdinalEncoding`, `CustomDescriptorEncoding`, `SpatialEncoding` | Flexible parameter space transformations |
| `multi_fidelity.py` | `MultiFidelityManager`, `FidelityLevel` | Multi-fidelity evaluation coordination |
| `parameter_importance.py` | `ParameterImportanceAnalyzer`, `ImportanceResult` | Attribute importance via surrogate model |
| `robust_optimizer.py` | `RobustOptimizer` | Robust design under noise/uncertainty |
| `stopping_rule.py` | `StoppingRule`, `StoppingDecision` | Convergence detection & early stopping |
| `transfer_learning.py` | `TransferLearningEngine`, `CampaignData` | Knowledge transfer from prior campaigns |
| `integration.py` | `InfrastructureStack`, `InfrastructureConfig` | Wire up all components; dependency injection |

**InfrastructureStack** integrates into `OptimizationEngine.run()` main loop, providing constraint handling, cost tracking, stopping rules, and auto-sampler selection.

---

### XVII. Platform Layer

#### 50. REST API (`api/`)

| File | Key Endpoints | Description |
|------|---------------|-------------|
| `app.py` | `create_app()` | FastAPI app factory with lifespan, CORS, health check |
| `routes/campaigns.py` | POST/GET/DELETE campaigns, start/stop/pause/resume | Campaign CRUD + lifecycle |
| `routes/campaigns.py` | GET batch, POST trials, GET result, GET checkpoint | Trial submission & result retrieval |
| `routes/advice.py` | POST `/advice` | Meta-learning recommendations |
| `routes/reports.py` | GET `/audit`, `/compliance`, POST `/compare` | Audit trails, compliance reports, campaign comparison |
| `routes/store.py` | GET `/query`, `/summary`, `/export` | Historical trial query & export |
| `routes/ws.py` | WebSocket `/{campaign_id}`, `/all_events` | Real-time event streaming |
| `routes/loop.py` | POST/GET/DELETE loop, iterate, ingest | CampaignLoop lifecycle |
| `routes/analysis.py` | POST top-k, ranking, outliers, correlation, fanova, symreg, pareto, diagnostics, molecular + causal, physics, hypothesis, robustness, hybrid endpoints | DataAnalysisPipeline endpoints |

---

#### 51. Platform Services (`platform/`)

| Class | Description |
|-------|-------------|
| `AuthManager` | API key management & workspace authentication |
| `CampaignManager` | Campaign CRUD & state transitions |
| `CampaignRunner` | Execute campaign logic; manage trial submission loop |
| `AsyncEventBus` | Pub-sub event broadcasting to WebSocket clients |
| `Workspace` | Workspace initialization & file-based persistence |
| `RAGIndex` | Retrieval-augmented generation index for meta-learning |

---

#### 52. CLI Application (`cli_app/`)

| Command Group | Commands | Description |
|---------------|----------|-------------|
| `campaign` | create, list, status, start, stop, pause, resume, delete | Campaign lifecycle operations |
| `store` | summary, query, export | Workspace trial store access |
| `meta-learning` | show, advice | Meta-learning inspection & advice |
| `server` | init, start | API server initialization & launch |

---

#### 53. Web Frontend (`web/`)

React TypeScript SPA for campaign visualization:

| Component | Description |
|-----------|-------------|
| `Dashboard` | Campaign overview grid with status cards |
| `CampaignDetail` | Single campaign inspection (KPIs, trials, timeline) |
| `Reports` | Audit logs, compliance reports, comparisons |
| `Compare` | Side-by-side campaign comparison |
| `KpiChart` | KPI evolution chart |
| `TrialTable` | Trial listing with sorting/filtering |
| `PhaseTimeline` | Campaign phase visualization |
| `AuditTrail` | Decision audit log display |
| `useCampaign` | Custom hook for campaign data fetch + polling |
| `useWebSocket` | Custom hook for WebSocket event subscription |
| `LoopView` | Interactive CampaignLoop management (create, iterate, ingest, view deliverables) |
| `AnalysisView` | Tab-based data analysis (Top-K, Correlation, fANOVA) with traced results |

---

### XVIII. Visualization & Analysis Layer (v3)

Pure-Python SVG-based visualization suite — zero external dependencies.

#### 54. Visualization Foundation (`visualization/`)

| Class / Protocol | Description |
|------------------|-------------|
| `PlotData` | Universal chart container: plot_type, data, metadata, svg; with `to_dict()` / `from_dict()` |
| `SurrogateModel` | `runtime_checkable` Protocol requiring `predict(x) -> (mean, uncertainty)` |
| `SVGCanvas` | Pure-Python SVG builder: rect, circle, line, polyline, polygon, text, path, group, defs |

---

#### 55. VSUP Colormaps (`visualization/colormaps.py`)

| Class | Description |
|-------|-------------|
| `VSUPColorMap` | Value-Suppressing Uncertainty Palette: dual-variable color encoding (hue=value, saturation=uncertainty) |
| | Supports `viridis`, `plasma`, `inferno` palettes with 5-stop interpolation |
| | `map(value, uncertainty)` -> (R, G, B, A); `batch_map()`; `color_to_hex()` |

---

#### 56. Space-Filling Diagnostics (`visualization/diagnostics.py`)

| Function | Description |
|----------|-------------|
| `plot_space_filling_metrics()` | Dashboard returning PlotData with discrepancy, coverage, min_distance |
| `_compute_star_discrepancy()` | Exact for d<=5, random approximation for d>5 |
| `_compute_coverage()` | Grid-based coverage percentage |
| `_compute_min_distance()` | Minimum pairwise L2 distance in normalized space |

---

#### 57. SHAP Explainability (`_analysis/` + `visualization/explainability.py`)

| Class / Function | Description |
|------------------|-------------|
| `KernelSHAPApproximator` | Kernel SHAP engine: exact enumeration (d<=11), random sampling (d>=12), weighted regression |
| `plot_shap_waterfall()` | SHAP waterfall chart (cumulative feature contributions) |
| `plot_shap_beeswarm()` | SHAP beeswarm plot (feature importance distribution) |
| `plot_shap_dependence()` | SHAP dependence plot (feature value vs. SHAP value with interaction coloring) |
| `plot_shap_force()` | SHAP force plot (push/pull from base value) |

---

#### 58. SDL Monitoring (`visualization/sdl_monitor.py`)

| Function | Description |
|----------|-------------|
| `plot_experiment_status_dashboard()` | Experiment progress: trials by status, throughput, time tracking |
| `plot_safety_monitoring()` | Safety panel: constraint violations, hardware alerts, severity timeline |
| `plot_human_in_the_loop()` | Operator panel: intervention frequency, decision overrides, approval rates |
| `plot_continuous_operation_timeline()` | Timeline: hardware utilization, experiment phases, maintenance windows |
| `SDLDashboardData` | Data container with autonomy_level, experiments, safety events, operator actions |

---

#### 59. Design Space Exploration (`visualization/design_space.py`)

| Function | Description |
|----------|-------------|
| `plot_latent_space_exploration()` | PCA + t-SNE 2D projections of parameter space |
| `plot_isom_landscape()` | Self-Organizing Map: competitive learning with Gaussian neighborhood |
| `plot_forward_inverse_design()` | Grid parameter space -> surrogate prediction -> feasible region filtering |

PCA uses `eigen_symmetric()` from `_math/linalg.py`. t-SNE implements pairwise affinities with binary search perplexity and gradient descent with early exaggeration.

---

#### 60. Hexbin Coverage (`visualization/parameter_space.py`)

| Class / Function | Description |
|------------------|-------------|
| `HexCell` | Hexagonal cell with axial coordinates and statistics |
| `plot_hexbin_coverage()` | Hexagonal binning coverage view with 3 coloring modes (density / predicted_mean / uncertainty) |

---

#### 61. LLM Visualization Assistant (`visualization/llm_assistant.py`)

| Class | Description |
|-------|-------------|
| `PlotSpec` | Plot specification (plot_type, filters, parameters, color_by, aggregation, title) |
| `LLMVisualizationAssistant` | Skeleton for LLM-driven interactive plot generation; `validate_spec()` implemented, `query_to_plot()` extensible |

---

### XIX. Campaign Engine (`campaign/`)

Closed-loop optimization engine with surrogate modeling, candidate ranking, and three-layer deliverable output.

#### 62. Fingerprint Surrogate (`campaign/surrogate.py`)

| Class | Description |
|-------|-------------|
| `FingerprintSurrogate` | Lightweight GP surrogate using fingerprint-based kernel. Fits on observation history, predicts mean + uncertainty for candidates |
| `SurrogateResult` | Prediction result with mean, std, and acquisition score |

#### 63. Candidate Ranker (`campaign/ranker.py`)

| Class | Description |
|-------|-------------|
| `CandidateRanker` | Rank candidates by acquisition score (UCB, EI, or PI). Produces sorted `RankedCandidate` list with predicted mean, std, and acquisition score |
| `RankedCandidate` | Single ranked candidate with rank, name, parameters, predictions |
| `RankedTable` | Sorted table of ranked candidates with objective metadata |

#### 64. Stage Gate Protocol (`campaign/stage_gate.py`)

| Class | Description |
|-------|-------------|
| `StageGateProtocol` | Decision logic for campaign progression: should we continue, expand, or stop? Based on convergence, budget, and improvement signals |
| `StageGateDecision` | Gate decision with action (continue/expand/stop) and rationale |

#### 65. Campaign Deliverable (`campaign/output.py`)

Three-layer structured output for each campaign iteration:

| Layer | Class | Contents |
|-------|-------|----------|
| Dashboard | `DashboardLayer` | Ranked candidate table + batch selection |
| Intelligence | `IntelligenceLayer` | Model metrics + learning report + Pareto summary |
| Reasoning | `ReasoningLayer` | Diagnostic summary + fANOVA result + execution traces |

| Class | Description |
|-------|-------------|
| `CampaignDeliverable` | Top-level container: iteration, timestamp, dashboard + intelligence + reasoning layers |
| `ModelMetrics` | Per-objective model metrics: training points, y statistics, fit duration |
| `LearningReport` | Comparison of model predictions vs. actuals: prediction errors, MAE, summary |

#### 66. Campaign Loop (`campaign/loop.py`)

| Class | Description |
|-------|-------------|
| `CampaignLoop` | Stateful closed-loop optimization: fit surrogates → rank candidates → build deliverable → ingest results → repeat |

**Loop lifecycle:**
```
create(snapshot, candidates) → run_iteration() → ingest_results(new_obs) → run_iteration() → ...
```

Each iteration produces a `CampaignDeliverable` with all three layers. Supports SMILES-based molecular encoding via `NGramTanimoto`.

---

### XX. Agent Layer (`agents/`)

Code execution enforcement system ensuring all quantitative claims are backed by actual computation.

#### 67. Execution Trace (`agents/execution_trace.py`)

| Class | Description |
|-------|-------------|
| `ExecutionTag` | Enum: `COMPUTED` (code ran), `ESTIMATED` (no code), `FAILED` (code errored) |
| `ExecutionTrace` | Proof-of-execution record: module, method, input/output summary, duration_ms, tag |
| `TracedResult` | Value + traces + aggregate tag. `is_computed` property for quick check |
| `trace_call()` | Helper: execute function, wrap result in TracedResult with timing |

**Principle:** Every quantitative conclusion must have a `TracedResult` with `tag=COMPUTED`.

#### 68. Data Analysis Pipeline (`agents/data_pipeline.py`)

| Method | Wraps | Purpose |
|--------|-------|---------|
| `run_top_k()` | Pure Python sorted | Top-K entries by value |
| `run_ranking()` | Pure Python sorted | Full ranking of all entries |
| `run_outlier_detection()` | Z-score statistics | Statistical outlier detection |
| `run_correlation()` | Pure Python Pearson | Pairwise correlation |
| `run_fanova()` | `InteractionMap` | Feature importance via fANOVA |
| `run_symreg()` | `EquationDiscovery` | Symbolic regression |
| `run_insight_report()` | `InsightReportGenerator` | Full fANOVA→SymReg→SVG pipeline |
| `run_confounder_detection()` | `ConfounderDetector` | Confounder detection |
| `run_pareto_analysis()` | `MultiObjectiveAnalyzer` | Pareto front computation |
| `run_diagnostics()` | `DiagnosticEngine` | Campaign health signals |
| `run_molecular_pipeline()` | `NGramTanimoto` + `GaussianProcessBO` | SMILES→fingerprint→GP→acquisition |
| `run_screening()` | `VariableScreener` | Variable importance screening |
| `run_causal_discovery()` | `CausalStructureLearner` | Causal DAG structure learning |
| `run_intervention()` | `InterventionalEngine` | do-operator causal intervention |
| `run_counterfactual()` | `CounterfactualReasoner` | SCM counterfactual reasoning |
| `run_physics_constrained_gp()` | `PhysicsConstraintModel` | Physics-constrained GP optimization |
| `run_ode_solve()` | `RK4Solver` | ODE numerical integration |
| `run_hypothesis_generate()` | `HypothesisGenerator` | Multi-source hypothesis generation |
| `run_hypothesis_test()` | `HypothesisTester` | BIC/Bayes factor hypothesis testing |
| `run_hypothesis_status()` | `HypothesisTracker` | Hypothesis lifecycle tracking |
| `run_bootstrap_ci()` | `BootstrapAnalyzer` | Bootstrap confidence intervals |
| `run_conclusion_robustness()` | `ConclusionRobustnessChecker` | Ranking/importance stability |
| `run_decision_sensitivity()` | `DecisionSensitivityAnalyzer` | Decision perturbation analysis |
| `run_cross_model_consistency()` | `CrossModelConsistency` | Cross-model rank agreement |
| `run_hybrid_fit()` | `HybridModel` | Theory + residual GP fitting |
| `run_hybrid_predict()` | `HybridModel` | Hybrid prediction with uncertainty |
| `run_discrepancy_analysis()` | `DiscrepancyAnalyzer` | Theory-data discrepancy detection |

Each method returns `TracedResult` with full execution provenance.

#### 69. Execution Guard (`agents/execution_guard.py`)

| Class | Description |
|-------|-------------|
| `ExecutionGuard` | Validates feedback payloads have execution traces for quantitative claims |
| `GuardMode` | `STRICT` (reject untraced claims) or `LENIENT` (tag as [ESTIMATED]) |

#### 70. Traced Scientific Agent (`agents/traced_agent.py`)

| Class | Description |
|-------|-------------|
| `TracedScientificAgent` | Optional base class with built-in pipeline and automatic trace collection |

#### 71. Scientific Orchestrator (`agents/orchestrator.py`)

| Class | Description |
|-------|-------------|
| `ScientificOrchestrator` | Multi-agent reasoning dispatcher with audit trail |
| `AuditEntry` | Per-agent audit record with execution traces |

#### 72. Safety Validator (`agents/safety.py`)

| Class | Description |
|-------|-------------|
| `LLMSafetyWrapper` | 7-check safety validation for LLM feedback: confidence, physics, hallucination, and execution guard |

#### 72a. Literature Mining Agent (`agents/literature/`)

| Class | Description |
|-------|-------------|
| `LiteratureAgent` | Mine literature for prior knowledge: parameter ranges, expected KPI values, domain constraints |
| `PriorTables` | Structured prior knowledge tables from literature |

#### 72e. Experiment Planner Agent (`agents/experiment_planner.py`)

| Class | Description |
|-------|-------------|
| `ExperimentPlannerAgent` | LLM-driven experiment planning with pragmatic fallback. Two modes: `PRAGMATIC` (rule-based: stagnating→explore, improving→exploit, early→space-fill) and `LLM_ENHANCED` (Claude API with structured prompts). Activates on milestone/stagnation events or every 10th iteration |
| `PlannerConfig` | Configuration: model_name, max_tokens, temperature, system_prompt |

#### 72b. Mechanism Design Agent (`agents/mechanism/`)

| Class | Description |
|-------|-------------|
| `MechanismAgent` | Generate mechanism hypotheses for observed phenomena |
| `MechanismTemplates` | Domain-specific mechanism templates (e.g., Arrhenius, Michaelis-Menten) |

#### 72c. Phase Structure Agent (`agents/phase_structure/`)

| Class | Description |
|-------|-------------|
| `PhaseStructureAgent` | Analyze phase diagrams and structure-property relationships |
| `ReferenceDB` | Reference phase structure database |

#### 72d. Symbolic Regression Agent (`agents/symreg/`)

| Class | Description |
|-------|-------------|
| `SymRegAgent` | Discover interpretable equations from data via genetic programming |

---

### XXI. Advanced Analysis (`explain/` + `anomaly/` + `confounder/` + `imputation/`)

#### 73. Interaction Map (`explain/interaction_map.py`)

| Class | Description |
|-------|-------------|
| `InteractionMap` | fANOVA-style feature importance decomposition using random forest variance analysis |

#### 74. Equation Discovery (`explain/equation_discovery.py`)

| Class | Description |
|-------|-------------|
| `EquationDiscovery` | Symbolic regression via genetic programming. Discovers interpretable equations from data |

#### 75. Insight Report Generator (`explain/report_generator.py`)

| Class | Description |
|-------|-------------|
| `InsightReportGenerator` | Full pipeline: fANOVA → symbolic regression → SVG visualization → human-readable report |

#### 76. Anomaly Detector (`anomaly/detector.py`)

Three-layer anomaly detection:

| Layer | Detection Method |
|-------|-----------------|
| Signal-level | Z-score outlier detection on diagnostic signals |
| KPI-level | Bayesian online changepoint detection (BOCPD) |
| GP-level | GP-based outlier detection using predictive variance |

#### 77. Confounder Governance (`confounder/`)

| Class | Description |
|-------|-------------|
| `ConfounderDetector` | Detect confounding variables via correlation analysis |
| `ConfounderCorrector` | 4 correction strategies: COVARIATE, NORMALIZE, FLAG, EXCLUDE |
| `ConfounderAuditTrail` | Full audit trail for confounder decisions |

#### 78. Deterministic Imputation (`imputation/`)

| Class | Description |
|-------|-------------|
| `DeterministicImputer` | 4 strategies: MEAN, MEDIAN, MODE, KNN. Full traceability with imputation log |

---

### XXII. Molecular & Representation (`representation/` + `candidate_pool/` + `extractors/`)

#### 79. Representation Layer (`representation/`)

| Class | Description |
|-------|-------------|
| `NGramTanimoto` | SMILES → n-gram fingerprint → Tanimoto kernel. Zero-dependency molecular encoding |
| `RepresentationProvider` | Swappable encoding interface for molecular inputs |

#### 80. Candidate Pool (`candidate_pool/`)

| Class | Description |
|-------|-------------|
| `CandidatePool` | External molecular library management with versioning and deduplication |

#### 81. Uncertainty-Aware Extractors (`extractors/`)

| Class | Description |
|-------|-------------|
| `EISExtractor` | EIS impedance extraction with uncertainty propagation |
| `UVVisExtractor` | UV-Vis spectral extraction with confidence intervals |
| `XRDExtractor` | XRD pattern extraction with measurement uncertainty |
| `DCCyclingExtractor` | DC cycling data extraction |

---

### XXIII. Workflow & Fidelity (`workflow/` + `fidelity/`)

#### 82. Workflow Engine (`workflow/`)

| Class | Description |
|-------|-------------|
| `WorkflowDAG` | Multi-stage experimental DAG with stage gates |
| `FidelityGraph` | Multi-fidelity experiment graph |
| `Simulator` | Experiment simulation for workflow planning |

#### 83. Fidelity Configuration (`fidelity/`)

| Class | Description |
|-------|-------------|
| `FidelityConfig` | Multi-fidelity cost modeling and level configuration |

---

### XXIV. Evaluation & Domain (`benchmark_protocol/` + `case_studies/` + `domain_knowledge/`)

#### 84. Benchmark Protocol (`benchmark_protocol/`)

| Class | Description |
|-------|-------------|
| `SDLBenchmark` | SDL benchmark evaluation with standardized metrics |
| `Leaderboard` | Benchmark leaderboard with ranking and export |

#### 85. Case Studies (`case_studies/`)

Real experimental benchmarks for validation:

| Case Study | Domain | Data |
|------------|--------|------|
| Perovskite | Materials | Crystal structure optimization |
| Zinc | Electrochemistry | Zinc plating optimization |
| Catalysis | Chemistry | Catalyst performance optimization |

#### 86. Domain Knowledge (`domain_knowledge/`)

| Class | Description |
|-------|-------------|
| `InstrumentSpecs` | Centralized instrument specifications (EIS, UV-Vis, XRD, DC-cycling) |
| `ConstraintLibrary` | Domain-specific constraint templates |

---

### XXV. Scientific Intelligence Layers

Five fundamental scientific reasoning layers that transform the system from an optimization tool into a scientific intelligence platform.

#### 87. Causal Discovery Engine (`causal/`)

| Class | Description |
|-------|-------------|
| `CausalGraph` | DAG data structure with `CausalNode` and `CausalEdge`. Edge types: causal/confounded/instrumental. Methods: `parents()`, `children()`, `ancestors()`, `descendants()`, `d_separated()` (Bayes-Ball), `topological_sort()` (Kahn's) |
| `CausalStructureLearner` | PC algorithm — conditional independence via partial correlations from precision matrix, Fisher z-transform, v-structure orientation, Meek rules R1-R3 |
| `InterventionalEngine` | `do()` operator via graph mutilation + backdoor/front-door adjustment. `find_valid_adjustment_set()` via backdoor criterion |
| `CausalEffectEstimator` | Average Treatment Effect (ATE), Conditional ATE (CATE), Natural Direct Effect (NDE) via mediation |
| `CounterfactualReasoner` | SCM three-step: abduction (infer noise U from factual), action (modify equations), prediction (compute counterfactual). `probability_of_necessity()` and `probability_of_sufficiency()` |

---

#### 88. Physics-Informed Modeling (`physics/`)

| Class | Description |
|-------|-------------|
| `PeriodicKernel` | `exp(-2 sin²(π|x-x'|/p) / l²)` for periodic phenomena |
| `LinearKernel` | Linear kernel with variance and bias parameters |
| `CompositeKernel` | Sum/product kernel composition |
| `SymmetryKernel` | Kernel for symmetric function modeling |
| `ArrheniusPrior` | GP mean function: `A * exp(-Ea / RT)` (chemical kinetics) |
| `MichaelisMentenPrior` | GP mean function: `Vmax * S / (Km + S)` (enzyme kinetics) |
| `PowerLawPrior` | GP mean function: `a * x^b` (scaling laws) |
| `RK4Solver` | 4th-order Runge-Kutta ODE solver. `solve()` and `solve_to_steady_state()`, pure Python |
| `PhysicsConstraintModel` | Conservation laws, monotonicity, physics bounds. `check_feasibility()` and `project_to_feasible()`. Integrates with `ConstraintEngine` |

---

#### 89. Hypothesis Engine (`hypothesis/`)

| Class | Description |
|-------|-------------|
| `Hypothesis` | Formal hypothesis object with lifecycle: PROPOSED → TESTING → SUPPORTED / REFUTED / INCONCLUSIVE. Serializable with `to_dict()` / `from_dict()` |
| `Prediction` | Predicted value + confidence interval + condition |
| `Evidence` | Prediction vs. observed outcome comparison |
| `HypothesisGenerator` | Generate hypotheses from multiple sources: `from_symreg()` (Pareto equations), `from_causal_graph()` (causal paths), `from_fanova()` (important features), `from_correlation()` |
| `HypothesisTester` | `compute_bic()`, `bayes_factor()`, `sequential_update()`, `check_falsification()` (≥3 consecutive misses). Includes safe recursive-descent expression parser |
| `HypothesisTracker` | Lifecycle management: `add()`, `update_with_observation()`, `suggest_discriminating_experiment()` (find where hypotheses disagree most), `get_status_report()` |

---

#### 90. Decision Robustness (`robustness/`)

| Class | Description |
|-------|-------------|
| `BootstrapAnalyzer` | Non-parametric bootstrap CI: `bootstrap_ci()`, `bootstrap_top_k()`, `bootstrap_correlation()`, `bootstrap_feature_importance()` |
| `ConclusionRobustnessChecker` | `check_ranking_stability()` (fraction of bootstraps with same top-K), `check_importance_stability()`, `check_pareto_stability()`, `comprehensive_robustness()` |
| `DecisionSensitivityAnalyzer` | `decision_sensitivity()` (perturb data, re-optimize, measure variation), `recommendation_confidence()`, `value_at_risk()` |
| `CrossModelConsistency` | `kendall_tau()` (O(n²)), `model_agreement()` (pairwise GP/RF/SymReg), `ensemble_confidence()`, `disagreement_regions()` |

---

#### 91. Theory-Data Hybrid (`hybrid/`)

| Class | Description |
|-------|-------------|
| `TheoryModel` | ABC for parametric theory models + concrete: `ArrheniusModel`, `MichaelisMentenModel`, `PowerLawModel`, `ODEModel` (wraps RK4Solver) |
| `ResidualGP` | GP on residuals r = y - theory(X). Fits via Cholesky from `_math/linalg`. Provides mean and uncertainty for residual predictions |
| `HybridModel` | Combined prediction: `theory(x) + GP_residual(x)`. `suggest_next()` with EI/UCB acquisition, `compare_to_theory_only()`, `theory_adequacy_score()` |
| `DiscrepancyAnalyzer` | `systematic_bias()`, `failure_regions()` (where |residual| > threshold), `model_adequacy_test()` (chi-squared), `suggest_theory_revision()` |

---

### XXVI. Measurement Uncertainty (`uncertainty/`)

| Class | Description |
|-------|-------------|
| `UncertaintyType` | Enum: ALEATORIC, EPISTEMIC, SYSTEMATIC, COMBINED |
| `UncertaintyEstimate` | Value + std_dev + confidence_interval + type + source |
| `PropagationEngine` | GUM-compliant uncertainty propagation through computation chains |

---

### XXVII. Dimension Analysis (`profiler/dimension_analyzer.py`)

| Class | Description |
|-------|-------------|
| `DimensionAnalyzer` | Analyze effective dimensionality: correlation-based redundancy detection, active subspace estimation, intrinsic dimensionality via eigenspectrum analysis |

---

### XXVIII. Preference Protocol (`preference/`)

| Class | Description |
|-------|-------------|
| `PreferenceProtocol` | Structured pairwise comparison protocol with active learning: selects the most informative pairs for preference elicitation |
| `PreferenceModel` | Preference model with utility scores, consistency metrics, and transitivity checks |

---

### XXIX. SDL Gap Closure Modules

Eight modules closing all identified self-driving laboratory community pain points.

#### 92. Data Integration Layer (`integration/`)

| Class | Description |
|-------|-------------|
| `CampaignExporter` | Export CampaignSnapshot to CSV, JSON, JSON-LD with schema versioning |
| `CampaignImporter` | Import from CSV/JSON with column mapping and type inference |
| `ProvenanceTracker` | Append-only provenance chain with W3C PROV-O JSON-LD export |
| `ProvenanceChain` | O(1) lookup, lineage walk, children query, duplicate-ID detection |
| `CSVConnector` / `JSONConnector` | File-based lab connectors for observation read + suggestion write |
| `InMemoryConnector` | In-process connector for testing and integration |

#### 93. Safety-First Optimization (`safety/`)

| Class | Description |
|-------|-------------|
| `HazardRegistry` | Register and classify hazards per parameter (8 categories, 5 severity levels) |
| `SafetyMonitor` | Real-time point/batch safety checking with graduated response (SAFE → WARNING → DANGER → EMERGENCY) |
| `EmergencyProtocol` | Escalation logic: warning accumulation → PAUSE → DANGER → FALLBACK → CRITICAL → STOP |
| `EmergencyLog` | Append-only audit trail for all emergency evaluations |

#### 94. Campaign Reproducibility (`reproducibility/`)

| Class | Description |
|-------|-------------|
| `CampaignLogger` | Append-only event logging (8 event types) with JSONL export/import |
| `CampaignReplayer` | Replay campaigns using recorded seeds; compute reproducibility score |
| `FAIRGenerator` | Generate FAIR-compliant metadata (schema.org JSON-LD) for campaign datasets |

#### 95. Human-in-the-Loop (`hitl/`)

| Class | Description |
|-------|-------------|
| `PriorRegistry` | Expert prior injection (Gaussian, Uniform, Constraint, Preference, Ranking) with confidence-weighted re-ranking |
| `AutonomyPolicy` | Progressive autonomy: MANUAL → SUPERVISED → COLLABORATIVE → AUTONOMOUS |
| `TrustTracker` | EMA-based trust score with auto-escalation/de-escalation thresholds |
| `SteeringEngine` | Interactive steering: FOCUS_REGION, AVOID_REGION, MODIFY, ACCEPT, REJECT with full audit trail |

#### 96. Community Benchmark Integration (`community_benchmarks/`)

| Class | Description |
|-------|-------------|
| `OLYMPUS_REGISTRY` | 6 built-in Olympus surfaces (photobleaching, crossed_barrel, colors_bob, hplc, benzylation, suzuki) |
| `OlympusSurface` | Tabular benchmark with nearest-neighbor evaluation; `to_benchmark()` for standard interface |
| `AccelerationFactor` | How much faster BO reaches target vs. random search |
| `EnhancementFactor` | How much better BO result is at fixed budget |
| `DegreeOfAutonomy` | 1 - (human_interventions / total_decisions) |
| `SDLMetricsCalculator` | All-in-one SDL performance evaluation (Nature Communications 2024 metrics) |

---

### XXX. Validation Testing (3-Tier Real Data Integration)

Three tiers of integration tests proving end-to-end system functionality:

| Tier | Tests | What it proves |
|------|-------|----------------|
| **Tier 1**: End-to-end pipeline | 10 | Full 8-stage pipeline on 50-row Suzuki coupling data (ingestion → store → snapshot → diagnostics → drift → modeling → recommendation → audit) |
| **Tier 2**: Scientific stress | 14 | Drift detection (-15/-25 shifts), batch effects (ANOVA F-stat), confounder flagging (operator/instrument), failure region identification, stabilization, model nonstationarity, meta-controller adaptation — all on 120-row 3-batch dataset |
| **Tier 3**: Closed-loop demo | 19 | 4-round closed-loop optimization with hidden Branin-Hoo objective (true optimum yield~92). Guided optimization beats random baseline. Hash-chained audit trail. Deterministic replay. Proves the system drives science forward, not just explains offline |

---

## End-to-End Usage

### Core Optimization Pipeline

```python
from optimization_copilot.core import CampaignSnapshot
from optimization_copilot.diagnostics import DiagnosticEngine
from optimization_copilot.profiler import ProblemProfiler
from optimization_copilot.meta_controller import MetaController
from optimization_copilot.explainability import DecisionExplainer

# 1. Construct experiment snapshot
snapshot = CampaignSnapshot(
    campaign_id="my_experiment",
    parameter_specs=[...],
    observations=[...],
    objective_names=["yield"],
    objective_directions=["maximize"],
)

# 2. Compute diagnostic signals
engine = DiagnosticEngine()
diagnostics = engine.compute(snapshot)

# 3. Analyze problem fingerprint
profiler = ProblemProfiler()
fingerprint = profiler.profile(snapshot)

# 4. Get strategy decision
controller = MetaController()
decision = controller.decide(snapshot, diagnostics.to_dict(), fingerprint, seed=42)

# 5. Generate decision report
explainer = DecisionExplainer()
report = explainer.explain(decision, fingerprint, diagnostics.to_dict())

print(f"Phase: {decision.phase.value}")
print(f"Backend: {decision.backend_name}")
print(f"Exploration: {decision.exploration_strength}")
print(f"Risk: {decision.risk_posture.value}")
print(f"Batch size: {decision.batch_size}")
print(f"Report: {report.summary}")
```

### Declarative DSL

```python
from optimization_copilot.dsl.spec import (
    OptimizationSpec, ParameterDef, ObjectiveDef, BudgetDef,
    ParamType, Direction,
)
from optimization_copilot.engine.engine import OptimizationEngine, EngineConfig

spec = OptimizationSpec(
    campaign_id="formulation_v2",
    parameters=[
        ParameterDef(name="temp", type=ParamType.CONTINUOUS, lower=60, upper=120),
        ParameterDef(name="pH", type=ParamType.CONTINUOUS, lower=5.0, upper=9.0),
        ParameterDef(name="catalyst", type=ParamType.CATEGORICAL, categories=["A", "B", "C"]),
    ],
    objectives=[ObjectiveDef(name="yield", direction=Direction.MAXIMIZE)],
    budget=BudgetDef(max_iterations=50),
)

engine = OptimizationEngine(config=EngineConfig(seed=42))
# Engine auto-completes: diagnostics -> fingerprint -> meta-control -> backend dispatch -> audit
```

### Data Import + Guided Modeling

```python
from optimization_copilot.store.store import ExperimentStore
from optimization_copilot.ingestion.agent import DataIngestionAgent
from optimization_copilot.problem_builder.builder import ProblemBuilder

# 1. Auto-import from CSV
store = ExperimentStore()
agent = DataIngestionAgent(store)
result = agent.ingest_csv("experiments.csv", campaign_id="formulation_v3")
print(f"Imported {result.n_imported} records, skipped {result.n_skipped}")

# 2. Export directly as CampaignSnapshot
snapshot = store.to_campaign_snapshot("formulation_v3")

# 3. Build optimization spec with Fluent API
spec = (
    ProblemBuilder()
    .add_continuous("temp", 60, 120)
    .add_continuous("pH", 5.0, 9.0)
    .add_categorical("catalyst", ["A", "B", "C"])
    .set_objective("yield", "maximize")
    .set_budget(max_iterations=50)
    .build()
)
```

### Cross-Project Meta-Learning

```python
from optimization_copilot.meta_learning import (
    MetaLearningAdvisor, CampaignOutcome, BackendPerformance,
)

# 1. Create meta-learning advisor
advisor = MetaLearningAdvisor()

# 2. Record completed campaign outcome
outcome = CampaignOutcome(
    campaign_id="formulation_v2",
    fingerprint=fingerprint,
    phase_transitions=[("cold_start", "learning", 10), ("learning", "exploitation", 25)],
    backend_performances=[
        BackendPerformance(backend_name="TPE", convergence_iteration=30,
                           final_best_kpi=0.95, regret=0.05, sample_efficiency=0.03,
                           failure_rate=0.02, drift_encountered=False, drift_score=0.0),
    ],
    failure_type_counts={},
    stabilization_used={},
    total_iterations=40,
    best_kpi=0.95,
    timestamp=1700000000.0,
)
advisor.learn_from_outcome(outcome)

# 3. Get meta-learning advice for next campaign
advice = advisor.advise(new_fingerprint)
print(f"Recommended backends: {advice.recommended_backends}")
print(f"Confidence: {advice.confidence:.2f}")
print(f"Reasons: {advice.reason_codes}")

# 4. Inject into MetaController
controller = MetaController(thresholds=advice.switching_thresholds)
scorer = BackendScorer(weights=advice.scoring_weights)

# 5. Persist (save meta-learning state across sessions)
state_json = advisor.to_json()
advisor = MetaLearningAdvisor.from_json(state_json)
```

### Compliance Auditing

```python
from optimization_copilot.replay.log import DecisionLog
from optimization_copilot.compliance.audit import AuditLog, verify_chain
from optimization_copilot.compliance.report import ComplianceReport

# Load decision log
log = DecisionLog.load("campaign_log.json")

# Build audit chain
audit_log = AuditLog.from_decision_log(log)
verification = verify_chain(audit_log)
assert verification.valid, f"Audit chain broken at entry {verification.first_broken_index}"

# Generate compliance report
report = ComplianceReport.from_audit_log(audit_log)
print(report.format_text())
```

---

## Cross-Domain Generalization (Verified)

Via problem fingerprinting + plugin architecture, the system has been validated across **10 domains x 10 seeds = 100 runs**:

| # | Domain | Landscape | Dims | Noise | Special Config |
|---|--------|-----------|------|-------|----------------|
| 1 | Electrochemistry | Rosenbrock | 4 | 0.15 | 5% failure rate |
| 2 | Chemical Synthesis | Sphere | 5 | 0.02 | Mixed variables |
| 3 | Formulation | Ackley | 8 | 0.05 | Bi-objective + constraints |
| 4 | Bioassay | Sphere | 2 | 0.40 | High noise |
| 5 | Polymers | Rastrigin | 10 | 0.08 | 10% failure + high-dim |
| 6 | Drug Discovery | Ackley | 7 | 0.10 | 25% failure + mixed vars |
| 7 | Process Engineering | Rosenbrock | 5 | 0.05 | Drift (drift_rate=0.02) |
| 8 | Materials Science | Rastrigin | 6 | 0.10 | 15% failure + mixed vars |
| 9 | Agricultural Optimization | Sphere | 4 | 0.20 | Bi-objective + drift |
| 10 | Energy Systems | Rosenbrock | 8 | 0.05 | Tri-objective + constraints + 5% failure |

**Validation conclusions:**
- No single-backend degradation: no backend dominates >80% globally
- Fingerprint bucket diversity: 10 domains produce >= 4 distinct fingerprint buckets
- Cold start safety: all domains produce valid decisions with empty portfolio
- High-failure graceful degradation: valid decisions + reason_codes at 25% failure rate

---

## Test Suite

**6,383 tests** across **147 test files**, all passing:

### Acceptance Tests

| Category | Tests | Validates |
|----------|-------|-----------|
| 1. End-to-end pipeline | 14 | Full pipeline runs, golden scenario regression |
| 2. Determinism & audit | 8 | Hash consistency, audit trail integrity |
| 3. Synthetic benchmarks | 11 | 4 landscapes x clean/noisy, AUC, step counts |
| 4. Constraints/failure/drift/MO | 12 | Constraint discovery, failure taxonomy, drift detection, Pareto |
| 5. Ablation/counterfactual/leaderboard | 27 | Module contribution, Kendall-tau, ranking stability |
| 6. Shadow mode | 11 | Agent vs. Baseline comparison, denylist compliance, safety parity |
| 7. Monitoring & SLO | 11 | p50 < 50ms, p95 < 200ms, drift FP < 15%, action FP < 5% |
| 8. Release gate v1 | 8 | Determinism 100%, zero safety violations, AUC >= 60%, regression < 2% |
| 9. Cross-domain generalization | 7 | 10-domain diversity, portfolio fallback, audit reports |
| 10. API/UX acceptance | 14 | Input validation, plugin degradation, audit chain export |

### Real Data Integration Tests (Tier 1-3)

| Category | Tests | Validates |
|----------|-------|-----------|
| Tier 1: End-to-end pipeline | 10 | Full 8-stage pipeline on synthetic Suzuki coupling data (50 rows) |
| Tier 2: Scientific stress | 14 | Drift/batch/confounder/failure detection on 120-row 3-batch dataset |
| Tier 3: Closed-loop optimization | 19 | 4-round closed-loop campaign with hidden objective, random baseline comparison, audit trail |
| Adversarial robustness | 14 | Label corruption, systematic shift, adversarial flip, confidence calibration |
| Continuous similarity | 26 | RBF kernel similarity, ordinal encoding, transfer learning fallback |

### Top Test Files by Count

| File | Tests |
|------|-------|
| `test_integration.py` | 146 |
| `test_acceptance_benchmarks.py` | 93 |
| `test_fidelity_importance.py` | 92 |
| `test_infrastructure.py` | 90 |
| `test_transfer_batch.py` | 90 |
| `test_plugins.py` | 87 |
| `test_encoding_robust.py` | 83 |
| `test_meta_learning.py` | 82 |
| `test_feature_extraction.py` | 80 |
| `test_benchmark_functions.py` | 70 |
| `test_ingestion.py` | 71 |
| `test_diagnostics.py` | 68 |
| `test_engine_infrastructure.py` | 66 |
| `test_math_utils.py` | 64 |
| `test_api.py` | 59 |
| `test_api_loop.py` | 15 |
| `test_api_analysis.py` | 14 |
| `test_campaign_*.py` | 100+ |
| `test_execution_trace.py` | 30+ |
| `test_data_pipeline.py` | 50+ |
| `test_execution_guard.py` | 40+ |
| `test_anomaly_*.py` | 60+ |
| `test_explain_*.py` | 50+ |
| `test_materials_e2e.py` | 23 |
| `test_causal.py` | 35 |
| `test_physics.py` | 58 |
| `test_hypothesis.py` | 22 |
| `test_robustness.py` | 37 |
| `test_hybrid.py` | 31 |
| `test_tier1_endtoend.py` | 10 |
| `test_tier2_stress.py` | 14 |
| `test_tier3_closedloop.py` | 19 |
| `test_adversarial_robustness.py` | 14 |
| `test_continuous_similarity.py` | 26 |
| `test_integration_package.py` | 18 |
| `test_safety_module.py` | 22 |
| `test_reproducibility.py` | 22 |
| `test_hitl.py` | 28 |
| `test_community_benchmarks.py` | 18 |
| `test_batch_enhancements.py` | 16 |
| `test_many_objective.py` | 20 |
| `test_experiment_planner.py` | 15 |
| ... (58 more files) | ... |
| **Total: 147 files** | **6,383** |

---

## User Pain Points -> Solution Mapping

> The following 10 pain points come from real feedback by materials science / self-driving laboratory (SDL) users.
> Each item annotates **implemented modules** and **enhancement status**.

### Pain Point 1: Unreliable UQ leads to poor AL/BO point selection

| Module | Capability |
|--------|-----------|
| `diagnostics/` | `model_uncertainty` signal (#9) monitors uncertainty level in real-time |
| `diagnostics/` | `miscalibration_score` (#15) + `overconfidence_rate` (#16) track UQ health |
| `meta_controller/` | Auto-switches to Conservative + high exploration when uncertainty is high |
| `counterfactual/` | Offline counterfactual: "what if a different UQ/backend was used?" |
| `sensitivity/` | Decision sensitivity to uncertainty analysis |

### Pain Point 2: Weak constraint / infeasible region handling

| Module | Capability |
|--------|-----------|
| `feasibility/` | Feasibility learning + failure probability surface + danger zone identification |
| `feasibility_first/` | Safety boundary learning + KNN feasibility classifier + adaptive scoring |
| `constraints/` | Implicit constraint auto-discovery from failure patterns |
| `infrastructure/constraint_engine` | Explicit inequality/equality constraint handling with feasibility GP |

### Pain Point 3: Ubiquitous non-stationarity (instrument drift, reagent aging)

| Module | Capability |
|--------|-----------|
| `drift/` | 4-strategy drift detection + action mapping with two-layer FP control |
| `nonstationary/` | Time weighting + seasonal detection + comprehensive assessment |
| `data_quality/` | Instrument drift detection (`InstrumentDrift`) |

### Pain Point 4: Ignored noise & measurement quality

| Module | Capability |
|--------|-----------|
| `diagnostics/` | `noise_estimate`, `variance_contraction`, `signal_to_noise_ratio` |
| `stabilization/` | N-sigma outlier removal, moving average smoothing, recency/quality reweighting |
| `data_quality/` | Noise decomposition (experimental vs. model vs. drift) + batch effect detection |

### Pain Point 5: Multi-objective & preference complexity

| Module | Capability |
|--------|-----------|
| `multi_objective/` | Pareto front detection, dominance ranking, trade-off analysis, weighted scoring |
| `preference/` | Bradley-Terry MM preference learning from pairwise comparisons |
| `backends/NSGA2Sampler` | Native multi-objective optimization with crowding distance |

### Pain Point 6: Batch point clustering

| Module | Capability |
|--------|-----------|
| `batch/` | 3 decorrelation strategies: maximin, coverage, hybrid |
| `infrastructure/batch_scheduler` | Async trial scheduling with batch queue management |

### Pain Point 7: Poor evaluation & reproducibility

| Module | Capability |
|--------|-----------|
| `core/hashing` | Deterministic hashing: same snapshot + seed -> identical decisions |
| `replay/` | Deterministic replay: VERIFY / COMPARE / WHAT_IF modes |
| `compliance/` | Hash-chain audit + tamper detection + one-click compliance reports |
| `benchmark/` | Standardized multi-landscape x multi-seed comparison + leaderboard |

### Pain Point 8: Diverse data formats (curves / spectra / images)

| Module | Capability |
|--------|-----------|
| `feature_extraction/` | EIS, UV-Vis, XRD extractors + version-locked registry + consistency checks |
| `latent/` | PCA embedding for curve -> latent vector dimensionality reduction |

### Pain Point 9: High data entry barrier

| Module | Capability |
|--------|-----------|
| `store/` | Unified experiment store with row-level append + snapshot bridge |
| `ingestion/` | CSV/JSON auto-import + column type inference + role guessing |
| `problem_builder/` | Fluent API + interactive guide mode |

### Pain Point 10: Starting from scratch every campaign

| Module | Capability |
|--------|-----------|
| `meta_learning/` (7 sub-modules) | Cross-project meta-learning: strategy, weights, thresholds, failure strategies, drift robustness |
| `infrastructure/transfer_learning` | Knowledge transfer from prior campaigns |

### Pain Point 11: No standardized data integration / provenance

| Module | Capability |
|--------|-----------|
| `integration/formats` | CSV/JSON/JSON-LD import/export with schema versioning |
| `integration/provenance` | W3C PROV-O compliant provenance chain with DAG lineage tracking |
| `integration/connectors` | Abstract lab connector interface with CSV, JSON, and in-memory adapters |

### Pain Point 12: Safety-first operation missing

| Module | Capability |
|--------|-----------|
| `safety/hazards` | 8-category hazard classification with 5 severity levels |
| `safety/monitor` | Real-time safety monitoring with graduated response (SAFE → EMERGENCY) |
| `safety/emergency` | Emergency escalation protocol with configurable thresholds and safe-region fallback |

### Pain Point 13: Campaign reproducibility / FAIR compliance

| Module | Capability |
|--------|-----------|
| `reproducibility/logger` | Append-only campaign event logging (8 event types) with JSONL serialization |
| `reproducibility/replay` | Deterministic campaign replay with reproducibility scoring |
| `reproducibility/fair` | FAIR metadata generation with schema.org JSON-LD export |

### Pain Point 14: Human-in-the-loop not supported

| Module | Capability |
|--------|-----------|
| `hitl/priors` | Expert prior injection with confidence-weighted candidate re-ranking |
| `hitl/autonomy` | Progressive autonomy (MANUAL → AUTONOMOUS) with EMA trust tracking |
| `hitl/steering` | Interactive optimization steering (focus/avoid regions, modify suggestions) |

### Pain Point 15: No community benchmark compatibility

| Module | Capability |
|--------|-----------|
| `community_benchmarks/olympus_compat` | 6 built-in Olympus surfaces with nearest-neighbor interpolation |
| `community_benchmarks/sdl_metrics` | Standardized SDL metrics: Acceleration Factor, Enhancement Factor, Degree of Autonomy |

### Pain Point 16: Many-objective / async batch / LLM planning

| Module | Capability |
|--------|-----------|
| `multi_objective/many_objective` | Hypervolume indicator, IGD, many-objective ranking |
| `multi_objective/interactive` | Interactive Pareto exploration with tradeoff analysis |
| `batch/async_executor` | PipeBO-style async execution with partial result incorporation |
| `batch/time_aware` | Time-aware scheduling minimizing wall-clock time |
| `agents/experiment_planner` | LLM-enhanced experiment planning with pragmatic fallback |

### Pain Point Coverage Summary

| # | Pain Point | Depth | Core Modules | Status |
|---|-----------|-------|--------------|--------|
| 1 | Unreliable UQ | Full | diagnostics, counterfactual, portfolio | All enhancements complete |
| 2 | Constraints / infeasible | Full | feasibility, constraints, feasibility_first | All enhancements complete |
| 3 | Non-stationarity | Full | drift, nonstationary, data_quality | All enhancements complete |
| 4 | Noise / measurement | Full | diagnostics, stabilization, data_quality, cost | All enhancements complete |
| 5 | Multi-objective | Full | multi_objective, preference, NSGA-II, many_objective, interactive | All enhancements complete |
| 6 | Batch clustering | Full | batch, feasibility_first, batch_scheduler, async_executor, time_aware | All enhancements complete |
| 7 | Reproducibility | Full | schema, replay, compliance, benchmark, reproducibility/ | All enhancements complete |
| 8 | Diverse data | Full | feature_extraction, latent | All enhancements complete |
| 9 | Data entry barrier | Full | store, ingestion, problem_builder | All enhancements complete |
| 10 | Starting from scratch | Full | meta_learning (7 sub-modules), transfer_learning | All enhancements complete |
| 11 | Data integration / provenance | Full | integration/ (formats, provenance, connectors) | **NEW** — complete |
| 12 | Safety-first operation | Full | safety/ (hazards, monitor, emergency) | **NEW** — complete |
| 13 | Reproducibility / FAIR | Full | reproducibility/ (logger, replay, fair) | **NEW** — complete |
| 14 | Human-in-the-loop | Full | hitl/ (priors, autonomy, steering) | **NEW** — complete |
| 15 | Community benchmarks | Full | community_benchmarks/ (olympus, sdl_metrics) | **NEW** — complete |
| 16 | Many-objective / async / LLM | Full | many_objective, interactive, async_executor, experiment_planner | **NEW** — complete |

---

## Technical Features

| Feature | Status |
|---------|--------|
| Pure Python, no heavy ML dependencies | Yes |
| Deterministic (same input -> same output) | Yes |
| Complete audit trail (hash-chain tamper-proof) | Yes |
| Compliance reports (one-click export) | Yes |
| Deterministic replay (VERIFY / COMPARE / WHAT_IF) | Yes |
| Plugin-extensible (marketplace + auto-culling) | Yes |
| Declarative DSL (JSON round-trip serialization) | Yes |
| Cross-domain generalization (10 domains verified) | Yes |
| Shadow mode (Agent vs. Baseline comparison) | Yes |
| SLO monitoring (latency p50/p95, drift FP, action FP) | Yes |
| Release gate automation (8 gate checks) | Yes |
| 6,383 tests (acceptance + unit/integration + 3-tier real data + SDL gap closure) | Yes |
| Type annotations throughout | Yes |
| Zero external runtime dependencies | Yes |
| Auto data import (CSV/JSON -> unified store -> snapshot) | Yes |
| Guided problem modeling (Fluent API + Guide mode) | Yes |
| Cross-project meta-learning (7 sub-learners) | Yes |
| Cold start safety (auto-fallback to static rules) | Yes |
| Meta-learning state persistence (JSON serialization) | Yes |
| 10 built-in optimization backends | Yes |
| 10+ infrastructure modules (constraints, cost, transfer, etc.) | Yes |
| REST API (FastAPI) + WebSocket streaming | Yes |
| CLI application (Click-based) | Yes |
| React TypeScript SPA dashboard | Yes |
| Pure-Python math library (28+ functions) | Yes |
| SVG-based visualization (12+ chart types) | Yes |
| KernelSHAP explainability (4 chart types) | Yes |
| SDL monitoring dashboards (4 panels) | Yes |
| Design space exploration (PCA / t-SNE / iSOM) | Yes |
| VSUP uncertainty-aware colormaps | Yes |
| Campaign engine (closed-loop iteration) | Yes |
| Code execution enforcement (traced results) | Yes |
| Agent layer (orchestrator + safety + guard) | Yes |
| Anomaly detection (3-layer) | Yes |
| Confounder governance (4 strategies) | Yes |
| Molecular encoding (NGramTanimoto) | Yes |
| Symbolic regression (genetic programming) | Yes |
| fANOVA interaction maps | Yes |
| Causal discovery (PC algorithm + do-operator) | Yes |
| Physics-informed kernels and priors | Yes |
| Hypothesis lifecycle management (BIC, Bayes factor) | Yes |
| Decision robustness (bootstrap, cross-model consistency) | Yes |
| Theory-data hybrid models (residual GP) | Yes |
| 27 REST API analysis endpoints | Yes |
| Continuous fingerprint similarity (RBF kernel) | Yes |
| Similarity-based meta-learning transfer | Yes |
| 4 scientific reasoning agents (Literature, Mechanism, Phase, SymReg) | Yes |
| GUM-compliant uncertainty propagation | Yes |
| Adversarial robustness testing | Yes |
| 3-tier real data integration tests (Tier 1 pipeline, Tier 2 stress, Tier 3 closed-loop) | Yes |
| Closed-loop optimization beats random baseline (proven) | Yes |
| Data integration with provenance (W3C PROV-O) | Yes |
| Safety-first optimization (graduated hazard response) | Yes |
| Campaign reproducibility with FAIR metadata | Yes |
| Human-in-the-loop (progressive autonomy, expert priors, steering) | Yes |
| Community benchmark compatibility (6 Olympus surfaces, SDL metrics) | Yes |
| Many-objective optimization (hypervolume, IGD) | Yes |
| Async batch execution with partial incorporation | Yes |
| LLM experiment planning with pragmatic fallback | Yes |

---

## Project Structure

```
optimization_copilot/
├── _analysis/          # Analysis engine (KernelSHAP)
├── agents/             # Agent layer: execution trace, pipeline, guard, orchestrator
│   ├── literature/     # Literature mining agent
│   ├── mechanism/      # Mechanism design agent
│   ├── phase_structure/ # Phase structure analysis agent
│   ├── symreg/         # Symbolic regression agent
│   └── experiment_planner.py  # LLM experiment planning agent
├── anomaly/            # Three-layer anomaly detection
├── campaign/           # Campaign engine: surrogate, ranker, stage_gate, output, loop
├── candidate_pool/     # External molecular library management
├── causal/            # Causal discovery engine (PC algorithm, do-operator, counterfactuals)
├── case_studies/       # Real experimental benchmarks
├── confounder/         # Confounder detection and correction
├── core/                # Data models + deterministic hashing
├── diagnostics/         # 17-signal diagnostic engine
├── profiler/            # 8-dimensional problem fingerprinting
├── meta_controller/     # Core intelligence: phase orchestration + strategy selection
├── backends/            # 10 built-in optimization algorithms
│   ├── builtin.py       # Random, LHS, TPE, Sobol, GP-BO, RF-BO, CMA-ES, DE, NSGA-II, TuRBO
│   └── _math/           # Pure-Python math library (linalg, stats, sobol, kernels, acquisition)
├── plugins/             # Plugin base class + registry + governance
├── portfolio/           # Algorithm portfolio learning + multi-dimensional scoring
├── marketplace/         # Plugin marketplace + health tracking + auto-culling
├── composer/            # Multi-stage pipeline orchestration
├── stabilization/       # Data cleaning and preprocessing
├── data_quality/        # Noise decomposition + batch effect detection
├── extractors/         # Uncertainty-aware KPI extractors
├── feature_extraction/  # Curve feature extraction (EIS, UV-Vis, XRD)
├── fidelity/           # Multi-fidelity cost configuration
├── imputation/         # Deterministic missing-value imputation
├── representation/     # Molecular encoding (NGramTanimoto)
├── robustness/        # Decision robustness (bootstrap, stability, consistency)
├── screening/           # High-dimensional variable screening
├── surgery/             # Dimensionality reduction surgery
├── latent/              # PCA latent space reduction
├── domain_knowledge/   # Instrument specs and constraint library
├── drift/               # Drift detection + strategy adaptation
├── nonstationary/       # Time weighting + seasonal detection
├── curriculum/          # Progressive difficulty management
├── feasibility/         # Feasibility learning + failure surface + failure taxonomy
├── feasibility_first/   # Safety boundary + feasibility classifier + safety-first scoring
├── hybrid/            # Theory-data hybrid models (residual GP, discrepancy detection)
├── hypothesis/        # Hypothesis lifecycle (generation, testing, tracking)
├── constraints/         # Implicit constraint discovery
├── multi_objective/     # Pareto front + many-objective + interactive exploration
├── preference/          # Preference learning (Bradley-Terry)
├── cost/                # Cost-aware analysis
├── batch/               # Batch diversification + async executor + time-aware scheduling
├── multi_fidelity/      # Multi-fidelity planning (successive halving)
├── explain/            # Interaction maps, equation discovery, insight reports
├── explainability/      # Human-readable decision explanations
├── explanation_graph/   # DAG-form explanation graphs
├── reasoning/           # Templated reasoning explanations
├── sensitivity/         # Parameter sensitivity analysis
├── compliance/          # Audit chain + compliance reports + compliance engine
├── schema/              # Versionable decision rules
├── replay/              # Decision log + deterministic replay engine
├── dsl/                 # Declarative DSL + conversion bridge + validation
├── engine/              # Full lifecycle orchestration engine + state management
├── benchmark/           # Benchmark runner + leaderboard
├── benchmark_generator/ # Synthetic benchmark generator
├── counterfactual/      # Counterfactual evaluation
├── validation/          # Golden scenarios + regression verification
├── uncertainty/        # Shared uncertainty types and propagation
├── workflow/           # Multi-stage experimental DAGs
├── benchmark_protocol/ # SDL benchmark evaluation
├── store/               # Unified experiment store
├── ingestion/           # Data auto-import agent
├── problem_builder/     # Guided problem modeling
├── meta_learning/       # Cross-project meta-learning (7 sub-modules)
├── infrastructure/      # Infrastructure stack (10+ modules)
├── physics/           # Physics-informed modeling (kernels, priors, ODE solver)
├── integration/       # Data integration (formats, provenance, connectors)
├── safety/            # Safety-first optimization (hazards, monitor, emergency)
├── reproducibility/   # Campaign reproducibility (logger, replay, FAIR)
├── hitl/              # Human-in-the-loop (priors, autonomy, steering)
├── community_benchmarks/  # Olympus compatibility + SDL performance metrics
├── uncertainty/       # Measurement uncertainty types and propagation
├── platform/            # Platform services (auth, campaign manager, events, workspace, RAG)
├── api/                 # FastAPI REST endpoints + WebSocket
├── cli_app/             # Click-based CLI application
├── visualization/       # SVG visualization (VSUP, SHAP, SDL, design space, hexbin)
├── _analysis/           # Analysis engine (KernelSHAP)
├── web/                 # React TypeScript SPA
└── config.py            # Environment configuration

tests/                   # 5,947 tests, 139 files
├── test_tier1_endtoend.py       # 10 tests: full 8-stage pipeline
├── test_tier2_stress.py         # 14 tests: drift/batch/confounder stress
├── test_tier3_closedloop.py     # 19 tests: closed-loop optimization
├── test_adversarial_robustness.py # 14 tests: adversarial attacks
├── test_continuous_similarity.py # 26 tests: RBF kernel similarity
├── test_acceptance.py           # Acceptance tests (categories 1-2)
├── test_acceptance_benchmarks.py # Acceptance tests (categories 3-10)
├── test_integration.py          # 146 integration tests
├── test_infrastructure.py       # 90 infrastructure tests
├── test_meta_learning.py        # 82 meta-learning tests
├── test_feature_extraction.py   # 80 feature extraction tests
├── test_viz_*.py                # 10 visualization test files
├── test_shap_values.py          # KernelSHAP tests
├── test_eigen_linalg.py         # Eigendecomposition tests
├── ... (63 more files)
└── total: 5,947 tests
```
