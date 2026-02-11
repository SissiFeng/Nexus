# Optimization Copilot — 功能概览

> **一句话定位：** 一个智能优化决策层，能够根据实验历史自动选择、切换、调整优化策略，并给出可追溯的决策解释。支持跨领域泛化、合规审计、确定性重放、数据自动导入与跨项目元学习。

---

## 架构总览

```
  Raw Data (CSV/JSON)                 OptimizationSpec (DSL声明)
        │                                      │
        ▼                                      │
┌──────────────────┐                           │
│ DataIngestionAgent│                           │
│ (自动解析/列推断)   │                           │
└───────┬──────────┘                           │
        ▼                                      │
┌──────────────────┐   ┌──────────────────┐    │
│ ExperimentStore  │──▶│ ProblemBuilder   │────┘
│ (统一实验存储)     │   │ (引导式建模)      │
└───────┬──────────┘   └──────────────────┘
        │                       │
        ▼                       ▼
                    ┌───────────────────────┐
                    │   OptimizationEngine  │
                    │  (全生命周期编排引擎)     │
                    └───────────┬───────────┘
                                │
         CampaignSnapshot (实验数据)
                │
        ┌───────┴────────┐
        ▼                ▼
┌──────────────┐  ┌───────────────┐  ┌────────────────┐
│ Diagnostic   │  │ Problem       │  │ Data Quality   │
│ Engine       │  │ Profiler      │  │ Engine         │
│ (17信号)      │  │ (8维指纹)      │  │ (噪声/批效应)   │
└──────┬───────┘  └──────┬────────┘  └───────┬────────┘
       │                 │                    │
       ▼                 ▼                    ▼
  ┌──────────────────────────────────────────────────┐
  │              Meta-Controller                      │
  │  阶段检测 → 策略选择 → 风险评估 → 后端调度           │
  │                                                   │
  │  Portfolio ◄── Drift ◄── NonStationary ◄── Cost   │
  │       ▲                                           │
  │       │  MetaLearningAdvisor (跨项目元学习)          │
  │       │  策略学习 + 权重调优 + 阈值学习 + 漂移鲁棒     │
  └────────────┬────────────┬────────────┬────────────┘
               │            │            │
      ┌────────┘    ┌───────┘    ┌───────┘
      ▼             ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│Stabilizer│ │Screener/ │ │Feasibility   │
│ 数据清洗   │ │Surgery   │ │First + Safety│
└──────────┘ │变量筛选    │ │ 可行域学习     │
             └──────────┘ └──────────────┘
               │
               ▼
      ┌─────────────────┐     ┌──────────────────┐
      │ DecisionExplainer│     │ Compliance Engine │
      │ + ExplanationGraph│    │ 审计链 + 合规报告    │
      └─────────────────┘     └──────────────────┘
               │                       │
               ▼                       ▼
         DecisionLog  ────→  ReplayEngine (确定性重放)
                │
                ▼
        ExperienceStore (跨campaign学习记忆)
```

---

## 模块清单（49 个模块）

### 一、核心智能

#### 1. 核心数据模型 (`core/`)

| 数据结构 | 说明 |
|---------|------|
| `CampaignSnapshot` | 优化 campaign 的完整快照：参数规格、观测历史、目标定义 |
| `StrategyDecision` | 输出决策：后端选择、探索强度、批次大小、风险态度、审计轨迹 |
| `ProblemFingerprint` | 8 维问题指纹，自动分类问题类型 |
| `Observation` | 单次实验观测（参数、KPI、失败标记、时间戳） |
| `StabilizeSpec` | 数据预处理策略（降噪、异常值、失败处理） |

**确定性哈希：** `snapshot_hash()` / `decision_hash()` / `diagnostics_hash()` — 相同输入 + 相同 seed = 完全相同的输出，每次决策都可追溯。

---

#### 2. 诊断信号引擎 (`diagnostics/`)

从实验历史中实时计算 **17 个健康信号**：

| # | 信号 | 含义 |
|---|------|------|
| 1 | `convergence_trend` | 最优值的收敛趋势（线性回归斜率） |
| 2 | `improvement_velocity` | 近期改善速度 vs 历史速度 |
| 3 | `variance_contraction` | KPI 方差是否在收缩（收敛标志） |
| 4 | `noise_estimate` | 近期 KPI 的噪声水平（变异系数） |
| 5 | `failure_rate` | 失败实验比例 |
| 6 | `failure_clustering` | 失败是否集中在近期 |
| 7 | `feasibility_shrinkage` | 可行域是否在缩小 |
| 8 | `parameter_drift` | 最优参数是否还在漂移 |
| 9 | `model_uncertainty` | 模型不确定性的代理指标 |
| 10 | `exploration_coverage` | 参数空间的探索覆盖率 |
| 11 | `kpi_plateau_length` | KPI 停滞了多少轮 |
| 12 | `best_kpi_value` | 当前最佳 KPI 值 |
| 13 | `data_efficiency` | 每次实验带来的平均改善 |
| 14 | `constraint_violation_rate` | 约束违反率 |
| 15 | `miscalibration_score` | UQ 校准误差 (ECE) |
| 16 | `overconfidence_rate` | 预测区间过窄比例 |
| 17 | `signal_to_noise_ratio` | KPI 信噪比（|mean|/std） |

---

#### 3. 问题指纹分析器 (`profiler/`)

自动从数据中推断问题的 **8 个维度**：

| 维度 | 分类值 | 分类依据 |
|------|--------|---------|
| 变量类型 | continuous / discrete / categorical / mixed | 参数规格 |
| 目标形式 | single / multi_objective / constrained | 目标数和约束 |
| 噪声水平 | low / medium / high | KPI 变异系数 |
| 成本分布 | uniform / heterogeneous | 时间戳间隔分析 |
| 失败信息量 | weak / strong | 失败点的参数多样性 |
| 数据规模 | tiny(<10) / small(<50) / moderate(50+) | 观测数量 |
| 时序特征 | static / time_series | 滞后 1 自相关 |
| 可行域 | wide / narrow / fragmented | 失败率 |

---

#### 4. 元控制器 (`meta_controller/`)

**核心智能模块** — 五阶段自动编排：

```
Cold Start ──→ Learning ──→ Exploitation
                  │               │
                  ▼               ▼
             Stagnation ←────────┘
                  │
                  ▼
            Termination
```

| 阶段 | 触发条件 | 探索强度 | 风险态度 | 推荐后端 |
|------|---------|---------|---------|---------|
| **Cold Start** | 观测数 < 10 | 0.9（高探索） | Conservative | LHS, Random |
| **Learning** | 数据充足，尚在学习 | 0.6（均衡） | Moderate | TPE, RF Surrogate |
| **Exploitation** | 收敛趋势强 + 不确定性低 | 0.2（强利用） | Aggressive | TPE, CMA-ES |
| **Stagnation** | KPI 长期停滞 / 失败激增 | 0.8（重启探索） | Conservative | Random, LHS |
| **Termination** | 协作终止判断 | 0.1 | — | TPE |

**自动适应特性：**
- 根据问题指纹自动调整后端优先级（如高噪声 → 优先 Random/LHS）
- 首选后端不可用时自动 fallback，并记录事件
- 探索强度根据覆盖率、噪声、数据规模动态微调
- 接受 `backend_policy`（allowlist/denylist）做治理约束
- 接受 `drift_report` 和 `cost_signals` 做自适应调整

---

### 二、算法管理

#### 5. 优化后端池 (`backends/` + `plugins/`)

**插件架构** — 可扩展的算法注册表：

内置 3 个算法后端：

| 后端 | 用途 | 特点 |
|------|------|------|
| `RandomSampler` | 基线 / Cold Start | 均匀随机采样 |
| `LatinHypercubeSampler` | 空间填充设计 | 分层采样，覆盖性好 |
| `TPESampler` | 有历史数据的优化 | 基于好/坏分组的贝叶斯方法 |

```python
class AlgorithmPlugin(ABC):
    def name(self) -> str: ...
    def fit(self, observations, parameter_specs) -> None: ...
    def suggest(self, n_suggestions, seed) -> list[dict]: ...
    def capabilities(self) -> dict: ...
```

支持 `BackendPolicy`（allowlist / denylist）治理策略。

---

#### 6. 算法组合学习器 (`portfolio/`)

| 类 | 说明 |
|----|------|
| `AlgorithmPortfolio` | 按问题指纹记录每个后端的历史表现，提供排名 |
| `BackendScorer` | 多维加权评分：历史表现 + 指纹匹配 + 不兼容惩罚 + 成本信号 |
| `BackendScore` | 评分明细（每个维度的贡献），支持可解释的后端选择 |
| `ScoringWeights` | 评分权重配置（history, fingerprint_match, incompatibility, cost） |

---

#### 7. 插件市场 (`marketplace/`)

| 类 | 说明 |
|----|------|
| `Marketplace` | 插件注册表 + 健康追踪 + 自动淘汰策略 |
| `CullPolicy` | 淘汰策略（根据失败率、历史表现自动下架不健康的插件） |
| `MarketplaceStatus` | 插件状态（active / probation / culled） |

---

#### 8. 流水线编排器 (`composer/`)

| 类 | 说明 |
|----|------|
| `AlgorithmComposer` | 选择、编排多阶段优化流水线（如：先筛选，后精调） |
| `PipelineStage` | 流水线阶段定义（后端、退出条件、转移规则） |
| `PIPELINE_TEMPLATES` | 内置模板：`exploration_first`、`screening_then_optimize`、`restart_on_stagnation` |

---

### 三、数据处理

#### 9. 数据稳定器 (`stabilization/`)

| 策略 | 说明 |
|------|------|
| 失败处理 | `exclude`（排除）/ `penalize`（保留标记）/ `impute`（用最差值填充） |
| 异常值剔除 | 基于 N-sigma 规则自动剔除 |
| 重加权 | `recency`（近期权重更高）/ `quality`（高质量数据权重更高） |
| 噪声平滑 | 移动平均窗口 |

---

#### 10. 数据质量引擎 (`data_quality/`)

| 类 | 说明 |
|----|------|
| `DataQualityReport` | 综合数据质量报告 |
| `NoiseDecomposition` | 噪声分解：实验噪声 vs 模型噪声 vs 系统漂移 |
| `BatchEffect` | 批效应检测：不同实验批次间的系统性偏差 |
| `InstrumentDrift` | 仪器漂移检测 |

---

#### 11. 特征提取器 (`feature_extraction/`)

| 类 | 说明 |
|----|------|
| `FeatureExtractor` | ABC 基类，从测量曲线中提取命名标量特征 |
| `CurveData` | 曲线数据容器 |
| `ExtractedFeatures` | 特征提取结果 |
| `EISNyquistExtractor` | EIS 阻抗谱提取器（溶液电阻、极化电阻、半圆直径、Warburg 斜率） |
| `UVVisExtractor` | UV-Vis 光谱提取器（峰位、峰高、半高宽 FWHM、总吸光度） |
| `XRDPatternExtractor` | XRD 衍射谱提取器（主峰角度、峰数、结晶度指数、背景水平） |
| `VersionLockedRegistry` | 带 allowlist/denylist + 版本锁定的提取器注册表 |
| `CurveEmbedder` | PCA 降维：曲线 → 低维 latent 向量（power iteration 实现） |
| `check_extractor_consistency()` | 一致性验证：确保同数据 + 同版本 → 同 KPI |

---

#### 12. 变量筛选器 (`screening/`)

- **重要性排序**：基于参数-KPI 相关性的重要性评分
- **方向性提示**：每个参数对 KPI 是正向还是负向影响
- **交互检测**：通过乘积项相关性检测参数间的交互作用
- **步长推荐**：根据重要性自动推荐搜索步长

---

#### 13. 参数手术器 (`surgery/`)

| 类 | 说明 |
|----|------|
| `Surgeon` | 基于筛选结果诊断并执行维度削减（排除不重要参数） |

---

#### 14. 潜在空间降维 (`latent/`)

| 类 | 说明 |
|----|------|
| `LatentTransform` | 纯 stdlib 实现的 PCA（幂迭代法），无需 numpy |
| `LatentOptimizer` | 控制何时/如何应用降维（高维问题自动触发） |

---

### 四、自适应智能

#### 15. 漂移检测器 (`drift/`)

| 类 | 说明 |
|----|------|
| `DriftDetector` | 多策略概念漂移检测（KPI 阶跃、参数-KPI 相关性变化、残差分析） |
| `DriftReport` | 漂移报告（`drift_detected` 标志 + 各维度检测结果） |
| `DriftStrategyAdapter` | 将漂移信号映射为策略调整（阶段重置、探索增强等） |
| `DriftAction` | 具体漂移响应动作 |

**两层误报控制：**
- **检测器 FP < 15%**：平稳数据上不频繁误报
- **动作 FP < 5%**：漂移误报不会实际导致策略变更

---

#### 16. 非平稳环境适配 (`nonstationary/`)

| 类 | 说明 |
|----|------|
| `NonStationaryAdapter` | 整合时间加权 + 季节检测 + 漂移信号的综合评估 |
| `SeasonalDetector` | 基于自相关分析的周期模式检测 |
| `TimeWeighter` | 基于时间衰减的观测权重计算 |

---

#### 17. 课程学习引擎 (`curriculum/`)

| 类 | 说明 |
|----|------|
| `CurriculumEngine` | 渐进式难度管理，参数排名 + 搜索范围逐步展宽 |

---

### 五、安全与可行性

#### 18. 可行域学习器 (`feasibility/`)

| 类 | 说明 |
|----|------|
| `FeasibilityLearner` | 从失败数据中学习安全区域、安全边界、危险区域 |
| `FailureSurface` | 失败概率曲面学习：估计安全边界和危险区域 |
| `FailureClassifier` | 结构化失败分类学（hardware / chemistry / data / protocol / unknown） |
| `FailureTaxonomy` | 失败分类结果 + 统计分布 |

---

#### 19. 安全优先评分 (`feasibility_first/`)

| 类 | 说明 |
|----|------|
| `SafetyBoundaryLearner` | 基于分位数估计学习保守安全参数边界 |
| `FeasibilityClassifier` | KNN 分类器预测候选方案可行性 + 置信度评分 |
| `FeasibilityFirstScorer` | 自适应混合可行性评分与目标评分（根据失败率动态调权） |

---

#### 20. 约束发现 (`constraints/`)

| 类 | 说明 |
|----|------|
| `ConstraintDiscoverer` | 从优化历史中发现隐式约束（阈值检测 + 交互检测） |
| `DiscoveredConstraint` | 发现的约束描述 |
| `ConstraintReport` | 约束发现报告 |

---

### 六、多目标与偏好

#### 21. 多目标优化分析 (`multi_objective/`)

- **Pareto 前沿检测**：非支配排序，找出当前最优解集
- **支配排名**：所有观测的逐层排名（Rank 1 = Pareto 前沿）
- **权衡分析**：目标对之间的相关性分析（冲突 / 协同 / 独立）
- **加权评分**：用户自定义权重的标量化评分

---

#### 22. 偏好学习 (`preference/`)

| 类 | 说明 |
|----|------|
| `PreferenceLearner` | 从成对偏好中学习效用评分（Bradley-Terry MM 算法） |

---

### 七、效率与成本

#### 23. 成本感知分析 (`cost/`)

| 类 | 说明 |
|----|------|
| `CostAnalyzer` | 成本感知优化分析：预算压力、效率指标、探索调整 |
| `CostSignals` | 成本信号（花费、效率、预算剩余），传入 MetaController 影响策略 |

---

#### 24. 批次多样化 (`batch/`)

| 类 | 说明 |
|----|------|
| `BatchDiversifier` | 批次多样化策略（maximin / coverage / hybrid） |
| `BatchPolicy` | 确保批内参数配置的多样性，避免重复采样 |

---

#### 25. 多保真度规划 (`multi_fidelity/`)

| 类 | 说明 |
|----|------|
| `MultiFidelityPlanner` | 两阶段优化规划：廉价筛选 + 昂贵精调 |
| `FidelityLevel` | 保真度级别定义 |
| `MultiFidelityPlan` | 包含逐次减半（successive halving）的执行计划 |

---

### 八、可解释性

#### 26. 决策解释器 (`explainability/`)

为每个策略决策生成**人类可读的报告**：

- **总结**：当前阶段 + 选择的策略 + 阶段变化
- **触发诊断**：哪些信号驱动了这个决策
- **阶段转换说明**：如果阶段变了，解释为什么
- **风险评估**：当前的风险态度和原因
- **覆盖状态**：参数空间探索了多少
- **不确定性评估**：对推荐的置信度

**原则：** 只报告算法实际计算的内容，不生成猜测性解释。

---

#### 27. 解释图 (`explanation_graph/`)

| 类 | 说明 |
|----|------|
| `GraphBuilder` | 从诊断、决策、失败曲面构建 DAG 形式的解释图 |

---

#### 28. 推理解释 (`reasoning/`)

| 类 | 说明 |
|----|------|
| `RewriteSuggestion` | 模板化人类可读解释（手术动作、campaign 状态） |
| `FailureCluster` | 失败聚类描述 |

---

#### 29. 参数敏感性分析 (`sensitivity/`)

| 类 | 说明 |
|----|------|
| `SensitivityAnalyzer` | 参数敏感性与决策稳定性分析（相关性 + 距离度量） |

---

### 九、合规与治理

#### 30. 合规审计系统 (`compliance/`)

| 类 | 说明 |
|----|------|
| `AuditEntry` | 单条审计记录（从 DecisionLogEntry 转换） |
| `AuditLog` | 哈希链式审计日志（每条记录链接前一条的哈希） |
| `verify_chain()` | 验证审计链完整性（检测篡改） |
| `ChainVerification` | 链验证结果（valid / first_broken_index） |
| `ComplianceReport` | 结构化合规报告：campaign 摘要、迭代日志、最终推荐、规则版本 |
| `ComplianceEngine` | 高层合规编排：审计日志 + 链验证 + 报告生成 |

**防篡改保证：** 如果任何一条记录被修改，`verify_chain()` 会精确报告第一个断裂位置。

---

#### 31. 决策规则引擎 (`schema/`)

| 类 | 说明 |
|----|------|
| `DecisionRule` | 可版本化、可审计的决策规则 |
| `RuleSignature` | 规则签名（用于固化 MetaController 的判断逻辑） |

---

#### 32. 决策重放 (`replay/`)

| 类 | 说明 |
|----|------|
| `DecisionLog` | 每轮审计轨迹：快照、诊断、决策、实验结果 |
| `DecisionLogEntry` | 单条日志（14 个字段），支持 JSON 序列化和文件读写 |
| `ReplayEngine` | 确定性重放引擎，三种模式： |
| | `VERIFY` — 验证历史决策可复现 |
| | `COMPARE` — 对比两个策略的历史选择 |
| | `WHAT_IF` — 假设分析（如果当时换了策略会怎样） |

---

### 十、实验编排

#### 33. 声明式 DSL (`dsl/`)

| 类 | 说明 |
|----|------|
| `OptimizationSpec` | 声明式优化规格（参数、目标、预算、约束） |
| `ParameterDef` | 参数定义（名称、类型、上下界、分类值） |
| `ObjectiveDef` | 目标定义（名称、方向 minimize/maximize） |
| `BudgetDef` | 预算定义（最大迭代、最大时间） |
| `SpecBridge` | DSL → 核心模型转换（ParameterSpec、CampaignSnapshot、ProblemFingerprint） |
| `SpecValidator` | 规格校验 + 人类可读错误消息 |

支持 JSON 序列化 / 反序列化，`to_dict()` / `from_dict()` 完整往返。

---

#### 34. 优化引擎 (`engine/`)

| 类 | 说明 |
|----|------|
| `OptimizationEngine` | 全生命周期编排：诊断 → 指纹 → 元控制 → 插件调度 → 结果记录 |
| `EngineConfig` | 引擎配置（最大迭代、批次大小、seed） |
| `EngineResult` | 引擎运行结果 |
| `CampaignState` | 可变 campaign 状态管理 + checkpoint / resume 序列化 |
| `Trial` | 单次试验的生命周期管理（pending → running → completed / failed） |
| `TrialBatch` | 批量试验操作 |

**输入校验：** 引擎拒绝空参数列表或空目标列表，给出清晰错误消息。

---

### 十一、基准测试

#### 35. 基准运行器 (`benchmark/`)

| 类 | 说明 |
|----|------|
| `BenchmarkRunner` | 在多景观 × 多 seed 下对比优化后端的标准化评估 |
| `BenchmarkResult` | 单次评估结果（AUC、best-so-far 序列、步数） |
| `Leaderboard` | 后端排名表，支持稳定性验证 |

---

#### 36. 合成基准生成器 (`benchmark_generator/`)

| 类 | 说明 |
|----|------|
| `SyntheticObjective` | 可配置的合成目标函数 |
| `LandscapeType` | 4 种景观：`SPHERE`、`ROSENBROCK`、`ACKLEY`、`RASTRIGIN` |
| `BenchmarkGenerator` | 生成完整的 campaign 场景（含噪声、失败、漂移、约束） |

参数维度：`n_dimensions`、`noise_sigma`、`failure_rate`、`failure_zones`、`constraints`、`n_objectives`、`drift_rate`、`has_categorical`、`categorical_effect`。

---

#### 37. 反事实评估 (`counterfactual/`)

| 类 | 说明 |
|----|------|
| `CounterfactualEvaluator` | 离线假设分析：如果换了后端，KPI 会怎样？ |
| `CounterfactualResult` | 反事实评估结果（加速比、KPI 范围） |

---

### 十二、验证系统

#### 38. 黄金场景 (`validation/`)

5 个黄金场景用于回归测试：

| 场景 | 模拟数据特征 | 期望行为 |
|------|------------|---------|
| `clean_convergence` | 30 次观测，单调递增 KPI | → Exploitation 阶段 |
| `cold_start` | 仅 4 次观测 | → Cold Start 阶段 |
| `failure_heavy` | 50% 失败率，集中在后半段 | → Stagnation 阶段 |
| `noisy_plateau` | 恒定 KPI + 微小噪声 | → Stagnation 阶段 |
| `mixed_variables` | 连续 + 分类参数 | → Learning 阶段 |

---

### 十三、数据入口层（Phase 1）

#### 39. 统一实验存储 (`store/`)

| 类 | 说明 |
|----|------|
| `ExperimentStore` | 持久化实验数据仓库：行级追加、campaign 级查询、快照导出 |
| `ExperimentRecord` | 单条实验记录（参数 + KPI + 元数据 + 时间戳） |
| `StoreQuery` | 灵活查询：按 campaign / 按参数范围 / 按时间窗口 / 按 KPI 范围 |
| `StoreStats` | 存储统计摘要：campaign 列表、参数列、总行数 |

**存储特性：**
- 纯 Python dict 后端，JSON 往返序列化
- 行级追加 + 批量导入（`add_records()` / `bulk_import()`）
- 自动 campaign ID 索引，O(1) campaign 查询
- 快照桥接：`to_campaign_snapshot()` 直接输出 `CampaignSnapshot`

---

#### 40. 数据导入代理 (`ingestion/`)

| 类 | 说明 |
|----|------|
| `DataIngestionAgent` | 自动解析 CSV/JSON + 列角色推断 + 交互式确认 |
| `ColumnProfile` | 列画像：类型推断、唯一值数、缺失率、数值范围 |
| `IngestionResult` | 导入结果：成功记录数 + 跳过数 + 错误列表 |
| `RoleMapping` | 列角色映射：parameter / kpi / metadata / ignore |

**导入能力：**
- CSV / JSON 自动格式检测
- 列类型推断（numeric / categorical / datetime / text）
- 基于启发式的角色猜测（参数列 vs KPI 列 vs 元数据列）
- 用户确认 → 自动写入 `ExperimentStore`
- 缺失值 / 类型不匹配的错误报告

---

#### 41. 问题建模器 (`problem_builder/`)

| 类 | 说明 |
|----|------|
| `ProblemBuilder` | Fluent API 构建 `OptimizationSpec`：链式调用 `.add_parameter().set_objective().build()` |
| `ProblemGuide` | 引导式对话建模：逐步询问参数、目标、约束，生成完整 spec |
| `BuilderValidation` | 构建期校验：参数名唯一性、边界合法性、目标方向检查 |

**建模方式：**
- **Fluent 模式**：程序化链式调用，适合脚本集成
- **Guide 模式**：交互式引导，适合新用户上手
- 两种模式均输出标准 `OptimizationSpec`，可直接传入 `OptimizationEngine`

---

### 十四、跨项目元学习（Phase 2）

#### 42. 元学习数据模型 (`meta_learning/models.py`)

| 数据结构 | 说明 |
|---------|------|
| `CampaignOutcome` | 完成的 campaign 总结：指纹、阶段转移、后端表现、失败类型、最佳 KPI |
| `BackendPerformance` | 单后端表现记录：收敛轮次、遗憾值、样本效率、失败率、漂移影响 |
| `ExperienceRecord` | 经验记录：`CampaignOutcome` + 指纹键 |
| `MetaLearningConfig` | 元学习配置：冷启动阈值、相似度衰减、EMA 学习率、近因半衰期 |
| `LearnedWeights` | 学习到的评分权重（gain / fail / cost / drift / incompatibility） |
| `LearnedThresholds` | 学习到的阶段切换阈值（冷启动观测数 / 学习平台期 / 利用增益阈值） |
| `FailureStrategy` | 失败类型 → 最佳稳定化策略映射 |
| `DriftRobustness` | 后端漂移鲁棒性评分（韧性分 + KPI 损失） |
| `MetaAdvice` | 元学习建议输出：推荐后端 + 评分权重 + 切换阈值 + 失败策略 + 漂移鲁棒后端 |

---

#### 43. 跨 campaign 经验存储 (`meta_learning/experience_store.py`)

| 类 | 说明 |
|----|------|
| `ExperienceStore` | 跨 campaign 持久化经验库：记录 outcome、按指纹查询、相似指纹检索 |

**核心能力：**
- 按 campaign ID 精确查询 / 按指纹键聚合查询
- **指纹相似度**：8 维逐维比较（枚举相等 → 1.0 / 不等 → 0.0），平均得相似度
- **近因加权**：`recency_halflife` 控制旧经验衰减
- JSON 序列化 / 反序列化

---

#### 44. 策略学习器 (`meta_learning/strategy_learner.py`)

| 类 | 说明 |
|----|------|
| `StrategyLearner` | 指纹 → 后端亲和力学习：精确匹配 + 相似匹配 fallback，KNN 风格排名 |

**算法：**
1. 精确匹配：查找相同指纹的历史经验
2. 相似匹配回退：按 `similarity_decay` 加权相似指纹经验
3. 排名公式：`weighted_avg(sample_efficiency) - weighted_avg(regret) - weighted_avg(failure_rate)`
4. 冷启动安全：经验不足时返回空列表，调用者回退到静态规则

---

#### 45. 权重调优器 (`meta_learning/weight_tuner.py`)

| 类 | 说明 |
|----|------|
| `WeightTuner` | 基于 EMA 学习最优 `ScoringWeights`：高失败 → 提高 fail 权重，高漂移 → 提高 drift 权重 |

**特性：** 信号驱动启发式调整 + 权重归一化（总和 = 1.0）+ 置信度饱和（~10 campaigns）

---

#### 46. 阈值学习器 (`meta_learning/threshold_learner.py`)

| 类 | 说明 |
|----|------|
| `ThresholdLearner` | 从阶段转移时机学习 `SwitchingThresholds`：EMA 融合好结果的转移轮次 |

**算法：** 从 `phase_transitions` 提取实际转移轮次 → 仅学习低遗憾（质量 > 1.0）的经验 → EMA 更新阈值

---

#### 47. 失败策略学习器 (`meta_learning/failure_learner.py`)

| 类 | 说明 |
|----|------|
| `FailureStrategyLearner` | 失败类型 → 稳定化方案映射：按平均 outcome 质量排名 |

**特性：** 跨指纹聚合（失败策略倾向于通用化）+ 有效性评分

---

#### 48. 漂移鲁棒性追踪器 (`meta_learning/drift_learner.py`)

| 类 | 说明 |
|----|------|
| `DriftRobustnessTracker` | 追踪各后端在漂移条件下的表现：韧性分 = 1.0 − KPI 损失 |

**特性：** 分离漂移/非漂移基线 → 计算 KPI 损失 → 排名推荐漂移鲁棒后端

---

#### 49. 元学习顾问 (`meta_learning/advisor.py`)

| 类 | 说明 |
|----|------|
| `MetaLearningAdvisor` | 顶层编排器：组合 5 个子学习器，输出统一 `MetaAdvice` |

**`advise()` 流程：**
1. `StrategyLearner.rank_backends()` → 推荐后端排名
2. `WeightTuner.suggest_weights()` → 学习到的评分权重
3. `ThresholdLearner.suggest_thresholds()` → 学习到的切换阈值
4. `FailureStrategyLearner.suggest_all()` → 失败调整策略
5. `DriftRobustnessTracker.rank_by_resilience()` → 漂移鲁棒后端
6. 计算综合置信度 + 生成 `reason_codes` 解释链

**`learn_from_outcome()` 流程：**
- 记录 outcome → 更新权重 → 更新阈值 → 更新失败策略 → 更新漂移追踪

**集成方式（纯注入，不修改已有文件）：**

| 已有组件 | 注入点 | MetaAdvice 提供 |
|---------|--------|----------------|
| `MetaController.decide(portfolio=...)` | portfolio 参数 | `recommended_backends` → `AlgorithmPortfolio` |
| `BackendScorer(weights=...)` | 构造器参数 | `scoring_weights` → `ScoringWeights` |
| `MetaController(thresholds=...)` | 构造器参数 | `switching_thresholds` → `SwitchingThresholds` |
| `FailureTaxonomy` | MetaController 消费 | `failure_adjustments` |
| `DriftReport` | MetaController 消费 | `drift_robust_backends` |

---

## 端到端使用流程

```python
from optimization_copilot.core import CampaignSnapshot
from optimization_copilot.diagnostics import DiagnosticEngine
from optimization_copilot.profiler import ProblemProfiler
from optimization_copilot.meta_controller import MetaController
from optimization_copilot.explainability import DecisionExplainer

# 1. 构造实验快照
snapshot = CampaignSnapshot(
    campaign_id="my_experiment",
    parameter_specs=[...],
    observations=[...],
    objective_names=["yield"],
    objective_directions=["maximize"],
)

# 2. 计算诊断信号
engine = DiagnosticEngine()
diagnostics = engine.compute(snapshot)

# 3. 分析问题指纹
profiler = ProblemProfiler()
fingerprint = profiler.profile(snapshot)

# 4. 获取策略决策
controller = MetaController()
decision = controller.decide(snapshot, diagnostics.to_dict(), fingerprint, seed=42)

# 5. 生成决策报告
explainer = DecisionExplainer()
report = explainer.explain(decision, fingerprint, diagnostics.to_dict())

print(f"阶段: {decision.phase.value}")
print(f"推荐后端: {decision.backend_name}")
print(f"探索强度: {decision.exploration_strength}")
print(f"风险态度: {decision.risk_posture.value}")
print(f"批次大小: {decision.batch_size}")
print(f"报告: {report.summary}")
```

### 声明式 DSL 使用

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
# engine 会自动完成：诊断 → 指纹 → 元控制 → 后端调度 → 审计记录
```

### 数据导入 + 自动建模（Phase 1）

```python
from optimization_copilot.store.store import ExperimentStore
from optimization_copilot.ingestion.agent import DataIngestionAgent
from optimization_copilot.problem_builder.builder import ProblemBuilder

# 1. 从 CSV 自动导入
store = ExperimentStore()
agent = DataIngestionAgent(store)
result = agent.ingest_csv("experiments.csv", campaign_id="formulation_v3")
print(f"导入 {result.n_imported} 条记录，跳过 {result.n_skipped} 条")

# 2. 从存储直接导出为 CampaignSnapshot
snapshot = store.to_campaign_snapshot("formulation_v3")

# 3. 用 Fluent API 构建优化规格
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

### 跨项目元学习（Phase 2）

```python
from optimization_copilot.meta_learning import (
    MetaLearningAdvisor, CampaignOutcome, BackendPerformance,
)
from optimization_copilot.profiler import ProblemProfiler

# 1. 创建元学习顾问
advisor = MetaLearningAdvisor()

# 2. 记录已完成 campaign 的 outcome
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

# 3. 下次新 campaign 获取元学习建议
advice = advisor.advise(new_fingerprint)
print(f"推荐后端: {advice.recommended_backends}")
print(f"置信度: {advice.confidence:.2f}")
print(f"原因: {advice.reason_codes}")

# 4. 注入 MetaController
controller = MetaController(thresholds=advice.switching_thresholds)
scorer = BackendScorer(weights=advice.scoring_weights)

# 5. 持久化（跨会话保存元学习状态）
state_json = advisor.to_json()
# ... 下次加载 ...
advisor = MetaLearningAdvisor.from_json(state_json)
```

### 合规审计使用

```python
from optimization_copilot.replay.log import DecisionLog
from optimization_copilot.compliance.audit import AuditLog, verify_chain
from optimization_copilot.compliance.report import ComplianceReport

# 加载决策日志
log = DecisionLog.load("campaign_log.json")

# 构建审计链
audit_log = AuditLog.from_decision_log(log)
verification = verify_chain(audit_log)
assert verification.valid, f"审计链断裂于第 {verification.first_broken_index} 条"

# 生成合规报告
report = ComplianceReport.from_audit_log(audit_log)
print(report.format_text())
```

---

## 跨领域泛化（已验证）

通过问题指纹 + 插件架构，系统经过 **10 个领域 × 10 个 seed = 100 组** 自动化验证：

| # | 领域 | 景观 | 维度 | 噪声 | 特殊配置 |
|---|------|-----|------|------|---------|
| 1 | 电化学 | Rosenbrock | 4 | 0.15 | 5% 失败率 |
| 2 | 化学合成 | Sphere | 5 | 0.02 | 混合变量 |
| 3 | 配方优化 | Ackley | 8 | 0.05 | 双目标 + 约束 |
| 4 | 生物检测 | Sphere | 2 | 0.40 | 高噪声 |
| 5 | 聚合物 | Rastrigin | 10 | 0.08 | 10% 失败 + 高维 |
| 6 | 药物发现 | Ackley | 7 | 0.10 | 25% 失败 + 混合变量 |
| 7 | 过程工程 | Rosenbrock | 5 | 0.05 | 漂移 (drift_rate=0.02) |
| 8 | 材料科学 | Rastrigin | 6 | 0.10 | 15% 失败 + 混合变量 |
| 9 | 农业优化 | Sphere | 4 | 0.20 | 双目标 + 漂移 |
| 10 | 能源系统 | Rosenbrock | 8 | 0.05 | 三目标 + 约束 + 5% 失败 |

**验证结论：**
- 无单一后端退化：全局无后端占比 > 80%
- 指纹桶多样性：10 个领域产生 ≥ 4 个不同的指纹桶
- 冷启动安全：空 portfolio 时所有领域都能正常决策
- 高失败域优雅降级：25% 失败率下仍产生合法决策 + reason_codes

---

## 测试体系

**1957 个测试**，51 个测试文件，全部通过（<1s）：

### 验收测试（`test_acceptance.py` + `test_acceptance_benchmarks.py`）

| 类别 | 测试数 | 验证内容 |
|------|-------|---------|
| 1. 端到端流水线 | 14 | 全链路运行、黄金场景回归 |
| 2. 确定性与审计 | 8 | 哈希一致性、审计轨迹完整性 |
| 3. 合成基准 | 11 | 4 景观 × clean/noisy、AUC、步数 |
| 4. 约束/失败/漂移/多目标 | 12 | 约束发现、失败分类、漂移检测、Pareto |
| 5. 消融/反事实/排行榜 | 27 | 模块贡献、Kendall-tau 一致性、排行稳定性 |
| 6. 影子模式 | 11 | Agent vs Baseline 对比、denylist 合规、安全平价 |
| 7. 监控与 SLO | 11 | p50 < 50ms、p95 < 200ms、漂移 FP < 15%、动作 FP < 5% |
| 8. 发布门 v1 | 8 | 确定性 100%、零安全违规、AUC ≥ 60%、回归 < 2% |
| 9. 跨领域泛化 | 7 | 10 领域多样性、portfolio fallback、审计报告 |
| 10. API/UX 验收 | 14 | 输入校验、插件降级、审计链导出 |
| **合计** | **155** | |

### 单元 / 集成测试（按文件前 10）

| 文件 | 测试数 |
|------|-------|
| `test_benchmark_generator.py` | 58 |
| `test_multi_fidelity.py` | 57 |
| `test_composer_templates.py` | 55 |
| `test_compliance.py` | 49 |
| `test_engine.py` | 47 |
| `test_dsl_validation.py` | 47 |
| `test_dsl.py` | 47 |
| `test_diagnostics.py` | 68 |
| `test_dsl_bridge.py` | 45 |
| `test_composer.py` | 44 |
| `test_meta_learning.py` | 82 |
| `test_store.py` | 33 |
| `test_ingestion.py` | 35 |
| `test_problem_builder.py` | 32 |
| ... | ... |
| **全部 51 个文件合计** | **1957** |

---

## 用户痛点 → 解法映射

> 以下 10 个痛点来自材料科学 / 自驱动实验室 (SDL) 用户的真实反馈。
> 每项标注**已落地模块**和**可强化方向**。

---

### 痛点 1：UQ 不可靠，AL/BO 会"选错点"

**问题**：材料领域 AL 很依赖不确定性量化 (UQ)，但 UQ 往往没校准，导致 acquisition 选点不稳、甚至越学越偏。

**已落地：**

| 模块 | 能力 |
|------|------|
| `diagnostics/` | `model_uncertainty` 信号（第 9 号）实时监控不确定性水平 |
| `diagnostics/` | `noise_estimate`（第 4 号）追踪 KPI 噪声，间接反映 UQ 质量 |
| `meta_controller/` | 不确定性高时自动切回 Conservative + 高探索（Stagnation 阶段） |
| `counterfactual/` | 离线反事实："如果当时用另一个 UQ/后端会怎样" |
| `benchmark/` + `portfolio/` | 跨后端对比，用证据链替代拍脑袋 |
| `sensitivity/` | 决策对不确定性的敏感性分析 |

**强化方向：**
- [x] `diagnostics/` 新增 `miscalibration_score`（预测区间 vs 实际落点的期望校准误差 ECE）
- [x] `diagnostics/` 新增 `overconfidence_rate`（预测区间过窄的比例）
- [x] MetaController：当 miscalibration > 阈值时，自动降低 exploitation 强度或切换到 UQ-robust 后端
- [x] 在 `explainability/` 中输出"UQ 健康度"段落

---

### 痛点 2：约束 / 不可行区处理弱

**问题**：自驱动实验常见"靠近 Pareto 前沿就更容易 infeasible"，约束与失败处理是核心难点。

**已落地：**

| 模块 | 能力 |
|------|------|
| `feasibility/` | 可行域学习 + 失败概率曲面（`FailureSurface`）+ 危险区域识别 |
| `feasibility/taxonomy` | 结构化失败分类：hardware / chemistry / data / protocol / unknown |
| `feasibility_first/` | 安全边界学习 + KNN 可行性分类 + 自适应 objective − λ·p_fail 评分 |
| `constraints/` | 隐式约束自动发现（从失败模式挖阈值 / 交互） |
| `stabilization/` | 失败处理策略（exclude / penalize / impute）落到 `StabilizeSpec` |
| `meta_controller/` | 失败率高时自动进入 Stagnation，切换到安全后端 |

**强化方向：**
- [x] 约束跨 campaign 迁移：新 campaign 复用历史 campaign 的已发现约束作为先验
- [x] 约束松紧追踪：随实验推进，安全边界是否在收紧 / 放宽（趋势诊断信号）
- [x] 在 Pareto 前沿附近自动提高可行性权重 λ（"越靠近前沿越谨慎"策略）

---

### 痛点 3：非平稳性普遍（仪器漂移、试剂老化、环境变化）

**问题**：SDL 场景非平稳性与漂移需要检测与持续适配，否则固定目标面会退化。

**已落地：**

| 模块 | 能力 |
|------|------|
| `drift/detector` | 4 种漂移检测：KPI 阶跃、平滑斜坡、相关性反转、残差分析 |
| `drift/strategy` | 漂移 → 真实动作映射：回到 Learning、缩短窗口、换 drift-robust 后端、重筛变量 |
| `nonstationary/adapter` | 整合时间加权 + 季节检测 + 漂移的综合评估 |
| `nonstationary/seasonal` | 自相关分析检测周期模式 |
| `nonstationary/weighter` | 时间衰减加权（近期数据权重更高） |
| `data_quality/` | 仪器漂移检测（`InstrumentDrift`） |
| SLO 验收测试 | 检测器 FP < 15%，动作 FP < 5% |

**强化方向：**
- [x] 自适应窗口大小：根据漂移强度自动调整训练窗口（强漂移 → 短窗口，弱漂移 → 长窗口）
- [x] 政权变化 (regime change) 检测：区分"渐进漂移"和"突变跳变"，采取不同策略
- [x] 漂移归因：识别漂移来源是哪个参数维度或环境因素

---

### 痛点 4：噪声与测量质量被忽略

**问题**：自动实验里"测量噪声"会直接影响成本与学习效率，很多 BO 工具完全忽略。

**已落地：**

| 模块 | 能力 |
|------|------|
| `diagnostics/` | `noise_estimate`（变异系数）、`variance_contraction`（收敛标志） |
| `stabilization/` | 异常值剔除（N-sigma）、噪声平滑（移动平均）、重加权 |
| `data_quality/` | 噪声分解（实验噪声 vs 模型噪声 vs 系统漂移）+ 批效应检测 |
| `meta_controller/` | 高噪声 → 自动优先 Random/LHS（鲁棒后端）+ 降低 exploitation 强度 |
| `cost/` | 成本信号影响探索 / 利用平衡 |

**强化方向：**
- [x] 重复测量建议：当 `noise_estimate` > 阈值时，建议对同一配置重复 N 次取均值
- [x] 异方差感知 (heteroscedastic)：不同参数区域噪声不同，对低噪声区域更积极 exploit
- [x] 信噪比 (SNR) 追踪信号：加入 `diagnostics/` 作为第 17 号信号

---

### 痛点 5：多目标与偏好难用

**问题**：材料设计常是多目标权衡（性能 / 稳定性 / 成本 / 可制造性），MOBO 实践复杂。

**已落地：**

| 模块 | 能力 |
|------|------|
| `multi_objective/` | Pareto 前沿检测、非支配排序、支配排名、权衡分析（冲突 / 协同 / 独立） |
| `preference/` | Bradley-Terry MM 偏好学习（从"A 比 B 好"学效用，不需精确权重） |
| 跨领域验证 | 配方（双目标 + 约束）、农业（双目标 + 漂移）、能源（三目标 + 约束）均已验证 |

**强化方向：**
- [x] Target band 约束：用户指定"目标 A 在 [lo, hi] 范围内即可，尽量优化目标 B"
- [x] 交互式 Pareto 导航：展示当前 Pareto 前沿，用户点选一个区域 → 系统聚焦该区域探索
- [x] Aspiration-level 细化：随实验推进，自动收紧/放宽各目标的期望水平

---

### 痛点 6：并行 batch 点聚集

**问题**：HTE / 机器人实验天然是 batch/并行；很多 acquisition 在 batch 下容易点聚集。

**已落地：**

| 模块 | 能力 |
|------|------|
| `batch/` | 3 种去相关策略：maximin、coverage、hybrid |
| `batch/` | `BatchPolicy` 确保批内参数配置的多样性 |
| `feasibility_first/` | 打分时结合可行域 / 失败面，避免选到危险区域 |
| 消融验收测试 | 验证 diversified batch 比 naive batch 有更高 diversity 和 coverage |

**强化方向：**
- [x] 批失败感知重规划：batch 内部分点失败后，自动补充替代点（不等下一轮）
- [x] 异步 batch：不等所有点跑完就开始规划下一批（流式更新）
- [x] batch size 自适应：根据当前阶段自动调整（Cold Start → 大 batch 覆盖，Exploitation → 小 batch 精调）

---

### 痛点 7：评价体系与可复现性差

**问题**：AL 工作流很多隐含设计假设和评估口径不统一，"看起来有效、换数据就不行"。

**已落地：**

| 模块 | 能力 |
|------|------|
| `core/hashing` | 确定性哈希：同 snapshot + 同 seed → 同决策 → 同哈希 |
| `schema/` | 可版本化决策规则（`DecisionRule` + `RuleSignature`） |
| `replay/` | 确定性重放：VERIFY / COMPARE / WHAT_IF 三种模式 |
| `compliance/` | 哈希链审计 + 防篡改 + 合规报告一键导出 |
| `benchmark/` | 标准化 multi-landscape × multi-seed 对比 + Leaderboard |
| `counterfactual/` | "当时用另一个后端会怎样"的离线证据链 |
| 发布门测试 | 确定性 100%、零安全违规、回归 < 2%、AUC ≥ 60% |

**强化方向：**
- [x] 跨实验室复现对比：不同 lab 的同类 campaign 对齐后横向比较
- [x] 决策规则版本 diff：两个版本的 `DecisionRule` 之间的行为差异自动分析
- [x] 复现性评分：对每个 campaign 给出"换种子 / 换后端 / 换窗口后结论变化程度"的量化指标

---

### 痛点 8：数据形态太杂（曲线 / 谱图 / 图像）

**问题**：材料实验输出常是 EIS / 谱图 / 曲线，不是单标量；如果 KPI 抽取不一致，优化会被输入污染。

**已落地：**

| 模块 | 能力 |
|------|------|
| `feature_extraction/` | `FeatureExtractor` ABC 基类，curve → scalar 的版本化提取器 |
| `feature_extraction/` | `CurveData` 容器 + `ExtractedFeatures` 结果，可审计可回放 |
| `compliance/` | `extractor_version` 可纳入审计链，避免"同名 KPI 不同含义" |

**强化方向：**
- [x] 内置提取器库：EIS 阻抗谱（Nyquist 特征）、UV-Vis 谱（峰位 / 峰高 / 半高宽）、XRD 晶型匹配
- [x] 提取器注册表 + 版本锁定：类似 `PluginRegistry`，对提取器做 allowlist / denylist + 自动版本追踪
- [x] 提取器一致性检查：相同原始数据 + 相同提取器版本 → 相同 KPI（纳入确定性保证）
- [x] 图像 / 谱图的 embedding 降维：对非标量数据先做 latent 压缩再进入优化循环

---

### 痛点 9：数据入口门槛高，手动构造 snapshot 繁琐

**问题**：使用优化系统前需要手动构造 `CampaignSnapshot`、定义 `ParameterSpec`、`ObjectiveDef` 等，对非编程用户门槛高。

**已落地：**

| 模块 | 能力 |
|------|------|
| `store/` | 统一实验存储：行级追加、campaign 查询、一键导出 `CampaignSnapshot` |
| `ingestion/` | CSV/JSON 自动导入 + 列类型推断 + 角色猜测（参数/KPI/元数据） |
| `problem_builder/` | Fluent API 链式建模 + 引导式对话建模，输出标准 `OptimizationSpec` |

---

### 痛点 10：每次新实验都从零开始，不积累经验

**问题**：每个新 campaign 都用固定静态规则选策略，完全不利用历史项目经验；即使同类问题已跑过很多次，也无法自动调优策略。

**已落地：**

| 模块 | 能力 |
|------|------|
| `meta_learning/experience_store` | 跨 campaign 经验持久化存储 + 指纹相似度检索 |
| `meta_learning/strategy_learner` | 指纹→后端亲和力学习，精确匹配 + 相似匹配 fallback |
| `meta_learning/weight_tuner` | EMA 自适应评分权重（高失败→提 fail 权重，高漂移→提 drift 权重） |
| `meta_learning/threshold_learner` | 从好结果的阶段转移时机学习最优切换阈值 |
| `meta_learning/failure_learner` | 失败类型→最佳稳定化策略的跨项目映射 |
| `meta_learning/drift_learner` | 各后端漂移鲁棒性追踪与排名 |
| `meta_learning/advisor` | 顶层编排：5 子学习器 → 统一 `MetaAdvice` → 注入 `MetaController` |

**冷启动安全：** 所有子学习器在经验不足时返回 `None`，调用者自动回退到静态规则，零风险。

---

### 痛点覆盖总结

| # | 痛点 | 落地深度 | 核心模块 | 强化项 |
|---|------|---------|---------|-------|
| 1 | UQ 不可靠 | ▓▓▓▓▓ | diagnostics, counterfactual, portfolio | ✅ miscalibration + overconfidence + UQ 健康度 |
| 2 | 约束 / 不可行区 | ▓▓▓▓▓ | feasibility, constraints, feasibility_first | ✅ 跨campaign迁移 + 松紧追踪 + Pareto-proximity λ |
| 3 | 非平稳性 | ▓▓▓▓▓ | drift, nonstationary, data_quality | ✅ 自适应窗口 + regime change + 漂移归因 |
| 4 | 噪声 / 测量质量 | ▓▓▓▓▓ | diagnostics, stabilization, data_quality, cost | ✅ 重复测量建议 + 异方差 + SNR |
| 5 | 多目标 / 偏好 | ▓▓▓▓▓ | multi_objective, preference | ✅ target band + Pareto 导航 + 自动 aspiration |
| 6 | batch 点聚集 | ▓▓▓▓▓ | batch, feasibility_first | ✅ 失败重规划 + 异步batch + 自适应size |
| 7 | 可复现性 | ▓▓▓▓▓ | schema, replay, compliance, benchmark, validation | ✅ 跨Lab对比 + 规则diff + 复现性评分 |
| 8 | 数据形态杂 | ▓▓▓▓▓ | feature_extraction | ✅ EIS/UV-Vis/XRD + 版本锁定 + 一致性 + embedding |
| 9 | 数据入口门槛高 | ▓▓▓▓▓ | store, ingestion, problem_builder | ✅ CSV/JSON自动导入 + 引导建模 + 快照桥接 |
| 10 | 每次从零开始 | ▓▓▓▓▓ | meta_learning (7 子模块) | ✅ 跨项目元学习 + 冷启动安全 + 状态持久化 |

> ▓ = 已落地（全部 25 项原有强化 + 2 项新痛点均已完成 ✅）

---

## 技术特性

| 特性 | 状态 |
|------|------|
| 纯 Python，无重型 ML 依赖 | ✅ |
| 确定性（相同输入 → 相同输出） | ✅ |
| 完整审计轨迹（哈希链防篡改） | ✅ |
| 合规报告（一键导出） | ✅ |
| 确定性重放（VERIFY / COMPARE / WHAT_IF） | ✅ |
| 插件式可扩展（含市场 + 自动淘汰） | ✅ |
| 声明式 DSL（JSON 往返序列化） | ✅ |
| 跨领域泛化（10 领域验证） | ✅ |
| 影子模式（Agent vs Baseline 对比） | ✅ |
| SLO 监控（延迟 p50/p95、漂移 FP、动作 FP） | ✅ |
| 发布门自动化（8 项门控检查） | ✅ |
| 1957 个测试（155 验收 + 1802 单元/集成） | ✅ |
| 类型标注 | ✅ |
| 零外部运行时依赖 | ✅ |
| 数据自动导入（CSV/JSON → 统一存储 → 快照） | ✅ |
| 引导式问题建模（Fluent API + 对话式 Guide） | ✅ |
| 跨项目元学习（指纹→后端亲和、权重/阈值自适应、失败策略、漂移鲁棒） | ✅ |
| 冷启动安全（元学习不足时自动回退静态规则） | ✅ |
| 元学习状态持久化（JSON 序列化/反序列化） | ✅ |

---

## 项目结构

```
optimization_copilot/
├── core/                # 数据模型 + 确定性哈希
├── diagnostics/         # 14 信号诊断引擎
├── profiler/            # 8 维问题指纹分析
├── meta_controller/     # 核心智能：阶段编排 + 策略选择
├── backends/            # 内置优化算法 (Random, LHS, TPE)
├── plugins/             # 插件基类 + 注册表 + 治理策略
├── portfolio/           # 算法组合学习 + 多维评分
├── marketplace/         # 插件市场 + 健康追踪 + 自动淘汰
├── composer/            # 多阶段流水线编排
├── stabilization/       # 数据清洗与预处理
├── data_quality/        # 噪声分解 + 批效应检测
├── feature_extraction/  # 曲线特征提取
├── screening/           # 高维变量筛选
├── surgery/             # 维度削减手术
├── latent/              # PCA 潜在空间降维
├── drift/               # 漂移检测 + 策略适配
├── nonstationary/       # 时间加权 + 季节检测
├── curriculum/          # 渐进式难度管理
├── feasibility/         # 可行域学习 + 失败曲面 + 失败分类学
├── feasibility_first/   # 安全边界 + 可行性分类 + 安全优先评分
├── constraints/         # 隐式约束发现
├── multi_objective/     # Pareto 前沿 + 多目标分析
├── preference/          # 偏好学习 (Bradley-Terry)
├── cost/                # 成本感知分析
├── batch/               # 批次多样化
├── multi_fidelity/      # 多保真度规划 (successive halving)
├── explainability/      # 人类可读的决策解释
├── explanation_graph/   # DAG 形式的解释图
├── reasoning/           # 模板化推理解释
├── sensitivity/         # 参数敏感性分析
├── compliance/          # 审计链 + 合规报告 + 合规引擎
├── schema/              # 可版本化决策规则
├── replay/              # 决策日志 + 确定性重放引擎
├── dsl/                 # 声明式 DSL + 转换桥 + 校验
├── engine/              # 全生命周期编排引擎 + 状态管理
├── benchmark/           # 基准运行器 + 排行榜
├── benchmark_generator/ # 合成基准生成器
├── counterfactual/      # 反事实评估
├── validation/          # 黄金场景 + 回归验证
├── store/               # 统一实验存储（Phase 1）
├── ingestion/           # 数据自动导入代理（Phase 1）
├── problem_builder/     # 引导式问题建模（Phase 1）
├── meta_learning/       # 跨项目元学习（Phase 2）
└── config.py            # 环境配置

tests/                   # 1957 tests, 51 files
├── test_acceptance.py           # 54 验收测试 (Cat 1-2)
├── test_acceptance_benchmarks.py # 101 验收测试 (Cat 3-10)
├── test_benchmark_generator.py  # 58
├── test_multi_fidelity.py       # 57
├── test_meta_learning.py        # 82
├── test_store.py                # 33
├── test_ingestion.py            # 35
├── test_problem_builder.py      # 32
├── ... (43 more files)
└── total: 1957 tests
```
