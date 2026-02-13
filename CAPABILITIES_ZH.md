# Optimization Copilot — 能力总览

> **一句话定位：** 基于实验历史自动选择、切换、调节优化策略的智能优化决策层，所有决策全链路可追溯可解释。支持跨领域泛化、合规审计、确定性回放、自动化数据导入、跨项目元学习，以及丰富的可视化功能。

---

## 架构总览

```
  原始数据 (CSV/JSON)                 OptimizationSpec (DSL)
        |                                      |
        v                                      |
+-----------------------+                      |
| DataIngestionAgent    |                      |
| (自动解析/推断)        |                      |
+----------+------------+                      |
           v                                   |
+-----------------------+   +----------------+ |
| ExperimentStore       |-->| ProblemBuilder |-+
| (统一数据仓库)         |   | (引导式建模)    |
+----------+------------+   +----------------+
           |                       |
           v                       v
                    +-------------------------+
                    |   OptimizationEngine    |
                    | (全生命周期引擎)          |
                    +-----------+-------------+
                                |
         CampaignSnapshot (实验数据)
                |
        +-------+--------+
        v                v
+--------------+  +---------------+  +----------------+
| Diagnostic   |  | Problem       |  | Data Quality   |
| Engine       |  | Profiler      |  | Engine         |
| (17 信号)    |  | (8维指纹)      |  | (噪声/批次)     |
+------+-------+  +------+--------+  +-------+--------+
       |                 |                    |
       v                 v                    v
  +----------------------------------------------------+
  |              Meta-Controller                        |
  |  阶段检测 -> 策略 -> 风险 -> 调度                     |
  |                                                     |
  |  Portfolio <-- Drift <-- NonStationary <-- Cost     |
  |       ^                                             |
  |       |  MetaLearningAdvisor (跨项目)                |
  |       |  策略 + 权重 + 阈值 + 漂移                    |
  +------------+------------+------------+--------------+
               |            |            |
      +--------+    +-------+    +-------+
      v             v            v
+----------+ +----------+ +--------------+
|Stabilizer| |Screener/ | |Feasibility   |
| (清洗)    | |Surgery   | |First + Safety|
+----------+ |(筛选)     | |(安全区域)     |
             +----------+ +--------------+
               |
               v
      +-----------------+     +------------------+
      |DecisionExplainer|     | Compliance Engine |
      |+ ExplanGraph    |     | 审计 + 报告       |
      +-----------------+     +------------------+
               |                       |
               v                       v
         DecisionLog  ----> ReplayEngine (确定性回放)
                |
                v
        ExperienceStore (跨 campaign 记忆)
                |
                v
  +---------------------------------------------------+
  |            平台层                                    |
  |  FastAPI + WebSocket + CLI + React SPA              |
  +---------------------------------------------------+
                |
                v
  +---------------------------------------------------+
  |         可视化 & 分析层                               |
  |  VSUP 颜色映射 | SHAP 图表 | SDL 监控                 |
  |  设计空间 (PCA/t-SNE/iSOM) | Hexbin 覆盖             |
  +---------------------------------------------------+
                |
                v
  +---------------------------------------------------+
  |         Campaign 引擎层                              |
  |  代理模型 → 排序器 → 阶段门 → 交付物                    |
  |  CampaignLoop（闭环迭代）                             |
  +---------------------------------------------------+
                |
                v
  +---------------------------------------------------+
  |         Agent 层（代码执行强制）                       |
  |  ExecutionTrace | DataAnalysisPipeline | Guard     |
  |  TracedScientificAgent | Orchestrator              |
  +---------------------------------------------------+
```

---

## 模块清单 (400+ 模块，分布在 79 个包中)

### I. 核心智能

#### 1. 核心数据模型 (`core/`)

| 数据结构 | 描述 |
|---------|------|
| `CampaignSnapshot` | 优化 campaign 完整快照：参数规范、观测历史、目标定义 |
| `StrategyDecision` | 输出决策：后端选择、探索强度、批次大小、风险姿态、审计轨迹 |
| `ProblemFingerprint` | 8 维问题指纹，用于自动问题分类 |
| `Observation` | 单次实验观测（参数、KPI、失败标记、时间戳） |
| `StabilizeSpec` | 数据预处理策略（降噪、异常值处理、失败处理） |

**确定性哈希：** `snapshot_hash()` / `decision_hash()` / `diagnostics_hash()` — 相同输入 + 相同种子 = 相同输出；每个决策均可追溯。

---

#### 2. 诊断信号引擎 (`diagnostics/`)

从实验历史实时计算 **17 个健康信号**：

| # | 信号 | 含义 |
|---|------|------|
| 1 | `convergence_trend` | 最优值收敛趋势（线性回归斜率） |
| 2 | `improvement_velocity` | 近期改进速率 vs. 历史速率 |
| 3 | `variance_contraction` | KPI 方差是否在收缩（收敛指标） |
| 4 | `noise_estimate` | 近期 KPI 噪声水平（变异系数） |
| 5 | `failure_rate` | 失败实验占比 |
| 6 | `failure_clustering` | 失败是否集中在最近的试验中 |
| 7 | `feasibility_shrinkage` | 可行区域是否在缩小 |
| 8 | `parameter_drift` | 最优参数是否仍在漂移 |
| 9 | `model_uncertainty` | 模型不确定性代理指标 |
| 10 | `exploration_coverage` | 参数空间探索覆盖率 |
| 11 | `kpi_plateau_length` | KPI 停滞轮数 |
| 12 | `best_kpi_value` | 当前最优 KPI 值 |
| 13 | `data_efficiency` | 每次实验的平均改进量 |
| 14 | `constraint_violation_rate` | 约束违反率 |
| 15 | `miscalibration_score` | UQ 校准误差（ECE） |
| 16 | `overconfidence_rate` | 预测区间过窄的比例 |
| 17 | `signal_to_noise_ratio` | KPI 信噪比（|均值|/标准差） |

---

#### 3. 问题指纹分析器 (`profiler/`)

从数据中自动推断 **8 个维度**：

| 维度 | 分类 | 依据 |
|------|------|------|
| 变量类型 | continuous / discrete / categorical / mixed | 参数规范 |
| 目标形式 | single / multi_objective / constrained | 目标数量与约束 |
| 噪声水平 | low / medium / high | KPI 变异系数 |
| 成本分布 | uniform / heterogeneous | 时间戳间隔分析 |
| 失败信息量 | weak / strong | 失败点的参数多样性 |
| 数据规模 | tiny(<10) / small(<50) / moderate(50+) | 观测数量 |
| 时间特征 | static / time_series | 滞后-1 自相关 |
| 可行区域 | wide / narrow / fragmented | 失败率 |

**连续向量编码**：`to_continuous_vector()` 将 8 个枚举维度通过领域感知序数编码映射到 [0,1]（如噪声：low=0.0, medium=0.5, high=1.0），加上归一化的有效维度数 — 实现 RBF 核相似度用于跨 campaign 迁移学习。

---

#### 4. 元控制器 (`meta_controller/`)

**核心智能模块** — 五阶段自动编排：

```
冷启动 --> 学习 --> 利用
              |          |
              v          v
         停滞 <---------+
              |
              v
           终止
```

| 阶段 | 触发条件 | 探索强度 | 风险姿态 | 推荐后端 |
|------|---------|---------|---------|---------|
| **冷启动** | 观测数 < 10 | 0.9（高探索） | 保守 | LHS, Random |
| **学习** | 数据充足，仍在学习 | 0.6（平衡） | 适中 | TPE, RF Surrogate |
| **利用** | 强收敛 + 低不确定性 | 0.2（利用） | 积极 | TPE, CMA-ES |
| **停滞** | 长期 KPI 平台 / 失败激增 | 0.8（重启） | 保守 | Random, LHS |
| **终止** | 协同终止判断 | 0.1 | — | TPE |

**自适应特性：**
- 根据问题指纹自动调整后端优先级（如高噪声 -> 优先 Random/LHS）
- 首选后端不可用时自动回退，记录事件日志
- 探索强度根据覆盖率、噪声、数据规模动态调节
- 接受 `backend_policy`（白名单/黑名单）用于治理约束
- 接受 `drift_report` 和 `cost_signals` 进行自适应调整

---

### II. 算法管理

#### 5. 优化后端池 (`backends/` + `plugins/`)

**插件架构** — 可扩展算法注册表，内置 **10 个后端**：

| 后端 | 用途 | 关键特性 |
|------|------|---------|
| `RandomSampler` | 基线 / 冷启动 | 均匀随机采样 |
| `LatinHypercubeSampler` | 空间填充设计 | 分层采样，覆盖良好 |
| `TPESampler` | 历史信息驱动优化 | 好/坏分割贝叶斯方法 |
| `SobolSampler` | 准随机设计 | 低差异序列（最多 21 维） |
| `GaussianProcessBO` | 贝叶斯优化 | GP 代理 + RBF 核 + EI 采集 |
| `RandomForestBO` | SMAC 风格优化 | 决策树集成；完整变量支持 |
| `CMAESSampler` | 进化策略 | 协方差矩阵自适应；rank-1/rank-mu 更新 |
| `DifferentialEvolution` | 基于种群优化 | 经典 DE/rand/1/bin；持久化种群 |
| `NSGA2Sampler` | 多目标优化 | 非支配排序 + 拥挤距离 |
| `TuRBOSampler` | 信赖域贝叶斯 | 动态信赖域根据成功/失败缩放 |

```python
class AlgorithmPlugin(ABC):
    def name(self) -> str: ...
    def fit(self, observations, parameter_specs) -> None: ...
    def suggest(self, n_suggestions, seed) -> list[dict]: ...
    def capabilities(self) -> dict: ...
```

支持 `BackendPolicy`（白名单 / 黑名单）治理。

---

#### 6. 算法组合学习器 (`portfolio/`)

| 类 | 描述 |
|----|------|
| `AlgorithmPortfolio` | 按指纹记录每个后端的历史表现，提供排名 |
| `BackendScorer` | 多维加权评分：历史 + 指纹匹配 + 不兼容惩罚 + 成本信号 |
| `BackendScore` | 分数分解（各维度贡献），支持可解释的后端选择 |
| `ScoringWeights` | 评分权重配置（history, fingerprint_match, incompatibility, cost） |

---

#### 7. 插件市场 (`marketplace/`)

| 类 | 描述 |
|----|------|
| `Marketplace` | 插件注册表 + 健康追踪 + 自动淘汰 |
| `CullPolicy` | 淘汰策略（根据失败率/性能自动下架不健康插件） |
| `MarketplaceStatus` | 插件状态（active / probation / culled） |

---

#### 8. 流水线编排器 (`composer/`)

| 类 | 描述 |
|----|------|
| `AlgorithmComposer` | 选择并编排多阶段优化流水线 |
| `PipelineStage` | 流水线阶段定义（后端、退出条件、转换规则） |
| `PIPELINE_TEMPLATES` | 内置模板：`exploration_first`、`screening_then_optimize`、`restart_on_stagnation` |

---

### III. 数据处理

#### 9. 数据稳定器 (`stabilization/`)

| 策略 | 描述 |
|------|------|
| 失败处理 | `exclude` / `penalize`（保留但标记）/ `impute`（用最差值填充） |
| 异常值移除 | 基于 N-sigma 规则的自动移除 |
| 重新加权 | `recency`（近期数据权重更高）/ `quality`（高质量数据权重更高） |
| 噪声平滑 | 滑动平均窗口 |

---

#### 10. 数据质量引擎 (`data_quality/`)

| 类 | 描述 |
|----|------|
| `DataQualityReport` | 全面的数据质量报告 |
| `NoiseDecomposition` | 噪声分解：实验噪声 vs. 模型噪声 vs. 系统漂移 |
| `BatchEffect` | 批次效应检测：实验批次间的系统性偏差 |
| `InstrumentDrift` | 仪器漂移检测 |

---

#### 11. 特征提取 (`feature_extraction/`)

| 类 | 描述 |
|----|------|
| `FeatureExtractor` | ABC 基类，从测量曲线中提取命名标量特征 |
| `CurveData` | 曲线数据容器 |
| `ExtractedFeatures` | 特征提取结果 |
| `EISNyquistExtractor` | EIS 阻抗谱（溶液电阻、极化电阻、半圆直径、Warburg 斜率） |
| `UVVisExtractor` | UV-Vis 光谱（峰位置、峰高度、FWHM、总吸光度） |
| `XRDPatternExtractor` | XRD 图谱（主峰角度、峰数量、结晶度指数、背景水平） |
| `VersionLockedRegistry` | 带白名单/黑名单 + 版本锁定的注册表 |
| `CurveEmbedder` | PCA 降维：曲线 -> 低维隐空间向量 |
| `check_extractor_consistency()` | 一致性验证：相同数据 + 相同版本 -> 相同 KPI |

---

#### 12. 变量筛选器 (`screening/`)

- **重要性排序**：基于参数-KPI 相关性的重要性评分
- **方向提示**：每个参数的正/负影响方向
- **交互检测**：乘积项相关性用于参数交互检测
- **步长建议**：基于重要性的自动搜索步长推荐

---

#### 13. 参数手术刀 (`surgery/`)

| 类 | 描述 |
|----|------|
| `Surgeon` | 根据筛选结果诊断并执行降维操作 |

---

#### 14. 隐空间降维 (`latent/`)

| 类 | 描述 |
|----|------|
| `LatentTransform` | 纯标准库 PCA（通过幂迭代实现，不依赖 numpy） |
| `LatentOptimizer` | 控制何时/如何应用降维（高维问题自动触发） |

---

### IV. 自适应智能

#### 15. 漂移检测器 (`drift/`)

| 类 | 描述 |
|----|------|
| `DriftDetector` | 多策略概念漂移检测（KPI 阶跃变化、参数-KPI 相关性偏移、残差分析） |
| `DriftReport` | 漂移报告（`drift_detected` 标记 + 各维度检测结果） |
| `DriftStrategyAdapter` | 将漂移信号映射为策略调整（阶段重置、探索增强等） |
| `DriftAction` | 具体的漂移响应动作 |

**两层误报控制：**
- **检测器 FP < 15%**：在平稳数据上很少误报
- **动作 FP < 5%**：漂移误报不会导致实际策略变更

---

#### 16. 非平稳自适应 (`nonstationary/`)

| 类 | 描述 |
|----|------|
| `NonStationaryAdapter` | 集成时间加权 + 季节性检测 + 漂移信号 |
| `SeasonalDetector` | 基于自相关的周期模式检测 |
| `TimeWeighter` | 时间衰减观测加权 |

---

#### 17. 课程学习引擎 (`curriculum/`)

| 类 | 描述 |
|----|------|
| `CurriculumEngine` | 渐进式难度管理，参数排序 + 搜索范围逐步扩展 |

---

### V. 安全性 & 可行性

#### 18. 可行性学习器 (`feasibility/`)

| 类 | 描述 |
|----|------|
| `FeasibilityLearner` | 从失败数据中学习安全区域、安全边界和危险区域 |
| `FailureSurface` | 失败概率曲面：估计安全边界和危险区域 |
| `FailureClassifier` | 结构化失败分类（硬件 / 化学 / 数据 / 协议 / 未知） |
| `FailureTaxonomy` | 失败分类结果 + 统计分布 |

---

#### 19. 安全优先评分 (`feasibility_first/`)

| 类 | 描述 |
|----|------|
| `SafetyBoundaryLearner` | 通过分位数估计学习保守安全边界 |
| `FeasibilityClassifier` | KNN 分类器预测候选点可行性 + 置信度分数 |
| `FeasibilityFirstScorer` | 可行性与目标分数的自适应混合（根据失败率动态加权） |

---

#### 20. 约束发现 (`constraints/`)

| 类 | 描述 |
|----|------|
| `ConstraintDiscoverer` | 从优化历史中发现隐式约束（阈值检测 + 交互检测） |
| `DiscoveredConstraint` | 已发现约束的描述 |
| `ConstraintReport` | 约束发现报告 |

---

### VI. 多目标 & 偏好

#### 21. 多目标优化 (`multi_objective/`)

- **Pareto 前沿检测**：非支配排序找到当前最优集
- **支配排序**：逐层排序（Rank 1 = Pareto 前沿）
- **权衡分析**：目标对之间的相关性分析（冲突 / 协同 / 独立）
- **加权评分**：用户自定义权重标量化

---

#### 22. 偏好学习 (`preference/`)

| 类 | 描述 |
|----|------|
| `PreferenceLearner` | 从成对比较中学习效用分数（Bradley-Terry MM 算法） |

---

### VII. 效率 & 成本

#### 23. 成本感知分析 (`cost/`)

| 类 | 描述 |
|----|------|
| `CostAnalyzer` | 成本感知优化分析：预算压力、效率指标、探索调整 |
| `CostSignals` | 成本信号（支出、效率、剩余预算），输入到 MetaController |

---

#### 24. 批次多样化 (`batch/`)

| 类 | 描述 |
|----|------|
| `BatchDiversifier` | 批次多样化策略（maximin / coverage / hybrid） |
| `BatchPolicy` | 确保批次内参数多样性，避免冗余采样 |

---

#### 25. 多保真度规划 (`multi_fidelity/`)

| 类 | 描述 |
|----|------|
| `MultiFidelityPlanner` | 两阶段优化：廉价筛选 + 昂贵精调 |
| `FidelityLevel` | 保真度级别定义 |
| `MultiFidelityPlan` | 执行计划，包含逐次减半 |

---

### VIII. 可解释性

#### 26. 决策解释器 (`explainability/`)

为每个策略决策生成 **人类可读报告**：

- **概要**：当前阶段 + 选定策略 + 阶段变化
- **触发诊断**：哪些信号驱动了此决策
- **阶段转换解释**：如果阶段发生变化，原因是什么
- **风险评估**：当前风险姿态及其理由
- **覆盖状态**：参数空间已探索多少
- **不确定性评估**：对推荐的置信度

**原则：** 只报告算法实际计算的内容 — 不做推测性解释。

---

#### 27. 解释图 (`explanation_graph/`)

| 类 | 描述 |
|----|------|
| `GraphBuilder` | 从诊断信号、决策和失败曲面构建 DAG 形式的解释图 |

---

#### 28. 推理解释 (`reasoning/`)

| 类 | 描述 |
|----|------|
| `RewriteSuggestion` | 模板化的人类可读解释（手术动作、campaign 状态） |
| `FailureCluster` | 失败簇描述 |

---

#### 29. 灵敏度分析 (`sensitivity/`)

| 类 | 描述 |
|----|------|
| `SensitivityAnalyzer` | 参数灵敏度与决策稳定性分析（相关性 + 距离指标） |

---

### IX. 合规 & 治理

#### 30. 合规审计系统 (`compliance/`)

| 类 | 描述 |
|----|------|
| `AuditEntry` | 单条审计记录（从 DecisionLogEntry 转换） |
| `AuditLog` | 哈希链审计日志（每条记录链接到前一条记录的哈希） |
| `verify_chain()` | 验证审计链完整性（检测篡改） |
| `ChainVerification` | 链验证结果（valid / first_broken_index） |
| `ComplianceReport` | 结构化合规报告：campaign 摘要、迭代日志、最终推荐、规则版本 |
| `ComplianceEngine` | 高层合规编排：审计日志 + 链验证 + 报告生成 |

**防篡改保证：** 如果任何记录被修改，`verify_chain()` 会精确定位首个断裂位置。

---

#### 31. 决策规则引擎 (`schema/`)

| 类 | 描述 |
|----|------|
| `DecisionRule` | 可版本化、可审计的决策规则 |
| `RuleSignature` | 规则签名（用于编纂 MetaController 判断逻辑） |

---

#### 32. 决策回放 (`replay/`)

| 类 | 描述 |
|----|------|
| `DecisionLog` | 逐轮审计轨迹：快照、诊断、决策、实验结果 |
| `DecisionLogEntry` | 单条日志条目（14 个字段），支持 JSON 序列化和文件 I/O |
| `ReplayEngine` | 确定性回放引擎，三种模式： |
| | `VERIFY` — 验证历史决策可复现 |
| | `COMPARE` — 对比两种策略的历史选择 |
| | `WHAT_IF` — 假设分析（如果使用不同策略会怎样？） |

---

### X. 实验编排

#### 33. 声明式 DSL (`dsl/`)

| 类 | 描述 |
|----|------|
| `OptimizationSpec` | 声明式优化规范（参数、目标、预算、约束） |
| `ParameterDef` | 参数定义（名称、类型、边界、分类值） |
| `ObjectiveDef` | 目标定义（名称、方向：minimize/maximize） |
| `BudgetDef` | 预算定义（最大迭代次数、最大时间） |
| `SpecBridge` | DSL -> 核心模型转换（ParameterSpec, CampaignSnapshot, ProblemFingerprint） |
| `SpecValidator` | 规范验证 + 人类可读错误消息 |

支持 JSON 序列化 / 反序列化，完整的 `to_dict()` / `from_dict()` 往返转换。

---

#### 34. 优化引擎 (`engine/`)

| 类 | 描述 |
|----|------|
| `OptimizationEngine` | 全生命周期编排：诊断 -> 指纹 -> 元控制 -> 插件调度 -> 结果记录 |
| `EngineConfig` | 引擎配置（最大迭代次数、批次大小、种子） |
| `EngineResult` | 引擎运行结果 |
| `CampaignState` | 可变 campaign 状态管理 + 检查点 / 恢复序列化 |
| `Trial` | 单次试验生命周期管理（pending -> running -> completed / failed） |
| `TrialBatch` | 批量试验操作 |

**输入验证：** 引擎拒绝空参数列表或空目标列表，并给出清晰错误消息。

---

### XI. 基准测试

#### 35. 基准运行器 (`benchmark/`)

| 类 | 描述 |
|----|------|
| `BenchmarkRunner` | 标准化评估：多景观 x 多种子的后端对比 |
| `BenchmarkResult` | 单次评估结果（AUC、best-so-far 序列、步数） |
| `Leaderboard` | 后端排名表，带稳定性验证 |

---

#### 36. 合成基准生成器 (`benchmark_generator/`)

| 类 | 描述 |
|----|------|
| `SyntheticObjective` | 可配置的合成目标函数 |
| `LandscapeType` | 4 种景观：`SPHERE`、`ROSENBROCK`、`ACKLEY`、`RASTRIGIN` |
| `BenchmarkGenerator` | 生成完整的 campaign 场景（含噪声、失败、漂移、约束） |

参数：`n_dimensions`、`noise_sigma`、`failure_rate`、`failure_zones`、`constraints`、`n_objectives`、`drift_rate`、`has_categorical`、`categorical_effect`。

---

#### 37. 反事实评估 (`counterfactual/`)

| 类 | 描述 |
|----|------|
| `CounterfactualEvaluator` | 离线假设分析：如果使用不同后端会怎样？ |
| `CounterfactualResult` | 反事实结果（加速比、KPI 范围） |

---

### XII. 验证

#### 38. 黄金场景 (`validation/`)

5 个黄金场景用于回归测试：

| 场景 | 模拟数据 | 预期行为 |
|------|---------|---------|
| `clean_convergence` | 30 个观测，KPI 单调递增 | -> 利用阶段 |
| `cold_start` | 仅 4 个观测 | -> 冷启动阶段 |
| `failure_heavy` | 50% 失败率，集中在后半段 | -> 停滞阶段 |
| `noisy_plateau` | 恒定 KPI + 微小噪声 | -> 停滞阶段 |
| `mixed_variables` | 连续 + 分类参数 | -> 学习阶段 |

---

### XIII. 数据录入层

#### 39. 统一实验仓库 (`store/`)

| 类 | 描述 |
|----|------|
| `ExperimentStore` | 持久化实验数据仓库：行级追加、campaign 级查询、快照导出 |
| `ExperimentRecord` | 单条实验记录（参数 + KPI + 元数据 + 时间戳） |
| `StoreQuery` | 灵活查询：按 campaign / 参数范围 / 时间窗口 / KPI 范围 |
| `StoreStats` | 存储摘要统计：campaign 列表、参数列、总行数 |

**存储特性：**
- 纯 Python dict 后端，JSON 往返序列化
- 行级追加 + 批量导入（`add_records()` / `bulk_import()`）
- 自动 campaign ID 索引，O(1) campaign 查询
- 快照桥接：`to_campaign_snapshot()` 直接输出 `CampaignSnapshot`

---

#### 40. 数据导入代理 (`ingestion/`)

| 类 | 描述 |
|----|------|
| `DataIngestionAgent` | 自动解析 CSV/JSON + 列角色推断 + 交互确认 |
| `ColumnProfile` | 列画像：类型推断、唯一值数、缺失率、数值范围 |
| `IngestionResult` | 导入结果：成功数 + 跳过数 + 错误列表 |
| `RoleMapping` | 列角色映射：parameter / kpi / metadata / ignore |

**导入能力：**
- CSV / JSON 自动格式检测
- 列类型推断（numeric / categorical / datetime / text）
- 基于启发式的角色猜测（参数 vs. KPI vs. 元数据列）
- 用户确认 -> 自动写入 `ExperimentStore`
- 缺失值 / 类型不匹配错误报告

---

#### 41. 问题建模器 (`problem_builder/`)

| 类 | 描述 |
|----|------|
| `ProblemBuilder` | 流式 API 构建 `OptimizationSpec`：链式 `.add_parameter().set_objective().build()` |
| `ProblemGuide` | 引导式交互建模：逐步提示参数、目标、约束 |
| `BuilderValidation` | 构建时验证：参数名唯一性、边界合法性、目标方向检查 |

**建模模式：**
- **流式模式**：编程式链式调用，适合脚本集成
- **引导模式**：交互式引导，适合新用户
- 两种模式输出标准 `OptimizationSpec`，可直接用于 `OptimizationEngine`

---

### XIV. 跨项目元学习（7 个子模块）

#### 42. 元学习数据模型 (`meta_learning/models.py`)

| 数据结构 | 描述 |
|---------|------|
| `CampaignOutcome` | 已完成 campaign 摘要：指纹、阶段转换、后端表现、失败类型、最优 KPI |
| `BackendPerformance` | 单后端表现记录：收敛迭代、遗憾值、样本效率、失败率、漂移影响 |
| `ExperienceRecord` | 经验记录：`CampaignOutcome` + 指纹键 |
| `MetaLearningConfig` | 元学习配置：冷启动阈值、相似度衰减、EMA 学习率、近期半衰期 |
| `LearnedWeights` | 学习到的评分权重（gain / fail / cost / drift / incompatibility） |
| `LearnedThresholds` | 学习到的阶段切换阈值（冷启动观测数 / 学习平台期 / 利用增益阈值） |
| `FailureStrategy` | 失败类型 -> 最佳稳定策略映射 |
| `DriftRobustness` | 后端漂移鲁棒性分数（韧性分数 + KPI 损失） |
| `MetaAdvice` | 元学习建议输出：推荐后端 + 评分权重 + 切换阈值 + 失败策略 + 抗漂移后端 |

---

#### 43. 跨 Campaign 经验仓库 (`meta_learning/experience_store.py`)

| 类 | 描述 |
|----|------|
| `ExperienceStore` | 跨 campaign 持久化经验仓库：记录结果、按指纹查询、检索相似指纹 |

**核心能力：**
- 按 campaign ID 精确查询 / 按指纹键聚合查询
- **连续指纹相似度**：RBF 核 `k(x,y) = exp(-||x-y||² / 2)` 在领域感知序数编码的连续向量上（9 维：8 个枚举序数 + 归一化维度数）。产生平滑的相似度梯度，替代二值匹配/不匹配
- **近期加权**：`recency_halflife` 控制旧经验衰减
- JSON 序列化 / 反序列化

**基于相似度的回退**：`WeightTuner` 和 `ThresholdLearner` 均支持基于相似度的查询 — 当没有精确指纹匹配时，查找最相似的已学习指纹（阈值 > 0.5）并迁移其学习到的参数。

---

#### 44. 策略学习器 (`meta_learning/strategy_learner.py`)

| 类 | 描述 |
|----|------|
| `StrategyLearner` | 指纹 -> 后端亲和力学习：精确匹配 + 相似匹配回退，KNN 风格排序 |

---

#### 45. 权重调优器 (`meta_learning/weight_tuner.py`)

| 类 | 描述 |
|----|------|
| `WeightTuner` | 基于 EMA 学习最优 `ScoringWeights`：高失败 -> 提高 fail 权重，高漂移 -> 提高 drift 权重 |

---

#### 46. 阈值学习器 (`meta_learning/threshold_learner.py`)

| 类 | 描述 |
|----|------|
| `ThresholdLearner` | 从阶段转换时机学习 `SwitchingThresholds`：EMA 融合高质量转换轮次 |

---

#### 47. 失败策略学习器 (`meta_learning/failure_learner.py`)

| 类 | 描述 |
|----|------|
| `FailureStrategyLearner` | 失败类型 -> 稳定策略映射：按平均结果质量排序 |

---

#### 48. 漂移鲁棒性追踪器 (`meta_learning/drift_learner.py`)

| 类 | 描述 |
|----|------|
| `DriftRobustnessTracker` | 追踪各后端在漂移下的表现：韧性 = 1.0 - KPI 损失 |

---

#### 49. 元学习顾问 (`meta_learning/advisor.py`)

| 类 | 描述 |
|----|------|
| `MetaLearningAdvisor` | 顶层编排器：组合 5 个子学习器，输出统一的 `MetaAdvice` |

**集成方式（纯注入，不修改现有文件）：**

| 现有组件 | 注入点 | MetaAdvice 提供 |
|---------|--------|----------------|
| `MetaController.decide(portfolio=...)` | portfolio 参数 | `recommended_backends` -> `AlgorithmPortfolio` |
| `BackendScorer(weights=...)` | 构造函数参数 | `scoring_weights` -> `ScoringWeights` |
| `MetaController(thresholds=...)` | 构造函数参数 | `switching_thresholds` -> `SwitchingThresholds` |
| `FailureTaxonomy` | MetaController 消费者 | `failure_adjustments` |
| `DriftReport` | MetaController 消费者 | `drift_robust_backends` |

---

### XV. 纯 Python 数学库 (`backends/_math/`)

零依赖数学原语，驱动所有后端：

| 模块 | 函数 | 用途 |
|------|------|------|
| `linalg.py` | `vec_dot`, `mat_mul`, `mat_vec`, `transpose`, `identity`, `mat_add`, `mat_scale`, `outer_product`, `cholesky`, `solve_lower`, `solve_upper`, `solve_cholesky`, `mat_inv`, `determinant`, `eigen_symmetric` | 线性代数（15 个函数） |
| `stats.py` | `norm_pdf`, `norm_cdf`, `norm_ppf`, `norm_logpdf`, `binary_entropy` | 统计分布（5 个函数） |
| `sobol.py` | `sobol_sequence`, `SOBOL_DIRECTION_NUMBERS` | 准随机序列（21 维） |
| `kernels.py` | `rbf_kernel`, `matern52_kernel`, `distance_matrix`, `kernel_matrix` | GP 核函数（4 个函数） |
| `acquisition.py` | `expected_improvement`, `upper_confidence_bound`, `probability_of_improvement`, `log_expected_improvement_per_cost` | 采集函数（4 个函数） |

**总计：28+ 纯 Python 数学函数** — Cholesky 分解、幂迭代特征分解、Sobol 序列、GP 核、采集函数，全部零外部依赖。

---

### XVI. 基础设施层 (`infrastructure/`)

企业级优化编排：

| 模块 | 关键类 | 用途 |
|------|--------|------|
| `auto_sampler.py` | `AutoSampler`, `SelectionResult` | 根据问题指纹自动选择最佳后端 |
| `batch_scheduler.py` | `BatchScheduler`, `AsyncTrial`, `TrialStatus` | 排队 & 调度试验批次；异步执行 |
| `constraint_engine.py` | `ConstraintEngine`, `Constraint`, `ConstraintEvaluation` | 处理不等式/等式约束；可行性 GP |
| `cost_tracker.py` | `CostTracker`, `TrialCost` | 追踪评估成本；成本感知优化 |
| `domain_encoding.py` | `EncodingPipeline`, `OneHotEncoding`, `OrdinalEncoding`, `CustomDescriptorEncoding`, `SpatialEncoding` | 灵活的参数空间变换 |
| `multi_fidelity.py` | `MultiFidelityManager`, `FidelityLevel` | 多保真度评估协调 |
| `parameter_importance.py` | `ParameterImportanceAnalyzer`, `ImportanceResult` | 基于代理模型的重要性归因 |
| `robust_optimizer.py` | `RobustOptimizer` | 噪声/不确定性下的鲁棒设计 |
| `stopping_rule.py` | `StoppingRule`, `StoppingDecision` | 收敛检测 & 提前停止 |
| `transfer_learning.py` | `TransferLearningEngine`, `CampaignData` | 从历史 campaign 迁移知识 |
| `integration.py` | `InfrastructureStack`, `InfrastructureConfig` | 组装所有组件；依赖注入 |

**InfrastructureStack** 集成到 `OptimizationEngine.run()` 主循环，提供约束处理、成本追踪、停止规则和自动采样器选择。

---

### XVII. 平台层

#### 50. REST API (`api/`)

| 文件 | 关键端点 | 描述 |
|------|---------|------|
| `app.py` | `create_app()` | FastAPI 应用工厂（lifespan、CORS、健康检查） |
| `routes/campaigns.py` | POST/GET/DELETE campaigns, start/stop/pause/resume | Campaign CRUD + 生命周期 |
| `routes/campaigns.py` | GET batch, POST trials, GET result, GET checkpoint | 试验提交 & 结果获取 |
| `routes/advice.py` | POST `/advice` | 元学习推荐 |
| `routes/reports.py` | GET `/audit`, `/compliance`, POST `/compare` | 审计轨迹、合规报告、campaign 对比 |
| `routes/store.py` | GET `/query`, `/summary`, `/export` | 历史试验查询 & 导出 |
| `routes/ws.py` | WebSocket `/{campaign_id}`, `/all_events` | 实时事件流 |
| `routes/loop.py` | POST/GET/DELETE loop, iterate, ingest | CampaignLoop 生命周期 |
| `routes/analysis.py` | POST top-k, ranking, outliers, correlation, fanova, symreg, pareto, diagnostics, molecular, causal, intervention, counterfactual, physics-gp, ode, hypothesis-generate, hypothesis-test, hypothesis-status, bootstrap, robustness, sensitivity, cross-model, hybrid-fit, hybrid-predict, discrepancy | DataAnalysisPipeline 端点 |

---

#### 51. 平台服务 (`platform/`)

| 类 | 描述 |
|----|------|
| `AuthManager` | API 密钥管理 & 工作区认证 |
| `CampaignManager` | Campaign CRUD & 状态转换 |
| `CampaignRunner` | 执行 campaign 逻辑；管理试验提交循环 |
| `AsyncEventBus` | 发布-订阅事件广播到 WebSocket 客户端 |
| `Workspace` | 工作区初始化 & 基于文件的持久化 |
| `RAGIndex` | 检索增强生成索引，用于元学习 |

---

#### 52. CLI 应用 (`cli_app/`)

| 命令组 | 命令 | 描述 |
|-------|------|------|
| `campaign` | create, list, status, start, stop, pause, resume, delete | Campaign 生命周期操作 |
| `store` | summary, query, export | 工作区试验仓库访问 |
| `meta-learning` | show, advice | 元学习检查 & 建议 |
| `server` | init, start | API 服务器初始化 & 启动 |

---

#### 53. Web 前端 (`web/`)

React TypeScript SPA，用于 campaign 可视化：

| 组件 | 描述 |
|------|------|
| `Dashboard` | Campaign 概览网格，包含状态卡片 |
| `CampaignDetail` | 单个 campaign 检查（KPI、试验、时间线） |
| `Reports` | 审计日志、合规报告、对比 |
| `Compare` | 并排 campaign 对比 |
| `KpiChart` | KPI 演变图 |
| `TrialTable` | 带排序/过滤的试验列表 |
| `PhaseTimeline` | Campaign 阶段可视化 |
| `AuditTrail` | 决策审计日志展示 |
| `useCampaign` | 自定义 Hook，用于 campaign 数据获取 + 轮询 |
| `useWebSocket` | 自定义 Hook，用于 WebSocket 事件订阅 |
| `LoopView` | 交互式 CampaignLoop 管理（创建、迭代、摄入、查看交付物） |
| `AnalysisView` | 基于标签的数据分析（Top-K、相关性、fANOVA），带追踪结果 |

---

### XVIII. 可视化 & 分析层 (v3)

纯 Python SVG 可视化套件 — 零外部依赖。

#### 54. 可视化基础 (`visualization/`)

| 类 / 协议 | 描述 |
|-----------|------|
| `PlotData` | 通用图表容器：plot_type, data, metadata, svg；支持 `to_dict()` / `from_dict()` |
| `SurrogateModel` | `runtime_checkable` 协议，要求 `predict(x) -> (mean, uncertainty)` |
| `SVGCanvas` | 纯 Python SVG 构建器：rect, circle, line, polyline, polygon, text, path, group, defs |

---

#### 55. VSUP 颜色映射 (`visualization/colormaps.py`)

| 类 | 描述 |
|----|------|
| `VSUPColorMap` | 值抑制不确定性调色板：双变量颜色编码（色调=值，饱和度=不确定性） |
| | 支持 `viridis`、`plasma`、`inferno` 调色板，5 站点插值 |
| | `map(value, uncertainty)` -> (R, G, B, A)；`batch_map()`；`color_to_hex()` |

---

#### 56. 空间填充诊断 (`visualization/diagnostics.py`)

| 函数 | 描述 |
|------|------|
| `plot_space_filling_metrics()` | 仪表板返回 PlotData，包含差异性、覆盖率、最小距离 |
| `_compute_star_discrepancy()` | d<=5 精确计算，d>5 随机近似 |
| `_compute_coverage()` | 基于网格的覆盖百分比 |
| `_compute_min_distance()` | 归一化空间中的最小成对 L2 距离 |

---

#### 57. SHAP 可解释性 (`_analysis/` + `visualization/explainability.py`)

| 类 / 函数 | 描述 |
|-----------|------|
| `KernelSHAPApproximator` | Kernel SHAP 引擎：精确枚举（d<=11）、随机采样（d>=12）、加权回归 |
| `plot_shap_waterfall()` | SHAP 瀑布图（累积特征贡献） |
| `plot_shap_beeswarm()` | SHAP 蜂群图（特征重要性分布） |
| `plot_shap_dependence()` | SHAP 依赖图（特征值 vs. SHAP 值，带交互着色） |
| `plot_shap_force()` | SHAP 力图（从基准值的推/拉） |

---

#### 58. SDL 监控 (`visualization/sdl_monitor.py`)

| 函数 | 描述 |
|------|------|
| `plot_experiment_status_dashboard()` | 实验进度：按状态分组的试验、吞吐量、时间追踪 |
| `plot_safety_monitoring()` | 安全面板：约束违反、硬件告警、严重性时间线 |
| `plot_human_in_the_loop()` | 操作员面板：干预频率、决策覆写、审批率 |
| `plot_continuous_operation_timeline()` | 时间线：硬件利用率、实验阶段、维护窗口 |
| `SDLDashboardData` | 数据容器，包含 autonomy_level、experiments、safety events、operator actions |

---

#### 59. 设计空间探索 (`visualization/design_space.py`)

| 函数 | 描述 |
|------|------|
| `plot_latent_space_exploration()` | PCA + t-SNE 参数空间二维投影 |
| `plot_isom_landscape()` | 自组织映射：竞争学习与高斯邻域 |
| `plot_forward_inverse_design()` | 网格参数空间 -> 代理预测 -> 可行区域过滤 |

PCA 使用 `_math/linalg.py` 的 `eigen_symmetric()`。t-SNE 实现了成对亲和力（二分搜索困惑度）和梯度下降（早期夸大）。

---

#### 60. Hexbin 覆盖 (`visualization/parameter_space.py`)

| 类 / 函数 | 描述 |
|-----------|------|
| `HexCell` | 六角形单元，带轴坐标和统计信息 |
| `plot_hexbin_coverage()` | 六角形分箱覆盖视图，3 种着色模式（density / predicted_mean / uncertainty） |

---

#### 61. LLM 可视化助手 (`visualization/llm_assistant.py`)

| 类 | 描述 |
|----|------|
| `PlotSpec` | 绘图规范（plot_type, filters, parameters, color_by, aggregation, title） |
| `LLMVisualizationAssistant` | LLM 驱动的交互式绘图生成骨架；`validate_spec()` 已实现，`query_to_plot()` 可扩展 |

---

### XIX. Campaign 引擎 (`campaign/`)

闭环优化引擎，包含代理模型、候选点排序和三层交付物输出。

#### 62. 指纹代理模型 (`campaign/surrogate.py`)

| 类 | 描述 |
|----|------|
| `FingerprintSurrogate` | 轻量级 GP 代理模型，使用指纹核。基于观测历史拟合，为候选点预测均值 + 不确定性 |
| `SurrogateResult` | 预测结果：均值、标准差、采集分数 |

#### 63. 候选点排序器 (`campaign/ranker.py`)

| 类 | 描述 |
|----|------|
| `CandidateRanker` | 按采集分数排序候选点（UCB、EI 或 PI）。生成排序后的 `RankedCandidate` 列表 |
| `RankedCandidate` | 单个排序候选点：排名、名称、参数、预测值 |
| `RankedTable` | 带目标元数据的排序候选点表 |

#### 64. 阶段门协议 (`campaign/stage_gate.py`)

| 类 | 描述 |
|----|------|
| `StageGateProtocol` | Campaign 进展决策逻辑：继续、扩展还是停止？基于收敛性、预算和改进信号 |
| `StageGateDecision` | 门决策：动作（continue/expand/stop）+ 理由 |

#### 65. Campaign 交付物 (`campaign/output.py`)

每次 campaign 迭代的三层结构化输出：

| 层 | 类 | 内容 |
|----|----|----|
| 仪表板 | `DashboardLayer` | 排序候选点表 + 批次选择 |
| 智能 | `IntelligenceLayer` | 模型指标 + 学习报告 + Pareto 摘要 |
| 推理 | `ReasoningLayer` | 诊断摘要 + fANOVA 结果 + 执行轨迹 |

| 类 | 描述 |
|----|------|
| `CampaignDeliverable` | 顶层容器：迭代次数、时间戳、仪表板 + 智能 + 推理层 |
| `ModelMetrics` | 每目标模型指标：训练点数、y 统计量、拟合耗时 |
| `LearningReport` | 模型预测与实际值对比：预测误差、MAE、摘要 |

#### 66. Campaign 循环 (`campaign/loop.py`)

| 类 | 描述 |
|----|------|
| `CampaignLoop` | 有状态闭环优化：拟合代理 → 排序候选 → 构建交付物 → 摄入结果 → 重复 |

**循环生命周期：**
```
create(snapshot, candidates) → run_iteration() → ingest_results(new_obs) → run_iteration() → ...
```

每次迭代生成包含三层的 `CampaignDeliverable`。通过 `NGramTanimoto` 支持 SMILES 分子编码。

---

### XX. Agent 层 (`agents/`)

代码执行强制系统，确保所有定量结论都有实际计算支撑。

#### 67. 执行轨迹 (`agents/execution_trace.py`)

| 类 | 描述 |
|----|------|
| `ExecutionTag` | 枚举：`COMPUTED`（代码已运行）、`ESTIMATED`（未运行代码）、`FAILED`（代码出错） |
| `ExecutionTrace` | 执行证明记录：模块、方法、输入/输出摘要、耗时、标签 |
| `TracedResult` | 值 + 轨迹 + 聚合标签。`is_computed` 属性用于快速检查 |
| `trace_call()` | 辅助函数：执行函数，用计时信息包装为 TracedResult |

**原则：** 每个定量结论必须有 `TracedResult`，且 `tag=COMPUTED`。

#### 68. 数据分析流水线 (`agents/data_pipeline.py`)

| 方法 | 包装 | 用途 |
|------|------|------|
| `run_top_k()` | 纯 Python sorted | 按值排序的 Top-K 条目 |
| `run_ranking()` | 纯 Python sorted | 所有条目的完整排序 |
| `run_outlier_detection()` | Z-score 统计 | 统计异常值检测 |
| `run_correlation()` | 纯 Python Pearson | 成对相关性 |
| `run_fanova()` | `InteractionMap` | 通过 fANOVA 的特征重要性 |
| `run_symreg()` | `EquationDiscovery` | 符号回归 |
| `run_insight_report()` | `InsightReportGenerator` | 完整 fANOVA→SymReg→SVG 流水线 |
| `run_confounder_detection()` | `ConfounderDetector` | 混杂因素检测 |
| `run_pareto_analysis()` | `MultiObjectiveAnalyzer` | Pareto 前沿计算 |
| `run_diagnostics()` | `DiagnosticEngine` | Campaign 健康信号 |
| `run_molecular_pipeline()` | `NGramTanimoto` + `GaussianProcessBO` | SMILES→指纹→GP→采集 |
| `run_screening()` | `VariableScreener` | 变量重要性筛选 |
| `run_causal_discovery()` | `CausalStructureLearner` | 因果 DAG 结构学习 |
| `run_intervention()` | `InterventionalEngine` | do 算子因果干预 |
| `run_counterfactual()` | `CounterfactualReasoner` | SCM 反事实推理 |
| `run_physics_constrained_gp()` | `PhysicsConstraintModel` | 物理约束 GP 优化 |
| `run_ode_solve()` | `RK4Solver` | ODE 数值积分 |
| `run_hypothesis_generate()` | `HypothesisGenerator` | 多源假设生成 |
| `run_hypothesis_test()` | `HypothesisTester` | BIC/贝叶斯因子假设检验 |
| `run_hypothesis_status()` | `HypothesisTracker` | 假设生命周期跟踪 |
| `run_bootstrap_ci()` | `BootstrapAnalyzer` | Bootstrap 置信区间 |
| `run_conclusion_robustness()` | `ConclusionRobustnessChecker` | 排名/重要性稳定性 |
| `run_decision_sensitivity()` | `DecisionSensitivityAnalyzer` | 决策扰动分析 |
| `run_cross_model_consistency()` | `CrossModelConsistency` | 跨模型排名一致性 |
| `run_hybrid_fit()` | `HybridModel` | 理论 + 残差 GP 拟合 |
| `run_hybrid_predict()` | `HybridModel` | 混合预测与不确定性 |
| `run_discrepancy_analysis()` | `DiscrepancyAnalyzer` | 理论-数据差异检测 |

每个方法返回带完整执行溯源的 `TracedResult`。

#### 69. 执行守卫 (`agents/execution_guard.py`)

| 类 | 描述 |
|----|------|
| `ExecutionGuard` | 验证反馈载荷中定量声明是否有执行轨迹 |
| `GuardMode` | `STRICT`（拒绝无轨迹声明）或 `LENIENT`（标记为 [ESTIMATED]） |

#### 70. 追踪科学代理 (`agents/traced_agent.py`)

| 类 | 描述 |
|----|------|
| `TracedScientificAgent` | 可选基类，内置流水线和自动轨迹收集 |

#### 71. 科学编排器 (`agents/orchestrator.py`)

| 类 | 描述 |
|----|------|
| `ScientificOrchestrator` | 多代理推理调度器，带审计轨迹 |
| `AuditEntry` | 每代理审计记录，包含执行轨迹 |

#### 72. 安全验证器 (`agents/safety.py`)

| 类 | 描述 |
|----|------|
| `LLMSafetyWrapper` | 7 项安全检查：置信度、物理、幻觉、执行守卫 |

#### 72a. 文献挖掘代理 (`agents/literature/`)

| 类 | 描述 |
|----|------|
| `LiteratureAgent` | 挖掘文献中的先验知识：参数范围、预期 KPI 值、领域约束 |
| `PriorTables` | 来自文献的结构化先验知识表 |

#### 72b. 机理设计代理 (`agents/mechanism/`)

| 类 | 描述 |
|----|------|
| `MechanismAgent` | 为观测到的现象生成机理假设 |
| `MechanismTemplates` | 领域特定机理模板（如 Arrhenius、Michaelis-Menten） |

#### 72c. 相结构代理 (`agents/phase_structure/`)

| 类 | 描述 |
|----|------|
| `PhaseStructureAgent` | 分析相图和结构-性能关系 |
| `ReferenceDB` | 参考相结构数据库 |

#### 72d. 符号回归代理 (`agents/symreg/`)

| 类 | 描述 |
|----|------|
| `SymRegAgent` | 将 EquationDiscovery 包装为代理接口，支持多目标 Pareto 前沿方程发现 |

---

### XXI. 高级分析 (`explain/` + `anomaly/` + `confounder/` + `imputation/`)

#### 73. 交互图 (`explain/interaction_map.py`)

| 类 | 描述 |
|----|------|
| `InteractionMap` | fANOVA 风格的特征重要性分解，使用随机森林方差分析 |

#### 74. 方程发现 (`explain/equation_discovery.py`)

| 类 | 描述 |
|----|------|
| `EquationDiscovery` | 通过遗传编程的符号回归。从数据中发现可解释方程 |

#### 75. 洞察报告生成器 (`explain/report_generator.py`)

| 类 | 描述 |
|----|------|
| `InsightReportGenerator` | 完整流水线：fANOVA → 符号回归 → SVG 可视化 → 人类可读报告 |

#### 76. 异常检测器 (`anomaly/detector.py`)

三层异常检测：

| 层 | 检测方法 |
|----|---------|
| 信号级 | 诊断信号的 Z-score 异常值检测 |
| KPI 级 | 贝叶斯在线变点检测（BOCPD） |
| GP 级 | 基于预测方差的 GP 异常值检测 |

#### 77. 混杂因素治理 (`confounder/`)

| 类 | 描述 |
|----|------|
| `ConfounderDetector` | 通过相关性分析检测混杂变量 |
| `ConfounderCorrector` | 4 种校正策略：COVARIATE、NORMALIZE、FLAG、EXCLUDE |
| `ConfounderAuditTrail` | 混杂因素决策的完整审计轨迹 |

#### 78. 确定性插补 (`imputation/`)

| 类 | 描述 |
|----|------|
| `DeterministicImputer` | 4 种策略：MEAN、MEDIAN、MODE、KNN。带插补日志的完整可追溯性 |

---

### XXII. 分子 & 表示 (`representation/` + `candidate_pool/` + `extractors/`)

#### 79. 表示层 (`representation/`)

| 类 | 描述 |
|----|------|
| `NGramTanimoto` | SMILES → n-gram 指纹 → Tanimoto 核。零依赖分子编码 |
| `RepresentationProvider` | 可替换的分子输入编码接口 |

#### 80. 候选池 (`candidate_pool/`)

| 类 | 描述 |
|----|------|
| `CandidatePool` | 外部分子库管理，带版本控制和去重 |

#### 81. 不确定性感知提取器 (`extractors/`)

| 类 | 描述 |
|----|------|
| `EISExtractor` | EIS 阻抗提取，带不确定性传播 |
| `UVVisExtractor` | UV-Vis 光谱提取，带置信区间 |
| `XRDExtractor` | XRD 图谱提取，带测量不确定性 |
| `DCCyclingExtractor` | DC 循环数据提取 |

---

### XXIII. 工作流 & 保真度 (`workflow/` + `fidelity/`)

#### 82. 工作流引擎 (`workflow/`)

| 类 | 描述 |
|----|------|
| `WorkflowDAG` | 带阶段门的多阶段实验 DAG |
| `FidelityGraph` | 多保真度实验图 |
| `Simulator` | 工作流规划的实验模拟 |

#### 83. 保真度配置 (`fidelity/`)

| 类 | 描述 |
|----|------|
| `FidelityConfig` | 多保真度成本建模和级别配置 |

---

### XXIV. 评估 & 领域 (`benchmark_protocol/` + `case_studies/` + `domain_knowledge/`)

#### 84. 基准协议 (`benchmark_protocol/`)

| 类 | 描述 |
|----|------|
| `SDLBenchmark` | SDL 基准评估，标准化指标 |
| `Leaderboard` | 基准排行榜，带排名和导出 |

#### 85. 案例研究 (`case_studies/`)

用于验证的真实实验基准：

| 案例 | 领域 | 数据 |
|------|------|------|
| 钙钛矿 | 材料 | 晶体结构优化 |
| 锌 | 电化学 | 镀锌优化 |
| 催化 | 化学 | 催化剂性能优化 |

#### 86. 领域知识 (`domain_knowledge/`)

| 类 | 描述 |
|----|------|
| `InstrumentSpecs` | 集中式仪器规范（EIS、UV-Vis、XRD、DC-cycling） |
| `ConstraintLibrary` | 领域特定约束模板 |

---

### XXV. 科学智能层

五个基础科学推理层，将系统从优化工具转变为科学智能平台。

#### 87. 因果发现引擎 (`causal/`)

| 类 | 描述 |
|-----|------|
| `CausalGraph` | DAG 数据结构，含 `CausalNode` 和 `CausalEdge`。边类型：因果/混杂/工具变量。方法：`parents()`、`children()`、`ancestors()`、`descendants()`、`d_separated()`（Bayes-Ball）、`topological_sort()`（Kahn 算法） |
| `CausalStructureLearner` | PC 算法 — 通过精度矩阵的偏相关进行条件独立性检验，Fisher z 变换，v-结构定向，Meek 规则 R1-R3 |
| `InterventionalEngine` | `do()` 算子：图割断 + 后门/前门调整公式。`find_valid_adjustment_set()` 基于后门准则 |
| `CausalEffectEstimator` | 平均处理效应（ATE）、条件 ATE（CATE）、自然直接效应（NDE）通过中介分析 |
| `CounterfactualReasoner` | SCM 三步法：溯因（从事实推断噪声 U）、干预（修改方程）、预测（计算反事实）。`probability_of_necessity()` 和 `probability_of_sufficiency()` |

---

#### 88. 物理先验建模 (`physics/`)

| 类 | 描述 |
|-----|------|
| `PeriodicKernel` | `exp(-2 sin²(π|x-x'|/p) / l²)` 用于周期性现象 |
| `LinearKernel` | 线性核，含方差和偏置参数 |
| `CompositeKernel` | 和/积核组合 |
| `SymmetryKernel` | 对称函数建模核 |
| `ArrheniusPrior` | GP 均值函数：`A * exp(-Ea / RT)`（化学动力学） |
| `MichaelisMentenPrior` | GP 均值函数：`Vmax * S / (Km + S)`（酶动力学） |
| `PowerLawPrior` | GP 均值函数：`a * x^b`（标度律） |
| `RK4Solver` | 四阶 Runge-Kutta ODE 求解器。`solve()` 和 `solve_to_steady_state()`，纯 Python |
| `PhysicsConstraintModel` | 守恒律、单调性、物理边界。`check_feasibility()` 和 `project_to_feasible()`。与 `ConstraintEngine` 集成 |

---

#### 89. 假设引擎 (`hypothesis/`)

| 类 | 描述 |
|-----|------|
| `Hypothesis` | 正式假设对象，生命周期：PROPOSED → TESTING → SUPPORTED / REFUTED / INCONCLUSIVE。可序列化 `to_dict()` / `from_dict()` |
| `Prediction` | 预测值 + 置信区间 + 条件 |
| `Evidence` | 预测与观测结果对比 |
| `HypothesisGenerator` | 多源假设生成：`from_symreg()`（Pareto 方程）、`from_causal_graph()`（因果路径）、`from_fanova()`（重要特征）、`from_correlation()` |
| `HypothesisTester` | `compute_bic()`、`bayes_factor()`、`sequential_update()`、`check_falsification()`（连续 ≥3 次未命中）。包含安全的递归下降表达式解析器 |
| `HypothesisTracker` | 生命周期管理：`add()`、`update_with_observation()`、`suggest_discriminating_experiment()`（找到假设分歧最大处）、`get_status_report()` |

---

#### 90. 决策鲁棒性 (`robustness/`)

| 类 | 描述 |
|-----|------|
| `BootstrapAnalyzer` | 非参数 Bootstrap 置信区间：`bootstrap_ci()`、`bootstrap_top_k()`、`bootstrap_correlation()`、`bootstrap_feature_importance()` |
| `ConclusionRobustnessChecker` | `check_ranking_stability()`（相同 Top-K 的 Bootstrap 比例）、`check_importance_stability()`、`check_pareto_stability()`、`comprehensive_robustness()` |
| `DecisionSensitivityAnalyzer` | `decision_sensitivity()`（扰动数据、重新优化、测量变异）、`recommendation_confidence()`、`value_at_risk()` |
| `CrossModelConsistency` | `kendall_tau()`（O(n²)）、`model_agreement()`（GP/RF/SymReg 两两对比）、`ensemble_confidence()`、`disagreement_regions()` |

---

#### 91. 理论-数据混合 (`hybrid/`)

| 类 | 描述 |
|-----|------|
| `TheoryModel` | 参数化理论模型 ABC + 具体实现：`ArrheniusModel`、`MichaelisMentenModel`、`PowerLawModel`、`ODEModel`（封装 RK4Solver） |
| `ResidualGP` | 残差 GP：r = y - theory(X)。通过 `_math/linalg` 的 Cholesky 分解拟合。提供残差预测的均值和不确定性 |
| `HybridModel` | 组合预测：`theory(x) + GP_residual(x)`。`suggest_next()` 支持 EI/UCB 采集函数、`compare_to_theory_only()`、`theory_adequacy_score()` |
| `DiscrepancyAnalyzer` | `systematic_bias()`、`failure_regions()`（|残差| > 阈值处）、`model_adequacy_test()`（卡方检验）、`suggest_theory_revision()` |

---

### XXVI. 测量不确定性 (`uncertainty/`)

| 类 | 描述 |
|----|------|
| `UncertaintyType` | 枚举：ALEATORIC、EPISTEMIC、SYSTEMATIC、COMBINED |
| `UncertaintyEstimate` | 值 + 标准差 + 置信区间 + 类型 + 来源 |
| `PropagationEngine` | GUM 合规的不确定性传播引擎，通过计算链传播 |

---

### XXVII. 维度分析 (`profiler/dimension_analyzer.py`)

| 类 | 描述 |
|----|------|
| `DimensionAnalyzer` | 分析有效维度：基于相关性的冗余检测、活跃子空间估计、通过特征谱分析的内在维度 |

---

### XXVIII. 偏好协议 (`preference/`)

| 类 | 描述 |
|----|------|
| `PreferenceProtocol` | 结构化成对比较协议，带主动学习：选择信息量最大的配对进行偏好获取 |
| `PreferenceModel` | 偏好模型，含效用分数、一致性指标和传递性检查 |

---

### XXIX. 验证测试（三层真实数据集成）

三层集成测试证明端到端系统功能：

| 层级 | 测试数 | 证明什么 |
|------|--------|---------|
| **Tier 1**：端到端流水线 | 10 | 完整 8 阶段流水线在 50 行 Suzuki 偶联数据上运行（导入 → 存储 → 快照 → 诊断 → 漂移 → 建模 → 推荐 → 审计） |
| **Tier 2**：科学压力测试 | 14 | 漂移检测（-15/-25 偏移）、批次效应（ANOVA F 统计量）、混杂因素标记（操作员/仪器）、失败区域识别、稳定化、模型非平稳性、元控制器自适应 — 全部在 120 行 3 批次数据集上 |
| **Tier 3**：闭环演示 | 19 | 4 轮闭环优化，隐藏 Branin-Hoo 目标函数（真实最优 yield≈92）。引导优化优于随机基线。哈希链审计轨迹。确定性回放。证明系统推动科学进步，而非仅仅解释离线数据 |

---

## 端到端使用方式

### 核心优化流水线

```python
from optimization_copilot.core import CampaignSnapshot
from optimization_copilot.diagnostics import DiagnosticEngine
from optimization_copilot.profiler import ProblemProfiler
from optimization_copilot.meta_controller import MetaController
from optimization_copilot.explainability import DecisionExplainer

# 1. 构建实验快照
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
print(f"后端: {decision.backend_name}")
print(f"探索强度: {decision.exploration_strength}")
print(f"风险姿态: {decision.risk_posture.value}")
print(f"批次大小: {decision.batch_size}")
print(f"报告: {report.summary}")
```

### 声明式 DSL

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
# 引擎自动完成：诊断 -> 指纹 -> 元控制 -> 后端调度 -> 审计
```

### 数据导入 + 引导式建模

```python
from optimization_copilot.store.store import ExperimentStore
from optimization_copilot.ingestion.agent import DataIngestionAgent
from optimization_copilot.problem_builder.builder import ProblemBuilder

# 1. 从 CSV 自动导入
store = ExperimentStore()
agent = DataIngestionAgent(store)
result = agent.ingest_csv("experiments.csv", campaign_id="formulation_v3")
print(f"已导入 {result.n_imported} 条记录，跳过 {result.n_skipped} 条")

# 2. 直接导出为 CampaignSnapshot
snapshot = store.to_campaign_snapshot("formulation_v3")

# 3. 使用流式 API 构建优化规范
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

### 跨项目元学习

```python
from optimization_copilot.meta_learning import (
    MetaLearningAdvisor, CampaignOutcome, BackendPerformance,
)

# 1. 创建元学习顾问
advisor = MetaLearningAdvisor()

# 2. 记录已完成的 campaign 结果
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

# 3. 获取新 campaign 的元学习建议
advice = advisor.advise(new_fingerprint)
print(f"推荐后端: {advice.recommended_backends}")
print(f"置信度: {advice.confidence:.2f}")
print(f"原因: {advice.reason_codes}")

# 4. 注入 MetaController
controller = MetaController(thresholds=advice.switching_thresholds)
scorer = BackendScorer(weights=advice.scoring_weights)

# 5. 持久化（跨会话保存元学习状态）
state_json = advisor.to_json()
advisor = MetaLearningAdvisor.from_json(state_json)
```

### 合规审计

```python
from optimization_copilot.replay.log import DecisionLog
from optimization_copilot.compliance.audit import AuditLog, verify_chain
from optimization_copilot.compliance.report import ComplianceReport

# 加载决策日志
log = DecisionLog.load("campaign_log.json")

# 构建审计链
audit_log = AuditLog.from_decision_log(log)
verification = verify_chain(audit_log)
assert verification.valid, f"审计链在条目 {verification.first_broken_index} 处断裂"

# 生成合规报告
report = ComplianceReport.from_audit_log(audit_log)
print(report.format_text())
```

---

## 跨领域泛化（已验证）

通过问题指纹 + 插件架构，系统已在 **10 个领域 x 10 个种子 = 100 次运行** 中验证：

| # | 领域 | 景观 | 维度 | 噪声 | 特殊配置 |
|---|------|------|------|------|---------|
| 1 | 电化学 | Rosenbrock | 4 | 0.15 | 5% 失败率 |
| 2 | 化学合成 | Sphere | 5 | 0.02 | 混合变量 |
| 3 | 制剂 | Ackley | 8 | 0.05 | 双目标 + 约束 |
| 4 | 生物检测 | Sphere | 2 | 0.40 | 高噪声 |
| 5 | 聚合物 | Rastrigin | 10 | 0.08 | 10% 失败 + 高维 |
| 6 | 药物发现 | Ackley | 7 | 0.10 | 25% 失败 + 混合变量 |
| 7 | 过程工程 | Rosenbrock | 5 | 0.05 | 漂移（drift_rate=0.02） |
| 8 | 材料科学 | Rastrigin | 6 | 0.10 | 15% 失败 + 混合变量 |
| 9 | 农业优化 | Sphere | 4 | 0.20 | 双目标 + 漂移 |
| 10 | 能源系统 | Rosenbrock | 8 | 0.05 | 三目标 + 约束 + 5% 失败 |

**验证结论：**
- 无单一后端退化：没有后端在全局范围内主导 >80%
- 指纹桶多样性：10 个领域产生 >= 4 个不同的指纹桶
- 冷启动安全：所有领域在空组合下都能产生有效决策
- 高失败率优雅降级：25% 失败率下仍有有效决策 + reason_codes

---

## 测试套件

**5,947 个测试**，分布在 **139 个测试文件** 中，全部通过（<30s）：

### 验收测试

| 类别 | 测试数 | 验证内容 |
|------|--------|---------|
| 1. 端到端流水线 | 14 | 完整流水线运行、黄金场景回归 |
| 2. 确定性 & 审计 | 8 | 哈希一致性、审计轨迹完整性 |
| 3. 合成基准 | 11 | 4 景观 x 干净/噪声，AUC，步数 |
| 4. 约束/失败/漂移/多目标 | 12 | 约束发现、失败分类、漂移检测、Pareto |
| 5. 消融/反事实/排行榜 | 27 | 模块贡献、Kendall-tau、排名稳定性 |
| 6. 影子模式 | 11 | Agent vs. Baseline 对比、黑名单合规、安全对等 |
| 7. 监控 & SLO | 11 | p50 < 50ms, p95 < 200ms, 漂移 FP < 15%, 动作 FP < 5% |
| 8. 发布门 v1 | 8 | 确定性 100%、零安全违规、AUC >= 60%、回归 < 2% |
| 9. 跨领域泛化 | 7 | 10 领域多样性、组合回退、审计报告 |
| 10. API/UX 验收 | 14 | 输入验证、插件降级、审计链导出 |

### 真实数据集成测试 (Tier 1-3)

| 类别 | 测试数 | 验证内容 |
|------|--------|---------|
| Tier 1：端到端流水线 | 10 | 完整 8 阶段流水线在合成 Suzuki 偶联数据上（50 行） |
| Tier 2：科学压力测试 | 14 | 120 行 3 批次数据集上的漂移/批次/混杂因素/失败检测 |
| Tier 3：闭环优化 | 19 | 4 轮闭环 campaign，隐藏目标函数，随机基线对比，审计轨迹 |
| 对抗鲁棒性 | 14 | 标签污染、系统偏移、对抗翻转、置信度校准 |
| 连续相似度 | 26 | RBF 核相似度、序数编码、迁移学习回退 |

### 按测试数量排序的主要测试文件

| 文件 | 测试数 |
|------|--------|
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
| ...（另外 58 个文件） | ... |
| **总计：139 个文件** | **5,947** |

---

## 用户痛点 -> 解决方案映射

> 以下 10 个痛点来自材料科学 / 自驱动实验室 (SDL) 用户的真实反馈。
> 每项标注了 **已实现模块** 和 **增强状态**。

### 痛点 1：不可靠的 UQ 导致 AL/BO 选点质量差

| 模块 | 能力 |
|------|------|
| `diagnostics/` | `model_uncertainty` 信号（#9）实时监控不确定性水平 |
| `diagnostics/` | `miscalibration_score`（#15）+ `overconfidence_rate`（#16）追踪 UQ 健康 |
| `meta_controller/` | 不确定性高时自动切换到保守 + 高探索 |
| `counterfactual/` | 离线反事实："如果使用不同的 UQ/后端会怎样？" |
| `sensitivity/` | 决策对不确定性的灵敏度分析 |

### 痛点 2：约束 / 不可行区域处理薄弱

| 模块 | 能力 |
|------|------|
| `feasibility/` | 可行性学习 + 失败概率曲面 + 危险区域识别 |
| `feasibility_first/` | 安全边界学习 + KNN 可行性分类器 + 自适应评分 |
| `constraints/` | 从失败模式自动发现隐式约束 |
| `infrastructure/constraint_engine` | 显式不等式/等式约束处理，含可行性 GP |

### 痛点 3：普遍的非平稳性（仪器漂移、试剂老化）

| 模块 | 能力 |
|------|------|
| `drift/` | 4 策略漂移检测 + 动作映射，两层误报控制 |
| `nonstationary/` | 时间加权 + 季节性检测 + 综合评估 |
| `data_quality/` | 仪器漂移检测（`InstrumentDrift`） |

### 痛点 4：忽视噪声 & 测量质量

| 模块 | 能力 |
|------|------|
| `diagnostics/` | `noise_estimate`、`variance_contraction`、`signal_to_noise_ratio` |
| `stabilization/` | N-sigma 异常值移除、滑动平均平滑、近期/质量重新加权 |
| `data_quality/` | 噪声分解（实验 vs. 模型 vs. 漂移）+ 批次效应检测 |

### 痛点 5：多目标 & 偏好复杂性

| 模块 | 能力 |
|------|------|
| `multi_objective/` | Pareto 前沿检测、支配排序、权衡分析、加权评分 |
| `preference/` | Bradley-Terry MM 从成对比较学习偏好 |
| `backends/NSGA2Sampler` | 原生多目标优化，带拥挤距离 |

### 痛点 6：批次点聚集

| 模块 | 能力 |
|------|------|
| `batch/` | 3 种去相关策略：maximin、coverage、hybrid |
| `infrastructure/batch_scheduler` | 异步试验调度，带批次队列管理 |

### 痛点 7：评估 & 可复现性差

| 模块 | 能力 |
|------|------|
| `core/hashing` | 确定性哈希：相同快照 + 种子 -> 相同决策 |
| `replay/` | 确定性回放：VERIFY / COMPARE / WHAT_IF 模式 |
| `compliance/` | 哈希链审计 + 篡改检测 + 一键合规报告 |
| `benchmark/` | 标准化多景观 x 多种子对比 + 排行榜 |

### 痛点 8：多样数据格式（曲线 / 光谱 / 图像）

| 模块 | 能力 |
|------|------|
| `feature_extraction/` | EIS、UV-Vis、XRD 提取器 + 版本锁定注册表 + 一致性检查 |
| `latent/` | PCA 嵌入，曲线 -> 隐空间向量降维 |

### 痛点 9：数据录入门槛高

| 模块 | 能力 |
|------|------|
| `store/` | 统一实验仓库，行级追加 + 快照桥接 |
| `ingestion/` | CSV/JSON 自动导入 + 列类型推断 + 角色猜测 |
| `problem_builder/` | 流式 API + 交互式引导模式 |

### 痛点 10：每次 campaign 从零开始

| 模块 | 能力 |
|------|------|
| `meta_learning/`（7 个子模块） | 跨项目元学习：策略、权重、阈值、失败策略、漂移鲁棒性 |
| `infrastructure/transfer_learning` | 从历史 campaign 迁移知识 |

### 痛点覆盖汇总

| # | 痛点 | 深度 | 核心模块 | 状态 |
|---|------|------|---------|------|
| 1 | 不可靠 UQ | 完整 | diagnostics, counterfactual, portfolio | 所有增强已完成 |
| 2 | 约束 / 不可行 | 完整 | feasibility, constraints, feasibility_first | 所有增强已完成 |
| 3 | 非平稳性 | 完整 | drift, nonstationary, data_quality | 所有增强已完成 |
| 4 | 噪声 / 测量 | 完整 | diagnostics, stabilization, data_quality, cost | 所有增强已完成 |
| 5 | 多目标 | 完整 | multi_objective, preference, NSGA-II | 所有增强已完成 |
| 6 | 批次聚集 | 完整 | batch, feasibility_first, batch_scheduler | 所有增强已完成 |
| 7 | 可复现性 | 完整 | schema, replay, compliance, benchmark | 所有增强已完成 |
| 8 | 多样数据 | 完整 | feature_extraction, latent | 所有增强已完成 |
| 9 | 数据录入门槛 | 完整 | store, ingestion, problem_builder | 所有增强已完成 |
| 10 | 从零开始 | 完整 | meta_learning（7 个子模块）, transfer_learning | 所有增强已完成 |

---

## 技术特性

| 特性 | 状态 |
|------|------|
| 纯 Python，无重型 ML 依赖 | 是 |
| 确定性（相同输入 -> 相同输出） | 是 |
| 完整审计轨迹（哈希链防篡改） | 是 |
| 合规报告（一键导出） | 是 |
| 确定性回放（VERIFY / COMPARE / WHAT_IF） | 是 |
| 插件可扩展（市场 + 自动淘汰） | 是 |
| 声明式 DSL（JSON 往返序列化） | 是 |
| 跨领域泛化（10 个领域已验证） | 是 |
| 影子模式（Agent vs. Baseline 对比） | 是 |
| SLO 监控（延迟 p50/p95、漂移 FP、动作 FP） | 是 |
| 发布门自动化（8 项门检查） | 是 |
| 5,947 个测试（验收 + 单元/集成 + 三层真实数据） | 是 |
| 全量类型注解 | 是 |
| 零外部运行时依赖 | 是 |
| 自动数据导入（CSV/JSON -> 统一仓库 -> 快照） | 是 |
| 引导式问题建模（流式 API + 引导模式） | 是 |
| 跨项目元学习（7 个子学习器） | 是 |
| 冷启动安全（自动回退到静态规则） | 是 |
| 元学习状态持久化（JSON 序列化） | 是 |
| 10 个内置优化后端 | 是 |
| 10+ 基础设施模块（约束、成本、迁移等） | 是 |
| REST API (FastAPI) + WebSocket 流 | 是 |
| CLI 应用（基于 Click） | 是 |
| React TypeScript SPA 仪表板 | 是 |
| 纯 Python 数学库（28+ 函数） | 是 |
| 基于 SVG 的可视化（12+ 图表类型） | 是 |
| KernelSHAP 可解释性（4 种图表类型） | 是 |
| SDL 监控仪表板（4 个面板） | 是 |
| 设计空间探索（PCA / t-SNE / iSOM） | 是 |
| VSUP 不确定性感知颜色映射 | 是 |
| Campaign 引擎（闭环迭代） | 是 |
| 代码执行强制（追踪结果） | 是 |
| Agent 层（编排器 + 安全 + 守卫） | 是 |
| 异常检测（三层） | 是 |
| 混杂因素治理（4 种策略） | 是 |
| 分子编码（NGramTanimoto） | 是 |
| 符号回归（遗传编程） | 是 |
| fANOVA 交互图 | 是 |
| 27 个 REST API 分析端点 | 是 |
| 因果发现（PC 算法 + do 算子） | 是 |
| 物理先验核和均值函数 | 是 |
| 假设生命周期管理（BIC、贝叶斯因子） | 是 |
| 决策鲁棒性（Bootstrap、跨模型一致性） | 是 |
| 理论-数据混合模型（残差 GP） | 是 |
| 连续指纹相似度（RBF 核） | 是 |
| 基于相似度的元学习迁移 | 是 |
| 4 个科学推理代理（文献、机理、相结构、符号回归） | 是 |
| GUM 合规不确定性传播 | 是 |
| 对抗鲁棒性测试 | 是 |
| 三层真实数据集成测试（Tier 1 流水线、Tier 2 压力、Tier 3 闭环） | 是 |
| 闭环优化优于随机基线（已证明） | 是 |

---

## 项目结构

```
optimization_copilot/
├── _analysis/           # 分析引擎（KernelSHAP）
├── agents/             # Agent 层：执行轨迹、流水线、守卫、编排器
│   ├── literature/     # 文献挖掘代理
│   ├── mechanism/      # 机理设计代理
│   ├── phase_structure/ # 相结构分析代理
│   └── symreg/         # 符号回归代理
├── anomaly/            # 三层异常检测
├── api/                 # FastAPI REST 端点 + WebSocket
├── backends/            # 10 个内置优化算法
│   ├── builtin.py       # Random, LHS, TPE, Sobol, GP-BO, RF-BO, CMA-ES, DE, NSGA-II, TuRBO
│   └── _math/           # 纯 Python 数学库（linalg, stats, sobol, kernels, acquisition）
├── batch/               # 批次多样化
├── benchmark/           # 基准运行器 + 排行榜
├── benchmark_generator/ # 合成基准生成器
├── benchmark_protocol/ # SDL 基准评估
├── campaign/           # Campaign 引擎：代理模型、排序器、阶段门、输出、循环
├── candidate_pool/     # 外部分子库管理
├── case_studies/       # 真实实验基准
├── causal/            # 因果发现引擎（PC 算法、do 算子、反事实）
├── cli_app/             # 基于 Click 的 CLI 应用
├── compliance/          # 审计链 + 合规报告 + 合规引擎
├── composer/            # 多阶段流水线编排
├── confounder/         # 混杂因素检测和校正
├── constraints/         # 隐式约束发现
├── core/                # 数据模型 + 确定性哈希
├── cost/                # 成本感知分析
├── counterfactual/      # 反事实评估
├── curriculum/          # 渐进式难度管理
├── data_quality/        # 噪声分解 + 批次效应检测
├── diagnostics/         # 17 信号诊断引擎
├── domain_knowledge/   # 仪器规范和约束库
├── drift/               # 漂移检测 + 策略自适应
├── dsl/                 # 声明式 DSL + 转换桥接 + 验证
├── engine/              # 全生命周期编排引擎 + 状态管理
├── explain/            # 交互图、方程发现、洞察报告
├── explainability/      # 人类可读决策解释
├── explanation_graph/   # DAG 形式解释图
├── extractors/         # 不确定性感知 KPI 提取器
├── feasibility/         # 可行性学习 + 失败曲面 + 失败分类
├── feasibility_first/   # 安全边界 + 可行性分类器 + 安全优先评分
├── hybrid/            # 理论-数据混合模型（残差 GP、差异检测）
├── hypothesis/        # 假设生命周期（生成、检验、跟踪）
├── feature_extraction/  # 曲线特征提取（EIS, UV-Vis, XRD）
├── fidelity/           # 多保真度成本配置
├── imputation/         # 确定性缺失值插补
├── infrastructure/      # 基础设施栈（10+ 模块）
├── ingestion/           # 数据自动导入代理
├── latent/              # PCA 隐空间降维
├── marketplace/         # 插件市场 + 健康追踪 + 自动淘汰
├── meta_controller/     # 核心智能：阶段编排 + 策略选择
├── meta_learning/       # 跨项目元学习（7 个子模块）
├── multi_fidelity/      # 多保真度规划（逐次减半）
├── multi_objective/     # Pareto 前沿 + 多目标分析
├── nonstationary/       # 时间加权 + 季节性检测
├── physics/           # 物理先验建模（核函数、先验、ODE 求解器）
├── platform/            # 平台服务（认证、campaign 管理、事件、工作区、RAG）
├── plugins/             # 插件基类 + 注册表 + 治理
├── portfolio/           # 算法组合学习 + 多维评分
├── preference/          # 偏好学习（Bradley-Terry）
├── problem_builder/     # 引导式问题建模
├── profiler/            # 8 维问题指纹分析
├── reasoning/           # 模板化推理解释
├── replay/              # 决策日志 + 确定性回放引擎
├── representation/     # 分子编码（NGramTanimoto）
├── robustness/        # 决策鲁棒性（Bootstrap、稳定性、一致性）
├── schema/              # 可版本化决策规则
├── screening/           # 高维变量筛选
├── sensitivity/         # 参数灵敏度分析
├── stabilization/       # 数据清洗和预处理
├── store/               # 统一实验仓库
├── surgery/             # 降维手术
├── uncertainty/        # 测量不确定性类型和传播
├── validation/          # 黄金场景 + 回归验证
├── visualization/       # SVG 可视化（VSUP, SHAP, SDL, 设计空间, hexbin）
├── web/                 # React TypeScript SPA
├── workflow/           # 多阶段实验 DAG
└── config.py            # 环境配置

tests/                   # 5,947 个测试，139 个文件
├── test_tier1_endtoend.py       # 10 个测试：完整 8 阶段流水线
├── test_tier2_stress.py         # 14 个测试：漂移/批次/混杂因素压力
├── test_tier3_closedloop.py     # 19 个测试：闭环优化
├── test_adversarial_robustness.py # 14 个测试：对抗攻击
├── test_continuous_similarity.py # 26 个测试：RBF 核相似度
├── test_acceptance.py           # 验收测试（类别 1-2）
├── test_acceptance_benchmarks.py # 验收测试（类别 3-10）
├── test_integration.py          # 146 个集成测试
├── test_infrastructure.py       # 90 个基础设施测试
├── test_meta_learning.py        # 82 个元学习测试
├── test_feature_extraction.py   # 80 个特征提取测试
├── test_viz_*.py                # 10 个可视化测试文件
├── test_shap_values.py          # KernelSHAP 测试
├── test_eigen_linalg.py         # 特征分解测试
├── ...（另外 63 个文件）
└── 总计：5,947 个测试
```
