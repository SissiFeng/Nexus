# Optimization Copilot — 功能概览

> **一句话定位：** 一个智能优化决策层，能够根据实验历史自动选择、切换、调整优化策略，并给出可追溯的决策解释。

---

## 架构总览

```
CampaignSnapshot (实验数据)
        │
        ▼
┌─────────────────┐     ┌──────────────────┐
│  Diagnostic      │     │  Problem          │
│  Engine (14信号)  │     │  Profiler (8维指纹) │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
    ┌────────────────────────────────┐
    │       Meta-Controller          │
    │  (阶段检测 → 策略选择 → 风险评估)  │
    └──────────────┬─────────────────┘
                   │
         ┌─────────┼──────────┐
         ▼         ▼          ▼
   ┌──────────┐ ┌────────┐ ┌──────────┐
   │Stabilizer│ │Screener│ │Feasibility│
   │ 数据清洗   │ │变量筛选  │ │ 可行域学习  │
   └──────────┘ └────────┘ └──────────┘
                   │
                   ▼
         ┌─────────────────┐
         │ DecisionExplainer│
         │  人类可读的决策报告  │
         └─────────────────┘
```

---

## 12 个核心模块

### 1. 核心数据模型 (`core/`)

| 数据结构 | 说明 |
|---------|------|
| `CampaignSnapshot` | 优化campaign的完整快照：参数规格、观测历史、目标定义 |
| `StrategyDecision` | 输出决策：后端选择、探索强度、批次大小、风险态度、审计轨迹 |
| `ProblemFingerprint` | 8维问题指纹，自动分类问题类型 |
| `Observation` | 单次实验观测（参数、KPI、失败标记、时间戳） |
| `StabilizeSpec` | 数据预处理策略（降噪、异常值、失败处理） |

**确定性哈希：** 所有输入输出均可通过 SHA256 哈希追踪，相同输入 + 相同 seed = 完全相同的输出。

---

### 2. 诊断信号引擎 (`diagnostics/`)

从实验历史中实时计算 **14 个健康信号**：

| # | 信号 | 含义 |
|---|------|------|
| 1 | `convergence_trend` | 最优值的收敛趋势 (线性回归斜率) |
| 2 | `improvement_velocity` | 近期改善速度 vs 历史速度 |
| 3 | `variance_contraction` | KPI方差是否在收缩（收敛标志） |
| 4 | `noise_estimate` | 近期KPI的噪声水平（变异系数） |
| 5 | `failure_rate` | 失败实验比例 |
| 6 | `failure_clustering` | 失败是否集中在近期 |
| 7 | `feasibility_shrinkage` | 可行域是否在缩小 |
| 8 | `parameter_drift` | 最优参数是否还在漂移 |
| 9 | `model_uncertainty` | 模型不确定性的代理指标 |
| 10 | `exploration_coverage` | 参数空间的探索覆盖率 |
| 11 | `kpi_plateau_length` | KPI停滞了多少轮 |
| 12 | `best_kpi_value` | 当前最佳KPI值 |
| 13 | `data_efficiency` | 每次实验带来的平均改善 |
| 14 | `constraint_violation_rate` | 约束违反率 |

---

### 3. 问题指纹分析器 (`profiler/`)

自动从数据中推断问题的 **8 个维度**：

| 维度 | 分类值 | 分类依据 |
|------|--------|---------|
| 变量类型 | continuous / discrete / categorical / mixed | 参数规格 |
| 目标形式 | single / multi_objective / constrained | 目标数和约束 |
| 噪声水平 | low / medium / high | KPI变异系数 |
| 成本分布 | uniform / heterogeneous | 时间戳间隔分析 |
| 失败信息量 | weak / strong | 失败点的参数多样性 |
| 数据规模 | tiny(<10) / small(<50) / moderate(50+) | 观测数量 |
| 时序特征 | static / time_series | 滞后1自相关 |
| 可行域 | wide / narrow / fragmented | 失败率 |

---

### 4. 元控制器 (`meta_controller/`)

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

**每个阶段的行为：**

| 阶段 | 触发条件 | 探索强度 | 风险态度 | 推荐后端 |
|------|---------|---------|---------|---------|
| **Cold Start** | 观测数 < 10 | 0.9 (高探索) | Conservative | LHS, Random |
| **Learning** | 数据充足，尚在学习 | 0.6 (均衡) | Moderate | TPE, RF Surrogate |
| **Exploitation** | 收敛趋势强 + 不确定性低 | 0.2 (强利用) | Aggressive | TPE, CMA-ES |
| **Stagnation** | KPI长期停滞 / 失败激增 | 0.8 (重启探索) | Conservative | Random, LHS |
| **Termination** | 协作终止判断 | 0.1 | — | TPE |

**自动适应特性：**
- 根据问题指纹自动调整后端优先级（如高噪声→优先Random/LHS）
- 首选后端不可用时自动fallback，并记录事件
- 探索强度根据覆盖率、噪声、数据规模动态微调

---

### 5. 优化后端池 (`backends/` + `plugins/`)

**插件架构** — 可扩展的算法注册表：

内置3个算法后端：

| 后端 | 用途 | 特点 |
|------|------|------|
| `RandomSampler` | 基线/Cold Start | 均匀随机采样 |
| `LatinHypercubeSampler` | 空间填充设计 | 分层采样，覆盖性好 |
| `TPESampler` | 有历史数据的优化 | 基于好/坏分组的贝叶斯方法 |

**插件接口：**
```python
class AlgorithmPlugin(ABC):
    def name(self) -> str: ...
    def fit(self, observations, parameter_specs) -> None: ...
    def suggest(self, n_suggestions, seed) -> list[dict]: ...
    def capabilities(self) -> dict: ...
```

注册自定义算法只需：
```python
registry = PluginRegistry()
registry.register(MyCustomOptimizer)
```

支持 allowlist/denylist 的 `BackendPolicy` 治理策略。

---

### 6. 数据稳定器 (`stabilization/`)

实验数据的自动清洗和预处理：

| 策略 | 说明 |
|------|------|
| **失败处理** | `exclude`(排除) / `penalize`(保留标记) / `impute`(用最差值填充) |
| **异常值剔除** | 基于 N-sigma 规则自动剔除 |
| **重加权** | `recency`(近期权重更高) / `quality`(高质量数据权重更高) |
| **噪声平滑** | 移动平均窗口 |

输出 `StabilizedData`，包含处理后的观测、被剔除的索引、平滑后的KPI序列。

---

### 7. 变量筛选器 (`screening/`)

当参数维度很高时，自动识别关键变量：

- **重要性排序**：基于参数-KPI相关性的重要性评分
- **方向性提示**：每个参数对KPI是正向还是负向影响
- **交互检测**：通过乘积项相关性检测参数间的交互作用
- **步长推荐**：根据重要性自动推荐搜索步长（重要参数→更精细）

---

### 8. 可行域学习器 (`feasibility/`)

从失败数据中学习安全区域：

- **安全边界**：基于成功实验推断每个参数的安全范围
- **危险区域识别**：检测失败聚集的参数子空间
- **失败密度图**：每个参数维度上失败点的分布
- **可行性检查**：给定参数组合，判断是否可能可行

```python
learner = FeasibilityLearner()
fmap = learner.learn(snapshot)
is_safe = learner.is_feasible({"temp": 85, "pressure": 3.5}, fmap)
```

---

### 9. 多目标优化分析 (`multi_objective/`)

支持多个优化目标的同时优化：

- **Pareto前沿检测**：非支配排序，找出当前最优解集
- **支配排名**：所有观测的逐层排名（Rank 1 = Pareto前沿）
- **权衡分析**：目标对之间的相关性分析（冲突/协同/独立）
- **加权评分**：用户自定义权重的标量化评分

---

### 10. 决策解释器 (`explainability/`)

为每个策略决策生成**人类可读的报告**：

```python
explainer = DecisionExplainer()
report = explainer.explain(decision, fingerprint, diagnostics)
```

报告包含：
- **总结**：当前阶段 + 选择的策略 + 阶段变化
- **触发诊断**：哪些信号驱动了这个决策
- **阶段转换说明**：如果阶段变了，解释为什么
- **风险评估**：当前的风险态度和原因
- **覆盖状态**：参数空间探索了多少
- **不确定性评估**：对推荐的置信度

**原则：** 只报告算法实际计算的内容，不生成猜测性解释。

---

### 11. 验证系统 (`validation/`)

**5 个黄金场景**用于回归测试：

| 场景 | 模拟数据特征 | 期望行为 |
|------|------------|---------|
| `clean_convergence` | 30次观测，单调递增KPI | → Exploitation阶段，Aggressive风险 |
| `cold_start` | 仅4次观测 | → Cold Start阶段，Conservative风险 |
| `failure_heavy` | 50%失败率，集中在后半段 | → Stagnation阶段，Conservative风险 |
| `noisy_plateau` | 恒定KPI + 微小噪声 | → Stagnation阶段（长期停滞） |
| `mixed_variables` | 连续 + 分类参数 | → Learning阶段 |

`ValidationRunner` 可一键运行全部场景并验证期望。

---

### 12. 确定性与审计 (`core/hashing.py`)

| 功能 | 说明 |
|------|------|
| `snapshot_hash()` | CampaignSnapshot 的 SHA256 指纹 |
| `decision_hash()` | StrategyDecision 的 SHA256 指纹 |
| `diagnostics_hash()` | 诊断向量的 SHA256 指纹 |

**保证：** 相同输入 + 相同 seed → 相同决策 → 相同哈希。每次决策都可追溯。

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

---

## 跨领域适用

通过问题指纹 + 插件架构，系统天然适用于：

- 电化学优化
- 合成路线优化
- 材料配方优化
- 聚合物加工
- 生物实验优化
- 多步骤实验工作流

无需修改核心代码，只需传入对应领域的 `CampaignSnapshot`。

---

## 技术特性

| 特性 | 状态 |
|------|------|
| 纯 Python，无重型 ML 依赖 | ✅ |
| 确定性（相同输入→相同输出） | ✅ |
| 完整审计轨迹 | ✅ |
| 插件式可扩展 | ✅ |
| 166 个测试（143 单元 + 23 集成） | ✅ |
| 5 个黄金回归场景 | ✅ |
| 类型标注 | ✅ |
| 零外部运行时依赖 | ✅ |

---

## 项目结构

```
optimization_copilot/
├── core/              # 数据模型 + 确定性哈希
├── diagnostics/       # 14信号诊断引擎
├── profiler/          # 8维问题指纹分析
├── meta_controller/   # 核心智能：阶段编排 + 策略选择
├── backends/          # 内置优化算法 (Random, LHS, TPE)
├── plugins/           # 插件基类 + 注册表 + 治理策略
├── stabilization/     # 数据清洗与预处理
├── screening/         # 高维变量筛选
├── feasibility/       # 可行域学习与安全检查
├── multi_objective/   # Pareto前沿 + 多目标分析
├── explainability/    # 人类可读的决策解释
└── validation/        # 黄金场景 + 回归验证

tests/
├── test_core.py           # 11 tests
├── test_diagnostics.py    # 38 tests
├── test_profiler.py       # 35 tests
├── test_plugins.py        # 27 tests
├── test_meta_controller.py # 14 tests
└── test_integration.py    # 23 tests (端到端 + 黄金场景)
                           # ─────────
                           # 166 tests total
```
