# Optimization Copilot v5 — 真实实验数据 Case Study

> **核心问题：** 10 个 benchmark 函数（Sphere, Rosenbrock, Ackley, Branin...）证明了系统工程没 bug，但无法证明在真实实验数据上有竞争力。发论文/推广需要 real-world validation。
>
> **目标：** 用你手上的锌电沉积数据 + 2-3 个公开数据集，构建可复现的 case study，展示平台在真实科研场景下的表现。
>
> **价值：** 从 "it works on toy problems" → "it works on YOUR data"。这是审稿人和用户最关心的跨越。
>
> **v5 在整体架构中的位置：** v5 既是 v1-v4 的验证层，也是 Layer 3 Agent 的首要验证数据集。锌电沉积数据将贯穿整个架构——从 v4 UQ 验证、v6c 异常检测验证、到 v7b 可解释性验证。v5 的 domain config 切换（electrochemistry → catalysis → perovskite）直接验证"换一份 YAML 即可适配新领域"的核心卖点。

---

## 目录

1. [为什么合成 Benchmark 不够](#1-为什么合成-benchmark-不够)
2. [Case Study 选择策略](#2-case-study-选择策略)
3. [Case Study 1: 锌电沉积添加剂优化（你的数据）](#3-case-study-1-锌电沉积添加剂优化)
4. [Case Study 2: 催化反应条件优化（公开数据）](#4-case-study-2-催化反应条件优化)
5. [Case Study 3: 钙钛矿薄膜组分优化（公开数据）](#5-case-study-3-钙钛矿薄膜组分优化)
6. [Domain Config 切换验证](#6-domain-config-切换验证)
7. [Benchmark 到 Case Study 的技术桥梁](#7-benchmark-到-case-study-的技术桥梁)
8. [评估框架](#8-评估框架)
9. [Agent 验证路线图](#9-agent-验证路线图)
10. [代码量估算 & 文件结构](#10-代码量估算--文件结构)
11. [实现优先级](#11-实现优先级)
12. [发表策略](#12-发表策略)

---

## 1. 为什么合成 Benchmark 不够

### 1.1 合成 vs 真实的鸿沟

| 维度 | 合成 Benchmark | 真实实验 |
|------|---------------|---------|
| **噪声** | 无噪声或加性高斯 | 异质、非高斯、偶发异常值 |
| **评估成本** | 0.001 秒/点 | 1 小时～1 天/点 |
| **约束** | 已知解析表达式 | 未知、隐式、可能失效 |
| **维度** | 连续实数 | 混合（连续+离散+分类） |
| **景观** | 确定性、平滑 | 随时间漂移、batch 效应 |
| **可重复性** | 完美 | 有限（同一条件CE可差 2-5%） |
| **多保真** | 无 | 快速筛选 vs 精确表征 |
| **领域知识** | 不需要 | domain config 决定提取质量 |

### 1.2 审稿人会问什么

1. "你的 constraint learning 在 Branin 上效果好，但真实实验的约束是什么样的？"
2. "异质噪声 GP 比同质 GP 好 0.3%—在合成数据上。真实数据呢？"
3. "你的 stopping rule 在 100 次评估后停止。真实实验做 100 次要花多少钱？"
4. "自动采样器切换策略的效果，在混合参数空间上验证过吗？"
5. "Domain config 驱动的 Extractor 真的能换领域不改代码？证据呢？" ← ★ 新架构带来的新问题

**结论：** 合成 benchmark 是 necessary but not sufficient。Case study 还需要验证 domain-configurable 设计的跨领域泛化能力。

---

## 2. Case Study 选择策略

### 2.1 选择标准

| 标准 | 权重 | 理由 |
|------|------|------|
| **与你的研究直接相关** | ★★★ | 发表时最有说服力 |
| **数据可获取** | ★★★ | 自有数据 > 公开数据 > 需许可 |
| **覆盖平台独有特性** | ★★★ | 约束学习、成本感知、异质噪声 |
| **验证 domain config 切换** | ★★★ | ★ 新增: 三个领域 = 三份 YAML |
| **领域多样性** | ★★ | 电化学 + 催化 + 薄膜 → 通用性 |
| **合理的参数空间维度** | ★★ | 5-15D 最佳（太低没意义，太高不现实） |
| **社区关注度** | ★ | 热门领域更容易被引 |

### 2.2 选定的三个 Case Study

```
┌────────────────────────────────────────────────────────────────┐
│ Case Study 1: 锌电沉积添加剂优化                               │
│ 来源: 你的实验数据                                              │
│ 维度: 7D 混合 (连续浓度)                                       │
│ 目标: CE↑, |Z|↓                                               │
│ Domain config: electrochemistry/                               │
│ 特性: 异质噪声, 约束 (总浓度), 小样本 (<50)                     │
│ 平台特性测试: v4 噪声传播, 约束学习, 停止准则                   │
│ Agent 验证: SpectrumAnalysis, AnomalyDetection, Hypothesis     │
├────────────────────────────────────────────────────────────────┤
│ Case Study 2: 催化反应条件优化                                  │
│ 来源: Perera et al. (2018) / EDBO benchmark                    │
│ 维度: 4-6D 混合 (温度/时间/浓度 + 催化剂类别)                   │
│ 目标: Yield↑                                                   │
│ Domain config: catalysis/                                      │
│ 特性: 分类参数, 高噪声, 已有 EDBO/CIME4R 对标                   │
│ 平台特性测试: DomainEncoding, AutoSampler, 迁移学习              │
│ Agent 验证: domain config 切换 (electrochemistry → catalysis)   │
├────────────────────────────────────────────────────────────────┤
│ Case Study 3: 钙钛矿薄膜组分/工艺优化                          │
│ 来源: Accelerated Discovery datasets (多个公开)                  │
│ 维度: 6-10D 混合 (组分比 + 温度 + 时间)                         │
│ 目标: PCE↑, Stability↑ (多目标)                                 │
│ Domain config: perovskite/                                     │
│ 特性: 混合空间约束 (组分和=100%), 多目标                         │
│ 平台特性测试: MOBO, 约束处理, 成本感知, 参数重要性              │
│ Agent 验证: 多目标场景下的 agent feedback                       │
└────────────────────────────────────────────────────────────────┘
```

### 2.3 三个 Case Study 共同验证的架构特性

```
              Case Study 1       Case Study 2       Case Study 3
              (锌电沉积)          (催化)             (钙钛矿)
              ──────────         ──────────         ──────────
v4 UQ链路:    ✅ 完整验证        ✅ yield UQ         ✅ 多目标 UQ
domain cfg:   electrochem/       catalysis/          perovskite/
              .yaml              .yaml               .yaml
Extractors:   EIS+DC+UVVis       Yield+HPLC          PCE+XRD+PL
              (config驱动)       (config驱动)        (config驱动)
spectrum      ✅ EIS rules       ✅ yield rules       ✅ XRD rules
  rules:
Agent验证:    首要验证集          跨领域泛化          多目标+MOBO
              (v6c, v7b)         (config切换)        (v6a multi-stage)
```

---

## 3. Case Study 1: 锌电沉积添加剂优化

### 3.1 数据描述

```python
# 你的实验设置 (从 0725 脚本重建)
search_space = {
    "Additive_A": {"type": "continuous", "range": [0.0, 1.0]},  # 体积比
    "Additive_B": {"type": "continuous", "range": [0.0, 1.0]},
    "Additive_C": {"type": "continuous", "range": [0.0, 1.0]},
    "Additive_D": {"type": "continuous", "range": [0.0, 1.0]},
    "Additive_E": {"type": "continuous", "range": [0.0, 1.0]},
    "Additive_F": {"type": "continuous", "range": [0.0, 1.0]},
    "Additive_G": {"type": "continuous", "range": [0.0, 1.0]},
    # 约束: Σ Additive_i ≤ 1.0 (其余为 ZnSO4 基底)
}

objectives = {
    "CE": {"direction": "maximize", "unit": "%"},
    # 可选第二目标:
    "|Z|@1kHz": {"direction": "minimize", "unit": "Ω"},
}

constraints = {
    "known_hard": {"sum_additives": "A+B+C+D+E+F+G <= 1.0"},
    "known_soft": {"CE_physical": "CE <= 105%"},
    "unknown": "to be learned"  # 某些组合可能导致实验失败
}

# 实验协议
protocol = {
    "deposition": {"current": -4, "unit": "mA", "duration": 3, "unit_t": "s"},
    "dissolution": {"current": +4, "unit": "mA", "duration": 3, "unit_t": "s"},
    "cycles": 20,
    "eis": {"freq_range": [10000, 1], "unit": "Hz"},
    "total_volume": 3000,  # µL
}
```

### 3.2 数据预处理管道

**文件：** `case_studies/zinc/data_loader.py` (~220 行)

```python
class ZincElectrodepositionDataLoader:
    """从你的实验 CSV + Squidstat 原始数据加载。
    
    数据来源:
    - CSV: 添加剂配方 (由 NIMO 生成) + CE 结果
    - Squidstat: 每个实验的 DC cycling + EIS 原始数据
    
    输出:
    - X: (n_experiments, 7) 添加剂浓度
    - Y: (n_experiments, n_objectives) 目标值
    - noise_vars: (n_experiments, n_objectives) 噪声方差 (v4 链路)
    - measurements: (n_experiments, n_kpis) MeasurementWithUncertainty (含 metadata)
    """
    
    def __init__(self, data_dir: str, domain_name: str = "electrochemistry"):
        self.data_dir = data_dir
        # ★ v4 新接口: DomainConfig 驱动
        self.domain_config = DomainConfig(domain_name)
        self._build_extractors()
    
    def _build_extractors(self):
        """根据 domain config 构建 extractor 列表。
        
        electrochemistry config 包含 squidstat 仪器参数 →
        自动启用 EIS + DC extractors。
        """
        from ..extractors.eis_uncertainty import EISUncertaintyExtractor
        from ..extractors.dc_uncertainty import DCCyclingUncertaintyExtractor
        
        self.eis_extractor = EISUncertaintyExtractor(self.domain_config)
        self.dc_extractor = DCCyclingUncertaintyExtractor(self.domain_config)
    
    def load_from_csv(self, csv_path: str) -> dict:
        """从 NIMO 格式 CSV 加载。
        
        CSV 格式 (你的 0725 脚本输出):
        A, B, C, D, E, F, G, CE, timestamp
        0.15, 0.30, 0.10, 0.05, 0.20, 0.10, 0.10, 95.3, 2024-07-25T...
        """
        ...
    
    def load_raw_electrochemistry(self, experiment_id: str) -> dict:
        """从 Squidstat 原始数据加载。
        
        返回:
        - dc_data: {current: [], voltage: [], time: []}
        - eis_data: [
            {freq: [], z_real: [], z_imag: [], phase: "initial"},
            {freq: [], z_real: [], z_imag: [], phase: "after_depo"},
            ...
          ]
        """
        ...
    
    def compute_kpis_with_uncertainty(
        self, raw_data: dict
    ) -> list['MeasurementWithUncertainty']:
        """调用 v4 Extractors 计算 KPI + 不确定性。
        
        ★ 使用 DomainConfig 驱动的 Extractor (v4 新接口):
        - 仪器精度参数从 electrochemistry/instruments.yaml 读取
        - 物理约束从 electrochemistry/physical_constraints.yaml 读取
        - 等效电路配置从 electrochemistry/eis_models.yaml 读取
        - 过程式规则从 electrochemistry/spectrum_rules.py 加载
        
        输出的 MeasurementWithUncertainty 包含 metadata:
        - raw_features: Nyquist 特征、电荷量等 (供 Agent 消费)
        - quality_flags: 质量标记 (供 AnomalyDetectionAgent 消费)
        - rule_diagnosis: 过程式规则诊断 (供 HypothesisAgent 消费)
        """
        measurements = []
        
        # DC → CE with uncertainty
        dc_results = self.dc_extractor.extract_with_uncertainty({
            "current": raw_data["dc_data"]["current"],
            "voltage": raw_data["dc_data"]["voltage"],
            "time": raw_data["dc_data"]["time"],
            "n_cycles": raw_data.get("n_cycles", 20),
        })
        measurements.extend(dc_results)
        
        # EIS → |Z|@1kHz, R_ct with uncertainty
        last_eis = raw_data["eis_data"][-1]  # after last dissolution
        eis_results = self.eis_extractor.extract_with_uncertainty({
            "z_real": last_eis["z_real"],
            "z_imag": last_eis["z_imag"],
            "frequencies": last_eis["freq"],
        })
        measurements.extend(eis_results)
        
        return measurements
    
    def load_full_dataset(self, csv_path: str) -> dict:
        """加载完整数据集: X + Y + noise_vars + measurements。
        
        用于 OfflineReplayBenchmark 和 Agent 验证。
        """
        csv_data = self.load_from_csv(csv_path)
        
        all_X = []
        all_measurements = []
        
        for exp_id in csv_data["experiment_ids"]:
            raw = self.load_raw_electrochemistry(exp_id)
            measurements = self.compute_kpis_with_uncertainty(raw)
            
            all_X.append(csv_data["params"][exp_id])
            all_measurements.append(measurements)
        
        return {
            "X": all_X,
            "measurements": all_measurements,  # ★ 完整 UQ + metadata
            # 便捷视图:
            "Y": [[m.value for m in ms] for ms in all_measurements],
            "noise_vars": [[m.variance for m in ms] for ms in all_measurements],
            "confidences": [[m.confidence for m in ms] for ms in all_measurements],
        }
```

### 3.3 Replay 优化实验

**核心思想：** 用你已有的实验数据做 offline replay，假装是在线优化。

```python
class OfflineReplayBenchmark:
    """离线回放 benchmark — 用真实数据评估优化策略。
    
    方法论:
    1. 建立真实数据的代理模型 (GP on all data)
    2. 用代理模型 + 噪声生成器作为"虚拟实验"
    3. 比较不同优化策略在该代理上的表现
    
    优点:
    - 无需额外实验成本
    - 可以跑 1000 次 Monte Carlo，统计显著
    - 真实的参数空间、噪声结构、约束
    
    缺点:
    - 代理模型不完美（但比合成函数真实得多）
    - 外推区域不可靠（限制搜索范围）
    """
    
    def __init__(self, X_real, Y_real, noise_vars_real):
        # 用全部真实数据建代理
        self.surrogate = self._fit_surrogate(X_real, Y_real, noise_vars_real)
        self.bounds = self._infer_bounds(X_real)
        self.noise_model = self._fit_noise_model(X_real, noise_vars_real)
    
    def evaluate(self, x: list[float]) -> tuple[float, float]:
        """虚拟实验: 代理预测 + 噪声采样。
        
        y = surrogate.predict(x) + ε
        ε ~ N(0, noise_model(x))
        """
        mu, pred_var = self.surrogate.predict(x)
        noise_var = self.noise_model.predict_noise(x)
        
        import random
        y = random.gauss(mu, (pred_var + noise_var) ** 0.5)
        
        return y, noise_var
    
    def run_comparison(
        self,
        strategies: list['OptimizationBackend'],
        n_trials: int = 30,
        n_repeats: int = 50,
        budget: int | None = None
    ) -> dict:
        """比较多个优化策略。
        
        对每个策略:
        1. 重复 n_repeats 次（不同随机种子）
        2. 每次给 budget 次评估机会
        3. 记录 best_so_far, regret, 约束违反数
        
        输出: 统计显著的性能对比
        """
        results = {}
        for strategy in strategies:
            strategy_results = []
            for repeat in range(n_repeats):
                history = self._run_single(strategy, budget or n_trials, seed=repeat)
                strategy_results.append(history)
            results[strategy.name] = strategy_results
        
        return self._compute_statistics(results)
```

### 3.4 展示的平台特性

| 平台特性 | 在锌电沉积中的体现 |
|----------|-------------------|
| **约束处理** | 添加剂总浓度约束 ΣA_i ≤ 1.0 |
| **未知约束学习** | 某些添加剂组合导致溶液沉淀 → 实验失败 |
| **异质噪声 GP (v4)** | EIS 质量差的实验自动降权 |
| **Domain Config 驱动 (v4)** | Squidstat 仪器参数从 YAML 注入 |
| **成本感知停止** | "再做 5 个实验能提升多少？" |
| **参数重要性** | 哪种添加剂影响最大？（fANOVA / SHAP） |
| **AutoSampler** | 初始 RE → PHYSBO → 收敛后自动停止 |
| **混合参数空间** | 7 个连续参数 + 单纯形约束 |
| **UQ metadata 链路 (v4)** | 每个实验的 quality_flags 追踪到最终 GP 权重 |

### 3.5 期望结论

```
1. 异质噪声 GP 在 < 30 个实验时优于同质 GP
   （通过 replay benchmark，50 次重复，p < 0.05）

2. 约束学习避免了 ~15% 的失败实验
   （对比无约束学习的策略）

3. 成本感知停止在第 ~25 个实验建议停止
   （余下实验的边际改进 < 0.5% CE）

4. 参数重要性分析揭示 Additive_A 和 Additive_D 贡献最大
   （SHAP 值 + fANOVA 一致）

5. VSUP 可视化帮助识别"看起来好但不确定性高"的区域

6. ★ Domain-config 驱动的 Extractor 正确加载 Squidstat 参数，
   输出的 MeasurementWithUncertainty.metadata 包含完整的
   Nyquist 特征 + quality_flags，可直接供 Agent 消费

7. ★ 过程式规则 (spectrum_rules.py) 正确诊断了 N 个异常实验
   （对比无规则的 baseline：confidence-only 判断）
```

---

## 4. Case Study 2: 催化反应条件优化

### 4.1 数据来源

**公开数据集：** Suzuki-Miyaura 交叉偶联反应优化

```
来源:
- Perera et al., Science (2018): "A platform for automated nanomole-scale 
  reaction screening and micromole-scale synthesis in flow"
- EDBO (Experimental Design via Bayesian Optimization): 
  GitHub: https://github.com/b-shields/edbo
- Shields et al., Nature (2021): "Bayesian reaction optimization"

参数空间:
- 催化剂 (categorical, 4 choices): Pd(OAc)2, Pd(PPh3)4, PdCl2, Pd2(dba)3
- 配体 (categorical, 6 choices): PPh3, XPhos, SPhos, BINAP, dppf, PCy3
- 碱 (categorical, 5 choices): K2CO3, Cs2CO3, KOtBu, Et3N, DBU
- 温度 (continuous): 40-120°C
- 时间 (continuous): 1-24h
- 浓度 (continuous): 0.05-0.5M

目标: Yield (%)
约束: 某些催化剂-配体组合不兼容（未知，需学习）
```

### 4.2 为什么选这个

1. **催化反应优化是 BO 文献中的经典应用** — 直接对标 EDBO、CIME4R
2. **混合参数空间** — 考验 `DomainEncoding`（one-hot, fingerprint）
3. **有已发表的 baseline** — Shields 的 EDBO 结果可以直接对比
4. **高噪声** — 反应 yield 变异性大，考验噪声处理
5. **约束丰富** — 催化剂-配体不兼容是典型的未知约束
6. **★ Domain config 切换验证** — 从 electrochemistry → catalysis，零代码修改

### 4.3 Domain Config: catalysis/

**文件：** `domain_knowledge/catalysis/instruments.yaml`

```yaml
# 催化反应的仪器参数
instruments:
  hplc:
    model: "generic HPLC"
    yield_accuracy_pct: 2.0        # yield ±2%
    detection_limit_pct: 0.5       # 低于 0.5% 不可靠
    retention_time_variance_s: 0.3 # 保留时间变异

  reactor:
    model: "flow_reactor"
    temperature_accuracy_c: 1.0    # ±1°C
    flow_rate_accuracy_pct: 0.5    # ±0.5%
    mixing_efficiency: 0.95        # 混合效率
```

**文件：** `domain_knowledge/catalysis/physical_constraints.yaml`

```yaml
physical_constraints:
  yield:
    min: 0
    max: 100
    unit: "%"
    warning_range: [95, 100]       # 接近 100% 可能是分析误差
    
  temperature:
    min: 20
    max: 200
    unit: "°C"
    
  concentration:
    min: 0.001
    max: 2.0
    unit: "M"

quality_thresholds:
  confidence_min: 0.5
  relative_uncertainty_max: 0.5

# 催化反应特有: 已知不兼容组合
known_incompatibilities:
  - catalyst: "Pd(OAc)2"
    ligand: "BINAP"
    reason: "oxidation state mismatch"
  - catalyst: "PdCl2"
    ligand: "PCy3"
    reason: "poor activation"
```

**文件：** `domain_knowledge/catalysis/spectrum_rules.py`

```python
"""催化反应的过程式规则。"""

def validate_yield_physics(
    yield_value: float, 
    yield_variance: float,
    temperature: float | None = None
) -> dict:
    """Yield 物理合理性校验。"""
    result = {"valid": True, "confidence_modifier": 1.0, "flags": []}
    
    if yield_value > 100:
        result["valid"] = False
        result["flags"].append("yield_above_100")
        result["confidence_modifier"] = 0.3
    elif yield_value > 95 and yield_variance < 1.0:
        result["flags"].append("suspiciously_high_yield")
        result["confidence_modifier"] = 0.8
    elif yield_value < 1.0:
        result["flags"].append("near_zero_yield")
        # 可能是约束违反（不兼容组合）而不是低活性
        result["confidence_modifier"] = 0.6
    
    return result


def check_catalyst_ligand_compatibility(
    catalyst: str,
    ligand: str,
    known_incompatibilities: list[dict]
) -> dict | None:
    """检查催化剂-配体兼容性。"""
    for entry in known_incompatibilities:
        if entry["catalyst"] == catalyst and entry["ligand"] == ligand:
            return {
                "diagnosis": "incompatible_combination",
                "confidence": 0.9,
                "evidence": entry["reason"],
                "suggestion": "skip this combination",
                "severity": "critical"
            }
    return None
```

### 4.4 实现要点

**文件：** `case_studies/catalysis/suzuki_benchmark.py` (~250 行)

```python
class SuzukiCouplingBenchmark:
    """Suzuki-Miyaura 偶联反应优化 benchmark。
    
    使用 Shields et al. 的公开数据构建代理模型:
    - 3955 个实验数据点
    - 4 个分类参数 + 2 个连续参数
    - yield 0-100%
    
    ★ 使用 catalysis/ domain config:
    - instruments.yaml → HPLC yield 精度
    - physical_constraints.yaml → yield [0, 100], 不兼容组合
    - spectrum_rules.py → validate_yield_physics()
    """
    
    def __init__(self, data_path: str = None):
        # ★ Domain config 驱动
        self.domain_config = DomainConfig("catalysis")
        
        if data_path:
            self.data = self._load_shields_data(data_path)
        else:
            self.data = self._load_embedded_subset()
    
    def evaluate(self, x: dict) -> float:
        """查表评估 (如果该组合在数据中) 或代理预测。
        
        x = {
            "catalyst": "Pd(OAc)2",
            "ligand": "SPhos",
            "base": "Cs2CO3",
            "temperature": 80.0,
            "time": 6.0,
            "concentration": 0.1
        }
        """
        # 先检查已知不兼容 (从 domain config)
        incompatibilities = self.domain_config.constraints.get(
            "known_incompatibilities", []
        )
        compat_check = self.domain_config.apply_rule(
            "check_catalyst_ligand_compatibility",
            catalyst=x["catalyst"],
            ligand=x["ligand"],
            known_incompatibilities=incompatibilities
        )
        if compat_check and compat_check["severity"] == "critical":
            return None  # 约束违反
        
        # 查表
        exact_match = self._lookup(x)
        if exact_match is not None:
            return exact_match
        
        # 代理预测
        return self.surrogate.predict(self._encode(x))[0]
    
    def get_known_constraints(self) -> list:
        """已知约束（从 domain config 读取）。"""
        return self.domain_config.constraints.get(
            "known_incompatibilities", []
        )
    
    def get_hidden_constraints(self) -> callable:
        """隐藏约束（用于测试约束学习）。
        
        从数据中学习: yield = 0 的组合 → 可能是约束违反。
        """
        def is_feasible(x):
            encoded = self._encode(x)
            neighbors = self._find_nearest(encoded, k=3)
            return all(n["yield"] > 5.0 for n in neighbors)
        return is_feasible
```

### 4.5 展示的平台特性

| 平台特性 | 在催化反应中的体现 |
|----------|-------------------|
| **DomainEncoding** | 催化剂/配体/碱 → one-hot / fingerprint |
| **AutoSampler 切换** | 初始 Sobol → GP-BO → 收敛 |
| **迁移学习** | 从 Pd(OAc)2 数据迁移到新催化剂 |
| **参数重要性** | 催化剂选择 vs 温度 vs 浓度 |
| **未知约束学习** | 不兼容组合的自动识别 |
| **对标 EDBO** | 直接比较 regret curve |
| **★ Domain config 切换** | electrochemistry → catalysis, 零代码修改 |
| **★ 过程式规则泛化** | validate_yield_physics() vs validate_ce_physics() |

---

## 5. Case Study 3: 钙钛矿薄膜组分优化

### 5.1 数据来源

```
来源:
- Accelerated Discovery datasets:
  - Sun et al., Joule (2019): CsPbI3 perovskite thin films
  - MacLeod et al., Science Advances (2020): Perovskite discovery
  - Häse et al., Chem (2020): Bayesian materials discovery
  
- Citrine Informatics 公开数据集
- Materials Project (间接)

参数空间:
- Cs/MA/FA 比例 (3 continuous, sum=1 → simplex constraint)
- Pb/Sn 比例 (1 continuous)
- Halide 比例 I/Br/Cl (3 continuous, sum=1)
- 退火温度 (continuous): 100-200°C
- 退火时间 (continuous): 5-60 min
- 前驱体浓度 (continuous): 0.5-1.5M

目标 (多目标):
- PCE (power conversion efficiency) ↑
- Stability (shelf life) ↑
- Bandgap → target value (1.3-1.5 eV)

约束:
- 组分和约束 (已知 hard)
- Pb 含量上限 (环保, 已知 soft)
- 相稳定性 (未知, 需学习)
```

### 5.2 为什么选这个

1. **钙钛矿是 2024-2026 材料科学最热的领域**
2. **多目标** — PCE vs Stability 是经典 trade-off
3. **simplex 约束** — 组分和=1 是化学中最常见的约束
4. **已有 SDL 验证** — MacLeod et al. 的 Ada robot 已证明 BO 有效
5. **数据集丰富** — 多个公开来源可组合
6. **★ 仪器组合不同** — 需要 XRD + PL extractor，验证 config 可组合性

### 5.3 Domain Config: perovskite/

**文件：** `domain_knowledge/perovskite/instruments.yaml`

```yaml
instruments:
  xrd:
    model: "lab XRD"
    instrument_broadening_deg: 0.08    # 不同于 electrochemistry 的 0.05
    two_theta_accuracy_deg: 0.02
    scherrer_k_range: [0.89, 0.94]

  pl:   # 光致发光 — 电化学没有这个仪器
    model: "PL spectrometer"
    wavelength_range_nm: [400, 900]
    intensity_noise_pct: 3.0
    peak_position_accuracy_nm: 0.5
    
  solar_simulator:
    model: "AM1.5G"
    pce_accuracy_pct: 0.5              # PCE ±0.5%
    jsc_accuracy_ma_cm2: 0.1
    voc_accuracy_mv: 5.0
    ff_accuracy_pct: 1.0
    
  spin_coater:
    speed_accuracy_rpm: 10
    time_accuracy_s: 0.5
```

**文件：** `domain_knowledge/perovskite/physical_constraints.yaml`

```yaml
physical_constraints:
  PCE:
    min: 0
    max: 33            # Shockley-Queisser limit for single junction
    unit: "%"
    warning_range: [28, 33]
    
  bandgap:
    min: 0.5
    max: 3.0
    unit: "eV"
    target_range: [1.3, 1.5]   # 最优带隙范围
    
  stability_hours:
    min: 0
    unit: "hours"
    
  composition:
    simplex_tolerance: 0.01    # Cs+MA+FA = 1.0 ± 0.01
    halide_simplex: true       # I+Br+Cl = 1.0

quality_thresholds:
  confidence_min: 0.5
  relative_uncertainty_max: 0.5

# 钙钛矿特有: 已知相稳定性约束
phase_stability:
  - condition: "FA > 0.85"
    issue: "δ-phase formation at room temperature"
  - condition: "Cs < 0.05 and Br < 0.1"
    issue: "poor phase stability"
```

### 5.4 实现要点

**文件：** `case_studies/perovskite/composition_benchmark.py` (~250 行)

```python
class PerovskiteCompositionBenchmark:
    """钙钛矿组分优化 benchmark。
    
    特点:
    - 多目标 (PCE + Stability)
    - Simplex 约束 (组分和 = 1)
    - 高维 (6-10D)
    - 公开数据支撑
    
    ★ 使用 perovskite/ domain config:
    - instruments.yaml → solar_simulator, XRD, PL 参数
    - physical_constraints.yaml → PCE [0, 33], simplex, phase stability
    """
    
    def __init__(self, dataset: str = "sun2019"):
        """
        dataset options:
        - "sun2019": CsPbI3 stability data
        - "macleod2020": Ada robot discovery data
        - "synthetic_realistic": 基于文献参数的合成数据
        """
        self.domain_config = DomainConfig("perovskite")
        self.data = self._load_dataset(dataset)
        self.surrogate_pce = self._build_surrogate("pce")
        self.surrogate_stability = self._build_surrogate("stability")
    
    def evaluate(self, x: dict) -> dict | None:
        """多目标评估。"""
        # simplex 约束检查 (从 domain config)
        simplex_tol = self.domain_config.constraints.get(
            "physical_constraints", {}
        ).get("composition", {}).get("simplex_tolerance", 0.01)
        
        cation_sum = x.get("Cs", 0) + x.get("MA", 0) + x.get("FA", 0)
        if abs(cation_sum - 1.0) > simplex_tol:
            return None  # 约束违反
        
        encoded = self._encode_composition(x)
        
        pce, pce_var = self.surrogate_pce.predict(encoded)
        stability, stab_var = self.surrogate_stability.predict(encoded)
        
        return {
            "pce": {"value": pce, "variance": pce_var},
            "stability": {"value": stability, "variance": stab_var}
        }
    
    def get_simplex_constraint(self) -> callable:
        """组分和约束: Cs + MA + FA = 1.0"""
        def check(x):
            return abs(x["Cs"] + x["MA"] + x["FA"] - 1.0) < 0.01
        return check
```

### 5.5 展示的平台特性

| 平台特性 | 在钙钛矿中的体现 |
|----------|-------------------|
| **NSGA-II / MOBO** | PCE vs Stability Pareto 前沿 |
| **约束处理** | Simplex (组分和), Pb 上限 |
| **超体积指标** | Pareto 前沿质量追踪 |
| **Radar 图** | >3 目标时的方案对比 |
| **成本感知** | 退火 1h vs 5min 的 trade-off |
| **批量调度** | 一次旋涂 8 个样品 |
| **★ 新仪器 config** | PL + solar simulator（electrochemistry 没有） |
| **★ 不同物理约束** | Shockley-Queisser limit vs CE limit |

---

## 6. Domain Config 切换验证

> 本节是 v5 新增内容。专门验证"同一套代码 + 不同 YAML = 不同领域"的核心架构卖点。

### 6.1 验证目标

```
命题: 给定 optimization_copilot 的 core code (v1-v4),
      仅通过替换 domain_knowledge/{domain}/ 下的 YAML + rules,
      即可适配新的实验体系，不需要修改任何 core code。

验证方式:
  Case Study 1 (electrochemistry) → Case Study 2 (catalysis)
  → Case Study 3 (perovskite)

  测量指标:
  1. 需要修改的 core code 行数 = 0 (hard requirement)
  2. 新领域只需编写: config YAML + spectrum_rules.py + data_loader.py
  3. Extractor + UQ Propagator + GP 全部自动适配
```

### 6.2 三个领域的 Config 对比

| Config 维度 | electrochemistry | catalysis | perovskite |
|-------------|-----------------|-----------|------------|
| **仪器数** | 3 (Squidstat, UV-Vis, XRD) | 2 (HPLC, reactor) | 4 (XRD, PL, solar sim, spin coater) |
| **Extractor 数** | 4 (EIS, DC, UV-Vis, XRD) | 1 (yield) | 3 (PCE, XRD, PL) |
| **物理约束** | CE [0,105], Rct>0 | Yield [0,100] | PCE [0,33], simplex |
| **过程式规则** | EIS 异常诊断, CE 校验 | Yield 校验, 兼容性 | 相稳定性检查 |
| **等效电路** | Randles, 2RC, ... | N/A | N/A |
| **特殊约束** | 添加剂总浓度 ≤1 | 催化剂-配体兼容 | 组分 simplex |

### 6.3 Config 切换自动化测试

```python
class DomainConfigSwitchTest:
    """验证 domain config 切换不破坏 core code。
    
    对每个领域:
    1. 加载 domain config
    2. 构建 Extractors
    3. 运行 UQ 链路 (Extractor → Propagator → GP)
    4. 验证输出类型一致 (MeasurementWithUncertainty)
    5. 验证 metadata 字段完整
    """
    
    def test_config_switch(self):
        domains = ["electrochemistry", "catalysis", "perovskite"]
        
        for domain in domains:
            config = DomainConfig(domain)
            
            # 1. Config 加载成功
            assert config.instruments is not None
            assert config.constraints is not None
            
            # 2. 所有声明的仪器有合法参数
            for inst_name, inst_spec in config.instruments.get("instruments", {}).items():
                assert isinstance(inst_spec, dict)
            
            # 3. 物理约束有 min/max
            for kpi_name, constraint in config.constraints.get(
                "physical_constraints", {}
            ).items():
                # 至少有 min 或 max 之一
                assert "min" in constraint or "max" in constraint
            
            # 4. Quality thresholds 存在
            thresholds = config.get_quality_thresholds()
            assert "confidence_min" in thresholds
    
    def test_extractor_output_consistency(self):
        """所有领域的 Extractor 输出相同的类型结构。"""
        # electrochemistry
        ec_config = DomainConfig("electrochemistry")
        ec_extractor = EISUncertaintyExtractor(ec_config)
        ec_results = ec_extractor.extract_with_uncertainty(mock_eis_data)
        
        for m in ec_results:
            assert isinstance(m, MeasurementWithUncertainty)
            assert "raw_features" in m.metadata
            assert "quality_flags" in m.metadata
        
        # catalysis (假设实现了 YieldExtractor)
        cat_config = DomainConfig("catalysis")
        cat_extractor = YieldUncertaintyExtractor(cat_config)
        cat_results = cat_extractor.extract_with_uncertainty(mock_yield_data)
        
        for m in cat_results:
            assert isinstance(m, MeasurementWithUncertainty)
            assert "raw_features" in m.metadata
            assert "quality_flags" in m.metadata
```

### 6.4 论文中的呈现方式

这个验证直接转化为论文的一个 subsection:

```
"4.4 Cross-Domain Generalization

To validate the domain-configurable design, we applied the same 
optimization framework to three distinct experimental domains—
zinc electrodeposition, Suzuki coupling catalysis, and perovskite 
thin-film optimization—by only modifying YAML configuration files 
(Table 3). No core code changes were required. The measurement 
uncertainty propagation pipeline correctly adapted to each domain's 
instrument specifications and physical constraints."

Table 3: Domain Config Comparison
(上面 6.2 的表格)
```

---

## 7. Benchmark 到 Case Study 的技术桥梁

### 7.1 通用 Case Study 框架

**文件：** `case_studies/base.py` (~200 行)

```python
from abc import ABC, abstractmethod

class ExperimentalBenchmark(ABC):
    """真实实验数据 benchmark 的基类。
    
    与合成 benchmark (BenchmarkFunction) 的区别:
    1. evaluate() 可能失败 (返回 None → 约束违反)
    2. evaluate() 有噪声 (noise_var 不为 0)
    3. 参数空间可以是混合的
    4. 有 known + unknown 约束
    5. 评估有成本 (不同参数组合成本不同)
    
    ★ 新增: 每个 benchmark 关联一个 DomainConfig
    """
    
    def __init__(self, domain_name: str | None = None):
        self.domain_config = DomainConfig(domain_name) if domain_name else None
    
    @abstractmethod
    def evaluate(self, x: dict) -> dict | None:
        """评估实验。
        
        Returns:
            dict with {objective_name: {"value": float, "variance": float}}
            or None if constraint violated (experiment failed)
        """
    
    @abstractmethod
    def get_search_space(self) -> dict:
        """参数空间定义。"""
    
    @abstractmethod
    def get_objectives(self) -> dict:
        """目标函数定义。"""
    
    def get_known_constraints(self) -> list:
        """已知约束。可从 domain config 读取。"""
        if self.domain_config:
            return self.domain_config.constraints.get(
                "known_incompatibilities", []
            )
        return []
    
    def get_evaluation_cost(self, x: dict) -> float:
        """评估成本（归一化）。"""
        return 1.0
    
    def is_feasible(self, x: dict) -> bool:
        """真实可行性（用于离线评估，不暴露给优化器）。"""
        return True
    
    def get_domain_config(self) -> 'DomainConfig | None':
        """返回关联的 domain config。"""
        return self.domain_config


class ReplayBenchmark(ExperimentalBenchmark):
    """基于真实数据的代理回放 benchmark。
    
    通用流程:
    1. 加载真实实验数据
    2. 建立代理模型
    3. 使用代理 + 噪声模型作为虚拟实验
    
    子类只需实现 _load_data() 和 _build_surrogate()。
    """
    
    def __init__(self, data_path: str = None, domain_name: str | None = None):
        super().__init__(domain_name)
        self.raw_data = self._load_data(data_path)
        self.surrogate = self._build_surrogate()
        self.noise_model = self._build_noise_model()
    
    def evaluate(self, x: dict) -> dict | None:
        if not self.is_feasible(x):
            return None
        
        encoded = self._encode(x)
        predictions = {}
        
        for obj_name, surrogate in self.surrogates.items():
            mu, var = surrogate.predict(encoded)
            noise_var = self.noise_model.predict_noise(encoded, obj_name)
            
            import random
            y = random.gauss(mu, (var + noise_var) ** 0.5)
            
            predictions[obj_name] = {
                "value": y,
                "variance": noise_var
            }
        
        return predictions
```

### 7.2 统一评估器

**文件：** `case_studies/evaluator.py` (~300 行)

```python
class CaseStudyEvaluator:
    """统一的 case study 评估框架。
    
    对每个 (benchmark, strategy) 组合:
    1. 运行 n_repeats 次独立实验
    2. 记录标准化指标
    3. 生成对比图表
    4. 统计检验 (Wilcoxon signed-rank test)
    """
    
    def __init__(self, benchmark: ExperimentalBenchmark):
        self.benchmark = benchmark
    
    def run_comparison(
        self,
        strategies: dict[str, 'OptimizationBackend'],
        budget: int,
        n_repeats: int = 30,
        metrics: list[str] = None
    ) -> 'ComparisonReport':
        """运行对比实验。
        
        metrics 可选:
        - "simple_regret": 最佳找到值 vs 全局最优
        - "cumulative_regret": 累积后悔
        - "feasibility_rate": 可行实验比例
        - "constraint_violations": 约束违反次数
        - "cost_adjusted_regret": 成本加权后悔
        - "hypervolume" (多目标): Pareto 超体积
        - "convergence_speed": 达到 90% 最优的速度
        """
        ...
    
    def generate_report(self, results: 'ComparisonReport') -> dict:
        """生成可视化报告。"""
        plots = {
            "convergence": self._plot_convergence(results),
            "box_comparison": self._plot_box(results),
            "significance": self._compute_significance(results),
            "parameter_importance": self._plot_importance(results),
            "constraint_accuracy": self._plot_constraint_learning(results),
        }
        return plots
```

### 7.3 Baselines 对比

```python
# 对每个 case study，比较以下策略:

baselines = {
    # --- 我们的平台 ---
    "Ours_GP-BO": GPBOBackend(noise_model="heteroscedastic"),
    "Ours_GP-BO_homo": GPBOBackend(noise_model="homoscedastic"),  # ablation
    "Ours_TuRBO": TuRBOBackend(),
    "Ours_MOBO": NSGAIIBackend(),  # for multi-objective
    
    # --- 外部 baseline ---
    "Random": RandomSearchBaseline(),
    "Sobol": SobolBaseline(),
    "PHYSBO": PHYSBOAdapter(),     # 你当前在用的
    "EDBO": EDBOAdapter(),          # Shields et al.
    "Optuna_TPE": OptunaTPEAdapter(),
    "CMA-ES": CMAESBackend(),
    
    # --- Ablation studies ---
    "Ours_no_constraints": GPBOBackend(constraint_engine=None),
    "Ours_no_cost_aware": GPBOBackend(cost_tracker=None),
    "Ours_no_transfer": GPBOBackend(transfer_learning=None),
    
    # --- ★ 新增: Domain config ablation ---
    "Ours_no_domain_config": GPBOBackend(
        noise_model="heteroscedastic",
        domain_config=None,         # 不用 domain YAML → 默认参数
    ),
    "Ours_no_spectrum_rules": GPBOBackend(
        noise_model="heteroscedastic",
        domain_config=DomainConfig("electrochemistry"),
        disable_rules=True,         # 有 YAML 但禁用 spectrum_rules.py
    ),
}
```

### 7.4 新增 Ablation Studies

| Ablation | 对比 | 验证什么 |
|----------|------|---------|
| hetero vs homo GP | `Ours_GP-BO` vs `Ours_GP-BO_homo` | v4 异质噪声的价值 |
| with vs without domain config | `Ours_GP-BO` vs `Ours_no_domain_config` | ★ YAML 注入仪器参数的价值 |
| with vs without spectrum rules | `Ours_GP-BO` vs `Ours_no_spectrum_rules` | ★ 过程式规则诊断的价值 |
| with vs without constraints | `Ours_GP-BO` vs `Ours_no_constraints` | 约束学习的价值 |
| with vs without cost awareness | `Ours_GP-BO` vs `Ours_no_cost_aware` | 成本感知的价值 |

**期望结论（domain config ablation）：**

```
1. Domain config 驱动 vs 默认参数:
   - 有 config: EIS |Z| 的 noise_var 正确反映低频噪声放大 → GP 更准
   - 无 config: 所有频率同等噪声 → GP 被低频噪声点误导
   期望: config 版在 <30 样本时 regret 低 10-20% (p < 0.05)

2. 有 spectrum_rules vs 无 spectrum_rules:
   - 有 rules: 异常 EIS 被自动诊断 (SEI, passivation, contact) → confidence 降低
   - 无 rules: 只靠 confidence 数值阈值 → 漏检部分异常
   期望: rules 版的 outlier detection F1 score 高 15-25%
```

---

## 8. 评估框架

### 8.1 指标体系

```python
@dataclass
class PerformanceMetrics:
    """标准化性能指标。"""
    
    # 基础
    best_value: float              # 最终找到的最优值
    simple_regret: float           # |best_found - global_opt|
    
    # 效率
    convergence_iteration: int     # 达到 90% 最优的迭代数
    area_under_curve: float        # convergence curve 下面积（越小越好）
    
    # 约束
    feasibility_rate: float        # 可行实验比例
    constraint_violations: int     # 约束违反总次数
    
    # 成本
    total_cost: float              # 总评估成本
    cost_adjusted_regret: float    # regret / cost
    
    # 多目标 (如适用)
    hypervolume: float | None = None
    pareto_front_size: int | None = None
    
    # 鲁棒性
    std_across_repeats: float = 0.0  # 重复实验间的标准差
    
    # ★ 新增: UQ 质量指标
    mean_confidence: float = 0.0      # 平均 KPI confidence
    outlier_detection_f1: float = 0.0 # 异常检测 F1 (需要 ground truth label)
    noise_calibration_error: float = 0.0  # |predicted_noise - actual_noise|
```

### 8.2 统计检验

```python
def statistical_comparison(
    results_a: list[float],  # 策略 A 的 n_repeats 次结果
    results_b: list[float],  # 策略 B 的 n_repeats 次结果
    test: str = "wilcoxon"
) -> dict:
    """统计显著性检验。
    
    使用 Wilcoxon signed-rank test (非参数):
    - 不需要正态性假设
    - 适合小样本 (n=30)
    - 配对比较（同一随机种子）
    
    Pure Python 实现。
    """
    ...
```

### 8.3 生成的图表清单

| 图表 | 用途 | 对标 |
|------|------|------|
| Convergence curves (mean ± shade) | 策略收敛速度对比 | 每篇 BO 论文都有 |
| Box plots of final performance | 鲁棒性对比 | 标准做法 |
| Heatmap of p-values | 统计显著性矩阵 | Bayesmark 风格 |
| Ablation bar chart | 各组件贡献度 | Nature MI 风格 |
| Constraint learning ROC | 约束预测准确率 | 独有 |
| Cost vs Performance Pareto | 成本效率 | 独有 |
| Parameter importance comparison | 跨方法的重要性一致性 | SHAP 论文风格 |
| ★ Domain config ablation | config vs no-config 对比 | 独有 |
| ★ Uncertainty budget across experiments | 各实验的方差贡献饼图 | 独有 (v4) |
| ★ Cross-domain config comparison | 三个领域的 config 差异表 | 独有 |

---

## 9. Agent 验证路线图

> 本节定义 v5 数据如何服务未来 Layer 3 Agent 的验证。
> v5 阶段不实现 Agent 代码，但数据结构和验证框架要 ready。

### 9.1 v5 作为 Agent 验证数据集

```
v5 锌电沉积数据的双重角色:

角色 1 (v5 阶段): BO 平台的 case study 验证
  - 证明 hetero GP > homo GP
  - 证明 domain config 有效
  - 证明 spectrum_rules 能提升 outlier detection

角色 2 (v6+ 阶段): Agent 的首要验证数据集
  - v6c AnomalyDetectionAgent:
    - 用 v5 zinc 数据中标注的异常实验 → ground truth
    - 对比: Agent 诊断 vs spectrum_rules 诊断 vs confidence-only
    
  - v7b SymRegAgent / Explainability:
    - 用 v5 zinc 数据的 GP 模型 → PySR 方程发现
    - 验证: 发现的方程是否 match 已知电化学规律
    
  - SpectrumAnalysisAgent:
    - 用 v5 zinc 数据的 EIS raw features → Agent 解读
    - 对比: Agent 解读 vs spectrum_rules 解读 vs 人工解读
    
  - HypothesisAgent:
    - 用 v5 zinc 优化轨迹 → 生成机理假设
    - 验证: 假设是否 match 已知添加剂作用机理
```

### 9.2 v5 需要预留的 Agent 验证接口

```python
class ZincElectrodepositionDataLoader:
    """... 已有方法 ..."""
    
    # ★ Agent 验证预留方法
    
    def get_anomaly_labels(self) -> dict[str, str]:
        """手工标注的异常实验 ground truth。
        
        用于评估 AnomalyDetectionAgent 的准确率。
        
        Returns:
            {experiment_id: anomaly_type}
            anomaly_type ∈ {"normal", "bubble", "contact_problem", 
                           "electrode_passivation", "solution_degradation"}
        
        来源: 你在做实验时的观察记录 + 事后 EIS 分析。
        """
        ...
    
    def get_optimization_trajectory(self) -> list[dict]:
        """按时间顺序的优化轨迹。
        
        用于评估 HypothesisAgent 是否能识别:
        - 优化拐点 (从 exploration → exploitation)
        - 平台期 (5 轮无改进)
        - 参数敏感度变化
        
        Returns:
            [{
                "round": int,
                "x": dict,
                "measurements": list[MeasurementWithUncertainty],
                "observation": ObservationWithNoise,
                "bo_acquisition": str,  # "exploration" or "exploitation"
            }]
        """
        ...
    
    def get_known_mechanisms(self) -> list[dict]:
        """已知的添加剂作用机理 (文献 + 你的理解)。
        
        用于评估 HypothesisAgent 生成的假设是否合理。
        
        Returns:
            [{
                "additive": str,
                "mechanism": str,
                "evidence": str,
                "literature": str,
            }]
        
        示例:
        - Additive_A (CTAB): 表面活性剂，高浓度引起 Zn²⁺ 络合
        - Additive_B (PEG): 粘度增加，抑制枝晶生长
        """
        ...
```

### 9.3 验证时间线

```
v5 阶段 (当前):
  ✅ 数据加载 + UQ 链路 + replay benchmark
  ✅ anomaly_labels 标注 (手工)
  ✅ optimization_trajectory 记录
  ✅ known_mechanisms 整理
  → 这些数据存在 case_studies/zinc/ 下，Agent 阶段直接消费

v6c 阶段 (AnomalyDetectionAgent):
  → 用 v5 anomaly_labels 评估 Agent 的 F1 score
  → 对比: Agent vs spectrum_rules vs confidence-only

v7b 阶段 (Explainability):
  → 用 v5 zinc GP 模型 → PySR → equation discovery
  → 用 known_mechanisms 评估方程的化学合理性

v6+ Agent 全面验证:
  → 用 v5 optimization_trajectory → Agent 全链路 replay
  → 模拟: 如果有 Agent 参与优化，能否更快收敛？
  → A/B: agent-enhanced BO vs plain BO on v5 replay
```

---

## 10. 代码量估算 & 文件结构

| 新增模块 | 文件 | 估算行数 | 与原版差异 |
|----------|------|---------|----------|
| 基类框架 | `case_studies/base.py` | ~220 | +20 (DomainConfig 集成) |
| 统一评估器 | `case_studies/evaluator.py` | ~320 | +20 (UQ 指标) |
| 锌电沉积数据加载 | `case_studies/zinc/data_loader.py` | ~280 | +80 (DomainConfig + agent 预留方法) |
| 锌电沉积 benchmark | `case_studies/zinc/benchmark.py` | ~250 | 同 |
| 锌电沉积标注数据 | `case_studies/zinc/annotations.py` | ~80 | ★ 全新 (anomaly labels + mechanisms) |
| 催化反应数据加载 | `case_studies/catalysis/data_loader.py` | ~150 | 同 |
| Suzuki benchmark | `case_studies/catalysis/suzuki_benchmark.py` | ~270 | +20 (DomainConfig) |
| 催化 domain config | `domain_knowledge/catalysis/*.yaml + .py` | ~100 | ★ 全新 |
| 钙钛矿数据加载 | `case_studies/perovskite/data_loader.py` | ~150 | 同 |
| 钙钛矿 benchmark | `case_studies/perovskite/composition_benchmark.py` | ~270 | +20 (DomainConfig) |
| 钙钛矿 domain config | `domain_knowledge/perovskite/*.yaml` | ~80 | ★ 全新 |
| Domain config 切换测试 | `tests/test_domain_switch.py` | ~120 | ★ 全新 |
| Baseline adapters | `case_studies/baselines/` | ~350 | +50 (config ablation variants) |
| 统计检验 | `case_studies/statistics.py` | ~150 | 同 |
| 报告生成 | `case_studies/reporting.py` | ~220 | +20 (新图表) |
| 测试 | `tests/test_case_studies/` | ~400 | 同 |
| **合计** | | **~3,410** | +约 610 行 |

```
optimization_copilot/
├── domain_knowledge/                         # v4 创建, v5 扩展
│   ├── schema.py
│   ├── loader.py
│   ├── electrochemistry/                     # v4 已有
│   │   ├── instruments.yaml
│   │   ├── physical_constraints.yaml
│   │   ├── eis_models.yaml
│   │   └── spectrum_rules.py
│   ├── catalysis/                            # ★ v5 新增
│   │   ├── instruments.yaml
│   │   ├── physical_constraints.yaml
│   │   └── spectrum_rules.py
│   └── perovskite/                           # ★ v5 新增
│       ├── instruments.yaml
│       └── physical_constraints.yaml
│
├── case_studies/                              # ★ v5 核心
│   ├── __init__.py
│   ├── base.py                               # + DomainConfig
│   ├── evaluator.py                          # + UQ 指标
│   ├── statistics.py
│   ├── reporting.py                          # + 新图表
│   │
│   ├── zinc/
│   │   ├── __init__.py
│   │   ├── data_loader.py                    # + DomainConfig + agent 预留
│   │   ├── benchmark.py
│   │   ├── annotations.py                    # ★ anomaly labels + mechanisms
│   │   └── README.md
│   │
│   ├── catalysis/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── suzuki_benchmark.py               # + DomainConfig
│   │   └── README.md
│   │
│   ├── perovskite/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── composition_benchmark.py          # + DomainConfig
│   │   └── README.md
│   │
│   └── baselines/
│       ├── __init__.py
│       ├── random_baseline.py
│       ├── physbo_adapter.py
│       ├── edbo_adapter.py
│       └── optuna_adapter.py
│
└── tests/
    ├── test_case_studies/
    └── test_domain_switch.py                  # ★ 跨领域切换测试
```

---

## 11. 实现优先级

| 优先级 | 模块 | 理由 |
|--------|------|------|
| **P0** | `base.py` + `evaluator.py` | 框架基础，其他都依赖 |
| **P0** | `zinc/` 全部 (含 annotations.py) | **你的数据**，最有发表价值 + Agent 验证基础 |
| **P1** | `catalysis/suzuki` + catalysis domain config | 有公开数据 + 已有 baseline (EDBO) + domain 切换验证 |
| **P1** | `statistics.py` + `reporting.py` | 对比结果需要统计支撑 |
| **P1** | `test_domain_switch.py` | ★ 验证核心架构卖点 |
| **P2** | `perovskite/` + perovskite domain config | 需要获取数据，工作量较大 |
| **P2** | `baselines/` adapters (含 config ablation) | 需要安装外部库或 mock |
| **P3** | 更多 case studies | 根据审稿意见补充 |

### 与 v4 的依赖关系

```
v4 (Uncertainty Propagation) ←── v5 (Case Studies)
  │                                    │
  │  types.py                          ├── zinc benchmark 需要 v4 的
  │  DomainConfig                      │   DomainConfig + Extractor + HeteroscedasticGP
  │  Extractors                        │
  │  HeteroscedasticGP                 ├── ablation study 需要对比
  │                                    │   hetero vs homo GP
  │                                    │   config vs no-config ← ★ 新增
  │                                    │   rules vs no-rules  ← ★ 新增
  │                                    │
  │                                    └── agent 验证数据预留
  │                                        (annotations, trajectory, mechanisms)
  │
  ├── v5 catalysis domain config → 验证 v4 DomainConfig 跨领域泛化
  └── v5 perovskite domain config → 验证 v4 DomainConfig 可组合性
```

**建议：** v4 P0-P1 → v5 P0 → v4 P2 → v5 P1-P2 交替推进。

---

## 12. 发表策略

### 12.1 单篇大文章 (Nature MI / Digital Discovery)

```
Title: "Measurement-Aware Bayesian Optimization for Self-Driving Laboratories:
        From Uncertainty Propagation to Real Experimental Validation"

Structure:
1. Introduction: SDL gap (measurement uncertainty ignored)
2. Method:
   a. Domain-configurable measurement uncertainty propagation (v4)
   b. Heteroscedastic GP with instrument-level noise
   c. Cross-domain generalization via YAML configuration
3. Results:
   a. Synthetic benchmarks (existing)
   b. Zinc electrodeposition (Case Study 1)
   c. Catalysis (Case Study 2)
   d. Perovskite (Case Study 3)
   e. ★ Cross-domain config switch validation
   f. ★ Ablation: config vs no-config, rules vs no-rules
4. Discussion: When does uncertainty-awareness matter?
5. Code availability: Open-source platform

Selling points:
- v4 + v5 组合 → 完整故事
- 3 个领域验证 → 通用性
- ★ Domain-configurable → 可扩展性（审稿人最关心）
- 开源平台 → 社区影响力
```

### 12.2 两篇拆分

```
Paper A (Methods): v4 + domain config → Digital Discovery / JCIM
- Focus on measurement uncertainty propagation framework
- ★ Emphasis on domain-configurable design
- Validate on zinc + catalysis (config switch)

Paper B (Application): v5 zinc 深入 → ACS Energy Letters / Batteries
- Focus on zinc electrodeposition optimization
- Use platform as tool, emphasize materials discovery
- ★ Include agent-ready data annotations for future work
```

### 12.3 时间线估算

| 里程碑 | 内容 | 预计时间 |
|--------|------|---------|
| M1 | v4 核心实现 (types + DomainConfig + propagation + hetero GP) | 2-3 周 |
| M2 | v5 锌数据集成 + 初步结果 + anomaly annotations | 2 周 |
| M3 | v5 催化 case study + catalysis domain config + baselines | 2-3 周 |
| M3.5 | ★ Domain config switch 验证 + ablation studies | 1 周 |
| M4 | 统计分析 + 图表生成 (含新图表) | 1-2 周 |
| M5 | 论文写作 | 3-4 周 |
| **总计** | | **11-15 周** |

---

## 附录 A: v5 → v6+ 数据交接清单

v5 为后续版本准备的数据和接口:

| 数据/接口 | v5 产出 | v6+ 消费者 |
|-----------|---------|-----------|
| `zinc/annotations.py` anomaly_labels | 手工标注异常实验 | v6c AnomalyDetectionAgent 验证 |
| `zinc/annotations.py` known_mechanisms | 添加剂作用机理 | HypothesisAgent 验证 |
| `zinc/data_loader.py` optimization_trajectory | 优化轨迹 | Agent 全链路 replay |
| `zinc/data_loader.py` load_full_dataset() | 完整 UQ + metadata | SpectrumAnalysisAgent 验证 |
| `catalysis/` domain config | 催化领域 YAML | Agent 跨领域泛化测试 |
| `perovskite/` domain config | 钙钛矿领域 YAML | Agent 跨领域泛化测试 |
| `test_domain_switch.py` | Config 切换测试 | Agent config 注入验证 |
| Ablation: config vs no-config | 量化 domain config 价值 | Agent 论文的 baseline |
| Ablation: rules vs no-rules | 量化过程式规则价值 | Agent (LLM) vs rules (Pragmatic) 对比基线 |
