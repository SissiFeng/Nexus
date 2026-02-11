# Optimization Copilot v4 — 测量不确定性传播链路

> **核心问题：** Feature Extractor（EIS/UV-Vis/XRD）的输出 KPI 本身有不确定性，但当前架构把它们当成精确值传给 BO。这条链路断了。
>
> **目标：** 让 `Extractor.extract() → (value, uncertainty)` → `GP.observe(y, noise_var=σ²)` 成为 first-class 的数据通路。
>
> **价值：** 对锌电沉积来说，EIS 拟合的 R_ct 本身就有 fitting uncertainty（equivalent circuit ambiguity、频率范围选择、噪声）；|Z|@1kHz 更稳定但仍有仪器噪声。如果 GP 能区分"这个点是高噪声测量"vs"这个点是低噪声测量"，代理模型的预测质量会显著提升。
>
> **v4 在整体架构中的位置：** v4 是 Layer 2 (Optimization Engine) 的 UQ 基础设施，同时为 Layer 3 (Scientific Reasoning Agents) 提供数据接口。Extractor 层的 `MeasurementWithUncertainty` 是 SpectrumAnalysisAgent 和 AnomalyDetectionAgent 的输入/输出标准格式。

---

## 目录

1. [问题定义](#1-问题定义)
2. [架构设计](#2-架构设计)
3. [核心数据类型](#3-核心数据类型)
4. [Domain-Configurable Extractors](#4-domain-configurable-extractors)
5. [Extractor 层：不确定性量化](#5-extractor-层不确定性量化)
6. [传播层：KPI → Objective](#6-传播层kpi--objective)
7. [GP 噪声模型集成](#7-gp-噪声模型集成)
8. [Agent Interface Hooks](#8-agent-interface-hooks)
9. [可视化增补](#9-可视化增补)
10. [锌电沉积实例走查](#10-锌电沉积实例走查)
11. [代码量估算 & 文件结构](#11-代码量估算--文件结构)
12. [实现优先级](#12-实现优先级)

---

## 1. 问题定义

### 1.1 当前架构的断点

```
当前流程（v2）:
┌───────────┐    ┌───────────┐    ┌────────────┐
│ Hardware  │───►│ Extractor │───►│  BO Engine │
│ (raw data)│    │ extract() │    │ observe(y) │
└───────────┘    └───────────┘    └────────────┘
                   返回: float       接收: float
                   ↑                  ↑
                   丢失了不确定性      假设同质噪声 σ² = const
```

```
目标流程（v4）:
┌───────────┐    ┌──────────────────┐    ┌─────────────┐    ┌──────────────────┐
│ Hardware  │───►│    Extractor     │───►│ UQ          │───►│   BO Engine      │
│ (raw data)│    │ extract_with_    │    │ Propagator  │    │ observe(y,       │
└───────────┘    │ uncertainty()    │    └──────┬──────┘    │  noise_var,      │
                 └──────────────────┘           │           │  metadata)       │
                   返回:                         │           └──────────────────┘
                   MeasurementWithUncertainty    │             GP 异质噪声
                   (μ, σ², confidence, metadata) │             + agent metadata
                                                 │
                                          ┌──────▼──────┐
                                          │  Layer 3    │
                                          │  Agents     │
                                          │  (consume   │
                                          │  UQ data)   │
                                          └─────────────┘
```

### 1.2 为什么这很重要

| 场景 | 没有不确定性传播 | 有不确定性传播 |
|------|---------------|--------------|
| EIS fitting 失败，R_ct 不可靠 | GP 把垃圾值当真，代理模型歪了 | GP 自动给这个点加大噪声，减轻其影响 |
| 某个 well 的 CE 因气泡异常低 | BO 避开这个区域（过度惩罚） | BO 知道这是高噪声点，愿意在附近重新探索 |
| 添加剂浓度低于仪器检测限 | 特征值精确到小数但无意义 | uncertainty flag 触发"增加浓度后重测" |
| 不同频率提取的 \|Z\| 噪声差异 | 所有频率同等权重 | 高频（低噪声）权重自然更大 |
| 异常检测需要量化"多异常" | 二值判断 (是/否) | AnomalyDetectionAgent 获得连续的 confidence 信号 |

### 1.3 v4 在三层架构中的角色

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Scientific Reasoning Agents                   │
│  SpectrumAnalysisAgent ← 消费 MeasurementWithUncertainty│
│  AnomalyDetectionAgent ← 消费 UQ + confidence           │
│  HypothesisAgent       ← 消费异常诊断结果                │
│  ...                                                    │
└──────────────────┬──────────────────────────────────────┘
                   │ OptimizationFeedback (prior_adjustment,
                   │   constraints, anomaly_verdict)
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Optimization Engine                           │
│  ★ v4: UQ 基础设施                                       │
│    Extractors → UQ Propagator → Heteroscedastic GP      │
│  v1-v3: BO backends, search space, visualization        │
│  v5+: case studies, anomaly detection, explainability   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Execution Layer                               │
│  Opentrons, Squidstat, sensors, raw data                │
└─────────────────────────────────────────────────────────┘
```

**关键设计原则：** v4 的 UQ 类型系统是 Layer 2 和 Layer 3 之间的共享语言。`MeasurementWithUncertainty` 既是 Extractor 的输出、GP 的输入，也是 Agent 的消费对象。

---

## 2. 架构设计

### 2.1 设计约束

来自整体架构（v6-v8 expansion document）的要求：

1. **Domain-agnostic core**：Extractor 和 UQ Propagator 不硬编码领域知识
2. **Domain knowledge via config**：仪器参数、物理阈值、异常模式从 YAML 注入
3. **Agent-ready interfaces**：数据类型预留 metadata 字段，supply Layer 3 需要的上下文
4. **Pragmatic-first**：Extractor 层全部确定性算法，不依赖 LLM

### 2.2 四层数据流

```
Layer 1: Raw Data
  Hardware → raw signals (voltage, current, impedance, spectra)
  
Layer 2a: Extraction + UQ (★ v4 核心)
  raw signals → MeasurementWithUncertainty[]
  Domain config 注入: 仪器精度、物理约束、阈值
  
Layer 2b: Propagation + GP
  MeasurementWithUncertainty[] → ObservationWithNoise
  ObservationWithNoise → HeteroscedasticGP.observe()
  
Layer 3: Agent Consumption (v6+ 启用)
  MeasurementWithUncertainty[] → SpectrumAnalysisAgent
  ObservationWithNoise + GP state → AnomalyDetectionAgent
  Agent feedback → GP prior adjustment / constraint update
```

---

## 3. 核心数据类型

**文件：** `uncertainty/types.py` (~100 行)

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MeasurementWithUncertainty:
    """带不确定性的测量值。
    
    所有 Extractor 的标准输出格式。
    同时是 Layer 3 Agent 的标准输入格式。
    """
    
    # --- 核心字段 ---
    value: float              # 提取的 KPI 值 (μ)
    variance: float           # 方差 (σ²)，表示测量不确定性
    confidence: float         # 置信度 [0, 1]，综合可靠性评估
    source: str               # 不确定性来源标识 (e.g., "EIS_Rct_ensemble")
    
    # --- 诊断字段 ---
    fit_residual: float | None = None    # 拟合残差
    n_points_used: int | None = None     # 使用的数据点数
    method: str = "direct"               # 提取方法标识
    
    # --- ★ Agent Interface 字段 (v4 新增) ---
    metadata: dict[str, Any] = field(default_factory=dict)
    """供 Layer 3 Agent 消费的扩展上下文。
    
    Extractor 层写入:
      metadata["raw_features"] = {...}     # 原始特征 (e.g., Nyquist 半圆参数)
      metadata["quality_flags"] = [...]    # 质量标记 (e.g., "low_freq_noise")
      metadata["instrument_state"] = {...} # 仪器状态 (e.g., 电极阻抗变化趋势)
    
    Agent 层写入 (v6+ 才启用):
      metadata["anomaly_verdict"] = {...}  # AnomalyDetectionAgent 诊断结果
      metadata["agent_diagnosis"] = {...}  # SpectrumAnalysisAgent 解读
    
    Pragmatic 阶段: Extractor 写入 raw_features + quality_flags
    LLM 阶段: Agent 额外写入 diagnosis + verdict
    """
    
    @property
    def std(self) -> float:
        return self.variance ** 0.5
    
    @property
    def relative_uncertainty(self) -> float:
        """相对不确定性 (CV)。"""
        if abs(self.value) < 1e-12:
            return float('inf')
        return self.std / abs(self.value)
    
    @property
    def is_reliable(self) -> bool:
        """快速判断：这个测量是否可靠。
        
        AnomalyDetectionAgent 的简化版——
        Pragmatic 阶段用这个，LLM 阶段用 Agent 的 anomaly_verdict。
        """
        return self.confidence >= 0.5 and self.relative_uncertainty < 0.5


@dataclass
class ObservationWithNoise:
    """传给 BO 的观测值，携带异质噪声。"""
    
    objective_value: float    # 目标函数值
    noise_variance: float     # 该点的噪声方差 σ²_i
    
    # --- 来源追溯 ---
    kpi_contributions: list[dict] | None = None  # 各 KPI 的贡献和不确定性
    
    # --- ★ Agent Interface 字段 (v4 新增) ---
    metadata: dict[str, Any] = field(default_factory=dict)
    """供 Orchestrator / Agent 消费的汇总上下文。
    
    Propagator 自动写入:
      metadata["min_confidence"]    = 0.78   # 所有 KPI 中最低的 confidence
      metadata["unreliable_kpis"]   = [...]  # confidence < 0.5 的 KPI 列表
      metadata["uncertainty_budget"] = {...}  # 各 KPI 的方差贡献比例
    
    Agent 层写入 (v6+):
      metadata["anomaly_verdict"]   = {...}  # 整体异常判定
      metadata["agent_feedback"]    = {...}  # Agent 对 GP 的建议
    """
```

### 3.1 与 v6-v8 Agent 架构的接口契约

```python
# 以下是 Layer 3 Agent 消费 v4 数据类型的接口预览。
# Agent 实际实现在 v6-v8 阶段，此处仅定义 v4 需要满足的契约。

class AgentContext:
    """Layer 3 Agent 从 Layer 2 获取的上下文包。
    
    v4 提供的字段（本版本实现）:
    """
    measurements: list[MeasurementWithUncertainty]  # 本轮所有 KPI
    observation: ObservationWithNoise                # 传播后的观测
    gp_state: dict                                   # GP 模型状态
    history: list[ObservationWithNoise]              # 历史观测
    domain_config: dict                              # 领域配置 (from YAML)


class OptimizationFeedback:
    """Layer 3 Agent 返回给 Layer 2 的反馈。
    
    v4 阶段: 仅 noise_override 生效（Extractor 层自检）
    v6+ 阶段: Agent 可通过此接口调整 GP 行为
    """
    noise_override: float | None = None     # 覆盖传播链路的噪声估计
    prior_adjustment: dict | None = None    # 调整 GP 先验 (v7b)
    constraint_update: dict | None = None   # 更新搜索空间约束 (v6a)
    rerun_suggested: bool = False           # 建议重新实验 (v6c)
    rerun_reason: str = ""
```

---

## 4. Domain-Configurable Extractors

### 4.1 设计理念

Extractor 的核心算法（插值、拟合、积分）是 domain-agnostic 的。但仪器参数、物理约束、质量阈值是领域相关的。这些领域知识通过 YAML 注入，不硬编码在 Python 中。

```
                 ┌────────────────────────────────┐
                 │  domain_knowledge/              │
                 │  electrochemistry/              │
                 │    ├── config.yaml              │
                 │    │   instruments:             │
                 │    │     squidstat:              │
                 │    │       z_accuracy: 0.001     │
                 │    │       phase_accuracy: 0.1   │
                 │    │       freq_range: [0.1, 1e5]│
                 │    │   physical_constraints:     │
                 │    │     CE: {min: 0, max: 105}  │
                 │    │     Rct: {min: 0}           │
                 │    │   quality_thresholds:       │
                 │    │     confidence_min: 0.5     │
                 │    │     rel_uncertainty_max: 0.5│
                 │    └── spectrum_rules.py         │
                 │        (过程式规则, YAML 表达不了) │
                 └───────────────┬────────────────┘
                                 │ load_domain()
                                 ▼
                 ┌────────────────────────────────┐
                 │  Extractor (domain-agnostic)   │
                 │  核心算法: 插值、NLLS、积分      │
                 │  + domain config 参数化          │
                 └────────────────────────────────┘
```

### 4.2 Domain Config Schema

**文件：** `domain_knowledge/electrochemistry/instruments.yaml`

```yaml
# 仪器级参数——Extractor 直接消费
instruments:
  squidstat:
    model: "Squidstat Plus"
    impedance:
      z_relative_accuracy: 0.001      # |Z| ±0.1%
      phase_accuracy_deg: 0.1         # phase ±0.1°
      freq_range_hz: [0.1, 100000]
      low_freq_noise_amplification:    # 低频噪声放大系数
        below_1hz: 3.0
        below_10hz: 1.5
    dc:
      current_accuracy_a: 1.0e-7      # ±100 nA
      voltage_accuracy_v: 1.0e-4      # ±0.1 mV
      sampling_rate_hz: 100
      zero_drift_a_per_hour: 5.0e-7   # 零点漂移

  uvvis:
    model: "Ocean Optics"
    wavelength_range_nm: [200, 900]
    absorbance_noise:
      low_abs: 0.002                   # A < 0.5
      mid_abs: 0.005                   # 0.5 < A < 2.0
      high_abs: 0.02                   # A > 2.0 (信噪比急剧下降)
    linear_range_max: 2.5             # Beer-Lambert 线性范围上限

  xrd:
    model: "lab XRD"
    instrument_broadening_deg: 0.05
    two_theta_accuracy_deg: 0.02
    scherrer_k_range: [0.89, 0.94]
```

**文件：** `domain_knowledge/electrochemistry/physical_constraints.yaml`

```yaml
# 物理约束——Extractor 用于 confidence 评估
# 未来 AnomalyDetectionAgent 也消费这些约束
physical_constraints:
  CE:
    min: 0
    max: 105         # CE > 105% 物理上不合理
    unit: "%"
    warning_range: [95, 105]  # 接近上限时降低 confidence
    
  Rct:
    min: 0
    unit: "Ω"
    typical_range: [1, 10000]  # 超出范围降低 confidence
    
  z_magnitude:
    min: 0
    unit: "Ω"
    
  absorbance:
    min: -0.05       # 微小负值可能是基线问题
    max: 4.0         # 超过 4.0 无意义
    unit: "AU"
    negative_confidence_penalty: 0.3
    
  crystallite_size:
    min: 1            # < 1nm 无意义
    max: 1000         # > 1μm Scherrer 不适用
    unit: "nm"

quality_thresholds:
  confidence_min: 0.5          # 低于此值标记 unreliable
  relative_uncertainty_max: 0.5  # 高于此值标记 noisy
```

**文件：** `domain_knowledge/electrochemistry/eis_models.yaml`

```yaml
# EIS 等效电路配置——供 R_ct ensemble extractor 使用
# 未来 SpectrumAnalysisAgent 也消费这个配置
equivalent_circuits:
  - name: "randles"
    formula: "R_s + (R_ct || C_dl)"
    params: ["R_s", "R_ct", "C_dl"]
    rct_index: 1
    init_bounds:
      R_s: [0.1, 100]
      R_ct: [1, 50000]
      C_dl: [1.0e-8, 1.0e-3]
      
  - name: "randles_warburg"
    formula: "R_s + (R_ct || C_dl) + W"
    params: ["R_s", "R_ct", "C_dl", "W_s", "W_n"]
    rct_index: 1
    init_bounds:
      R_s: [0.1, 100]
      R_ct: [1, 50000]
      C_dl: [1.0e-8, 1.0e-3]
      W_s: [0.1, 1000]
      W_n: [0.3, 0.7]
      
  - name: "2rc"
    formula: "R_s + (R1 || C1) + (R2 || C2)"
    params: ["R_s", "R1", "C1", "R2", "C2"]
    rct_index: 1    # R1 interpreted as R_ct
    init_bounds:
      R_s: [0.1, 100]
      R1: [1, 50000]
      C1: [1.0e-8, 1.0e-3]
      R2: [1, 50000]
      C2: [1.0e-8, 1.0e-3]

model_selection:
  method: "aic_weighted_ensemble"
  min_models_for_ensemble: 2
  convergence_max_iter: 200
```

### 4.3 Domain Config Loader

```python
import yaml
from pathlib import Path
from importlib import import_module


class DomainConfig:
    """加载和管理领域配置。
    
    支持 YAML (声明式, ~80% 领域知识) 
    + Python rules (过程式, ~20% 领域知识)。
    
    同一份 config 被 Extractor (v4) 和 Agent (v6+) 共同消费。
    """
    
    def __init__(self, domain_name: str, config_dir: str = "domain_knowledge"):
        self.domain_name = domain_name
        self.base_path = Path(config_dir) / domain_name
        
        # 声明式知识: YAML 文件
        self.instruments = self._load_yaml("instruments.yaml")
        self.constraints = self._load_yaml("physical_constraints.yaml")
        self.eis_models = self._load_yaml("eis_models.yaml")
        
        # 过程式知识: Python 规则 (可选)
        self.rules = self._load_rules()
    
    def _load_yaml(self, filename: str) -> dict:
        path = self.base_path / filename
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_rules(self):
        """加载过程式规则（YAML 表达不了的复杂逻辑）。
        
        e.g., domain_knowledge/electrochemistry/spectrum_rules.py
        包含 interpret_eis_anomaly(), validate_cv_shape() 等纯函数。
        """
        try:
            module_path = f"domain_knowledge.{self.domain_name}.spectrum_rules"
            return import_module(module_path)
        except ImportError:
            return None
    
    def get_instrument_spec(self, instrument: str) -> dict:
        """获取仪器参数。Extractor 调用。"""
        return self.instruments.get("instruments", {}).get(instrument, {})
    
    def get_constraint(self, kpi_name: str) -> dict:
        """获取物理约束。Extractor + AnomalyDetectionAgent 调用。"""
        return self.constraints.get("physical_constraints", {}).get(kpi_name, {})
    
    def get_quality_thresholds(self) -> dict:
        """获取质量阈值。"""
        return self.constraints.get("quality_thresholds", {})
    
    def get_eis_circuits(self) -> list[dict]:
        """获取 EIS 等效电路配置。"""
        return self.eis_models.get("equivalent_circuits", [])
    
    def apply_rule(self, rule_name: str, **kwargs):
        """调用过程式规则。
        
        Pragmatic 阶段: Extractor 直接调用
        LLM 阶段: SpectrumAnalysisAgent 可选择调用或用 LLM 推理替代
        """
        if self.rules and hasattr(self.rules, rule_name):
            return getattr(self.rules, rule_name)(**kwargs)
        return None
```

### 4.4 过程式规则示例

**文件：** `domain_knowledge/electrochemistry/spectrum_rules.py`

```python
"""电化学领域的过程式规则。

这些规则太复杂、太条件化，YAML 表达不了。
但它们仍然是确定性的（不依赖 LLM）。

Extractor 在 Pragmatic 模式下直接调用这些函数。
SpectrumAnalysisAgent 在 LLM 模式下可选择调用或用 LLM 推理替代。
"""


def interpret_eis_anomaly(
    nyquist_features: dict, 
    history: list[dict]
) -> dict | None:
    """诊断 EIS 异常。
    
    YAML 无法表达的原因: 需要同时考虑 Nyquist 特征 + 历史趋势，
    且有多级条件分支。
    
    Args:
        nyquist_features: {"n_semicircles": int, "second_semicircle_freq": float, ...}
        history: 最近 5 轮的 KPI 历史
        
    Returns:
        diagnosis dict 或 None (无异常)
    """
    # 场景1: 新出现第二个半圆 + CE 突降
    if (nyquist_features.get("n_semicircles", 1) >= 2
        and nyquist_features.get("second_semicircle_freq", 999) < 1.0
        and len(history) >= 2
        and history[-1].get("CE", 100) - history[-2].get("CE", 100) < -10):
        return {
            "diagnosis": "SEI_layer_formation",
            "confidence": 0.7,
            "evidence": "second semicircle at low freq + CE drop >10%",
            "suggestion": "reduce current density",
            "severity": "warning"
        }
    
    # 场景2: R_ct 持续上升 → 钝化
    if len(history) >= 3:
        rct_trend = [h.get("Rct", 0) for h in history[-3:]]
        if all(rct_trend[i] < rct_trend[i+1] for i in range(len(rct_trend)-1)):
            pct_increase = (rct_trend[-1] - rct_trend[0]) / max(rct_trend[0], 1)
            if pct_increase > 0.5:
                return {
                    "diagnosis": "electrode_passivation",
                    "confidence": 0.6,
                    "evidence": f"Rct increased {pct_increase:.0%} over 3 rounds",
                    "suggestion": "check electrode surface, consider polishing",
                    "severity": "warning"
                }
    
    # 场景3: |Z| 全频段突然飙升 → 接触问题
    if nyquist_features.get("z_1khz_ratio_to_median", 1.0) > 3.0:
        return {
            "diagnosis": "contact_problem",
            "confidence": 0.8,
            "evidence": "|Z|@1kHz is 3x median → likely loose connection",
            "suggestion": "re-check electrode connections, flag for rerun",
            "severity": "critical"
        }
    
    return None


def validate_ce_physics(ce_value: float, ce_variance: float) -> dict:
    """CE 物理合理性校验。
    
    YAML 能表达 min/max，但不能表达:
    - CE 接近 100% 时的特殊处理
    - 结合 variance 的联合判断
    """
    result = {"valid": True, "confidence_modifier": 1.0, "flags": []}
    
    if ce_value > 100 and ce_value <= 105:
        # 微超 100% 可能是积分误差，不一定是异常
        if ce_variance < 1.0:
            # 低方差但超 100% → 可能是系统偏差
            result["flags"].append("systematic_bias_suspected")
            result["confidence_modifier"] = 0.7
        else:
            # 高方差且超 100% → 正常波动
            result["confidence_modifier"] = 0.9
    elif ce_value > 105:
        result["valid"] = False
        result["flags"].append("physically_impossible")
        result["confidence_modifier"] = 0.3
    elif ce_value < 30:
        result["flags"].append("abnormally_low")
        result["confidence_modifier"] = 0.5
    
    return result
```

---

## 5. Extractor 层：不确定性量化

### 5.1 Extractor 基类

```python
from abc import ABC, abstractmethod


class UncertaintyExtractor(ABC):
    """所有 Extractor 的基类。
    
    Extractor 是 domain-agnostic 的算法引擎，
    通过 DomainConfig 注入仪器参数和领域约束。
    
    Layer 3 的 SpectrumAnalysisAgent 包装 Extractor:
    - Pragmatic: 直接调用 Extractor + domain rules
    - LLM: 调用 Extractor + LLM 解读 metadata
    """
    
    def __init__(self, domain_config: DomainConfig):
        self.config = domain_config
    
    @abstractmethod
    def extract_with_uncertainty(
        self, raw_data: dict
    ) -> list[MeasurementWithUncertainty]:
        """从原始数据提取 KPI，带不确定性。
        
        Returns:
            列表——一个 Extractor 可能输出多个 KPI
            (e.g., EIS → |Z|@1kHz, R_ct, C_dl)
        """
        ...
    
    def _apply_physical_constraints(
        self, 
        measurement: MeasurementWithUncertainty,
        kpi_name: str
    ) -> MeasurementWithUncertainty:
        """用领域约束调整 confidence。从 YAML 读取阈值。"""
        constraint = self.config.get_constraint(kpi_name)
        if not constraint:
            return measurement
        
        confidence = measurement.confidence
        
        # 检查硬约束
        if "min" in constraint and measurement.value < constraint["min"]:
            confidence *= 0.3
            measurement.metadata["quality_flags"] = measurement.metadata.get(
                "quality_flags", []
            ) + [f"{kpi_name}_below_min"]
        
        if "max" in constraint and measurement.value > constraint["max"]:
            confidence *= 0.3
            measurement.metadata["quality_flags"] = measurement.metadata.get(
                "quality_flags", []
            ) + [f"{kpi_name}_above_max"]
        
        # 检查典型范围
        typical = constraint.get("typical_range")
        if typical and not (typical[0] <= measurement.value <= typical[1]):
            confidence *= 0.7
            measurement.metadata["quality_flags"] = measurement.metadata.get(
                "quality_flags", []
            ) + [f"{kpi_name}_outside_typical"]
        
        measurement.confidence = confidence
        return measurement
```

### 5.2 EIS Extractor

**文件：** `extractors/eis_uncertainty.py` (~350 行)

```python
class EISUncertaintyExtractor(UncertaintyExtractor):
    """EIS 测量的不确定性量化。
    
    四种不确定性来源：
    
    1. 仪器噪声 (instrument noise)
       - 从 domain config 读取 Squidstat spec
       - 频率越低，噪声越大（从 config 读取 amplification factor）
       - 从重复测量或仪器 spec 估计
    
    2. 拟合不确定性 (fitting uncertainty)
       - Equivalent circuit fitting 的参数标准差
       - 来自 Jacobian 矩阵: cov(θ) = σ² (J^T J)^{-1}
       - 适用于 R_ct, R_s 等拟合参数
    
    3. 模型选择不确定性 (model selection)
       - 不同 equivalent circuit 给出不同 R_ct
       - 从 domain config 读取 circuit 列表
       - 用多模型 ensemble 量化
    
    4. 实验变异性 (experimental variability)
       - 同一条件重复实验的变异
       - 从历史数据估计
    """
    
    def __init__(self, domain_config: DomainConfig):
        super().__init__(domain_config)
        self.instrument_spec = domain_config.get_instrument_spec("squidstat")
        self.eis_circuits = domain_config.get_eis_circuits()
    
    def extract_with_uncertainty(
        self, raw_data: dict
    ) -> list[MeasurementWithUncertainty]:
        """提取所有 EIS KPI。"""
        results = []
        
        z_real = raw_data["z_real"]
        z_imag = raw_data["z_imag"]
        freqs = raw_data["frequencies"]
        
        # KPI 1: |Z|@1kHz (最稳健)
        z_1khz = self.extract_impedance_at_freq(z_real, z_imag, freqs, 1000.0)
        results.append(z_1khz)
        
        # KPI 2: R_ct (ensemble)
        rct = self.extract_rct_with_uncertainty(z_real, z_imag, freqs)
        results.append(rct)
        
        return results
    
    def extract_impedance_at_freq(
        self,
        z_real: list[float],
        z_imag: list[float],
        frequencies: list[float],
        target_freq: float
    ) -> MeasurementWithUncertainty:
        """提取特定频率的 |Z|，带不确定性。
        
        这是最简单最稳健的 EIS KPI——不依赖 fitting。
        """
        # 1. 找到最近的频率点（或插值）
        z_at_freq, interp_uncertainty = self._interpolate(
            frequencies, z_real, z_imag, target_freq
        )
        
        # 2. 计算 |Z|
        z_magnitude = (z_at_freq[0]**2 + z_at_freq[1]**2) ** 0.5
        
        # 3. 仪器噪声估计 (从 domain config)
        z_accuracy = self.instrument_spec.get("impedance", {}).get(
            "z_relative_accuracy", 0.001
        )
        instrument_var = (z_magnitude * z_accuracy) ** 2
        
        # 低频噪声放大 (从 domain config)
        noise_amp = self.instrument_spec.get("impedance", {}).get(
            "low_freq_noise_amplification", {}
        )
        if target_freq < 1.0:
            instrument_var *= noise_amp.get("below_1hz", 3.0) ** 2
        elif target_freq < 10.0:
            instrument_var *= noise_amp.get("below_10hz", 1.5) ** 2
        
        # 4. 合并不确定性
        total_var = interp_uncertainty**2 + instrument_var
        
        # 5. 写入 metadata 供 Agent 消费
        nyquist_features = self._extract_nyquist_features(z_real, z_imag, frequencies)
        
        measurement = MeasurementWithUncertainty(
            value=z_magnitude,
            variance=total_var,
            confidence=self._compute_confidence(total_var, z_magnitude),
            source=f"EIS_|Z|@{target_freq:.0f}Hz",
            method="interpolation",
            metadata={
                "raw_features": nyquist_features,
                "quality_flags": [],
                "instrument_noise_var": instrument_var,
                "interp_uncertainty": interp_uncertainty,
            }
        )
        
        return self._apply_physical_constraints(measurement, "z_magnitude")
    
    def extract_rct_with_uncertainty(
        self,
        z_real: list[float],
        z_imag: list[float],
        frequencies: list[float]
    ) -> MeasurementWithUncertainty:
        """提取 R_ct，带拟合 + 模型选择不确定性。
        
        核心算法:
        1. 对每个 equivalent circuit (从 domain config)，用 NLLS 拟合
        2. 每个 fit 的 R_ct 有自己的 fitting uncertainty
        3. 多模型 ensemble: μ = weighted_mean(R_ct_i), σ² = inter + intra
        
        加权方式: AIC/BIC 权重
        w_i = exp(-0.5 * ΔAIC_i) / Σ exp(-0.5 * ΔAIC_j)
        """
        results = []
        
        # 从 domain config 读取电路配置 (不硬编码)
        circuits = self.eis_circuits
        if not circuits:
            # fallback: 默认 Randles
            circuits = [{"name": "randles", "params": ["R_s", "R_ct", "C_dl"], "rct_index": 1}]
        
        for circuit_cfg in circuits:
            circuit_name = circuit_cfg["name"]
            
            params, residual = self._fit_circuit(
                z_real, z_imag, frequencies, circuit_cfg
            )
            
            if params is None:
                continue
            
            # 拟合不确定性: cov(θ) = s² (J^T J)^{-1}
            jacobian = self._compute_jacobian(
                z_real, z_imag, frequencies, circuit_cfg, params
            )
            s_squared = residual / (2 * len(frequencies) - len(params))
            
            JtJ = self._matrix_multiply_transpose(jacobian)
            param_cov = self._matrix_inverse(JtJ)
            
            rct_idx = circuit_cfg["rct_index"]
            rct_value = params[rct_idx]
            rct_var = s_squared * param_cov[rct_idx][rct_idx]
            
            # AIC = n * ln(RSS/n) + 2k
            n = 2 * len(frequencies)
            k = len(params)
            aic = n * _log(residual / n) + 2 * k
            
            results.append({
                "circuit": circuit_name,
                "rct": rct_value,
                "rct_var": rct_var,
                "aic": aic,
                "residual": residual,
                "all_params": dict(zip(circuit_cfg["params"], params))
            })
        
        if not results:
            return MeasurementWithUncertainty(
                value=float('nan'), variance=float('inf'),
                confidence=0.0, source="EIS_Rct_fit_failed",
                metadata={"quality_flags": ["all_fits_failed"]}
            )
        
        # AIC 加权 ensemble
        min_aic = min(r["aic"] for r in results)
        weights = []
        for r in results:
            w = _exp(-0.5 * (r["aic"] - min_aic))
            weights.append(w)
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]
        
        rct_mean = sum(w * r["rct"] for w, r in zip(weights, results))
        
        # 总方差 = 模型内方差 + 模型间方差
        intra_var = sum(w * r["rct_var"] for w, r in zip(weights, results))
        inter_var = sum(w * (r["rct"] - rct_mean)**2 for w, r in zip(weights, results))
        total_var = intra_var + inter_var
        
        measurement = MeasurementWithUncertainty(
            value=rct_mean,
            variance=total_var,
            confidence=self._compute_confidence(total_var, rct_mean),
            source="EIS_Rct_ensemble",
            fit_residual=min(r["residual"] for r in results),
            method="multi_circuit_ensemble",
            metadata={
                "raw_features": {
                    "circuit_results": results,
                    "model_weights": dict(zip(
                        [r["circuit"] for r in results], weights
                    )),
                    "intra_model_var": intra_var,
                    "inter_model_var": inter_var,
                },
                "quality_flags": [],
            }
        )
        
        # 物理约束检查
        measurement = self._apply_physical_constraints(measurement, "Rct")
        
        # 过程式规则检查 (如果有)
        if self.config.rules:
            nyquist_features = measurement.metadata.get("raw_features", {})
            rule_result = self.config.apply_rule(
                "interpret_eis_anomaly", 
                nyquist_features=nyquist_features,
                history=[]  # 需要外部传入历史; v6 orchestrator 负责
            )
            if rule_result:
                measurement.metadata["rule_diagnosis"] = rule_result
                if rule_result.get("severity") == "critical":
                    measurement.confidence *= 0.3
        
        return measurement
    
    def _extract_nyquist_features(self, z_real, z_imag, frequencies) -> dict:
        """提取 Nyquist 图特征 → 写入 metadata。
        
        这些特征是 SpectrumAnalysisAgent 的输入。
        Pragmatic 阶段: 直接用 spectrum_rules.py 解读
        LLM 阶段: Agent 用这些特征 + LLM 推理
        """
        # 简化实现
        z_magnitudes = [(r**2 + i**2)**0.5 for r, i in zip(z_real, z_imag)]
        
        return {
            "n_points": len(frequencies),
            "freq_range": [min(frequencies), max(frequencies)],
            "z_real_range": [min(z_real), max(z_real)],
            "z_imag_range": [min(z_imag), max(z_imag)],
            "z_1khz_value": self._interp_at_freq(z_magnitudes, frequencies, 1000),
            "n_semicircles": self._count_semicircles(z_real, z_imag),
            # 更多特征由具体实现填充...
        }
    
    def _estimate_instrument_noise(self, z_real, z_imag, freqs, target_freq):
        """从 Nyquist 图的局部平滑度估计仪器噪声。
        
        原理: 真实阻抗谱是平滑的，高频震荡 = 噪声。
        方法: 局部多项式拟合的残差 → 噪声估计。
        """
        ...

    def _fit_circuit(self, z_real, z_imag, freqs, circuit_cfg):
        """Pure Python NLLS 拟合等效电路。
        
        从 circuit_cfg (domain config) 读取:
        - 参数名称和数量
        - 初始值范围
        - 电路拓扑
        
        使用 Levenberg-Marquardt 算法 (Pure Python)。
        """
        ...
    
    def _compute_jacobian(self, z_real, z_imag, freqs, circuit_cfg, params):
        """数值 Jacobian。"""
        ...
    
    def _matrix_multiply_transpose(self, J):
        """J^T J。"""
        ...
    
    def _matrix_inverse(self, M):
        """矩阵求逆 (小矩阵，Gauss-Jordan)。"""
        ...
    
    def _compute_confidence(self, variance, value):
        """基于相对不确定性计算 confidence [0, 1]。"""
        if abs(value) < 1e-12:
            return 0.0
        ru = (variance ** 0.5) / abs(value)
        # sigmoid-like mapping: ru=0 → conf=1, ru=0.5 → conf≈0.5, ru>1 → conf→0
        return max(0.0, min(1.0, 1.0 - ru))
```

### 5.3 DC Cycling Extractor

**文件：** `extractors/dc_uncertainty.py` (~150 行)

```python
class DCCyclingUncertaintyExtractor(UncertaintyExtractor):
    """库仑效率 (CE) 的不确定性量化。"""
    
    def __init__(self, domain_config: DomainConfig):
        super().__init__(domain_config)
        self.dc_spec = domain_config.get_instrument_spec("squidstat").get("dc", {})
    
    def extract_with_uncertainty(
        self, raw_data: dict
    ) -> list[MeasurementWithUncertainty]:
        return [self.extract_ce_with_uncertainty(
            raw_data["current"], raw_data["voltage"], raw_data["time"]
        )]
    
    def extract_ce_with_uncertainty(
        self,
        current: list[float],
        voltage: list[float],
        time: list[float]
    ) -> MeasurementWithUncertainty:
        """CE 不确定性来源：
        
        1. 电流积分误差
           - 采样率有限 → 梯形积分截断误差
           - 估计: O(h² * max|f''|) where h = sampling interval
        
        2. 电流零点偏移 (从 domain config 读取)
           - Squidstat 零点漂移
           - 对长时间实验影响大
        
        3. 截止电压判断
           - 沉积/溶解切换点的判断误差
        
        4. 气泡/接触问题
           - 异常大的方差 → 置信度 flag
        """
        # 1. 分段积分
        q_deposit, q_deposit_var = self._integrate_with_uncertainty(
            current, time, phase="deposition"
        )
        q_dissolve, q_dissolve_var = self._integrate_with_uncertainty(
            current, time, phase="dissolution"
        )
        
        # 2. CE = Q_dissolve / Q_deposit
        ce = abs(q_dissolve / q_deposit) * 100
        
        # 3. 误差传播: CE = f(Q_d, Q_s)
        rel_var = (q_deposit_var / q_deposit**2) + (q_dissolve_var / q_dissolve**2)
        ce_var = ce**2 * rel_var
        
        # 4. 零点漂移贡献 (从 domain config)
        zero_drift = self.dc_spec.get("zero_drift_a_per_hour", 5e-7)
        t_total = time[-1] - time[0]  # seconds
        drift_charge = zero_drift * (t_total / 3600) * t_total
        drift_var = (drift_charge / max(abs(q_deposit), 1e-12) * 100) ** 2
        ce_var += drift_var
        
        # 5. 物理约束 + 过程式规则
        confidence = self._compute_confidence(ce_var, ce)
        
        if self.config.rules:
            physics_check = self.config.apply_rule(
                "validate_ce_physics", ce_value=ce, ce_variance=ce_var
            )
            if physics_check:
                confidence *= physics_check.get("confidence_modifier", 1.0)
                quality_flags = physics_check.get("flags", [])
            else:
                quality_flags = []
        else:
            quality_flags = []
            if ce > 105:
                confidence *= 0.3
                quality_flags.append("CE_above_105")
        
        measurement = MeasurementWithUncertainty(
            value=ce,
            variance=ce_var,
            confidence=confidence,
            source="DC_CE",
            n_points_used=len(current),
            method="trapezoid_integration",
            metadata={
                "raw_features": {
                    "q_deposit": q_deposit,
                    "q_dissolve": q_dissolve,
                    "n_cycles": raw_data.get("n_cycles", 1),
                    "t_total_s": t_total,
                },
                "quality_flags": quality_flags,
                "drift_contribution_var": drift_var,
            }
        )
        
        return self._apply_physical_constraints(measurement, "CE")
    
    def _integrate_with_uncertainty(self, current, time, phase):
        """梯形积分 + 截断误差估计。"""
        ...
    
    def _compute_confidence(self, variance, value):
        if abs(value) < 1e-12:
            return 0.0
        ru = (variance ** 0.5) / abs(value)
        return max(0.0, min(1.0, 1.0 - ru))
```

### 5.4 UV-Vis Extractor

**文件：** `extractors/uvvis_uncertainty.py` (~120 行)

```python
class UVVisUncertaintyExtractor(UncertaintyExtractor):
    """UV-Vis 光谱 KPI 的不确定性量化。"""
    
    def __init__(self, domain_config: DomainConfig):
        super().__init__(domain_config)
        self.uvvis_spec = domain_config.get_instrument_spec("uvvis")
    
    def extract_with_uncertainty(
        self, raw_data: dict
    ) -> list[MeasurementWithUncertainty]:
        results = []
        if "target_wavelength" in raw_data:
            results.append(self.extract_absorbance_with_uncertainty(
                raw_data["wavelengths"],
                raw_data["absorbance"],
                raw_data["target_wavelength"],
                raw_data.get("baseline")
            ))
        return results
    
    def extract_absorbance_with_uncertainty(
        self,
        wavelengths: list[float],
        absorbance: list[float],
        target_wavelength: float,
        baseline: list[float] | None = None
    ) -> MeasurementWithUncertainty:
        """吸光度提取的不确定性来源：
        
        1. 仪器噪声 (从 domain config, 分三段)
        2. 基线校正不确定性
        3. Beer-Lambert 线性范围 (从 domain config)
        """
        # 1. 插值到目标波长
        abs_value, interp_var = self._interpolate(
            wavelengths, absorbance, target_wavelength
        )
        
        # 2. 仪器噪声 (分段, 从 config)
        noise_cfg = self.uvvis_spec.get("absorbance_noise", {})
        if abs_value < 0.5:
            inst_noise = noise_cfg.get("low_abs", 0.002)
        elif abs_value < 2.0:
            inst_noise = noise_cfg.get("mid_abs", 0.005)
        else:
            inst_noise = noise_cfg.get("high_abs", 0.02)
        
        total_var = interp_var + inst_noise**2
        
        # 3. Beer-Lambert 范围检查
        quality_flags = []
        confidence = self._compute_confidence(total_var, abs_value)
        linear_max = self.uvvis_spec.get("linear_range_max", 2.5)
        if abs_value > linear_max:
            confidence *= 0.5
            quality_flags.append("above_linear_range")
        
        measurement = MeasurementWithUncertainty(
            value=abs_value,
            variance=total_var,
            confidence=confidence,
            source=f"UVVis_A@{target_wavelength:.0f}nm",
            method="interpolation",
            metadata={
                "raw_features": {
                    "wavelength": target_wavelength,
                    "absorbance": abs_value,
                    "linear_range_max": linear_max,
                },
                "quality_flags": quality_flags,
            }
        )
        
        return self._apply_physical_constraints(measurement, "absorbance")
```

### 5.5 XRD Extractor

**文件：** `extractors/xrd_uncertainty.py` (~120 行)

```python
class XRDUncertaintyExtractor(UncertaintyExtractor):
    """XRD 峰位、晶粒尺寸的不确定性量化。"""
    
    def __init__(self, domain_config: DomainConfig):
        super().__init__(domain_config)
        self.xrd_spec = domain_config.get_instrument_spec("xrd")
    
    def extract_crystallite_size_with_uncertainty(
        self,
        two_theta: list[float],
        intensity: list[float],
        peak_position: float
    ) -> MeasurementWithUncertainty:
        """Scherrer 公式: D = Kλ / (β cosθ)
        
        不确定性来源 (参数从 domain config):
        1. 峰宽 β 的拟合误差（Voigt fitting）
        2. 仪器展宽校正 (from config: instrument_broadening_deg)
        3. Scherrer 常数 K 的不确定性 (from config: scherrer_k_range)
        4. 各向异性时，不同 (hkl) 给出不同 D
        """
        inst_broad = self.xrd_spec.get("instrument_broadening_deg", 0.05)
        k_range = self.xrd_spec.get("scherrer_k_range", [0.89, 0.94])
        k_mean = (k_range[0] + k_range[1]) / 2
        k_var = ((k_range[1] - k_range[0]) / 4) ** 2  # ~uniform → σ ≈ range/4
        
        # 峰拟合 → β + fitting uncertainty
        beta, beta_var = self._fit_peak(two_theta, intensity, peak_position)
        
        # 仪器展宽校正: β_sample = sqrt(β_obs² - β_inst²)
        beta_obs_sq = beta**2
        beta_inst_sq = inst_broad**2
        if beta_obs_sq <= beta_inst_sq:
            # 峰宽小于仪器展宽 → 无法提取晶粒尺寸
            return MeasurementWithUncertainty(
                value=float('nan'), variance=float('inf'),
                confidence=0.0, source="XRD_crystallite",
                metadata={"quality_flags": ["peak_narrower_than_instrument"]}
            )
        
        beta_sample = (beta_obs_sq - beta_inst_sq) ** 0.5
        
        # Scherrer
        wavelength = 1.5406  # Cu Kα, Å
        theta_rad = peak_position / 2 * 3.14159 / 180
        D = k_mean * wavelength / (beta_sample * _cos(theta_rad))
        
        # 误差传播: δD/D = sqrt((δK/K)² + (δβ/β)²)
        rel_var_k = k_var / k_mean**2
        rel_var_beta = beta_var / beta**2
        D_var = D**2 * (rel_var_k + rel_var_beta)
        
        measurement = MeasurementWithUncertainty(
            value=D * 10,  # Å → nm
            variance=D_var * 100,  # Å² → nm²
            confidence=self._compute_confidence(D_var * 100, D * 10),
            source="XRD_crystallite",
            method="scherrer",
            metadata={
                "raw_features": {
                    "peak_position_2theta": peak_position,
                    "beta_obs_deg": beta,
                    "beta_sample_deg": beta_sample,
                    "scherrer_k": k_mean,
                },
                "quality_flags": [],
            }
        )
        
        return self._apply_physical_constraints(measurement, "crystallite_size")
```

---

## 6. 传播层：KPI → Objective

**文件：** `uncertainty/propagation.py` (~250 行)

```python
class UncertaintyPropagator:
    """多 KPI 到单目标函数的不确定性传播。
    
    场景：objective = f(CE, Rct, crystallite_size, ...)
    需要把各 KPI 的不确定性传播到 objective 的不确定性。
    
    同时计算 metadata 汇总，供 Agent / Orchestrator 消费。
    """
    
    def linear_propagation(
        self,
        kpi_measurements: list[MeasurementWithUncertainty],
        weights: list[float]
    ) -> ObservationWithNoise:
        """线性组合: objective = Σ w_i * kpi_i
        
        σ²_obj = Σ w_i² * σ²_i
        """
        obj_value = sum(w * m.value for w, m in zip(weights, kpi_measurements))
        obj_var = sum(w**2 * m.variance for w, m in zip(weights, kpi_measurements))
        
        contributions = [
            {"name": m.source, "value": m.value, 
             "weight": w, "var_contribution": w**2 * m.variance,
             "var_fraction": (w**2 * m.variance) / max(obj_var, 1e-20)}
            for w, m in zip(weights, kpi_measurements)
        ]
        
        # ★ 汇总 metadata 供 Agent 消费
        meta = self._aggregate_metadata(kpi_measurements, contributions)
        
        return ObservationWithNoise(
            objective_value=obj_value,
            noise_variance=obj_var,
            kpi_contributions=contributions,
            metadata=meta
        )
    
    def nonlinear_propagation(
        self,
        kpi_measurements: list[MeasurementWithUncertainty],
        objective_func: callable,
        jacobian_func: callable | None = None
    ) -> ObservationWithNoise:
        """非线性函数的 delta-method 传播。
        
        σ²_obj ≈ J^T Σ_kpi J
        """
        values = [m.value for m in kpi_measurements]
        obj_value = objective_func(*values)
        
        if jacobian_func is None:
            jacobian = self._numerical_jacobian(objective_func, values)
        else:
            jacobian = jacobian_func(*values)
        
        obj_var = sum(
            jacobian[i]**2 * kpi_measurements[i].variance
            for i in range(len(kpi_measurements))
        )
        
        contributions = [
            {"name": m.source, "value": m.value,
             "jacobian": jacobian[i],
             "var_contribution": jacobian[i]**2 * m.variance,
             "var_fraction": (jacobian[i]**2 * m.variance) / max(obj_var, 1e-20)}
            for i, m in enumerate(kpi_measurements)
        ]
        
        meta = self._aggregate_metadata(kpi_measurements, contributions)
        
        return ObservationWithNoise(
            objective_value=obj_value,
            noise_variance=obj_var,
            kpi_contributions=contributions,
            metadata=meta
        )
    
    def monte_carlo_propagation(
        self,
        kpi_measurements: list[MeasurementWithUncertainty],
        objective_func: callable,
        n_samples: int = 10000
    ) -> ObservationWithNoise:
        """Monte Carlo 不确定性传播。"""
        import random as _rnd
        
        samples = []
        for _ in range(n_samples):
            kpi_sample = [
                _rnd.gauss(m.value, m.std) for m in kpi_measurements
            ]
            try:
                obj_sample = objective_func(*kpi_sample)
                samples.append(obj_sample)
            except (ValueError, ZeroDivisionError):
                continue
        
        obj_mean = sum(samples) / len(samples)
        obj_var = sum((s - obj_mean)**2 for s in samples) / (len(samples) - 1)
        
        contributions = [
            {"name": m.source, "value": m.value, "variance": m.variance}
            for m in kpi_measurements
        ]
        
        meta = self._aggregate_metadata(kpi_measurements, contributions)
        
        return ObservationWithNoise(
            objective_value=obj_mean,
            noise_variance=obj_var,
            kpi_contributions=contributions,
            metadata=meta
        )
    
    def _aggregate_metadata(
        self, 
        measurements: list[MeasurementWithUncertainty],
        contributions: list[dict]
    ) -> dict:
        """汇总所有 KPI 的 metadata → ObservationWithNoise.metadata。
        
        这是 AnomalyDetectionAgent 和 Orchestrator 的快速入口。
        """
        confidences = [m.confidence for m in measurements]
        
        unreliable = [
            m.source for m in measurements if not m.is_reliable
        ]
        
        all_flags = []
        for m in measurements:
            all_flags.extend(m.metadata.get("quality_flags", []))
        
        # 不确定性预算 (哪个 KPI 贡献了最多的方差)
        budget = {}
        for c in contributions:
            frac = c.get("var_fraction", 0)
            budget[c["name"]] = round(frac, 4)
        
        return {
            "min_confidence": min(confidences) if confidences else 0.0,
            "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "unreliable_kpis": unreliable,
            "all_quality_flags": all_flags,
            "uncertainty_budget": budget,
            # v6+ Agent 会额外写入:
            # "anomaly_verdict": {...},
            # "agent_feedback": {...},
        }
    
    def _numerical_jacobian(self, func, values, eps=1e-6):
        """数值 Jacobian (中心差分)。"""
        n = len(values)
        jac = []
        for i in range(n):
            v_plus = values.copy()
            v_minus = values.copy()
            v_plus[i] += eps
            v_minus[i] -= eps
            jac.append((func(*v_plus) - func(*v_minus)) / (2 * eps))
        return jac
```

---

## 7. GP 噪声模型集成

**文件：** `backends/gp_heteroscedastic.py` (~300 行)

### 7.1 异质噪声 GP

```python
class HeteroscedasticGP:
    """支持每个观测点不同噪声方差的 GP。
    
    标准 GP: y_i = f(x_i) + ε, ε ~ N(0, σ²)      ← 同质噪声
    异质 GP: y_i = f(x_i) + ε_i, ε_i ~ N(0, σ²_i)  ← v4 需要的
    
    关键修改：
    - K_y = K(X, X) + diag(σ²_1, ..., σ²_n)
      而不是 K_y = K(X, X) + σ² I
    """
    
    def __init__(self, kernel='matern52', lengthscales=None):
        self.kernel_type = kernel
        self.lengthscales = lengthscales
        self._X = []
        self._y = []
        self._noise_vars = []
        self._metadata = []       # ★ v4 新增: 存储每个观测的 metadata
    
    def observe(
        self, 
        x: list[float], 
        y: float, 
        noise_var: float,
        metadata: dict | None = None   # ★ v4 新增
    ):
        """添加带噪声方差的观测。
        
        Args:
            x: 输入参数
            y: 观测值
            noise_var: 该观测的噪声方差 σ²_i
            metadata: 来自 ObservationWithNoise.metadata 的上下文
                      v4 阶段: 存储 uncertainty_budget, quality_flags
                      v6+ 阶段: 存储 anomaly_verdict, agent_feedback
        """
        self._X.append(x)
        self._y.append(y)
        self._noise_vars.append(noise_var)
        self._metadata.append(metadata or {})
    
    def predict(self, x_new: list[float]) -> tuple[float, float]:
        """预测均值和方差。
        
        μ(x*) = k(x*, X) [K(X,X) + diag(σ²)]^{-1} y
        σ²(x*) = k(x*,x*) - k(x*,X) [K(X,X) + diag(σ²)]^{-1} k(X,x*)
        """
        n = len(self._X)
        
        K = self._compute_kernel_matrix(self._X, self._X)
        
        # 关键：异质噪声对角矩阵
        for i in range(n):
            K[i][i] += self._noise_vars[i]
        
        L = self._cholesky(K)
        alpha = self._cholesky_solve(L, self._y)
        
        k_star = [self._kernel(x_new, xi) for xi in self._X]
        
        mu = sum(k_star[i] * alpha[i] for i in range(n))
        
        v = self._cholesky_solve_vec(L, k_star)
        k_star_star = self._kernel(x_new, x_new)
        var = k_star_star - sum(k_star[i] * v[i] for i in range(n))
        
        return mu, max(var, 1e-10)
    
    def get_model_state(self) -> dict:
        """导出 GP 状态供 Layer 3 Agent 消费。
        
        AgentContext.gp_state 的数据来源。
        """
        return {
            "n_observations": len(self._X),
            "X": [x[:] for x in self._X],
            "y": self._y[:],
            "noise_vars": self._noise_vars[:],
            "metadata": self._metadata[:],
            "kernel_type": self.kernel_type,
            "lengthscales": self.lengthscales,
        }
    
    def compute_noise_impact(self) -> list[dict]:
        """诊断：各观测点的噪声对模型的影响。
        
        返回每个点的 "effective weight"——噪声越大，权重越低。
        用途:
        - 可视化 (v4)
        - AnomalyDetectionAgent 的输入 (v6c)
        """
        n = len(self._X)
        K = self._compute_kernel_matrix(self._X, self._X)
        
        # 同质噪声下的权重
        K_homo = [row[:] for row in K]
        avg_noise = sum(self._noise_vars) / n
        for i in range(n):
            K_homo[i][i] += avg_noise
        
        # 异质噪声下的权重
        K_hetero = [row[:] for row in K]
        for i in range(n):
            K_hetero[i][i] += self._noise_vars[i]
        
        alpha_homo = self._solve(K_homo, self._y)
        alpha_hetero = self._solve(K_hetero, self._y)
        
        return [
            {
                "index": i,
                "noise_var": self._noise_vars[i],
                "weight_homo": abs(alpha_homo[i]),
                "weight_hetero": abs(alpha_hetero[i]),
                "weight_ratio": abs(alpha_hetero[i]) / max(abs(alpha_homo[i]), 1e-12),
                "metadata": self._metadata[i],  # ★ 传递 metadata
            }
            for i in range(n)
        ]
    
    # --- 内核实现 (与原版相同) ---
    
    def _kernel(self, x1, x2):
        """Matérn 5/2 kernel."""
        ...
    
    def _compute_kernel_matrix(self, X1, X2):
        ...
    
    def _cholesky(self, K):
        ...
    
    def _cholesky_solve(self, L, b):
        ...
    
    def _cholesky_solve_vec(self, L, v):
        ...
    
    def _solve(self, A, b):
        ...
```

### 7.2 与 OptimizationEngine 的接线

```python
# engine.py 修改点

class OptimizationEngine:
    """v4 修改: observe() 接受 noise_var + metadata。"""
    
    def __init__(self, domain_config: DomainConfig, ...):
        self.domain_config = domain_config
        self.extractors = self._build_extractors(domain_config)
        self.propagator = UncertaintyPropagator()
        # ...
    
    def _build_extractors(self, config: DomainConfig) -> list[UncertaintyExtractor]:
        """根据 domain config 构建 extractor 列表。
        
        哪些 extractor 启用由 domain config 的 instruments 段决定。
        """
        extractors = []
        instruments = config.instruments.get("instruments", {})
        
        if "squidstat" in instruments:
            extractors.append(EISUncertaintyExtractor(config))
            extractors.append(DCCyclingUncertaintyExtractor(config))
        if "uvvis" in instruments:
            extractors.append(UVVisUncertaintyExtractor(config))
        if "xrd" in instruments:
            extractors.append(XRDUncertaintyExtractor(config))
        
        return extractors
    
    def run_iteration(self):
        # 1. Hardware 执行实验
        raw_data = self.hardware.execute(params)
        
        # 2. ★ v4: 带不确定性的 KPI 提取
        all_measurements = []
        for extractor in self.extractors:
            measurements = extractor.extract_with_uncertainty(raw_data)
            all_measurements.extend(measurements)
        
        # 3. ★ v4: 不确定性传播
        observation = self.propagator.linear_propagation(
            all_measurements,
            self.objective_weights
        )
        
        # 4. ★ v4: observe 带噪声方差 + metadata
        self.backend.observe(
            x=params,
            y=observation.objective_value,
            noise_var=observation.noise_variance,
            metadata=observation.metadata          # ← v4 新增
        )
        
        # 5. ★ v6+ 预留: Agent 消费点
        # (v4 阶段此处为空; v6 启用后由 Orchestrator 接管)
        # agent_context = AgentContext(
        #     measurements=all_measurements,
        #     observation=observation,
        #     gp_state=self.backend.get_model_state(),
        #     history=self._history,
        #     domain_config=self.domain_config,
        # )
        # self.orchestrator.on_observation(agent_context)
        
        # 6. Suggest next (不变)
        next_params = self.backend.suggest()
        return next_params
```

---

## 8. Agent Interface Hooks

> 本节定义 v4 为 Layer 3 Agent 预留的接口点。
> v4 阶段这些接口不被调用（无 Agent 代码），但数据结构已就位。
> v6c (AnomalyDetection) 和 v7b (Explainability) 是第一批消费者。

### 8.1 接口总览

```
v4 提供给 Layer 3 的数据:
┌──────────────────────────────────────────────────────┐
│                                                      │
│  MeasurementWithUncertainty.metadata                 │
│  ├── raw_features: {...}    ← SpectrumAnalysisAgent  │
│  ├── quality_flags: [...]   ← AnomalyDetectionAgent  │
│  └── rule_diagnosis: {...}  ← 过程式规则的初步诊断    │
│                                                      │
│  ObservationWithNoise.metadata                       │
│  ├── min_confidence         ← 快速异常筛查            │
│  ├── unreliable_kpis        ← 需要关注的 KPI          │
│  └── uncertainty_budget     ← SymRegAgent 分析        │
│                                                      │
│  GP.get_model_state()                                │
│  ├── X, y, noise_vars       ← HypothesisAgent        │
│  └── metadata per point     ← 全链路追溯              │
│                                                      │
│  DomainConfig                                        │
│  ├── instruments.yaml       ← 仪器参数               │
│  ├── physical_constraints   ← 异常阈值               │
│  └── spectrum_rules.py      ← 过程式规则              │
│                                                      │
└──────────────────────────────────────────────────────┘

v6+ Agent 写回 Layer 2 的反馈:
┌──────────────────────────────────────────────────────┐
│  OptimizationFeedback                                │
│  ├── noise_override         → GP 噪声调整            │
│  ├── prior_adjustment       → GP 先验调整 (v7b)      │
│  ├── constraint_update      → 搜索空间约束 (v6a)     │
│  └── rerun_suggested        → 建议重新实验 (v6c)     │
└──────────────────────────────────────────────────────┘
```

### 8.2 SpectrumAnalysisAgent 如何消费 v4 数据

```python
# ★ 这是 v6+ 的代码预览，v4 不实现
# 展示 v4 的数据类型如何被 Agent 消费

class SpectrumAnalysisAgent:
    """感知层 Agent: '这个数据长什么样，质量如何'
    
    消费 v4 的 MeasurementWithUncertainty:
    - Pragmatic: 调用 Extractor（已在 v4 实现）+ domain rules
    - LLM: 调用 Extractor + LLM 解读 metadata.raw_features
    """
    
    def analyze(self, context: AgentContext) -> dict:
        # v4 已经提供了所有需要的数据
        for m in context.measurements:
            
            # 1. raw_features 包含 Nyquist 特征、电荷量、吸光度等
            features = m.metadata.get("raw_features", {})
            
            # 2. quality_flags 包含初步质量问题
            flags = m.metadata.get("quality_flags", [])
            
            # 3. rule_diagnosis 包含过程式规则的诊断 (如果有)
            rule_diag = m.metadata.get("rule_diagnosis")
            
            # Pragmatic: 直接使用以上信息
            # LLM: 将 features 描述给 LLM，获得更深层的解读
            
        return {"spectrum_quality": ..., "feature_summary": ...}
```

### 8.3 AnomalyDetectionAgent 如何消费 v4 数据

```python
# ★ v6c 的代码预览

class AnomalyDetectionAgent:
    """消费 v4 的 UQ 数据进行三层异常检测。
    
    Layer 2 异常检测 (确定性, v6c 实现):
      1. Signal-level: 从 MeasurementWithUncertainty.metadata.raw_features
      2. KPI-level: 从 MeasurementWithUncertainty.confidence + quality_flags
      3. GP-level: 从 GP.get_model_state() + ObservationWithNoise.noise_variance
    
    Layer 3 异常诊断 (Agent):
      Chain: Anomaly → SpectrumAnalysisAgent → HypothesisAgent
    """
    
    def detect(self, context: AgentContext) -> dict:
        anomalies = []
        
        # 1. Signal-level: v4 metadata 已包含
        for m in context.measurements:
            if not m.is_reliable:
                anomalies.append({
                    "level": "kpi",
                    "source": m.source,
                    "confidence": m.confidence,
                    "flags": m.metadata.get("quality_flags", []),
                })
        
        # 2. GP-level: 利用 v4 的 GP state
        gp = context.gp_state
        # Mahalanobis distance, LOO residuals, entropy change...
        
        # 3. Aggregate
        observation_meta = context.observation.metadata
        if observation_meta.get("min_confidence", 1.0) < 0.5:
            anomalies.append({
                "level": "observation",
                "detail": f"unreliable KPIs: {observation_meta.get('unreliable_kpis')}",
            })
        
        return {"anomalies": anomalies, "verdict": ...}
```

### 8.4 为什么 v4 要做这些预留

不做预留的代价：v6c 需要 Extractor 返回 raw features，但 v4 的 Extractor 只返回 float → 要么重构 Extractor（破坏接口），要么在 Agent 层重新解析原始数据（重复计算）。

做了预留（+~30 行 metadata 相关代码），v6c 直接消费 metadata，零重构。

---

## 9. 可视化增补

**文件：** `visualization/uncertainty_flow.py` (~200 行)

```python
def plot_uncertainty_budget(
    kpi_measurements: list[MeasurementWithUncertainty],
    observation: ObservationWithNoise
) -> PlotData:
    """不确定性预算图——各 KPI 对总不确定性的贡献。
    
    数据来源: ObservationWithNoise.metadata["uncertainty_budget"]
    
    示例输出:
    ┌────────────────────────────────────────┐
    │ Uncertainty Budget                     │
    │                                        │
    │ CE (35%)     ████████                  │
    │ Rct (52%)    █████████████             │ ← 瓶颈！
    │ |Z|@1kHz (8%)██                        │
    │ Other (5%)   █                         │
    └────────────────────────────────────────┘
    """

def plot_noise_impact_on_gp(
    gp_model: 'HeteroscedasticGP'
) -> PlotData:
    """异质噪声对 GP 的影响可视化。
    
    散点图叠加:
    - 点大小 ∝ 1/σ²_i (噪声小的点更大 = 更有影响力)
    - 颜色: 观测值
    - 误差线: ±2σ_i
    - GP 均值曲面 + 置信带
    """

def plot_measurement_reliability_timeline(
    history: list[list[MeasurementWithUncertainty]]
) -> PlotData:
    """测量可靠性时间线——跟踪 extractor 的健康状况。
    
    横轴: 实验序号
    纵轴: 各 KPI 的 relative_uncertainty
    警告线: confidence < 0.5 的点标红
    
    用途: 发现仪器漂移、传感器退化
    v6c 的 drift detection 会消费这条时间线
    """

def plot_confidence_heatmap(
    history: list[list[MeasurementWithUncertainty]],
    kpi_names: list[str]
) -> PlotData:
    """信度热力图——所有 KPI × 所有实验的 confidence 矩阵。
    
    快速发现: 哪些实验的哪些 KPI 不可靠。
    AnomalyDetectionAgent 的可视化入口。
    """
```

---

## 10. 锌电沉积实例走查

### 10.1 Domain Config 加载

```python
# 启动时
config = DomainConfig("electrochemistry")

# config 包含:
# - instruments.yaml → squidstat 精度参数
# - physical_constraints.yaml → CE [0, 105], Rct > 0 等
# - eis_models.yaml → randles, randles_warburg, 2rc
# - spectrum_rules.py → interpret_eis_anomaly(), validate_ce_physics()
```

### 10.2 完整数据流

```
实验 #42: Additive_A=15%, Additive_B=30%, ZnSO4_conc=1.0M

1. Opentrons 配液
2. Squidstat 执行:
   a. OCV → EIS (10kHz→1Hz)     → Nyquist_initial
   b. 沉积 (−4mA, 3s)
   c. OCV → EIS (10kHz→1Hz)     → Nyquist_after_deposition
   d. 溶解 (+4mA, 3s)
   e. OCV → EIS (10kHz→1Hz)     → Nyquist_after_dissolution
   f. 重复 b-e × 20 cycles

3. KPI 提取 (with uncertainty + metadata):
   ┌────────────────────────────────────────────────────────────────────┐
   │ KPI          │ Value  │ σ     │ Confidence │ Method    │ metadata │
   ├──────────────┼────────┼───────┼────────────┼───────────┼──────────┤
   │ CE           │ 95.3%  │ 1.2%  │ 0.95       │ trapezoid │ q_dep,   │
   │              │        │       │            │           │ q_dis,   │
   │              │        │       │            │           │ drift_var│
   │ |Z|@1kHz     │ 12.3 Ω │ 0.4 Ω │ 0.92       │ interp    │ nyquist  │
   │              │        │       │            │           │ features │
   │ R_ct         │ 245 Ω  │ 38 Ω  │ 0.78       │ ensemble  │ circuit  │
   │              │        │       │            │ (3 models)│ weights, │
   │              │        │       │            │           │ inter/   │
   │              │        │       │            │           │ intra var│
   └──────────────┴────────┴───────┴────────────┴───────────┴──────────┘

4. 不确定性传播:
   objective = 0.6 * normalized_CE + 0.4 * normalized_Z
   σ²_obj = 0.000271
   
   metadata = {
     "min_confidence": 0.78,        # R_ct 是瓶颈
     "unreliable_kpis": [],          # 都 >0.5，无 unreliable
     "uncertainty_budget": {
       "DC_CE": 0.76,               # CE 贡献 76% 方差
       "EIS_|Z|@1kHz": 0.24         # |Z| 贡献 24%
     }
   }

5. GP 更新:
   gp.observe(
       x=[0.15, 0.30, 1.0],
       y=0.823,
       noise_var=0.000271,
       metadata={...}               # ★ v4 新增
   )
   
   对比: 实验 #13（EIS 质量差，CE 有气泡）
   gp.observe(
       x=[0.05, 0.45, 0.8],
       y=0.621,
       noise_var=0.0085,             # 噪声方差 30x 大！
       metadata={
         "min_confidence": 0.35,
         "unreliable_kpis": ["EIS_Rct_ensemble"],
         "all_quality_flags": ["contact_problem", "CE_bubble"]
       }
   )
   → GP 自动降低实验 #13 的权重

6. (v6+ 才启用) Agent 消费:
   # AnomalyDetectionAgent 看到实验 #13 的 min_confidence=0.35
   # → 触发 SpectrumAnalysisAgent 检查 Nyquist features
   # → 触发 HypothesisAgent 诊断原因
   # → 建议 rerun 或标记为 outlier
```

### 10.3 与 NIMO/PHYSBO 的接口

```python
# 你的现有 NIMO pipeline 中，PHYSBO 调用修改:

# 之前:
physbo_optimizer.register(x, -ce_value)  # 只传值

# 之后:
physbo_optimizer.register(
    x, 
    -ce_measurement.value,
    noise_var=ce_measurement.variance  # ← 新增
)
# 注: PHYSBO 原生不支持 per-point noise
# 需要在 adapter 层做转换:
# 方案 A: 用我们自己的 HeteroscedasticGP 替换 PHYSBO 的 GP
# 方案 B: 低噪声点 → 小 jitter, 高噪声点 → 复制+抖动 (data augmentation)
```

---

## 11. 代码量估算 & 文件结构

| 新增模块 | 文件 | 估算行数 | 与原版差异 |
|----------|------|---------|----------|
| 核心数据类型 | `uncertainty/types.py` | ~100 | +40 (metadata 字段 + AgentContext/Feedback 预定义) |
| Domain Config | `domain_knowledge/electrochemistry/*.yaml` | ~120 | ★ 全新 |
| Domain Config Loader | `domain_knowledge/loader.py` | ~80 | ★ 全新 |
| 过程式规则 | `domain_knowledge/electrochemistry/spectrum_rules.py` | ~80 | ★ 全新 |
| Extractor 基类 | `extractors/base.py` | ~60 | ★ 全新 |
| EIS 不确定性量化 | `extractors/eis_uncertainty.py` | ~380 | +30 (metadata 写入 + config 注入) |
| DC Cycling 不确定性 | `extractors/dc_uncertainty.py` | ~160 | +10 (metadata + config) |
| UV-Vis 不确定性 | `extractors/uvvis_uncertainty.py` | ~130 | +10 |
| XRD 不确定性 | `extractors/xrd_uncertainty.py` | ~130 | +10 |
| 不确定性传播层 | `uncertainty/propagation.py` | ~280 | +30 (_aggregate_metadata) |
| 异质噪声 GP | `backends/gp_heteroscedastic.py` | ~330 | +30 (metadata 存储 + get_model_state) |
| 可视化（4 种图） | `visualization/uncertainty_flow.py` | ~250 | +50 (confidence_heatmap) |
| Engine 接线 | `engine.py` 修改 | ~70 | +20 (config + agent 预留) |
| 测试 | `tests/test_uncertainty_*` | ~400 | 同 |
| **合计** | | **~2,570** | +约 570 行 |

```
optimization_copilot/
├── domain_knowledge/                        # ★ v4 新增 (整体架构共享)
│   ├── schema.py                            # YAML 验证 (~40)
│   ├── loader.py                            # DomainConfig (~80)
│   ├── electrochemistry/
│   │   ├── instruments.yaml                 # 仪器参数
│   │   ├── physical_constraints.yaml        # 物理约束
│   │   ├── eis_models.yaml                  # 等效电路
│   │   └── spectrum_rules.py                # 过程式规则
│   └── catalysis/                           # 未来领域 (v6+)
│       ├── instruments.yaml
│       └── ...
│
├── uncertainty/                             # ★ v4 核心
│   ├── __init__.py
│   ├── types.py                             # MeasurementWithUncertainty + metadata
│   └── propagation.py                       # 传播 + metadata 汇总
│
├── extractors/                              # ★ v4 核心
│   ├── __init__.py
│   ├── base.py                              # UncertaintyExtractor ABC
│   ├── eis_uncertainty.py                   # EIS: config 驱动
│   ├── dc_uncertainty.py                    # CE: config 驱动
│   ├── uvvis_uncertainty.py                 # UV-Vis
│   └── xrd_uncertainty.py                   # XRD
│
├── backends/
│   ├── gp_heteroscedastic.py                # ★ v4: 异质噪声 GP + metadata
│   └── ...
│
├── visualization/
│   ├── uncertainty_flow.py                  # ★ v4: 不确定性预算/影响/时间线/热力图
│   └── ...
│
├── agents/                                  # v6+ 实现 (v4 仅定义接口 in types.py)
│   ├── base.py                              # (placeholder)
│   └── ...
│
└── engine.py                                # 修改: config + observe(metadata) + agent 预留
```

---

## 12. 实现优先级

| 优先级 | 模块 | 理由 |
|--------|------|------|
| **P0** | `types.py` (含 metadata) | 定义接口，其他一切依赖此 |
| **P0** | `domain_knowledge/loader.py` + electrochemistry YAMLs | 所有 Extractor 依赖 config |
| **P0** | `gp_heteroscedastic.py` | 核心能力，无此则链路断裂 |
| **P1** | `extractors/base.py` | Extractor 基类 |
| **P1** | `dc_uncertainty.py` (CE) | 最稳定的 KPI，最先受益 |
| **P1** | `eis_uncertainty.py` (\|Z\|@freq) | 简单模式先行 |
| **P2** | `eis_uncertainty.py` (R_ct ensemble) | 需要 NLLS fitter，工作量大 |
| **P2** | `propagation.py` (含 _aggregate_metadata) | 传播链路 + Agent 数据汇总 |
| **P2** | `spectrum_rules.py` | 过程式规则，Agent 的 Pragmatic 基础 |
| **P2** | `uncertainty_flow.py` (可视化) | 调试和发表都需要 |
| **P3** | `uvvis_uncertainty.py` + `xrd_uncertainty.py` | 等仪器接入 SDL 时再做 |
| **P3** | Engine 接线 | 与 v2 InfrastructureStack 集成同步 |

---

## 附录 A: 发表价值

**论文角度：**

这条链路在 SDL 文献中几乎没有被系统化讨论过。现有 SDL 平台（Atinary、Telescope、AC2）都把 KPI 当成精确值传给优化器。

**可以形成的 contribution：**
1. 形式化定义了 "measurement-aware Bayesian optimization"
2. 证明异质噪声 GP 在真实实验数据上优于同质噪声 GP（用锌电沉积数据）
3. 不确定性预算图指导实验改进方向（比优化参数更有价值）
4. **Domain-configurable 设计：** 同一套 UQ 框架，换一份 YAML 即可适配新领域——这是可扩展性的核心卖点

**目标期刊：** Digital Discovery (RSC), Matter (Cell Press), 或 Nature Machine Intelligence 的 methods section

## 附录 B: v4 → v6+ 接口清单

以下是 v4 为后续版本预留的所有接口点汇总:

| 接口 | v4 写入 | v6+ 消费者 | 数据内容 |
|------|---------|-----------|---------|
| `MeasurementWithUncertainty.metadata["raw_features"]` | Extractor | SpectrumAnalysisAgent | Nyquist 特征、电荷量等 |
| `MeasurementWithUncertainty.metadata["quality_flags"]` | Extractor | AnomalyDetectionAgent | 质量标记列表 |
| `MeasurementWithUncertainty.metadata["rule_diagnosis"]` | spectrum_rules.py | HypothesisAgent | 过程式规则的初步诊断 |
| `ObservationWithNoise.metadata["min_confidence"]` | Propagator | Orchestrator | 最低 confidence → 触发异常检测 |
| `ObservationWithNoise.metadata["unreliable_kpis"]` | Propagator | AnomalyDetectionAgent | 不可靠 KPI 列表 |
| `ObservationWithNoise.metadata["uncertainty_budget"]` | Propagator | SymRegAgent, v7b | 方差贡献比例 |
| `GP.get_model_state()` | GP | HypothesisAgent, SymRegAgent | 完整 GP 状态 |
| `GP.observe(metadata=)` | Engine | GP (存储) → Agent (检索) | 全链路追溯 |
| `DomainConfig` | YAML + rules | 所有 Agent | 领域知识注入 |
| `OptimizationFeedback` | Agent (v6+) | GP / Engine | noise_override, prior, constraint |
