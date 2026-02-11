# Optimization Copilot — 可视化层 v3 增补开发规范

> **约束前提：** 纯 Python stdlib，零外部依赖，与 v2 `VisualizationEngine` + `PlotData` 架构完全兼容。
>
> **来源：** 对 Optuna Dashboard v0.20、CIME4R (Bayer/JKU)、Ax 1.0、pymoo、BOXVIA、BayesianOptimization、SHAP、SDL 监控平台、GP surrogate 文献、Design Space Exploration 研究的全面调研（80+ 来源）。
>
> **定位：** 以下为 v2 方案（3,698 行，40+ 图表）中**尚未覆盖**的补充特性，新增 ~12 种可视化 + 2 个分析模块。

---

## 目录

1. [LLM 接口预留](#1-llm-接口预留)
2. [VSUP 不确定性色彩引擎](#2-vsup-不确定性色彩引擎)
3. [六边形分箱覆盖视图](#3-六边形分箱覆盖视图)
4. [SHAP 交互分析（4 种图 + 计算引擎）](#4-shap-交互分析)
5. [空间填充质量度量](#5-空间填充质量度量)
6. [SDL 实时监控可视化（4 种图）](#6-sdl-实时监控可视化)
7. [Design Space 高级探索（3 种图）](#7-design-space-高级探索)
8. [竞品对标总结](#8-竞品对标总结)
9. [代码量估算 & 实现优先级](#9-代码量估算--实现优先级)
10. [修订后的图表总表](#10-修订后的图表总表)

---

## 1. LLM 接口预留

**背景：** Optuna Dashboard v0.20.0 (2025-11) 引入 LLM (GPT-4/5) 集成 —— 自然语言试验过滤 + 自动图表生成。这代表了优化可视化的新范式。

| Optuna 功能 | 描述 | 我们的定位 |
|------|------|-----------|
| **自然语言试验过滤** | "show trials where accuracy > 0.9 and learning\_rate < 0.01" | 🔮 Phase 5+ |
| **自动图表生成** | LLM 根据文本 prompt 自动生成 Plotly.js 图表 | 🔮 Phase 5+ |
| **Browser-only 模式** | SQLite3 Wasm + Rust，无需 Python 安装 | 我们用 Pure Python SVG，不需要 |
| **VS Code 扩展** | .db 文件直接可视化 | ✅ JSON + SVG 已支持 IDE 集成 |
| **偏好优化** | 用户比较试验，GP 采样建议新方向 | 🔮 人机协作方向 |

**策略：** 我们的 `PlotData` JSON Schema 天然可作为 LLM 结构化输出目标，无需改造渲染层。

**文件：** `visualization/llm_assistant.py` (~80 行骨架)

```python
class LLMVisualizationAssistant:
    """LLM 驱动的自然语言可视化。
    
    预留架构：
    1. 用户输入自然语言 query
    2. LLM 解析为结构化 PlotSpec (JSON Schema)
    3. VisualizationEngine 渲染
    
    PlotSpec 兼容当前 PlotData 格式，无需改造渲染层。
    """
    
    def query_to_plot(self, query: str, study_data: dict) -> 'PlotData':
        """
        示例：
        query = "展示 learning_rate < 0.01 且 accuracy > 0.9 的试验的参数分布"
        → 自动生成 parallel coordinate plot with filter
        """
        # Step 1: NL → PlotSpec (LLM call)
        plot_spec = self._parse_query(query, study_data)
        # Step 2: PlotSpec → PlotData (deterministic)
        return self._render(plot_spec)
```

---

## 2. VSUP 不确定性色彩引擎

**来源：** CIME4R (Bayer/JKU, 2024) 引入 **VSUP (Value-Suppressing Uncertainty Palettes)** —— 一种双变量色彩映射技术，同时编码目标值 + 不确定性，自动**视觉淡化高不确定性区域**。

**科研意义：** 防止研究者过度信任不确定区域的预测，是信任校准的视觉工具。

**文件：** `visualization/colormaps.py` (~120 行)

```python
class VSUPColorMap:
    """Value-Suppressing Uncertainty Palette.
    
    核心原理：
    - 色相 (hue) 编码目标值 (如 yield: 红→蓝)
    - 饱和度 (saturation) 编码不确定性 (高不确定 → 灰/淡)
    - 效果：高确定性区域鲜艳，低确定性区域自动"褪色"
    """
    
    def __init__(self, value_cmap: str = "viridis", 
                 uncertainty_range: tuple[float, float] = (0.0, 1.0)):
        self.value_cmap = value_cmap
        self.uncertainty_range = uncertainty_range
    
    def map(self, value: float, uncertainty: float) -> tuple[int, int, int, int]:
        """返回 RGBA 颜色。
        
        value: 归一化目标值 [0, 1]
        uncertainty: 归一化不确定性 [0, 1]
        
        Returns: (R, G, B, A) 0-255
        """
        # 基础色：来自 value
        base_r, base_g, base_b = self._value_to_rgb(value)
        
        # 抑制系数：uncertainty → 饱和度降低
        suppress = 1.0 - uncertainty  # 高不确定 → 接近 0
        
        # 混合到灰色
        gray = 200  # 背景灰
        r = int(base_r * suppress + gray * (1 - suppress))
        g = int(base_g * suppress + gray * (1 - suppress))
        b = int(base_b * suppress + gray * (1 - suppress))
        
        return (r, g, b, 255)
    
    def _value_to_rgb(self, value: float) -> tuple[int, int, int]:
        """viridis-style 色彩映射 (Pure Python)。"""
        # 简化版 viridis: 插值 5 个关键色
        stops = [
            (0.0, (68, 1, 84)),      # 深紫
            (0.25, (59, 82, 139)),    # 蓝紫
            (0.5, (33, 145, 140)),    # 青绿
            (0.75, (94, 201, 98)),    # 绿
            (1.0, (253, 231, 37)),    # 黄
        ]
        # 线性插值
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if t0 <= value <= t1:
                frac = (value - t0) / (t1 - t0)
                return tuple(int(c0[j] + frac * (c1[j] - c0[j])) for j in range(3))
        return stops[-1][1]
```

**集成点：** GP 后验可视化 (3.1)、约束边界可视化 (3.3)、投影视图均可使用 VSUP 替代单变量色彩。

---

## 3. 六边形分箱覆盖视图

**来源：** CIME4R 用六边形分箱（hexagonal binning）显示整个参数空间（包括未探索的组合），而非仅展示已有实验点。

**文件：** `visualization/parameter_space.py` (追加 ~150 行)

```python
def plot_hexbin_coverage(
    search_space: dict,           # 参数空间定义
    observed_points: list[dict],  # 已观测点
    predicted_surface: 'SurrogateModel | None' = None,
    hex_size: int = 20,           # 六边形大小
    color_by: str = "density"     # density | predicted_mean | uncertainty
) -> PlotData:
    """参数空间六边形分箱覆盖视图。
    
    对标 CIME4R 聚合视图。
    
    三种着色模式：
    - density: 每个 hex 中的已观测点密度（探索覆盖率）
    - predicted_mean: 代理模型预测均值
    - uncertainty: 预测不确定性（指导下一步探索方向）
    
    科研价值：
    - 展示参数空间中哪些区域被充分探索，哪些仍是盲区
    - 定量化 space-filling quality
    
    实现要点：
    - 六边形网格坐标: axial coordinates (q, r)
    - 点到 hex 分配: cube_round() 最近邻
    - 仅支持 2D（高维需先 PCA 降维）
    """
    # 1. 生成六边形网格覆盖 search_space
    hexes = _generate_hex_grid(search_space, hex_size)
    
    # 2. 将 observed_points 分配到对应 hex
    for point in observed_points:
        hex_id = _point_to_hex(point, hex_size)
        hexes[hex_id].count += 1
    
    # 3. 着色
    if color_by == "density":
        colors = {h: hex.count / max_count for h, hex in hexes.items()}
    elif color_by == "predicted_mean" and predicted_surface:
        colors = {h: predicted_surface.predict(hex.center)[0] for h, hex in hexes.items()}
    elif color_by == "uncertainty" and predicted_surface:
        colors = {h: predicted_surface.predict(hex.center)[1] for h, hex in hexes.items()}
    
    # 4. 生成 PlotData (hex 多边形 + 色彩)
    return PlotData(
        plot_type="hexbin",
        data={"hexagons": hexes, "colors": colors, "observed": observed_points},
        metadata={"color_by": color_by, "hex_size": hex_size}
    )
```

---

## 4. SHAP 交互分析

**来源：** SHAP (SHapley Additive exPlanations) 是模型解释的事实标准。v2 已有 PDP/ALE/Sobol/ARD，以下补充 SHAP 的四种经典图表 + Pure Python 计算引擎。

### 4.1 SHAP 值计算引擎

**文件：** `_analysis/shap_values.py` (~300 行)

```python
class KernelSHAPApproximator:
    """基于核近似的 SHAP 值计算。
    
    适用于任何黑盒 surrogate model (GP, RF)。
    使用加权线性回归近似 Shapley 值。
    
    算法:
    1. 对每个特征子集 S ⊆ {1,...,d}:
       - 构建 mask: 保留 S 中的参数，其余用边际分布采样
       - 计算 surrogate(mask) 的均值
    2. 用 SHAP kernel 加权的线性回归求解 φ_i
    
    注: 对 d < 12 使用穷举，d ≥ 12 使用采样近似。
    复杂度: O(2^d) 穷举 或 O(d × n_samples) 采样
    """
    
    def __init__(self, model: 'SurrogateModel', n_samples: int = 1000):
        self.model = model
        self.n_samples = n_samples
    
    def compute(self, x: list[float], X_background: list[list[float]]) -> list[float]:
        """计算 x 的 SHAP 值。
        
        Args:
            x: 待解释的点
            X_background: 背景数据集（用于边际化）
        
        Returns:
            shap_values: 各参数的 SHAP 值，sum(φ_i) ≈ f(x) - E[f(X)]
        """
        d = len(x)
        if d <= 11:
            return self._exact_shap(x, X_background)
        else:
            return self._sampled_shap(x, X_background)
    
    def _exact_shap(self, x, X_bg):
        """穷举所有 2^d 个子集。"""
        d = len(x)
        n_subsets = 1 << d  # 2^d
        
        # SHAP kernel weight: π(|S|) = (d-1) / (C(d,|S|) * |S| * (d-|S|))
        # 空集和全集权重为 inf → 特殊处理
        
        # 构建线性系统 Zw = y
        Z = []  # 子集指示矩阵
        y = []  # f(mask) 均值
        w = []  # kernel 权重
        
        for mask_int in range(1, n_subsets - 1):  # 排除空集和全集
            mask = [(mask_int >> j) & 1 for j in range(d)]
            s = sum(mask)
            
            # 计算 E[f(x_S, X_Sc)]
            f_vals = []
            for x_bg in X_bg:
                x_masked = [x[j] if mask[j] else x_bg[j] for j in range(d)]
                mu, _ = self.model.predict(x_masked)
                f_vals.append(mu)
            
            Z.append(mask)
            y.append(sum(f_vals) / len(f_vals))
            
            # Kernel weight
            weight = (d - 1) / (_comb(d, s) * s * (d - s))
            w.append(weight)
        
        # 加权线性回归: min_φ Σ w_i (y_i - Z_i · φ)^2
        # 解: φ = (Z^T W Z)^{-1} Z^T W y
        return self._weighted_regression(Z, y, w, d)
    
    def _sampled_shap(self, x, X_bg):
        """采样近似（高维场景）。"""
        d = len(x)
        Z, y, w = [], [], []
        
        for _ in range(self.n_samples):
            # 随机选择子集大小
            s = _random_subset_size(d)
            # 随机选择 s 个特征
            mask = [0] * d
            indices = _random_sample(range(d), s)
            for idx in indices:
                mask[idx] = 1
            
            # 计算条件期望
            f_vals = []
            for x_bg in _random_sample(X_bg, min(100, len(X_bg))):
                x_masked = [x[j] if mask[j] else x_bg[j] for j in range(d)]
                mu, _ = self.model.predict(x_masked)
                f_vals.append(mu)
            
            Z.append(mask)
            y.append(sum(f_vals) / len(f_vals))
            weight = (d - 1) / (_comb(d, s) * s * (d - s)) if 0 < s < d else 1.0
            w.append(weight)
        
        return self._weighted_regression(Z, y, w, d)
```

### 4.2 SHAP 可视化（4 种图）

**文件：** `visualization/explainability.py` (追加 ~250 行)

```python
def plot_shap_waterfall(
    trial_index: int,
    shap_values: list[float],     # 每个参数的 SHAP 值
    feature_names: list[str],
    base_value: float             # 模型基准预测值
) -> PlotData:
    """SHAP Waterfall Plot — 单次实验的特征贡献分解。
    
    展示从 base_value 开始，每个参数如何将预测值
    "推"向最终结果。
    
    SVG 渲染：
    - 横向瀑布图
    - 红色条：正贡献（增加目标值）
    - 蓝色条：负贡献（降低目标值）
    - 累积线连接各步
    - 标注：参数名 = 参数值 (φ = SHAP值)
    """

def plot_shap_beeswarm(
    shap_matrix: list[list[float]],  # shape: (n_trials, n_params)
    feature_values: list[list[float]], # 对应的参数实际值
    feature_names: list[str]
) -> PlotData:
    """SHAP Beeswarm Plot — 全局特征重要性 + 值→贡献关系。
    
    Y 轴: 参数（按重要性排序，即 mean(|SHAP|) 降序）
    X 轴: SHAP 值
    颜色: 参数实际值 (低→蓝, 高→红)
    
    优于简单重要性条形图，因为同时展示：
    - 哪些参数重要（散布宽度）
    - 高/低值如何影响目标（颜色-位置关系）
    - 非线性/非单调关系（同色散布在两侧）
    
    SVG 渲染要点：
    - 每行 jitter 防重叠
    - 色彩条 legend 显示参数值范围
    """

def plot_shap_dependence(
    feature_idx: int,
    shap_values: list[float],        # 该参数的 SHAP 值
    feature_values: list[float],     # 该参数的实际值
    interaction_feature_values: list[float] | None = None,
    interaction_name: str | None = None
) -> PlotData:
    """SHAP Dependence Plot — 参数→贡献 + 交互效应。
    
    X 轴: 参数值
    Y 轴: SHAP 值
    颜色: 交互特征值（自动选择最强交互，或手动指定）
    
    科研价值：
    - 发现非线性关系（如 pH 在 4-6 范围内效应反转）
    - 揭示隐藏交互（如温度 × 催化剂浓度）
    
    自动交互特征选择：
    - 对每个候选特征 j ≠ i:
      计算 corr(SHAP_i, feature_j)
    - 选择相关性最强的特征作为着色变量
    """

def plot_shap_force(
    trial_index: int,
    shap_values: list[float],
    feature_names: list[str],
    feature_values: list[float],
    base_value: float
) -> PlotData:
    """SHAP Force Plot — 横向力图展示特征推拉。
    
    红色箭头: 正向推动（增加目标值）的参数
    蓝色箭头: 负向拉动（降低目标值）的参数
    宽度: |SHAP值|
    
    SVG 渲染：
    - 中心线: 最终预测值
    - 向右: 正贡献参数（红色楔形）
    - 向左: 负贡献参数（蓝色楔形）
    - 标注: 参数名 = 值
    
    适合单次实验的快速解释。
    """
```

---

## 5. 空间填充质量度量

**来源：** 空间填充质量在科研中很重要 —— Sobol vs LHS vs Random 的采样质量需要量化。v2 有 `plot_coverage_analysis`，以下补充定量化度量。

**文件：** `visualization/diagnostics.py` (追加 ~200 行)

```python
def plot_space_filling_metrics(
    points: list[list[float]],
    bounds: list[tuple[float, float]],
    metrics: list[str] = ["discrepancy", "coverage", "min_distance"]
) -> PlotData:
    """空间填充质量仪表板。
    
    三种互补度量：
    
    1. **Star Discrepancy (D*)** — 偏差度
       - 衡量点集分布的均匀性
       - D* = sup |F_n(x) - F_uniform(x)| over all axis-aligned boxes
       - 越接近 0 越好，Sobol 优于 Random
       - 计算: O(n^(d/2) * log(n))
    
    2. **Coverage Percentage** — 覆盖率
       - 将参数空间网格化，计算"被至少一个观测点覆盖"的网格比例
       - grid_resolution 自适应: min(100, 10^d) 个网格点
       - 输出: percentage (0-100%)
    
    3. **Minimum Inter-point Distance** — 最小间距
       - 所有点对之间的最小归一化距离
       - 检测聚集（clustering）现象
       - 高值 = 点分布均匀, 低值 = 有聚集
    
    可视化:
    - 主图: 所有度量随 n_trials 的演变曲线
    - 子图: 2D 投影覆盖热图 (每对参数)
    - 参考线: Random / LHS / Sobol 的理论值
    """

def _compute_star_discrepancy(points: list[list[float]], 
                               bounds: list[tuple[float, float]]) -> float:
    """计算星差异 (Star Discrepancy)。
    
    使用 Niederreiter 方法的简化版本。
    对 d ≤ 5 使用精确计算（检查所有由点坐标定义的 box），
    d > 5 使用随机近似（10000 随机 box 采样）。
    """
    d = len(bounds)
    # 归一化到 [0,1]^d
    normalized = []
    for p in points:
        np_ = [(p[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0]) for i in range(d)]
        normalized.append(np_)
    
    n = len(normalized)
    max_disc = 0.0
    
    if d <= 5:
        # 精确计算: 检查所有由点坐标定义的 box
        for point in normalized:
            vol = 1.0
            for xi in point:
                vol *= xi
            count = sum(1 for q in normalized 
                       if all(q[j] <= point[j] for j in range(d)))
            disc = abs(count / n - vol)
            max_disc = max(max_disc, disc)
    else:
        # 随机近似
        import random as _rnd
        for _ in range(10000):
            corner = [_rnd.random() for _ in range(d)]
            vol = 1.0
            for xi in corner:
                vol *= xi
            count = sum(1 for q in normalized
                       if all(q[j] <= corner[j] for j in range(d)))
            disc = abs(count / n - vol)
            max_disc = max(max_disc, disc)
    
    return max_disc

def _compute_coverage(points, bounds, resolution=50):
    """网格覆盖率。"""
    d = len(bounds)
    if d > 6:
        resolution = max(5, int(100 ** (1/d)))
    
    # 网格化
    grid_counts = {}
    for p in points:
        cell = tuple(
            min(resolution - 1, int((p[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0]) * resolution))
            for i in range(d)
        )
        grid_counts[cell] = grid_counts.get(cell, 0) + 1
    
    total_cells = resolution ** d
    covered_cells = len(grid_counts)
    return covered_cells / total_cells * 100

def _compute_min_distance(points, bounds):
    """最小归一化点间距。"""
    d = len(bounds)
    n = len(points)
    
    # 归一化
    normalized = []
    for p in points:
        np_ = [(p[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0]) for i in range(d)]
        normalized.append(np_)
    
    min_dist = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            dist = sum((normalized[i][k] - normalized[j][k]) ** 2 for k in range(d)) ** 0.5
            min_dist = min(min_dist, dist)
    
    return min_dist
```

---

## 6. SDL 实时监控可视化

**来源：** v2 聚焦后优化分析。以下补充 SDL (自驱实验室) 在线运行时的**实时监控**需求，对标 Atinary SDLabs、Telescope Innovations、Trilobio。

**文件：** `visualization/sdl_monitor.py` (新增 ~400 行)

```python
class SDLDashboardData:
    """SDL 实时监控仪表板数据生成。
    
    SDL 自主等级 (Royal Society Open Science, 2025):
    - Level 1: 手动
    - Level 2: 半自动
    - Level 3: 闭环自动 (DMTA cycle)
    - Level 4: 自主发现
    
    我们的目标: Level 3-4 监控。
    """

# ──── 6.1 实验状态仪表板 ────

def plot_experiment_status_dashboard(
    queue: list[dict],        # 排队实验
    running: list[dict],      # 正在执行
    completed: list[dict],    # 已完成
    hardware_status: dict     # 各硬件状态
) -> PlotData:
    """SDL 实验状态仪表板。
    
    四象限布局:
    ┌──────────────┬──────────────┐
    │  实验队列     │  硬件状态    │
    │  (甘特图)     │  (状态灯)    │
    ├──────────────┼──────────────┤
    │  进度总览     │  异常告警    │
    │  (进度条)     │  (时序图)    │
    └──────────────┴──────────────┘
    
    硬件状态含:
    - Robot (Opentrons): idle / moving / dispensing / error
    - Potentiostat (Squidstat): idle / measuring / error
    - Sensors: pH, temperature, pressure
    - PLC: 连接状态
    
    数据结构:
    hardware_status = {
        "opentrons": {"state": "dispensing", "progress": 0.6, "error": None},
        "squidstat": {"state": "measuring", "channel": 3, "error": None},
        "ph_sensor": {"value": 5.2, "in_range": True},
        "temperature": {"value": 25.1, "in_range": True},
    }
    """

# ──── 6.2 安全监控 ────

def plot_safety_monitoring(
    constraint_history: list[dict],   # 约束满足历史
    anomaly_scores: list[float],      # 异常检测分数
    threshold: float = 3.0            # 异常阈值 (标准差)
) -> PlotData:
    """安全监控 — 约束违反 + 异常检测。
    
    上面板: 各约束的满足/违反时间线
    - 绿色段: 约束满足
    - 红色段: 约束违反
    - 标注: 违反详情 (如 "pH = 3.2 < 4.0 下限")
    
    下面板: 异常分数 + 阈值线
    - 蓝色线: 异常分数随时间变化
    - 红色虚线: 阈值
    - 红色区域: 超阈值事件
    
    叠加: 算法建议 vs 人类 override 标记
    """

# ──── 6.3 人机协作 ────

def plot_human_in_the_loop(
    proposed_experiments: list[dict],  # 算法建议的实验
    human_decisions: list[str],        # approve / reject / modify
    reasoning: list[str] | None = None # 人类决策理由
) -> PlotData:
    """人机协作决策可视化。
    
    科研场景中，研究者常需要审核算法建议:
    
    可视化内容:
    1. 拟议实验参数 + 算法决策理由 (白话文摘要)
       - "建议高温 + 低浓度: GP 预测此区域 yield 高 (μ=0.85±0.12)"
    2. 人类决策标注
       - ✅ approve (绿色)
       - ❌ reject (红色)
       - ✏️ modify (橙色) + 修改内容
    3. 信任校准指标
       - Override 率随时间变化 (理想: 先高后低)
       - 被拒绝实验的事后分析 (人类对还是算法对?)
    
    参考: Nature Communications (2025) — 
    "SDL 需要视觉符号 + 白话语言摘要，让研究者高效审核实验提案"
    """

# ──── 6.4 连续运行时间线 ────

def plot_continuous_operation_timeline(
    start_time: str,
    operations: list[dict],       # [{time, type, status, details}]
    failures: list[dict],         # 故障事件
    recovery_actions: list[dict]  # 自动恢复动作
) -> PlotData:
    """24/7 连续运行时间线。
    
    横轴: 时间 (天/小时)
    泳道 (swim lanes): 各设备运行状态
    - Robot:        ████░░████████░░████
    - Potentiostat: ░░████████░░████████
    - Analysis:     ░░░░██░░░░██░░░░██░░
    
    事件标记:
    - 🔴 故障 (红色三角)
    - 🟢 恢复 (绿色三角)
    - 🟡 维护 (黄色菱形)
    
    指标叠加: 吞吐量 (实验/小时) 趋势线
    
    适用: 夜间/周末无人值守运行的事后审查
    """
```

---

## 7. Design Space 高级探索

**来源：** 2023-2025 Design Space Exploration 研究，包含 VAE latent space、iSOM、forward/inverse design 等方法。

**文件：** `visualization/design_space.py` (新增 ~350 行)

```python
# ──── 7.1 低维投影 ────

def plot_latent_space_exploration(
    X_observed: list[list[float]],     # 原始参数空间点
    Y_observed: list[float],           # 目标值
    method: str = "pca",               # pca | tsne
    color_by: str = "objective"        # objective | uncertainty | iteration
) -> PlotData:
    """设计空间低维投影可视化。
    
    支持两种 Pure Python 降维方法:
    
    1. PCA:
       - 协方差矩阵特征分解 (利用 _math/ 基础设施)
       - 保留前 2-3 主成分
       - 优点: 可解释的投影方向 + 方差贡献比
       - 复杂度: O(d^3 + n*d^2)
    
    2. t-SNE (简化版):
       - 对称化 SNE 梯度下降
       - Barnes-Hut 近似: O(n log n)
       - 保留局部结构
       - 复杂度: O(n^2) 精确版, O(n log n) 近似版
       - 注: n > 500 时切换到近似
    
    着色模式:
    - objective: 目标值渐变色 (viridis)
    - uncertainty: GP 预测不确定性 (VSUP 色彩)
    - iteration: 迭代序号 (时间演化)
    
    对标: CIME4R 投影视图
    """

# ──── 7.2 iSOM 景观 ────

def plot_isom_landscape(
    X_observed: list[list[float]],
    Y_observed: list[float],
    grid_size: tuple[int, int] = (10, 10)
) -> PlotData:
    """iSOM (Interactive Self-Organizing Map) 可视化。
    
    将高维参数空间映射到 2D 网格:
    - 每个网格节点代表一个"原型"参数组合
    - 颜色: 该区域的目标值均值
    - 标记: 实际实验点落在哪些节点
    - U-matrix: 相邻节点距离 (揭示参数空间中的自然分界)
    
    优势: 比 t-SNE 更稳定，拓扑保持性好
    适用: 化学参数空间导航 (催化剂发现)
    
    Pure Python 实现:
    - 竞争学习 + 高斯邻域函数
    - 邻域半径指数衰减
    - 学习率线性衰减
    - 迭代次数: n_iterations = 500 * grid_size[0] * grid_size[1]
    """

# ──── 7.3 Forward / Inverse 设计 ────

def plot_forward_inverse_design(
    parameter_space: dict,
    objective_space: dict,
    mapping_model: 'SurrogateModel',
    target_objectives: list[float] | None = None
) -> PlotData:
    """双向设计空间可视化 (Forward + Inverse)。
    
    Forward Design (参数 → 目标):
    - 研究者选择参数范围 → 看预测目标分布
    - 可视化: 参数空间热图 → 箭头 → 目标空间散点
    
    Inverse Design (目标 → 参数):
    - 研究者设定目标值 → 看可行参数区域
    - 可视化: 目标空间标记 → 箭头 → 参数空间高亮区域
    
    实现方式:
    - Forward: 网格化参数空间 → surrogate.predict() → 目标分布
    - Inverse: 给定目标 ± 容差 → 在参数网格中筛选满足条件的点 → 高亮
    
    可视化布局:
    ┌──────────────────┐    ┌──────────────────┐
    │   参数空间        │ ←→ │   目标空间        │
    │   (等高线/热图)   │    │   (散点/分布)     │
    │   [交互式选择]    │    │   [目标设定]      │
    └──────────────────┘    └──────────────────┘
    
    科研价值: 
    - "我想要 yield > 90%, 需要什么参数?" → 直接看可行域
    - "如果我把温度从 80°C 改到 100°C, 目标怎么变?" → 即时反馈
    """
```

---

## 8. 竞品对标总结

| 功能 | Optuna v4.7+ | Ax 1.0 | CIME4R | pymoo | **Ours v3** |
|------|-------------|--------|--------|-------|-------------|
| 图表类型总数 | ~17 | ~8 | ~8 | ~10 | **~52** |
| GP 后验可视化 | ✗ | ✅ | ✗ | ✗ | ✅ |
| 采集函数分解 | ✗ | ✗ | ✗ | ✗ | **✅★** |
| 约束边界可视化 | ✗ | ✗ | ✗ | ✗ | **✅★** |
| 未知约束学习 | ✗ | ✗ | ✗ | ✗ | **✅★★** |
| 成本调整后悔 | ✗ | ✗ | ✗ | ✗ | **✅★** |
| 迁移学习可视化 | ✗ | ✗ | ✗ | ✗ | **✅★** |
| VSUP 不确定性色彩 | ✗ | ✗ | ✅ | ✗ | **✅** |
| 六边形分箱覆盖 | ✗ | ✗ | ✅ | ✗ | **✅** |
| SHAP 交互分析 | ✗ | ✗ | ✗ | ✗ | **✅★** |
| 空间填充度量 | ✗ | ✗ | ✗ | ✗ | **✅★** |
| SDL 实时监控 | ✗ | ✗ | ✗ | ✗ | **✅★★** |
| 人机协作可视化 | ✗ | ✗ | ✗ | ✗ | **✅★★** |
| 安全异常监控 | ✗ | ✗ | ✗ | ✗ | **✅★** |
| Design Space 探索 | ✗ | ✗ | ✅ (投影) | ✗ | **✅** (PCA+tSNE+iSOM) |
| Forward/Inverse 设计 | ✗ | ✗ | ✗ | ✗ | **✅★** |
| LLM 自动图表 | ✅ (GPT) | ✗ | ✗ | ✗ | 🔮 接口预留 |
| 偏好优化可视化 | ✅ | ✗ | ✗ | ✗ | 🔮 人机协作方向 |
| Pure Python SVG | ✗ | ✗ | ✗ | ✗ | **✅★★★** |
| 零外部依赖 | ✗ | ✗ | ✗ | ✗ | **✅★★★** |

**★ = 平台独有, ★★ = 业界首创, ★★★ = 架构优势**

---

## 9. 代码量估算 & 实现优先级

### 9.1 代码量

| 新增模块 | 文件 | 估算行数 |
|----------|------|---------|
| VSUP 色彩引擎 | `colormaps.py` | ~120 |
| 六边形分箱 | `parameter_space.py` (追加) | ~150 |
| SHAP 值计算 | `_analysis/shap_values.py` | ~300 |
| SHAP 可视化 (4 种图) | `explainability.py` (追加) | ~250 |
| 空间填充度量 | `diagnostics.py` (追加) | ~200 |
| SDL 实时监控 (4 种图) | `sdl_monitor.py` (新增) | ~400 |
| Design Space 探索 (3 种图) | `design_space.py` (新增) | ~350 |
| LLM 接口预留 | `llm_assistant.py` (骨架) | ~80 |
| **合计** | | **~1,850** |

### 9.2 实现优先级

| 优先级 | 模块 | 理由 |
|--------|------|------|
| **P0 (与 v2 可视化同步)** | VSUP 色彩、空间填充度量 | GP 后验和覆盖分析的质量提升，改动小 |
| **P1 (紧随 v2 之后)** | SHAP 计算 + 4 种图 | 科研解释的核心需求，填补 XAI 空白 |
| **P2 (SDL 集成时)** | SDL 监控 4 面板 | 与 SDL 硬件集成同步开发 |
| **P3 (高级分析)** | Design Space 探索 3 图 | 需要 PCA/tSNE 实现，独立模块 |
| **P4 (六边形分箱)** | hexbin 覆盖 | 增强覆盖分析，非关键路径 |
| **Future** | LLM 接口 | 骨架先放，等 LLM 集成时再充实 |

### 9.3 文件结构

```
optimization_copilot/
├── _analysis/
│   ├── shap_values.py          # ★ 新增: KernelSHAP Pure Python
│   └── ...
├── visualization/
│   ├── colormaps.py             # ★ 新增: VSUP + viridis
│   ├── sdl_monitor.py           # ★ 新增: SDL 实时监控 4 面板
│   ├── design_space.py          # ★ 新增: PCA/tSNE/iSOM/FwdInv
│   ├── llm_assistant.py         # ★ 新增: LLM 接口骨架
│   ├── parameter_space.py       # 追加: hexbin_coverage
│   ├── explainability.py        # 追加: SHAP waterfall/beeswarm/dependence/force
│   ├── diagnostics.py           # 追加: space_filling_metrics
│   └── ...
```

---

## 10. 修订后的图表总表

**共 52 种可视化类型 + 1 🔮，覆盖 9 个层级：**

| 层级 | 图表类型 | 数量 | 对标 |
|------|---------|------|------|
| **L1: 基础追踪** | 优化历史、收敛曲线、成本追踪、停止诊断、Phase 时间线 | 5 | Optuna |
| **L2: 参数分析** | 切片图、等高线、平行坐标、参数重要性、排序图、覆盖分析、交互热图、**六边形分箱**、**空间填充度量** | 9 | Optuna + CIME4R |
| **L3: 算法解释** | GP 后验、采集函数、约束边界、Surrogate 诊断、Trust Region、**VSUP 不确定性编码** | 6 | Ax + 独有 |
| **L4: 多目标** | Pareto 前沿、超体积收敛、RadViz、雷达图、Pareto 演化、指标仪表板 | 6 | pymoo |
| **L5: XAI 解释** | PDP、ALE、Sobol、ARD lengthscale、迁移诊断、**SHAP waterfall/beeswarm/dependence/force** | 9 | SHAP + 独有 |
| **L6: 对比 & 导出** | 算法对比、多保真对比、出版导出、报告生成、EDF | 5 | Optuna |
| **L7: SDL 监控** ★ | 实验状态仪表板、安全监控、人机协作、连续运行时间线 | 4 | 无竞品 |
| **L8: 设计空间** ★ | 低维投影 (PCA/tSNE)、iSOM、Forward/Inverse 设计 | 3 | CIME4R |
| **L9: 高级渲染** | SVG 引擎、HTML 报告、JSON 数据层、**LLM 接口预留** | 4+1🔮 | — |
| **合计** | | **52 + 1🔮** | |
