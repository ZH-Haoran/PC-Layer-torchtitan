# visualize

SVD 分析结果的可视化模块，包含两个绘图工具：

- **`plotter`** — 绘制某一步的奇异值分布直方图（静态快照）
- **`metrics_plotter`** — 绘制指标随训练步数变化的曲线（动态趋势）

两者的输入都是 `svd_analyzer.py` 生成的 `singular_values_step_*.json` 文件。

---

## Environment
First run:
```bash
export LD_LIBRARY_PATH=/usr/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
```
for setup.

## plotter — 奇异值直方图

对单个 step 的 JSON 文件，按权重类型（wq, wk, ...）和层号绘制奇异值分布直方图。

### 快速开始

```bash
# 画一个 JSON 的奇异值直方图
python3 -m visualize.plotter path/to/singular_values_step_100.json

# 比较两个实验同一步的分布
python3 -m visualize.plotter path/to/exp_A/step_100.json path/to/exp_B/step_100.json
```

### 参数速查

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `json_files` | 位置参数 | (必填) | 一个或多个 `singular_values_step_*.json` 文件路径 |
| `--normalized` | flag | off | 画 S/\|\|W\|\| 归一化直方图（需要 JSON 中有 `weight_norm`） |
| `--topsv` | flag | off | 标记最大奇异值 σ₁ 的竖虚线和数值 |
| `--logy` | flag | off | y 轴使用 log scale |
| `--labels` | 多选 | 自动 | 自定义图例标签，每个 JSON 文件对应一个 |
| `--fmt` | 单选 | `pdf` | 输出格式：`pdf` 或 `png` |

### 两种绘图模式

**默认模式** — 奇异值直方图：
- 单个 JSON：同时画原始 SV 和 PC 后 SV（如果有）
- 多个 JSON：每个 JSON 画 PC 后 SV（如果有），方便对比

**`--normalized` 模式** — S/\|\|W\|\| 归一化直方图：
- 将奇异值除以权重范数，x 轴归一化到 [0, 1] 附近
- 根据 `pc_level` 自动画 cutoff 竖线（如 pc_level=1 对应 cutoff=0.8）
- 用于观察 PC 实际"砍"了多少比例的奇异值

### 用法示例

```bash
# 比较两个实验，标记 σ₁，y 轴 log scale
python3 -m visualize.plotter exp_A/step_100.json exp_B/step_100.json \
    --topsv --logy \
    --labels "AdamW" "Muon"

# 画归一化直方图，观察 PC cutoff
python3 -m visualize.plotter exp/step_100.json --normalized
```

### 输出目录结构

```
# 默认模式
step_100/
  wq/layer0.pdf
  wk/layer0.pdf
  ...

# --normalized 模式
step_100/normalized_weights/
  wq/layer0.pdf
  ...
```

---

## metrics_plotter — 指标趋势曲线

从实验目录中的多个 step JSON 文件提取指标，绘制 metric-vs-step 曲线。

### 快速开始

```bash
# 画一个实验的所有指标
python3 -m visualize.metrics_plotter path/to/exp_dir

# 比较两个实验
python3 -m visualize.metrics_plotter path/to/exp_A path/to/exp_B
```

### 输入格式

每个实验目录下需要包含 `singular_values_step_*.json` 文件，由 `svd_analyzer.py` 生成。

### 参数速查

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `experiment_dirs` | 位置参数 | (必填) | 一个或多个实验目录路径 |
| `--metrics` | 多选 | 全部 | 要绘制的指标名称 |
| `--labels` | 多选 | 自动 | 自定义图例标签，每个实验目录对应一个 |
| `--plot-levels` | 多选 | 全部 | 绘图粒度：`per_layer`、`per_block`、`global` |
| `--x-unit` | 单选 | `step` | X 轴单位：`step` 或 `token` |
| `--total-tokens` | float | - | 训练总 token 数（单位：B），`--x-unit token` 时必填 |
| `--output-dir` | 路径 | 自动 | 输出目录 |
| `--fmt` | 单选 | `pdf` | 输出格式：`pdf` 或 `png` |

### 可用指标 (`--metrics`)

| 指标名 | 含义 | 全局聚合方式 |
|--------|------|-------------|
| `modified_condition_number` | top-10% 均值 / bottom-10% 均值 | geometric mean |
| `quantile_condition_number` | q90 / q10 | geometric mean |
| `svd_entropy` | 归一化 SVD 熵，范围 [0, 1] | mean |
| `weight_norm` | 权重范数（仅 per_layer） | mean |
| `max_singular_value` | 最大奇异值（PC 后） | mean |
| `original_max_singular_value` | 最大奇异值（PC 前） | mean |

### 绘图粒度 (`--plot-levels`)

- **`per_layer`** — 每个层的每个权重矩阵单独一张图
- **`per_block`** — 按权重类型聚合（如所有层的 wq 画一张图），聚合方式为 mean
- **`global`** — 所有层所有权重聚合成一张图

### 用法示例

```bash
# 只看 per_block 和 global 级别的条件数
python3 -m visualize.metrics_plotter path/to/exp \
    --metrics modified_condition_number \
    --plot-levels per_block global

# 比较两个实验，自定义图例
python3 -m visualize.metrics_plotter path/to/exp_A path/to/exp_B \
    --labels "AdamW lr=3e-4" "Muon lr=1e-3"

# X 轴用 token 数（假设训练了 1000 步，总共 1.3B tokens）
python3 -m visualize.metrics_plotter path/to/exp \
    --x-unit token --total-tokens 1.3
```

step 会被线性换算为 token 数：`x = step / max_step * total_tokens`。

```bash
# 完整示例
python3 -m visualize.metrics_plotter \
    path/to/exp_A path/to/exp_B \
    --metrics svd_entropy modified_condition_number \
    --labels "Baseline" "PC-Layer" \
    --plot-levels per_block global \
    --x-unit token --total-tokens 1.3 \
    --output-dir ./comparison_plots
```

### 输出目录结构

```
output_root/
  modified_condition_number/
    per_layer/
      layers.0.attention.wq.pdf
      ...
    per_block/
      wq.pdf
      wk.pdf
      ...
    global.pdf
  svd_entropy/
    ...
```
