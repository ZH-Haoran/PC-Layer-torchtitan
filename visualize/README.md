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

---

## svd_analyzer — 从 ckpt 生成 SV JSON

`svd_analyzer.py` 是 SVD 计算的入口（被 `train.py` 在 `--visualize.enable` 模式下调用）。它会：加载某些 step 的 ckpt 权重、在指定的 val batch 上跑一次 forward 用于校验 ckpt 是否被正确加载（打印 val loss）、对每个 weight matrix 做 SVD，并写 `singular_values_step_{step}.json` 到输出目录。

### 给定实验名后的查找流程

实验名形如 `wsm10010000_PC_ADAMW_1B_baseline`（== `metrics.wandb_comment`）。

1. **找 ckpt 文件夹**。路径由 `train.py` 的拼接规则决定：
   ```
   {checkpoint.folder}/{model.name}_{model.flavor}/{optimizer.name}/{wandb_comment}/step-{N}/
   ```
   绝大多数 1B run 的 `checkpoint.folder` 是 `/data/zrs/sunruoyu-folder/checkpoints/llama2_pc_layer_cosine`，所以本例即：
   ```
   /data/zrs/sunruoyu-folder/checkpoints/llama2_pc_layer_cosine/llama2_1B/AdamW/wsm10010000_PC_ADAMW_1B_baseline/
   ```
   `ls` 这个目录看 `step-*` 子目录就知道实际存了哪些 step（注意保存频率由 `checkpoint.interval` 决定，不一定每 1k 都有）。

2. **找当时的 config toml**。和 ckpt 平级存有当时训练用的 toml：
   ```
   /data/zrs/sunruoyu-folder/checkpoints/llama2_pc_layer_cosine/configs/{wandb_comment}/*.toml
   ```
   例如 `configs/wsm10010000_PC_ADAMW_1B_baseline/adamw_10010000.toml`。**必须用这个 toml**（而非 `train_configs/` 里的当前版本），否则模型架构 / PC 设置可能和 ckpt 不匹配。

3. **选择要做 SVD 的 step 列表**。SVD 比较贵，通常不每个 ckpt 都做。把目标 step 写成逗号分隔传给 `--visualize.step`；缺失的 step 会被 warn 跳过。`-1` 表示全部可用 step。

### 跑脚本

直接复用 `train_configs/llama_scripts/visualize_adamw_82.sh` 的模板，把 `CONFIG_FILE` 和 `STEPS` 改成目标实验的值即可。最小命令示例：

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate pc
cd /data/zrs/sunruoyu-folder/PC-Layer-torchtitan-0427-split
export USE_LIBUV=1 CC=gcc
export LD_LIBRARY_PATH=/usr/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

CONFIG_FILE=/data/zrs/sunruoyu-folder/checkpoints/llama2_pc_layer_cosine/configs/wsm10010000_PC_ADAMW_1B_baseline/adamw_10010000.toml
STEPS="1,4000,8000,12000,16000,20000,24000,28000,32000,36000,40000,44000,48000,52000,56000,60000,61100"

torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 --role rank --tee 3 \
    train.py --job.config_file ${CONFIG_FILE} \
        --visualize.enable \
        --visualize.step "${STEPS}"
```

### 输出位置

JSON 写到 cwd 下的：
```
visualization_output/{model.name}_{model.flavor}_{optimizer.name}_{wandb_comment}/singular_values_step_{N}.json
```
即本例的 `visualization_output/llama2_1B_AdamW_wsm10010000_PC_ADAMW_1B_baseline/`。

### 提示

- **校验**：每个 step 在 SVD 前会跑 5 个 val batch，打印 `[Checkpoint Verification] Step N - Val loss: ...`。如果某一步明显偏离训练曲线，说明 ckpt 没正确加载，需检查 config 是否和当时一致。
- **崩溃恢复**：偶发 CUDA NVML 错误会让 run 中途挂掉。已写入的 JSON 不丢，直接把剩下的 step 重新传给 `--visualize.step` 续跑即可。
- **采样建议**：步频太密的 step 列表收益有限。一般 1B 这种 ~61k 步的 run，每 4k–5k 一个采样点足够画 metrics_plotter 的趋势曲线；记得首末两步（如 `1` 和 `61100`）单独加上。

---

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
| `condition_number` | max sv / min sv（PC block 上自动叠加 pre-PC 虚线） | geometric mean |
| `svd_entropy` | 归一化 SVD 熵，范围 [0, 1] | mean |
| `max_singular_value` | 最大奇异值（PC block 上自动叠加 pre-PC 虚线） | mean |
| `min_singular_value` | 最小奇异值（PC block 上自动叠加 pre-PC 虚线，y 轴 log scale） | mean |

`max_singular_value` / `min_singular_value` / `condition_number` 会自动在 PC 启用的 block 上叠加 pre-PC 虚线作为参考；不启用 PC 的实验只画一条线。`min_singular_value` 默认使用 log scale 的 y 轴。

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
