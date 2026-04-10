# PC Layer Triton Operator: Summary

## 做了什么

把 `torchtitan/pc_layer/pc_layer.py` 中 PCTransform 的 element-wise 操作用 Triton kernel 融合，减少 kernel launch 次数和显存带宽消耗。

### 新增的 Triton kernel

| Kernel | 计算 | 用途 |
|--------|------|------|
| `_add_scaled_identity_kernel` | `αI + βM`（方阵） | 多项式求值（level 1-4） |
| `_axpby_kernel` | `αA + βB`（任意同形 tensor） | Newton-Schulz 迭代（level 5） |

每个 kernel 配有 `torch.autograd.Function` 包装器，backward 是解析式的（标量乘 grad）。

### PCTransform 改动

- `preconditionertall/wide`：用 `fused_add_scaled_identity` 替换 `scalar*I + scalar*M` 链式表达式，去掉 `torch.eye()`
- `apply_preconditioner`：合并 `scale_constant × W_norm × gamma` 为一次标量乘
- `modified_F` 路径完全不动

### 未改动

`LearnableGamma`、`PCLinear`、`pc_normalize`、op-norm 相关逻辑、module-level utilities 全部不变。

---

## Benchmark 结果

测试环境：单卡，float32。

### 速度

| 矩阵尺寸 | 效果 |
|----------|------|
| 小（512×256） | Triton 反而慢 0.6-0.8x，kernel launch 开销占主导 |
| 中（2048×1024） | 基本持平 ~1.05x |
| 大（4096×4096） | 基本持平 ~1.01-1.03x，GEMM 占绝大部分时间 |

**结论**：速度上基本持平。element-wise 操作在整体计算中占比很小，GEMM（cuBLAS）是瓶颈。

### Forward 显存

| 矩阵尺寸 | Level | 原始 | 融合后 | 节省 |
|----------|-------|------|--------|------|
| 4096×4096 | 1 | 512MB | 320MB | **37.5%** |
| 4096×4096 | 4 | 704MB | 576MB | **18.2%** |
| 2048×1024 | 4 | 64MB | 52MB | **18.7%** |

**结论**：forward 峰值显存有明显改善（18-37%），来源是不再分配 `scalar*I`、`scalar*M` 等中间 tensor。

### Forward+Backward 显存

几乎无差别。原因：GEMM 的 backward 仍需保存操作数，这部分占大头。如果 torchtitan 开了 activation checkpointing（包在 TransformerBlock 上），中间量本来就会被重算，所以 backward 显存由 AC 管理。

### 端到端训练对比（2× H200, Llama-2 271M, pc_level=4, F-norm, 100 steps）

测试环境：2× NVIDIA H200, FSDP2 data-parallel, `batch_size=5, seq_len=8192, grad_accumulation_steps=8`, AdamW lr=8.485e-4。

| | Baseline（无 Triton） | Triton Kernel |
|---|---|---|
| **总 wall time** | 3m58s | 4m08s |
| **启动/初始化** | ~2m03s | ~2m13s |
| **Step 1（含编译）** | 6.2s | 7.7s |
| **Step 2-100 稳态** | ~1m41s（99 steps） | ~1m40s（99 steps） |
| **稳态每步** | ~1.02s/step | ~1.01s/step |
| **稳态 wps** | ~156k-167k | ~148k-167k |
| **峰值显存 active** | 42.35 GiB | 42.35 GiB |
| **峰值显存 reserved** | 44.61 GiB | 44.61 GiB |
| **Loss (step 1)** | 10.8524 | 10.8523 |
| **Loss (step 100)** | 5.4765 | 5.4811 |

**结论**：

- **稳态速度持平**：~1.01 vs ~1.02 s/step，差异在噪声范围内。element-wise 操作在整体 forward+backward 中占比极小，GEMM 是绝对瓶颈。
- **峰值显存一致**：forward+backward 中 GEMM 操作数保存占主导，与单卡 micro-benchmark 结论一致。
- **数值正确**：Loss 在两版本间几乎完全一致（step 1 差 0.0001，step 100 差 0.005），确认 Triton kernel 数值等价。
- **Triton autotune 开销**：首次 step 多 ~1.5s（一次性编译），后续零开销。

---

## DTensor 兼容性

### 问题

FSDP2 `fully_shard` 后，forward 时权重是 `DTensor(Replicate)`。Triton kernel 只能处理 plain tensor。

### 方案：`to_local()` → Triton → `from_local()`

对 DTensor 输入，取出 local tensor 跑 Triton kernel，再包回 DTensor：

```python
def fused_add_scaled_identity(M, alpha, beta):
    if isinstance(M, DTensor):
        local_out = _AddScaledIdentityOp.apply(M.to_local(), alpha, beta)
        return DTensor.from_local(local_out, device_mesh=M.device_mesh,
                                  placements=M.placements, run_check=False)
    return _AddScaledIdentityOp.apply(M, alpha, beta)
```

### 验证结果

| 测试 | Forward | Backward |
|------|---------|----------|
| `add_scaled_identity` DTensor(Replicate) | OK | OK |
| `axpby` DTensor(Replicate) | OK | OK |
| PCTransform e2e DTensor(Replicate), `pc_norm_type="none"` | OK | OK |
| PCTransform e2e DTensor(Replicate), `pc_norm_type="F"` + `recover_w_norm` | OK | OK |

`to_local()` 对 `Replicate` 无通信、无拷贝（返回同一块显存的 view），autograd 梯度正确回传。

### 踩坑：原始 fallback 代码有 bug

最初的 DTensor fallback 写的是：

```python
I = torch.eye(N, device=M.device, dtype=M.dtype)
return alpha * I + beta * M  # M 是 DTensor
```

PyTorch **不允许** plain tensor 和 DTensor 混合运算，直接报错。所以 `to_local` 方案不只是优化，也是在修 bug。

### 踩坑 & 修复：`weight.norm()` 的 backward 在 DTensor 下不可用

Frobenius norm (`pc_norm_type="F"`) 的 backward 内部用了 `aten.masked_fill_`，DTensor 没有注册这个 op 的 sharding strategy，导致 backward 报错：

```
NotImplementedError: Operator aten.masked_fill_.Scalar does not have a sharding strategy registered.
```

此外，即使绕过 `masked_fill_` 问题，如果 `W_norm` 是 plain tensor 而 `weight` 是 DTensor，后续 `weight / W_norm` 和 `W_preconditioned * combined_scale`（`combined_scale` 包含 `W_norm.detach()`）会触发：

```
RuntimeError: aten.mul.Tensor: got mixed torch.Tensor and DTensor
```

**修复方案**：对 DTensor 输入，先 `to_local()`（Replicate 零拷贝）或 `full_tensor()`（Shard 需 all-gather），在 local tensor 上算 norm，再用 `DTensor.from_local(..., placements=[Replicate()])` 包回 DTensor，保证整条计算链都在 DTensor 语义下：

```python
elif model_config.pc_norm_type == "F":
    if isinstance(weight, DTensor):
        if all(isinstance(p, Replicate) for p in weight.placements):
            w_local = weight.to_local()
        else:
            w_local = weight.full_tensor()
        W_norm_local = w_local.norm() + model_config.pc_norm_eps
        W_norm = DTensor.from_local(
            W_norm_local,
            device_mesh=weight.device_mesh,
            placements=[Replicate()],
            run_check=False,
        )
    else:
        W_norm = weight.norm() + model_config.pc_norm_eps
```

**验证**：`pc_norm_type="F"` + `recover_w_norm=True` + DTensor(Replicate) 的 forward/backward 均与 plain tensor 参考对齐（atol=1e-4）。

### TP（Tensor Parallelism）分析

当 `tensor_parallel_degree > 1` 时：

- `W` 是 `Shard(0)` 或 `Shard(1)`
- `wtw = W.t().mm(W)` → DTensor dispatch 做 all-reduce → 结果是 `Replicate`
- 多项式操作在 `Replicate` gram 矩阵上 → `to_local()` 安全
- `fused_axpby(W, WT, ...)` 两者同为 `Shard(k)` → element-wise 在每个 shard 上独立做，正确

目前仅在 1 卡上验证了 `Replicate`。多卡 TP（`Shard`）场景需要在多卡环境下进一步验证。

### 多卡 FSDP2 验证

见上方「端到端训练对比」。2× H200 FSDP2 下 100 steps 正常完成，Triton kernel + DTensor `to_local()/from_local()` 在分布式下正常工作，与 baseline 数值一致。

---

## 关于 `@triton.autotune`

kernel 开头的 `@triton.autotune` 会在首次调用时对多个配置（BLOCK_SIZE=1024/2048/4096）做 benchmark，选出最快的缓存起来。

- 首次调用多花几秒（一次性）
- 之后零开销
- 不影响数值正确性

---

## 如何复现测试

### 单元测试

```bash
# 正确性测试（46 cases，覆盖所有 level/shape/norm）
PYTHONPATH=. python test/test_pc_layer_fused.py

# 性能 benchmark（单卡，forward/backward 速度和显存）
PYTHONPATH=. python test/bench_pc_layer_fused.py

# DTensor autograd 兼容性（需要 1 卡 torchrun）
PYTHONPATH=. torchrun --nproc_per_node=1 test/test_dtensor_autograd.py
```

### 端到端训练对比

**Triton 版本**（`/data_hss/torchtitan-clean`）：

```bash
cd /data_hss/torchtitan-clean
NGPU=2 bash run_llama_2_pc_layer.sh --training.steps 100 --training.warmup_steps 10
```

**Baseline 版本**（`/data_hss/torchtitian`，无 Triton kernel）：

```bash
cd /data_hss/torchtitian
# 需要先准备匹配的 config（从 clean 版复制，补上 baseline 独有字段）
# 已存放在 train_configs/_bench_100steps.toml
NGPU=2 bash run_llama_2_pc_layer.sh --job.config_file ./train_configs/_bench_100steps.toml --training.steps 100 --training.warmup_steps 10
```

两边使用相同的 config：Llama-2 271M, `pc_level=4`, `pc_norm_type='F'`, `recover_w_norm=true`, `learnable_gamma=true`, `batch_size=5`, `seq_len=8192`, `grad_accumulation_steps=8`, AdamW lr=8.485e-4, 2× GPU FSDP2。

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `torchtitan/pc_layer/pc_layer.py` | 主要改动：Triton kernel + autograd wrapper + PCTransform 重构 |
| `test/test_pc_layer_fused.py` | 正确性测试（46 cases，覆盖所有 level/shape/norm） |
| `test/bench_pc_layer_fused.py` | 性能 benchmark |
| `test/test_dtensor_autograd.py` | DTensor autograd 兼容性验证 |

---

## 未来可选优化

1. **Frobenius norm 融合**：将 `weight.norm() + eps` 和 `weight / W_norm` 融合成一个 Triton kernel（global reduction + broadcast division）。对大矩阵可省一次完整读写。目前 `pc_norm_type="F"` 的 DTensor 兼容性已通过 `to_local()` 方案解决，Triton 融合是纯性能优化。
2. **多卡 TP（Shard）场景验证**：当前 Triton kernel + `to_local()` 方案仅验证了 FSDP2 Replicate 场景。`tensor_parallel_degree > 1` 时权重为 `Shard(0/1)`，需在多卡 TP 环境下验证 `fused_axpby` 对 Shard DTensor 的正确性。

---

# 第二轮优化：`torch.addmm` Horner 多项式融合

## 背景：第一轮（Triton）为何没有改进

经过分析，Triton kernel 方案存在三个根本问题：

1. **`@triton.autotune` 首次编译开销**：step 1 多出 ~1.5s（7.7s vs 6.2s baseline）
2. **Triton kernel launch 延迟比 PyTorch 内置算子更高**，加上自定义 `torch.autograd.Function` 的 Python wrapper 开销，element-wise 算子的速度反而更慢
3. **优化对象选错了**：被融合的 `αI + βM` 在整体 forward+backward 中占比极小，GEMM（cuBLAS）才是瓶颈，融合 element-wise 无实质收益

## 新方案：`torch.addmm` Horner 求值

### 核心思路

Horner 多项式每一步的形式是 `αI + (gram @ T)`，恰好对应 `torch.addmm` 的语义：

```
torch.addmm(eye, gram, T, beta=α, alpha=1)  =  α·I + 1·(gram @ T)
```

这将原本的**两个** kernel launch（GEMM + element-wise add）合并为**一个** cuBLAS 调用，不引入任何额外分配。

### 实现

新增 `_HORNER_COEFFS` 查找表和 `_horner_poly()` 函数，统一处理 level 1-4：

```python
_HORNER_COEFFS = {
    1: [(-0.507, 1.507)],
    2: [(0.560, -1.643), (1.0, 2.083)],
    3: [(-1.283, 4.023), (1.0, -4.649), (1.0, 2.909)],
    4: [(2.890, -10.351), (1.0, 14.097), (1.0, -9.261), (1.0, 3.625)],
}

def _horner_poly(gram, eye, coeffs):
    beta_0, alpha_0 = coeffs[0]
    S = alpha_0 * eye + beta_0 * gram          # 第一步：纯 scalar 运算
    for beta_k, alpha_k in coeffs[1:]:
        S = torch.addmm(eye, gram, S, beta=alpha_k, alpha=beta_k)  # GEMM + 加对角线，一次调用
    return S
```

Level 5（Newton-Schulz）改用 `torch.addmm` + `torch.add`：

```python
T = torch.addmm(gram, gram, gram, beta=b, alpha=c_coeff)  # b·gram + c·gram²
WT = weight.mm(T)
weight = torch.add(WT, weight, alpha=a)                    # WT + a·weight
```

### 与旧方案对比

| | Triton 方案 | `addmm` 方案 |
|--|--|--|
| 额外依赖 | `triton`，需编译 | 无，纯 PyTorch |
| Step 1 开销 | +1.5s autotune | 无 |
| DTensor 兼容 | 需要 `to_local()/from_local()` wrapper | 原生支持，无需 wrapper |
| 代码行数 | +160 行（kernel + wrapper） | +25 行 |
| 显存节省来源 | 消除 `scalar*I`、`scalar*M` 中间 tensor | 同上，且不需要 eye 常驻 |

`pc_normalize` 中 `weight.norm()` backward 的 DTensor bug fix 予以保留（与 Triton 无关，是独立修复）。

---

## Benchmark 结果（第二轮）

测试环境：2× NVIDIA H200, FSDP2, `batch_size=5, seq_len=8192, grad_accumulation_steps=8`, AdamW lr=8.485e-4, Llama-2 271M, `pc_level=4`, `pc_norm_type='F'`。

### 单卡 micro-benchmark（forward-only, float32）

| Shape | Level | Fwd Speedup | Fwd 显存节省 | F+B Speedup |
|-------|-------|------------|------------|-------------|
| 512×256 | 1 | 0.95x | **30.7%** | 0.97x |
| 512×256 | 4 | **1.56x** | **18.7%** | 0.99x |
| 512×256 | 5 | **1.71x** | 7.3% | **1.11x** |
| 2048×1024 | 1 | **1.21x** | **30.8%** | 1.01x |
| 2048×1024 | 4 | **1.26x** | **18.7%** | 1.00x |
| 2048×1024 | 5 | **1.06x** | 7.3% | **1.12x** |
| 4096×4096 | 4 | **1.01x** | **18.2%** | 1.00x |
| 4096×4096 | 5 | 1.01x | 6.9% | 1.00x |

forward-only 有明显提速（小/中矩阵 1.2-1.7x）；forward+backward 与 baseline 持平（GEMM backward 是瓶颈）。

### 端到端训练对比（AC 关闭）

| | Baseline | addmm（本轮） | Triton（上轮） |
|--|--|--|--|
| **Step 1** | 6.27s | **5.84s** | 7.7s |
| **稳态 s/step** | 1.02 | **0.99** | 1.01 |
| **Peak active** | 42.35 GiB | 42.35 GiB | 42.35 GiB |
| **Peak reserved** | 44.61 GiB | 44.61 GiB | 44.61 GiB |
| **Loss @100** | 5.4765 | 5.4912 | 5.4811 |

### 端到端训练对比（AC 开启，`mode='full'`）

#### 表 1：Baseline 自身，AC 关闭 vs AC 开启

| | AC off | AC on (full) | 变化 |
|--|--------|-------------|------|
| **Step 1** | 6.27s | 6.27s | — |
| **稳态 s/step** | 1.02 | 1.35 | **+32%** |
| **Peak active** | 42.35 GiB | 18.31 GiB | **−56.8%** |
| **Peak reserved** | 44.61 GiB | 21.19 GiB | **−52.5%** |
| **Loss @100** | 5.4765 | 5.4802 | ≈ |

AC 用 +32% 时间换取 ~57% 的显存节省。

#### 表 2：AC 开启后，addmm vs Baseline

| | addmm（本轮） | Baseline | 差异 |
|--|--|--|--|
| **Step 1** | **5.79s** | 6.27s | **−7.7%** |
| **稳态 s/step** | **1.29** | 1.35 | **−4.3%** |
| **Peak active** | 18.31 GiB | 18.31 GiB | 相同 |
| **Peak reserved** | 21.19 GiB | 21.19 GiB | 相同 |
| **Loss @100** | 5.4795 | 5.4802 | ≈ |

AC 开启后 addmm 的速度优势比 AC 关闭时更明显（4.3% vs ~3%）：AC 重算 forward，`addmm` 消除的 kernel launch 和中间分配被多执行一次，优化效果叠加放大。

---

## DTensor 兼容性（第二轮）

`torch.addmm` 和 `torch.add` 是标准 ATen op，DTensor 原生支持，无需任何 wrapper。`pc_normalize` 中 `weight.norm()` backward 的 `to_local()` 修复予以保留。

| 测试 | Forward | Backward |
|------|---------|----------|
| `torch.addmm` DTensor(Replicate) | OK | OK |
| `torch.add(alpha=...)` DTensor(Replicate) | OK | OK |
| PCTransform e2e DTensor(Replicate), `pc_norm_type="none"` | OK | OK |
| PCTransform e2e DTensor(Replicate), `pc_norm_type="F"` + `recover_w_norm` | OK | OK |

---

## 如何复现测试（第二轮）

```bash
# 正确性测试（46 cases）
PYTHONPATH=. python test/test_pc_layer_fused.py

# 性能 benchmark（单卡）
PYTHONPATH=. python test/bench_pc_layer_fused.py

# DTensor 兼容性
PYTHONPATH=. torchrun --nproc_per_node=1 test/test_dtensor_autograd.py

# 端到端（AC 关闭）
cd /data_hss/torchtitan-clean
NGPU=2 bash run_llama_2_pc_layer.sh --training.steps 100 --training.warmup_steps 10

# 端到端（AC 开启，需先在 train_configs/llama2_pc_layer.toml 中设 mode='full'）
NGPU=2 bash run_llama_2_pc_layer.sh --training.steps 100 --training.warmup_steps 10
```

## 文件清单（第二轮变更）

| 文件 | 说明 |
|------|------|
| `torchtitan/pc_layer/pc_layer.py` | 移除全部 Triton 代码；新增 `_HORNER_COEFFS`、`_horner_poly()`；level 5 改用 `torch.addmm`+`torch.add`；保留 DTensor norm bug fix |
| `test/test_pc_layer_fused.py` | 移除 Triton kernel 直接测试；新增 `test_horner_poly_forward/backward`（共 46 cases） |
| `test/test_dtensor_autograd.py` | 更新为测试 `torch.addmm`/`torch.add` 的 DTensor 兼容性 |

## 未来可选优化（更新）

1. **Frobenius norm 融合**：将 `weight.norm() + eps` 和 `weight / W_norm` 融合为单个 kernel（global reduction + broadcast division），对大矩阵可省一次完整读写。
2. **多卡 TP（Shard）场景验证**：目前仅验证了 FSDP2 Replicate 场景，`tensor_parallel_degree > 1` 时需在多卡 TP 环境下进一步验证。
