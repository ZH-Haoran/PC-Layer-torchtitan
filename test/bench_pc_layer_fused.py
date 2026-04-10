"""
Benchmark: Triton fused ops vs pure torch ops for PC layer.

Measures forward time, backward time, and peak memory for both implementations
across different matrix sizes and pc_levels.
"""

import torch
import torch.nn as nn
import time
from dataclasses import dataclass

# ── Reference (pure torch, original code) ────────────────────────────

_POLAR_EXPRESS_COEFFS = [
    (7.2086, -15.5131, 9.0178),
    (3.9623, -2.5813, 0.4542),
    (3.9466, -2.5765, 0.4544),
    (3.8991, -2.5671, 0.4566),
    (3.7186, -2.5308, 0.4653),
    (3.1390, -2.3073, 0.4733),
    (2.1715, -1.5246, 0.3885),
    (1.8648, -1.2224, 0.3577),
]


def ref_preconditioner(weight, pc_level, norm_eps=1e-7, scale=1.0,
                       recover_norm=True, gamma=None):
    W_norm = weight.norm() + norm_eps
    W_hat = weight / W_norm
    r, c = W_hat.shape

    if r >= c:
        I = torch.eye(c, device=weight.device, dtype=weight.dtype)
        wtw = W_hat.t().mm(W_hat)
        if pc_level == 1:
            W_hat = W_hat.mm(1.507 * I - 0.507 * wtw)
        elif pc_level == 2:
            W_hat = W_hat.mm(2.083 * I + wtw.mm(-1.643 * I + 0.560 * wtw))
        elif pc_level == 3:
            W_hat = W_hat.mm(2.909 * I + wtw.mm(-4.649 * I + wtw.mm(4.023 * I - 1.283 * wtw)))
        elif pc_level == 4:
            W_hat = W_hat.mm(3.625 * I + wtw.mm(-9.261 * I + wtw.mm(14.097 * I + wtw.mm(-10.351 * I + 2.890 * wtw))))
        elif pc_level == 5:
            for i, (a, b, c_coeff) in enumerate(_POLAR_EXPRESS_COEFFS):
                if i > 0:
                    wtw = W_hat.t().mm(W_hat)
                W_hat = a * W_hat + W_hat.mm(b * wtw + c_coeff * wtw.mm(wtw))
    else:
        I = torch.eye(r, device=weight.device, dtype=weight.dtype)
        wwt = W_hat.mm(W_hat.t())
        if pc_level == 1:
            W_hat = (1.507 * I - 0.507 * wwt).mm(W_hat)
        elif pc_level == 2:
            W_hat = (2.083 * I + wwt.mm(-1.643 * I + 0.560 * wwt)).mm(W_hat)
        elif pc_level == 3:
            W_hat = (2.909 * I + wwt.mm(-4.649 * I + wwt.mm(4.023 * I - 1.283 * wwt))).mm(W_hat)
        elif pc_level == 4:
            W_hat = (3.625 * I + wwt.mm(-9.261 * I + wwt.mm(14.097 * I + wwt.mm(-10.351 * I + 2.890 * wwt)))).mm(W_hat)
        elif pc_level == 5:
            for i, (a, b, c_coeff) in enumerate(_POLAR_EXPRESS_COEFFS):
                if i > 0:
                    wwt = W_hat.mm(W_hat.t())
                W_hat = a * W_hat + (b * wwt + c_coeff * wwt.mm(wwt)).mm(W_hat)

    W_pc = W_hat * scale
    if recover_norm:
        W_pc = W_pc * W_norm.detach()
    if gamma is not None:
        W_pc = W_pc * gamma
    return W_pc


# ── Fused (Triton) ──────────────────────────────────────────────────

from torchtitan.pc_layer.pc_layer import PCTransform


@dataclass
class _BenchConfig:
    pc_level: int = 1
    pc_norm_type: str = "F"
    pc_norm_eps: float = 1e-7
    scale_constant: float = 1.0
    recover_w_norm: bool = True
    learnable_gamma: bool = True
    gamma_init_value: float = 1.0
    pc_op_beta: float = 0.0
    power_iter: int = 5


def benchmark_fn(fn, warmup=10, repeat=50):
    """Benchmark a function, return median time in ms."""
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times.sort()
    return times[len(times) // 2]  # median


def measure_peak_memory(fn, repeat=5):
    """Run fn and return peak GPU memory delta in MB."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()

    for _ in range(repeat):
        fn()
        torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()
    return (peak - base) / (1024 * 1024)


def run_benchmark(r, c, pc_level, dtype=torch.float32):
    torch.manual_seed(42)
    W_data = torch.randn(r, c, device="cuda", dtype=dtype)
    W_data = W_data / W_data.norm()
    gamma_val = torch.tensor([1.5], device="cuda", dtype=dtype)

    cfg = _BenchConfig(pc_level=pc_level, learnable_gamma=True)
    pc = PCTransform(cfg)

    # ── Forward time ──
    def fwd_ref():
        W = W_data.clone().requires_grad_(True)
        g = gamma_val.clone().requires_grad_(True)
        return ref_preconditioner(W, pc_level, gamma=g)

    def fwd_fused():
        W = W_data.clone().requires_grad_(True)
        g = gamma_val.clone().requires_grad_(True)
        return pc(W, gamma=g)

    t_fwd_ref = benchmark_fn(fwd_ref)
    t_fwd_fused = benchmark_fn(fwd_fused)

    # ── Forward + Backward time ──
    def fwd_bwd_ref():
        W = W_data.clone().requires_grad_(True)
        g = gamma_val.clone().requires_grad_(True)
        out = ref_preconditioner(W, pc_level, gamma=g)
        out.sum().backward()

    def fwd_bwd_fused():
        W = W_data.clone().requires_grad_(True)
        g = gamma_val.clone().requires_grad_(True)
        out = pc(W, gamma=g)
        out.sum().backward()

    t_fb_ref = benchmark_fn(fwd_bwd_ref)
    t_fb_fused = benchmark_fn(fwd_bwd_fused)

    # ── Peak memory (forward only) ──
    torch.cuda.empty_cache()
    mem_ref = measure_peak_memory(fwd_ref)
    torch.cuda.empty_cache()
    mem_fused = measure_peak_memory(fwd_fused)

    # ── Peak memory (forward + backward) ──
    torch.cuda.empty_cache()
    mem_fb_ref = measure_peak_memory(fwd_bwd_ref)
    torch.cuda.empty_cache()
    mem_fb_fused = measure_peak_memory(fwd_bwd_fused)

    return {
        "fwd_ref_ms": t_fwd_ref,
        "fwd_fused_ms": t_fwd_fused,
        "fwd_speedup": t_fwd_ref / t_fwd_fused,
        "fb_ref_ms": t_fb_ref,
        "fb_fused_ms": t_fb_fused,
        "fb_speedup": t_fb_ref / t_fb_fused,
        "mem_fwd_ref_mb": mem_ref,
        "mem_fwd_fused_mb": mem_fused,
        "mem_fwd_save_pct": (1 - mem_fused / mem_ref) * 100 if mem_ref > 0 else 0,
        "mem_fb_ref_mb": mem_fb_ref,
        "mem_fb_fused_mb": mem_fb_fused,
        "mem_fb_save_pct": (1 - mem_fb_fused / mem_fb_ref) * 100 if mem_fb_ref > 0 else 0,
    }


if __name__ == "__main__":
    configs = [
        # (rows, cols, pc_level)
        (512, 256, 1),
        (512, 256, 4),
        (512, 256, 5),
        (2048, 1024, 1),
        (2048, 1024, 4),
        (2048, 1024, 5),
        (4096, 4096, 1),
        (4096, 4096, 4),
        (4096, 4096, 5),
        # wide
        (256, 512, 4),
        (1024, 2048, 4),
    ]

    print(f"{'Shape':>14s} {'Lvl':>3s} | "
          f"{'Fwd Ref':>8s} {'Fwd Fuse':>8s} {'Speedup':>7s} | "
          f"{'F+B Ref':>8s} {'F+B Fuse':>8s} {'Speedup':>7s} | "
          f"{'Mem Fwd Ref':>11s} {'Mem Fwd F':>9s} {'Save%':>6s} | "
          f"{'Mem FB Ref':>10s} {'Mem FB F':>9s} {'Save%':>6s}")
    print("-" * 155)

    for r, c, lvl in configs:
        res = run_benchmark(r, c, lvl)
        shape_str = f"{r}x{c}"
        print(f"{shape_str:>14s} {lvl:>3d} | "
              f"{res['fwd_ref_ms']:>7.3f}ms {res['fwd_fused_ms']:>7.3f}ms {res['fwd_speedup']:>6.2f}x | "
              f"{res['fb_ref_ms']:>7.3f}ms {res['fb_fused_ms']:>7.3f}ms {res['fb_speedup']:>6.2f}x | "
              f"{res['mem_fwd_ref_mb']:>10.1f}MB {res['mem_fwd_fused_mb']:>8.1f}MB {res['mem_fwd_save_pct']:>5.1f}% | "
              f"{res['mem_fb_ref_mb']:>9.1f}MB {res['mem_fb_fused_mb']:>8.1f}MB {res['mem_fb_save_pct']:>5.1f}%")
