"""
Correctness test for fused Triton PC layer operators.

Compares the new fused implementation (Triton kernels) against a pure-torch
reference for all pc_levels (0-5), both tall and wide matrices, and both
F and op norm types. Checks forward values and backward gradients.
"""

import pytest
import torch
from dataclasses import dataclass
from typing import Optional


# ── Reference implementation (pure torch, no Triton) ─────────────────

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


def _ref_preconditionertall(weight, pc_level, gram=None):
    if pc_level == 0:
        return weight
    _, c = weight.shape
    I = torch.eye(c, device=weight.device, dtype=weight.dtype)
    wtw = gram if gram is not None else weight.t().mm(weight)
    if pc_level == 1:
        weight = weight.mm(1.507 * I - 0.507 * wtw)
    elif pc_level == 2:
        weight = weight.mm(2.083 * I + wtw.mm(-1.643 * I + 0.560 * wtw))
    elif pc_level == 3:
        weight = weight.mm(2.909 * I + wtw.mm(-4.649 * I + wtw.mm(4.023 * I - 1.283 * wtw)))
    elif pc_level == 4:
        weight = weight.mm(3.625 * I + wtw.mm(-9.261 * I + wtw.mm(14.097 * I + wtw.mm(-10.351 * I + 2.890 * wtw))))
    elif pc_level == 5:
        for i, (a, b, c_coeff) in enumerate(_POLAR_EXPRESS_COEFFS):
            if i > 0:
                wtw = weight.t().mm(weight)
            weight = a * weight + weight.mm(b * wtw + c_coeff * wtw.mm(wtw))
    return weight


def _ref_preconditionerwide(weight, pc_level, gram=None):
    if pc_level == 0:
        return weight
    r, _ = weight.shape
    I = torch.eye(r, device=weight.device, dtype=weight.dtype)
    wwt = gram if gram is not None else weight.mm(weight.t())
    if pc_level == 1:
        weight = (1.507 * I - 0.507 * wwt).mm(weight)
    elif pc_level == 2:
        weight = (2.083 * I + wwt.mm(-1.643 * I + 0.560 * wwt)).mm(weight)
    elif pc_level == 3:
        weight = (2.909 * I + wwt.mm(-4.649 * I + wwt.mm(4.023 * I - 1.283 * wwt))).mm(weight)
    elif pc_level == 4:
        weight = (3.625 * I + wwt.mm(-9.261 * I + wwt.mm(14.097 * I + wwt.mm(-10.351 * I + 2.890 * wwt)))).mm(weight)
    elif pc_level == 5:
        for i, (a, b, c_coeff) in enumerate(_POLAR_EXPRESS_COEFFS):
            if i > 0:
                wwt = weight.mm(weight.t())
            weight = a * weight + (b * wwt + c_coeff * wwt.mm(wwt)).mm(weight)
    return weight


def _ref_apply_preconditioner(weight, pc_level, norm_type, norm_eps, scale_constant,
                               recover_w_norm, gamma, op_norm):
    # normalize
    if norm_type == "F":
        W_norm = weight.norm() + norm_eps
    elif norm_type == "op":
        W_norm = op_norm
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")
    W_hat = weight / W_norm

    # precondition
    r, c = W_hat.shape
    if r >= c:
        W_pc = _ref_preconditionertall(W_hat, pc_level)
    else:
        W_pc = _ref_preconditionerwide(W_hat, pc_level)

    # post-scale
    W_pc = W_pc * scale_constant
    if recover_w_norm:
        W_pc = W_pc * W_norm.detach()
    if gamma is not None:
        W_pc = W_pc * gamma

    return W_pc


# ── Minimal config dataclass for PCTransform ─────────────────────────

@dataclass
class _TestConfig:
    pc_level: int = 1
    pc_norm_type: str = "F"
    pc_norm_eps: float = 1e-7
    scale_constant: float = 1.0
    recover_w_norm: bool = True
    learnable_gamma: bool = False
    gamma_init_value: float = 1.0
    pc_op_beta: float = 0.0
    power_iter: int = 5


# ── Import fused implementation ──────────────────────────────────────

from torchtitan.pc_layer.pc_layer import PCTransform


# ── Tests ────────────────────────────────────────────────────────────

def _make_weight(r, c, device="cuda", dtype=torch.float32):
    W = torch.randn(r, c, device=device, dtype=dtype)
    W = W / W.norm()  # normalize so polynomial doesn't blow up
    return W


@pytest.mark.parametrize("pc_level", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (48, 48)])
def test_forward_F_norm(pc_level, shape):
    """Forward values match between fused and reference (Frobenius norm)."""
    torch.manual_seed(42)
    r, c = shape
    W = _make_weight(r, c)

    cfg = _TestConfig(pc_level=pc_level, pc_norm_type="F",
                      recover_w_norm=True, scale_constant=1.0,
                      learnable_gamma=False)

    ref = _ref_apply_preconditioner(W, pc_level, "F", cfg.pc_norm_eps,
                                     cfg.scale_constant, True, None, None)

    pc = PCTransform(cfg)
    out = pc(W)

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("pc_level", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("shape", [(64, 32), (32, 64)])
def test_forward_op_norm(pc_level, shape):
    """Forward values match between fused and reference (operator norm)."""
    torch.manual_seed(42)
    r, c = shape
    W = _make_weight(r, c)
    # Simulate precomputed op_norm
    op_norm = torch.linalg.matrix_norm(W, ord=2) + 1e-7

    cfg = _TestConfig(pc_level=pc_level, pc_norm_type="op",
                      recover_w_norm=True, scale_constant=1.0,
                      learnable_gamma=False)

    ref = _ref_apply_preconditioner(W, pc_level, "op", cfg.pc_norm_eps,
                                     cfg.scale_constant, True, None, op_norm)

    pc = PCTransform(cfg)
    out = pc(W, op_norm=op_norm)

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("pc_level", [1, 3, 5])
@pytest.mark.parametrize("shape", [(64, 32), (32, 64)])
def test_backward_grad_weight(pc_level, shape):
    """Gradient w.r.t. weight matches between fused and reference."""
    torch.manual_seed(42)
    r, c = shape
    W_data = _make_weight(r, c)

    cfg = _TestConfig(pc_level=pc_level, pc_norm_type="F",
                      recover_w_norm=True, learnable_gamma=False)

    # Reference gradient
    W_ref = W_data.clone().requires_grad_(True)
    out_ref = _ref_apply_preconditioner(W_ref, pc_level, "F", cfg.pc_norm_eps,
                                         cfg.scale_constant, True, None, None)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    grad_ref = W_ref.grad.clone()

    # Fused gradient
    W_fused = W_data.clone().requires_grad_(True)
    pc = PCTransform(cfg)
    out_fused = pc(W_fused)
    loss_fused = out_fused.sum()
    loss_fused.backward()
    grad_fused = W_fused.grad.clone()

    torch.testing.assert_close(grad_fused, grad_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("pc_level", [1, 4])
def test_backward_grad_gamma(pc_level):
    """Gradient w.r.t. learnable gamma matches between fused and reference."""
    torch.manual_seed(42)
    r, c = 64, 32
    W_data = _make_weight(r, c)
    gamma_val = torch.tensor([1.5], device="cuda", dtype=torch.float32)

    cfg = _TestConfig(pc_level=pc_level, pc_norm_type="F",
                      recover_w_norm=True, learnable_gamma=True)

    # Reference
    W_ref = W_data.clone()
    g_ref = gamma_val.clone().requires_grad_(True)
    out_ref = _ref_apply_preconditioner(W_ref, pc_level, "F", cfg.pc_norm_eps,
                                         cfg.scale_constant, True, g_ref, None)
    out_ref.sum().backward()
    grad_gamma_ref = g_ref.grad.clone()

    # Fused
    W_fused = W_data.clone()
    g_fused = gamma_val.clone().requires_grad_(True)
    pc = PCTransform(cfg)
    out_fused = pc(W_fused, gamma=g_fused)
    out_fused.sum().backward()
    grad_gamma_fused = g_fused.grad.clone()

    torch.testing.assert_close(grad_gamma_fused, grad_gamma_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("pc_level", [1, 2, 3, 4, 5])
def test_fused_add_scaled_identity_kernel(pc_level):
    """Directly test the fused_add_scaled_identity kernel."""
    from torchtitan.pc_layer.pc_layer import fused_add_scaled_identity

    torch.manual_seed(42)
    N = 64
    M = torch.randn(N, N, device="cuda", dtype=torch.float32)
    alpha, beta = 3.14, -2.71

    ref = alpha * torch.eye(N, device="cuda") + beta * M
    out = fused_add_scaled_identity(M, alpha, beta)

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_fused_axpby_kernel():
    """Directly test the fused_axpby kernel."""
    from torchtitan.pc_layer.pc_layer import fused_axpby

    torch.manual_seed(42)
    A = torch.randn(128, 64, device="cuda", dtype=torch.float32)
    B = torch.randn(128, 64, device="cuda", dtype=torch.float32)
    alpha, beta = 0.5, -1.3

    ref = alpha * A + beta * B
    out = fused_axpby(A, B, alpha, beta)

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_fused_add_scaled_identity_backward():
    """Backward of fused_add_scaled_identity produces correct gradients."""
    from torchtitan.pc_layer.pc_layer import fused_add_scaled_identity

    torch.manual_seed(42)
    N = 32
    M = torch.randn(N, N, device="cuda", requires_grad=True)
    alpha, beta = 2.0, 0.7

    out = fused_add_scaled_identity(M, alpha, beta)
    out.sum().backward()

    # d(alpha*I + beta*M)/dM = beta
    expected_grad = torch.full_like(M, beta)
    torch.testing.assert_close(M.grad, expected_grad, atol=1e-5, rtol=1e-5)


def test_fused_axpby_backward():
    """Backward of fused_axpby produces correct gradients."""
    from torchtitan.pc_layer.pc_layer import fused_axpby

    torch.manual_seed(42)
    A = torch.randn(32, 64, device="cuda", requires_grad=True)
    B = torch.randn(32, 64, device="cuda", requires_grad=True)
    alpha, beta = 1.5, -0.8

    out = fused_axpby(A, B, alpha, beta)
    out.sum().backward()

    torch.testing.assert_close(A.grad, torch.full_like(A, alpha), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(B.grad, torch.full_like(B, beta), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
