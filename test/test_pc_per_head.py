"""Numerical equivalence tests for per-head Q/K/V preconditioning."""
import torch
import pytest

from torchtitan.pc_layer.pc_layer import PCTransform, PCLinear


class _Cfg:
    def __init__(self, **kw):
        self.pc_norm_type = kw.get("pc_norm_type", "F")
        self.pc_norm_eps = kw.get("pc_norm_eps", 1e-7)
        self.pc_level = kw.get("pc_level", 2)
        self.scale_constant = kw.get("scale_constant", 1.0)
        self.recover_w_norm = kw.get("recover_w_norm", False)
        self.learnable_gamma = kw.get("learnable_gamma", False)
        self.gamma_init_value = kw.get("gamma_init_value", 1.0)
        self.pc_op_beta = kw.get("pc_op_beta", 0.0)
        self.power_iter = kw.get("power_iter", 10)
        self.power_iter_warmup_steps = kw.get("power_iter_warmup_steps", 0)
        self.power_iter_warmup_value = kw.get("power_iter_warmup_value", 0)


@pytest.mark.parametrize("pc_level", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("pc_norm_type", ["none", "F", "modified_F"])
def test_h1_matches_full_matrix(pc_level, pc_norm_type):
    """With H=1 and the whole matrix being a single head, per-head path must
    numerically match the full-matrix path."""
    torch.manual_seed(0)
    head_dim, in_features = 16, 128  # head wide
    W = torch.randn(head_dim, in_features, dtype=torch.float64)

    cfg = _Cfg(pc_norm_type=pc_norm_type, pc_level=pc_level, recover_w_norm=True)
    pc = PCTransform(cfg)

    full = pc.apply_preconditioner(weight=W, model_config=cfg)
    per_head = pc.apply_preconditioner(weight=W, model_config=cfg, num_heads=1)

    # equal_nan: pc_norm_type='none' + high pc_level can blow up to NaN on both
    # paths identically; we only require the two paths agree bitwise.
    torch.testing.assert_close(full, per_head, rtol=1e-10, atol=1e-10, equal_nan=True)


@pytest.mark.parametrize("pc_level", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("pc_norm_type", ["F", "modified_F"])
def test_block_independence(pc_level, pc_norm_type):
    """Per-head path on a stack of independent head sub-matrices should match
    applying the full-matrix path to each head separately."""
    torch.manual_seed(1)
    H, head_dim, in_features = 4, 16, 128
    heads = [torch.randn(head_dim, in_features, dtype=torch.float64) for _ in range(H)]
    W = torch.cat(heads, dim=0)  # [H*head_dim, in_features]

    cfg = _Cfg(pc_norm_type=pc_norm_type, pc_level=pc_level, recover_w_norm=True)
    pc = PCTransform(cfg)

    per_head = pc.apply_preconditioner(weight=W, model_config=cfg, num_heads=H)
    per_head_blocks = per_head.view(H, head_dim, in_features)

    for h in range(H):
        expected = pc.apply_preconditioner(weight=heads[h], model_config=cfg)
        torch.testing.assert_close(per_head_blocks[h], expected, rtol=1e-9, atol=1e-9)


def test_scale_and_gamma_per_head():
    """Per-head gamma ([H]) should scale each head block independently."""
    torch.manual_seed(2)
    H, head_dim, in_features = 3, 8, 64
    W = torch.randn(H * head_dim, in_features, dtype=torch.float64)

    cfg = _Cfg(pc_norm_type="F", pc_level=2,
               recover_w_norm=False, scale_constant=1.0,
               learnable_gamma=True)
    pc = PCTransform(cfg)

    gamma = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    no_gamma = pc.apply_preconditioner(weight=W, model_config=cfg, num_heads=H)
    with_gamma = pc.apply_preconditioner(weight=W, model_config=cfg, num_heads=H, gamma=gamma)

    no_gamma_blocks = no_gamma.view(H, head_dim, in_features)
    with_gamma_blocks = with_gamma.view(H, head_dim, in_features)
    for h in range(H):
        torch.testing.assert_close(with_gamma_blocks[h], no_gamma_blocks[h] * gamma[h],
                                   rtol=1e-10, atol=1e-10)


def test_per_head_op_norm_state_convergence():
    """Per-head power iteration should converge to each head's top singular value."""
    torch.manual_seed(3)
    H, head_dim, in_features = 4, 8, 64

    cfg = _Cfg(pc_norm_type="op", pc_level=0, power_iter=200,
               power_iter_warmup_steps=0, power_iter_warmup_value=0,
               pc_op_beta=0.0)
    linear = torch.nn.Linear(in_features, H * head_dim, bias=False)
    # Stable weights; each head has its own distinct spectrum.
    with torch.no_grad():
        W = torch.randn(H * head_dim, in_features, dtype=torch.float32)
        linear.weight.copy_(W)

    pcl = PCLinear(linear, cfg, layer_id=0, num_heads=H, per_head=True)
    pcl.update_op_state(step=1_000_000)  # bypass warmup

    # Compute per-head top singular values
    W3 = W.view(H, head_dim, in_features)
    expected = torch.linalg.svdvals(W3)[:, 0]  # [H]
    got = pcl._compute_op_norm_from_state(linear.weight) - cfg.pc_norm_eps  # [H]

    # Accept ~1e-3 relative error (power iteration converges geometrically)
    torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)


def test_non_per_head_path_unchanged():
    """pc_qkv_per_head=False path should be bit-identical to the old API with
    num_heads=None (i.e. no regression for non-QKV or disabled QKV-per-head)."""
    torch.manual_seed(4)
    W = torch.randn(64, 128, dtype=torch.float64)
    cfg = _Cfg(pc_norm_type="modified_F", pc_level=3, recover_w_norm=True)
    pc = PCTransform(cfg)
    out_no_kw = pc.apply_preconditioner(weight=W, model_config=cfg)
    out_with_kw = pc.apply_preconditioner(weight=W, model_config=cfg, num_heads=None)
    torch.testing.assert_close(out_no_kw, out_with_kw, rtol=0, atol=0)


def test_op_norm_per_head_end_to_end():
    """PCLinear forward with pc_norm_type='op' per-head should run and produce
    an orthogonal-ish weight per head after NS iterations."""
    torch.manual_seed(5)
    H, head_dim, in_features = 4, 16, 128

    cfg = _Cfg(pc_norm_type="op", pc_level=5,
               power_iter=30, power_iter_warmup_steps=0, power_iter_warmup_value=0,
               recover_w_norm=False, scale_constant=1.0)
    linear = torch.nn.Linear(in_features, H * head_dim, bias=False)
    pcl = PCLinear(linear, cfg, layer_id=0, num_heads=H, per_head=True)
    pcl.update_op_state(step=1_000_000)

    x = torch.randn(2, in_features)
    y = pcl(x)
    assert y.shape == (2, H * head_dim)
    assert torch.isfinite(y).all()

    # Check each head's preconditioned weight is close to row-orthogonal
    # (W_h W_h^T ≈ I scaled by some factor).
    w = pcl.pc(linear.weight, gamma=None,
              op_norm=pcl._compute_op_norm_from_state(linear.weight),
              num_heads=H)
    w3 = w.view(H, head_dim, in_features)
    for h in range(H):
        gram = w3[h] @ w3[h].T
        # After polar-express (level 5), gram should be close to identity.
        I = torch.eye(head_dim, dtype=gram.dtype, device=gram.device)
        assert (gram - I).abs().max() < 0.05, f"head {h}: gram - I max = {(gram - I).abs().max().item()}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
