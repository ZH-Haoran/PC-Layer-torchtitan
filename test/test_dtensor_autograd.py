"""
Verify autograd compatibility of PCTransform with DTensor(Replicate) inputs.
Run with: torchrun --nproc_per_node=1 test/test_dtensor_autograd.py
"""

import os
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, init_device_mesh


def test_addmm_dtensor(mesh):
    """torch.addmm works correctly with DTensor(Replicate) inputs."""
    print("=== [Replicate] addmm DTensor compat ===")
    torch.manual_seed(42)
    N = 64

    local_M = torch.randn(N, N, device="cuda", requires_grad=True)
    M_dt = DTensor.from_local(local_M, device_mesh=mesh, placements=[Replicate()], run_check=False)
    eye = torch.eye(N, device="cuda")
    eye_dt = DTensor.from_local(eye, device_mesh=mesh, placements=[Replicate()], run_check=False)

    alpha, beta = 2.5, -0.7
    out_dt = torch.addmm(eye_dt, M_dt, M_dt, beta=alpha, alpha=beta)
    loss_dt = out_dt.to_local().sum()
    loss_dt.backward()
    grad_dt = local_M.grad.clone()
    local_M.grad = None

    local_M2 = local_M.detach().clone().requires_grad_(True)
    eye2 = torch.eye(N, device="cuda")
    out_ref = torch.addmm(eye2, local_M2, local_M2, beta=alpha, alpha=beta)
    out_ref.sum().backward()
    grad_ref = local_M2.grad.clone()

    fwd_ok = torch.allclose(out_dt.to_local(), out_ref, atol=1e-5)
    bwd_ok = torch.allclose(grad_dt, grad_ref, atol=1e-5)
    print(f"  forward match:  {fwd_ok}")
    print(f"  backward match: {bwd_ok}")
    return fwd_ok and bwd_ok


def test_torch_add_dtensor(mesh):
    """torch.add with alpha works correctly with DTensor(Replicate) inputs."""
    print("=== [Replicate] torch.add DTensor compat ===")
    torch.manual_seed(42)

    local_A = torch.randn(64, 64, device="cuda", requires_grad=True)
    local_B = torch.randn(64, 64, device="cuda", requires_grad=True)
    A_dt = DTensor.from_local(local_A, device_mesh=mesh, placements=[Replicate()], run_check=False)
    B_dt = DTensor.from_local(local_B, device_mesh=mesh, placements=[Replicate()], run_check=False)

    alpha = 0.8
    out_dt = torch.add(A_dt, B_dt, alpha=alpha)
    loss_dt = out_dt.to_local().sum()
    loss_dt.backward()
    grad_A_dt = local_A.grad.clone()
    grad_B_dt = local_B.grad.clone()
    local_A.grad = None
    local_B.grad = None

    local_A2 = local_A.detach().clone().requires_grad_(True)
    local_B2 = local_B.detach().clone().requires_grad_(True)
    out_ref = torch.add(local_A2, local_B2, alpha=alpha)
    out_ref.sum().backward()
    grad_A_ref = local_A2.grad.clone()
    grad_B_ref = local_B2.grad.clone()

    fwd_ok = torch.allclose(out_dt.to_local(), out_ref, atol=1e-5)
    bwd_A_ok = torch.allclose(grad_A_dt, grad_A_ref, atol=1e-5)
    bwd_B_ok = torch.allclose(grad_B_dt, grad_B_ref, atol=1e-5)
    print(f"  forward match:    {fwd_ok}")
    print(f"  backward A match: {bwd_A_ok}")
    print(f"  backward B match: {bwd_B_ok}")
    return fwd_ok and bwd_A_ok and bwd_B_ok


def test_end_to_end_pctransform(mesh, norm_type="none"):
    """Test full PCTransform with DTensor(Replicate) input."""
    from dataclasses import dataclass
    from torchtitan.pc_layer.pc_layer import PCTransform

    print(f"=== [Replicate] full PCTransform e2e (norm={norm_type}) ===")

    @dataclass
    class Cfg:
        pc_level: int = 4
        pc_norm_type: str = norm_type
        pc_norm_eps: float = 1e-7
        scale_constant: float = 1.0
        recover_w_norm: bool = True
        learnable_gamma: bool = False
        gamma_init_value: float = 1.0

    torch.manual_seed(42)
    local_W = torch.randn(64, 32, device="cuda")
    local_W = local_W / local_W.norm()

    cfg = Cfg()
    pc = PCTransform(cfg)

    W_plain = local_W.clone().requires_grad_(True)
    out_plain = pc(W_plain)

    W_local = local_W.clone().requires_grad_(True)
    W_dt = DTensor.from_local(W_local, device_mesh=mesh, placements=[Replicate()], run_check=False)
    out_dt = pc(W_dt)

    fwd_ok = torch.allclose(out_dt.to_local(), out_plain, atol=1e-5)
    print(f"  forward match (DTensor vs plain): {fwd_ok}")

    out_dt.to_local().sum().backward()
    grad_dt = W_local.grad.clone()

    out_plain.sum().backward()
    grad_plain = W_plain.grad.clone()

    bwd_ok = torch.allclose(grad_dt, grad_plain, atol=1e-4)
    print(f"  backward match: {bwd_ok}")
    if not bwd_ok:
        print(f"  max diff: {(grad_dt - grad_plain).abs().max().item()}")

    return fwd_ok and bwd_ok


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    ok1 = test_addmm_dtensor(mesh)
    ok2 = test_torch_add_dtensor(mesh)
    ok3 = test_end_to_end_pctransform(mesh, norm_type="none")
    ok4 = test_end_to_end_pctransform(mesh, norm_type="F")

    print(f"\nAll passed: {ok1 and ok2 and ok3 and ok4}")

    dist.destroy_process_group()
