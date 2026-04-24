"""Full-SVD spectral-norm tracking for validating the power-iteration approximation.

Every `svd_freq` steps, compute a full SVD on selected weight matrices and record
the top singular value (ground truth) alongside the current power-iter estimate
u^T W v from the associated PCLinear module, so the approximation gap can be
quantified over training.

Results are appended to a JSONL file (one record per snapshot, plus a metadata
header line) — safe against crashes and incremental writes.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed._tensor import Replicate
from torch.distributed.tensor import DTensor

DEFAULT_TARGET_LAYERS: List[int] = [0, 1, 9, 16, 17]
DEFAULT_TARGET_MATRICES: List[str] = ["wo", "w1", "w2", "w3"]
DEFAULT_SVD_FREQ: int = 200


def _materialize_full_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return an unsharded view of `weight`, collectives included if needed.

    Every rank must call this together when the weight is a sharded DTensor,
    because `full_tensor()` triggers an all-gather.
    """
    if isinstance(weight, DTensor):
        if all(isinstance(p, Replicate) for p in weight.placements):
            return weight.to_local()
        return weight.full_tensor()
    return weight


def _resolve_matrix_module(layer_module: torch.nn.Module, matrix_name: str):
    """Find the linear-like submodule for a named matrix (wo/w1/w2/w3).

    Llama convention: `wo` lives under `attention`, `w1/w2/w3` under `feed_forward`.
    Returns None if the path doesn't exist on this layer.
    """
    if matrix_name == "wo":
        parent = getattr(layer_module, "attention", None)
    elif matrix_name in ("w1", "w2", "w3"):
        parent = getattr(layer_module, "feed_forward", None)
    else:
        return None
    if parent is None:
        return None
    return getattr(parent, matrix_name, None)


def _get_weight_tensor(module) -> Optional[torch.Tensor]:
    """Unwrap PCLinear or return nn.Linear.weight directly."""
    if module is None:
        return None
    if hasattr(module, "linear") and hasattr(module.linear, "weight"):
        return module.linear.weight
    if hasattr(module, "weight"):
        return module.weight
    return None


def _power_iter_estimate(module, full_weight: torch.Tensor) -> Optional[float]:
    """u^T W v with the currently-stored op_u, op_v from the PCLinear module.

    Returns None when the module isn't a PCLinear using op-norm, or its buffers
    haven't been initialized.
    """
    # Late import to avoid circular import at module load time.
    from torchtitan.pc_layer.pc_layer import PCLinear

    if not isinstance(module, PCLinear) or not module._uses_op_norm():
        return None
    if not module._has_valid_op_state(module.linear.weight):
        return None
    op_u = module.op_u.to(dtype=full_weight.dtype, device=full_weight.device)
    op_v = module.op_v.to(dtype=full_weight.dtype, device=full_weight.device)
    wv = torch.mv(full_weight, op_v)
    sigma_approx = torch.dot(op_u, wv) + module.model_args.pc_norm_eps
    return float(sigma_approx.item())


class SVDTracker:
    """Periodically dump full-SVD spectral norms for selected weights.

    All ranks must call `maybe_log` on the same step; collective operations
    inside (DTensor.full_tensor) require every rank to participate. Only rank 0
    writes to disk.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_config,
        output_dir: str,
        target_layers: Optional[List[int]] = None,
        target_matrices: Optional[List[str]] = None,
        svd_freq: int = DEFAULT_SVD_FREQ,
        global_rank: int = 0,
        run_tag: str = "",
    ):
        self.model = model
        self.model_config = model_config
        self.target_layers = list(target_layers) if target_layers is not None else list(DEFAULT_TARGET_LAYERS)
        self.target_matrices = list(target_matrices) if target_matrices is not None else list(DEFAULT_TARGET_MATRICES)
        self.svd_freq = int(svd_freq)
        self.global_rank = int(global_rank)
        self.is_rank_zero = self.global_rank == 0

        out_path = Path(output_dir)
        if self.is_rank_zero:
            out_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_part = f"_{run_tag}" if run_tag else ""
        self.jsonl_path = out_path / f"svd_spectral_norm{tag_part}_{timestamp}.jsonl"

        if self.is_rank_zero:
            metadata = {
                "_type": "metadata",
                "target_layers": self.target_layers,
                "target_matrices": self.target_matrices,
                "svd_freq": self.svd_freq,
                "started_at": timestamp,
                "run_tag": run_tag,
                "pc_config": {
                    "power_iter": getattr(model_config, "power_iter", None),
                    "pc_norm_type": getattr(model_config, "pc_norm_type", None),
                    "pc_op_beta": getattr(model_config, "pc_op_beta", None),
                    "pc_norm_eps": getattr(model_config, "pc_norm_eps", None),
                    "pc_level": getattr(model_config, "pc_level", None),
                    "power_iter_warmup_steps": getattr(model_config, "power_iter_warmup_steps", None),
                    "power_iter_warmup_value": getattr(model_config, "power_iter_warmup_value", None),
                    "precondition_mlp": getattr(model_config, "precondition_mlp", None),
                    "precondition_o": getattr(model_config, "precondition_o", None),
                    "precondition_qk": getattr(model_config, "precondition_qk", None),
                    "precondition_v": getattr(model_config, "precondition_v", None),
                },
            }
            with open(self.jsonl_path, "w") as f:
                f.write(json.dumps(metadata) + "\n")

    def should_compute(self, step: int) -> bool:
        return self.svd_freq > 0 and step > 0 and step % self.svd_freq == 0

    @torch.no_grad()
    def _compute_snapshot(self, step: int) -> Dict:
        layers_record: Dict[str, Dict] = {}
        layers = getattr(self.model, "layers", None)
        if layers is None:
            return {"step": step, "layers": {}}

        for layer_id in self.target_layers:
            key = str(layer_id)
            # ModuleDict supports `in` on string keys.
            if key not in layers:
                continue
            layer_module = layers[key]
            matrix_records: Dict[str, Dict] = {}
            for mat_name in self.target_matrices:
                module = _resolve_matrix_module(layer_module, mat_name)
                weight = _get_weight_tensor(module)
                if weight is None:
                    continue

                # All ranks gather the full weight; only rank 0 uses it.
                full_weight = _materialize_full_weight(weight).detach()
                # Cast to float32 for numerically stable SVD (FSDP mixed-precision may leave bf16).
                svd_input = full_weight.to(dtype=torch.float32)

                # Skip SVD on non-rank-0 to save compute — but we still needed
                # the full_tensor() collective above.
                if not self.is_rank_zero:
                    continue

                try:
                    S = torch.linalg.svdvals(svd_input)
                except Exception as e:
                    matrix_records[mat_name] = {"error": repr(e), "shape": list(svd_input.shape)}
                    continue

                sigma_1 = float(S[0].item())
                sigma_2 = float(S[1].item()) if S.numel() > 1 else float("nan")
                sigma_min = float(S[-1].item())
                fro_norm = float(torch.linalg.norm(S).item())
                rec = {
                    "shape": list(svd_input.shape),
                    "sigma_1_svd": sigma_1,
                    "sigma_2_svd": sigma_2,
                    "sigma_min_svd": sigma_min,
                    "fro_norm_svd": fro_norm,
                    "sigma_gap_abs": sigma_1 - sigma_2,
                    "sigma_ratio_2_over_1": (sigma_2 / sigma_1) if sigma_1 > 0 else float("nan"),
                    "condition_number": (sigma_1 / sigma_min) if sigma_min > 0 else float("inf"),
                }

                sigma_pi = _power_iter_estimate(module, svd_input)
                if sigma_pi is not None:
                    rec["sigma_1_power_iter"] = sigma_pi
                    if sigma_1 > 0:
                        rec["relative_error"] = abs(sigma_1 - sigma_pi) / sigma_1

                matrix_records[mat_name] = rec
            if matrix_records or self.is_rank_zero:
                layers_record[key] = matrix_records
        return {"step": step, "layers": layers_record}

    def maybe_log(self, step: int) -> Optional[Dict]:
        """Entry point called once per training step (all ranks).

        Returns the snapshot on rank 0 at measurement steps, else None.
        """
        if not self.should_compute(step):
            return None
        snapshot = self._compute_snapshot(step)
        if self.is_rank_zero:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(snapshot) + "\n")
            return snapshot
        return None
