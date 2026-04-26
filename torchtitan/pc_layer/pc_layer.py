import warnings
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.distributed._tensor import Replicate


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


class LearnableGamma(nn.Module):
    def __init__(self, shape=(1,)):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(*shape))

    @torch.no_grad()
    def reset_parameters(self, init_gamma: float):
        if self.v is None or self.v.is_meta:
            return
        self.v.fill_(float(init_gamma))

    def value(self) -> torch.Tensor:
        return self.v


class PCTransform(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config

    def forward(self, weight, gamma=None, op_norm=None, return_norm=False, num_heads=None):
        return self.apply_preconditioner(weight=weight, model_config=self.model_config, gamma=gamma, op_norm=op_norm, return_norm=return_norm, num_heads=num_heads)

    def apply_preconditioner(self, weight=None, model_config=None, gamma=None, op_norm=None, return_norm=False, num_heads=None):
        if num_heads is not None:
            return self._apply_preconditioner_per_head(
                weight=weight, model_config=model_config,
                gamma=gamma, op_norm=op_norm, return_norm=return_norm,
                num_heads=num_heads,
            )

        W_normalized, W_norm, cache = self.pc_normalize(weight=weight, model_config=model_config, op_norm=op_norm)
        r, c = W_normalized.shape

        gram_for_pc = None
        gram_kind = None
        if cache is not None and 'gram' in cache:
            s = cache['divisor']
            gram_for_pc = cache['gram'] / (s * s)
            gram_kind = cache['gram_kind']

        if r >= c:
            use_gram = gram_for_pc if gram_kind == 'wtw' else None
            W_preconditioned = self.preconditionertall(
                weight=W_normalized, model_config=model_config, gram=use_gram
            )
        else:
            use_gram = gram_for_pc if gram_kind == 'wwt' else None
            W_preconditioned = self.preconditionerwide(
                weight=W_normalized, model_config=model_config, gram=use_gram
            )

        W_preconditioned *= model_config.scale_constant
        if model_config.recover_w_norm:
            norm_for_recover = W_norm.detach()
            W_preconditioned = W_preconditioned * norm_for_recover
        if model_config.learnable_gamma and gamma is not None:
            gamma = gamma.to(dtype=W_preconditioned.dtype, device=W_preconditioned.device)
            W_preconditioned = W_preconditioned * gamma

        if return_norm:
            return W_preconditioned, W_norm
        return W_preconditioned

    # ── per-head path ──────────────────────────────────────────────────

    @staticmethod
    def _materialize_local(weight):
        """Return a plain local 2D tensor view of `weight`, collecting across shards if needed."""
        if isinstance(weight, DTensor):
            if all(isinstance(p, Replicate) for p in weight.placements):
                return weight.to_local()
            return weight.full_tensor()
        return weight

    @staticmethod
    def _rewrap_like(local_tensor, ref):
        """If `ref` is a DTensor, wrap `local_tensor` as a Replicate DTensor on the same mesh."""
        if isinstance(ref, DTensor):
            return DTensor.from_local(
                local_tensor,
                device_mesh=ref.device_mesh,
                placements=[Replicate()],
                run_check=False,
            )
        return local_tensor

    def _apply_preconditioner_per_head(self, *, weight, model_config, gamma, op_norm, return_norm, num_heads):
        local_w = self._materialize_local(weight)
        out, in_ = local_w.shape
        if out % num_heads != 0:
            raise ValueError(
                f"per-head precondition: out_features={out} not divisible by num_heads={num_heads}"
            )
        head_dim = out // num_heads
        # Use reshape (not view) because to_local() may yield a non-contiguous tensor.
        W3 = local_w.reshape(num_heads, head_dim, in_)

        W3_normalized, W_norm, cache = self._pc_normalize_per_head(
            W3=W3, model_config=model_config, op_norm=op_norm,
        )  # W_norm: [H]

        gram_for_pc = None
        if cache is not None and 'gram' in cache:
            s = cache['divisor'].view(num_heads, 1, 1)
            gram_for_pc = cache['gram'] / (s * s)

        W3_pre = self._preconditionerwide_batched(
            weight=W3_normalized, model_config=model_config, gram=gram_for_pc,
        )  # [H, head_dim, in_]

        W3_pre = W3_pre * model_config.scale_constant
        if model_config.recover_w_norm:
            W3_pre = W3_pre * W_norm.detach().view(num_heads, 1, 1)
        if model_config.learnable_gamma and gamma is not None:
            gamma = gamma.to(dtype=W3_pre.dtype, device=W3_pre.device)
            # gamma shape is [H]; broadcast as [H, 1, 1] across each head block.
            W3_pre = W3_pre * gamma.view(num_heads, 1, 1)

        W_out = W3_pre.reshape(out, in_)
        W_out = self._rewrap_like(W_out, weight)

        if return_norm:
            return W_out, W_norm
        return W_out

    def _pc_normalize_per_head(self, *, W3, model_config, op_norm):
        H, hd, in_ = W3.shape
        cache = None

        if model_config.pc_norm_type == 'none':
            if model_config.pc_level != 0:
                warnings.warn(
                    "pc_norm_type is None but pc_level != 0: weight is not normalized before applying preconditioner. "
                    "This may lead to unexpected behavior.",
                    UserWarning
                )
            W_norm = torch.ones(H, dtype=W3.dtype, device=W3.device)

        elif model_config.pc_norm_type == "F":
            W_norm = W3.reshape(H, -1).norm(dim=-1) + model_config.pc_norm_eps  # [H]

        elif model_config.pc_norm_type == "modified_F":
            # Each head is wide (hd < in_), so use W W^T.
            gram = torch.bmm(W3, W3.transpose(-1, -2))  # [H, hd, hd]
            gram2 = torch.bmm(gram, gram)
            fro = torch.linalg.matrix_norm(gram2, dim=(-2, -1), ord='fro')  # [H]
            W_norm = (fro ** 0.25) + model_config.pc_norm_eps
            cache = {'gram': gram, 'gram_kind': 'wwt', 'divisor': W_norm}

        elif model_config.pc_norm_type == "op":
            if op_norm is None:
                raise ValueError(
                    "pc_norm_type='op' requires op_norm to be pre-computed by PCLinear."
                )
            # op_norm is expected to be a [H] tensor in the per-head path.
            W_norm = op_norm

        else:
            raise ValueError(f"Unknown pc_norm_type: {model_config.pc_norm_type}")

        W3_normalized = W3 / W_norm.view(H, 1, 1)
        return W3_normalized, W_norm, cache

    def _preconditionerwide_batched(self, *, weight, model_config, gram=None):
        pc_level = model_config.pc_level
        if pc_level == 0:
            return weight

        H, hd, _ = weight.shape
        I = torch.eye(hd, device=weight.device, dtype=weight.dtype).unsqueeze(0)  # [1, hd, hd]
        wwt = gram if gram is not None else torch.bmm(weight, weight.transpose(-1, -2))

        def bmm(a, b):
            return torch.bmm(a, b)

        if pc_level == 1:
            weight = bmm(1.507 * I - 0.507 * wwt, weight)
        elif pc_level == 2:
            weight = bmm(2.083 * I + bmm(wwt, -1.643 * I + 0.560 * wwt), weight)
        elif pc_level == 3:
            weight = bmm(2.909 * I + bmm(wwt, -4.649 * I + bmm(wwt, 4.023 * I - 1.283 * wwt)), weight)
        elif pc_level == 4:
            weight = bmm(3.625 * I + bmm(wwt, -9.261 * I + bmm(wwt, 14.097 * I + bmm(wwt, -10.351 * I + 2.890 * wwt))), weight)
        elif pc_level == 5:
            for i, (a, b, c_coeff) in enumerate(_POLAR_EXPRESS_COEFFS):
                if i > 0:
                    wwt = torch.bmm(weight, weight.transpose(-1, -2))
                weight = a * weight + bmm(b * wwt + c_coeff * bmm(wwt, wwt), weight)
        else:
            raise ValueError("No pre-conditioner provided")
        return weight

    def pc_normalize(self, weight=None, model_config=None, op_norm=None):
        if weight.ndim != 2:
            raise ValueError("Weight must be a 2D tensor")
        r, c = weight.shape
        cache = None

        if model_config.pc_norm_type == 'none':
            if model_config.pc_level != 0:
                warnings.warn(
                    "pc_norm_type is None but pc_level != 0: weight is not normalized before applying preconditioner. "
                    "This may lead to unexpected behavior.",
                    UserWarning
                )
            W_norm = torch.tensor(1.0, dtype=weight.dtype, device=weight.device)
        elif model_config.pc_norm_type == "F":
            W_norm = weight.norm() + model_config.pc_norm_eps

        elif model_config.pc_norm_type == "modified_F":
            if r <= c:
                gram = weight.mm(weight.T)
                gram_kind = 'wwt'
            else:
                gram = weight.T.mm(weight)
                gram_kind = 'wtw'

            gram2 = gram.mm(gram)
            fro = torch.linalg.matrix_norm(gram2, ord='fro')
            W_norm = (fro ** 0.25) + model_config.pc_norm_eps
            cache = {'gram': gram, 'gram_kind': gram_kind, 'divisor': W_norm}

        elif model_config.pc_norm_type == "op":
            if op_norm is None:
                raise ValueError(
                    "pc_norm_type='op' requires op_norm to be pre-computed by PCLinear."
                )
            W_norm = op_norm

        else:
            raise ValueError(f"Unknown pc_norm_type: {model_config.pc_norm_type}")

        normalized_weight = weight / W_norm
        return normalized_weight, W_norm, cache

    def preconditionertall(self, weight=None, model_config=None, gram=None):
        pc_level = model_config.pc_level
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
            # Polar express iterative Newton-Schulz (tall: apply from right)
            for i, (a, b, c_coeff) in enumerate(_POLAR_EXPRESS_COEFFS):
                if i > 0:
                    wtw = weight.t().mm(weight)
                weight = a * weight + weight.mm(b * wtw + c_coeff * wtw.mm(wtw))
        else:
            raise ValueError("No pre-conditioner provided")
        return weight

    def preconditionerwide(self, weight=None, model_config=None, pc_level=None, gram=None):
        pc_level = model_config.pc_level
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
            # Polar express iterative Newton-Schulz (wide: apply from left)
            for i, (a, b, c_coeff) in enumerate(_POLAR_EXPRESS_COEFFS):
                if i > 0:
                    wwt = weight.mm(weight.t())
                weight = a * weight + (b * wwt + c_coeff * wwt.mm(wwt)).mm(weight)
        else:
            raise ValueError("No pre-conditioner provided")
        return weight
    

class PCLinear(nn.Module):
    def __init__(self, linear: nn.Linear, model_args, layer_id: int,
                 num_heads: int = None, per_head: bool = False):
        super().__init__()
        self.linear = linear
        self.model_args = model_args
        self.layer_id = layer_id
        self.pc = PCTransform(model_args)

        self.per_head = bool(per_head)
        self.num_heads = int(num_heads) if (self.per_head and num_heads is not None) else None
        if self.per_head:
            if self.num_heads is None:
                raise ValueError("per_head=True requires num_heads to be provided")
            out_features, _ = linear.weight.shape
            if out_features % self.num_heads != 0:
                raise ValueError(
                    f"out_features {out_features} not divisible by num_heads {self.num_heads}"
                )
            self.head_dim_pc = out_features // self.num_heads

        if model_args.pc_norm_type == "op":
            out_features, in_features = linear.weight.shape
            if self.per_head:
                self.register_buffer("op_u", torch.empty(self.num_heads, self.head_dim_pc), persistent=True)
                self.register_buffer("op_v", torch.empty(self.num_heads, in_features), persistent=True)
            else:
                self.register_buffer("op_u", torch.empty(out_features), persistent=True)
                self.register_buffer("op_v", torch.empty(in_features), persistent=True)

        if model_args.learnable_gamma:
            gamma_shape = (self.num_heads,) if self.per_head else (1,)
            self.gamma = LearnableGamma(shape=gamma_shape)
        else:
            self.gamma = None

        # 保险：避免 meta 参数在 __init__ 时没法 fill_
        self._gamma_inited_after_materialize = False

    @torch.no_grad()
    def _maybe_init_gamma(self):
        if self.gamma is None or self._gamma_inited_after_materialize:
            return
        v = self.gamma.v
        if v is not None and (not v.is_meta):
            self.gamma.reset_parameters(self.model_args.gamma_init_value)
            self._gamma_inited_after_materialize = True

    # ── OP norm helpers ────────────────────────────────────────────────

    def _uses_op_norm(self):
        return self.model_args.pc_norm_type == "op"

    def _normalize_vector(self, vec):
        return vec / (vec.norm() + self.model_args.pc_norm_eps)

    def _normalize_rows(self, mat):
        return mat / (mat.norm(dim=-1, keepdim=True) + self.model_args.pc_norm_eps)

    @torch.no_grad()
    def update_op_state(self, step=None):
        if not self._uses_op_norm():
            return
        weight = self.linear.weight
        op_weight = self._get_weight_for_op(weight)
        self._initialize_op_state_if_needed(op_weight)
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            u, v = self._compute_updated_op_state(op_weight, step=step)
            self.op_u.copy_(u)
            self.op_v.copy_(v)
        self._broadcast_op_state()

    @torch.no_grad()
    def _broadcast_op_state(self):
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(self.op_u, src=0)
            dist.broadcast(self.op_v, src=0)

    def _get_weight_for_op(self, weight):
        if isinstance(weight, DTensor):
            if all(isinstance(p, Replicate) for p in weight.placements):
                return weight.to_local()
            return weight.full_tensor()
        return weight

    def _wrap_scalar_like_weight(self, scalar, weight):
        if isinstance(weight, DTensor):
            return DTensor.from_local(
                scalar,
                device_mesh=weight.device_mesh,
                placements=[Replicate()],
                run_check=False,
            )
        return scalar

    @torch.no_grad()
    def _random_unit_vector(self, size, weight):
        vec = torch.randn(size, device=weight.device, dtype=weight.dtype)
        return self._normalize_vector(vec)

    @torch.no_grad()
    def _random_unit_rows(self, n_rows, n_cols, weight):
        mat = torch.randn(n_rows, n_cols, device=weight.device, dtype=weight.dtype)
        return self._normalize_rows(mat)

    def _has_valid_op_state(self, weight):
        # dtype is intentionally excluded: op_u/op_v are stored in the update
        # dtype (e.g. float32) but forward may see a cast weight (e.g. bfloat16
        # from FSDP mixed-precision).  Casting happens lazily in
        # _compute_op_norm_from_state.
        if self.per_head:
            H = self.num_heads
            hd = self.head_dim_pc
            in_features = weight.size(1)
            if (
                self.op_u.shape != (H, hd)
                or self.op_v.shape != (H, in_features)
                or self.op_u.device != weight.device
                or self.op_v.device != weight.device
            ):
                return False
            if (
                not torch.isfinite(self.op_u).all()
                or not torch.isfinite(self.op_v).all()
                or self.op_u.norm(dim=-1).min() == 0
                or self.op_v.norm(dim=-1).min() == 0
            ):
                return False
            return True

        if (
            self.op_u.numel() != weight.size(0)
            or self.op_v.numel() != weight.size(1)
            or self.op_u.device != weight.device
            or self.op_v.device != weight.device
        ):
            return False
        # After meta-device materialization, buffers may contain garbage
        # (NaN/Inf/zero). Detect this so _initialize_op_state_if_needed
        # properly reinitializes them.
        if (
            not torch.isfinite(self.op_u).all()
            or not torch.isfinite(self.op_v).all()
            or self.op_u.norm() == 0
            or self.op_v.norm() == 0
        ):
            return False
        return True

    @torch.no_grad()
    def _initialize_op_state_if_needed(self, weight):
        if not self._uses_op_norm() or self._has_valid_op_state(weight):
            return
        if self.per_head:
            self.op_u = self._random_unit_rows(self.num_heads, self.head_dim_pc, weight)
            self.op_v = self._random_unit_rows(self.num_heads, weight.size(1), weight)
        else:
            self.op_u = self._random_unit_vector(weight.size(0), weight)
            self.op_v = self._random_unit_vector(weight.size(1), weight)

    def _ensure_op_state(self, weight):
        if self._uses_op_norm() and not self._has_valid_op_state(weight):
            raise RuntimeError(
                f"OP state is not initialized for layer {self.layer_id}. "
                "Call update_op_state() or update_model_op_state() before forward."
            )

    @torch.no_grad()
    def _compute_updated_op_state(self, weight, step=None):
        beta = float(getattr(self.model_args, "pc_op_beta", 0.0))
        beta = max(0.0, min(1.0, beta))

        warmup_steps = int(self.model_args.power_iter_warmup_steps)
        warmup_value = int(self.model_args.power_iter_warmup_value)
        if step is not None and step <= warmup_steps:
            n_iters = warmup_value
        else:
            n_iters = self.model_args.power_iter

        if self.per_head:
            H = self.num_heads
            hd = self.head_dim_pc
            W3 = weight.reshape(H, hd, -1)  # [H, hd, in]

            u_rand = self._random_unit_rows(H, hd, weight)
            v_rand = self._random_unit_rows(H, weight.size(1), weight)

            u = self._normalize_rows(beta * self.op_u + (1.0 - beta) * u_rand)
            v = self._normalize_rows(beta * self.op_v + (1.0 - beta) * v_rand)

            for _ in range(n_iters):
                v = self._normalize_rows(
                    torch.bmm(W3.transpose(-1, -2), u.unsqueeze(-1)).squeeze(-1)
                )
                u = self._normalize_rows(
                    torch.bmm(W3, v.unsqueeze(-1)).squeeze(-1)
                )
            return u, v

        u_rand = self._random_unit_vector(weight.size(0), weight)
        v_rand = self._random_unit_vector(weight.size(1), weight)

        u = beta * self.op_u + (1.0 - beta) * u_rand
        v = beta * self.op_v + (1.0 - beta) * v_rand
        u = self._normalize_vector(u)
        v = self._normalize_vector(v)

        for _ in range(n_iters):
            v = self._normalize_vector(torch.mv(weight.T, u))
            u = self._normalize_vector(torch.mv(weight, v))

        return u, v

    def _compute_op_norm_from_state(self, weight):
        op_weight = self._get_weight_for_op(weight)
        self._ensure_op_state(op_weight)
        # Cast op_u/op_v to match the forward weight dtype (e.g. bfloat16 under
        # FSDP mixed precision) so torch.mv/dot don't error on dtype mismatch.
        op_u = self.op_u.to(dtype=op_weight.dtype)
        op_v = self.op_v.to(dtype=op_weight.dtype)
        if self.per_head:
            H = self.num_heads
            hd = self.head_dim_pc
            W3 = op_weight.reshape(H, hd, -1)  # [H, hd, in]
            Wv = torch.bmm(W3, op_v.unsqueeze(-1)).squeeze(-1)  # [H, hd]
            W_norm_local = (op_u * Wv).sum(dim=-1) + self.model_args.pc_norm_eps  # [H]
            # per-head path consumes local [H] directly; skip DTensor wrapping.
            return W_norm_local
        wv = torch.mv(op_weight, op_v)
        W_norm_local = torch.dot(op_u, wv) + self.model_args.pc_norm_eps
        return self._wrap_scalar_like_weight(W_norm_local, weight)

    # ── forward ───────────────────────────────────────────────────────

    def forward(self, x):
        self._maybe_init_gamma()
        g = self.gamma.value() if self.gamma is not None else None
        nh = self.num_heads if self.per_head else None
        if self._uses_op_norm():
            op_norm = self._compute_op_norm_from_state(self.linear.weight)
            w = self.pc(self.linear.weight, gamma=g, op_norm=op_norm, num_heads=nh)
        else:
            w = self.pc(self.linear.weight, gamma=g, num_heads=nh)
        return F.linear(x, w, self.linear.bias)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


# ── Module-level utilities ────────────────────────────────────────────

def iter_pc_linear_modules(module):
    for submodule in module.modules():
        if isinstance(submodule, PCLinear):
            yield submodule


def model_uses_op_norm(module):
    return any(submodule._uses_op_norm() for submodule in iter_pc_linear_modules(module))


@torch.no_grad()
def update_model_op_state(module, step=None):
    for submodule in iter_pc_linear_modules(module):
        if submodule._uses_op_norm():
            submodule.update_op_state(step=step)
