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


_HORNER_COEFFS = {
    1: [(-0.507, 1.507)],
    2: [(0.560, -1.643), (1.0, 2.083)],
    3: [(-1.283, 4.023), (1.0, -4.649), (1.0, 2.909)],
    4: [(2.890, -10.351), (1.0, 14.097), (1.0, -9.261), (1.0, 3.625)],
}


def _horner_poly(gram, eye, coeffs):
    """Evaluate a matrix polynomial in Horner form.

    *coeffs* is a list of (beta, alpha) pairs from innermost to outermost.
    The first step uses scalar arithmetic; subsequent steps fuse the GEMM
    and the alpha*I addition into a single ``torch.addmm`` call.
    """
    beta_0, alpha_0 = coeffs[0]
    S = alpha_0 * eye + beta_0 * gram
    for beta_k, alpha_k in coeffs[1:]:
        S = torch.addmm(eye, gram, S, beta=alpha_k, alpha=beta_k)
    return S


class LearnableGamma(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(1))

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

    def forward(self, weight, gamma=None, op_norm=None, return_norm=False):
        return self.apply_preconditioner(weight=weight, model_config=self.model_config, gamma=gamma, op_norm=op_norm, return_norm=return_norm)

    def apply_preconditioner(self, weight=None, model_config=None, gamma=None, op_norm=None, return_norm=False):
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

        combined_scale = model_config.scale_constant
        if model_config.recover_w_norm:
            combined_scale = combined_scale * W_norm.detach()
        if model_config.learnable_gamma and gamma is not None:
            gamma = gamma.to(dtype=W_preconditioned.dtype, device=W_preconditioned.device)
            combined_scale = combined_scale * gamma
        W_preconditioned = W_preconditioned * combined_scale

        if return_norm:
            return W_preconditioned, W_norm
        return W_preconditioned

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

        gram = gram if gram is not None else weight.t().mm(weight)
        N = gram.shape[0]

        if pc_level in _HORNER_COEFFS:
            eye = torch.eye(N, device=gram.device, dtype=gram.dtype)
            T = _horner_poly(gram, eye, _HORNER_COEFFS[pc_level])
            weight = weight.mm(T)
        elif pc_level == 5:
            for i, (a, b, c_coeff) in enumerate(_POLAR_EXPRESS_COEFFS):
                if i > 0:
                    gram = weight.t().mm(weight)
                T = torch.addmm(gram, gram, gram, beta=b, alpha=c_coeff)
                WT = weight.mm(T)
                weight = torch.add(WT, weight, alpha=a)
        else:
            raise ValueError("No pre-conditioner provided")
        return weight

    def preconditionerwide(self, weight=None, model_config=None, pc_level=None, gram=None):
        pc_level = model_config.pc_level
        if pc_level == 0:
            return weight

        gram = gram if gram is not None else weight.mm(weight.t())
        N = gram.shape[0]

        if pc_level in _HORNER_COEFFS:
            eye = torch.eye(N, device=gram.device, dtype=gram.dtype)
            T = _horner_poly(gram, eye, _HORNER_COEFFS[pc_level])
            weight = T.mm(weight)
        elif pc_level == 5:
            for i, (a, b, c_coeff) in enumerate(_POLAR_EXPRESS_COEFFS):
                if i > 0:
                    gram = weight.mm(weight.t())
                T = torch.addmm(gram, gram, gram, beta=b, alpha=c_coeff)
                TW = T.mm(weight)
                weight = torch.add(TW, weight, alpha=a)
        else:
            raise ValueError("No pre-conditioner provided")
        return weight


class PCLinear(nn.Module):
    def __init__(self, linear: nn.Linear, model_args, layer_id: int):
        super().__init__()
        self.linear = linear
        self.model_args = model_args
        self.layer_id = layer_id
        self.pc = PCTransform(model_args)
        if model_args.pc_norm_type == "op":
            out_features, in_features = linear.weight.shape
            self.register_buffer("op_u", torch.empty(out_features), persistent=True)
            self.register_buffer("op_v", torch.empty(in_features), persistent=True)

        if model_args.learnable_gamma:
            self.gamma = LearnableGamma()
        else:
            self.gamma = None

        self._gamma_inited_after_materialize = False

    @torch.no_grad()
    def _maybe_init_gamma(self):
        if self.gamma is None or self._gamma_inited_after_materialize:
            return
        v = self.gamma.v
        if v is not None and (not v.is_meta):
            self.gamma.reset_parameters(self.model_args.gamma_init_value)
            self._gamma_inited_after_materialize = True

    def _uses_op_norm(self):
        return self.model_args.pc_norm_type == "op"

    def _normalize_vector(self, vec):
        return vec / (vec.norm() + self.model_args.pc_norm_eps)

    @torch.no_grad()
    def update_op_state(self):
        if not self._uses_op_norm():
            return
        weight = self.linear.weight
        op_weight = self._get_weight_for_op(weight)
        self._initialize_op_state_if_needed(op_weight)
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            u, v = self._compute_updated_op_state(op_weight)
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

    def _has_valid_op_state(self, weight):
        if (
            self.op_u.numel() != weight.size(0)
            or self.op_v.numel() != weight.size(1)
            or self.op_u.device != weight.device
            or self.op_v.device != weight.device
        ):
            return False
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
        self.op_u = self._random_unit_vector(weight.size(0), weight)
        self.op_v = self._random_unit_vector(weight.size(1), weight)

    def _ensure_op_state(self, weight):
        if self._uses_op_norm() and not self._has_valid_op_state(weight):
            raise RuntimeError(
                f"OP state is not initialized for layer {self.layer_id}. "
                "Call update_op_state() or update_model_op_state() before forward."
            )

    @torch.no_grad()
    def _compute_updated_op_state(self, weight):
        beta = float(getattr(self.model_args, "pc_op_beta", 0.0))
        beta = max(0.0, min(1.0, beta))

        u_rand = self._random_unit_vector(weight.size(0), weight)
        v_rand = self._random_unit_vector(weight.size(1), weight)

        u = beta * self.op_u + (1.0 - beta) * u_rand
        v = beta * self.op_v + (1.0 - beta) * v_rand
        u = self._normalize_vector(u)
        v = self._normalize_vector(v)

        for _ in range(self.model_args.power_iter):
            v = self._normalize_vector(torch.mv(weight.T, u))
            u = self._normalize_vector(torch.mv(weight, v))

        return u, v

    def _compute_op_norm_from_state(self, weight):
        op_weight = self._get_weight_for_op(weight)
        self._ensure_op_state(op_weight)
        op_u = self.op_u.to(dtype=op_weight.dtype)
        op_v = self.op_v.to(dtype=op_weight.dtype)
        wv = torch.mv(op_weight, op_v)
        W_norm_local = torch.dot(op_u, wv) + self.model_args.pc_norm_eps
        return self._wrap_scalar_like_weight(W_norm_local, weight)

    def forward(self, x):
        self._maybe_init_gamma()
        g = self.gamma.value() if self.gamma is not None else None
        if self._uses_op_norm():
            op_norm = self._compute_op_norm_from_state(self.linear.weight)
            w = self.pc(self.linear.weight, gamma=g, op_norm=op_norm)
        else:
            w = self.pc(self.linear.weight, gamma=g)
        return F.linear(x, w, self.linear.bias)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


def iter_pc_linear_modules(module):
    for submodule in module.modules():
        if isinstance(submodule, PCLinear):
            yield submodule


def model_uses_op_norm(module):
    return any(submodule._uses_op_norm() for submodule in iter_pc_linear_modules(module))


@torch.no_grad()
def update_model_op_state(module):
    for submodule in iter_pc_linear_modules(module):
        if submodule._uses_op_norm():
            submodule.update_op_state()
