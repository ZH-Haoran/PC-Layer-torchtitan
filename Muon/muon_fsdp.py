import os
import math
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_map, tree_flatten
from typing import Generator, Literal, Sequence
from torch.distributed.tensor import distribute_tensor, DTensor
from itertools import chain, cycle, islice, repeat
# from utils import to_local, to_dist

_COEFFICIENT_SETS = {
    "simple": [
        (3.4445, -4.7750, 2.0315),
    ],
    "quintic": [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        (7.2086, -15.5131, 9.0178),
        (3.9623, -2.5813, 0.4542),
        (3.9466, -2.5765, 0.4544),
        (3.8991, -2.5671, 0.4566),
        (3.7186, -2.5308, 0.4653),
        (3.1390, -2.3073, 0.4733),
        (2.1715, -1.5246, 0.3885),
        (1.8648, -1.2224, 0.3577),
    ],
    "aol": [
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ],
}


def get_coefficient_iterator(
    steps: int,
    coefficient_sets: Sequence[tuple[float, float, float]],
    mode: Literal["cycle", "repeat_last"] = "cycle",
):
    """Iterate through coefficient sets with configurable end behavior using itertools.

    Args:
        steps: The number of tuples to yield.
        coefficient_sets: A sequence of (a, b, c) coefficient tuples.
        mode: Iteration mode:
            - "cycle": After the last element, restart from beginning.
            - "repeat_last": After the last element, keep yielding the last tuple.

    Yields:
        Tuples (a, b, c) from coefficient_sets according to the specified mode.

    Raises:
        ValueError: If coefficient_sets is empty.
        ValueError: If an invalid mode is provided.
    """
    if not coefficient_sets:
        raise ValueError("coefficient_sets must be non-empty.")

    base = cycle(coefficient_sets) if mode == "cycle" else chain(coefficient_sets, repeat(coefficient_sets[-1]))
    return islice(base, steps)


# @torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7, coefficient_type="simple", use_bf16=True):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Args:
        G: Input tensor to orthogonalize.
        steps: Number of Newton-Schulz iterations.
        eps: Small constant for numerical stability.
        coefficient_type: Type of coefficient set to use ("simple", "quintic", "polar_express", "aol").
        use_bf16: Whether to convert to bfloat16 during computation. Default is True.
    """
    assert len(G.shape) == 2
    if coefficient_type not in _COEFFICIENT_SETS:
        raise ValueError(f"Invalid coefficient type: {coefficient_type}. Must be one of {list(_COEFFICIENT_SETS.keys())}")

    coefficient_sets = _COEFFICIENT_SETS[coefficient_type]
    # For polar_express, use repeat_last mode; otherwise use cycle mode
    iter_mode = "repeat_last" if coefficient_type == "polar_express" else "cycle"
    coeff_iter = get_coefficient_iterator(steps, coefficient_sets, mode=iter_mode)

    X = G.bfloat16() if use_bf16 else G
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for a, b, c in coeff_iter:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
        lr_adjust: The learning rate adjustment method ("moonlight", "spectral_mup", "kellerjordan").
        muon_coefficient_type: The type of coefficient set for Newton-Schulz iterations ("simple", "quintic", "polar_express", "aol").
        use_bf16: Whether to use bfloat16 during Newton-Schulz computation. Default is True.
    """
    def __init__(self, muon_params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 adamw_params=None, adamw_betas=(0.95, 0.95), adamw_eps=1e-8, adamw_wd=0, lr_adjust='moonlight',
                 muon_coefficient_type='simple', use_bf16=True):

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_betas=adamw_betas, adamw_eps=adamw_eps, adamw_wd=adamw_wd)

        self.lr_adjust = lr_adjust
        self.coefficient_type = muon_coefficient_type
        self.use_bf16 = use_bf16

        # handle list of params or list of dicts
        if isinstance(muon_params, Generator):
            muon_params = list(muon_params)
        if isinstance(adamw_params, Generator):
            adamw_params = list(adamw_params)
        elif adamw_params is None:
            adamw_params = []

        super().__init__([*muon_params, *adamw_params], defaults)

        # Sort parameters into those for which we will use Muon, and those for which we will not
        # we cant pickle booleans for saving, so we will use 1=True, 0=False
        def assign_muon(p):
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]['use_muon'] = 1
            else:
                self.state[p]['use_muon'] = 0

        if isinstance(muon_params[0], dict):
            for group in muon_params:
                for p in group['params']:
                    assign_muon(p)
        else:
            for p in muon_params:
                assign_muon(p)

        def assign_adamw(p):
            # Do not use Muon for parameters in adamw_params
            self.state[p]['use_muon'] = 0

        if len(adamw_params) and isinstance(adamw_params[0], dict):
            for group in adamw_params:
                for p in group['params']:
                    assign_adamw(p)
        else:
            for p in adamw_params:
                assign_adamw(p)

        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
            
    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        if self.lr_adjust == 'moonlight':
            adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        elif self.lr_adjust == 'spectral_mup':
            adjusted_ratio = math.sqrt(A / B)
        elif self.lr_adjust == 'keller_jordan':
            # Suggested by Muon (https://kellerjordan.github.io/posts/muon/)
            adjusted_ratio = math.sqrt(max(1, A / B))
        else:
            raise ValueError(f"Unknown lr_adjust method: {self.lr_adjust}")

        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group['momentum']
            for i, p in enumerate(group['params']):
                if self.state[p]['use_muon'] == 1:
                    g = p.grad
                    if g is None:
                        continue
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf

                    meta = None
                    if isinstance(g, DTensor):
                        g, meta = to_local(g, keep_sharded=False)
                    # gives NaNs when done with Dtensor, instead of throwing a typical op not supported error, quite sneaky
                    g = zeropower_via_newtonschulz5(g, steps=group['ns_steps'], coefficient_type=self.coefficient_type, use_bf16=self.use_bf16)
                    
                    adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                    
                    if meta is not None:
                        g = to_dist(g, **meta)
                        
                    # g *= max(1, g.size(0)/g.size(1))**0.5
                    
                    # apply weight decay
                    p.data.mul_(1 - lr * group['adamw_wd'])
                    
                    g = g.view_as(p.data).type_as(p.data)
                    p.data.add_(g, alpha=-adjusted_lr)
                else:
                    # these are all pointwise so we can stay in Dtensor
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if 'step' not in state:
                        state['step'] = 0
                        state['moment1'] = torch.zeros_like(g)
                        state['moment2'] = torch.zeros_like(g)
                    state['step'] += 1
                    step = state['step']
                    buf1 = state['moment1']
                    buf2 = state['moment2']
                    buf1.lerp_(g, 1-group['adamw_betas'][0])
                    buf2.lerp_(g.square(), 1-group['adamw_betas'][1])

                    g = buf1 / (group['adamw_eps'] + buf2.sqrt())

                    bias_correction1 = 1 - group['adamw_betas'][0]**step
                    bias_correction2 = 1 - group['adamw_betas'][1]**step
                    scale = bias_correction1 / bias_correction2**0.5
                    p.data.mul_(1 - lr * group['adamw_wd'])
                    p.data.add_(g, alpha=-lr/scale)
                    

def to_dist(x, from_local=False, **meta):
    if from_local:
        return DTensor.from_local(
            x,
            device_mesh=meta["device_mesh"],
            placements=meta["placements"],
            shape=meta["shape"],
            stride=meta["stride"],
        )
    else:
        return distribute_tensor(x, device_mesh=meta["device_mesh"], placements=meta["placements"])


def to_local(x, keep_sharded=False):
    if isinstance(x, DTensor):
        meta = dict(
            device_mesh=x.device_mesh,
            placements=x.placements,
            shape=x.shape,
            stride=x.stride(),
        )
        if keep_sharded:
            return x.to_local(), meta
        else:
            return x.full_tensor(), meta

    return x, None