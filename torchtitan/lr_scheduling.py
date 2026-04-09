# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

from torch.optim.lr_scheduler import LambdaLR
from torchtitan.config_manager import JobConfig

# global states for scheduling
# these are needed as LambdaLR does not support argument passing
_warmup_steps = 0
_total_steps = 0
_decay_steps = 0
_stable_steps = 0
_lr_decay_type = "linear"
# new: cosine end ratio
_cosine_end_ratio = 0.1


def linear_warmup_linear_decay(current_step: int) -> float:
    """Computes linear warmup followed by linear decay.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    if current_step < _warmup_steps:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        curr_adjustment = float(current_step / (_warmup_steps + 1))

    else:
        # linear decay
        normalized_step = _decay_steps - (current_step - _warmup_steps)
        curr_adjustment = 1 - (_decay_steps - normalized_step) / _decay_steps

    return curr_adjustment


def linear_warmup_cosine_decay(current_step: int) -> float:
    """
    Linear warmup (0 -> 1) then cosine decay (1 -> _cosine_end_ratio).
    The returned value is a multiplicative factor for LambdaLR.

    - Warmup: steps [0, _warmup_steps-1], factor ramps 0->1 (with +1 offset)
    - Decay : steps [_warmup_steps, _total_steps-1], cosine 1->end_ratio
    """
    if current_step < _warmup_steps:
        return float(current_step + 1) / float(_warmup_steps + 1)

    # decay progress in [0, 1]
    # note: _decay_steps is the number of steps after warmup
    # we clamp to be safe in case scheduler.step() is called extra times
    denom = max(1, _decay_steps)
    progress = float(current_step - _warmup_steps) / float(denom)
    progress = min(max(progress, 0.0), 1.0)

    # cosine from 1 -> 0
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    # map to [end_ratio, 1]
    factor = _cosine_end_ratio + (1.0 - _cosine_end_ratio) * cosine
    return factor


def wsd_schedule(current_step: int) -> float:
    """
    WSD (Warmup–Stable–Decay) schedule.
    - Warmup: linear ramp-up 0 → 1
    - Stable: constant 1.0
    - Decay: linear decay 1 → 0
    """
    warmup_stable_steps = _warmup_steps + _stable_steps
    if current_step < _warmup_steps:
        # warmup
        return float(current_step + 1) / float(_warmup_steps + 1)

    elif current_step < warmup_stable_steps:
        # stable region (flat)
        return 1.0

    else:
        # decay region
        decay_progress = float(current_step - warmup_stable_steps) / _decay_steps
        assert decay_progress >= 0
        
        if _lr_decay_type == "linear":
            factor = 1 - decay_progress
        elif _lr_decay_type == "sqrt": 
            factor = 1 - math.sqrt(decay_progress)
        elif _lr_decay_type == "cosine": 
            factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        
        return factor


def get_lr_schedulers(optimizers, job_config: JobConfig):
    def _get_lr_scheduler(optimizer):
        """Build a linear warmup and linear decay scheduler"""
        global _warmup_steps, _decay_steps, _stable_steps, _total_steps, _lr_decay_type, _cosine_end_ratio
        _warmup_steps = int(job_config.training.warmup_steps)
        _total_steps = int(job_config.training.steps)
        _scheduler_type = str(job_config.training.lr_scheduler_type)

        if _scheduler_type == "wsd": 
            _lr_decay_type = str(job_config.training.lr_decay_type)
            _decay_steps = int(job_config.training.decay_steps)
            _stable_steps = _total_steps - _warmup_steps - _decay_steps
            lr_lambda = wsd_schedule
        elif _scheduler_type == "linear_warmup_linear_decay":
            _decay_steps = max(1, _total_steps - _warmup_steps)
            _stable_steps = 0
            lr_lambda = linear_warmup_linear_decay
        # linear warmup + cosine decay to 10% of peak lr
        elif _scheduler_type == "cosine":
            _cosine_end_ratio = float(job_config.training.cosine_end_ratio)
            _decay_steps = max(1, _total_steps - _warmup_steps)
            _stable_steps = 0
            lr_lambda = linear_warmup_cosine_decay
        else: 
            raise

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return warmup_scheduler

    class SchedulersContainer:
        """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

        def __init__(self, schedulers):
            self.schedulers = schedulers

        def step(self):
            for schedulers in self.schedulers:
                schedulers.step()

    return SchedulersContainer(
        [_get_lr_scheduler(optimizer) for optimizer in optimizers]
    )
