# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies PTD parallelisms and various training techniques to the
# GPT model, i.e. activation checkpointing, tensor parallelism, etc.

from typing import Dict, Tuple

import torch

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torch.utils.checkpoint import checkpoint

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.logging_utils import logger
from torchtitan.parallelisms.parallelize_llama import (
    get_tp_parallel_strategy,
    _mixed_precision_dtype,
)


def checkpoint_wrapper(module, config):
    """Apply activation checkpointing wrapper."""
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )
    from collections import defaultdict

    no_recompute_list = {
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
    }

    if config.mode == "selective" and config.selective_ac_option == "op":
        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                to_save = func in no_recompute_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )
                return (
                    CheckpointPolicy.MUST_SAVE
                    if to_save
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )
            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            context_fn=selective_checkpointing_context_fn,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    elif config.mode == "full":
        return ptd_checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    elif config.mode == "selective" and config.selective_ac_option.isdigit():
        ac_freq = int(config.selective_ac_option)
        checkpoint_wrapper.__dict__.setdefault("_count", 0)
        checkpoint_wrapper._count += 1
        if not ac_freq or checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                checkpoint_fn=checkpoint,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            return module
    else:
        return module


def apply_tp(model, world_mesh, parallel_dims, job_config: JobConfig):
    """
    Apply tensor parallelism to GPT model.
    """
    tp_mesh = world_mesh["tp"]
    (
        row_parallel_strategy,
        col_parallel_strategy,
        prepare_module_input,
    ) = get_tp_parallel_strategy(job_config)
    loss_parallel = parallel_dims.loss_parallel_enabled

    # Parallelize token embeddings (RowwiseParallel - shard along vocab dim)
    model = parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "output": col_parallel_strategy(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
            "norm": SequenceParallel(),
        },
    )

    # Apply tensor + sequence parallelism to every transformer block
    for layer_id, transformer_block in model.layers.items():
        layer_plan = {
            # Attention - GPT uses wq, wk, wv, wo naming
            "attn": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "attn.wq": col_parallel_strategy(),
            "attn.wk": col_parallel_strategy(),
            "attn.wv": col_parallel_strategy(),
            "attn.wo": row_parallel_strategy(output_layouts=Shard(1)),
            # LayerNorms (ln_1 and ln_2 in GPT)
            "ln_1": SequenceParallel(),
            # MLP - GPT uses c_fc (up proj) and c_proj (down proj)
            "mlp": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "mlp.c_fc": col_parallel_strategy(),  # Up projection (like w1/w3 in Llama)
            "mlp.c_proj": row_parallel_strategy(output_layouts=Shard(1)),  # Down projection (like w2)
            "ln_2": SequenceParallel(),
        }

        # Adjust attention module to use the local number of heads
        attn_layer = transformer_block.attn
        attn_layer.n_head = attn_layer.n_head // tp_mesh.size()

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if job_config.experimental.enable_async_tensor_parallel:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group
        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info("Applied Tensor Parallelism to GPT model")
    return model


def apply_ac(model, job_config: JobConfig):
    """
    Apply activation checkpointing to the model.
    """
    ac_config = job_config.activation_checkpoint

    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = checkpoint_wrapper(transformer_block, ac_config)
        model.layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to GPT model")
    return model


def apply_compile(model, job_config: JobConfig):
    """
    Apply torch.compile to the model.
    """
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(transformer_block, dynamic=False)
        model.layers.register_module(layer_id, transformer_block)

    ac_config = job_config.activation_checkpoint
    if ac_config.mode == "selective" and ac_config.selective_ac_option == "op":
        torch._dynamo.config._experimental_support_context_fn_in_torch_utils_checkpoint = True

    logger.info("Compiled each GPT Block with torch.compile")
    return model


def apply_dp(model, world_mesh, parallel_dims, job_config: JobConfig):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    dp_mesh = world_mesh["dp"] if world_mesh.ndim > 1 else world_mesh
    assert dp_mesh.mesh_dim_names == ("dp",), dp_mesh.mesh_dim_names

    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    for layer_id, transformer_block in model.layers.items():
        reshard_after_forward = (
            int(layer_id) < len(model.layers) - 1 and not parallel_dims.pp_enabled
        )
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
        model.layers[layer_id] = transformer_block

    model = fully_shard(
        model, **fsdp_config, reshard_after_forward=not parallel_dims.pp_enabled
    )

    logger.info("Applied FSDP to GPT model")
    return model


def parallelize_gpt(model, world_mesh, parallel_dims, job_config: JobConfig):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the GPT model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    if parallel_dims.tp_enabled:
        model = apply_tp(model, world_mesh, parallel_dims, job_config)

    if job_config.activation_checkpoint.mode != "none":
        model = apply_ac(model, job_config)

    if job_config.training.compile:
        model = apply_compile(model, job_config)

    if parallel_dims.dp_enabled:
        model = apply_dp(model, world_mesh, parallel_dims, job_config)

    return model


def pipeline_gpt(
    model, world_mesh, parallel_dims, job_config: JobConfig, device, model_config: Dict
):
    """
    Pipeline parallelism for GPT model.
    Currently not implemented - raises NotImplementedError.
    """
    raise NotImplementedError(
        "Pipeline parallelism for GPT model is not yet implemented. "
        "Please use tensor parallelism and data parallelism instead."
    )
