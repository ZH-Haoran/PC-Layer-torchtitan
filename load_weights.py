# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import gc
import os
import time

from dataclasses import dataclass, field
from datetime import timedelta
from io import BytesIO
from timeit import default_timer as timer
from typing import Any, Dict, List

import numpy as np

import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor.parallel import loss_parallel
from torchtitan.checkpoint import CheckpointManager
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_hf_data_loader, create_tokenizer
from torchtitan.float8_linear import build_fp8_linear
from torchtitan.logging_utils import init_logger, logger
from torchtitan.lr_scheduling import get_lr_schedulers
from torchtitan.metrics import build_gpu_memory_monitor, build_metric_logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.parallelisms import (
    models_parallelize_fns,
    models_pipelining_fns,
    ParallelDims,
)
from torchtitan.parallelisms.pipelining_utils import build_pipeline_schedule
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torchtitan.utils import (
    Color,
    dist_max,
    dist_mean,
    get_metrics_rank,
    get_num_flop_per_token,
    get_num_params,
    get_peak_flops,
    init_distributed,
    NoColor,
    set_pg_timeouts,
)
import torch.distributed as dist
from adam_mini import Adam_mini
import wandb

from matplotlib import pyplot as plt
import json
from torchtitan.models.llama.model import apply_preconditioner, pc_normalize

@dataclass
class TrainState(Stateful):
    step: int = 0
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        # Only checkpoint global_avg_losses and global_max_losses per log frequency
        # to avoid sync overhead in every iteration.
        global_avg_losses_bytes = BytesIO()
        torch.save(self.global_avg_losses, global_avg_losses_bytes)
        global_max_losses_bytes = BytesIO()
        torch.save(self.global_max_losses, global_max_losses_bytes)
        log_steps_bytes = BytesIO()
        torch.save(self.log_steps, log_steps_bytes)
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "global_avg_losses": global_avg_losses_bytes,
            "global_max_losses": global_max_losses_bytes,
            "log_steps": log_steps_bytes,
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        state_dict["global_avg_losses"].seek(0)
        self.global_avg_losses = torch.load(
            state_dict["global_avg_losses"], weights_only=False
        )
        state_dict["global_max_losses"].seek(0)
        self.global_max_losses = torch.load(
            state_dict["global_max_losses"], weights_only=False
        )
        state_dict["log_steps"].seek(0)
        self.log_steps = torch.load(state_dict["log_steps"], weights_only=False)


def build_optimizers(model_parts, job_config: JobConfig, world_mesh=None):
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """

    def _build_optimizer(model, world_mesh=None):
        name = job_config.optimizer.name
        lr = job_config.optimizer.lr
        fused = False # job_config.optimizer.fused

        # Common parameters for both optimizers
        optimizer_kwargs = {
            "lr": lr,
            "betas": (0.9, 0.95),
            "weight_decay": 0.1,
            "fused": fused,
            "foreach": not fused,
        }
        if name == "Adam":
            # TODO: make the optimizer options configurable by toml/cmd args
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        elif name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
            print(f"======>>>>> Using AdamW optimizer")
            logger.info(f"======>>>>> Using AdamW optimizer, lr = {lr}")
        elif name == "adam_mini":
            optimizer = Adam_mini(
                        named_parameters = model.named_parameters(),
                        lr = lr,
                        betas = (0.9,0.95),
                        weight_decay = 0.1,
                        dim = model_config.dim,
                        n_heads = model_config.n_heads,
                        n_kv_heads = model_config.n_kv_heads,
                        )
            #optimizer.embd_names = {"embed", "embd", "wte"}  # move to mlp
            # Output layers. Use one lr per token
            #optimizer.output_names = {"lm_head.weight", "output.weight"}  # move output to mlp
            # Query and Keys. User one lr per head
            #optimizer.wqk_names = {}
            # Values. Use one lr per neuron
            # it is okay to set self.wv_names to be empty and use a single lr for the whole v. But this  will bring extra all_reduce operations
            optimizer.wv_names = {}
            # attn_proj. Use one lr per neuron
            #optimizer.attn_proj_names = {}
            #optimizer.adam_block_names = {}
            #optimizer.adam_block_names = {"embed", "embd", "wte", "lm_head.weight", "output.weight"}
            # MLPs. Use one lr per neuron
            #optimizer.mlp_names = {}
            logger.info(f"======>>>>> Using Adam-mini optimizer, lr = {lr}")
        else:
            raise NotImplementedError(f"Optimizer {name} not added.")

        return optimizer

    class OptimizersContainer:
        """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages"""

        def __init__(self, optimizers):
            self.optimizers = optimizers

        def step(self):
            for optimizer in self.optimizers:
                optimizer.step()
            #pass

        def zero_grad(self):
            for optimizer in self.optimizers:
                optimizer.zero_grad()

    return OptimizersContainer([_build_optimizer(model) for model in model_parts])



def validate(job_config, model, data_loader_val, loss_fn, current_step):
    model.eval()
    loss_list = []
    total_tokens = 0
    num_val_batch = job_config.training.num_val_batch # limit the number of data for validation
    current_batch_idx = 0
    logger.info(f"Calculating validation loss...")
    with torch.no_grad():
        for batch in data_loader_val:
            current_batch_idx  += 1
            if current_batch_idx > num_val_batch:
                break
            input_ids, labels = batch
            input_ids = input_ids.cuda()
            labels = labels.cuda()
            pred = model(input_ids)
            loss = loss_fn(pred, labels)
            loss_list.append( loss.item() )
            total_tokens += labels.numel()

    avg_loss = np.mean(loss_list)
    logger.info(f"Validation completed: step: {current_step}, val loss: {avg_loss} val token: {total_tokens}")
    model.train()
    return avg_loss


def svd_all_params(model_config,job_config, train_state, model,global_rank):

    model.eval()

    os.makedirs(f"figures/{job_config.metrics.wandb_comment}", exist_ok=True)
    
    os.makedirs(f"files/{job_config.metrics.wandb_comment}", exist_ok=True)
    
    # read dict from json file if the file exists, otherwise create an empty dict
    if os.path.exists(f'files/{job_config.metrics.wandb_comment}/pc_kappa_dict_{job_config.metrics.wandb_comment}.json'):
        with open(f'files/{job_config.metrics.wandb_comment}/pc_kappa_dict_{job_config.metrics.wandb_comment}.json', 'r') as f:
            pc_kappa_dict = json.load(f)

        # with open(f'files/pc_kappa_normalized_dict_{job_config.metrics.wandb_comment}.json', 'r') as f:
        #     pc_kappa_normalized_dict = json.load(f)

        with open(f'files/{job_config.metrics.wandb_comment}/pc_kappa_preconditioned_dict_{job_config.metrics.wandb_comment}.json', 'r') as f:
            pc_kappa_preconditioned_dict = json.load(f)

    else:
        pc_kappa_dict = {}
        #pc_kappa_normalized_dict = {}
        pc_kappa_preconditioned_dict = {}
        # initialize the dict with empty lists
        for name, param in model.named_parameters():
            # only consider the first layers and the last layer
            if ('feed_forward' in name.lower() and param.ndim == 2):
                pc_kappa_dict[name] = []
                #pc_kappa_normalized_dict[name] = []
                pc_kappa_preconditioned_dict[name] = []

   

    # SVD over every 2D weight matrix
    for name, param in model.named_parameters():


        # only consider the first layers and the last layer
        if ('feed_forward' in name.lower() and param.ndim == 2):
            
            # need to all gather before svd
            power_iter = model_config.power_iter
            local_param = param.data.to_local()
            # all gather local_param
            world_size = dist.get_world_size()
            gathered_param = [torch.zeros_like(local_param) for _ in range(world_size)]
            dist.all_gather(gathered_param, local_param)
            param = torch.cat(gathered_param, dim=0)

            print(f"SVDing name: {name}, param.shape: {param.shape}")

                #param_normalized_op,_ = pc_normalize(param)
                #param_normalized_F = normalize_W(param, pc_norm_type = "F")
                
            

            
            r, c = param.size()
            # print(f"before all gather: name: {name}, r: {r}, c: {c}")
            # do we need allgather before svd?
            #param = param.all_gather()
            #print(f"after all gather: name: {name}, r: {r}, c: {c}")
            _, S, _ = torch.svd(param)
            #S = S.to_local()
            sin_num = max(1, int(S.shape[0] * 0.1))
            kappa = S[0] / (S[-sin_num:]).mean()    
            S = sorted(S.detach().clone().cpu().tolist())

            # _, S_normalized_op, _ = torch.svd(param_normalized_op)
            # #S_normalized_op = S_normalized_op.to_local()

            # sin_num = max(1, int(S_normalized_op.shape[0] * 0.1))
            # kappa_normalized_op = S_normalized_op[0] / (S_normalized_op[-sin_num:]).mean()    
            # S_normalized_op = sorted(S_normalized_op.detach().clone().cpu().tolist())

            param_preconditioned_op = apply_preconditioner(weight = param, pc_level = model_config.pc_level, pc_norm_type = model_config.pc_norm_type, pc_norm_eps = model_config.pc_norm_eps, power_iter = model_config.power_iter)
            
            _, S_preconditioned_op, _ = torch.svd(param_preconditioned_op)
            #S_preconditioned_op = S_preconditioned_op.to_local()
            sin_num = max(1, int(S_preconditioned_op.shape[0] * 0.1))
            kappa_preconditioned_op = S_preconditioned_op[0] / (S_preconditioned_op[-sin_num:]).mean()    
            S_preconditioned_op = sorted(S_preconditioned_op.detach().clone().cpu().tolist())

           
            if global_rank == 0:
                pc_kappa_dict[name].append(kappa.item())
                #pc_kappa_normalized_dict[name].append(kappa_normalized_op.item())
                pc_kappa_preconditioned_dict[name].append(kappa_preconditioned_op.item())

                # plot the list of kappa
                plt.figure(figsize=(10, 5))
                plt.title(f"Kappa of {name}")
                plt.plot(pc_kappa_dict[name], label = "kappa of W")
                # plt.plot(pc_kappa_normalized_dict[name], label = "kappa of W / ||W||_op")
                plt.plot(pc_kappa_preconditioned_dict[name], label = "kappa of P(WW^T)W")
                plt.legend()
                plt.savefig(f"figures/{job_config.metrics.wandb_comment}/kappa_{name}.png")
                plt.close()
                # save the dict to a json file
                with open(f'files/{job_config.metrics.wandb_comment}/pc_kappa_dict_{job_config.metrics.wandb_comment}.json', 'w') as f:
                    json.dump(pc_kappa_dict, f)

                # with open(f'files/pc_kappa_normalized_dict_{job_config.metrics.wandb_comment}.json', 'w') as f:
                #     json.dump(pc_kappa_normalized_dict, f)
                
                with open(f'files/{job_config.metrics.wandb_comment}/pc_kappa_preconditioned_dict_{job_config.metrics.wandb_comment}.json', 'w') as f:
                    json.dump(pc_kappa_preconditioned_dict, f)

                logger.info(f"SVDing {name}, S len = {len(S)}")
                # plot the histogram of the singular values
                plt.figure(figsize=(10, 5))
                plt.title(f"Singular values of {name} at step {train_state.step}")
                plt.hist(S, bins = max(r, c) - 50, density=True, label = "W")
                #plt.hist(S_normalized_op, bins = max(r, c) - 50, density=True, label = f"W / ||W||_op with power iter = {power_iter}")
                plt.hist(S_preconditioned_op, bins = max(r, c) - 50, density=True, label = f"P(WW^T)W, normalized by op-norm with power method with {power_iter} steps, PC-level = {job_config.model.pc_level}") 
                plt.legend()
                plt.ylim(0, 50)
                plt.savefig(f"figures/{job_config.metrics.wandb_comment}/singular_values_step_{train_state.step}_{name}.png")
                plt.close()



# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    # used for colorful printing
    color = Color if job_config.metrics.enable_color_printing else NoColor

    # take control of garbage collection to avoid stragglers
    _gc_freq = job_config.training.gc_freq
    gc.disable()
    gc.collect(1)

    # init distributed
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp=job_config.training.data_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    init_distributed(job_config)

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0
        dp_mesh = None # is this correct?

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    model_name = job_config.model.name

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = create_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    # build dataloader
    data_loader = build_hf_data_loader(
        job_config.training.dataset,
        job_config.training.dataset_path,
        tokenizer,
        job_config.training.batch_size,
        job_config.training.seq_len,
        dp_degree,
        dp_rank,
    )

    # validation dataloader use c4 mini

    data_loader_val = build_hf_data_loader(
        "c4_mini",
        "./torchtitan/datasets/c4_mini/",
        tokenizer,
        job_config.training.batch_size,
        job_config.training.seq_len,
        dp_degree,
        dp_rank,
    )

    # loss_parallel enables dispatching to efficient loss operators
    loss_parallel_ctx = (
        loss_parallel if parallel_dims.loss_parallel_enabled else contextlib.nullcontext
    )

    # loss fn can be shared by pipeline-parallel or non-pp execution
    def loss_fn(pred, labels):
        return F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    global model_config
    model_config = models_config[model_name][job_config.model.flavor]

    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = job_config.training.seq_len
    model_config.precondition_w1 = job_config.model.precondition_w1
    model_config.precondition_w2 = job_config.model.precondition_w2
    model_config.precondition_w3 = job_config.model.precondition_w3
    model_config.power_iter = job_config.model.power_iter
    model_config.pc_norm_type = job_config.model.pc_norm_type
    model_config.pc_norm_eps = job_config.model.pc_norm_eps
    model_config.pc_level = job_config.model.pc_level


    run_id = job_config.job.description + job_config.model.flavor + job_config.optimizer.name + str(job_config.optimizer.lr) + job_config.metrics.wandb_comment
    logger.info(f"=========> Currently running: {run_id}")
    logger.info(f"Building {model_name} {job_config.model.flavor} with {model_config}")


 
    with torch.device("meta"):
        whole_model = model_cls.from_model_args(model_config)

    # apply fp8 linear module swap
    if job_config.training.fp8_linear:
        build_fp8_linear(whole_model, job_config)

    # log model size
    model_param_count = get_num_params(whole_model)
    num_flop_per_token = get_num_flop_per_token(
        get_num_params(whole_model, exclude_embedding=True),
        model_config,
        job_config.training.seq_len,
    )
    logger.info(
        f"{color.blue}Model {model_name} {job_config.model.flavor} "
        f"{color.red}size: {model_param_count:,} total parameters{color.reset} "
        f"num_flop_per_token: {num_flop_per_token:,} "
    )

    # initialize GPU memory monitor before applying parallelisms to the model
    gpu_memory_monitor = build_gpu_memory_monitor()
    # obtain the peak flops of bf16 type for MFU calculation
    gpu_peak_flops = get_peak_flops(gpu_memory_monitor.device_name)

    if parallel_dims.pp_enabled:
        stages, model_parts = models_pipelining_fns[model_name](
            whole_model, world_mesh, parallel_dims, job_config, device, model_config
        )
    else:
        # In 1D/2D cases or PP with simple schedules, model_parts is just one item
        # for PP with looped schedules, each item is one stage-model-chunk
        # we iterate all model_parts for applying SPMD parallelism, compilation, optimizer, and checkpointing
        model_parts = [whole_model]

    # apply PT-D DP/TP parallelisms and activation checkpointing
    model_parts = [
        models_parallelize_fns[model_name](m, world_mesh, parallel_dims, job_config)
        for m in model_parts
    ]

    init_device = "cpu" if job_config.checkpoint.create_seed_checkpoint else "cuda"
    for model in model_parts:
        model.to_empty(device=init_device)

    if parallel_dims.pp_enabled:
        pp_schedule = build_pipeline_schedule(
            job_config, parallel_dims, stages, loss_fn
        )
    else:
        # If PP is enabled, we can't rely on init_weights, because some layers are missing.
        # In the future, we may make init_weights handle missing layers, but also have to consider RNG seed propagation.
        # allocate sharded model on GPU and initialize weights via DTensor
        whole_model.init_weights()

    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
    logger.info(
        f"GPU memory usage for model: "
        f"{gpu_mem_stats.max_reserved_gib:.2f}GiB"
        f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = build_optimizers(model_parts, job_config, world_mesh=dp_mesh)
    lr_schedulers = get_lr_schedulers(optimizers.optimizers, job_config)


    metric_logger = build_metric_logger(
        job_config, metrics_log_rank=get_metrics_rank(world_mesh, parallel_dims), run_id = run_id
    )

    if job_config.metrics.enable_wandb and global_rank == 0:
        # if torch.distributed.get_rank() == 0 
        logger.info("Initializing wandb")
        run_id = job_config.model.name + job_config.model.flavor + job_config.optimizer.name + str(job_config.optimizer.lr) + job_config.metrics.wandb_comment
        wandb.init(project=job_config.job.description, name= run_id)


    train_state = TrainState()

    # train loop
    for model in model_parts:
        model.train()

    # load initial checkpoint
    job_config.checkpoint.folder +=  f"/{job_config.model.name}_{job_config.model.flavor}/{job_config.optimizer.name}"
    
    checkpoint = CheckpointManager(
        model_parts=model_parts,
        optimizers=optimizers.optimizers,
        lr_schedulers=lr_schedulers.schedulers,
        dataloader=data_loader,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert (
            world_size == 1
        ), "Must create seed-checkpoint using one gpu, to disable sharding"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint_loaded = checkpoint.load(resume = True) #job_config.checkpoint.resume



    if parallel_dims.pp_enabled and not checkpoint_loaded:
        raise RuntimeError(
            "Pipeline Parallelism requires meta-initialization and loading seed checkpoint. "
            "Please run `./create_seed_checkpoint.sh` and rerun training with `--checkpoint.enable_checkpoint`"
        )

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
                "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(data_loader)

    checkpoint.reset()

    # variables used to keep info for metrics logging
    losses_since_last_log: List[float] = []
    ntokens_since_last_log = 0
    ntokens_total_train = 0
    data_loading_times: List[float] = []
    time_last_log = timer()
    gpu_memory_monitor.reset_peak_stats()

    # train loop
    logger.info(f"Training starts at step {train_state.step + 1}")

    svd_all_params(model_config,job_config, train_state, model,global_rank)

if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    destroy_process_group()