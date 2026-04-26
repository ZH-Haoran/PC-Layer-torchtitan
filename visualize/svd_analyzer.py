"""SVD analysis module: load checkpoint, compute SVD, save singular values.

Supports PCLinear layers: computes SVD for both original weights and PC-transformed weights.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm



# Mapping from weight key to PC config flag
_WEIGHT_KEY_TO_PC_FLAG = {
    'wq': 'precondition_qk', 'wk': 'precondition_qk',
    'wv': 'precondition_v', 'wo': 'precondition_o',
    'w1': 'precondition_w1', 'w2': 'precondition_w2', 'w3': 'precondition_w3',
    'c_fc': 'precondition_mlp', 'c_proj': 'precondition_mlp',
}


def _get_weight_key(layer_name: str) -> Optional[str]:
    """Extract weight key from layer name."""
    for key in _WEIGHT_KEY_TO_PC_FLAG:
        if f'.{key}' in layer_name or layer_name.endswith(f'.{key}'):
            return key
    return None


def _should_apply_pc(layer_name: str, model_config) -> bool:
    """Check if PC should be applied to this layer based on config."""
    weight_key = _get_weight_key(layer_name)
    if weight_key is None:
        return False
    pc_flag = _WEIGHT_KEY_TO_PC_FLAG.get(weight_key)
    if pc_flag is None:
        return False
    return getattr(model_config, pc_flag, False)


def extract_weight_matrices(model) -> Dict[str, torch.Tensor]:
    """Extract attention and MLP weight matrices from model.

    Supports both Llama and GPT naming conventions:
    - Llama: layers.{i}.attention.wq/wk/wv/wo, layers.{i}.feed_forward.w1/w2/w3
    - GPT: transformer.h.{i}.attn.wq/wk/wv/wo, transformer.h.{i}.mlp.c_fc/c_proj

    Handles PCLinear wrapper: layers.{i}.attention.wq.linear.weight

    Args:
        model: PyTorch model

    Returns:
        Dict mapping layer names to weight tensors
    """
    weights = {}
    weight_keys = ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3', 'c_fc', 'c_proj']

    for name, param in model.named_parameters():
        # Check if this is a weight parameter we care about
        if any(key in name for key in weight_keys) and name.endswith('weight'):
            # Extract layer name (handle PCLinear wrapper)
            parts = name.split('.')
            if parts[-2] == 'linear' and len(parts) >= 5:
                # Remove .linear.weight suffix for PCLinear
                layer_key = '.'.join(parts[:-2])
            else:
                # Remove .weight suffix
                layer_key = '.'.join(parts[:-1])

            weights[layer_key] = param.detach()

    return weights


def compute_svd(weights: Dict[str, torch.Tensor], step: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """Compute SVD for each weight matrix.

    Args:
        weights: Dict mapping layer names to weight tensors
        step: Optional checkpoint step number for progress display

    Returns:
        Dict mapping layer names to singular values
    """
    svd_results = {}

    desc = f"Computing SVD (step {step})" if step is not None else "Computing SVD"
    for name, weight in tqdm(weights.items(), desc=desc, unit="layer"):
        # SVD: weight = U @ S @ V^T
        _, S, _ = torch.linalg.svd(weight, full_matrices=False)
        svd_results[name] = {
            "shape": list(weight.shape),
            "singular_values": S.cpu().tolist()
        }

    return svd_results


def extract_op_norms(model) -> Dict[str, torch.Tensor]:
    """Extract approximate op norms from PCLinear modules.

    Primary path: if op_u/op_v were saved in the checkpoint (persistent buffers),
    use _compute_op_norm_from_state() — identical to training-time approximation.

    Fallback path: if op_u/op_v are absent or mis-sized (e.g. checkpoint was saved
    before op-norm tracking), run power iteration from random initialization for
    model_args.power_iter steps and compute u^T W v + eps directly.

    Only called for modules where pc_norm_type == 'op'.

    Args:
        model: PyTorch model with loaded checkpoint weights

    Returns:
        Dict mapping module names to op norm tensors
    """
    from torchtitan.pc_layer.pc_layer import PCLinear

    op_norms = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, PCLinear):
            continue
        if not module._uses_op_norm():
            continue

        weight = module.linear.weight
        op_weight = module._get_weight_for_op(weight)

        with torch.no_grad():
            if module._has_valid_op_state(weight):
                # Checkpoint contains valid op_u/op_v: reuse them directly
                op_norm = module._compute_op_norm_from_state(weight)
            else:
                # Fallback: random init + power_iter steps of power iteration
                u = module._random_unit_vector(op_weight.size(0), op_weight)
                v = module._random_unit_vector(op_weight.size(1), op_weight)
                for _ in range(module.model_args.power_iter):
                    v = module._normalize_vector(torch.mv(op_weight.T, u))
                    u = module._normalize_vector(torch.mv(op_weight, v))
                wv = torch.mv(op_weight, v)
                op_norm_local = torch.dot(u, wv) + module.model_args.pc_norm_eps
                op_norm = module._wrap_scalar_like_weight(op_norm_local, weight)

        op_norms[module_name] = op_norm
    return op_norms


def apply_pc_transform(weight: torch.Tensor, model_config, op_norm=None, return_norm=False):
    """Apply PC transform to a weight matrix using config settings.

    Args:
        weight: Weight tensor (2D)
        model_config: Model config with PC settings
        op_norm: Optional pre-computed op norm (required when pc_norm_type == 'op')
        return_norm: If True, return (W_preconditioned, W_norm) tuple

    Returns:
        PC-transformed weight tensor, or (W_preconditioned, W_norm) if return_norm=True
    """
    from torchtitan.pc_layer.pc_layer import PCTransform

    pc = PCTransform(model_config)
    return pc(weight, op_norm=op_norm, return_norm=return_norm)


def compute_svd_with_pc(weights: Dict[str, torch.Tensor], model_config, step: Optional[int] = None, op_norms: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Dict[str, Any]]:
    """Compute SVD for weights, including PC-transformed versions where applicable.

    For layers with PC enabled, computes both:
    - SVD of original weight
    - SVD of PC-transformed weight

    Args:
        weights: Dict mapping layer names to weight tensors
        model_config: Model config with PC settings
        step: Optional checkpoint step number for progress display

    Returns:
        Dict mapping layer names to singular values (with optional PC versions)
    """
    svd_results = {}

    desc = f"Computing SVD (step {step})" if step is not None else "Computing SVD"
    for name, weight in tqdm(weights.items(), desc=desc, unit="layer"):
        # SVD of original weight
        _, S, _ = torch.linalg.svd(weight, full_matrices=False)
        result = {
            "shape": list(weight.shape),
            "singular_values": S.cpu().tolist()
        }

        # Check if PC should be applied to this layer
        if _should_apply_pc(name, model_config):
            # For op norm: use pre-loaded PCLinear op_u/op_v approximation
            pc_norm_type = getattr(model_config, 'pc_norm_type', 'F')
            op_norm_val = op_norms.get(name) if (pc_norm_type == 'op' and op_norms) else None

            # return_norm=True: W_norm is consistent with pc_norm_type (F or op)
            pc_weight, W_norm = apply_pc_transform(
                weight, model_config, op_norm=op_norm_val, return_norm=True
            )
            _, S_pc, _ = torch.linalg.svd(pc_weight, full_matrices=False)
            result["singular_values_pc"] = S_pc.cpu().tolist()
            result["weight_norm"] = float(W_norm.item())

        svd_results[name] = result

    return svd_results


def has_any_pc_enabled(model_config) -> bool:
    """Check if any PC flag is enabled in the config."""
    pc_flags = ['precondition_w1', 'precondition_w2', 'precondition_w3',
                'precondition_o', 'precondition_qk', 'precondition_v']
    return any(getattr(model_config, flag, False) for flag in pc_flags)


def compute_val_loss(model, data_loader, num_batches: int = 1, device: str = "cuda") -> float:
    """Compute validation loss on a few batches to verify checkpoint loading.

    Args:
        model: PyTorch model
        data_loader: Validation data loader
        num_batches: Number of batches to compute loss on (from config: training.num_val_batch)
        device: Device to run on

    Returns:
        Average loss over the batches
    """
    model.eval()
    loss_list = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), total=num_batches, desc="Validating", unit="batch"):
            if i >= num_batches:
                break
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            pred = model(input_ids)
            loss = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))
            loss_list.append(loss.item())

    avg_loss = np.mean(loss_list) if loss_list else float('nan')
    return avg_loss


def build_val_dataloader(job_config, tokenizer) -> Optional[Any]:
    """Build validation data loader for loss verification.

    Args:
        job_config: JobConfig with training settings
        tokenizer: Tokenizer instance

    Returns:
        DataLoader for validation, or None if dataset path not configured
    """
    from torchtitan.datasets.hf_datasets import build_hf_data_loader
    from torchtitan.logging_utils import logger

    # Check if validation dataset path is configured
    dataset_val_path = getattr(job_config.training, 'dataset_val_path', None)
    if not dataset_val_path:
        logger.warning("No validation dataset path configured (training.dataset_val_path), skipping val loss check")
        return None

    batch_size = getattr(job_config.training, 'batch_size', 1)
    seq_len = getattr(job_config.training, 'seq_len', 2048)

    try:
        data_loader = build_hf_data_loader(
            "fineweb",
            dataset_val_path,
            tokenizer,
            batch_size,
            seq_len,
            world_size=1,  # Single GPU for visualization
            rank=0,
            infinite=False,
        )
        return data_loader
    except Exception as e:
        from torchtitan.logging_utils import logger
        logger.warning(f"Failed to build validation dataloader: {e}")
        return None


def save_singular_values(svd_results: Dict[str, Dict[str, Any]],
                         metadata: Dict[str, Any],
                         output_dir: str = "visualization_output",
                         step: Optional[int] = None) -> str:
    """Save SVD results to JSON file.

    Args:
        svd_results: Dict of layer names to singular values
        metadata: Metadata about the analysis
        output_dir: Output directory
        step: Optional checkpoint step number. If provided, included in filename.

    Returns:
        Path to saved JSON file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if step is not None:
        filename = f"singular_values_step_{step}.json"
    else:
        filename = "singular_values.json"
    filepath = output_path / filename

    data = {
        "metadata": metadata,
        "layers": svd_results
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    return str(filepath)


def get_output_dir(job_config) -> str:
    """Get output directory path based on config.

    Format: visualization_output/{model.name}_{model.flavor}_{optimizer.name}_{metrics.wandb_comment}
    """
    model_name = job_config.model.name
    model_flavor = job_config.model.flavor
    optimizer_name = job_config.optimizer.name
    wandb_comment = job_config.metrics.wandb_comment

    dir_name = f"{model_name}_{model_flavor}_{optimizer_name}_{wandb_comment}"
    return f"visualization_output/{dir_name}"


def list_checkpoint_steps(checkpoint_folder: str) -> List[int]:
    """List all checkpoint step numbers in the folder.

    Args:
        checkpoint_folder: Path to the checkpoint folder

    Returns:
        Sorted list of step numbers
    """
    if not os.path.isdir(checkpoint_folder):
        return []

    step_counts = []
    for filename in os.listdir(checkpoint_folder):
        match = re.search(r"step-(\d+)", filename)
        metadata_probe = os.path.join(checkpoint_folder, filename, ".metadata")
        if match and os.path.isfile(metadata_probe):
            step_counts.append(int(match.group(1)))

    return sorted(step_counts)


def parse_visualize_steps(step_config: str, available_steps: List[int]) -> List[int]:
    """Parse the visualize.step config and return filtered list of steps.

    Args:
        step_config: String from config, either "-1" for all steps or comma-separated list
        available_steps: List of all available checkpoint steps

    Returns:
        Sorted list of steps to process
    """
    step_config = step_config.strip()

    if step_config == "-1" or step_config == "":
        return available_steps

    # Parse comma-separated list
    try:
        requested_steps = [int(s.strip()) for s in step_config.split(",")]
    except ValueError:
        raise ValueError(f"Invalid visualize.step format: '{step_config}'. Use -1 for all steps or comma-separated list like '1,2000,60000'")

    # Filter to only include steps that exist
    valid_steps = [s for s in requested_steps if s in available_steps]
    missing_steps = [s for s in requested_steps if s not in available_steps]

    if missing_steps:
        from torchtitan.logging_utils import logger
        logger.warning(f"Requested steps not found in checkpoints: {missing_steps}")

    return sorted(valid_steps)


def run_visualize(job_config) -> List[str]:
    """Main entry point for visualization mode.

    Loads model, iterates through all checkpoint steps, computes SVD, saves results
    to separate JSON filens per step.

    Args:
        job_config: JobConfig object with model and checkpoint settings

    Returns:
        List of paths to saved JSON files
    """
    import torch.distributed as dist
    from torchtitan.models import model_name_to_cls, models_config
    from torchtitan.checkpoint import CheckpointManager
    from torchtitan.logging_utils import init_logger, logger
    from torchtitan.utils import init_distributed

    init_logger()
    logger.info("Starting visualization mode...")

    # Initialize distributed (needed for checkpoint loading)
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('LOCAL_RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    init_distributed(job_config)

    model_name = job_config.model.name
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]

    # Copy PC-related settings from job_config.model to model_config
    # This is critical for proper model architecture matching the checkpoint
    MODEL_CONFIG_KEYS = {
        'norm_type',
        'precondition_w1',
        'precondition_w2',
        'precondition_w3',
        'precondition_o',
        'precondition_qk',
        'precondition_v',
        'pc_qkv_per_head',
        'power_iter',
        'pc_norm_type',
        'pc_norm_eps',
        'pc_level',
        'recover_w_norm',
        'scale_constant',
        'learnable_gamma',
        'gamma_init_value',
        'log_signal_propagation',
        'log_gradients',
    }
    for key in MODEL_CONFIG_KEYS:
        if hasattr(job_config.model, key):
            setattr(model_config, key, getattr(job_config.model, key))

    # Get vocab size from tokenizer (use default if not available)
    try:
        from torchtitan.datasets import create_tokenizer
        tokenizer_type = {
            'llama2': 'sentencepiece',
            'llama3': 'tiktoken',
            'gpt2': 'sentencepiece'
        }.get(model_name, 'tiktoken')
        tokenizer = create_tokenizer(tokenizer_type, job_config.model.tokenizer_path)
        model_config.vocab_size = tokenizer.n_words
    except Exception as e:
        logger.warning(f"Could not load tokenizer, using default vocab_size: {e}")
        model_config.vocab_size = getattr(model_config, 'vocab_size', 32000)

    model_config.max_seq_len = getattr(job_config.training, 'seq_len', 2048)

    logger.info(f"Building {model_name} {job_config.model.flavor}")

    # Build checkpoint path using the same logic as train.py
    checkpoint_path = (
        f"{job_config.checkpoint.folder}/"
        f"{job_config.model.name}_{job_config.model.flavor}/"
        f"{job_config.optimizer.name}/"
        f"{job_config.metrics.wandb_comment}"
    )
    logger.info(f"Checkpoint folder: {checkpoint_path}")

    # List all available checkpoint steps
    all_steps = list_checkpoint_steps(checkpoint_path)
    if not all_steps:
        logger.warning(f"No checkpoints found in {checkpoint_path}")
        return []

    logger.info(f"Found {len(all_steps)} checkpoint steps: {all_steps}")

    # Filter steps based on visualize.step config
    step_config = getattr(job_config.visualize, 'step', '-1')
    steps_to_process = parse_visualize_steps(step_config, all_steps)

    if not steps_to_process:
        logger.warning(f"No valid checkpoint steps to process (requested: '{step_config}')")
        return []

    logger.info(f"Will process {len(steps_to_process)} checkpoint steps: {steps_to_process}")

    # Check if PC is enabled (use job_config.model, not model_config)
    pc_enabled = has_any_pc_enabled(job_config.model)

    output_dir = get_output_dir(job_config)
    saved_paths = []

    # Build validation dataloader for loss verification (only once)
    val_dataloader = None
    tokenizer_for_val = None
    try:
        from torchtitan.datasets import create_tokenizer
        tokenizer_type = {
            'llama2': 'sentencepiece',
            'llama3': 'tiktoken',
            'gpt2': 'sentencepiece'
        }.get(model_name, 'tiktoken')
        tokenizer_for_val = create_tokenizer(tokenizer_type, job_config.model.tokenizer_path)
        val_dataloader = build_val_dataloader(job_config, tokenizer_for_val)
    except Exception as e:
        logger.warning(f"Could not build validation dataloader: {e}")

    # Process each checkpoint step
    for step in steps_to_process:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing checkpoint step {step}")
        logger.info(f"{'='*50}")

        # Build model with meta init (fresh model for each checkpoint)
        with torch.device("meta"):
            model = model_cls.from_model_args(model_config)

        # Move to device and initialize
        model.to_empty(device=device)
        model.init_weights()

        # Create a simple train state for checkpoint loading
        class TrainState:
            def __init__(self):
                self.step = 0
                self.log_steps = []
                self.global_avg_losses = []
                self.global_max_losses = []

        train_state = TrainState()

        # Create dummy optimizer and lr_scheduler (required by CheckpointManager)
        dummy_optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
        dummy_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(dummy_optimizer)

        # Modify checkpoint folder to match train.py behavior
        # This is critical for finding the correct checkpoint path
        original_checkpoint_folder = job_config.checkpoint.folder
        job_config.checkpoint.folder = (
            f"{original_checkpoint_folder}/"
            f"{job_config.model.name}_{job_config.model.flavor}/"
            f"{job_config.optimizer.name}/"
            f"{job_config.metrics.wandb_comment}"
        )
        logger.info(f"Checkpoint folder: {job_config.checkpoint.folder}")

        # Load checkpoint for specific step - only load model weights
        checkpoint = CheckpointManager(
            model_parts=[model],
            optimizers=[dummy_optimizer],
            lr_schedulers=[dummy_lr_scheduler],
            dataloader=None,
            states={"train_state": train_state},
            job_config=job_config,
        )

        # Only load model weights, skip optimizer/lr_scheduler/dataloader/train_state
        # This avoids state_dict structure mismatches
        import torch.distributed.checkpoint as dcp
        from torchtitan.checkpoint import ModelWrapper

        model_wrapper = ModelWrapper(model)
        checkpoint_id = checkpoint._create_checkpoint_id(step)
        logger.info(f"Loading model weights from checkpoint at step {step}")
        dcp.load({"model": model_wrapper}, checkpoint_id=checkpoint_id)

        # Restore original folder (in case it's needed elsewhere)
        job_config.checkpoint.folder = original_checkpoint_folder

        model.eval()

        # Verify checkpoint by computing val loss
        val_loss = None
        if val_dataloader is not None:
            try:
                if hasattr(val_dataloader.dataset, "reset"):
                    val_dataloader.dataset.reset()
                # num_val_batch = getattr(job_config.training, 'num_val_batch', 1)
                num_val_batch = 5
                val_loss = compute_val_loss(model, val_dataloader, num_batches=num_val_batch, device=str(device))
                logger.info(f"[Checkpoint Verification] Step {step} - Val loss: {val_loss:.4f} (over {num_val_batch} batches)")
            except Exception as e:
                logger.warning(f"Failed to compute val loss for step {step}: {e}")

        # Extract weights and compute SVD
        logger.info("Extracting weight matrices...")
        weights = extract_weight_matrices(model)
        logger.info(f"Found {len(weights)} weight matrices")

        logger.info("Computing SVD...")
        if pc_enabled:
            logger.info("PC enabled, computing SVD for both original and PC-transformed weights")
            op_norms = extract_op_norms(model)
            svd_results = compute_svd_with_pc(weights, job_config.model, step=step, op_norms=op_norms)
        else:
            svd_results = compute_svd(weights, step=step)

        # Prepare metadata
        metadata = {
            "model_name": job_config.model.name,
            "model_flavor": job_config.model.flavor,
            "optimizer_name": job_config.optimizer.name,
            "wandb_comment": job_config.metrics.wandb_comment,
            "checkpoint_path": checkpoint_path,
            "checkpoint_step": step,
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "num_layers": len(svd_results),
            "pc_enabled": pc_enabled,
            "val_loss": val_loss,
        }
        if pc_enabled:
            metadata["pc_config"] = {
                "precondition_w1": getattr(job_config.model, 'precondition_w1', False),
                "precondition_w2": getattr(job_config.model, 'precondition_w2', False),
                "precondition_w3": getattr(job_config.model, 'precondition_w3', False),
                "precondition_o": getattr(job_config.model, 'precondition_o', False),
                "precondition_qk": getattr(job_config.model, 'precondition_qk', False),
                "precondition_v": getattr(job_config.model, 'precondition_v', False),
                "pc_norm_type": getattr(job_config.model, 'pc_norm_type', 'F'),
                "pc_level": getattr(job_config.model, 'pc_level', 0),
                "recover_w_norm": getattr(job_config.model, 'recover_w_norm', False),
                "learnable_gamma": getattr(job_config.model, 'learnable_gamma', False),
                "scale_constant": getattr(job_config.model, 'scale_constant', None),
                "learnable_gamma": getattr(job_config.model, 'learnable_gamma', False),
                "gamma_init_value": getattr(job_config.model, 'gamma_init_value', None),
            }

        # Save results (JSON) with step number in filename
        output_path = save_singular_values(svd_results, metadata, output_dir, step=step)
        logger.info(f"Saved singular values to: {output_path}")
        saved_paths.append(output_path)

        # Clean up model to free memory
        del model, dummy_optimizer, dummy_lr_scheduler, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    return saved_paths
