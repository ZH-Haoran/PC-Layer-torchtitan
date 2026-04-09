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


from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Load your pre-trained LLaMA model and tokenizer
model_name = "path_to_your_pretrained_llama_model"  # Replace with your model path
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Load the MMLU dataset (choose a specific sub-task)
dataset = load_dataset("mmlu", "all")  # Specify the appropriate sub-task, if needed

# Define the function to evaluate the model
def evaluate_mmlu(model, tokenizer, dataset, task_name='arithmetic', batch_size=8):
    # Extract a subset of the task
    task_data = dataset[task_name]

    model.eval()
    all_preds = []
    all_labels = []

    # Iterate over the dataset in batches
    for i in range(0, len(task_data), batch_size):
        batch = task_data[i:i + batch_size]
        questions = [example['input'] for example in batch]
        answers = [example['output'] for example in batch]

        # Tokenize the questions and answers
        encodings = tokenizer(questions, padding=True, truncation=True, return_tensors="pt", max_length=512)
        labels = tokenizer(answers, padding=True, truncation=True, return_tensors="pt", max_length=512)['input_ids']

        # Move tensors to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        encodings = {key: value.to(device) for key, value in encodings.items()}
        labels = labels.to(device)

        # Forward pass to get predictions
        with torch.no_grad():
            outputs = model.generate(input_ids=encodings['input_ids'], max_length=512, num_return_sequences=1)

        # Decode the generated token ids to strings
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Collect the predictions and true labels
        all_preds.extend(preds)
        all_labels.extend(answers)

    # Calculate evaluation metric (e.g., accuracy)
    acc = accuracy_score(all_labels, all_preds)
    return acc

# Evaluate the model on the MMLU task
acc = evaluate_mmlu(model, tokenizer, dataset, task_name='arithmetic')  # You can change task_name to another task
print(f"Accuracy on MMLU arithmetic task: {acc}")
