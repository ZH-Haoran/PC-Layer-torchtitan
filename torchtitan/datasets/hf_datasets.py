# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from typing import Any, Dict, List, Optional
import os
import glob
import mmap

import torch
import numpy as np
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

try:
    from torchdata.stateful_dataloader import StatefulDataLoader
except ImportError as e:
    raise ImportError(
        "Please install the latest torchdata nightly to use StatefulDataloader via:"
        "pip3 install --pre torchdata --index-url https://download.pytorch.org/whl/nightly"
    ) from e

from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging_utils import logger

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

# map from dataset name to a local directory, or
# a dataset repository on the HF hub
_supported_datasets = {
    "c4_mini": "torchtitan/datasets/c4_mini",
    "c4": "allenai/c4",
    "fineweb_100B": "/zhaokunxiang/fineweb_100B_tokenized_with_idx",
}


class HuggingFaceDataset(IterableDataset, Stateful):
    """PyTorch Representation of the HuggingFace Dataset.

    Args:
        dataset_name (str): name of the dataset to load
        dataset_path (Optional[str]):
            Path to the dataset in the file system. If provided, data will be loaded
            from this path instead of downloaded.
        tokenizer (Tokenizer):
            Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite (bool): whether to loop infinitely over the dataset

    We currently support the c4 dataset and a subset of it:
    c4_mini (45K training entries)
    c4 (177M training entries - this dataset is streamed due to the size)

    >> c4 (EN) <<:
    c4 cleaned, English version
    Data input format (c4):
    {
    'url': 'https://klyq.com/beginners-bbq-class-taking-place-in-missoula/',
    'text': 'Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at ...',
    'timestamp': '2019-04-25T12:57:54Z'
    }

    Example use (c4):
    >>> ds = HuggingFaceDataset(dataset_name="c4", dataset_path=None, tokenizer=tokenizer)
    >>> for batch in Dataloader(ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        # allow user to pass in a (local or HF hub) path to use unsupported datasets
        if dataset_name not in _supported_datasets:
            if dataset_path:
                logger.warning(
                    f"Dataset {dataset_name} is not tested or verfied. "
                    f"Recommended datasets are: {list(_supported_datasets.keys())}."
                )
            else:
                raise ValueError(
                    f"Dataset {dataset_name} is not supported. "
                    f"Supported datasets are: {list(_supported_datasets.keys())}."
                )

        if not dataset_path:
            dataset_path = _supported_datasets[dataset_name]
        logger.info(f"Preparing {dataset_name} dataset from {dataset_path}")

        if dataset_name == "c4":
            # c4 is huge, and requires both streaming and language selection
            # (we default to en)
            ds = load_dataset(dataset_path, name="en", split="train", streaming=True)
        elif dataset_name == "c4_mini":
            ds = load_dataset(dataset_path, split="train")
        else:
            ds = load_dataset(dataset_path, split="train", streaming = True)

        # TODO: support shuffling and checkpointing
        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite

        # variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                sample_text = sample["text"]
                sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
                self._all_tokens.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._all_tokens) >= max_buffer_token_len:
                    x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._all_tokens = self._all_tokens[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield input, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data.")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(
                    f"Dataset {self.dataset_name} is being re-looped. "
                    "Loss related metrics might be misleading."
                )

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        # Skip samples
        if isinstance(self._data, IterableDataset):
            it = iter(self._data)
            # Naively iterate through the samples as skip may not be supported
            for _ in range(self._sample_idx):
                next(it)
            return it

        # As skipping to the end throws an error in case of map-style dataset, return an empty iterator
        if self._sample_idx == len(self._data):
            return iter([])
        return iter(self._data.skip(self._sample_idx))

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]

    def state_dict(self):
        return {"token_buffer": self._all_tokens, "sample_idx": self._sample_idx}


class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(self, dp_rank: int, hf_ds: IterableDataset, batch_size: int, 
                num_workers: int = 0, prefetch_factor: int = 2, pin_memory: bool = False, 
                persistent_workers: bool = False, **kwargs):
        
        if num_workers > 0:
            super().__init__(
                hf_ds,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                **kwargs,
            )
        else:
            super().__init__(
                hf_ds,
                batch_size=batch_size,
                pin_memory=pin_memory,
                **kwargs,
            )
        
        # super().__init__(hf_ds, batch_size)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid, don't log a warning
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}."
            )
            return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


class BinIdxDataset(IterableDataset, Stateful):
    """PyTorch Dataset for bin+idx format data.
    
    This dataset reads tokenized data from binary files (.bin) with corresponding
    index files (.idx) that store document offsets.
    
    Args:
        dataset_path (str): Path to directory containing .bin and .idx files
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite (bool): whether to loop infinitely over the dataset
    
    File format:
    - .bin files: contain uint32 token sequences concatenated together
    - .idx files: contain int64 offsets marking document boundaries in .bin files
    """
    
    def __init__(
        self,
        dataset_path: str,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        self.dataset_path = dataset_path
        self.seq_len = seq_len
        self.world_size = world_size
        self.rank = rank
        self.infinite = infinite
        
        # Find all bin/idx file pairs
        bin_files = sorted(glob.glob(os.path.join(dataset_path, "*.bin")))
        idx_files = sorted(glob.glob(os.path.join(dataset_path, "*.idx")))
        
        if len(bin_files) != len(idx_files):
            raise ValueError(f"Mismatch between bin files ({len(bin_files)}) and idx files ({len(idx_files)})")
        
        # Distribute files across workers
        files_per_rank = len(bin_files) // world_size
        start_idx = rank * files_per_rank
        if rank == world_size - 1:
            end_idx = len(bin_files)  # Last rank gets remaining files
        else:
            end_idx = start_idx + files_per_rank
            
        self.bin_files = bin_files[start_idx:end_idx]
        self.idx_files = idx_files[start_idx:end_idx]
        
        logger.info(f"Rank {rank} processing {len(self.bin_files)} files: {self.bin_files[0]} to {self.bin_files[-1]}")

        # ADD: 预先把 idx + bin memmap 到进程内，避免反复 open/close
        self._doc_offsets = [
            self._load_document_offsets(p) for p in self.idx_files
        ]
        self._bin_memmaps = [
            np.memmap(p, mode="r", dtype=np.uint32) for p in self.bin_files
        ]
        
        # Variables for checkpointing
        self._current_file_idx = 0
        self._current_doc_idx = 0
        self._all_tokens: List[int] = []
        
    def _load_document_offsets(self, idx_file_path: str) -> np.ndarray:
        """Load document offsets from idx file."""
        return np.fromfile(idx_file_path, dtype=np.int64)
    
    # DELETE
    # def _load_tokens_from_range(self, bin_file_path: str, start_offset: int, end_offset: int) -> np.ndarray:
    #     """Load tokens from bin file within specified range."""
    #     with open(bin_file_path, 'rb') as f:
    #         f.seek(start_offset * 4)  # uint32 = 4 bytes
    #         num_tokens = end_offset - start_offset
    #         tokens = np.fromfile(f, dtype=np.uint32, count=num_tokens)
    #     return tokens.astype(np.int64)  # Convert to int64 for torch
    # DELETE
    # def _load_tokens_from_range(self, bin_file_path: str, start_offset: int, end_offset: int) -> np.ndarray:
    #     """Load tokens from bin file within specified range."""
    #     with open(bin_file_path, 'rb') as f:
    #         f.seek(start_offset * 4)  # uint32 = 4 bytes
    #         num_tokens = end_offset - start_offset
    #         tokens = np.fromfile(f, dtype=np.uint32, count=num_tokens)
    #     return tokens.astype(np.int64)  # Convert to int64 for torch
    
    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len
        
        while True:
            # Process files assigned to this rank
            for file_idx in range(self._current_file_idx, len(self.bin_files)):
                # DELETE
                # bin_file = self.bin_files[file_idx]
                # idx_file = self.idx_files[file_idx]
                # ADD
                bin_tokens = self._bin_memmaps[file_idx]
                doc_offsets = self._doc_offsets[file_idx]
                
                logger.info(f"Rank {self.rank} processing file {self.bin_files[file_idx]}")
                
                
                # DELETE
                # Load document offsets
                # doc_offsets = self._load_document_offsets(idx_file)
                
                # Process documents starting from checkpoint position
                for doc_idx in range(self._current_doc_idx, len(doc_offsets) - 1):
                    start_offset = doc_offsets[doc_idx]
                    end_offset = doc_offsets[doc_idx + 1]
                    
                    # Load tokens for this document
                    # ADD
                    doc_tokens = bin_tokens[start_offset:end_offset].astype(np.int64)
                    # DELETE
                    # doc_tokens = self._load_tokens_from_range(bin_file, start_offset, end_offset)
                    self._all_tokens.extend(doc_tokens.tolist())
                    
                    # Yield sequences when we have enough tokens
                    while len(self._all_tokens) >= max_buffer_token_len:
                        x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                        self._all_tokens = self._all_tokens[max_buffer_token_len:]
                        input_tokens = x[:-1]
                        label_tokens = x[1:]
                        yield input_tokens, label_tokens
                    
                    self._current_doc_idx += 1
                
                # Reset document index for next file
                self._current_doc_idx = 0
                self._current_file_idx += 1
            
            if not self.infinite:
                logger.warning("BinIdx dataset has run out of data.")
                break
            else:
                # Reset for next iteration
                self._current_file_idx = 0
                self._current_doc_idx = 0
                logger.warning("BinIdx dataset is being re-looped.")
    
    def load_state_dict(self, state_dict):
        self._current_file_idx = state_dict.get("current_file_idx", 0)
        self._current_doc_idx = state_dict.get("current_doc_idx", 0)
        self._all_tokens = state_dict.get("token_buffer", [])

    def state_dict(self):
        return {
            "current_file_idx": self._current_file_idx,
            "current_doc_idx": self._current_doc_idx,
            "token_buffer": self._all_tokens,
        }
    def reset(self):
        """Resets the dataset to the beginning for a new epoch."""
        self._current_file_idx = 0
        self._current_doc_idx = 0
        self._all_tokens = []
        logger.info(f"Rank {self.rank} BinIdxDataset has been reset.")


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    world_size,
    rank,
    infinite: bool = True, 
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    **kwargs,
):
    # Check if this is a bin+idx dataset
    if dataset_path and _is_binidx_dataset(dataset_path):
        if not dataset_path:
            dataset_path = _supported_datasets.get(dataset_name)
        
        logger.info(f"Using BinIdxDataset for {dataset_name} from {dataset_path}")
        ds = BinIdxDataset(
            dataset_path, seq_len, world_size, rank, infinite
        )
    else:
        # Use original HuggingFace dataset
        ds = HuggingFaceDataset(
            dataset_name, dataset_path, tokenizer, seq_len, world_size, rank, infinite
        )

    return DPAwareDataLoader(
        rank, 
        ds, 
        batch_size=batch_size, 
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **kwargs,
    )


def _is_binidx_dataset(dataset_path: str) -> bool:
    """Check if dataset_path contains bin+idx files."""
    if not os.path.exists(dataset_path):
        return False
    bin_files = glob.glob(os.path.join(dataset_path, "*.bin"))
    idx_files = glob.glob(os.path.join(dataset_path, "*.idx"))
    return len(bin_files) > 0 and len(idx_files) > 0
