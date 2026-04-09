# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.gpt.model import GPTConfig, GPT

__all__ = ["GPT", "GPTConfig"]

# GPT-2 style model configurations
# Using GPTConfig with torchtitan-compatible defaults
gpt_configs = {
    "debugmodel": GPTConfig(
        block_size=512,
        vocab_size=32000,  # Will be overridden by tokenizer
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.0,
        bias=False,  # Match Llama style (no bias)
        flash_attn=True,
    ),
    "small": GPTConfig(
        block_size=1024,
        vocab_size=32000,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=False,
        flash_attn=True,
    ),
    "124M": GPTConfig(
        block_size=1024,
        vocab_size=50257,  # Original GPT-2 vocab
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,  # Original GPT-2 uses bias
        flash_attn=True,
    ),
    "350M": GPTConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=24,
        n_head=16,
        n_embd=1024,
        dropout=0.0,
        bias=True,
        flash_attn=True,
    ),
    "medium": GPTConfig(
        block_size=2048,
        vocab_size=32000,
        n_layer=16,
        n_head=16,
        n_embd=1024,
        dropout=0.0,
        bias=False,
        flash_attn=True,
    ),
    "large": GPTConfig(
        block_size=2048,
        vocab_size=32000,
        n_layer=24,
        n_head=24,
        n_embd=1536,
        dropout=0.0,
        bias=False,
        flash_attn=True,
    ),
    "770M": GPTConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=36,
        n_head=20,
        n_embd=1280,
        dropout=0.0,
        bias=True,
        flash_attn=True,
    ),
    "1.5B": GPTConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=48,
        n_head=25,
        n_embd=1600,
        dropout=0.0,
        bias=True,
        flash_attn=True,
    ),
}
