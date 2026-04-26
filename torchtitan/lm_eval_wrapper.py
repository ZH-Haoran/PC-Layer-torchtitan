# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from typing import Any, Optional, Sequence, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F

from torchtitan.models import (
    model_name_to_cls,
    model_name_to_tokenizer,
    models_config,
)
from torchtitan.models.llama.model import ModelArgs

try:
    from lm_eval.api.model import LM
    from lm_eval import utils
except ImportError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    harness_dir = os.path.join(repo_root, "lm-evaluation-harness")
    if os.path.isdir(harness_dir) and harness_dir not in sys.path:
        sys.path.insert(0, harness_dir)
    try:
        from lm_eval.api.model import LM
        from lm_eval import utils
    except ImportError as e:  # pragma: no cover - runtime dependency
        raise ImportError(
            "lm-eval-harness is required. Install it or add it to PYTHONPATH."
        ) from e

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py<3.11 fallback
    import tomli as tomllib


def _build_tokenizer(tokenizer_kind: str, tokenizer_path: str):
    if tokenizer_kind == "sentencepiece":
        from torchtitan.datasets.tokenizer.sentencepiece import SentencePieceTokenizer

        return SentencePieceTokenizer(tokenizer_path)
    if tokenizer_kind == "tiktoken":
        from torchtitan.datasets.tokenizer.tiktoken import TikTokenizer

        return TikTokenizer(tokenizer_path)
    raise ValueError(f"Unsupported tokenizer kind: {tokenizer_kind}")


def _resolve_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    name = dtype.lower()
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _load_toml_config(config_path: Optional[str]) -> dict[str, Any]:
    if not config_path:
        return {}
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _find_latest_step_dir(checkpoint_root: str) -> Optional[str]:
    if not os.path.isdir(checkpoint_root):
        return None
    max_step = None
    max_step_dir = None
    for name in os.listdir(checkpoint_root):
        if not name.startswith("step-"):
            continue
        try:
            step = int(name.split("step-", 1)[1])
        except ValueError:
            continue
        if max_step is None or step > max_step:
            max_step = step
            max_step_dir = os.path.join(checkpoint_root, name)
    return max_step_dir


def _resolve_checkpoint_dir(
    checkpoint_path: Optional[str],
    *,
    config: dict[str, Any],
    step: Optional[int],
) -> str:
    # Explicit checkpoint path has priority.
    if checkpoint_path:
        if os.path.basename(checkpoint_path).startswith("step-"):
            return checkpoint_path
        if step is not None:
            return os.path.join(checkpoint_path, f"step-{step}")
        latest = _find_latest_step_dir(checkpoint_path)
        if latest is None:
            raise ValueError(
                f"No step-* checkpoint found under checkpoint_path={checkpoint_path}"
            )
        return latest

    # Build the same checkpoint hierarchy as train.py.
    model_section = config.get("model", {})
    optimizer_section = config.get("optimizer", {})
    metrics_section = config.get("metrics", {})
    checkpoint_section = config.get("checkpoint", {})

    base = checkpoint_section.get("folder")
    if not base:
        raise ValueError("checkpoint.folder is required in config when checkpoint_path is not provided")

    model_name = model_section.get("name")
    flavor = model_section.get("flavor")
    optimizer_name = optimizer_section.get("name")
    wandb_comment = metrics_section.get("wandb_comment", "")
    if not model_name or not flavor or not optimizer_name:
        raise ValueError(
            "model.name, model.flavor, optimizer.name must exist in config to resolve checkpoint path"
        )

    suffix = os.path.join(f"{model_name}_{flavor}", optimizer_name, wandb_comment)
    checkpoint_root = (
        base
        if os.path.normpath(base).endswith(os.path.normpath(suffix))
        else os.path.join(base, suffix)
    )

    if step is not None:
        return os.path.join(checkpoint_root, f"step-{step}")

    latest = _find_latest_step_dir(checkpoint_root)
    if latest is None:
        raise ValueError(f"No step-* checkpoint found under checkpoint root: {checkpoint_root}")
    return latest


def _build_model_args(
    model_name: str,
    flavor: str,
    vocab_size: int,
    norm_type: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    config_path: Optional[str] = None,
) -> ModelArgs:
    cfg = _load_toml_config(config_path)

    model_section = cfg.get("model", {})
    training_section = cfg.get("training", {})

    model_name = model_section.get("name", model_name)
    flavor = model_section.get("flavor", flavor)

    base_args = models_config[model_name][flavor]
    args_dict = asdict(base_args)
    args_dict["vocab_size"] = vocab_size

    if norm_type is None:
        norm_type = model_section.get("norm_type")
    if norm_type:
        args_dict["norm_type"] = norm_type

    if max_seq_len is None:
        max_seq_len = training_section.get("seq_len")
    if max_seq_len:
        args_dict["max_seq_len"] = max_seq_len

    for key in (
        "precondition_w1",
        "precondition_w2",
        "precondition_w3",
        "precondition_qk",
        "precondition_v",
        "precondition_o",
        "pc_level",
        "pc_norm_type",
        "pc_norm_eps",
        "pc_op_beta",
        "recover_w_norm",
        "scale_constant",
        "learnable_gamma",
        "gamma_init_value",
    ):
        if key in model_section:
            args_dict[key] = model_section[key]

    return ModelArgs(**args_dict)


def _load_state_dict_from_distcp(
    model: torch.nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> None:
    created_pg = False
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29600",
            world_size=1,
            rank=0,
        )
        created_pg = True

    try:
        # Path 1: checkpoint stores model in stateful format.
        wrapped = {"model": model}
        try:
            dcp.load(wrapped, checkpoint_id=checkpoint_path)
            return
        except Exception:
            pass

        # Path 2: checkpoint stores a plain state_dict (possibly nested under "model").
        loaded: dict[str, Any] = {}
        dcp.load(loaded, checkpoint_id=checkpoint_path)
        if "model" in loaded and isinstance(loaded["model"], dict):
            loaded = loaded["model"]
        model.load_state_dict(loaded, strict=strict)
    finally:
        if created_pg:
            dist.destroy_process_group()


def build_titan_model_and_tokenizer(
    checkpoint_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    *,
    model_name: str = "llama2",
    flavor: str = "1B",
    norm_type: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    config_path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: Union[str, torch.dtype] = "auto",
    strict: bool = False,
    step: Optional[int] = None,
) -> tuple[torch.nn.Module, Any]:
    cfg = _load_toml_config(config_path)
    model_section = cfg.get("model", {})
    model_name = model_section.get("name", model_name)
    flavor = model_section.get("flavor", flavor)
    tokenizer_path = model_section.get("tokenizer_path", tokenizer_path)
    if not tokenizer_path:
        raise ValueError("tokenizer_path is required (or must exist in model.tokenizer_path of config)")

    checkpoint_dir = _resolve_checkpoint_dir(
        checkpoint_path,
        config=cfg,
        step=step,
    )
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    tokenizer_kind = model_name_to_tokenizer[model_name]
    tokenizer = _build_tokenizer(tokenizer_kind, tokenizer_path)

    model_args = _build_model_args(
        model_name=model_name,
        flavor=flavor,
        vocab_size=tokenizer.n_words,
        norm_type=norm_type,
        max_seq_len=max_seq_len,
        config_path=config_path,
    )

    model_cls = model_name_to_cls[model_name]
    model = model_cls.from_model_args(model_args)
    model = model.to(device=torch.device(device), dtype=_resolve_dtype(dtype))

    # FIX: Recompute freqs_cis after dtype conversion to avoid complex->real corruption
    # When model.to(dtype=bfloat16) is called, complex64 freqs_cis gets corrupted to bfloat16
    # because bfloat16 doesn't support complex numbers. We need to recompute it.
    model.freqs_cis = model._precompute_freqs_cis().to(device=model.freqs_cis.device)

    _load_state_dict_from_distcp(model, checkpoint_dir, strict=strict)
    model.eval()
    return model, tokenizer


class TitanWrapper(LM):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        device: Union[str, torch.device] = "cuda",
        max_seq_len: Optional[int] = None,
        max_gen_toks: int = 256,
        add_bos: bool = True,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self._device = torch.device(device)
        self.model.to(self._device)

        self._max_seq_len = max_seq_len or getattr(
            getattr(model, "model_args", object()), "max_seq_len", 2048
        )
        self._max_gen_toks = max_gen_toks
        self._add_bos = add_bos

        self._bos_id = getattr(tokenizer, "bos_id", None)
        self._eos_id = getattr(tokenizer, "eos_id", None)

    def _tok_encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        return list(self.tokenizer.encode(text, bos=bos, eos=eos))

    def _tok_decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(list(tokens))

    def _get_stop_sequences(self, gen_kwargs: dict[str, Any]) -> list[Union[str, list[int]]]:
        """Get stop sequences from gen_kwargs, ensuring EOS token is included.

        This aligns with HFLM's handle_stop_sequences behavior.
        """
        stop = gen_kwargs.get("until", [])
        if isinstance(stop, str):
            stop = [stop]
        else:
            stop = list(stop)

        # 自动添加 EOS token 到停止序列（与 HFLM handle_stop_sequences 一致）
        if self._eos_id is not None:
            eos_str = self._tok_decode([self._eos_id])
            if eos_str and eos_str not in stop:
                stop.append(eos_str)

        return stop

    def _score_sequence_from(
        self, tokens: list[int], score_from_pos: int
    ) -> tuple[float, bool]:
        if len(tokens) <= 1 or score_from_pos >= len(tokens):
            return 0.0, True

        input_tokens = torch.tensor(tokens[:-1], dtype=torch.long, device=self._device).unsqueeze(0)
        target_tokens = torch.tensor(tokens[1:], dtype=torch.long, device=self._device)

        with torch.no_grad():
            logits = self.model(input_tokens)
            log_probs = F.log_softmax(logits[0], dim=-1)

        # 从 score_from_pos 开始计算所有 continuation tokens 的 log-likelihood 求和
        # 这与 HuggingFace HFLM 的实现保持一致
        start = max(score_from_pos - 1, 0)
        target_slice = target_tokens[start:]
        logits_slice = log_probs[start:]

        # 计算所有 continuation tokens 的 log-prob 求和
        total_logprob = 0.0
        for i, target in enumerate(target_slice):
            total_logprob += float(logits_slice[i, target].item())

        # greedy 检查：所有 continuation tokens 都必须是 argmax
        greedy_tokens = logits_slice.argmax(dim=-1)
        all_greedy = bool((greedy_tokens == target_slice).all().item())

        return float(total_logprob), all_greedy

    def _score_continuation(
        self, context_ids: list[int], continuation_ids: list[int]
    ) -> tuple[float, bool]:
        """Score continuation tokens given context tokens.

        NOTE: BOS token should be included in context_ids by the caller if needed.
        This method does NOT add BOS token - that's the caller's responsibility.
        """
        if not continuation_ids:
            return 0.0, True

        # BOS handling is done by the caller (loglikelihood/loglikelihood_rolling)
        # We do NOT add BOS here to avoid double-adding
        tokens = context_ids + continuation_ids
        score_from = len(context_ids)

        # 使用 max_length + 1 进行截断，与 HFLM 保持一致
        # 最后一个 token 是 target，输入时不需要包含它
        if len(tokens) <= self._max_seq_len + 1:
            return self._score_sequence_from(tokens, score_from)

        # Long-sequence fallback: score token-by-token with rolling context.
        total_logprob = 0.0
        all_greedy = True
        with torch.no_grad():
            for target_pos in range(max(score_from, 1), len(tokens)):
                # 滚动窗口，保留最多 max_seq_len - 1 个输入 token
                context_start = max(0, target_pos - (self._max_seq_len - 1))
                in_tokens = tokens[context_start:target_pos]
                x = torch.tensor(in_tokens, dtype=torch.long, device=self._device).unsqueeze(0)
                logits = self.model(x)[0, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                tgt = tokens[target_pos]
                total_logprob += float(log_probs[tgt].item())
                all_greedy = all_greedy and int(log_probs.argmax().item()) == int(tgt)
        return total_logprob, all_greedy

    def _encode_pair(
        self, context: str, continuation: str, add_bos: bool = False
    ) -> tuple[list[int], list[int]]:
        """Encode a context-continuation pair into separate token ID lists.

        Args:
            context: The conditioning text (must be non-empty).
            continuation: The text to score.
            add_bos: Whether to prepend BOS token to the context.

        Returns:
            A (context_enc, continuation_enc) tuple of token ID lists.
        """
        # Keep trailing spaces with continuation for stable token-boundary behavior.
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole = self._tok_encode(context + continuation, bos=False, eos=False)
        context_ids = self._tok_encode(context, bos=False, eos=False)
        continuation_ids = whole[len(context_ids) :]

        # 如果需要添加 BOS token（与 HFLM add_bos_token 配置一致）
        if add_bos and self._bos_id is not None:
            context_ids = [self._bos_id] + context_ids

        return context_ids, continuation_ids

    def _normalize_gen_kwargs(self, gen_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Normalize generation kwargs for consistent handling with HuggingFace models."""
        kwargs = gen_kwargs.copy()

        # 处理 stop sequences
        until = kwargs.get("until", [])
        if not isinstance(until, list):
            until = [until]
        kwargs["until"] = until

        # 处理 max_gen_toks 及其别名（与 HuggingFace normalize_gen_kwargs 保持一致）
        max_gen_toks = kwargs.pop("max_gen_toks", kwargs.pop("max_new_tokens",
                     kwargs.pop("max_tokens",
                     kwargs.pop("max_completion_tokens", self._max_gen_toks))))
        kwargs["max_gen_toks"] = int(max_gen_toks)

        # 处理 temperature
        temperature = float(kwargs.get("temperature", 0.0))
        kwargs["temperature"] = temperature

        # 处理 do_sample
        do_sample = kwargs.get("do_sample", None)
        if temperature == 0.0 and do_sample is None:
            kwargs["do_sample"] = False

        # 如果 do_sample 为 False 且 temperature 为 0.0，移除 temperature（与 HuggingFace 一致）
        if do_sample is False and temperature == 0.0:
            kwargs.pop("temperature", None)

        # 处理 top_p
        top_p = float(kwargs.get("top_p", 1.0))
        kwargs["top_p"] = top_p

        # 处理 top_k
        top_k = int(kwargs.get("top_k", 0))
        kwargs["top_k"] = top_k

        return kwargs

    def _prefix_token(self) -> int:
        if self._bos_id is not None and self._bos_id >= 0:
            return int(self._bos_id)
        if self._eos_id is not None and self._eos_id >= 0:
            return int(self._eos_id)
        raise ValueError("Tokenizer must expose a valid bos_id or eos_id")

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        """Compute log-likelihood of continuations given contexts.

        BOS token handling (aligned with HFLM):
        - Empty context: Use prefix_token (BOS) as the context token
        - Non-empty context: Add BOS token if self._add_bos is True (matches HFLM add_bos_token)

        This matches HuggingFace HFLM behavior where BOS can be controlled via add_bos_token config.
        """
        results: list[tuple[float, bool]] = []

        for req in requests:
            context, continuation = req.args
            if context == "":
                # Empty context: use prefix token (BOS) as context
                # This matches HFLM's TemplateLM.loglikelihood behavior
                continuation_ids = self._tok_encode(continuation, bos=False, eos=False)
                if not continuation_ids:
                    results.append((0.0, True))
                    continue
                prefix = self._prefix_token()
                # Handle case where continuation already starts with prefix token
                if continuation_ids[0] == prefix:
                    context_ids = continuation_ids[:1]  # Use first token as context
                    continuation_ids = continuation_ids[1:]
                else:
                    context_ids = [prefix]  # Add BOS as context
            else:
                # Non-empty context: 根据 self._add_bos 决定是否添加 BOS
                # 这与 HFLM 的 add_bos_token 配置行为一致
                context_ids, continuation_ids = self._encode_pair(
                    context, continuation, add_bos=self._add_bos
                )

            results.append(self._score_continuation(context_ids, continuation_ids))
        return results

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        outputs: list[float] = []
        prefix = self._prefix_token()

        for req in requests:
            text = req.args[0]
            tokens = self._tok_encode(text, bos=False, eos=False)
            if self._add_bos:
                tokens = [prefix] + tokens

            # 使用滑动窗口计算 rolling loglikelihood
            # context_len=1 表示每个窗口的 context 只有 1 个 token
            rolling_token_windows: list[tuple[list[int], list[int]]] = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=tokens,
                        prefix_token=prefix,
                        max_seq_len=self._max_seq_len,
                        context_len=1,
                    ),
                )
            )

            # 计算所有窗口的 log-likelihood 总和
            total_score = 0.0
            for context_ids, continuation_ids in rolling_token_windows:
                if continuation_ids:  # 只计算非空的 continuation
                    score, _ = self._score_continuation(context_ids, continuation_ids)
                    total_score += score

            outputs.append(total_score)
        return outputs

    def _should_stop(
        self, generated_tokens: list[int], stop_sequences: list[Union[str, list[int]]]
    ) -> bool:
        if not stop_sequences:
            return False
        generated_text = self._tok_decode(generated_tokens)
        for stop in stop_sequences:
            if isinstance(stop, str) and stop and stop in generated_text:
                return True
            if isinstance(stop, list) and stop and generated_tokens[-len(stop) :] == stop:
                return True
        return False

    def _trim_by_stop(self, text: str, stop_sequences: list[Union[str, list[int]]]) -> str:
        stop_strings = [s for s in stop_sequences if isinstance(s, str) and s]
        if not stop_strings:
            return text
        cut = len(text)
        for stop in stop_strings:
            idx = text.find(stop)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut]

    def generate_until(self, requests: list[Any]) -> list[str]:
        outputs: list[str] = []
        prefix = self._prefix_token()

        for req in requests:
            context = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}

            # 归一化生成参数，与 HuggingFace 保持一致
            gen_kwargs = self._normalize_gen_kwargs(gen_kwargs)
            max_new_tokens = gen_kwargs["max_gen_toks"]
            temperature = gen_kwargs.get("temperature")  # 可能为 None（贪心模式）
            top_p = gen_kwargs["top_p"]
            top_k = gen_kwargs["top_k"]
            do_sample = gen_kwargs["do_sample"]
            stop_sequences = self._get_stop_sequences(gen_kwargs)

            current = self._tok_encode(context, bos=False, eos=False)
            if self._add_bos and (not current or current[0] != prefix):
                current = [prefix] + current

            generated: list[int] = []

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    in_tokens = current[-(self._max_seq_len - 1) :] if len(current) >= self._max_seq_len else current
                    x = torch.tensor(in_tokens, dtype=torch.long, device=self._device).unsqueeze(0)
                    logits = self.model(x)[0, -1, :]

                    # 采样逻辑：贪心、top-k、top-p、温度采样
                    if temperature is None or (do_sample is False):
                        # 贪心解码：直接选择概率最高的词
                        next_token = int(torch.argmax(logits).item())
                    else:
                        # 采样模式
                        probs = F.softmax(logits / temperature, dim=-1)

                        # 应用 top-k 过滤
                        if top_k > 0 and top_k < probs.shape[-1]:
                            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
                            # 创建一个新的概率分布，只保留 top-k 的值
                            probs = torch.zeros_like(probs)
                            probs.scatter_(-1, top_k_indices.unsqueeze(0), top_k_probs)

                        # 应用 top-p (nucleus) 过滤
                        if top_p < 1.0:
                            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                            # 找到累计概率达到 top_p 的位置
                            sorted_indices_to_remove = cumulative_probs > top_p
                            # 确保至少保留一个 token
                            sorted_indices_to_remove[..., 0] = False
                            # 创建新的概率分布，只保留 nucleus 内的 tokens
                            probs = torch.zeros_like(probs)
                            probs.scatter_(-1, sorted_indices[..., ~sorted_indices_to_remove], sorted_probs[..., ~sorted_indices_to_remove])

                        # 重新归一化
                        probs = probs / probs.sum(dim=-1, keepdim=True)

                        # 从采样分布中采样
                        next_token = int(torch.multinomial(probs, num_samples=1).item())

                    current.append(next_token)
                    generated.append(next_token)

                    if self._eos_id is not None and next_token == int(self._eos_id):
                        break
                    if self._should_stop(generated, stop_sequences):
                        break

            text = self._tok_decode(generated)
            outputs.append(self._trim_by_stop(text, stop_sequences))
        return outputs
