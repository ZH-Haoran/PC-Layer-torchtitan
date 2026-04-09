"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
import time

from torchtitan.pc_layer.pc_layer import PCLinear


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class CausalSelfAttention(nn.Module):
    """
    Multi-head attention module with optional PC Layer support.
    """

    def __init__(self, config, layer_id: int):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.layer_id = layer_id
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.device = config.device
        self.flash = config.flash_attn
        self.dropout = config.dropout

        # Create raw nn.Linear layers first
        wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        wk = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        wv = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

        # Wrap with PCLinear based on config flags
        self.wq = PCLinear(wq, config, layer_id) if config.precondition_qk else wq
        self.wk = PCLinear(wk, config, layer_id) if config.precondition_qk else wk
        self.wv = PCLinear(wv, config, layer_id) if config.precondition_v else wv
        self.wo = PCLinear(wo, config, layer_id) if config.precondition_o else wo

        self.model_args = config

    def init_weights(self):
        """Initialize weights for this attention layer."""
        def get_linear(module):
            if isinstance(module, PCLinear):
                return module.linear
            return module

        # Align with reference: all attention linear layers use std=0.02
        for linear in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(get_linear(linear).weight, mean=0.0, std=0.02)

        # Initialize gamma parameters in PCLinear
        for linear in (self.wq, self.wk, self.wv, self.wo):
            if isinstance(linear, PCLinear):
                linear._maybe_init_gamma()

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        # PCLinear will automatically apply preconditioner in forward
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim)

        # for training
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            mask = None
            if seqlen > 1:
                mask = torch.full(
                    (1, 1, seqlen, seqlen), float("-inf"), device=self.device
                )
                mask = torch.triu(mask, diagonal=1).type_as(x)
            scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # PCLinear will automatically apply preconditioner in forward
        return self.wo(output)


class MLP(nn.Module):
    """
    FeedForward module with optional PC Layer support.
    """

    def __init__(self, config, layer_id: int):
        super().__init__()

        self.layer_id = layer_id
        hidden_dim = 4 * config.n_embd

        # Create raw nn.Linear layers first
        c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)

        # Wrap with PCLinear based on config flag
        if config.precondition_mlp:
            self.c_fc = PCLinear(c_fc, config, layer_id)
            self.c_proj = PCLinear(c_proj, config, layer_id)
        else:
            self.c_fc = c_fc
            self.c_proj = c_proj

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.model_args = config

    def init_weights(self, init_std: float):
        """Initialize weights for this MLP layer."""
        def get_linear(module):
            if isinstance(module, PCLinear):
                return module.linear
            return module

        # Align with reference: c_fc uses std=0.02, c_proj uses scaled std
        nn.init.normal_(get_linear(self.c_fc).weight, mean=0.0, std=0.02)
        nn.init.normal_(get_linear(self.c_proj).weight, mean=0.0, std=init_std)

        # Initialize gamma parameters in PCLinear
        for linear in (self.c_fc, self.c_proj):
            if isinstance(linear, PCLinear):
                linear._maybe_init_gamma()

    def forward(self, x):
        # PCLinear will automatically apply preconditioner in forward
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class ResidualAdd(nn.Module):
    """
    Residual connection module for hook-based monitoring.
    Separates the residual addition F(x) = x + f(x) into a distinct module.
    """
    def forward(self, x, delta):
        return x + delta


class Block(nn.Module):
    """
    Transformer Block with optional PC Layer support.
    """

    def __init__(self, config, layer_id: int):
        super().__init__()

        self.layer_id = layer_id
        self.num_layers = config.n_layer

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer_id)
        self.attn_residual = ResidualAdd()  # For hook-based monitoring
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config, layer_id)
        self.ffn_residual = ResidualAdd()  # For hook-based monitoring

        # Align with reference: scaled init for residual projections per GPT-2 paper
        self.weight_init_std = 0.02 / math.sqrt(2 * self.num_layers)

        self.model_args = config

    def init_weights(self):
        """Initialize weights for this block."""
        self.ln_1.reset_parameters()
        self.ln_2.reset_parameters()
        self.attn.init_weights()
        self.mlp.init_weights(self.weight_init_std)

    def forward(self, x):
        x = self.attn_residual(x, self.attn(self.ln_1(x)))
        x = self.ffn_residual(x, self.mlp(self.ln_2(x)))
        return x

@dataclass
class GPTConfig:
    # GPT-specific config
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    flash_attn: bool = False
    device: str = 'cuda'

    # Compatibility with torchtitan MODEL_CONFIG_KEYS (research features - defaults for GPT)
    norm_type: str = "layernorm"
    depth_init: bool = True  # If True, each block uses layer_id for init; if False, uses total layers
    precondition_mlp: bool = False
    precondition_qk: bool = False
    precondition_v: bool = False
    precondition_o: bool = False
    power_iter: int = 5
    # PC norm type: "F" (Frobenius), "modified_F", "op" (operator norm), None (no normalization)
    pc_norm_type: Optional[str] = "F"
    pc_norm_eps: float = 1e-7
    pc_op_beta: float = 0.0
    pc_level: int = 0
    recover_w_norm: bool = False
    # Ablation: whether to detach W_norm in the normalization division (weight / W_norm)
    # Ablation: whether to detach W_norm when recovering it after preconditioning (W_pc * W_norm)
    scale_constant: float = 1.0
    learnable_gamma: bool = False
    gamma_init_value: float = 1.0
    log_signal_propagation: bool = False
    log_gradients: bool = False

    # Aliases for compatibility with torchtitan (map dim -> n_embd, n_layers -> n_layer, n_heads -> n_head)
    @property
    def dim(self):
        return self.n_embd

    @dim.setter
    def dim(self, value):
        self.n_embd = value

    @property
    def n_layers(self):
        return self.n_layer

    @n_layers.setter
    def n_layers(self, value):
        self.n_layer = value

    @property
    def n_heads(self):
        return self.n_head

    @n_heads.setter
    def n_heads(self, value):
        self.n_head = value

    @property
    def max_seq_len(self):
        return self.block_size

    @max_seq_len.setter
    def max_seq_len(self, value):
        self.block_size = value

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.model_args = config  # Store for compatibility with torchtitan hooks
        print('config.vocab_size', config.vocab_size)
        print('config.n_embd', config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, layer_id) for layer_id in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: share weights between token embedding and output head
        self.transformer.wte.weight = self.lm_head.weight

        # Don't init weights here - will be done via init_weights() for meta device compatibility

    @classmethod
    def from_model_args(cls, config):
        """
        Factory method to create a GPT model from a GPTConfig.
        Compatible with torchtitan's model instantiation pattern.
        """
        return cls(config)

    def init_weights(self):
        """
        Initialize weights. Called after materializing the model from meta device.
        Compatible with torchtitan's initialization pattern.
        """
        # Initialize embeddings
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.transformer.wpe.weight, mean=0.0, std=0.02)

        # Initialize each block
        for block in self.transformer.h:
            block.init_weights()

        # Initialize final layer norm
        self.transformer.ln_f.reset_parameters()

        # Initialize output head - align with reference: std=0.02
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        # Report number of parameters
        n_params = self.get_num_params()
        for pn, p in self.named_parameters():
            print(f'name = {pn}, num para = {p.numel()}, ratio = {p.numel()/n_params}')

        print(f'number of parameters: {n_params}')
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    @property
    def layers(self):
        """
        Property to access transformer layers.
        Returns a ModuleDict with string keys for compatibility with torchtitan parallelisms.
        """
        return nn.ModuleDict({str(i): self.transformer.h[i] for i in range(len(self.transformer.h))})

    @property
    def tok_embeddings(self):
        """Alias for token embeddings (torchtitan compatibility)"""
        return self.transformer.wte

    @property
    def norm(self):
        """Alias for final norm (torchtitan compatibility)"""
        return self.transformer.ln_f

    @property
    def output(self):
        """Alias for output head (torchtitan compatibility)"""
        return self.lm_head
   



    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens):
        """
        Forward pass compatible with torchtitan training pipeline.
        Returns logits only (loss is computed externally in train.py).
        """
        device = tokens.device
        b, t = tokens.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(tokens) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Return full logits for training (loss computed externally)
        logits = self.lm_head(x)
        return logits

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, use_sgd = False):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        
        
        if use_sgd: 
            optimizer = torch.optim.SGD(optim_groups, lr = learning_rate, momentum = 0.9)
        
        else: 
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    
            print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
