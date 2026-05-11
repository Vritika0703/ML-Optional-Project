#!/usr/bin/env python3
"""
Core Transformer model definition.
Decoder-only GPT-style transformer, compatible with both
Standard Parameterization (SP) and µP (via the mup package).

Adapted from nanoGPT (github.com/karpathy/nanoGPT) with modifications:
- Configurable for µP via use_mup flag
- Flash attention support
- Clean separation of SP vs µP attention scaling
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    context_length: int = 512   # max sequence length
    d_model: int = 384          # embedding dimension
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536            # feed-forward hidden dim
    dropout: float = 0.1
    bias: bool = False          # use bias in linear layers?
    use_mup: bool = False       # use µP parameterization


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    When use_mup=True: attention scale is 1/d_head (µP),
    otherwise: 1/sqrt(d_head) (standard).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.use_mup = config.use_mup
        self.dropout = config.dropout

        # QKV projection
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Check for flash attention
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.context_length, config.context_length)).view(
                    1, 1, config.context_length, config.context_length
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # B, nh, T, dh

        q, k, v = reshape(q), reshape(k), reshape(v)

        if self.flash:
            # µP: scale = 1/d (vs standard 1/sqrt(d))
            scale = (1.0 / self.d_head) if self.use_mup else None
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                scale=scale,
            )
        else:
            scale = (1.0 / self.d_head) if self.use_mup else (1.0 / math.sqrt(self.d_head))
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(self.act(self.fc(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.d_model, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.d_model, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full GPT-style model
# ─────────────────────────────────────────────────────────────────────────────

class SVGTransformer(nn.Module):
    """Decoder-only transformer for SVG language modeling."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.d_model),   # token embeddings
            wpe=nn.Embedding(config.context_length, config.d_model),  # position embeddings
            drop=nn.Dropout(config.dropout),
            blocks=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
            ln_f=LayerNorm(config.d_model, config.bias),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (embed and unembed)
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)
        # Scale residual projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.size()
        assert T <= self.config.context_length, f"Sequence length {T} > context {self.config.context_length}"

        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])  # only last token for generation
            return logits, None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_remove] = float("-inf")
                logits = torch.gather(sorted_logits, 1, sorted_idx.argsort(1))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return idx

    def num_parameters(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def configure_optimizers(self, weight_decay: float, lr: float, betas: tuple, device_type: str):
        """Separate weight decay and no-weight-decay parameter groups."""
        decay = set()
        no_decay = set()
        whitelist = (nn.Linear,)
        blacklist = (nn.LayerNorm, LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter = decay & no_decay
        union = decay | no_decay
        assert len(inter) == 0, f"Conflict: {inter}"
        assert len(param_dict.keys() - union) == 0

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
        kwargs = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **kwargs)
        return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# µP variant — wraps SVGTransformer with mup package modifications
# ─────────────────────────────────────────────────────────────────────────────

def make_mup_model(config: ModelConfig, base_d_model: int = 128) -> "SVGTransformer":
    """
    Create a model with µP parameterization using the `mup` package.
    The base_d_model is the width of the "base/proxy" model used for shape initialization.
    """
    try:
        from mup import make_base_shapes, set_base_shapes, MuReadout, MuSharedReadout
    except ImportError:
        raise ImportError("Install mup: pip install mup")

    config.use_mup = True

    # Build base (delta) model at the smallest width
    base_config = ModelConfig(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=base_d_model,
        n_layers=config.n_layers,
        n_heads=max(1, config.n_heads * base_d_model // config.d_model),
        d_ff=base_d_model * 4,
        dropout=config.dropout,
        bias=config.bias,
        use_mup=True,
    )

    base_model = SVGTransformer(base_config)
    delta_model = SVGTransformer(config)  # target model

    # Set µP shapes
    set_base_shapes(delta_model, base_model, delta=base_model)
    return delta_model


def make_mup_optimizer(model, lr: float, weight_decay: float, betas: tuple):
    """µP-aware optimizer — uses per-param learning rates scaled by width."""
    try:
        from mup import MuAdamW
        return MuAdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    except ImportError:
        raise ImportError("Install mup: pip install mup")
