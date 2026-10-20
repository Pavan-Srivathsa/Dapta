"""A from-scratch, tiny Llama-architecture model: RMSNorm, RoPE, GQA
attention, SwiGLU FFN. Built standalone (not by patching HF's
LlamaDecoderLayer) so the zFLoRA reconstruction in zflora.py can hook
exactly the points PLANNING.md Day 1 needs, without chasing transformers'
internal layer API across versions.

Mirrors Qwen3-VL's language-backbone structure closely enough for an
architecture prototype: pre-norm, RoPE, grouped-query attention, SwiGLU.
Random-init only -- this is for proving the zFLoRA mechanism trains, not
for matching any real model's weights or quality.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class Attention(nn.Module):
    """Standard GQA attention. Wo's forward signature exposes attn_output
    (pre-Wo) separately from the final projection, because the zFLoRA
    reconstruction needs to attach its backward adapter to that same
    attn_output tensor, in parallel with Wo -- see zflora.py.
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)

        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        attn = attn + causal_mask.to(attn.device)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)

        attn_output = out  # pre-Wo, this is what the backward adapter also consumes
        main = self.o_proj(attn_output)
        return main, attn_output


class SwiGLUFFN(nn.Module):
    """gate_proj/up_proj/down_proj, matching train.py's LORA_TARGET_MODULES
    naming. forward() exposes the pre-down-proj gated hidden state, because
    the zFLoRA reconstruction's forward adapter reads FFN input and feeds
    the next layer, attached around this block -- see zflora.py.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated)


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        self.input_norm = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, num_kv_heads)
        self.post_attn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        attn_main, _ = self.attn(self.input_norm(x), cos, sin)
        x = x + attn_main
        x = x + self.ffn(self.post_attn_norm(x))
        return x


class TinyLlama(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 32,
        intermediate_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_size, intermediate_size, num_heads, num_kv_heads) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        head_dim = hidden_size // num_heads
        cos, sin = build_rope_cache(max_seq_len, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        x = self.embed(input_ids)
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.final_norm(x)
        return self.lm_head(x)

    def freeze_base(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)
