"""Standard LoRA baseline, attached the same way train.py / factory_gpu.py
attach it in production: q_proj/k_proj/v_proj/o_proj. Used only as the
comparison point for the zFLoRA reconstruction in zflora.py -- same base
model, same rank, same training budget.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from tiny_llama import TinyLlama


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear base layer with a standard LoRA delta:
    Z = WX + BAX, scaled by alpha/r. B zero-init so the adapted model
    starts identical to the frozen base, same as PEFT's default.
    """

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: int) -> None:
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad_(False)
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        delta = (x @ self.lora_a.T) @ self.lora_b.T
        return base + delta * self.scaling


def apply_standard_lora(base: TinyLlama, rank: int, alpha: int) -> TinyLlama:
    base.freeze_base()
    for layer in base.layers:
        layer.attn.q_proj = LoRALinear(layer.attn.q_proj, rank, alpha)
        layer.attn.k_proj = LoRALinear(layer.attn.k_proj, rank, alpha)
        layer.attn.v_proj = LoRALinear(layer.attn.v_proj, rank, alpha)
        layer.attn.o_proj = LoRALinear(layer.attn.o_proj, rank, alpha)
    return base


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
