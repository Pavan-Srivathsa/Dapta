"""Our reconstruction of the zFLoRA architecture (Gowda, Song, Goka, Lee,
Samsung Research, EMNLP 2025, arXiv:2510.25784), for PLANNING.md Day 1/2.

No public reference code exists and the paper's own Section 4 is prose +
an unlabeled block diagram (Figure 4), not equations -- this is a
best-effort reconstruction, not a reproduction. Where we made a judgment
call because the paper underspecifies it, it's called out below.

Reconstructed wiring, grounded in the paper's text:
  - A "backward adapter" (B, the up-projection r->d) attaches to the
    attention sub-block, in parallel with Wo. It consumes an r-dim "aux"
    channel carried in from the previous layer and adds a d-dim
    contribution directly into the residual -- no separate merge step,
    matching the paper's "yield an automatically merged output" claim.
  - A "forward adapter" (A, the down-projection d->r) attaches within the
    FFN sub-block. It consumes the FFN's own d-dim input (same input
    gate_proj/up_proj see) and produces the r-dim aux update carried to
    *the next layer's* backward adapter -- matching "pairing ... spanning
    across MHA and FFN blocks."
  - The aux channel is expanded once before layer 0 (initial_expand) and
    merged back to d-dim once after the last layer, before the LM head
    (final_merge) -- directly from the paper's own wording.

Judgment call (paper doesn't specify): we treat the aux channel as
additive/residual across layers (forward adapters accumulate into it)
rather than being overwritten each layer, since that's the natural reading
of "carried forward through layer-norm after the addition of residual
component." If Day 2's quality check shows a large, consistent gap vs
standard LoRA, this is the first assumption to revisit.

Both injection points (backward_adapters, final_merge) are zero-init, so
at initialization this model computes exactly what the frozen base would
-- same stability property as standard LoRA's B zero-init.
"""

from __future__ import annotations

import torch
from torch import nn

from tiny_llama import TinyLlama


class BackwardAdapter(nn.Module):
    """B: aux_dim -> hidden_size. Attaches to the attention sub-block."""

    def __init__(self, aux_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(aux_dim, hidden_size, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, aux_in: torch.Tensor) -> torch.Tensor:
        return self.proj(aux_in)


class ForwardAdapter(nn.Module):
    """A: hidden_size -> aux_dim. Attaches within the FFN sub-block."""

    def __init__(self, hidden_size: int, aux_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, aux_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, ffn_input: torch.Tensor) -> torch.Tensor:
        return self.proj(ffn_input)


class ZFLoRAModel(nn.Module):
    def __init__(self, base: TinyLlama, aux_dim: int) -> None:
        super().__init__()
        base.freeze_base()
        self.base = base
        self.aux_dim = aux_dim
        hidden_size = base.hidden_size
        num_layers = len(base.layers)

        self.initial_expand = nn.Linear(hidden_size, aux_dim, bias=False)
        nn.init.normal_(self.initial_expand.weight, std=0.02)

        self.backward_adapters = nn.ModuleList(
            [BackwardAdapter(aux_dim, hidden_size) for _ in range(num_layers)]
        )
        self.forward_adapters = nn.ModuleList(
            [ForwardAdapter(hidden_size, aux_dim) for _ in range(num_layers)]
        )

        self.final_merge = nn.Linear(aux_dim, hidden_size, bias=False)
        nn.init.zeros_(self.final_merge.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        x = self.base.embed(input_ids)
        cos = self.base.rope_cos[:seq_len]
        sin = self.base.rope_sin[:seq_len]

        aux = self.initial_expand(x)

        for i, layer in enumerate(self.base.layers):
            normed = layer.input_norm(x)
            attn_main, _ = layer.attn(normed, cos, sin)
            x = x + attn_main + self.backward_adapters[i](aux)

            ffn_input = layer.post_attn_norm(x)
            x = x + layer.ffn(ffn_input)
            aux = aux + self.forward_adapters[i](ffn_input)

        x = self.base.final_norm(x)
        x = x + self.final_merge(aux)
        return self.base.lm_head(x)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
