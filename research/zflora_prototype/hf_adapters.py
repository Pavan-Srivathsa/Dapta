"""Adapter implementations for a real Hugging Face causal LM, using
forward hooks instead of reimplementing the model (unlike tiny_llama.py's
Day 1 approach). This is deliberately the mechanism Day 3 will reuse
against Qwen3-VL's language backbone -- same hook points
(`self_attn.o_proj`, `mlp`, `lm_head`), just a different checkpoint.
Works on any Llama-family model using the standard
q_proj/k_proj/v_proj/o_proj + gate_proj/up_proj/down_proj naming
(Llama, Qwen2, Qwen3 all match this).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class HFLoRALinear(nn.Module):
    """Same delta as standard_lora.py's LoRALinear, generalized to accept
    a base layer that may have a bias (Qwen2's q/k/v_proj do).
    """

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: int) -> None:
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        delta = (x @ self.lora_a.T) @ self.lora_b.T
        return base + delta * self.scaling


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def get_language_model(model: nn.Module) -> nn.Module:
    """Qwen2/Llama: model.model has .layers directly. Qwen3-VL (and other
    VL models wrapping a separate vision tower): model.model.language_model
    has .layers, and model.model.visual is the frozen vision tower we are
    deliberately not touching -- confirmed against
    transformers/models/qwen3_vl/modeling_qwen3_vl.py on the training pod
    (Qwen3VLModel.language_model = Qwen3VLTextModel._from_config(...)).
    """
    inner = model.model
    if hasattr(inner, "language_model"):
        return inner.language_model
    return inner


def get_hidden_size(model: nn.Module) -> int:
    config = model.config
    text_config = getattr(config, "text_config", None)
    return text_config.hidden_size if text_config is not None else config.hidden_size


def apply_standard_lora_hf(model: nn.Module, rank: int, alpha: int) -> nn.Module:
    """Attaches LoRA at q/k/v/o_proj, matching backend/dapta/train.py's
    default LORA_TARGET_MODULES for non-nda-intake skills."""
    freeze_all(model)
    for layer in get_language_model(model).layers:
        attn = layer.self_attn
        attn.q_proj = HFLoRALinear(attn.q_proj, rank, alpha)
        attn.k_proj = HFLoRALinear(attn.k_proj, rank, alpha)
        attn.v_proj = HFLoRALinear(attn.v_proj, rank, alpha)
        attn.o_proj = HFLoRALinear(attn.o_proj, rank, alpha)
    return model


class _AuxCarryingLayer(nn.Module):
    """Wraps one decoder layer so the aux channel enters and leaves it as
    explicit tensors instead of only through ZFLoRAHookWrapper._aux.

    Why this exists (Day 3 finding): HF's own gradient checkpointing
    re-runs each layer's forward during backward. The o_proj/mlp hooks
    read and mutate the shared self._aux, so on that re-run they saw
    whatever aux the *last* layer left behind, and every adapter except
    final_merge got silently wrong gradients (measured 87%-550% off on
    Qwen2.5-0.5B, no error raised). Here, `run` resets owner._aux from
    its own argument before calling the layer, so a recompute under
    non-reentrant checkpointing replays exactly the original forward,
    and autograd sees aux as a real input/output of the layer.
    """

    def __init__(self, layer: nn.Module, owner: "ZFLoRAHookWrapper", index: int) -> None:
        super().__init__()
        self.layer = layer
        self.index = index
        # Plain attribute, not a submodule: owner already contains us.
        object.__setattr__(self, "_owner", owner)
        if hasattr(layer, "attention_type"):
            self.attention_type = layer.attention_type

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        owner = self._owner
        aux = owner.initial_expand(hidden_states) if self.index == 0 else owner._aux

        def run(h, a, *run_args, **run_kwargs):
            owner._aux = a
            out = self.layer(h, *run_args, **run_kwargs)
            return out, owner._aux

        if owner.checkpoint_layers and self.training and torch.is_grad_enabled():
            out, aux = checkpoint(run, hidden_states, aux, *args, use_reentrant=False, **kwargs)
        else:
            out, aux = run(hidden_states, aux, *args, **kwargs)
        owner._aux = aux
        return out


class ZFLoRAHookWrapper(nn.Module):
    """Reconstructed zFLoRA (see zflora.py's docstring for the full
    rationale) implemented via forward hooks on an unmodified HF model,
    instead of reimplementing the decoder layer. Hook points:

      - layer 0's _AuxCarryingLayer: captures the embedded input, seeds
        the aux channel via initial_expand.
      - forward hook on each layer's self_attn.o_proj: adds
        backward_adapters[i](aux) into the attention output, in parallel
        with Wo (same as zflora.py's `x + attn_main + backward(...)`).
      - pre-hook on each layer's mlp: observes the FFN's input (the same
        tensor gate_proj/up_proj see) and updates aux via
        forward_adapters[i], without altering the FFN's own computation.
      - pre-hook on lm_head: injects final_merge(aux) into the final
        hidden state before logits are computed.

    self._aux is reset at the start of every forward() call by layer 0,
    so this is safe for sequential (non-overlapping) forward calls --
    exactly what a normal train/eval loop does.

    Do NOT enable HF's model.gradient_checkpointing_enable() on the base;
    pass checkpoint_layers=True instead (see _AuxCarryingLayer for why).
    """

    def __init__(self, model: nn.Module, aux_dim: int, checkpoint_layers: bool = False) -> None:
        super().__init__()
        if getattr(model, "is_gradient_checkpointing", False):
            raise ValueError(
                "HF gradient checkpointing corrupts zFLoRA adapter gradients; "
                "leave it off and pass checkpoint_layers=True instead."
            )
        freeze_all(model)
        self.model = model
        self.aux_dim = aux_dim
        self.checkpoint_layers = checkpoint_layers
        hidden_size = get_hidden_size(model)
        num_layers = len(get_language_model(model).layers)

        self.initial_expand = nn.Linear(hidden_size, aux_dim, bias=False)
        nn.init.normal_(self.initial_expand.weight, std=0.02)
        self.backward_adapters = nn.ModuleList(
            [nn.Linear(aux_dim, hidden_size, bias=False) for _ in range(num_layers)]
        )
        for adapter in self.backward_adapters:
            nn.init.zeros_(adapter.weight)
        self.forward_adapters = nn.ModuleList(
            [nn.Linear(hidden_size, aux_dim, bias=False) for _ in range(num_layers)]
        )
        for adapter in self.forward_adapters:
            nn.init.normal_(adapter.weight, std=0.02)
        self.final_merge = nn.Linear(aux_dim, hidden_size, bias=False)
        nn.init.zeros_(self.final_merge.weight)

        self._aux: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        layers = get_language_model(self.model).layers

        for i, layer in enumerate(layers):
            def backward_hook(_module, _input, output, i=i):
                return output + self.backward_adapters[i](self._aux)

            layer.self_attn.o_proj.register_forward_hook(backward_hook)

            def forward_hook(_module, args, i=i):
                ffn_input = args[0]
                self._aux = self._aux + self.forward_adapters[i](ffn_input)

            layer.mlp.register_forward_pre_hook(forward_hook)

        for i in range(len(layers)):
            layers[i] = _AuxCarryingLayer(layers[i], self, i)

        def merge_hook(_module, args):
            hidden_states = args[0]
            return (hidden_states + self.final_merge(self._aux),)

        self.model.lm_head.register_forward_pre_hook(merge_hook)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def generate(self, *args, **kwargs):
        """Passthrough so eval code can call .generate() uniformly whether
        it holds a raw HF model (standard LoRA) or this wrapper -- the
        hooks are attached to self.model's own submodules either way."""
        return self.model.generate(*args, **kwargs)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
