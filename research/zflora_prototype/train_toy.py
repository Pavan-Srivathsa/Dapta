"""PLANNING.md Day 1 pass gate: prove the zFLoRA reconstruction (zflora.py)
trains on a frozen tiny Llama-style base, on CPU, and compare it against a
standard-LoRA baseline on the same base, same rank budget, same steps.

This does NOT validate the paper's latency claim -- that's a GPU
kernel-launch-count effect with no CPU analogue. See PLANNING.md Day 1
"Explicit non-goal". What this validates: the reconstructed architecture
is mechanically sound (correct shapes, gradients reach only adapter
params, loss goes down) before we spend GPU time porting it to Qwen3-VL.

Usage: .venv/bin/python research/zflora_prototype/train_toy.py
"""

from __future__ import annotations

import argparse
import copy

import torch
from torch import nn

from standard_lora import apply_standard_lora, count_trainable_params as count_lora_params
from tiny_llama import TinyLlama
from zflora import ZFLoRAModel, count_trainable_params as count_zflora_params


def make_dataset(num_sequences: int, seq_len: int, vocab_size: int, seed: int, task: str = "pair-sum") -> torch.Tensor:
    """Two task variants, to separate "can it learn a pointwise remap" from
    "can it learn new cross-position attention routing on a random base":

      pair-sum:  y[t] = (y[t-1] + y[t-2]) % vocab_size, t >= 2 -- requires
                 combining two previous positions, i.e. needs attention to
                 route information across positions.
      increment: y[t] = (y[t-1] + 1) % vocab_size -- purely pointwise, each
                 position only needs its own immediately-prior token.
    """
    generator = torch.Generator().manual_seed(seed)
    data = torch.zeros(num_sequences, seq_len, dtype=torch.long)
    data[:, 0] = torch.randint(0, vocab_size, (num_sequences,), generator=generator)
    data[:, 1] = torch.randint(0, vocab_size, (num_sequences,), generator=generator)
    if task == "pair-sum":
        for t in range(2, seq_len):
            data[:, t] = (data[:, t - 1] + data[:, t - 2]) % vocab_size
    elif task == "increment":
        for t in range(2, seq_len):
            data[:, t] = (data[:, t - 1] + 1) % vocab_size
    else:
        raise ValueError(f"unknown task: {task}")
    return data


def evaluate(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        accuracy = (logits.argmax(-1) == labels).float().mean().item()
    model.train()
    return loss.item(), accuracy


def train_model(name: str, model: nn.Module, train_ids, train_labels, val_ids, val_labels, steps: int, lr: float) -> None:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    start_loss, start_acc = evaluate(model, val_ids, val_labels)
    for step in range(steps):
        optimizer.zero_grad()
        logits = model(train_ids)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), train_labels.reshape(-1))
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 5) == 0:
            val_loss, val_acc = evaluate(model, val_ids, val_labels)
            print(f"  [{name}] step {step + 1:>4}/{steps}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    final_loss, final_acc = evaluate(model, val_ids, val_labels)
    print(f"  [{name}] DONE  val_loss {start_loss:.4f} -> {final_loss:.4f}   val_acc {start_acc:.3f} -> {final_acc:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=24)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--intermediate-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--train-sequences", type=int, default=256)
    parser.add_argument("--val-sequences", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", choices=["pair-sum", "increment"], default="pair-sum")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_data = make_dataset(args.train_sequences, args.seq_len, args.vocab_size, seed=args.seed, task=args.task)
    val_data = make_dataset(args.val_sequences, args.seq_len, args.vocab_size, seed=args.seed + 1, task=args.task)
    train_ids, train_labels = train_data[:, :-1], train_data[:, 1:]
    val_ids, val_labels = val_data[:, :-1], val_data[:, 1:]

    def fresh_base() -> TinyLlama:
        torch.manual_seed(args.seed)  # identical base init across conditions
        return TinyLlama(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            max_seq_len=args.seq_len,
        )

    print("=" * 70)
    print("Condition 0: frozen base, untrained (reference floor)")
    base_only = fresh_base()
    base_only.freeze_base()
    loss0, acc0 = evaluate(base_only, val_ids, val_labels)
    print(f"  [frozen-base] val_loss={loss0:.4f}  val_acc={acc0:.3f}")

    print("=" * 70)
    print(f"Condition 1: standard LoRA (rank={args.rank}, targets q/k/v/o)")
    lora_model = apply_standard_lora(fresh_base(), rank=args.rank, alpha=args.rank * 2)
    print(f"  trainable params: {count_lora_params(lora_model):,}")
    train_model("standard-LoRA", lora_model, train_ids, train_labels, val_ids, val_labels, args.steps, args.lr)

    print("=" * 70)
    print(f"Condition 2: zFLoRA reconstruction (aux_dim={args.rank})")
    zflora_model = ZFLoRAModel(fresh_base(), aux_dim=args.rank)
    print(f"  trainable params: {count_zflora_params(zflora_model):,}")
    train_model("zFLoRA", zflora_model, train_ids, train_labels, val_ids, val_labels, args.steps, args.lr)

    print("=" * 70)
    num_layers = args.num_layers
    lora_extra_matmuls = num_layers * 4 * 2  # q,k,v,o each add an A and a B matmul
    zflora_extra_matmuls = num_layers * 2 + 2  # backward+forward per layer, plus one-time expand/merge
    print("Structural proxy only (NOT a latency measurement -- see module docstring):")
    print(f"  standard LoRA adapter matmuls beyond the base forward pass: {lora_extra_matmuls}")
    print(f"  zFLoRA reconstruction adapter matmuls beyond the base forward pass: {zflora_extra_matmuls}")


if __name__ == "__main__":
    main()
