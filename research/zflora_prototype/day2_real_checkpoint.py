"""PLANNING.md Day 2: does the attention-routing gap found on Day 1's
random-init toy model (research/zflora_prototype/README.md) actually show
up when fine-tuning a real *pretrained* checkpoint, or was that an
artifact of starting from random attention weights?

Task: tiny structured-extraction mini-task, shaped like Dapta's actual
skills (free text -> JSON), not abstract arithmetic:
  "Sentence: The red ball is here.\nJSON:" -> ' {"color": "red", "object": "ball"}'

Base model: Qwen/Qwen2.5-0.5B (real Qwen family, same q/k/v/o_proj +
gate/up/down_proj naming as Qwen3-VL's language backbone, runs on CPU).

Usage: .venv/bin/python research/zflora_prototype/day2_real_checkpoint.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import random

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from hf_adapters import ZFLoRAHookWrapper, apply_standard_lora_hf, count_trainable_params

MODEL_ID = "Qwen/Qwen2.5-0.5B"
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]
OBJECTS = ["ball", "car", "hat", "book", "cup", "shoe", "box", "kite"]


def build_examples() -> list[dict]:
    combos = list(itertools.product(COLORS, OBJECTS))
    random.Random(0).shuffle(combos)
    examples = []
    for color, obj in combos:
        prompt = f"Sentence: The {color} {obj} is here.\nJSON:"
        completion = f' {{"color": "{color}", "object": "{obj}"}}'
        examples.append({"prompt": prompt, "completion": completion, "color": color, "object": obj})
    return examples


def tokenize_batch(tokenizer, examples: list[dict], device: torch.device):
    prompt_ids_list, label_ids_list = [], []
    for ex in examples:
        prompt_ids = tokenizer(ex["prompt"], add_special_tokens=False).input_ids
        completion_ids = tokenizer(ex["completion"], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        prompt_ids_list.append(prompt_ids)
        label_ids_list.append([-100] * len(prompt_ids) + completion_ids)

    full_ids_list = [p + l[len(p):] for p, l in zip(prompt_ids_list, label_ids_list)]
    max_len = max(len(ids) for ids in full_ids_list)

    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    for row, (full_ids, label_ids) in enumerate(zip(full_ids_list, label_ids_list)):
        length = len(full_ids)
        input_ids[row, :length] = torch.tensor(full_ids)
        labels[row, :length] = torch.tensor(label_ids)
        attention_mask[row, :length] = 1
    return input_ids.to(device), labels.to(device), attention_mask.to(device)


def completion_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    return nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1), ignore_index=-100
    )


@torch.no_grad()
def exact_match_eval(model, tokenizer, examples: list[dict], device: torch.device) -> float:
    correct = 0
    for ex in examples:
        prompt_ids = tokenizer(ex["prompt"], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        # use_cache=False: this prototype's aux channel is recomputed from
        # scratch on every hooked forward call (see hf_adapters.py), so KV
        # caching would only see the newest token and reset aux wrongly.
        # Correct but slow -- fine for a 16-example sanity check, not for
        # production serving (a Day 5 concern, not a Day 2 one).
        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=24,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = tokenizer.decode(output_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True).strip()
        try:
            parsed = json.loads(generated)
            correct += int(parsed == {"color": ex["color"], "object": ex["object"]})
        except json.JSONDecodeError:
            pass
    return correct / len(examples)


def train_condition(name: str, model, tokenizer, train_examples, val_examples, device, steps: int, lr: float):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    input_ids, labels, attention_mask = tokenize_batch(tokenizer, train_examples, device)

    for step in range(steps):
        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits if hasattr(output, "logits") else output
        loss = completion_loss(logits, labels)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 5) == 0:
            print(f"  [{name}] step {step + 1:>4}/{steps}  train_loss={loss.item():.4f}")

    accuracy = exact_match_eval(model, tokenizer, val_examples, device)
    print(f"  [{name}] DONE  val exact-match JSON accuracy = {accuracy:.3f}  ({len(val_examples)} examples)")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = build_examples()
    train_examples, val_examples = examples[:48], examples[48:]
    print(f"{len(train_examples)} train / {len(val_examples)} val examples, base model {MODEL_ID}")

    print("=" * 70)
    print(f"Condition 1: standard LoRA (rank={args.rank}, targets q/k/v/o) on real Qwen2.5-0.5B")
    lora_model = apply_standard_lora_hf(
        AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32), rank=args.rank, alpha=args.rank * 2
    )
    print(f"  trainable params: {count_trainable_params(lora_model):,}")
    lora_acc = train_condition("standard-LoRA", lora_model, tokenizer, train_examples, val_examples, device, args.steps, args.lr)

    print("=" * 70)
    print(f"Condition 2: zFLoRA reconstruction (aux_dim={args.rank}) on real Qwen2.5-0.5B")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    zflora_model = ZFLoRAHookWrapper(base, aux_dim=args.rank)
    print(f"  trainable params: {count_trainable_params(zflora_model):,}")
    zflora_acc = train_condition("zFLoRA", zflora_model, tokenizer, train_examples, val_examples, device, args.steps, args.lr)

    print("=" * 70)
    print(f"Day 2 result: standard-LoRA val acc = {lora_acc:.3f}   zFLoRA reconstruction val acc = {zflora_acc:.3f}")


if __name__ == "__main__":
    main()
