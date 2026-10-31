"""PLANNING.md Day 3: port the zFLoRA reconstruction (validated on a real
small checkpoint in Day 2) onto Qwen3-VL-8B-Instruct's language backbone,
on the real nda-intake dataset (artifacts/datasets/kleister-nda/), text
only -- vision tower untouched, matching the existing project decision to
adapt the language backbone first (docs/PROJECT_CONTEXT.md).

Same 4-bit QLoRA envelope as production (docs/TRAINING_RUNBOOK.md,
backend/dapta/train.py): NF4 double-quant, bf16 compute, gradient
checkpointing, SDPA attention, micro-batch 1. Standard LoRA uses HF's own
gradient checkpointing; zFLoRA uses ZFLoRAHookWrapper(checkpoint_layers=
True) instead, because HF's silently corrupts the zFLoRA adapter gradients
(see _AuxCarryingLayer in hf_adapters.py).

Pass gate (PLANNING.md) is dev-set *accuracy*, so after training each
condition is scored by greedy generation with backend/dapta/factory_gpu.py's
own score() -- the same field-F1 / exact-form metric behind the shipped
nda-intake adapter's 70.75% field F1 (that number is on the test split at
6144 prompt tokens; this compares both conditions on dev under identical
settings). zFLoRA generates with use_cache=False (the Day 2 KV-cache
limitation, scoped to Day 5), so its eval is slower, not less correct.
Results are also written to --out as JSON.

Deliberate scope reduction vs production, disclosed here rather than
silently done: prompts are truncated to --max-prompt-tokens (default
3072) tokens from the start of the document, instead of the full 6144
production uses, to keep this go/no-go check fast on a single smoke run.
Some NDA fields that appear late in a long document may become
unanswerable under this truncation -- that's a task-difficulty variable,
not a bug in the adapter mechanism being tested.

Usage (on the GPU pod):
  HF_TOKEN=... python3 day3_qwen3vl.py --steps 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from hf_adapters import ZFLoRAHookWrapper, apply_standard_lora_hf, count_trainable_params

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from dapta.factory_gpu import _parse_json, score  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DATASET_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "datasets" / "kleister-nda"


def load_rows(split: str) -> list[dict]:
    rows = []
    with open(DATASET_DIR / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_prompt_ids(tokenizer, row: dict, max_prompt_tokens: int) -> list[int]:
    system_text = row.get("system", "Follow the requested output schema exactly.")
    user_text = row["prompt"]

    def template(text: str) -> list[int]:
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": text},
        ]
        # Render to text, then tokenize: apply_chat_template(tokenize=True)
        # returns a list on transformers 4.x but a BatchEncoding on 5.x.
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return tokenizer(text, add_special_tokens=False).input_ids

    prompt_ids = template(user_text)
    overflow = len(prompt_ids) - max_prompt_tokens
    if overflow > 0:
        # The schema instructions come first and the NDA text last
        # ("...NDA:\n<document>"), so cut from the end of the user text and
        # re-template. Slicing the templated ids would instead drop the
        # end-of-turn and assistant header the model must generate after.
        text_ids = tokenizer(user_text, add_special_tokens=False).input_ids
        while overflow > 0:
            text_ids = text_ids[: len(text_ids) - overflow]
            prompt_ids = template(tokenizer.decode(text_ids))
            overflow = len(prompt_ids) - max_prompt_tokens
    return prompt_ids


def completion_text(row: dict) -> str:
    completion = row["completion"]
    if not isinstance(completion, str):
        completion = json.dumps(completion, ensure_ascii=False, separators=(",", ":"))
    return completion


def build_example(tokenizer, row: dict, max_prompt_tokens: int) -> dict:
    prompt_ids = build_prompt_ids(tokenizer, row, max_prompt_tokens)
    completion = completion_text(row)

    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    eos_id = tokenizer.eos_token_id
    full_ids = prompt_ids + completion_ids + [eos_id]
    labels = [-100] * len(prompt_ids) + completion_ids + [eos_id]
    return {"input_ids": full_ids, "labels": labels}


def collate(examples: list[dict], pad_id: int, device: torch.device):
    max_len = max(len(ex["input_ids"]) for ex in examples)
    input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    for row, ex in enumerate(examples):
        length = len(ex["input_ids"])
        input_ids[row, :length] = torch.tensor(ex["input_ids"])
        labels[row, :length] = torch.tensor(ex["labels"])
        attention_mask[row, :length] = 1
    return input_ids.to(device), labels.to(device), attention_mask.to(device)


def completion_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    return nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]).float(), shift_labels.reshape(-1), ignore_index=-100
    )


def load_base_model(hf_gradient_checkpointing: bool) -> Qwen3VLForConditionalGeneration:
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=os.environ.get("HF_TOKEN"),
        quantization_config=quantization,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model.config.use_cache = False
    if hf_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model


def cast_new_adapter_params(model: nn.Module, reference: nn.Module) -> None:
    """Newly constructed nn.Linear adapters default to fp32 on CPU; the
    frozen 4-bit base computes in bf16. Mismatched dtypes here is exactly
    the 'Float but found BFloat16' failure TRAINING_RUNBOOK.md documents
    for the frozen base -- same fix, applied to our new adapter params.
    """
    device = next(reference.parameters()).device
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(device=device, dtype=torch.bfloat16)


def train_condition(name: str, model, tokenizer, train_rows, dev_rows, device, steps: int, lr: float, max_prompt_tokens: int, batch_size: int):
    train_examples = [build_example(tokenizer, row, max_prompt_tokens) for row in train_rows]
    dev_examples = [build_example(tokenizer, row, max_prompt_tokens) for row in dev_rows]
    pad_id = tokenizer.pad_token_id

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    rng = torch.Generator().manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(steps):
        idx = torch.randperm(len(train_examples), generator=rng)[:batch_size].tolist()
        batch = [train_examples[i] for i in idx]
        input_ids, labels, attention_mask = collate(batch, pad_id, device)

        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits if hasattr(output, "logits") else output
        loss = completion_loss(logits, labels)
        loss.backward()
        if step == 0:
            # Cheap guard against the class of bug that broke zFLoRA under
            # HF checkpointing: every adapter tensor must receive a gradient.
            dead = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or not p.grad.any())]
            if dead:
                raise RuntimeError(f"{name}: {len(dead)} trainable tensors got no gradient, e.g. {dead[:3]}")
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 5) == 0:
            print(f"  [{name}] step {step + 1:>4}/{steps}  train_loss={loss.item():.4f}")

    train_seconds = time.perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    print(f"  [{name}] trained in {train_seconds:.0f}s, peak CUDA memory {peak_gib:.1f} GiB")

    model.eval()
    dev_loss_total, dev_tokens = 0.0, 0
    with torch.no_grad():
        for ex in dev_examples:
            input_ids, labels, attention_mask = collate([ex], pad_id, device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits if hasattr(output, "logits") else output
            loss = completion_loss(logits, labels)
            n_completion_tokens = (labels[:, 1:] != -100).sum().item()
            dev_loss_total += loss.item() * n_completion_tokens
            dev_tokens += n_completion_tokens
    dev_loss = dev_loss_total / max(1, dev_tokens)
    print(f"  [{name}] dev completion-token loss = {dev_loss:.4f}  ({len(dev_examples)} examples)")
    return {"dev_loss": dev_loss, "train_seconds": train_seconds, "peak_train_memory_gib": peak_gib}


def generate_and_score(name: str, model, tokenizer, dev_rows, device, max_prompt_tokens: int, use_cache: bool, max_new_tokens: int) -> dict:
    predictions, elapsed = [], []
    input_tokens = output_tokens = 0
    for index, row in enumerate(dev_rows):
        input_ids = torch.tensor([build_prompt_ids(tokenizer, row, max_prompt_tokens)], device=device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ids = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed.append(time.perf_counter() - started)
        generated = ids[0, input_ids.shape[1]:]
        predictions.append(_parse_json(tokenizer.decode(generated, skip_special_tokens=True)))
        input_tokens += input_ids.shape[1]
        output_tokens += generated.shape[0]
        print(f"  [{name}] generated {index + 1}/{len(dev_rows)} ({elapsed[-1]:.1f}s)")
    cases = [{"id": row.get("id"), "output": row["completion"]} for row in dev_rows]
    result = score(cases, predictions, {}, elapsed, input_tokens, output_tokens)
    print(
        f"  [{name}] dev field_f1={result['field_f1']:.4f}  exact_form_rate={result['exact_form_rate']:.4f}"
        f"  schema_pass_rate={result['schema_pass_rate']:.4f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("day3_results.json"))
    args = parser.parse_args()

    device = torch.device("cuda")
    train_rows = load_rows("train")
    dev_rows = load_rows("dev")
    print(f"{len(train_rows)} train / {len(dev_rows)} dev rows from {DATASET_DIR}")

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=os.environ.get("HF_TOKEN"))
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 70)
    print(f"Condition 1: standard LoRA (rank={args.rank}, targets q/k/v/o) on {MODEL_ID}")
    lora_model = apply_standard_lora_hf(load_base_model(hf_gradient_checkpointing=True), rank=args.rank, alpha=args.rank * 2)
    cast_new_adapter_params(lora_model, lora_model)
    results = {"args": {k: str(v) for k, v in vars(args).items()}, "model": MODEL_ID, "revision": MODEL_REVISION}
    results["standard_lora"] = {"trainable_params": count_trainable_params(lora_model)}
    print(f"  trainable params: {results['standard_lora']['trainable_params']:,}")
    results["standard_lora"].update(train_condition(
        "standard-LoRA", lora_model, tokenizer, train_rows, dev_rows, device, args.steps, args.lr, args.max_prompt_tokens, args.batch_size
    ))
    results["standard_lora"]["eval"] = generate_and_score(
        "standard-LoRA", lora_model, tokenizer, dev_rows, device, args.max_prompt_tokens, use_cache=True, max_new_tokens=args.max_new_tokens
    )
    args.out.write_text(json.dumps(results, indent=2))
    del lora_model
    torch.cuda.empty_cache()

    print("=" * 70)
    print(f"Condition 2: zFLoRA reconstruction (aux_dim={args.rank}) on {MODEL_ID}")
    base = load_base_model(hf_gradient_checkpointing=False)
    zflora_model = ZFLoRAHookWrapper(base, aux_dim=args.rank, checkpoint_layers=True)
    cast_new_adapter_params(zflora_model, base)
    results["zflora"] = {"trainable_params": count_trainable_params(zflora_model)}
    print(f"  trainable params: {results['zflora']['trainable_params']:,}")
    results["zflora"].update(train_condition(
        "zFLoRA", zflora_model, tokenizer, train_rows, dev_rows, device, args.steps, args.lr, args.max_prompt_tokens, args.batch_size
    ))
    results["zflora"]["eval"] = generate_and_score(
        "zFLoRA", zflora_model, tokenizer, dev_rows, device, args.max_prompt_tokens, use_cache=False, max_new_tokens=args.max_new_tokens
    )
    args.out.write_text(json.dumps(results, indent=2))

    print("=" * 70)
    for key in ("standard_lora", "zflora"):
        r = results[key]
        print(
            f"Day 3 {key:>13}: params={r['trainable_params']:,}  dev_loss={r['dev_loss']:.4f}"
            f"  field_f1={r['eval']['field_f1']:.4f}  exact={r['eval']['exact_form_rate']:.4f}"
            f"  peak_mem={r['peak_train_memory_gib']:.1f}GiB"
        )
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
