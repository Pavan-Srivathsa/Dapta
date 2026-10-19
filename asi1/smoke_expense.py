"""Train one tiny expense-line adapter and print its held-out output.

The public ASI agent stays closed. This operator script runs on the same
machine as that agent, unloads the inference model, trains at most six steps
or two minutes, then generates one line that was not in the training set.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


MAX_EXAMPLES = 10
MAX_STEPS = 6
MAX_TRAIN_SECONDS = 120
MAX_LENGTH = 256
MAX_NEW_TOKENS = 48

PROMPT = (
    "Convert the expense line into JSON with merchant, category, and amount. "
    "category is meals, travel, or supplies. Return JSON only."
)

TRAIN = (
    {"id": "train-1", "input": "Coffee at Blue Bottle for 4.50", "output": {"merchant": "Blue Bottle", "category": "meals", "amount": "4.50"}},
    {"id": "train-2", "input": "Lunch at Sweetgreen for 14.20", "output": {"merchant": "Sweetgreen", "category": "meals", "amount": "14.20"}},
    {"id": "train-3", "input": "Dinner at Nobu for 86.00", "output": {"merchant": "Nobu", "category": "meals", "amount": "86.00"}},
    {"id": "train-4", "input": "Uber to the airport for 38.00", "output": {"merchant": "Uber", "category": "travel", "amount": "38.00"}},
    {"id": "train-5", "input": "Train to San Jose for 12.75", "output": {"merchant": "Train", "category": "travel", "amount": "12.75"}},
    {"id": "train-6", "input": "Notebook from Staples for 6.40", "output": {"merchant": "Staples", "category": "supplies", "amount": "6.40"}},
)

PROBE = {
    "id": "probe-1",
    "input": "Taxi to the hotel for 22.00",
    "output": {"merchant": "Taxi", "category": "travel", "amount": "22.00"},
}


def example_count() -> int:
    return len(TRAIN) + 1


def _case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": case["input"]}]},
        ],
        "completion": [
            {"role": "assistant", "content": [{
                "type": "text",
                "text": json.dumps(case["output"], ensure_ascii=False, separators=(",", ":")),
            }]},
        ],
    }


def _parse_json(text: str) -> Any:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(value[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None


def train(output_dir: Path) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from dapta.factory_gpu import restore_frozen_compute_dtype

    if example_count() > MAX_EXAMPLES:
        raise RuntimeError("smoke dataset exceeds 10 examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("loading Qwen3-VL-8B in 4-bit", flush=True)
    base_model = os.environ.get("QWEN_BASE_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    processor = AutoProcessor.from_pretrained(base_model)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation=os.environ.get("QWEN_ATTN_IMPLEMENTATION", "sdpa"),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    restore_frozen_compute_dtype(model, torch)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    class Stop(TrainerCallback):
        started = 0.0
        stopped_by_time_limit = False

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, state, control, kwargs
            self.started = time.perf_counter()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if state.global_step >= MAX_STEPS or (
                self.started and time.perf_counter() - self.started >= MAX_TRAIN_SECONDS
            ):
                if self.started and time.perf_counter() - self.started >= MAX_TRAIN_SECONDS:
                    self.stopped_by_time_limit = True
                control.should_training_stop = True
            return control

    stop = Stop()
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(output_dir / "trainer"),
            num_train_epochs=1,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            max_length=MAX_LENGTH,
            completion_only_loss=True,
            packing=False,
            bf16=True,
            fp16=False,
            tf32=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="no",
            save_strategy="no",
            logging_steps=1,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            seed=20261018,
        ),
        train_dataset=Dataset.from_list([_case(case) for case in TRAIN]),
        processing_class=processor,
        peft_config=lora,
        callbacks=[stop],
    )
    print(
        f"training at most {MAX_STEPS} steps or {MAX_TRAIN_SECONDS} seconds",
        flush=True,
    )
    started = time.perf_counter()
    train_result = trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    model = trainer.model
    model.eval()
    model.config.use_cache = True
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": PROBE["input"]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    completion = processor.batch_decode(
        generated[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    parsed = _parse_json(completion)
    result = {
        "adapter": "expense-line-smoke",
        "base_model": base_model,
        "adapter_dir": str(output_dir),
        "train_examples": len(TRAIN),
        "held_out_examples": 1,
        "max_steps": MAX_STEPS,
        "max_train_seconds": MAX_TRAIN_SECONDS,
        "training_seconds": round(training_seconds, 3),
        "global_step": int(getattr(train_result, "global_step", 0) or trainer.state.global_step),
        "stopped_by_time_limit": stop.stopped_by_time_limit,
        "probe_input": PROBE["input"],
        "expected": PROBE["output"],
        "model_output": completion,
        "parsed_output": parsed,
        "matched": parsed == PROBE["output"],
    }
    (output_dir / "smoke_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    root = Path(os.environ.get("DAPTA_ASI1_SMOKE_DIR", "artifacts/asi1-smoke/expense-line"))
    train(root)


if __name__ == "__main__":
    main()
