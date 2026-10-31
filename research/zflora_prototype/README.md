# zFLoRA Day 1 prototype

Small, from-scratch tiny Llama-architecture model (CPU, random init, no
downloads) used to validate that our reconstruction of zFLoRA (see
`zflora.py`'s module docstring for the exact wiring and which parts are
our judgment calls vs the paper's own wording) is mechanically sound
before porting it to Qwen3-VL on GPU. See `../../PLANNING.md` for the
full multi-day plan this is Day 1 of.

## Files

- `tiny_llama.py` -- the frozen base model (RMSNorm, RoPE, GQA attention,
  SwiGLU FFN).
- `standard_lora.py` -- standard LoRA baseline, attached at q/k/v/o like
  `backend/dapta/train.py` does in production.
- `zflora.py` -- our reconstruction of the zFLoRA architecture.
- `train_toy.py` -- trains both adapters on a synthetic next-token task
  and reports loss/accuracy.

## Run it

```
.venv/bin/python research/zflora_prototype/train_toy.py --task increment
.venv/bin/python research/zflora_prototype/train_toy.py --task pair-sum
```

## Day 1 result (recorded here, not just in a terminal scrollback)

Two synthetic tasks, same tiny random-init base, same rank/aux-dim (4),
same step budget:

- **`increment`** (`y[t] = x[t-1] + 1`, purely pointwise, no cross-position
  dependency): zFLoRA reconstruction **matches or beats** standard LoRA --
  val_acc 0.928 vs 0.926, val_loss 0.39 vs 1.05 -- using **less than half
  the trainable params** (768 vs 1,792).
- **`pair-sum`** (`y[t] = x[t-1] + x[t-2]`, requires combining two prior
  positions -- i.e. requires learning new cross-position attention routing
  on a randomly-initialized, frozen base): zFLoRA reconstruction plateaus
  around val_acc 0.08 even at 5x the steps and higher LR, while standard
  LoRA reaches 0.30.

**Why**: our wiring attaches adapters at the attention output projection
(`Wo`) and inside the FFN -- matching the paper's own description -- and
never touches `q_proj`/`k_proj`. That means it can reshape/bias per-token
output after attention has already happened, but it cannot change *what
the (frozen) attention mechanism attends to*. Standard LoRA, by directly
adding a delta to `q_proj`/`k_proj`/`v_proj`, can. On a random base that
has no useful attention pattern to begin with, that's a real structural
disadvantage for tasks that need new routing.

**Why this isn't necessarily a problem for Day 3**: Qwen3-VL's base is
pretrained, not random -- it already has broadly sensible attention
patterns. Skill fine-tuning plausibly looks more like `increment`
(reweight/bias what the model already attends to and produces) than
`pair-sum` (learn attention routing from nothing). This is now a specific,
checkable question for Day 2's quality check, not a guess: does the
accuracy gap we saw on `pair-sum` show up on a real small pretrained
checkpoint, or does it behave like `increment`?

**Pass gate status**: met. Both conditions train, gradients reach only
adapter parameters, loss decreases. The `increment` vs `pair-sum` split is
a bonus finding beyond the minimum gate, and the one thing Day 2 should
specifically test for.

## Day 2 result: real pretrained checkpoint (`day2_real_checkpoint.py`)

Switched implementation approach: `hf_adapters.py` uses PyTorch forward
hooks on an **unmodified real HF model** (`Qwen/Qwen2.5-0.5B`) instead of
reimplementing the decoder layer. Hook points: `self_attn.o_proj`
(backward adapter), `mlp` pre-hook (forward adapter, observes FFN input),
`lm_head` pre-hook (final merge). This is deliberately the same mechanism
Day 3 will point at Qwen3-VL's language backbone, not throwaway code.

Task: a tiny structured-extraction mini-task shaped like Dapta's actual
skills (`"Sentence: The red ball is here.\nJSON:"` ->
`' {"color": "red", "object": "ball"}'`), 48 train / 16 val, completion-
only loss (prompt tokens masked), exact-match JSON eval via greedy
generation.

**Result at 80 steps**: standard LoRA and the zFLoRA reconstruction both
reach **1.000 val exact-match accuracy**, with zFLoRA using **358,400
trainable params vs LoRA's 1,081,344 (about a third)** and converging
almost as fast (train loss near-zero by step 32 for both).

**This confirms the Day 1 working hypothesis directly**: the attention-
routing ceiling seen on the random-init toy model (`pair-sum` task) does
not show up here. A pretrained base already has useful attention
patterns; fine-tuning a skill onto it is closer to `increment` (bias what
the model already does) than `pair-sum` (learn routing from nothing) --
and zFLoRA's wiring, which can't touch `q_proj`/`k_proj`, is fine for
that regime.

**New limitation found, relevant to Day 5 (serving), not Day 2's gate**:
this hook-based implementation recomputes the aux channel from scratch on
every hooked forward call. Under vLLM/HF's normal KV-cached incremental
decoding, each decode step only runs a forward pass over the *new* token,
so the aux channel would be wrongly reset from just that one token instead
of reflecting the accumulated sequence. Eval here used `use_cache=False`
(correct but slow -- full recompute every decode step) as a workaround.
Production serving needs either an aux-channel KV-cache analogue or a
different integration point -- added to Day 5's scope below, not solved
here.

**Pass gate status**: met, and the specific question Day 1 flagged is
answered: real pretrained base, no measurable accuracy gap, at 1/3 the
adapter parameters.

## Day 3 prep: bugs found before spending GPU time (`day3_qwen3vl.py`)

Reviewed and CPU-smoke-tested the Day 3 script against Qwen2.5-0.5B
(same chat template family) before any pod run. Three real bugs, all
fixed:

1. **HF gradient checkpointing silently corrupts zFLoRA gradients.** The
   production QLoRA envelope enables `model.gradient_checkpointing_enable()`,
   which re-runs each decoder layer during backward. The o_proj/mlp hooks
   read and mutate the shared `self._aux`, so on that re-run they saw the
   aux left behind by the *last* layer. Measured on Qwen2.5-0.5B: every
   adapter except `final_merge` got gradients 87%-550% off, no error, no
   `None` grads -- the run would have looked like it was training. Fix:
   `_AuxCarryingLayer` wraps each decoder layer so aux is an explicit
   input/output under non-reentrant checkpointing
   (`ZFLoRAHookWrapper(..., checkpoint_layers=True)`); the wrapper now
   refuses a base with HF checkpointing on. Verified: identical logits and
   0.0 relative gradient difference vs no checkpointing across all 50
   adapter tensors. Day 2 re-run after the change: still 1.000 / 1.000.
2. **`apply_chat_template(tokenize=True)` returns a BatchEncoding on
   transformers 5.x**, not a list, so prompt building would have crashed
   (or, wrapped in `list()`, produced 2-token prompts). Now renders text
   then tokenizes, which works on 4.x and 5.x.
3. **Truncation sliced the templated ids**, dropping `<|im_end|>` and the
   assistant header. Now truncates the NDA text (which comes last in the
   user turn) and re-templates.

Also added, because the PLANNING.md gate is dev *accuracy* not loss:
greedy-generation eval scored by `backend/dapta/factory_gpu.py`'s own
`score()` (field F1 / exact form), a first-step check that every trainable
tensor gets a gradient, peak-memory reporting, micro-batch 1 (matches the
runbook), and results written to `day3_results.json`.

Note for reading Day 3 numbers: at the default `--max-prompt-tokens 3072`,
20 of 25 dev NDAs are truncated, and dates/signature blocks often sit at
the end. LoRA-vs-zFLoRA is still a fair head-to-head, but it is not
comparable to the shipped adapter's 70.75% field F1 (test split, 6144
tokens) unless run with `--max-prompt-tokens 6144`.
