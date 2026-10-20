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
