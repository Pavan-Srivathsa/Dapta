# zFLoRA implementation plan

Branch: `zflora-implementation`. This document is the entry point for a
multi-day effort and should be updated as each stage completes or as
assumptions get corrected by evidence — treat the "Open questions / risks"
section as living, not historical.

## Goal

Dapta currently serves standard QLoRA adapters (see `docs/TRAINING_RUNBOOK.md`,
`backend/dapta/train.py`, `backend/dapta/factory_gpu.py`). Standard LoRA
adapters measurably add latency over the unadapted base model — this is the
entire problem zFLoRA targets. The goal of this effort is to get an adapted
Qwen3-VL skill as close to **base-model latency** as possible, so an adapter
never looks slower than just asking the base model.

**Explicit, corrected framing**: no adapter technique makes a model faster
than its own unmodified base model — the adapter still does the base
model's forward pass plus something. The target is **matching base latency
with near-zero overhead**, not beating it. "Beat the base model" means on
*task accuracy*, while *matching* it on latency — not literally
outrunning it.

## What zFLoRA is (grounded in the primary source)

Paper: Gowda, Song, Goka, Lee (Samsung Research), *zFLoRA: Zero-Latency
Fused Low-Rank Adapters*, EMNLP 2025 Main, [arXiv:2510.25784](https://arxiv.org/abs/2510.25784v1).
No public code release exists (checked — nothing on GitHub under the
authors' names or "zFLoRA").

- Standard LoRA computes `Z = WX + BAX` as separate sequential ops (`WX`,
  `AX`, `B(AX)`, then add). The paper measures this costing **1.3–2.5x**
  base-model prefill latency and **1.3–1.6x** decode latency on an H100,
  *even when served through vLLM's existing multi-LoRA path* — the
  overhead is real on exactly the serving stack Dapta already uses.
- zFLoRA restructures *where* the forward (down, `A`) and backward (up,
  `B`) adapter pieces attach: the backward adapter attaches to the
  attention output projection (`Wo`), the forward adapter attaches within
  the FFN block, and the low-rank signal between them is **carried through
  the residual stream** across the MHA→FFN boundary of the same layer,
  rather than computed and merged locally at every adapted linear layer.
- Consequence: the hidden/residual stream is `d+r` wide for the **entire
  model stack**, not just at adapted layers. It's expanded once before
  layer 1 and merged back to `d` once before the LM head. `Wq/Wk/Wv` and
  RoPE only ever see the original `d`-wide slice, which is why RoPE isn't
  disturbed (a "uniform" width-everywhere variant the paper also tried does
  disturb RoPE and measurably hurts quality — avoid that variant).
- Reported result (H100, vLLM, FP16, 8B model, 8192-token input): base
  267ms / LoRA 354ms / zFLoRA 249ms TTFT — i.e. zFLoRA slightly *beat* the
  base-model number in their table, within measurement noise of "zero
  overhead." Decode-step latency differences are smaller in absolute terms
  (base 7.93ms / LoRA 10.50ms / zFLoRA 7.97ms per token) but follow the
  same pattern. Task accuracy on 18 commonsense/math/dialogue tasks tracks
  standard LoRA and full fine-tuning closely (within ~1 point in most
  cases).

**This is a model-architecture change, not a PEFT config.** It requires
patching the decoder-layer forward pass (residual add, Add&Norm, the `Wo`
and FFN hookpoints), not attaching a `peft.LoraConfig` to existing
`nn.Linear` modules — there's no PEFT hook for "carry an extra channel
through the global residual stream." The paper's own architecture section
(§4) is prose + an unlabeled block diagram, **no equations or pseudocode**
for the actual zFLoRA wiring (only for the simpler FFA/FBA/pf-LoRA
baselines it rejects along the way). Our implementation is therefore a
**reconstruction** from the figure and prose, not a reproduction of
verified reference code — flag every place this matters.

## The real tension with how Dapta serves adapters

zFLoRA's "zero overhead" number is for a **single specialized model** —
their "Fused" deployment variant literally folds the adapter into the base
weights at deploy time. There is no "resident shared base + hot-swap a
small delta" story in the paper; swapping to a different zFLoRA adapter
means swapping which weights are loaded entirely. That's exactly what
vLLM's multi-LoRA (punica/sgmv kernels) gives Dapta *today* for standard
LoRA, and is central to the "hot-swappable adapter, revert to base
instantly" pitch.

Integrating zFLoRA into vLLM's dynamic multi-adapter serving would mean
**registering a custom model architecture in vLLM** (a model whose hidden
width is `d+r` for one adapter and `d` for the base/another adapter isn't
expressible through vLLM's existing LoRA-adapter loading API) — real
engineering on top of the training-side reconstruction. We are not
committing to that path yet; see Day 5 below.

## Staged plan

Each stage has a pass/fail gate, not a vibe check — keep that discipline
(matches [[user-rejects-subjective-tasks]] generally: prefer
exactly-checkable outcomes at each stage).

### Day 1 — Small-model CPU prototype, no GPU, no vLLM

- Build a tiny from-scratch Llama-architecture model (`transformers`
  `LlamaConfig` + `LlamaForCausalLM`, small dims, random init, no
  downloads — instant on CPU) to mirror Qwen3-VL's language backbone
  structure (RMSNorm, RoPE, SwiGLU, GQA-compatible).
- Implement our reconstructed zFLoRA decoder layer as a patched forward
  pass on top of it: input expansion before layer 1, backward adapter at
  `Wo` consuming the carried aux channel, forward adapter in the FFN
  producing the next aux channel, final merge before the LM head. Base
  weights frozen; only adapter params trainable.
- Also implement a standard-LoRA tiny baseline on the same toy model for
  comparison.
- **Pass gate**: forward pass runs with correct shapes; gradients flow
  only to adapter params; training loss decreases on a tiny synthetic
  next-token task over a few hundred steps, for both the zFLoRA
  reconstruction and the standard-LoRA baseline, to confirm the mechanism
  is at least trainable and mechanically sound.
- **Explicit non-goal**: CPU wall-clock timing here does *not* validate
  the paper's latency claim. The claim is about GPU kernel-launch overhead
  (sequential CUDA kernel launches vs one fused matmul); CPU execution
  doesn't have the same async-launch cost structure. Day 1 proves
  "trains and is architecturally coherent," not "is faster." Op/node
  count in the computation graph (fewer sequential matmuls vs standard
  LoRA) is a fair structural proxy to record here, wall-clock is not.

### Day 2 — Quality sanity check on a small real checkpoint

- Repeat the Day 1 setup on a small *real* pretrained checkpoint (e.g. a
  small open Llama-family model) and a small real task, comparing zFLoRA
  reconstruction vs standard LoRA on eval loss / accuracy, not just loss
  going down.
- Use any quality gap (or lack of one) to pressure-test which wiring
  assumptions we got right vs wrong versus the paper's Figure 4 — the
  figure's exact per-projection assignment wasn't fully recoverable from
  the text extraction; if our reconstruction clearly underperforms
  standard LoRA by a wide margin, that's a signal the wiring is off, not
  that the technique fails.
- **Pass gate**: zFLoRA reconstruction gets within a small, defined margin
  of standard LoRA's eval accuracy on the same small task (paper shows
  ~0–1.5 point gaps on 1B/3B models — use that as the target band, not an
  exact reproduction).

### Day 3 — Port to Qwen3-VL's language backbone (GPU, test pod)

- Move the validated reconstruction onto `Qwen/Qwen3-VL-8B-Instruct`'s
  language backbone only (vision tower frozen, matching the existing
  project decision in `docs/PROJECT_CONTEXT.md` to adapt the language
  backbone first). Scope to one text-only skill first: `nda-intake`
  (lowest risk — no image pipeline involved).
- Run on Pavan's test pod (`sheer_pink_turkey`, A40) — see
  [[runpod-environments]] for access.
- **Pass gate**: trains without crashing inside the existing QLoRA memory
  envelope documented in `docs/TRAINING_RUNBOOK.md` (4-bit NF4, micro-batch
  1, gradient checkpointing); dev-set accuracy roughly matches the
  already-shipped standard-LoRA `nda-intake` adapter.

### Day 4 — Measure real latency on Qwen3-VL (GPU, test pod)

- Reproduce the paper's own methodology: TTFT and per-token latency at
  several input lengths, comparing base / standard LoRA / zFLoRA
  reconstruction, all served the same way (plain HF `generate` first,
  since vLLM integration isn't built yet).
- **Pass gate**: zFLoRA reconstruction measurably closes the gap to base
  latency versus the existing standard-LoRA adapter, on this hardware, on
  this model. If it doesn't, stop here and report why before investing in
  Day 5 — this is the actual go/no-go checkpoint for the whole effort.

### Day 5 — Decide and build the serving path

Only after Day 4 confirms a real win. Two options, pick based on what Day
4 showed and actual demo requirements:

- **A. Custom vLLM model registration** — true dynamic hot-swap serving
  of multiple zFLoRA adapters on a shared resident base. Highest effort,
  matches Dapta's existing multi-adapter architecture most closely.
- **B. Merge-and-swap-whole-model** — skip dynamic LoRA swapping
  entirely; fold each active zFLoRA adapter into its own full model
  checkpoint and swap which checkpoint is loaded when the routed skill
  changes. Much less engineering, but swap latency becomes "reload a full
  model" instead of "swap a small delta" — only acceptable if Dapta's
  demo doesn't need sub-second adapter switching.

### Day 6+ — Integrate into Dapta's factory pipeline

- Wire the validated path into `backend/dapta/train.py` /
  `backend/dapta/factory_gpu.py` behind an explicit opt-in (new skill
  config or env flag), not as a replacement for the existing QLoRA recipe,
  until it's proven on more than one skill.
- Extend to the image-bearing skills (`receipt-to-expense`,
  `csv-to-dashboard`) only after the text-only path is solid — the vision
  tower interaction with a widened residual stream is unvalidated and
  likely the riskiest remaining unknown.

## Open questions / risks (keep current)

- Exact per-projection wiring inside the FFN block (forward adapter on
  the gate projection vs the down projection) isn't fully recoverable
  from the paper's text extraction — Day 2's quality check is partly a
  proxy for resolving this empirically.
- No public reference implementation exists to diff against — everything
  here is our own reconstruction until Day 2/4 data says otherwise.
- Paper validates text-only LLaMA 1B/3B/8B only. Qwen3-VL's multimodal
  fusion and GQA-vs-MHA details are untested territory.
- The vLLM serving story (Day 5) is explicitly unsolved by the paper
  itself — budget for this being the hardest part, not the training side.

## Status

- [ ] Day 1 — small-model CPU prototype
- [ ] Day 2 — quality sanity check
- [ ] Day 3 — port to Qwen3-VL backbone
- [ ] Day 4 — latency measurement, go/no-go
- [ ] Day 5 — serving path decision + build
- [ ] Day 6+ — factory pipeline integration
