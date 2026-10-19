# Continuation handoff for Claude

Written: October 18, 2026, about 11:20 PM America/Los_Angeles.
Repo: `/Users/pavan/Dapta`. GitHub: `adiv-1/Dapta`.
This file is the session handoff. `docs/AGENT_HANDOFF.md` is still the production
takeover document. Read both before changing paid factory, RunPod, or Base44
behavior. Do not commit secrets. Do not start a paid factory run unless the user
asks.

Nothing in this session was committed.

## What the user wants, in order

1. Work from the newest GitHub branch, then a new branch named exactly `Depta-dev`.
2. Diagnose why the Base44 website throws an error.
3. Put 5 to 6 base models and about 30 adapters in the directory, using 4 to 5
   tasks that already exist. A task must be able to switch models and adapters
   and run on the efficient choice.
4. Add a RunPod login so a user can connect their own GPU and train.
5. They asked for that plan as a diagram, a timeline, and code. That plan was
   not written. This file replaces it.
6. Fix ASI:One so the chat side works completely, the same product as the web
   app. That part was implemented locally and tested. It was not deployed.

## Checkout right now

- Branch: `Depta-dev` (local only, no upstream).
- Commit: `3350d92` (`Add clearly labeled demo benchmark projections`), which is
  `origin/main` as of October 18, 2026, 10:28 AM PT.
- `Depta-dev` was created from that commit. Uncommitted `creao-integration` work
  was stashed first so the switch could happen.
- Working tree on top of `3350d92` contains the ASI companion plus the ASI fixes
  below. `git status` shows those files as added or modified. Do not assume they
  are on GitHub.

Stashes, do not drop them:

- `stash@{0}`: `On creao-integration: WIP on creao-integration before Depta-dev`.
  The tracked half was the pre-fix ASI comparison work. The untracked half
  (`stash@{0}^3`) still holds files that were **not** restored:
  `asi1/run-smoke-receipt.sh`, `asi1/smoke_receipt.py`, `creao/`, `reports/`,
  `research_notes/Receipt adapter integration/`.
- `stash@{1}`: `On asi1-integration: pre-merge WIP: factory limits + asi1 training access (06:39)`.

`creao-integration` (`44620c5`, local, not on GitHub) is 9 commits ahead of an
older main and is missing `3350d92`. It is where ASI was developed. This branch
now has that companion in the working tree, plus the fixes in this file.
`origin/main` does not contain `asi1/`.

## Already done: ASI:One

ASI:One is not a second product. `asi1/` is an Agentverse chat companion over the
existing controller (`:8000`) and Qwen worker (`:8002`). Handle on Agentverse:
`@dapta-lahacks`. Companion port is **8004**. vLLM stays on loopback **8003**.
Do not put the companion back on 8003.

### Chat behavior that tests cover

- `list adapters`, `status`, `help`, `cancel`.
- A leading `@handle` is stripped. `@dapta-lahacks` alone returns help and does
  not call the router.
- Every other message is classified by `POST /v1/route` as `existing_adapter`,
  `create_adapter`, or `unsupported`. The user must reply `confirm`.
- Text execution calls `POST /v1/agent/tools/run_comparison`, polls
  `GET /v1/runs/{id}` for up to 180 seconds, and prints Qwen LoRA, Claude, and
  GPT. It does not call the Qwen worker directly.
- An image is kept in process memory (`LoraFactoryOrchestrator.pending_images`)
  until confirm, cancel, or execution. It is not written to
  `artifacts/asi1-agent.sqlite3`.
- The image is sent to the worker only when the selected adapter metadata has
  `accepts_image: true`. Otherwise the image is dropped and the text comparison
  path is used.
- Image execution calls `POST {DAPTA_QWEN_URL}/generate` with `image_url`.
- An image-only Agentverse message is sent as
  `Extract the fields from this receipt image.`
- New LoRA builds stay off unless `DAPTA_ASI1_ENABLE_FACTORY_BUILDS=true` and a
  one-time code from `./asi1/training-access.sh` is supplied. The code is stored
  only as a SHA-256 digest. One code claims one job. Caps remain 10 examples and
  5 minutes. Do not enable public builds.

### Registry and worker

`_seed_demo_data` in `backend/dapta/api_v2.py` now registers two adapters when
`DAPTA_SEED_DEMO_DATA=true`:

| adapter_id | skill | input | evidence |
| --- | --- | --- | --- |
| `nda-intake/qwen3-vl-8b-v1` | `nda-intake` | text | measured Kleister benchmark, do not change the numbers |
| `receipt-to-expense/qwen3-vl-8b-cord-v1` | `receipt-to-expense` | one JPEG, PNG, or WebP | imported CORD LoRA, **not** a frozen DAPTA benchmark |

Receipt metadata includes `accepts_image: true`. Skill text is
`backend/dapta/skills/receipt-cord-extractor/SKILL.md`. Artifact URI:
`hf://adityave/dapta-lora-adapters/receipt-to-expense/qwen3-vl-8b-cord-v1`.
The source weights are the Apache-2.0 adapter
`steven0226/vlm-receipt-extractor` at revision
`ecd48f99ccf61b39074883d88bb21bb52cc4ccba`. Declared upstream base is
`unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`. DAPTA loads it on
`Qwen/Qwen3-VL-8B-Instruct`. A direct GPU smoke on CORD test image 12 returned
valid JSON in 7.1s; one item name was misread (`RLUS WANITA` vs `BLUS WANITA`).
Do not describe that as a completed benchmark.

`POST /generate` on the vLLM backend (`DAPTA_QWEN_BACKEND=vllm`, the production
default) now downloads the HTTPS image, checks type and size, and sends vLLM a
`data:image/jpeg;base64,...` URL. The transformers rollback path still downloads
inside `ModelRuntime.generate`. Non-receipt adapters reject `image_url` with
HTTP 422.

### Files touched for this

New under `asi1/`: agent, client, orchestrator, session store, validation,
training-access helper, smoke expense script, start/stop, tests.
Edited: `backend/dapta/api_v2.py`, `backend/dapta/model_server.py`,
`backend/tests/test_api.py`, `backend/tests/test_model_server.py`,
`backend/tests/test_vllm_backend.py`, plus the receipt skill above.

### Tests that passed locally

```bash
PYTHONPATH=. python3 -m unittest discover -s asi1/tests -v
PYTHONPATH=backend .venv/bin/python -m unittest \
  backend.tests.test_vllm_backend \
  backend.tests.test_model_server \
  backend.tests.test_api.ApiTests.test_v1_seed_registry_auth_and_agent_tools
```

44 ASI tests and the worker/registry tests above passed on October 18, 2026.
Use `.venv/bin/python` for backend tests. System `python3` does not have FastAPI.

### How to try ASI:One after deploy

Text:

```text
@dapta-lahacks Extract fields from an NDA
```

Reply `confirm`, then:

```text
Document: This Mutual NDA is effective as of March 7, 2025, between Northwind Labs LLC and Contoso Research Inc. This agreement is governed by the laws of the State of California. The term is 2 years.
```

Expected comparison: Qwen LoRA, Claude, and GPT on
`nda-intake/qwen3-vl-8b-v1`. Use synthetic text only.

Receipt: attach one public HTTPS JPEG, PNG, or WebP, say it is a receipt, reply
`confirm`. The held image should run the CORD adapter. Do not send a real
personal receipt.

### Not deployed

SSH to the last known pod `root@69.30.85.12` port `22029` with
`.runpod_dapta_ed25519` returned **connection refused** on October 18, 2026,
about 11:15 PM PT. The key is gitignored. Do not commit it. Resolve the current
pod from the RunPod console or the ignored local `.env` / `runtime.env`. Do not
hard-code the address into Git.

Until the controller and `./asi1/start.sh` are restarted from this working tree,
`@dapta-lahacks` is still the previously deployed agent. The live registry will
not show the receipt adapter until that controller restart with
`DAPTA_SEED_DEMO_DATA=true`.

Chat still cannot draw the website's charts, stage timeline, or adapter cards.
Say that plainly if asked for a pixel clone.

## Diagnosed, not fixed: Base44 website

Live app: <https://dapta-bf7a516c.base44.app> (private, project
`6ad441d282a59638bf7a516c`).

On October 18, 2026, about 11:09 PM PT the page rendered, and the banner said
**`user worker threw an exception`**. Four `POST`s to
`https://base44.app/api/apps/6ad441d282a59638bf7a516c/functions/dapta-proxy`
returned that failure. The UI then showed **Prior adapters 0** because
`/v1/adapters` never arrived. Sentry org `dapta-hl` (`https://us.sentry.io`)
had no new frontend issue for this crash. The newest unresolved issues were
hours older (`DAPTA-B` polling, factory `ValueError` / `RuntimeError` /
`UnboundLocalError`, OpenRouter errors). They are not this banner.

The phrase is a Base44 function-worker failure, not a string in
`frontend/base44/functions/dapta-proxy/entry.ts`. Likely causes to check, in
order:

1. Base44 function logs for `dapta-proxy` (the response body was not captured;
   a follow-up `fetch` from the page failed with `TypeError: Failed to fetch`).
2. `secrets.get("DAPTA_CONTROLLER_URL")` and `DAPTA_CONTROL_TOKEN`. If either is
   missing, the function is supposed to return the NDA fallback or HTTP 503
   `"Dapta controller is not configured"`, not an uncaught exception. An
   uncaught throw means something else blew up: `base44.auth.me()`,
   `mirrorSnapshots` only runs after `upstream.ok`, or the Deno worker failed
   before the handler returned.
3. The controller URL from the pod. If the pod is down, `fetch` to it should
   still become a handled error unless the worker itself crashes.

Do not put the controller token in the browser bundle. Base44 secrets only.

## Not done: directory, switching, RunPod login

The user asked for this and then asked for ASI first. Do this on `Depta-dev`
after ASI is deployed, unless they say otherwise.

### Catalog

Use the five tasks that already exist. Do not invent a sixth task or fake
measured accuracy.

| task | source today | evidence |
| --- | --- | --- |
| NDA intake | `backend/dapta/skills/nda-intake/SKILL.md`, seeded adapter | measured |
| CSV to dashboard | factory job `31d6e12e-3da0-4c85-8bf5-e080a809db98`, benchmark `needs_repair` | weights exist, scores were truncated at 768 tokens |
| Receipt to expense | seeded in this working tree; research skill also under `research/adapter-assets/skills/receipt-to-expense/` | imported LoRA, one smoke image, not a frozen benchmark |
| Three-statement model | `research/adapter-assets/skills/three-statement-model/SKILL.md` | skill only |
| Document redaction | `research/adapter-assets/skills/reference/doc-redaction-app/SKILL.md` | skill only |

`frontend/src/App.tsx` also has illustrative rows for "Support Code Router" and
"CSV Cleaner". Those are labeled demo projections. Do not put them in the
registry as real adapters.

Six base models, one of which is live:

1. `Qwen/Qwen3-VL-8B-Instruct` — production base. Vision. The only one that has run.
2. `Qwen/Qwen2.5-VL-7B-Instruct` — proposed vision.
3. `Qwen/Qwen2.5-7B-Instruct` — proposed text.
4. `meta-llama/Llama-3.1-8B-Instruct` — proposed text.
5. `google/gemma-2-9b-it` — proposed text.
6. `microsoft/Phi-3.5-vision-instruct` — proposed vision.

30 directory slots = 5 tasks × 6 bases. Only two slots may be marked runnable
today: NDA on Qwen3-VL, and the imported receipt LoRA on Qwen3-VL. The other 28
are catalog slots with `evidence_status: "catalog"` until a training run
publishes weights. Do not copy NDA accuracy onto them.

### Switching

Today `WorkflowRouter` (`backend/dapta/routing.py`) picks one adapter. JEV is
first; the fallback is `SequenceMatcher` plus keyword hits, threshold 0.55.
The web comparison then always runs three conditions (`qwen_lora`, `frontier_a`,
`frontier_b`) for that one adapter.

Change the route result so a task can list the adapters for that `skill_id`,
mark one as the default, and let the user pick another before `run_comparison`
or image execute. Default rank, only among slots that have real weights:

- measured evidence before catalog
- then higher accuracy
- then lower latency
- then lower cost

Label the score as a ranking rule, not as a new benchmark. ASI chat needs the
same choice: after `confirm`, if more than one runnable adapter exists for that
skill, say which one is selected and accept `use <adapter-id>` before the
document or image.

UI today: `AdapterDrawer` in `frontend/src/Conversation.tsx` is a flat list.
The website only shows adapters returned by `GET /v1/adapters`. Base44's
`registryFallback` in `frontend/base44/functions/dapta-proxy/entry.ts` still
returns only the NDA adapter when controller secrets are missing. If the
directory grows, update that fallback or the empty-secret case will keep hiding
every other specialist.

### RunPod login

There is no user RunPod login. Training uses the one shared pod from
`RUNPOD_*` in server env (`backend/dapta/factory_runtime.py`,
`deploy/runpod-start.sh`). The user wants a person to connect **their** RunPod
account so training uses their GPU.

Implement it server-side:

- A signed-in user submits a RunPod API key (and then picks a GPU pod).
- The controller checks the key with RunPod (`myself { id }` or the current
  REST equivalent). Do not log the key.
- Store it in the controller database, not in the browser bundle, not in a
  Base44 client, and not in git.
- Return only a connected/not-connected status and a masked hint.
- A factory job for that user uses their pod. The shared A40 remains the
  default when they are not connected.
- Keep the existing budget and time caps. Connecting a GPU does not by itself
  enable public ASI builds.

`frontend/` has no settings screen for this. Add the control next to the
factory build flow, and a short ASI command such as `connect runpod` that
tells the user to paste the key in the web app rather than in Agentverse chat.
Do not ask the public agent to store a raw RunPod key.

## Suggested order for Claude

1. Do not commit unless the user asks. Do not `git stash drop`. Do not push
   `Depta-dev` unless asked.
2. Re-read `docs/AGENT_HANDOFF.md` before any live restart.
3. Find the current RunPod address. If SSH works, deploy this working tree,
   restart the controller with seed data on, restart `./asi1/start.sh` on port
   8004, and run one synthetic NDA comparison plus one public receipt image
   through `@dapta-lahacks`.
4. Open Base44 function logs and fix `user worker threw an exception`. Confirm
   the banner is gone and Prior adapters shows NDA and the receipt specialist.
5. Add the catalog, the per-task switch, and the RunPod connect flow described
   above. Keep catalog rows visibly distinct from measured rows.
6. Extend ASI so the switch works in chat. Re-run the ASI unittest command and
   the backend tests listed above.

## Constraints

- Synthetic documents only in chat and in issues. No real NDAs or personal
  receipts.
- Do not invent benchmark percentages. Illustrative UI numbers in `App.tsx`
  must stay labeled illustrative.
- Do not enable `DAPTA_ASI1_ENABLE_FACTORY_BUILDS` on the public agent except
  through `./asi1/training-access.sh` for a supervised window.
- Controller token, Qwen token, RunPod API key, Hugging Face token, and
  `runtime.env` stay off the client and out of git.
- The shared GPU cannot run inference and training at the same time. Training
  unloads the worker and reloads it in `finally`.
