# DAPTA Adapter Factory for ASI:One

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

This companion registers one public **DAPTA Adapter Factory** agent on Agentverse. It uses
the Agent Chat Protocol to expose DAPTA's existing adapter factory and private Qwen LoRA
worker without changing or publicly exposing either service.

## Capabilities

- list registered DAPTA adapters;
- classify any request as `existing_adapter`, `create_adapter`, or `unsupported`;
- route a stable workflow to an existing adapter or a new build;
- require explicit confirmation before starting a paid adapter build;
- report real factory job state;
- execute a ready Qwen LoRA adapter on privacy-safe input; and
- for text documents, run the same three-model comparison as the DAPTA web app
  (Qwen LoRA, Claude, and GPT).

Kimi, OpenRouter, JEV, Devin, RunPod training, evaluation, Hugging Face publishing, the
SQLite ledger and Sentry remain internal DAPTA services. They are not separate Agentverse
agents.

## Local verification

The orchestration tests use fake services and do not spend provider credits:

```bash
PYTHONPATH=. python -m unittest discover -s asi1/tests -v
```

## Run beside DAPTA

Use a separate environment so the existing backend dependency set stays unchanged:

```bash
python3.11 -m venv asi1/.venv
asi1/.venv/bin/pip install -r asi1/requirements.txt
cp asi1/.env.example asi1/.env
set -a
source asi1/.env
set +a
asi1/.venv/bin/python -m asi1.agent
```

On RunPod, the companion can be managed independently from the existing controller and
worker:

```bash
./asi1/start.sh
./asi1/stop.sh
```

Generate one unique seed, keep it private and never change it after registration. Connect
the Inspector URL printed at startup to Agentverse using **Mailbox**, then configure the
name, handle and Innovation Lab category. Keep this process running during the demo.
When it runs on RunPod, forward port `8004` over SSH before opening the Inspector URL on
your computer; registration through a localhost Inspector is not directly reachable from
the remote pod. Confirm the profile publishes `AgentChatProtocol` and is marked active.

`DAPTA_ASI1_ENABLE_FACTORY_BUILDS` defaults to `false`, so the public agent can execute
ready adapters without allowing arbitrary users to spend model or GPU credits. Use the
one-shot access helper for a supervised smoke build. It generates an expiring code,
allows that code to claim one idempotent factory job, caps that job at 10 examples and
5 minutes of LoRA training, and restarts only the companion:

```bash
./asi1/training-access.sh enable --ttl-seconds 600
./asi1/training-access.sh disable
```

`disable` is the kill switch. Run it from the repository root on the machine that hosts
the agent. It turns builds off, deletes the one-time code, deletes the controller access
file, clears pending build confirmations, and restarts only the ASI companion. After that,
the public agent cannot start another build, and the controller rejects `create_adapter`
even if the companion has not restarted yet. A build that was already created keeps the
10-example, 5-minute cap and continues independently in DAPTA.

## Safe demo

Use synthetic document text only. A reliable sequence is:

1. `list adapters`
2. Describe a workflow or test case and verify the reported category.
3. Reply `confirm`.
4. For `existing_adapter`, send `Document: <synthetic input>`.
5. During a supervised build window, send
   `Access code <one-time-code>. Create a LoRA adapter that ...`.
6. Verify `create_adapter`, then reply `confirm` after accepting the paid
   compute/model cost.
7. Use `status <job-id>` to follow the real factory state.

The primary qualifying workflow should execute an already-ready adapter completely inside
ASI:One. A new build is a secondary workflow unless it genuinely reaches `ready` during the
same conversation.
