from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
import uuid
from typing import Any, Protocol

from .dapta_client import DaptaClientError
from .session_store import AgentSession, SessionStore
from .validation import validate_output


JOB_ID = re.compile(
    r"\b(?:job-)?[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
LEADING_AGENT_MENTION = re.compile(
    r"^@[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\b[\s,:-]*"
)
ADAPTER_ID = re.compile(r"\badapter\s+([A-Za-z0-9._/-]{3,300})", re.IGNORECASE)
DOCUMENT_MARKER = re.compile(r"\b(?:document|input|receipt|invoice)\s*:\s*", re.IGNORECASE)
ACCESS_CODE = re.compile(
    r"\baccess(?:\s+code)?\s*[:=]?\s*([A-Za-z0-9_-]{16,128})\b",
    re.IGNORECASE,
)
MAX_MESSAGE_CHARS = 500_000
MAX_WORKFLOW_CHARS = 20_000
MIN_WORKFLOW_CHARS = 3  # matches the controller's RouteRequest.workflow min_length
COMPARISON_POLL_SECONDS = 180
COMPARISON_POLL_INTERVAL = 1.5
CONDITION_LABELS = {
    "qwen_lora": "Qwen LoRA",
    "frontier_a": "Claude",
    "frontier_b": "GPT",
}


class FactoryClient(Protocol):
    def health(self) -> dict[str, Any]: ...
    def list_adapters(self) -> list[dict[str, Any]]: ...
    def route_workflow(self, workflow: str) -> dict[str, Any]: ...
    def create_adapter(self, *, workflow: str, idempotency_key: str,
                       skill_id: str | None = None) -> dict[str, Any]: ...
    def get_job_status(self, job_id: str) -> dict[str, Any]: ...
    def run_comparison(self, *, workflow: str, document_text: str, adapter_id: str,
                       idempotency_key: str, attachment_name: str | None = None
                       ) -> dict[str, Any]: ...
    def get_run(self, run_id: str) -> dict[str, Any]: ...
    def execute_adapter(self, *, run_id: str, workflow: str, document_text: str,
                        adapter_id: str, image_url: str | None = None) -> dict[str, Any]: ...


class LoraFactoryOrchestrator:
    CATEGORY_LABELS = {
        "existing_adapter": "run an existing specialist",
        "create_adapter": "create a new LoRA",
        "unsupported": "not a specialist task",
    }

    def __init__(self, client: FactoryClient, sessions: SessionStore,
                 *, default_adapter_id: str | None = None,
                 factory_builds_enabled: bool = False,
                 factory_build_code_sha256: str | None = None,
                 factory_build_expires_at: float | None = None,
                 sleeper: Any = time.sleep):
        self.client = client
        self.sessions = sessions
        self.default_adapter_id = default_adapter_id
        self.factory_builds_enabled = factory_builds_enabled
        self.factory_build_code_sha256 = factory_build_code_sha256
        self.factory_build_expires_at = factory_build_expires_at
        self.sleeper = sleeper
        # Image URLs stay in memory for the next confirmation only. They are not written
        # to the session database.
        self.pending_images: dict[str, str] = {}

    def handle(
        self, sender: str, text: str, *, image_url: str | None = None
    ) -> list[str]:
        message = text.strip()
        if not message:
            return ["Please describe a workflow, request adapter status, or provide a document to process."]
        message = LEADING_AGENT_MENTION.sub("", message, count=1).strip()
        if not message:
            return [self._help()]
        if len(message) > MAX_MESSAGE_CHARS:
            return ["The message is too large. Send no more than 500,000 characters."]
        session = self.sessions.get(sender)
        if image_url:
            self.pending_images[sender] = image_url
        elif not self._keeps_attachment(session):
            self.pending_images.pop(sender, None)
        try:
            lowered = message.lower()
            if self._is_connect_runpod(lowered):
                return [self._connect_runpod_instructions()]
            if lowered in {"cancel", "cancel build", "never mind", "nevermind"}:
                return [self._cancel(session)]
            if (
                isinstance(session.pending_action, str)
                and session.pending_action.startswith("confirm:")
                and self._is_confirmation(message)
            ):
                return [self._confirm_category(session, message, image_url=image_url)]
            if session.pending_action == "confirm_create" and self._is_confirmation(lowered):
                return [self._confirm_create(session)]
            if self._is_confirmation(message):
                return ["There is no categorized action awaiting confirmation."]
            if session.pending_action == "await_document":
                marker = DOCUMENT_MARKER.search(message)
                document_text = message[marker.end():].strip() if marker else message
                return [self._execute(session, document_text, image_url=image_url)]
            if self._is_help(lowered):
                return [self._help()]
            if self._is_list(lowered):
                return [self._list_adapters(session)]
            if self._is_status(lowered):
                return [self._status(session, message)]
            if self._is_explicit_run(lowered):
                return [self._prepare_run(session, message, image_url=image_url)]
            return [self._classify(session, message)]
        except DaptaClientError as error:
            return [f"The DAPTA operation could not be completed: {error}"]

    def _classify(self, session: AgentSession, message: str) -> str:
        access_match = ACCESS_CODE.search(message)
        workflow = ACCESS_CODE.sub("", message, count=1).strip(" \t\r\n.,;:-")
        if not workflow:
            return "Please describe the workflow or test case you want DAPTA to handle."

        marker = DOCUMENT_MARKER.search(workflow)
        routing_workflow = workflow[:marker.start()].strip() if marker else workflow
        if not routing_workflow:
            return (
                "Describe the workflow first, then send the test case as "
                "`Document: ...` after confirmation."
            )
        if len(routing_workflow) > MAX_WORKFLOW_CHARS:
            return "The workflow description must be no more than 20,000 characters."
        if len(routing_workflow) < MIN_WORKFLOW_CHARS:
            return (
                "There is no categorized action awaiting confirmation. "
                "Describe a workflow (at least 3 characters) or say `help`."
            )

        decision = self.client.route_workflow(routing_workflow)
        category = str(decision.get("decision") or "unsupported")
        if category not in self.CATEGORY_LABELS:
            category = "unsupported"

        session.pending_action = f"confirm:{category}"
        session.pending_workflow = routing_workflow
        session.idempotency_key = None
        session.build_access_hash = None
        if category == "existing_adapter" and decision.get("adapter_id"):
            session.adapter_id = str(decision["adapter_id"])
        elif category != "existing_adapter":
            session.adapter_id = None
            self.pending_images.pop(session.sender, None)

        access_note = ""
        if category == "create_adapter":
            if access_match:
                supplied_hash = hashlib.sha256(
                    access_match.group(1).encode("utf-8")
                ).hexdigest()
                if hmac.compare_digest(
                    supplied_hash, self.factory_build_code_sha256 or ""
                ):
                    session.build_access_hash = supplied_hash
                else:
                    access_note = (
                        " The supplied access code is invalid; include a valid code "
                        "with your confirmation."
                    )
            elif self._active_access_error() is None:
                access_note = (
                    " Include the one-time access code with your confirmation."
                )

        self.sessions.save(session)

        detail = (
            f" Adapter: `{session.adapter_id}`."
            if category == "existing_adapter" and session.adapter_id
            else ""
        )
        limit = (
            " A new LoRA uses at most 10 examples and stops training after 5 minutes."
            if category == "create_adapter"
            else ""
        )
        held_image = self.pending_images.get(session.sender)
        privacy_note = (
            " For privacy, resend the test case as `Document: ...` after confirmation."
            if marker and category == "existing_adapter"
            else ""
        )
        if held_image and category == "existing_adapter":
            privacy_note += " The attached image is held until you confirm or cancel."
        reason = str(decision.get("reason") or "").strip()
        reason_line = f"\n{reason}" if reason else ""
        return (
            f"Category `{category}`: {self.CATEGORY_LABELS[category]}."
            f"{detail}{limit}{access_note}{privacy_note}{reason_line}\n\n"
            "Reply `confirm` to proceed, or `cancel`."
        )

    def _confirm_category(
        self, session: AgentSession, message: str, *, image_url: str | None = None
    ) -> str:
        category = str(session.pending_action or "").split(":", 1)[-1]
        if category == "unsupported":
            self._clear_pending_create(session)
            return (
                "Confirmed: this is not a specialist task. "
                "No adapter was run and no LoRA training was started."
            )
        if category == "existing_adapter":
            adapter_id = session.adapter_id
            if not adapter_id:
                self._clear_pending_create(session)
                return "DAPTA did not name an adapter for this request. Say `list adapters`."
            return self._prepare_run(
                session,
                f"run adapter {adapter_id}",
                image_url=image_url or self.pending_images.get(session.sender),
            )

        access_match = ACCESS_CODE.search(message)
        if access_match:
            supplied_hash = hashlib.sha256(
                access_match.group(1).encode("utf-8")
            ).hexdigest()
            if hmac.compare_digest(
                supplied_hash, self.factory_build_code_sha256 or ""
            ):
                session.build_access_hash = supplied_hash
            else:
                return (
                    "A valid one-time access code is required for this "
                    "controlled build window."
                )
        access_error = self._active_access_error()
        if access_error:
            self._clear_pending_create(session)
            return access_error
        if not session.build_access_hash:
            return (
                "Reply `confirm access code <code>` using the one-time code for "
                "this controlled build window."
            )
        session.pending_action = "confirm_create"
        session.idempotency_key = f"asi1-{uuid.uuid4()}"
        self.sessions.save(session)
        return self._confirm_create(session)

    def _confirm_create(self, session: AgentSession) -> str:
        if not session.pending_workflow or not session.idempotency_key:
            return "There is no adapter build awaiting confirmation."
        access_error = self._active_access_error()
        access_hash = session.build_access_hash or ""
        if access_error or not hmac.compare_digest(
            access_hash, self.factory_build_code_sha256 or ""
        ):
            self._clear_pending_create(session)
            return access_error or "This build authorization is no longer valid."

        readiness = self.client.health()
        required = {
            "factory_autostart": "factory runner",
            "openrouter_configured": "OpenRouter/JEV",
            "devin_configured": "Devin integration",
            "training_configured": "GPU training",
            "publishing_configured": "adapter publishing",
        }
        unavailable = [label for key, label in required.items() if not readiness.get(key)]
        if unavailable:
            return (
                "DAPTA is not ready to start the build. Missing or inactive: "
                + ", ".join(unavailable)
                + ". No factory job was created."
            )
        if not self.sessions.claim_build(
            access_hash=access_hash,
            sender=session.sender,
            idempotency_key=session.idempotency_key,
        ):
            self._clear_pending_create(session)
            return "This one-time build authorization has already been used."
        job = self.client.create_adapter(
            workflow=session.pending_workflow,
            idempotency_key=session.idempotency_key,
        )
        session.job_id = str(job.get("job_id") or "") or None
        session.pending_action = None
        session.idempotency_key = None
        session.build_access_hash = None
        self.sessions.save(session)
        if not session.job_id:
            raise DaptaClientError("DAPTA did not return a factory job ID")
        status = str(job.get("status") or "queued")
        self.sessions.record_build_job(access_hash=access_hash, job_id=session.job_id)
        return (
            f"Adapter factory job `{session.job_id}` was created with status `{status}`. "
            "Ask `status` at any time. I will not claim the adapter is ready until DAPTA reports it."
        )

    def _status(self, session: AgentSession, message: str) -> str:
        match = JOB_ID.search(message)
        job_id = match.group(0) if match else session.job_id
        if not job_id:
            return "I do not have a factory job for this conversation yet."
        job = self.client.get_job_status(job_id)
        session.job_id = job_id
        if job.get("adapter_id"):
            session.adapter_id = str(job["adapter_id"])
        self.sessions.save(session)
        status = str(job.get("status") or "unknown")
        progress = job.get("progress") if isinstance(job.get("progress"), dict) else {}
        percent = progress.get("percent")
        detail = f", {percent}% complete" if isinstance(percent, (int, float)) else ""
        response = f"Factory job `{job_id}` is `{status}`{detail}."
        if status == "ready" and session.adapter_id:
            response += f" Ready adapter: `{session.adapter_id}`."
        if status == "failed":
            failure_message = str(job.get("failure_message") or "")
            if "adapter failed evaluation" in failure_message:
                response += (
                    " Training completed, but the adapter did not pass the quality "
                    "gate and was not published."
                )
            else:
                response += " The build failed; inspect the DAPTA job evidence before retrying."
        return response

    def _list_adapters(self, session: AgentSession) -> str:
        adapters = self.client.list_adapters()
        if not adapters:
            return "DAPTA has no registered adapters."
        lines = []
        for adapter in adapters[:10]:
            adapter_id = str(adapter.get("adapter_id") or "unknown")
            version = str(adapter.get("version") or "unversioned")
            metadata = adapter.get("metadata") if isinstance(adapter.get("metadata"), dict) else {}
            name = str(metadata.get("name") or adapter.get("skill_id") or adapter_id)
            description = str(metadata.get("description") or "").strip()
            status = str(metadata.get("status") or metadata.get("benchmark_status") or "ready")
            line = f"- **{name}** `{adapter_id}` ({version}, {status})"
            if description:
                line += f"\n  {description}"
            lines.append(line)
        if len(adapters) == 1:
            session.adapter_id = str(adapters[0].get("adapter_id") or "") or None
            self.sessions.save(session)
        return "Registered DAPTA adapters:\n" + "\n".join(lines)

    def _prepare_run(
        self, session: AgentSession, message: str, *, image_url: str | None = None
    ) -> str:
        adapter_match = ADAPTER_ID.search(message)
        if adapter_match and adapter_match.group(1).lower() not in {"on", "with"}:
            session.adapter_id = adapter_match.group(1)
        session.adapter_id = session.adapter_id or self.default_adapter_id
        if not session.adapter_id:
            return "Choose an adapter first by saying `list adapters`, then name the adapter to run."
        adapters = self.client.list_adapters()
        selected = next(
            (
                adapter
                for adapter in adapters
                if str(adapter.get("adapter_id") or "") == session.adapter_id
            ),
            None,
        )
        if selected is None:
            return f"Adapter `{session.adapter_id}` is not registered in DAPTA."
        metadata = selected.get("metadata")
        if isinstance(metadata, dict):
            skill_prompt = metadata.get("skill_prompt")
            if isinstance(skill_prompt, str) and skill_prompt.strip():
                session.pending_workflow = skill_prompt.strip()
        marker = DOCUMENT_MARKER.search(message)
        if marker and message[marker.end():].strip():
            return self._execute(
                session,
                message[marker.end():].strip(),
                image_url=image_url if image_url and self._accepts_image(selected) else None,
            )
        remembered = image_url or self.pending_images.get(session.sender)
        active_image = remembered if remembered and self._accepts_image(selected) else None
        if remembered and active_image is None:
            self.pending_images.pop(session.sender, None)
        if active_image:
            return self._execute(
                session, "[attached receipt image]", image_url=active_image
            )
        session.pending_action = "await_document"
        if not session.pending_workflow:
            session.pending_workflow = "Execute the selected learned workflow and return valid JSON."
        self.sessions.save(session)
        return (
            f"Ready to run `{session.adapter_id}`. Send the synthetic input as `Document: ...`, "
            "or attach one receipt image. Do not send personal or confidential information."
        )

    def _execute(
        self,
        session: AgentSession,
        document_text: str,
        *,
        image_url: str | None = None,
    ) -> str:
        adapter_id = session.adapter_id or self.default_adapter_id
        if not adapter_id:
            session.pending_action = None
            self.sessions.save(session)
            return "No adapter is selected. Say `list adapters` first."
        workflow = session.pending_workflow or "Execute the selected learned workflow."
        if image_url:
            return self._execute_lora(
                session,
                workflow=workflow,
                document_text=document_text,
                adapter_id=adapter_id,
                image_url=image_url,
            )
        run = self.client.run_comparison(
            workflow=workflow,
            document_text=document_text,
            adapter_id=adapter_id,
            idempotency_key=f"asi1-{uuid.uuid4()}",
        )
        run_id = str(run.get("run_id") or "")
        if not run_id:
            raise DaptaClientError("DAPTA did not return a comparison run ID")
        completed = self._wait_for_run(run_id)
        session.pending_action = None
        self.pending_images.pop(session.sender, None)
        self.sessions.save(session)
        return self._render_comparison(completed, adapter_id=adapter_id)

    def _execute_lora(
        self,
        session: AgentSession,
        *,
        workflow: str,
        document_text: str,
        adapter_id: str,
        image_url: str | None = None,
    ) -> str:
        run_id = f"asi1-{uuid.uuid4()}"
        result = self.client.execute_adapter(
            run_id=run_id,
            workflow=workflow,
            document_text=document_text,
            adapter_id=adapter_id,
            image_url=image_url,
        )
        output = result.get("output")
        validation = validate_output(output)
        session.pending_action = None
        self.pending_images.pop(session.sender, None)
        self.sessions.save(session)
        rendered = json.dumps(output, indent=2, ensure_ascii=False)
        if len(rendered) > 8000:
            rendered = rendered[:7997] + "..."
        return (
            f"LoRA execution completed with adapter `{adapter_id}`.\n"
            f"Run ID: `{run_id}`\n"
            f"Latency: {result.get('latency_ms', 'unknown')} ms\n"
            f"Tokens: {result.get('input_tokens', 'unknown')} input / "
            f"{result.get('output_tokens', 'unknown')} output\n"
            f"Validation: `{json.dumps(validation, separators=(',', ':'))}`\n\n"
            f"```json\n{rendered}\n```"
        )

    def _wait_for_run(self, run_id: str) -> dict[str, Any]:
        deadline = time.time() + COMPARISON_POLL_SECONDS
        run = self.client.get_run(run_id)
        while str(run.get("status") or "") not in {"completed", "failed"}:
            if time.time() >= deadline:
                raise DaptaClientError("The comparison is still running. Ask `status` later.")
            self.sleeper(COMPARISON_POLL_INTERVAL)
            run = self.client.get_run(run_id)
        if str(run.get("status") or "") == "failed":
            raise DaptaClientError(str(run.get("error_message") or "comparison failed"))
        return run

    def _render_comparison(self, run: dict[str, Any], *, adapter_id: str) -> str:
        results = run.get("results") if isinstance(run.get("results"), dict) else {}
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        sections = [
            f"Same task. Every model. Adapter `{adapter_id}`.",
            f"Run ID: `{run.get('run_id')}`",
        ]
        for key, label in CONDITION_LABELS.items():
            condition = results.get(key) if isinstance(results.get(key), dict) else {}
            status = str(condition.get("status") or "missing")
            metrics = condition.get("metrics") if isinstance(condition.get("metrics"), dict) else {}
            latency = metrics.get("latency_ms")
            tokens = None
            if metrics.get("input_tokens") is not None and metrics.get("output_tokens") is not None:
                tokens = f"{metrics['input_tokens']} in / {metrics['output_tokens']} out"
            cost = metrics.get("cost_usd")
            header = f"## {label} (`{status}`)"
            extras = []
            if latency is not None:
                extras.append(f"{latency} ms")
            if tokens:
                extras.append(tokens)
            if cost not in (None, ""):
                extras.append(f"${cost}")
            if extras:
                header += "\n" + " · ".join(str(item) for item in extras)
            if condition.get("error"):
                sections.append(f"{header}\n{condition['error']}")
                continue
            output = condition.get("output")
            rendered = json.dumps(output, indent=2, ensure_ascii=False)
            if len(rendered) > 4000:
                rendered = rendered[:3997] + "..."
            sections.append(f"{header}\n```json\n{rendered}\n```")
        best = summary.get("best_overall")
        if best:
            sections.append(f"Best overall: **{CONDITION_LABELS.get(str(best), best)}**.")
        return "\n\n".join(sections)

    def _cancel(self, session: AgentSession) -> str:
        self.pending_images.pop(session.sender, None)
        session.pending_action = None
        session.pending_workflow = None
        session.idempotency_key = None
        session.build_access_hash = None
        self.sessions.save(session)
        return "The pending action was cancelled. No adapter build was started."

    def _active_access_error(self) -> str | None:
        if not self.factory_builds_enabled:
            return (
                "New adapter builds are disabled for this public agent. A DAPTA operator "
                "can temporarily enable them for a controlled build window."
            )
        if not self.factory_build_code_sha256 or self.factory_build_expires_at is None:
            return "The controlled build window is not safely configured. No build was started."
        if time.time() >= self.factory_build_expires_at:
            return "The controlled build window has expired. No build was started."
        return None

    @staticmethod
    def _keeps_attachment(session: AgentSession) -> bool:
        action = session.pending_action or ""
        return action == "await_document" or action.startswith("confirm:")

    @staticmethod
    def _accepts_image(adapter: dict[str, Any]) -> bool:
        metadata = adapter.get("metadata") if isinstance(adapter.get("metadata"), dict) else {}
        return bool(metadata.get("accepts_image"))

    def _clear_pending_create(self, session: AgentSession) -> None:
        self.pending_images.pop(session.sender, None)
        session.pending_action = None
        session.pending_workflow = None
        session.idempotency_key = None
        session.build_access_hash = None
        self.sessions.save(session)

    @staticmethod
    def _is_confirmation(message: str) -> bool:
        lowered = message.strip().lower()
        if lowered in {"confirm", "confirm build", "yes", "yes, confirm", "proceed"}:
            return True
        return lowered.startswith(("confirm access ", "confirm access code "))

    @staticmethod
    def _is_help(message: str) -> bool:
        return message in {"help", "/help", "what can you do?", "what can you do"}

    @staticmethod
    def _is_list(message: str) -> bool:
        return "list" in message and ("adapter" in message or "skill" in message)

    @staticmethod
    def _is_status(message: str) -> bool:
        return "status" in message or bool(JOB_ID.search(message))

    @staticmethod
    def _is_explicit_run(message: str) -> bool:
        return message.startswith("run adapter")

    @staticmethod
    def _is_connect_runpod(message: str) -> bool:
        return "runpod" in message and ("connect" in message or "gpu" in message)

    @staticmethod
    def _connect_runpod_instructions() -> str:
        # Never accept a Runpod key here: Agentverse chat is not a trusted channel for it,
        # and the controller only accepts a key from a signed-in Base44 session (see
        # dapta-proxy/entry.ts and api_v2.py's require_user_id). Point the user there instead.
        return (
            "To connect your own Runpod GPU, open the Dapta web app, sign in, and go to "
            "Skills & Adapters → Bring your own GPU. Paste your Runpod API key there, not "
            "here — Agentverse chat is not a safe place to send it, and I will not store one "
            "sent this way."
        )

    @staticmethod
    def _help() -> str:
        return (
            "I am DAPTA on ASI:One — the same factory as the web app, in chat.\n"
            "JEV routes every request to an existing specialist, a new LoRA build, "
            "or unsupported.\n\n"
            "Try:\n"
            "- `Extract fields from an NDA`\n"
            "- `list adapters`\n"
            "- reply `confirm`, then send `Document: ...` — I will compare "
            "Qwen LoRA, Claude, and GPT on the same input\n"
            "- attach one receipt image for the CORD specialist\n"
            "- `status` to follow a factory job\n"
            "- `run adapter <adapter-id>` followed by `Document: ...`\n"
            "- `connect runpod` for how to use your own GPU"
        )
