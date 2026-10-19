from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from .dapta_client import DaptaClient
from .orchestrator import LoraFactoryOrchestrator
from .session_store import SessionStore


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _optional_float(name: str) -> float | None:
    value = os.environ.get(name, "").strip()
    return float(value) if value else None


def extract_image_url(item: Any) -> str | None:
    resources = item.resource if isinstance(item.resource, list) else [item.resource]
    for resource in resources:
        candidates = [getattr(resource, "uri", None)]
        metadata = getattr(resource, "metadata", None) or {}
        if isinstance(metadata, dict):
            candidates.extend(
                metadata.get(key) for key in ("url", "uri", "source", "image_url")
            )
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            parsed = urlparse(candidate)
            if parsed.scheme == "https" and parsed.netloc:
                return candidate
    return None


def build_orchestrator() -> LoraFactoryOrchestrator:
    required = (
        "DAPTA_AGENT_SEED",
        "DAPTA_CONTROLLER_URL",
        "DAPTA_CONTROL_TOKEN",
        "DAPTA_QWEN_URL",
        "DAPTA_QWEN_TOKEN",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("Missing required configuration: " + ", ".join(missing))
    client = DaptaClient(
        controller_url=os.environ["DAPTA_CONTROLLER_URL"],
        controller_token=os.environ["DAPTA_CONTROL_TOKEN"],
        qwen_url=os.environ["DAPTA_QWEN_URL"],
        qwen_token=os.environ["DAPTA_QWEN_TOKEN"],
    )
    session_path = Path(os.environ.get("DAPTA_ASI1_STATE_PATH", "artifacts/asi1-agent.sqlite3"))
    session_path.parent.mkdir(parents=True, exist_ok=True)
    return LoraFactoryOrchestrator(
        client,
        SessionStore(session_path),
        default_adapter_id=os.environ.get("DAPTA_DEFAULT_ADAPTER_ID") or None,
        factory_builds_enabled=_enabled("DAPTA_ASI1_ENABLE_FACTORY_BUILDS"),
        factory_build_code_sha256=(
            os.environ.get("DAPTA_ASI1_FACTORY_BUILD_CODE_SHA256") or None
        ),
        factory_build_expires_at=_optional_float(
            "DAPTA_ASI1_FACTORY_BUILD_EXPIRES_AT"
        ),
    )


def main() -> None:
    # Imported lazily so unit tests for the orchestration layer do not need the
    # Agentverse runtime installed in DAPTA's existing backend environment.
    from uagents import Agent, Context, Protocol
    from uagents_core.contrib.protocols.chat import (
        ChatAcknowledgement,
        ChatMessage,
        MetadataContent,
        ResourceContent,
        StartSessionContent,
        TextContent,
        chat_protocol_spec,
    )

    orchestrator = build_orchestrator()
    agent = Agent(
        name=os.environ.get("DAPTA_AGENT_NAME", "dapta-lora-factory"),
        seed=os.environ["DAPTA_AGENT_SEED"],
        port=int(os.environ.get("DAPTA_AGENT_PORT", "8004")),
        mailbox=True,
        publish_agent_details=True,
    )
    chat = Protocol(spec=chat_protocol_spec)

    @chat.on_message(ChatMessage)
    async def handle_message(ctx: Context, sender: str, message: ChatMessage) -> None:
        await ctx.send(
            sender,
            ChatAcknowledgement(
                timestamp=datetime.now(timezone.utc),
                acknowledged_msg_id=message.msg_id,
            ),
        )
        if any(isinstance(item, StartSessionContent) for item in message.content):
            await ctx.send(
                sender,
                ChatMessage(
                    timestamp=datetime.now(timezone.utc),
                    msg_id=uuid4(),
                    content=[
                        MetadataContent(
                            type="metadata", metadata={"attachments": "true"}
                        )
                    ],
                ),
            )
        text = message.text() if hasattr(message, "text") else " ".join(
            item.text for item in message.content if isinstance(item, TextContent)
        )
        image_urls = [
            url
            for item in message.content
            if isinstance(item, ResourceContent)
            for url in [extract_image_url(item)]
            if url
        ]
        if len(image_urls) > 1:
            replies = ["Send exactly one receipt image at a time."]
        elif any(
            isinstance(item, ResourceContent) for item in message.content
        ) and not image_urls:
            replies = ["The attachment did not contain a usable HTTPS image URL."]
        elif not text.strip() and not image_urls:
            return
        else:
            replies = await asyncio.to_thread(
                orchestrator.handle,
                sender,
                text or "Extract the fields from this receipt image.",
                image_url=image_urls[0] if image_urls else None,
            )
        for reply in replies:
            await ctx.send(
                sender,
                ChatMessage(
                    timestamp=datetime.now(timezone.utc),
                    msg_id=uuid4(),
                    content=[TextContent(type="text", text=reply)],
                ),
            )

    @chat.on_message(ChatAcknowledgement)
    async def handle_acknowledgement(
        ctx: Context, sender: str, message: ChatAcknowledgement
    ) -> None:
        del ctx, sender, message

    agent.include(chat, publish_manifest=True)
    agent.run()


if __name__ == "__main__":
    main()
