"""Per-user RunPod GPU connection: the "bring your own GPU" login.

A signed-in Dapta user can connect their own RunPod account so a future
factory job can provision training on their GPU instead of the shared one.
This module owns validation and storage only; it does not provision anything
itself, and nothing here enables public factory builds on its own (see
`factory_limits.training_access_open`).

RunPod's v2 API documents no dedicated "whoami" endpoint, so the key is
verified with the cheapest side-effect-free call that still proves it: listing
the user's own pods (`GET /v2/pods`), capped to one result. A 401 means the
key is wrong; any other 2xx means it's real. The raw key is never logged, and
is returned to the caller only once, by the browser that just typed it, inside
the same request that stored it being echoed back as a masked hint — after
that, only the masked hint and a connected/not-connected flag ever leave this
module over the API.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

RUNPOD_API_BASE = "https://api.runpod.io/v2"
_KEY_SHAPE = re.compile(r"^[A-Za-z0-9_-]{10,256}$")


class RunpodConnectionError(RuntimeError):
    """RunPod could not be reached, or returned something other than 401/2xx."""


class InvalidRunpodKey(RunpodConnectionError):
    """RunPod rejected the key, or it doesn't look like one."""


def _mask(api_key: str) -> str:
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:4]}…{api_key[-4:]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_runpod_key(api_key: str, *, client: httpx.Client | None = None) -> None:
    """Raise InvalidRunpodKey/RunpodConnectionError if the key doesn't check out."""
    if not _KEY_SHAPE.match(api_key):
        raise InvalidRunpodKey("that doesn't look like a Runpod API key")
    owns_client = client is None
    http_client = client or httpx.Client(timeout=15.0)
    try:
        response = http_client.get(
            f"{RUNPOD_API_BASE}/pods",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"limit": 1},
        )
    except httpx.HTTPError as error:
        raise RunpodConnectionError("could not reach Runpod to verify the key") from error
    finally:
        if owns_client:
            http_client.close()
    if response.status_code == 401:
        raise InvalidRunpodKey("Runpod rejected this API key")
    if response.status_code >= 400:
        raise RunpodConnectionError(f"Runpod returned {response.status_code} while verifying the key")


class RunpodConnectionStore:
    """One RunPod connection per Dapta user, in the same sqlite file as the job ledger."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_runpod_connections (
                    user_id TEXT PRIMARY KEY,
                    api_key TEXT NOT NULL,
                    masked_hint TEXT NOT NULL,
                    connected_at TEXT NOT NULL,
                    last_verified_at TEXT NOT NULL
                )
                """
            )

    def connect(
        self, user_id: str, api_key: str, *, client: httpx.Client | None = None
    ) -> dict[str, Any]:
        if not user_id:
            raise ValueError("user_id is required")
        verify_runpod_key(api_key, client=client)
        now = _now()
        masked = _mask(api_key)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_runpod_connections(
                    user_id, api_key, masked_hint, connected_at, last_verified_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    api_key=excluded.api_key,
                    masked_hint=excluded.masked_hint,
                    last_verified_at=excluded.last_verified_at
                """,
                (user_id, api_key, masked, now, now),
            )
            connection.commit()
        return {"connected": True, "masked_hint": masked, "connected_at": now}

    def status(self, user_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT masked_hint, connected_at FROM user_runpod_connections WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return {"connected": False}
        return {"connected": True, "masked_hint": row["masked_hint"], "connected_at": row["connected_at"]}

    def disconnect(self, user_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM user_runpod_connections WHERE user_id = ?", (user_id,)
            )
            connection.commit()
        return cursor.rowcount > 0

    def get_api_key(self, user_id: str) -> str | None:
        """Internal use only, for a future job to provision on the user's own GPU.

        Never exposed over the HTTP API -- only `status()`'s masked hint is.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT api_key FROM user_runpod_connections WHERE user_id = ?", (user_id,)
            ).fetchone()
        return row["api_key"] if row else None
