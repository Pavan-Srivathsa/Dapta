from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class AgentSession:
    sender: str
    pending_action: str | None = None
    pending_workflow: str | None = None
    idempotency_key: str | None = None
    job_id: str | None = None
    adapter_id: str | None = None
    build_access_hash: str | None = None


class SessionStore:
    """Small persistent state store. Document contents are deliberately excluded."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._initialize()

    def get(self, sender: str) -> AgentSession:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM asi1_sessions WHERE sender = ?", (sender,)
            ).fetchone()
        if row is None:
            return AgentSession(sender=sender)
        return AgentSession(**dict(row))

    def save(self, session: AgentSession) -> None:
        values = asdict(session)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO asi1_sessions(
                    sender, pending_action, pending_workflow, idempotency_key,
                    job_id, adapter_id, build_access_hash
                ) VALUES(
                    :sender, :pending_action, :pending_workflow, :idempotency_key,
                    :job_id, :adapter_id, :build_access_hash
                )
                ON CONFLICT(sender) DO UPDATE SET
                    pending_action = excluded.pending_action,
                    pending_workflow = excluded.pending_workflow,
                    idempotency_key = excluded.idempotency_key,
                    job_id = excluded.job_id,
                    adapter_id = excluded.adapter_id,
                    build_access_hash = excluded.build_access_hash
                """,
                values,
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS asi1_sessions(
                    sender TEXT PRIMARY KEY,
                    pending_action TEXT,
                    pending_workflow TEXT,
                    idempotency_key TEXT,
                    job_id TEXT,
                    adapter_id TEXT,
                    build_access_hash TEXT
                )
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(asi1_sessions)")
            }
            if "build_access_hash" not in columns:
                connection.execute(
                    "ALTER TABLE asi1_sessions ADD COLUMN build_access_hash TEXT"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS asi1_build_claims(
                    access_hash TEXT PRIMARY KEY,
                    sender TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    job_id TEXT,
                    claimed_at TEXT NOT NULL
                )
                """
            )

    def claim_build(
        self, *, access_hash: str, sender: str, idempotency_key: str
    ) -> bool:
        """Atomically reserve one access code for one idempotent build."""
        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO asi1_build_claims(
                        access_hash, sender, idempotency_key, claimed_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        access_hash,
                        sender,
                        idempotency_key,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                existing = connection.execute(
                    """
                    SELECT sender, idempotency_key FROM asi1_build_claims
                    WHERE access_hash = ?
                    """,
                    (access_hash,),
                ).fetchone()
                return bool(
                    existing
                    and existing["sender"] == sender
                    and existing["idempotency_key"] == idempotency_key
                )

    def record_build_job(self, *, access_hash: str, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE asi1_build_claims SET job_id = ? WHERE access_hash = ?",
                (job_id, access_hash),
            )

    def clear_pending_builds(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE asi1_sessions
                SET pending_action = NULL, pending_workflow = NULL,
                    idempotency_key = NULL, build_access_hash = NULL
                WHERE pending_action IN ('confirm_create', 'confirm:create_adapter')
                """
            )
            return cursor.rowcount

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection
