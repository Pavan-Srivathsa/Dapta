from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from asi1.session_store import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_state_survives_new_store_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.sqlite3"
            first = SessionStore(path)
            session = first.get("agent1")
            session.job_id = "job-123"
            session.adapter_id = "receipt/qwen-v1"
            first.save(session)

            restored = SessionStore(path).get("agent1")

            self.assertEqual(restored.job_id, "job-123")
            self.assertEqual(restored.adapter_id, "receipt/qwen-v1")

    def test_clear_pending_builds_preserves_existing_job_and_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory) / "sessions.sqlite3")
            for index, sender in enumerate(("agent1", "agent2")):
                session = store.get(sender)
                session.pending_action = (
                    "confirm_create" if index == 0 else "confirm:create_adapter"
                )
                session.pending_workflow = "synthetic workflow"
                session.idempotency_key = f"key-{sender}"
                session.build_access_hash = "digest"
                session.job_id = "existing-job"
                session.adapter_id = "existing-adapter"
                store.save(session)

            self.assertEqual(store.clear_pending_builds(), 2)
            for sender in ("agent1", "agent2"):
                session = store.get(sender)
                self.assertIsNone(session.pending_action)
                self.assertIsNone(session.pending_workflow)
                self.assertIsNone(session.idempotency_key)
                self.assertIsNone(session.build_access_hash)
                self.assertEqual(session.job_id, "existing-job")
                self.assertEqual(session.adapter_id, "existing-adapter")

    def test_build_claim_is_atomic_and_idempotent_for_one_request(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory) / "sessions.sqlite3")

            self.assertTrue(
                store.claim_build(
                    access_hash="digest", sender="agent1", idempotency_key="key-1"
                )
            )
            self.assertTrue(
                store.claim_build(
                    access_hash="digest", sender="agent1", idempotency_key="key-1"
                )
            )
            self.assertFalse(
                store.claim_build(
                    access_hash="digest", sender="agent2", idempotency_key="key-2"
                )
            )


if __name__ == "__main__":
    unittest.main()
