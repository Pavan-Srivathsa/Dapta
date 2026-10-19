from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asi1.session_store import SessionStore
from asi1.training_access import disable, enable


def env_values(path: Path) -> dict[str, str]:
    return {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    }


class TrainingAccessTests(unittest.TestCase):
    def test_enable_stores_only_digest_and_expiry(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "asi1.training_access.time.time", return_value=1_000
        ):
            env_path = Path(directory) / ".env"
            env_path.write_text("KEEP=value\n", encoding="utf-8")

            access_path = Path(directory) / "access.json"
            code, expires_at = enable(env_path, ttl_seconds=600, access_path=access_path)

            values = env_values(env_path)
            self.assertEqual(values["KEEP"], "value")
            self.assertEqual(values["DAPTA_ASI1_ENABLE_FACTORY_BUILDS"], "true")
            self.assertEqual(expires_at, 1_600)
            self.assertEqual(
                values["DAPTA_ASI1_FACTORY_BUILD_CODE_SHA256"],
                hashlib.sha256(code.encode("utf-8")).hexdigest(),
            )
            self.assertNotIn(code, env_path.read_text(encoding="utf-8"))
            self.assertEqual(
                access_path.read_text(encoding="utf-8"),
                '{"enabled":true,"expires_at":1600}\n',
            )

    def test_disable_clears_pending_and_turns_access_off(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            state_path = Path(directory) / "sessions.sqlite3"
            access_path = Path(directory) / "access.json"
            access_path.write_text('{"enabled":true,"expires_at":2000}\n', encoding="utf-8")
            env_path.write_text(
                "DAPTA_ASI1_ENABLE_FACTORY_BUILDS=true\n"
                "DAPTA_ASI1_FACTORY_BUILD_CODE_SHA256=digest\n"
                "DAPTA_ASI1_FACTORY_BUILD_EXPIRES_AT=2000\n",
                encoding="utf-8",
            )
            store = SessionStore(state_path)
            session = store.get("agent1")
            session.pending_action = "confirm_create"
            session.pending_workflow = "workflow"
            session.idempotency_key = "key"
            store.save(session)

            cleared = disable(env_path, state_path, access_path)

            values = env_values(env_path)
            self.assertEqual(cleared, 1)
            self.assertEqual(values["DAPTA_ASI1_ENABLE_FACTORY_BUILDS"], "false")
            self.assertEqual(values["DAPTA_ASI1_FACTORY_BUILD_CODE_SHA256"], "")
            self.assertEqual(values["DAPTA_ASI1_FACTORY_BUILD_EXPIRES_AT"], "")
            self.assertIsNone(store.get("agent1").pending_action)
            self.assertFalse(access_path.exists())


if __name__ == "__main__":
    unittest.main()
