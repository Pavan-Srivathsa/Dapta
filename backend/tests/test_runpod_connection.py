import tempfile
import unittest
from pathlib import Path

import httpx

from dapta.runpod_connection import (
    InvalidRunpodKey,
    RunpodConnectionError,
    RunpodConnectionStore,
    verify_runpod_key,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class VerifyRunpodKeyTests(unittest.TestCase):
    def test_well_formed_key_accepted_by_runpod_passes(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            self.assertEqual(request.url.path, "/v2/pods")
            self.assertEqual(request.headers["authorization"], "Bearer real-looking-key-123")
            return httpx.Response(200, json={"pods": []})

        verify_runpod_key("real-looking-key-123", client=_client(handler))
        self.assertEqual(len(seen), 1)

    def test_runpod_401_raises_invalid_key_without_a_second_call(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"title": "Unauthorized"})

        with self.assertRaises(InvalidRunpodKey):
            verify_runpod_key("wrong-but-well-formed-key", client=_client(handler))

    def test_malformed_key_is_rejected_before_any_network_call(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not call Runpod with an obviously malformed key")

        with self.assertRaises(InvalidRunpodKey):
            verify_runpod_key("has spaces and @#$ symbols", client=_client(handler))

    def test_unexpected_server_error_raises_connection_error_not_invalid_key(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"title": "Internal Server Error"})

        with self.assertRaises(RunpodConnectionError):
            verify_runpod_key("real-looking-key-123", client=_client(handler))


class RunpodConnectionStoreTests(unittest.TestCase):
    def test_connect_requires_a_passing_verification_then_masks_the_key(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"pods": []})

        with tempfile.TemporaryDirectory() as directory:
            store = RunpodConnectionStore(Path(directory) / "runpod.sqlite3")
            result = store.connect("user-1", "real-looking-key-123", client=_client(handler))

            self.assertTrue(result["connected"])
            self.assertEqual(result["masked_hint"], "real…-123")
            self.assertNotIn("real-looking-key-123", str(result))

            status = store.status("user-1")
            self.assertTrue(status["connected"])
            self.assertEqual(status["masked_hint"], "real…-123")
            # The real key is retrievable internally (for a future job to use) but never
            # appears in anything returned over the API-shaped methods.
            self.assertEqual(store.get_api_key("user-1"), "real-looking-key-123")

    def test_status_for_an_unconnected_user_says_so_without_touching_runpod(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunpodConnectionStore(Path(directory) / "runpod.sqlite3")
            self.assertEqual(store.status("nobody"), {"connected": False})

    def test_disconnect_removes_the_connection(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"pods": []})

        with tempfile.TemporaryDirectory() as directory:
            store = RunpodConnectionStore(Path(directory) / "runpod.sqlite3")
            store.connect("user-2", "real-looking-key-123", client=_client(handler))

            self.assertTrue(store.disconnect("user-2"))
            self.assertEqual(store.status("user-2"), {"connected": False})
            self.assertFalse(store.disconnect("user-2"), "a second disconnect has nothing to remove")
