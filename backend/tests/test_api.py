import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@unittest.skipUnless(HAS_FASTAPI, "FastAPI test dependencies are not installed")
class ApiTests(unittest.TestCase):
    def test_job_and_read_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "dapta.sqlite3"
            environment = {
                "DAPTA_LEDGER_PATH": str(database),
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_FACTORY_MODEL": "moonshotai/kimi-k3",
                "OPENROUTER_CRITIC_MODEL": "openai/gpt-oss-120b",
            }
            with patch.dict(os.environ, environment, clear=False):
                from dapta.api import create_app

                client = TestClient(create_app(database))
                created = client.post(
                    "/jobs",
                    headers={"Idempotency-Key": "api-test-001"},
                    json={
                        "workflow": "Turn receipt images into a structured expense report.",
                        "skill_id": "expense-report",
                    },
                )
                self.assertEqual(created.status_code, 202)
                job_id = created.json()["job_id"]
                self.assertEqual(client.get(f"/jobs/{job_id}").status_code, 200)
                self.assertEqual(client.get("/adapters").json(), [])
                self.assertEqual(client.get("/benchmarks").json(), [])
                self.assertEqual(client.post("/compare").status_code, 501)

    def test_v1_seed_registry_auth_and_agent_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "dapta.sqlite3"
            environment = {
                "DAPTA_LEDGER_PATH": str(database),
                "DAPTA_CONTROL_TOKEN": "controller-test-token",
                "DAPTA_SEED_DEMO_DATA": "true",
            }
            with patch.dict(os.environ, environment, clear=False):
                from dapta.api import create_app

                client = TestClient(create_app(database))
                self.assertEqual(client.get("/v1/adapters").status_code, 401)
                headers = {"X-Dapta-Token": "controller-test-token"}
                adapters = client.get("/v1/adapters", headers=headers).json()
                by_id = {item["adapter_id"]: item for item in adapters}
                self.assertIn("nda-intake/qwen3-vl-8b-v1", by_id)
                receipt = by_id["receipt-to-expense/qwen3-vl-8b-cord-v1"]
                self.assertTrue(receipt["metadata"]["accepts_image"])
                benchmarks = client.get("/v1/benchmarks", headers=headers).json()
                self.assertEqual(benchmarks[0]["test_documents"], 83)
                tools = client.get("/v1/agent/tools", headers=headers).json()
                self.assertEqual(
                    {tool["name"] for tool in tools},
                    {"list_adapters", "run_comparison", "create_adapter", "get_job_status"},
                )
                directory = client.get("/v1/directory", headers=headers).json()
                self.assertEqual(directory["integrated_base_model"], "Qwen/Qwen3-VL-8B-Instruct")
                integrated_entry = next(
                    entry for entry in directory["base_models"]
                    if entry["model_id"] == "Qwen/Qwen3-VL-8B-Instruct"
                )
                self.assertIn("nda-intake/qwen3-vl-8b-v1", integrated_entry["adapter_ids"])

                catalog_ids = {
                    "csv-to-dashboard/qwen2.5-7b-catalog-v0": "Qwen/Qwen2.5-7B-Instruct",
                    "three-statement-model/qwen3-vl-30b-a3b-catalog-v0": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                    "pii-redaction/qwen3-vl-2b-catalog-v0": "Qwen/Qwen3-VL-2B-Instruct",
                }
                for adapter_id, base_model in catalog_ids.items():
                    self.assertIn(adapter_id, by_id, f"{adapter_id} missing from /v1/adapters")
                    self.assertEqual(by_id[adapter_id]["metadata"]["status"], "catalog")
                    self.assertEqual(by_id[adapter_id]["base_model"], base_model)
                    entry = next(e for e in directory["base_models"] if e["model_id"] == base_model)
                    self.assertEqual(entry["status"], "catalog_only")
                    self.assertIn(adapter_id, entry["adapter_ids"])

    def test_catalog_adapter_cannot_be_run(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "dapta.sqlite3"
            environment = {
                "DAPTA_LEDGER_PATH": str(database),
                "DAPTA_CONTROL_TOKEN": "controller-test-token",
                "DAPTA_SEED_DEMO_DATA": "true",
            }
            with patch.dict(os.environ, environment, clear=False):
                from dapta.api import create_app

                client = TestClient(create_app(database))
                headers = {
                    "X-Dapta-Token": "controller-test-token",
                    "Idempotency-Key": "catalog-run-test-001",
                }
                response = client.post(
                    "/v1/comparisons",
                    headers=headers,
                    json={
                        "workflow": "Clean this CSV and build a dashboard.",
                        "document_text": "a,b\n1,2\n",
                        "adapter_id": "csv-to-dashboard/qwen2.5-7b-catalog-v0",
                    },
                )
                self.assertEqual(response.status_code, 409)
                self.assertIn("not trained yet", response.json()["detail"])

                tool_response = client.post(
                    "/v1/agent/tools/run_comparison",
                    headers={"X-Dapta-Token": "controller-test-token"},
                    json={
                        "arguments": {
                            "workflow": "Clean this CSV and build a dashboard.",
                            "document_text": "a,b\n1,2\n",
                            "adapter_id": "csv-to-dashboard/qwen2.5-7b-catalog-v0",
                            "idempotency_key": "catalog-run-test-002",
                        },
                    },
                )
                self.assertEqual(tool_response.status_code, 409)

    def test_runpod_connection_requires_sign_in_then_validates_stores_and_disconnects(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "dapta.sqlite3"
            environment = {
                "DAPTA_LEDGER_PATH": str(database),
                "DAPTA_CONTROL_TOKEN": "controller-test-token",
            }
            with patch.dict(os.environ, environment, clear=False):
                from dapta.api import create_app

                client = TestClient(create_app(database))
                headers = {"X-Dapta-Token": "controller-test-token"}

                # No X-Dapta-User-Id: this is the proxy's job, so the controller refuses.
                no_user = client.get("/v1/users/me/runpod-connection", headers=headers)
                self.assertEqual(no_user.status_code, 401)

                signed_in = {**headers, "X-Dapta-User-Id": "base44-user-1"}
                not_connected = client.get("/v1/users/me/runpod-connection", headers=signed_in)
                self.assertEqual(not_connected.json(), {"connected": False})

                def fake_transport(request: httpx.Request) -> httpx.Response:
                    self.assertEqual(request.headers["authorization"], "Bearer real-looking-key-123")
                    return httpx.Response(200, json={"pods": []})

                with patch(
                    "dapta.runpod_connection.httpx.Client",
                    return_value=httpx.Client(transport=httpx.MockTransport(fake_transport)),
                ):
                    connected = client.post(
                        "/v1/users/me/runpod-connection",
                        headers=signed_in,
                        json={"api_key": "real-looking-key-123"},
                    )
                self.assertEqual(connected.status_code, 200)
                self.assertEqual(connected.json()["masked_hint"], "real…-123")
                self.assertNotIn("real-looking-key-123", connected.text)

                status = client.get("/v1/users/me/runpod-connection", headers=signed_in).json()
                self.assertTrue(status["connected"])

                # A different signed-in user sees their own (unconnected) status, not user 1's.
                other_user = {**headers, "X-Dapta-User-Id": "base44-user-2"}
                self.assertEqual(
                    client.get("/v1/users/me/runpod-connection", headers=other_user).json(),
                    {"connected": False},
                )

                with patch(
                    "dapta.runpod_connection.httpx.Client",
                    return_value=httpx.Client(
                        transport=httpx.MockTransport(lambda request: httpx.Response(401))
                    ),
                ):
                    rejected = client.post(
                        "/v1/users/me/runpod-connection",
                        headers=signed_in,
                        json={"api_key": "wrong-but-well-formed-key"},
                    )
                self.assertEqual(rejected.status_code, 422)
                # Rejection doesn't disturb the already-stored valid connection.
                self.assertTrue(client.get("/v1/users/me/runpod-connection", headers=signed_in).json()["connected"])

                disconnected = client.post("/v1/users/me/runpod-connection/disconnect", headers=signed_in)
                self.assertEqual(disconnected.json(), {"connected": False})
                self.assertEqual(
                    client.get("/v1/users/me/runpod-connection", headers=signed_in).json(), {"connected": False}
                )

    def test_agent_training_requires_the_access_file_and_stays_small(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "dapta.sqlite3"
            access = root / "access.json"
            environment = {
                "DAPTA_LEDGER_PATH": str(database),
                "DAPTA_CONTROL_TOKEN": "controller-test-token",
                "DAPTA_FACTORY_AUTOSTART": "false",
                "DAPTA_TRAINING_ACCESS_PATH": str(access),
            }
            with patch.dict(os.environ, environment, clear=False):
                from dapta.api import create_app
                from dapta.jobs import JobStore

                client = TestClient(create_app(database))
                headers = {"X-Dapta-Token": "controller-test-token"}
                body = {
                    "idempotency_key": "smoke-access-001",
                    "arguments": {
                        "workflow": "Classify synthetic receipts into expense categories.",
                        "idempotency_key": "smoke-access-001",
                    },
                }
                closed = client.post("/v1/agent/tools/create_adapter", headers=headers, json=body)
                self.assertEqual(closed.status_code, 403)
                operator = client.post(
                    "/jobs",
                    headers={"Idempotency-Key": "operator-job-001"},
                    json={"workflow": "Turn receipt images into a structured expense report."},
                )
                self.assertEqual(operator.status_code, 202)

                access.write_text('{"enabled":true,"expires_at":9999999999}\n', encoding="utf-8")
                opened = client.post("/v1/agent/tools/create_adapter", headers=headers, json=body)
                self.assertEqual(opened.status_code, 200)
                limits = JobStore(database).get_artifact(opened.json()["job_id"], "training_limits.json")
                self.assertEqual(limits["profile"], "smoke")
                self.assertEqual(limits["max_examples"], 10)
                self.assertEqual(limits["max_train_seconds"], 120)


if __name__ == "__main__":
    unittest.main()
