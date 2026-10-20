from __future__ import annotations

import hashlib
import tempfile
import time
import unittest
from pathlib import Path

from asi1.orchestrator import LoraFactoryOrchestrator
from asi1.session_store import SessionStore


class Client:
    JOB_ID = "06138942-d2cc-4484-9dde-f09d26b29fea"

    def __init__(self):
        self.calls = []
        self.readiness = {
            "factory_autostart": True,
            "openrouter_configured": True,
            "devin_configured": True,
            "training_configured": True,
            "publishing_configured": True,
        }
        self.route = {
            "decision": "create_adapter",
            "reason": "No existing specialist matched",
        }
        self.adapters = [
            {
                "adapter_id": "receipt/qwen-v1",
                "version": "v1",
                "metadata": {"skill_prompt": "Extract and validate receipt fields."},
            },
            {"adapter_id": "nda/qwen-v1", "version": "v1"},
        ]
        self.job = {"job_id": self.JOB_ID, "status": "queued"}
        self.status = {
            "job_id": self.JOB_ID,
            "status": "training",
            "progress": {"percent": 50},
        }
        self.execution = {
            "output": {"subtotal": "10.00", "tax": "0.80", "total": "10.80"},
            "input_tokens": 20,
            "output_tokens": 8,
            "latency_ms": 1200,
        }
        self.comparison = {
            "run_id": "run-compare-1",
            "status": "queued",
            "adapter_id": "receipt/qwen-v1",
        }
        self.comparison_result = {
            "run_id": "run-compare-1",
            "status": "completed",
            "adapter_id": "receipt/qwen-v1",
            "results": {
                "qwen_lora": {
                    "status": "completed",
                    "output": {"subtotal": "10.00", "tax": "0.80", "total": "10.80"},
                    "metrics": {
                        "latency_ms": 1200,
                        "input_tokens": 20,
                        "output_tokens": 8,
                        "cost_usd": "0.001",
                    },
                },
                "frontier_a": {
                    "status": "completed",
                    "output": {"subtotal": "10.00", "tax": "0.80", "total": "10.80"},
                    "metrics": {"latency_ms": 2100, "cost_usd": "0.02"},
                },
                "frontier_b": {
                    "status": "completed",
                    "output": {"subtotal": "10.00", "tax": "0.80", "total": "10.80"},
                    "metrics": {"latency_ms": 1800, "cost_usd": "0.03"},
                },
            },
            "summary": {"best_overall": "qwen_lora"},
        }

    def health(self):
        self.calls.append(("health",))
        return self.readiness

    def list_adapters(self):
        self.calls.append(("list",))
        return self.adapters

    def route_workflow(self, workflow):
        self.calls.append(("route", workflow))
        return self.route

    def create_adapter(self, *, workflow, idempotency_key, skill_id=None):
        self.calls.append(("create", workflow, idempotency_key, skill_id))
        return self.job

    def get_job_status(self, job_id):
        self.calls.append(("status", job_id))
        return self.status

    def execute_adapter(
        self, *, run_id, workflow, document_text, adapter_id, image_url=None
    ):
        self.calls.append(
            ("execute", run_id, workflow, document_text, adapter_id, image_url)
        )
        return self.execution

    def run_comparison(
        self, *, workflow, document_text, adapter_id, idempotency_key, attachment_name=None
    ):
        self.calls.append(
            ("compare", workflow, document_text, adapter_id, idempotency_key, attachment_name)
        )
        return self.comparison

    def get_run(self, run_id):
        self.calls.append(("get_run", run_id))
        return self.comparison_result


class OrchestratorTests(unittest.TestCase):
    ACCESS_CODE = "demo-access-code-123456"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.client = Client()
        self.sessions = SessionStore(Path(self.temp.name) / "sessions.sqlite3")
        self.orchestrator = LoraFactoryOrchestrator(
            self.client,
            self.sessions,
            default_adapter_id="receipt/qwen-v1",
            factory_builds_enabled=True,
            factory_build_code_sha256=hashlib.sha256(
                self.ACCESS_CODE.encode("utf-8")
            ).hexdigest(),
            factory_build_expires_at=time.time() + 3600,
            sleeper=lambda _seconds: None,
        )

    def tearDown(self):
        self.temp.cleanup()

    def prepare_build(self, sender, workflow):
        return self.orchestrator.handle(
            sender, f"Access code {self.ACCESS_CODE}. {workflow}"
        )

    def test_connect_runpod_points_to_the_web_app_without_touching_the_client(self):
        reply = self.orchestrator.handle("agent1", "connect runpod")

        self.assertIn("Skills & Adapters", reply[0])
        self.assertIn("Bring your own GPU", reply[0])
        self.assertEqual(self.client.calls, [], "connect runpod must never call the controller")

    def test_connect_runpod_never_stores_a_key_sent_in_chat(self):
        reply = self.orchestrator.handle("agent1", "connect runpod key rp_abc123")

        self.assertIn("not here", reply[0])
        self.assertEqual(self.client.calls, [])

    def test_create_requires_confirmation_before_factory_call(self):
        first = self.prepare_build(
            "agent1", "Create a LoRA adapter that extracts synthetic receipts"
        )

        self.assertIn("Category `create_adapter`", first[0])
        self.assertIn("Reply `confirm`", first[0])
        self.assertTrue(any(call[0] == "route" for call in self.client.calls))
        self.assertFalse(any(call[0] == "create" for call in self.client.calls))

        second = self.orchestrator.handle("agent1", "confirm")

        self.assertIn(Client.JOB_ID, second[0])
        create_calls = [call for call in self.client.calls if call[0] == "create"]
        self.assertEqual(len(create_calls), 1)
        self.assertTrue(create_calls[0][2].startswith("asi1-"))

    def test_second_confirmation_never_duplicates_job(self):
        self.prepare_build("agent1", "Build a LoRA adapter for receipt extraction")
        self.orchestrator.handle("agent1", "confirm build")

        reply = self.orchestrator.handle("agent1", "confirm build")

        self.assertIn("no categorized action", reply[0])
        self.assertEqual(
            len([call for call in self.client.calls if call[0] == "create"]), 1
        )

    def test_existing_adapter_is_selected_instead_of_training(self):
        self.client.route = {
            "decision": "existing_adapter",
            "adapter_id": "nda/qwen-v1",
            "reason": "matched",
        }

        first = self.prepare_build(
            "agent1", "Create a LoRA adapter that extracts NDA fields"
        )
        self.assertIn("Category `existing_adapter`", first[0])

        reply = self.orchestrator.handle("agent1", "confirm")

        self.assertIn("Ready to run `nda/qwen-v1`", reply[0])
        self.assertIn("Document:", reply[0])
        self.assertEqual(self.sessions.get("agent1").adapter_id, "nda/qwen-v1")
        self.assertFalse(any(call[0] == "create" for call in self.client.calls))

    def test_status_uses_current_job_and_reports_real_progress(self):
        session = self.sessions.get("agent1")
        session.job_id = Client.JOB_ID
        self.sessions.save(session)

        reply = self.orchestrator.handle("agent1", "status")

        self.assertIn("training", reply[0])
        self.assertIn("50%", reply[0])

    def test_status_accepts_an_explicit_uuid_job_id(self):
        reply = self.orchestrator.handle(
            "agent1", f"status for {Client.JOB_ID}"
        )

        self.assertIn("training", reply[0])
        self.assertIn(("status", Client.JOB_ID), self.client.calls)

    def test_factory_readiness_blocks_job_creation(self):
        self.client.readiness["training_configured"] = False
        self.prepare_build("agent1", "Build a LoRA adapter for receipt extraction")

        reply = self.orchestrator.handle("agent1", "confirm build")

        self.assertIn("GPU training", reply[0])
        self.assertIn("No factory job was created", reply[0])
        self.assertFalse(any(call[0] == "create" for call in self.client.calls))

    def test_run_collects_document_then_compares_models(self):
        prompt = self.orchestrator.handle("agent1", "run adapter receipt/qwen-v1")
        self.assertIn("Document:", prompt[0])

        reply = self.orchestrator.handle(
            "agent1", "Document: Merchant ACME, subtotal 10.00, tax 0.80, total 10.80"
        )

        self.assertIn("Same task. Every model.", reply[0])
        self.assertIn("## Qwen LoRA", reply[0])
        self.assertIn("## Claude", reply[0])
        self.assertIn("## GPT", reply[0])
        self.assertIn("Best overall: **Qwen LoRA**", reply[0])
        compare = next(call for call in self.client.calls if call[0] == "compare")
        self.assertEqual(
            compare[2], "Merchant ACME, subtotal 10.00, tax 0.80, total 10.80"
        )
        self.assertEqual(compare[3], "receipt/qwen-v1")
        self.assertEqual(compare[1], "Extract and validate receipt fields.")
        self.assertFalse(any(call[0] == "execute" for call in self.client.calls))

    def test_confirmed_receipt_keeps_the_image_without_storing_it(self):
        self.client.route = {
            "decision": "existing_adapter",
            "adapter_id": "receipt/qwen-v1",
            "reason": "matched the receipt specialist",
        }
        self.client.adapters[0]["metadata"]["accepts_image"] = True
        image_url = "https://storage.example.test/receipt.jpg"

        classified = self.orchestrator.handle(
            "agent1", "Extract this receipt", image_url=image_url
        )
        self.assertIn("held until you confirm", classified[0])
        self.assertFalse(any(call[0] == "execute" for call in self.client.calls))
        database = (Path(self.temp.name) / "sessions.sqlite3").read_bytes()
        self.assertNotIn(image_url.encode(), database)

        reply = self.orchestrator.handle("agent1", "confirm")

        self.assertIn("LoRA execution completed", reply[0])
        execute = next(call for call in self.client.calls if call[0] == "execute")
        self.assertEqual(execute[-1], image_url)
        self.assertNotIn(image_url, self.orchestrator.pending_images.values())

    def test_attached_image_is_forwarded_to_selected_adapter(self):
        image_url = "https://storage.example.test/receipt.jpg"

        self.client.adapters[0]["metadata"]["accepts_image"] = True
        reply = self.orchestrator.handle(
            "agent1",
            "run adapter receipt/qwen-v1",
            image_url=image_url,
        )

        self.assertIn("LoRA execution completed", reply[0])
        execute = next(call for call in self.client.calls if call[0] == "execute")
        self.assertEqual(execute[-3], "[attached receipt image]")
        self.assertEqual(execute[-2], "receipt/qwen-v1")
        self.assertEqual(execute[-1], image_url)

    def test_agent_mention_is_stripped_before_explicit_command(self):
        reply = self.orchestrator.handle(
            "agent1", "@dapta-lahacks run adapter nda/qwen-v1"
        )

        self.assertIn("Ready to run `nda/qwen-v1`", reply[0])
        self.assertFalse(any(call[0] == "route" for call in self.client.calls))

    def test_agent_mention_only_returns_help_without_routing(self):
        reply = self.orchestrator.handle("agent1", "@dapta-lahacks")

        self.assertIn("DAPTA on ASI:One", reply[0])
        self.assertFalse(any(call[0] == "route" for call in self.client.calls))

    def test_unknown_adapter_is_not_sent_to_the_worker(self):
        reply = self.orchestrator.handle("agent1", "run adapter missing/qwen-v1")

        self.assertIn("not registered", reply[0])
        self.assertFalse(any(call[0] == "execute" for call in self.client.calls))

    def test_document_is_not_persisted(self):
        self.orchestrator.handle("agent1", "run adapter receipt/qwen-v1")
        document = "PRIVATE-SYNTHETIC-DOCUMENT"
        self.orchestrator.handle("agent1", document)

        database = (Path(self.temp.name) / "sessions.sqlite3").read_bytes()
        self.assertNotIn(document.encode(), database)

    def test_cancel_never_starts_build(self):
        self.prepare_build("agent1", "Create a LoRA adapter for receipt extraction")

        reply = self.orchestrator.handle("agent1", "cancel")

        self.assertIn("No adapter build was started", reply[0])
        self.assertFalse(any(call[0] == "create" for call in self.client.calls))

    def test_public_factory_builds_can_be_disabled(self):
        orchestrator = LoraFactoryOrchestrator(
            self.client,
            self.sessions,
            factory_builds_enabled=False,
        )

        classified = orchestrator.handle(
            "agent1", "Create a LoRA adapter for receipt extraction"
        )
        reply = orchestrator.handle("agent1", "confirm")

        self.assertIn("Category `create_adapter`", classified[0])
        self.assertIn("disabled", reply[0])
        self.assertFalse(any(call[0] == "create" for call in self.client.calls))

    def test_short_reply_after_disabled_build_does_not_round_trip_to_controller(self):
        orchestrator = LoraFactoryOrchestrator(
            self.client,
            self.sessions,
            factory_builds_enabled=False,
        )
        orchestrator.handle("agent1", "Create a LoRA adapter for receipt extraction")
        orchestrator.handle("agent1", "confirm")
        calls_before = len(self.client.calls)

        reply = orchestrator.handle("agent1", "ok")

        self.assertEqual(len(self.client.calls), calls_before)
        self.assertIn("no categorized action", reply[0])

    def test_wrong_access_code_never_prepares_or_routes_build(self):
        reply = self.orchestrator.handle(
            "agent1",
            "Access code wrong-access-code-123456. Create a LoRA adapter for receipts",
        )

        self.assertIn("Category `create_adapter`", reply[0])
        self.assertIn("supplied access code is invalid", reply[0])
        self.assertEqual(
            self.sessions.get("agent1").pending_action, "confirm:create_adapter"
        )
        self.assertFalse(any(call[0] == "create" for call in self.client.calls))

    def test_confirmation_rechecks_revoked_access(self):
        self.prepare_build("agent1", "Create a LoRA adapter for receipt extraction")
        revoked = LoraFactoryOrchestrator(
            self.client,
            self.sessions,
            factory_builds_enabled=False,
            factory_build_code_sha256=self.orchestrator.factory_build_code_sha256,
            factory_build_expires_at=time.time() + 3600,
        )

        reply = revoked.handle("agent1", "confirm build")

        self.assertIn("disabled", reply[0])
        self.assertIsNone(self.sessions.get("agent1").pending_action)
        self.assertFalse(any(call[0] == "create" for call in self.client.calls))

    def test_access_code_claims_only_one_factory_job(self):
        self.prepare_build("agent1", "Create a LoRA adapter for receipt extraction")
        self.orchestrator.handle("agent1", "confirm build")
        self.prepare_build("agent2", "Create a LoRA adapter for invoice extraction")

        reply = self.orchestrator.handle("agent2", "confirm build")

        self.assertIn("already been used", reply[0])
        self.assertEqual(
            len([call for call in self.client.calls if call[0] == "create"]), 1
        )

    def test_expired_access_fails_before_any_live_call(self):
        expired = LoraFactoryOrchestrator(
            self.client,
            self.sessions,
            factory_builds_enabled=True,
            factory_build_code_sha256=self.orchestrator.factory_build_code_sha256,
            factory_build_expires_at=time.time() - 1,
        )

        reply = expired.handle(
            "agent1",
            f"Access code {self.ACCESS_CODE}. Create a LoRA adapter for receipts",
        )
        reply = expired.handle("agent1", "confirm")

        self.assertIn("expired", reply[0])
        self.assertFalse(any(call[0] == "create" for call in self.client.calls))

    def test_plaintext_access_code_is_not_persisted(self):
        self.prepare_build("agent1", "Create a LoRA adapter for receipt extraction")

        database = (Path(self.temp.name) / "sessions.sqlite3").read_bytes()

        self.assertNotIn(self.ACCESS_CODE.encode("utf-8"), database)

    def test_create_can_receive_access_code_with_confirmation(self):
        first = self.orchestrator.handle(
            "agent1", "Create a LoRA adapter for synthetic receipt extraction"
        )

        self.assertIn("Category `create_adapter`", first[0])
        self.assertIn("one-time access code", first[0])

        reply = self.orchestrator.handle(
            "agent1", f"confirm access code {self.ACCESS_CODE}"
        )

        self.assertIn(Client.JOB_ID, reply[0])

    def test_arbitrary_existing_request_is_classified_before_execution(self):
        self.client.route = {
            "decision": "existing_adapter",
            "adapter_id": "nda/qwen-v1",
            "reason": "JEV selected the NDA specialist",
        }

        classified = self.orchestrator.handle(
            "agent1", "Extract the parties and term from an NDA"
        )

        self.assertIn("Category `existing_adapter`", classified[0])
        self.assertFalse(any(call[0] == "execute" for call in self.client.calls))

        confirmed = self.orchestrator.handle("agent1", "confirm")

        self.assertIn("Document:", confirmed[0])
        self.assertFalse(any(call[0] == "execute" for call in self.client.calls))

    def test_inline_document_is_not_persisted_before_confirmation(self):
        self.client.route = {
            "decision": "existing_adapter",
            "adapter_id": "nda/qwen-v1",
            "reason": "matched",
        }
        document = "PRIVATE-SYNTHETIC-NDA-TEXT"

        reply = self.orchestrator.handle(
            "agent1", f"Extract NDA fields. Document: {document}"
        )

        self.assertIn("resend the test case", reply[0])
        database = (Path(self.temp.name) / "sessions.sqlite3").read_bytes()
        self.assertNotIn(document.encode(), database)

    def test_unsupported_request_requires_confirmation_and_does_nothing(self):
        self.client.route = {
            "decision": "unsupported",
            "reason": "No stable specialist workflow",
        }

        classified = self.orchestrator.handle(
            "agent1", "What is the weather tomorrow morning?"
        )
        confirmed = self.orchestrator.handle("agent1", "confirm")

        self.assertIn("Category `unsupported`", classified[0])
        self.assertIn("No adapter was run", confirmed[0])
        self.assertFalse(any(call[0] == "create" for call in self.client.calls))
        self.assertFalse(any(call[0] == "execute" for call in self.client.calls))

    def test_failed_quality_gate_is_reported_without_claiming_training_failed(self):
        self.client.status = {
            "job_id": Client.JOB_ID,
            "status": "failed",
            "failure_code": "ValueError",
            "failure_message": "adapter failed evaluation after 1 data rounds",
        }

        reply = self.orchestrator.handle(
            "agent1", f"status {Client.JOB_ID}"
        )

        self.assertIn("Training completed", reply[0])
        self.assertIn("not published", reply[0])

    def test_oversized_messages_are_rejected_before_execution(self):
        reply = self.orchestrator.handle("agent1", "x" * 500_001)

        self.assertIn("too large", reply[0])
        self.assertEqual(self.client.calls, [])


if __name__ == "__main__":
    unittest.main()
