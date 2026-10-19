import unittest

from dapta.model_server import (
    GenerateRequest,
    NDA_SHORT_SYSTEM,
    NDA_USER_INSTRUCTION,
    RECEIPT_ADAPTER_ID,
    RECEIPT_PROMPT,
    generation_messages,
)


class ModelServerPromptTests(unittest.TestCase):
    def test_nda_adapter_uses_the_compact_benchmark_contract_with_schema_guardrails(self):
        request = GenerateRequest(
            run_id="run-1",
            workflow="Extract an NDA",
            document_text="Example agreement text",
            adapter_id="nda-intake/qwen3-vl-8b-v1",
        )
        messages = generation_messages(request)
        self.assertEqual(messages[0]["content"], NDA_SHORT_SYSTEM)
        self.assertEqual(messages[1]["content"], NDA_USER_INSTRUCTION + request.document_text)
        self.assertIn('"effective_date"', messages[1]["content"])
        self.assertIn('"parties"', messages[1]["content"])
        self.assertIn("never a JSON number", messages[0]["content"])
        self.assertIn('"3_years"', messages[1]["content"])

    def test_other_adapters_keep_the_generic_workflow_prompt(self):
        request = GenerateRequest(
            run_id="run-1",
            workflow="Clean a CSV",
            document_text="a,b\n1,2",
            adapter_id="generated/csv-clean-dashboard-v1/v1",
        )
        messages = generation_messages(request)
        self.assertIn("Clean a CSV", messages[1]["content"])

    def test_receipt_adapter_uses_image_and_frozen_extraction_contract(self):
        request = GenerateRequest(
            run_id="run-1",
            workflow="Extract receipt fields",
            document_text="[attached receipt image]",
            adapter_id=RECEIPT_ADAPTER_ID,
            image_url="https://storage.example.test/receipt.jpg",
        )

        messages = generation_messages(request)

        self.assertEqual(messages[0]["content"][0]["type"], "image_url")
        self.assertEqual(
            messages[0]["content"][0]["image_url"]["url"], request.image_url
        )
        self.assertEqual(messages[0]["content"][1], {"type": "text", "text": RECEIPT_PROMPT})
        self.assertIn('"unit_price"', RECEIPT_PROMPT)
        self.assertIn("output only the JSON object", RECEIPT_PROMPT)


if __name__ == "__main__":
    unittest.main()
