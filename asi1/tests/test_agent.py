from types import SimpleNamespace
import unittest

from asi1.agent import extract_image_url


class AgentAttachmentTests(unittest.TestCase):
    def test_extracts_https_resource_uri(self):
        item = SimpleNamespace(
            resource=SimpleNamespace(
                uri="https://storage.example.test/receipt.jpg",
                metadata={"mime_type": "image/jpeg"},
            )
        )

        self.assertEqual(
            extract_image_url(item), "https://storage.example.test/receipt.jpg"
        )

    def test_extracts_metadata_url_from_resource_list(self):
        item = SimpleNamespace(
            resource=[
                SimpleNamespace(
                    uri="agent-storage://asset/123",
                    metadata={
                        "mime_type": "image/png",
                        "image_url": "https://storage.example.test/receipt.png",
                    },
                )
            ]
        )

        self.assertEqual(
            extract_image_url(item), "https://storage.example.test/receipt.png"
        )

    def test_rejects_non_https_resource(self):
        item = SimpleNamespace(
            resource=SimpleNamespace(
                uri="http://127.0.0.1/private.jpg",
                metadata={"mime_type": "image/jpeg"},
            )
        )

        self.assertIsNone(extract_image_url(item))


if __name__ == "__main__":
    unittest.main()
