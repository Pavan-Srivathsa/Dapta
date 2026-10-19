import unittest

from dapta.model_directory import (
    BASE_MODELS,
    BASE_MODELS_BY_ID,
    directory_snapshot,
    efficient_adapter,
)


class ModelDirectoryTests(unittest.TestCase):
    def test_exactly_one_base_model_is_marked_integrated(self):
        integrated = [spec for spec in BASE_MODELS if spec.status == "integrated"]
        self.assertEqual(len(integrated), 1)
        self.assertEqual(integrated[0].model_id, "Qwen/Qwen3-VL-8B-Instruct")

    def test_prefers_adapter_already_on_loaded_base_model(self):
        candidates = [
            {"adapter_id": "a", "base_model": "Qwen/Qwen2.5-7B-Instruct"},
            {"adapter_id": "b", "base_model": "Qwen/Qwen3-VL-8B-Instruct"},
        ]
        selection = efficient_adapter(candidates, loaded_base_model="Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(selection.adapter["adapter_id"], "b")
        self.assertFalse(selection.requires_model_switch)
        self.assertIsNone(selection.switch_target_base_model)

    def test_flags_switch_instead_of_silently_reloading(self):
        candidates = [{"adapter_id": "a", "base_model": "Qwen/Qwen2.5-7B-Instruct"}]
        selection = efficient_adapter(candidates, loaded_base_model="Qwen/Qwen3-VL-8B-Instruct")
        self.assertTrue(selection.requires_model_switch)
        self.assertEqual(selection.switch_target_base_model, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(selection.adapter["adapter_id"], "a")

    def test_no_candidates_returns_no_adapter_without_switch(self):
        selection = efficient_adapter([], loaded_base_model="Qwen/Qwen3-VL-8B-Instruct")
        self.assertIsNone(selection.adapter)
        self.assertFalse(selection.requires_model_switch)

    def test_no_loaded_model_still_requires_explicit_switch(self):
        candidates = [{"adapter_id": "a", "base_model": "Qwen/Qwen3-VL-8B-Instruct"}]
        selection = efficient_adapter(candidates, loaded_base_model=None)
        self.assertTrue(selection.requires_model_switch)
        self.assertEqual(selection.switch_target_base_model, "Qwen/Qwen3-VL-8B-Instruct")

    def test_directory_snapshot_groups_adapters_by_base_model(self):
        adapters = [
            {"adapter_id": "nda-intake/qwen3-vl-8b-v1", "base_model": "Qwen/Qwen3-VL-8B-Instruct"},
        ]
        snapshot = directory_snapshot(adapters)
        self.assertEqual(snapshot["adapter_count"], 1)
        self.assertEqual(snapshot["integrated_base_model"], "Qwen/Qwen3-VL-8B-Instruct")
        integrated_entry = next(
            entry for entry in snapshot["base_models"] if entry["model_id"] == "Qwen/Qwen3-VL-8B-Instruct"
        )
        self.assertIn("nda-intake/qwen3-vl-8b-v1", integrated_entry["adapter_ids"])

    def test_base_models_by_id_matches_catalog(self):
        self.assertEqual(len(BASE_MODELS_BY_ID), len(BASE_MODELS))
        for spec in BASE_MODELS:
            self.assertIs(BASE_MODELS_BY_ID[spec.model_id], spec)


if __name__ == "__main__":
    unittest.main()
