import unittest

from gui_qt.settings_schema import (
    ADVANCED_SETTING_FIELD_BY_KEY,
    apply_advanced_settings,
    filter_gemini_rotation_models,
    read_advanced_settings,
    recommended_advanced_settings,
    validate_advanced_settings,
)


class GuiSettingsSchemaTests(unittest.TestCase):
    def test_read_advanced_settings_uses_defaults_for_missing_config(self):
        values = read_advanced_settings({})

        self.assertEqual(values["sync_chunk_size"], 40)
        self.assertEqual(values["batch_chunk_size"], 60)
        self.assertEqual(values["batch_temperature"], 0.2)
        self.assertEqual(values["batch_source_index_min_similarity"], 0.72)
        self.assertEqual(values["context_storage_game_dir_name"], "translation_context")
        self.assertEqual(values["batch_rag_store_dir"], "")

    def test_read_advanced_settings_coerces_valid_existing_values(self):
        values = read_advanced_settings(
            {
                "sync": {
                    "chunk_size": "7",
                    "rag": {
                        "enabled": True,
                        "min_similarity": "0.5",
                        "store_dir": " logs/sync_rag ",
                    },
                },
                "batch": {
                    "temperature": "0.6",
                    "source_index": {"top_k": "3"},
                    "story_memory": {"include_scene_summary": False},
                },
            }
        )

        self.assertEqual(values["sync_chunk_size"], 7)
        self.assertTrue(values["sync_rag_enabled"])
        self.assertEqual(values["sync_rag_min_similarity"], 0.5)
        self.assertEqual(values["sync_rag_store_dir"], "logs/sync_rag")
        self.assertEqual(values["batch_temperature"], 0.6)
        self.assertEqual(values["batch_source_index_top_k"], 3)
        self.assertFalse(values["batch_story_memory_include_scene_summary"])

    def test_validate_advanced_settings_rejects_out_of_range_values(self):
        values = recommended_advanced_settings()
        values["batch_chunk_size"] = 0
        values["batch_temperature"] = 2.5
        values["sync_rag_min_similarity"] = -0.1
        values["context_storage_game_dir_name"] = ""

        errors = validate_advanced_settings(values)

        self.assertIn("batch_chunk_size", errors)
        self.assertIn("batch_temperature", errors)
        self.assertIn("sync_rag_min_similarity", errors)
        self.assertIn("context_storage_game_dir_name", errors)

    def test_list_and_json_fields_round_trip_from_text(self):
        config = {"game_root": "C:/Game/work", "batch": {}}
        values = recommended_advanced_settings()
        values["game_root"] = "C:/Game/work"
        values["include_files"] = "chapter01.rpy\nchapter02.rpy"
        values["include_prefixes"] = "routes/common, routes/extra"
        values["batch_safety_settings"] = '[{"category":"sexually_explicit","threshold":"BLOCK_NONE"}]'
        values["prepare_unpack_command"] = '["python", "unpack.py"]'
        values["batch_macro_setting"] = "Use a concise voice.\nKeep honorifics."

        saved = apply_advanced_settings(config, values)

        self.assertEqual(saved["include_files"], ["chapter01.rpy", "chapter02.rpy"])
        self.assertEqual(saved["include_prefixes"], ["routes/common", "routes/extra"])
        self.assertEqual(
            saved["batch"]["safety_settings"],
            [{"category": "sexually_explicit", "threshold": "BLOCK_NONE"}],
        )
        self.assertEqual(saved["prepare"]["unpack_command"], ["python", "unpack.py"])
        self.assertEqual(saved["batch"]["macro_setting"], "Use a concise voice.\nKeep honorifics.")

    def test_invalid_json_field_reports_error(self):
        values = recommended_advanced_settings()
        values["game_root"] = "C:/Game/work"
        values["batch_safety_settings"] = '[{"bad"'

        errors = validate_advanced_settings(values)

        self.assertIn("batch_safety_settings", errors)
        self.assertIn("有效 JSON", errors["batch_safety_settings"])

    def test_validate_advanced_settings_allows_empty_paths(self):
        values = recommended_advanced_settings()
        values["batch_rag_store_dir"] = ""
        values["batch_source_index_store_dir"] = ""
        values["batch_story_memory_graph_file"] = ""

        errors = validate_advanced_settings(values)

        self.assertNotIn("batch_rag_store_dir", errors)
        self.assertNotIn("batch_source_index_store_dir", errors)
        self.assertNotIn("batch_story_memory_graph_file", errors)

    def test_apply_advanced_settings_preserves_unknown_fields(self):
        config = {
            "game_root": "C:/Game/work",
            "sync": {"model": "gemini-sync", "custom": 1},
            "batch": {
                "model": "gemini-batch",
                "rag": {"enabled": True, "unknown_rag": "keep"},
                "source_index": {"enabled": True},
            },
            "unknown_top": {"keep": True},
        }
        values = recommended_advanced_settings()
        values["game_root"] = "C:/Game/work"
        values["sync_chunk_size"] = 12
        values["batch_temperature"] = 0.4
        values["batch_rag_top_k_history"] = 9
        values["batch_source_index_store_dir"] = "C:/ctx/source"

        saved = apply_advanced_settings(config, values)

        self.assertIs(saved, config)
        self.assertEqual(saved["game_root"], "C:/Game/work")
        self.assertEqual(saved["sync"]["model"], "gemini-sync")
        self.assertEqual(saved["sync"]["custom"], 1)
        self.assertEqual(saved["batch"]["model"], "gemini-batch")
        self.assertTrue(saved["batch"]["rag"]["enabled"])
        self.assertEqual(saved["batch"]["rag"]["unknown_rag"], "keep")
        self.assertTrue(saved["batch"]["source_index"]["enabled"])
        self.assertEqual(saved["unknown_top"], {"keep": True})
        self.assertEqual(saved["sync"]["chunk_size"], 12)
        self.assertEqual(saved["batch"]["temperature"], 0.4)
        self.assertEqual(saved["batch"]["rag"]["top_k_history"], 9)
        self.assertEqual(saved["batch"]["source_index"]["store_dir"], "C:/ctx/source")

    def test_field_registry_exposes_expected_paths(self):
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["batch_source_index_char_limit"].path,
            ("batch", "source_index", "char_limit"),
        )
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["sync_story_memory_graph_file"].path,
            ("sync", "story_memory", "graph_file"),
        )
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["model_rotation_models"].kind,
            "gemini_model_list",
        )

    def test_gemini_model_list_rejects_unknown_ids(self):
        with self.assertRaises(ValueError):
            filter_gemini_rotation_models(
                ["gemini-3.1-flash-lite", "totally-fake-model"],
                reject_unknown=True,
            )
        self.assertEqual(
            filter_gemini_rotation_models(
                ["gemini-3.1-flash-lite", "totally-fake-model"],
                reject_unknown=False,
            ),
            ["gemini-3.1-flash-lite"],
        )

    def test_catalog_gemini_lists_strip_builtins_on_write(self):
        config = {"game_root": "C:/Game/work"}
        values = recommended_advanced_settings()
        values["game_root"] = "C:/Game/work"
        values["catalog_gemini_models"] = [
            "gemini-3.1-flash-lite",
            "gemini-custom-extra",
        ]
        values["catalog_gemini_embedding_models"] = [
            "gemini-embedding-001",
            "gemini-embedding-custom",
        ]
        saved = apply_advanced_settings(config, values)
        self.assertEqual(saved["model_catalog"]["gemini"], ["gemini-custom-extra"])
        self.assertEqual(
            saved["model_catalog"]["gemini_embedding"],
            ["gemini-embedding-custom"],
        )
        self.assertEqual(
            ADVANCED_SETTING_FIELD_BY_KEY["catalog_gemini_models"].kind,
            "gemini_catalog_list",
        )


if __name__ == "__main__":
    unittest.main()
