"""Qt-free tests for applying collected Settings values onto translator_config."""
from __future__ import annotations

import copy
import unittest

from gui_qt.settings.save_apply import (
    SettingsSaveExtras,
    apply_collected_settings,
)


class SettingsSaveApplyTests(unittest.TestCase):
    def test_applies_context_storage_and_project_flags(self) -> None:
        config = {
            "sync": {"rag": {}},
            "batch": {
                "rag": {"enabled": False},
                "source_index": {"enabled": True},
                "model": "gemini-3.1-flash-lite",
            },
        }
        result = apply_collected_settings(
            config,
            {
                "context_storage_location": "game",
                "rag_enabled": True,
                "source_index_enabled": True,
                "bootstrap_on_build": False,
                "sync_source_index_enabled": False,
                "sync_project_analysis_inject_enabled": False,
                "sync_backend": "gemini",
                "sync_model": "gemini-sync",
                "litellm_model": "",
                "batch_model": "gemini-3.1-flash-lite",
                "sync_embedding_model": "gemini-embedding-001",
                "batch_embedding_model": "gemini-embedding-001",
            },
            original_config=config,
            extras=SettingsSaveExtras(game_root=None),
        )
        self.assertTrue(result.ok)
        self.assertEqual(config["context_storage"]["location"], "game")
        self.assertEqual(config["context_storage"]["game_dir_name"], "translation_context")
        self.assertEqual(config["sync"]["backend"], "gemini")
        self.assertEqual(config["sync"]["model"], "gemini-sync")
        self.assertTrue(result.project_context_flags["rag_enabled"])
        self.assertFalse(result.project_context_flags["bootstrap_on_build"])

    def test_preserves_legacy_context_storage_dir_name(self) -> None:
        config = {"context_storage": {"directory_name": "my_context"}}
        result = apply_collected_settings(
            config,
            {"context_storage_location": "tool"},
            original_config=config,
        )
        self.assertTrue(result.ok)
        self.assertEqual(config["context_storage"]["location"], "tool")
        self.assertEqual(config["context_storage"]["game_dir_name"], "my_context")

    def test_blocks_litellm_without_model(self) -> None:
        config: dict = {"sync": {}, "batch": {}}
        result = apply_collected_settings(
            config,
            {"sync_backend": "litellm", "litellm_model": ""},
            original_config=config,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.block_page, "litellm")
        self.assertIn("provider 前缀", result.block_message or "")

    def test_blocks_invalid_custom_provider_save(self) -> None:
        config: dict = {"sync": {}, "batch": {}}
        result = apply_collected_settings(
            config,
            {"sync_backend": "gemini"},
            original_config=config,
            extras=SettingsSaveExtras(
                custom_providers_modified=True,
                custom_providers_load_error="bad json",
            ),
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.block_page, "litellm")
        self.assertTrue(result.block_warning)
        self.assertIn("bad json", result.block_message or "")

    def test_writes_theme_without_clobbering_sync(self) -> None:
        config: dict = {"sync": {"backend": "litellm", "model": "openai/gpt"}}
        result = apply_collected_settings(
            config,
            {"theme": "dark"},
            original_config=copy.deepcopy(config),
        )
        self.assertTrue(result.ok)
        self.assertEqual(config["gui"]["theme"], "dark")
        self.assertEqual(config["sync"]["backend"], "litellm")
        self.assertEqual(config["sync"]["model"], "openai/gpt")

    def test_converts_tuple_of_pairs_custom_providers(self) -> None:
        config: dict = {"sync": {}, "batch": {}}
        result = apply_collected_settings(
            config,
            {
                "sync_backend": "gemini",
                "custom_litellm_providers": (
                    (("id", "opencode-go"), ("base_url", "https://example.invalid")),
                ),
            },
            original_config=copy.deepcopy(config),
        )
        self.assertTrue(result.ok)
        self.assertEqual(
            config["sync"]["custom_litellm_providers"],
            [{"id": "opencode-go", "base_url": "https://example.invalid"}],
        )

    def test_empty_modified_custom_providers_clears_key(self) -> None:
        config: dict = {
            "sync": {"custom_litellm_providers": [{"id": "keep-me"}]},
            "batch": {},
        }
        result = apply_collected_settings(
            config,
            {"sync_backend": "gemini", "custom_litellm_providers": ()},
            original_config=copy.deepcopy(config),
            extras=SettingsSaveExtras(custom_providers_modified=True),
        )
        self.assertTrue(result.ok)
        self.assertNotIn("custom_litellm_providers", config["sync"])

    def test_unmodified_empty_custom_providers_preserve_on_disk(self) -> None:
        config: dict = {
            "sync": {"custom_litellm_providers": [{"id": "keep-me"}]},
            "batch": {},
        }
        result = apply_collected_settings(
            config,
            {"sync_backend": "gemini", "custom_litellm_providers": ()},
            original_config=copy.deepcopy(config),
        )
        self.assertTrue(result.ok)
        self.assertEqual(
            config["sync"]["custom_litellm_providers"],
            [{"id": "keep-me"}],
        )

    def test_explicit_disabled_thinking_writes_empty_key(self) -> None:
        config: dict = {
            "sync": {},
            "batch": {"model": "gemini-3.1-flash-lite"},
        }
        result = apply_collected_settings(
            config,
            {
                "batch_model": "gemini-3.1-flash-lite",
                "batch_thinking_level": "",
            },
            original_config=copy.deepcopy(config),
            extras=SettingsSaveExtras(batch_thinking_user_changed=True),
        )
        self.assertTrue(result.ok)
        self.assertEqual(config["batch"]["thinking_level"], "")

    def test_advanced_validation_errors_do_not_apply(self) -> None:
        config: dict = {"sync": {}, "batch": {}}
        original = copy.deepcopy(config)
        result = apply_collected_settings(
            config,
            {"context_storage_game_dir_name": ""},
            original_config=original,
        )
        self.assertFalse(result.ok)
        self.assertIn("context_storage_game_dir_name", result.advanced_errors)
        self.assertEqual(config, original)

    def test_restores_original_global_project_analysis_keys(self) -> None:
        config = {
            "sync": {},
            "batch": {
                "chunk_size": 8,
                "project_analysis": {
                    "enabled": True,
                    "inject_published_brief": True,
                },
            },
        }
        original = copy.deepcopy(config)
        result = apply_collected_settings(
            config,
            {"batch_chunk_size": 12},
            original_config=original,
            extras=SettingsSaveExtras(game_root="C:/Game/work"),
        )
        self.assertTrue(result.ok)
        self.assertEqual(config["batch"]["chunk_size"], 12)
        self.assertTrue(config["batch"]["project_analysis"]["enabled"])
        self.assertTrue(config["batch"]["project_analysis"]["inject_published_brief"])



class ModelRoutingSaveApplyTests(unittest.TestCase):
    def test_collected_model_routing_is_applied_with_unknown_fields(self) -> None:
        section = {
            "schema_version": 1,
            "providers": {
                "gemini-main": {
                    "label": "Gemini",
                    "adapter": "gemini",
                    "provider": "gemini",
                    "credential_ref": {
                        "kind": "api_keys_json",
                        "name": "api_keys",
                        "env_name": "GEMINI_API_KEY",
                    },
                }
            },
            "profiles": {
                "gemini-main": {
                    "label": "Gemini Main",
                    "provider_id": "gemini-main",
                    "model": "gemini-3.5-flash",
                    "future_profile_field": "keep",
                }
            },
            "defaults": {
                "primary_profile_id": "gemini-main",
                "execution_strategy": "sync",
            },
            "routes": {},
            "future_section_field": {"keep": True},
        }
        config = {"model_routing": {"schema_version": 1}}

        result = apply_collected_settings(
            config,
            {"model_routing": copy.deepcopy(section)},
            original_config=copy.deepcopy(config),
        )

        self.assertTrue(result.ok, result)
        self.assertEqual(config["model_routing"], section)

    def test_invalid_collected_model_routing_blocks_save(self) -> None:
        config = {"model_routing": {"schema_version": 1}}
        invalid = {
            "schema_version": 1,
            "providers": {},
            "profiles": {},
            "defaults": {"primary_profile_id": "", "execution_strategy": "sync"},
            "routes": {},
        }

        result = apply_collected_settings(
            config,
            {"model_routing": copy.deepcopy(invalid)},
            original_config=copy.deepcopy(config),
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.block_page, "profiles")
        self.assertEqual(config["model_routing"], {"schema_version": 1})

    def test_explicit_removal_drops_invalid_model_routing(self) -> None:
        config = {"model_routing": {"schema_version": 1}, "sync": {}}

        result = apply_collected_settings(
            config,
            {"model_routing": None, "sync_model": "gemini-3.1-flash-lite"},
            original_config=copy.deepcopy(config),
        )

        self.assertTrue(result.ok, result)
        self.assertNotIn("model_routing", config)
        self.assertEqual(config["sync"]["model"], "gemini-3.1-flash-lite")

    def test_legacy_invalid_section_still_blocks_models_page(self) -> None:
        config = {"model_routing": {"schema_version": 1}}

        result = apply_collected_settings(
            config,
            {"sync_model": "gemini-3.1-flash-lite"},
            original_config=copy.deepcopy(config),
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.block_page, "models")


if __name__ == "__main__":
    unittest.main()
