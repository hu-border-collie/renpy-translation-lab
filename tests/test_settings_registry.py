"""Registry tests for #202 Settings page identity and key ownership."""
from __future__ import annotations

import unittest

from gui_qt.settings.registry import (
    ADVANCED_CONFIG_KEYS,
    CONTEXT_CONFIG_KEYS,
    PROJECT_CONFIG_KEYS,
    SETTINGS_CONFIG_PAGE_KEYS,
    SETTINGS_PAGE_SPECS,
    SettingsPageRegistry,
    SettingsPageSpec,
    WORKSPACE_MANAGED_KEYS,
    build_default_registry,
)
from gui_qt.settings_schema import (
    ADVANCED_SETTING_FIELDS,
    CONTEXT_PRIMARY_SETTING_KEYS,
)


class SettingsRegistryTests(unittest.TestCase):
    def test_default_registry_has_the_eleven_current_pages(self) -> None:
        registry = build_default_registry()
        self.assertEqual(
            registry.keys(),
            (
                "workspace",
                "project",
                "api_keys",
                "models",
                "profiles",
                "litellm",
                "extensions",
                "context",
                "appearance",
                "shortcuts",
                "advanced",
            ),
        )
        self.assertEqual(len(SETTINGS_PAGE_SPECS), 11)

    def test_legacy_tuple_is_derived_from_spec_objects(self) -> None:
        registry = build_default_registry()
        self.assertEqual(
            SETTINGS_PAGE_SPECS,
            tuple(
                (spec.key, spec.nav_label, spec.builder_name)
                for spec in registry.specs()
            ),
        )
        for spec in registry.specs():
            self.assertEqual(
                spec.builder_name,
                f"_create_{spec.key}_settings_page",
                spec.key,
            )

    def test_every_config_key_has_exactly_one_owner(self) -> None:
        registry = build_default_registry()
        owners: dict[str, str] = {}
        for spec in registry.specs():
            for key in spec.config_keys:
                self.assertNotIn(key, owners, key)
                owners[key] = spec.key
        all_advanced = {field.key for field in ADVANCED_SETTING_FIELDS}
        self.assertTrue(
            (all_advanced - WORKSPACE_MANAGED_KEYS).issubset(owners)
        )
        self.assertEqual(
            {
                "sync_model",
                "sync_embedding_model",
                "batch_model",
                "batch_embedding_model",
                "batch_thinking_level",
                "model_routing",
                "sync_backend",
                "litellm_model",
                "custom_litellm_providers",
                "theme",
                "rag_enabled",
                "source_index_enabled",
                "bootstrap_on_build",
                "sync_source_index_enabled",
                "sync_project_analysis_inject_enabled",
                "context_storage_location",
            },
            set(owners) - all_advanced,
        )
        self.assertEqual(
            set(CONTEXT_PRIMARY_SETTING_KEYS) & set(owners),
            set(CONTEXT_PRIMARY_SETTING_KEYS),
        )
        self.assertEqual(registry.owner_of("batch_model"), "models")
        self.assertEqual(registry.owner_of("glossary_file"), "project")
        self.assertEqual(registry.owner_of("sync_rag_enabled"), "context")
        self.assertEqual(registry.owner_of("catalog_gemini_models"), "advanced")
        self.assertIsNone(registry.owner_of("game_root"))

    def test_workspace_managed_key_is_not_page_owned(self) -> None:
        self.assertIn("game_root", WORKSPACE_MANAGED_KEYS)
        self.assertNotIn("game_root", PROJECT_CONFIG_KEYS)
        self.assertNotIn("game_root", ADVANCED_CONFIG_KEYS)

    def test_context_page_owns_primary_and_project_analysis_keys(self) -> None:
        registry = build_default_registry()
        context_keys = registry.config_keys_for("context")
        self.assertTrue(CONTEXT_PRIMARY_SETTING_KEYS.issubset(context_keys))
        self.assertIn("rag_enabled", context_keys)
        self.assertIn("context_storage_location", context_keys)
        self.assertEqual(context_keys, CONTEXT_CONFIG_KEYS)

    def test_config_pages_match_current_main_window_contract(self) -> None:
        registry = build_default_registry()
        self.assertEqual(
            registry.config_page_keys(),
            SETTINGS_CONFIG_PAGE_KEYS,
        )
        self.assertEqual(
            SETTINGS_CONFIG_PAGE_KEYS,
            frozenset(
                {
                    "project",
                    "api_keys",
                    "models",
                    "profiles",
                    "litellm",
                    "context",
                    "appearance",
                    "advanced",
                }
            ),
        )

    def test_lazy_attr_map_resolves_to_registered_pages(self) -> None:
        registry = build_default_registry()
        for attr in (
            "_games_registry_panel",
            "batch_model_combo",
            "sync_backend_combo",
            "theme_combo",
        ):
            page_key = registry.page_for_lazy_attr(attr)
            self.assertIsNotNone(page_key)
            self.assertTrue(registry.has(page_key or ""))

    def test_config_keys_for_pages_is_a_union(self) -> None:
        registry = build_default_registry()
        keys = registry.config_keys_for_pages({"models", "appearance"})
        self.assertEqual(keys, frozenset({"sync_model", "sync_embedding_model",
                                           "batch_model", "batch_embedding_model",
                                           "batch_thinking_level", "theme"}))

    def test_duplicate_page_key_is_rejected(self) -> None:
        registry = SettingsPageRegistry(
            [SettingsPageSpec("a", "A", "_build_a")]
        )
        with self.assertRaises(ValueError):
            registry.register(SettingsPageSpec("a", "Other", "_build_other"))

    def test_duplicate_config_key_ownership_is_rejected(self) -> None:
        registry = SettingsPageRegistry(
            [
                SettingsPageSpec(
                    "a", "A", "_build_a", config_keys=frozenset({"shared"})
                )
            ]
        )
        with self.assertRaises(ValueError):
            registry.register(
                SettingsPageSpec(
                    "b", "B", "_build_b", config_keys=frozenset({"shared"})
                )
            )

    def test_unknown_lazy_attr_page_is_rejected(self) -> None:
        registry = SettingsPageRegistry(
            [SettingsPageSpec("a", "A", "_build_a")]
        )
        with self.assertRaises(ValueError):
            registry.set_lazy_attr_map({"widget": "missing"})

    def test_unknown_page_lookup_raises(self) -> None:
        registry = build_default_registry()
        with self.assertRaises(KeyError):
            registry.get("missing")
        with self.assertRaises(KeyError):
            registry.config_keys_for("missing")


if __name__ == "__main__":
    unittest.main()
