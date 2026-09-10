"""GUI integration tests for the #202 Settings coordinator."""
from __future__ import annotations

import unittest
import warnings
from pathlib import Path
from unittest import mock

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
    from gui_qt.settings.page_contract import SettingsIssue
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    MainWindow = None  # type: ignore[assignment,misc]
    SettingsIssue = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


def _process(app: QApplication, rounds: int = 8) -> None:
    for _ in range(rounds):
        app.processEvents()


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiSettingsCoordinatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()
        _process(self._app, 2)

    def _activate_window(self) -> None:
        self.window.show()
        self.window.activateWindow()
        if QApplication.focusWidget() is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                QApplication.setActiveWindow(self.window)

    def test_litellm_page_is_migrated_settings_page(self) -> None:
        from gui_qt.settings.litellm_page import LiteLLMSettingsPage

        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value={"sync": {}, "batch": {}},
        ):
            self.window._ensure_settings_page("litellm")
        page = self.window._settings_coordinator.page("litellm")
        self.assertIsInstance(page, LiteLLMSettingsPage)
        collected = self.window._settings_coordinator.collect()
        self.assertTrue(
            set(collected).issubset(
                self.window._settings_registry.config_keys_for("litellm")
            )
        )

    def test_context_page_is_migrated_settings_page(self) -> None:
        from gui_qt.settings.context_page import ContextSettingsPage

        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value={
                "sync": {"rag": {"enabled": True}},
                "batch": {},
            },
        ):
            self.window._ensure_settings_page("context")
        page = self.window._settings_coordinator.page("context")
        self.assertIsInstance(page, ContextSettingsPage)
        collected = self.window._settings_coordinator.collect()
        self.assertEqual(
            set(collected),
            self.window._settings_registry.config_keys_for("context"),
        )
        self.assertTrue(collected["sync_rag_enabled"])

    def test_context_page_first_open_after_other_pages_loads_disk_values(
        self,
    ) -> None:
        config = {
            "batch": {
                "rag": {"enabled": True, "bootstrap_on_build": False},
                "project_analysis": {
                    "enabled": True,
                    "inject_published_brief": True,
                    "model": "analysis-model",
                    "thinking_level": "high",
                },
                "story_memory": {"enabled": True},
            },
            "sync": {
                "rag": {"enabled": True},
                "story_memory": {"enabled": True},
                "project_analysis": {"inject_published_brief": True},
            },
            "context_storage": {"location": "game"},
        }
        with (
            mock.patch.object(
                self.window.state,
                "load_translator_config",
                return_value=config,
            ),
            mock.patch.object(
                self.window.state,
                "get_game_root",
                return_value=None,
            ),
        ):
            self.window._ensure_settings_page("advanced")
            self.assertFalse(self.window._settings_coordinator.is_built("context"))
            self.window._ensure_settings_page("context")
        collected = self.window._settings_coordinator.collect()
        self.assertTrue(collected["rag_enabled"])
        self.assertFalse(collected["bootstrap_on_build"])
        self.assertEqual(collected["context_storage_location"], "game")
        self.assertTrue(collected["sync_rag_enabled"])
        self.assertTrue(collected["sync_project_analysis_inject_enabled"])
        self.assertEqual(collected["batch_project_analysis_model"], "analysis-model")
        self.assertEqual(
            collected["batch_project_analysis_thinking_level"], "high"
        )

    def test_shortcuts_page_is_migrated_settings_page(self) -> None:
        from gui_qt.settings.shortcuts_page import ShortcutsSettingsPage

        self.window._ensure_settings_page("shortcuts")
        page = self.window._settings_coordinator.page("shortcuts")
        self.assertIsInstance(page, ShortcutsSettingsPage)
        collected = self.window._settings_coordinator.collect()
        self.assertEqual(collected, {})
        self.assertEqual(
            set(collected),
            self.window._settings_registry.config_keys_for("shortcuts"),
        )

    def test_api_keys_page_is_migrated_settings_page(self) -> None:
        from gui_qt.settings.api_keys_page import ApiKeysSettingsPage

        self.window._ensure_settings_page("api_keys")
        page = self.window._settings_coordinator.page("api_keys")
        self.assertIsInstance(page, ApiKeysSettingsPage)
        collected = self.window._settings_coordinator.collect()
        self.assertEqual(collected, {})
        self.assertEqual(
            set(collected),
            self.window._settings_registry.config_keys_for("api_keys"),
        )
        self.assertEqual(self.window.api_btn.text(), "管理 Gemini API Key")
        self.assertEqual(
            self.window.litellm_keys_manage_btn.text(), "管理 Provider Key"
        )

    def test_extensions_page_is_migrated_settings_page(self) -> None:
        from gui_qt.settings.extensions_page import ExtensionsSettingsPage

        self.window._ensure_settings_page("extensions")
        page = self.window._settings_coordinator.page("extensions")
        self.assertIsInstance(page, ExtensionsSettingsPage)
        collected = self.window._settings_coordinator.collect()
        self.assertEqual(collected, {})
        self.assertEqual(
            set(collected),
            self.window._settings_registry.config_keys_for("extensions"),
        )
        self.assertEqual(
            self.window.relation_analyzer_install_btn.objectName(),
            "relation_analyzer_install_btn",
        )

    def test_workspace_page_is_migrated_settings_page(self) -> None:
        from gui_qt.settings.workspace_page import WorkspaceSettingsPage

        self.window._ensure_settings_page("workspace")
        page = self.window._settings_coordinator.page("workspace")
        self.assertIsInstance(page, WorkspaceSettingsPage)
        collected = self.window._settings_coordinator.collect()
        self.assertEqual(collected, {})
        self.assertEqual(
            set(collected),
            self.window._settings_registry.config_keys_for("workspace"),
        )
        self.assertEqual(
            self.window._games_registry_panel.objectName(),
            "games_registry_panel",
        )

    def test_appearance_page_is_migrated_settings_page(self) -> None:
        from gui_qt.settings.appearance_page import AppearanceSettingsPage

        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value={"gui": {"theme": "dark"}},
        ):
            self.window._ensure_settings_page("appearance")
        page = self.window._settings_coordinator.page("appearance")
        self.assertIsInstance(page, AppearanceSettingsPage)
        collected = self.window._settings_coordinator.collect()
        self.assertEqual(
            set(collected),
            self.window._settings_registry.config_keys_for("appearance"),
        )
        self.assertEqual(collected["theme"], "dark")

    def test_appearance_page_first_open_after_other_pages_loads_disk_values(
        self,
    ) -> None:
        config = {"gui": {"theme": "light"}}
        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value=config,
        ):
            self.window._ensure_settings_page("advanced")
            self.assertFalse(self.window._settings_coordinator.is_built("appearance"))
            self.window._ensure_settings_page("appearance")
        collected = self.window._settings_coordinator.collect()
        self.assertEqual(collected["theme"], "light")

    def test_opening_other_page_keeps_unsaved_theme_preview(self) -> None:
        config = {"gui": {"theme": "system"}}
        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value=config,
        ):
            self.window._ensure_settings_page("appearance")
            page = self.window._settings_coordinator.page("appearance")
            page.theme_combo.setCurrentIndex(page.theme_combo.findData("dark"))
            self.assertEqual(page.collect()["theme"], "dark")
            self.assertEqual(self.window._theme_preference, "dark")
            self.window._ensure_settings_page("advanced")
        self.assertEqual(page.collect()["theme"], "dark")
        self.assertEqual(self.window._theme_preference, "dark")

    def test_materializing_other_pages_restores_unsaved_theme_preview(self) -> None:
        config = {"gui": {"theme": "system"}}
        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value=config,
        ):
            self.window._ensure_settings_page("appearance")
            page = self.window._settings_coordinator.page("appearance")
            page.theme_combo.setCurrentIndex(page.theme_combo.findData("dark"))
            self.assertEqual(self.window._theme_preference, "dark")
            self.window._ensure_settings_pages_for_config()
        self.assertEqual(page.collect()["theme"], "dark")
        self.assertEqual(self.window._theme_preference, "dark")

    def test_appearance_reset_syncs_host_theme_preference(self) -> None:
        config = {"gui": {"theme": "system"}}
        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value=config,
        ):
            self.window._ensure_settings_page("appearance")
            page = self.window._settings_coordinator.page("appearance")
            page.theme_combo.setCurrentIndex(page.theme_combo.findData("dark"))
            self.assertEqual(self.window._theme_preference, "dark")
            self.window._settings_coordinator.reset(pages={"appearance"})
        self.assertEqual(page.collect()["theme"], "system")
        self.assertEqual(self.window._theme_preference, "system")

    def test_advanced_page_is_migrated_settings_page(self) -> None:
        from gui_qt.settings.advanced_page import AdvancedSettingsPage

        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value={"sync": {"chunk_size": 42}},
        ):
            self.window._ensure_settings_page("advanced")
        page = self.window._settings_coordinator.page("advanced")
        self.assertIsInstance(page, AdvancedSettingsPage)
        collected = self.window._settings_coordinator.collect()
        self.assertEqual(
            set(collected),
            self.window._settings_registry.config_keys_for("advanced"),
        )
        self.assertEqual(collected["sync_chunk_size"], 42)

    def test_advanced_page_first_open_after_other_pages_loads_disk_values(
        self,
    ) -> None:
        config = {
            "sync": {"chunk_size": 42},
            "model_catalog": {"gemini": ["gemini-experimental-foo"]},
        }
        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value=config,
        ):
            self.window._ensure_settings_page("project")
            self.assertFalse(self.window._settings_coordinator.is_built("advanced"))
            self.window._ensure_settings_page("advanced")
        collected = self.window._settings_coordinator.collect()
        self.assertEqual(collected["sync_chunk_size"], 42)
        self.assertEqual(collected["catalog_gemini_models"], ["gemini-experimental-foo"])

    def test_project_page_is_migrated_settings_page(self) -> None:
        from gui_qt.settings.project_page import ProjectSettingsPage

        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value={"tl_subdir": "custom_tl"},
        ):
            self.window._ensure_settings_page("project")
        page = self.window._settings_coordinator.page("project")
        self.assertIsInstance(page, ProjectSettingsPage)
        collected = self.window._settings_coordinator.collect()
        self.assertEqual(
            set(collected),
            self.window._settings_registry.config_keys_for("project"),
        )
        self.assertEqual(collected["tl_subdir"], "custom_tl")

    def test_models_page_is_migrated_settings_page(self) -> None:
        from gui_qt.settings.models_page import ModelsSettingsPage

        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value={"sync": {}, "batch": {}},
        ):
            self.window._ensure_settings_page("models")
        page = self.window._settings_coordinator.page("models")
        self.assertIsInstance(page, ModelsSettingsPage)
        collected = self.window._settings_coordinator.collect()
        self.assertEqual(
            set(collected),
            self.window._settings_registry.config_keys_for("models"),
        )

    def test_coordinator_tracks_lazy_build_only_target_page(self) -> None:
        self.window._focus_settings_section("advanced")
        coordinator = self.window._settings_coordinator
        self.assertEqual(coordinator.built_keys(), ("advanced",))
        self.assertTrue(coordinator.is_built("advanced"))
        self.assertFalse(coordinator.is_built("models"))
        self.assertFalse(coordinator.is_built("workspace"))

    def test_project_page_first_open_loads_owned_fields(self) -> None:
        config = {
            "tl_subdir": "custom_tl",
            "glossary_file": "custom_glossary.json",
            "prepare": {"enabled": True},
        }
        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value=config,
        ):
            self.window._ensure_settings_page("project")
        widgets = self.window._advanced_setting_widgets
        self.assertEqual(widgets["tl_subdir"].text(), "custom_tl")
        self.assertEqual(
            widgets["glossary_file"].text(),
            "custom_glossary.json",
        )
        self.assertTrue(widgets["prepare_enabled"].isChecked())
        self.assertTrue(self.window._settings_coordinator.is_loaded("project"))

    def test_materializing_other_pages_preserves_project_edits(self) -> None:
        config = {"tl_subdir": "saved_tl", "prepare": {"enabled": False}}
        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value=config,
        ):
            self.window._ensure_settings_page("project")
        widget = self.window._advanced_setting_widgets["tl_subdir"]
        self.assertEqual(widget.text(), "saved_tl")
        widget.setText("unsaved_tl")
        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value=config,
        ):
            self.window._ensure_settings_pages_for_config()
        self.assertEqual(widget.text(), "unsaved_tl")
        self.assertTrue(self.window._config_tab_has_unsaved_changes())
        current = self.window._current_config_ui_snapshot()
        self.assertTrue(self.window._settings_coordinator.is_dirty(current))
        self.assertEqual(
            self.window._settings_coordinator.baseline().get("tl_subdir"),
            "saved_tl",
        )

    def test_coordinator_collect_returns_owned_page_keys(self) -> None:
        self.window._focus_settings_section("models")
        collected = self.window._settings_coordinator.collect()
        self.assertTrue(collected)
        self.assertTrue(
            set(collected).issubset(
                self.window._settings_registry.config_keys_for("models")
            )
        )
        self.assertNotIn("theme", collected)
        self.assertNotIn("glossary_file", collected)

    def test_coordinator_reset_reloads_page_from_disk(self) -> None:
        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value={"tl_subdir": "saved_tl"},
        ):
            self.window._ensure_settings_page("project")
        widget = self.window._advanced_setting_widgets["tl_subdir"]
        widget.setText("unsaved_tl")
        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value={"tl_subdir": "saved_tl"},
        ):
            self.window._settings_coordinator.reset(pages={"project"})
        self.assertEqual(widget.text(), "saved_tl")

    def test_coordinator_load_applies_snapshot_to_legacy_page(self) -> None:
        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value={"tl_subdir": "disk_tl"},
        ):
            self.window._ensure_settings_page("project")
        # coordinator.load() must use the in-memory snapshot, not re-read disk.
        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            side_effect=AssertionError("coordinator.load must not read disk"),
        ):
            self.window._settings_coordinator.load(
                {"tl_subdir": "snapshot_tl"},
                pages={"project"},
            )
        self.assertEqual(
            self.window._advanced_setting_widgets["tl_subdir"].text(),
            "snapshot_tl",
        )

    def test_coordinator_focus_issue_populates_and_switches_page(self) -> None:
        self._activate_window()
        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value={"glossary_file": "focused.json"},
        ):
            issue = SettingsIssue("project", "glossary_file", "invalid")
            self.assertTrue(
                self.window._settings_coordinator.focus_issue(issue)
            )
        self.assertEqual(
            self.window.settings_nav.currentRow(),
            self.window._settings_nav_rows["project"],
        )
        widget = self.window._advanced_setting_widgets["glossary_file"]
        self.assertEqual(widget.text(), "focused.json")
        self.assertIs(QApplication.focusWidget(), widget)

    def test_first_open_settings_pages_inherit_task_lock(self) -> None:
        empty = {"sync": {}, "batch": {}}
        self.window._set_task_running(True)
        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value=empty,
        ):
            self.window._ensure_settings_page("models")
            self.window._ensure_settings_page("context")
            self.window._ensure_settings_page("project")
            self.window._ensure_settings_page("appearance")
        models = self.window._settings_coordinator.page("models")
        context = self.window._settings_coordinator.page("context")
        project = self.window._settings_coordinator.page("project")
        appearance = self.window._settings_coordinator.page("appearance")
        self.assertFalse(models.sync_model_combo.isEnabled())
        self.assertFalse(context.rag_enabled_cb.isEnabled())
        self.assertFalse(project._prepare_renpy_sdk_browse_btn.isEnabled())
        self.assertFalse(appearance.theme_combo.isEnabled())

    def test_deleting_last_provider_on_first_save_clears_disk(self) -> None:
        provider = {
            "id": "opencode-go",
            "base_url": "https://opencode.ai/zen/go/v1",
        }
        config = {
            "sync": {"backend": "gemini", "custom_litellm_providers": [provider]},
            "batch": {},
        }
        with mock.patch.object(
            self.window.state,
            "load_translator_config",
            return_value=config,
        ):
            self.window._ensure_settings_page("litellm")
        self.assertEqual(
            set(self.window._settings_pages_built).intersection(
                self.window._settings_registry.config_page_keys()
            ),
            {"litellm"},
        )
        self.window.custom_provider_table.selectRow(0)
        with (
            mock.patch(
                "gui_qt.app.message_box_question",
                return_value="yes",
            ),
            mock.patch("gui_qt.app.load_provider_api_key", return_value=""),
            mock.patch("gui_qt.app.load_provider_key_store"),
        ):
            self.window._on_delete_custom_litellm_provider()
        self.assertEqual(self.window._custom_litellm_providers, {})
        self.assertTrue(self.window._custom_litellm_providers_modified)

        with (
            mock.patch.object(
                self.window.state,
                "get_game_root",
                return_value=Path("C:/Game/work"),
            ),
            mock.patch.object(
                self.window.state,
                "load_translator_config",
                return_value=config,
            ),
            mock.patch.object(
                self.window.state,
                "save_translator_config",
            ) as save_config,
            mock.patch("gui_qt.app.message_box_information"),
            mock.patch("gui_qt.app.load_provider_api_key", return_value=""),
            mock.patch("gui_qt.app.load_provider_key_store"),
            mock.patch("project_context_settings.save_project_context_settings"),
        ):
            saved = self.window._on_save_config()
        self.assertTrue(saved)
        saved_payload = save_config.call_args.args[0]
        self.assertNotIn("custom_litellm_providers", saved_payload["sync"])


if __name__ == "__main__":
    unittest.main()
