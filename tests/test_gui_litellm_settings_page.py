import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gui_qt.litellm_catalog_cache import LiteLLMCatalogCache

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QCompleter, QGroupBox, QLineEdit

    from gui_qt.app import MainWindow
except ImportError as exc:
    MainWindow = None
    QApplication = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiLiteLLMSettingsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls.cache = LiteLLMCatalogCache(
            Path(cls._temp_dir.name) / "litellm_catalog.json"
        )
        cls.window = MainWindow(litellm_catalog_cache=cls.cache)

    @classmethod
    def tearDownClass(cls):
        cls.window.close()
        cls.window.deleteLater()
        cls._app.processEvents()
        cls._temp_dir.cleanup()

    def setUp(self):
        self.cache = LiteLLMCatalogCache(
            Path(self._temp_dir.name) / f"{self._testMethodName}.json"
        )
        self.window._litellm_cache = self.cache
        self.window._populate_litellm_providers((), selected="")
        self.window._set_litellm_models("", ())
        self.window.litellm_api_key_edit.clear()
        self.window._litellm_saved_key_status.clear()
        self.window._refresh_litellm_catalog_status()
        self.window._on_sync_backend_changed(-1)

    @classmethod
    def _load_provider_choices(cls):
        cls.window._populate_litellm_providers(
            ("anthropic", "openai"),
            selected="openai",
        )

    def test_fresh_state_does_not_preselect_provider_or_model(self):
        self.assertEqual(self.window.litellm_provider_combo.count(), 0)
        self.assertEqual(self.window.litellm_provider_combo.currentText(), "")
        self.assertEqual(self.window.litellm_model_combo.count(), 0)
        self.assertEqual(self.window.litellm_model_combo.currentText(), "")
        self.assertFalse(self.window.litellm_refresh_models_btn.isEnabled())
        self.assertFalse(self.window.litellm_api_key_edit.isEnabled())

    def test_litellm_has_independent_settings_page(self):
        self.assertIn("litellm", self.window._settings_nav_rows)
        row = self.window._settings_nav_rows["litellm"]
        page = self.window.settings_stack.widget(row)
        self.assertEqual(page.objectName(), "settings_litellm_scroll")
        titles = {group.title() for group in page.findChildren(QGroupBox)}
        self.assertIn("LiteLLM 同步替代后端", titles)
        self.assertTrue(
            any(title.startswith("Provider 凭据") for title in titles)
        )
        self.assertTrue(self.window.litellm_model_combo.isEditable())
        self.assertTrue(self.window.litellm_provider_combo.isEditable())
        provider_completer = self.window.litellm_provider_combo.completer()
        self.assertEqual(
            provider_completer.caseSensitivity(),
            Qt.CaseSensitivity.CaseInsensitive,
        )
        self.assertEqual(
            provider_completer.filterMode(),
            Qt.MatchFlag.MatchContains,
        )
        self.assertEqual(
            provider_completer.completionMode(),
            QCompleter.CompletionMode.PopupCompletion,
        )
        self.assertEqual(
            self.window.litellm_api_key_edit.echoMode(),
            QLineEdit.EchoMode.Password,
        )
        self.assertEqual(self.window.litellm_refresh_models_btn.text(), "联网加载模型")
        self.assertEqual(
            self.window.litellm_refresh_providers_btn.text(), "联网加载供应商"
        )
        self.assertEqual(self.window.litellm_test_connection_btn.text(), "测试连接")
        self.assertTrue(self.window.litellm_version_label.wordWrap())
        self.assertEqual(self.window.litellm_version_label.minimumWidth(), 0)
        actions = {
            action.objectName(): action
            for action in self.window.litellm_model_combo.lineEdit().actions()
        }
        self.assertIn("combo_popup_action", actions)
        self.assertFalse(actions["combo_popup_action"].icon().isNull())

    def test_common_providers_are_listed_before_dynamic_providers(self):
        self.window._populate_litellm_providers(
            ("zzz_custom", "deepseek", "openrouter", "anthropic", "openai", "aaa_custom"),
        )
        values = tuple(
            self.window.litellm_provider_combo.itemData(index)
            for index in range(self.window.litellm_provider_combo.count())
        )
        self.assertEqual(
            values,
            ("openai", "anthropic", "openrouter", "deepseek", "aaa_custom", "zzz_custom"),
        )
        completer = self.window.litellm_provider_combo.completer()
        completer.setCompletionPrefix("router")
        self.assertEqual(completer.completionCount(), 1)
        self.assertEqual(completer.currentCompletion(), "OpenRouter")


    def test_typed_model_prefix_takes_priority_for_credentials(self):

        self._load_provider_choices()
        self.window.litellm_provider_combo.setCurrentIndex(
            self.window.litellm_provider_combo.findData("openai")
        )
        self.window.litellm_api_key_edit.setText("unsaved-openai-key")
        self.window.litellm_model_combo.setEditText("azure/my-deployment")
        self.assertEqual(self.window._current_litellm_provider(), "azure")
        self.assertEqual(self.window.litellm_api_key_edit.text(), "")
        self.window.litellm_provider_combo.setCurrentIndex(
            self.window.litellm_provider_combo.findData("openai")
        )
        self.window.litellm_model_combo.setEditText("gpt-custom")
        self.assertEqual(self.window._litellm_model_text(), "openai/gpt-custom")

    def test_model_selection_enables_test_connection_button(self):
        litellm_index = self.window.sync_backend_combo.findData("litellm")
        self.window.sync_backend_combo.setCurrentIndex(litellm_index)
        self._load_provider_choices()
        self.window.litellm_provider_combo.setCurrentIndex(
            self.window.litellm_provider_combo.findData("openai")
        )
        self.window.litellm_model_combo.setEditText("")
        self.window._on_sync_backend_changed(-1)
        self.assertFalse(self.window.litellm_test_connection_btn.isEnabled())

        self.window.litellm_model_combo.setEditText("openai/gpt-test")
        self.assertTrue(self.window.litellm_test_connection_btn.isEnabled())

        self.window.litellm_model_combo.setEditText("")
        self.assertFalse(self.window.litellm_test_connection_btn.isEnabled())

    def test_model_prefix_switch_reloads_cached_models_and_gates_credentials(self):
        litellm_index = self.window.sync_backend_combo.findData("litellm")
        self.window.sync_backend_combo.setCurrentIndex(litellm_index)
        self.cache.update_models(
            "openai",
            ["openai/gpt-a", "openai/gpt-b"],
            source="online",
        )
        self.cache.update_models(
            "ollama",
            ["ollama/llama3", "ollama/mistral"],
            source="ollama",
        )
        self.window._populate_litellm_providers(
            ("openai", "ollama"),
            selected="",
        )
        self.window.litellm_provider_combo.setCurrentIndex(
            self.window.litellm_provider_combo.findData("openai")
        )
        self.window.litellm_api_key_edit.setText("unsaved-openai-key")
        self.assertTrue(self.window.litellm_api_key_edit.isEnabled())
        openai_models = {
            self.window.litellm_model_combo.itemText(index)
            for index in range(self.window.litellm_model_combo.count())
        }
        self.assertEqual(openai_models, {"openai/gpt-a", "openai/gpt-b"})

        self.window.litellm_model_combo.setEditText("ollama/llama3")

        self.assertEqual(self.window._current_litellm_provider(), "ollama")
        self.assertEqual(self.window.litellm_api_key_edit.text(), "")
        self.assertFalse(self.window.litellm_api_key_edit.isEnabled())
        model_items = {
            self.window.litellm_model_combo.itemText(index)
            for index in range(self.window.litellm_model_combo.count())
        }
        self.assertEqual(model_items, {"ollama/llama3", "ollama/mistral"})
        self.assertEqual(self.window.litellm_model_combo.currentText(), "ollama/llama3")
        self.assertTrue(self.window.litellm_test_connection_btn.isEnabled())

    def test_display_label_free_text_resolves_to_provider_id(self):
        self.window._populate_litellm_providers(("openai", "ollama"), selected="")
        self.window.litellm_provider_combo.setCurrentIndex(-1)
        self.window.litellm_provider_combo.setEditText("Ollama（本地）")
        self.assertEqual(self.window._litellm_provider_combo_value(), "ollama")
        self.window.litellm_provider_combo.setEditText("Google Gemini")
        self.assertEqual(self.window._litellm_provider_combo_value(), "gemini")

    def test_switching_provider_does_not_relabel_bare_model_as_new_provider(self):
        self._load_provider_choices()
        openai_index = self.window.litellm_provider_combo.findData("openai")
        anthropic_index = self.window.litellm_provider_combo.findData("anthropic")
        self.window.litellm_provider_combo.setCurrentIndex(openai_index)
        try:
            self.window.litellm_model_combo.setEditText("gpt-custom")
            self.window.litellm_provider_combo.setCurrentIndex(anthropic_index)

            self.assertEqual(self.window.litellm_model_combo.currentText(), "")
        finally:

            self.window.litellm_provider_combo.setCurrentIndex(openai_index)

    def test_unchanged_provider_focus_out_preserves_unsaved_fields(self):
        self._load_provider_choices()
        self.window.litellm_api_key_edit.setText("unsaved-key")
        self.window.litellm_model_combo.setEditText("gpt-custom")

        self.window._on_litellm_provider_changed()

        self.assertEqual(self.window.litellm_api_key_edit.text(), "unsaved-key")
        self.assertEqual(self.window.litellm_model_combo.currentText(), "gpt-custom")

    def test_cancel_provider_discards_typed_key_but_does_not_delete_credential(self):
        self._load_provider_choices()
        self.window.litellm_provider_combo.setCurrentIndex(
            self.window.litellm_provider_combo.findData("openai")
        )
        self.window.litellm_api_key_edit.setText("unsaved-secret")
        with mock.patch("gui_qt.app.delete_provider_api_key") as delete_key:
            self.window._on_clear_litellm_provider()

        self.assertEqual(self.window._current_litellm_provider(), "")
        self.assertEqual(self.window.litellm_model_combo.currentText(), "")
        self.assertEqual(self.window.litellm_api_key_edit.text(), "")
        self.assertEqual(self.cache.selected_provider, "")
        delete_key.assert_not_called()

    def test_catalog_refresh_preserves_custom_model(self):
        self._load_provider_choices()
        openai_index = self.window.litellm_provider_combo.findData("openai")
        self.window.litellm_provider_combo.setCurrentIndex(openai_index)
        self.window.litellm_model_combo.setEditText("gpt-custom")
        self.window._litellm_catalog_worker = None

        self.window._on_litellm_models_loaded(
            "openai",
            ("openai/gpt-current",),
            None,
            "online",
        )

        self.assertEqual(self.window.litellm_model_combo.currentText(), "gpt-custom")

    def test_catalog_failure_keeps_cached_models(self):
        self.cache.update_models(
            "openai",
            ["openai/cached-model"],
            source="online",
            litellm_version="1.2.3",
        )
        self._load_provider_choices()
        self.window.litellm_provider_combo.setCurrentIndex(
            self.window.litellm_provider_combo.findData("openai")
        )
        self.window._litellm_catalog_worker = None

        with mock.patch("gui_qt.app.QMessageBox.warning"):
            self.window._on_litellm_models_loaded(
                "openai",
                (),
                "offline",
            )

        self.assertEqual(
            self.cache.models("openai").values,
            ("openai/cached-model",),
        )

    def test_configured_model_takes_priority_over_cached_selection(self):
        self.cache.update_models(
            "anthropic",
            ["anthropic/cached-model"],
            source="online",
        )
        self.cache.select_provider("anthropic")
        self.cache.select_model("anthropic", "anthropic/cached-model")

        self.window._restore_configured_litellm_model("openai/configured-model")

        self.assertEqual(self.window._current_litellm_provider(), "openai")
        self.assertEqual(
            self.window.litellm_model_combo.currentText(),
            "openai/configured-model",
        )

    def test_saved_credential_is_reported_as_environment_override(self):
        self._load_provider_choices()
        with (
            mock.patch("gui_qt.app.load_provider_api_key", return_value="saved") as load_key,
            mock.patch.dict("gui_qt.app.os.environ", {"OPENAI_API_KEY": "env"}, clear=True),
        ):
            self.window.litellm_provider_combo.setCurrentIndex(
                self.window.litellm_provider_combo.findData("openai")
            )
            self.window.litellm_model_combo.setEditText("openai/test")
            self.window._refresh_litellm_credential_status()
            self.window._refresh_litellm_credential_status()
            self.window._refresh_litellm_credential_status()
        status = self.window.litellm_credential_status_label.text()
        self.assertIn("系统凭据管理器中已保存密钥", status)
        self.assertIn("OPENAI_API_KEY", status)
        load_key.assert_called_once_with("openai")

    def test_gemini_key_page_does_not_contain_litellm_controls(self):
        row = self.window._settings_nav_rows["api_keys"]
        page = self.window.settings_stack.widget(row)
        titles = {group.title() for group in page.findChildren(QGroupBox)}
        self.assertEqual(titles, {"Gemini API Key"})
        self.assertNotIn(
            self.window.install_litellm_btn,
            page.findChildren(type(self.window.install_litellm_btn)),
        )

    def test_empty_litellm_model_cannot_be_saved(self):
        self.window.sync_backend_combo.setCurrentIndex(
            self.window.sync_backend_combo.findData("litellm")
        )
        self.window.litellm_model_combo.setEditText("")
        with (
            mock.patch.object(
                self.window.state,
                "get_game_root",
                return_value=Path("C:/Game/work"),
            ),
            mock.patch.object(
                self.window.state,
                "load_translator_config",
                return_value={"sync": {}, "batch": {}},
            ),
            mock.patch.object(self.window.state, "save_translator_config") as save_config,
            mock.patch("gui_qt.app.QMessageBox.information") as information,
        ):
            saved = self.window._on_save_config()
        self.assertFalse(saved)
        save_config.assert_not_called()
        information.assert_called_once()

    def test_worker_completion_reapplies_current_backend_gating(self):
        gemini_index = self.window.sync_backend_combo.findData("gemini")
        litellm_index = self.window.sync_backend_combo.findData("litellm")
        self.window.sync_backend_combo.setCurrentIndex(gemini_index)
        try:
            self.window._litellm_connection_worker = None
            self.window._on_litellm_connection_tested(True, "连接成功")
            self.assertFalse(self.window.litellm_test_connection_btn.isEnabled())

            self.window._litellm_catalog_worker = None
            with mock.patch("gui_qt.app.QMessageBox.warning"):
                self.window._on_litellm_models_loaded(
                    "openai",
                    (),
                    "catalog failed",
                )
            self.assertFalse(self.window.litellm_refresh_models_btn.isEnabled())
        finally:
            self.window.sync_backend_combo.setCurrentIndex(litellm_index)

    def test_models_page_is_gemini_only(self):
        # Materialize the lazy models settings page before inspecting widgets.
        _ = self.window.sync_model_combo
        row = self.window._settings_nav_rows["models"]
        page = self.window.settings_stack.widget(row)
        titles = {group.title() for group in page.findChildren(QGroupBox)}
        self.assertIn("Gemini 同步翻译", titles)
        self.assertIn("批量离线翻译", titles)
        self.assertNotIn("LiteLLM 同步替代后端", titles)
        self.assertFalse(self.window.sync_model_combo.isEditable())

        gemini_index = self.window.sync_backend_combo.findData("gemini")
        litellm_index = self.window.sync_backend_combo.findData("litellm")
        self.window.sync_backend_combo.setCurrentIndex(gemini_index)
        self.window.sync_backend_combo.setCurrentIndex(litellm_index)
        try:
            self.assertFalse(self.window.sync_model_combo.isEnabled())
            self.assertTrue(self.window.sync_embedding_combo.isEnabled())
            self.assertIn("LiteLLM", self.window.sync_model_combo.toolTip())
        finally:
            self.window.sync_backend_combo.setCurrentIndex(gemini_index)
        self.assertTrue(self.window.sync_model_combo.isEnabled())
        self.assertEqual(self.window.sync_model_combo.toolTip(), "")


if __name__ == "__main__":
    unittest.main()
