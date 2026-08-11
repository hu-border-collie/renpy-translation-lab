import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gui_qt.litellm_catalog_cache import LiteLLMCatalogCache
from litellm_provider_config import custom_provider_registry

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QCompleter, QGroupBox, QLineEdit

    from gui_qt.app import _RETIRED_LITELLM_WARMUP_WORKERS, MainWindow
except ImportError as exc:
    MainWindow = None
    _RETIRED_LITELLM_WARMUP_WORKERS = None
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
        # Build the LiteLLM page once against an empty config so tests never
        # load a developer's real translator_config.json (which may register
        # custom providers and break the fresh-state / duplicate-id tests).
        with mock.patch.object(
            cls.window.state,
            "load_translator_config",
            return_value={"sync": {}, "batch": {}},
        ):
            cls.window._ensure_settings_page("litellm")

    @classmethod
    def tearDownClass(cls):
        # Tests intentionally leave some settings dirty; avoid opening the
        # interactive unsaved-config prompt during headless teardown.
        with mock.patch.object(
            cls.window,
            "_confirm_unsaved_config_before_close",
            return_value=True,
        ):
            cls.window.close()
        cls.window.deleteLater()
        cls._app.processEvents()
        cls._temp_dir.cleanup()

    def setUp(self):
        _RETIRED_LITELLM_WARMUP_WORKERS.clear()
        self.cache = LiteLLMCatalogCache(
            Path(self._temp_dir.name) / f"{self._testMethodName}.json"
        )
        self.window._litellm_cache = self.cache
        self.window._populate_litellm_providers((), selected="")
        self.window._set_litellm_models("", ())
        self.window._litellm_saved_key_status.clear()
        self.window._custom_litellm_providers = {}
        self.window._custom_litellm_providers_load_error = ""
        self.window._custom_litellm_providers_modified = False
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
        self.assertFalse(self.window.litellm_manage_keys_btn.isEnabled())

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
        self.assertEqual(self.window.litellm_manage_keys_btn.text(), "管理密钥…")
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
        self.window.litellm_model_combo.setEditText("azure/my-deployment")
        self.assertEqual(self.window._current_litellm_provider(), "azure")
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
        self.window._on_sync_backend_changed(-1)
        self.assertTrue(self.window.litellm_manage_keys_btn.isEnabled())
        openai_models = {
            self.window.litellm_model_combo.itemText(index)
            for index in range(self.window.litellm_model_combo.count())
        }
        self.assertEqual(openai_models, {"openai/gpt-a", "openai/gpt-b"})

        self.window.litellm_model_combo.setEditText("ollama/llama3")

        self.assertEqual(self.window._current_litellm_provider(), "ollama")
        self.assertFalse(self.window.litellm_manage_keys_btn.isEnabled())
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

    def test_unchanged_provider_focus_out_preserves_model_text(self):
        self._load_provider_choices()
        self.window.litellm_model_combo.setEditText("gpt-custom")

        self.window._on_litellm_provider_changed()

        self.assertEqual(self.window.litellm_model_combo.currentText(), "gpt-custom")

    def test_cancel_provider_clears_selection_without_touching_keyring(self):
        self._load_provider_choices()
        self.window.litellm_provider_combo.setCurrentIndex(
            self.window.litellm_provider_combo.findData("openai")
        )
        with mock.patch("gui_qt.app.store_provider_key_store") as store_keys:
            self.window._on_clear_litellm_provider()

        self.assertEqual(self.window._current_litellm_provider(), "")
        self.assertEqual(self.window.litellm_model_combo.currentText(), "")
        self.assertEqual(self.cache.selected_provider, "")
        store_keys.assert_not_called()

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

        with mock.patch("gui_qt.app.message_box_warning"):
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
        from litellm_provider_config import ProviderApiKeyStore

        self._load_provider_choices()
        store = ProviderApiKeyStore(
            keys=("sk-test-secret-value", "sk-second-zzzz"),
            active_index=0,
        )
        with (
            mock.patch(
                "gui_qt.app.load_provider_key_store",
                return_value=store,
            ) as load_store,
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
        self.assertIn("已保存 2 把密钥", status)
        self.assertIn("********alue", status)
        self.assertIn("********zzzz", status)
        self.assertNotIn("sk-test-secret-value", status)
        self.assertIn("OPENAI_API_KEY", status)
        load_store.assert_called_once_with("openai")

    def test_manage_keys_opens_dialog_and_saves_multi_key_store(self):
        from litellm_provider_config import ProviderApiKeyStore
        from PySide6.QtWidgets import QDialog

        self._load_provider_choices()
        self.window.litellm_provider_combo.setCurrentIndex(
            self.window.litellm_provider_combo.findData("openai")
        )
        dialog = mock.Mock()
        dialog.exec.return_value = QDialog.DialogCode.Accepted
        dialog.result_keys.return_value = ["key-a", "key-b"]
        dialog.result_active_index.return_value = 1
        with (
            mock.patch(
                "gui_qt.app.load_provider_key_store",
                return_value=ProviderApiKeyStore(keys=("old",), active_index=0),
            ),
            mock.patch("gui_qt.app.ApiKeyDialog", return_value=dialog) as dialog_cls,
            mock.patch("gui_qt.app.store_provider_key_store") as store_keys,
        ):
            self.window._on_manage_litellm_keys()
        dialog_cls.assert_called_once()
        self.assertTrue(dialog_cls.call_args.kwargs.get("support_active_key"))
        store_keys.assert_called_once()
        self.assertEqual(store_keys.call_args.args[0], "openai")
        self.assertEqual(store_keys.call_args.args[1], ["key-a", "key-b"])
        self.assertEqual(store_keys.call_args.kwargs.get("active_index"), 1)

    def test_missing_key_prompts_before_model_catalog_load(self):
        self.window._populate_litellm_providers(("deepseek",), selected="deepseek")
        self.window.sync_backend_combo.setCurrentIndex(
            self.window.sync_backend_combo.findData("litellm")
        )
        self.window._on_sync_backend_changed(-1)
        self.assertEqual(self.window._current_litellm_provider(), "deepseek")
        with (
            mock.patch("gui_qt.app.load_provider_api_key", return_value=""),
            mock.patch("gui_qt.app.message_box_question") as question,
            mock.patch("gui_qt.app.LiteLLMModelCatalogWorker") as worker_cls,
        ):
            question.return_value = "no"
            self.window._on_refresh_litellm_models()
            worker_cls.assert_not_called()
            question.assert_called_once()
            self.assertIn("API Key", question.call_args.args[1])

            question.reset_mock()
            worker_cls.reset_mock()
            question.return_value = "yes"
            worker = mock.Mock()
            worker_cls.return_value = worker
            self.window._on_refresh_litellm_models()
            worker_cls.assert_called_once()
            self.assertEqual(worker_cls.call_args.args[0], "deepseek")
            self.assertEqual(worker_cls.call_args.kwargs.get("api_key"), "")
            worker.start.assert_called_once()

    def test_missing_key_shows_catalog_tip_for_official_providers(self):
        self.window._populate_litellm_providers(("deepseek",), selected="deepseek")
        self.assertEqual(self.window._litellm_provider_combo_value(), "deepseek")
        with mock.patch("gui_qt.app.load_provider_api_key", return_value=""):
            self.window._refresh_litellm_catalog_status()
        text = self.window.litellm_catalog_status_label.text()
        self.assertIn("官方列表需先保存 API Key", text)
        self.assertIn("DeepSeek", text)

    def test_keys_page_hosts_gemini_and_litellm_provider_sections(self):
        row = self.window._settings_nav_rows["api_keys"]
        page = self.window.settings_stack.widget(row)
        titles = {group.title() for group in page.findChildren(QGroupBox)}
        self.assertEqual(titles, {"Gemini API Key", "LiteLLM Provider 密钥"})
        self.assertEqual(self.window.api_btn.text(), "管理 Gemini API Key")
        self.assertEqual(self.window.litellm_keys_manage_btn.text(), "管理 Provider Key")
        self.assertTrue(self.window.litellm_keys_provider_combo.isEditable())
        ids = {
            self.window.litellm_keys_provider_combo.itemData(index)
            for index in range(self.window.litellm_keys_provider_combo.count())
        }
        self.assertIn("openai", ids)
        self.assertIn("gemini", ids)
        self.assertIn("azure", ids)
        self.assertIn("vertex_ai", ids)
        self.assertNotIn("ollama", ids)
        self.assertNotIn(
            self.window.install_litellm_btn,
            page.findChildren(type(self.window.install_litellm_btn)),
        )

    def test_keys_page_provider_list_includes_online_catalog_cache(self):
        self.cache.update_providers(
            ("custom_vendor", "openai", "mistral"),
            source="online",
        )
        self.window._populate_litellm_keys_provider_combo()
        ids = {
            self.window.litellm_keys_provider_combo.itemData(index)
            for index in range(self.window.litellm_keys_provider_combo.count())
        }
        self.assertIn("custom_vendor", ids)
        self.assertIn("mistral", ids)
        self.assertIn("deepseek", ids)

    def test_keys_page_free_typed_provider_uses_current_text(self):
        combo = self.window.litellm_keys_provider_combo
        combo.setCurrentIndex(combo.findData("openai"))
        combo.setEditText("custom_vendor")

        self.assertEqual(self.window._litellm_keys_page_provider(), "custom_vendor")

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
            mock.patch("gui_qt.app.message_box_information") as information,
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
            with mock.patch("gui_qt.app.message_box_warning"):
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

    def _dialog_result(self, *, provider_id="opencode-go"):
        from PySide6.QtWidgets import QDialog

        dialog = mock.Mock()
        dialog.exec.return_value = QDialog.DialogCode.Accepted
        entry = {
            "id": provider_id,
            "label": "OpenCode Go",
            "base_url": "https://opencode.ai/zen/go/v1",
            "models_url": "https://opencode.ai/zen/go/v1/models",
            "api_key_env": "OPENCODE_GO_API_KEY",
        }
        dialog.result_provider.return_value = entry
        return dialog, entry

    def test_add_custom_provider_populates_registry_and_dropdowns(self):
        self.window._ensure_settings_page("litellm")
        dialog, _entry = self._dialog_result()
        with (
            mock.patch(
                "gui_qt.app.CustomLiteLLMProviderDialog",
                return_value=dialog,
            ) as dialog_cls,
            mock.patch("gui_qt.app.load_provider_api_key", return_value=""),
            mock.patch("gui_qt.app.load_provider_key_store"),
            mock.patch("gui_qt.app.message_box_warning") as warning,
        ):
            self.window._on_add_custom_litellm_provider()

        dialog_cls.assert_called_once()
        warning.assert_not_called()
        self.assertIn("opencode-go", self.window._custom_litellm_providers)
        provider = self.window._custom_litellm_providers["opencode-go"]
        self.assertEqual(provider.label, "OpenCode Go")
        self.assertEqual(provider.base_url, "https://opencode.ai/zen/go/v1")
        self.assertEqual(provider.api_key_env, "OPENCODE_GO_API_KEY")
        # Table row and both provider dropdowns show the custom label.
        self.assertEqual(self.window.custom_provider_table.rowCount(), 1)
        combo = self.window.litellm_provider_combo
        index = combo.findData("opencode-go")
        self.assertGreaterEqual(index, 0)
        self.assertEqual(combo.itemText(index), "OpenCode Go")
        keys_combo = self.window.litellm_keys_provider_combo
        self.assertGreaterEqual(keys_combo.findData("opencode-go"), 0)
        self.assertEqual(
            self.window._custom_provider_entries()[0]["id"],
            "opencode-go",
        )

    def test_edit_custom_provider_updates_fields_and_keeps_id(self):
        self.window._ensure_settings_page("litellm")
        self.window._custom_litellm_providers = {
            "opencode-go": custom_provider_registry(
                [
                    {
                        "id": "opencode-go",
                        "label": "OpenCode Go",
                        "base_url": "https://old.example.com/v1",
                    }
                ]
            )["opencode-go"],
        }
        self.window._refresh_custom_provider_table()
        self.window.custom_provider_table.selectRow(0)
        dialog = mock.Mock()
        from PySide6.QtWidgets import QDialog

        dialog.exec.return_value = QDialog.DialogCode.Accepted
        dialog.result_provider.return_value = {
            "id": "opencode-go",
            "label": "OpenCode Go 新端点",
            "base_url": "https://new.example.com/v1",
            "models_url": "https://new.example.com/v1/models",
        }
        with mock.patch(
            "gui_qt.app.CustomLiteLLMProviderDialog",
            return_value=dialog,
        ) as dialog_cls, mock.patch("gui_qt.app.load_provider_api_key", return_value=""), mock.patch("gui_qt.app.load_provider_key_store"):
            self.window._on_edit_custom_litellm_provider()

        provider = self.window._custom_litellm_providers["opencode-go"]
        self.assertEqual(provider.base_url, "https://new.example.com/v1")
        self.assertEqual(provider.label, "OpenCode Go 新端点")
        self.assertEqual(provider.id, "opencode-go")
        self.assertTrue(dialog_cls.call_args.kwargs.get("provider"))
        self.assertEqual(
            dialog_cls.call_args.kwargs["provider"]["id"],
            "opencode-go",
        )
        self.assertNotIn(
            "opencode-go",
            dialog_cls.call_args.kwargs["reserved"],
        )

    def test_edit_custom_provider_real_dialog_accepts_changed_fields(self):
        """Drive the real dialog's accept path with edited mutable fields."""
        from gui_qt.custom_provider_dialog import CustomLiteLLMProviderDialog

        self.window._ensure_settings_page("litellm")
        self.window._custom_litellm_providers = {
            "opencode-go": custom_provider_registry(
                [
                    {
                        "id": "opencode-go",
                        "label": "OpenCode Go",
                        "base_url": "https://old.example.com/v1",
                    }
                ]
            )["opencode-go"],
        }
        self.window._refresh_custom_provider_table()
        self.window.custom_provider_table.selectRow(0)

        real_dialog = CustomLiteLLMProviderDialog(
            self.window,
            provider={
                "id": "opencode-go",
                "label": "OpenCode Go",
                "base_url": "https://old.example.com/v1",
            },
            reserved=frozenset({"opencode-go"}),
            title="编辑自定义 Provider",
        )
        self.assertTrue(real_dialog.id_edit.isReadOnly())
        real_dialog.base_url_edit.setText("https://new.example.com/v1")
        self.assertIsNone(real_dialog._on_accept())
        self.assertTrue(real_dialog.error_label.isHidden())
        entry = real_dialog.result_provider()
        self.assertEqual(entry["id"], "opencode-go")
        self.assertEqual(entry["base_url"], "https://new.example.com/v1")
        self.assertNotIn("requires_key", entry)
        # New-provider mode still rejects a reserved id through the real dialog.
        add_dialog = CustomLiteLLMProviderDialog(
            self.window,
            reserved=frozenset({"opencode-go"}),
        )
        add_dialog.id_edit.setText("opencode-go")
        self.assertIsNone(add_dialog._on_accept())
        self.assertFalse(add_dialog.error_label.isHidden())
        self.assertIn("冲突", add_dialog.error_label.text())

    def test_delete_custom_provider_requires_confirmation(self):
        self.window._ensure_settings_page("litellm")
        self.cache.update_models(
            "opencode-go",
            ["opencode-go/gpt-4o-mini"],
            source="opencode-go",
        )
        self.window._custom_litellm_providers = {
            "opencode-go": custom_provider_registry(
                [
                    {
                        "id": "opencode-go",
                        "base_url": "https://opencode.ai/zen/go/v1",
                    }
                ]
            )["opencode-go"],
        }
        self.window._refresh_custom_provider_table()
        self.window.custom_provider_table.selectRow(0)
        with (
            mock.patch("gui_qt.app.message_box_question", return_value="no"),
            mock.patch("gui_qt.app.load_provider_api_key", return_value=""),
            mock.patch("gui_qt.app.load_provider_key_store"),
        ):
            self.window._on_delete_custom_litellm_provider()
        self.assertIn("opencode-go", self.window._custom_litellm_providers)

        with (
            mock.patch("gui_qt.app.message_box_question", return_value="yes"),
            mock.patch("gui_qt.app.load_provider_api_key", return_value=""),
            mock.patch("gui_qt.app.load_provider_key_store"),
        ):
            self.window._on_delete_custom_litellm_provider()
        self.assertEqual(self.window._custom_litellm_providers, {})
        self.assertEqual(self.window.custom_provider_table.rowCount(), 0)
        self.assertEqual(self.cache.models("opencode-go").values, ())

    def test_custom_provider_loaded_from_config_and_saved_back(self):
        config = {
            "sync": {
                "backend": "litellm",
                "litellm_model": "opencode-go/gpt-4o-mini",
                "custom_litellm_providers": [
                    {
                        "id": "opencode-go",
                        "label": "OpenCode Go",
                        "base_url": "https://opencode.ai/zen/go/v1",
                    }
                ],
            },
            "batch": {},
        }
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
        ):
            if "litellm" not in self.window._settings_pages_built:
                self.window._ensure_settings_page("litellm")
            self.window._load_config_to_ui(pages={"litellm"})

        self.assertIn("opencode-go", self.window._custom_litellm_providers)
        self.assertEqual(self.window.custom_provider_table.rowCount(), 1)
        index = self.window.litellm_provider_combo.findData("opencode-go")
        self.assertGreaterEqual(index, 0)
        self.assertEqual(
            self.window.litellm_provider_combo.itemText(index),
            "OpenCode Go",
        )

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
        entries = saved_payload["sync"]["custom_litellm_providers"]
        self.assertEqual(entries[0]["id"], "opencode-go")
        self.assertEqual(entries[0]["base_url"], "https://opencode.ai/zen/go/v1")

    def test_custom_provider_model_catalog_worker_receives_registry(self):
        self.window._ensure_settings_page("litellm")
        self.window._custom_litellm_providers = custom_provider_registry(
            [
                {
                    "id": "opencode-go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                }
            ]
        )
        self.window._populate_litellm_providers((), selected="opencode-go")
        self.window.sync_backend_combo.setCurrentIndex(
            self.window.sync_backend_combo.findData("litellm")
        )
        self.window._on_sync_backend_changed(-1)
        self.assertEqual(self.window._current_litellm_provider(), "opencode-go")

        with (
            mock.patch(
                "gui_qt.app.load_provider_api_key",
                return_value="custom-key",
            ),
            mock.patch("gui_qt.app.LiteLLMModelCatalogWorker") as worker_cls,
        ):
            worker = mock.Mock()
            worker_cls.return_value = worker
            self.window._on_refresh_litellm_models()

        self.assertEqual(worker_cls.call_args.args[0], "opencode-go")
        self.assertEqual(
            worker_cls.call_args.kwargs.get("custom_providers"),
            self.window._custom_litellm_providers,
        )
        worker.start.assert_called_once()
        self.window._litellm_catalog_worker = None

    def test_custom_provider_connection_test_requires_key(self):
        self.window._ensure_settings_page("litellm")
        self.window._custom_litellm_providers = custom_provider_registry(
            [
                {
                    "id": "opencode-go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                }
            ]
        )
        self.window._populate_litellm_providers((), selected="opencode-go")
        self.window.sync_backend_combo.setCurrentIndex(
            self.window.sync_backend_combo.findData("litellm")
        )
        self.window._on_sync_backend_changed(-1)
        self.window.litellm_model_combo.setEditText("opencode-go/gpt-4o-mini")
        with (
            mock.patch(
                "gui_qt.app.load_provider_api_key",
                return_value="",
            ),
            mock.patch(
                "gui_qt.app.LiteLLMConnectionTestWorker",
            ) as worker_cls,
            mock.patch("gui_qt.app.message_box_information") as information,
        ):
            self.window._on_test_litellm_connection()

        worker_cls.assert_not_called()
        information.assert_called_once()
        self.assertIn("请先保存 API Key", information.call_args.args[1])

    def test_custom_provider_connection_test_uses_api_key_env(self):
        self.window._ensure_settings_page("litellm")
        self.window._custom_litellm_providers = custom_provider_registry(
            [
                {
                    "id": "opencode-go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                    "api_key_env": "OPENCODE_GO_API_KEY",
                }
            ]
        )
        self.window._populate_litellm_providers((), selected="opencode-go")
        self.window.sync_backend_combo.setCurrentIndex(
            self.window.sync_backend_combo.findData("litellm")
        )
        self.window._on_sync_backend_changed(-1)
        self.window.litellm_model_combo.setEditText("opencode-go/gpt-4o-mini")
        try:
            with (
                mock.patch(
                    "gui_qt.app.load_provider_api_key",
                    return_value="",
                ),
                mock.patch(
                    "gui_qt.app.LiteLLMConnectionTestWorker",
                ) as worker_cls,
                mock.patch("gui_qt.app.message_box_information") as information,
                mock.patch.dict(
                    "gui_qt.app.os.environ",
                    {"OPENCODE_GO_API_KEY": "env-custom-key"},
                    clear=True,
                ),
            ):
                worker = mock.Mock()
                worker_cls.return_value = worker
                self.window._on_test_litellm_connection()

            information.assert_not_called()
            worker_cls.assert_called_once()
            self.assertEqual(worker_cls.call_args.args[1], "env-custom-key")
            worker.start.assert_called_once()
        finally:
            self.window._litellm_connection_worker = None

    def test_custom_provider_model_catalog_uses_api_key_env(self):
        self.window._ensure_settings_page("litellm")
        self.window._custom_litellm_providers = custom_provider_registry(
            [
                {
                    "id": "opencode-go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                    "api_key_env": "OPENCODE_GO_API_KEY",
                }
            ]
        )
        self.window._populate_litellm_providers((), selected="opencode-go")
        self.window.sync_backend_combo.setCurrentIndex(
            self.window.sync_backend_combo.findData("litellm")
        )
        self.window._on_sync_backend_changed(-1)
        try:
            with (
                mock.patch(
                    "gui_qt.app.load_provider_api_key",
                    return_value="",
                ),
                mock.patch("gui_qt.app.LiteLLMModelCatalogWorker") as worker_cls,
                mock.patch("gui_qt.app.message_box_information") as information,
                mock.patch.dict(
                    "gui_qt.app.os.environ",
                    {"OPENCODE_GO_API_KEY": "env-custom-key"},
                    clear=True,
                ),
            ):
                worker = mock.Mock()
                worker_cls.return_value = worker
                self.window._on_refresh_litellm_models()

            information.assert_not_called()
            worker_cls.assert_called_once()
            self.assertEqual(worker_cls.call_args.kwargs.get("api_key"), "env-custom-key")
            worker.start.assert_called_once()
        finally:
            self.window._litellm_catalog_worker = None

    def test_keyless_custom_provider_catalog_skips_key_gate(self):
        self.window._ensure_settings_page("litellm")
        self.window._custom_litellm_providers = custom_provider_registry(
            [
                {
                    "id": "local-vllm",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "requires_key": False,
                }
            ]
        )
        self.window._populate_litellm_providers((), selected="local-vllm")
        self.window.sync_backend_combo.setCurrentIndex(
            self.window.sync_backend_combo.findData("litellm")
        )
        self.window._on_sync_backend_changed(-1)
        try:
            with (
                mock.patch(
                    "gui_qt.app.load_provider_api_key",
                    return_value="",
                ),
                mock.patch("gui_qt.app.LiteLLMModelCatalogWorker") as worker_cls,
                mock.patch("gui_qt.app.message_box_information") as information,
            ):
                worker = mock.Mock()
                worker_cls.return_value = worker
                self.window._on_refresh_litellm_models()

            information.assert_not_called()
            worker_cls.assert_called_once()
            self.assertEqual(worker_cls.call_args.kwargs.get("api_key"), "")
            worker.start.assert_called_once()
        finally:
            self.window._litellm_catalog_worker = None

    def test_delete_current_provider_clears_selection(self):
        self.window._ensure_settings_page("litellm")
        self.window._custom_litellm_providers = custom_provider_registry(
            [
                {
                    "id": "opencode-go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                }
            ]
        )
        self.window._populate_litellm_providers((), selected="opencode-go")
        self.window.sync_backend_combo.setCurrentIndex(
            self.window.sync_backend_combo.findData("litellm")
        )
        self.window.litellm_model_combo.setEditText("opencode-go/gpt-4o-mini")
        self.window._refresh_custom_provider_table()
        self.window.custom_provider_table.selectRow(0)
        with (
            mock.patch(
                "gui_qt.app.message_box_question",
                return_value="yes",
            ) as question,
            mock.patch("gui_qt.app.load_provider_api_key", return_value=""),
            mock.patch("gui_qt.app.load_provider_key_store"),
        ):
            self.window._on_delete_custom_litellm_provider()

        self.assertIn("当前正在使用", question.call_args.args[2])
        self.assertEqual(self.window._custom_litellm_providers, {})
        self.assertEqual(self.window._current_litellm_provider(), "")
        self.assertEqual(self.window.litellm_model_combo.currentText(), "")

    def test_refresh_custom_provider_table_restores_selection(self):
        self.window._ensure_settings_page("litellm")
        self.window._custom_litellm_providers = custom_provider_registry(
            [
                {
                    "id": "opencode-go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                },
                {
                    "id": "other-vendor",
                    "base_url": "https://other.example.com/v1",
                },
            ]
        )
        self.window._refresh_custom_provider_table()
        rows = self.window.custom_provider_table.rowCount()
        self.assertEqual(rows, 2)
        target = None
        for row in range(rows):
            item = self.window.custom_provider_table.item(row, 0)
            if item.text() == "opencode-go":
                target = row
        self.assertIsNotNone(target)
        self.window.custom_provider_table.selectRow(target)

        self.window._refresh_custom_provider_table()

        selected = self.window._selected_custom_provider()
        self.assertEqual(selected, "opencode-go")
        self.assertTrue(self.window.custom_provider_edit_btn.isEnabled())

    def test_invalid_custom_provider_config_is_ignored_on_load(self):
        config = {
            "sync": {
                "backend": "gemini",
                "custom_litellm_providers": [
                    {"id": "bad", "base_url": "not-a-url"},
                ],
            },
            "batch": {},
        }
        with (
            mock.patch.object(
                self.window.state,
                "load_translator_config",
                return_value=config,
            ),
            mock.patch.object(self.window, "_append_log") as append_log,
        ):
            if "litellm" not in self.window._settings_pages_built:
                self.window._ensure_settings_page("litellm")
            self.window._load_config_to_ui(pages={"litellm"})

        self.assertEqual(self.window._custom_litellm_providers, {})
        append_log.assert_called_once()
        self.assertIn("custom_litellm_providers", append_log.call_args.args[0])
        status_text = self.window.custom_provider_status_label.text()
        self.assertIn("已忽略无效的 custom_litellm_providers 配置", status_text)
        self.assertNotEqual(status_text, "尚未注册自定义 Provider。")

    def test_invalid_config_is_preserved_on_save(self):
        """Data-loss guard: an invalid provider list must survive an unrelated save."""
        self.window._ensure_settings_page("litellm")
        config = {
            "sync": {
                "backend": "gemini",
                "custom_litellm_providers": [
                    {"id": "valid-one", "base_url": "https://valid.example.com"},
                    {"id": "bad", "base_url": "not-a-url"},
                ],
            },
            "batch": {},
        }
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
        ):
            self.window._load_config_to_ui(pages={"litellm"})

        self.assertEqual(self.window._custom_litellm_providers, {})
        self.assertTrue(self.window._custom_litellm_providers_load_error)

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
        self.assertEqual(
            saved_payload["sync"]["custom_litellm_providers"],
            config["sync"]["custom_litellm_providers"],
        )

    def test_restore_config_snapshot_degrades_on_invalid_providers(self):
        self.window._ensure_settings_page("litellm")
        with (
            mock.patch.object(self.window, "_append_log") as append_log,
            mock.patch.object(
                self.window.state,
                "load_translator_config",
                return_value={"sync": {}, "batch": {}},
            ),
        ):
            self.window._restore_config_ui_snapshot(
                {
                    "custom_litellm_providers": (
                        (("id", "bad"), ("base_url", "not-a-url")),
                    ),
                }
            )

        self.assertEqual(self.window._custom_litellm_providers, {})
        self.assertTrue(self.window._custom_litellm_providers_load_error)
        append_log.assert_called_once()
        self.assertIn("快照", append_log.call_args.args[0])

    def test_deleting_all_providers_clears_config_on_save(self):
        """User-emptied registry must remove the key even when disk has entries."""
        self.window._ensure_settings_page("litellm")
        self.window.sync_backend_combo.setCurrentIndex(
            self.window.sync_backend_combo.findData("gemini")
        )
        self.window._custom_litellm_providers = custom_provider_registry(
            [
                {
                    "id": "opencode-go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                }
            ]
        )
        self.window._refresh_custom_provider_table()
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

        config = {
            "sync": {
                "backend": "gemini",
                "custom_litellm_providers": [
                    {"id": "opencode-go", "base_url": "https://opencode.ai/zen/go/v1"},
                ],
            },
            "batch": {},
        }
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

    def test_cached_catalog_ids_do_not_block_same_provider_readd(self):
        """Historical catalog cache ids must not shadow a re-added provider."""
        self.window._ensure_settings_page("litellm")
        self.cache.update_providers(
            ("opencode-go", "openai"),
            source="online",
        )
        reserved = self.window._reserved_custom_provider_ids()
        self.assertIn("openai", reserved)  # built-in stays reserved
        self.assertNotIn("opencode-go", reserved)  # cache history is ignored

    def test_save_blocked_when_config_load_failed_and_modified(self):
        """A save that would drop valid disk entries is blocked after a bad load."""
        self.window._ensure_settings_page("litellm")
        config = {
            "sync": {
                "backend": "gemini",
                "custom_litellm_providers": [
                    {"id": "valid-one", "base_url": "https://valid.example.com"},
                    {"id": "bad", "base_url": "not-a-url"},
                ],
            },
            "batch": {},
        }
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
        ):
            self.window._load_config_to_ui(pages={"litellm"})
        self.assertTrue(self.window._custom_litellm_providers_load_error)

        # Simulate the user adding a new provider after the failed load.
        from PySide6.QtWidgets import QDialog

        dialog = mock.Mock()
        dialog.exec.return_value = QDialog.DialogCode.Accepted
        dialog.result_provider.return_value = {
            "id": "new-one",
            "base_url": "https://new.example.com/v1",
        }
        with (
            mock.patch(
                "gui_qt.app.CustomLiteLLMProviderDialog",
                return_value=dialog,
            ),
            mock.patch("gui_qt.app.load_provider_api_key", return_value=""),
            mock.patch("gui_qt.app.load_provider_key_store"),
        ):
            self.window._on_add_custom_litellm_provider()
        self.assertIn("new-one", self.window._custom_litellm_providers)
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
            mock.patch("gui_qt.app.message_box_warning") as warning,
        ):
            saved = self.window._on_save_config()
        self.assertFalse(saved)
        save_config.assert_not_called()
        warning.assert_called_once()
        self.assertIn("无效", warning.call_args.args[1])

    def test_restore_snapshot_skips_malformed_entries(self):
        self.window._ensure_settings_page("litellm")
        with (
            mock.patch.object(self.window, "_append_log") as append_log,
            mock.patch.object(
                self.window.state,
                "load_translator_config",
                return_value={"sync": {}, "batch": {}},
            ),
        ):
            self.window._restore_config_ui_snapshot(
                {
                    "custom_litellm_providers": (
                        ("good",),
                        (("id", "opencode-go"), ("base_url", "https://ok.example.com")),
                    ),
                }
            )

        self.assertIn("opencode-go", self.window._custom_litellm_providers)
        self.assertFalse(self.window._custom_litellm_providers_load_error)
        append_log.assert_not_called()

    def test_detach_litellm_warmup_keeps_running_worker_alive(self):
        """Shutdown must not drop the only reference to a running QThread."""
        self.window._ensure_settings_page("litellm")
        worker = mock.Mock()
        worker.isRunning.return_value = True
        self.window._litellm_module_warmup_worker = worker

        self.window._detach_litellm_module_warmup()

        self.assertIsNone(self.window._litellm_module_warmup_worker)
        self.assertIn(worker, _RETIRED_LITELLM_WARMUP_WORKERS)
        worker.setParent.assert_called_once_with(None)
        worker.deleteLater.assert_not_called()

    def test_detach_litellm_warmup_deletes_finished_worker(self):
        self.window._ensure_settings_page("litellm")
        worker = mock.Mock()
        worker.isRunning.return_value = False
        self.window._litellm_module_warmup_worker = worker

        self.window._detach_litellm_module_warmup()

        self.assertIsNone(self.window._litellm_module_warmup_worker)
        worker.deleteLater.assert_called_once_with()
        self.assertNotIn(worker, _RETIRED_LITELLM_WARMUP_WORKERS)

    def test_module_warmup_refreshes_without_full_reload(self):
        """Warmup revalidation must not reset unsaved page selections."""
        self.window._ensure_settings_page("litellm")
        config = {"sync": {"backend": "gemini"}, "batch": {}}
        with (
            mock.patch.object(
                self.window.state,
                "load_translator_config",
                return_value=config,
            ),
            mock.patch.object(
                self.window,
                "_load_config_to_ui",
            ) as reload_page,
            mock.patch.object(
                self.window,
                "_after_custom_providers_changed",
            ) as refresh,
        ):
            self.window._on_litellm_module_warmed(None)

        reload_page.assert_not_called()
        refresh.assert_called_once_with()
        self.assertFalse(self.window._custom_litellm_providers_load_error)

    def test_module_warmup_skips_refresh_when_providers_edited(self):
        self.window._ensure_settings_page("litellm")
        self.window._custom_litellm_providers_modified = True
        with (
            mock.patch.object(
                self.window.state,
                "load_translator_config",
            ) as load,
            mock.patch.object(
                self.window,
                "_after_custom_providers_changed",
            ) as refresh,
        ):
            self.window._on_litellm_module_warmed(None)

        load.assert_not_called()
        refresh.assert_not_called()

    def test_module_warmup_revalidation_error_sets_load_error(self):
        self.window._ensure_settings_page("litellm")
        config = {
            "sync": {
                "backend": "gemini",
                "custom_litellm_providers": [
                    {"id": "bad", "base_url": "not-a-url"},
                ],
            },
            "batch": {},
        }
        with (
            mock.patch.object(
                self.window.state,
                "load_translator_config",
                return_value=config,
            ),
            mock.patch.object(
                self.window,
                "_after_custom_providers_changed",
            ),
            mock.patch.object(self.window, "_append_log"),
        ):
            self.window._on_litellm_module_warmed(None)

        self.assertTrue(self.window._custom_litellm_providers_load_error)
        self.assertEqual(self.window._custom_litellm_providers, {})

    def test_module_warmup_cleans_retired_workers(self):
        self.window._ensure_settings_page("litellm")
        retired = mock.Mock()
        retired.isRunning.return_value = False
        _RETIRED_LITELLM_WARMUP_WORKERS.add(retired)
        config = {"sync": {"backend": "gemini"}, "batch": {}}
        with (
            mock.patch.object(
                self.window.state,
                "load_translator_config",
                return_value=config,
            ),
            mock.patch.object(
                self.window,
                "_after_custom_providers_changed",
            ),
        ):
            self.window._on_litellm_module_warmed(None)

        self.assertNotIn(retired, _RETIRED_LITELLM_WARMUP_WORKERS)
        retired.deleteLater.assert_called_once_with()

    def test_warmup_worker_is_not_a_background_task(self):
        """The warmup import must not gate shutdown with a task dialog."""
        self.window._ensure_settings_page("litellm")
        worker = mock.Mock()
        worker.isRunning.return_value = True
        self.window._litellm_module_warmup_worker = worker
        with mock.patch.object(
            self.window,
            "findChildren",
            return_value=[worker],
        ):
            self.assertEqual(self.window._owned_background_threads(), ())

        # A retired (detached, still importing) worker is excluded as well.
        self.window._litellm_module_warmup_worker = None
        _RETIRED_LITELLM_WARMUP_WORKERS.add(worker)
        with mock.patch.object(
            self.window,
            "findChildren",
            return_value=[worker],
        ):
            self.assertEqual(self.window._owned_background_threads(), ())


if __name__ == "__main__":
    unittest.main()
