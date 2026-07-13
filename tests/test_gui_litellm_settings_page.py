import unittest
from pathlib import Path
from unittest import mock

try:
    from PySide6.QtWidgets import QApplication, QGroupBox, QLineEdit

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
        cls.window = MainWindow()

    @classmethod
    def tearDownClass(cls):
        cls.window.close()
        cls.window.deleteLater()
        cls._app.processEvents()

    def test_litellm_has_independent_settings_page(self):
        self.assertIn("litellm", self.window._settings_nav_rows)
        row = self.window._settings_nav_rows["litellm"]
        page = self.window.settings_stack.widget(row)
        self.assertEqual(page.objectName(), "settings_litellm_scroll")
        titles = {group.title() for group in page.findChildren(QGroupBox)}
        self.assertIn("LiteLLM 同步替代后端", titles)
        self.assertIn("Provider 凭据", titles)
        self.assertTrue(self.window.litellm_model_combo.isEditable())
        self.assertGreater(self.window.litellm_provider_combo.count(), 1)
        self.assertEqual(
            self.window.litellm_api_key_edit.echoMode(),
            QLineEdit.EchoMode.Password,
        )
        self.assertEqual(self.window.litellm_refresh_models_btn.text(), "联网更新列表")
        self.assertEqual(self.window.litellm_test_connection_btn.text(), "测试连接")
        actions = {
            action.objectName(): action
            for action in self.window.litellm_model_combo.lineEdit().actions()
        }
        self.assertIn("combo_popup_action", actions)
        self.assertFalse(actions["combo_popup_action"].icon().isNull())

    def test_typed_model_prefix_takes_priority_for_credentials(self):
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

    def test_switching_provider_does_not_relabel_bare_model_as_new_provider(self):
        openai_index = self.window.litellm_provider_combo.findData("openai")
        anthropic_index = self.window.litellm_provider_combo.findData("anthropic")
        self.window.litellm_provider_combo.setCurrentIndex(openai_index)
        try:
            self.window.litellm_model_combo.setEditText("gpt-custom")
            self.window.litellm_provider_combo.setCurrentIndex(anthropic_index)

            self.assertNotEqual(
                self.window.litellm_model_combo.currentText(),
                "anthropic/gpt-custom",
            )
            self.assertTrue(
                self.window.litellm_model_combo.currentText().startswith("anthropic/")
            )
        finally:

            self.window.litellm_provider_combo.setCurrentIndex(openai_index)
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
        row = self.window._settings_nav_rows["models"]
        page = self.window.settings_stack.widget(row)
        titles = {group.title() for group in page.findChildren(QGroupBox)}
        self.assertIn("Gemini 同步翻译", titles)
        self.assertIn("批量离线翻译", titles)
        self.assertNotIn("LiteLLM 同步替代后端", titles)
        actions = {
            action.objectName(): action
            for action in self.window.sync_model_combo.lineEdit().actions()
        }
        self.assertIn("combo_popup_action", actions)
        self.assertFalse(actions["combo_popup_action"].icon().isNull())

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
