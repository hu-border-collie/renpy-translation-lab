import unittest
from pathlib import Path
from unittest import mock

try:
    from PySide6.QtWidgets import QApplication, QGroupBox

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

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()

    def test_litellm_has_independent_settings_page(self):
        self.assertIn("litellm", self.window._settings_nav_rows)
        row = self.window._settings_nav_rows["litellm"]
        page = self.window.settings_stack.widget(row)
        self.assertEqual(page.objectName(), "settings_litellm_scroll")
        titles = {group.title() for group in page.findChildren(QGroupBox)}
        self.assertIn("LiteLLM 同步替代后端", titles)
        self.assertIn("Provider 凭据状态", titles)

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
        self.window.state = mock.Mock()
        self.window.state.get_game_root.return_value = Path("C:/Game/work")
        self.window.state.load_translator_config.return_value = {
            "sync": {},
            "batch": {},
        }
        self.window.sync_backend_combo.setCurrentIndex(
            self.window.sync_backend_combo.findData("litellm")
        )
        self.window.litellm_model_combo.setEditText("")
        with mock.patch("gui_qt.app.QMessageBox.information") as information:
            saved = self.window._on_save_config()
        self.assertFalse(saved)
        self.window.state.save_translator_config.assert_not_called()
        information.assert_called_once()

    def test_models_page_is_gemini_only(self):
        row = self.window._settings_nav_rows["models"]
        page = self.window.settings_stack.widget(row)
        titles = {group.title() for group in page.findChildren(QGroupBox)}
        self.assertIn("Gemini 同步翻译", titles)
        self.assertIn("批量离线翻译", titles)
        self.assertNotIn("LiteLLM 同步替代后端", titles)


if __name__ == "__main__":
    unittest.main()
