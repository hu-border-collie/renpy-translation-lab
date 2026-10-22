import unittest
from unittest import mock

try:
    from gui_qt.app import MainWindow
except ImportError as exc:
    MainWindow = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class _State:
    def __init__(self, config):
        self.config = config

    def load_translator_config(self):
        return self.config


@unittest.skipIf(MainWindow is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class GuiSyncBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow.__new__(MainWindow)
        from gui_qt.settings.litellm_page import LiteLLMSettingsPage
        self.page = LiteLLMSettingsPage(start_warmup=False)
        self.addCleanup(self.page.deleteLater)
        self.addCleanup(self.page.widget.deleteLater)
        self.addCleanup(self.page.request_shutdown)

    def test_saved_sync_backend_defaults_to_gemini(self):
        self.window.state = _State({})
        self.assertEqual(self.window._saved_sync_backend(), "gemini")

    def test_saved_sync_backend_reads_explicit_litellm(self):
        self.window.state = _State({"sync": {"backend": "litellm"}})
        self.assertEqual(self.window._saved_sync_backend(), "litellm")

    def test_litellm_hint_reports_missing_optional_dependency(self):
        self.page.load({"sync_backend": "litellm"})
        with mock.patch("gui_qt.settings.litellm_page.importlib.util.find_spec", return_value=None):
            self.page._on_sync_backend_changed(0)
        self.assertIn("LiteLLM", self.page.sync_backend_hint.text())
        self.assertIn("尚未安装", self.page.sync_backend_hint.text())
        self.assertIn("不使用 Gemini API Key", self.page.sync_backend_hint.text())

    def test_gemini_hint_preserves_batch_as_recommended_path(self):
        self.page.load({"sync_backend": "gemini"})
        self.page._on_sync_backend_changed(0)
        self.assertIn("推荐路径", self.page.sync_backend_hint.text())
        self.assertIn("Gemini Batch", self.page.sync_backend_hint.text())


if __name__ == "__main__":
    unittest.main()
