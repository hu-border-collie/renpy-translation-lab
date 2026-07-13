import unittest
from unittest import mock

try:
    from gui_qt.app import MainWindow
except ImportError as exc:
    MainWindow = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class _Combo:
    def __init__(self, data="gemini"):
        self.data = data

    def currentData(self):
        return self.data


class _Label:
    def __init__(self):
        self.value = ""

    def setText(self, value):
        self.value = value


class _State:
    def __init__(self, config):
        self.config = config

    def load_translator_config(self):
        return self.config


@unittest.skipIf(MainWindow is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class GuiSyncBackendTests(unittest.TestCase):
    def setUp(self):
        self.window = MainWindow.__new__(MainWindow)

    def test_saved_sync_backend_defaults_to_gemini(self):
        self.window.state = _State({})
        self.assertEqual(self.window._saved_sync_backend(), "gemini")

    def test_saved_sync_backend_reads_explicit_litellm(self):
        self.window.state = _State({"sync": {"backend": "litellm"}})
        self.assertEqual(self.window._saved_sync_backend(), "litellm")

    def test_litellm_hint_reports_missing_optional_dependency(self):
        self.window.sync_backend_combo = _Combo("litellm")
        self.window.sync_backend_hint = _Label()
        with mock.patch("gui_qt.app.importlib.util.find_spec", return_value=None):
            self.window._on_sync_backend_changed(0)
        self.assertIn("LiteLLM", self.window.sync_backend_hint.value)
        self.assertIn("尚未安装", self.window.sync_backend_hint.value)
        self.assertIn("不使用 Gemini API Key", self.window.sync_backend_hint.value)

    def test_gemini_hint_preserves_batch_as_recommended_path(self):
        self.window.sync_backend_combo = _Combo("gemini")
        self.window.sync_backend_hint = _Label()
        self.window._on_sync_backend_changed(0)
        self.assertIn("推荐路径", self.window.sync_backend_hint.value)
        self.assertIn("Gemini Batch", self.window.sync_backend_hint.value)


if __name__ == "__main__":
    unittest.main()
