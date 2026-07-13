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
    def __init__(self, data):
        self.data = data

    def currentData(self):
        return self.data


class _Label:
    def __init__(self):
        self.value = ""

    def setText(self, value):
        self.value = value


class _Button:
    def __init__(self):
        self.visible = None
        self.enabled = None
        self.text = ""

    def setVisible(self, value):
        self.visible = value

    def setEnabled(self, value):
        self.enabled = value

    def setText(self, value):
        self.text = value


@unittest.skipIf(MainWindow is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class GuiLiteLLMInstallTests(unittest.TestCase):
    def setUp(self):
        self.window = MainWindow.__new__(MainWindow)
        self.window.sync_backend_combo = _Combo("litellm")
        self.window.sync_backend_hint = _Label()
        self.window.install_litellm_btn = _Button()

    def test_missing_dependency_shows_enabled_install_button(self):
        with mock.patch("gui_qt.app.importlib.util.find_spec", return_value=None):
            self.window._on_sync_backend_changed(0)
        self.assertTrue(self.window.install_litellm_btn.visible)
        self.assertTrue(self.window.install_litellm_btn.enabled)
        self.assertEqual(self.window.install_litellm_btn.text, "安装 LiteLLM")

    def test_installed_dependency_hides_install_button(self):
        with mock.patch("gui_qt.app.importlib.util.find_spec", return_value=object()):
            self.window._on_sync_backend_changed(0)
        self.assertFalse(self.window.install_litellm_btn.visible)
        self.assertIn("已安装", self.window.sync_backend_hint.value)

    def test_gemini_backend_hides_install_button(self):
        self.window.sync_backend_combo = _Combo("gemini")
        self.window._on_sync_backend_changed(0)
        self.assertFalse(self.window.install_litellm_btn.visible)


if __name__ == "__main__":
    unittest.main()
