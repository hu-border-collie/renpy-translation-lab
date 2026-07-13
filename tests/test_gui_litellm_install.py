import unittest
from unittest import mock

try:
    from gui_qt.app import MainWindow, QProcess
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

    def setEnabled(self, value):
        self.enabled = value


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


class _Process:
    def errorString(self):
        return "cannot start"


class _ProgressBar:
    def __init__(self):
        self.visible = None
        self.range_values = None
        self.format = ""

    def setVisible(self, value):
        self.visible = value

    def setRange(self, minimum, maximum):
        self.range_values = (minimum, maximum)

    def setFormat(self, value):
        self.format = value


@unittest.skipIf(MainWindow is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class GuiLiteLLMInstallTests(unittest.TestCase):
    def setUp(self):
        self.window = MainWindow.__new__(MainWindow)
        self.window.sync_backend_combo = _Combo("litellm")
        self.window.sync_backend_hint = _Label()
        self.window.install_litellm_btn = _Button()
        self.window.sync_model_combo = _Combo(None)
        self.window.litellm_install_progress = _ProgressBar()
        self.window._refresh_litellm_install_action_gating = mock.Mock()

    def test_missing_dependency_shows_enabled_install_button(self):
        with mock.patch("gui_qt.app.importlib.util.find_spec", return_value=None):
            self.window._on_sync_backend_changed(0)
        self.assertTrue(self.window.install_litellm_btn.visible)
        self.assertTrue(self.window.install_litellm_btn.enabled)
        self.assertEqual(self.window.install_litellm_btn.text, "安装 LiteLLM")
        self.assertFalse(self.window.litellm_install_progress.visible)
        self.assertTrue(self.window.sync_model_combo.enabled)

    def test_installing_shows_busy_progress_and_disables_litellm_model(self):
        self.window._litellm_install_running = mock.Mock(return_value=True)
        with mock.patch("gui_qt.app.importlib.util.find_spec", return_value=None):
            self.window._on_sync_backend_changed(0)
        self.assertTrue(self.window.litellm_install_progress.visible)
        self.assertEqual(self.window.litellm_install_progress.range_values, (0, 0))
        self.assertIn("后台安装", self.window.litellm_install_progress.format)
        self.assertFalse(self.window.sync_model_combo.enabled)
        self.assertFalse(self.window.install_litellm_btn.enabled)
        self.assertIn("正在后台安装", self.window.sync_backend_hint.value)

    def test_failed_to_start_clears_install_state_and_restores_ui(self):
        self.window._litellm_install_process = _Process()
        self.window._append_log = mock.Mock()
        self.window._on_sync_backend_changed = mock.Mock()
        with mock.patch("gui_qt.app.QMessageBox.warning") as warning:
            self.window._on_litellm_install_error(
                QProcess.ProcessError.FailedToStart
            )
        self.assertIsNone(self.window._litellm_install_process)
        self.window._on_sync_backend_changed.assert_called_once_with(-1)
        warning.assert_called_once()
        self.assertIn(
            "cannot start",
            self.window._append_log.call_args.args[0],
        )

    def test_non_start_error_waits_for_finished_cleanup(self):
        process = _Process()
        self.window._litellm_install_process = process
        self.window._append_log = mock.Mock()
        self.window._on_sync_backend_changed = mock.Mock()
        with mock.patch("gui_qt.app.QMessageBox.warning") as warning:
            self.window._on_litellm_install_error(
                QProcess.ProcessError.Crashed
            )
        self.assertIs(self.window._litellm_install_process, process)
        self.window._on_sync_backend_changed.assert_not_called()
        warning.assert_not_called()

    def test_installed_dependency_hides_install_button(self):
        with mock.patch("gui_qt.app.importlib.util.find_spec", return_value=object()):
            self.window._on_sync_backend_changed(0)
        self.assertFalse(self.window.install_litellm_btn.visible)
        self.assertIn("已安装", self.window.sync_backend_hint.value)

    def test_gemini_backend_hides_install_button(self):
        self.window.sync_backend_combo = _Combo("gemini")
        self.window._litellm_install_running = mock.Mock(return_value=True)
        self.window._on_sync_backend_changed(0)
        self.assertFalse(self.window.install_litellm_btn.visible)
        self.assertTrue(self.window.litellm_install_progress.visible)
        self.assertTrue(self.window.sync_model_combo.enabled)


if __name__ == "__main__":
    unittest.main()
