import unittest
from unittest import mock

try:
    from gui_qt.app import MainWindow
    from gui_qt.optional_feature_install import OptionalFeatureInstallController
    from gui_qt.work_modes import WorkMode
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

    def setToolTip(self, value):
        self.tooltip = value


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
        OptionalFeatureInstallController._active_feature_id = None
        OptionalFeatureInstallController._active_controller = None
        self.window = MainWindow.__new__(MainWindow)
        self.window.sync_backend_combo = _Combo("litellm")
        self.window.sync_backend_hint = _Label()
        self.window.install_litellm_btn = _Button()
        self.window.sync_model_combo = _Combo(None)
        self.window.litellm_model_combo = _Combo(None)
        self.window.litellm_install_progress = _ProgressBar()
        self.window._litellm_install = None
        self.window._litellm_install_active = False
        self.window._refresh_litellm_install_action_gating = mock.Mock()

    def tearDown(self):
        OptionalFeatureInstallController._active_feature_id = None
        OptionalFeatureInstallController._active_controller = None

    def test_missing_dependency_shows_enabled_install_button(self):
        with mock.patch("gui_qt.app.importlib.util.find_spec", return_value=None):
            self.window._on_sync_backend_changed(0)
        self.assertTrue(self.window.install_litellm_btn.visible)
        self.assertTrue(self.window.install_litellm_btn.enabled)
        self.assertEqual(self.window.install_litellm_btn.text, "安装 LiteLLM")
        self.assertFalse(self.window.litellm_install_progress.visible)
        self.assertTrue(self.window.litellm_model_combo.enabled)

    def test_installing_shows_busy_progress_and_disables_litellm_model(self):
        self.window._litellm_install_running = mock.Mock(return_value=True)
        with mock.patch("gui_qt.app.importlib.util.find_spec", return_value=None):
            self.window._on_sync_backend_changed(0)
        self.assertTrue(self.window.litellm_install_progress.visible)
        self.assertEqual(self.window.litellm_install_progress.range_values, (0, 0))
        self.assertIn("后台安装", self.window.litellm_install_progress.format)
        self.assertFalse(self.window.litellm_model_combo.enabled)
        self.assertFalse(self.window.install_litellm_btn.enabled)
        self.assertIn("正在后台安装", self.window.sync_backend_hint.value)

    def test_failed_install_restores_ui_via_controller_finished(self):
        self.window._on_sync_backend_changed = mock.Mock()
        self.window._refresh_litellm_version_label = mock.Mock()
        self.window._on_check_litellm_version = mock.Mock()
        self.window._selected_sync_backend = mock.Mock(return_value="gemini")
        self.window.statusBar = mock.Mock(return_value=mock.Mock())
        with mock.patch("gui_qt.app.QMessageBox.warning") as warning:
            self.window._on_litellm_install_finished(
                "litellm",
                False,
                "无法启动安装进程：cannot start",
            )
        self.window._on_sync_backend_changed.assert_called_once_with(-1)
        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[1], "LiteLLM 安装失败")
        self.assertIn("cannot start", warning.call_args.args[2])

    def test_successful_install_refreshes_version_and_models(self):
        self.window._on_sync_backend_changed = mock.Mock()
        self.window._refresh_litellm_version_label = mock.Mock()
        self.window._on_check_litellm_version = mock.Mock()
        self.window._on_refresh_litellm_models = mock.Mock()
        self.window._selected_sync_backend = mock.Mock(return_value="litellm")
        status_bar = mock.Mock()
        self.window.statusBar = mock.Mock(return_value=status_bar)
        with mock.patch("gui_qt.app.QMessageBox.warning") as warning:
            self.window._on_litellm_install_finished(
                "litellm",
                True,
                "LiteLLM 安装完成。",
            )
        warning.assert_not_called()
        status_bar.showMessage.assert_called_once()
        self.window._refresh_litellm_version_label.assert_called_once()
        self.window._on_check_litellm_version.assert_called_once()
        self.window._on_refresh_litellm_models.assert_called_once()
        self.window._on_sync_backend_changed.assert_called_once_with(-1)

    def test_install_uses_shared_optional_feature_controller(self):
        controller = mock.Mock()
        controller.is_running.return_value = False
        controller.start_install.return_value = (True, "started")
        self.window._ensure_litellm_install_controller = mock.Mock(
            return_value=controller
        )
        self.window._on_sync_backend_changed = mock.Mock()
        self.window._on_install_litellm()
        controller.start_install.assert_called_once_with()
        self.window._on_sync_backend_changed.assert_called_once_with(-1)

    def test_install_blocks_when_controller_reports_busy(self):
        controller = mock.Mock()
        controller.is_running.return_value = True
        self.window._ensure_litellm_install_controller = mock.Mock(
            return_value=controller
        )
        self.window._on_install_litellm()
        controller.start_install.assert_not_called()

    def test_install_blocks_all_litellm_sync_modes_but_not_batch(self):
        self.window._litellm_install_active = True
        self.window._saved_sync_backend = mock.Mock(return_value="litellm")
        for mode in (
            WorkMode.SYNC_TRANSLATION,
            WorkMode.SYNC_KEYWORD_EXTRACTION,
            WorkMode.SYNC_REVISION,
        ):
            with self.subTest(mode=mode):
                self.assertTrue(self.window._litellm_install_blocks_mode(mode))
        self.assertFalse(
            self.window._litellm_install_blocks_mode(WorkMode.BATCH_TRANSLATION)
        )

    def test_install_guard_uses_saved_litellm_when_ui_selects_gemini(self):
        self.window._litellm_install_active = True
        self.window.sync_backend_combo = _Combo("gemini")
        self.window._saved_sync_backend = mock.Mock(return_value="litellm")
        self.assertTrue(
            self.window._litellm_install_blocks_mode(WorkMode.SYNC_TRANSLATION)
        )

    def test_install_guard_allows_saved_gemini_when_ui_selects_litellm(self):
        self.window._litellm_install_active = True
        self.window.sync_backend_combo = _Combo("litellm")
        self.window._saved_sync_backend = mock.Mock(return_value="gemini")
        self.assertFalse(
            self.window._litellm_install_blocks_mode(WorkMode.SYNC_TRANSLATION)
        )

    def test_translate_availability_stays_disabled_after_ui_refresh(self):
        self.window._litellm_install_blocks_mode = mock.Mock(return_value=True)
        spec = mock.Mock(mode=WorkMode.SYNC_TRANSLATION, implemented=True)
        self.assertFalse(
            self.window._translate_button_enabled(
                spec=spec,
                bootstrap_ready=True,
                running=False,
            )
        )

    def test_start_action_checks_install_after_unsaved_config_resolution(self):
        self.window._current_work_mode = mock.Mock(
            return_value=WorkMode.SYNC_KEYWORD_EXTRACTION
        )
        self.window.state = mock.Mock()
        self.window.state.get_game_root.return_value = "C:/Game/work"
        self.window._confirm_unsaved_config_before_workflow = mock.Mock(
            return_value=True
        )
        self.window._litellm_install_blocks_mode = mock.Mock(return_value=True)
        with mock.patch("gui_qt.app.QMessageBox.information") as information:
            self.window._on_start_translation()
        self.window._confirm_unsaved_config_before_workflow.assert_called_once()
        self.window._litellm_install_blocks_mode.assert_called_once_with(
            WorkMode.SYNC_KEYWORD_EXTRACTION
        )
        information.assert_called_once()
        self.assertIn("正在安装", information.call_args.args[1])

    def test_installed_dependency_offers_update_button(self):
        with (
            mock.patch("gui_qt.app.importlib.util.find_spec", return_value=object()),
            mock.patch("gui_qt.app.installed_litellm_version", return_value="1.83.7"),
        ):
            self.window._on_sync_backend_changed(0)
        self.assertTrue(self.window.install_litellm_btn.visible)
        self.assertEqual(self.window.install_litellm_btn.text, "更新 LiteLLM")
        self.assertIn("已安装", self.window.sync_backend_hint.value)

    def test_python_314_compatible_limit_disables_repeated_update(self):
        self.window.litellm_version_label = _Label()
        self.window._litellm_latest_version = "1.92.0"
        self.window._litellm_latest_compatible_version = "1.83.7"
        self.window._litellm_latest_requires_python = ">=3.10,<3.14"

        with (
            mock.patch("gui_qt.app.importlib.util.find_spec", return_value=object()),
            mock.patch("gui_qt.app.installed_litellm_version", return_value="1.83.7"),
        ):
            self.window._refresh_litellm_version_label()
            self.window._on_sync_backend_changed(0)

        self.assertIn("不支持当前", self.window.litellm_version_label.value)
        self.assertIn("\n兼容最新版 1.83.7", self.window.litellm_version_label.value)
        self.assertEqual(
            self.window.install_litellm_btn.text,
            "当前 Python 可用最新版",
        )
        self.assertFalse(self.window.install_litellm_btn.enabled)

    def test_gemini_backend_hides_install_button(self):
        self.window.sync_backend_combo = _Combo("gemini")
        self.window._litellm_install_running = mock.Mock(return_value=True)
        self.window._on_sync_backend_changed(0)
        self.assertFalse(self.window.install_litellm_btn.visible)
        self.assertTrue(self.window.litellm_install_progress.visible)
        self.assertFalse(self.window.litellm_model_combo.enabled)


if __name__ == "__main__":
    unittest.main()
