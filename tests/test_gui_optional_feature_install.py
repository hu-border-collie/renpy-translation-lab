"""GUI tests for Settings → Extensions optional install surface."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
    from gui_qt.optional_feature_install import (
        OptionalFeatureInstallController,
        action_enabled_for_status,
        build_litellm_controller,
    )
    from gui_qt.work_modes import WorkMode
    from optional_feature import (
        FeatureInstallState,
        FeatureStatus,
        litellm_feature,
        relation_analyzer_feature,
    )
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    MainWindow = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


def _status(
    state: FeatureInstallState,
    *,
    action: str = "安装并启用",
    message: str = "status",
    versions: dict[str, str] | None = None,
) -> FeatureStatus:
    return FeatureStatus(
        feature_id="relation_analyzer",
        state=state,
        installed_versions=versions or {},
        missing=(),
        outdated=(),
        message=message,
        action_label=action,
    )


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiOptionalFeatureInstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        OptionalFeatureInstallController._active_feature_id = None
        OptionalFeatureInstallController._active_controller = None
        self.window = MainWindow()
        self.window.resize(1400, 900)
        self.window.show()
        self.window._activate_shell_route("settings")
        row = self.window._settings_nav_rows["extensions"]
        self.window.settings_nav.setCurrentRow(row)
        for _ in range(8):
            self._app.processEvents()

    def tearDown(self) -> None:
        OptionalFeatureInstallController._active_feature_id = None
        OptionalFeatureInstallController._active_controller = None
        self.window.close()
        self.window.deleteLater()

    def test_extensions_page_owns_relation_analyzer_card(self) -> None:
        self.assertIn("extensions", self.window._settings_nav_rows)
        self.assertEqual(
            self.window.settings_nav.item(
                self.window._settings_nav_rows["extensions"]
            ).text(),
            "扩展",
        )
        self.assertTrue(hasattr(self.window, "relation_analyzer_install_btn"))
        self.assertTrue(hasattr(self.window, "relation_analyzer_status_label"))
        self.assertTrue(hasattr(self.window, "relation_analyzer_install_progress"))
        page = self.window.settings_stack.currentWidget()
        self.assertIsNotNone(page.findChild(type(self.window.relation_analyzer_install_btn), "relation_analyzer_install_btn"))

    def test_status_transitions_update_action_and_progress(self) -> None:
        self.window._apply_relation_analyzer_status(
            _status(FeatureInstallState.NOT_INSTALLED, action="安装并启用")
        )
        self.assertEqual(self.window.relation_analyzer_install_btn.text(), "安装并启用")
        self.assertTrue(self.window.relation_analyzer_install_btn.isEnabled())
        self.assertFalse(self.window.relation_analyzer_install_progress.isVisible())

        self.window._apply_relation_analyzer_status(
            _status(FeatureInstallState.PARTIALLY_INSTALLED, action="修复安装")
        )
        self.assertEqual(self.window.relation_analyzer_install_btn.text(), "修复安装")
        self.assertTrue(self.window.relation_analyzer_install_btn.isEnabled())

        self.window._apply_relation_analyzer_status(
            _status(FeatureInstallState.UPDATE_AVAILABLE, action="更新")
        )
        self.assertEqual(self.window.relation_analyzer_install_btn.text(), "更新")

        self.window._apply_relation_analyzer_status(
            _status(FeatureInstallState.INSTALLED, action="已启用")
        )
        self.assertEqual(self.window.relation_analyzer_install_btn.text(), "已启用")
        self.assertFalse(self.window.relation_analyzer_install_btn.isEnabled())

        self.window._apply_relation_analyzer_status(
            _status(FeatureInstallState.INSTALLING, action="正在安装…")
        )
        self.assertFalse(self.window.relation_analyzer_install_btn.isEnabled())
        self.assertTrue(self.window.relation_analyzer_install_progress.isVisible())

        self.window._apply_relation_analyzer_status(
            _status(FeatureInstallState.FAILED, action="修复安装")
        )
        self.assertTrue(self.window.relation_analyzer_failure_label.isVisible())

    def test_repeated_clicks_do_not_start_second_install(self) -> None:
        controller = self.window._ensure_relation_analyzer_install_controller()
        with mock.patch.object(
            controller,
            "start_install",
            return_value=(True, "started"),
        ) as start:
            self.window._on_install_relation_analyzer()
            controller._active = True
            OptionalFeatureInstallController._active_controller = controller
            OptionalFeatureInstallController._active_feature_id = "relation_analyzer"
            self.window._on_install_relation_analyzer()
        self.assertEqual(start.call_count, 1)

    def test_process_failure_marks_failed_and_restores_action(self) -> None:
        controller = self.window._ensure_relation_analyzer_install_controller()
        self.window._append_log = mock.Mock()
        with mock.patch("gui_qt.app.QMessageBox.warning") as warning:
            controller.finished.emit(
                "relation_analyzer",
                False,
                "无法启动安装进程：cannot start",
            )
            for _ in range(4):
                self._app.processEvents()
        warning.assert_called_once()
        self.assertIn("relation_analyzer", self.window._optional_feature_last_failed)
        self.window._refresh_relation_analyzer_extension_ui()
        self.assertTrue(self.window.relation_analyzer_install_btn.isEnabled())

    def test_translation_actions_remain_available_while_analyzer_installs(self) -> None:
        controller = self.window._ensure_relation_analyzer_install_controller()
        controller._active = True
        OptionalFeatureInstallController._active_controller = controller
        OptionalFeatureInstallController._active_feature_id = "relation_analyzer"
        # Analyzer install must not share LiteLLM's mode-blocking gate.
        self.window._litellm_install_active = False
        self.window._saved_sync_backend = mock.Mock(return_value="gemini")
        for mode in (
            WorkMode.BATCH_TRANSLATION,
            WorkMode.SYNC_TRANSLATION,
            WorkMode.SYNC_KEYWORD_EXTRACTION,
            WorkMode.SYNC_REVISION,
        ):
            with self.subTest(mode=mode):
                self.assertFalse(self.window._litellm_install_blocks_mode(mode))
        spec = SimpleNamespace(mode=WorkMode.BATCH_TRANSLATION, implemented=True)
        with mock.patch.object(
            self.window,
            "_translation_requires_doctor_check",
            return_value=False,
        ):
            self.assertTrue(
                self.window._translate_button_enabled(
                    spec=spec,
                    bootstrap_ready=True,
                    running=False,
                )
            )

    def test_action_enabled_helper(self) -> None:
        self.assertTrue(
            action_enabled_for_status(_status(FeatureInstallState.NOT_INSTALLED))
        )
        self.assertFalse(
            action_enabled_for_status(_status(FeatureInstallState.INSTALLED))
        )
        self.assertFalse(
            action_enabled_for_status(_status(FeatureInstallState.INSTALLING))
        )

    def test_controller_blocks_concurrent_optional_feature_install(self) -> None:
        analyzer = OptionalFeatureInstallController(relation_analyzer_feature())
        litellm = build_litellm_controller()
        analyzer._active = True
        OptionalFeatureInstallController._active_controller = analyzer
        OptionalFeatureInstallController._active_feature_id = "relation_analyzer"
        started, message = litellm.start_install()
        self.assertFalse(started)
        self.assertIn("relation_analyzer", message)

    def test_litellm_builder_prefers_platform_lock_when_available(self) -> None:
        feature = litellm_feature()
        controller = build_litellm_controller()
        self.assertEqual(controller.feature.feature_id, "litellm")
        if feature.lock_relative_path:
            self.assertTrue(controller.prefer_hash_lock)
            self.assertTrue(feature.lock_path().is_file())
        else:
            self.assertFalse(controller.prefer_hash_lock)

    def test_failed_to_start_emits_finished_and_releases_mutex(self) -> None:
        controller = OptionalFeatureInstallController(relation_analyzer_feature())
        finished = mock.Mock()
        controller.finished.connect(finished)
        process = mock.Mock()
        process.errorString.return_value = "cannot start"
        controller._process = process
        controller._active = True
        OptionalFeatureInstallController._active_controller = controller
        OptionalFeatureInstallController._active_feature_id = "relation_analyzer"
        from PySide6.QtCore import QProcess

        controller._on_error(QProcess.ProcessError.FailedToStart)
        self.assertFalse(controller.is_running())
        self.assertIsNone(OptionalFeatureInstallController._active_controller)
        finished.assert_called_once()
        self.assertFalse(finished.call_args.args[1])
        self.assertIn("cannot start", finished.call_args.args[2])

    def test_non_start_error_waits_for_finished_cleanup(self) -> None:
        controller = OptionalFeatureInstallController(relation_analyzer_feature())
        finished = mock.Mock()
        controller.finished.connect(finished)
        process = mock.Mock()
        process.errorString.return_value = "crashed"
        controller._process = process
        controller._active = True
        OptionalFeatureInstallController._active_controller = controller
        OptionalFeatureInstallController._active_feature_id = "relation_analyzer"
        from PySide6.QtCore import QProcess

        controller._on_error(QProcess.ProcessError.Crashed)
        self.assertTrue(controller.is_running())
        self.assertIs(OptionalFeatureInstallController._active_controller, controller)
        finished.assert_not_called()


if __name__ == "__main__":
    unittest.main()
