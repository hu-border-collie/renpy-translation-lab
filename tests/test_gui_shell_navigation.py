"""Contracts for the unified application shell introduced by the v3 redesign."""
from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
    from gui_qt.check_report import WritebackSummary
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    WritebackSummary = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    Qt = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from gui_qt.work_modes import WorkMode, WorkbenchNavItem
from tests import gui_test_support


def _workbench_route(item: WorkbenchNavItem) -> str:
    return f"workbench:{item.value}"


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiShellNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        cls._app = app or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()

    def _route_item(self, route: str):
        row = self.window._shell_nav_rows[route]
        return self.window.shell_nav.item(row)

    def test_single_primary_sidebar_and_default_batch_route(self) -> None:
        batch_route = _workbench_route(WorkbenchNavItem.BATCH_TRANSLATION)

        self.assertFalse(self.window.shell_nav.isHidden())
        self.assertTrue(self.window.tab_widget.tabBar().isHidden())
        self.assertTrue(self.window.workbench_nav.isHidden())
        # Settings categories remain a horizontal page-local selector.
        self.assertFalse(self.window.settings_nav.isHidden())
        self.assertEqual(len(self.window._shell_nav_rows), 7)
        self.assertEqual(self.window._current_shell_route(), batch_route)
        self.assertEqual(
            self.window.shell_nav.currentRow(),
            self.window._shell_nav_rows[batch_route],
        )
        self.assertTrue(self.window.global_project_bar.isHidden())
        self.assertIs(
            self.window.global_project_bar.parentWidget(),
            self.window.workbench_primary,
        )

        self.assertEqual(self.window.app_sidebar.width(), 224)
        self.assertFalse(hasattr(self.window, "sidebar_brand_mark"))
        self.assertFalse(hasattr(self.window, "sidebar_collapse_btn"))
        self.assertTrue(self.window.global_project_bar.isHidden())
        self.assertEqual(self.window.global_project_bar_label.text(), "当前项目")
        for route in self.window._shell_nav_rows:
            item = self._route_item(route)
            self.assertTrue(item.icon().isNull(), route)
            self.assertTrue(item.text(), route)

        # Tabler icons are reserved for actions; navigation stays text-led.
        self.assertFalse(self.window.header_log_btn.icon().isNull())
        self.assertFalse(self.window.batch_translation_page.buttons["start"].icon().isNull())
        self.assertFalse(self.window.save_config_btn.icon().isNull())

        self.window._activate_shell_route("project_prepare")
        self.assertFalse(self.window.global_project_bar.isHidden())

    def test_routes_reuse_existing_workbench_settings_and_diagnostics(self) -> None:
        keywords_route = _workbench_route(WorkbenchNavItem.KEYWORDS)
        self.window._activate_shell_route(keywords_route)
        self.assertEqual(self.window._work_mode, WorkMode.KEYWORD_EXTRACTION)
        self.assertIs(
            self.window.workbench_stack.currentWidget(),
            self.window.keywords_page,
        )

        self.window._activate_shell_route("settings")
        self.assertIs(self.window.tab_widget.currentWidget(), self.window._config_tab)
        context_row = self.window._settings_nav_rows["context"]
        self.window.settings_nav.setCurrentRow(context_row)
        self.assertEqual(self.window.settings_stack.currentIndex(), context_row)
        self.assertEqual(self.window._current_shell_route(), "settings")

        self.window._activate_shell_route("diagnostics")
        self.assertIs(
            self.window.tab_widget.currentWidget(),
            self.window._diagnostics_tab,
        )
        self.assertEqual(self.window._current_shell_route(), "diagnostics")
        self.assertTrue(self.window.header_log_btn.isChecked())

        self.window._activate_shell_route(keywords_route)
        self.assertIs(
            self.window.workbench_stack.currentWidget(),
            self.window.keywords_page,
        )

    def test_repeated_header_log_click_keeps_diagnostics_selected(self) -> None:
        self.window._activate_shell_route("diagnostics")
        self.assertTrue(self.window.header_log_btn.isChecked())

        self.window.header_log_btn.click()

        self.assertEqual(self.window._current_shell_route(), "diagnostics")
        self.assertTrue(self.window.header_log_btn.isChecked())

    def test_safe_non_writeback_result_does_not_claim_apply_is_available(self) -> None:
        self.window._set_writeback_summary(
            WritebackSummary(
                status="safe",
                heading="关键词报告已生成",
                message="报告已生成，不需要写回。",
                facts=[],
                findings=[],
                can_apply=False,
            )
        )

        badge_text = self.window.safety_status_label.text()
        self.assertIn("无需写回", badge_text)
        self.assertNotIn("可安全写回", badge_text)

    def test_programmatic_navigation_keeps_shell_selection_in_sync(self) -> None:
        revision_route = _workbench_route(WorkbenchNavItem.REVISION)

        self.window._focus_settings_section("litellm")
        self.assertEqual(self.window._current_shell_route(), "settings")
        self.assertEqual(
            self.window.shell_nav.currentRow(),
            self.window._shell_nav_rows["settings"],
        )

        self.window._set_work_mode(
            WorkMode.REVISION,
            refresh_manifest_writeback=False,
        )
        # Mode changes do not silently pull the user out of settings.
        self.assertEqual(self.window._current_shell_route(), "settings")
        self.window._focus_workbench_main_tab()
        self.assertEqual(self.window._current_shell_route(), revision_route)
        self.assertEqual(
            self.window.shell_nav.currentRow(),
            self.window._shell_nav_rows[revision_route],
        )

        self.window._expand_diagnostics_log(switch_tab=True)
        self.assertEqual(self.window._current_shell_route(), "diagnostics")
        self.assertEqual(self.window.shell_nav.currentRow(), -1)

    def test_running_lock_preserves_diagnostics_and_active_task_access(self) -> None:
        batch_route = _workbench_route(WorkbenchNavItem.BATCH_TRANSLATION)
        self.window._activate_shell_route(batch_route)
        self.window._set_task_running(True)

        self.assertTrue(
            bool(self._route_item(batch_route).flags() & Qt.ItemFlag.ItemIsEnabled)
        )
        for nav_item in WorkbenchNavItem:
            route = _workbench_route(nav_item)
            if route == batch_route:
                continue
            self.assertFalse(
                bool(self._route_item(route).flags() & Qt.ItemFlag.ItemIsEnabled),
                route,
            )
        self.assertTrue(
            bool(
                self._route_item("project_prepare").flags()
                & Qt.ItemFlag.ItemIsEnabled
            )
        )
        self.assertTrue(
            bool(self._route_item("settings").flags() & Qt.ItemFlag.ItemIsEnabled)
        )

        self.window._activate_shell_route("settings")
        self.assertEqual(self.window._current_shell_route(), "settings")
        self.window._expand_diagnostics_log(switch_tab=True)
        self.assertEqual(self.window._current_shell_route(), "diagnostics")
        self.window._activate_shell_route(batch_route)
        self.assertEqual(self.window._current_shell_route(), batch_route)

        self.window._set_task_running(False)
        for route in self.window._shell_nav_rows:
            self.assertTrue(
                bool(self._route_item(route).flags() & Qt.ItemFlag.ItemIsEnabled),
                route,
            )

    @patch("gui_qt.app.DoctorWorker")
    def test_doctor_launch_uses_unified_project_environment_route(self, worker_cls) -> None:
        worker = worker_cls.return_value
        worker.completed.connect = MagicMock()
        worker.start = MagicMock()
        self.window.state.get_game_root = lambda: Path("C:/DemoGame/work")
        self.window._confirm_unsaved_config_before_workflow = lambda: True
        self.window._activate_shell_route("project_prepare")

        self.window._on_run_doctor()

        self.assertEqual(self.window._current_shell_route(), "project_prepare")
        self.assertTrue(self.window.workbench_stack.isHidden())
        worker.start.assert_called_once_with()
        self.window._doctor_worker = None
        self.window._active_command = ""
        self.window._set_task_running(False)

    @patch("gui_qt.app.DoctorWorker")
    def test_doctor_running_allows_shell_page_navigation(self, worker_cls) -> None:
        """Environment check must not trap the user on 项目与环境."""
        worker = worker_cls.return_value
        worker.isRunning = MagicMock(return_value=True)
        worker.completed.connect = MagicMock()
        worker.start = MagicMock()
        self.window.state.get_game_root = lambda: Path("C:/DemoGame/work")
        self.window._confirm_unsaved_config_before_workflow = lambda: True
        self.window._activate_shell_route("project_prepare")

        self.window._on_run_doctor()
        self.assertEqual(self.window._active_command, "doctor")
        self.assertTrue(self.window._task_running)
        self.assertFalse(self.window._context_switching_locked())

        for route in self.window._shell_nav_rows:
            self.assertTrue(
                bool(self._route_item(route).flags() & Qt.ItemFlag.ItemIsEnabled),
                route,
            )

        keywords_route = _workbench_route(WorkbenchNavItem.KEYWORDS)
        settings_route = "settings"
        batch_route = _workbench_route(WorkbenchNavItem.BATCH_TRANSLATION)

        self.window._activate_shell_route(keywords_route)
        self.assertEqual(self.window._current_shell_route(), keywords_route)
        self.assertFalse(self.window.workbench_stack.isHidden())

        self.window._activate_shell_route(settings_route)
        self.assertEqual(self.window._current_shell_route(), settings_route)

        self.window._activate_shell_route(batch_route)
        self.assertEqual(self.window._current_shell_route(), batch_route)

        self.window._activate_shell_route("project_prepare")
        self.assertEqual(self.window._current_shell_route(), "project_prepare")
        self.assertTrue(self.window.workbench_stack.isHidden())

        # Global cancel remains available; project-bar doctor becomes red "停止检查".
        self.assertTrue(self.window.kill_btn.isEnabled())
        self.assertTrue(self.window.doctor_btn.isEnabled())
        self.assertEqual(self.window.doctor_btn.text(), "停止检查")
        self.assertEqual(self.window.doctor_btn.objectName(), "doctor_stop_btn")
        self.assertEqual(self.window.header_task_status_label.text(), "环境检查中")
        self.assertFalse(self.window.translate_btn.isEnabled())
        # Task pages must not pretend they own the doctor job.
        self.assertFalse(self.window._task_page_running_chrome())
        self.assertFalse(self.window.batch_translation_page.buttons["stop"].isEnabled())
        self.assertFalse(self.window.keywords_page.stop_btn.isEnabled())
        self.assertFalse(self.window.sync_translation_page.stop_btn.isEnabled())
        self.assertFalse(self.window.revision_page.stop_btn.isEnabled())
        self.assertFalse(self.window.context_library_page.stop_btn.isEnabled())

        self.window._doctor_worker = None
        self.window._active_command = ""
        self.window._set_task_running(False)
        self.assertEqual(self.window.doctor_btn.text(), "环境检查")
        self.assertEqual(self.window.doctor_btn.objectName(), "secondary_btn")

    @patch("gui_qt.app.DoctorWorker")
    def test_doctor_running_uses_soft_work_mode_switch(self, worker_cls) -> None:
        """Page switches during doctor must not reload manifests from disk."""
        worker = worker_cls.return_value
        worker.isRunning = MagicMock(return_value=True)
        worker.completed.connect = MagicMock()
        worker.start = MagicMock()
        self.window.state.get_game_root = lambda: Path("C:/DemoGame/work")
        self.window._confirm_unsaved_config_before_workflow = lambda: True
        self.window._activate_shell_route("project_prepare")
        self.window._on_run_doctor()

        keywords_route = _workbench_route(WorkbenchNavItem.KEYWORDS)
        with patch.object(self.window, "_set_work_mode") as set_mode:
            self.window._activate_shell_route(keywords_route)
            set_mode.assert_called()
            kwargs = set_mode.call_args.kwargs
            self.assertFalse(kwargs.get("refresh_manifest_writeback", True))

        self.window._doctor_worker = None
        self.window._active_command = ""
        self.window._set_task_running(False)

    @patch("gui_qt.app.DoctorWorker")
    def test_doctor_stop_control_stays_on_project_bar(self, worker_cls) -> None:
        """Cancel environment check from 项目与环境, not via foreign page stop buttons."""
        worker = worker_cls.return_value
        worker.isRunning = MagicMock(return_value=True)
        worker.completed.connect = MagicMock()
        worker.start = MagicMock()
        self.window.state.get_game_root = lambda: Path("C:/DemoGame/work")
        self.window._confirm_unsaved_config_before_workflow = lambda: True
        self.window._activate_shell_route("project_prepare")
        self.window._on_run_doctor()

        batch_route = _workbench_route(WorkbenchNavItem.BATCH_TRANSLATION)
        self.window._activate_shell_route(batch_route)
        # Idle-style chrome: start remains visible (disabled), stop is not armed.
        start = self.window.batch_translation_page.buttons["start"]
        stop = self.window.batch_translation_page.buttons["stop"]
        self.assertFalse(start.isHidden())
        self.assertFalse(start.isEnabled())
        self.assertFalse(stop.isEnabled())

        self.window._activate_shell_route("project_prepare")
        self.assertEqual(self.window.doctor_btn.text(), "停止检查")
        with patch.object(self.window, "_on_kill") as on_kill:
            self.window.doctor_btn.click()
            on_kill.assert_called_once_with()

        self.window._doctor_worker = None
        self.window._active_command = ""
        self.window._set_task_running(False)

    def test_generate_template_uses_project_bar_cancel(self) -> None:
        self.window.state.get_game_root = lambda: Path("C:/DemoGame/work")
        self.window._confirm_unsaved_config_before_workflow = lambda: True
        self.window.runner.run = MagicMock()

        self.window._on_generate_template()

        self.assertEqual(self.window._active_command, "generate_template")
        self.assertEqual(self.window._current_shell_route(), "project_prepare")
        self.assertEqual(self.window.doctor_btn.text(), "停止生成")
        self.assertEqual(self.window.doctor_btn.objectName(), "doctor_stop_btn")
        self.assertTrue(self.window.doctor_btn.isEnabled())
        self.assertEqual(self.window.header_task_status_label.text(), "生成模板中")
        self.assertEqual(self.window.kill_btn.text(), "停止生成")
        with patch.object(self.window, "_on_kill") as on_kill:
            self.window.doctor_btn.click()
            on_kill.assert_called_once_with()

        self.window._active_command = ""
        self.window._set_task_running(False)
        self.assertEqual(self.window.doctor_btn.text(), "环境检查")
        self.assertEqual(self.window.kill_btn.text(), "停止")

    def test_bootstrap_launch_preserves_and_can_return_to_project_route(self) -> None:
        self.window.state.get_game_root = lambda: Path("C:/DemoGame/work")
        self.window._confirm_unsaved_config_before_workflow = lambda: True
        self.window.runner.run = MagicMock()

        self.window._on_bootstrap_work()

        self.assertEqual(self.window._current_shell_route(), "project_prepare")
        self.assertEqual(self.window.workbench_status_tabs.currentIndex(), 0)
        project_item = self._route_item("project_prepare")
        self.assertTrue(bool(project_item.flags() & Qt.ItemFlag.ItemIsEnabled))
        # Visible cancel on project bar (legacy kill lives on a hidden panel).
        self.assertEqual(self.window.bootstrap_work_btn.text(), "停止准备")
        self.assertEqual(self.window.bootstrap_work_btn.objectName(), "bootstrap_stop_btn")
        self.assertTrue(self.window.bootstrap_work_btn.isEnabled())
        self.assertEqual(self.window.header_task_status_label.text(), "准备工作目录中")
        self.window._activate_shell_route("settings")
        self.window._activate_shell_route("project_prepare")
        self.assertEqual(self.window._current_shell_route(), "project_prepare")
        self.assertEqual(self.window.workbench_status_tabs.currentIndex(), 0)
        self.window.runner.run.assert_called_once()
        self.window._active_command = ""
        self.window._set_task_running(False)
        self.assertEqual(self.window.bootstrap_work_btn.text(), "准备工作目录")
        self.assertEqual(self.window.bootstrap_work_btn.objectName(), "secondary_btn")
        for route in self.window._shell_nav_rows:
            self.assertTrue(
                bool(self._route_item(route).flags() & Qt.ItemFlag.ItemIsEnabled),
                route,
            )

    def test_route_owns_environment_progress_and_optional_writeback_tabs(self) -> None:
        batch_route = _workbench_route(WorkbenchNavItem.BATCH_TRANSLATION)
        self.window._activate_shell_route(batch_route)
        self.window.workbench_status_tabs.setCurrentIndex(2)
        tab_bar = self.window.workbench_status_tabs.tabBar()

        self.assertFalse(self.window.workbench_status_card.isHidden())
        self.assertFalse(self.window.workbench_status_tabs.isHidden())
        self.assertFalse(tab_bar.isTabVisible(0))
        self.assertTrue(tab_bar.isTabVisible(1))
        self.assertTrue(tab_bar.isTabVisible(2))
        self.assertTrue(self.window.global_project_bar.isHidden())
        self.assertFalse(hasattr(self.window, "workbench_status_toggle_btn"))
        self.assertFalse(hasattr(self.window, "workbench_status_header"))

        self.window._activate_shell_route("project_prepare")
        self.assertFalse(self.window.workbench_status_card.isHidden())
        self.assertFalse(self.window.workbench_status_tabs.isHidden())
        self.assertTrue(tab_bar.isTabVisible(0))
        self.assertFalse(tab_bar.isTabVisible(1))
        self.assertFalse(tab_bar.isTabVisible(2))
        self.assertFalse(self.window.global_project_bar.isHidden())
        self.assertEqual(self.window.workbench_status_tabs.currentIndex(), 0)

        self.window._activate_shell_route(batch_route)
        self.assertFalse(self.window.workbench_status_card.isHidden())
        self.assertFalse(self.window.workbench_status_tabs.isHidden())
        self.assertFalse(tab_bar.isTabVisible(0))
        self.assertTrue(tab_bar.isTabVisible(1))
        self.assertTrue(tab_bar.isTabVisible(2))
        self.assertEqual(self.window.workbench_status_tabs.currentIndex(), 2)

    def test_workflow_empty_cta_navigates_to_project_prepare(self) -> None:
        """Task-page empty CTA must open 项目与环境, not a hidden status tab."""
        batch_route = _workbench_route(WorkbenchNavItem.BATCH_TRANSLATION)
        self.window._activate_shell_route(batch_route)
        self.window.state.get_game_root = lambda: None  # type: ignore[method-assign]
        self.window._workflow = None
        self.window._writeback_manifest_path = ""
        self.window._sync_workbench_empty_states()

        empty = self.window.workflow_empty_state
        self.assertFalse(empty.isHidden())
        btn = empty._action_btn
        self.assertIsNotNone(btn)
        assert btn is not None
        self.assertEqual(btn.text(), "去环境检查")

        btn.click()
        self._app.processEvents()

        self.assertEqual(self.window._current_shell_route(), "project_prepare")
        self.assertEqual(self.window.workbench_status_tabs.currentIndex(), 0)
        self.assertTrue(self.window.workbench_status_tabs.tabBar().isTabVisible(0))
        self.assertTrue(self.window.workbench_stack.isHidden())
        self.assertFalse(self.window.global_project_bar.isHidden())

    def test_project_route_does_not_pollute_task_status_sessions(self) -> None:
        batch_route = _workbench_route(WorkbenchNavItem.BATCH_TRANSLATION)
        keyword_route = _workbench_route(WorkbenchNavItem.KEYWORDS)
        self.window._activate_shell_route(batch_route)
        self.window.workbench_status_tabs.setCurrentIndex(2)

        self.window._activate_shell_route("project_prepare")
        self.assertEqual(self.window.workbench_status_tabs.currentIndex(), 0)

        self.window._activate_shell_route(keyword_route)
        self.assertEqual(
            self.window._mode_sessions[WorkMode.BATCH_TRANSLATION].stage_index,
            2,
        )
        self.window._activate_shell_route(batch_route)
        self.assertEqual(self.window.workbench_status_tabs.currentIndex(), 2)

    def test_project_switch_resets_saved_task_status_behind_special_route(self) -> None:
        batch_route = _workbench_route(WorkbenchNavItem.BATCH_TRANSLATION)
        self.window._activate_shell_route(batch_route)
        self.window.workbench_status_tabs.setCurrentIndex(2)
        self.window._activate_shell_route("project_prepare")

        with (
            patch.object(
                self.window.state,
                "set_game_root",
                return_value=(Path("C:/Games/Example/work"), False),
            ),
            patch.object(self.window.runner, "is_running", return_value=False),
            patch.object(self.window, "_is_doctor_running", return_value=False),
            patch.object(self.window, "_load_config_to_ui"),
            patch.object(self.window, "_refresh_diagnostics_context"),
            patch.object(self.window, "_invalidate_manifest_caches"),
            patch.object(self.window, "_apply_work_mode_ui"),
        ):
            self.assertTrue(
                self.window._switch_game_root("C:/Games/Example/work")
            )

        self.assertEqual(self.window._current_shell_route(), "project_prepare")
        self.assertEqual(self.window.workbench_status_tabs.currentIndex(), 0)
        self.assertEqual(self.window._shell_task_status_index, 1)
        self.assertEqual(
            self.window._mode_sessions[WorkMode.BATCH_TRANSLATION].stage_index,
            1,
        )
        self.window._activate_shell_route(batch_route)
        self.assertEqual(self.window.workbench_status_tabs.currentIndex(), 1)


if __name__ == "__main__":
    unittest.main()
