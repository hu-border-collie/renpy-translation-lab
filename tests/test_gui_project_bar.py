"""Tests for global project bar and writeback issues collapse (GUI IA P0b / #159)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
    from gui_qt.check_report import WritebackSummary
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    WritebackSummary = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiProjectBarAndWritebackCollapseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()

    def test_global_project_bar_widgets_exist(self) -> None:
        self.assertTrue(hasattr(self.window, "global_project_bar"))
        self.assertTrue(hasattr(self.window, "global_project_path_edit"))
        self.assertTrue(hasattr(self.window, "global_switch_project_btn"))
        self.assertTrue(hasattr(self.window, "global_browse_project_btn"))
        # Legacy alias still points at browse control for enable/disable callers.
        self.assertIs(self.window.select_btn, self.window.global_browse_project_btn)
        self.assertIs(self.window.project_path_edit, self.window.global_project_path_edit)

    def test_refresh_project_label_updates_global_bar(self) -> None:
        self.window.state.get_game_root = lambda: "C:/games/Demo/work"  # type: ignore[method-assign]
        self.window._refresh_project_label()
        self.assertEqual(
            self.window.global_project_path_edit.text(),
            "C:/games/Demo/work",
        )

    def test_task_running_disables_global_project_switch_controls(self) -> None:
        self.window._set_task_running(True)
        self.assertFalse(self.window.global_switch_project_btn.isEnabled())
        self.assertFalse(self.window.global_browse_project_btn.isEnabled())
        self.window._set_task_running(False)
        self.assertTrue(self.window.global_switch_project_btn.isEnabled())
        self.assertTrue(self.window.global_browse_project_btn.isEnabled())

    def test_select_project_respects_dirty_leave_guard(self) -> None:
        calls: list[str] = []
        self.window._confirm_unsaved_config_before_registry_switch = (  # type: ignore[method-assign]
            lambda: calls.append("guard") or False
        )
        self.window._switch_game_root = lambda _path: calls.append("switch") or True  # type: ignore[method-assign]

        self.window._on_select_project()

        self.assertEqual(calls, ["guard"])

    def test_select_project_blocked_while_running(self) -> None:
        calls: list[str] = []
        self.window._task_running = True
        self.window._confirm_unsaved_config_before_registry_switch = (  # type: ignore[method-assign]
            lambda: calls.append("guard") or True
        )
        self.window._on_select_project()
        self.assertEqual(calls, [])

    def test_global_switch_opens_workspace_section(self) -> None:
        focused: list[str] = []
        self.window._focus_settings_section = lambda key: focused.append(key)  # type: ignore[method-assign]
        self.window._on_global_switch_project()
        self.assertEqual(focused, ["workspace"])

    def test_writeback_issues_panel_collapsed_by_default(self) -> None:
        self.assertFalse(self.window._writeback_issues_expanded)
        self.assertTrue(self.window.writeback_issues_panel.isHidden())

    def test_writeback_issues_auto_expand_on_warn(self) -> None:
        summary = WritebackSummary(
            status="warn",
            heading="需处理",
            message="检查为需处理",
            facts=[],
            findings=[],
            can_apply=False,
            manifest_path="C:/m.json",
        )
        # Minimal stubs so update path can run without full manifest IO.
        self.window._writeback_issues_ready = lambda _s: True  # type: ignore[method-assign]
        self.window._load_writeback_manifest = lambda: None  # type: ignore[method-assign]
        self.window._apply_failure_report_ready = lambda *_a, **_k: False  # type: ignore[method-assign]
        self.window._retry_button_mode = lambda *_a, **_k: "hidden"  # type: ignore[method-assign]
        self.window._remediation_ready = lambda *_a, **_k: False  # type: ignore[method-assign]

        from gui_qt.work_modes import WorkMode

        self.window._work_mode = WorkMode.BATCH_TRANSLATION
        self.window._update_writeback_action_buttons(summary, running=False)

        self.assertTrue(self.window._writeback_issues_expanded)
        self.assertFalse(self.window.writeback_issues_panel.isHidden())
        self.assertFalse(self.window.writeback_issues_badge.isHidden())
        self.assertEqual(self.window.writeback_issues_badge.text(), "有待处理问题")

    def test_apply_btn_stays_outside_issues_panel(self) -> None:
        panel = self.window.writeback_issues_panel
        self.assertNotIn(self.window.apply_btn, panel.findChildren(type(self.window.apply_btn)))
        # Primary apply is a sibling layout control, not inside the collapse panel.
        self.assertIsNot(self.window.apply_btn.parentWidget(), panel)


if __name__ == "__main__":
    unittest.main()
