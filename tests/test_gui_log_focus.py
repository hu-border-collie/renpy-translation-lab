"""Tests for workbench log drawer and dual log-focus APIs (GUI IA P0a / #158)."""
from __future__ import annotations

import unittest

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiLogFocusTests(unittest.TestCase):
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

    def test_log_views_share_document(self) -> None:
        self.assertIs(
            self.window.workbench_log_view.document(),
            self.window.log_view.document(),
        )
        self.window.log_view.setPlainText("shared-line")
        self.assertIn("shared-line", self.window.workbench_log_view.toPlainText())

    def test_clear_log_clears_shared_document(self) -> None:
        self.window.log_view.setPlainText("to-clear")
        self.window._clear_log_view()
        self.assertEqual(self.window.log_view.toPlainText(), "")
        self.assertEqual(self.window.workbench_log_view.toPlainText(), "")

    def test_show_workbench_log_drawer_does_not_switch_main_tab(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)
        self.window._set_workbench_log_drawer_expanded(False)

        self.window._show_workbench_log_drawer()

        self.assertIs(self.window.tab_widget.currentWidget(), workbench)
        self.assertTrue(self.window._workbench_log_drawer_expanded)
        # Window may not be shown in offscreen tests; isHidden tracks setVisible.
        self.assertFalse(self.window.workbench_log_body.isHidden())
        self.assertEqual(self.window.workbench_log_toggle_btn.text(), "折叠")

    def test_expand_diagnostics_log_switches_to_diagnostics_tab(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)
        idle_sizes = list(self.window.diagnostics_splitter.sizes())

        self.window._expand_diagnostics_log()

        self.assertIs(
            self.window.tab_widget.currentWidget(),
            self.window._diagnostics_tab,
        )
        # Animation may still be running; force end sizes for a stable assertion.
        total = max(sum(self.window.diagnostics_splitter.sizes()), sum(idle_sizes), 1)
        target_context = int(total * 0.32)
        self.window.diagnostics_splitter.setSizes([target_context, total - target_context])
        sizes = self.window.diagnostics_splitter.sizes()
        self.assertGreater(sizes[1], 0)
        self.assertLess(sizes[0], total)

    def test_expand_diagnostics_log_without_switch_keeps_workbench(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)

        self.window._expand_diagnostics_log(switch_tab=False)

        self.assertIs(self.window.tab_widget.currentWidget(), workbench)

    def test_reveal_log_for_active_context_prefers_drawer_off_diagnostics(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)
        self.window._set_workbench_log_drawer_expanded(False)

        self.window._reveal_log_for_active_context()

        self.assertIs(self.window.tab_widget.currentWidget(), workbench)
        self.assertTrue(self.window._workbench_log_drawer_expanded)

    def test_reveal_log_for_active_context_on_diagnostics_expands_log(self) -> None:
        self.window.tab_widget.setCurrentWidget(self.window._diagnostics_tab)
        self.window._set_workbench_log_drawer_expanded(False)

        self.window._reveal_log_for_active_context()

        self.assertIs(
            self.window.tab_widget.currentWidget(),
            self.window._diagnostics_tab,
        )
        # Drawer should not be required when already on diagnostics.
        self.assertFalse(self.window._workbench_log_drawer_expanded)

    def test_deprecated_focus_log_tab_aliases_workbench_drawer(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)
        self.window._set_workbench_log_drawer_expanded(False)

        self.window._focus_log_tab()

        self.assertIs(self.window.tab_widget.currentWidget(), workbench)
        self.assertTrue(self.window._workbench_log_drawer_expanded)

    def test_probe_entrypoint_uses_expand_diagnostics_log(self) -> None:
        """Regression: diagnostics tools must not be rewritten to workbench-only reveal."""
        import inspect
        from gui_qt import app as app_module

        source = inspect.getsource(app_module.MainWindow._on_run_probe)
        self.assertIn("_expand_diagnostics_log", source)
        self.assertNotIn("_show_workbench_log_drawer", source)

    def test_start_translation_entrypoint_uses_workbench_drawer(self) -> None:
        import inspect
        from gui_qt import app as app_module

        source = inspect.getsource(app_module.MainWindow._on_start_translation)
        self.assertIn("_show_workbench_log_drawer", source)


if __name__ == "__main__":
    unittest.main()
