"""Cold-start guards: keep MainWindow construction off expensive work."""
from __future__ import annotations

import unittest
import subprocess
import sys
from pathlib import Path
from unittest import mock

try:
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import (
        MainWindow,
        _SETTINGS_CONFIG_PAGE_KEYS,
        _SETTINGS_PAGE_SPECS,
    )
    from gui_qt.project_state import ProjectState
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    MainWindow = None  # type: ignore[assignment,misc]
    ProjectState = None  # type: ignore[assignment,misc]
    _SETTINGS_CONFIG_PAGE_KEYS = frozenset()  # type: ignore[assignment,misc]
    _SETTINGS_PAGE_SPECS = ()  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests.support import gui_test_support


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiStartupPerfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def tearDown(self) -> None:
        window = getattr(self, "window", None)
        if window is not None:
            gui_test_support.close_main_window(window)
            window.deleteLater()
            self._app.processEvents()

    def test_startup_skips_manifest_history_scan(self) -> None:
        """Probe/split readiness must not walk logs/ during MainWindow().__init__."""
        with mock.patch.object(
            ProjectState,
            "get_latest_manifest_path_for_mode",
            autospec=True,
            return_value=None,
        ) as latest_for_mode:
            self.window = MainWindow()
            latest_for_mode.assert_not_called()

    def test_startup_builds_no_settings_section_bodies(self) -> None:
        self.window = MainWindow()
        self.assertEqual(self.window._settings_pages_built, set())
        self.assertNotIn("batch_model_combo", self.window.__dict__)
        self.assertNotIn("_games_registry_panel", self.window.__dict__)
        for index, (key, _label, _builder) in enumerate(_SETTINGS_PAGE_SPECS):
            page = self.window.settings_stack.widget(index)
            self.assertIsNotNone(page)
            assert page is not None
            self.assertEqual(page.objectName(), f"settings_{key}_placeholder")

    def test_load_config_to_ui_does_not_materialize_settings_pages(self) -> None:
        """_load_theme_to_ui must not getattr(theme_combo) into full config build."""
        self.window = MainWindow()
        self.assertEqual(self.window._settings_pages_built, set())
        self.window._load_config_to_ui(refresh_task_gates=False)
        self.assertEqual(self.window._settings_pages_built, set())
        self.assertNotIn("theme_combo", self.window.__dict__)
        self.assertNotIn("batch_model_combo", self.window.__dict__)

    def test_opening_workspace_builds_only_workspace(self) -> None:
        self.window = MainWindow()
        self.window._focus_settings_section("workspace")
        self.assertEqual(self.window._settings_pages_built, {"workspace"})
        self.assertIn("_games_registry_panel", self.window.__dict__)
        self.assertNotIn("batch_model_combo", self.window.__dict__)

    def test_first_config_section_builds_only_that_page(self) -> None:
        """Tab switches must not materialize every config section at once."""
        self.window = MainWindow()
        self.window._focus_settings_section("models")
        self.assertEqual(self.window._settings_pages_built, {"models"})
        self.assertIn("batch_model_combo", self.window.__dict__)
        self.assertNotIn("advanced", self.window._settings_pages_built)
        self.assertNotIn("workspace", self.window._settings_pages_built)
        self.assertNotIn("extensions", self.window._settings_pages_built)
        self.assertNotIn("shortcuts", self.window._settings_pages_built)

        self.window._focus_settings_section("advanced")
        self.assertEqual(
            self.window._settings_pages_built,
            {"models", "advanced"},
        )
        self.assertFalse(
            _SETTINGS_CONFIG_PAGE_KEYS.issubset(self.window._settings_pages_built)
        )

    def test_partial_page_load_preserves_dirty_baseline(self) -> None:
        """Opening another settings tab must not mark prior edits as saved."""
        self.window = MainWindow()
        self.window._focus_settings_section("models")
        combo = self.window.batch_model_combo
        original = combo.currentText()
        dirty_value = f"{original}-dirty-marker" if original else "dirty-marker"
        combo.setEditText(dirty_value) if combo.isEditable() else combo.setCurrentText(dirty_value)
        # Force a non-default value even if setCurrentText no-ops on empty lists.
        if combo.currentText() == original:
            combo.addItem(dirty_value)
            combo.setCurrentText(dirty_value)
        self.assertTrue(self.window._config_tab_has_unsaved_changes())
        saved_before = dict(self.window._config_ui_saved_snapshot)

        self.window._focus_settings_section("context")
        self.assertTrue(
            self.window._config_tab_has_unsaved_changes(),
            "partial load of context must not fold dirty models into baseline",
        )
        self.assertEqual(
            self.window._config_ui_saved_snapshot.get("batch_model"),
            saved_before.get("batch_model"),
        )
        self.assertNotEqual(
            self.window._current_config_ui_snapshot().get("batch_model"),
            self.window._config_ui_saved_snapshot.get("batch_model"),
        )

    def test_doctor_details_toggle_has_stable_fixed_width(self) -> None:
        self.window = MainWindow()
        toggle = self.window.doctor_details_toggle
        self.assertGreaterEqual(toggle.minimumWidth(), 96)
        self.assertEqual(toggle.minimumWidth(), toggle.maximumWidth())

    def test_import_does_not_load_connection_test_event_loop(self) -> None:
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import gui_qt.app, sys; assert 'asyncio' not in sys.modules"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_same_project_bar_mode_keeps_layout(self) -> None:
        self.window = MainWindow()
        layout = self.window.global_project_bar_layout
        with mock.patch.object(layout, "removeWidget", wraps=layout.removeWidget) as remove:
            self.window._set_project_bar_compact(True)
            self.window._set_project_bar_compact(True)
            remove.assert_not_called()
            self.window._set_project_bar_compact(False)
            self.assertEqual(remove.call_count, 3)
            self.window._set_project_bar_compact(False)
            self.assertEqual(remove.call_count, 3)

    def test_repeated_shell_refresh_queues_one_layout_followup(self) -> None:
        self.window = MainWindow()
        self.window._reflow_button_bars_deferred()
        with mock.patch("gui_qt.app.QTimer.singleShot") as schedule:
            for _ in range(4):
                self.window._reflow_button_bars()
            self.assertEqual(schedule.call_count, 1)
            self.window._reflow_button_bars_deferred()
            self.window._reflow_button_bars()
            self.assertEqual(schedule.call_count, 2)

    def test_status_tab_change_still_refreshes_readiness(self) -> None:
        self.window = MainWindow()
        with (
            mock.patch.object(
                self.window,
                "_update_probe_btn_enabled",
            ) as probe,
            mock.patch.object(
                self.window,
                "_update_split_btn_enabled",
            ) as split,
        ):
            self.window._sync_workbench_status_chrome(refresh_readiness=True)
            probe.assert_called_once()
            split.assert_called_once()

        with (
            mock.patch.object(
                self.window,
                "_update_probe_btn_enabled",
            ) as probe,
            mock.patch.object(
                self.window,
                "_update_split_btn_enabled",
            ) as split,
        ):
            self.window._sync_workbench_status_chrome(refresh_readiness=False)
            probe.assert_not_called()
            split.assert_not_called()

    def test_repeated_settings_navigation_does_not_read_task_manifests(self) -> None:
        self.window = MainWindow()
        self.window._focus_settings_section("workspace")
        self._app.processEvents()
        for route in ("project_prepare", "workbench:translation"):
            with self.subTest(route=route):
                self.window._activate_shell_route(route)
                self._app.processEvents()
                with mock.patch.object(
                    self.window,
                    "_current_diagnostics_manifest",
                    return_value=("", None),
                ) as read_manifest:
                    for _ in range(3):
                        for target in ("settings", route):
                            self.window.shell_nav.setCurrentRow(
                                self.window._shell_nav_rows[target]
                            )
                            self._app.processEvents()
                            self.assertEqual(self.window._current_shell_route(), target)
                    read_manifest.assert_not_called()

    def test_settings_keeps_hidden_project_preparation_layout(self) -> None:
        self.window = MainWindow()
        self.window._activate_shell_route("project_prepare")
        tabs = self.window.workbench_status_tabs
        visibility = [tabs.isTabVisible(i) for i in range(tabs.count())]
        layout = self.window.global_project_bar_layout
        with mock.patch.object(layout, "removeWidget", wraps=layout.removeWidget) as remove:
            for _ in range(3):
                self.window._activate_shell_route("settings")
                self._app.processEvents()
                self.assertEqual(tabs.currentIndex(), 0)
                self.assertEqual(
                    [tabs.isTabVisible(i) for i in range(tabs.count())], visibility
                )
                self.window._activate_shell_route("project_prepare")
                self._app.processEvents()
            remove.assert_not_called()

    def test_settings_keeps_batch_result_tab_selected(self) -> None:
        self.window = MainWindow()
        self.window._workflow = mock.Mock(manifest_path="")
        route = "workbench:translation"
        self.window._sync_workbench_status_surface(route)
        self.window.workbench_status_tabs.setCurrentIndex(2)
        for _ in range(3):
            self.window._activate_shell_route("settings")
            self._app.processEvents()
            self.assertEqual(self.window.workbench_status_tabs.currentIndex(), 2)
            self.window._activate_shell_route(route)
            self._app.processEvents()
            self.assertEqual(self.window.workbench_status_tabs.currentIndex(), 2)

    def test_deferred_context_refresh_does_not_restart_during_shutdown(self) -> None:
        self.window = MainWindow()
        self.window._shutdown_requested = True
        with mock.patch('gui_qt.app.ContextLibraryStatusJob') as job, mock.patch.object(
            self.window, '_render_context_library_panel',
        ) as render:
            self.window._refresh_context_library_panel_async(force=True)
        job.assert_not_called()
        render.assert_not_called()

    def test_automatic_status_scan_exits_without_task_confirmation(self) -> None:
        self.window = MainWindow()
        job = mock.Mock()
        self.window._context_library_status_job = job
        with mock.patch.object(
            self.window, '_confirm_active_tasks_before_close',
        ) as confirm, mock.patch('gui_qt.app.QTimer.singleShot'):
            event = QCloseEvent()
            self.window.closeEvent(event)
            self.assertFalse(event.isAccepted())
            confirm.assert_not_called()
            job.request_cancel.assert_called_once()
            self.assertTrue(self.window._shutdown_requested)
            self.assertTrue(self.window._shutdown_coordinator.in_progress)
            self.window._on_context_library_status_ready(None)
            self.window._shutdown_coordinator.check_now()
            self.assertTrue(self.window._shutdown_close_ready)
            final_event = QCloseEvent()
            self.window.closeEvent(final_event)
            self.assertTrue(final_event.isAccepted())

    def test_manual_task_still_confirms_without_automatic_scan_label(self) -> None:
        self.window = MainWindow()
        job = mock.Mock()
        self.window._context_library_status_job = job
        self.window._task_running = True
        try:
            with mock.patch.object(
                self.window, '_confirm_active_tasks_before_close', return_value=False,
            ) as confirm:
                event = QCloseEvent()
                self.window.closeEvent(event)
            self.assertFalse(event.isAccepted())
            confirm.assert_called_once_with(('当前工作台任务',))
            job.request_cancel.assert_not_called()
            self.assertFalse(self.window._shutdown_requested)
        finally:
            self.window._task_running = False
            self.window._context_library_status_job = None
