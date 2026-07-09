"""Tests for workbench left nav + per-mode sessions (GUI IA P1a / #160)."""
from __future__ import annotations

import unittest

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
    from gui_qt.work_modes import WorkMode, WorkbenchNavItem
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    Qt = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from gui_qt.work_modes import (
    WORKBENCH_NAV_ORDER,
    default_work_mode_for_nav,
    workbench_nav_for_work_mode,
    workbench_nav_spec,
)
from gui_qt.workbench_session import WorkbenchModeSession
from tests import gui_test_support


class WorkbenchNavMetaTests(unittest.TestCase):
    def test_nav_order_starts_with_batch(self) -> None:
        self.assertEqual(WORKBENCH_NAV_ORDER[0], WorkbenchNavItem.BATCH_TRANSLATION)

    def test_nav_for_work_modes(self) -> None:
        self.assertEqual(
            workbench_nav_for_work_mode(WorkMode.KEYWORD_EXTRACTION),
            WorkbenchNavItem.KEYWORDS,
        )
        self.assertEqual(
            workbench_nav_for_work_mode(WorkMode.BOOTSTRAP_SOURCE_INDEX),
            WorkbenchNavItem.CONTEXT,
        )

    def test_default_mode_for_keywords_is_batch_keyword(self) -> None:
        self.assertEqual(
            default_work_mode_for_nav(WorkbenchNavItem.KEYWORDS),
            WorkMode.KEYWORD_EXTRACTION,
        )

    def test_keywords_nav_shows_submode(self) -> None:
        self.assertTrue(workbench_nav_spec(WorkbenchNavItem.KEYWORDS).show_submode)
        self.assertFalse(workbench_nav_spec(WorkbenchNavItem.BATCH_TRANSLATION).show_submode)


class WorkbenchSessionTests(unittest.TestCase):
    def test_empty_session_flag(self) -> None:
        self.assertTrue(WorkbenchModeSession().is_empty())
        self.assertFalse(
            WorkbenchModeSession(writeback_manifest_path="C:/m.json").is_empty()
        )


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiWorkbenchNavTests(unittest.TestCase):
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

    def test_default_nav_is_batch_translation(self) -> None:
        self.assertEqual(self.window._work_mode, WorkMode.BATCH_TRANSLATION)
        current = self.window.workbench_nav.currentItem()
        self.assertIsNotNone(current)
        assert current is not None
        self.assertEqual(
            current.data(Qt.ItemDataRole.UserRole),
            WorkbenchNavItem.BATCH_TRANSLATION.value,
        )

    def test_nav_has_five_items(self) -> None:
        self.assertEqual(self.window.workbench_nav.count(), 5)
        self.assertEqual(self.window.workbench_stack.count(), 5)

    def test_switch_nav_preserves_writeback_path_session(self) -> None:
        self.window._writeback_manifest_path = "C:/batch/manifest.json"
        self.window._keyword_merge_candidates_path = ""
        self.window._completed_manifest_snapshot = {
            "manifest_path": "C:/batch/manifest.json",
            "heading": "done",
        }

        self.window._set_work_mode(
            WorkMode.KEYWORD_EXTRACTION,
            refresh_manifest_writeback=False,
        )
        self.assertEqual(self.window._work_mode, WorkMode.KEYWORD_EXTRACTION)
        # Active session for keywords starts clean.
        self.assertEqual(self.window._writeback_manifest_path, "")
        self.assertIsNone(self.window._completed_manifest_snapshot)

        # Batch session still holds prior state.
        batch_session = self.window._mode_sessions[WorkMode.BATCH_TRANSLATION]
        self.assertEqual(batch_session.writeback_manifest_path, "C:/batch/manifest.json")
        self.assertIsNotNone(batch_session.completed_manifest_snapshot)

        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.assertEqual(
            self.window._writeback_manifest_path,
            "C:/batch/manifest.json",
        )
        self.assertEqual(
            self.window._completed_manifest_snapshot,
            {"manifest_path": "C:/batch/manifest.json", "heading": "done"},
        )

    def test_running_disables_workbench_nav(self) -> None:
        self.window._set_task_running(True)
        self.assertFalse(self.window.workbench_nav.isEnabled())
        self.window._set_task_running(False)
        self.assertTrue(self.window.workbench_nav.isEnabled())

    def test_keywords_nav_shows_submode_combo(self) -> None:
        self.window._set_work_mode(
            WorkMode.KEYWORD_EXTRACTION,
            refresh_manifest_writeback=False,
        )
        self.assertFalse(self.window.work_submode_combo.isHidden())
        self.assertGreaterEqual(self.window.work_submode_combo.count(), 2)

        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.assertTrue(self.window.work_submode_combo.isHidden())

    def test_game_root_switch_clears_sessions(self) -> None:
        self.window._mode_sessions[WorkMode.BATCH_TRANSLATION] = WorkbenchModeSession(
            writeback_manifest_path="C:/old.json",
        )
        self.window._clear_all_mode_sessions()
        self.assertEqual(self.window._mode_sessions, {})


if __name__ == "__main__":
    unittest.main()
