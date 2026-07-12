"""Tests for workbench left nav + per-mode sessions (GUI IA P1a / #160)."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    Qt = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

# Pure work_modes imports stay outside the GUI try so meta tests run without PySide6.
from gui_qt.work_modes import (
    WORKBENCH_NAV_ORDER,
    WorkMode,
    WorkbenchNavItem,
    default_work_mode_for_nav,
    workbench_nav_for_work_mode,
    workbench_nav_spec,
)
from gui_qt.workbench_session import WorkbenchModeSession
from gui_qt.workbench import WorkbenchPage, WorkbenchPageActions
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

    def test_page_contract_exposes_migration_boundary(self) -> None:
        self.assertIn("supported_modes", WorkbenchPage.__annotations__)
        for method in ("set_action_callbacks", "activate", "set_task_running", "reset_project"):
            self.assertTrue(hasattr(WorkbenchPage, method))
        self.assertIsNone(WorkbenchPageActions().start)


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
        self.assertTrue(self.window.work_submode_combo.isHidden())
        self.assertFalse(self.window.keywords_page.mode_combo.isHidden())
        self.assertGreaterEqual(self.window.keywords_page.mode_combo.count(), 2)

        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.assertTrue(self.window.work_submode_combo.isHidden())

    def test_keywords_nav_restores_last_selected_submode_without_session_data(self) -> None:
        self.window._set_work_mode(WorkMode.SYNC_KEYWORD_EXTRACTION, refresh_manifest_writeback=False)
        self.window._set_work_mode(WorkMode.BATCH_TRANSLATION, refresh_manifest_writeback=False)
        self.window.workbench_nav.setCurrentRow(WORKBENCH_NAV_ORDER.index(WorkbenchNavItem.KEYWORDS))
        self.assertEqual(self.window._work_mode, WorkMode.SYNC_KEYWORD_EXTRACTION)

    def test_revision_nav_restores_last_selected_submode_without_session_data(self) -> None:
        self.window._set_work_mode(WorkMode.SYNC_REVISION, refresh_manifest_writeback=False)
        self.window._set_work_mode(WorkMode.BATCH_TRANSLATION, refresh_manifest_writeback=False)
        self.window.workbench_nav.setCurrentRow(WORKBENCH_NAV_ORDER.index(WorkbenchNavItem.REVISION))
        self.assertEqual(self.window._work_mode, WorkMode.SYNC_REVISION)

    def test_game_root_switch_clears_sessions(self) -> None:
        """Drive real _switch_game_root wiring, not only the clear helper."""
        self.window._set_work_mode(
            WorkMode.SYNC_KEYWORD_EXTRACTION,
            refresh_manifest_writeback=False,
        )
        self.window._mode_sessions[WorkMode.KEYWORD_EXTRACTION] = WorkbenchModeSession(
            writeback_manifest_path="C:/keyword-old.json",
        )
        self.window._mode_sessions[WorkMode.BATCH_TRANSLATION] = WorkbenchModeSession(
            writeback_manifest_path="C:/batch-old.json",
        )
        self.window._writeback_manifest_path = "C:/batch-old.json"

        with (
            mock.patch.object(
                self.window.state,
                "set_game_root",
                return_value=(Path("C:/Games/Example/work"), False),
            ),
            mock.patch.object(self.window.runner, "is_running", return_value=False),
            mock.patch.object(self.window, "_is_doctor_running", return_value=False),
            mock.patch.object(self.window, "_load_config_to_ui"),
            mock.patch.object(self.window, "_refresh_diagnostics_context"),
            mock.patch.object(self.window, "_invalidate_manifest_caches"),
            mock.patch.object(self.window, "_apply_work_mode_ui"),
        ):
            ok = self.window._switch_game_root("C:/Games/Example/work")

        self.assertTrue(ok)
        # The active keyword page receives a new stale session, never the old path.
        keyword_session = self.window._mode_sessions[WorkMode.KEYWORD_EXTRACTION]
        self.assertEqual(keyword_session.writeback_manifest_path, "")
        # Active mode is re-captured after clear; old writeback path must be gone.
        self.assertEqual(self.window._writeback_manifest_path, "")
        active = self.window._mode_sessions.get(self.window._work_mode)
        if active is not None:
            self.assertEqual(active.writeback_manifest_path, "")
        self.assertEqual(
            self.window._last_mode_by_nav[WorkbenchNavItem.KEYWORDS],
            WorkMode.KEYWORD_EXTRACTION,
        )
        self.assertEqual(self.window._work_mode, WorkMode.KEYWORD_EXTRACTION)

    def test_project_switch_resets_dormant_keywords_page(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        page = self.window.keywords_page
        page.set_controls(
            start_enabled=True,
            resume_enabled=True,
            resume_visible=True,
            resume_label="继续提取",
            merge_enabled=True,
            merge_message="关键词候选已就绪。",
        )

        with (
            mock.patch.object(
                self.window.state,
                "set_game_root",
                return_value=(Path("C:/Games/Example/work"), False),
            ),
            mock.patch.object(self.window.runner, "is_running", return_value=False),
            mock.patch.object(self.window, "_is_doctor_running", return_value=False),
            mock.patch.object(self.window, "_load_config_to_ui"),
            mock.patch.object(self.window, "_refresh_diagnostics_context"),
            mock.patch.object(self.window, "_invalidate_manifest_caches"),
            mock.patch.object(self.window, "_apply_work_mode_ui"),
        ):
            self.assertTrue(self.window._switch_game_root("C:/Games/Example/work"))

        self.assertFalse(page.start_btn.isEnabled())
        self.assertFalse(page.merge_btn.isEnabled())
        self.assertIn("项目已切换", page.result_hint.text())

    def test_project_switch_resets_dormant_revision_page(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        page = self.window.revision_page
        page.set_controls(
            start_enabled=True,
            resume_enabled=True,
            resume_visible=True,
            resume_label="继续订正",
            writeback_enabled=True,
            result_message="订正预览已通过。",
        )

        with (
            mock.patch.object(
                self.window.state,
                "set_game_root",
                return_value=(Path("C:/Games/Example/work"), False),
            ),
            mock.patch.object(self.window.runner, "is_running", return_value=False),
            mock.patch.object(self.window, "_is_doctor_running", return_value=False),
            mock.patch.object(self.window, "_load_config_to_ui"),
            mock.patch.object(self.window, "_refresh_diagnostics_context"),
            mock.patch.object(self.window, "_invalidate_manifest_caches"),
            mock.patch.object(self.window, "_apply_work_mode_ui"),
        ):
            self.assertTrue(self.window._switch_game_root("C:/Games/Example/work"))

        self.assertFalse(page.start_btn.isEnabled())
        self.assertFalse(page.writeback_btn.isEnabled())
        self.assertIn("项目已切换", page.result_hint.text())


if __name__ == "__main__":
    unittest.main()
