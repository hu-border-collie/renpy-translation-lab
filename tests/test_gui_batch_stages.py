"""Tests for shared workbench status tabs (env check / progress / result)."""
from __future__ import annotations

import unittest

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import (
        MainWindow,
        _BATCH_STAGE_EXECUTE,
        _BATCH_STAGE_PREPARE,
        _BATCH_STAGE_RESULT,
    )
    from gui_qt.work_modes import WorkMode
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiBatchStageTests(unittest.TestCase):
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

    def test_batch_mode_uses_flat_status_tabs(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.assertFalse(hasattr(self.window, "batch_stage_bar"))
        self.assertEqual(self.window.workbench_status_tabs.tabText(0), "环境检查")
        self.assertEqual(self.window.workbench_status_tabs.tabText(1), "翻译进度")
        self.assertEqual(self.window.workbench_status_tabs.tabText(2), "写回")
        tab_bar = self.window.workbench_status_tabs.tabBar()
        self.assertIsNotNone(tab_bar)
        assert tab_bar is not None
        self.assertFalse(tab_bar.isHidden())

    def test_non_batch_mode_uses_mode_specific_tab_labels(self) -> None:
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.assertEqual(self.window.workbench_status_tabs.tabText(0), "环境检查")
        self.assertEqual(self.window.workbench_status_tabs.tabText(1), "翻译进度")
        self.assertEqual(self.window.workbench_status_tabs.tabText(2), "写回说明")
        tab_bar = self.window.workbench_status_tabs.tabBar()
        self.assertIsNotNone(tab_bar)
        assert tab_bar is not None
        self.assertFalse(tab_bar.isHidden())

    def test_status_tab_switch_updates_stage_index(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._focus_workbench_status_tab(_BATCH_STAGE_PREPARE)
        self.assertEqual(
            self.window.workbench_status_tabs.currentIndex(),
            _BATCH_STAGE_EXECUTE,
        )

        self.window._focus_workbench_status_tab(_BATCH_STAGE_RESULT)
        self.assertEqual(
            self.window.workbench_status_tabs.currentIndex(),
            _BATCH_STAGE_RESULT,
        )

    def test_stage_index_restored_with_mode_session(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._focus_workbench_status_tab(_BATCH_STAGE_RESULT)
        self.assertEqual(
            self.window.workbench_status_tabs.currentIndex(),
            _BATCH_STAGE_RESULT,
        )

        self.window._set_work_mode(
            WorkMode.KEYWORD_EXTRACTION,
            refresh_manifest_writeback=False,
        )
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.assertEqual(
            self.window.workbench_status_tabs.currentIndex(),
            _BATCH_STAGE_RESULT,
        )

    def test_hidden_environment_stage_is_not_saved_as_task_progress(self) -> None:
        """Environment belongs to the project route, so task sessions stay on progress."""
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._focus_workbench_status_tab(_BATCH_STAGE_PREPARE)
        self.assertEqual(
            self.window.workbench_status_tabs.currentIndex(),
            _BATCH_STAGE_EXECUTE,
        )

        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.assertEqual(
            self.window.workbench_status_tabs.currentIndex(),
            _BATCH_STAGE_EXECUTE,
        )
        self.assertEqual(self.window.workbench_status_tabs.tabText(0), "环境检查")

    def test_focus_helpers_map_to_stages(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._focus_workbench_status_tab(0)
        self.assertEqual(self.window._current_batch_stage_index(), _BATCH_STAGE_EXECUTE)
        self.window._focus_workbench_status_tab(1)
        self.assertEqual(self.window._current_batch_stage_index(), _BATCH_STAGE_EXECUTE)
        self.window._focus_workbench_status_tab(2)
        self.assertEqual(self.window._current_batch_stage_index(), _BATCH_STAGE_RESULT)

    def test_batch_task_groups_stay_owned_by_batch_page_across_status_tabs(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        page = self.window.batch_translation_page
        for stage in (
            _BATCH_STAGE_EXECUTE,
            _BATCH_STAGE_RESULT,
        ):
            self.window._focus_workbench_status_tab(stage)
            self.assertIs(self.window.workbench_stack.currentWidget(), page)
            self.assertFalse(
                page.buttons["probe"].isHidden(),
                msg=f"probe should stay on batch (stage {stage})",
            )
            self.assertFalse(
                page.buttons["split"].isHidden(),
                msg=f"split should stay on batch (stage {stage})",
            )

        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.assertTrue(page.isHidden())


if __name__ == "__main__":
    unittest.main()
