"""Tests for batch translation 准备/执行/结果 stages (GUI IA P1b / #161)."""
from __future__ import annotations

import unittest

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import (
        MainWindow,
        _BATCH_STAGE_EXECUTE,
        _BATCH_STAGE_LABELS,
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

    def test_batch_mode_shows_stage_bar_and_labels(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.assertFalse(self.window.batch_stage_bar.isHidden())
        self.assertEqual(self.window.workbench_status_tabs.tabText(0), _BATCH_STAGE_LABELS[0])
        self.assertEqual(self.window.workbench_status_tabs.tabText(1), _BATCH_STAGE_LABELS[1])
        self.assertEqual(self.window.workbench_status_tabs.tabText(2), _BATCH_STAGE_LABELS[2])
        self.assertEqual(len(self.window._batch_stage_buttons), 3)
        # Stage strip replaces the flat tab bar chrome in batch mode.
        tab_bar = self.window.workbench_status_tabs.tabBar()
        self.assertIsNotNone(tab_bar)
        assert tab_bar is not None
        self.assertTrue(tab_bar.isHidden())

    def test_non_batch_mode_hides_stage_bar(self) -> None:
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.assertTrue(self.window.batch_stage_bar.isHidden())
        self.assertEqual(self.window.workbench_status_tabs.tabText(0), "环境检查")
        tab_bar = self.window.workbench_status_tabs.tabBar()
        self.assertIsNotNone(tab_bar)
        assert tab_bar is not None
        self.assertFalse(tab_bar.isHidden())

    def test_stage_button_switches_status_tab(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._on_batch_stage_clicked(_BATCH_STAGE_PREPARE)
        self.assertEqual(
            self.window.workbench_status_tabs.currentIndex(),
            _BATCH_STAGE_PREPARE,
        )
        self.assertTrue(self.window._batch_stage_buttons[_BATCH_STAGE_PREPARE].isChecked())

        self.window._on_batch_stage_clicked(_BATCH_STAGE_RESULT)
        self.assertEqual(
            self.window.workbench_status_tabs.currentIndex(),
            _BATCH_STAGE_RESULT,
        )
        self.assertTrue(self.window._batch_stage_buttons[_BATCH_STAGE_RESULT].isChecked())

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
        self.assertTrue(self.window._batch_stage_buttons[_BATCH_STAGE_RESULT].isChecked())

    def test_prepare_stage_restored_with_mode_session(self) -> None:
        """PREPARE (0) must survive nav round-trip; not collapse to default EXECUTE."""
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._focus_workbench_status_tab(_BATCH_STAGE_PREPARE)
        self.assertEqual(
            self.window.workbench_status_tabs.currentIndex(),
            _BATCH_STAGE_PREPARE,
        )
        self.assertTrue(self.window._batch_stage_buttons[_BATCH_STAGE_PREPARE].isChecked())

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
            _BATCH_STAGE_PREPARE,
        )
        self.assertTrue(self.window._batch_stage_buttons[_BATCH_STAGE_PREPARE].isChecked())
        self.assertIn("环境检查", self.window.batch_stage_hint.text())

    def test_focus_helpers_map_to_stages(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._focus_workbench_status_tab(0)
        self.assertEqual(self.window._current_batch_stage_index(), _BATCH_STAGE_PREPARE)
        self.window._focus_workbench_status_tab(1)
        self.assertEqual(self.window._current_batch_stage_index(), _BATCH_STAGE_EXECUTE)
        self.window._focus_workbench_status_tab(2)
        self.assertEqual(self.window._current_batch_stage_index(), _BATCH_STAGE_RESULT)


if __name__ == "__main__":
    unittest.main()
