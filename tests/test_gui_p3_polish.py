"""GUI IA P3 polish: empty states, resume gate, splitter idle restore (#166)."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from gui_qt.work_modes import WorkMode
from tests import gui_test_support


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiP3PolishTests(unittest.TestCase):
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

    def test_resume_disabled_without_resumable_task(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window.state.get_game_root = lambda: Path("C:/game/work")  # type: ignore[method-assign]
        self.window.state.get_latest_manifest_path_for_mode = lambda *_a, **_k: None  # type: ignore[method-assign]
        self.window._update_resume_btn_enabled(running=False)
        self.assertFalse(self.window.resume_btn.isEnabled())
        tip = self.window.resume_btn.toolTip()
        self.assertTrue(tip)
        self.assertIn("未找到", tip)

    def test_resume_enabled_when_manifest_loadable(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window.state.get_game_root = lambda: Path("C:/game/work")  # type: ignore[method-assign]
        self.window.state.get_latest_manifest_path_for_mode = (  # type: ignore[method-assign]
            lambda *_a, **_k: "C:/game/work/logs/manifest.json"
        )
        self.window.state.load_resume_manifest = lambda *_a, **_k: {"version": 1}  # type: ignore[method-assign]
        self.window._update_resume_btn_enabled(running=False)
        self.assertTrue(self.window.resume_btn.isEnabled())

    def test_doctor_empty_state_visible_before_check(self) -> None:
        self.window._doctor_check_completed = False
        self.window._sync_workbench_empty_states()
        self.assertFalse(self.window.doctor_empty_state.isHidden())
        self.window._doctor_check_completed = True
        self.window._sync_workbench_empty_states()
        self.assertTrue(self.window.doctor_empty_state.isHidden())

    def test_workflow_empty_state_without_project(self) -> None:
        self.window.state.get_game_root = lambda: None  # type: ignore[method-assign]
        self.window._workflow = None
        self.window._writeback_manifest_path = ""
        self.window._sync_workbench_empty_states()
        self.assertFalse(self.window.workflow_empty_state.isHidden())

    def test_workflow_empty_cta_button_fully_visible(self) -> None:
        """Empty-state CTA must not be clipped by the progress column."""
        from PySide6.QtCore import QPoint, QRect
        from PySide6.QtWidgets import QScrollArea

        from gui_qt.work_modes import WorkMode

        self.window.resize(1100, 700)
        self.window.show()
        for _ in range(8):
            self._app.processEvents()
        self.window._set_work_mode(WorkMode.BOOTSTRAP_RAG, refresh_manifest_writeback=False)
        self.window.state.get_game_root = lambda: None  # type: ignore[method-assign]
        self.window._workflow = None
        self.window._writeback_manifest_path = ""
        self.window._set_workflow_summary(
            "idle",
            "上下文库",
            "选择项目后可预建记忆库或原文索引。",
            [],
        )
        if hasattr(self.window, "workbench_status_tabs"):
            self.window.workbench_status_tabs.setCurrentIndex(1)
        for _ in range(6):
            self._app.processEvents()
        # Flush deferred ensureWidgetVisible scroll.
        self.window._ensure_workflow_empty_cta_visible()
        for _ in range(12):
            self._app.processEvents()

        empty = self.window.workflow_empty_state
        self.assertFalse(empty.isHidden())
        btn = empty._action_btn
        self.assertIsNotNone(btn)
        assert btn is not None
        self.assertEqual(btn.text(), "去环境检查")
        self.assertFalse(btn.isHidden())
        # Button geometry fully inside empty-state widget.
        btn_in_empty = QRect(btn.mapTo(empty, QPoint(0, 0)), btn.size())
        self.assertTrue(
            empty.rect().contains(btn_in_empty),
            msg=f"btn {btn_in_empty} not in empty {empty.rect()}",
        )
        # After auto-scroll, button must lie inside each ancestor scroll viewport.
        parent = btn.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                viewport = parent.viewport()
                mapped = QRect(btn.mapTo(viewport, QPoint(0, 0)), btn.size())
                self.assertTrue(
                    viewport.rect().contains(mapped),
                    msg=(
                        f"btn {mapped} not in scroll {parent.objectName()} "
                        f"viewport {viewport.rect()}"
                    ),
                )
            parent = parent.parentWidget()
        # Competing summary chrome must yield space to the empty CTA.
        self.assertTrue(self.window.workflow_message_label.isHidden())

    def test_restore_diagnostics_splitter_idle(self) -> None:
        self.window.resize(1280, 900)
        self.window.show()
        self.window.tab_widget.setCurrentWidget(self.window._diagnostics_tab)
        for _ in range(6):
            self._app.processEvents()
        self.window.diagnostics_splitter.setSizes([200, 400])
        for _ in range(4):
            self._app.processEvents()
        before = self.window.diagnostics_splitter.sizes()
        self.window._restore_diagnostics_splitter_idle()
        after = self.window.diagnostics_splitter.sizes()
        self.assertEqual(len(after), 2)
        # Idle should prefer a larger context share than a 200/400 running layout.
        self.assertGreaterEqual(after[0], before[0])


if __name__ == "__main__":
    unittest.main()
