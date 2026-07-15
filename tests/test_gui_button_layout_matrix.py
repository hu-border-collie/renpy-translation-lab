"""Multi-width button layout audit for the main GUI shell."""
from __future__ import annotations

import unittest

try:
    from PySide6.QtWidgets import QApplication, QPushButton

    from gui_qt.app import MainWindow
    from gui_qt.responsive_layout import FlowButtonBar, find_overlapping_buttons
    from gui_qt.work_modes import WorkMode
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support

# Representative shell widths after left-nav / chrome: window sizes users actually hit.
_WINDOW_SIZES = (
    (960, 700),
    (1024, 720),
    (1100, 760),
    (1280, 800),
    (1440, 900),
    (1600, 960),
)


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiButtonLayoutMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self.window = MainWindow()
        # Force all recovery buttons visible so the densest writeback strip is exercised.
        self.window._set_work_mode(WorkMode.BATCH_TRANSLATION, refresh_manifest_writeback=False)
        for name in (
            "recheck_btn",
            "check_issues_btn",
            "retry_btn",
            "retry_followup_btn",
            "repair_btn",
            "apply_failure_btn",
            "remediation_btn",
            "keyword_merge_writeback_btn",
            "split_submit_btn",
            "probe_btn",
            "split_btn",
            "resume_btn",
        ):
            btn = getattr(self.window, name, None)
            if btn is not None:
                btn.setVisible(True)
        self.window._set_writeback_issues_expanded(True)
        # Dense execute chrome (P2a advanced tools + main actions).
        if hasattr(self.window, "_focus_workbench_status_tab"):
            self.window._focus_workbench_status_tab(1)
        self.window._reflow_button_bars()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()

    def _assert_no_button_overlap(self, *, width: int, height: int) -> None:
        self.window.resize(width, height)
        self.window.show()
        for _ in range(8):
            self._app.processEvents()
        self.window._reflow_button_bars()
        for _ in range(4):
            self._app.processEvents()

        overlaps = find_overlapping_buttons(self.window, min_overlap_px=4)
        # Filter pairs that are the same physical control under different parents (none expected).
        serious = [
            hit
            for hit in overlaps
            if hit[0] != hit[1]
        ]
        self.assertEqual(
            serious,
            [],
            msg=f"Overlapping buttons at {width}x{height}: {serious[:8]}",
        )

    def test_no_button_overlap_across_common_window_sizes(self) -> None:
        for width, height in _WINDOW_SIZES:
            with self.subTest(width=width, height=height):
                self._assert_no_button_overlap(width=width, height=height)

    def test_no_overlap_batch_execute_and_result_dense(self) -> None:
        """Regression: advanced tools / stacked actions / writeback recovery strips."""
        for stage in (1, 2):
            self.window._focus_workbench_status_tab(stage)
            for name in (
                "doctor_btn",
                "bootstrap_work_btn",
                "translate_btn",
                "resume_btn",
                "probe_btn",
                "split_btn",
                "apply_btn",
                "recheck_btn",
                "check_issues_btn",
                "retry_btn",
                "retry_followup_btn",
                "repair_btn",
                "apply_failure_btn",
                "remediation_btn",
            ):
                btn = getattr(self.window, name, None)
                if btn is not None:
                    btn.setVisible(True)
            self.window._set_writeback_issues_expanded(True)
            for width, height in ((960, 700), (1100, 760), (1280, 800), (1600, 960)):
                with self.subTest(stage=stage, width=width, height=height):
                    self.window.resize(width, height)
                    self.window.show()
                    for _ in range(8):
                        self._app.processEvents()
                    self.window._reflow_button_bars()
                    for _ in range(20):
                        self._app.processEvents()
                    overlaps = [
                        hit
                        for hit in find_overlapping_buttons(self.window, min_overlap_px=4)
                        if hit[0] != hit[1]
                    ]
                    self.assertEqual(
                        overlaps,
                        [],
                        msg=f"stage={stage} {width}x{height}: {overlaps[:8]}",
                    )

    def test_action_panel_fits_single_row_without_prep_buttons(self) -> None:
        # Project prep moved to the global bar; task actions are translate-only.
        self.window.resize(960, 700)
        self.window.show()
        for _ in range(6):
            self._app.processEvents()
        self.window.action_panel.reflow(force=True)
        for _ in range(4):
            self._app.processEvents()
        self.assertEqual(self.window.action_panel._prep_buttons, [])
        # Translate + 停止 fit a single row at typical workbench width.
        self.assertTrue(self.window.action_panel._is_wide)

    def test_flow_bars_exist_for_main_strips(self) -> None:
        self.assertIsInstance(self.window.global_project_actions, FlowButtonBar)
        self.assertIsInstance(self.window.writeback_primary_bar, FlowButtonBar)
        self.assertIsInstance(self.window.writeback_issues_panel, FlowButtonBar)

    def test_batch_page_keeps_writeback_controls_on_status_page(self) -> None:
        """Task actions stay on the batch page; writeback remains result-owned."""
        self.window.resize(1600, 900)
        self.window.show()
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        page = self.window.batch_translation_page

        self.assertNotIn("apply", page.buttons)
        self.assertFalse(hasattr(page, "writeback_row"))
        self.assertNotIn(self.window.apply_btn, page.findChildren(QPushButton))
        self.assertIs(
            self.window.apply_btn.parentWidget(),
            self.window.writeback_primary_bar,
        )
        self.assertEqual(
            self.window.writeback_primary_bar.parentWidget().objectName(),
            "workbench_writeback_page",
        )

        page.set_controls(
            {
                "probe": (True, True, "试跑样本请求"),
                "split": (True, True, "拆分翻译包"),
            }
        )
        self.assertIs(page.buttons["probe"].parentWidget(), page.main_bar)
        self.assertIs(page.buttons["split"].parentWidget(), page.main_bar)
        self.assertFalse(page.buttons["probe"].isHidden())
        self.assertFalse(page.buttons["split"].isHidden())
        self.assertFalse(hasattr(page, "more_toggle_btn"))

class FlowButtonBarUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if QApplication is None:
            return
        cls._app = QApplication.instance() or QApplication([])

    @unittest.skipIf(QApplication is None, "GUI unavailable")
    def test_flow_bar_uses_multiple_rows_when_narrow(self) -> None:
        bar = FlowButtonBar(spacing=8)
        for label in ("查看问题清单", "生成补译包", "继续补译", "同步修补", "查看写回失败报告", "补救命令"):
            btn = QPushButton(label)
            bar.add_widget(btn, min_width=100)
        bar.finish_setup()
        bar.resize(360, 120)
        bar.show()
        self._app.processEvents()
        bar.reflow(force=True)
        self._app.processEvents()
        # More than one row layout item means wrapping happened.
        self.assertGreaterEqual(bar._root.count(), 2)

    @unittest.skipIf(QApplication is None, "GUI unavailable")
    def test_flow_bar_single_row_when_wide(self) -> None:
        bar = FlowButtonBar(spacing=8)
        for label in ("查看问题清单", "生成补译包", "继续补译", "同步修补", "重新检查", "补救命令"):
            btn = QPushButton(label)
            bar.add_widget(btn, min_width=100)
        bar.finish_setup()
        bar.resize(1200, 80)
        bar.show()
        self._app.processEvents()
        bar.reflow(force=True)
        self._app.processEvents()
        self.assertEqual(bar._root.count(), 1)


if __name__ == "__main__":
    unittest.main()
