import unittest
from unittest import mock

try:
    from PySide6.QtWidgets import QApplication, QPushButton

    from gui_qt.widgets.responsive_layout import FlowButtonBar, ResponsiveActionPanel
except ImportError as exc:
    ResponsiveActionPanel = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(
    ResponsiveActionPanel is None,
    f"GUI dependencies are unavailable: {IMPORT_ERROR}",
)
class ResponsiveActionPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def _build_panel(self) -> ResponsiveActionPanel:
        panel = ResponsiveActionPanel(compact_width=700)
        panel.add_prep_button(QPushButton("环境检查"))
        panel.add_prep_button(QPushButton("准备工作目录"))
        panel.add_translate_button(QPushButton("开始翻译"))
        panel.add_translate_button(QPushButton("继续翻译"))
        panel.add_translate_trailing(QPushButton("停止"))
        panel.finish_setup()
        return panel

    def test_merges_rows_when_wide(self):
        panel = self._build_panel()
        panel.resize(1200, 80)
        panel.show()
        self._app.processEvents()
        self.assertTrue(panel._is_wide)
        self.assertEqual(panel._root.count(), 1)

    def test_stacks_rows_when_narrow(self):
        panel = self._build_panel()
        panel.resize(520, 140)
        panel.show()
        self._app.processEvents()
        self.assertFalse(panel._is_wide)
        self.assertEqual(panel._root.count(), 3)

    def test_stacks_when_wide_enough_for_breakpoint_but_not_buttons(self):
        """Left-nav workbench can be ~700–800px yet still too tight for one button row."""
        panel = self._build_panel()
        # compact_width is 700; estimated button row is larger → must stack.
        panel.resize(720, 140)
        panel.show()
        self._app.processEvents()
        self.assertFalse(panel._is_wide)
        self.assertEqual(panel._root.count(), 3)

    def test_stacked_rows_do_not_overlap(self):
        from gui_qt.widgets.responsive_layout import find_overlapping_buttons

        panel = self._build_panel()
        panel.resize(640, 120)
        panel.show()
        for _ in range(6):
            self._app.processEvents()
        panel.reflow(force=True)
        for _ in range(4):
            self._app.processEvents()
        self.assertFalse(panel._is_wide)
        self.assertEqual(find_overlapping_buttons(panel), [])
        # Translate-row buttons must sit fully below prep-row bottoms.
        prep = [b for b in panel._prep_buttons if not b.isHidden()]
        translate = [b for b in panel._translate_buttons if not b.isHidden()]
        if prep and translate:
            prep_bottom = max(b.geometry().bottom() for b in prep)
            translate_top = min(b.geometry().top() for b in translate)
            self.assertGreaterEqual(translate_top, prep_bottom)

    def test_reflow_does_not_leave_orphan_separators_blocking_clicks(self):
        """Regression: old VLine separators sat at 640x480 and stole mouse events."""
        from PySide6.QtWidgets import QFrame

        panel = self._build_panel()
        panel.resize(1200, 80)
        panel.show()
        for _ in range(4):
            self._app.processEvents()
        # Many reflows (as on resize/mode switch) used to accumulate separators.
        for width in (1200, 500, 1100, 480, 1280, 640, 900):
            panel.resize(width, 140)
            panel.reflow(force=True)
            for _ in range(3):
                self._app.processEvents()
        seps = [
            fr
            for fr in panel.findChildren(QFrame)
            if fr.objectName() == "action_separator" and fr.parentWidget() is panel
        ]
        # At most one live separator from the current layout mode.
        self.assertLessEqual(len(seps), 1)
        for fr in seps:
            self.assertLessEqual(fr.width(), 4)
            self.assertLessEqual(fr.height(), 40)
        # Buttons must be the topmost widget at their center (not a ghost separator).
        for btn in panel._prep_buttons + panel._translate_buttons:
            if btn.isHidden():
                continue
            center = btn.mapTo(panel, btn.rect().center())
            hit = panel.childAt(center)
            self.assertIs(hit, btn, msg=f"{btn.text()} blocked by {hit}")

    def test_resize_does_not_hang(self):
        panel = self._build_panel()
        panel.resize(900, 140)
        panel.show()
        for width in (900, 500, 1000, 480, 960):
            panel.resize(width, 140)
            for _ in range(5):
                self._app.processEvents()
        self.assertIn(panel._is_wide, (True, False))

    def test_unchanged_reflow_keeps_layout_and_detects_hidden_buttons(self):
        panel = self._build_panel()
        panel.setProperty("flowWidthHint", 1400)
        panel.reflow()
        row = panel._root.itemAt(0).layout()
        with mock.patch.object(panel, "_rebuild_layout", wraps=panel._rebuild_layout) as rebuild:
            panel.reflow()
            rebuild.assert_not_called()
            panel._prep_buttons[0].hide()
            panel.reflow()
            rebuild.assert_called_once()
        self.assertTrue(panel._is_wide)
        new_row = panel._root.itemAt(0).layout()
        self.assertIsNot(new_row, row)
        self.assertNotIn(
            panel._prep_buttons[0],
            [new_row.itemAt(index).widget() for index in range(new_row.count())],
        )


@unittest.skipIf(ResponsiveActionPanel is None, f"GUI unavailable: {IMPORT_ERROR}")
class FlowButtonBarPerfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.bar = FlowButtonBar()
        self.bar.setProperty("flowWidthHint", 400)
        self.buttons = [QPushButton(str(index)) for index in range(3)]
        for button in self.buttons:
            self.bar.add_widget(button, min_width=100)
        self.bar.finish_setup()
        self.bar.reflow()

    def tearDown(self):
        self.bar.close()
        self.bar.deleteLater()
        self._app.processEvents()

    def test_unchanged_refresh_preserves_layout(self):
        row = self.bar._root.itemAt(0).layout()
        with mock.patch.object(self.bar, "_rebuild", wraps=self.bar._rebuild) as rebuild:
            for _ in range(10):
                self.bar.reflow()
            rebuild.assert_not_called()
        self.assertIs(self.bar._root.itemAt(0).layout(), row)

    def test_visibility_width_height_and_alignment_changes_update_layout(self):
        self.buttons[0].hide()
        self.bar.reflow()
        row = self.bar._root.itemAt(0).layout()
        self.assertEqual(row.itemAt(0).widget(), self.buttons[1])

        self.bar.setProperty("flowWidthHint", 150)
        self.bar.reflow()
        self.assertEqual(self.bar._row_count, 2)
        self.assertEqual(self.bar.minimumHeight(), 84)

        self.buttons[1].setMinimumHeight(52)
        self.bar.reflow()
        self.assertEqual(self.bar.minimumHeight(), 112)

        self.bar.add_stretch_end(False)
        self.bar.reflow()
        for index in range(self.bar._root.count()):
            row = self.bar._root.itemAt(index).layout()
            self.assertIsNotNone(row.itemAt(row.count() - 1).widget())

    def test_deferred_refreshes_coalesce_and_skip_unchanged_layout(self):
        with mock.patch.object(self.bar, "_rebuild", wraps=self.bar._rebuild) as rebuild:
            self.bar._schedule_reflow()
            self.bar._schedule_reflow()
            self._app.processEvents()
            rebuild.assert_not_called()
        self.assertFalse(self.bar._reflow_scheduled)
        with mock.patch.object(self.bar, "reflow", wraps=self.bar.reflow) as reflow:
            self.bar._schedule_reflow()
            self.bar._schedule_reflow()
            self._app.processEvents()
            reflow.assert_called_once()


if __name__ == "__main__":
    unittest.main()
