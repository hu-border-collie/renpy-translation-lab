import unittest

try:
    from PySide6.QtWidgets import QApplication, QPushButton

    from gui_qt.responsive_layout import ResponsiveActionPanel
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
        from gui_qt.responsive_layout import find_overlapping_buttons

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


if __name__ == "__main__":
    unittest.main()