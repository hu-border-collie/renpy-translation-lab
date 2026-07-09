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