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
        panel.resize(1000, 80)
        panel.show()
        self._app.processEvents()
        self.assertTrue(panel._is_wide)
        self.assertEqual(panel._root.count(), 1)

    def test_stacks_rows_when_narrow(self):
        panel = self._build_panel()
        panel.resize(520, 120)
        panel.show()
        self._app.processEvents()
        self.assertFalse(panel._is_wide)
        self.assertEqual(panel._root.count(), 3)

    def test_resize_does_not_hang(self):
        panel = self._build_panel()
        panel.resize(900, 80)
        panel.show()
        for width in (900, 500, 1000, 480, 960):
            panel.resize(width, 80)
            for _ in range(5):
                self._app.processEvents()
        self.assertIn(panel._is_wide, (True, False))


if __name__ == "__main__":
    unittest.main()