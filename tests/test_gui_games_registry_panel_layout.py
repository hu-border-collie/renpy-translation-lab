"""Layout regression for settings · workspace registry toolbar."""
from __future__ import annotations

import unittest

try:
    from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QScrollArea

    from gui_qt.app import MainWindow
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiGamesRegistryPanelLayoutTests(unittest.TestCase):
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

    def test_workspace_toolbar_fits_settings_viewport_without_hscroll(self) -> None:
        self.window.resize(960, 700)
        self.window.show()
        for _ in range(6):
            self._app.processEvents()
        self.window._focus_settings_section("workspace")
        for _ in range(12):
            self._app.processEvents()

        page = self.window.settings_stack.currentWidget()
        self.assertIsInstance(page, QScrollArea)
        assert isinstance(page, QScrollArea)
        panel = self.window._games_registry_panel
        self.assertIsNotNone(panel)
        assert panel is not None

        # Toolbar must wrap instead of forcing a ~1250px min width.
        self.assertLessEqual(panel.sizeHint().width(), page.viewport().width() + 80)
        hbar = page.horizontalScrollBar()
        self.assertIsNotNone(hbar)
        assert hbar is not None
        self.assertEqual(hbar.maximum(), 0)

        # No header tool buttons should intersect.
        items: list[tuple[str, object]] = []
        from PySide6.QtCore import QRect

        for wgt in list(panel.findChildren(QLabel)) + list(panel.findChildren(QPushButton)):
            if not wgt.isVisible() or wgt.width() <= 1 or wgt.height() <= 1:
                continue
            top_left = wgt.mapTo(panel, wgt.rect().topLeft())
            if top_left.y() > 160:
                continue
            rect = QRect(top_left.x(), top_left.y(), wgt.width(), wgt.height())
            items.append((wgt.text() or wgt.objectName() or "?", rect))
        hits: list[tuple[str, str]] = []
        for i, (name_a, rect_a) in enumerate(items):
            for name_b, rect_b in items[i + 1 :]:
                inter = rect_a.intersected(rect_b)
                if inter.width() >= 3 and inter.height() >= 3:
                    hits.append((name_a, name_b))
        self.assertEqual(hits, [])

    def test_workspace_status_and_filter_combos_share_widths(self) -> None:
        """Play/translation status and engine/translation filters must match widths."""
        self.window.resize(960, 700)
        self.window.show()
        for _ in range(6):
            self._app.processEvents()
        self.window._focus_settings_section("workspace")
        for _ in range(12):
            self._app.processEvents()

        panel = self.window._games_registry_panel
        self.assertIsNotNone(panel)
        assert panel is not None

        play = panel._play_status_combo
        translation = panel._translation_status_combo
        self.assertEqual(play.width(), translation.width())
        self.assertGreaterEqual(play.width(), 120)

        engine = panel._engine_filter_combo
        tl_filter = panel._translation_filter_combo
        self.assertEqual(engine.width(), tl_filter.width())
        self.assertEqual(engine.minimumWidth(), tl_filter.minimumWidth())


if __name__ == "__main__":
    unittest.main()
