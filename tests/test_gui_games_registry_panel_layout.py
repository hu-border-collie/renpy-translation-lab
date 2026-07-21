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

    def test_workspace_actions_are_grouped_and_details_follow_selection(self) -> None:
        panel = self.window._games_registry_panel
        root_layout = panel.layout()
        title_row = root_layout.itemAt(0).layout()
        self.assertEqual(
            title_row.itemAt(0).widget().objectName(), "diagnostics_section_label"
        )
        self.assertEqual(
            [
                widget.text() if hasattr(widget, "text") else widget.objectName()
                for widget in panel._toolbar._items
            ],
            ["刷新当前", "刷新全部", "games_registry_mode_host", "停止"],
        )
        self.assertEqual(
            [
                widget.text() if hasattr(widget, "text") else widget.objectName()
                for widget in panel._maintenance_toolbar._items
            ],
            ["扫描新项目", "导入游戏…", "打开分区时自动扫描新项目"],
        )
        self.assertEqual(
            [widget.text() for widget in panel._registry_toolbar._items],
            ["从 GAMES.md 导入", "同步 GAMES.md"],
        )

        self.window.resize(960, 700)
        self.window.show()
        self.window._focus_settings_section("workspace")
        for _ in range(12):
            self._app.processEvents()
        panel._table.clearSelection()
        for _ in range(4):
            self._app.processEvents()
        self.assertTrue(panel._edit_group.isHidden())
        if panel._table.rowCount() > 0:
            panel._table.selectRow(0)
            for _ in range(4):
                self._app.processEvents()
            self.assertFalse(panel._edit_group.isHidden())

    def test_table_column_path_is_last_stretch_and_drag_clamps_eui_min(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QHeaderView

        from gui_qt.games_registry_table import REGISTRY_TABLE_PATH_COLUMN

        self.window.resize(1200, 800)
        self.window.show()
        self.window._focus_settings_section("workspace")
        for _ in range(12):
            self._app.processEvents()
        panel = self.window._games_registry_panel
        assert panel is not None
        header = panel._table.horizontalHeader()
        # Path is last flex column (EUI: one flex; Carbon-style resource list).
        self.assertEqual(REGISTRY_TABLE_PATH_COLUMN, panel._table.columnCount() - 1)
        self.assertTrue(header.stretchLastSection())
        self.assertEqual(
            header.sectionResizeMode(REGISTRY_TABLE_PATH_COLUMN),
            QHeaderView.ResizeMode.Stretch,
        )
        self.assertEqual(
            header.sectionResizeMode(0),
            QHeaderView.ResizeMode.Interactive,
        )
        # Layout status (enum) min must cover longest known label sample.
        layout_min = panel._header_min_width(1)
        self.assertGreaterEqual(layout_min, panel._header_min_width(0))

        panel._reset_table_column_layout()
        min_name = panel._header_min_width(0)
        # Shrink below header min → clamp.
        panel._on_table_section_resized(0, 200, 10)
        self.assertGreaterEqual(panel._table.columnWidth(0), min_name)
        # Path stretch: resize handler is a no-op.
        path_col = REGISTRY_TABLE_PATH_COLUMN
        path_before = panel._table.columnWidth(path_col)
        panel._on_table_section_resized(path_col, path_before, path_before + 80)
        # Widen name freely; path absorbs leftover via last-section Stretch.
        panel._on_table_section_resized(0, min_name, min_name + 60)
        self.assertEqual(panel._table.columnWidth(0), min_name + 60)

        self.assertEqual(
            panel._table.textElideMode(),
            Qt.TextElideMode.ElideRight,
        )
        # Headers: 项目 | 目录状态 | 版本 | 游玩 | 翻译 | 路径
        headers = [
            panel._table.horizontalHeaderItem(i).text()
            for i in range(panel._table.columnCount())
        ]
        self.assertEqual(
            headers,
            ["项目", "目录状态", "版本", "游玩", "翻译", "路径"],
        )

    def test_detail_translation_status_matches_annotated_table_value(self) -> None:
        """Annotated statuses like 已完成（6.7 增量） must not collapse to 待确认."""
        from gui_qt.games_registry_panel import _set_status_combo_value
        from gui_qt.widget_helpers import NoWheelComboBox
        from games_registry import TRANSLATION_STATUSES

        combo = NoWheelComboBox()
        combo.addItems(sorted(TRANSLATION_STATUSES))
        _set_status_combo_value(combo, "已完成（6.7 增量）")
        self.assertEqual(combo.currentText(), "已完成（6.7 增量）")
        _set_status_combo_value(combo, "已完成")
        self.assertEqual(combo.currentText(), "已完成")

    def test_workspace_status_and_filter_combos_share_widths(self) -> None:
        """Play/translation status and engine/translation filters must match widths."""
        from gui_qt.games_registry_panel import _combo_natural_width

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
        self.assertEqual(play.minimumWidth(), play.maximumWidth())
        self.assertEqual(translation.minimumWidth(), translation.maximumWidth())
        # Content-sized pair — must not stretch to the form field column.
        self.assertLess(play.width(), panel._name_edit.width() // 2)
        # Must fit longest item under current style/font (no clipping).
        for combo in (play, translation):
            # Temporarily unlock to measure true natural width of this combo alone.
            saved = combo.width()
            natural = _combo_natural_width(combo)
            combo.setFixedWidth(saved)
            self.assertGreaterEqual(saved, natural)

        # Simulate first-show remeasure: fixed width must keep the pair equal.
        panel._reflow_uniform_combos()
        for _ in range(4):
            self._app.processEvents()
        self.assertEqual(play.width(), translation.width())

        engine = panel._engine_filter_combo
        tl_filter = panel._translation_filter_combo
        sort_combo = panel._sort_combo
        self.assertEqual(engine.width(), tl_filter.width())
        self.assertEqual(engine.width(), sort_combo.width())
        for combo in (engine, tl_filter, sort_combo):
            self.assertEqual(combo.minimumWidth(), combo.maximumWidth())
            saved = combo.width()
            natural = _combo_natural_width(combo)
            combo.setFixedWidth(saved)
            self.assertGreaterEqual(saved, natural)


if __name__ == "__main__":
    unittest.main()
