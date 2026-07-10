"""Broad GUI control audit: buttons/combos clip, overlap, and known equal pairs."""
from __future__ import annotations

import unittest

try:
    from PySide6.QtCore import QPoint, QRect, QSize
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QPushButton,
        QScrollArea,
        QStyle,
        QStyleOptionComboBox,
    )

    from gui_qt.app import MainWindow
    from gui_qt.responsive_layout import find_overlapping_buttons
    from gui_qt.widget_helpers import NoWheelComboBox
    from gui_qt.work_modes import WorkMode
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


def _process(app: QApplication, n: int = 6) -> None:
    for _ in range(n):
        app.processEvents()


def _combo_natural_width(combo: QComboBox) -> int:
    metrics = combo.fontMetrics()
    longest = ""
    longest_px = 0
    for index in range(combo.count()):
        text = combo.itemText(index)
        advance = metrics.horizontalAdvance(text)
        if advance > longest_px:
            longest_px = advance
            longest = text
    clone = NoWheelComboBox()
    for index in range(combo.count()):
        clone.addItem(combo.itemText(index))
    clone.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    opt = QStyleOptionComboBox()
    combo.initStyleOption(opt)
    opt.currentText = longest
    styled = combo.style().sizeFromContents(
        QStyle.ContentsType.CT_ComboBox,
        opt,
        QSize(longest_px, metrics.height()),
        combo,
    )
    return max(clone.sizeHint().width(), styled.width())


def _is_mapped_visible(widget, window) -> bool:
    """False when the widget is scrolled out of a QScrollArea viewport."""
    if not widget.isVisible() or widget.width() <= 1 or widget.height() <= 1:
        return False
    rect = QRect(widget.mapTo(window, QPoint(0, 0)), widget.size())
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QScrollArea):
            viewport = parent.viewport()
            view_rect = QRect(viewport.mapTo(window, QPoint(0, 0)), viewport.size())
            rect = rect.intersected(view_rect)
            if rect.isEmpty():
                return False
        parent = parent.parentWidget()
    window_rect = QRect(0, 0, window.width(), window.height())
    return not rect.intersected(window_rect).isEmpty()


def _button_text_fits(btn: QPushButton) -> bool:
    text = (btn.text() or "").replace("&", "")
    if not text:
        return True
    need = btn.fontMetrics().horizontalAdvance(text)
    return btn.width() >= need + 10


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiControlLayoutAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self.window = MainWindow()
        self.window.resize(1100, 760)
        self.window.show()
        _process(self._app, 8)

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()

    def _assert_visible_controls_healthy(self, *, context: str) -> None:
        overlaps = find_overlapping_buttons(self.window, min_overlap_px=3)
        self.assertEqual(overlaps, [], msg=f"button overlap @ {context}: {overlaps[:6]}")

        for btn in self.window.findChildren(QPushButton):
            if not _is_mapped_visible(btn, self.window):
                continue
            self.assertTrue(
                _button_text_fits(btn),
                msg=(
                    f"button clip @ {context}: "
                    f"{btn.objectName() or btn.text()!r} w={btn.width()} text={btn.text()!r}"
                ),
            )

        for combo in self.window.findChildren(QComboBox):
            if not _is_mapped_visible(combo, self.window):
                continue
            if combo.count() <= 0:
                continue
            natural = _combo_natural_width(combo)
            self.assertGreaterEqual(
                combo.width() + 1,
                natural,
                msg=(
                    f"combo clip @ {context}: "
                    f"{combo.objectName() or combo.currentText()!r} "
                    f"w={combo.width()} natural={natural}"
                ),
            )

    def test_work_modes_controls_no_clip_or_overlap(self) -> None:
        for mode in WorkMode:
            with self.subTest(mode=str(mode)):
                try:
                    self.window._set_work_mode(mode, refresh_manifest_writeback=False)
                except TypeError:
                    self.window._set_work_mode(mode)
                _process(self._app, 6)
                if hasattr(self.window, "_reflow_button_bars"):
                    self.window._reflow_button_bars()
                _process(self._app, 4)
                self._assert_visible_controls_healthy(context=f"mode={mode}")

    def test_settings_sections_controls_no_clip_or_overlap(self) -> None:
        self.window.tab_widget.setCurrentWidget(self.window._config_tab)
        _process(self._app, 6)
        for section_id in dict(self.window._settings_nav_rows):
            with self.subTest(section=section_id):
                self.window._focus_settings_section(section_id)
                _process(self._app, 10)
                if hasattr(self.window, "_reflow_button_bars"):
                    self.window._reflow_button_bars()
                _process(self._app, 4)
                self._assert_visible_controls_healthy(context=f"settings={section_id}")

        # Settings model/theme form combos should share the field column width.
        self.window._focus_settings_section("models")
        _process(self._app, 8)
        model_pair = (
            self.window.sync_model_combo,
            self.window.sync_embedding_combo,
            self.window.batch_model_combo,
            self.window.batch_embedding_combo,
            self.window.batch_thinking_combo,
        )
        widths = {combo.width() for combo in model_pair if combo.isVisible()}
        self.assertEqual(len(widths), 1, msg=f"model combos unequal widths: {widths}")

    def test_workspace_registry_combo_pairs(self) -> None:
        self.window.tab_widget.setCurrentWidget(self.window._config_tab)
        self.window._focus_settings_section("workspace")
        _process(self._app, 12)
        panel = self.window._games_registry_panel
        self.assertIsNotNone(panel)
        assert panel is not None
        panel._reflow_uniform_combos()
        _process(self._app, 4)

        self.assertEqual(
            panel._engine_filter_combo.width(),
            panel._translation_filter_combo.width(),
        )
        self.assertEqual(
            panel._engine_filter_combo.width(),
            panel._sort_combo.width(),
        )
        self.assertEqual(
            panel._play_status_combo.width(),
            panel._translation_status_combo.width(),
        )
        # Compact — not stretched to the full form field like the name line edit.
        self.assertLess(
            panel._play_status_combo.width(),
            panel._name_edit.width() // 2,
        )


if __name__ == "__main__":
    unittest.main()
