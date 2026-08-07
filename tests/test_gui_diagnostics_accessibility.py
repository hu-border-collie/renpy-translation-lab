"""Diagnostics hierarchy and accessibility regression tests."""
from __future__ import annotations

import unittest
import warnings

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QPushButton, QScrollArea

    from gui_qt.app import MainWindow
    from gui_qt.widget_helpers import ArrowKeyButtonFilter
    from gui_qt.responsive_layout import FlowButtonBar
    from gui_qt.theme_tokens import DARK_TOKENS, LIGHT_TOKENS
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    QPushButton = None  # type: ignore[assignment,misc]
    MainWindow = None  # type: ignore[assignment,misc]
    FlowButtonBar = None  # type: ignore[assignment,misc]
    DARK_TOKENS = {}  # type: ignore[assignment,misc]
    LIGHT_TOKENS = {}  # type: ignore[assignment,misc]
    Qt = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


def _activate_window(window) -> None:
    """Activate a window for keyboard-focus tests.

    Prefers the non-deprecated ``activateWindow``; the offscreen test platform
    never activates windows, so fall back to the deprecated
    ``QApplication.setActiveWindow`` there.
    """
    window.show()
    window.activateWindow()
    if QApplication.focusWidget() is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            QApplication.setActiveWindow(window)


def _relative_luminance(color: str) -> float:
    channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]

    def linear(value: float) -> float:
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (linear(value) for value in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(foreground: str, background: str) -> float:
    light, dark = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (light + 0.05) / (dark + 0.05)


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiDiagnosticsAccessibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()

    def test_diagnostics_uses_one_flat_responsive_toolbar(self) -> None:
        panel = self.window.diagnostics_action_panel
        self.assertIsInstance(panel, FlowButtonBar)
        self.assertEqual(panel.objectName(), "diagnostics_action_panel")
        self.assertEqual(panel.parentWidget().objectName(), "diagnostics_tab")
        self.assertNotEqual(panel.parentWidget().objectName(), "action_frame")
        self.assertLessEqual(len(self.window.diagnostics_hint_label.text()), 40)

        texts = {
            button.text()
            for button in panel.findChildren(QPushButton)
            if not button.isHidden()
        }
        self.assertEqual(texts, {"刷新上下文", "翻译 A/B 对比", "清空日志"})

    def test_diagnostics_toolbar_and_tabs_accept_keyboard_focus(self) -> None:
        for button in (
            self.window.refresh_diagnostics_btn,
            self.window.compare_variants_btn,
            self.window.clear_log_btn,
        ):
            self.assertEqual(button.focusPolicy(), Qt.FocusPolicy.StrongFocus)
        self.assertEqual(
            self.window.diagnostics_inner_tabs.tabBar().focusPolicy(),
            Qt.FocusPolicy.StrongFocus,
        )

    def test_doctor_details_toggle_accepts_keyboard_focus(self) -> None:
        toggle = self.window.doctor_details_toggle
        self.assertEqual(toggle.focusPolicy(), Qt.FocusPolicy.StrongFocus)
        self.assertEqual(toggle.accessibleName(), "更多详情")

    def test_split_status_table_accepts_keyboard_focus(self) -> None:
        table = self.window.split_status_table
        self.assertEqual(table.focusPolicy(), Qt.FocusPolicy.StrongFocus)
        self.assertEqual(table.accessibleName(), "拆分包状态表")

    def test_outer_tab_bar_is_excluded_from_focus_chain(self) -> None:
        # The outer tab bar is hidden (the sidebar switches pages); keeping it
        # focusable stalls Tab traversal because QTabWidget claims the event
        # for an invisible tab bar (#299).
        self.assertEqual(
            self.window.tab_widget.tabBar().focusPolicy(),
            Qt.FocusPolicy.NoFocus,
        )

    def test_sidebar_and_stage_tabs_are_excluded_from_focus_chain(self) -> None:
        # Arrow keys on the sidebar would switch pages and on the stage tab bar
        # would flip batch stages while the user explores page content; both
        # navigation rails stay out of the Tab chain (#299).
        self.assertEqual(
            self.window.shell_nav.focusPolicy(),
            Qt.FocusPolicy.NoFocus,
        )
        self.assertEqual(
            self.window.workbench_status_tabs.tabBar().focusPolicy(),
            Qt.FocusPolicy.NoFocus,
        )

    def test_scroll_areas_are_excluded_from_focus_chain(self) -> None:
        # A scroll area has no visible focus frame; Tab stopping on it between
        # buttons reads as a jumping focus box (#299).
        for name in (
            "workbench_content_scroll",
            "workflow_summary_scroll",
            "writeback_summary_scroll",
            "diagnostics_context_scroll",
            "diagnostics_commands_scroll",
        ):
            with self.subTest(scroll=name):
                scroll = self.window.findChild(QScrollArea, name)
                self.assertIsNotNone(scroll)
                self.assertEqual(scroll.focusPolicy(), Qt.FocusPolicy.NoFocus)

    def test_header_log_button_activates_with_enter(self) -> None:
        self.window.tab_widget.setCurrentWidget(self.window._workbench_tab)
        self.window.header_log_btn.setFocus()

        QTest.keyClick(self.window.header_log_btn, Qt.Key.Key_Enter)

        self.assertEqual(
            self.window.tab_widget.currentWidget(),
            self.window._diagnostics_tab,
        )
        self.assertTrue(self.window.header_log_btn.isChecked())

    def test_doctor_details_toggle_activates_with_enter(self) -> None:
        toggle = self.window.doctor_details_toggle
        toggle.setVisible(True)
        toggle.setFocus()
        before = self.window._doctor_details_expanded

        QTest.keyClick(toggle, Qt.Key.Key_Return)

        self.assertNotEqual(self.window._doctor_details_expanded, before)

    def test_arrow_key_button_filter_swallows_direction_keys(self) -> None:
        app = QApplication.instance()
        app_filter = getattr(app, "_renpy_lab_arrow_filter", None)
        self.assertIsNotNone(app_filter)
        inside = QPushButton(self.window)
        outside = QPushButton()
        for key in (
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
        ):
            event = QKeyEvent(
                QKeyEvent.Type.KeyPress,
                key,
                Qt.KeyboardModifier.NoModifier,
            )
            self.assertTrue(app_filter.eventFilter(inside, event))
            # Dialog buttons (top-level windows) keep arrow-key navigation.
            self.assertFalse(app_filter.eventFilter(outside, event))

    def test_arrow_keys_keep_focus_on_button(self) -> None:
        app = QApplication.instance()
        _activate_window(self.window)
        self.window.setFocus()
        self.window.header_log_btn.setFocus()
        self.assertEqual(app.focusWidget(), self.window.header_log_btn)

        QTest.keyClick(app.focusWidget(), Qt.Key.Key_Down)
        QTest.keyClick(app.focusWidget(), Qt.Key.Key_Right)

        self.assertEqual(app.focusWidget(), self.window.header_log_btn)

    def test_arrow_keys_switch_shell_page_when_unfocused(self) -> None:
        initial = self.window.shell_nav.currentRow()
        total = self.window.shell_nav.count()
        self.assertGreater(total, 0)

        down = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Down,
            Qt.KeyboardModifier.NoModifier,
        )
        self.window.keyPressEvent(down)
        self.assertEqual(
            self.window.shell_nav.currentRow(),
            (initial + 1) % total,
        )

        up = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Up,
            Qt.KeyboardModifier.NoModifier,
        )
        self.window.keyPressEvent(up)
        self.assertEqual(self.window.shell_nav.currentRow(), initial)

    def test_arrow_keys_skip_shell_section_headers(self) -> None:
        section_rows = {
            self.window.shell_nav.row(item)
            for item in self.window._shell_section_items
        }
        self.assertTrue(section_rows)
        self.window.shell_nav.setCurrentRow(0)
        down = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Down,
            Qt.KeyboardModifier.NoModifier,
        )
        for _ in range(30):
            self.window.keyPressEvent(down)
            self.assertNotIn(self.window.shell_nav.currentRow(), section_rows)

    def test_clicking_inert_background_returns_focus_to_window(self) -> None:
        app = QApplication.instance()
        _activate_window(self.window)
        self.window.setFocus()
        self.window.header_log_btn.setFocus()

        QTest.mouseClick(self.window.header_log_btn, Qt.MouseButton.LeftButton)
        self.assertEqual(app.focusWidget(), self.window.header_log_btn)

        # A label does not consume clicks; the window takes the focus back so
        # arrow-key page switching works again.
        QTest.mouseClick(self.window.shell_breadcrumb_label, Qt.MouseButton.LeftButton)
        app.processEvents()
        self.assertEqual(app.focusWidget(), self.window)

    def test_disabled_button_text_meets_normal_text_contrast(self) -> None:
        for theme, tokens in (("light", LIGHT_TOKENS), ("dark", DARK_TOKENS)):
            with self.subTest(theme=theme, kind="default"):
                self.assertGreaterEqual(
                    _contrast(tokens["fg_button_disabled"], tokens["bg_button_disabled"]),
                    4.5,
                )
            with self.subTest(theme=theme, kind="secondary"):
                self.assertGreaterEqual(
                    _contrast(
                        tokens["fg_secondary_disabled"],
                        tokens["bg_secondary_disabled"],
                    ),
                    4.5,
                )


if __name__ == "__main__":
    unittest.main()
