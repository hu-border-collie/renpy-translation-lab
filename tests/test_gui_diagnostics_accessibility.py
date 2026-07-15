"""Diagnostics hierarchy and accessibility regression tests."""
from __future__ import annotations

import unittest

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QPushButton

    from gui_qt.app import MainWindow
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
        self.window.close()
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
