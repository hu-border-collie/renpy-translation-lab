"""Tests for the vendored, theme-aware Tabler icon provider."""
from __future__ import annotations

from pathlib import Path
import unittest

try:
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QPushButton

    from gui_qt.icon_provider import set_tabler_button_icon, tabler_icon
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


@gui_test_support.skip_unless_gui(QApplication is None, IMPORT_ERROR)
class GuiIconProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        cls._app = app or QApplication([])
        cls._resources = Path(__file__).resolve().parents[1] / "gui_qt" / "resources"

    def test_vendored_icon_renders_in_light_and_dark_themes(self) -> None:
        light = tabler_icon(self._resources, "language", dark=False)
        dark = tabler_icon(self._resources, "language", dark=True)

        self.assertFalse(light.isNull())
        self.assertFalse(dark.isNull())
        self.assertFalse(light.pixmap(18, 18, QIcon.Mode.Normal).isNull())
        self.assertFalse(dark.pixmap(18, 18, QIcon.Mode.Disabled).isNull())
        self.assertNotEqual(
            light.pixmap(18, 18).cacheKey(),
            dark.pixmap(18, 18).cacheKey(),
        )

    def test_button_helper_sets_consistent_icon_size(self) -> None:
        button = QPushButton("开始翻译")

        set_tabler_button_icon(
            button,
            self._resources,
            "language",
            dark=False,
            role="on_accent",
        )

        self.assertFalse(button.icon().isNull())
        self.assertEqual(button.iconSize(), QSize(18, 18))

    def test_missing_icon_fails_softly(self) -> None:
        icon = tabler_icon(self._resources, "does-not-exist", dark=False)

        self.assertTrue(icon.isNull())

    def test_unknown_role_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tabler_icon(self._resources, "language", dark=False, role="unknown")


if __name__ == "__main__":
    unittest.main()
