"""Independent construction tests for the #202 Phase D Shortcuts Settings page."""
from __future__ import annotations

import unittest

from gui_qt.settings.page_contract import SettingsIssue, SettingsPage

try:
    from PySide6.QtWidgets import QApplication, QLabel
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    QLabel = None  # type: ignore[misc,assignment]
    IMPORT_ERROR = exc
    ShortcutsSettingsPage = None  # type: ignore[misc,assignment]
    shortcut_catalog = None  # type: ignore[misc,assignment]
else:
    IMPORT_ERROR = None
    from gui_qt.settings.shortcuts_page import ShortcutsSettingsPage, shortcut_catalog

from tests import gui_test_support


@gui_test_support.skip_unless_gui(ShortcutsSettingsPage is None, IMPORT_ERROR)
class ShortcutsSettingsPageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.page = ShortcutsSettingsPage(
            nav_labels=("项目与环境", "批量翻译", "设置")
        )

    def tearDown(self) -> None:
        self.page.widget.deleteLater()
        self._app.processEvents()

    def test_page_satisfies_settings_protocol(self) -> None:
        self.assertIsInstance(self.page, SettingsPage)
        self.assertEqual(self.page.page_key, "shortcuts")
        self.assertEqual(self.page.nav_label, "快捷键")
        self.assertEqual(self.page.config_keys, frozenset())
        self.assertEqual(self.page.immediate_action_ids, frozenset())

    def test_collect_is_empty_and_load_reset_are_noops(self) -> None:
        self.assertEqual(self.page.collect(), {})
        self.page.load({"theme": "dark"})
        self.assertEqual(self.page.collect(), {})
        self.page.reset()
        self.assertEqual(self.page.collect(), {})
        self.assertEqual(self.page.validate(), [])
        self.assertFalse(
            self.page.focus_issue(SettingsIssue("shortcuts", "theme", "n/a"))
        )

    def test_catalog_lists_core_bindings(self) -> None:
        labels = [
            widget.text()
            for widget in self.page.body.findChildren(QLabel)
            if widget.text().strip()
        ]
        joined = "\n".join(labels)
        for needle in (
            "Ctrl+D",
            "Ctrl+T",
            "Ctrl+K",
            "Ctrl+L",
            "Ctrl+Shift+L",
            "Ctrl+S",
            "Ctrl+1",
            "Ctrl+0",
            "项目与环境",
            "导航",
        ):
            self.assertIn(needle, joined)

    def test_shortcut_catalog_helper_caps_nav_at_nine(self) -> None:
        groups = shortcut_catalog([f"页{i}" for i in range(1, 12)])
        nav_rows = dict(groups)["导航"]
        self.assertEqual(nav_rows[-1], ("Ctrl+0", "打开「诊断与运行日志」"))
        self.assertEqual(nav_rows[0][0], "Ctrl+1")
        self.assertEqual(nav_rows[8][0], "Ctrl+9")
        self.assertEqual(len(nav_rows), 10)


if __name__ == "__main__":
    unittest.main()
