"""Independent construction tests for the #202 Phase D Appearance Settings page."""
from __future__ import annotations

import unittest

from gui_qt.settings.page_contract import SettingsIssue, SettingsPage
from gui_qt.theme_helpers import THEME_DARK, THEME_SYSTEM

try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
    AppearanceSettingsPage = None  # type: ignore[misc,assignment]
else:
    IMPORT_ERROR = None
    from gui_qt.settings.appearance_page import AppearanceSettingsPage

from tests import gui_test_support


@gui_test_support.skip_unless_gui(AppearanceSettingsPage is None, IMPORT_ERROR)
class AppearanceSettingsPageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.page = AppearanceSettingsPage()

    def tearDown(self) -> None:
        self.page.widget.deleteLater()
        self._app.processEvents()

    def test_page_satisfies_settings_protocol(self) -> None:
        self.assertIsInstance(self.page, SettingsPage)
        self.assertEqual(self.page.page_key, "appearance")
        self.assertEqual(self.page.nav_label, "外观")
        self.assertEqual(self.page.config_keys, frozenset({"theme"}))
        self.assertIn("preview_theme", self.page.immediate_action_ids)
        self.assertIn("download_fonts", self.page.immediate_action_ids)

    def test_load_collect_round_trip_theme(self) -> None:
        self.page.load({"theme": THEME_DARK})
        collected = self.page.collect()
        self.assertEqual(collected["theme"], THEME_DARK)
        self.assertEqual(set(collected), set(self.page.config_keys))

    def test_reset_restores_last_loaded_baseline(self) -> None:
        self.page.load({"theme": THEME_DARK})
        self.page.theme_combo.setCurrentIndex(
            self.page.theme_combo.findData(THEME_SYSTEM)
        )
        self.page.reset()
        self.assertEqual(self.page.collect()["theme"], THEME_DARK)

    def test_restore_does_not_replace_baseline(self) -> None:
        self.page.load({"theme": THEME_DARK})
        self.page.load({"theme": THEME_SYSTEM}, restore=True)
        self.assertEqual(self.page.collect()["theme"], THEME_SYSTEM)
        self.page.reset()
        self.assertEqual(self.page.collect()["theme"], THEME_DARK)

    def test_focus_issue_targets_theme_combo(self) -> None:
        issue = SettingsIssue("appearance", "theme", "invalid")
        self.assertTrue(self.page.focus_issue(issue))

    def test_set_task_running_disables_theme_keeps_cancel_download(self) -> None:
        self.assertTrue(self.page.theme_combo.isEnabled())
        self.page.set_task_running(True)
        self.assertFalse(self.page.theme_combo.isEnabled())
        self.assertFalse(self.page.download_fonts_btn.isEnabled())
        busy_page = AppearanceSettingsPage(is_font_install_running=lambda: True)
        try:
            busy_page.set_task_running(True)
            self.assertTrue(busy_page.download_fonts_btn.isEnabled())
        finally:
            busy_page.widget.deleteLater()
            self._app.processEvents()
        self.page.set_task_running(False)
        self.assertTrue(self.page.theme_combo.isEnabled())

    def test_theme_preview_uses_host_callback(self) -> None:
        calls: list[str] = []
        page = AppearanceSettingsPage(
            on_theme_preview=lambda preference: calls.append(preference)
        )
        try:
            page.theme_combo.setCurrentIndex(page.theme_combo.findData(THEME_DARK))
            self.assertEqual(calls, [THEME_DARK])
            page.load({"theme": THEME_SYSTEM})
            self.assertEqual(calls, [THEME_DARK])
        finally:
            page.widget.deleteLater()
            self._app.processEvents()

    def test_download_fonts_uses_host_callback(self) -> None:
        calls: list[str] = []
        page = AppearanceSettingsPage(
            on_download_fonts=lambda: calls.append("download")
        )
        try:
            page.download_fonts_btn.click()
            self.assertEqual(calls, ["download"])
        finally:
            page.widget.deleteLater()
            self._app.processEvents()


if __name__ == "__main__":
    unittest.main()
