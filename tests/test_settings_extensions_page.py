"""Independent construction tests for the #202 Phase D Extensions Settings page."""
from __future__ import annotations

import unittest

from gui_qt.settings.page_contract import SettingsIssue, SettingsPage

try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
    ExtensionsSettingsPage = None  # type: ignore[misc,assignment]
else:
    IMPORT_ERROR = None
    from gui_qt.settings.extensions_page import ExtensionsSettingsPage

from tests import gui_test_support


@gui_test_support.skip_unless_gui(ExtensionsSettingsPage is None, IMPORT_ERROR)
class ExtensionsSettingsPageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.page = ExtensionsSettingsPage()

    def tearDown(self) -> None:
        self.page.widget.deleteLater()
        self._app.processEvents()

    def test_page_satisfies_settings_protocol(self) -> None:
        self.assertIsInstance(self.page, SettingsPage)
        self.assertEqual(self.page.page_key, "extensions")
        self.assertEqual(self.page.nav_label, "扩展")
        self.assertEqual(self.page.config_keys, frozenset())
        self.assertEqual(
            self.page.immediate_action_ids,
            frozenset({"install_relation_analyzer"}),
        )

    def test_collect_is_empty_and_load_reset_are_noops(self) -> None:
        self.assertEqual(self.page.collect(), {})
        self.page.load({"theme": "dark"})
        self.assertEqual(self.page.collect(), {})
        self.page.reset()
        self.assertEqual(self.page.validate(), [])

    def test_action_buttons_use_host_callbacks(self) -> None:
        calls: list[str] = []
        page = ExtensionsSettingsPage(
            on_install_relation_analyzer=lambda: calls.append("install"),
            on_open_docs=lambda: calls.append("docs"),
        )
        try:
            page.relation_analyzer_install_btn.click()
            page.relation_analyzer_docs_btn.click()
            self.assertEqual(calls, ["install", "docs"])
        finally:
            page.widget.deleteLater()
            self._app.processEvents()

    def test_attach_widget_aliases_copies_owned_controls(self) -> None:
        host = type("Host", (), {})()
        self.page.attach_widget_aliases(host)
        self.assertIs(
            host.relation_analyzer_install_btn,
            self.page.relation_analyzer_install_btn,
        )
        self.assertIs(
            host.relation_analyzer_status_label,
            self.page.relation_analyzer_status_label,
        )
        self.assertIs(
            host.relation_analyzer_docs_btn,
            self.page.relation_analyzer_docs_btn,
        )

    def test_focus_issue_targets_install_button(self) -> None:
        self.assertTrue(
            self.page.focus_issue(
                SettingsIssue("extensions", "install_relation_analyzer", "")
            )
        )

    def test_set_task_running_does_not_disable_install(self) -> None:
        self.assertTrue(self.page.relation_analyzer_install_btn.isEnabled())
        self.page.set_task_running(True)
        self.assertTrue(self.page.relation_analyzer_install_btn.isEnabled())
        self.assertTrue(self.page.relation_analyzer_docs_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
