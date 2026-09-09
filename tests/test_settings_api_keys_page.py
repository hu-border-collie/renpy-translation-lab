"""Independent construction tests for the #202 Phase D API Keys Settings page."""
from __future__ import annotations

import unittest

from gui_qt.settings.page_contract import SettingsIssue, SettingsPage

try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
    ApiKeysSettingsPage = None  # type: ignore[misc,assignment]
else:
    IMPORT_ERROR = None
    from gui_qt.settings.api_keys_page import ApiKeysSettingsPage

from tests import gui_test_support


@gui_test_support.skip_unless_gui(ApiKeysSettingsPage is None, IMPORT_ERROR)
class ApiKeysSettingsPageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.page = ApiKeysSettingsPage()

    def tearDown(self) -> None:
        self.page.widget.deleteLater()
        self._app.processEvents()

    def test_page_satisfies_settings_protocol(self) -> None:
        self.assertIsInstance(self.page, SettingsPage)
        self.assertEqual(self.page.page_key, "api_keys")
        self.assertEqual(self.page.nav_label, "密钥")
        self.assertEqual(self.page.config_keys, frozenset())
        self.assertEqual(
            self.page.immediate_action_ids,
            frozenset({"manage_gemini_keys", "manage_litellm_keys"}),
        )

    def test_collect_is_empty_and_load_reset_are_noops(self) -> None:
        self.assertEqual(self.page.collect(), {})
        self.page.load({"theme": "dark"})
        self.assertEqual(self.page.collect(), {})
        self.page.reset()
        self.assertEqual(self.page.validate(), [])

    def test_manage_buttons_use_host_callbacks(self) -> None:
        calls: list[str] = []
        page = ApiKeysSettingsPage(
            on_manage_gemini_keys=lambda: calls.append("gemini"),
            on_manage_litellm_keys=lambda: calls.append("litellm"),
        )
        try:
            page.api_btn.click()
            page.litellm_keys_manage_btn.click()
            self.assertEqual(calls, ["gemini", "litellm"])
        finally:
            page.widget.deleteLater()
            self._app.processEvents()

    def test_provider_combo_notifies_host(self) -> None:
        calls: list[str] = []
        page = ApiKeysSettingsPage(
            on_provider_changed=lambda: calls.append("changed"),
        )
        try:
            page.litellm_keys_provider_combo.addItem("OpenAI", "openai")
            page.litellm_keys_provider_combo.setCurrentIndex(0)
            self.assertEqual(calls, ["changed"])
        finally:
            page.widget.deleteLater()
            self._app.processEvents()

    def test_attach_widget_aliases_copies_owned_controls(self) -> None:
        host = type("Host", (), {})()
        self.page.attach_widget_aliases(host)
        self.assertIs(host.api_btn, self.page.api_btn)
        self.assertIs(host.api_status_label, self.page.api_status_label)
        self.assertIs(
            host.litellm_keys_manage_btn, self.page.litellm_keys_manage_btn
        )
        self.assertIs(
            host.litellm_keys_provider_combo, self.page.litellm_keys_provider_combo
        )

    def test_focus_issue_targets_owned_controls(self) -> None:
        self.assertTrue(
            self.page.focus_issue(SettingsIssue("api_keys", "manage_gemini_keys", ""))
        )
        self.assertTrue(
            self.page.focus_issue(SettingsIssue("api_keys", "manage_litellm_keys", ""))
        )

    def test_set_task_running_disables_key_actions(self) -> None:
        self.assertTrue(self.page.api_btn.isEnabled())
        self.page.set_task_running(True)
        self.assertFalse(self.page.api_btn.isEnabled())
        self.assertFalse(self.page.litellm_keys_manage_btn.isEnabled())
        self.assertFalse(self.page.litellm_keys_provider_combo.isEnabled())
        self.page.set_task_running(False)
        self.assertTrue(self.page.api_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
