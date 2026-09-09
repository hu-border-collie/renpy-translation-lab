"""Independent construction tests for the #202 Phase D Context Settings page."""
from __future__ import annotations

import unittest

from gui_qt.settings.page_contract import SettingsIssue, SettingsPage
from gui_qt.settings.registry import CONTEXT_CONFIG_KEYS

try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
    ContextSettingsPage = None  # type: ignore[misc,assignment]
else:
    IMPORT_ERROR = None
    from gui_qt.settings.context_page import ContextSettingsPage

from tests import gui_test_support


@gui_test_support.skip_unless_gui(ContextSettingsPage is None, IMPORT_ERROR)
class ContextSettingsPageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.page = ContextSettingsPage()

    def tearDown(self) -> None:
        self.page.widget.deleteLater()
        self._app.processEvents()

    def test_page_satisfies_settings_protocol(self) -> None:
        self.assertIsInstance(self.page, SettingsPage)
        self.assertEqual(self.page.page_key, "context")
        self.assertEqual(self.page.nav_label, "上下文")
        self.assertEqual(self.page.config_keys, CONTEXT_CONFIG_KEYS)
        self.assertIn("rag_enabled", self.page.config_keys)
        self.assertIn("sync_rag_enabled", self.page.config_keys)
        self.assertIn("batch_project_analysis_model", self.page.config_keys)

    def test_load_collect_round_trip_owned_keys(self) -> None:
        self.page.load(
            {
                "rag_enabled": True,
                "source_index_enabled": True,
                "bootstrap_on_build": False,
                "sync_source_index_enabled": True,
                "sync_project_analysis_inject_enabled": True,
                "context_storage_location": "game",
                "sync_rag_enabled": True,
                "batch_project_analysis_model": "analysis-model",
                "batch_project_analysis_thinking_level": "high",
            }
        )
        collected = self.page.collect()
        self.assertTrue(collected["rag_enabled"])
        self.assertEqual(collected["context_storage_location"], "game")
        self.assertTrue(collected["sync_rag_enabled"])
        self.assertEqual(collected["batch_project_analysis_model"], "analysis-model")
        self.assertEqual(collected["batch_project_analysis_thinking_level"], "high")
        self.assertEqual(set(collected), set(self.page.config_keys))

    def test_reset_restores_last_loaded_baseline(self) -> None:
        self.page.load({"rag_enabled": True, "context_storage_location": "tool"})
        self.page.rag_enabled_cb.setChecked(False)
        self.page.context_storage_game_cb.setChecked(True)
        self.page.reset()
        collected = self.page.collect()
        self.assertTrue(collected["rag_enabled"])
        self.assertEqual(collected["context_storage_location"], "tool")

    def test_restore_does_not_replace_baseline(self) -> None:
        self.page.load({"rag_enabled": True})
        self.page.load({"rag_enabled": False}, restore=True)
        self.assertFalse(self.page.collect()["rag_enabled"])
        self.page.reset()
        self.assertTrue(self.page.collect()["rag_enabled"])

    def test_focus_issue_targets_owned_widget(self) -> None:
        issue = SettingsIssue("context", "sync_rag_enabled", "invalid")
        self.assertTrue(self.page.focus_issue(issue))
        issue = SettingsIssue("context", "rag_enabled", "invalid")
        self.assertTrue(self.page.focus_issue(issue))

    def test_set_task_running_disables_config_controls(self) -> None:
        self.assertTrue(self.page.rag_enabled_cb.isEnabled())
        self.page.set_task_running(True)
        self.assertFalse(self.page.rag_enabled_cb.isEnabled())
        self.assertFalse(self.page.field_widgets["sync_rag_enabled"].isEnabled())
        self.assertTrue(self.page.analysis_advanced_btn.isEnabled())
        self.page.set_task_running(False)
        self.assertTrue(self.page.rag_enabled_cb.isEnabled())

    def test_open_analysis_advanced_uses_host_callback(self) -> None:
        calls: list[str] = []
        page = ContextSettingsPage(
            on_open_analysis_advanced=lambda: calls.append("advanced")
        )
        try:
            page.analysis_advanced_btn.click()
            self.assertEqual(calls, ["advanced"])
        finally:
            page.widget.deleteLater()
            self._app.processEvents()


if __name__ == "__main__":
    unittest.main()
