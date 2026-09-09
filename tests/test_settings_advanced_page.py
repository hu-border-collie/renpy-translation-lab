"""Independent construction tests for the #202 Phase D Advanced Settings page."""
from __future__ import annotations

import unittest

from gui_qt.settings.page_contract import SettingsIssue, SettingsPage
from gui_qt.settings.registry import ADVANCED_CONFIG_KEYS

try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
    AdvancedSettingsPage = None  # type: ignore[misc,assignment]
else:
    IMPORT_ERROR = None
    from gui_qt.settings.advanced_page import AdvancedSettingsPage

from tests import gui_test_support


@gui_test_support.skip_unless_gui(AdvancedSettingsPage is None, IMPORT_ERROR)
class AdvancedSettingsPageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.page = AdvancedSettingsPage()

    def tearDown(self) -> None:
        self.page.widget.deleteLater()
        self._app.processEvents()

    def test_page_satisfies_settings_protocol(self) -> None:
        self.assertIsInstance(self.page, SettingsPage)
        self.assertEqual(self.page.page_key, "advanced")
        self.assertEqual(self.page.nav_label, "高级")
        self.assertEqual(self.page.config_keys, ADVANCED_CONFIG_KEYS)
        self.assertIn("sync_chunk_size", self.page.config_keys)
        self.assertIn("catalog_gemini_models", self.page.config_keys)
        self.assertIn("model_rotation_models", self.page.config_keys)
        self.assertNotIn("tl_subdir", self.page.config_keys)
        self.assertNotIn("sync_rag_enabled", self.page.config_keys)
        self.assertNotIn("game_root", self.page.config_keys)

    def test_load_collect_round_trip_owned_keys(self) -> None:
        self.page.load(
            {
                "sync_chunk_size": 42,
                "model_rotation_enabled": True,
                "catalog_gemini_models": ["gemini-experimental-foo"],
                "catalog_gemini_embedding_models": ["gemini-embedding-custom"],
            }
        )
        collected = self.page.collect()
        self.assertEqual(collected["sync_chunk_size"], 42)
        self.assertTrue(collected["model_rotation_enabled"])
        self.assertEqual(collected["catalog_gemini_models"], ["gemini-experimental-foo"])
        self.assertEqual(
            collected["catalog_gemini_embedding_models"],
            ["gemini-embedding-custom"],
        )
        self.assertEqual(set(collected), set(self.page.config_keys))

    def test_reset_restores_last_loaded_baseline(self) -> None:
        self.page.load({"sync_chunk_size": 42})
        self.page.field_widgets["sync_chunk_size"].setValue(7)
        self.page.reset()
        self.assertEqual(self.page.collect()["sync_chunk_size"], 42)

    def test_restore_does_not_replace_baseline(self) -> None:
        self.page.load({"sync_chunk_size": 42})
        self.page.load({"sync_chunk_size": 7}, restore=True)
        self.assertEqual(self.page.collect()["sync_chunk_size"], 7)
        self.page.reset()
        self.assertEqual(self.page.collect()["sync_chunk_size"], 42)

    def test_focus_issue_targets_owned_widget(self) -> None:
        issue = SettingsIssue("advanced", "sync_chunk_size", "invalid")
        self.assertTrue(self.page.focus_issue(issue))

    def test_set_task_running_disables_config_controls(self) -> None:
        widget = self.page.field_widgets["sync_chunk_size"]
        self.assertTrue(widget.isEnabled())
        self.page.set_task_running(True)
        self.assertFalse(widget.isEnabled())
        self.assertFalse(self.page.field_widgets["catalog_gemini_models"].isEnabled())
        self.page.set_task_running(False)
        self.assertTrue(widget.isEnabled())

    def test_rotation_checklist_tracks_enabled_toggle(self) -> None:
        enabled = self.page.field_widgets["model_rotation_enabled"]
        checklist = self.page.field_widgets["model_rotation_models"]
        enabled.setChecked(False)
        self.assertFalse(checklist.isEnabled())
        enabled.setChecked(True)
        self.assertTrue(checklist.isEnabled())
        self.page.set_task_running(True)
        self.assertFalse(checklist.isEnabled())
        self.page.set_task_running(False)
        self.assertTrue(checklist.isEnabled())


if __name__ == "__main__":
    unittest.main()
