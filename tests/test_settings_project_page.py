"""Independent construction tests for the #202 Phase D Project Settings page."""
from __future__ import annotations

import unittest

from gui_qt.settings.page_contract import SettingsIssue, SettingsPage, SettingsPageActions
from gui_qt.settings.registry import PROJECT_CONFIG_KEYS

try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
    ProjectSettingsPage = None  # type: ignore[misc,assignment]
else:
    IMPORT_ERROR = None
    from gui_qt.settings.project_page import ProjectSettingsPage

from tests import gui_test_support


@gui_test_support.skip_unless_gui(ProjectSettingsPage is None, IMPORT_ERROR)
class ProjectSettingsPageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.page = ProjectSettingsPage()

    def tearDown(self) -> None:
        self.page.widget.deleteLater()
        self._app.processEvents()

    def test_page_satisfies_settings_protocol(self) -> None:
        self.assertIsInstance(self.page, SettingsPage)
        self.assertEqual(self.page.page_key, "project")
        self.assertEqual(self.page.nav_label, "项目")
        self.assertEqual(self.page.config_keys, PROJECT_CONFIG_KEYS)
        self.assertIn("tl_subdir", self.page.config_keys)
        self.assertIn("prepare_enabled", self.page.config_keys)
        self.assertNotIn("game_root", self.page.config_keys)

    def test_load_collect_round_trip_owned_keys(self) -> None:
        self.page.load(
            {
                "tl_subdir": "game/tl/custom",
                "glossary_file": "terms.json",
                "prepare_enabled": False,
                "include_files": ["script.rpy", "options.rpy"],
            }
        )
        collected = self.page.collect()
        self.assertEqual(collected["tl_subdir"], "game/tl/custom")
        self.assertEqual(collected["glossary_file"], "terms.json")
        self.assertFalse(collected["prepare_enabled"])
        self.assertEqual(collected["include_files"], "script.rpy\noptions.rpy")
        self.assertEqual(set(collected), set(self.page.config_keys))

    def test_reset_restores_last_loaded_baseline(self) -> None:
        self.page.load({"tl_subdir": "game/tl/saved"})
        self.page.field_widgets["tl_subdir"].setText("game/tl/dirty")
        self.page.reset()
        self.assertEqual(self.page.collect()["tl_subdir"], "game/tl/saved")

    def test_focus_issue_targets_owned_widget(self) -> None:
        issue = SettingsIssue("project", "glossary_file", "missing")
        self.assertTrue(self.page.focus_issue(issue))

    def test_set_task_running_disables_config_fields(self) -> None:
        widget = self.page.field_widgets["tl_subdir"]
        self.assertTrue(widget.isEnabled())
        self.page.set_task_running(True)
        self.assertFalse(widget.isEnabled())
        self.assertFalse(self.page._prepare_renpy_sdk_browse_btn.isEnabled())
        self.assertTrue(self.page.settings_go_workspace_btn.isEnabled())
        self.page.set_task_running(False)
        self.assertTrue(widget.isEnabled())

    def test_navigate_goes_through_injected_callback(self) -> None:
        calls: list[str] = []
        self.page.set_action_callbacks(
            SettingsPageActions(navigate=lambda key: calls.append(key))
        )
        self.page.settings_go_workspace_btn.click()
        self.assertEqual(calls, ["workspace"])

    def test_game_root_display_is_not_a_config_key(self) -> None:
        self.page.set_game_root_display("/tmp/work")
        self.assertEqual(self.page.settings_project_root_value.text(), "/tmp/work")
        self.assertNotIn("game_root", self.page.collect())


if __name__ == "__main__":
    unittest.main()
