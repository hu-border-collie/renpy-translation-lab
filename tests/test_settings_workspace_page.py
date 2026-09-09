"""Independent construction tests for the #202 Phase D Workspace Settings page."""
from __future__ import annotations

import unittest

from gui_qt.settings.page_contract import SettingsIssue, SettingsPage

try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
    WorkspaceSettingsPage = None  # type: ignore[misc,assignment]
else:
    IMPORT_ERROR = None
    from gui_qt.settings.workspace_page import WorkspaceSettingsPage

from tests import gui_test_support


@gui_test_support.skip_unless_gui(WorkspaceSettingsPage is None, IMPORT_ERROR)
class WorkspaceSettingsPageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.page = WorkspaceSettingsPage()

    def tearDown(self) -> None:
        self.page.widget.deleteLater()
        self._app.processEvents()

    def test_page_satisfies_settings_protocol(self) -> None:
        self.assertIsInstance(self.page, SettingsPage)
        self.assertEqual(self.page.page_key, "workspace")
        self.assertEqual(self.page.nav_label, "项目列表")
        self.assertEqual(self.page.config_keys, frozenset())
        self.assertEqual(
            self.page.immediate_action_ids,
            frozenset({"switch_project", "refresh_registry", "import_projects"}),
        )

    def test_collect_is_empty_and_load_reset_are_noops(self) -> None:
        self.assertEqual(self.page.collect(), {})
        self.page.load({"theme": "dark"})
        self.assertEqual(self.page.collect(), {})
        self.page.reset()
        self.assertEqual(self.page.validate(), [])

    def test_embeds_games_registry_panel(self) -> None:
        panel = self.page._games_registry_panel
        self.assertEqual(panel.objectName(), "games_registry_panel")

    def test_attach_widget_aliases_copies_panel(self) -> None:
        host = type("Host", (), {})()
        self.page.attach_widget_aliases(host)
        self.assertIs(host._games_registry_panel, self.page._games_registry_panel)

    def test_focus_issue_targets_panel(self) -> None:
        self.assertTrue(
            self.page.focus_issue(SettingsIssue("workspace", "switch_project", ""))
        )
        self.assertTrue(
            self.page.focus_issue(SettingsIssue("workspace", "refresh_registry", ""))
        )

    def test_set_task_running_gates_panel(self) -> None:
        panel = self.page._games_registry_panel
        self.assertFalse(panel._host_task_running)
        self.page.set_task_running(True)
        self.assertTrue(panel._host_task_running)
        self.page.set_task_running(False)
        self.assertFalse(panel._host_task_running)


if __name__ == "__main__":
    unittest.main()
