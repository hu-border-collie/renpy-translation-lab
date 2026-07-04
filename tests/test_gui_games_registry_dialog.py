"""Tests for the games registry dialog."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PySide6.QtWidgets import QApplication

import games_registry as registry
from gui_qt.games_registry_dialog import GamesRegistryDialog
from gui_qt.games_registry_actions import RegistryActionResult


class GuiGamesRegistryDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_dialog_loads_rows_and_accepts_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            payload = {
                "projects": [
                    {
                        "id": "demo",
                        "name": "Example",
                        "path": "Game_Example",
                        "version": "1.0",
                        "play_status": "待确认",
                        "translation_status": "待翻译",
                    }
                ]
            }
            project_root = workspace / "Game_Example"
            (project_root / "work").mkdir(parents=True)
            (project_root / "original" / "game").mkdir(parents=True)
            registry_path = workspace / registry.REGISTRY_FILENAME
            registry_path.write_text(json.dumps(payload), encoding="utf-8")

            dialog = GamesRegistryDialog(None, workspace_root=workspace)
            self.assertEqual(dialog._table.rowCount(), 1)
            self.assertIn("Example", dialog._table.item(0, 0).text())

            dialog._table.selectRow(0)
            with mock.patch.object(dialog, "accept") as accept_mock:
                dialog._switch_to_selected()
                accept_mock.assert_called_once()
            self.assertTrue(dialog.selected_project_root().endswith("Game_Example"))

    def test_refresh_current_project_updates_status_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            payload = {
                "projects": [
                    {
                        "id": "demo",
                        "name": "Example",
                        "path": "Game_Example",
                        "translation_status_source": "doctor",
                    }
                ]
            }
            project_root = workspace / "Game_Example"
            (project_root / "work").mkdir(parents=True)
            registry_path = workspace / registry.REGISTRY_FILENAME
            registry_path.write_text(json.dumps(payload), encoding="utf-8")

            dialog = GamesRegistryDialog(None, workspace_root=workspace)
            dialog._table.selectRow(0)
            with mock.patch(
                "gui_qt.games_registry_dialog.refresh_registry_projects",
                return_value=RegistryActionResult(True, "已快速刷新项目 Example。"),
            ) as refresh_mock:
                dialog._refresh_current_project()
                refresh_mock.assert_called_once_with(
                    workspace,
                    project_id="demo",
                    refresh_everything=False,
                    mode="lite",
                    on_progress=dialog._on_refresh_progress,
                )
            self.assertIn("已快速刷新项目 Example", dialog._status_label.text())

    def test_refresh_current_uses_deep_mode_when_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            payload = {
                "projects": [
                    {
                        "id": "demo",
                        "name": "Example",
                        "path": "Game_Example",
                    }
                ]
            }
            project_root = workspace / "Game_Example"
            (project_root / "work").mkdir(parents=True)
            registry_path = workspace / registry.REGISTRY_FILENAME
            registry_path.write_text(json.dumps(payload), encoding="utf-8")

            dialog = GamesRegistryDialog(None, workspace_root=workspace)
            dialog._table.selectRow(0)
            dialog._refresh_mode_combo.setCurrentIndex(1)
            with mock.patch(
                "gui_qt.games_registry_dialog.refresh_registry_projects",
                return_value=RegistryActionResult(True, "已深度刷新项目 Example。"),
            ) as refresh_mock:
                dialog._refresh_current_project()
                refresh_mock.assert_called_once_with(
                    workspace,
                    project_id="demo",
                    refresh_everything=False,
                    mode="deep",
                    on_progress=dialog._on_refresh_progress,
                )

    def test_manual_translation_status_shows_in_tooltip(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            payload = {
                "projects": [
                    {
                        "id": "demo",
                        "name": "Example",
                        "path": "Game_Example",
                        "translation_status": "待反编译",
                        "translation_status_source": "manual",
                    }
                ]
            }
            registry_path = workspace / registry.REGISTRY_FILENAME
            registry_path.write_text(json.dumps(payload), encoding="utf-8")

            dialog = GamesRegistryDialog(None, workspace_root=workspace)
            tooltip = dialog._table.item(0, 0).toolTip()
            self.assertIn("人工维护", tooltip)


if __name__ == "__main__":
    unittest.main()