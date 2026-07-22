"""Tests for the games registry dialog."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import games_registry as registry

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.games_registry_actions import RegistryActionResult
    from gui_qt.games_registry_dialog import GamesRegistryDialog
    from gui_qt.games_registry_panel import GamesRegistryPanel
    from gui_qt.games_registry_worker import RegistryRefreshWorker
except ImportError as exc:
    GamesRegistryDialog = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(GamesRegistryDialog is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class GuiGamesRegistryDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_panel_switch_invokes_callback(self):
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

            switched = []
            panel = GamesRegistryPanel(
                None,
                workspace_root=workspace,
                on_switch_project=lambda target: switched.append(target) or True,
            )
            panel._table.selectRow(0)
            panel._switch_to_selected()
            self.assertEqual(len(switched), 1)
            self.assertTrue(switched[0].endswith("Game_Example"))

    def test_panel_empty_workspace_shows_empty_state_and_blocks_actions(self):
        panel = GamesRegistryPanel(None, workspace_root=None, auto_discover_on_show=False)
        self.assertIsNone(panel.workspace_root())
        # isHidden() reflects setVisible without requiring a shown parent window.
        self.assertFalse(panel._workspace_empty_state.isHidden())
        self.assertTrue(panel._workspace_body.isHidden())
        with mock.patch(
            "gui_qt.games_registry_panel.message_box_information"
        ) as info:
            self.assertIsNone(panel._require_workspace())
            info.assert_called()
        with mock.patch(
            "gui_qt.games_registry_panel.message_box_information"
        ) as info:
            with mock.patch(
                "gui_qt.games_registry_panel.discover_registry_projects"
            ) as discover:
                panel._discover_new_projects()
                discover.assert_not_called()
                info.assert_called()

    def test_panel_host_task_disables_choose_workspace_and_empty_action(self):
        panel = GamesRegistryPanel(None, workspace_root=None, auto_discover_on_show=False)
        panel.set_host_task_running(True)
        self.assertFalse(panel._choose_workspace_btn.isEnabled())
        if panel._workspace_empty_state._action_btn is not None:
            self.assertFalse(panel._workspace_empty_state._action_btn.isEnabled())
        panel.set_host_task_running(False)
        self.assertTrue(panel._choose_workspace_btn.isEnabled())

    def test_panel_choose_workspace_host_success_and_revert_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            # Setup dialog already wrote registry; host only persists config.
            (workspace / registry.REGISTRY_FILENAME).write_text(
                json.dumps({"projects": []}),
                encoding="utf-8",
            )

            from gui_qt.workspace_setup_dialog import WorkspaceSetupDialogResult

            fake_result = WorkspaceSetupDialogResult(
                workspace=workspace,
                message="工作区已接入：测试",
                project_count=0,
                sdk_message="已跳过 SDK 配置。",
            )

            # Mirror MainWindow._on_workspace_changed success path.
            def on_ok(path: Path, panel_ref: list) -> None:
                panel_ref[0].set_workspace_root(path)

            panel_box: list = []
            panel = GamesRegistryPanel(
                None,
                workspace_root=None,
                on_workspace_changed=lambda p: on_ok(p, panel_box),
                auto_discover_on_show=False,
            )
            panel_box.append(panel)

            class _AcceptDialog:
                DialogCode = type("DC", (), {"Accepted": 1, "Rejected": 0})()

                def __init__(self, *args, **kwargs):
                    pass

                def exec(self):
                    return self.DialogCode.Accepted

                def result_payload(self):
                    return fake_result

            with mock.patch(
                "gui_qt.workspace_setup_dialog.WorkspaceSetupDialog",
                _AcceptDialog,
            ):
                with mock.patch(
                    "gui_qt.games_registry_panel.message_box_information"
                ) as info:
                    panel._choose_workspace()
            self.assertIsNotNone(panel.workspace_root())
            self.assertEqual(
                Path(panel.workspace_root()).resolve(),
                workspace.resolve(),
            )
            self.assertTrue(panel._workspace_empty_state.isHidden())
            self.assertFalse(panel._workspace_body.isHidden())
            info.assert_called()
            # Summary must include workspace outcome and SDK status from dialog result.
            summary_text = "\n".join(str(c) for c in info.call_args[0])
            self.assertIn("工作区已接入：测试", summary_text)
            self.assertIn("已跳过 SDK 配置。", summary_text)

            # Failure path: host reverts to previous (None).
            def on_fail(path: Path, panel_ref: list) -> None:
                # Simulate persist failure: keep previous root.
                panel_ref[0].set_workspace_root(None)

            panel2_box: list = []
            panel2 = GamesRegistryPanel(
                None,
                workspace_root=None,
                on_workspace_changed=lambda p: on_fail(p, panel2_box),
                auto_discover_on_show=False,
            )
            panel2_box.append(panel2)
            with mock.patch(
                "gui_qt.workspace_setup_dialog.WorkspaceSetupDialog",
                _AcceptDialog,
            ):
                with mock.patch(
                    "gui_qt.games_registry_panel.message_box_information"
                ) as info2:
                    panel2._choose_workspace()
            self.assertIsNone(panel2.workspace_root())
            self.assertFalse(panel2._workspace_empty_state.isHidden())
            self.assertTrue(panel2._workspace_body.isHidden())
            info2.assert_not_called()

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
            # Path column is last (EUI flex); name stays column 0.
            path_col = dialog._table.columnCount() - 1
            self.assertEqual(dialog._table.item(0, path_col).text(), "Game_Example")
            with mock.patch.object(dialog, "accept") as accept_mock:
                dialog._switch_to_selected()
                accept_mock.assert_called_once()
            # Switch prefers effective work/ when present.
            selected = dialog.selected_project_root().replace("\\", "/")
            self.assertIn("Game_Example", selected)
            self.assertTrue(
                selected.rstrip("/").endswith("Game_Example/work")
                or selected.rstrip("/").endswith("Game_Example")
            )

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
            with mock.patch.object(RegistryRefreshWorker, "start") as start_mock:
                dialog._refresh_current_project()
                start_mock.assert_called_once()
            with mock.patch(
                "gui_qt.games_registry_panel.prompt_render_games_md_after_refresh",
                return_value=RegistryActionResult(True, "已跳过 GAMES.md 同步。"),
            ):
                dialog._on_refresh_completed(
                    RegistryActionResult(True, "已快速刷新项目 Example。")
                )
            self.assertIn("已快速刷新项目 Example", dialog._status_label.text())

    def test_refresh_keeps_table_enabled_for_scrolling(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry_path = workspace / registry.REGISTRY_FILENAME
            registry_path.write_text(json.dumps({"projects": []}), encoding="utf-8")

            dialog = GamesRegistryDialog(None, workspace_root=workspace)
            dialog._set_refresh_busy(True)

            self.assertTrue(dialog._table.isEnabled())
            self.assertFalse(dialog._refresh_all_btn.isEnabled())
            self.assertTrue(dialog._stop_refresh_btn.isEnabled())

    def test_switch_is_ignored_while_refresh_running(self):
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
            dialog._panel._table.selectRow(0)
            dialog._panel._refresh_worker = mock.MagicMock()
            dialog._panel._refresh_worker.isRunning.return_value = True

            with mock.patch.object(dialog, "accept") as accept_mock:
                dialog._panel._switch_to_selected()
                dialog._panel._on_row_activated(0, 0)
                accept_mock.assert_not_called()

    def test_stop_refresh_requests_worker_cancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry_path = workspace / registry.REGISTRY_FILENAME
            registry_path.write_text(json.dumps({"projects": []}), encoding="utf-8")

            dialog = GamesRegistryDialog(None, workspace_root=workspace)
            worker = mock.MagicMock()
            worker.isRunning.return_value = True
            dialog._panel._refresh_worker = worker
            dialog._panel._on_stop_refresh()
            worker.request_stop.assert_called_once()

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
            with mock.patch.object(RegistryRefreshWorker, "start") as start_mock:
                dialog._refresh_current_project()
                start_mock.assert_called_once()
            worker = dialog._refresh_worker
            self.assertIsNotNone(worker)
            self.assertEqual(worker._mode, "deep")

    def test_dialog_exposes_setup_and_edit_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry_path = workspace / registry.REGISTRY_FILENAME
            registry_path.write_text(json.dumps({"projects": []}), encoding="utf-8")

            dialog = GamesRegistryDialog(None, workspace_root=workspace)
            self.assertEqual(dialog._table.columnCount(), 6)
            self.assertTrue(hasattr(dialog, "_import_md_btn"))
            self.assertTrue(hasattr(dialog, "_discover_btn"))
            self.assertTrue(hasattr(dialog, "_sync_md_btn"))
            self.assertTrue(hasattr(dialog, "_save_fields_btn"))

    def test_save_selected_project_fields_updates_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            payload = {
                "projects": [
                    {
                        "id": "demo",
                        "name": "Example",
                        "path": "Game_Example",
                        "translation_status_source": "scan",
                    }
                ]
            }
            registry_path = workspace / registry.REGISTRY_FILENAME
            registry_path.write_text(json.dumps(payload), encoding="utf-8")

            dialog = GamesRegistryDialog(None, workspace_root=workspace)
            dialog._table.selectRow(0)
            dialog._play_status_combo.setCurrentText("进行中")
            dialog._translation_status_combo.setCurrentText("待润色")
            dialog._notes_edit.setPlainText("下一步：校对。")
            dialog._name_edit.setText("Example Renamed")
            dialog._save_selected_project_fields()

            data = registry.load_registry(registry_path)
            project = data["projects"][0]
            self.assertEqual(project["name"], "Example Renamed")
            self.assertEqual(project["play_status"], "进行中")
            self.assertEqual(project["translation_status"], "待润色")
            self.assertEqual(project["notes"], "下一步：校对。")
            self.assertEqual(project["translation_status_source"], "manual")

    def test_search_filter_limits_visible_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            payload = {
                "projects": [
                    {"id": "a", "name": "Alpha", "path": "Game_Alpha"},
                    {"id": "b", "name": "Beta", "path": "Game_Beta"},
                ]
            }
            registry_path = workspace / registry.REGISTRY_FILENAME
            registry_path.write_text(json.dumps(payload), encoding="utf-8")

            dialog = GamesRegistryDialog(None, workspace_root=workspace)
            dialog._search_edit.setText("Beta")
            dialog._apply_filters()
            self.assertEqual(dialog._table.rowCount(), 1)
            self.assertEqual(dialog._table.item(0, 0).text(), "Beta")

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