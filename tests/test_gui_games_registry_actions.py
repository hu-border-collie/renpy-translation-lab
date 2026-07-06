"""Tests for GUI games registry actions."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import games_registry as registry
from gui_qt.games_registry_actions import (
    discover_registry_projects,
    import_registry_from_games_md,
    record_apply_batch_for_game_root,
    refresh_registry_projects,
    render_registry_games_md,
    save_registry_project_fields,
)
from gui_qt.games_registry_view import find_project_id_for_game_root


class GuiGamesRegistryActionsTests(unittest.TestCase):
    def _write_registry(self, workspace: Path, projects: list[dict]) -> None:
        payload = {"projects": projects}
        (workspace / registry.REGISTRY_FILENAME).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def test_find_project_id_for_game_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "Game_Example"
            (project_root / "work").mkdir(parents=True)
            (project_root / "original" / "game").mkdir(parents=True)
            self._write_registry(
                workspace,
                [{"id": "demo", "name": "Example", "path": "Game_Example"}],
            )
            project_id = find_project_id_for_game_root(
                workspace_root=workspace,
                game_root=project_root / "work",
            )
            self.assertEqual(project_id, "demo")

    def test_refresh_registry_projects_updates_translation_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "Game_Example"
            (project_root / "work").mkdir(parents=True)
            self._write_registry(
                workspace,
                [
                    {
                        "id": "demo",
                        "name": "Example",
                        "path": "Game_Example",
                        "translation_status_source": "doctor",
                    }
                ],
            )
            result = refresh_registry_projects(workspace, project_id="demo")
            self.assertTrue(result.ok)
            self.assertIn("快速刷新", result.message)

            data = registry.load_registry(workspace / registry.REGISTRY_FILENAME)
            project = data["projects"][0]
            self.assertIn("auto", project)
            self.assertTrue(project.get("translation_status"))

    def test_record_apply_batch_for_game_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "Game_Example"
            work_dir = project_root / "work"
            work_dir.mkdir(parents=True)
            (project_root / "original" / "game").mkdir(parents=True)
            self._write_registry(
                workspace,
                [{"id": "demo", "name": "Example", "path": "Game_Example"}],
            )

            manifest_dir = work_dir / "logs" / "batch_jobs" / "job_001"
            manifest_dir.mkdir(parents=True)
            manifest = {
                "status": "applied",
                "apply_summary": {"files_changed": 2, "lines_changed": 40},
            }
            manifest_path = manifest_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch(
                "gui_qt.games_registry_actions.refresh_project",
                side_effect=lambda data, project_id, workspace_root, **kwargs: data["projects"][0],
            ):
                result = record_apply_batch_for_game_root(
                    workspace,
                    game_root=work_dir,
                    manifest_path=manifest_path,
                )
            self.assertTrue(result.ok)
            data = registry.load_registry(workspace / registry.REGISTRY_FILENAME)
            self.assertEqual(data["projects"][0]["auto"]["last_batch_id"], "job_001")

    def test_record_apply_batch_reports_manifest_io_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "Game_Example"
            work_dir = project_root / "work"
            work_dir.mkdir(parents=True)
            (project_root / "original" / "game").mkdir(parents=True)
            self._write_registry(
                workspace,
                [{"id": "demo", "name": "Example", "path": "Game_Example"}],
            )

            with mock.patch(
                "gui_qt.games_registry_actions.record_batch",
                side_effect=OSError("permission denied"),
            ):
                result = record_apply_batch_for_game_root(
                    workspace,
                    game_root=work_dir,
                    manifest_path=work_dir / "manifest.json",
                )
            self.assertFalse(result.ok)
            self.assertIn("registry 批次记录失败", result.message)

    def test_render_registry_games_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_registry(
                workspace,
                [{"id": "demo", "name": "Example", "path": "Game_Example"}],
            )
            result = render_registry_games_md(workspace)
            self.assertTrue(result.ok)
            self.assertTrue((workspace / registry.GAMES_MD_FILENAME).is_file())

    def test_import_registry_from_games_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / registry.GAMES_MD_FILENAME).write_text(
                "\n".join(
                    [
                        "| Alpha | `Game_Alpha` | 1.0 | ready | 待确认 | 待翻译 | 下一步。 |",
                    ]
                ),
                encoding="utf-8",
            )
            result = import_registry_from_games_md(workspace)
            self.assertTrue(result.ok)
            data = registry.load_registry(workspace / registry.REGISTRY_FILENAME)
            self.assertEqual(data["projects"][0]["path"], "Game_Alpha")

    def test_discover_registry_projects_adds_new_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "Game_New").mkdir()
            with mock.patch(
                "gui_qt.games_registry_actions.refresh_project",
                side_effect=lambda data, project_id, workspace_root, **kwargs: data["projects"][-1],
            ):
                result = discover_registry_projects(workspace)
            self.assertTrue(result.ok)
            self.assertIn("Game_New", result.message)
            data = registry.load_registry(workspace / registry.REGISTRY_FILENAME)
            self.assertEqual(data["projects"][0]["path"], "Game_New")

    def test_save_registry_project_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_registry(
                workspace,
                [
                    {
                        "id": "demo",
                        "name": "Example",
                        "path": "Game_Example",
                        "translation_status_source": "scan",
                    }
                ],
            )
            result = save_registry_project_fields(
                workspace,
                project_id="demo",
                play_status="进行中",
                translation_status="待润色",
                notes="校对术语。",
            )
            self.assertTrue(result.ok)
            project = registry.load_registry(workspace / registry.REGISTRY_FILENAME)["projects"][0]
            self.assertEqual(project["play_status"], "进行中")
            self.assertEqual(project["translation_status_source"], "manual")


if __name__ == "__main__":
    unittest.main()