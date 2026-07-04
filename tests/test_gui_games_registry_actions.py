"""Tests for GUI games registry actions."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import games_registry as registry
from gui_qt.games_registry_actions import (
    record_apply_batch_for_game_root,
    refresh_registry_projects,
    render_registry_games_md,
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
                side_effect=lambda data, project_id, workspace_root: data["projects"][0],
            ):
                result = record_apply_batch_for_game_root(
                    workspace,
                    game_root=work_dir,
                    manifest_path=manifest_path,
                )
            self.assertTrue(result.ok)
            data = registry.load_registry(workspace / registry.REGISTRY_FILENAME)
            self.assertEqual(data["projects"][0]["auto"]["last_batch_id"], "job_001")

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


if __name__ == "__main__":
    unittest.main()