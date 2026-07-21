"""Tests for GUI games registry view helpers."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gui_qt.games_registry_view import (
    REGISTRY_SORT_NAME_DESC,
    RegistryRow,
    filter_and_sort_registry_rows,
    load_registry_rows,
    registry_row_from_project,
    row_matches_game_root,
)
import games_registry as registry


class GuiGamesRegistryViewTests(unittest.TestCase):
    def test_load_registry_rows_sorts_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            payload = {
                "projects": [
                    {"id": "b", "name": "Beta", "path": "Game_Beta"},
                    {"id": "a", "name": "Alpha", "path": "Game_Alpha"},
                ]
            }
            registry_path = workspace / registry.REGISTRY_FILENAME
            registry_path.write_text(json.dumps(payload), encoding="utf-8")

            rows, _ = load_registry_rows(workspace_root=workspace, registry_path=registry_path)
            self.assertEqual([row.name for row in rows], ["Alpha", "Beta"])

    def test_load_registry_rows_reports_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            rows, message = load_registry_rows(workspace_root=workspace)
            self.assertEqual(rows, [])
            self.assertIn("未找到", message)
            self.assertIn("扫描新项目", message)

    def test_registry_row_uses_layout_status_or_auto_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            row = registry_row_from_project(
                workspace,
                {
                    "id": "demo",
                    "name": "Example",
                    "path": "Game_Example",
                    "auto": {"doctor_layout": "ready"},
                },
            )
            self.assertEqual(row.layout_status, "就绪")

    def test_registry_row_resolves_nested_work_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "Game_Example"
            (project_root / "work").mkdir(parents=True)
            (project_root / "original" / "game").mkdir(parents=True)

            row = registry_row_from_project(
                workspace,
                {"id": "demo", "name": "Example", "path": "Game_Example"},
            )
            self.assertTrue(row.work_dir.endswith("/work") or row.work_dir.endswith("\\work"))

    def test_row_matches_game_root_compares_normalized_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "Game_Example" / "work"
            work_dir.mkdir(parents=True)
            row = RegistryRow(
                project_id="demo",
                name="Example",
                path="Game_Example",
                version="1.0",
                layout_status="ready",
                play_status="待确认",
                translation_status="待翻译",
                notes="",
                engine="renpy",
                in_renpy_pipeline=True,
                work_dir=str(work_dir),
            )
            self.assertTrue(row_matches_game_root(row, work_dir))

    def test_filter_and_sort_registry_rows(self):
        rows = [
            RegistryRow(
                project_id="b",
                name="Beta",
                path="Game_Beta",
                version="1.0",
                layout_status="ready",
                play_status="待确认",
                translation_status="待翻译",
                notes="",
                engine="renpy",
                in_renpy_pipeline=True,
                work_dir="",
            ),
            RegistryRow(
                project_id="a",
                name="Alpha",
                path="Game_Alpha",
                version="1.0",
                layout_status="ready",
                play_status="待确认",
                translation_status="已完成",
                notes="术语",
                engine="unity",
                in_renpy_pipeline=False,
                work_dir="",
            ),
        ]
        filtered = filter_and_sort_registry_rows(
            rows,
            search_text="术语",
            engine_filter="unity",
            sort_key=REGISTRY_SORT_NAME_DESC,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "Alpha")


if __name__ == "__main__":
    unittest.main()