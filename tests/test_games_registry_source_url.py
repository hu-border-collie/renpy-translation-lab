"""Source URL and Markdown compatibility coverage for the games registry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import games_registry as registry


OLD_MD = """| 项目 | 路径 | 当前版本 | 目录状态 | 游玩状态 | 翻译状态 | 备注 / 下一步 |
|---|---|---|---|---|---|---|
| Alpha | `Game_Alpha` | 1.0 | 就绪 | 待确认 | 待翻译 | 旧表。 |
"""

SOURCE_MD = """| 项目 | 路径 | 来源 | 当前版本 | 目录状态 | 游玩状态 | 翻译状态 | 备注 / 下一步 |
|---|---|---|---|---|---|---|---|
| Alpha | `Game_Alpha` | [itch.io](<https://studio.itch.io/alpha>) | 1.1 | 就绪 | 进行中 | 翻译中 | 新表。 |
"""


class GamesRegistrySourceUrlTests(unittest.TestCase):
    def test_normalize_source_url_and_site_labels(self) -> None:
        self.assertEqual(registry.normalize_source_url(""), "")
        self.assertEqual(
            registry.normalize_source_url("  https://example.com/game  "),
            "https://example.com/game",
        )
        self.assertEqual(
            registry.source_site_name("https://studio.itch.io/game"),
            "itch.io",
        )
        self.assertEqual(
            registry.source_site_name("https://store.steampowered.com/app/1"),
            "Steam",
        )
        self.assertEqual(
            registry.source_site_name("https://www.example.com/game"),
            "example.com",
        )
        for invalid in (
            "ftp://example.com/game",
            "https://",
            "example.com",
            "https://exa mple.com",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "http://"):
                    registry.normalize_source_url(invalid)

    def test_parse_old_and_source_aware_tables(self) -> None:
        legacy = registry.parse_games_md_table(OLD_MD)
        self.assertEqual(len(legacy), 1)
        self.assertNotIn("source_url", legacy[0])

        source_aware = registry.parse_games_md_table(SOURCE_MD)
        self.assertEqual(source_aware[0]["source_url"], "https://studio.itch.io/alpha")

        invalid = SOURCE_MD.replace(
            "[itch.io](<https://studio.itch.io/alpha>)",
            "ftp://example.com/alpha",
        )
        with self.assertRaisesRegex(ValueError, "GAMES.md 第 3 行"):
            registry.parse_games_md_table(invalid)

    def test_render_source_link_and_roundtrip_escaped_cells(self) -> None:
        payload = {
            "updated_at": "2026-07-29T00:00:00+00:00",
            "projects": [
                {
                    "name": "Pipe | Game",
                    "path": "Game_Pipe",
                    "source_url": "https://studio.itch.io/pipe-game?ref=table",
                    "version": "1.0",
                    "layout_status": "ready",
                    "play_status": "待确认",
                    "translation_status": "待翻译",
                    "notes": "a | b",
                    "auto": {},
                }
            ],
        }
        rendered = registry.render_games_md(payload)
        self.assertIn("| 项目 | 路径 | 来源 | 当前版本 |", rendered)
        self.assertIn(
            "[itch.io](<https://studio.itch.io/pipe-game?ref=table>)",
            rendered,
        )
        parsed = registry.parse_games_md_table(rendered)
        self.assertEqual(parsed[0]["name"], "Pipe | Game")
        self.assertEqual(parsed[0]["notes"], "a | b")
        self.assertEqual(
            parsed[0]["source_url"],
            "https://studio.itch.io/pipe-game?ref=table",
        )

    def test_import_merge_preserves_legacy_source_and_updates_new_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry_path = workspace / registry.REGISTRY_FILENAME
            md_path = workspace / registry.GAMES_MD_FILENAME
            registry.save_registry(
                registry_path,
                {
                    "projects": [
                        {
                            "id": "game_alpha",
                            "name": "Alpha",
                            "path": "Game_Alpha",
                            "source_url": "https://old.example.com/alpha",
                            "auto": {},
                        }
                    ]
                },
            )

            md_path.write_text(OLD_MD, encoding="utf-8")
            legacy_merge = registry.import_from_games_md(
                md_path=md_path,
                registry_path=registry_path,
                workspace_root=workspace,
                merge=True,
            )
            self.assertEqual(
                legacy_merge["projects"][0]["source_url"],
                "https://old.example.com/alpha",
            )

            md_path.write_text(SOURCE_MD, encoding="utf-8")
            source_merge = registry.import_from_games_md(
                md_path=md_path,
                registry_path=registry_path,
                workspace_root=workspace,
                merge=True,
            )
            self.assertEqual(
                source_merge["projects"][0]["source_url"],
                "https://studio.itch.io/alpha",
            )

            md_path.write_text(
                SOURCE_MD.replace(
                    "[itch.io](<https://studio.itch.io/alpha>)",
                    "",
                ),
                encoding="utf-8",
            )
            cleared = registry.import_from_games_md(
                md_path=md_path,
                registry_path=registry_path,
                workspace_root=workspace,
                merge=True,
            )
            self.assertEqual(cleared["projects"][0]["source_url"], "")

    def test_new_projects_default_empty_and_refresh_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "Game_New").mkdir()
            payload = registry.empty_registry(workspace)
            added, _ = registry.merge_discovered_projects(
                payload,
                workspace_root=workspace,
                refresh_new=False,
            )
            self.assertEqual(added, 1)
            project = payload["projects"][0]
            self.assertEqual(project["source_url"], "")

            project["source_url"] = "https://example.com/new"
            with (
                mock.patch.object(registry, "scan_project_auto", return_value={}),
                mock.patch.object(registry, "detect_game_version", return_value=("", "")),
            ):
                registry.refresh_project(
                    payload,
                    project["id"],
                    workspace_root=workspace,
                )
            self.assertEqual(project["source_url"], "https://example.com/new")

    def test_manual_update_validates_before_mutating_and_can_clear(self) -> None:
        payload = {
            "projects": [
                {
                    "id": "demo",
                    "name": "Demo",
                    "source_url": "https://example.com/old",
                }
            ]
        }
        with self.assertRaises(ValueError):
            registry.update_project_manual_fields(
                payload,
                "demo",
                name="Changed",
                source_url="file:///tmp/demo",
            )
        self.assertEqual(payload["projects"][0]["name"], "Demo")
        self.assertEqual(
            payload["projects"][0]["source_url"],
            "https://example.com/old",
        )

        registry.update_project_manual_fields(
            payload,
            "demo",
            source_url="https://example.com/new",
        )
        self.assertEqual(
            payload["projects"][0]["source_url"],
            "https://example.com/new",
        )
        registry.update_project_manual_fields(payload, "demo", source_url="")
        self.assertEqual(payload["projects"][0]["source_url"], "")


if __name__ == "__main__":
    unittest.main()
