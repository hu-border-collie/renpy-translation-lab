"""Tests for workspace games registry."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import games_registry as registry


SAMPLE_MD = """# 游戏状态总表

更新时间：2026-07-03（示例）

## 项目状态

| 项目 | 路径 | 当前版本 | 目录状态 | 游玩状态 | 翻译状态 | 备注 / 下一步 |
|---|---|---|---|---|---|---|
| Glory Hounds | `Game_GloryHounds` | 6.7 | 已整理 | 待确认 | 已完成（6.7 增量） | 术语已提取。 |
| Stranded | `Game_Stranded` | 0.4.0 | 已建 work | 待确认 | 未开始 | work 为空。 |
| Stranded | `Game_Stranded` | 0.4.0 | duplicate | 待确认 | 未开始 | 应被去重。 |
| Lookouts | `Game_Lookouts` | 1.3 | Unity 包 | 待确认 | 待确认 | 非 Ren'Py。 |
"""


class GamesRegistryTests(unittest.TestCase):
    def test_parse_games_md_table_dedupes_by_path(self):
        projects = registry.parse_games_md_table(SAMPLE_MD)
        paths = [project["path"] for project in projects]
        self.assertEqual(len(projects), 3)
        self.assertEqual(paths.count("Game_Stranded"), 1)

    def test_slugify_project_id_handles_nested_paths(self):
        self.assertEqual(
            registry.slugify_project_id("Game_Adastra_Universe/Adastra", "Adastra"),
            "game_adastra_universe_adastra",
        )

    def test_import_from_games_md_writes_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            md_path = workspace / "GAMES.md"
            md_path.write_text(SAMPLE_MD, encoding="utf-8")
            registry_path = workspace / registry.REGISTRY_FILENAME

            data = registry.import_from_games_md(
                md_path=md_path,
                registry_path=registry_path,
                workspace_root=workspace,
            )

            self.assertEqual(len(data["projects"]), 3)
            self.assertTrue(registry_path.is_file())
            loaded = registry.load_registry(registry_path)
            self.assertEqual(loaded["projects"][0]["id"], "game_gloryhounds")

    def test_normalize_translation_status_maps_unknown_to_default(self):
        self.assertEqual(registry.normalize_translation_status("  乱写状态  "), "待确认")
        self.assertEqual(registry.normalize_translation_status("待翻译"), "待翻译")
        self.assertEqual(registry.normalize_translation_status("已完成（6.7 增量）"), "已完成（6.7 增量）")

    def test_render_games_md_escapes_pipe_characters(self):
        payload = {
            "updated_at": "2026-07-04T12:00:00+00:00",
            "update_summary": "测试生成",
            "projects": [
                {
                    "name": "Pipe | Game",
                    "path": "Game_Pipe",
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
        self.assertIn("Pipe \\| Game", rendered)
        self.assertIn("a \\| b", rendered)

    def test_render_games_md_contains_projects_and_marker(self):
        payload = {
            "updated_at": "2026-07-04T12:00:00+00:00",
            "update_summary": "测试生成",
            "projects": [
                {
                    "name": "Alpha",
                    "path": "Game_Alpha",
                    "version": "1.0",
                    "layout_status": "ready",
                    "play_status": "待确认",
                    "translation_status": "待翻译",
                    "notes": "下一步：初译。",
                    "auto": {},
                }
            ],
        }
        rendered = registry.render_games_md(payload)
        self.assertIn("generated from games_registry.json", rendered)
        self.assertIn("| Alpha | `Game_Alpha` | 1.0 | ready | 待确认 | 待翻译 | 下一步：初译。 |", rendered)

    def test_suggest_translation_status_for_empty_work(self):
        project = {"translation_status_source": "doctor"}
        auto = {
            "in_renpy_pipeline": True,
            "work_exists": True,
            "work_empty": True,
            "original_rpa_count": 2,
            "original_editable_rpy_count": 0,
            "tl_rpy_files": 0,
        }
        self.assertEqual(registry.suggest_translation_status(project, auto), "待反编译")

    def test_suggest_translation_status_respects_manual_override(self):
        project = {
            "translation_status": "已完成",
            "translation_status_source": "manual",
        }
        auto = {
            "in_renpy_pipeline": True,
            "work_empty": True,
            "tl_rpy_files": 0,
        }
        self.assertEqual(registry.suggest_translation_status(project, auto), "已完成")

    def test_detect_game_version_from_options_rpy(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "Game_Example"
            options = project_root / "work" / "game" / "options.rpy"
            options.parent.mkdir(parents=True)
            options.write_text('define config.version = "2.5.1"\n', encoding="utf-8")

            version, source = registry.detect_game_version(project_root)
            self.assertEqual(version, "2.5.1")
            self.assertEqual(source, "detected")

    def test_collect_tl_counts_tracks_translated_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp) / "tl"
            tl_dir.mkdir()
            (tl_dir / "dialogue.rpy").write_text(
                '\n'.join(
                    [
                        "translate schinese strings:",
                        '    old "Hello"',
                        '    new "你好"',
                        '    old "Bye"',
                        '    new "Bye"',
                    ]
                ),
                encoding="utf-8",
            )
            counts = registry.collect_tl_counts(tl_dir)
            self.assertEqual(counts["new_lines"], 2)
            self.assertEqual(counts["translated_new_lines"], 1)

    def test_record_batch_updates_auto_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = Path(tmp) / "job_001"
            manifest_dir.mkdir()
            manifest = {
                "status": "applied",
                "apply_summary": {"files_changed": 3, "lines_changed": 120},
            }
            manifest_path = manifest_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            payload = {
                "projects": [
                    {
                        "id": "demo",
                        "translation_status_source": "doctor",
                        "translation_status": "待润色",
                        "auto": {},
                    }
                ]
            }
            project = registry.record_batch(
                payload,
                project_id="demo",
                manifest_path=manifest_path,
            )
            self.assertIsNotNone(project)
            self.assertEqual(project["auto"]["last_batch_id"], "job_001")
            self.assertIn("3 文件", project["auto"]["last_batch_summary"])

    def test_record_batch_preserves_zero_file_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = Path(tmp) / "job_zero"
            manifest_dir.mkdir()
            manifest = {
                "status": "applied",
                "apply_summary": {"files_changed": 0, "lines_changed": 0},
            }
            manifest_path = manifest_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            payload = {
                "projects": [
                    {
                        "id": "demo",
                        "translation_status_source": "doctor",
                        "auto": {},
                    }
                ]
            }
            project = registry.record_batch(
                payload,
                project_id="demo",
                manifest_path=manifest_path,
            )
            self.assertIsNotNone(project)
            self.assertIn("0 文件", project["auto"]["last_batch_summary"])
            self.assertIn("0 行", project["auto"]["last_batch_summary"])

    def test_lite_scan_skips_doctor_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "Game_Example"
            (project_root / "work" / "game" / "tl" / "schinese").mkdir(parents=True)
            (project_root / "original" / "game").mkdir(parents=True)
            project = {
                "id": "game_example",
                "path": "Game_Example",
                "engine": "renpy",
                "in_renpy_pipeline": True,
            }
            with mock.patch.object(
                registry,
                "_doctor_layout_snapshot",
                return_value=("ready", "existing_tl_only"),
            ) as doctor_mock:
                auto = registry.scan_project_auto(workspace, project, deep=False)
                doctor_mock.assert_not_called()
            self.assertEqual(auto["refresh_mode"], registry.REFRESH_MODE_LITE)
            self.assertEqual(auto["doctor_layout"], "attention")

    def test_deep_scan_uses_doctor_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "Game_Example"
            (project_root / "work").mkdir(parents=True)
            (project_root / "original" / "game").mkdir(parents=True)
            project = {
                "id": "game_example",
                "path": "Game_Example",
                "engine": "renpy",
                "in_renpy_pipeline": True,
            }
            with mock.patch.object(
                registry,
                "_doctor_layout_snapshot",
                return_value=("ready", "existing_tl_only"),
            ) as doctor_mock:
                auto = registry.scan_project_auto(workspace, project, deep=True)
                doctor_mock.assert_called_once()
            self.assertEqual(auto["refresh_mode"], registry.REFRESH_MODE_DEEP)
            self.assertEqual(auto["doctor_layout"], "ready")

    def test_refresh_all_stops_when_cancelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            payload = {
                "projects": [
                    {"id": "a", "name": "Alpha", "path": "Game_Alpha"},
                    {"id": "b", "name": "Beta", "path": "Game_Beta"},
                ]
            }
            calls: list[str] = []

            def fake_refresh(registry, project_id, **kwargs):
                calls.append(project_id)
                return registry["projects"][0]

            with mock.patch.object(registry, "refresh_project", side_effect=fake_refresh):
                count, cancelled = registry.refresh_all(
                    payload,
                    workspace_root=workspace,
                    should_cancel=lambda: len(calls) >= 1,
                )

            self.assertEqual(count, 1)
            self.assertTrue(cancelled)

    def test_refresh_project_updates_auto_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "Game_Example"
            tl_dir = project_root / "work" / "game" / "tl" / "schinese"
            tl_dir.mkdir(parents=True)
            (tl_dir / "common.rpy").write_text(
                '\n'.join(
                    [
                        "# line 1",
                        '    new "hello"',
                    ]
                ),
                encoding="utf-8",
            )
            (project_root / "original" / "game").mkdir(parents=True)

            payload = {
                "workspace_root": workspace.as_posix(),
                "projects": [
                    {
                        "id": "game_example",
                        "name": "Example",
                        "path": "Game_Example",
                        "engine": "renpy",
                        "in_renpy_pipeline": True,
                        "translation_status_source": "scan",
                        "translation_status": "",
                    }
                ],
            }
            registry.refresh_project(payload, "game_example", workspace_root=workspace)
            auto = payload["projects"][0]["auto"]
            self.assertEqual(auto["refresh_mode"], registry.REFRESH_MODE_LITE)
            self.assertEqual(auto["tl_rpy_files"], 1)
            self.assertEqual(payload["projects"][0]["translation_status_source"], "scan")
            self.assertIn(
                payload["projects"][0]["translation_status"],
                {"待翻译", "待润色", "未开始", "待提取"},
            )

    def test_iter_workspace_project_paths_finds_game_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "Game_Alpha").mkdir()
            (workspace / "Game_Beta").mkdir()
            (workspace / "renpy-translation-lab").mkdir()
            adastra = workspace / "Game_Adastra_Universe" / "Adastra"
            adastra.mkdir(parents=True)

            paths = registry.iter_workspace_project_paths(workspace)
            self.assertEqual(
                paths,
                ["Game_Alpha", "Game_Beta", "Game_Adastra_Universe/Adastra"],
            )

    def test_merge_discovered_projects_adds_new_game_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "Game_New").mkdir()
            payload = registry.empty_registry(workspace)
            with mock.patch.object(
                registry,
                "refresh_project",
                side_effect=lambda data, project_id, **kwargs: registry.find_project(data, project_id),
            ):
                added_count, added_paths = registry.merge_discovered_projects(
                    payload,
                    workspace_root=workspace,
                    refresh_new=True,
                )
            self.assertEqual(added_count, 1)
            self.assertEqual(added_paths, ["Game_New"])
            self.assertEqual(payload["projects"][0]["path"], "Game_New")
            self.assertEqual(payload["projects"][0]["name"], "New")

    def test_refresh_project_syncs_layout_status_from_auto(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "Game_Example"
            (project_root / "work").mkdir(parents=True)
            (project_root / "original" / "game").mkdir(parents=True)
            payload = {
                "projects": [
                    {
                        "id": "game_example",
                        "path": "Game_Example",
                        "layout_status": "旧状态",
                        "engine": "renpy",
                        "in_renpy_pipeline": True,
                    }
                ]
            }
            with mock.patch.object(
                registry,
                "scan_project_auto",
                return_value={
                    "doctor_layout": "ready",
                    "doctor_mode": "existing_tl_only",
                    "refresh_mode": registry.REFRESH_MODE_LITE,
                },
            ):
                registry.refresh_project(payload, "game_example", workspace_root=workspace)
            self.assertEqual(payload["projects"][0]["layout_status"], "ready")

    def test_update_project_manual_fields_marks_translation_manual(self):
        payload = {
            "projects": [
                {
                    "id": "demo",
                    "play_status": "待确认",
                    "translation_status": "待翻译",
                    "translation_status_source": "scan",
                    "notes": "",
                }
            ]
        }
        project = registry.update_project_manual_fields(
            payload,
            "demo",
            play_status="进行中",
            translation_status="待润色",
            notes="下一步：校对。",
        )
        self.assertEqual(project["play_status"], "进行中")
        self.assertEqual(project["translation_status"], "待润色")
        self.assertEqual(project["translation_status_source"], "manual")
        self.assertEqual(project["notes"], "下一步：校对。")

    def test_import_from_games_md_merge_updates_existing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            md_path = workspace / "GAMES.md"
            md_path.write_text(SAMPLE_MD, encoding="utf-8")
            registry_path = workspace / registry.REGISTRY_FILENAME
            registry.save_registry(
                registry_path,
                {
                    "projects": [
                        {
                            "id": "game_gloryhounds",
                            "name": "Old Name",
                            "path": "Game_GloryHounds",
                            "notes": "保留",
                            "auto": {"last_refresh_at": "2026-01-01T00:00:00+00:00"},
                        }
                    ]
                },
            )

            data = registry.import_from_games_md(
                md_path=md_path,
                registry_path=registry_path,
                workspace_root=workspace,
                merge=True,
            )
            by_path = {project["path"]: project for project in data["projects"]}
            self.assertEqual(by_path["Game_GloryHounds"]["name"], "Glory Hounds")
            self.assertEqual(by_path["Game_GloryHounds"]["auto"]["last_refresh_at"], "2026-01-01T00:00:00+00:00")

    def test_remove_project_and_manual_name_update(self):
        payload = {
            "projects": [
                {"id": "demo", "name": "Old", "path": "Game_Example"},
                {"id": "other", "name": "Other", "path": "Game_Other"},
            ]
        }
        removed = registry.remove_project(payload, "demo")
        self.assertEqual(removed["name"], "Old")
        self.assertEqual(len(payload["projects"]), 1)

        project = registry.update_project_manual_fields(
            payload,
            "other",
            name="Renamed",
        )
        self.assertEqual(project["name"], "Renamed")


if __name__ == "__main__":
    unittest.main()