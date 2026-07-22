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
        self.assertIn("| Alpha | `Game_Alpha` | 1.0 | 就绪 | 待确认 | 待翻译 | 下一步：初译。 |", rendered)

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

    def test_detect_game_version_from_cache_build_info(self):
        """Ren'Py often stores build_info under game/cache/, not game/ root."""
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "Game_Example"
            build_info = project_root / "original" / "game" / "cache" / "build_info.json"
            build_info.parent.mkdir(parents=True)
            build_info.write_text(
                '{"name": "Example", "version": "1.6", "time": 1.0, "info": {}}',
                encoding="utf-8",
            )

            version, source = registry.detect_game_version(project_root)
            self.assertEqual(version, "1.6")
            self.assertEqual(source, "detected")

    def test_format_layout_status_label_maps_machine_codes(self):
        self.assertEqual(registry.format_layout_status_label("ready"), "就绪")
        self.assertEqual(registry.format_layout_status_label("attention"), "需关注")
        self.assertEqual(registry.format_layout_status_label("switch_to_work"), "建议使用 work")
        self.assertEqual(registry.format_layout_status_label("failed"), "不可用")
        self.assertEqual(registry.format_layout_status_label("non_renpy"), "非 Ren'Py")
        self.assertEqual(registry.format_layout_status_label(""), "待确认")
        # Legacy free-form Chinese notes collapse to short labels.
        self.assertEqual(
            registry.format_layout_status_label(
                "已建 `original/work/build`；散放解压目录已移入 `original/`"
            ),
            "需关注",
        )
        self.assertEqual(
            registry.format_layout_status_label(
                "已建 `original/work/build`；非 Ren'Py，TyranoScript / NW.js 结构"
            ),
            "非 Ren'Py",
        )
        self.assertEqual(registry.format_layout_status_label("Unity 包"), "非 Ren'Py")

    def test_format_doctor_mode_label_maps_machine_codes(self):
        self.assertEqual(
            registry.format_doctor_mode_label("existing_tl_only"),
            "已有 TL 模板",
        )
        self.assertEqual(registry.format_doctor_mode_label(""), "—")

    def test_sync_layout_status_normalizes_non_renpy_freeform(self):
        project = {
            "layout_status": "已建 original/work/build；非 Ren'Py，TyranoScript",
            "in_renpy_pipeline": False,
            "engine": "tyrano",
            "auto": {"doctor_layout": "", "in_renpy_pipeline": False, "engine": "tyrano"},
        }
        registry.sync_layout_status_from_auto(project)
        self.assertEqual(project["layout_status"], "non_renpy")

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

    def test_doctor_layout_snapshot_prefers_subprocess(self):
        class _FakeQueue:
            def __init__(self, payload):
                self._payload = payload

            def get(self, timeout=None):
                return self._payload

        class _FakeProcess:
            def __init__(self, *args, **kwargs):
                self._alive = True

            def start(self):
                self._alive = False

            def join(self, timeout=None):
                return None

            def is_alive(self):
                return self._alive

            def terminate(self):
                self._alive = False

            def kill(self):
                self._alive = False

        class _FakeCtx:
            def Queue(self, maxsize=0):
                return _FakeQueue(("ready", "existing_tl_only"))

            def Process(self, **kwargs):
                return _FakeProcess()

        with mock.patch("multiprocessing.get_context", return_value=_FakeCtx()):
            layout, mode = registry._doctor_layout_snapshot("C:/game/work", True)
        self.assertEqual((layout, mode), ("ready", "existing_tl_only"))

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

    def test_default_workspace_root_is_unset_without_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "translator_config.json"
            with mock.patch.object(registry, "translator_config_path", return_value=config_path):
                self.assertIsNone(registry.default_workspace_root())
                self.assertIsNone(registry.load_configured_workspace_root(config_path))
                with self.assertRaises(ValueError):
                    registry.require_workspace_root()
                with self.assertRaises(ValueError):
                    registry.default_registry_path()

    def test_configured_workspace_root_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "translator_config.json"
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            # Preserve unrelated keys when updating workspace_root.
            config_path.write_text(
                json.dumps({"game_root": "C:/games/Example/work", "sync": {"model": "x"}}),
                encoding="utf-8",
            )
            saved = registry.save_configured_workspace_root(workspace, config_path)
            self.assertEqual(saved, workspace.resolve())
            loaded = registry.load_configured_workspace_root(config_path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.resolve(), workspace.resolve())
            data = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(data["game_root"], "C:/games/Example/work")
            self.assertEqual(data["sync"]["model"], "x")
            with mock.patch.object(registry, "translator_config_path", return_value=config_path):
                required = registry.require_workspace_root()
            self.assertEqual(required, workspace.resolve())
            required_explicit = registry.require_workspace_root(workspace)
            self.assertEqual(required_explicit, workspace.resolve())

    def test_save_configured_workspace_root_refuses_corrupt_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "translator_config.json"
            config_path.write_text("{not-json", encoding="utf-8")
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            with self.assertRaises(ValueError):
                registry.save_configured_workspace_root(workspace, config_path)
            self.assertEqual(config_path.read_text(encoding="utf-8"), "{not-json")

    def test_cli_resolve_paths_requires_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "missing_config.json"
            with mock.patch.object(registry, "translator_config_path", return_value=config_path):
                parser = registry.build_arg_parser()
                args = parser.parse_args(["show"])
                with self.assertRaises(SystemExit) as ctx:
                    registry.resolve_paths(args)
                self.assertEqual(ctx.exception.code, 2)

            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            parser = registry.build_arg_parser()
            args = parser.parse_args(["--workspace", str(workspace), "show"])
            resolved_ws, registry_path, md_path = registry.resolve_paths(args)
            # Windows tempfile may use 8.3 short paths; compare fully resolved forms.
            expected_ws = workspace.resolve()
            self.assertEqual(resolved_ws, expected_ws)
            self.assertEqual(registry_path.resolve(), (expected_ws / registry.REGISTRY_FILENAME))
            self.assertEqual(md_path.resolve(), (expected_ws / registry.GAMES_MD_FILENAME))

    def test_try_read_registry_missing_ok_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / registry.REGISTRY_FILENAME
            status, data, err = registry.try_read_registry(path)
            self.assertEqual(status, "missing")
            self.assertIsNone(data)
            self.assertEqual(err, "")

            registry.save_registry(path, registry.empty_registry(Path(tmp)))
            status, data, err = registry.try_read_registry(path)
            self.assertEqual(status, "ok")
            self.assertIsNotNone(data)
            self.assertEqual(err, "")

            path.write_text("{not-json", encoding="utf-8")
            status, data, err = registry.try_read_registry(path)
            self.assertEqual(status, "corrupt")
            self.assertIsNone(data)
            self.assertIn("无法解析", err)

            path.write_text("[]\n", encoding="utf-8")
            status, data, err = registry.try_read_registry(path)
            self.assertEqual(status, "corrupt")
            self.assertIsNone(data)

    def test_plan_and_apply_empty_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            config_path = Path(tmp) / "translator_config.json"
            plan = registry.plan_workspace_setup(workspace)
            self.assertTrue(plan.ok)
            self.assertEqual(plan.scene, registry.WorkspaceScene.EMPTY)
            self.assertTrue(plan.will_create_empty_registry)

            result = registry.apply_workspace_setup(
                plan,
                registry.WorkspaceSetupOptions(
                    persist_workspace_root=True,
                    config_path=config_path,
                ),
            )
            self.assertTrue(result.ok, result.message)
            self.assertTrue(result.created_registry)
            registry_path = workspace / registry.REGISTRY_FILENAME
            self.assertTrue(registry_path.is_file())
            loaded = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["projects"], [])
            self.assertEqual(
                registry.load_configured_workspace_root(config_path),
                workspace.resolve(),
            )

    def test_plan_attach_existing_registry_preserves_projects(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry_path = workspace / registry.REGISTRY_FILENAME
            original = {
                "schema_version": 1,
                "workspace_root": workspace.as_posix(),
                "projects": [
                    {
                        "id": "game_keep",
                        "name": "Keep",
                        "path": "Game_Keep",
                        "notes": "人工备注",
                        "play_status": "进行中",
                    }
                ],
            }
            registry_path.write_text(
                json.dumps(original, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            plan = registry.plan_workspace_setup(workspace)
            self.assertTrue(plan.ok)
            self.assertEqual(plan.scene, registry.WorkspaceScene.REGISTRY_OK)
            self.assertTrue(plan.will_attach_registry)
            self.assertFalse(plan.suggest_import_md)

            result = registry.apply_workspace_setup(
                plan,
                registry.WorkspaceSetupOptions(persist_workspace_root=False),
            )
            self.assertTrue(result.ok, result.message)
            self.assertFalse(result.created_registry)
            loaded = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["projects"][0]["notes"], "人工备注")
            self.assertEqual(loaded["projects"][0]["play_status"], "进行中")
            self.assertEqual(result.project_count, 1)

    def test_apply_import_md_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "GAMES.md").write_text(SAMPLE_MD, encoding="utf-8")
            plan = registry.plan_workspace_setup(workspace)
            self.assertEqual(plan.scene, registry.WorkspaceScene.GAMES_MD_ONLY)
            self.assertTrue(plan.suggest_import_md)

            result = registry.apply_workspace_setup(
                plan,
                registry.options_from_plan(plan, persist_workspace_root=False),
            )
            self.assertTrue(result.ok, result.message)
            self.assertTrue(result.imported_md)
            self.assertEqual(result.project_count, 3)

    def test_apply_discover_game_dirs_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "Game_Alpha").mkdir()
            (workspace / "Game_Beta").mkdir()
            plan = registry.plan_workspace_setup(workspace)
            self.assertEqual(plan.scene, registry.WorkspaceScene.GAME_DIRS_ONLY)
            self.assertTrue(plan.suggest_discover)
            self.assertEqual(set(plan.undiscovered_paths), {"Game_Alpha", "Game_Beta"})

            with mock.patch.object(
                registry,
                "refresh_project",
                side_effect=lambda data, project_id, **kwargs: registry.find_project(
                    data, project_id
                ),
            ):
                result = registry.apply_workspace_setup(
                    plan,
                    registry.options_from_plan(plan, persist_workspace_root=False),
                )
            self.assertTrue(result.ok, result.message)
            self.assertEqual(result.discovered_count, 2)
            self.assertEqual(result.project_count, 2)

            # Idempotent re-run
            plan2 = registry.plan_workspace_setup(workspace)
            with mock.patch.object(
                registry,
                "refresh_project",
                side_effect=lambda data, project_id, **kwargs: registry.find_project(
                    data, project_id
                ),
            ):
                result2 = registry.apply_workspace_setup(
                    plan2,
                    registry.WorkspaceSetupOptions(
                        discover=True,
                        persist_workspace_root=False,
                    ),
                )
            self.assertTrue(result2.ok, result2.message)
            self.assertEqual(result2.discovered_count, 0)
            self.assertEqual(result2.project_count, 2)

    def test_apply_mixed_does_not_duplicate_or_clobber_manual(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "Game_Keep").mkdir()
            (workspace / "Game_New").mkdir()
            (workspace / "GAMES.md").write_text(SAMPLE_MD, encoding="utf-8")
            registry_path = workspace / registry.REGISTRY_FILENAME
            registry.save_registry(
                registry_path,
                {
                    "projects": [
                        {
                            "id": "game_keep",
                            "name": "Keep",
                            "path": "Game_Keep",
                            "notes": "勿覆盖",
                            "play_status": "已玩完",
                            "auto": {"marker": 1},
                        }
                    ]
                },
            )
            plan = registry.plan_workspace_setup(workspace)
            self.assertEqual(plan.scene, registry.WorkspaceScene.MIXED)
            self.assertIn("Game_New", plan.undiscovered_paths)

            with mock.patch.object(
                registry,
                "refresh_project",
                side_effect=lambda data, project_id, **kwargs: registry.find_project(
                    data, project_id
                ),
            ):
                result = registry.apply_workspace_setup(
                    plan,
                    registry.WorkspaceSetupOptions(
                        import_md=True,
                        discover=True,
                        persist_workspace_root=False,
                    ),
                )
            self.assertTrue(result.ok, result.message)
            loaded = json.loads(registry_path.read_text(encoding="utf-8"))
            by_path = {p["path"]: p for p in loaded["projects"]}
            self.assertEqual(by_path["Game_Keep"]["notes"], "勿覆盖")
            self.assertEqual(by_path["Game_Keep"]["play_status"], "已玩完")
            self.assertEqual(by_path["Game_Keep"]["auto"]["marker"], 1)
            self.assertIn("Game_New", by_path)
            # SAMPLE_MD paths also merged
            self.assertIn("Game_GloryHounds", by_path)

    def test_corrupt_registry_refuses_apply_and_preserves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry_path = workspace / registry.REGISTRY_FILENAME
            corrupt = "{not-valid-json"
            registry_path.write_text(corrupt, encoding="utf-8")
            config_path = Path(tmp) / "translator_config.json"
            plan = registry.plan_workspace_setup(workspace)
            self.assertFalse(plan.ok)
            self.assertEqual(plan.scene, registry.WorkspaceScene.REGISTRY_CORRUPT)

            result = registry.apply_workspace_setup(
                plan,
                registry.WorkspaceSetupOptions(
                    persist_workspace_root=True,
                    config_path=config_path,
                ),
            )
            self.assertFalse(result.ok)
            self.assertEqual(registry_path.read_text(encoding="utf-8"), corrupt)
            self.assertFalse(config_path.exists())

    def test_not_directory_plan_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "not_a_dir.txt"
            file_path.write_text("x", encoding="utf-8")
            plan = registry.plan_workspace_setup(file_path)
            self.assertFalse(plan.ok)
            self.assertEqual(plan.scene, registry.WorkspaceScene.NOT_DIRECTORY)
            result = registry.apply_workspace_setup(plan)
            self.assertFalse(result.ok)

    def test_missing_path_requires_create_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "new_ws"
            plan = registry.plan_workspace_setup(workspace)
            self.assertTrue(plan.ok)
            self.assertEqual(plan.scene, registry.WorkspaceScene.MISSING_PATH)

            refused = registry.apply_workspace_setup(
                plan,
                registry.WorkspaceSetupOptions(
                    create_directory=False,
                    persist_workspace_root=False,
                ),
            )
            self.assertFalse(refused.ok)
            self.assertFalse(workspace.exists())

            ok = registry.apply_workspace_setup(
                plan,
                registry.WorkspaceSetupOptions(
                    create_directory=True,
                    persist_workspace_root=False,
                ),
            )
            self.assertTrue(ok.ok, ok.message)
            self.assertTrue(workspace.is_dir())
            self.assertTrue((workspace / registry.REGISTRY_FILENAME).is_file())

    def test_discover_stays_inside_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            outside = Path(tmp) / "outside"
            workspace.mkdir()
            outside.mkdir()
            (workspace / "Game_Inside").mkdir()
            (outside / "Game_Outside").mkdir()
            plan = registry.plan_workspace_setup(workspace)
            self.assertEqual(plan.game_dir_paths, ("Game_Inside",))
            self.assertNotIn("Game_Outside", plan.game_dir_paths)
            self.assertNotIn("outside/Game_Outside", plan.game_dir_paths)

    def test_dry_run_cli_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            config_path = Path(tmp) / "translator_config.json"
            with mock.patch.object(registry, "translator_config_path", return_value=config_path):
                code = registry.main(
                    ["--workspace", str(workspace), "setup", "--dry-run"]
                )
            self.assertEqual(code, 0)
            self.assertFalse((workspace / registry.REGISTRY_FILENAME).exists())
            self.assertFalse(config_path.exists())

    def test_cli_setup_requires_workspace(self):
        code = registry.main(["setup", "--dry-run"])
        self.assertEqual(code, 2)

    def test_cli_setup_empty_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            config_path = Path(tmp) / "translator_config.json"
            with mock.patch.object(registry, "translator_config_path", return_value=config_path):
                code = registry.main(
                    [
                        "--workspace",
                        str(workspace),
                        "setup",
                        "--no-persist-config",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue((workspace / registry.REGISTRY_FILENAME).is_file())
            self.assertFalse(config_path.exists())


if __name__ == "__main__":
    unittest.main()