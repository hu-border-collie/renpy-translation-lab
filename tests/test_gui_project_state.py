import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gui_qt.project_state import ProjectState
from gui_qt.work_modes import WorkMode
import translator_runtime as runtime


class GuiProjectStateTests(unittest.TestCase):
    def assert_same_path(self, left, right) -> None:
        self.assertEqual(
            runtime.canonical_abs_path(left),
            runtime.canonical_abs_path(right),
        )

    def make_state(self, root: Path) -> ProjectState:
        state = ProjectState.__new__(ProjectState)
        state.tool_root = root
        state.batch_script = root / "gemini_translate_batch.py"
        state.api_keys_path = root / "api_keys.json"
        state.config_path = root / "translator_config.json"
        state._game_root = None
        state._game_root_redirect_from = None
        state._manifest_file_cache = {}
        return state

    def test_save_api_keys_preserves_existing_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            state.api_keys_path.write_text(
                json.dumps(
                    {
                        "api_keys": ["old-1", "old-2"],
                        "batch_size": 5,
                        "legacy": {"enabled": True},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            state.save_api_keys(["new-1", "old-2"])

            saved = json.loads(state.api_keys_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["api_keys"], ["new-1", "old-2"])
            self.assertEqual(saved["batch_size"], 5)
            self.assertEqual(saved["legacy"], {"enabled": True})

    def test_api_key_status_uses_file_keys_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            state.api_keys_path.write_text(
                json.dumps({"api_keys": ["file-key", "your-key-here", " "]}),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}):
                self.assertEqual(state.get_api_key_status(), (1, "file"))

    def test_api_key_status_ignores_file_placeholders_before_env_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            state.api_keys_path.write_text(
                json.dumps({"api_keys": ["your-key-here", "replace-me", " "]}),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}):
                self.assertEqual(state.get_api_key_status(), (1, "environment"))

    def test_api_key_status_treats_only_placeholders_as_unconfigured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            state.api_keys_path.write_text(
                json.dumps({"api_keys": ["your_gemini_api_key", "paste-api-key"]}),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(state.get_api_key_status(), (0, ""))

    def test_api_key_status_falls_back_to_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)

            with patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": "env-key",
                    "GEMINI_API_KEY_2": " ",
                    "GEMINI_API_KEY_3": "env-key-3",
                },
            ):
                self.assertEqual(state.get_api_key_status(), (2, "environment"))

    def test_api_keys_path_uses_legacy_data_file_when_root_file_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = workspace / "renpy-translation-lab"
            root.mkdir()
            legacy_path = workspace / "data" / "api_keys.json"
            legacy_path.parent.mkdir()
            legacy_path.write_text(
                json.dumps({"api_keys": ["legacy-key"]}),
                encoding="utf-8",
            )
            state = self.make_state(root)

            self.assertEqual(state._resolve_api_keys_path(), legacy_path)

    def test_api_keys_path_prefers_root_file_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = workspace / "renpy-translation-lab"
            root.mkdir()
            root_path = root / "api_keys.json"
            root_path.write_text(
                json.dumps({"api_keys": ["root-key"]}),
                encoding="utf-8",
            )
            legacy_path = workspace / "data" / "api_keys.json"
            legacy_path.parent.mkdir()
            legacy_path.write_text(
                json.dumps({"api_keys": ["legacy-key"]}),
                encoding="utf-8",
            )
            state = self.make_state(root)

            self.assertEqual(state._resolve_api_keys_path(), root_path)

    def test_logs_dir_uses_legacy_root_when_root_api_keys_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = workspace / "renpy-translation-lab"
            root.mkdir()
            state = self.make_state(root)

            self.assertEqual(state.get_logs_dir(), workspace / "logs" / "batch_jobs")

    def test_logs_dir_prefers_tool_root_when_root_api_keys_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "api_keys.json").write_text(
                json.dumps({"api_keys": ["root-key"]}),
                encoding="utf-8",
            )
            state = self.make_state(root)

            self.assertEqual(state.get_logs_dir(), root / "logs" / "batch_jobs")

    def test_latest_manifest_path_uses_cli_log_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = workspace / "renpy-translation-lab"
            root.mkdir()
            jobs_dir = workspace / "logs" / "batch_jobs"
            jobs_dir.mkdir(parents=True)
            manifest = jobs_dir / "job1" / "manifest.json"
            manifest.parent.mkdir()
            manifest.write_text("{}", encoding="utf-8")
            (jobs_dir / "latest_manifest.txt").write_text(str(manifest), encoding="utf-8")
            state = self.make_state(root)

            self.assertEqual(state.get_latest_manifest_path(), manifest)

    def test_load_resume_manifest_accepts_current_translation_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            state._game_root = root / "Game Work"
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "mode": "translation",
                        "base_dir": str(state._game_root),
                        "job_name": "batches/example",
                    }
                ),
                encoding="utf-8",
            )

            loaded = state.load_resume_manifest(manifest)

            self.assertEqual(loaded["job_name"], "batches/example")

    def test_load_resume_manifest_rejects_non_translation_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            state._game_root = root / "Game Work"
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"mode": "revision", "base_dir": str(state._game_root)}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "不是批量翻译任务"):
                state.load_resume_manifest(manifest)

    def test_load_resume_manifest_accepts_revision_for_revision_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            state._game_root = root / "Game Work"
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"mode": "revision", "base_dir": str(state._game_root)}),
                encoding="utf-8",
            )

            loaded = state.load_resume_manifest(manifest, work_mode=WorkMode.REVISION)

            self.assertEqual(loaded["mode"], "revision")

    def test_load_resume_manifest_rejects_other_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            state._game_root = root / "Game Work"
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"mode": "translation", "base_dir": str(root / "Other Work")}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "属于其他项目"):
                state.load_resume_manifest(manifest)

    def test_load_resume_manifest_allows_missing_game_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            manifest = root / "manifest.json"
            base_dir = root / "Game Work"
            manifest.write_text(
                json.dumps(
                    {
                        "mode": "translation",
                        "base_dir": str(base_dir),
                        "job_name": "batches/example",
                    }
                ),
                encoding="utf-8",
            )

            loaded = state.load_resume_manifest(manifest)

            self.assertEqual(loaded["base_dir"], str(base_dir))
            self.assertEqual(loaded["job_name"], "batches/example")

    def test_save_api_keys_preserves_existing_file_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            state.api_keys_path.write_text(
                json.dumps({"api_keys": ["old-key"]}),
                encoding="utf-8",
            )
            existing_mode = state.api_keys_path.stat().st_mode & 0o777
            open_calls = []
            real_open = os.open

            def recording_open(path, flags, mode):
                open_calls.append((Path(path), flags, mode))
                return real_open(path, flags, mode)

            with (
                patch("gui_qt.project_state.os.open", side_effect=recording_open),
                patch("gui_qt.project_state.os.chmod") as chmod_mock,
            ):
                state.save_api_keys(["new-key"])

            self.assertEqual(
                open_calls,
                [(
                    state.api_keys_path.with_suffix(".tmp"),
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    existing_mode,
                )],
            )
            chmod_mock.assert_called_once()
            chmod_path, chmod_mode = chmod_mock.call_args.args
            self.assertEqual(Path(chmod_path), state.api_keys_path.with_suffix(".tmp"))
            self.assertEqual(chmod_mode, existing_mode)

    def test_save_new_api_keys_creates_temp_file_restrictively(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            open_calls = []
            real_open = os.open

            def recording_open(path, flags, mode):
                open_calls.append((Path(path), flags, mode))
                return real_open(path, flags, mode)

            with patch("gui_qt.project_state.os.open", side_effect=recording_open):
                state.save_api_keys(["new-key"])

            self.assertEqual(
                open_calls,
                [(
                    state.api_keys_path.with_suffix(".tmp"),
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )],
            )

    def test_save_api_keys_rejects_invalid_json_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            state.api_keys_path.write_text("{bad json", encoding="utf-8")

            with self.assertRaises(ValueError):
                state.save_api_keys(["new-key"])

            self.assertEqual(state.api_keys_path.read_text(encoding="utf-8"), "{bad json")

    def test_save_api_keys_wraps_write_errors_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            state.api_keys_path.write_text(
                json.dumps({"api_keys": ["old-key"]}),
                encoding="utf-8",
            )

            with patch("gui_qt.project_state.os.replace", side_effect=OSError("denied")):
                with self.assertRaisesRegex(ValueError, "Failed to write JSON file"):
                    state.save_api_keys(["new-key"])

            saved = json.loads(state.api_keys_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["api_keys"], ["old-key"])
            self.assertFalse(state.api_keys_path.with_suffix(".tmp").exists())

    def test_take_game_root_redirect_from_clears_pending_notice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "Game Example"
            work = project / "work"
            original_game = project / "original" / "game"
            work.mkdir(parents=True)
            original_game.mkdir(parents=True)
            state = self.make_state(root)
            state.config_path.write_text("{}", encoding="utf-8")
            state.set_game_root(project)

            self.assertEqual(state.take_game_root_redirect_from(), project)
            self.assertIsNone(state.take_game_root_redirect_from())

    def test_set_game_root_auto_redirects_to_nested_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "Game Example"
            work = project / "work"
            original_game = project / "original" / "game"
            work.mkdir(parents=True)
            original_game.mkdir(parents=True)
            state = self.make_state(root)
            state.config_path.write_text("{}", encoding="utf-8")

            effective, adjusted = state.set_game_root(project)
            self.assertTrue(adjusted)
            self.assert_same_path(effective, work)
            saved = json.loads(state.config_path.read_text(encoding="utf-8"))
            self.assert_same_path(saved["game_root"], work)

    def test_set_game_root_preserves_translator_config_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            game_root = root / "Game Work"
            state.config_path.write_text(
                json.dumps(
                    {
                        "game_root": "old",
                        "batch": {"model": "gemini-test"},
                        "include_files": ["script.rpy"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            state.set_game_root(game_root)

            saved = json.loads(state.config_path.read_text(encoding="utf-8"))
            self.assert_same_path(saved["game_root"], game_root)
            self.assertEqual(saved["batch"]["model"], "gemini-test")
            self.assert_same_path(
                saved["batch"]["macro_setting_file"],
                game_root / "macro_setting.md",
            )
            self.assert_same_path(saved["glossary_file"], game_root / "glossary.json")
            self.assertEqual(saved["include_files"], ["script.rpy"])
            self.assert_same_path(state.get_game_root(), game_root)

    def test_set_game_root_canonicalizes_windows_short_path(self):
        if os.name != "nt":
            self.skipTest("Windows-only short-path normalization")

        import ctypes

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            game_dir = root / "Game Work"
            game_dir.mkdir(parents=True, exist_ok=True)
            long_path = str(game_dir.resolve())
            buffer = ctypes.create_unicode_buffer(260)
            if ctypes.windll.kernel32.GetShortPathNameW(long_path, buffer, len(buffer)) == 0:
                self.skipTest("Could not resolve Windows short path for temp directory")

            short_path = buffer.value
            if os.path.normcase(short_path) == os.path.normcase(long_path):
                self.skipTest("Short-path generation disabled on this system")

            state.set_game_root(Path(short_path))

            saved = json.loads(state.config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["game_root"], runtime.canonical_abs_path(long_path))
            self.assert_same_path(state.get_game_root(), long_path)

    def test_set_game_root_rejects_invalid_json_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            game_root = root / "Game Work"
            state.config_path.write_text("{bad json", encoding="utf-8")

            with self.assertRaises(ValueError):
                state.set_game_root(game_root)

            self.assertEqual(state.config_path.read_text(encoding="utf-8"), "{bad json")
            self.assertIsNone(state.get_game_root())

    def test_load_translator_config_returns_empty_dict_for_non_object_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            state.config_path.write_text('["not", "an", "object"]', encoding="utf-8")

            self.assertEqual(state.load_translator_config(), {})

    def test_save_model_config_preserves_other_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            state.config_path.write_text(
                json.dumps(
                    {
                        "game_root": "some_root",
                        "sync": {
                            "model": "gemini-old",
                            "chunk_size": 40,
                        },
                        "batch": {
                            "model": "gemini-old-batch",
                            "thinking_level": "",
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            config = state.load_translator_config()
            config.setdefault("sync", {})["model"] = "gemini-new"
            config.setdefault("batch", {})["model"] = "gemini-new-batch"
            config.setdefault("batch", {})["thinking_level"] = "high"
            state.save_translator_config(config)

            saved = json.loads(state.config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["game_root"], "some_root")
            self.assertEqual(saved["sync"]["model"], "gemini-new")
            self.assertEqual(saved["sync"]["chunk_size"], 40)
            self.assertEqual(saved["batch"]["model"], "gemini-new-batch")
            self.assertEqual(saved["batch"]["thinking_level"], "high")

    def test_get_latest_manifest_path_for_mode_walks_logs_and_filters_correctly(self):
        from gui_qt.work_modes import WorkMode
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)

            logs_dir = root / "logs" / "batch_jobs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            dir1 = logs_dir / "job1"
            dir1.mkdir()
            manifest1 = dir1 / "manifest.json"
            manifest1.write_text(json.dumps({
                "mode": "keyword_extraction",
                "base_dir": "C:/different/project",
            }), encoding="utf-8")

            dir2 = logs_dir / "job2"
            dir2.mkdir()
            manifest2 = dir2 / "manifest.json"
            manifest2.write_text(json.dumps({
                "mode": "translation",
                "base_dir": "C:/correct/project",
            }), encoding="utf-8")

            dir3 = logs_dir / "job3"
            dir3.mkdir()
            manifest3 = dir3 / "manifest.json"
            manifest3.write_text(json.dumps({
                "mode": "keyword_extraction",
                "base_dir": "C:/correct/project",
            }), encoding="utf-8")

            dir4 = logs_dir / "job4"
            dir4.mkdir()
            manifest4 = dir4 / "manifest.json"
            manifest4.write_text(json.dumps({
                "mode": "keyword_extraction",
                "base_dir": "C:/correct/project",
            }), encoding="utf-8")

            os.utime(manifest4, (1000, 1000))
            os.utime(manifest3, (2000, 2000))

            state.get_logs_dir = lambda: logs_dir
            state._normalized_path_text = lambda p: os.path.normcase(os.path.abspath(p))

            game_root = Path("C:/correct/project")
            latest = state.get_latest_manifest_path_for_mode(game_root, WorkMode.KEYWORD_EXTRACTION)

            self.assertIsNotNone(latest)
            self.assertEqual(Path(latest).resolve(), manifest3.resolve())

    def test_get_latest_submitted_manifest_ignores_newer_local_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            logs_dir = root / "logs" / "batch_jobs"
            remote = logs_dir / "remote" / "manifest.json"
            local = logs_dir / "new-local" / "manifest.json"
            remote.parent.mkdir(parents=True)
            local.parent.mkdir(parents=True)
            remote.write_text(
                json.dumps(
                    {
                        "mode": "keyword_extraction",
                        "base_dir": "C:/correct/project",
                        "job_name": "batches/remote-keywords",
                    }
                ),
                encoding="utf-8",
            )
            local.write_text(
                json.dumps(
                    {
                        "mode": "keyword_extraction",
                        "base_dir": "C:/correct/project",
                        "job_name": "",
                    }
                ),
                encoding="utf-8",
            )
            (logs_dir / "latest_manifest.txt").write_text(
                str(local), encoding="utf-8"
            )
            state.get_logs_dir = lambda: logs_dir
            state._normalized_path_text = lambda p: os.path.normcase(os.path.abspath(p))

            submitted = state.get_latest_submitted_manifest_path_for_mode(
                Path("C:/correct/project"),
                WorkMode.KEYWORD_EXTRACTION,
            )

            self.assertEqual(Path(submitted).resolve(), remote.resolve())
    def test_get_latest_manifest_path_for_mode_stops_on_unreadable_latest_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)

            logs_dir = root / "logs" / "batch_jobs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            corrupt_dir = logs_dir / "job_corrupt"
            corrupt_dir.mkdir()
            corrupt_manifest = corrupt_dir / "manifest.json"
            corrupt_manifest.write_text("{", encoding="utf-8")
            (logs_dir / "latest_manifest.txt").write_text(str(corrupt_manifest), encoding="utf-8")

            old_dir = logs_dir / "job_old"
            old_dir.mkdir()
            old_manifest = old_dir / "manifest.json"
            old_manifest.write_text(json.dumps({
                "mode": "keyword_extraction",
                "base_dir": "C:/correct/project",
            }), encoding="utf-8")

            state.get_logs_dir = lambda: logs_dir
            state._normalized_path_text = lambda p: os.path.normcase(os.path.abspath(p))

            latest = state.get_latest_manifest_path_for_mode(
                Path("C:/correct/project"),
                WorkMode.KEYWORD_EXTRACTION,
            )

            self.assertIsNone(latest)

    def test_get_latest_manifest_path_for_mode_reuses_history_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            logs_dir = root / "logs" / "batch_jobs"
            manifest_dir = logs_dir / "job1"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest = manifest_dir / "manifest.json"
            manifest.write_text(json.dumps({
                "mode": "keyword_extraction",
                "base_dir": "C:/correct/project",
            }), encoding="utf-8")

            state.get_logs_dir = lambda: logs_dir
            state._normalized_path_text = lambda p: os.path.normcase(os.path.abspath(p))
            game_root = Path("C:/correct/project")

            first = state.get_latest_manifest_path_for_mode(
                game_root,
                WorkMode.KEYWORD_EXTRACTION,
            )
            first_index = state._manifest_history_index()
            second = state.get_latest_manifest_path_for_mode(
                game_root,
                WorkMode.KEYWORD_EXTRACTION,
            )
            second_index = state._manifest_history_index()

            self.assertEqual(Path(first).resolve(), manifest.resolve())
            self.assertEqual(Path(second).resolve(), manifest.resolve())
            self.assertIs(first_index, second_index)

    def test_load_manifest_file_reuses_mtime_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"mode": "translation", "base_dir": "C:/game/work"}),
                encoding="utf-8",
            )

            read_calls = {"count": 0}
            original_read = state._read_json_object

            def counting_read(path, description):
                read_calls["count"] += 1
                return original_read(path, description)

            with patch.object(state, "_read_json_object", side_effect=counting_read):
                first = state.load_manifest_file(manifest)
                second = state.load_manifest_file(manifest)

            self.assertEqual(first.get("mode"), "translation")
            self.assertIs(second, first)
            self.assertEqual(read_calls["count"], 1)

    def test_invalidate_manifest_file_cache_forces_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"mode": "translation", "base_dir": "C:/game/work"}),
                encoding="utf-8",
            )

            read_calls = {"count": 0}
            original_read = state._read_json_object

            def counting_read(path, description):
                read_calls["count"] += 1
                return original_read(path, description)

            with patch.object(state, "_read_json_object", side_effect=counting_read):
                state.load_manifest_file(manifest)
                state.invalidate_manifest_file_cache(manifest)
                state.load_manifest_file(manifest)

            self.assertEqual(read_calls["count"], 2)

    def test_load_manifest_file_keeps_lite_and_full_cache_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            manifest = root / "manifest.json"
            payload = {
                "mode": "translation",
                "base_dir": "C:/game/work",
                "padding": "x" * 300_000,
                "chunks": [{"key": "chunk-1"}],
                "last_check_summary": {"safety_level": "safe"},
            }
            manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            lite = state.load_manifest_file(manifest, lite=True)
            full = state.load_manifest_file(manifest, lite=False)

            self.assertNotIn("chunks", lite)
            self.assertIn("chunks", full)
            self.assertIsNot(lite, full)

    def test_invalidate_manifest_file_cache_clears_lite_and_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"mode": "translation", "base_dir": "C:/game/work"}),
                encoding="utf-8",
            )

            first = state.load_manifest_file(manifest)
            state.invalidate_manifest_file_cache(manifest)
            second = state.load_manifest_file(manifest)

            self.assertEqual(first.get("mode"), "translation")
            self.assertIsNot(second, first)

if __name__ == "__main__":
    unittest.main()
