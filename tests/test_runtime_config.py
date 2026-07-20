"""Tests for RuntimeConfig / defaults-first reload isolation (issue #216)."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import translator_runtime as runtime


def _snapshot_sensitive_runtime():
    return {
        "API_KEYS": list(runtime.API_KEYS),
        "MODELS": list(runtime.MODELS),
        "CURRENT_KEY_INDEX": runtime.CURRENT_KEY_INDEX,
        "CURRENT_MODEL_INDEX": runtime.CURRENT_MODEL_INDEX,
        "BASE_DIR": runtime.BASE_DIR,
        "TL_SUBDIR": runtime.TL_SUBDIR,
        "TL_DIR": runtime.TL_DIR,
        "WORK_GAME_DIR": runtime.WORK_GAME_DIR,
        "SOURCE_GAME_DIR": runtime.SOURCE_GAME_DIR,
        "PREP_LANGUAGE": runtime.PREP_LANGUAGE,
        "PREP_ALLOW_SHELL_COMMANDS": runtime.PREP_ALLOW_SHELL_COMMANDS,
        "PREP_UNPACK_COMMAND": runtime.PREP_UNPACK_COMMAND,
        "PREP_TEMPLATE_COMMAND": runtime.PREP_TEMPLATE_COMMAND,
        "PREP_RENPY_SDK_DIR": runtime.PREP_RENPY_SDK_DIR,
        "GLOSSARY_FILE": runtime.GLOSSARY_FILE,
        "MAX_ITEMS": runtime.MAX_ITEMS,
        "MAX_CHARS": runtime.MAX_CHARS,
        "SYNC_MAX_OUTPUT_TOKENS": runtime.SYNC_MAX_OUTPUT_TOKENS,
        "SYNC_BACKEND": runtime.SYNC_BACKEND,
        "INCLUDE_FILES": set(runtime.INCLUDE_FILES),
        "INCLUDE_PREFIXES": set(runtime.INCLUDE_PREFIXES),
        "SYNC_RAG_ENABLED": runtime.SYNC_RAG_ENABLED,
        "SYNC_STORY_MEMORY_ENABLED": runtime.SYNC_STORY_MEMORY_ENABLED,
        "ENV_GAME_ROOT": runtime.ENV_GAME_ROOT,
    }


def _restore_sensitive_runtime(snapshot):
    for key, value in snapshot.items():
        setattr(runtime, key, value)


class RuntimeConfigObjectTests(unittest.TestCase):
    def test_default_runtime_config_is_independent_copy(self):
        a = runtime.default_runtime_config()
        b = runtime.default_runtime_config()
        a.api_keys.append("mutated")
        a.models.append("extra-model")
        a.include_files.add("x.rpy")
        self.assertEqual(b.api_keys, [])
        self.assertEqual(b.models, list(runtime.DEFAULT_MODELS))
        self.assertEqual(b.include_files, set())
        self.assertEqual(a.prep_language, runtime.DEFAULT_PREP_LANGUAGE)
        self.assertEqual(a.tl_subdir, runtime.DEFAULT_TL_SUBDIR)

    def test_apply_and_snapshot_round_trip(self):
        snapshot = _snapshot_sensitive_runtime()
        try:
            cfg = runtime.default_runtime_config()
            cfg.api_keys = ["key-a"]
            cfg.prep_language = "japanese"
            cfg.tl_subdir = "game/tl/japanese"
            cfg.base_dir = "C:/games/Example/work"
            cfg.tl_dir = "C:/games/Example/work/game/tl/japanese"
            cfg.work_game_dir = "C:/games/Example/work/game"
            cfg.max_items = 11
            applied = runtime.apply_runtime_config(cfg)
            self.assertEqual(runtime.API_KEYS, ["key-a"])
            self.assertEqual(runtime.PREP_LANGUAGE, "japanese")
            self.assertEqual(runtime.TL_SUBDIR, "game/tl/japanese")
            self.assertEqual(runtime.MAX_ITEMS, 11)

            snapped = runtime.snapshot_runtime_config()
            self.assertEqual(snapped.api_keys, ["key-a"])
            self.assertEqual(snapped.prep_language, "japanese")
            self.assertEqual(snapped.tl_subdir, "game/tl/japanese")
            self.assertEqual(snapped.max_items, 11)
            self.assertIsInstance(applied, runtime.RuntimeConfig)
            self.assertIs(runtime.ProjectContext, runtime.RuntimeConfig)
            got = runtime.get_runtime_config()
            self.assertEqual(got.api_keys, ["key-a"])
        finally:
            _restore_sensitive_runtime(snapshot)


class DefaultsFirstReloadTests(unittest.TestCase):
    def test_omitted_prep_language_and_tl_subdir_do_not_leak(self):
        snapshot = _snapshot_sensitive_runtime()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                work_a = workspace / "work_a"
                work_b = workspace / "work_b"
                work_a.mkdir()
                work_b.mkdir()

                config_a = {
                    "game_root": str(work_a),
                    "tl_subdir": "game/tl/japanese",
                    "prepare": {"language": "japanese"},
                }
                config_b = {
                    "game_root": str(work_b),
                    # intentionally omit tl_subdir and prepare.language
                }
                path_a = workspace / "config_a.json"
                path_b = workspace / "config_b.json"
                path_a.write_text(json.dumps(config_a), encoding="utf-8")
                path_b.write_text(json.dumps(config_b), encoding="utf-8")

                with (
                    mock.patch.object(runtime, "ROOT_DIR", str(workspace / "tool")),
                    mock.patch.object(runtime, "TOOL_DIR", str(workspace / "tool")),
                    mock.patch.dict(os.environ, {}, clear=True),
                ):
                    with mock.patch.object(runtime, "TRANSLATOR_CONFIG", str(path_a)):
                        runtime.load_translator_settings()
                    self.assertEqual(runtime.PREP_LANGUAGE, "japanese")
                    self.assertEqual(runtime.TL_SUBDIR, "game/tl/japanese")
                    self.assertTrue(
                        runtime.TL_DIR.replace("\\", "/").endswith("work_a/game/tl/japanese")
                    )

                    with mock.patch.object(runtime, "TRANSLATOR_CONFIG", str(path_b)):
                        runtime.load_translator_settings()
                    self.assertEqual(runtime.PREP_LANGUAGE, runtime.DEFAULT_PREP_LANGUAGE)
                    self.assertEqual(runtime.TL_SUBDIR, runtime.DEFAULT_TL_SUBDIR)
                    self.assertTrue(
                        runtime.TL_DIR.replace("\\", "/").endswith(
                            f"work_b/{runtime.DEFAULT_TL_SUBDIR.replace(chr(92), '/')}"
                        )
                    )
        finally:
            _restore_sensitive_runtime(snapshot)

    def test_api_keys_rebuild_and_do_not_leak_across_loads(self):
        snapshot = _snapshot_sensitive_runtime()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                keys_a = workspace / "keys_a.json"
                keys_b = workspace / "keys_b.json"
                keys_a.write_text(
                    json.dumps({"api_keys": ["project-a-key"]}),
                    encoding="utf-8",
                )
                # Empty keys list must clear previous project keys.
                keys_b.write_text(json.dumps({"api_keys": []}), encoding="utf-8")

                with mock.patch("sys.stdout", io.StringIO()), mock.patch.dict(os.environ, {}, clear=True):
                    with mock.patch.object(runtime, "CONFIG_FILE", str(keys_a)):
                        runtime.load_config(require_api_key=False)
                    self.assertEqual(runtime.API_KEYS, ["project-a-key"])

                    with mock.patch.object(runtime, "CONFIG_FILE", str(keys_b)):
                        runtime.load_config(require_api_key=False)
                    self.assertEqual(runtime.API_KEYS, [])

                    # Failed parse must not keep previous keys either.
                    runtime.API_KEYS = ["stale-key"]
                    bad = workspace / "keys_bad.json"
                    bad.write_text("{not-json", encoding="utf-8")
                    with mock.patch.object(runtime, "CONFIG_FILE", str(bad)):
                        runtime.load_config(require_api_key=False)
                    self.assertEqual(runtime.API_KEYS, [])
        finally:
            _restore_sensitive_runtime(snapshot)

    def test_project_switch_via_load_runtime_config(self):
        snapshot = _snapshot_sensitive_runtime()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                work_a = workspace / "proj_a" / "work"
                work_b = workspace / "proj_b" / "work"
                work_a.mkdir(parents=True)
                work_b.mkdir(parents=True)

                translator_a = workspace / "translator_a.json"
                translator_b = workspace / "translator_b.json"
                keys_a = workspace / "keys_a.json"
                keys_b = workspace / "keys_b.json"

                translator_a.write_text(
                    json.dumps(
                        {
                            "game_root": str(work_a),
                            "tl_subdir": "game/tl/korean",
                            "prepare": {"language": "korean"},
                            "sync": {"backend": "litellm", "model": "openai/a"},
                        }
                    ),
                    encoding="utf-8",
                )
                translator_b.write_text(
                    json.dumps(
                        {
                            "game_root": str(work_b),
                            # omit language / tl_subdir / sync
                        }
                    ),
                    encoding="utf-8",
                )
                keys_a.write_text(json.dumps({"api_keys": ["key-a"]}), encoding="utf-8")
                keys_b.write_text(json.dumps({"api_keys": ["key-b"]}), encoding="utf-8")

                with (
                    mock.patch.object(runtime, "ROOT_DIR", str(workspace / "tool")),
                    mock.patch.object(runtime, "TOOL_DIR", str(workspace / "tool")),
                    mock.patch.dict(os.environ, {}, clear=True),
                    mock.patch("sys.stdout", io.StringIO()),
                ):
                    with (
                        mock.patch.object(runtime, "TRANSLATOR_CONFIG", str(translator_a)),
                        mock.patch.object(runtime, "CONFIG_FILE", str(keys_a)),
                    ):
                        cfg_a = runtime.load_runtime_config(require_api_key=False)
                    self.assertEqual(cfg_a.api_keys, ["key-a"])
                    self.assertEqual(cfg_a.prep_language, "korean")
                    self.assertEqual(cfg_a.tl_subdir, "game/tl/korean")
                    self.assertEqual(cfg_a.sync_backend, "litellm")
                    self.assertEqual(runtime.API_KEYS, ["key-a"])

                    with (
                        mock.patch.object(runtime, "TRANSLATOR_CONFIG", str(translator_b)),
                        mock.patch.object(runtime, "CONFIG_FILE", str(keys_b)),
                    ):
                        cfg_b = runtime.load_runtime_config(require_api_key=False)
                    self.assertEqual(cfg_b.api_keys, ["key-b"])
                    self.assertEqual(cfg_b.prep_language, runtime.DEFAULT_PREP_LANGUAGE)
                    self.assertEqual(cfg_b.tl_subdir, runtime.DEFAULT_TL_SUBDIR)
                    self.assertEqual(cfg_b.sync_backend, runtime.DEFAULT_SYNC_BACKEND)
                    self.assertEqual(runtime.API_KEYS, ["key-b"])
                    self.assertEqual(runtime.PREP_LANGUAGE, runtime.DEFAULT_PREP_LANGUAGE)
                    # Models fall back to defaults when project B omits sync.models
                    # after a full defaults-first load_runtime_config.
                    self.assertEqual(runtime.MODELS, list(runtime.DEFAULT_MODELS))
        finally:
            _restore_sensitive_runtime(snapshot)

    def test_failed_translator_config_parse_uses_defaults(self):
        snapshot = _snapshot_sensitive_runtime()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                work = workspace / "work"
                work.mkdir()
                # Seed "stale" state as if a previous project was loaded.
                runtime.PREP_LANGUAGE = "japanese"
                runtime.TL_SUBDIR = "game/tl/japanese"
                runtime.API_KEYS = ["stale"]

                bad = workspace / "translator_config.json"
                bad.write_text("{broken", encoding="utf-8")

                with (
                    mock.patch.object(runtime, "TRANSLATOR_CONFIG", str(bad)),
                    mock.patch.object(runtime, "ROOT_DIR", str(workspace / "tool")),
                    mock.patch.object(runtime, "TOOL_DIR", str(workspace / "tool")),
                    mock.patch.dict(os.environ, {"GAME_ROOT": str(work)}, clear=True),
                    mock.patch("sys.stdout", io.StringIO()),
                ):
                    runtime.load_translator_settings()

                self.assertEqual(runtime.PREP_LANGUAGE, runtime.DEFAULT_PREP_LANGUAGE)
                self.assertEqual(runtime.TL_SUBDIR, runtime.DEFAULT_TL_SUBDIR)
                # load_translator_settings does not clear API keys (owned by load_config).
                self.assertEqual(runtime.API_KEYS, ["stale"])
        finally:
            _restore_sensitive_runtime(snapshot)

    def test_repeated_reload_same_config_is_stable(self):
        snapshot = _snapshot_sensitive_runtime()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                work = workspace / "work"
                work.mkdir()
                config_path = workspace / "translator_config.json"
                config_path.write_text(
                    json.dumps(
                        {
                            "game_root": str(work),
                            "tl_subdir": "game/tl/japanese",
                            "prepare": {"language": "japanese"},
                        }
                    ),
                    encoding="utf-8",
                )
                with (
                    mock.patch.object(runtime, "TRANSLATOR_CONFIG", str(config_path)),
                    mock.patch.object(runtime, "ROOT_DIR", str(workspace / "tool")),
                    mock.patch.object(runtime, "TOOL_DIR", str(workspace / "tool")),
                    mock.patch.dict(os.environ, {}, clear=True),
                ):
                    runtime.load_translator_settings()
                    first = runtime.snapshot_runtime_config()
                    runtime.load_translator_settings()
                    second = runtime.snapshot_runtime_config()

                self.assertEqual(first.prep_language, "japanese")
                self.assertEqual(second.prep_language, "japanese")
                self.assertEqual(first.tl_subdir, second.tl_subdir)
                self.assertEqual(first.base_dir, second.base_dir)
        finally:
            _restore_sensitive_runtime(snapshot)


class RuntimeConfigScopeTests(unittest.TestCase):
    def test_scope_applies_config_and_restores_previous(self):
        snapshot = _snapshot_sensitive_runtime()
        try:
            before = runtime.snapshot_runtime_config()
            job = before.copy()
            job.prep_language = "korean"
            job.api_keys = ["scope-key"]
            seen = {}
            with runtime.runtime_config_scope(job) as active:
                seen["lang"] = runtime.PREP_LANGUAGE
                seen["keys"] = list(runtime.API_KEYS)
                seen["active_lang"] = active.prep_language
            self.assertEqual(seen["lang"], "korean")
            self.assertEqual(seen["keys"], ["scope-key"])
            self.assertEqual(seen["active_lang"], "korean")
            self.assertEqual(runtime.PREP_LANGUAGE, before.prep_language)
            self.assertEqual(runtime.API_KEYS, before.api_keys)
        finally:
            _restore_sensitive_runtime(snapshot)

    def test_scope_restores_after_exception(self):
        snapshot = _snapshot_sensitive_runtime()
        try:
            before_lang = runtime.PREP_LANGUAGE
            job = runtime.snapshot_runtime_config().copy()
            job.prep_language = "japanese"
            with self.assertRaises(RuntimeError):
                with runtime.runtime_config_scope(job):
                    self.assertEqual(runtime.PREP_LANGUAGE, "japanese")
                    raise RuntimeError("job failed")
            self.assertEqual(runtime.PREP_LANGUAGE, before_lang)
        finally:
            _restore_sensitive_runtime(snapshot)

    def test_scope_reload_translator_settings_is_restored(self):
        snapshot = _snapshot_sensitive_runtime()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                work = workspace / "work"
                work.mkdir()
                config_path = workspace / "translator_config.json"
                config_path.write_text(
                    json.dumps(
                        {
                            "game_root": str(work),
                            "tl_subdir": "game/tl/japanese",
                            "prepare": {"language": "japanese"},
                        }
                    ),
                    encoding="utf-8",
                )
                before_lang = runtime.PREP_LANGUAGE
                with (
                    mock.patch.object(runtime, "TRANSLATOR_CONFIG", str(config_path)),
                    mock.patch.object(runtime, "ROOT_DIR", str(workspace / "tool")),
                    mock.patch.object(runtime, "TOOL_DIR", str(workspace / "tool")),
                    mock.patch.dict(os.environ, {}, clear=True),
                ):
                    with runtime.runtime_config_scope(reload_translator_settings=True):
                        self.assertEqual(runtime.PREP_LANGUAGE, "japanese")
                    self.assertEqual(runtime.PREP_LANGUAGE, before_lang)
        finally:
            _restore_sensitive_runtime(snapshot)


if __name__ == "__main__":
    unittest.main()
