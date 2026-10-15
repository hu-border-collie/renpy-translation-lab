"""Keep translator_config.example.json in sync with managed settings (#424)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import gui_qt.settings_schema as settings_schema


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "translator_config.example.json"


def _advanced_fields() -> dict[str, settings_schema.SettingField]:
    fields: dict[str, settings_schema.SettingField] = {}
    for name in dir(settings_schema):
        value = getattr(settings_schema, name)
        if not isinstance(value, tuple) or not value:
            continue
        if not all(isinstance(item, settings_schema.SettingField) for item in value):
            continue
        for field in value:
            fields[field.key] = field
    return fields


def _example_config() -> dict:
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def _value_at(config: dict, path: tuple[str, ...]):
    current = config
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None, False
        current = current[part]
    return current, True


class ConfigExampleSyncTests(unittest.TestCase):
    def test_new_project_batch_defaults_match_cli_gui_and_example(self):
        # A fresh interpreter avoids previous tests' mutated Batch globals and
        # inspects import defaults without loading private config or API keys.
        code = (
            "import json; import gemini_translate_batch as b; "
            "print(json.dumps({'model':b.BATCH_MODEL, 'chunk_size':b.BATCH_TARGET_SIZE, "
            "'max_source_chars':b.BATCH_TARGET_CHARS, 'max_output_tokens':b.BATCH_MAX_OUTPUT_TOKENS, "
            "'context_before':b.BATCH_CONTEXT_BEFORE, 'context_after':b.BATCH_CONTEXT_AFTER, "
            "'temperature':b.BATCH_TEMPERATURE, 'thinking_level':b.BATCH_THINKING_LEVEL, "
            "'rag_enabled':b.RAG_ENABLED, 'source_index_enabled':b.SOURCE_INDEX_ENABLED}))"
        )
        runtime_defaults = json.loads(subprocess.check_output(
            [sys.executable, "-B", "-c", code], cwd=EXAMPLE_PATH.parent, text=True,
        ))
        example = _example_config()["batch"]
        advanced = settings_schema.read_advanced_settings({})
        expected = {
            "model": "gemini-3.1-flash-lite", "chunk_size": 60,
            "max_source_chars": 18000, "max_output_tokens": 12288,
            "context_before": 30, "context_after": 10, "temperature": 0.2,
            "thinking_level": "", "rag_enabled": False, "source_index_enabled": False,
        }
        self.assertEqual(runtime_defaults, expected)
        for key in ("chunk_size", "max_source_chars", "max_output_tokens",
                    "context_before", "context_after", "temperature"):
            with self.subTest(key=key):
                self.assertEqual(example[key], runtime_defaults[key])
                self.assertEqual(advanced[f"batch_{key}"], runtime_defaults[key])
        basic = settings_schema.BASIC_RECOMMENDED_VALUES
        self.assertEqual(basic["batch_model"], runtime_defaults["model"])
        self.assertEqual(basic["batch_thinking_level"], example["thinking_level"])
        for key in ("rag", "source_index"):
            self.assertEqual(basic[f"{key}_enabled"], example[key]["enabled"])

    def test_new_rag_default_preserves_saved_project_retrieval_choices(self):
        from project_context_settings import default_context_flags_from_config

        saved = {"batch": {"rag": {"enabled": True}, "source_index": {"enabled": True}}}
        flags = default_context_flags_from_config(saved)
        self.assertTrue(flags["rag_enabled"])
        self.assertTrue(flags["source_index_enabled"])
        defaults = default_context_flags_from_config({})
        self.assertEqual(defaults["rag_enabled"], settings_schema.BASIC_RECOMMENDED_VALUES["rag_enabled"])
        self.assertEqual(defaults["source_index_enabled"], settings_schema.BASIC_RECOMMENDED_VALUES["source_index_enabled"])

    def test_example_covers_every_advanced_setting_field(self):
        config = _example_config()
        missing = []
        for field in _advanced_fields().values():
            _value, present = _value_at(config, field.path)
            if not present:
                missing.append(f"{field.key} ({'.'.join(field.path)})")
        self.assertEqual(
            missing,
            [],
            "translator_config.example.json is missing managed settings: "
            + ", ".join(missing),
        )

    def test_example_values_pass_managed_setting_normalization(self):
        config = _example_config()
        invalid = []
        for field in _advanced_fields().values():
            value, present = _value_at(config, field.path)
            if not present:
                continue
            try:
                settings_schema.normalize_for_write(field, value)
            except ValueError as exc:
                invalid.append(f"{field.key}: {exc}")
        self.assertEqual(invalid, [])

    def test_example_keeps_required_top_level_sections(self):
        config = _example_config()
        for section in (
            "game_root",
            "tl_subdir",
            "generation",
            "context_storage",
            "prepare",
            "sync",
            "batch",
            "model_routing",
        ):
            with self.subTest(section=section):
                self.assertIn(section, config)


if __name__ == "__main__":
    unittest.main()
