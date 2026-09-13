"""Keep translator_config.example.json in sync with managed settings (#424)."""

from __future__ import annotations

import json
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
