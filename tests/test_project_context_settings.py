import json
import tempfile
import unittest
from pathlib import Path

from project_context_settings import (
    apply_project_context_settings_to_config,
    default_context_flags_from_config,
    load_project_context_settings,
    project_context_settings_path,
    resolve_batch_context_flags,
    save_project_context_settings,
)
from gui_qt.bootstrap_report import read_batch_context_flags


class ProjectContextSettingsTests(unittest.TestCase):
    def test_defaults_from_global_config(self):
        flags = default_context_flags_from_config(
            {
                "batch": {
                    "rag": {"enabled": True, "bootstrap_on_build": False},
                    "source_index": {"enabled": True},
                }
            }
        )
        self.assertTrue(flags["rag_enabled"])
        self.assertTrue(flags["source_index_enabled"])
        self.assertFalse(flags["bootstrap_on_build"])

    def test_project_file_overrides_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_project_context_settings(
                root,
                {
                    "rag_enabled": False,
                    "source_index_enabled": True,
                    "bootstrap_on_build": False,
                },
            )
            path = project_context_settings_path(root)
            self.assertTrue(Path(path).is_file())
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertFalse(data["batch_rag_enabled"])
            self.assertTrue(data["batch_source_index_enabled"])
            self.assertEqual(list(root.glob(".project_context_settings.json.*.tmp")), [])

            global_config = {
                "batch": {
                    "rag": {"enabled": True, "bootstrap_on_build": True},
                    "source_index": {"enabled": False},
                }
            }
            flags = resolve_batch_context_flags(global_config, root)
            self.assertFalse(flags["rag_enabled"])
            self.assertTrue(flags["source_index_enabled"])
            self.assertFalse(flags["bootstrap_on_build"])

            applied = apply_project_context_settings_to_config(dict(global_config), root)
            self.assertFalse(applied["batch"]["rag"]["enabled"])
            self.assertTrue(applied["batch"]["source_index"]["enabled"])

    def test_missing_project_file_uses_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(load_project_context_settings(root))
            flags = resolve_batch_context_flags(
                {
                    "batch": {
                        "rag": {"enabled": True},
                        "source_index": {"enabled": False},
                    }
                },
                root,
            )
            self.assertTrue(flags["rag_enabled"])
            self.assertFalse(flags["source_index_enabled"])

    def test_read_batch_context_flags_honors_game_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_project_context_settings(
                root,
                {
                    "rag_enabled": True,
                    "source_index_enabled": True,
                    "bootstrap_on_build": True,
                },
            )
            flags = read_batch_context_flags(
                {
                    "batch": {
                        "rag": {"enabled": False, "bootstrap_on_build": False},
                        "source_index": {"enabled": False},
                    }
                },
                game_root=str(root),
            )
            self.assertTrue(flags["rag_enabled"])
            self.assertTrue(flags["source_index_enabled"])
            self.assertTrue(flags["bootstrap_on_build"])


if __name__ == "__main__":
    unittest.main()
