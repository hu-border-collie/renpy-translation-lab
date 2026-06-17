import json
import tempfile
import unittest
from pathlib import Path

from gui_qt.project_state import ProjectState


class GuiProjectStateTests(unittest.TestCase):
    def make_state(self, root: Path) -> ProjectState:
        state = ProjectState.__new__(ProjectState)
        state.tool_root = root
        state.batch_script = root / "gemini_translate_batch.py"
        state.api_keys_path = root / "api_keys.json"
        state.config_path = root / "translator_config.json"
        state._game_root = None
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

    def test_save_api_keys_rejects_invalid_json_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.make_state(root)
            state.api_keys_path.write_text("{bad json", encoding="utf-8")

            with self.assertRaises(ValueError):
                state.save_api_keys(["new-key"])

            self.assertEqual(state.api_keys_path.read_text(encoding="utf-8"), "{bad json")

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
            self.assertEqual(Path(saved["game_root"]), game_root)
            self.assertEqual(saved["batch"], {"model": "gemini-test"})
            self.assertEqual(saved["include_files"], ["script.rpy"])
            self.assertEqual(state.get_game_root(), game_root)

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


if __name__ == "__main__":
    unittest.main()
