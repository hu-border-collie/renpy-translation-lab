import unittest
from pathlib import Path

try:
    from gui_qt.app import MainWindow
except ImportError as exc:
    # Skip when PySide6 is missing or Qt system libraries are unavailable (e.g. headless CI).
    MainWindow = None  # type: ignore[assignment]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(MainWindow is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class GuiAppConfigHelperTests(unittest.TestCase):
    def setUp(self):
        self.window = MainWindow.__new__(MainWindow)

    def test_sync_models_for_save_preserves_fallback_models(self):
        models = self.window._sync_models_for_save(
            ["gemini-primary", "gemini-fallback", "gemini-primary", "", 123],
            "gemini-new",
        )

        self.assertEqual(models, ["gemini-new", "gemini-primary", "gemini-fallback"])

    def test_sync_models_for_save_keeps_existing_models_when_selection_is_empty(self):
        models = self.window._sync_models_for_save(
            ["gemini-primary", "gemini-fallback"],
            "",
        )

        self.assertEqual(models, ["gemini-primary", "gemini-fallback"])

    def test_missing_batch_thinking_uses_cli_default_for_supported_model(self):
        thinking_level = self.window._batch_thinking_value_for_load(
            {},
            "gemini-3.5-flash",
        )

        self.assertEqual(thinking_level, "minimal")

    def test_explicit_empty_batch_thinking_overrides_supported_model_default(self):
        thinking_level = self.window._batch_thinking_value_for_load(
            {"thinking_level": ""},
            "gemini-3.5-flash",
        )

        self.assertEqual(thinking_level, "")

    def test_empty_batch_thinking_is_saved_after_user_change_for_supported_model(self):
        should_save = self.window._should_save_batch_thinking_level(
            {},
            "gemini-3.5-flash",
            "",
            True,
        )

        self.assertTrue(should_save)

    def test_empty_batch_thinking_is_not_saved_for_implicit_supported_model_switch(self):
        should_save = self.window._should_save_batch_thinking_level(
            {},
            "gemini-3.5-flash",
            "",
            False,
        )

        self.assertFalse(should_save)

    def test_empty_batch_thinking_is_not_added_for_unsupported_model(self):
        should_save = self.window._should_save_batch_thinking_level(
            {},
            "gemini-2.5-flash",
            "",
            True,
        )

        self.assertFalse(should_save)

    def test_supported_model_switch_defaults_empty_implicit_selection_to_minimal(self):
        thinking_level = self.window._batch_thinking_value_for_model_change(
            "gemini-3.5-flash",
            "",
            False,
            False,
        )

        self.assertEqual(thinking_level, "minimal")

    def test_supported_model_switch_preserves_explicit_empty_selection(self):
        thinking_level = self.window._batch_thinking_value_for_model_change(
            "gemini-3.5-flash",
            "",
            False,
            True,
        )

        self.assertIsNone(thinking_level)

    def _prepare_bootstrap_window(self, config):
        class FakeRunner:
            def __init__(self):
                self.calls = []

            def run(self, script, args):
                self.calls.append((script, args))

        class FakeState:
            def get_game_root(self):
                return Path("C:/Game/work")

            def get_batch_script_path(self):
                return Path("C:/tool/gemini_translate_batch.py")

            def load_translator_config(self):
                return config

        self.window.state = FakeState()
        self.window.runner = FakeRunner()
        self.window.log_view = type("FakeLogView", (), {"clear": lambda _self: None})()
        self.window._focus_log_tab = lambda: None
        self.window._set_bootstrap_summary = lambda _summary: None
        self.window._append_log = lambda _text: None
        self.window._set_task_running = lambda _running: None
        return self.window.runner

    def test_bootstrap_rag_uses_skip_prepare(self):
        runner = self._prepare_bootstrap_window(
            {"batch": {"rag": {"enabled": True}}}
        )

        self.window._on_bootstrap_rag()

        self.assertEqual(
            runner.calls,
            [
                (
                    Path("C:/tool/gemini_translate_batch.py"),
                    ["bootstrap-rag", "--skip-prepare"],
                )
            ],
        )

    def test_bootstrap_source_index_uses_skip_prepare(self):
        runner = self._prepare_bootstrap_window(
            {"batch": {"source_index": {"enabled": True}}}
        )

        self.window._on_bootstrap_source_index()

        self.assertEqual(
            runner.calls,
            [
                (
                    Path("C:/tool/gemini_translate_batch.py"),
                    ["bootstrap-source-index", "--skip-prepare"],
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
