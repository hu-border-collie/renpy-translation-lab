import unittest

try:
    from gui_qt.app import MainWindow
except ImportError as exc:
    if getattr(exc, "name", None) != "PySide6":
        raise
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

    def test_empty_batch_thinking_is_saved_for_supported_model(self):
        should_save = self.window._should_save_batch_thinking_level(
            {},
            "gemini-3.5-flash",
            "",
        )

        self.assertTrue(should_save)

    def test_empty_batch_thinking_is_not_added_for_unsupported_model(self):
        should_save = self.window._should_save_batch_thinking_level(
            {},
            "gemini-2.5-flash",
            "",
        )

        self.assertFalse(should_save)


if __name__ == "__main__":
    unittest.main()
