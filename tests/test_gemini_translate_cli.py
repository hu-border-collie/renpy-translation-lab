import unittest
from unittest import mock

import gemini_translate


class GeminiTranslateCliTests(unittest.TestCase):
    def test_default_command_generates_preview(self):
        with (
            mock.patch.object(gemini_translate.runtime, "initialize_runtime_logging"),
            mock.patch.object(gemini_translate.runtime, "run_translation") as run_preview,
            mock.patch.object(gemini_translate.runtime, "apply_sync_translation_preview") as apply_preview,
        ):
            result = gemini_translate.main([])

        self.assertEqual(result, 0)
        run_preview.assert_called_once_with(prepare=False)
        apply_preview.assert_not_called()

    def test_apply_command_does_not_run_translation_api(self):
        with (
            mock.patch.object(gemini_translate.runtime, "initialize_runtime_logging"),
            mock.patch.object(gemini_translate.runtime, "run_translation") as run_preview,
            mock.patch.object(gemini_translate.runtime, "apply_sync_translation_preview") as apply_preview,
        ):
            result = gemini_translate.main(["--apply", "C:/run/manifest.json"])

        self.assertEqual(result, 0)
        apply_preview.assert_called_once_with("C:/run/manifest.json")
        run_preview.assert_not_called()

    def test_prepare_is_explicit_preview_option(self):
        with (
            mock.patch.object(gemini_translate.runtime, "initialize_runtime_logging"),
            mock.patch.object(gemini_translate.runtime, "run_translation") as run_preview,
        ):
            gemini_translate.main(["--prepare"])

        run_preview.assert_called_once_with(prepare=True)


if __name__ == "__main__":
    unittest.main()
