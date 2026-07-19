import io
import unittest
from unittest import mock

import translator_runtime as runtime


class LiteLLMRuntimeIntegrationTests(unittest.TestCase):
    def test_sync_runner_does_not_require_gemini_key_for_litellm(self):
        previous_backend = runtime.SYNC_BACKEND
        previous_keys = runtime.API_KEYS

        def select_litellm():
            runtime.SYNC_BACKEND = "litellm"

        try:
            runtime.API_KEYS = []
            with (
                mock.patch.object(runtime, "load_config") as load_config,
                mock.patch.object(
                    runtime,
                    "load_translator_settings",
                    side_effect=select_litellm,
                ),
                mock.patch.object(runtime, "load_glossary"),
                mock.patch.object(runtime, "run_prepare_steps"),
                mock.patch.object(runtime.os, "walk", return_value=[]),
                mock.patch.object(runtime, "load_progress", return_value={}),
                mock.patch.object(
                    runtime.sync_translation_preview,
                    "create_sync_preview",
                    return_value=("manifest.json", {"summary": {}}),
                ),
                mock.patch("sys.stdout", output := io.StringIO()),
            ):
                runtime.run_translation()

            load_config.assert_called_once_with(require_api_key=False)
            self.assertIn("Sync backend: litellm", output.getvalue())
            self.assertIn("not required for LiteLLM", output.getvalue())
        finally:
            runtime.SYNC_BACKEND = previous_backend
            runtime.API_KEYS = previous_keys

    def test_sync_runner_still_requires_gemini_key_for_gemini(self):
        previous_backend = runtime.SYNC_BACKEND
        previous_keys = runtime.API_KEYS

        def select_gemini():
            runtime.SYNC_BACKEND = "gemini"

        try:
            runtime.API_KEYS = []
            with (
                mock.patch.object(runtime, "load_config"),
                mock.patch.object(
                    runtime,
                    "load_translator_settings",
                    side_effect=select_gemini,
                ),
                mock.patch("sys.stdout"),
            ):
                with self.assertRaisesRegex(SystemExit, "No API keys available"):
                    runtime.run_translation()
        finally:
            runtime.SYNC_BACKEND = previous_backend
            runtime.API_KEYS = previous_keys

    def test_runtime_uses_explicit_litellm_backend_without_gemini_client(self):
        fake_result = type("Result", (), {
            "parsed": None,
            "response_text": '[{"id":"a","translation":"你好"}]',
            "response_payload": {},
            "finish_reason": "stop",
        })()
        fake_backend = mock.Mock()
        fake_backend.generate.return_value = fake_result
        items = [{"id": "a", "text": "Hello"}]

        with (
            mock.patch.object(runtime, "SYNC_BACKEND", "litellm"),
            mock.patch.object(runtime, "get_current_model", return_value="openai/test"),
            mock.patch.object(runtime, "create_genai_client") as create_gemini,
            mock.patch("litellm_sync_backend.LiteLLMSyncBackend", return_value=fake_backend),
        ):
            result = runtime.call_gemini_sdk("prompt", items)

        create_gemini.assert_not_called()
        request = fake_backend.generate.call_args.args[0]
        self.assertEqual(request.model, "openai/test")
        self.assertEqual(result, [{"id": "a", "translation": "你好"}])

    def test_sync_settings_default_to_gemini_and_accept_litellm(self):
        previous = runtime.SYNC_BACKEND
        try:
            runtime.load_sync_translation_settings({"sync": {}})
            self.assertEqual(runtime.SYNC_BACKEND, "gemini")
            runtime.load_sync_translation_settings({
                "sync": {"backend": "litellm", "model": "openai/test"}
            })
            self.assertEqual(runtime.SYNC_BACKEND, "litellm")
            self.assertEqual(runtime.MODELS, ["openai/test"])
        finally:
            runtime.SYNC_BACKEND = previous

    def test_unknown_backend_fails_before_model_call(self):
        with self.assertRaises(ValueError) as captured:
            runtime.load_sync_translation_settings({"sync": {"backend": "automatic"}})
        self.assertIn("Unsupported sync backend", str(captured.exception))


if __name__ == "__main__":
    unittest.main()
