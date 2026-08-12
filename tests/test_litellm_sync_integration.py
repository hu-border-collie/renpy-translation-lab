import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch_mod


class LiteLLMSyncIntegrationTests(unittest.TestCase):
    @staticmethod
    def _fake_result(provider="litellm", model="openai/test"):
        return type("Result", (), {
            "response_payload": {"choices": []},
            "response_text": "[]",
            "finish_reason": "stop",
            "usage_metadata": {"total_tokens": 3},
            "provider": provider,
            "model": model,
            "execution_mode": "sync",
        })()

    def test_sync_runner_uses_litellm_only_when_explicitly_selected(self):
        fake_result = self._fake_result()
        fake_backend = mock.Mock()
        fake_backend.generate.return_value = fake_result

        with (
            mock.patch.object(batch_mod, "SYNC_BACKEND", "litellm"),
            mock.patch.object(batch_mod, "SYNC_MODEL", "openai/test"),
            mock.patch.object(batch_mod, "SYNC_TIMEOUT_SECONDS", 45),
            mock.patch.object(batch_mod, "create_batch_client") as create_client,
            mock.patch("litellm_sync_backend.LiteLLMSyncBackend", return_value=fake_backend),
        ):
            result = batch_mod.run_sync_request(
                {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
                model_name="gemini-default",
            )

        create_client.assert_not_called()
        request = fake_backend.generate.call_args.args[0]
        self.assertEqual(request.model, "openai/test")
        self.assertEqual(request.config["timeout"], 45)
        self.assertEqual(result["provider"], "litellm")
        self.assertEqual(result["execution_mode"], "sync")

    def test_sync_runner_passes_timeout_to_gemini_backend(self):
        fake_backend = mock.Mock()
        fake_backend.generate.return_value = self._fake_result(
            provider="gemini",
            model="gemini-test",
        )
        with (
            mock.patch.object(batch_mod, "SYNC_BACKEND", "gemini"),
            mock.patch.object(batch_mod, "SYNC_MODEL", "gemini-test"),
            mock.patch.object(batch_mod, "SYNC_TIMEOUT_SECONDS", 45),
            mock.patch.object(batch_mod, "create_batch_client"),
            mock.patch.object(
                batch_mod,
                "GeminiSyncBackend",
                return_value=fake_backend,
            ),
        ):
            batch_mod.run_sync_request({"contents": []}, "gemini-default")

        request = fake_backend.generate.call_args.args[0]
        self.assertEqual(request.config["timeout"], 45)

    def test_litellm_rejects_gemini_api_key_index(self):
        with (
            mock.patch.object(batch_mod, "SYNC_BACKEND", "litellm"),
            mock.patch.object(batch_mod, "SYNC_MODEL", "openai/test"),
        ):
            with self.assertRaises(SystemExit) as captured:
                batch_mod.run_sync_request({}, "gemini-default", api_key_index=0)
        self.assertIn("only supported by the Gemini", str(captured.exception))

    def test_sync_manifest_records_effective_request_timeout(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            batch_mod,
            "SYNC_TIMEOUT_SECONDS",
            45,
        ):
            manifest_path = batch_mod.make_sync_manifest(
                package_dir=tmp,
                mode=batch_mod.MANIFEST_MODE_TRANSLATION,
                display_name="timeout-test",
                chunks=[],
                request_rows=[],
                settings={"max_output_tokens": 64},
            )

            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

        self.assertEqual(manifest["settings"]["timeout_seconds"], 45)


if __name__ == "__main__":
    unittest.main()
