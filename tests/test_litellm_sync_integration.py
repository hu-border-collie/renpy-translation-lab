import unittest
from unittest import mock

import gemini_translate_batch as batch_mod


class LiteLLMSyncIntegrationTests(unittest.TestCase):
    def test_sync_runner_uses_litellm_only_when_explicitly_selected(self):
        fake_result = type("Result", (), {
            "response_payload": {"choices": []},
            "response_text": "[]",
            "finish_reason": "stop",
            "usage_metadata": {"total_tokens": 3},
            "provider": "litellm",
            "model": "openai/test",
            "execution_mode": "sync",
        })()
        fake_backend = mock.Mock()
        fake_backend.generate.return_value = fake_result

        with (
            mock.patch.object(batch_mod, "SYNC_BACKEND", "litellm"),
            mock.patch.object(batch_mod, "SYNC_MODEL", "openai/test"),
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
        self.assertEqual(result["provider"], "litellm")
        self.assertEqual(result["execution_mode"], "sync")

    def test_litellm_rejects_gemini_api_key_index(self):
        with (
            mock.patch.object(batch_mod, "SYNC_BACKEND", "litellm"),
            mock.patch.object(batch_mod, "SYNC_MODEL", "openai/test"),
        ):
            with self.assertRaises(SystemExit) as captured:
                batch_mod.run_sync_request({}, "gemini-default", api_key_index=0)
        self.assertIn("only supported by the Gemini", str(captured.exception))


if __name__ == "__main__":
    unittest.main()
