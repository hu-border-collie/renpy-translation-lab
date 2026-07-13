import builtins
import unittest
from unittest import mock

from litellm_sync_backend import (
    LiteLLMBackendError,
    LiteLLMCapabilityError,
    LiteLLMSyncBackend,
    LiteLLMUnavailableError,
)
from sync_model_backend import SyncGenerationRequest, SyncModelBackend


class LiteLLMSyncBackendTests(unittest.TestCase):
    def test_success_normalizes_response_and_maps_config(self):
        calls = []

        def completion(**kwargs):
            calls.append(kwargs)
            return {
                "choices": [{
                    "message": {"content": '[{"id":"a","translation":"你好"}]'},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 8, "completion_tokens": 6},
            }

        backend = LiteLLMSyncBackend(completion=completion)
        self.assertIsInstance(backend, SyncModelBackend)
        result = backend.generate(SyncGenerationRequest(
            model="openai/gpt-test",
            contents="Translate this",
            config={
                "temperature": 0.2,
                "max_output_tokens": 100,
                "response_json_schema": {"type": "array"},
            },
        ))

        self.assertEqual(calls[0]["model"], "openai/gpt-test")
        self.assertEqual(calls[0]["messages"], [{"role": "user", "content": "Translate this"}])
        self.assertEqual(calls[0]["max_tokens"], 100)
        self.assertEqual(calls[0]["response_format"]["type"], "json_schema")
        self.assertEqual(result.provider, "litellm")
        self.assertEqual(result.execution_mode, "sync")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage_metadata["prompt_tokens"], 8)

    def test_converts_gemini_style_system_instruction_and_contents(self):
        calls = []
        backend = LiteLLMSyncBackend(completion=lambda **kwargs: calls.append(kwargs) or {"choices": []})
        backend.generate(SyncGenerationRequest(
            "anthropic/test",
            [{"role": "user", "parts": [{"text": "hello"}]}],
            {"system_instruction": {"parts": [{"text": "rules"}]}},
        ))
        self.assertEqual(calls[0]["messages"], [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "hello"},
        ])

    def test_missing_dependency_is_reported_only_when_selected(self):
        backend = LiteLLMSyncBackend()
        original_import = builtins.__import__

        def reject_litellm(name, *args, **kwargs):
            if name == "litellm":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=reject_litellm):
            with self.assertRaises(LiteLLMUnavailableError) as captured:
                backend.generate(SyncGenerationRequest("openai/test", "hello"))
        self.assertEqual(captured.exception.category, "missing_dependency")
        self.assertIn("select Gemini Batch", str(captured.exception))

    def test_rejects_gemini_only_safety_settings_before_request(self):
        backend = LiteLLMSyncBackend(completion=lambda **kwargs: {})
        with self.assertRaises(LiteLLMCapabilityError) as captured:
            backend.generate(SyncGenerationRequest(
                "openai/test", "hello", {"safety_settings": [{"category": "x"}]}
            ))
        self.assertEqual(captured.exception.category, "unsupported_capability")

    def test_classifies_rate_limit_and_service_unavailable(self):
        class RateLimitError(Exception):
            pass

        backend = LiteLLMSyncBackend(completion=lambda **kwargs: (_ for _ in ()).throw(RateLimitError()))
        with self.assertRaises(LiteLLMBackendError) as captured:
            backend.generate(SyncGenerationRequest("openai/test", "hello"))
        self.assertEqual(captured.exception.category, "rate_limit")


if __name__ == "__main__":
    unittest.main()
