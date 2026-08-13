import asyncio
import builtins
import os
import unittest
from unittest import mock

from litellm_sync_backend import (
    LiteLLMBackendError,
    LiteLLMCapabilityError,
    LiteLLMSyncBackend,
    LiteLLMUnavailableError,
)
from litellm_provider_config import custom_provider_registry
from sync_model_backend import SyncGenerationRequest, SyncModelBackend


class LiteLLMSyncBackendTests(unittest.TestCase):
    def _custom_registry(self):
        return custom_provider_registry(
            [
                {
                    "id": "opencode-go",
                    "label": "OpenCode Go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                    "api_key_env": "OPENCODE_GO_API_KEY",
                }
            ]
        )

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
                "timeout": 12,
                "response_json_schema": {"type": "array"},
            },
        ))

        self.assertEqual(calls[0]["model"], "openai/gpt-test")
        self.assertEqual(calls[0]["messages"], [{"role": "user", "content": "Translate this"}])
        self.assertEqual(calls[0]["max_tokens"], 100)
        self.assertEqual(calls[0]["timeout"], 12)
        self.assertEqual(calls[0]["response_format"]["type"], "json_schema")
        self.assertEqual(result.provider, "litellm")
        self.assertEqual(result.execution_mode, "sync")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage_metadata["prompt_tokens"], 8)

    def test_async_success_uses_acompletion(self):
        calls = []

        async def completion(**kwargs):
            calls.append(kwargs)
            return {
                "choices": [{
                    "message": {"content": "OK"},
                    "finish_reason": "stop",
                }],
                "usage": {"total_tokens": 3},
            }

        backend = LiteLLMSyncBackend(async_completion=completion)
        result = asyncio.run(
            backend.generate_async(
                SyncGenerationRequest(
                    "openai/test",
                    "hello",
                    {
                        "timeout": 7,
                        "temperature": 0.3,
                        "max_output_tokens": 42,
                        "response_json_schema": {"type": "object"},
                    },
                )
            )
        )

        self.assertEqual(calls[0]["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(calls[0]["timeout"], 7)
        self.assertEqual(calls[0]["temperature"], 0.3)
        self.assertEqual(calls[0]["max_tokens"], 42)
        self.assertEqual(calls[0]["response_format"]["type"], "json_schema")
        self.assertEqual(result.response_text, "OK")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage_metadata["total_tokens"], 3)

    def test_new_gemini_model_omits_sampling_but_keeps_other_config(self):
        calls = []

        def completion(**kwargs):
            calls.append(kwargs)
            return {
                "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                "usage": {},
            }

        backend = LiteLLMSyncBackend(completion=completion)
        backend.generate(SyncGenerationRequest(
            model="gemini/gemini-3.6-flash",
            contents="hello",
            config={"temperature": 0.2, "max_output_tokens": 42},
        ))

        self.assertNotIn("temperature", calls[0])
        self.assertEqual(calls[0]["max_tokens"], 42)

    def test_preserves_provider_reported_cost_from_hidden_params(self):
        class Response:
            _hidden_params = {
                "response_cost": 0.00125,
                "response_cost_currency": "USD",
            }

            def model_dump(self):
                return {
                    "choices": [{
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 6},
                }

        backend = LiteLLMSyncBackend(completion=lambda **kwargs: Response())
        result = backend.generate(
            SyncGenerationRequest("openai/test", "hello")
        )

        self.assertEqual(
            result.response_payload["_hidden_params"]["response_cost"], 0.00125
        )

    def test_uses_json_object_for_deepseek(self):
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
        backend.generate(SyncGenerationRequest(
            model="deepseek/deepseek-v4-flash",
            contents="Translate this",
            config={
                "response_json_schema": {"type": "array"},
            },
        ))

        self.assertEqual(calls[0]["model"], "deepseek/deepseek-v4-flash")
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})

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

    def test_explicit_api_key_is_passed_without_entering_config_payload(self):
        calls = []
        backend = LiteLLMSyncBackend(
            completion=lambda **kwargs: calls.append(kwargs) or {"choices": []},
            api_key="secret-test-key",
        )

        backend.generate(SyncGenerationRequest("openai/test", "hello"))

        self.assertEqual(calls[0]["api_key"], "secret-test-key")

    def test_saved_provider_key_is_passed_to_litellm(self):
        calls = []
        backend = LiteLLMSyncBackend(
            completion=lambda **kwargs: calls.append(kwargs) or {"choices": []}
        )

        with mock.patch(
            "litellm_provider_config.load_provider_api_key",
            return_value="stored-secret",
        ) as load_key:
            backend.generate(SyncGenerationRequest("anthropic/test", "hello"))

        load_key.assert_called_once_with("anthropic")
        self.assertEqual(calls[0]["api_key"], "stored-secret")

    def test_missing_keyring_leaves_auth_to_litellm_environment(self):
        calls = []
        backend = LiteLLMSyncBackend(
            completion=lambda **kwargs: calls.append(kwargs) or {"choices": []}
        )
        with mock.patch(
            "litellm_provider_config.load_provider_api_key",
            side_effect=RuntimeError("no keyring"),
        ):
            backend.generate(SyncGenerationRequest("openai/test", "hello"))
        self.assertNotIn("api_key", calls[0])

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
            status_code = 429

        backend = LiteLLMSyncBackend(completion=lambda **kwargs: (_ for _ in ()).throw(RateLimitError()))
        with self.assertRaises(LiteLLMBackendError) as captured:
            backend.generate(SyncGenerationRequest("openai/test", "hello"))
        self.assertEqual(captured.exception.category, "rate_limit")

    def test_custom_provider_rewrites_model_and_passes_api_base(self):
        calls = []
        backend = LiteLLMSyncBackend(
            completion=lambda **kwargs: calls.append(kwargs) or {"choices": []},
            custom_providers=self._custom_registry(),
        )
        with mock.patch(
            "litellm_provider_config.load_provider_api_key",
            return_value="stored-custom-key",
        ):
            backend.generate(SyncGenerationRequest(
                "opencode-go/gpt-4o-mini",
                "Translate this",
                {"response_json_schema": {"type": "array"}},
            ))

        self.assertEqual(calls[0]["model"], "openai/gpt-4o-mini")
        self.assertEqual(calls[0]["api_base"], "https://opencode.ai/zen/go/v1")
        self.assertEqual(calls[0]["api_key"], "stored-custom-key")
        # Capability decision uses the original custom id → json_object.
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})

    def test_unknown_provider_uses_prompt_only_json_capability(self):
        calls = []
        backend = LiteLLMSyncBackend(
            completion=lambda **kwargs: calls.append(kwargs) or {"choices": []},
        )
        backend.generate(SyncGenerationRequest(
            "future-provider/model-x",
            "Translate this",
            {"response_json_schema": {"type": "object"}},
        ))

        self.assertNotIn("response_format", calls[0])

    def test_schema_name_comes_from_named_envelope(self):
        calls = []
        backend = LiteLLMSyncBackend(
            completion=lambda **kwargs: calls.append(kwargs) or {"choices": []},
        )
        backend.generate(SyncGenerationRequest(
            "openai/gpt-test",
            "Translate this",
            {
                "response_json_schema": {
                    "type": "object",
                    "properties": {"revisions": {"type": "array"}},
                }
            },
        ))

        schema = calls[0]["response_format"]["json_schema"]
        self.assertEqual(schema["name"], "revisions_response")

    def test_custom_provider_env_key_fallback_when_keyring_empty(self):
        calls = []
        backend = LiteLLMSyncBackend(
            completion=lambda **kwargs: calls.append(kwargs) or {"choices": []},
            custom_providers=self._custom_registry(),
        )
        with (
            mock.patch(
                "litellm_provider_config.load_provider_api_key",
                return_value="",
            ),
            mock.patch.dict(
                os.environ,
                {"OPENCODE_GO_API_KEY": "env-custom-key"},
                clear=True,
            ),
        ):
            backend.generate(SyncGenerationRequest("opencode-go/gpt-4o-mini", "hello"))

        self.assertEqual(calls[0]["model"], "openai/gpt-4o-mini")
        self.assertEqual(calls[0]["api_key"], "env-custom-key")

    def test_custom_provider_async_path_rewrites_identically(self):
        calls = []

        async def completion(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "OK"}}], "usage": {}}

        backend = LiteLLMSyncBackend(
            async_completion=completion,
            custom_providers=self._custom_registry(),
        )
        with mock.patch(
            "litellm_provider_config.load_provider_api_key",
            return_value="stored-custom-key",
        ):
            result = asyncio.run(
                backend.generate_async(
                    SyncGenerationRequest("opencode-go/model-x", "hello")
                )
            )

        self.assertEqual(calls[0]["model"], "openai/model-x")
        self.assertEqual(calls[0]["api_base"], "https://opencode.ai/zen/go/v1")
        self.assertEqual(result.response_text, "OK")

    def test_builtin_provider_ignores_custom_registry(self):
        calls = []
        backend = LiteLLMSyncBackend(
            completion=lambda **kwargs: calls.append(kwargs) or {"choices": []},
            custom_providers=self._custom_registry(),
        )
        with mock.patch(
            "litellm_provider_config.load_provider_api_key",
            return_value="openai-key",
        ):
            backend.generate(SyncGenerationRequest(
                "openai/gpt-test",
                "hello",
                {"response_json_schema": {"type": "array"}},
            ))

        self.assertEqual(calls[0]["model"], "openai/gpt-test")
        self.assertNotIn("api_base", calls[0])
        self.assertEqual(calls[0]["response_format"]["type"], "json_schema")

    def test_custom_provider_blocks_implicit_openai_key_fallback(self):
        """Regression: OPENAI_API_KEY must never reach a third-party api_base."""
        calls = []
        backend = LiteLLMSyncBackend(
            completion=lambda **kwargs: calls.append(kwargs) or {"choices": []},
            custom_providers=self._custom_registry(),
        )
        with (
            mock.patch(
                "litellm_provider_config.load_provider_api_key",
                return_value="",
            ),
            mock.patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "official-openai-key"},
                clear=True,
            ),
        ):
            with self.assertRaises(LiteLLMBackendError) as captured:
                backend.generate(SyncGenerationRequest("opencode-go/gpt-4o-mini", "hello"))

        self.assertEqual(calls, [])
        self.assertEqual(captured.exception.category, "authentication")
        self.assertIn("需要 API Key", str(captured.exception))

    def test_custom_provider_without_key_requirement_skips_key_gate(self):
        calls = []
        registry = custom_provider_registry(
            [
                {
                    "id": "local-vllm",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "requires_key": False,
                }
            ]
        )
        backend = LiteLLMSyncBackend(
            completion=lambda **kwargs: calls.append(kwargs) or {"choices": []},
            custom_providers=registry,
        )
        with (
            mock.patch(
                "litellm_provider_config.load_provider_api_key",
                return_value="",
            ),
            mock.patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "official-openai-key"},
                clear=True,
            ),
        ):
            backend.generate(SyncGenerationRequest("local-vllm/llama-3", "hello"))

        self.assertNotIn("api_key", calls[0])
        self.assertEqual(calls[0]["api_base"], "http://127.0.0.1:8000/v1")
        self.assertEqual(calls[0]["model"], "openai/llama-3")


if __name__ == "__main__":
    unittest.main()
