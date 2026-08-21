import asyncio
import io
import json
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from PySide6.QtCore import Qt
    from gui_qt.litellm_worker import (
        BudgetExhausted,
        CANCELLED_MESSAGE_PREFIX,
        CATALOG_TIMEOUT_SECONDS,
        CATALOG_TOTAL_BUDGET_SECONDS,
        CONNECTION_TEST_MAX_OUTPUT_TOKENS,
        CONNECTION_TEST_PROMPT,
        CONNECTION_TEST_RESPONSE_SCHEMA,
        CONNECTION_TEST_TIMEOUT_SECONDS,
        LiteLLMConnectionTestWorker,
        LiteLLMModelCatalogWorker,
        LiteLLMModuleWarmupWorker,
        LiteLLMProviderCatalogWorker,
        LiteLLMVersionWorker,
        is_cancelled_message,
    )
    from litellm_sync_backend import LiteLLMBackendError
    from litellm_provider_config import custom_provider_registry
except ImportError as exc:
    LiteLLMConnectionTestWorker = None
    LiteLLMModuleWarmupWorker = None
    custom_provider_registry = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def _record_connection_completed(worker, completed: list) -> None:
    worker.completed.connect(
        lambda success, message, identity="": completed.append(
            (success, message, identity)
        )
    )


@unittest.skipIf(
    LiteLLMConnectionTestWorker is None,
    f"GUI dependencies are unavailable: {IMPORT_ERROR}",
)
class LiteLLMConnectionTestWorkerTests(unittest.TestCase):
    def test_connection_test_passes_a_bounded_timeout(self):
        backend = mock.Mock()
        backend.generate_async = mock.AsyncMock(
            return_value=SimpleNamespace(response_text='{"ok":true}')
        )
        completed = []
        worker = LiteLLMConnectionTestWorker("openai/test", "typed-secret")
        _record_connection_completed(worker, completed)

        with mock.patch(
            "gui_qt.litellm_worker.model_profile.build_sync_backend",
            return_value=backend,
        ):
            worker.run()

        request = backend.generate_async.call_args.args[0]
        self.assertEqual(
            request.config["timeout"],
            CONNECTION_TEST_TIMEOUT_SECONDS,
        )
        self.assertEqual(request.contents, CONNECTION_TEST_PROMPT)
        self.assertEqual(
            request.config["max_output_tokens"],
            CONNECTION_TEST_MAX_OUTPUT_TOKENS,
        )
        self.assertEqual(
            request.config["response_json_schema"],
            CONNECTION_TEST_RESPONSE_SCHEMA,
        )
        self.assertEqual(
            completed,
            [(True, "连接成功。已通过最小 JSON 响应校验。", "")],
        )

    def test_connection_test_echoes_operation_identity(self):
        backend = mock.Mock()
        backend.generate_async = mock.AsyncMock(
            return_value=SimpleNamespace(response_text='{"ok":true}')
        )
        completed = []
        worker = LiteLLMConnectionTestWorker(
            "openai/test",
            operation_identity="digest-abc",
        )
        _record_connection_completed(worker, completed)

        with mock.patch(
            "gui_qt.litellm_worker.model_profile.build_sync_backend",
            return_value=backend,
        ):
            worker.run()

        self.assertEqual(completed[0][0], True)
        self.assertEqual(completed[0][2], "digest-abc")

    def test_connection_test_omits_sampling_for_new_gemini_model(self):
        backend = mock.Mock()
        backend.generate_async = mock.AsyncMock(
            return_value=SimpleNamespace(response_text='{"ok":true}')
        )
        worker = LiteLLMConnectionTestWorker("gemini/gemini-3.6-flash")

        with mock.patch(
            "gui_qt.litellm_worker.model_profile.build_sync_backend",
            return_value=backend,
        ):
            worker.run()

        request = backend.generate_async.call_args.args[0]
        self.assertNotIn("temperature", request.config)
        self.assertEqual(
            request.config["max_output_tokens"],
            CONNECTION_TEST_MAX_OUTPUT_TOKENS,
        )

    def test_connection_test_rejects_empty_invalid_or_mismatched_json(self):
        responses = (
            "",
            "not json",
            '{"ok":false}',
            '{"ok":1}',
            '{"ok":true,"detail":"provider-secret"}',
        )
        for response_text in responses:
            with self.subTest(response_text=response_text):
                backend = mock.Mock()
                backend.generate_async = mock.AsyncMock(
                    return_value=SimpleNamespace(response_text=response_text)
                )
                completed = []
                worker = LiteLLMConnectionTestWorker("openai/test")
                _record_connection_completed(worker, completed)

                with mock.patch(
                    "gui_qt.litellm_worker.model_profile.build_sync_backend",
                    return_value=backend,
                ):
                    worker.run()

                self.assertEqual(len(completed), 1)
                self.assertFalse(completed[0][0])
                self.assertIn("invalid_response", completed[0][1])
                self.assertNotIn("provider-secret", completed[0][1])

    def test_connection_test_prefers_parsed_response_when_available(self):
        cases = (
            (
                SimpleNamespace(parsed={"ok": True}, response_text="not json"),
                True,
            ),
            (
                SimpleNamespace(
                    parsed={"ok": False, "detail": "provider-secret"},
                    response_text='{"ok":true}',
                ),
                False,
            ),
        )
        for result, expected_success in cases:
            with self.subTest(parsed=result.parsed):
                backend = mock.Mock()
                backend.generate_async = mock.AsyncMock(return_value=result)
                completed = []
                worker = LiteLLMConnectionTestWorker("openai/test")
                _record_connection_completed(worker, completed)

                with mock.patch(
                    "gui_qt.litellm_worker.model_profile.build_sync_backend",
                    return_value=backend,
                ):
                    worker.run()

                self.assertEqual(len(completed), 1)
                self.assertEqual(completed[0][0], expected_success)
                self.assertNotIn("provider-secret", completed[0][1])
                if not expected_success:
                    self.assertIn("invalid_response", completed[0][1])

    def test_connection_test_preflight_reports_missing_provider_prefix(self):
        completed = []
        worker = LiteLLMConnectionTestWorker("gpt-4o-mini")
        _record_connection_completed(worker, completed)

        with mock.patch(
            "gui_qt.litellm_worker.model_profile.build_sync_backend",
        ) as build_backend:
            worker.run()

        build_backend.assert_not_called()
        self.assertEqual(len(completed), 1)
        self.assertFalse(completed[0][0])
        message = completed[0][1]
        self.assertIn("MODEL_PROFILE_INVALID", message)
        self.assertIn("<provider>/<model>", message)
        self.assertNotIn("JSON 能力", message)
        self.assertNotIn("unsupported_capability", message)

    def test_connection_error_never_includes_provider_exception_text(self):
        backend = mock.Mock()
        backend.generate_async.side_effect = LiteLLMBackendError(
            "provider echoed stored-secret",
            category="authentication",
        )
        completed = []
        worker = LiteLLMConnectionTestWorker("openai/test")
        _record_connection_completed(worker, completed)

        with mock.patch(
            "gui_qt.litellm_worker.model_profile.build_sync_backend",
            return_value=backend,
        ):
            worker.run()

        self.assertFalse(completed[0][0])
        self.assertIn("authentication", completed[0][1])
        self.assertNotIn("stored-secret", completed[0][1])
        self.assertNotIn("provider echoed", completed[0][1])

    def test_connection_test_cancel_before_generate_emits_cancelled(self):
        backend = mock.Mock()
        completed = []
        worker = LiteLLMConnectionTestWorker("openai/test")
        _record_connection_completed(worker, completed)
        worker.request_cancel()

        with mock.patch(
            "gui_qt.litellm_worker.model_profile.build_sync_backend",
            return_value=backend,
        ):
            worker.run()

        backend.generate_async.assert_not_called()
        self.assertEqual(len(completed), 1)
        self.assertFalse(completed[0][0])
        self.assertTrue(is_cancelled_message(completed[0][1]))
        self.assertTrue(completed[0][1].startswith(CANCELLED_MESSAGE_PREFIX))

    def test_connection_test_cancel_after_generate_discards_success(self):
        backend = mock.Mock()
        backend.generate_async = mock.AsyncMock(
            return_value=SimpleNamespace(response_text='{"ok":true}')
        )
        completed = []
        worker = LiteLLMConnectionTestWorker("openai/test")
        _record_connection_completed(worker, completed)

        def generate_and_cancel(request):
            worker.request_cancel()
            return SimpleNamespace(response_text='{"ok":true}')

        backend.generate_async.side_effect = generate_and_cancel
        with mock.patch(
            "gui_qt.litellm_worker.model_profile.build_sync_backend",
            return_value=backend,
        ):
            worker.run()

        self.assertEqual(len(completed), 1)
        self.assertFalse(completed[0][0])
        self.assertTrue(is_cancelled_message(completed[0][1]))


    def test_connection_test_cancel_during_async_request_cancels_task(self):
        started = threading.Event()
        cancelled = threading.Event()
        backend = mock.Mock()

        async def generate_async(request):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        backend.generate_async = generate_async
        completed = []
        worker = LiteLLMConnectionTestWorker("openai/test")
        worker.completed.connect(
            lambda success, message, identity="": completed.append(
                (success, message, identity)
            ),
            Qt.ConnectionType.DirectConnection,
        )

        with mock.patch(
            "gui_qt.litellm_worker.model_profile.build_sync_backend",
            return_value=backend,
        ):
            thread = threading.Thread(target=worker.run)
            thread.start()
            self.assertTrue(started.wait(2))
            worker.request_cancel()
            thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertTrue(cancelled.is_set())
        self.assertEqual(len(completed), 1)
        self.assertFalse(completed[0][0])
        self.assertTrue(is_cancelled_message(completed[0][1]))

    def test_model_catalog_prefers_provider_native_catalog(self):
        payload = {"data": [{"id": "gpt-current"}, {"id": "text-embedding-3-small"}]}
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(json.dumps(payload).encode("utf-8"))
        completed = []
        worker = LiteLLMModelCatalogWorker("openai", api_key="sk-test")
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )

        with mock.patch("gui_qt.litellm_worker.urlopen", return_value=response):
            worker.run()

        self.assertEqual(completed, [(("openai/gpt-current",), "openai", None)])

    def test_openrouter_catalog_uses_official_models_endpoint(self):
        payload = {
            "data": [
                {
                    "id": "openai/gpt-5",
                    "architecture": {"output_modalities": ["text"]},
                },
                {
                    "id": "anthropic/claude-sonnet",
                    "architecture": {"output_modalities": ["text"]},
                },
            ]
        }
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(json.dumps(payload).encode("utf-8"))
        completed = []
        worker = LiteLLMModelCatalogWorker("openrouter", api_key="or-secret")
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )

        with mock.patch("gui_qt.litellm_worker.urlopen", return_value=response) as urlopen:
            worker.run()

        request = urlopen.call_args.args[0]
        self.assertIn("openrouter.ai/api/v1/models", request.full_url)
        self.assertEqual(request.get_header("Authorization"), "Bearer or-secret")
        self.assertEqual(completed[0][1], "openrouter")
        self.assertIsNone(completed[0][2])
        self.assertEqual(
            completed[0][0],
            (
                "openrouter/anthropic/claude-sonnet",
                "openrouter/openai/gpt-5",
            ),
        )

    def test_openai_catalog_requires_key_then_uses_official_endpoint(self):
        completed = []
        worker = LiteLLMModelCatalogWorker("openai", api_key="")
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )
        litellm_payload = {
            "gpt-from-litellm": {"litellm_provider": "openai", "mode": "chat"},
        }
        litellm_response = mock.MagicMock()
        litellm_response.__enter__.return_value = io.BytesIO(
            json.dumps(litellm_payload).encode("utf-8")
        )

        with mock.patch("gui_qt.litellm_worker.urlopen", return_value=litellm_response):
            worker.run()

        self.assertEqual(completed[0][1], "online")
        self.assertIn("请先保存 OpenAI API Key", completed[0][2])
        self.assertEqual(completed[0][0], ("openai/gpt-from-litellm",))

        completed.clear()
        official_payload = {"data": [{"id": "gpt-5"}, {"id": "text-embedding-3-large"}]}
        official_response = mock.MagicMock()
        official_response.__enter__.return_value = io.BytesIO(
            json.dumps(official_payload).encode("utf-8")
        )
        worker = LiteLLMModelCatalogWorker("openai", api_key="sk-test")
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )
        with mock.patch(
            "gui_qt.litellm_worker.urlopen", return_value=official_response
        ) as urlopen:
            worker.run()
        request = urlopen.call_args.args[0]
        self.assertIn("api.openai.com/v1/models", request.full_url)
        self.assertEqual(request.get_header("Authorization"), "Bearer sk-test")
        self.assertEqual(completed[0], (("openai/gpt-5",), "openai", None))

    def test_anthropic_catalog_sends_version_header(self):
        payload = {"data": [{"id": "claude-sonnet-4-5-20250929"}]}
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(json.dumps(payload).encode("utf-8"))
        completed = []
        worker = LiteLLMModelCatalogWorker("anthropic", api_key="ant-key")
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )
        with mock.patch("gui_qt.litellm_worker.urlopen", return_value=response) as urlopen:
            worker.run()
        request = urlopen.call_args.args[0]
        self.assertIn("api.anthropic.com/v1/models", request.full_url)
        self.assertEqual(request.get_header("X-api-key"), "ant-key")
        self.assertEqual(request.get_header("Anthropic-version"), "2023-06-01")
        self.assertEqual(
            completed[0],
            (("anthropic/claude-sonnet-4-5-20250929",), "anthropic", None),
        )

    def test_ollama_catalog_reads_local_tags(self):
        payload = {"models": [{"name": "llama3:latest"}]}
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(json.dumps(payload).encode("utf-8"))
        completed = []
        worker = LiteLLMModelCatalogWorker("ollama")
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )
        with mock.patch("gui_qt.litellm_worker.urlopen", return_value=response) as urlopen:
            worker.run()
        self.assertIn("127.0.0.1:11434/api/tags", urlopen.call_args.args[0].full_url)
        self.assertEqual(completed[0], (("ollama/llama3:latest",), "ollama", None))

    def test_custom_provider_catalog_uses_configured_models_url(self):
        registry = custom_provider_registry(
            [
                {
                    "id": "opencode-go",
                    "label": "OpenCode Go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                }
            ]
        )
        payload = {
            "data": [
                {"id": "gpt-4o-mini"},
                {"id": "text-embedding-3-small"},
            ]
        }
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(
            json.dumps(payload).encode("utf-8")
        )
        completed = []
        worker = LiteLLMModelCatalogWorker(
            "opencode-go",
            api_key="custom-secret",
            custom_providers=registry,
        )
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )

        with mock.patch("gui_qt.litellm_worker.urlopen", return_value=response) as urlopen:
            worker.run()

        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://opencode.ai/zen/go/v1/models",
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer custom-secret")
        self.assertEqual(
            completed,
            [(("opencode-go/gpt-4o-mini",), "opencode-go", None)],
        )

    def test_custom_provider_catalog_requires_saved_key(self):
        registry = custom_provider_registry(
            [
                {
                    "id": "opencode-go",
                    "label": "OpenCode Go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                }
            ]
        )
        completed = []
        worker = LiteLLMModelCatalogWorker(
            "opencode-go",
            api_key="",
            custom_providers=registry,
        )
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )

        with mock.patch("gui_qt.litellm_worker.urlopen") as urlopen:
            worker.run()

        urlopen.assert_not_called()
        self.assertEqual(completed[0][0], ())
        self.assertIn("请先保存 OpenCode Go API Key", completed[0][2])

    def test_custom_provider_catalog_uses_api_key_env_when_keyring_empty(self):
        registry = custom_provider_registry(
            [
                {
                    "id": "opencode-go",
                    "label": "OpenCode Go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                    "api_key_env": "OPENCODE_GO_API_KEY",
                }
            ]
        )
        payload = {"data": [{"id": "gpt-4o-mini"}]}
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(
            json.dumps(payload).encode("utf-8")
        )
        completed = []
        worker = LiteLLMModelCatalogWorker(
            "opencode-go",
            api_key="",
            custom_providers=registry,
        )
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )

        with (
            mock.patch("gui_qt.litellm_worker.urlopen", return_value=response) as urlopen,
            mock.patch.dict(
                "gui_qt.litellm_worker.os.environ",
                {"OPENCODE_GO_API_KEY": "env-custom-key"},
                clear=True,
            ),
        ):
            worker.run()

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer env-custom-key")
        self.assertEqual(
            completed,
            [(("opencode-go/gpt-4o-mini",), "opencode-go", None)],
        )

    def test_keyless_custom_provider_catalog_does_not_require_key(self):
        registry = custom_provider_registry(
            [
                {
                    "id": "local-vllm",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "requires_key": False,
                }
            ]
        )
        payload = {"data": [{"id": "llama-3"}]}
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(
            json.dumps(payload).encode("utf-8")
        )
        completed = []
        worker = LiteLLMModelCatalogWorker(
            "local-vllm",
            api_key="",
            custom_providers=registry,
        )
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )

        with mock.patch("gui_qt.litellm_worker.urlopen", return_value=response) as urlopen:
            worker.run()

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), None)
        self.assertEqual(completed, [(("local-vllm/llama-3",), "local-vllm", None)])

    def test_connection_test_forwards_custom_providers_to_backend(self):
        registry = custom_provider_registry(
            [
                {
                    "id": "opencode-go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                    "api_key_env": "OPENCODE_GO_API_KEY",
                }
            ]
        )
        backend = mock.Mock()
        backend.generate_async = mock.AsyncMock(
            return_value=SimpleNamespace(response_text='{"ok":true}')
        )
        completed = []
        worker = LiteLLMConnectionTestWorker(
            "opencode-go/gpt-4o-mini",
            "custom-secret",
            custom_providers=registry,
        )
        _record_connection_completed(worker, completed)

        with mock.patch(
            "gui_qt.litellm_worker.model_profile.build_sync_backend",
            return_value=backend,
        ) as backend_cls:
            worker.run()

        self.assertEqual(backend_cls.call_args.kwargs["custom_providers"], registry)
        self.assertEqual(
            backend_cls.call_args.kwargs["diagnostic_api_key"], "custom-secret",
        )
        self.assertEqual(backend_cls.call_args.args[0].adapter, "litellm")
        request = backend.generate_async.call_args.args[0]
        self.assertEqual(request.model, "opencode-go/gpt-4o-mini")
        self.assertEqual(
            completed,
            [(True, "连接成功。已通过最小 JSON 响应校验。", "")],
        )

    def test_openrouter_falls_back_to_litellm_subset_then_local(self):
        litellm_payload = {
            "openrouter/openai/gpt-subset": {
                "litellm_provider": "openrouter",
                "mode": "chat",
            }
        }
        litellm_response = mock.MagicMock()
        litellm_response.__enter__.return_value = io.BytesIO(
            json.dumps(litellm_payload).encode("utf-8")
        )
        completed = []
        worker = LiteLLMModelCatalogWorker("openrouter")
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )

        def fake_urlopen(request, timeout=0):
            url = getattr(request, "full_url", str(request))
            if "openrouter.ai" in url:
                raise OSError("openrouter down")
            return litellm_response

        with mock.patch("gui_qt.litellm_worker.urlopen", side_effect=fake_urlopen):
            worker.run()

        self.assertEqual(completed[0][0], ("openrouter/openai/gpt-subset",))
        self.assertEqual(completed[0][1], "online")
        self.assertIn("OpenRouter 官方列表失败", completed[0][2])

    def test_model_catalog_failure_does_not_inject_local_defaults(self):
        completed = []
        worker = LiteLLMModelCatalogWorker("openai")
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )

        with mock.patch(
            "gui_qt.litellm_worker.urlopen",
            side_effect=OSError("offline"),
        ):
            worker.run()

        self.assertEqual(completed[0][0], ())
        self.assertEqual(completed[0][1], "")
        self.assertIn("联网加载模型失败", completed[0][2])

    def test_provider_catalog_is_loaded_only_when_worker_runs(self):
        payload = {
            "gpt-current": {"litellm_provider": "openai", "mode": "chat"},
            "claude-current": {"litellm_provider": "anthropic", "mode": "chat"},
        }
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(
            json.dumps(payload).encode("utf-8")
        )
        completed = []
        worker = LiteLLMProviderCatalogWorker()
        worker.completed.connect(
            lambda providers, source, error: completed.append((providers, source, error))
        )
        with mock.patch("gui_qt.litellm_worker.urlopen", return_value=response):
            worker.run()
        self.assertIn("openai", completed[0][0])
        self.assertIn("anthropic", completed[0][0])
        self.assertEqual(completed[0][1:], ("online", None))

    def test_provider_catalog_failure_reports_empty_result(self):
        completed = []
        worker = LiteLLMProviderCatalogWorker()
        worker.completed.connect(
            lambda providers, source, error: completed.append(
                (providers, source, error)
            )
        )
        with mock.patch(
            "gui_qt.litellm_worker.urlopen",
            side_effect=OSError("offline"),
        ):
            worker.run()
        self.assertEqual(completed[0][0], ())
        self.assertEqual(completed[0][1], "")
        self.assertIn("联网加载供应商失败", completed[0][2])

    def test_provider_catalog_cancel_before_fetch_emits_cancelled(self):
        completed = []
        worker = LiteLLMProviderCatalogWorker()
        worker.completed.connect(
            lambda providers, source, error: completed.append(
                (providers, source, error)
            )
        )
        worker.request_cancel()
        with mock.patch("gui_qt.litellm_worker.urlopen") as urlopen:
            worker.run()
        urlopen.assert_not_called()
        self.assertEqual(completed[0][0], ())
        self.assertEqual(completed[0][1], "")
        self.assertTrue(is_cancelled_message(completed[0][2]))

    def test_model_catalog_cancel_skips_fallback_after_native_failure(self):
        completed = []
        worker = LiteLLMModelCatalogWorker("openai", api_key="sk-test")
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )

        def fail_then_would_fallback(request, timeout=0):
            worker.request_cancel()
            raise OSError("native down")

        with mock.patch(
            "gui_qt.litellm_worker.urlopen",
            side_effect=fail_then_would_fallback,
        ) as urlopen:
            worker.run()

        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(completed[0][0], ())
        self.assertTrue(is_cancelled_message(completed[0][2]))

    def test_model_catalog_request_cancel_closes_active_response(self):
        worker = LiteLLMModelCatalogWorker("openai", api_key="sk-test")
        response = mock.MagicMock()
        worker._active_response = response
        worker.request_cancel()
        self.assertTrue(worker.is_cancelled())
        response.close.assert_called_once()

    def test_model_catalog_emits_progress_then_falls_back_within_budget(self):
        import time

        litellm_payload = {
            "openai/gpt-subset": {"litellm_provider": "openai", "mode": "chat"},
        }
        litellm_response = mock.MagicMock()
        litellm_response.__enter__.return_value = io.BytesIO(
            json.dumps(litellm_payload).encode("utf-8")
        )
        completed = []
        progress = []
        timeouts = []
        worker = LiteLLMModelCatalogWorker("openai", api_key="sk-test")
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )
        worker.progress.connect(progress.append)

        def fake_urlopen(request, timeout=0):
            timeouts.append(float(timeout))
            url = getattr(request, "full_url", str(request))
            if "api.openai.com" in url:
                # Simulate the first hop consuming most of the shared budget.
                worker._deadline = time.monotonic() + (
                    CATALOG_TOTAL_BUDGET_SECONDS - 18.0
                )
                raise OSError("openai down")
            return litellm_response

        with mock.patch("gui_qt.litellm_worker.urlopen", side_effect=fake_urlopen):
            worker.run()

        self.assertEqual(completed[0][0], ("openai/gpt-subset",))
        self.assertEqual(completed[0][1], "online")
        self.assertIn("官方列表失败", completed[0][2])
        self.assertTrue(any("官方模型列表" in item for item in progress))
        self.assertTrue(any("改用 LiteLLM 子集" in item for item in progress))
        self.assertEqual(len(timeouts), 2)
        self.assertLessEqual(timeouts[0], CATALOG_TIMEOUT_SECONDS)
        # Fallback hop must use remaining budget (~17s), not a fresh full cap.
        self.assertLess(timeouts[1], CATALOG_TIMEOUT_SECONDS)
        self.assertGreater(timeouts[1], 10.0)
        self.assertAlmostEqual(timeouts[1], CATALOG_TOTAL_BUDGET_SECONDS - 18.0, delta=1.0)

    def test_model_catalog_skips_fallback_when_budget_already_exhausted(self):
        completed = []
        progress = []
        worker = LiteLLMModelCatalogWorker("openai", api_key="sk-test")
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )
        worker.progress.connect(progress.append)

        def slow_then_fail(request, timeout=0):
            # Consume the whole shared budget inside the first hop.
            worker._deadline = worker._deadline - CATALOG_TOTAL_BUDGET_SECONDS
            raise OSError("native timed out")

        with mock.patch("gui_qt.litellm_worker.urlopen", side_effect=slow_then_fail) as urlopen:
            worker.run()

        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(completed[0][0], ())
        self.assertIn("总时限", completed[0][2])
        self.assertIn(f"{CATALOG_TOTAL_BUDGET_SECONDS:g}", completed[0][2])
        self.assertTrue(any("官方模型列表" in item for item in progress))
        self.assertFalse(any("改用 LiteLLM 子集" in item for item in progress))

    def test_budget_exhausted_message_uses_started_budget_seconds(self):
        import time

        provider_worker = LiteLLMProviderCatalogWorker()
        provider_worker._start_budget(CATALOG_TIMEOUT_SECONDS)
        provider_worker._deadline = time.monotonic() - 1.0
        with self.assertRaises(BudgetExhausted) as ctx:
            provider_worker._remaining_timeout()
        message = str(ctx.exception)
        self.assertIn(f"{CATALOG_TIMEOUT_SECONDS:g}", message)
        self.assertNotIn(f"{CATALOG_TOTAL_BUDGET_SECONDS:g}", message)

        model_worker = LiteLLMModelCatalogWorker("openai", api_key="sk-test")
        model_worker._start_budget(CATALOG_TOTAL_BUDGET_SECONDS)
        model_worker._deadline = time.monotonic() - 1.0
        with self.assertRaises(BudgetExhausted) as ctx_model:
            model_worker._remaining_timeout()
        self.assertIn(f"{CATALOG_TOTAL_BUDGET_SECONDS:g}", str(ctx_model.exception))

    def test_version_worker_reads_latest_stable_version_from_pypi(self):
        payload = {
            "info": {"version": "1.92.0", "requires_python": ">=3.10,<3.14"},
            "releases": {
                "1.83.7": [
                    {"requires_python": ">=3.9,<4.0", "yanked": False},
                ],
                "1.92.0": [
                    {"requires_python": ">=3.10,<3.14", "yanked": False},
                ],
            },
        }
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(json.dumps(payload).encode("utf-8"))
        completed = []
        worker = LiteLLMVersionWorker()
        worker.completed.connect(
            lambda installed, latest, compatible, requirement, error: completed.append(
                (installed, latest, compatible, requirement, error)
            )
        )

        with (
            mock.patch("gui_qt.litellm_worker.urlopen", return_value=response),
            mock.patch("gui_qt.litellm_worker.sys.version_info", (3, 14, 0)),
            mock.patch(
                "gui_qt.litellm_worker.installed_litellm_version",
                return_value="1.83.7",
            ),
        ):
            worker.run()

        self.assertEqual(
            completed,
            [("1.83.7", "1.92.0", "1.83.7", ">=3.10,<3.14", None)],
        )

    def test_version_worker_reports_metadata_errors_and_completes(self):
        completed = []
        worker = LiteLLMVersionWorker()
        worker.completed.connect(
            lambda installed, latest, compatible, requirement, error: completed.append(
                (installed, latest, compatible, requirement, error)
            )
        )

        with mock.patch(
            "gui_qt.litellm_worker.installed_litellm_version",
            side_effect=RuntimeError("broken metadata"),
        ):
            worker.run()

        self.assertEqual(completed, [("", "", "", "", "broken metadata")])


@unittest.skipIf(
    LiteLLMModuleWarmupWorker is None,
    f"GUI dependencies are unavailable: {IMPORT_ERROR}",
)
class LiteLLMModuleWarmupWorkerTests(unittest.TestCase):
    def test_warmup_emits_cached_module(self):
        fake_litellm = SimpleNamespace(models_by_provider={"warmed_prefix": ()})
        completed = []
        worker = LiteLLMModuleWarmupWorker()
        worker.completed.connect(lambda module: completed.append(module))

        with mock.patch(
            "gui_qt.litellm_worker.warm_litellm_module",
            return_value=fake_litellm,
        ):
            worker.run()

        self.assertEqual(completed, [fake_litellm])

    def test_warmup_emits_none_when_probe_fails(self):
        completed = []
        worker = LiteLLMModuleWarmupWorker()
        worker.completed.connect(lambda module: completed.append(module))

        with mock.patch(
            "gui_qt.litellm_worker.warm_litellm_module",
            side_effect=RuntimeError("broken import"),
        ):
            worker.run()

        self.assertEqual(completed, [None])

if __name__ == "__main__":
    unittest.main()
