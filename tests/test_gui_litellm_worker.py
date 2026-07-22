import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from gui_qt.litellm_worker import (
        CONNECTION_TEST_TIMEOUT_SECONDS,
        LiteLLMConnectionTestWorker,
        LiteLLMModelCatalogWorker,
        LiteLLMVersionWorker,
    )
    from litellm_sync_backend import LiteLLMBackendError
except ImportError as exc:
    LiteLLMConnectionTestWorker = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(
    LiteLLMConnectionTestWorker is None,
    f"GUI dependencies are unavailable: {IMPORT_ERROR}",
)
class LiteLLMConnectionTestWorkerTests(unittest.TestCase):
    def test_connection_test_passes_a_bounded_timeout(self):
        backend = mock.Mock()
        backend.generate.return_value = SimpleNamespace(response_text="OK")
        completed = []
        worker = LiteLLMConnectionTestWorker("openai/test", "typed-secret")
        worker.completed.connect(lambda success, message: completed.append((success, message)))

        with mock.patch(
            "gui_qt.litellm_worker.LiteLLMSyncBackend",
            return_value=backend,
        ):
            worker.run()

        request = backend.generate.call_args.args[0]
        self.assertEqual(
            request.config["timeout"],
            CONNECTION_TEST_TIMEOUT_SECONDS,
        )
        self.assertEqual(completed, [(True, "连接成功。模型返回：OK")])

    def test_connection_error_never_includes_provider_exception_text(self):
        backend = mock.Mock()
        backend.generate.side_effect = LiteLLMBackendError(
            "provider echoed stored-secret",
            category="authentication",
        )
        completed = []
        worker = LiteLLMConnectionTestWorker("openai/test")
        worker.completed.connect(lambda success, message: completed.append((success, message)))

        with mock.patch(
            "gui_qt.litellm_worker.LiteLLMSyncBackend",
            return_value=backend,
        ):
            worker.run()

        self.assertFalse(completed[0][0])
        self.assertIn("authentication", completed[0][1])
        self.assertNotIn("stored-secret", completed[0][1])
        self.assertNotIn("provider echoed", completed[0][1])


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

    def test_model_catalog_marks_local_fallback_as_possibly_stale(self):
        completed = []
        worker = LiteLLMModelCatalogWorker("openai")
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )

        with (
            mock.patch("gui_qt.litellm_worker.urlopen", side_effect=OSError("offline")),
            mock.patch(
                "gui_qt.litellm_worker.models_for_provider",
                return_value=("openai/local-model",),
            ),
        ):
            worker.run()

        self.assertEqual(completed[0][0], ("openai/local-model",))
        self.assertEqual(completed[0][1], "local")
        self.assertIn("已改用本地目录", completed[0][2])

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

if __name__ == "__main__":
    unittest.main()
