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


    def test_model_catalog_prefers_official_online_data(self):
        payload = {
            "gpt-current": {"litellm_provider": "openai", "mode": "chat"},
        }
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(json.dumps(payload).encode("utf-8"))
        completed = []
        worker = LiteLLMModelCatalogWorker("openai")
        worker.completed.connect(
            lambda models, source, error: completed.append((models, source, error))
        )

        with mock.patch("gui_qt.litellm_worker.urlopen", return_value=response):
            worker.run()

        self.assertEqual(completed, [(('openai/gpt-current',), "online", None)])

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
        payload = {"info": {"version": "1.92.0"}}
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(json.dumps(payload).encode("utf-8"))
        completed = []
        worker = LiteLLMVersionWorker()
        worker.completed.connect(lambda installed, latest, error: completed.append((installed, latest, error)))

        with (
            mock.patch("gui_qt.litellm_worker.urlopen", return_value=response),
            mock.patch(
                "gui_qt.litellm_worker.installed_litellm_version",
                return_value="1.83.7",
            ),
        ):
            worker.run()

        self.assertEqual(completed, [("1.83.7", "1.92.0", None)])

if __name__ == "__main__":
    unittest.main()
