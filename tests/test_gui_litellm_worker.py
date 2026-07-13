import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from gui_qt.litellm_worker import (
        CONNECTION_TEST_TIMEOUT_SECONDS,
        LiteLLMConnectionTestWorker,
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


if __name__ == "__main__":
    unittest.main()
