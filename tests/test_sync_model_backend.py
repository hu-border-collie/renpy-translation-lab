import unittest

from sync_model_backend import GeminiSyncBackend, SyncGenerationRequest, SyncModelBackend


class _Models:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _Client:
    def __init__(self, response):
        self.models = _Models(response)


class _Response:
    parsed = [{"id": "a", "translation": "你好"}]


class SyncModelBackendTests(unittest.TestCase):
    def test_gemini_adapter_standardizes_metadata_and_payload(self):
        client = _Client(_Response())
        backend = GeminiSyncBackend(
            client,
            serialize_response=lambda response: {
                "text": "[]", "finish_reason": "STOP", "usage": {"total_tokens": 12},
            },
            extract_text=lambda payload: payload["text"],
            extract_finish_reason=lambda payload: payload["finish_reason"],
            extract_usage=lambda payload: payload["usage"],
        )
        self.assertIsInstance(backend, SyncModelBackend)
        result = backend.generate(SyncGenerationRequest(
            model="gemini-test", contents="prompt", config={"temperature": 0.2},
        ))
        self.assertEqual(client.models.calls, [{
            "model": "gemini-test", "contents": "prompt", "config": {"temperature": 0.2},
        }])
        self.assertEqual(result.provider, "gemini")
        self.assertEqual(result.model, "gemini-test")
        self.assertEqual(result.execution_mode, "sync")
        self.assertEqual(result.response_text, "[]")
        self.assertEqual(result.finish_reason, "STOP")
        self.assertEqual(result.usage_metadata, {"total_tokens": 12})
        self.assertEqual(result.parsed, _Response.parsed)

    def test_request_config_is_copied_before_sdk_call(self):
        config = {"nested": {"enabled": True}}
        client = _Client(_Response())
        backend = GeminiSyncBackend(
            client,
            serialize_response=lambda response: {},
            extract_text=lambda payload: "",
            extract_finish_reason=lambda payload: "",
        )
        backend.generate(SyncGenerationRequest("gemini-test", [], config))
        self.assertIsNot(client.models.calls[0]["config"], config)
        self.assertEqual(client.models.calls[0]["config"], config)


if __name__ == "__main__":
    unittest.main()
