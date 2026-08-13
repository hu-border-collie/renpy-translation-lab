import unittest

from sync_model_backend import (
    DEFAULT_SYNC_TIMEOUT_SECONDS,
    MAX_SYNC_TIMEOUT_SECONDS,
    MIN_SYNC_TIMEOUT_SECONDS,
    GeminiSyncBackend,
    SyncGenerationRequest,
    SyncBackendError,
    SyncModelBackend,
    normalize_sync_timeout_seconds,
    sync_error_category,
    sync_error_summary,
    sync_recovery_decision,
)


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
            "model": "gemini-test",
            "contents": "prompt",
            "config": {
                "temperature": 0.2,
                "http_options": {
                    "timeout": DEFAULT_SYNC_TIMEOUT_SECONDS * 1000,
                },
            },
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
        self.assertEqual(config, {"nested": {"enabled": True}})
        self.assertEqual(
            client.models.calls[0]["config"],
            {
                "nested": {"enabled": True},
                "http_options": {
                    "timeout": DEFAULT_SYNC_TIMEOUT_SECONDS * 1000,
                },
            },
        )

    def test_gemini_adapter_keeps_named_schema_and_removes_internal_mode_hint(self):
        schema = {
            "type": "object",
            "properties": {"translations": {"type": "array"}},
        }
        client = _Client(_Response())
        backend = GeminiSyncBackend(
            client,
            serialize_response=lambda response: {},
            extract_text=lambda payload: "",
            extract_finish_reason=lambda payload: "",
        )

        backend.generate(
            SyncGenerationRequest(
                "gemini-test",
                [],
                {
                    "response_json_schema": schema,
                    "structured_output_mode": "strict_json_schema",
                },
            )
        )

        sent = client.models.calls[0]["config"]
        self.assertEqual(sent["response_json_schema"], schema)
        self.assertNotIn("structured_output_mode", sent)

    def test_gemini_timeout_seconds_becomes_http_options_milliseconds(self):
        config = {
            "timeout": 12,
            "http_options": {"headers": {"X-Test": "yes"}},
        }
        client = _Client(_Response())
        backend = GeminiSyncBackend(
            client,
            serialize_response=lambda response: {},
            extract_text=lambda payload: "",
            extract_finish_reason=lambda payload: "",
        )

        backend.generate(SyncGenerationRequest("gemini-test", [], config))

        self.assertEqual(config["timeout"], 12)
        self.assertEqual(
            client.models.calls[0]["config"]["http_options"],
            {"headers": {"X-Test": "yes"}, "timeout": 12_000},
        )
        self.assertNotIn("timeout", client.models.calls[0]["config"])

    def test_gemini_provider_failure_is_classified_without_secret_text(self):
        client = _Client(_Response())
        error = RuntimeError("provider echoed fake-secret-value")
        error.status_code = 401

        def fail(**_kwargs):
            raise error

        client.models.generate_content = fail
        backend = GeminiSyncBackend(
            client,
            serialize_response=lambda response: {},
            extract_text=lambda payload: "",
            extract_finish_reason=lambda payload: "",
        )

        with self.assertRaises(SyncBackendError) as captured:
            backend.generate(SyncGenerationRequest("gemini-test", [], {}))

        self.assertEqual(captured.exception.category, "authentication")
        self.assertEqual(
            captured.exception.request_metadata,
            {"provider": "gemini"},
        )
        self.assertNotIn("fake-secret-value", str(captured.exception))

    def test_gemini_response_extraction_failure_is_invalid_response(self):
        backend = GeminiSyncBackend(
            _Client(_Response()),
            serialize_response=lambda _response: (_ for _ in ()).throw(
                ValueError("provider response contained fake-secret-value")
            ),
            extract_text=lambda payload: "",
            extract_finish_reason=lambda payload: "",
        )

        with self.assertRaises(SyncBackendError) as captured:
            backend.generate(SyncGenerationRequest("gemini-test", [], {}))

        self.assertEqual(captured.exception.category, "invalid_response")
        self.assertNotIn("fake-secret-value", str(captured.exception))

    def test_sync_timeout_normalization_has_finite_bounds(self):
        self.assertEqual(
            normalize_sync_timeout_seconds(None),
            DEFAULT_SYNC_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            normalize_sync_timeout_seconds(True),
            DEFAULT_SYNC_TIMEOUT_SECONDS,
        )
        self.assertEqual(normalize_sync_timeout_seconds(1), MIN_SYNC_TIMEOUT_SECONDS)
        self.assertEqual(normalize_sync_timeout_seconds(9999), MAX_SYNC_TIMEOUT_SECONDS)
        self.assertEqual(normalize_sync_timeout_seconds("45"), 45)

    def test_new_gemini_model_omits_sampling_before_sdk_call(self):
        config = {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 20,
            "max_output_tokens": 1024,
        }
        client = _Client(_Response())
        backend = GeminiSyncBackend(
            client,
            serialize_response=lambda response: {},
            extract_text=lambda payload: "",
            extract_finish_reason=lambda payload: "",
        )
        backend.generate(SyncGenerationRequest("gemini-3.5-flash-lite", [], config))
        self.assertEqual(
            client.models.calls[0]["config"],
            {
                "max_output_tokens": 1024,
                "http_options": {
                    "timeout": DEFAULT_SYNC_TIMEOUT_SECONDS * 1000,
                },
            },
        )
        self.assertIn("temperature", config)

    def test_recovery_decisions_are_category_driven(self):
        cases = (
            (401, "authentication", False, False),
            (429, "rate_limit", True, False),
            (408, "timeout", True, False),
            (503, "service_unavailable", True, False),
        )
        for status, category, retry, split in cases:
            with self.subTest(status=status):
                error = RuntimeError("provider details")
                error.status_code = status
                decision = sync_recovery_decision(error)
                self.assertEqual(decision.category, category)
                self.assertEqual(decision.retry_same_request, retry)
                self.assertEqual(decision.split_request, split)

        invalid = RuntimeError("bad payload")
        invalid.reason_code = "empty_response_text"
        decision = sync_recovery_decision(invalid)
        self.assertEqual(decision.category, "invalid_response")
        self.assertTrue(decision.split_request)
        self.assertFalse(decision.retry_same_request)

    def test_explicit_category_wins_and_safe_summary_hides_details(self):
        error = RuntimeError("provider echoed secret-value")
        error.category = "authentication"
        error.status_code = 429

        self.assertEqual(sync_error_category(error), "authentication")
        summary = sync_error_summary(error)
        self.assertIn("authentication", summary)
        self.assertNotIn("secret-value", summary)

        error.category = "secret-value"
        self.assertEqual(sync_error_category(error), "rate_limit")


if __name__ == "__main__":
    unittest.main()
