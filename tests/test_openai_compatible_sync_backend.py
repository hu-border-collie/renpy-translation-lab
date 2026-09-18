"""Offline contract tests for the direct OpenAI-compatible sync backend (#431 S1)."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import unittest
from unittest import mock

from openai_compatible_sync_backend import (
    HTTPResponse,
    OpenAICompatibleSyncBackend,
)
from sync_model_backend import SyncBackendError, SyncGenerationRequest


def json_response(
    payload: dict,
    *,
    status: int = 200,
    headers: dict | None = None,
) -> HTTPResponse:
    return HTTPResponse(
        status=status,
        headers=dict(headers or {"Content-Type": "application/json"}),
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )


def chat_payload(
    *,
    content: str = '{"translations": []}',
    reasoning: str = "",
    finish_reason: str = "stop",
    usage: dict | None = None,
) -> dict:
    message: dict = {"content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": dict(
            usage
            or {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        ),
    }


class RecordingTransport:
    """Capture one request and return a queued response/exception."""

    def __init__(self, response=None, *, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, request, timeout):
        self.calls.append(
            {
                "url": request.full_url,
                "headers": dict(request.headers),
                "body": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response


def header_value(headers: dict, name: str) -> str | None:
    """Case-insensitive header lookup; urllib normalizes header casing."""

    target = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == target:
            return value
    return None


def make_backend(
    transport,
    *,
    credential_ref: dict | None = None,
    params: dict | None = None,
    structured_output_mode: str = "prompt_only_json",
    extra_headers: dict | None = None,
    api_key: str | None = None,
) -> OpenAICompatibleSyncBackend:
    return OpenAICompatibleSyncBackend(
        provider="openai",
        base_url="https://api.example/v1",
        credential_ref=credential_ref or {"kind": "none"},
        params=params,
        structured_output_mode=structured_output_mode,
        extra_headers=extra_headers,
        api_key=api_key,
        transport=transport,
    )


class RequestShapeTests(unittest.TestCase):
    def test_chat_completions_request_shape(self) -> None:
        transport = RecordingTransport(json_response(chat_payload()))
        backend = make_backend(transport, params={"temperature": 0.2, "max_output_tokens": 256})
        result = backend.generate(
            SyncGenerationRequest(
                model="org/model-x",
                contents="hello",
                config={
                    "system_instruction": "system prompt",
                    "timeout": 30,
                },
            )
        )

        call = transport.calls[0]
        self.assertEqual(call["url"], "https://api.example/v1/chat/completions")
        self.assertEqual(call["timeout"], 30)
        self.assertEqual(
            call["body"]["messages"],
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertEqual(call["body"]["model"], "org/model-x")
        self.assertEqual(call["body"]["temperature"], 0.2)
        self.assertEqual(call["body"]["max_tokens"], 256)
        self.assertNotIn("timeout", call["body"])
        self.assertNotIn("max_output_tokens", call["body"])
        self.assertFalse(call["body"]["stream"])
        self.assertNotIn("response_format", call["body"])
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.execution_mode, "sync")
        self.assertEqual(result.response_text, '{"translations": []}')
        self.assertEqual(result.parsed, {"translations": []})
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage_metadata["total_tokens"], 18)

    def test_base_url_already_ending_in_chat_completions_is_kept(self) -> None:
        transport = RecordingTransport(json_response(chat_payload()))
        backend = OpenAICompatibleSyncBackend(
            provider="openai",
            base_url="https://api.example/v1/chat/completions",
            credential_ref={"kind": "none"},
            transport=transport,
        )
        backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(
            transport.calls[0]["url"],
            "https://api.example/v1/chat/completions",
        )

    def test_base_url_without_scheme_is_unsupported(self) -> None:
        backend = OpenAICompatibleSyncBackend(
            provider="openai",
            base_url="api.example.com/v1/chat/completions",
            credential_ref={"kind": "none"},
            transport=RecordingTransport(json_response(chat_payload())),
        )
        with self.assertRaises(SyncBackendError) as captured:
            backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(captured.exception.category, "unsupported_capability")
        self.assertEqual(
            captured.exception.request_metadata["reason"],
            "invalid_base_url",
        )

    def test_base_url_with_query_preserves_query(self) -> None:
        transport = RecordingTransport(json_response(chat_payload()))
        backend = OpenAICompatibleSyncBackend(
            provider="azure",
            base_url=(
                "https://x.example/openai/deployments/d"
                "?api-version=2024-02-01"
            ),
            credential_ref={"kind": "none"},
            transport=transport,
        )
        result = backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(
            transport.calls[0]["url"],
            "https://x.example/openai/deployments/d/chat/completions"
            "?api-version=2024-02-01",
        )
        self.assertEqual(
            result.request_metadata["request_url"],
            "https://x.example/openai/deployments/d/chat/completions",
        )

    def test_message_list_contents_are_preserved(self) -> None:
        transport = RecordingTransport(json_response(chat_payload(content="ok")))
        backend = make_backend(transport)
        backend.generate(
            SyncGenerationRequest(
                model="m",
                contents=[
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                ],
            )
        )
        self.assertEqual(
            transport.calls[0]["body"]["messages"],
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
            ],
        )

    def test_profile_params_override_request_defaults(self) -> None:
        transport = RecordingTransport(json_response(chat_payload()))
        backend = make_backend(transport, params={"temperature": 0.9})
        backend.generate(
            SyncGenerationRequest(
                model="m",
                contents="x",
                config={"temperature": 0.1, "max_output_tokens": 64},
            )
        )
        self.assertEqual(transport.calls[0]["body"]["temperature"], 0.9)
        self.assertEqual(transport.calls[0]["body"]["max_tokens"], 64)
        self.assertNotIn("max_output_tokens", transport.calls[0]["body"])
        self.assertNotIn("timeout", transport.calls[0]["body"])


class StructuredOutputTests(unittest.TestCase):
    schema = {
        "type": "object",
        "properties": {"translations": {"type": "array"}},
    }

    def test_strict_json_schema_mode(self) -> None:
        transport = RecordingTransport(json_response(chat_payload()))
        backend = make_backend(transport, structured_output_mode="strict_json_schema")
        backend.generate(
            SyncGenerationRequest(
                model="m",
                contents="x",
                config={"response_json_schema": self.schema},
            )
        )
        response_format = transport.calls[0]["body"]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "translations_response")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(response_format["json_schema"]["schema"], self.schema)

    def test_json_object_mode(self) -> None:
        transport = RecordingTransport(json_response(chat_payload()))
        backend = make_backend(transport, structured_output_mode="json_object")
        backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(
            transport.calls[0]["body"]["response_format"],
            {"type": "json_object"},
        )

    def test_prompt_only_mode_sends_no_response_format(self) -> None:
        transport = RecordingTransport(json_response(chat_payload()))
        backend = make_backend(transport, structured_output_mode="prompt_only_json")
        backend.generate(
            SyncGenerationRequest(
                model="m",
                contents="x",
                config={"response_json_schema": self.schema},
            )
        )
        self.assertNotIn("response_format", transport.calls[0]["body"])

    def test_strict_mode_without_schema_is_unsupported(self) -> None:
        backend = make_backend(
            RecordingTransport(json_response(chat_payload())),
            structured_output_mode="strict_json_schema",
        )
        with self.assertRaises(SyncBackendError) as captured:
            backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(captured.exception.category, "unsupported_capability")
        self.assertEqual(
            captured.exception.request_metadata["reason"],
            "missing_response_schema",
        )

    def test_unknown_mode_is_unsupported(self) -> None:
        backend = make_backend(
            RecordingTransport(json_response(chat_payload())),
            structured_output_mode="script_mode",
        )
        with self.assertRaises(SyncBackendError) as captured:
            backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(captured.exception.category, "unsupported_capability")

    def test_safety_settings_fail_closed(self) -> None:
        backend = make_backend(RecordingTransport(json_response(chat_payload())))
        with self.assertRaises(SyncBackendError) as captured:
            backend.generate(
                SyncGenerationRequest(
                    model="m",
                    contents="x",
                    config={"safety_settings": [{"category": "HARM"}]},
                )
            )
        self.assertEqual(captured.exception.category, "unsupported_capability")
        self.assertEqual(
            captured.exception.request_metadata["reason"],
            "gemini_only_option",
        )

    def test_empty_safety_settings_is_ignored(self) -> None:
        transport = RecordingTransport(json_response(chat_payload()))
        backend = make_backend(transport)
        backend.generate(
            SyncGenerationRequest(
                model="m",
                contents="x",
                config={"safety_settings": []},
            )
        )
        self.assertNotIn("safety_settings", transport.calls[0]["body"])


class CredentialAndHeaderTests(unittest.TestCase):
    def test_explicit_api_key_sets_bearer_header(self) -> None:
        transport = RecordingTransport(json_response(chat_payload()))
        backend = make_backend(transport, api_key="secret-key")
        backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(
            header_value(transport.calls[0]["headers"], "Authorization"),
            "Bearer secret-key",
        )

    def test_env_credential_is_resolved(self) -> None:
        transport = RecordingTransport(json_response(chat_payload()))
        backend = make_backend(
            transport,
            credential_ref={"kind": "env", "name": "MY_OPENAI_KEY"},
        )
        with mock.patch.dict(os.environ, {"MY_OPENAI_KEY": "env-key"}):
            backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(
            header_value(transport.calls[0]["headers"], "Authorization"),
            "Bearer env-key",
        )

    def test_missing_env_credential_is_authentication(self) -> None:
        backend = make_backend(
            RecordingTransport(json_response(chat_payload())),
            credential_ref={"kind": "env", "name": "MISSING_OPENAI_KEY"},
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SyncBackendError) as captured:
                backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(captured.exception.category, "authentication")
        self.assertEqual(
            captured.exception.request_metadata["reason"],
            "credential_env_missing",
        )

    def test_keyring_reader_import_failure_is_missing_dependency(self) -> None:
        backend = make_backend(
            RecordingTransport(json_response(chat_payload())),
            credential_ref={"kind": "keyring", "name": "openai"},
        )
        with mock.patch.dict("sys.modules", {"litellm_provider_config": None}):
            with self.assertRaises(SyncBackendError) as captured:
                backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(captured.exception.category, "missing_dependency")
        self.assertEqual(
            captured.exception.request_metadata["reason"],
            "credential_reader_unavailable",
        )

    def test_keyring_reader_error_is_provider_error(self) -> None:
        backend = make_backend(
            RecordingTransport(json_response(chat_payload())),
            credential_ref={"kind": "keyring", "name": "openai"},
        )
        with mock.patch(
            "litellm_provider_config.load_provider_api_key",
            side_effect=RuntimeError("keyring backend exploded"),
        ):
            with self.assertRaises(SyncBackendError) as captured:
                backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(captured.exception.category, "provider_error")
        self.assertEqual(
            captured.exception.request_metadata["reason"],
            "credential_reader_error",
        )

    def test_keyless_credential_sends_no_authorization(self) -> None:
        transport = RecordingTransport(json_response(chat_payload()))
        backend = make_backend(transport, credential_ref={"kind": "none"})
        backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertIsNone(header_value(transport.calls[0]["headers"], "Authorization"))

    def test_extra_headers_are_merged(self) -> None:
        transport = RecordingTransport(json_response(chat_payload()))
        backend = make_backend(
            transport,
            extra_headers={"X-Client-Version": "1.2.3"},
        )
        backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(
            header_value(transport.calls[0]["headers"], "X-Client-Version"),
            "1.2.3",
        )

    def test_non_ascii_extra_header_is_unsupported(self) -> None:
        backend = make_backend(
            RecordingTransport(json_response(chat_payload())),
            extra_headers={"X-Trace": "中文"},
        )
        with self.assertRaises(SyncBackendError) as captured:
            backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(captured.exception.category, "unsupported_capability")
        self.assertEqual(
            captured.exception.request_metadata["reason"],
            "invalid_extra_header",
        )

    def test_extra_headers_cannot_override_authorization(self) -> None:
        transport = RecordingTransport(json_response(chat_payload()))
        backend = make_backend(
            transport,
            api_key="real-key",
            extra_headers={"Authorization": "Bearer attacker"},
        )
        with self.assertRaises(SyncBackendError) as captured:
            backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(captured.exception.category, "unsupported_capability")
        self.assertEqual(
            captured.exception.request_metadata["reason"],
            "sensitive_extra_header",
        )
        self.assertEqual(transport.calls, [])

    def test_extra_headers_reject_crlf(self) -> None:
        backend = make_backend(
            RecordingTransport(json_response(chat_payload())),
            extra_headers={"X-Trace": "value\r\nX-Evil: 1"},
        )
        with self.assertRaises(SyncBackendError) as captured:
            backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(captured.exception.category, "unsupported_capability")
        self.assertEqual(
            captured.exception.request_metadata["reason"],
            "invalid_extra_header",
        )


class ResponseAndErrorTests(unittest.TestCase):
    def test_reasoning_is_not_mixed_into_final_text(self) -> None:
        transport = RecordingTransport(
            json_response(
                chat_payload(
                    content='{"translations": []}',
                    reasoning="internal chain",
                )
            )
        )
        result = make_backend(transport).generate(
            SyncGenerationRequest(model="m", contents="x")
        )
        self.assertEqual(result.response_text, '{"translations": []}')
        self.assertTrue(result.output_diagnostics["reasoning_present"])
        self.assertEqual(result.output_diagnostics["reasoning_chars"], len("internal chain"))

    def test_content_parts_are_joined(self) -> None:
        payload = chat_payload()
        payload["choices"][0]["message"]["content"] = [
            {"type": "text", "text": "part-a"},
            {"type": "text", "text": "part-b"},
        ]
        transport = RecordingTransport(json_response(payload))
        result = make_backend(transport).generate(
            SyncGenerationRequest(model="m", contents="x")
        )
        self.assertEqual(result.response_text, "part-apart-b")

    def test_http_errors_map_to_shared_categories(self) -> None:
        cases = (
            (401, b'{"error": "bad key"}', "authentication"),
            (429, b'{"error": "slow down"}', "rate_limit"),
            (503, b'{"error": "down"}', "service_unavailable"),
        )
        for status, body, expected in cases:
            with self.subTest(status=status):
                backend = make_backend(
                    RecordingTransport(HTTPResponse(status, {}, body))
                )
                with self.assertRaises(SyncBackendError) as captured:
                    backend.generate(SyncGenerationRequest(model="m", contents="x"))
                self.assertEqual(captured.exception.category, expected)

    def test_structured_output_rejection_is_unsupported(self) -> None:
        backend = make_backend(
            RecordingTransport(
                HTTPResponse(
                    400,
                    {},
                    b'{"error": "response_format json_schema is not supported"}',
                )
            ),
            structured_output_mode="strict_json_schema",
        )
        with self.assertRaises(SyncBackendError) as captured:
            backend.generate(
                SyncGenerationRequest(
                    model="m",
                    contents="x",
                    config={"response_json_schema": {"type": "object"}},
                )
            )
        self.assertEqual(captured.exception.category, "unsupported_capability")
        self.assertEqual(
            captured.exception.request_metadata["reason"],
            "structured_output_unsupported",
        )

    def test_timeout_is_classified(self) -> None:
        backend = make_backend(
            RecordingTransport(error=TimeoutError("timed out"))
        )
        with self.assertRaises(SyncBackendError) as captured:
            backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(captured.exception.category, "timeout")

    def test_invalid_json_response_is_invalid_response(self) -> None:
        backend = make_backend(
            RecordingTransport(HTTPResponse(200, {}, b"not json"))
        )
        with self.assertRaises(SyncBackendError) as captured:
            backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(captured.exception.category, "invalid_response")
        self.assertEqual(
            captured.exception.request_metadata["reason"],
            "response_not_json",
        )

    def test_missing_choices_is_invalid_response(self) -> None:
        backend = make_backend(RecordingTransport(json_response({"id": "x"})))
        with self.assertRaises(SyncBackendError) as captured:
            backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(captured.exception.category, "invalid_response")

    def test_empty_content_is_invalid_response(self) -> None:
        backend = make_backend(
            RecordingTransport(json_response(chat_payload(content="")))
        )
        with self.assertRaises(SyncBackendError) as captured:
            backend.generate(SyncGenerationRequest(model="m", contents="x"))
        self.assertEqual(captured.exception.category, "invalid_response")
        self.assertEqual(
            captured.exception.request_metadata["reason"],
            "empty_response_text",
        )


class AsyncWrapperTests(unittest.TestCase):
    def test_generate_async_propagates_cancellation(self) -> None:
        release = threading.Event()

        def blocking_transport(_request, _timeout):
            release.wait(timeout=2)
            return json_response(chat_payload())

        backend = make_backend(blocking_transport)

        async def cancel_running_request():
            task = asyncio.create_task(
                backend.generate_async(
                    SyncGenerationRequest(model="m", contents="x")
                )
            )
            await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        try:
            asyncio.run(cancel_running_request())
        finally:
            release.set()

    def test_generate_async_returns_the_same_result(self) -> None:
        transport = RecordingTransport(json_response(chat_payload()))
        backend = make_backend(transport)
        result = asyncio.run(
            backend.generate_async(SyncGenerationRequest(model="m", contents="x"))
        )
        self.assertEqual(result.response_text, '{"translations": []}')
        self.assertEqual(len(transport.calls), 1)


if __name__ == "__main__":
    unittest.main()
