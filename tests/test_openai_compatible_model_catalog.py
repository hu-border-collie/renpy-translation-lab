"""Offline tests for direct OpenAI-compatible model catalog discovery (#431 S2)."""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock

import model_profiles_editor as editor
import openai_compatible_model_catalog as catalog
from model_profile import ADAPTER_OPENAI_COMPATIBLE
from openai_compatible_connection import OpenAICompatibleConnection
from sync_model_backend import SyncBackendError


def models_response(payload: dict, *, status: int = 200) -> catalog.CatalogResponse:
    return catalog.CatalogResponse(
        status=status,
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )


def header_value(headers: dict, name: str) -> str | None:
    """Case-insensitive header lookup (urllib normalizes header casing)."""

    target = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == target:
            return value
    return None


class RecordingTransport:
    def __init__(self, response=None, *, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, request, timeout):
        self.calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "headers": dict(request.headers),
                "timeout": timeout,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response


def direct_section(*, models_url: str = "", credential_kind: str = "none"):
    section = editor.empty_section()
    section = editor.add_provider(
        section,
        label="OpenAI",
        adapter=ADAPTER_OPENAI_COMPATIBLE,
        provider="openai",
        base_url="https://api.example/v1",
        models_url=models_url,
        credential_kind=credential_kind,
        credential_name="openai",
        credential_env_name="OPENAI_API_KEY",
        extra_headers={"X-Client-Version": "1.0"},
    )
    provider_id = editor.provider_ids(section)[0]
    section = editor.add_profile(
        section,
        label="OpenAI GPT",
        provider_id=provider_id,
        model="gpt-4.1-mini",
    )
    profile_id = editor.profile_ids(section)[0]
    section = editor.set_defaults(
        section,
        primary_profile_id=profile_id,
        execution_strategy="sync",
    )
    return section, provider_id, profile_id


class ParseModelPayloadTests(unittest.TestCase):
    def test_parse_keeps_ids_and_slashes(self) -> None:
        payload = {
            "data": [
                {"id": "org/model-b"},
                {"id": "model-a"},
                {"id": "model-a"},
                {"id": ""},
                "not-an-object",
            ]
        }
        self.assertEqual(
            catalog.parse_models_payload(payload),
            ("model-a", "org/model-b"),
        )

    def test_parse_rejects_non_mapping_or_missing_data(self) -> None:
        self.assertEqual(catalog.parse_models_payload([]), ())
        self.assertEqual(catalog.parse_models_payload({"data": {}}), ())
        self.assertEqual(catalog.parse_models_payload({"data": []}), ())


class FetchModelTests(unittest.TestCase):
    def test_fetch_models_builds_get_request_and_headers(self) -> None:
        transport = RecordingTransport(
            models_response({"data": [{"id": "gpt-4.1-mini"}]})
        )
        connection = OpenAICompatibleConnection(
            provider="openai",
            base_url="https://api.example/v1",
            credential_ref={"kind": "env", "name": "MY_OPENAI_KEY"},
            extra_headers={"X-Client-Version": "1.0"},
        )
        with mock.patch.dict(os.environ, {"MY_OPENAI_KEY": "test-key"}):
            models = catalog.fetch_models(
                connection,
                timeout_seconds=30,
                transport=transport,
            )
        self.assertEqual(models, ("gpt-4.1-mini",))
        call = transport.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["url"], "https://api.example/v1/models")
        self.assertEqual(call["timeout"], 30)
        self.assertEqual(
            header_value(call["headers"], "Authorization"),
            "Bearer test-key",
        )
        self.assertEqual(
            header_value(call["headers"], "X-Client-Version"),
            "1.0",
        )
        self.assertIsNone(header_value(call["headers"], "Content-Type"))

    def test_explicit_models_url_wins_and_query_is_redacted_helper(self) -> None:
        transport = RecordingTransport(
            models_response({"data": [{"id": "m"}]})
        )
        connection = OpenAICompatibleConnection(
            provider="azure",
            base_url="https://api.example",
            models_url="https://api.example/openai/models?api-version=2024-02-01",
            credential_ref={"kind": "none"},
        )
        catalog.fetch_models(connection, transport=transport)
        self.assertEqual(
            transport.calls[0]["url"],
            "https://api.example/openai/models?api-version=2024-02-01",
        )
        self.assertEqual(
            catalog.redacted_endpoint(connection.provider_models_url()),
            "https://api.example/openai/models",
        )

    def test_http_errors_map_to_shared_categories(self) -> None:
        cases = (
            (401, "authentication"),
            (429, "rate_limit"),
            (503, "service_unavailable"),
        )
        for status, expected in cases:
            with self.subTest(status=status):
                connection = OpenAICompatibleConnection(
                    provider="openai",
                    base_url="https://api.example/v1",
                    credential_ref={"kind": "none"},
                )
                with self.assertRaises(SyncBackendError) as captured:
                    catalog.fetch_models(
                        connection,
                        transport=RecordingTransport(
                            catalog.CatalogResponse(status=status, body=b"{}")
                        ),
                    )
                self.assertEqual(captured.exception.category, expected)

    def test_invalid_or_empty_catalog_payload_fails(self) -> None:
        connection = OpenAICompatibleConnection(
            provider="openai",
            base_url="https://api.example/v1",
            credential_ref={"kind": "none"},
        )
        with self.assertRaises(SyncBackendError) as captured:
            catalog.fetch_models(
                connection,
                transport=RecordingTransport(
                    catalog.CatalogResponse(status=200, body=b"not json")
                ),
            )
        self.assertEqual(captured.exception.category, "invalid_response")
        with self.assertRaises(SyncBackendError) as captured:
            catalog.fetch_models(
                connection,
                transport=RecordingTransport(
                    models_response({"data": []})
                ),
            )
        self.assertEqual(captured.exception.request_metadata["reason"], "catalog_empty")

    def test_connection_for_profile_reads_provider_fields(self) -> None:
        section, _provider_id, profile_id = direct_section(
            models_url="https://api.example/v1/models"
        )
        connection = catalog.connection_for_profile(section, profile_id)
        self.assertEqual(connection.provider, "openai")
        self.assertEqual(connection.base_url, "https://api.example/v1")
        self.assertEqual(connection.provider_models_url(), "https://api.example/v1/models")
        self.assertEqual(connection.extra_headers, {"X-Client-Version": "1.0"})

    def test_connection_for_profile_rejects_gemini_profile(self) -> None:
        section = editor.empty_section()
        section = editor.add_provider(
            section,
            label="Gemini",
            adapter="gemini",
            provider="gemini",
            credential_kind="api_keys_json",
            credential_name="api_keys",
        )
        provider_id = editor.provider_ids(section)[0]
        section = editor.add_profile(
            section,
            label="Gemini Main",
            provider_id=provider_id,
            model="gemini-3.5-flash",
        )
        profile_id = editor.profile_ids(section)[0]
        with self.assertRaises(SyncBackendError) as captured:
            catalog.connection_for_profile(section, profile_id)
        self.assertEqual(captured.exception.category, "unsupported_capability")

    def test_sensitive_extra_header_is_rejected_on_catalog_path(self) -> None:
        connection = OpenAICompatibleConnection(
            provider="openai",
            base_url="https://api.example/v1",
            credential_ref={"kind": "none"},
            extra_headers={"Authorization": "Bearer leaked"},
        )
        with self.assertRaises(SyncBackendError) as captured:
            catalog.fetch_models(
                connection,
                transport=RecordingTransport(
                    models_response({"data": [{"id": "m"}]})
                ),
            )
        self.assertEqual(captured.exception.category, "unsupported_capability")
        self.assertEqual(
            captured.exception.request_metadata["reason"],
            "sensitive_extra_header",
        )


if __name__ == "__main__":
    unittest.main()
