"""Offline coverage for native Gemini catalog discovery and adapter dispatch."""

from __future__ import annotations

import json
import socket
import unittest
import urllib.error
from unittest import mock

from providers import model_profiles_editor as editor
from providers import profile_model_catalog as catalog
from providers.sync_model_backend import SyncBackendError


def response(models, **extra):
    return catalog.CatalogResponse(200, json.dumps({"models": models, **extra}).encode())


def model(name, method="generateContent"):
    return {"name": f"models/{name}", "supportedGenerationMethods": [method]}


class ProfileModelCatalogTests(unittest.TestCase):
    def setUp(self):
        self.connection = catalog.GeminiCatalogConnection(
            credential_ref={"kind": "env", "name": "CATALOG_TEST_KEY"},
        )

    def fetch(self, transport, connection=None, **kwargs):
        return catalog.fetch_models(
            connection or self.connection, transport=transport,
            credential_loader=lambda _: "SECRET_CATALOG_KEY", **kwargs,
        )

    def test_profiles_resolve_native_connection_without_loading_credentials(self):
        section = editor.initial_section(
            provider_label="Gemini", profile_label="Gemini Main", model="gemini-4-argon",
        )
        profile_id = editor.profile_ids(section)[0]
        with mock.patch("providers.model_capability_probe.default_credential_loader") as load:
            connection = catalog.connection_for_profile(section, profile_id)
        load.assert_not_called()
        self.assertEqual(catalog.adapter_for_profile(section, profile_id), "gemini")
        self.assertIsInstance(connection, catalog.GeminiCatalogConnection)
        self.assertEqual(connection.provider_models_url(), f"{catalog.GEMINI_API_BASE}/models")

    def test_unknown_profile_and_adapter_are_unsupported(self):
        section = editor.empty_section()
        self.assertEqual(catalog.adapter_for_profile(section, "missing"), "")
        with self.assertRaises(SyncBackendError) as caught:
            catalog.connection_for_profile(section, "missing")
        self.assertEqual(caught.exception.category, "unsupported_capability")

    def test_malformed_provider_reference_does_not_break_catalog_availability(self):
        section = {"profiles": {"bad": {"provider_id": []}}, "providers": {}}
        self.assertEqual(catalog.adapter_for_profile(section, "bad"), "")

    def test_unsupported_gemini_credential_reference_never_falls_back_to_other_keys(self):
        for ref in ({"kind": "keyring", "name": "other-provider"},
                    {"kind": "api_keys_json", "name": "other-slot"}):
            connection = catalog.GeminiCatalogConnection(ref)
            transport, loader = mock.Mock(), mock.Mock(return_value="other-key")
            with self.subTest(ref=ref), self.assertRaises(SyncBackendError) as caught:
                catalog.fetch_models(connection, transport=transport, credential_loader=loader)
            self.assertEqual(caught.exception.category, "unsupported_capability")
            loader.assert_not_called()
            transport.assert_not_called()

    def test_lists_all_pages_and_filters_generation_with_one_timeout_budget(self):
        transport = mock.Mock(side_effect=[
            response([
                model("gemini-4-argon"), model("gemini-embedding-2", "embedContent"),
                None, {"name": "models/missing-method"},
            ], nextPageToken="next/token+value"),
            response([model("gemini-3.8-flash"), model("gemini-4-argon")]),
        ])
        with mock.patch.object(catalog.time, "monotonic", side_effect=[100, 100, 101, 103, 104]):
            models = self.fetch(transport, timeout_seconds=15)
        self.assertEqual(models, ("gemini-3.8-flash", "gemini-4-argon"))
        first, second = transport.call_args_list
        self.assertEqual(first.args[0].get_method(), "GET")
        self.assertEqual(first.args[0].get_header("X-goog-api-key"), "SECRET_CATALOG_KEY")
        self.assertNotIn("SECRET_CATALOG_KEY", first.args[0].full_url)
        self.assertIn("pageSize=1000", first.args[0].full_url)
        self.assertIn("pageToken=next%2Ftoken%2Bvalue", second.args[0].full_url)
        self.assertEqual((first.args[1], second.args[1]), (15, 12))

    def test_embedding_profile_only_lists_embedding_models(self):
        connection = catalog.GeminiCatalogConnection(self.connection.credential_ref, purpose="embedding")
        models = self.fetch(mock.Mock(return_value=response([
            model("gemini-4-argon"), model("gemini-embedding-2", "embedContent"),
            model("gemini-embedding-001", "embedContent"),
        ])), connection)
        self.assertEqual(models, ("gemini-embedding-001", "gemini-embedding-2"))

    def test_env_credential_is_resolved_only_from_the_profile_reference(self):
        transport = mock.Mock(return_value=response([model("gemini-4-argon")]))
        with mock.patch.dict("os.environ", {
            "CATALOG_TEST_KEY": "profile-key", "GEMINI_API_KEY": "other-key",
        }, clear=True):
            catalog.fetch_models(self.connection, transport=transport)
        self.assertEqual(transport.call_args.args[0].get_header("X-goog-api-key"), "profile-key")

    def test_api_keys_json_reference_uses_existing_gemini_loader(self):
        connection = catalog.GeminiCatalogConnection({"kind": "api_keys_json", "name": "api_keys"})
        transport = mock.Mock(return_value=response([model("gemini-4-argon")]))
        with mock.patch("providers.model_capability_probe._resolve_gemini_api_key", return_value="file-key"):
            catalog.fetch_models(connection, transport=transport)
        self.assertEqual(transport.call_args.args[0].get_header("X-goog-api-key"), "file-key")

    def test_missing_credential_never_sends_request(self):
        transport = mock.Mock()
        with self.assertRaises(SyncBackendError) as caught:
            catalog.fetch_models(self.connection, transport=transport, credential_loader=lambda _: "")
        self.assertEqual(caught.exception.category, "authentication")
        transport.assert_not_called()

    def test_http_failures_use_safe_categories_without_echoing_response(self):
        for status, category in ((400, "authentication"), (401, "authentication"),
                                 (403, "authentication"), (429, "rate_limit"),
                                 (503, "service_unavailable"), (404, "provider_error")):
            with self.subTest(status=status), self.assertRaises(SyncBackendError) as caught:
                self.fetch(mock.Mock(return_value=catalog.CatalogResponse(status, b"SECRET_PROVIDER_BODY")))
            self.assertEqual(caught.exception.category, category)
            self.assertNotIn("SECRET", str(caught.exception))

    def test_transport_failures_use_safe_categories(self):
        for error, category in ((socket.timeout("SECRET"), "timeout"),
                                (urllib.error.URLError(socket.timeout("SECRET")), "timeout"),
                                (urllib.error.URLError("SECRET"), "provider_error"),
                                (OSError("SECRET"), "provider_error")):
            with self.subTest(error=type(error)), self.assertRaises(SyncBackendError) as caught:
                self.fetch(mock.Mock(side_effect=error))
            self.assertEqual(caught.exception.category, category)
            self.assertNotIn("SECRET", str(caught.exception))

    def test_invalid_or_empty_catalog_never_looks_successful(self):
        for body in (b"not-json SECRET", b"[]", b"{}", b'{"models":{}}',
                     b'{"models":[]}', response([model("embed-only", "embedContent")]).body):
            with self.subTest(body=body), self.assertRaises(SyncBackendError) as caught:
                self.fetch(mock.Mock(return_value=catalog.CatalogResponse(200, body)))
            self.assertEqual(caught.exception.category, "invalid_response")
            self.assertNotIn("SECRET", str(caught.exception))

    def test_timeout_expiry_does_not_request_another_page(self):
        transport = mock.Mock(return_value=response([model("gemini-4-argon")], nextPageToken="more"))
        with mock.patch.object(catalog.time, "monotonic", side_effect=[0, 0, 16]):
            with self.assertRaises(SyncBackendError) as caught:
                self.fetch(transport, timeout_seconds=15)
        self.assertEqual(caught.exception.category, "timeout")
        self.assertEqual(transport.call_count, 1)


    def test_repeated_page_token_and_page_limit_reject_partial_catalog(self):
        page = response([model("gemini-4-argon")], nextPageToken="same-token")
        transport = mock.Mock(return_value=page)
        with self.assertRaises(SyncBackendError) as caught:
            self.fetch(transport)
        self.assertEqual(caught.exception.category, "invalid_response")
        self.assertEqual(transport.call_count, 2)
        with mock.patch.object(catalog, "MAX_GEMINI_CATALOG_PAGES", 1):
            with self.assertRaises(SyncBackendError) as caught:
                self.fetch(mock.Mock(return_value=page))
        self.assertEqual(caught.exception.category, "invalid_response")


class LiteLLMProfileCatalogTests(unittest.TestCase):
    def section(self, provider="deepseek", *, kind="env", base_url="", models_url=""):
        section = editor.add_provider(
            editor.empty_section(), label="Catalog connection", adapter="litellm", provider=provider,
            base_url=base_url, models_url=models_url, credential_kind=kind,
            credential_name="PROFILE_CATALOG_KEY" if kind != "none" else "",
        )
        return editor.add_profile(
            section, label="Main", provider_id=editor.provider_ids(section)[0], model=f"{provider}/saved-model",
        )

    def test_deepseek_catalog_returns_prefixed_ids_using_only_profile_credentials(self):
        section = self.section()
        loader = mock.Mock(return_value="PROFILE_SECRET")
        transport = mock.Mock(return_value=catalog.CatalogResponse(
            200, json.dumps({"data": [{"id": "new-model"}, {"id": "deepseek/other-model"}]}).encode(),
        ))
        with mock.patch("providers.litellm_provider_config.warm_litellm_module") as sdk:
            connection = catalog.connection_for_profile(section, editor.profile_ids(section)[0])
            loader.assert_not_called()
            models = catalog.fetch_models(connection, credential_loader=loader, transport=transport)
        sdk.assert_not_called()
        self.assertEqual(models, ("deepseek/new-model", "deepseek/other-model"))
        self.assertEqual(loader.call_args.args[0]["credential_ref"]["name"], "PROFILE_CATALOG_KEY")
        request = transport.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/models")
        self.assertEqual(request.get_header("Authorization"), "Bearer PROFILE_SECRET")
        self.assertEqual(request.get_method(), "GET")

    def test_explicit_custom_endpoint_is_used_without_falling_back(self):
        section = self.section("relay", models_url="https://catalog.example/custom/models", kind="none")
        connection = catalog.connection_for_profile(section, editor.profile_ids(section)[0])
        transport = mock.Mock(return_value=catalog.CatalogResponse(503, b"SECRET_RESPONSE"))
        loader = mock.Mock()
        with self.assertRaises(SyncBackendError) as caught:
            catalog.fetch_models(connection, transport=transport, credential_loader=loader)
        self.assertEqual(caught.exception.category, "service_unavailable")
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(transport.call_args.args[0].full_url, "https://catalog.example/custom/models")
        loader.assert_not_called()

    def test_custom_base_url_overrides_known_provider_native_address(self):
        section = self.section(base_url="https://gateway.example/v1")
        connection = catalog.connection_for_profile(section, editor.profile_ids(section)[0])
        self.assertEqual(connection.provider_models_url(), "https://gateway.example/v1/models")

    def test_native_authentication_style_is_preserved(self):
        section = self.section("anthropic")
        connection = catalog.connection_for_profile(section, editor.profile_ids(section)[0])
        transport = mock.Mock(return_value=catalog.CatalogResponse(200, b'{"data":[{"id":"example-text"}]}'))
        models = catalog.fetch_models(connection, transport=transport, credential_loader=lambda _: "KEY")
        request = transport.call_args.args[0]
        self.assertEqual(request.get_header("X-api-key"), "KEY")
        self.assertEqual(request.get_header("Anthropic-version"), "2023-06-01")
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(models, ("anthropic/example-text",))

    def test_missing_required_credentials_fail_before_request(self):
        for kind in ("none", "env"):
            section = self.section(kind=kind)
            connection = catalog.connection_for_profile(section, editor.profile_ids(section)[0])
            transport = mock.Mock()
            with self.subTest(kind=kind), self.assertRaises(SyncBackendError) as caught:
                catalog.fetch_models(connection, transport=transport, credential_loader=lambda _: "")
            self.assertEqual(caught.exception.category, "authentication")
            transport.assert_not_called()

    def test_openrouter_anonymous_catalog_keeps_litellm_prefix_and_vendor_slash(self):
        section = self.section("openrouter", kind="none")
        connection = catalog.connection_for_profile(section, editor.profile_ids(section)[0])
        transport = mock.Mock(return_value=catalog.CatalogResponse(200, b'{"data":[{"id":"vendor/model"}]}'))
        models = catalog.fetch_models(connection, transport=transport)
        self.assertEqual(models, ("openrouter/vendor/model",))
        self.assertIsNone(transport.call_args.args[0].get_header("Authorization"))

    def test_unknown_endpoint_is_unsupported_but_saved_model_is_an_offline_choice(self):
        section = self.section("unknown", kind="none")
        profile_id = editor.profile_ids(section)[0]
        self.assertEqual(catalog.suggested_models(section, profile_id), ("unknown/saved-model",))
        with self.assertRaises(SyncBackendError) as caught:
            catalog.connection_for_profile(section, profile_id)
        self.assertEqual(caught.exception.category, "unsupported_capability")

    def test_malformed_connection_identity_and_suggestions_are_safe(self):
        for section in ({"profiles": [1]}, {"profiles": {"bad": {"provider_id": []}}}):
            self.assertIsInstance(catalog.connection_identity(section, "bad"), str)
            self.assertEqual(catalog.suggested_models(section, "bad"), ())

if __name__ == "__main__":
    unittest.main()
