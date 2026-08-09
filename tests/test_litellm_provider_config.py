import unittest
from types import SimpleNamespace

from litellm_provider_config import (
    KEYRING_SERVICE,
    CustomLiteLLMProvider,
    ProviderApiKeyStore,
    _decode_provider_key_store,
    catalog_source_label,
    custom_provider_from_mapping,
    custom_provider_registry,
    delete_provider_api_key,
    load_provider_api_key,
    load_provider_api_keys,
    load_provider_key_store,
    latest_compatible_litellm_version,
    models_for_provider,
    models_from_ollama_payload,
    models_from_openai_compatible_payload,
    models_from_openrouter_payload,
    models_from_remote_catalog,
    native_catalog_endpoint,
    parse_custom_litellm_providers,
    provider_display_label,
    providers_from_remote_catalog,
    reserved_litellm_provider_ids,
    resolve_provider_id,
    sort_provider_ids,
    store_provider_api_key,
    store_provider_key_store,
    credential_provider_candidates,
    validate_custom_provider_id,
    validate_custom_provider_url,
    version_key,
    provider_from_model,
    python_requirement_allows,
)


class _FakeKeyring:
    def __init__(self):
        self.values = {}

    def get_password(self, service, username):
        return self.values.get((service, username))

    def set_password(self, service, username, password):
        self.values[(service, username)] = password

    def delete_password(self, service, username):
        del self.values[(service, username)]


class LiteLLMProviderConfigTests(unittest.TestCase):
    def test_key_is_stored_by_provider_in_keyring(self):
        keyring = _FakeKeyring()
        store_provider_api_key("OpenAI", " secret-value ", keyring)

        self.assertEqual(load_provider_api_key("openai", keyring), "secret-value")
        self.assertEqual(
            keyring.values[(KEYRING_SERVICE, "openai")],
            "secret-value",
        )
        self.assertTrue(delete_provider_api_key("openai", keyring))
        self.assertEqual(load_provider_api_key("openai", keyring), "")
        self.assertFalse(delete_provider_api_key("openai", keyring))

    def test_multi_key_store_preserves_active_index_and_legacy_plaintext(self):
        keyring = _FakeKeyring()
        # Legacy single plaintext secret remains readable.
        keyring.set_password(KEYRING_SERVICE, "deepseek", "legacy-key")
        self.assertEqual(load_provider_api_key("deepseek", keyring), "legacy-key")
        self.assertEqual(load_provider_api_keys("deepseek", keyring), ("legacy-key",))

        store_provider_key_store(
            "deepseek",
            ("key-one", "key-two", "key-three"),
            keyring,
            active_index=1,
        )
        store = load_provider_key_store("deepseek", keyring)
        self.assertEqual(store.keys, ("key-one", "key-two", "key-three"))
        self.assertEqual(store.active_index, 1)
        self.assertEqual(store.active_key(), "key-two")
        self.assertEqual(load_provider_api_key("deepseek", keyring), "key-two")
        encoded = keyring.values[(KEYRING_SERVICE, "deepseek")]
        self.assertIn('"version":1', encoded)
        self.assertIn("key-two", encoded)

        store_provider_key_store("deepseek", ProviderApiKeyStore(), keyring)
        self.assertEqual(load_provider_api_keys("deepseek", keyring), ())
        self.assertNotIn((KEYRING_SERVICE, "deepseek"), keyring.values)

    def test_decode_unrecognized_json_object_keeps_opaque_secret(self):
        opaque = '{"token":"still-a-secret","not":"our-schema"}'
        store = _decode_provider_key_store(opaque)
        self.assertEqual(store.keys, (opaque,))
        self.assertEqual(store.active_key(), opaque)

        empty_keys = '{"version":1,"keys":[],"active_index":0}'
        store = _decode_provider_key_store(empty_keys)
        self.assertEqual(store.keys, (empty_keys,))

    def test_store_provider_key_store_active_index_overrides_mapping(self):
        keyring = _FakeKeyring()
        store_provider_key_store(
            "openai",
            {"keys": ["a", "b"], "active_index": 0},
            keyring,
            active_index=1,
        )
        self.assertEqual(load_provider_api_key("openai", keyring), "b")

    def test_ollama_does_not_use_keyring(self):
        keyring = _FakeKeyring()
        self.assertEqual(load_provider_api_key("ollama", keyring), "")
        self.assertFalse(delete_provider_api_key("ollama", keyring))

    def test_credential_provider_candidates_merge_common_and_extra(self):
        providers = credential_provider_candidates(
            ("zzz_custom", "deepseek", "openai"),
            include_ollama=False,
        )
        self.assertIn("openai", providers)
        self.assertIn("anthropic", providers)
        self.assertIn("gemini", providers)
        self.assertIn("azure", providers)
        self.assertIn("vertex_ai", providers)
        self.assertIn("zzz_custom", providers)
        self.assertNotIn("ollama", providers)
        # Common ids stay ahead of free-form extras.
        self.assertLess(providers.index("openai"), providers.index("zzz_custom"))

    def test_catalog_filters_non_text_models_and_adds_provider_prefix(self):
        fake_litellm = SimpleNamespace(
            models_by_provider={"openai": ("gpt-text", "dall-e-test", "openai/gpt-prefixed")},
            model_cost={
                "gpt-text": {"mode": "chat"},
                "dall-e-test": {"mode": "image_generation"},
                "openai/gpt-prefixed": {"mode": "responses"},
            },
        )

        models = models_for_provider("openai", fake_litellm)

        self.assertIn("openai/gpt-text", models)
        self.assertIn("openai/gpt-prefixed", models)
        self.assertNotIn("openai/dall-e-test", models)

    def test_local_catalog_treats_missing_mode_as_text_like_remote_catalog(self):
        fake_litellm = SimpleNamespace(
            models_by_provider={"openai": ("gpt-without-mode",)},
            model_cost={"gpt-without-mode": {}},
        )

        self.assertEqual(
            models_for_provider("openai", fake_litellm),
            ("openai/gpt-without-mode",),
        )

    def test_local_catalog_does_not_mix_hardcoded_default_into_real_results(self):
        fake_litellm = SimpleNamespace(
            models_by_provider={"openai": ("gpt-current",)},
            model_cost={"gpt-current": {"mode": "chat"}},
        )

        self.assertEqual(
            models_for_provider("openai", fake_litellm),
            ("openai/gpt-current",),
        )

    def test_remote_catalog_filters_by_provider_and_text_mode(self):
        catalog = {
            "gpt-current": {"litellm_provider": "openai", "mode": "chat"},
            "openai/gpt-responses": {
                "litellm_provider": "openai",
                "mode": "responses",
            },
            "dall-e": {"litellm_provider": "openai", "mode": "image_generation"},
            "claude": {"litellm_provider": "anthropic", "mode": "chat"},
        }

        self.assertEqual(
            models_from_remote_catalog("openai", catalog),
            ("openai/gpt-current", "openai/gpt-responses"),
        )

    def test_remote_catalog_discovers_dynamic_and_native_providers(self):
        catalog = {
            "custom/model": {
                "litellm_provider": "custom_provider",
                "mode": "chat",
            },
            "other/model": {"mode": "chat"},
        }
        providers = providers_from_remote_catalog(catalog)
        self.assertIn("custom_provider", providers)
        self.assertIn("other", providers)
        self.assertIn("ollama", providers)

    def test_provider_sort_places_common_choices_before_dynamic_catalog(self):
        providers = sort_provider_ids(
            ("zzz_custom", "deepseek", "anthropic", "aaa_custom", "openai")
        )
        self.assertEqual(
            providers,
            ("openai", "anthropic", "deepseek", "aaa_custom", "zzz_custom"),
        )

    def test_provider_sort_accepts_mapping_views(self):
        providers = sort_provider_ids(
            {"zzz_custom": None, "openai": None, "anthropic": None}.keys()
        )

        self.assertEqual(providers, ("openai", "anthropic", "zzz_custom"))

    def test_openrouter_payload_prefixes_and_skips_non_text_and_aliases(self):
        payload = {
            "data": [
                {
                    "id": "openai/gpt-5",
                    "architecture": {"output_modalities": ["text"]},
                },
                {
                    "id": "openrouter/auto",
                    "architecture": {"output_modalities": ["text"]},
                },
                {
                    "id": "~openai/gpt-latest",
                    "architecture": {"output_modalities": ["text"]},
                },
                {
                    "id": "black-forest-labs/flux",
                    "architecture": {"output_modalities": ["image"]},
                },
                {
                    "id": "google/gemini-flash",
                    "architecture": {"output_modalities": ["text", "image"]},
                },
            ]
        }

        self.assertEqual(
            models_from_openrouter_payload(payload),
            (
                "openrouter/auto",
                "openrouter/google/gemini-flash",
                "openrouter/openai/gpt-5",
            ),
        )

    def test_openai_compatible_payload_filters_non_text_and_prefixes(self):
        payload = {
            "data": [
                {"id": "gpt-5"},
                {"id": "text-embedding-3-large"},
                {"id": "whisper-1"},
                {"id": "openai/gpt-4o"},
            ]
        }
        self.assertEqual(
            models_from_openai_compatible_payload("openai", payload),
            ("openai/gpt-4o", "openai/gpt-5"),
        )
        self.assertEqual(
            models_from_openai_compatible_payload(
                "deepseek",
                {"data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]},
            ),
            ("deepseek/deepseek-chat", "deepseek/deepseek-reasoner"),
        )

    def test_ollama_payload_uses_local_model_names(self):
        payload = {
            "models": [
                {"name": "llama3:latest"},
                {"name": "ollama/mistral"},
                {"model": "qwen2.5"},
            ]
        }
        self.assertEqual(
            models_from_ollama_payload(payload),
            ("ollama/llama3:latest", "ollama/mistral", "ollama/qwen2.5"),
        )

    def test_native_catalog_endpoints_cover_supported_providers(self):
        for provider, _label in (
            ("openai", "OpenAI"),
            ("anthropic", "Anthropic"),
            ("openrouter", "OpenRouter"),
            ("deepseek", "DeepSeek"),
            ("xai", "xAI"),
            ("ollama", "Ollama"),
        ):
            endpoint = native_catalog_endpoint(provider)
            self.assertIsNotNone(endpoint)
            self.assertEqual(endpoint.provider, provider)
            self.assertTrue(endpoint.url)
        self.assertIn("OpenAI 官方模型列表", catalog_source_label("openai"))
        self.assertIn("Ollama 本机", catalog_source_label("ollama"))
        self.assertIn("LiteLLM 官方在线目录", catalog_source_label("online"))
        self.assertEqual(catalog_source_label("local"), "目录来源：未知。")

    def test_resolve_provider_id_maps_display_labels_and_known_ids(self):
        self.assertEqual(resolve_provider_id("openai"), "openai")
        self.assertEqual(resolve_provider_id("OpenAI"), "openai")
        self.assertEqual(resolve_provider_id("Ollama（本地）"), "ollama")
        self.assertEqual(resolve_provider_id("Google Gemini"), "gemini")
        self.assertEqual(resolve_provider_id("Azure OpenAI"), "azure")
        self.assertEqual(resolve_provider_id("Google Vertex AI"), "vertex_ai")
        self.assertEqual(resolve_provider_id("MyCustomProvider"), "mycustomprovider")
        self.assertEqual(resolve_provider_id(""), "")

    def test_python_requirement_rejects_litellm_latest_on_python_314(self):
        self.assertFalse(
            python_requirement_allows(">=3.10,<3.14", (3, 14, 0))
        )
        self.assertTrue(
            python_requirement_allows(">=3.9,<4.0", (3, 14, 0))
        )

    def test_latest_compatible_version_respects_python_requirement(self):
        releases = {
            "1.83.7": [
                {"requires_python": ">=3.9,<4.0", "yanked": False},
            ],
            "1.92.0": [
                {"requires_python": ">=3.10,<3.14", "yanked": False},
            ],
            "1.93.0rc1": [
                {"requires_python": ">=3.10,<4.0", "yanked": False},
            ],
        }

        self.assertEqual(
            latest_compatible_litellm_version(releases, (3, 14, 0)),
            "1.83.7",
        )


    def test_stable_versions_compare_numerically(self):
        self.assertLess(version_key("1.83.7"), version_key("1.92.0"))

    def test_provider_is_derived_from_model_prefix(self):
        self.assertEqual(provider_from_model(" OpenRouter/openai/gpt-5 "), "openrouter")
        self.assertEqual(provider_from_model("gpt-5"), "")

    def test_custom_provider_parse_derives_models_url_and_env(self):
        raw = [
            {
                "id": "opencode-go",
                "label": "OpenCode Go",
                "base_url": "https://opencode.ai/zen/go/v1",
                "api_key_env": "OPENCODE_GO_API_KEY",
            }
        ]
        providers = parse_custom_litellm_providers(raw)
        self.assertEqual(len(providers), 1)
        provider = providers[0]
        self.assertEqual(provider.id, "opencode-go")
        self.assertEqual(provider.label, "OpenCode Go")
        self.assertEqual(provider.base_url, "https://opencode.ai/zen/go/v1")
        self.assertEqual(provider.models_url, "https://opencode.ai/zen/go/v1/models")
        self.assertEqual(provider.api_key_env, "OPENCODE_GO_API_KEY")

        registry = custom_provider_registry(raw)
        self.assertIsInstance(registry["opencode-go"], CustomLiteLLMProvider)

    def test_custom_provider_rejects_non_http_urls(self):
        with self.assertRaises(ValueError):
            validate_custom_provider_url("ftp://example.com", field_name="base_url")
        with self.assertRaises(ValueError):
            validate_custom_provider_url("opencode.ai/zen", field_name="base_url")
        with self.assertRaises(ValueError):
            validate_custom_provider_url("", field_name="base_url")
        self.assertEqual(
            validate_custom_provider_url("http://127.0.0.1:8000/v1", field_name="base_url"),
            "http://127.0.0.1:8000/v1",
        )
        with self.assertRaises(ValueError):
            custom_provider_from_mapping(
                {"id": "bad", "base_url": "not-a-url"},
                index=0,
            )

    def test_custom_provider_id_rules(self):
        self.assertEqual(validate_custom_provider_id("opencode-go"), "opencode-go")
        self.assertEqual(validate_custom_provider_id(" my_vendor_2 "), "my_vendor_2")
        self.assertEqual(validate_custom_provider_id("Uppercase"), "uppercase")
        with self.assertRaises(ValueError):
            validate_custom_provider_id("has space")
        with self.assertRaises(ValueError):
            validate_custom_provider_id("")
        # Built-in and common LiteLLM prefixes are reserved.
        for reserved in ("openai", "anthropic", "gemini", "azure", "ollama", "mistral"):
            with self.subTest(reserved=reserved):
                with self.assertRaises(ValueError):
                    validate_custom_provider_id(reserved)
        self.assertIn("openai", reserved_litellm_provider_ids())

    def test_custom_provider_id_merges_installed_litellm_table(self):
        fake_litellm = SimpleNamespace(models_by_provider={"some_custom_prefix": ()})
        reserved = reserved_litellm_provider_ids(fake_litellm)
        self.assertIn("some_custom_prefix", reserved)
        self.assertIn("openai", reserved)

    def test_custom_provider_duplicate_ids_are_rejected(self):
        raw = [
            {"id": "dup", "base_url": "https://a.example.com"},
            {"id": "dup", "base_url": "https://b.example.com"},
        ]
        with self.assertRaises(ValueError) as captured:
            parse_custom_litellm_providers(raw)
        self.assertIn("重复", str(captured.exception))

    def test_custom_provider_requires_list_shape(self):
        self.assertEqual(parse_custom_litellm_providers(None), ())
        with self.assertRaises(ValueError):
            parse_custom_litellm_providers({"id": "x", "base_url": "https://x"})

    def test_custom_provider_env_name_must_be_valid(self):
        with self.assertRaises(ValueError):
            custom_provider_from_mapping(
                {
                    "id": "vendor",
                    "base_url": "https://vendor.example.com",
                    "api_key_env": "BAD ENV NAME",
                }
            )
        provider = custom_provider_from_mapping(
            {"id": "vendor", "base_url": "https://vendor.example.com", "api_key_env": "_KEY1"}
        )
        self.assertEqual(provider.api_key_env, "_KEY1")

    def test_custom_provider_endpoint_label_and_source(self):
        registry = custom_provider_registry(
            [
                {
                    "id": "opencode-go",
                    "label": "OpenCode Go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                }
            ]
        )
        endpoint = native_catalog_endpoint("opencode-go", registry)
        self.assertIsNotNone(endpoint)
        self.assertEqual(endpoint.url, "https://opencode.ai/zen/go/v1/models")
        self.assertEqual(endpoint.label, "OpenCode Go")
        self.assertEqual(endpoint.source, "opencode-go")
        self.assertEqual(endpoint.auth, "bearer")
        self.assertTrue(endpoint.require_key)
        self.assertEqual(endpoint.payload_style, "openai")
        # Built-in endpoints still win over custom entries.
        self.assertEqual(
            native_catalog_endpoint("openai", registry).url,
            "https://api.openai.com/v1/models",
        )
        self.assertEqual(provider_display_label("opencode-go", registry), "OpenCode Go")
        self.assertEqual(provider_display_label("opencode-go"), "opencode-go")
        self.assertIn(
            "OpenCode Go 官方模型列表",
            catalog_source_label("opencode-go", registry),
        )
        self.assertEqual(catalog_source_label("opencode-go"), "目录来源：未知。")
        self.assertEqual(catalog_source_label("unknown"), "目录来源：未知。")


if __name__ == "__main__":
    unittest.main()
