import unittest
from types import SimpleNamespace

from litellm_provider_config import (
    KEYRING_SERVICE,
    catalog_source_label,
    delete_provider_api_key,
    load_provider_api_key,
    latest_compatible_litellm_version,
    models_for_provider,
    models_from_ollama_payload,
    models_from_openai_compatible_payload,
    models_from_openrouter_payload,
    models_from_remote_catalog,
    native_catalog_endpoint,
    version_key,
    provider_from_model,
    python_requirement_allows,
    store_provider_api_key,
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

    def test_ollama_does_not_use_keyring(self):
        keyring = _FakeKeyring()
        self.assertEqual(load_provider_api_key("ollama", keyring), "")
        self.assertFalse(delete_provider_api_key("ollama", keyring))

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


if __name__ == "__main__":
    unittest.main()
